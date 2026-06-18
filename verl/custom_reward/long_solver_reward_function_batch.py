"""Batch reward function for long-form solver GRPO training.

Scores solver outputs based on format compliance (think/tool/answer) and
normalized rubric accuracy from an LLM grader. Unlike the challenger reward
which measures task *difficulty*, this directly rewards higher rubric scores.

Usage::

    custom_reward_function.name=compute_long_solver_score_batch
    custom_reward_function.path=verl/custom_reward/long_solver_reward_function_batch.py

Note:
    Heavy dependencies (torch, numpy) are imported lazily inside the entry
    point so that lightweight helpers can be imported and tested without GPU
    libraries.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from typing import Any

from scope.utils.parse_grader import (
    aggregate_per_rubric_results,
    parse_grader_response,
)
from scope.utils.parse_solver import parse_solver_response
from verl.custom_reward.batch_reward_rollout import BatchMultiTurnRollout
from verl.custom_reward.benchmark_evaluators import BENCHMARK_EVALUATORS
from verl.custom_reward.long_reward_function import (
    ASSISTANT_PATTERN,
    INFORMATION_PATTERN,
    THINK_PATTERN,
    compute_rubric_sum_with_recovery,
    count_valid_function_calls_search,
    count_valid_search_r1_calls,
    count_valid_tool_calls,
)
try:
    from verl.custom_reward.reward_function import em_check
except ImportError:
    # reward_function.py transitively imports sglang which may not be
    # installed in lightweight test environments.  Fall back to a local
    # copy that has no heavy dependencies.
    import string

    def _normalize_answer(s: str) -> str:
        """Normalize answer for exact match comparison."""
        s = s.lower()
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())

    def em_check(prediction: str, golden_answers: str | list[str]) -> int:
        """Check exact match after normalization.

        Args:
            prediction: The predicted answer string.
            golden_answers: A single gold answer or list of gold answers.

        Returns:
            int: 1 if prediction matches any gold answer, 0 otherwise.
        """
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = _normalize_answer(prediction)
        for golden_answer in golden_answers:
            if _normalize_answer(golden_answer) == normalized_prediction:
                return 1
        return 0

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Pattern to extract the first user message from a ChatML conversation
USER_MESSAGE_PATTERN = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"


def _extract_evidence_text(raw_response: str) -> str:
    """Extract ``<information>`` blocks from a multi-turn solver response.

    Used by the ``answer_with_evidence`` grader input mode to prepend
    retrieved evidence before the extracted answer, giving the grader
    visibility into the search results that informed the answer.

    Args:
        raw_response: Full decoded multi-turn solver response containing
            interleaved ``<search>``, ``<information>``, ``<think>``, and
            ``<answer>`` blocks.

    Returns:
        Formatted evidence string with section header, or empty string
        if no information blocks were found.
    """
    matches = re.findall(INFORMATION_PATTERN, raw_response, re.DOTALL)
    blocks = [m.strip() for m in matches if m.strip()]
    if not blocks:
        return ""
    formatted = "\n\n".join(
        f"<information>\n{b}\n</information>" for b in blocks
    )
    return f"## Retrieved Evidence\n{formatted}"


def _tokenize_for_ngram(text: str) -> list[str]:
    """Lowercase word tokenization for n-gram overlap computation.

    Args:
        text: Raw text string.

    Returns:
        List of lowercase word tokens (alphanumeric sequences).
    """
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def _compute_ngram_overlap_ratio(
    answer_text: str, doc_text: str, n: int,
) -> float:
    """Fraction of answer n-grams that appear in the document text.

    Used to detect verbatim copying from retrieved ``<information>``
    blocks into the solver's ``<answer>``.

    Args:
        answer_text: The extracted answer text.
        doc_text: Concatenated retrieved document text.
        n: N-gram size (e.g. 8).

    Returns:
        Overlap ratio in [0, 1].  Returns 0.0 if the answer has fewer
        than *n* tokens.
    """
    a_tok = _tokenize_for_ngram(answer_text)
    d_tok = _tokenize_for_ngram(doc_text)
    if len(a_tok) < n:
        return 0.0
    a_ng = [tuple(a_tok[i : i + n]) for i in range(len(a_tok) - n + 1)]
    d_set = set(
        tuple(d_tok[i : i + n]) for i in range(len(d_tok) - n + 1)
    )
    return sum(1 for ng in a_ng if ng in d_set) / len(a_ng)


def _format_rubric_list_from_ground_truth(ground_truth: dict[str, Any]) -> str:
    """Format rubrics from a ground_truth dict as a numbered list.

    Unlike ``format_rubric_list`` which takes an ``ExtractedTask``, this
    operates on the raw ground_truth dict stored in parquet data (accessed
    via ``reward_model.ground_truth``).

    Args:
        ground_truth: Dict with keys ``"rubrics"`` (list of str) and
            optionally ``"priorities"`` (list of str | None).

    Returns:
        str: Numbered rubric list, e.g. ``"1. rubric1\\n2. rubric2"``.
    """
    rubrics = ground_truth.get("rubrics", [])
    return "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rubrics))


def _extract_task_prompt(raw_message: str) -> str:
    """Extract the task prompt from the first user message in a conversation.

    Looks for the first ``<|im_start|>user`` block and extracts the
    ``"Question: ..."`` portion. If no ``Question:`` delimiter is found,
    returns the full first user message.

    Args:
        raw_message: Full decoded ChatML conversation from ``input_ids``.

    Returns:
        str: The extracted task prompt text.
    """
    user_match = re.search(USER_MESSAGE_PATTERN, raw_message, re.DOTALL)
    if not user_match:
        return ""
    content = user_match.group(1).strip()
    # Split on "Question: " to get the actual question
    if "Question: " in content:
        return content.split("Question: ", 1)[1].strip()
    return content


def _count_tokens(text: str, processing_class: Any) -> int:
    """Count the number of tokens in a text string.

    Args:
        text: The text to tokenize.
        processing_class: Tokenizer instance with an ``encode`` method.

    Returns:
        int: Number of tokens in the text.
    """
    return len(processing_class.encode(text, add_special_tokens=False))


def compute_length_penalty(
    answer_tokens: int,
    soft_limit: int,
    hard_limit: int,
    floor: float,
) -> float:
    """Cosine length penalty for answer token counts.

    Returns 1.0 for answers at or below ``soft_limit``, smoothly decays
    via a cosine curve to ``floor`` at ``hard_limit``, and clamps to
    ``floor`` for anything above.

    Args:
        answer_tokens: Number of tokens in the answer.
        soft_limit: Token count below which no penalty is applied.
        hard_limit: Token count at which the penalty reaches ``floor``.
        floor: Minimum penalty multiplier (applied at and above
            ``hard_limit``).

    Returns:
        float: Penalty multiplier in ``[floor, 1.0]``.
    """
    import math

    if answer_tokens <= soft_limit:
        return 1.0
    if answer_tokens >= hard_limit:
        return floor
    t = (answer_tokens - soft_limit) / (hard_limit - soft_limit)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t))


def _truncate_to_tokens(
    text: str,
    max_tokens: int,
    processing_class: Any,
) -> tuple[str, bool]:
    """Truncate text to at most ``max_tokens`` tokens.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens to keep.
        processing_class: Tokenizer instance with ``encode`` and ``decode``
            methods.

    Returns:
        tuple[str, bool]: The (possibly truncated) text and a boolean
            indicating whether truncation occurred.
    """
    token_ids = processing_class.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text, False
    truncated_ids = token_ids[:max_tokens]
    return processing_class.decode(truncated_ids, skip_special_tokens=True), True


def compute_long_solver_format_scores(
    raw_messages: list[str],
    responses: list[str],
    solver_format: str = "tool_call",
    processing_class: Any | None = None,
    think_token_budget: int | None = None,
    answer_token_limit: int | None = None,
) -> tuple[list[float], list[bool]]:
    """Compute format compliance scores for solver outputs.

    Evaluates three components and averages them:
    1. **Think reward**: Fraction of assistant turns containing ``<think>`` tags.
       When ``think_token_budget`` is set, a turn whose ``<think>`` content
       exceeds the budget does NOT count toward ``think_count``, reducing
       the think reward for that item.
    2. **Tool reward**: Ratio of valid search tool calls to
       ``(total_assistant_turns - 1)``, since the last turn should be the
       answer, not a tool call.
    3. **Answer reward**: 1.0 if ``parse_solver_response()`` yields a valid
       answer (has ``<answer>`` tags), else 0.0. When ``answer_token_limit``
       is set, answers exceeding the limit get ``answer_reward = 0.0``.

    Args:
        raw_messages: Decoded ``input_ids`` strings (full ChatML conversations).
        responses: Decoded response strings (policy outputs).
        solver_format: Tool interaction format. ``"tool_call"`` counts
            ``<tool_call>`` JSON tags; ``"search_r1"`` counts ``<search>``
            tags. Defaults to ``"tool_call"``.
        processing_class: Tokenizer instance. Required when
            ``think_token_budget`` or ``answer_token_limit`` is set.
        think_token_budget: Maximum allowed tokens per ``<think>`` block.
            ``None`` disables think budget enforcement.
        answer_token_limit: Maximum allowed tokens for the ``<answer>``
            content. ``None`` disables answer limit enforcement.

    Returns:
        tuple[list[float], list[bool]]: A tuple of (format_scores,
            answer_over_limits) where format_scores are in [0, 1] for
            each item, and answer_over_limits is True for items whose
            answer exceeded the token limit.
    """
    format_scores: list[float] = []
    answer_over_limits: list[bool] = []
    for raw_msg, response in zip(raw_messages, responses):
        # Combine full conversation for assistant turn analysis
        full_text = raw_msg + response
        assistant_turns = re.findall(ASSISTANT_PATTERN, full_text, re.DOTALL)
        num_turns = len(assistant_turns)

        # Think reward: fraction of assistant turns with <think> tags
        if num_turns > 0:
            think_count = 0
            for turn in assistant_turns:
                think_match = re.search(THINK_PATTERN, turn, re.DOTALL)
                if not think_match:
                    continue
                # Check budget if enabled
                if (
                    think_token_budget is not None
                    and processing_class is not None
                ):
                    think_content = think_match.group(1)
                    token_count = _count_tokens(
                        think_content, processing_class
                    )
                    if token_count > think_token_budget:
                        continue  # violation — don't count this turn
                think_count += 1
            think_reward = think_count / num_turns
        else:
            think_reward = 0.0

        # Tool reward: ratio of valid tool calls to (turns - 1)
        # Last turn is the answer turn, so tool calls expected in earlier turns.
        # Count only in assistant turns to avoid spurious matches from user
        # messages (e.g., invalid-action nudge text containing <search> tags).
        if num_turns > 1:
            non_answer_turns = assistant_turns[:-1]
            if solver_format in ("function_calls", "function_calls_xml"):
                valid_calls = sum(
                    count_valid_function_calls_search(t)
                    for t in non_answer_turns
                )
            elif solver_format == "search_r1":
                valid_calls = sum(
                    count_valid_search_r1_calls(t)
                    for t in non_answer_turns
                )
            elif solver_format == "call_tool":
                from drtulu.call_tool_parser import count_valid_call_tool_calls
                valid_calls = sum(
                    count_valid_call_tool_calls(t)
                    for t in non_answer_turns
                )
            elif solver_format == "webthinker":
                from webthinker.call_tool_parser import count_valid_search_queries
                valid_calls = sum(
                    count_valid_search_queries(t)
                    for t in non_answer_turns
                )
            elif solver_format == "webexplorer":
                from webexplorer.call_tool_parser import count_valid_tool_calls as count_valid_webexplorer_calls
                valid_calls = sum(
                    count_valid_webexplorer_calls(t)
                    for t in non_answer_turns
                )
            else:
                valid_calls = sum(
                    count_valid_tool_calls(t) for t in non_answer_turns
                )
            tool_reward = min(valid_calls / (num_turns - 1), 1.0)
        else:
            tool_reward = 0.0

        # Answer reward: valid answer tag present
        # parse_solver_response already handles function_calls answer format
        solver_result = parse_solver_response(response)
        answer_reward = 1.0 if solver_result.is_valid else 0.0

        # Check answer token limit (track for logging only — no longer
        # zeroes answer_reward; length penalty is applied to accuracy instead)
        answer_over = False
        if (
            answer_token_limit is not None
            and processing_class is not None
            and solver_result.is_valid
            and solver_result.answer
        ):
            answer_tokens = _count_tokens(
                solver_result.answer, processing_class
            )
            if answer_tokens > answer_token_limit:
                answer_over = True

        answer_over_limits.append(answer_over)

        format_score = (think_reward + tool_reward + answer_reward) / 3.0
        format_scores.append(format_score)

    return format_scores, answer_over_limits


def compute_long_solver_format_errors(
    raw_messages: list[str],
    responses: list[str],
    solver_format: str = "tool_call",
    processing_class: Any | None = None,
    think_token_budget: int | None = None,
    answer_token_limit: int | None = None,
) -> list[list[str]]:
    """Detect specific format compliance errors for solver outputs.

    Unlike ``compute_long_solver_format_scores()`` which returns a scalar,
    this returns human-readable error strings suitable for privileged
    teacher feedback in SDPO.

    Checks three categories:
    1. **Think errors**: Any assistant turns missing ``<think>`` tags.
    2. **Search errors**: Non-final turns with fewer valid search calls
       than expected.
    3. **Answer errors**: Final response missing a valid ``<answer>`` tag.
    4. **Budget errors**: Think blocks exceeding ``think_token_budget`` or
       answer exceeding ``answer_token_limit``.

    Args:
        raw_messages: Decoded ``input_ids`` strings (full ChatML conversations).
        responses: Decoded response strings (policy outputs).
        solver_format: Tool interaction format. ``"tool_call"`` counts
            ``<tool_call>`` JSON tags; ``"search_r1"`` counts ``<search>``
            tags. Defaults to ``"tool_call"``.
        processing_class: Tokenizer instance. Required when
            ``think_token_budget`` or ``answer_token_limit`` is set.
        think_token_budget: Maximum allowed tokens per ``<think>`` block.
            ``None`` disables think budget error reporting.
        answer_token_limit: Maximum allowed tokens for the ``<answer>``
            content. ``None`` disables answer limit error reporting.

    Returns:
        list[list[str]]: Per-sample list of error message strings.
            Empty list means fully compliant.
    """
    all_errors: list[list[str]] = []
    for raw_msg, response in zip(raw_messages, responses):
        errors: list[str] = []
        full_text = raw_msg + response
        assistant_turns = re.findall(ASSISTANT_PATTERN, full_text, re.DOTALL)
        num_turns = len(assistant_turns)

        # Think errors: any assistant turns missing <think> tags
        if num_turns > 0:
            think_count = sum(
                1 for turn in assistant_turns
                if re.search(THINK_PATTERN, turn, re.DOTALL)
            )
            if think_count < num_turns:
                errors.append(
                    "Some of your reasoning turns are missing <think>...</think> "
                    "tags. Every assistant turn should begin with reasoning "
                    "inside <think> tags."
                )

        # Think budget errors
        if (
            think_token_budget is not None
            and processing_class is not None
            and num_turns > 0
        ):
            for turn in assistant_turns:
                think_match = re.search(THINK_PATTERN, turn, re.DOTALL)
                if think_match:
                    think_content = think_match.group(1)
                    token_count = _count_tokens(
                        think_content, processing_class
                    )
                    if token_count > think_token_budget:
                        errors.append(
                            f"Your <think> block exceeded the "
                            f"{think_token_budget}-token budget "
                            f"({token_count} tokens). Keep reasoning concise."
                        )
                        break  # one error message per item is sufficient

        # Search errors: non-final turns with fewer valid search calls.
        # Count only in assistant turns to avoid spurious matches from user
        # messages (e.g., invalid-action nudge text containing <search> tags).
        if num_turns > 1:
            non_answer_turns = assistant_turns[:-1]
            if solver_format in ("function_calls", "function_calls_xml"):
                valid_calls = sum(
                    count_valid_function_calls_search(t)
                    for t in non_answer_turns
                )
            elif solver_format == "search_r1":
                valid_calls = sum(
                    count_valid_search_r1_calls(t)
                    for t in non_answer_turns
                )
            elif solver_format == "call_tool":
                from drtulu.call_tool_parser import count_valid_call_tool_calls
                valid_calls = sum(
                    count_valid_call_tool_calls(t)
                    for t in non_answer_turns
                )
            elif solver_format == "webthinker":
                from webthinker.call_tool_parser import count_valid_search_queries
                valid_calls = sum(
                    count_valid_search_queries(t)
                    for t in non_answer_turns
                )
            elif solver_format == "webexplorer":
                from webexplorer.call_tool_parser import count_valid_tool_calls as count_valid_webexplorer_calls
                valid_calls = sum(
                    count_valid_webexplorer_calls(t)
                    for t in non_answer_turns
                )
            else:
                valid_calls = sum(
                    count_valid_tool_calls(t) for t in non_answer_turns
                )
            if valid_calls < (num_turns - 1):
                errors.append(
                    "Your intermediate turns should contain search queries "
                    "wrapped in <search>query</search>."
                )

        # Answer errors: valid answer tag present
        solver_result = parse_solver_response(response)
        if not solver_result.is_valid:
            errors.append(
                "Your final response must contain your answer wrapped "
                "in <answer>answer</answer>."
            )

        # Answer token limit error
        if (
            answer_token_limit is not None
            and processing_class is not None
            and solver_result.is_valid
            and solver_result.answer
        ):
            answer_tokens = _count_tokens(
                solver_result.answer, processing_class
            )
            if answer_tokens > answer_token_limit:
                errors.append(
                    f"Your <answer> exceeded the {answer_token_limit}-token "
                    f"limit ({answer_tokens} tokens). Provide a more concise "
                    f"answer."
                )

        all_errors.append(errors)
    return all_errors


def _compute_token_budget_stats(
    raw_messages: list[str],
    responses: list[str],
    processing_class: Any,
    think_token_budget: int | None = None,
    answer_token_limit: int | None = None,
) -> list[dict[str, int | float]]:
    """Compute per-item token budget statistics for WandB logging.

    Args:
        raw_messages: Decoded ``input_ids`` strings (full ChatML conversations).
        responses: Decoded response strings (policy outputs).
        processing_class: Tokenizer instance with an ``encode`` method.
        think_token_budget: Maximum allowed tokens per ``<think>`` block.
            ``None`` means no budget is enforced.
        answer_token_limit: Maximum allowed tokens for the ``<answer>``
            content. ``None`` means no limit is enforced.

    Returns:
        list[dict[str, int | float]]: Per-item dicts with keys:
            ``think_budget_violations`` (int), ``answer_over_limit`` (int),
            ``avg_think_tokens`` (float), ``answer_tokens`` (int).
    """
    stats: list[dict[str, int | float]] = []
    for raw_msg, response in zip(raw_messages, responses):
        full_text = raw_msg + response
        assistant_turns = re.findall(ASSISTANT_PATTERN, full_text, re.DOTALL)

        # Think stats
        think_token_counts: list[int] = []
        think_violations = 0
        for turn in assistant_turns:
            think_match = re.search(THINK_PATTERN, turn, re.DOTALL)
            if think_match:
                tc = _count_tokens(think_match.group(1), processing_class)
                think_token_counts.append(tc)
                if think_token_budget is not None and tc > think_token_budget:
                    think_violations += 1

        avg_think = (
            sum(think_token_counts) / len(think_token_counts)
            if think_token_counts
            else 0.0
        )

        # Answer stats
        solver_result = parse_solver_response(response)
        answer_tc = 0
        answer_over = 0
        if solver_result.is_valid and solver_result.answer:
            answer_tc = _count_tokens(solver_result.answer, processing_class)
            if (
                answer_token_limit is not None
                and answer_tc > answer_token_limit
            ):
                answer_over = 1

        stats.append({
            "think_budget_violations": think_violations,
            "answer_over_limit": answer_over,
            "avg_think_tokens": avg_think,
            "answer_tokens": answer_tc,
        })
    return stats


def compute_long_solver_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Compute reward scores for a batch of long-form solver outputs.

    Combines format compliance scoring with normalized rubric accuracy from
    an LLM grader. The solver's own multi-turn response is graded directly
    (no external solver rollouts needed, unlike the challenger reward).

    Args:
        data_sources: Data source identifiers per item (unused).
        solution_strs: Solution strings per item (unused).
        ground_truths: Ground truth dicts per item, each with keys
            ``"target"``, ``"rubrics"`` (list[str]), ``"priorities"``
            (list[str | None]).
        extra_infos: Extra info dicts per item (unused).
        **kwargs: Must contain:
            data: DataProto batch from the framework.
            processing_class: Tokenizer instance.
            model_name (str): Model identifier for grader server.
            grader_base_url (str): URL of the grader server.
            grader_template_path (str): Path to grader prompt template.
            format_weight (float): Weight for format score component.
            acc_weight (float): Weight for accuracy score component.
            search_weight (float): Weight for search reward component.
                Defaults to 0.0 (disabled) for backward compatibility.
            search_reward_max_turns (int): Maximum number of search turns
                that contribute to search reward. Defaults to 3.
            grader_min_coverage (float): Minimum rubric coverage fraction.
            grader_pad_value (float): Pad value for missing assessments.
            grader_retry (bool): Enable grader retry for v2 format.
            retrieval_url (str): URL of retrieval server endpoint.
            debug_print_first (bool): Print debug info for first item.

    Returns:
        list[dict[str, Any]]: Per-item dicts with keys ``"score"``
            (combined reward), ``"format_reward"``, ``"acc_reward"``,
            ``"search_reward"`` (normalized search usage, 0-1),
            ``"number_of_valid_search"`` (int count of valid search/tool
            calls), ``"task_prompt"``, ``"rubric_text"``,
            ``"solver_response"``, ``"grader_output"``,
            ``"extracted_answer"`` (parsed answer from solver output),
            ``"gold_answer"`` (ground truth target answer), and
            ``"format_errors"`` (list of error strings for SDPO).
    """
    import time as _time

    import numpy as np

    _reward_t0 = _time.perf_counter()

    batch = kwargs["data"]
    processing_class = kwargs["processing_class"]

    model_name = kwargs.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
    grader_base_url = kwargs.get("grader_base_url", "http://127.0.0.1:8001")
    retrieval_url = kwargs.get(
        "retrieval_url", "http://127.0.0.1:8000/retrieve"
    )
    grader_template_path = kwargs.get(
        "grader_template_path", "scope/prompts/grader_per_rubric.txt"
    )
    format_weight = kwargs.get("format_weight", 0.5)
    acc_weight = kwargs.get("acc_weight", 1.0)
    search_weight = kwargs.get("search_weight", 0.0)
    _raw_search_max = kwargs.get("search_reward_max_turns", None)
    search_reward_max_turns: int = (
        int(_raw_search_max)
        if _raw_search_max not in (None, "", "null")
        else 3
    )
    grader_min_coverage = kwargs.get("grader_min_coverage", 0.5)
    grader_pad_value = kwargs.get("grader_pad_value", 0.0)
    grader_retry = kwargs.get("grader_retry", False)
    debug_print_first = kwargs.get("debug_print_first", False)
    solver_format = kwargs.get("solver_format", "tool_call")

    # OpenAI-compatible remote grader configuration (OpenRouter, Fireworks,
    # Together, ...). All optional with back-compat defaults — leaving them
    # unset keeps the historical local-server behavior.
    grader_api_mode = kwargs.get("grader_api_mode", "auto")
    grader_api_base_url = kwargs.get("grader_api_base_url", None)
    grader_api_key_env = kwargs.get("grader_api_key_env", "OPENAI_API_KEY")
    _raw_provider_order = kwargs.get("grader_provider_order", None)
    if isinstance(_raw_provider_order, str):
        # Hydra list-quoting on the CLI is brittle; accept a plain comma-
        # separated string as fallback.
        grader_provider_order: list[str] | None = [
            s.strip() for s in _raw_provider_order.split(",") if s.strip()
        ] or None
    elif _raw_provider_order is None:
        grader_provider_order = None
    else:
        grader_provider_order = list(_raw_provider_order)
    grader_allow_fallbacks = bool(kwargs.get("grader_allow_fallbacks", True))
    grader_require_parameters = bool(
        kwargs.get("grader_require_parameters", False)
    )
    _raw_request_timeout = kwargs.get("grader_request_timeout", None)
    grader_request_timeout: float | None = (
        float(_raw_request_timeout)
        if _raw_request_timeout not in (None, "", "null")
        else None
    )
    _raw_app_max_retries = kwargs.get("grader_app_max_retries", None)
    grader_app_max_retries: int | None = (
        int(_raw_app_max_retries)
        if _raw_app_max_retries not in (None, "", "null")
        else None
    )
    openai_concurrency = int(kwargs.get("openai_concurrency", 16))

    # Token budget enforcement
    _raw_think_budget = kwargs.get("think_token_budget", None)
    think_token_budget: int | None = (
        int(_raw_think_budget)
        if _raw_think_budget not in (None, "", "null")
        else None
    )
    _raw_answer_limit = kwargs.get("answer_token_limit", None)
    answer_token_limit: int | None = (
        int(_raw_answer_limit)
        if _raw_answer_limit not in (None, "", "null")
        else None
    )

    # Soft length penalty parameters (v1.9.10+)
    _raw_soft_limit = kwargs.get("answer_soft_limit", None)
    answer_soft_limit: int | None = (
        int(_raw_soft_limit)
        if _raw_soft_limit not in (None, "", "null")
        else None
    )
    _raw_penalty_floor = kwargs.get("answer_length_penalty_floor", None)
    answer_length_penalty_floor: float = (
        float(_raw_penalty_floor)
        if _raw_penalty_floor not in (None, "", "null")
        else 0.05
    )

    # No-answer zero: when True, rollouts without a parseable <answer> tag
    # get final score=0 instead of accumulating format/search credit.
    # This prevents incentivizing search-only loops during training.
    no_answer_zero = kwargs.get("no_answer_zero", False)

    # Per-rubric grading: grade each rubric independently in its own LLM call
    grader_per_rubric = kwargs.get("grader_per_rubric", False)
    if not grader_per_rubric and "grader_per_rubric" in grader_template_path:
        grader_per_rubric = True

    # Grader input mode: controls what the grader sees as `{response}`.
    #   "answer_only"          (default): extracted <answer> block only
    #   "answer_with_evidence": retrieved <information> blocks + answer
    grader_input_mode = kwargs.get("grader_input_mode", "answer_only")

    # Copy penalty: penalize verbatim copying from <information> blocks.
    # When copy_penalty_ngram_n is None (default), the penalty is disabled.
    _raw_cp_ngram = kwargs.get("copy_penalty_ngram_n", None)
    copy_penalty_ngram_n: int | None = (
        int(_raw_cp_ngram)
        if _raw_cp_ngram not in (None, "", "null")
        else None
    )
    _raw_cp_threshold = kwargs.get("copy_penalty_threshold", None)
    copy_penalty_threshold: float = (
        float(_raw_cp_threshold)
        if _raw_cp_threshold not in (None, "", "null")
        else 0.15
    )
    _raw_cp_deduction = kwargs.get("copy_penalty_deduction", None)
    copy_penalty_deduction: float = (
        float(_raw_cp_deduction)
        if _raw_cp_deduction not in (None, "", "null")
        else 0.2
    )

    # Read grader template once
    with open(grader_template_path, "r") as f:
        grader_template = f.read()

    # Grader format. Per-rubric grading synthesizes v2-style score records.
    grader_format = kwargs.get("grader_format", "xml")
    if grader_format == "xml" and grader_template:
        if "<scores>" in grader_template:
            grader_format = "v2"
    if grader_per_rubric:
        grader_format = "v2"

    # Step 1: Decode batch
    raw_messages = [
        processing_class.decode(batch.batch["input_ids"][i])
        for i in range(len(batch))
    ]
    responses = [
        processing_class.decode(
            batch.batch["responses"][i], skip_special_tokens=True,
        )
        for i in range(len(batch))
    ]

    # Step 2: Format scoring
    format_scores, answer_over_limits = compute_long_solver_format_scores(
        raw_messages, responses, solver_format=solver_format,
        processing_class=processing_class,
        think_token_budget=think_token_budget,
        answer_token_limit=answer_token_limit,
    )
    format_errors = compute_long_solver_format_errors(
        raw_messages, responses, solver_format=solver_format,
        processing_class=processing_class,
        think_token_budget=think_token_budget,
        answer_token_limit=answer_token_limit,
    )

    # Compute token budget stats for WandB logging
    if think_token_budget is not None or answer_token_limit is not None:
        token_budget_stats = _compute_token_budget_stats(
            raw_messages, responses, processing_class,
            think_token_budget=think_token_budget,
            answer_token_limit=answer_token_limit,
        )
    else:
        token_budget_stats = [
            {
                "think_budget_violations": 0,
                "answer_over_limit": 0,
                "avg_think_tokens": 0.0,
                "answer_tokens": 0,
            }
            for _ in range(len(raw_messages))
        ]

    # Note: answer_over_limits is still tracked for logging/stats but no
    # longer used to skip grading or zero accuracy (soft penalty replaces
    # the hard cutoff as of v1.9.10).

    # Count valid search calls per item (for wandb tracking).
    # Count only in assistant turns to avoid spurious matches from user
    # messages (e.g., invalid-action nudge text containing <search> tags).
    valid_search_counts: list[int] = []
    all_assistant_turns: list[list[str]] = []
    for raw_msg, response in zip(raw_messages, responses):
        full_text = raw_msg + response
        turns = re.findall(ASSISTANT_PATTERN, full_text, re.DOTALL)
        all_assistant_turns.append(turns)
        if solver_format in ("function_calls", "function_calls_xml"):
            valid_search_counts.append(
                sum(count_valid_function_calls_search(t) for t in turns)
            )
        elif solver_format == "search_r1":
            valid_search_counts.append(
                sum(count_valid_search_r1_calls(t) for t in turns)
            )
        else:
            valid_search_counts.append(
                sum(count_valid_tool_calls(t) for t in turns)
            )

    # Compute search reward per item: min(searches, max_turns) / max_turns.
    # Rewards the solver for actually using search, capped at max_turns.
    search_rewards: list[float] = [
        min(sc, search_reward_max_turns) / search_reward_max_turns
        for sc in valid_search_counts
    ]

    # Compute ratio of valid actions (search + answer) to total assistant turns
    valid_action_ratios: list[float] = []
    for raw_msg, response, search_count, assist_turns in zip(
        raw_messages, responses, valid_search_counts, all_assistant_turns,
    ):
        num_assistant_turns = len(assist_turns)
        if num_assistant_turns == 0:
            valid_action_ratios.append(0.0)
        else:
            has_answer = 1.0 if parse_solver_response(response).is_valid else 0.0
            ratio = (search_count + has_answer) / num_assistant_turns
            valid_action_ratios.append(min(ratio, 1.0))

    # Normalize ground_truths: parse JSON strings to dicts
    for i in range(len(batch)):
        if isinstance(ground_truths[i], str):
            try:
                ground_truths[i] = json.loads(ground_truths[i])
            except (json.JSONDecodeError, TypeError):
                pass

    # Step 2.5: Exact match for items with target answers (validation data)
    accuracy_scores: list[float] = [0.0] * len(batch)
    extracted_answers: list[str] = [""] * len(batch)
    gold_answers: list[str] = [""] * len(batch)
    em_handled: set[int] = set()
    for i in range(len(batch)):
        gt = ground_truths[i]
        if gt is not None and isinstance(gt, dict):
            target = gt.get("target")
            if target is not None:
                gold_answers[i] = (
                    json.dumps(target) if isinstance(target, list)
                    else str(target)
                )
                solver_result = parse_solver_response(responses[i])
                if solver_result.is_valid and solver_result.answer:
                    extracted_answers[i] = solver_result.answer
                    accuracy_scores[i] = float(
                        em_check(solver_result.answer, target)
                    )
                em_handled.add(i)

    # Early return: if all format scores are 0, skip grading entirely
    # (but still return EM results if any items were handled by EM)
    if all(s == 0.0 for s in format_scores) and not em_handled:
        return [
            {
                "score": search_weight * search_rewards[i] if not no_answer_zero else 0.0,
                "format_reward": 0.0,
                "acc_reward": 0.0,
                "search_reward": search_rewards[i],
                "number_of_valid_search": valid_search_counts[i],
                "ratio_of_valid_action": valid_action_ratios[i],
                "task_prompt": "",
                "rubric_text": "",
                "solver_response": "",
                "grader_output": "",
                "extracted_answer": "",
                "gold_answer": "",
                "format_errors": format_errors[i],
                "length_penalty": 1.0,
                "answer_truncated": 0,
                "ngram_overlap_ratio": 0.0,
                "copy_penalty": 0.0,
                **token_budget_stats[i],
            }
            for i in range(len(batch))
        ]

    # Step 3: Extract task prompts and rubrics from ground_truths
    task_prompts: list[str] = []
    rubric_texts: list[str] = []
    num_rubrics_list: list[int] = []

    for i in range(len(batch)):
        task_prompt = _extract_task_prompt(raw_messages[i])
        task_prompts.append(task_prompt)

        gt = ground_truths[i]
        if gt is not None and isinstance(gt, dict) and gt.get("rubrics"):
            rubric_text = _format_rubric_list_from_ground_truth(gt)
            rubric_texts.append(rubric_text)
            num_rubrics_list.append(len(gt["rubrics"]))
        else:
            rubric_texts.append("")
            num_rubrics_list.append(0)

    # Step 4: Build grader prompts for items with valid format + rubrics
    grader_prompts: list[str] = []
    grader_messages_list: list[list[dict[str, str]]] = []
    grader_to_item_idx: list[int] = []
    grader_raw_contents: list[str] = []
    grader_expected_rubrics: list[int] = []
    # Per-rubric mode: flat map from grader prompt -> (item_idx, rubric_idx)
    flat_grader_map: list[tuple[int, int]] = []

    # Optional override for grader model (e.g., "gpt-5.2")
    grader_model_name = kwargs.get("grader_model_name", None)

    # Use a separate OpenAI judge for validation when specified
    is_validation = kwargs.get("is_validation", False)
    val_grader_model_name = kwargs.get("val_grader_model_name", None)
    if is_validation and val_grader_model_name:
        grader_model_name = val_grader_model_name
        if debug_print_first:
            print(f"[SolverReward] Using validation judge: {grader_model_name}")

    # Grader temperature: default 0.6 for local grader (Qwen3 thinking mode
    # requires non-greedy decoding to avoid repetition loops).
    grader_temperature = float(kwargs.get("grader_temperature", 0.6))
    val_grader_temperature = kwargs.get("val_grader_temperature", None)
    if is_validation and val_grader_temperature is not None:
        grader_temperature = float(val_grader_temperature)
        if debug_print_first:
            print(f"[SolverReward] Using validation grader temperature: {grader_temperature}")

    # Step 2.8: Benchmark-specific evaluation dispatch (validation only)
    benchmark_handled: set[int] = set()
    benchmark_extras: list[dict] = [{} for _ in range(len(batch))]
    benchmark_grader_outputs: dict[int, str] = {}
    per_rubric_raw: dict[int, dict[int, str]] = defaultdict(dict)

    with BatchMultiTurnRollout(
        solver_base_url=grader_base_url,  # not used for solving, just init
        retrieval_url=retrieval_url,
        model_name=model_name,
        processing_class=processing_class,
        openai_concurrency=openai_concurrency,
        grader_api_mode=grader_api_mode,
        grader_api_base_url=grader_api_base_url,
        grader_api_key_env=grader_api_key_env,
        grader_provider_order=grader_provider_order,
        grader_allow_fallbacks=grader_allow_fallbacks,
        grader_require_parameters=grader_require_parameters,
        grader_request_timeout=grader_request_timeout,
        grader_app_max_retries=grader_app_max_retries,
    ) as rollout:

        if is_validation:
            benchmark_items: dict[str, list[int]] = defaultdict(list)
            for i in range(len(batch)):
                ds = data_sources[i]
                if ds in BENCHMARK_EVALUATORS:
                    benchmark_items[ds].append(i)

            for ds, indices in benchmark_items.items():
                print(
                    f"[SolverReward] Evaluating {ds}: {len(indices)} items",
                    flush=True,
                )
                evaluator_fn = BENCHMARK_EVALUATORS[ds]
                ds_responses = [responses[i] for i in indices]
                ds_ground_truths = [ground_truths[i] for i in indices]

                # Extract answer from <answer> tags for each response.
                # Invalid parses get empty string (scored 0 by evaluators).
                ds_extracted = []
                for resp in ds_responses:
                    solver_result = parse_solver_response(resp)
                    ds_extracted.append(
                        solver_result.answer
                        if solver_result.is_valid and solver_result.answer
                        else ""
                    )

                eval_results = evaluator_fn(
                    ds_extracted,
                    ds_ground_truths,
                    rollout,
                    grader_model_name,
                    grader_temperature,
                )

                if eval_results:
                    ds_avg = float(np.mean([r["score"] for r in eval_results]))
                    print(
                        f"[SolverReward] {ds} done: avg={ds_avg:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[SolverReward] {ds} done: no results",
                        flush=True,
                    )

                for idx, result in zip(indices, eval_results):
                    accuracy_scores[idx] = result["score"]
                    benchmark_handled.add(idx)
                    benchmark_grader_outputs[idx] = result.get(
                        "grader_output", ""
                    )
                    benchmark_extras[idx] = {
                        k: v for k, v in result.items()
                        if k not in ("score", "grader_output")
                    }

        # Backfill non-benchmark items with None so every dict has the same keys
        all_benchmark_keys: set[str] = set()
        for extras in benchmark_extras:
            all_benchmark_keys.update(extras.keys())
        if all_benchmark_keys:
            for i in range(len(batch)):
                for key in all_benchmark_keys:
                    if key not in benchmark_extras[i]:
                        benchmark_extras[i][key] = None

        # Emergency cap: answers above 2× answer_token_limit skip grading
        # entirely and get accuracy=0 (OOM safety net).
        emergency_cap: int | None = (
            2 * answer_token_limit if answer_token_limit is not None else None
        )
        emergency_cap_set: set[int] = set()

        # Track per-item length penalty and truncation for logging
        length_penalties: list[float] = [1.0] * len(batch)
        answer_truncated: list[bool] = [False] * len(batch)

        # Pre-compute length penalties and truncation for ALL items with
        # valid answers (including EM-handled items, since the penalty
        # should apply uniformly to the accuracy component).
        answer_n_tokens_list: list[int] = [0] * len(batch)
        for i in range(len(batch)):
            if not extracted_answers[i]:
                solver_result = parse_solver_response(responses[i])
                if solver_result.is_valid and solver_result.answer:
                    extracted_answers[i] = solver_result.answer
            if extracted_answers[i]:
                answer_n_tokens_list[i] = _count_tokens(
                    extracted_answers[i], processing_class
                )
                n_tok = answer_n_tokens_list[i]

                # Emergency cap
                if emergency_cap is not None and n_tok > emergency_cap:
                    accuracy_scores[i] = 0.0
                    length_penalties[i] = 0.0
                    emergency_cap_set.add(i)
                    continue

                # Truncation flag
                if (
                    answer_token_limit is not None
                    and n_tok > answer_token_limit
                ):
                    answer_truncated[i] = True

                # Soft length penalty
                if (
                    answer_soft_limit is not None
                    and answer_token_limit is not None
                ):
                    length_penalties[i] = compute_length_penalty(
                        n_tok,
                        answer_soft_limit,
                        answer_token_limit,
                        answer_length_penalty_floor,
                    )

        for i in range(len(batch)):
            # Skip items with zero format score, no rubrics, already
            # handled by exact match, benchmark evaluation, or
            # emergency cap
            if (
                format_scores[i] == 0.0
                or num_rubrics_list[i] == 0
                or i in em_handled
                or i in benchmark_handled
                or i in emergency_cap_set
            ):
                continue

            if not extracted_answers[i]:
                continue

            # Truncate answer for grading (grader only sees first
            # answer_token_limit tokens to eliminate tail-content bias)
            grading_answer = extracted_answers[i]
            if answer_truncated[i] and answer_token_limit is not None:
                grading_answer, _ = _truncate_to_tokens(
                    extracted_answers[i], answer_token_limit, processing_class,
                )

            # Prepend retrieved evidence when grader_input_mode requests it
            if grader_input_mode == "answer_with_evidence":
                evidence = _extract_evidence_text(responses[i])
                if evidence:
                    grading_answer = (
                        f"{evidence}\n\n## Answer\n{grading_answer}"
                    )

            if grader_per_rubric:
                # Per-rubric mode: one grader call per rubric
                gt = ground_truths[i]
                rubrics = gt["rubrics"] if isinstance(gt, dict) else []
                for rubric_idx, rubric_text in enumerate(rubrics):
                    grader_content = grader_template.format(
                        prompt=task_prompts[i],
                        response=grading_answer,
                        rubric=f"{rubric_idx + 1}. {rubric_text}",
                    )
                    grader_msg = [{"role": "user", "content": grader_content}]
                    grader_raw = processing_class.apply_chat_template(
                        grader_msg, add_generation_prompt=True, tokenize=False,
                    )
                    grader_prompts.append(grader_raw)
                    grader_messages_list.append(grader_msg)
                    grader_raw_contents.append(grader_content)
                    flat_grader_map.append((i, rubric_idx))
                grader_to_item_idx.append(i)
                grader_expected_rubrics.append(num_rubrics_list[i])
            else:
                grader_content = grader_template.format(
                    prompt=task_prompts[i],
                    response=grading_answer,
                    rubric_list=rubric_texts[i],
                )
                grader_msg = [{"role": "user", "content": grader_content}]
                grader_raw = processing_class.apply_chat_template(
                    grader_msg, add_generation_prompt=True, tokenize=False,
                )
                grader_prompts.append(grader_raw)
                grader_messages_list.append(grader_msg)
                grader_to_item_idx.append(i)
                grader_raw_contents.append(grader_content)
                grader_expected_rubrics.append(num_rubrics_list[i])

        # Step 5: Batch grading
        print(
            f"[SolverReward] Grading phase: {len(grader_prompts)} prompts"
            f" from {len(grader_to_item_idx)} items (per_rubric={grader_per_rubric})",
            flush=True,
        )
        raw_grader_outputs = (
            rollout.grade_batch(
                grader_prompts,
                grader_base_url,
                grader_model_name=grader_model_name,
                grader_messages=grader_messages_list,
                temperature=grader_temperature,
            )
            if grader_prompts
            else []
        )

        if grader_per_rubric:
            # Step 5.5a: Per-rubric retry for failed individual calls
            retry_flat_indices: list[int] = []
            for fi, output in enumerate(raw_grader_outputs):
                parsed = parse_grader_response(output, output_format="per_rubric")
                if not parsed.is_valid:
                    retry_flat_indices.append(fi)

            if grader_retry and retry_flat_indices:
                print(
                    f"[SolverReward] Per-rubric retry: {len(retry_flat_indices)}"
                    f" failed out of {len(raw_grader_outputs)}, retrying...",
                    flush=True,
                )
                retry_prompts_pr: list[str] = []
                retry_messages_pr: list[list[dict[str, str]]] = []
                for fi in retry_flat_indices:
                    nudge = (
                        "Your previous response could not be parsed. "
                        "Output ONLY a <score> block with exactly one "
                        "score (0 or 1). Nothing else."
                    )
                    retry_conv = [
                        {"role": "user", "content": grader_raw_contents[fi]},
                        {"role": "assistant", "content": raw_grader_outputs[fi]},
                        {"role": "user", "content": nudge},
                    ]
                    retry_raw = processing_class.apply_chat_template(
                        retry_conv, add_generation_prompt=True, tokenize=False,
                    )
                    retry_prompts_pr.append(retry_raw)
                    retry_messages_pr.append(retry_conv)

                retry_outputs_pr = rollout.grade_batch(
                    retry_prompts_pr,
                    grader_base_url,
                    grader_model_name=grader_model_name,
                    grader_messages=retry_messages_pr,
                    temperature=grader_temperature,
                )
                n_retry_ok = 0
                for ri, fi in enumerate(retry_flat_indices):
                    raw_grader_outputs[fi] = retry_outputs_pr[ri]
                    reparsed = parse_grader_response(
                        retry_outputs_pr[ri], output_format="per_rubric"
                    )
                    if reparsed.is_valid:
                        n_retry_ok += 1

                print(
                    f"[SolverReward] Per-rubric grader retry: "
                    f"{len(retry_flat_indices)} failed, "
                    f"{n_retry_ok} recovered after retry"
                )
        else:
            # Step 5.5: Grader retry (v2 format only)
            if grader_retry and grader_format == "v2" and raw_grader_outputs:
                retry_indices: list[int] = []
                for gi, output in enumerate(raw_grader_outputs):
                    parsed = parse_grader_response(output, output_format="v2")
                    if (
                        not parsed.is_valid
                        or parsed.num_assessments != grader_expected_rubrics[gi]
                    ):
                        retry_indices.append(gi)

                if retry_indices:
                    retry_prompts_grader: list[str] = []
                    retry_messages_list_r: list[list[dict[str, str]]] = []
                    for gi in retry_indices:
                        n_rubrics = grader_expected_rubrics[gi]
                        nudge = (
                            "Your previous response could not be parsed. "
                            f"Output ONLY a <scores> block with exactly "
                            f"{n_rubrics} comma-separated scores "
                            "(each 0, 0.5, or 1). Nothing else."
                        )
                        retry_conv = [
                            {"role": "user", "content": grader_raw_contents[gi]},
                            {"role": "assistant", "content": raw_grader_outputs[gi]},
                            {"role": "user", "content": nudge},
                        ]
                        retry_raw = processing_class.apply_chat_template(
                            retry_conv,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                        retry_prompts_grader.append(retry_raw)
                        retry_messages_list_r.append(retry_conv)

                    retry_outputs = rollout.grade_batch(
                        retry_prompts_grader,
                        grader_base_url,
                        grader_model_name=grader_model_name,
                        grader_messages=retry_messages_list_r,
                        temperature=grader_temperature,
                    )
                    for ri, gi in enumerate(retry_indices):
                        raw_grader_outputs[gi] = retry_outputs[ri]

                    n_retry_succeeded = 0
                    for gi in retry_indices:
                        reparsed = parse_grader_response(
                            raw_grader_outputs[gi], output_format="v2"
                        )
                        if (
                            reparsed.is_valid
                            and reparsed.num_assessments
                            == grader_expected_rubrics[gi]
                        ):
                            n_retry_succeeded += 1

                    print(
                        f"[SolverReward] Grader retry: {len(retry_indices)} failed,"
                        f" {n_retry_succeeded} recovered after retry"
                    )

    # Step 6: Reconstruct per-item grader outputs
    grader_outputs: list[str] = [""] * len(batch)
    if grader_per_rubric:
        # Unmap flat outputs -> per-item, per-rubric scores
        per_rubric_raw: dict[int, dict[int, str]] = defaultdict(dict)
        for fi, (item_i, rubric_i) in enumerate(flat_grader_map):
            per_rubric_raw[item_i][rubric_i] = raw_grader_outputs[fi]
        # Build synthetic <scores> strings for logging
        for item_i, rubric_outputs in per_rubric_raw.items():
            n_rubrics = num_rubrics_list[item_i]
            scores_parts: list[str] = []
            for ri in range(n_rubrics):
                if ri in rubric_outputs:
                    parsed = parse_grader_response(
                        rubric_outputs[ri], output_format="per_rubric"
                    )
                    if parsed.is_valid and parsed.assessments:
                        scores_parts.append(f"{parsed.assessments[0].score:g}")
                    else:
                        scores_parts.append("?")
                else:
                    scores_parts.append("?")
            grader_outputs[item_i] = f"<scores>{', '.join(scores_parts)}</scores>"
    else:
        for gi, item_i in enumerate(grader_to_item_idx):
            grader_outputs[item_i] = raw_grader_outputs[gi]

    # Step 6.5: Populate grader_outputs for benchmark-handled items
    for i in benchmark_handled:
        grader_output_from_bench = benchmark_grader_outputs.get(i, "")
        if grader_output_from_bench:
            grader_outputs[i] = grader_output_from_bench

    # Step 7: Parse grader outputs and compute rubric accuracy
    # (accuracy_scores already initialized in Step 2.5 with EM results)
    for item_i in grader_to_item_idx:
        num_rubrics = num_rubrics_list[item_i]

        if grader_per_rubric:
            rubric_outputs = per_rubric_raw.get(item_i, {})
            if not rubric_outputs:
                continue
            per_rubric_parsed = [
                parse_grader_response(
                    rubric_outputs.get(ri, ""),
                    output_format="per_rubric",
                )
                for ri in range(num_rubrics)
            ]
            grader_result = aggregate_per_rubric_results(per_rubric_parsed)
        else:
            grader_output = grader_outputs[item_i]
            if not grader_output:
                continue
            grader_result = parse_grader_response(
                grader_output, output_format=grader_format
            )

        rubric_sum, info = compute_rubric_sum_with_recovery(
            grader_result, num_rubrics, grader_min_coverage, grader_pad_value
        )
        if rubric_sum is not None and num_rubrics > 0:
            accuracy_scores[item_i] = rubric_sum / num_rubrics
        else:
            logger.warning(
                "[SolverReward] Grading failed item %d: info=%s",
                item_i,
                info,
            )

    # Step 7.5: Apply soft length penalty to accuracy scores.
    # Applied to all items (including EM) except benchmark-evaluated items
    # (benchmarks have their own scoring and answers aren't length-gamed).
    for i in range(len(batch)):
        if i not in benchmark_handled:
            accuracy_scores[i] *= length_penalties[i]

    # Step 7.6: Copy penalty — penalize verbatim copying from <information>
    # blocks.  If the n-gram overlap between the answer and retrieved docs
    # exceeds a threshold, deduct a fixed amount from the accuracy score.
    copy_penalties: list[float] = [0.0] * len(batch)
    ngram_overlaps: list[float] = [0.0] * len(batch)
    if copy_penalty_ngram_n is not None:
        for i in range(len(batch)):
            if i in benchmark_handled or not extracted_answers[i]:
                continue
            info_blocks = re.findall(
                INFORMATION_PATTERN, responses[i], re.DOTALL,
            )
            if not info_blocks:
                continue
            overlap = _compute_ngram_overlap_ratio(
                extracted_answers[i],
                " ".join(info_blocks),
                copy_penalty_ngram_n,
            )
            ngram_overlaps[i] = overlap
            if overlap > copy_penalty_threshold:
                copy_penalties[i] = copy_penalty_deduction
                accuracy_scores[i] = max(
                    0.0, accuracy_scores[i] - copy_penalty_deduction,
                )

    # Step 8: Final combined score
    final_scores = [
        format_weight * format_scores[i]
        + acc_weight * accuracy_scores[i]
        + search_weight * search_rewards[i]
        for i in range(len(batch))
    ]
    # When no_answer_zero is enabled, rollouts without a parseable <answer>
    # tag get score=0 to avoid incentivizing search-only loops.
    if no_answer_zero:
        for i in range(len(batch)):
            if not extracted_answers[i]:
                final_scores[i] = 0.0

    if debug_print_first:
        em_scores = [accuracy_scores[i] for i in em_handled]
        benchmark_scores = [accuracy_scores[i] for i in benchmark_handled]
        rubric_indices = [
            i for i in range(len(batch))
            if i not in em_handled and i not in benchmark_handled
        ]
        rubric_scores = [accuracy_scores[i] for i in rubric_indices]
        em_avg = float(np.mean(em_scores)) if em_scores else 0.0
        rubric_avg = float(np.mean(rubric_scores)) if rubric_scores else 0.0
        benchmark_avg = float(np.mean(benchmark_scores)) if benchmark_scores else 0.0
        print(
            f"[SolverReward] Format rewards: "
            f"Avg {np.mean(format_scores):.4f}, "
            f"Max {np.max(format_scores):.4f}\n"
            f"[SolverReward] Accuracy rewards: "
            f"Avg {np.mean(accuracy_scores):.4f}, "
            f"Max {np.max(accuracy_scores):.4f}\n"
            f"[SolverReward] EM items: {len(em_handled)}, "
            f"EM avg: {em_avg:.4f}\n"
            f"[SolverReward] Rubric items: {len(rubric_indices)}, "
            f"Rubric avg: {rubric_avg:.4f}\n"
            f"[SolverReward] Benchmark items: {len(benchmark_handled)}, "
            f"Benchmark avg: {benchmark_avg:.4f}\n"
            f"[SolverReward] Search rewards: "
            f"weight={search_weight}, max_turns={search_reward_max_turns}, "
            f"Avg {np.mean(search_rewards):.4f}, "
            f"Max {np.max(search_rewards):.4f}\n"
            f"[SolverReward] Final rewards: "
            f"Avg {np.mean(final_scores):.4f}, "
            f"Max {np.max(final_scores):.4f}\n"
            f"[SolverReward] Task prompt (first): "
            f"{task_prompts[0][:200]}...\n"
            f"[SolverReward] Response (first): "
            f"{responses[0][:200]}..."
        )
        # Token budget violation summary
        if think_token_budget is not None or answer_token_limit is not None:
            total_think_violations = sum(
                s["think_budget_violations"] for s in token_budget_stats
            )
            total_answer_over = sum(
                s["answer_over_limit"] for s in token_budget_stats
            )
            avg_think_tok = float(np.mean(
                [s["avg_think_tokens"] for s in token_budget_stats]
            ))
            avg_answer_tok = float(np.mean(
                [s["answer_tokens"] for s in token_budget_stats]
            ))
            avg_penalty = float(np.mean(length_penalties))
            n_truncated = sum(answer_truncated)
            n_emergency = len(emergency_cap_set)
            print(
                f"[SolverReward] Token budgets: "
                f"think_budget={think_token_budget}, "
                f"answer_limit={answer_token_limit}, "
                f"soft_limit={answer_soft_limit}\n"
                f"[SolverReward] Think violations: {total_think_violations}, "
                f"Answer over limit: {total_answer_over}\n"
                f"[SolverReward] Avg think tokens: {avg_think_tok:.1f}, "
                f"Avg answer tokens: {avg_answer_tok:.1f}\n"
                f"[SolverReward] Length penalty: avg={avg_penalty:.4f}, "
                f"truncated={n_truncated}, emergency_cap={n_emergency}"
            )
        # Copy penalty summary
        if copy_penalty_ngram_n is not None:
            n_penalized = sum(1 for p in copy_penalties if p > 0)
            avg_overlap = float(np.mean(ngram_overlaps))
            max_overlap = float(np.max(ngram_overlaps))
            print(
                f"[SolverReward] Copy penalty: "
                f"ngram={copy_penalty_ngram_n}, "
                f"threshold={copy_penalty_threshold}, "
                f"deduction={copy_penalty_deduction}\n"
                f"[SolverReward] Overlap: "
                f"avg={avg_overlap:.4f}, "
                f"max={max_overlap:.4f}, "
                f"penalized={n_penalized}/{len(batch)}"
            )

    if is_validation and debug_print_first:
        _reward_elapsed = _time.perf_counter() - _reward_t0
        print(
            f"[SolverReward] Total reward computation: {_reward_elapsed:.1f}s",
            flush=True,
        )

    return [
        {
            "score": final_scores[i],
            "format_reward": format_scores[i],
            "acc_reward": accuracy_scores[i],
            "search_reward": search_rewards[i],
            "number_of_valid_search": valid_search_counts[i],
            "ratio_of_valid_action": valid_action_ratios[i],
            "task_prompt": task_prompts[i],
            "rubric_text": rubric_texts[i],
            "solver_response": responses[i],
            "grader_output": grader_outputs[i],
            "extracted_answer": extracted_answers[i],
            "gold_answer": gold_answers[i],
            "format_errors": format_errors[i],
            "length_penalty": length_penalties[i],
            "answer_truncated": int(answer_truncated[i]),
            "ngram_overlap_ratio": ngram_overlaps[i],
            "copy_penalty": copy_penalties[i],
            **token_budget_stats[i],
            **benchmark_extras[i],
        }
        for i in range(len(batch))
    ]
