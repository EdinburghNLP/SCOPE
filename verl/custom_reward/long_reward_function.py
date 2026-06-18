"""Reward function for long-form challenger training.

Scores challenger outputs based on format compliance (think/tool/structure)
and task difficulty measured via solver rollouts + grader rubric evaluation.

Usage::

    # Referenced in training config:
    custom_reward_function.name=compute_long_challenger_score_batch
    custom_reward_function.path=verl/custom_reward/long_reward_function.py

Note:
    Heavy dependencies (torch, sglang, verl framework modules) are imported
    lazily inside ``compute_long_challenger_score_batch`` so that lightweight
    helper functions can be imported and tested without GPU libraries.
"""

import re
import json
import logging
import os
import string
import asyncio
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

from scope.utils.parse_challenger import validate_final_turn, extract_task
from scope.utils.parse_grader import (
    aggregate_per_rubric_results,
    parse_grader_response,
)
from scope.utils.parse_solver import parse_solver_response

# Patterns inlined to avoid importing verl.prompts (which may pull in heavy deps)
ASSISTANT_PATTERN = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
THINK_PATTERN = r"^\s*<think>(.*?)</think>"
TOOL_CALL_PATTERN = r"<tool_call>(.*?)</tool_call>"
SEARCH_R1_PATTERN = r"<search>(.*?)</search>"
INFORMATION_PATTERN = r"<information>(.*?)</information>"

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SOLVER_PROMPT_MAX_LENGTH = 16384


@dataclass
class FormatScoreComponents:
    """Per-item breakdown of format compliance scoring components.

    Attributes:
        format_score: float, the averaged format score in [0, 1].
        think_reward: float, fraction of assistant turns with <think> tags.
        tool_reward: float, ratio of valid tool calls to expected count.
        structure_reward: float, 1.0 if final turn has valid XML task structure.
    """

    format_score: float
    think_reward: float
    tool_reward: float
    structure_reward: float


def normalize_answer(s: str) -> str:
    """Normalize answer string for exact match comparison.

    Lowercases, strips punctuation, removes articles (a/an/the), and
    collapses whitespace.

    Args:
        s: Raw answer string.

    Returns:
        str: Normalized answer.
    """
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_short_form_answer_reward(
    response: str,
    source_document: str,
    full_conversation: str = "",
) -> float:
    """Compute answer quality reward for short_form_qa challenger outputs.

    Extracts the last ``<answer>`` tag, checks grounding against the last
    retrieved ``<information>`` block (matching the prompt instruction that
    the answer must come from the last search result), and applies length
    penalty.

    Args:
        response: The challenger's full response text.
        source_document: Unused, kept for API compatibility.
        full_conversation: The full multi-turn conversation text (prompt +
            response). Used to extract the last ``<information>`` block for
            grounding.

    Returns:
        float: Answer reward in [0.0, 1.0].
    """
    ans_matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not ans_matches:
        return 0.0
    ans = ans_matches[-1].strip()
    if not ans:
        return 0.0

    # Normalize answer (lowercase + strip punctuation + collapse whitespace)
    norm_ans = ans.lower()
    norm_ans = "".join(ch for ch in norm_ans if ch not in string.punctuation)
    norm_ans = " ".join(norm_ans.split())

    # Grounding check: answer must appear in the last retrieved document
    # (matching prompt: "answer from the LAST search result")
    grounding_text = ""
    if full_conversation:
        info_blocks = re.findall(
            r"<information>(.*?)</information>", full_conversation, re.DOTALL,
        )
        if info_blocks:
            grounding_text = info_blocks[-1]

    if not grounding_text:
        # No retrieved docs available — cannot verify grounding
        return 0.0

    norm_doc = grounding_text.lower()
    norm_doc = "".join(ch for ch in norm_doc if ch not in string.punctuation)
    norm_doc = " ".join(norm_doc.split())
    if norm_ans not in ("yes", "no") and norm_ans not in norm_doc:
        return 0.0

    # Length penalty (count normalized words after punctuation removal)
    word_count = len(norm_ans.split())
    if 1 <= word_count <= 5:
        return 1.0
    elif 6 <= word_count <= 10:
        return 0.5
    else:
        return 0.0


def count_valid_tool_calls(response):
    """Count valid search tool calls in a response string.

    A valid tool call is a JSON object inside ``<tool_call>`` tags with
    ``name='search'`` and ``arguments.query_list`` as a non-empty list of
    strings.

    Args:
        response: str, the response text to scan for tool calls.

    Returns:
        int: Number of valid tool calls found.
    """
    valid_count = 0
    tool_matches = re.findall(TOOL_CALL_PATTERN, response, re.DOTALL)
    for match in tool_matches:
        try:
            parsed = json.loads(match.strip())
            if (
                isinstance(parsed, dict)
                and parsed.get("name") == "search"
                and isinstance(parsed.get("arguments"), dict)
                and isinstance(parsed["arguments"].get("query_list"), list)
                and len(parsed["arguments"]["query_list"]) > 0
                and all(isinstance(q, str) for q in parsed["arguments"]["query_list"])
            ):
                valid_count += 1
        except Exception:
            continue
    return valid_count


def count_valid_search_r1_calls(response):
    """Count valid search queries in ``<search>`` tags in a response string.

    A valid search query is a non-empty string inside ``<search>...</search>``
    tags. Unlike ``count_valid_tool_calls``, no JSON parsing is needed.

    Args:
        response: str, the response text to scan for search tags.

    Returns:
        int: Number of valid search queries found.
    """
    matches = re.findall(SEARCH_R1_PATTERN, response, re.DOTALL)
    return sum(1 for m in matches if m.strip())


FUNCTION_CALLS_SEARCH_PATTERN = (
    r'<function_calls>\s*search\s*\(\s*query\s*=\s*["\'](.+?)["\']\s*\)'
)
FUNCTION_CALLS_TASK_PATTERN = (
    r'<function_calls>\s*task\s*\('
)
FUNCTION_CALLS_ANSWER_PATTERN = (
    r'<function_calls>\s*answer\s*\(\s*answer\s*=\s*["\'](.+?)["\']\s*\)'
)


def count_valid_function_calls_search(response):
    """Count valid search function calls in ``<function_calls>search(query=...)`` format.

    Args:
        response: str, the response text to scan for function_calls search patterns.

    Returns:
        int: Number of valid search function calls found.
    """
    matches = re.findall(FUNCTION_CALLS_SEARCH_PATTERN, response, re.DOTALL)
    return sum(1 for m in matches if m.strip())


def format_rubric_list(extracted):
    """Format rubrics from an ExtractedTask as a numbered list.

    Args:
        extracted: ExtractedTask with rubrics list.

    Returns:
        str: Numbered rubric list, e.g. "1. rubric1\\n2. rubric2".
    """
    return "\n".join(f"{i + 1}. {r}" for i, r in enumerate(extracted.rubrics))


def compute_continuous_difficulty(avg_rubric_sum, n):
    """Compute continuous difficulty score from average rubric sum.

    Higher difficulty means solvers score lower on rubrics, indicating the
    challenger created a harder task. Returns 0 for trivial edge cases.

    Args:
        avg_rubric_sum: float, average rubric score sum across K solver rollouts.
        n: int, number of rubrics (max possible score per rollout).

    Returns:
        float: Difficulty in [0, 1]. 0 if task is trivially easy/hard or n <= 1.
    """
    if n <= 1:
        return 0.0
    if avg_rubric_sum <= 0 or avg_rubric_sum >= n:
        return 0.0
    return min((n - avg_rubric_sum) / (n - 1), 1.0)


def compute_tent_difficulty(avg_rubric_sum, n, target=0.5):
    """Compute tent (triangular) difficulty score peaking at a target normalized score.

    Unlike ``compute_continuous_difficulty`` which monotonically rewards harder
    tasks, this function peaks at a configurable target and penalizes both
    trivially easy (score near 1) and impossibly hard (score near 0) tasks.

    Math:
        ``normalized = avg_rubric_sum / n``
        - ``normalized in [0, target]``: ``reward = normalized / target``
        - ``normalized in [target, 1]``: ``reward = (1 - normalized) / (1 - target)``

    Args:
        avg_rubric_sum: float, average rubric score sum across K solver rollouts.
        n: int, number of rubrics (max possible score per rollout).
        target: float, target normalized score where reward peaks at 1.0.
            Must be in (0, 1) exclusive.

    Returns:
        float: Difficulty in [0, 1]. 0 at boundaries and for invalid inputs.
    """
    if n <= 0:
        return 0.0
    if target <= 0.0 or target >= 1.0:
        return 0.0
    if avg_rubric_sum <= 0 or avg_rubric_sum >= n:
        return 0.0
    normalized = avg_rubric_sum / n
    if normalized <= target:
        return normalized / target
    else:
        return (1.0 - normalized) / (1.0 - target)


def compute_difficulty(avg_rubric_sum, n, difficulty_fn="continuous", difficulty_target=0.5):
    """Dispatch to the appropriate difficulty scoring function.

    Args:
        avg_rubric_sum: float, average rubric score sum across K solver rollouts.
        n: int, number of rubrics (max possible score per rollout).
        difficulty_fn: str, difficulty function name. ``"continuous"`` for
            monotone decreasing, ``"tent"`` for triangular peaking at target,
            or ``"normalized"`` for raw normalized rubric score
            (``avg_rubric_sum / n``).
        difficulty_target: float, target normalized score for tent function.
            Ignored when ``difficulty_fn="continuous"`` or ``"normalized"``.

    Returns:
        float: Difficulty score in [0, 1].

    Raises:
        ValueError: If ``difficulty_fn`` is not ``"continuous"``, ``"tent"``,
            or ``"normalized"``.
    """
    if difficulty_fn == "continuous":
        return compute_continuous_difficulty(avg_rubric_sum, n)
    elif difficulty_fn == "tent":
        return compute_tent_difficulty(avg_rubric_sum, n, target=difficulty_target)
    elif difficulty_fn == "normalized":
        if n <= 0:
            return 0.0
        return max(0.0, min(avg_rubric_sum / n, 1.0))
    else:
        raise ValueError(
            f"Unknown difficulty_fn={difficulty_fn!r}. "
            f"Expected 'continuous', 'tent', or 'normalized'."
        )


def compute_rubric_sum_with_recovery(grader_result, num_rubrics, min_coverage=0.5, pad_value=0.0):
    """Compute rubric score sum with recovery for mismatched assessment counts.

    Handles cases where the grader returns fewer or more assessments than
    the expected number of rubrics by padding or truncating.

    Args:
        grader_result: GraderParseResult from parse_grader_response().
        num_rubrics: int, expected number of rubric assessments.
        min_coverage: float, minimum fraction of rubrics that must be assessed
            for the result to be usable (default 0.5).
        pad_value: float, value to pad missing assessments with (default 0.0).

    Returns:
        tuple[float | None, dict]: (rubric_sum, recovery_info). rubric_sum is
            None if the result should be discarded. recovery_info contains
            metadata about any recovery applied.
    """
    if not grader_result.assessments:
        return None, {"reason": "empty"}

    if num_rubrics == 0:
        return 0.0, {}

    actual = len(grader_result.assessments)
    scores = [a.score for a in grader_result.assessments]

    if actual == num_rubrics:
        return sum(scores), {"recovery": False}
    elif actual < num_rubrics:
        coverage = actual / num_rubrics
        if coverage >= min_coverage:
            padded_sum = sum(scores) + (num_rubrics - actual) * pad_value
            return padded_sum, {
                "recovery": True,
                "padded": num_rubrics - actual,
                "coverage": coverage,
            }
        else:
            return None, {"reason": "insufficient_coverage", "coverage": coverage}
    else:
        truncated_sum = sum(scores[:num_rubrics])
        return truncated_sum, {
            "recovery": True,
            "truncated": actual - num_rubrics,
        }


def extract_documents_from_conversation(raw_message: str) -> str:
    """Extract source document and all tool response contents from a conversation.

    Combines the ``## Source Document`` section from the initial prompt with
    all ``<tool_response>`` contents (retrieved search results) from the
    multi-turn conversation. Used to provide full context for rubric
    generation in the v5 pipeline.

    Args:
        raw_message: str, the full decoded conversation with chat markers.

    Returns:
        str: Combined document text (source document + tool responses).
    """
    parts: list[str] = []

    # Extract source document section from user prompt
    source_doc_match = re.search(
        r"## Source Document\s*\n(.*?)(?=\n## |\Z)", raw_message, re.DOTALL
    )
    if source_doc_match:
        parts.append(source_doc_match.group(1).strip())

    # Extract all tool_response contents
    tool_responses = re.findall(
        r"<tool_response>(.*?)</tool_response>", raw_message, re.DOTALL
    )
    for tr in tool_responses:
        stripped = tr.strip()
        if stripped:
            parts.append(stripped)

    # Extract all <information> contents (search_r1 format)
    info_responses = re.findall(INFORMATION_PATTERN, raw_message, re.DOTALL)
    for ir in info_responses:
        stripped = ir.strip()
        if stripped:
            parts.append(stripped)

    return "\n\n---\n\n".join(parts)


def extract_per_search_turn_docs(raw_message: str) -> str:
    """Extract per-search-turn documents from a conversation.

    Supports both legacy search_r1 format (``<search>`` / ``<information>``)
    and native tool_call format (``<tool_call>`` / ``<tool_response>``).

    Pairs each query with its corresponding result block and formats them as
    numbered search turns. Used by the retrieval quality gate.

    Args:
        raw_message: The full decoded conversation with chat markers.

    Returns:
        Formatted string with each search turn's query and retrieved docs,
        or empty string if no search turns found.
    """
    # --- Legacy search_r1 format ---
    queries = re.findall(SEARCH_R1_PATTERN, raw_message, re.DOTALL)
    if queries:
        info_blocks = re.findall(INFORMATION_PATTERN, raw_message, re.DOTALL)
        parts: list[str] = []
        for i, query in enumerate(queries):
            docs = info_blocks[i].strip() if i < len(info_blocks) else "(no results)"
            parts.append(f"[Search turn {i + 1}: \"{query.strip()}\"]\n{docs}")
        return "\n\n".join(parts)

    # --- Native tool_call format ---
    import json as _json
    tool_call_blocks = re.findall(TOOL_CALL_PATTERN, raw_message, re.DOTALL)
    tool_response_blocks = re.findall(r"<tool_response>(.*?)</tool_response>", raw_message, re.DOTALL)

    if not tool_call_blocks:
        return ""

    parts = []
    resp_idx = 0
    for tc_raw in tool_call_blocks:
        query_str = "(unknown query)"
        try:
            tc = _json.loads(tc_raw.strip())
            args = tc.get("arguments", {})
            ql = args.get("query_list", [])
            if isinstance(ql, list):
                # Normalize: items may be strings or dicts
                strs = []
                for item in ql:
                    if isinstance(item, str):
                        strs.append(item)
                    elif isinstance(item, dict):
                        for k in ("query", "text", "content", "q"):
                            if k in item:
                                strs.append(str(item[k]))
                                break
                query_str = "; ".join(strs) if strs else "(empty query)"
            elif isinstance(ql, str):
                query_str = ql
        except Exception:
            pass

        if resp_idx < len(tool_response_blocks):
            resp_raw = tool_response_blocks[resp_idx].strip()
            resp_idx += 1
            # Try to extract the "result" field from the JSON response
            try:
                resp_obj = _json.loads(resp_raw)
                docs = resp_obj.get("result", resp_raw)
            except Exception:
                docs = resp_raw
        else:
            docs = "(no results)"

        parts.append(f"[Search turn {len(parts) + 1}: \"{query_str}\"]\n{docs}")

    return "\n\n".join(parts)


def extract_documents_per_turn_from_conversation(
    raw_message: str,
) -> tuple[str, list[dict[str, str]]]:
    """Extract source document and per-turn search results as structured data.

    Unlike :func:`extract_per_search_turn_docs` which returns a formatted
    string, this function returns structured ``(source_doc, turns)`` for
    use by the V19 per-turn rubric generation pipeline.

    Supports both ``search_r1`` format (``<search>`` / ``<information>``)
    and native ``tool_call`` format (``<tool_call>`` / ``<tool_response>``).

    Args:
        raw_message: The full decoded conversation with chat markers.

    Returns:
        tuple[str, list[dict[str, str]]]: A tuple of
            ``(source_doc, turns)`` where ``source_doc`` is the text of
            the source document section and ``turns`` is a list of dicts
            with ``"query"`` and ``"docs"`` keys.
    """
    source_doc = ""
    turns: list[dict[str, str]] = []

    # Extract source document section from user prompt
    source_doc_match = re.search(
        r"## Source Document\s*\n(.*?)(?=\n## |\Z)", raw_message, re.DOTALL
    )
    if source_doc_match:
        source_doc = source_doc_match.group(1).strip()

    # --- Legacy search_r1 format ---
    queries = re.findall(SEARCH_R1_PATTERN, raw_message, re.DOTALL)
    if queries:
        info_blocks = re.findall(INFORMATION_PATTERN, raw_message, re.DOTALL)
        for i, query in enumerate(queries):
            docs = info_blocks[i].strip() if i < len(info_blocks) else ""
            turns.append({"query": query.strip(), "docs": docs})
        return source_doc, turns

    # --- Native tool_call format ---
    import json as _json

    tool_call_blocks = re.findall(TOOL_CALL_PATTERN, raw_message, re.DOTALL)
    tool_response_blocks = re.findall(
        r"<tool_response>(.*?)</tool_response>", raw_message, re.DOTALL
    )

    resp_idx = 0
    for tc_raw in tool_call_blocks:
        query_str = ""
        try:
            tc = _json.loads(tc_raw.strip())
            args = tc.get("arguments", {})
            ql = args.get("query_list", [])
            if isinstance(ql, list):
                strs = []
                for item in ql:
                    if isinstance(item, str):
                        strs.append(item)
                    elif isinstance(item, dict):
                        for k in ("query", "text", "content", "q"):
                            if k in item:
                                strs.append(str(item[k]))
                                break
                query_str = "; ".join(strs) if strs else ""
            elif isinstance(ql, str):
                query_str = ql
        except Exception:
            pass

        if resp_idx < len(tool_response_blocks):
            resp_raw = tool_response_blocks[resp_idx].strip()
            resp_idx += 1
            try:
                resp_obj = _json.loads(resp_raw)
                docs = resp_obj.get("result", resp_raw)
            except Exception:
                docs = resp_raw
        else:
            docs = ""

        turns.append({"query": query_str, "docs": docs if isinstance(docs, str) else str(docs)})

    return source_doc, turns


def dedup_docs_across_turns(
    turns: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Remove duplicate documents across search turns.

    Splits each turn's doc block into individual documents by the
    ``Doc N (Title: ...)`` pattern, fingerprints each by its first 200
    chars of body text, and removes docs already seen in a previous turn.
    Turns where all docs are duplicates get empty ``docs`` values.

    If the ``Doc N (Title:`` pattern is not found, falls back to splitting
    on double-newlines and fingerprinting the full chunk.

    Args:
        turns: List of dicts with ``"query"`` and ``"docs"`` keys.

    Returns:
        list[dict[str, str]]: Same structure with duplicate docs removed.
    """
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    doc_header_pattern = r"(?=Doc \d+ \(Title:)"

    for turn in turns:
        raw = turn["docs"]
        if not raw.strip():
            result.append({"query": turn["query"], "docs": ""})
            continue

        # Try splitting by Doc N (Title: ...) pattern
        chunks = re.split(doc_header_pattern, raw)
        if len(chunks) <= 1:
            # Fallback: split on double-newlines
            chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]

        kept: list[str] = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            # Extract body text (remove Doc N header if present)
            body = re.sub(r'Doc \d+ \(Title: "[^"]*"\)\s*', "", chunk).strip()
            fingerprint = body[:200]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            kept.append(chunk)

        result.append({"query": turn["query"], "docs": "\n".join(kept)})

    return result


def build_perturn_prompts(
    task_prompt: str,
    task_type: str,
    source_doc: str,
    turns: list[dict[str, str]],
    all_docs: str,
    initial_template: str,
    turn_template: str,
    synthesis_template: str,
) -> list[dict[str, str]]:
    """Build all rubric prompts for one task (V19 per-turn system).

    All prompts are independent (noprev variant) and can be batched in
    a single ``llm.generate()`` call. Turns with empty ``docs`` (all
    duplicates removed) are skipped.

    Args:
        task_prompt: The task question text.
        task_type: Task type string (e.g. ``"long_form_qa"``).
        source_doc: Source document text.
        turns: List of dicts with ``"query"`` and ``"docs"`` keys
            (after doc dedup).
        all_docs: All documents combined (for synthesis prompt).
        initial_template: Template for the source document stage.
        turn_template: Template for each search turn stage.
        synthesis_template: Template for the synthesis stage.

    Returns:
        list[dict[str, str]]: List of dicts with ``"stage"`` (label like
            ``"initial"``, ``"turn_1"``, ``"synthesis"``) and ``"prompt"``
            (filled template string) keys.
    """
    prompts: list[dict[str, str]] = []

    # Initial: source document
    prompts.append({
        "stage": "initial",
        "prompt": initial_template
            .replace("{task_type}", task_type)
            .replace("{task_prompt}", task_prompt)
            .replace("{documents}", source_doc),
    })

    # Per-turn (only turns with non-empty docs after dedup)
    for i, turn in enumerate(turns):
        if not turn["docs"].strip():
            continue  # all docs were duplicates — skip
        prompts.append({
            "stage": f"turn_{i + 1}",
            "prompt": turn_template
                .replace("{task_type}", task_type)
                .replace("{task_prompt}", task_prompt)
                .replace("{documents}", turn["docs"])
                .replace("{search_query}", turn["query"]),
        })

    # Synthesis (sees ALL docs)
    prompts.append({
        "stage": "synthesis",
        "prompt": synthesis_template
            .replace("{task_type}", task_type)
            .replace("{task_prompt}", task_prompt)
            .replace("{documents}", all_docs),
    })

    return prompts


def dedup_rubrics(
    rubric_texts: list[str],
    rubric_priorities: list[str | None],
    threshold: float = 0.6,
) -> tuple[list[str], list[str | None]]:
    """Remove duplicate rubrics based on word overlap.

    Iterates rubrics in order (initial first, synthesis last). For each
    rubric, computes word overlap with all previously kept rubrics. If
    overlap exceeds the threshold, the rubric is dropped.

    ``overlap = |words_A & words_B| / min(|words_A|, |words_B|)``

    Using ``min()`` means: if the shorter rubric shares 60%+ of its words
    with a longer one, it's considered a duplicate.

    Args:
        rubric_texts: List of rubric text strings.
        rubric_priorities: Parallel list of priority values per rubric.
        threshold: Word overlap threshold for duplicate detection.

    Returns:
        tuple[list[str], list[str | None]]: Filtered ``(texts, priorities)``
            with duplicates removed.
    """
    kept_texts: list[str] = []
    kept_priorities: list[str | None] = []

    for text, priority in zip(rubric_texts, rubric_priorities):
        words = set(text.lower().split())
        is_dup = False
        for prev_text in kept_texts:
            prev_words = set(prev_text.lower().split())
            if not words or not prev_words:
                continue
            overlap = len(words & prev_words) / min(len(words), len(prev_words))
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            kept_texts.append(text)
            kept_priorities.append(priority)

    return kept_texts, kept_priorities


def _check_submit_task_tool_call(content: str) -> float:
    """Check if content contains a valid submit_task tool call.

    Args:
        content: Assistant turn content to check.

    Returns:
        1.0 if a valid submit_task tool call with non-empty task_prompt
        is found, 0.0 otherwise.
    """
    from scope.utils.parse_challenger import _extract_submit_task_tool_call
    return 1.0 if _extract_submit_task_tool_call(content) is not None else 0.0


def compute_long_challenger_format_scores_detailed(
    raw_messages, responses, num_search_turns=None, require_rubrics=True,
    challenger_format="tool_call", original_messages=None,
):
    """Compute format compliance scores with per-component breakdown.

    Same logic as ``compute_long_challenger_format_scores`` but returns
    ``FormatScoreComponents`` objects with individual component values
    for diagnostic analysis.

    Three components averaged equally:
    1. Think reward: fraction of assistant turns containing <think> tags.
    2. Tool reward: ratio of valid tool calls to expected count.
    3. Structure reward: 1.0 if final turn has valid XML task structure.

    When ``num_search_turns`` is not None (v5 mode), tool reward uses
    per-turn validation against the exact expected search turn count, and
    structure reward skips rubric validation.

    Args:
        raw_messages: list[str], decoded full conversations with chat markers.
        responses: list[str], decoded response texts (skip_special_tokens=True).
        num_search_turns: int | list[int | None] | None, exact number of
            mandatory search turns for v5 mode. Can be a scalar (applied to
            all items), a per-item list, or None (v4 logic).
        require_rubrics: bool, whether to require rubrics in structure
            validation. Defaults to True (v4 behavior).
        challenger_format: str, format style for tool call validation.
            ``"tool_call"`` (default) uses ``<tool_call>`` JSON format,
            ``"search_r1"`` uses ``<search>query</search>`` format.
        original_messages: list[list[dict | Message]] | None, original
            conversation messages per item with ``<think>`` blocks intact.
            Used for think_reward when the chat template strips thinking
            from previous turns (e.g. thinking-hidden mode). Falls back
            to regex on decoded ``raw_messages`` when None.

    Returns:
        list[FormatScoreComponents]: Per-item format score breakdowns.
    """
    results = []
    for idx, (raw_msg, response) in enumerate(zip(raw_messages, responses)):
        # Resolve per-item search turns
        if isinstance(num_search_turns, list):
            item_turns = num_search_turns[idx]
        else:
            item_turns = num_search_turns

        # Think reward: fraction of assistant turns with <think> tags
        assistant_turns = re.findall(ASSISTANT_PATTERN, raw_msg, re.DOTALL)
        think_reward = 0.0
        if assistant_turns:
            if original_messages is not None and idx < len(original_messages):
                # Use original messages which preserve <think> blocks that
                # the thinking-hidden chat template strips from input_ids.
                orig_msgs = original_messages[idx]
                orig_assistant_contents = [
                    m["content"] if isinstance(m, dict) else m.content
                    for m in orig_msgs
                    if (m.get("role") if isinstance(m, dict) else m.role) == "assistant"
                ]
                if orig_assistant_contents:
                    think_count = sum(
                        1 for content in orig_assistant_contents
                        if re.search(r"<think>", content)
                    )
                    think_reward = think_count / len(orig_assistant_contents)
            else:
                # Fallback: use decoded raw_msg (non-thinking-hidden templates)
                think_count = sum(
                    1 for turn in assistant_turns
                    if re.match(THINK_PATTERN, turn.strip(), re.DOTALL)
                )
                think_reward = think_count / len(assistant_turns)

        if item_turns is not None:
            # v5 mode: per-turn tool call + response validation
            if challenger_format == "search_r1":
                tool_reward = _compute_search_r1_tool_reward(
                    raw_msg, assistant_turns, item_turns
                )
            elif challenger_format in ("function_calls", "function_calls_xml"):
                tool_reward = _compute_function_calls_tool_reward(
                    raw_msg, assistant_turns, item_turns
                )
            else:
                tool_reward = _compute_v5_tool_reward(
                    raw_msg, assistant_turns, item_turns
                )
            # Structure reward: validate the task turn.
            # Use the LAST assistant turn for all formats — invalid-action
            # nudges and error recovery can insert extra assistant turns
            # that shift fixed indices.
            if assistant_turns:
                task_turn_content = assistant_turns[-1]
                validation = validate_final_turn(
                    task_turn_content, require_rubrics=require_rubrics,
                    challenger_format=challenger_format,
                )
                structure_reward = 1.0 if validation.is_valid else 0.0
                # Fallback: accept submit_task tool call (hermes format)
                if not validation.is_valid and challenger_format == "tool_call":
                    structure_reward = _check_submit_task_tool_call(
                        task_turn_content
                    )
            else:
                structure_reward = 0.0
        else:
            # v4 mode: original logic
            if challenger_format == "search_r1":
                total_tool_calls = count_valid_search_r1_calls(response)
            elif challenger_format in ("function_calls", "function_calls_xml"):
                total_tool_calls = count_valid_function_calls_search(response)
            else:
                total_tool_calls = count_valid_tool_calls(response)
            expected = max(len(assistant_turns) - 1, 0)
            if expected == 0 and total_tool_calls == 0:
                tool_reward = 1.0
            elif expected == 0:
                tool_reward = 0.0
            else:
                tool_reward = min(total_tool_calls / expected, 1.0)

            # Structure reward: valid final turn
            validation = validate_final_turn(
                response, require_rubrics=require_rubrics,
                challenger_format=challenger_format,
            )
            structure_reward = 1.0 if validation.is_valid else 0.0
            # Fallback: accept submit_task tool call (hermes format)
            if not validation.is_valid and challenger_format == "tool_call":
                structure_reward = _check_submit_task_tool_call(response)

        format_score = (think_reward + tool_reward + structure_reward) / 3.0
        results.append(FormatScoreComponents(
            format_score=format_score,
            think_reward=think_reward,
            tool_reward=tool_reward,
            structure_reward=structure_reward,
        ))

    return results


def compute_long_challenger_format_scores(
    raw_messages, responses, num_search_turns=None, require_rubrics=True,
    challenger_format="tool_call", original_messages=None,
):
    """Compute format compliance scores for long challenger outputs.

    Thin wrapper around ``compute_long_challenger_format_scores_detailed``
    that returns only the aggregate format scores for backward compatibility.

    Three components averaged equally:
    1. Think reward: fraction of assistant turns containing <think> tags.
    2. Tool reward: ratio of valid tool calls to expected count.
    3. Structure reward: 1.0 if final turn has valid XML task structure.

    Args:
        raw_messages: list[str], decoded full conversations with chat markers.
        responses: list[str], decoded response texts (skip_special_tokens=True).
        num_search_turns: int | list[int | None] | None, exact number of
            mandatory search turns for v5 mode. Can be a scalar (applied to
            all items), a per-item list, or None (v4 logic).
        require_rubrics: bool, whether to require rubrics in structure
            validation. Defaults to True (v4 behavior).
        challenger_format: str, format style for tool call validation.
            ``"tool_call"`` (default) or ``"search_r1"``.
        original_messages: list[list[dict | Message]] | None, original
            conversation messages per item. See
            ``compute_long_challenger_format_scores_detailed`` for details.

    Returns:
        list[float]: Format scores in [0, 1] for each item.
    """
    detailed = compute_long_challenger_format_scores_detailed(
        raw_messages, responses, num_search_turns=num_search_turns,
        require_rubrics=require_rubrics, challenger_format=challenger_format,
        original_messages=original_messages,
    )
    return [c.format_score for c in detailed]


# Pattern to extract user turns (tool responses are wrapped in user turns)
_USER_TURN_PATTERN = re.compile(
    r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL
)
_TOOL_RESPONSE_PATTERN = re.compile(
    r"<tool_response>(.*?)</tool_response>", re.DOTALL
)


def _compute_v5_tool_reward(raw_msg, assistant_turns, num_search_turns):
    """Compute v5 tool reward based on exact search turn count.

    For each of the first ``num_search_turns`` assistant turns, checks:
    1. The assistant turn has a valid ``<tool_call>`` (valid JSON, name=search,
       non-empty query_list).
    2. A subsequent tool response exists in the conversation.

    Args:
        raw_msg: str, full decoded conversation with chat markers.
        assistant_turns: list[str], extracted assistant turn contents.
        num_search_turns: int, exact number of mandatory search turns.

    Returns:
        float: Fraction of valid search turns (0.0 to 1.0).
    """
    if num_search_turns == 0:
        return 1.0

    # Extract user turns to find tool responses
    user_turns = _USER_TURN_PATTERN.findall(raw_msg)

    valid_count = 0
    for turn_idx in range(min(num_search_turns, len(assistant_turns))):
        turn_content = assistant_turns[turn_idx]

        # Check for valid tool call in this assistant turn
        tool_call_match = re.search(TOOL_CALL_PATTERN, turn_content, re.DOTALL)
        if not tool_call_match:
            continue

        try:
            parsed = json.loads(tool_call_match.group(1).strip())
            if not (
                isinstance(parsed, dict)
                and parsed.get("name") == "search"
                and isinstance(parsed.get("arguments"), dict)
                and isinstance(parsed["arguments"].get("query_list"), list)
                and len(parsed["arguments"]["query_list"]) > 0
            ):
                continue
        except Exception:
            continue

        # Check that a subsequent tool response exists
        # User turns after the first correspond to tool responses
        # user_turn[0] is the initial prompt, user_turn[turn_idx+1] should
        # contain the tool response for assistant_turn[turn_idx]
        response_idx = turn_idx + 1  # offset by initial user prompt
        if response_idx < len(user_turns):
            user_content = user_turns[response_idx]
            if _TOOL_RESPONSE_PATTERN.search(user_content):
                valid_count += 1

    return valid_count / num_search_turns


def _compute_search_r1_tool_reward(raw_msg, assistant_turns, num_search_turns):
    """Compute search_r1 tool reward based on valid vs invalid search attempts.

    Walks ALL assistant turns (not just the first ``num_search_turns``) to
    handle error-recovery conversations where invalid-action nudges insert
    extra turns.  A turn counts as:

    - **valid**: has a well-formed ``<search>query</search>`` tag.
    - **invalid**: contains an opening ``<search>`` tag but no valid
      ``<search>query</search>`` pair (format mistake).

    Formula: ``max(valid - invalid, 0) / num_search_turns``, capped at 1.0.

    Args:
        raw_msg: str, full decoded conversation with chat markers (unused,
            kept for API compatibility).
        assistant_turns: list[str], extracted assistant turn contents.
        num_search_turns: int, exact number of mandatory search turns.

    Returns:
        float: Tool reward in [0.0, 1.0].
    """
    if num_search_turns == 0:
        return 1.0

    valid_count = 0
    invalid_count = 0
    for turn_content in assistant_turns:
        search_match = re.search(SEARCH_R1_PATTERN, turn_content, re.DOTALL)
        if search_match and search_match.group(1).strip():
            valid_count += 1
        elif "<search>" in turn_content:
            # Opening tag present but no valid closing pair → format mistake
            invalid_count += 1

    return min(max(valid_count - invalid_count, 0) / num_search_turns, 1.0)


def _compute_function_calls_tool_reward(raw_msg, assistant_turns, num_search_turns):
    """Compute function_calls tool reward based on exact search turn count.

    Walks through the conversation sequentially to find valid
    (search call, information response) pairs. This approach is robust
    to extra assistant turns caused by invalid-action nudges, which
    shift indices and break fixed-offset lookups.

    A valid pair requires:
    1. An assistant turn with a valid ``<function_calls>search(query="...")`` call.
    2. The immediately following user turn contains ``<information>...</information>``.

    Args:
        raw_msg: str, full decoded conversation with chat markers.
        assistant_turns: list[str], extracted assistant turn contents (unused,
            kept for API consistency with other tool reward functions).
        num_search_turns: int, exact number of mandatory search turns.

    Returns:
        float: Fraction of valid search-information pairs (0.0 to 1.0).
    """
    if num_search_turns == 0:
        return 1.0

    # Parse all turns sequentially: (role, content) pairs
    all_turns = re.findall(
        r'<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>', raw_msg, re.DOTALL
    )

    valid_count = 0
    pending_search = False

    for role, content in all_turns:
        if role == "assistant":
            fc_match = re.search(
                FUNCTION_CALLS_SEARCH_PATTERN, content, re.DOTALL
            )
            pending_search = bool(
                fc_match and fc_match.group(1).strip()
            )
        elif role in ("user", "environment", "tool") and pending_search:
            if re.search(INFORMATION_PATTERN, content, re.DOTALL):
                valid_count += 1
            pending_search = False
        else:
            pending_search = False

    return min(valid_count / num_search_turns, 1.0)


def compute_long_challenger_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """Compute reward scores for a batch of long challenger outputs.

    Combines format compliance scoring with task difficulty estimation via
    solver rollouts and grader evaluation.

    Args:
        data_sources: list[str], data source identifiers per item (unused).
        solution_strs: list[str], solution strings per item (unused).
        ground_truths: list[str], ground truth strings per item (unused).
        extra_infos: list[dict], extra info per item (unused).
        **kwargs: Must contain:
            data: DataProto batch from the framework.
            processing_class: Tokenizer instance.
            config: Rollout configuration dict.
            model_name: str, model identifier for SGLang servers.
            solver_base_url: str, URL of the solver SGLang server.
            grader_base_url: str, URL of the grader SGLang server.
            reward_rollout_n: int, number of solver rollouts per item (K).
            solver_template_path: str, path to solver prompt template.
            grader_template_path: str, path to grader prompt template.
            format_weight: float, weight for format score component.
            difficulty_weight: float, weight for difficulty score component.
            reward_mode: str, reward combination mode. ``"additive"``
                (default) uses ``format_weight * format + difficulty_weight *
                difficulty``. ``"gated"`` uses ``difficulty_weight *
                difficulty`` when format is perfect (1.0), else 0.
            grader_min_coverage: float, minimum rubric coverage for grader.
            grader_pad_value: float, pad value for missing rubric assessments.
            debug_print_first: bool, whether to print debug info for first item.
            solver_server_type: str, inference server type for solver
                (``"sglang"`` or ``"vllm"``). Defaults to ``"sglang"``.
            grader_server_type: str, inference server type for grader
                (``"sglang"`` or ``"vllm"``). Defaults to ``"sglang"``.
            max_concurrency: int, maximum concurrent HTTP requests to
                inference servers. Defaults to 64.
            difficulty_fn: str, difficulty scoring function name.
                ``"continuous"`` (default) for monotone decreasing or
                ``"tent"`` for triangular peaking at ``difficulty_target``.
            difficulty_target: float, target normalized score for tent
                difficulty function. Defaults to 0.5.

    Returns:
        list[dict[str, float]]: Per-item dicts with keys ``"score"`` (combined
            reward), ``"format_reward"``, and ``"difficulty_reward"``.
    """
    # Lazy imports for heavy dependencies (GPU/framework libs)
    import torch
    import numpy as np
    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask
    import verl.utils.torch_functional as verl_F
    from verl.custom_reward.reward_rollout import MultiTurnRewardRollout
    from verl.custom_reward.grader_client import GraderClient

    batch = kwargs["data"]
    processing_class = kwargs["processing_class"]
    assert "qwen" in type(processing_class).__name__.lower()

    solver_max_prompt_length = kwargs.get(
        "solver_max_prompt_length", SOLVER_PROMPT_MAX_LENGTH
    )

    rollout_config = deepcopy(kwargs["config"])
    rollout_config["prompt_length"] = solver_max_prompt_length

    model_name = kwargs.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
    solver_base_url = kwargs.get("solver_base_url", "http://127.0.0.1:8001")
    grader_base_url = kwargs.get("grader_base_url", "http://127.0.0.1:8002")
    K = kwargs.get("reward_rollout_n", 4)
    solver_template_path = kwargs.get("solver_template_path", "scope/prompts/solver_search_r1.txt")
    grader_template_path = kwargs.get("grader_template_path", "scope/prompts/grader_per_rubric.txt")
    format_weight = kwargs.get("format_weight", 0.5)
    difficulty_weight = kwargs.get("difficulty_weight", 1.0)
    reward_mode = kwargs.get("reward_mode", "additive")
    grader_min_coverage = kwargs.get("grader_min_coverage", 0.5)
    grader_pad_value = kwargs.get("grader_pad_value", 0.0)
    debug_print_first = kwargs.get("debug_print_first", False)
    solver_server_type = kwargs.get("solver_server_type", "sglang")
    grader_server_type = kwargs.get("grader_server_type", "sglang")
    max_concurrency = kwargs.get("max_concurrency", 64)
    difficulty_fn = kwargs.get("difficulty_fn", "continuous")
    difficulty_target = kwargs.get("difficulty_target", 0.5)
    grader_retry = kwargs.get("grader_retry", False)

    # Per-rubric grading: grade each rubric independently in its own LLM call
    grader_per_rubric = kwargs.get("grader_per_rubric", False)
    if not grader_per_rubric and "grader_per_rubric" in grader_template_path:
        grader_per_rubric = True

    # Read templates once
    with open(solver_template_path, "r") as f:
        solver_template = f.read()
    with open(grader_template_path, "r") as f:
        grader_template = f.read()

    # Pre-compute max_turns for solver template formatting
    max_turns = rollout_config.get("max_turns", 5)

    # Step 1: Decode batch
    raw_messages = [
        processing_class.decode(batch.batch["input_ids"][i]) for i in range(len(batch))
    ]
    responses = [
        processing_class.decode(
            batch.batch["responses"][i], skip_special_tokens=True,
        ) for i in range(len(batch))
    ]

    # Build per-item search turns from extra_infos
    challenger_format = kwargs.get("challenger_format", "tool_call")
    per_item_turns: list[int | None] = []
    for i in range(len(batch)):
        ei = extra_infos[i] if extra_infos[i] is not None else {}
        per_item_turns.append(ei.get("num_search_turns", None))

    # Count valid search calls per rollout for logging
    if challenger_format == "search_r1":
        valid_search_counts = [count_valid_search_r1_calls(r) for r in responses]
    elif challenger_format in ("function_calls", "function_calls_xml"):
        valid_search_counts = [count_valid_function_calls_search(r) for r in responses]
    else:
        valid_search_counts = [count_valid_tool_calls(r) for r in responses]

    # Extract original messages for think_reward (preserves <think> blocks
    # that the thinking-hidden chat template strips from input_ids).
    original_messages = None
    if hasattr(batch, "non_tensor_batch") and "messages" in batch.non_tensor_batch:
        original_messages = []
        for i in range(len(batch)):
            entry = batch.non_tensor_batch["messages"][i]
            if isinstance(entry, dict) and "messages" in entry:
                original_messages.append(entry["messages"])
            else:
                original_messages.append(entry)

    # Step 2: Format scoring (v5 with per-item turns when available)
    format_scores = compute_long_challenger_format_scores(
        raw_messages, responses,
        num_search_turns=per_item_turns,
        challenger_format=challenger_format,
        original_messages=original_messages,
    )

    # Step 3: Extract tasks for valid items
    require_rubrics = challenger_format not in ("tool_call",)
    extracted_tasks = {}
    for idx in range(len(batch)):
        if format_scores[idx] > 0:
            extracted, errors = extract_task(
                responses[idx],
                require_rubrics=require_rubrics,
                challenger_format=challenger_format,
            )
            if extracted is None or (require_rubrics and len(extracted.rubrics) == 0):
                format_scores[idx] = 0.0
            else:
                extracted_tasks[idx] = extracted

    # Step 4: Build solver batch
    gen_batch_ids, gen_batch = [], defaultdict(list)
    for idx in range(len(batch)):
        if format_scores[idx] > 0 and idx in extracted_tasks:
            gen_batch_ids.append(idx)
            extracted = extracted_tasks[idx]

            row_dict, messages = {}, [
                {"role": "user", "content": solver_template.format(
                    question=extracted.task_prompt.strip(),
                    max_search_turns=max_turns,
                )}
            ]

            raw_prompt = processing_class.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = processing_class(
                raw_prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=rollout_config["prompt_length"],
                pad_token_id=processing_class.pad_token_id,
                left_pad=True,
                truncation="left",
            )
            position_ids = compute_position_id_with_mask(attention_mask)

            row_dict["index"] = idx
            row_dict["raw_prompt"] = messages
            row_dict["full_prompts"] = raw_prompt
            row_dict["input_ids"] = input_ids[0]
            row_dict["attention_mask"] = attention_mask[0]
            row_dict["position_ids"] = position_ids[0]

            row_dict["tools_kwargs"] = {
                "search": {
                    "create_kwargs": {
                        "data_source": "search_zero",
                        "ground_truth": "",
                        "question": "",
                    }
                }
            }
            row_dict["interaction_kwargs"] = {}

            for key, value in row_dict.items():
                gen_batch[key].append(value)

    if len(gen_batch_ids) == 0:
        return [
            {
                "score": 0.0 if reward_mode == "gated" else format_weight * s,
                "format_reward": s,
                "difficulty_reward": 0.0,
                "avg_normalized_rubric_score": float("nan"),
                "number_of_valid_search": valid_search_counts[idx],
            }
            for idx, s in enumerate(format_scores)
        ]

    for key in gen_batch:
        if isinstance(gen_batch[key][0], torch.Tensor):
            gen_batch[key] = torch.stack(gen_batch[key])
        else:
            gen_batch[key] = np.array(gen_batch[key])

    gen_batch_proto = DataProto.from_single_dict(gen_batch)
    gen_batch_proto = gen_batch_proto.repeat(repeat_times=K, interleave=True)

    # Step 5: Async solver + grader
    async def _compute():
        # Solver rollouts
        async with MultiTurnRewardRollout(
            config=rollout_config,
            processing_class=processing_class,
            model_name=model_name,
            base_url=solver_base_url,
            server_type=solver_server_type,
            max_concurrency=max_concurrency,
        ) as rollout:
            solver_outputs = await rollout.generate_sequences(gen_batch_proto)

        # Extract solver responses (last turn content)
        solver_responses = [
            x["messages"][-1].content for x in solver_outputs.non_tensor_batch["messages"]
        ]

        # Build grader prompts (skip unparseable solver responses)
        grader_prompts = []
        grader_raw_contents: list[str] = []
        grader_to_solver_idx: list[int] = []
        # Per-rubric mode: flat map from grader prompt -> (solver_idx, rubric_idx)
        flat_grader_map: list[tuple[int, int]] = []

        for i, solver_resp in enumerate(solver_responses):
            solver_result = parse_solver_response(solver_resp)
            if not solver_result.is_valid:
                continue
            item_idx = gen_batch_ids[i // K]
            extracted = extracted_tasks[item_idx]

            if grader_per_rubric:
                for rubric_idx, rubric_text in enumerate(extracted.rubrics):
                    grader_content = grader_template.format(
                        prompt=extracted.task_prompt,
                        response=solver_result.answer,
                        rubric=f"{rubric_idx + 1}. {rubric_text}",
                    )
                    grader_msg = [{"role": "user", "content": grader_content}]
                    grader_raw = processing_class.apply_chat_template(
                        grader_msg, add_generation_prompt=True, tokenize=False
                    )
                    grader_prompts.append(grader_raw)
                    grader_raw_contents.append(grader_content)
                    flat_grader_map.append((i, rubric_idx))
                grader_to_solver_idx.append(i)
            else:
                rubric_list = format_rubric_list(extracted)
                grader_content = grader_template.format(
                    prompt=extracted.task_prompt,
                    response=solver_result.answer,
                    rubric_list=rubric_list,
                )
                grader_msg = [{"role": "user", "content": grader_content}]
                grader_raw = processing_class.apply_chat_template(
                    grader_msg, add_generation_prompt=True, tokenize=False
                )
                grader_prompts.append(grader_raw)
                grader_to_solver_idx.append(i)

        # Grade all at once
        async with GraderClient(base_url=grader_base_url, model_name=model_name, server_type=grader_server_type, max_concurrency=max_concurrency) as grader:
            raw_grader_outputs = await grader.grade_batch(grader_prompts)

        if grader_per_rubric:
            # Per-rubric retry for failed individual calls
            if grader_retry:
                retry_flat_indices: list[int] = []
                for fi, output in enumerate(raw_grader_outputs):
                    parsed = parse_grader_response(output, output_format="per_rubric")
                    if not parsed.is_valid:
                        retry_flat_indices.append(fi)

                if retry_flat_indices:
                    retry_prompts_pr: list[str] = []
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

                    async with GraderClient(
                        base_url=grader_base_url, model_name=model_name,
                        server_type=grader_server_type, max_concurrency=max_concurrency,
                    ) as grader_retry_client:
                        retry_outputs_pr = await grader_retry_client.grade_batch(retry_prompts_pr)
                    n_retry_ok = 0
                    for ri, fi in enumerate(retry_flat_indices):
                        raw_grader_outputs[fi] = retry_outputs_pr[ri]
                        reparsed = parse_grader_response(
                            retry_outputs_pr[ri], output_format="per_rubric"
                        )
                        if reparsed.is_valid:
                            n_retry_ok += 1

                    print(
                        f"[AsyncReward] Per-rubric grader retry: "
                        f"{len(retry_flat_indices)} failed, "
                        f"{n_retry_ok} recovered after retry"
                    )

            # Unmap flat outputs -> per-solver, per-rubric
            per_rubric_raw: dict[int, dict[int, str]] = {}
            for fi, (solver_i, rubric_i) in enumerate(flat_grader_map):
                per_rubric_raw.setdefault(solver_i, {})[rubric_i] = raw_grader_outputs[fi]

            # Reconstruct synthetic <scores> strings for logging
            grader_outputs: list[str] = [""] * len(solver_responses)
            for solver_i, rubric_outputs in per_rubric_raw.items():
                item_idx = gen_batch_ids[solver_i // K]
                n_rubrics = len(extracted_tasks[item_idx].rubrics)
                parts: list[str] = []
                for ri in range(n_rubrics):
                    if ri in rubric_outputs:
                        parsed = parse_grader_response(
                            rubric_outputs[ri], output_format="per_rubric"
                        )
                        if parsed.is_valid and parsed.assessments:
                            parts.append(f"{parsed.assessments[0].score:g}")
                        else:
                            parts.append("?")
                    else:
                        parts.append("?")
                grader_outputs[solver_i] = f"<scores>{', '.join(parts)}</scores>"

            return solver_responses, grader_outputs, per_rubric_raw
        else:
            # Reconstruct full-length list ("" for skipped/invalid solver responses)
            grader_outputs: list[str] = [""] * len(solver_responses)
            for gi, solver_i in enumerate(grader_to_solver_idx):
                grader_outputs[solver_i] = raw_grader_outputs[gi]

            return solver_responses, grader_outputs, {}

    loop = asyncio.get_event_loop()
    solver_responses, grader_outputs, per_rubric_raw = loop.run_until_complete(_compute())

    # Step 6: Parse grader outputs and compute difficulty
    difficulty_scores = []
    normalized_rubric_scores = []
    for item_i, idx in enumerate(gen_batch_ids):
        extracted = extracted_tasks[idx]
        num_rubrics = len(extracted.rubrics)
        item_grader_outputs = grader_outputs[item_i * K : (item_i + 1) * K]

        valid_sums = []
        for j_offset, grader_output in enumerate(item_grader_outputs):
            j = item_i * K + j_offset  # absolute solver index

            if grader_per_rubric and j in per_rubric_raw:
                rubric_outputs = per_rubric_raw[j]
                per_rubric_parsed = [
                    parse_grader_response(
                        rubric_outputs.get(ri, ""),
                        output_format="per_rubric",
                    )
                    for ri in range(num_rubrics)
                ]
                grader_result = aggregate_per_rubric_results(per_rubric_parsed)
            else:
                grader_result = parse_grader_response(grader_output)

            rubric_sum, info = compute_rubric_sum_with_recovery(
                grader_result, num_rubrics, grader_min_coverage, grader_pad_value
            )
            if rubric_sum is not None:
                valid_sums.append(rubric_sum)

        if valid_sums:
            avg_rubric_sum = sum(valid_sums) / len(valid_sums)
            difficulty = compute_difficulty(avg_rubric_sum, num_rubrics, difficulty_fn, difficulty_target)
            normalized = avg_rubric_sum / num_rubrics if num_rubrics > 0 else float("nan")
        else:
            difficulty = 0.0
            normalized = float("nan")  # no valid grader results

        difficulty_scores.append(difficulty)
        normalized_rubric_scores.append(normalized)

    # Step 7: Final reward
    if reward_mode == "gated":
        final_scores = [0.0] * len(format_scores)
        for item_i, idx in enumerate(gen_batch_ids):
            if format_scores[idx] == 1.0:
                final_scores[idx] = difficulty_weight * difficulty_scores[item_i]
    else:
        final_scores = [format_weight * s for s in format_scores]
        for item_i, idx in enumerate(gen_batch_ids):
            final_scores[idx] += difficulty_weight * difficulty_scores[item_i]

    if debug_print_first and gen_batch_ids:
        first_idx = gen_batch_ids[0]
        first_extracted = extracted_tasks[first_idx]
        grouped_responses = [
            solver_responses[i:i + K] for i in range(0, len(solver_responses), K)
        ]
        print(
            f"Raw format rewards: Avg {np.mean(format_scores):.4f}, Max {np.max(format_scores):.4f}\n"
            f"Final rewards: Avg {np.mean(final_scores):.4f}, Max {np.max(final_scores):.4f}\n"
            f"Task prompt: {first_extracted.task_prompt[:200]}...\n"
            f"Rubrics: {first_extracted.rubrics}\n"
            f"Difficulty: {difficulty_scores[0]:.4f}\n"
            f"Solver response (first): {grouped_responses[0][0][:200]}..."
        )

    difficulty_lookup = {
        idx: difficulty_scores[item_i]
        for item_i, idx in enumerate(gen_batch_ids)
    }
    normalized_rubric_lookup = {
        idx: normalized_rubric_scores[item_i]
        for item_i, idx in enumerate(gen_batch_ids)
    }
    return [
        {
            "score": final_scores[idx],
            "format_reward": format_scores[idx],
            "difficulty_reward": difficulty_lookup.get(idx, 0.0),
            "avg_normalized_rubric_score": normalized_rubric_lookup.get(idx, float("nan")),
            "number_of_valid_search": valid_search_counts[idx],
        }
        for idx in range(len(batch))
    ]
