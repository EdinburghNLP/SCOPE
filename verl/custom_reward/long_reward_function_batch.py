"""Batch reward function for long-form challenger GRPO training.

Drop-in replacement for ``long_reward_function.py`` that uses synchronous
batch processing instead of per-request async HTTP calls. Reduces HTTP
requests from O(N * turns) to O(turns) by processing all conversations
in lockstep waves.

Usage: Set ``custom_reward_function.path`` in the shell script to this file
and ``custom_reward_function.name`` to ``compute_long_challenger_score_batch``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any

from scope.utils.parse_challenger import extract_task
from scope.utils.parse_grader import (
    aggregate_per_rubric_results,
    parse_grader_response,
)
from scope.utils.parse_quality import (
    compute_quality_gate,
    parse_quality_gates,
    parse_quality_score,
)
from scope.utils.parse_rubric import parse_rubric_output
from scope.utils.parse_solver import parse_solver_response
from verl.custom_reward.batch_reward_rollout import BatchMultiTurnRollout
from verl.custom_reward.long_reward_function import (
    SOLVER_PROMPT_MAX_LENGTH,
    build_perturn_prompts,
    compute_continuous_difficulty,
    compute_difficulty,
    compute_long_challenger_format_scores,
    compute_rubric_sum_with_recovery,
    compute_short_form_answer_reward,
    normalize_answer,
    count_valid_function_calls_search,
    count_valid_search_r1_calls,
    count_valid_tool_calls,
    dedup_docs_across_turns,
    dedup_rubrics,
    extract_documents_from_conversation,
    extract_documents_per_turn_from_conversation,
    extract_per_search_turn_docs,
    format_rubric_list,
    SEARCH_R1_PATTERN,
    TOOL_CALL_PATTERN,
)

logger = logging.getLogger(__name__)

# Simplified tool schemas for the function_calls format.
# The actual SearchTool uses query_list (array) but the function_calls
# format exposes a single-query interface: search(query="...").
# The answer tool is included so the solver knows to call answer() as
# its final action (matching the with_tools_v2 design).
_FC_SOLVER_TOOL_SCHEMA_DICTS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for relevant information based on the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": (
                "Submit the final answer after completing all search turns. "
                "Call this exactly once as your final action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The answer to the question.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

# Search-only schemas for function_calls_xml format (answer uses XML, not FC)
_FC_SEARCH_ONLY_TOOL_SCHEMA_DICTS: list[dict[str, Any]] = [
    s for s in _FC_SOLVER_TOOL_SCHEMA_DICTS if s["function"]["name"] == "search"
]


def _load_tool_schemas(config_path: str) -> list[dict[str, Any]]:
    """Load tool schemas from a YAML tool config file.

    Reads the ``tool_schema`` section from each tool entry without
    instantiating the tool classes (avoids heavy dependencies).

    Args:
        config_path: Path to the tool config YAML file
            (e.g. ``"config/search_tool_config.yaml"``).

    Returns:
        list[dict[str, Any]]: List of OpenAI-format tool schema dicts.
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return [
        tool["tool_schema"]
        for tool in config.get("tools", [])
        if "tool_schema" in tool
    ]


def _get_last_assistant_content(messages: list[dict[str, str]]) -> str:
    """Extract the content of the last assistant message.

    Args:
        messages: Conversation history as list of role/content dicts.

    Returns:
        str: Content of the last assistant message, or empty string if none.
    """
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


def compute_long_challenger_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, float | str]]:
    """Compute reward scores for a batch of long challenger outputs.

    Combines format compliance scoring with task difficulty estimation via
    synchronous batch solver rollouts and grader evaluation. This is a
    drop-in replacement for the original async version in
    ``long_reward_function.py``.

    Args:
        data_sources: Data source identifiers per item (unused).
        solution_strs: Solution strings per item (unused).
        ground_truths: Ground truth strings per item (unused).
        extra_infos: Extra info dicts per item (unused).
        **kwargs: Must contain:
            data: DataProto batch from the framework.
            processing_class: Tokenizer instance.
            config: Rollout configuration dict.
            model_name (str): Model identifier for inference servers.
            solver_base_url (str): URL of the solver server.
            grader_base_url (str): URL of the grader server.
            retrieval_url (str): URL of the retrieval server endpoint.
            reward_rollout_n (int): Number of solver rollouts per item (K).
            solver_template_path (str): Path to solver prompt template.
            grader_template_path (str): Path to grader prompt template.
            format_weight (float): Weight for format score component.
            difficulty_weight (float): Weight for difficulty score component.
            reward_mode (str): Reward combination mode. ``"additive"``
                (default) uses ``format_weight * format + difficulty_weight *
                difficulty``. ``"gated"`` uses ``difficulty_weight *
                difficulty`` when format is perfect (1.0), else 0.
            grader_min_coverage (float): Minimum rubric coverage for grader.
            grader_pad_value (float): Pad value for missing rubric assessments.
            debug_print_first (bool): Whether to print debug info for first item.

    Returns:
        list[dict[str, float | str]]: Per-item dicts with keys ``"score"``
            (combined reward), ``"format_reward"``, ``"difficulty_reward"``,
            ``"task_prompt"``, ``"rubric_text"``, ``"rubric_raw_output"``,
            ``"solver_responses"`` (JSON string), and ``"grader_outputs"``
            (JSON string).
    """
    import numpy as np

    batch = kwargs["data"]
    processing_class = kwargs["processing_class"]

    solver_max_prompt_length = kwargs.get(
        "solver_max_prompt_length", SOLVER_PROMPT_MAX_LENGTH
    )

    rollout_config = deepcopy(kwargs["config"])
    rollout_config["prompt_length"] = solver_max_prompt_length

    model_name = kwargs.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
    grader_model_name: str | None = kwargs.get("grader_model_name", None)
    solver_base_url = kwargs.get("solver_base_url", "http://127.0.0.1:8001")
    grader_base_url = kwargs.get("grader_base_url", "http://127.0.0.1:8002")
    retrieval_url = kwargs.get(
        "retrieval_url", "http://127.0.0.1:8000/retrieve"
    )
    K = kwargs.get("reward_rollout_n", 4)
    solver_template_path = kwargs.get(
        "solver_template_path", "scope/prompts/solver_search_r1.txt"
    )
    grader_template_path = kwargs.get(
        "grader_template_path", "scope/prompts/grader_per_rubric.txt"
    )
    format_weight = kwargs.get("format_weight", 0.5)
    difficulty_weight = kwargs.get("difficulty_weight", 1.0)
    reward_mode = kwargs.get("reward_mode", "additive")
    grader_min_coverage = kwargs.get("grader_min_coverage", 0.5)
    grader_pad_value = kwargs.get("grader_pad_value", 0.0)
    debug_print_first = kwargs.get("debug_print_first", False)

    # Grader format. Per-rubric grading synthesizes v2-style score records.
    grader_format = kwargs.get("grader_format", "xml")

    # v5 kwargs (backward-compatible defaults)
    separate_rubric_generation = kwargs.get("separate_rubric_generation", False)
    rubric_template_path = kwargs.get("rubric_template_path", None)
    rubric_format = kwargs.get("rubric_format", "xml")

    # V19 per-turn rubric generation mode
    rubric_mode = kwargs.get("rubric_mode", "single")  # "single" or "v19_perturn"
    rubric_initial_template_path = kwargs.get("rubric_initial_template_path", None)
    rubric_turn_template_path = kwargs.get("rubric_turn_template_path", None)
    rubric_synthesis_template_path = kwargs.get("rubric_synthesis_template_path", None)
    rubric_dedup_threshold = float(kwargs.get("rubric_dedup_threshold", 0.6))
    rubric_min_count = int(kwargs.get("rubric_min_count", 3))

    # Solver/grader retry and stop tokens
    solver_retry = kwargs.get("solver_retry", False)
    solver_stop_tokens = kwargs.get("solver_stop_tokens", None)
    grader_retry = kwargs.get("grader_retry", False)

    # Rubric generation max tokens (thinking models need more budget)
    rubric_max_tokens = int(kwargs.get("rubric_max_tokens", 8192))

    # Per-rubric grading: grade each rubric independently in its own LLM call
    grader_per_rubric = kwargs.get("grader_per_rubric", False)
    if not grader_per_rubric and "grader_per_rubric" in grader_template_path:
        grader_per_rubric = True
    if grader_per_rubric:
        grader_format = "v2"

    # Difficulty function configuration
    difficulty_fn = kwargs.get("difficulty_fn", "continuous")
    difficulty_target = kwargs.get("difficulty_target", 0.5)

    # Challenger format (tool_call or search_r1)
    challenger_format = kwargs.get("challenger_format", "tool_call")

    # Solver format (tool_call or search_r1)
    solver_format = kwargs.get("solver_format", "tool_call")

    # Explicit tool call parser type (e.g. "qwen25") to bypass auto-detection
    # that incorrectly picks "glm" for Qwen3 models.
    tool_call_parser_type = kwargs.get("tool_call_parser_type", None)

    # Solver HTTP timeout (seconds).  The default of 600s can be too short
    # when many rollouts hit a single-GPU solver (e.g. Challenger Iter 2).
    solver_timeout: float = float(kwargs.get("solver_timeout", 3600))

    # Quality grading (opt-in via comma-separated gate names)
    # New interface: quality_gates="entity,no_leakage" (subset of gates)
    # Legacy interface: quality_enabled=True (all 3 gates)
    _qg_raw = str(kwargs.get("quality_gates", "")).strip()
    if _qg_raw:
        quality_gate_names = parse_quality_gates(_qg_raw)
    else:
        # Backward compat: quality_enabled=True → all 3 gates
        _qe_raw = kwargs.get("quality_enabled", False)
        _qe_bool = _qe_raw if isinstance(_qe_raw, bool) else str(_qe_raw).lower() == "true"
        quality_gate_names = list(("entity", "no_leakage", "retrieval")) if _qe_bool else []
    quality_enabled = len(quality_gate_names) > 0

    _gate_template_keys = {
        "entity": "quality_entity_template_path",
        "no_leakage": "quality_no_leakage_template_path",
        "retrieval": "quality_retrieval_template_path",
        "source_relevance": "quality_source_relevance_template_path",
    }
    quality_template_paths = {
        gate: kwargs.get(_gate_template_keys[gate], "")
        for gate in quality_gate_names
    }
    quality_max_tokens = kwargs.get("quality_max_tokens", 256)
    quality_required_sum = int(kwargs.get("quality_required_sum", len(quality_gate_names)))

    # Routed retrieval gate: separate template for multi-search tasks (2+ turns)
    _retrieval_multi_path = kwargs.get("quality_retrieval_multi_template_path", "")
    quality_retrieval_multi_template: str | None = None
    if _retrieval_multi_path:
        if os.path.isfile(_retrieval_multi_path):
            with open(_retrieval_multi_path) as f:
                quality_retrieval_multi_template = f.read()
        else:
            logger.warning("quality_retrieval_multi_template_path set to %r but file not found", _retrieval_multi_path)

    # Tool response truncation
    max_tool_response_length = int(
        kwargs.get("max_tool_response_length", 500)
    )
    tool_response_truncation_unit = kwargs.get(
        "tool_response_truncation_unit", "char"
    )

    # Tool schemas for solver tool-use prompt injection
    # search_r1: skip injection (tools described in prompt text only)
    # function_calls: inject simplified search(query) schema so OLMo's
    #   chat template produces function-calling system instructions
    # tool_call: inject real schemas from config
    tool_config_path = kwargs.get("tool_config_path", None)
    tool_schemas: list[dict[str, Any]] | None = None
    if tool_config_path and solver_format == "function_calls":
        tool_schemas = _FC_SOLVER_TOOL_SCHEMA_DICTS
    elif tool_config_path and solver_format == "function_calls_xml":
        tool_schemas = _FC_SEARCH_ONLY_TOOL_SCHEMA_DICTS
    elif tool_config_path and solver_format not in ("search_r1",):
        tool_schemas = _load_tool_schemas(tool_config_path)

    # Read templates once
    with open(solver_template_path, "r") as f:
        solver_template = f.read()
    with open(grader_template_path, "r") as f:
        grader_template = f.read()

    rubric_template: str | None = None
    if separate_rubric_generation and rubric_template_path:
        with open(rubric_template_path, "r") as f:
            rubric_template = f.read()

    # V19 per-turn rubric templates
    rubric_initial_template: str | None = None
    rubric_turn_template: str | None = None
    rubric_synthesis_template: str | None = None
    if rubric_mode == "v19_perturn" and rubric_initial_template_path and rubric_turn_template_path and rubric_synthesis_template_path:
        with open(rubric_initial_template_path, "r") as f:
            rubric_initial_template = f.read()
        with open(rubric_turn_template_path, "r") as f:
            rubric_turn_template = f.read()
        with open(rubric_synthesis_template_path, "r") as f:
            rubric_synthesis_template = f.read()

    # Load quality grading templates (only for active gates)
    quality_templates: list[str] = []
    if quality_enabled:
        for gate in quality_gate_names:
            qpath = quality_template_paths.get(gate, "")
            if qpath:
                with open(qpath, "r") as f:
                    quality_templates.append(f.read())
            else:
                logger.warning(
                    "Quality gate %r enabled but template path not set; "
                    "disabling quality gate.",
                    gate,
                )
                quality_enabled = False
                quality_templates = []
                break

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

    # Build per-item search turns from extra_infos (v5 per-sample turns)
    per_item_turns: list[int | None] = []
    for i in range(len(batch)):
        ei = extra_infos[i] if extra_infos[i] is not None else {}
        item_turns = ei.get("num_search_turns", None)
        per_item_turns.append(item_turns)

    # Count valid search calls per rollout for logging
    if challenger_format in ("function_calls", "function_calls_xml"):
        valid_search_counts = [
            count_valid_function_calls_search(r) for r in responses
        ]
    elif challenger_format == "search_r1":
        valid_search_counts = [
            count_valid_search_r1_calls(r) for r in responses
        ]
    else:
        valid_search_counts = [
            count_valid_tool_calls(r) for r in responses
        ]

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

    # Step 2: Format scoring (v5 or v4 depending on config)
    if separate_rubric_generation:
        format_scores = compute_long_challenger_format_scores(
            raw_messages, responses,
            num_search_turns=per_item_turns, require_rubrics=False,
            challenger_format=challenger_format,
            original_messages=original_messages,
        )
    else:
        format_scores = compute_long_challenger_format_scores(
            raw_messages, responses,
            challenger_format=challenger_format,
            original_messages=original_messages,
        )

    # Pre-compute task types and source docs from extra_infos
    task_types: list[str] = []
    source_docs: list[str] = []
    for i in range(len(batch)):
        ei = extra_infos[i] if extra_infos[i] is not None else {}
        task_types.append(ei.get("task_type", ""))
        source_docs.append(ei.get("source_document", ""))

    # Step 2.5: Answer reward for short_form_qa items
    for idx in range(len(batch)):
        if task_types[idx] == "short_form_qa":
            full_conversation = raw_messages[idx] + "\n" + responses[idx]
            answer_reward = compute_short_form_answer_reward(
                responses[idx], source_docs[idx], full_conversation,
            )
            if answer_reward == 0.0:
                # No valid <answer> tag → reject item entirely so it doesn't
                # poison EM difficulty with an empty gold answer downstream.
                format_scores[idx] = 0.0
            else:
                # Recalculate: was (think + tool + structure) / 3
                # New: (think + tool + structure + answer) / 4 = (old * 3 + answer) / 4
                format_scores[idx] = (format_scores[idx] * 3 + answer_reward) / 4

    # Collection dicts for extra output fields
    rubric_lookup: dict[int, str] = {}
    task_prompt_lookup: dict[int, str] = {}
    rubric_raw_lookup: dict[int, str] = {}
    solver_lookup: dict[int, list[str]] = {}
    grader_lookup: dict[int, list[str]] = {}

    # Step 3: Extract tasks for valid items
    require_rubrics = not separate_rubric_generation
    extracted_tasks: dict[int, Any] = {}
    for idx in range(len(batch)):
        if format_scores[idx] > 0:
            extracted, errors = extract_task(
                responses[idx], require_rubrics=require_rubrics,
                challenger_format=challenger_format,
            )
            if extracted is None:
                format_scores[idx] = 0.0
            elif require_rubrics and len(extracted.rubrics) == 0:
                format_scores[idx] = 0.0
            else:
                extracted_tasks[idx] = extracted
                task_prompt_lookup[idx] = extracted.task_prompt

    # Extract gold answers for short_form_qa items (from <answer> tag).
    # Only store non-empty answers — empty reference_answer means the
    # challenger failed to emit <answer>, and the item should already
    # have format_score=0.0 from Step 2.5.
    gold_answers: dict[int, str] = {}
    for idx in extracted_tasks:
        if task_types[idx] == "short_form_qa":
            ref = extracted_tasks[idx].reference_answer
            if ref:
                gold_answers[idx] = ref
            else:
                logger.warning(
                    "short_form_qa item %d has empty reference_answer; "
                    "skipping EM difficulty", idx
                )

    # Capture rubric text for v4 items (rubrics already in extracted task)
    if not separate_rubric_generation:
        for idx, ext in extracted_tasks.items():
            rubric_lookup[idx] = format_rubric_list(ext)

    # Step 3.5: Rubric generation (v5 only — skip short_form_qa items)
    if separate_rubric_generation and extracted_tasks:
        # Filter out short_form_qa items from rubric generation
        rubric_candidate_idxs = {
            idx for idx in extracted_tasks
            if task_types[idx] != "short_form_qa"
        }

        if rubric_mode == "v19_perturn" and rubric_initial_template:
            # V19 per-turn rubric generation
            rubric_prompts: list[str] = []
            rubric_prompt_task_map: list[tuple[int, str]] = []

            for idx in sorted(rubric_candidate_idxs):
                extracted = extracted_tasks[idx]
                full_conversation = raw_messages[idx] + "\n" + responses[idx]
                source_doc, turns = extract_documents_per_turn_from_conversation(
                    full_conversation
                )
                turns = dedup_docs_across_turns(turns)
                all_docs = extract_documents_from_conversation(full_conversation)
                ei = extra_infos[idx] if extra_infos[idx] is not None else {}
                task_type = ei.get("task_type", "long_form_qa")

                task_prompts_list = build_perturn_prompts(
                    extracted.task_prompt, task_type,
                    source_doc, turns, all_docs,
                    rubric_initial_template, rubric_turn_template,
                    rubric_synthesis_template,
                )
                for p in task_prompts_list:
                    rubric_msg = [{"role": "user", "content": p["prompt"]}]
                    rubric_raw = processing_class.apply_chat_template(
                        rubric_msg, add_generation_prompt=True, tokenize=False,
                    )
                    rubric_prompts.append(rubric_raw)
                    rubric_prompt_task_map.append((idx, p["stage"]))

            if rubric_prompts:
                with BatchMultiTurnRollout(
                    solver_base_url=grader_base_url,
                    retrieval_url=retrieval_url,
                    processing_class=processing_class,
                    model_name=model_name,
                    max_turns=1,
                    max_tokens=rubric_max_tokens,
                    max_prompt_length=solver_max_prompt_length,
                ) as rubric_rollout:
                    rubric_outputs = rubric_rollout.grade_batch(
                        rubric_prompts, grader_base_url,
                        max_tokens=rubric_max_tokens,
                        grader_model_name=grader_model_name,
                    )

                # Group outputs by task, parse, dedup
                task_rubric_texts: dict[int, list[str]] = defaultdict(list)
                task_rubric_priorities: dict[int, list[str | None]] = defaultdict(list)
                task_raw_outputs: dict[int, list[str]] = defaultdict(list)

                for (batch_idx, stage), rubric_output in zip(
                    rubric_prompt_task_map, rubric_outputs
                ):
                    task_raw_outputs[batch_idx].append(rubric_output)
                    # Per-turn outputs may have fewer rubrics than the
                    # global MIN_RUBRICS (3).  Accept any non-zero count
                    # per turn; the combined total is validated below.
                    rubric_result = parse_rubric_output(
                        rubric_output, output_format=rubric_format,
                        min_rubrics=1,
                    )
                    if rubric_result.is_valid:
                        task_rubric_texts[batch_idx].extend(rubric_result.rubrics)
                        task_rubric_priorities[batch_idx].extend(
                            rubric_result.priorities
                        )

                # Dedup rubrics and attach to tasks
                for idx in sorted(rubric_candidate_idxs):
                    texts = task_rubric_texts.get(idx, [])
                    priorities = task_rubric_priorities.get(idx, [])
                    texts, priorities = dedup_rubrics(
                        texts, priorities, threshold=rubric_dedup_threshold
                    )
                    rubric_raw_lookup[idx] = "\n---\n".join(
                        task_raw_outputs.get(idx, [])
                    )

                    if len(texts) >= rubric_min_count:
                        extracted_tasks[idx].rubrics = texts
                        extracted_tasks[idx].priorities = priorities
                        rubric_lookup[idx] = format_rubric_list(
                            extracted_tasks[idx]
                        )
                    else:
                        format_scores[idx] = 0.0
                        del extracted_tasks[idx]

        elif rubric_template:
            # Legacy single-template rubric generation
            rubric_prompts_legacy: list[str] = []
            rubric_prompt_indices: list[int] = []

            for idx in sorted(rubric_candidate_idxs):
                extracted = extracted_tasks[idx]
                full_conversation = raw_messages[idx] + "\n" + responses[idx]
                documents = extract_documents_from_conversation(full_conversation)
                ei = extra_infos[idx] if extra_infos[idx] is not None else {}
                task_type = ei.get("task_type", "long_form_qa")

                rubric_content = rubric_template.replace(
                    "{task_prompt}", extracted.task_prompt
                ).replace(
                    "{documents}", documents
                ).replace(
                    "{task_type}", task_type
                )

                rubric_msg = [{"role": "user", "content": rubric_content}]
                rubric_raw = processing_class.apply_chat_template(
                    rubric_msg, add_generation_prompt=True, tokenize=False,
                )
                rubric_prompts_legacy.append(rubric_raw)
                rubric_prompt_indices.append(idx)

            if rubric_prompts_legacy:
                with BatchMultiTurnRollout(
                    solver_base_url=grader_base_url,
                    retrieval_url=retrieval_url,
                    processing_class=processing_class,
                    model_name=model_name,
                    max_turns=1,
                    max_tokens=rubric_max_tokens,
                    max_prompt_length=solver_max_prompt_length,
                ) as rubric_rollout:
                    rubric_outputs = rubric_rollout.grade_batch(
                        rubric_prompts_legacy, grader_base_url,
                        max_tokens=rubric_max_tokens,
                        grader_model_name=grader_model_name,
                    )

                for i, rubric_output in enumerate(rubric_outputs):
                    idx = rubric_prompt_indices[i]
                    rubric_raw_lookup[idx] = rubric_output
                    rubric_result = parse_rubric_output(
                        rubric_output, output_format=rubric_format
                    )
                    if rubric_result.is_valid:
                        extracted_tasks[idx].rubrics = rubric_result.rubrics
                        extracted_tasks[idx].priorities = rubric_result.priorities
                        rubric_lookup[idx] = format_rubric_list(
                            extracted_tasks[idx]
                        )
                    else:
                        format_scores[idx] = 0.0
                        del extracted_tasks[idx]

    # Step 3.7: Quality grading (before solver rollouts to save compute)
    quality_scores: dict[int, int] = {}
    per_gate_quality: dict[int, dict[str, int]] = {}
    if quality_enabled and quality_templates and extracted_tasks:
        quality_prompts: list[str] = []

        for idx in sorted(extracted_tasks.keys()):
            task_prompt = extracted_tasks[idx].task_prompt
            ei = extra_infos[idx] if extra_infos[idx] is not None else {}
            source_doc = ei.get("source_document", "")
            if not source_doc:
                # Fallback: extract from raw_messages (decoded prompt)
                m = re.search(
                    r"## Source Document\s*\n(.*?)(?=\n## |\Z)",
                    raw_messages[idx], re.DOTALL,
                )
                if m:
                    source_doc = m.group(1).strip()
            # Extract per-search-turn docs for retrieval quality gate
            # Combine input + response to capture the full conversation
            full_conversation = raw_messages[idx] + "\n" + responses[idx]
            retrieved_docs = extract_per_search_turn_docs(full_conversation)
            # Count search turns for routed retrieval gate (handles both formats)
            n_search_turns = len(
                re.findall(SEARCH_R1_PATTERN, full_conversation, re.DOTALL)
            ) or len(
                re.findall(TOOL_CALL_PATTERN, full_conversation, re.DOTALL)
            )
            # Extract source title for template variable
            _title_m = re.search(
                r'\(Title: "?(.*?)"?\)', source_doc
            )
            source_title = _title_m.group(1) if _title_m else "unknown"
            for gate_idx, tmpl in enumerate(quality_templates):
                # Routed retrieval gate: swap template for multi-search tasks
                effective_tmpl = tmpl
                if (
                    quality_retrieval_multi_template
                    and gate_idx < len(quality_gate_names)
                    and quality_gate_names[gate_idx] == "retrieval"
                    and n_search_turns >= 2
                ):
                    effective_tmpl = quality_retrieval_multi_template
                content = (
                    effective_tmpl.replace("{task_prompt}", task_prompt)
                    .replace("{source_document}", source_doc)
                    .replace("{search_turns}", retrieved_docs)
                    .replace("{source_title}", source_title)
                )
                msg = [{"role": "user", "content": content}]
                raw = processing_class.apply_chat_template(
                    msg, add_generation_prompt=True, tokenize=False,
                )
                quality_prompts.append(raw)

        if quality_prompts:
            with BatchMultiTurnRollout(
                solver_base_url=grader_base_url,
                retrieval_url=retrieval_url,
                processing_class=processing_class,
                model_name=model_name,
                max_turns=1,
                max_tokens=quality_max_tokens,
                max_prompt_length=solver_max_prompt_length,
            ) as quality_rollout:
                quality_outputs = quality_rollout.grade_batch(
                    quality_prompts, grader_base_url,
                    max_tokens=quality_max_tokens,
                    grader_model_name=grader_model_name,
                )

            # Parse scores and compute per-task quality gate
            parsed_quality = [parse_quality_score(o) for o in quality_outputs]

            # Group by task idx (n_gates scores per task)
            n_gates = len(quality_templates)
            task_ids_sorted = sorted(extracted_tasks.keys())
            for i, idx in enumerate(task_ids_sorted):
                scores_for_task = parsed_quality[i * n_gates : (i + 1) * n_gates]
                # short_form_qa: auto-pass source_relevance gate since the
                # task targets the last retrieved doc, not the source doc.
                if task_types[idx] == "short_form_qa":
                    for g, gate in enumerate(quality_gate_names):
                        if gate == "source_relevance":
                            scores_for_task[g] = 1
                quality_ok = 1 if compute_quality_gate(scores_for_task, required=quality_required_sum) else 0
                quality_scores[idx] = quality_ok
                # Store per-gate scores for logging
                per_gate_quality[idx] = {
                    gate: int(scores_for_task[g] or 0)
                    for g, gate in enumerate(quality_gate_names)
                }

            n_passed = sum(1 for v in quality_scores.values() if v == 1)
            n_total = len(quality_scores)
            print(
                f"[BatchRollout] Quality gate: {n_passed}/{n_total} passed "
                f"({100.0 * n_passed / max(n_total, 1):.1f}%)"
            )

    # Pre-compute max_turns for solver template formatting
    max_turns = rollout_config.get("multi_turn", {}).get("max_assistant_turns", 5)

    # Step 4: Build solver messages (simplified — no torch/DataProto)
    # Skip items that failed quality gate to save solver rollout compute.
    gen_batch_ids: list[int] = []
    solver_messages: list[list[dict[str, str]]] = []
    for idx in range(len(batch)):
        if quality_enabled and quality_scores.get(idx, 0) == 0:
            continue
        if format_scores[idx] > 0 and idx in extracted_tasks:
            gen_batch_ids.append(idx)
            extracted = extracted_tasks[idx]
            messages = [
                {
                    "role": "user",
                    "content": solver_template.format_map(
                        defaultdict(str,
                            question=extracted.task_prompt.strip(),
                            max_search_turns=str(max_turns - 1),
                        )
                    ),
                }
            ]
            solver_messages.append(messages)

    if not gen_batch_ids:
        return [
            {
                "score": 0.0 if reward_mode == "gated" else format_weight * s,
                "format_reward": s,
                "difficulty_reward": 0.0,
                "avg_normalized_rubric_score": float("nan"),
                "quality_reward": quality_scores.get(idx, 0) if quality_enabled else 1,
                "number_of_valid_search": valid_search_counts[idx],
                "task_prompt": task_prompt_lookup.get(idx, ""),
                "rubric_text": rubric_lookup.get(idx, ""),
                "rubric_raw_output": rubric_raw_lookup.get(idx, ""),
                "solver_responses": "[]",
                "grader_outputs": "[]",
            }
            for idx, s in enumerate(format_scores)
        ]

    # Repeat each message K times for multiple rollouts
    repeated_messages: list[list[dict[str, str]]] = []
    repeated_ids: list[int] = []
    for idx, msgs in zip(gen_batch_ids, solver_messages):
        for _ in range(K):
            repeated_messages.append(deepcopy(msgs))
            repeated_ids.append(idx)

    # Step 5: Synchronous batch solver rollout
    max_tokens = rollout_config.get("response_length", 2048)
    per_rubric_results: dict[int, dict[int, str]] = defaultdict(dict)

    with BatchMultiTurnRollout(
        solver_base_url=solver_base_url,
        retrieval_url=retrieval_url,
        processing_class=processing_class,
        model_name=model_name,
        max_turns=max_turns,
        max_tokens=max_tokens,
        max_prompt_length=solver_max_prompt_length,
        max_tool_response_length=max_tool_response_length,
        tool_response_truncation_unit=tool_response_truncation_unit,
        tool_schemas=tool_schemas,
        solver_format=solver_format,
        solver_retry=solver_retry,
        solver_stop=solver_stop_tokens,
        timeout=solver_timeout,
        tool_call_parser_type=tool_call_parser_type,
    ) as rollout:
        final_histories = rollout.generate_sequences_batch(repeated_messages)

        # Extract solver responses (last assistant content)
        solver_responses = [
            _get_last_assistant_content(hist) for hist in final_histories
        ]

        # Parse solver responses to filter out unparseable ones
        solver_parse_results = [
            parse_solver_response(resp) for resp in solver_responses
        ]

        # Build grader prompts (skip unparseable solver responses)
        grader_prompts: list[str] = []
        grader_to_solver_idx: list[int] = []
        grader_raw_contents: list[str] = []  # pre-template grader prompts
        grader_expected_rubrics: list[int] = []  # expected rubric counts
        # Per-rubric mode: flat map from grader prompt -> (solver_idx, rubric_idx)
        flat_grader_map: list[tuple[int, int]] = []

        if grader_per_rubric:
            # Per-rubric mode: one grader call per (solver_response, rubric)
            for i in range(len(solver_responses)):
                if not solver_parse_results[i].is_valid:
                    continue
                item_idx = repeated_ids[i]
                # Skip short_form_qa — uses EM grading, not rubric grading
                if task_types[item_idx] == "short_form_qa":
                    continue
                extracted = extracted_tasks[item_idx]
                for rubric_idx, rubric_text in enumerate(extracted.rubrics):
                    grader_content = grader_template.format(
                        prompt=extracted.task_prompt,
                        response=solver_parse_results[i].answer,
                        rubric=f"{rubric_idx + 1}. {rubric_text}",
                    )
                    grader_msg = [{"role": "user", "content": grader_content}]
                    grader_raw = processing_class.apply_chat_template(
                        grader_msg, add_generation_prompt=True, tokenize=False,
                    )
                    grader_prompts.append(grader_raw)
                    grader_raw_contents.append(grader_content)
                    flat_grader_map.append((i, rubric_idx))
                grader_to_solver_idx.append(i)
                grader_expected_rubrics.append(len(extracted.rubrics))
        else:
            # All-at-once mode: one grader call per solver response
            for i in range(len(solver_responses)):
                if not solver_parse_results[i].is_valid:
                    continue
                item_idx = repeated_ids[i]
                # Skip short_form_qa — uses EM grading, not rubric grading
                if task_types[item_idx] == "short_form_qa":
                    continue
                extracted = extracted_tasks[item_idx]
                rubric_list = format_rubric_list(extracted)
                grader_content = grader_template.format(
                    prompt=extracted.task_prompt,
                    response=solver_parse_results[i].answer,
                    rubric_list=rubric_list,
                )
                grader_msg = [{"role": "user", "content": grader_content}]
                grader_raw = processing_class.apply_chat_template(
                    grader_msg, add_generation_prompt=True, tokenize=False,
                )
                grader_prompts.append(grader_raw)
                grader_to_solver_idx.append(i)
                grader_raw_contents.append(grader_content)
                grader_expected_rubrics.append(len(extracted.rubrics))

        # Step 6: Batch grading (only for parseable solver responses)
        raw_grader_outputs = (
            rollout.grade_batch(
                grader_prompts, grader_base_url,
                grader_model_name=grader_model_name,
            )
            if grader_prompts
            else []
        )

        if grader_per_rubric:
            # Step 6.5a: Per-rubric retry for failed individual calls
            retry_flat_indices: list[int] = []
            for fi, output in enumerate(raw_grader_outputs):
                parsed = parse_grader_response(output, output_format="per_rubric")
                if not parsed.is_valid:
                    retry_flat_indices.append(fi)

            if grader_retry and retry_flat_indices:
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

                retry_outputs_pr = rollout.grade_batch(
                    retry_prompts_pr, grader_base_url,
                    grader_model_name=grader_model_name,
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
                    f"[BatchRollout] Per-rubric grader retry: "
                    f"{len(retry_flat_indices)} failed, "
                    f"{n_retry_ok} recovered after retry"
                )

            # Unmap flat outputs -> per-solver, per-rubric scores
            # grader_per_rubric_outputs[solver_idx][rubric_idx] = GraderParseResult
            per_rubric_results: dict[int, dict[int, str]] = defaultdict(dict)
            for fi, (solver_i, rubric_i) in enumerate(flat_grader_map):
                per_rubric_results[solver_i][rubric_i] = raw_grader_outputs[fi]

            # Reconstruct synthetic grader_outputs strings for logging
            grader_outputs: list[str] = [""] * len(solver_responses)
            for solver_i, rubric_outputs in per_rubric_results.items():
                item_idx = repeated_ids[solver_i]
                n_rubrics = len(extracted_tasks[item_idx].rubrics)
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
                grader_outputs[solver_i] = f"<scores>{', '.join(scores_parts)}</scores>"
        else:
            # Step 6.5: Grader retry (v2 format only)
            if grader_retry and grader_format == "v2" and raw_grader_outputs:
                retry_indices: list[int] = []
                for gi, output in enumerate(raw_grader_outputs):
                    parsed = parse_grader_response(output, output_format="v2")
                    if not parsed.is_valid or parsed.num_assessments != grader_expected_rubrics[gi]:
                        retry_indices.append(gi)

                if retry_indices:
                    retry_prompts_grader: list[str] = []
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
                            retry_conv, add_generation_prompt=True, tokenize=False,
                        )
                        retry_prompts_grader.append(retry_raw)

                    retry_outputs = rollout.grade_batch(
                        retry_prompts_grader, grader_base_url,
                        grader_model_name=grader_model_name,
                    )
                    for ri, gi in enumerate(retry_indices):
                        raw_grader_outputs[gi] = retry_outputs[ri]

                    n_retry_succeeded = 0
                    for gi in retry_indices:
                        reparsed = parse_grader_response(
                            raw_grader_outputs[gi], output_format="v2"
                        )
                        if reparsed.is_valid and reparsed.num_assessments == grader_expected_rubrics[gi]:
                            n_retry_succeeded += 1

                    print(
                        f"[BatchRollout] Grader retry: {len(retry_indices)} failed,"
                        f" {n_retry_succeeded} recovered after retry"
                    )

            # Reconstruct full-length list ("" for skipped responses)
            grader_outputs: list[str] = [""] * len(solver_responses)
            for gi, solver_i in enumerate(grader_to_solver_idx):
                grader_outputs[solver_i] = raw_grader_outputs[gi]

        # Log solver parse rate
        if solver_parse_results:
            n_total = len(solver_parse_results)
            n_unparsed = sum(
                1 for r in solver_parse_results if not r.is_valid
            )
            print(
                f"Solver answer parse: {n_unparsed}/{n_total} "
                f"({100.0 * n_unparsed / n_total:.1f}%) not parsable"
            )

    # Capture solver and grader outputs per item
    for item_i, idx in enumerate(gen_batch_ids):
        solver_lookup[idx] = solver_responses[item_i * K : (item_i + 1) * K]
        grader_lookup[idx] = grader_outputs[item_i * K : (item_i + 1) * K]

    # Step 7: Parse grader outputs and compute difficulty (reuse helpers)
    difficulty_scores: list[float] = []
    normalized_rubric_scores: list[float] = []
    for item_i, idx in enumerate(gen_batch_ids):
        if task_types[idx] == "short_form_qa" and idx in gold_answers:
            # EM-based difficulty for short_form_qa
            gold = gold_answers[idx]
            em_results: list[float] = []
            for j in range(item_i * K, (item_i + 1) * K):
                if solver_parse_results[j].is_valid and solver_parse_results[j].answer:
                    pred_norm = normalize_answer(solver_parse_results[j].answer)
                    gold_norm = normalize_answer(gold)
                    em_results.append(float(pred_norm == gold_norm))
                else:
                    em_results.append(0.0)
            n = len(em_results)
            num_correct = sum(em_results)
            if n <= 1 or num_correct == 0 or num_correct == n:
                difficulty = 0.0
            else:
                difficulty = (n - num_correct) / (n - 1)
            difficulty_scores.append(difficulty)
            normalized_rubric_scores.append(float("nan"))  # EM-based, no rubric scores
        else:
            # Rubric-based difficulty for other task types
            extracted = extracted_tasks[idx]
            num_rubrics = len(extracted.rubrics)

            valid_sums: list[float] = []
            for j in range(item_i * K, (item_i + 1) * K):
                if not solver_parse_results[j].is_valid:
                    continue

                if grader_per_rubric:
                    # Aggregate per-rubric results for this solver response
                    rubric_outputs_for_j = per_rubric_results.get(j, {})
                    if not rubric_outputs_for_j:
                        continue
                    per_rubric_parsed = [
                        parse_grader_response(
                            rubric_outputs_for_j.get(ri, ""),
                            output_format="per_rubric",
                        )
                        for ri in range(num_rubrics)
                    ]
                    grader_result = aggregate_per_rubric_results(per_rubric_parsed)
                else:
                    grader_output = grader_outputs[j]
                    if not grader_output:
                        continue
                    grader_result = parse_grader_response(
                        grader_output, output_format=grader_format
                    )

                rubric_sum, _info = compute_rubric_sum_with_recovery(
                    grader_result, num_rubrics, grader_min_coverage, grader_pad_value
                )
                if rubric_sum is not None:
                    valid_sums.append(rubric_sum)

            if valid_sums:
                avg_rubric_sum = sum(valid_sums) / len(valid_sums)
                difficulty = compute_difficulty(
                    avg_rubric_sum, num_rubrics, difficulty_fn, difficulty_target
                )
                normalized = avg_rubric_sum / num_rubrics if num_rubrics > 0 else float("nan")
            else:
                difficulty = 0.0
                normalized = float("nan")  # no valid grader results

            difficulty_scores.append(difficulty)
            normalized_rubric_scores.append(normalized)

    # Step 8: Final reward (quality gate zeroes difficulty for failed items)
    if reward_mode == "gated":
        final_scores = [0.0] * len(format_scores)
        for item_i, idx in enumerate(gen_batch_ids):
            quality_ok = quality_scores.get(idx, 0) if quality_enabled else 1
            if format_scores[idx] == 1.0 and quality_ok == 1:
                final_scores[idx] = difficulty_weight * difficulty_scores[item_i]
    else:
        final_scores = [format_weight * s for s in format_scores]
        for item_i, idx in enumerate(gen_batch_ids):
            quality_ok = quality_scores.get(idx, 0) if quality_enabled else 1
            final_scores[idx] += difficulty_weight * difficulty_scores[item_i] * quality_ok

    if debug_print_first and gen_batch_ids:
        first_idx = gen_batch_ids[0]
        first_extracted = extracted_tasks[first_idx]
        grouped_responses = [
            solver_responses[i : i + K]
            for i in range(0, len(solver_responses), K)
        ]
        print(
            f"Raw format rewards: Avg {np.mean(format_scores):.4f}, "
            f"Max {np.max(format_scores):.4f}\n"
            f"Final rewards: Avg {np.mean(final_scores):.4f}, "
            f"Max {np.max(final_scores):.4f}\n"
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
    results = []
    for idx in range(len(batch)):
        # avg_normalized_rubric_score: only count rubric-graded items that
        # passed quality gate.  NaN for everything else so wandb mean is
        # not diluted by format failures, EM items, or gated-out tasks.
        quality_ok = quality_scores.get(idx, 0) if quality_enabled else 1
        raw_norm = normalized_rubric_lookup.get(idx, float("nan"))
        if quality_ok == 0:
            raw_norm = float("nan")
        entry = {
            "score": final_scores[idx],
            "format_reward": format_scores[idx],
            "difficulty_reward": difficulty_lookup.get(idx, 0.0),
            "avg_normalized_rubric_score": raw_norm,
            "quality_reward": quality_ok,
            "number_of_valid_search": valid_search_counts[idx],
            "task_prompt": task_prompt_lookup.get(idx, ""),
            "rubric_text": rubric_lookup.get(idx, ""),
            "rubric_raw_output": rubric_raw_lookup.get(idx, ""),
            "solver_responses": json.dumps(solver_lookup.get(idx, [])),
            "grader_outputs": json.dumps(grader_lookup.get(idx, [])),
        }
        # Per-gate quality scores for wandb logging and rollout .jsonl
        for gate in quality_gate_names:
            entry[f"quality_{gate}_score"] = (
                per_gate_quality.get(idx, {}).get(gate, 0)
                if quality_enabled else 1
            )
        results.append(entry)
    return results
