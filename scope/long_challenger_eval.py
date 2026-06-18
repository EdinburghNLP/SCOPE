"""Standalone CLI for long-form challenger evaluation with difficulty filtering.

Reads intermediate parquet from ``main_generation_long.py``, runs rubric
generation, K solver rollouts, grading, and difficulty-based filtering.
Outputs a filtered parquet suitable for solver training.

Usage::

    python -m scope.long_challenger_eval \
        --input_path intermediate.parquet \
        --output_path filtered.parquet \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --solver_base_url http://127.0.0.1:8001 \
        --grader_base_url http://127.0.0.1:8001 \
        --retrieval_url http://127.0.0.1:8000/retrieve \
        --rubric_template_path scope/prompts/rubric.txt \
        --solver_template_path scope/prompts/solver_search_r1.txt \
        --grader_template_path scope/prompts/grader_per_rubric.txt \
        --tool_config_path config/search_tool_config.yaml \
        --reward_rollout_n 8 \
        --difficulty_min 0.2 --difficulty_max 0.8
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import datasets as hf_datasets
import numpy as np
import pandas as pd

from scope.utils.parse_challenger import ExtractedTask, extract_task
from scope.utils.parse_grader import (
    aggregate_per_rubric_results,
    parse_grader_response,
)
from scope.utils.parse_solver import parse_solver_response
from scope.utils.parse_quality import (
    compute_quality_gate,
    parse_quality_gates,
    parse_quality_score,
)
from scope.utils.parse_rubric import parse_rubric_output
from verl.custom_reward.batch_reward_rollout import BatchMultiTurnRollout
from verl.custom_reward.long_reward_function import (
    SEARCH_R1_PATTERN,
    build_perturn_prompts,
    compute_difficulty,
    compute_rubric_sum_with_recovery,
    dedup_docs_across_turns,
    dedup_rubrics,
    extract_documents_from_conversation,
    extract_documents_per_turn_from_conversation,
    extract_per_search_turn_docs,
    format_rubric_list,
    normalize_answer,
)

logger = logging.getLogger(__name__)


def _load_tool_schemas(config_path: str) -> list[dict[str, Any]]:
    """Load tool schemas from a YAML tool config file.

    Args:
        config_path: Path to the tool config YAML file.

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


def generate_rubrics(
    df: pd.DataFrame,
    rubric_template: str,
    rollout: BatchMultiTurnRollout,
    grader_base_url: str,
    processing_class: Any,
    rubric_format: str = "xml",
    grader_model_name: str | None = None,
    grader_processing_class: Any = None,
    rubric_mode: str = "single",
    initial_template: str | None = None,
    turn_template: str | None = None,
    synthesis_template: str | None = None,
    rubric_dedup_threshold: float = 0.6,
    rubric_min_count: int = 3,
) -> pd.DataFrame:
    """Stage A: Generate rubrics for tasks without them.

    Supports two modes: ``"single"`` (legacy single-template) and
    ``"v19_perturn"`` (per-turn rubric generation with doc/rubric dedup).

    Args:
        df: Input DataFrame with ``metadata`` column containing
            ``raw_context`` and ``task_type``, and ``prompt`` column.
        rubric_template: Template string with ``{task_prompt}``,
            ``{task_type}``, and ``{documents}`` placeholders.
        rollout: BatchMultiTurnRollout instance for LLM calls.
        grader_base_url: Base URL for the grader/rubric server.
        processing_class: Tokenizer for chat template formatting.
        rubric_format: Output format for rubric parsing (``"xml"`` or
            ``"json"``).
        grader_model_name: Model name for grader server. When ``None``,
            ``grade_batch`` uses the rollout's default model.
        grader_processing_class: Tokenizer for the grader model. When
            ``None``, falls back to ``processing_class``.
        rubric_mode: ``"single"`` for legacy or ``"v19_perturn"`` for
            per-turn generation.
        initial_template: V19 initial (source doc) template string.
        turn_template: V19 per-turn template string.
        synthesis_template: V19 synthesis template string.
        rubric_dedup_threshold: Word overlap threshold for rubric dedup.
        rubric_min_count: Minimum rubrics after dedup to keep a task.

    Returns:
        pd.DataFrame: Filtered DataFrame with rubrics populated in
            ``reward_model.ground_truth``.
    """
    if grader_processing_class is None:
        grader_processing_class = processing_class

    # Extract common fields from each row
    rows_data: list[dict] = []
    for idx, row in df.iterrows():
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        raw_context = metadata.get("raw_context", "")
        task_type = metadata.get("task_type", "long_form_qa")

        prompt_msgs = row.get("prompt", [])
        if isinstance(prompt_msgs, str):
            prompt_msgs = json.loads(prompt_msgs)
        task_prompt = ""
        if prompt_msgs and isinstance(prompt_msgs, list):
            content = prompt_msgs[0].get("content", "")
            if "Question:" in content:
                task_prompt = content.split("Question:")[-1].strip()
            else:
                task_prompt = content

        rows_data.append({
            "idx": idx,
            "raw_context": raw_context,
            "task_type": task_type or "long_form_qa",
            "task_prompt": task_prompt,
        })

    if not rows_data:
        return df

    use_perturn = (
        rubric_mode == "v19_perturn"
        and initial_template
        and turn_template
        and synthesis_template
    )

    if use_perturn:
        # V19 per-turn rubric generation
        rubric_prompts: list[str] = []
        rubric_prompt_task_map: list[tuple[int, str]] = []  # (row_pos, stage)

        for row_pos, rd in enumerate(rows_data):
            source_doc, turns = extract_documents_per_turn_from_conversation(
                rd["raw_context"]
            )
            turns = dedup_docs_across_turns(turns)
            all_docs = extract_documents_from_conversation(rd["raw_context"])

            task_prompts_list = build_perturn_prompts(
                rd["task_prompt"], rd["task_type"],
                source_doc, turns, all_docs,
                initial_template, turn_template, synthesis_template,
            )
            for p in task_prompts_list:
                rubric_msg = [{"role": "user", "content": p["prompt"]}]
                rubric_raw = grader_processing_class.apply_chat_template(
                    rubric_msg, add_generation_prompt=True, tokenize=False,
                )
                rubric_prompts.append(rubric_raw)
                rubric_prompt_task_map.append((row_pos, p["stage"]))

        print(f"[Stage A] V19 per-turn: generating {len(rubric_prompts)} prompts for {len(rows_data)} tasks...")
        rubric_outputs = rollout.grade_batch(
            rubric_prompts, grader_base_url, grader_model_name=grader_model_name,
        )

        # Group outputs by row, parse, dedup
        from collections import defaultdict
        task_rubric_texts: dict[int, list[str]] = defaultdict(list)
        task_rubric_priorities: dict[int, list[str | None]] = defaultdict(list)
        task_raw_outputs: dict[int, list[str]] = defaultdict(list)

        for (row_pos, stage), rubric_output in zip(
            rubric_prompt_task_map, rubric_outputs
        ):
            task_raw_outputs[row_pos].append(rubric_output)
            rubric_result = parse_rubric_output(
                rubric_output, output_format=rubric_format
            )
            if rubric_result.is_valid:
                task_rubric_texts[row_pos].extend(rubric_result.rubrics)
                task_rubric_priorities[row_pos].extend(rubric_result.priorities)

        valid_mask = []
        for row_pos, rd in enumerate(rows_data):
            texts = task_rubric_texts.get(row_pos, [])
            priorities = task_rubric_priorities.get(row_pos, [])
            texts, priorities = dedup_rubrics(
                texts, priorities, threshold=rubric_dedup_threshold
            )
            raw_output = "\n---\n".join(task_raw_outputs.get(row_pos, []))

            if len(texts) >= rubric_min_count:
                idx = rd["idx"]
                rm = df.at[idx, "reward_model"]
                if isinstance(rm, str):
                    rm = json.loads(rm)
                rm = deepcopy(rm)
                rm["ground_truth"]["rubrics"] = texts
                rm["ground_truth"]["priorities"] = priorities
                df.at[idx, "reward_model"] = rm

                meta = df.at[idx, "metadata"]
                if isinstance(meta, str):
                    meta = json.loads(meta)
                meta = deepcopy(meta)
                meta["num_rubrics"] = len(texts)
                meta["rubric_raw_output"] = raw_output
                df.at[idx, "metadata"] = meta
                valid_mask.append(True)
            else:
                valid_mask.append(False)
    else:
        # Legacy single-template rubric generation
        rubric_prompts_legacy: list[str] = []
        rubric_indices: list[int] = []

        for rd in rows_data:
            documents = extract_documents_from_conversation(rd["raw_context"])
            rubric_content = rubric_template.replace(
                "{task_prompt}", rd["task_prompt"]
            ).replace(
                "{documents}", documents
            ).replace(
                "{task_type}", rd["task_type"]
            )
            rubric_msg = [{"role": "user", "content": rubric_content}]
            rubric_raw = grader_processing_class.apply_chat_template(
                rubric_msg, add_generation_prompt=True, tokenize=False,
            )
            rubric_prompts_legacy.append(rubric_raw)
            rubric_indices.append(rd["idx"])

        if not rubric_prompts_legacy:
            return df

        print(f"[Stage A] Generating rubrics for {len(rubric_prompts_legacy)} tasks...")
        rubric_outputs = rollout.grade_batch(
            rubric_prompts_legacy, grader_base_url,
            grader_model_name=grader_model_name,
        )

        valid_mask = []
        for i, rubric_output in enumerate(rubric_outputs):
            idx = rubric_indices[i]
            rubric_result = parse_rubric_output(
                rubric_output, output_format=rubric_format
            )
            if rubric_result.is_valid and 3 <= len(rubric_result.rubrics) <= 10:
                rm = df.at[idx, "reward_model"]
                if isinstance(rm, str):
                    rm = json.loads(rm)
                rm = deepcopy(rm)
                rm["ground_truth"]["rubrics"] = rubric_result.rubrics
                rm["ground_truth"]["priorities"] = rubric_result.priorities
                df.at[idx, "reward_model"] = rm

                meta = df.at[idx, "metadata"]
                if isinstance(meta, str):
                    meta = json.loads(meta)
                meta = deepcopy(meta)
                meta["num_rubrics"] = len(rubric_result.rubrics)
                meta["rubric_raw_output"] = rubric_output
                df.at[idx, "metadata"] = meta
                valid_mask.append(True)
            else:
                valid_mask.append(False)

    n_valid = sum(valid_mask)
    n_total = len(rows_data)
    print(
        f"[Stage A] Rubric generation: {n_valid}/{n_total} valid "
        f"({100 * n_valid / max(n_total, 1):.1f}%)"
    )

    valid_indices = {
        rows_data[i]["idx"] for i, v in enumerate(valid_mask) if v
    }
    return df.loc[df.index.isin(valid_indices)].reset_index(drop=True)


def run_solver_rollouts(
    df: pd.DataFrame,
    solver_template: str,
    rollout: BatchMultiTurnRollout,
    K: int,
    max_search_turns: int = 4,
) -> list[list[list[dict[str, str]]]]:
    """Stage B: Run K solver rollouts per task.

    Args:
        df: DataFrame with ``prompt`` column containing solver messages.
        solver_template: Solver template (unused here, prompts already formatted).
        rollout: BatchMultiTurnRollout instance.
        K: Number of solver rollouts per task.
        max_search_turns: Max search turns (for informational logging).

    Returns:
        list[list[list[dict[str, str]]]]: Per-task list of K conversation
            histories (each history is a list of message dicts).
    """
    # Build repeated messages for K rollouts per task
    all_messages: list[list[dict[str, str]]] = []
    for _, row in df.iterrows():
        prompt_msgs = row["prompt"]
        if isinstance(prompt_msgs, str):
            prompt_msgs = json.loads(prompt_msgs)
        for _ in range(K):
            all_messages.append(deepcopy(prompt_msgs))

    n_tasks = len(df)
    print(
        f"[Stage B] Running {K} solver rollouts x {n_tasks} tasks = "
        f"{len(all_messages)} total conversations..."
    )

    if not all_messages:
        return []

    histories = rollout.generate_sequences_batch(all_messages)

    # Group by task
    grouped: list[list[list[dict[str, str]]]] = []
    for i in range(0, len(histories), K):
        grouped.append(histories[i : i + K])

    return grouped


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


def grade_rollouts(
    df: pd.DataFrame,
    solver_histories: list[list[list[dict[str, str]]]],
    grader_template: str,
    rollout: BatchMultiTurnRollout,
    grader_base_url: str,
    processing_class: Any,
    grader_format: str = "v2",
    grader_retry: bool = False,
    grader_model_name: str | None = None,
    grader_processing_class: Any = None,
    grader_per_rubric: bool = False,
) -> list[list[str]]:
    """Stage C: Grade all solver rollouts.

    Args:
        df: DataFrame with reward_model containing rubrics.
        solver_histories: Per-task list of K conversation histories.
        grader_template: Template with ``{prompt}``, ``{response}``,
            ``{rubric_list}`` placeholders (or ``{rubric}`` for per-rubric).
        rollout: BatchMultiTurnRollout instance.
        grader_base_url: Base URL for grader server.
        processing_class: Tokenizer for chat template formatting.
        grader_format: Grader output format (``"v2"``, ``"xml"``, ``"json"``).
        grader_retry: Whether to retry failed grader parses.
        grader_model_name: Model name for grader server. When ``None``,
            ``grade_batch`` uses the rollout's default model.
        grader_processing_class: Tokenizer for the grader model. When
            ``None``, falls back to ``processing_class``.
        grader_per_rubric: Whether to grade each rubric independently
            in its own LLM call (per-rubric mode).

    Returns:
        list[list[str]]: Per-task list of K grader output strings.
    """
    if grader_processing_class is None:
        grader_processing_class = processing_class
    grader_prompts: list[str] = []
    grader_raw_contents: list[str] = []
    grader_expected_rubrics: list[int] = []
    # Maps grader prompt index -> flat rollout index for reconstruction
    grader_to_flat: list[int] = []
    # Per-rubric mode: flat map from grader prompt -> (flat_rollout_idx, rubric_idx)
    flat_grader_map: list[tuple[int, int]] = []
    n_skipped = 0

    K = len(solver_histories[0]) if solver_histories else 0
    total_rollouts = sum(len(hists) for hists in solver_histories)

    for task_i, row in df.iterrows():
        rm = row["reward_model"]
        if isinstance(rm, str):
            rm = json.loads(rm)
        gt = rm["ground_truth"]

        # Skip items with EM target (short_form_qa) — no rubrics to grade.
        # Their difficulty is computed via EM in compute_difficulty_scores().
        if gt.get("target") is not None:
            continue

        rubrics = gt["rubrics"]

        # Extract task prompt
        prompt_msgs = row["prompt"]
        if isinstance(prompt_msgs, str):
            prompt_msgs = json.loads(prompt_msgs)
        task_prompt = ""
        if prompt_msgs and isinstance(prompt_msgs, list):
            content = prompt_msgs[0].get("content", "")
            if "Question:" in content:
                task_prompt = content.split("Question:")[-1].strip()
            else:
                task_prompt = content

        rubric_list = "\n".join(
            f"{i + 1}. {r}" for i, r in enumerate(rubrics)
        )

        for k, hist in enumerate(solver_histories[task_i]):
            flat_idx = task_i * K + k
            raw_resp = _get_last_assistant_content(hist)
            solver_result = parse_solver_response(raw_resp)
            if not solver_result.is_valid:
                n_skipped += 1
                continue

            if grader_per_rubric:
                # Per-rubric: one prompt per (solver_response, rubric)
                for rubric_idx, rubric_text in enumerate(rubrics):
                    grader_content = grader_template.format(
                        prompt=task_prompt,
                        response=solver_result.answer,
                        rubric=f"{rubric_idx + 1}. {rubric_text}",
                    )
                    grader_msg = [{"role": "user", "content": grader_content}]
                    grader_raw = grader_processing_class.apply_chat_template(
                        grader_msg, add_generation_prompt=True, tokenize=False,
                    )
                    grader_prompts.append(grader_raw)
                    grader_raw_contents.append(grader_content)
                    flat_grader_map.append((flat_idx, rubric_idx))
                grader_to_flat.append(flat_idx)
                grader_expected_rubrics.append(len(rubrics))
            else:
                grader_content = grader_template.format(
                    prompt=task_prompt,
                    response=solver_result.answer,
                    rubric_list=rubric_list,
                )
                grader_msg = [{"role": "user", "content": grader_content}]
                grader_raw = grader_processing_class.apply_chat_template(
                    grader_msg, add_generation_prompt=True, tokenize=False,
                )
                grader_prompts.append(grader_raw)
                grader_raw_contents.append(grader_content)
                grader_expected_rubrics.append(len(rubrics))
                grader_to_flat.append(flat_idx)

    n_valid = len(grader_prompts)
    print(
        f"[Stage C] Grading {n_valid} solver responses "
        f"({n_skipped} skipped, no valid answer)"
        f"{' [per-rubric mode]' if grader_per_rubric else ''}..."
    )

    if not grader_prompts:
        return [[] for _ in range(len(df))]

    raw_grader_outputs = rollout.grade_batch(
        grader_prompts, grader_base_url, grader_model_name=grader_model_name,
    )

    if grader_per_rubric:
        # Per-rubric retry for failed individual calls
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
                retry_raw = grader_processing_class.apply_chat_template(
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
                f"[Stage C] Per-rubric retry: {len(retry_flat_indices)} failed, "
                f"{n_retry_ok} recovered"
            )

        # Unmap flat outputs -> per-rollout synthetic <scores> strings
        per_rubric_raw: dict[int, dict[int, str]] = defaultdict(dict)
        for fi, (flat_i, rubric_i) in enumerate(flat_grader_map):
            per_rubric_raw[flat_i][rubric_i] = raw_grader_outputs[fi]

        all_outputs: list[str] = [""] * total_rollouts
        for flat_i, rubric_outputs in per_rubric_raw.items():
            # Find num_rubrics from the task
            task_i = flat_i // K if K > 0 else 0
            rm = df.iloc[task_i]["reward_model"]
            if isinstance(rm, str):
                rm = json.loads(rm)
            n_rubrics = len(rm["ground_truth"]["rubrics"])
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
            all_outputs[flat_i] = f"<scores>{', '.join(scores_parts)}</scores>"
    else:
        # Grader retry (v2 format only)
        if grader_retry and grader_format == "v2" and raw_grader_outputs:
            retry_indices: list[int] = []
            for gi, output in enumerate(raw_grader_outputs):
                parsed = parse_grader_response(output, output_format="v2")
                if not parsed.is_valid or parsed.num_assessments != grader_expected_rubrics[gi]:
                    retry_indices.append(gi)

            if retry_indices:
                retry_prompts: list[str] = []
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
                    retry_raw = grader_processing_class.apply_chat_template(
                        retry_conv, add_generation_prompt=True, tokenize=False,
                    )
                    retry_prompts.append(retry_raw)

                retry_outputs = rollout.grade_batch(
                    retry_prompts, grader_base_url,
                    grader_model_name=grader_model_name,
                )
                for ri, gi in enumerate(retry_indices):
                    raw_grader_outputs[gi] = retry_outputs[ri]

                print(
                    f"[Stage C] Grader retry: {len(retry_indices)} failed parses retried"
                )

        # Reconstruct full flat list (empty string for skipped rollouts)
        all_outputs: list[str] = [""] * total_rollouts
        for gi, flat_i in enumerate(grader_to_flat):
            all_outputs[flat_i] = raw_grader_outputs[gi]

    # Group by task
    grouped: list[list[str]] = []
    for i in range(0, total_rollouts, K):
        grouped.append(all_outputs[i : i + K])

    return grouped


def compute_difficulty_scores(
    df: pd.DataFrame,
    grader_outputs: list[list[str]],
    solver_histories: list[list[list[dict[str, str]]]] | None = None,
    grader_format: str = "v2",
    grader_min_coverage: float = 0.5,
    grader_pad_value: float = 0.0,
    difficulty_fn: str = "tent",
    difficulty_target: float = 0.5,
) -> list[float]:
    """Stage D: Compute difficulty scores from grader outputs.

    For rubric-based items, difficulty is computed from grader scores.
    For EM-based items (``ground_truth.target`` set, e.g. short_form_qa),
    difficulty is computed via exact match of solver answers against the
    gold target: ``(n - num_correct) / (n - 1)``.

    Args:
        df: DataFrame with reward_model containing rubrics.
        grader_outputs: Per-task list of K grader output strings.
        solver_histories: Per-task list of K solver conversation histories.
            Required for EM-based difficulty (items with target set).
            When ``None``, EM items get difficulty 0.0.
        grader_format: Grader output format.
        grader_min_coverage: Minimum rubric coverage for grader recovery.
        grader_pad_value: Pad value for missing rubric assessments.
        difficulty_fn: Difficulty function (``"tent"`` or ``"continuous"``).
        difficulty_target: Target normalized score for tent function.

    Returns:
        tuple[list[float], list[dict | None]]: Per-task difficulty scores and
            optional per-task detail dicts.  For EM items the detail dict
            contains ``em_results`` (list of 0/1 per rollout),
            ``solver_answers`` (raw answers), ``gold``, and ``num_correct``.
            For rubric items the entry is ``None``.
    """
    scores: list[float] = []
    details: list[dict | None] = []

    for pos, (_task_i, row) in enumerate(df.iterrows()):
        rm = row["reward_model"]
        if isinstance(rm, str):
            rm = json.loads(rm)
        gt = rm["ground_truth"]

        # EM-based difficulty for items with target set (short_form_qa).
        # Parse solver answers, compare against gold target via exact match.
        if gt.get("target") is not None:
            gold = gt["target"]
            if not gold or solver_histories is None:
                scores.append(0.0)
                details.append({"em_results": [], "solver_answers": [],
                                "gold": gold or "", "num_correct": 0})
                continue

            gold_norm = normalize_answer(gold)
            em_results: list[float] = []
            solver_answers: list[str] = []
            for hist in solver_histories[pos]:
                raw_resp = _get_last_assistant_content(hist)
                solver_result = parse_solver_response(raw_resp)
                if solver_result.is_valid and solver_result.answer:
                    pred_norm = normalize_answer(solver_result.answer)
                    em_results.append(float(pred_norm == gold_norm))
                    solver_answers.append(solver_result.answer)
                else:
                    em_results.append(0.0)
                    solver_answers.append("")

            n = len(em_results)
            num_correct = sum(em_results)
            if n <= 1 or num_correct == 0 or num_correct == n:
                difficulty = 0.0
            else:
                difficulty = (n - num_correct) / (n - 1)
            scores.append(difficulty)
            details.append({
                "em_results": em_results,
                "solver_answers": solver_answers,
                "gold": gold,
                "num_correct": int(num_correct),
            })
            continue

        num_rubrics = len(gt["rubrics"])

        valid_sums: list[float] = []
        for grader_output in grader_outputs[pos]:
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
        else:
            difficulty = 0.0

        scores.append(difficulty)
        details.append(None)

    return scores, details


def filter_by_prompt_length(
    df: pd.DataFrame,
    processing_class: Any,
    max_prompt_length: int,
    tool_schemas: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Filter tasks whose rendered solver prompt exceeds max_prompt_length tokens.

    Renders each task's prompt through the chat template (with tool schemas
    if provided) and drops tasks that exceed the token limit.  This prevents
    solver training crashes from over-length prompts.

    Args:
        df: DataFrame with ``prompt`` column containing solver message lists.
        processing_class: Tokenizer used for chat template rendering.
        max_prompt_length: Maximum allowed prompt length in tokens.
        tool_schemas: Optional tool schemas included in chat template
            rendering (adds tool description tokens to the prompt).

    Returns:
        pd.DataFrame: Filtered DataFrame with over-length tasks removed.
    """
    keep = []
    dropped_lengths: list[int] = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        rendered = processing_class.apply_chat_template(
            prompt, tools=tool_schemas, tokenize=True,
            add_generation_prompt=True,
        )
        if len(rendered) <= max_prompt_length:
            keep.append(True)
        else:
            keep.append(False)
            dropped_lengths.append(len(rendered))

    if dropped_lengths:
        print(
            f"Prompt length filter: dropped {len(dropped_lengths)}/{len(df)} tasks "
            f"exceeding {max_prompt_length} tokens "
            f"(max={max(dropped_lengths)}, min={min(dropped_lengths)})"
        )

    return df[keep].reset_index(drop=True)


def filter_by_difficulty(
    df: pd.DataFrame,
    difficulty_scores: list[float],
    difficulty_min: float = 0.2,
    difficulty_max: float = 0.8,
    difficulty_details: list[dict | None] | None = None,
) -> pd.DataFrame:
    """Filter DataFrame by difficulty score range.

    Args:
        df: Input DataFrame.
        difficulty_scores: Per-task difficulty scores.
        difficulty_min: Minimum difficulty (inclusive).
        difficulty_max: Maximum difficulty (inclusive).
        difficulty_details: Optional per-task detail dicts from
            ``compute_difficulty_scores``.  EM items include
            ``em_results``, ``solver_answers``, ``gold``, ``num_correct``.

    Returns:
        pd.DataFrame: Filtered DataFrame with difficulty metadata added.
    """
    keep_mask = []
    for i, score in enumerate(difficulty_scores):
        # Update metadata with difficulty
        meta = df.iloc[i]["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        meta = deepcopy(meta)
        meta["difficulty_score"] = score
        if difficulty_details and difficulty_details[i] is not None:
            meta["difficulty_detail"] = difficulty_details[i]
        df.iat[i, df.columns.get_loc("metadata")] = meta

        keep_mask.append(difficulty_min <= score <= difficulty_max)

    filtered = df[keep_mask].reset_index(drop=True)
    return filtered


def cap_tasks_per_prompt(
    df: pd.DataFrame,
    max_per_prompt: int,
    prompt_counts: dict[int, int] | None = None,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Cap the number of tasks kept per source prompt.

    After difficulty filtering, multiple tasks from the same input prompt may
    survive.  This function limits how many are retained, preventing
    over-representation of any single prompt/context.

    Args:
        df: DataFrame with a ``metadata`` column containing a
            ``source_item_idx`` key that identifies the originating prompt.
        max_per_prompt: Maximum tasks to keep per source prompt.
        prompt_counts: Running counts from previous batches.  Pass ``None``
            (or ``{}``) on the first call; the returned dict should be
            forwarded to subsequent calls for cross-batch correctness.

    Returns:
        tuple[pd.DataFrame, dict[int, int]]: A tuple of the filtered
            DataFrame and the updated prompt counts dict.
    """
    if prompt_counts is None:
        prompt_counts = {}

    keep_mask: list[bool] = []
    for _, row in df.iterrows():
        meta = row.get("metadata", {})
        if isinstance(meta, str):
            meta = json.loads(meta)

        source_idx = meta.get("source_item_idx")
        if source_idx is None:
            # No source tracking — keep the row unconditionally
            keep_mask.append(True)
            continue

        current = prompt_counts.get(source_idx, 0)
        if current < max_per_prompt:
            prompt_counts[source_idx] = current + 1
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    filtered = df[keep_mask].reset_index(drop=True)
    removed = len(df) - len(filtered)
    if removed > 0:
        print(
            f"[Per-prompt cap] Removed {removed} tasks exceeding "
            f"max_tasks_per_prompt={max_per_prompt} "
            f"({len(filtered)} kept from {len(df)})"
        )
    return filtered, prompt_counts


def filter_by_quality(
    df: pd.DataFrame,
    quality_templates: list[str],
    rollout: BatchMultiTurnRollout,
    grader_base_url: str,
    processing_class: Any,
    grader_model_name: str | None = None,
    grader_processing_class: Any = None,
    max_tokens: int = 256,
    required_sum: int = -1,
    quality_gate_names: list[str] | None = None,
    quality_retrieval_multi_template: str | None = None,
) -> pd.DataFrame:
    """Stage 0: Filter tasks by quality grading before rubric generation.

    Runs binary graders on each task prompt for the active quality gates
    and keeps only rows where all active gates score ``1``.

    Args:
        df: Input DataFrame with ``reward_model`` column containing
            ``ground_truth.task_prompt``, or ``prompt`` column with
            solver messages from which the task prompt can be extracted.
        quality_templates: List of template strings (one per active gate),
            each with a ``{task_prompt}`` placeholder.
        rollout: BatchMultiTurnRollout instance for LLM calls.
        grader_base_url: Base URL for the grader server.
        processing_class: Tokenizer for chat template formatting.
        grader_model_name: Model name for grader server. When ``None``,
            ``grade_batch`` uses the rollout's default model.
        grader_processing_class: Tokenizer for the grader model. When
            ``None``, falls back to ``processing_class``.
        max_tokens: Maximum tokens for quality grader responses.
        required_sum: Required sum of scores to pass. Default ``-1``
            means all active gates must pass.
        quality_gate_names: Names of active gates, matching order of
            ``quality_templates``. Used for metadata column names.
            Defaults to ``["entity", "no_leakage", "retrieval"]``
            for backward compatibility.

    Returns:
        pd.DataFrame: Filtered DataFrame with ``quality_<gate>`` and
            ``quality_sum`` metadata columns added.
    """
    if grader_processing_class is None:
        grader_processing_class = processing_class

    n_gates = len(quality_templates)
    if quality_gate_names is None:
        quality_gate_names = ["entity", "no_leakage", "retrieval"][:n_gates]
    if required_sum < 0:
        required_sum = n_gates

    # Extract task prompts, source documents, and search turn data from each row
    task_prompts: list[str] = []
    source_docs: list[str] = []
    search_turn_docs: list[str] = []
    search_turn_counts: list[int] = []
    source_titles: list[str] = []
    for _, row in df.iterrows():
        rm = row.get("reward_model", {})
        if isinstance(rm, str):
            rm = json.loads(rm)
        gt = rm.get("ground_truth", {})
        task_prompt = gt.get("task_prompt", "")

        # Fallback: extract from solver prompt messages
        if not task_prompt:
            prompt_msgs = row.get("prompt", [])
            if isinstance(prompt_msgs, str):
                prompt_msgs = json.loads(prompt_msgs)
            if isinstance(prompt_msgs, np.ndarray):
                prompt_msgs = prompt_msgs.tolist()
            if prompt_msgs and isinstance(prompt_msgs, list):
                content = prompt_msgs[0].get("content", "")
                if "Question:" in content:
                    task_prompt = content.split("Question:")[-1].strip()
                else:
                    task_prompt = content

        task_prompts.append(task_prompt)

        # Parse metadata once for source doc fallback and search turn extraction
        meta = row.get("metadata", {})
        if isinstance(meta, str):
            meta = json.loads(meta) if meta else {}
        raw_ctx = meta.get("raw_context", "") if isinstance(meta, dict) else ""

        # Extract source document for source_relevance gate
        ei = row.get("extra_info", {})
        if isinstance(ei, str):
            ei = json.loads(ei) if ei else {}
        if isinstance(ei, dict):
            source_doc = ei.get("source_document", "")
        else:
            source_doc = ""
        if not source_doc and raw_ctx:
            m = re.search(
                r"## Source Document\s*\n(.*?)(?=\n## |\Z)",
                raw_ctx, re.DOTALL,
            )
            if m:
                source_doc = m.group(1).strip()
        source_docs.append(source_doc)

        # Extract source title
        _title_m = re.search(r'\(Title: "?(.*?)"?\)', source_doc)
        source_titles.append(_title_m.group(1) if _title_m else "unknown")

        # Extract per-search-turn docs and count for retrieval gate routing
        retrieved = extract_per_search_turn_docs(raw_ctx) if raw_ctx else ""
        n_turns = len(re.findall(SEARCH_R1_PATTERN, raw_ctx, re.DOTALL)) if raw_ctx else 0
        search_turn_docs.append(retrieved)
        search_turn_counts.append(n_turns)

    # Build n_gates * N quality grading prompts
    quality_prompts: list[str] = []
    for i, task_prompt in enumerate(task_prompts):
        for gate_idx, tmpl in enumerate(quality_templates):
            # Routed retrieval gate: swap template for multi-search tasks
            effective_tmpl = tmpl
            if (
                quality_retrieval_multi_template
                and gate_idx < len(quality_gate_names)
                and quality_gate_names[gate_idx] == "retrieval"
                and search_turn_counts[i] >= 2
            ):
                effective_tmpl = quality_retrieval_multi_template
            content = (
                effective_tmpl.replace("{task_prompt}", task_prompt)
                .replace("{source_document}", source_docs[i])
                .replace("{search_turns}", search_turn_docs[i])
                .replace("{source_title}", source_titles[i])
            )
            msg = [{"role": "user", "content": content}]
            raw = grader_processing_class.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False,
            )
            quality_prompts.append(raw)

    if not quality_prompts:
        return df

    n_tasks = len(task_prompts)
    print(
        f"[Stage 0] Quality grading {n_tasks} tasks "
        f"({len(quality_prompts)} prompts, gates={quality_gate_names})..."
    )

    quality_outputs = rollout.grade_batch(
        quality_prompts, grader_base_url,
        max_tokens=max_tokens,
        grader_model_name=grader_model_name,
    )

    # Parse scores and compute per-task quality gate
    parsed_scores = [parse_quality_score(o) for o in quality_outputs]

    # Build per-gate columns dynamically
    gate_cols: dict[str, list[int | None]] = {g: [] for g in quality_gate_names}
    quality_sum_col: list[int] = []
    keep_mask: list[bool] = []

    for i in range(n_tasks):
        task_scores = parsed_scores[i * n_gates : (i + 1) * n_gates]
        for j, gate_name in enumerate(quality_gate_names):
            gate_cols[gate_name].append(task_scores[j])

        gate = compute_quality_gate(task_scores, required=required_sum)
        if any(s is None for s in task_scores):
            quality_sum_col.append(-1)
        else:
            quality_sum_col.append(sum(task_scores))  # type: ignore[arg-type]
        keep_mask.append(gate)

    # Add metadata columns
    df = df.copy()
    for gate_name in quality_gate_names:
        df[f"quality_{gate_name}"] = gate_cols[gate_name]
    df["quality_sum"] = quality_sum_col

    n_passed = sum(keep_mask)
    print(
        f"[Stage 0] Quality filter: {n_passed}/{n_tasks} passed "
        f"({100 * n_passed / max(n_tasks, 1):.1f}%)"
    )

    filtered = df[keep_mask].reset_index(drop=True)
    return filtered


def _run_solver_stages(
    df: pd.DataFrame,
    args: argparse.Namespace,
    rollout: BatchMultiTurnRollout,
    rubric_template: str,
    solver_template: str,
    processing_class: Any,
    K: int,
    grader_model_name: str | None = None,
    grader_processing_class: Any = None,
    quality_templates: list[str] | None = None,
    quality_gate_names: list[str] | None = None,
    quality_retrieval_multi_template: str | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, list[list[list[dict[str, str]]]]] | None:
    """Run pipeline stages 0–B: quality filter, rubric gen, solver rollouts.

    Separated from grader stages to allow async overlap: the caller can
    run grader stages for batch N-1 in a background thread while this
    function runs solver stages for batch N in the main thread.

    Args:
        df: Input DataFrame chunk to process.
        args: Parsed CLI arguments.
        rollout: BatchMultiTurnRollout instance.
        rubric_template: Rubric generation template string.
        solver_template: Solver prompt template string.
        processing_class: Tokenizer instance.
        K: Number of solver rollouts per task.
        grader_model_name: Model name for grader server.
        grader_processing_class: Tokenizer for the grader model.
        quality_templates: List of quality grading template strings.
        quality_gate_names: Names of active quality gates.
        quality_retrieval_multi_template: Multi-retrieval quality
            template string.
        tool_schemas: Optional tool schemas for prompt length calculation.

    Returns:
        Tuple of ``(prepared_df, solver_histories)`` or ``None`` if df
        is empty after quality/rubric filtering.
    """
    if grader_processing_class is None:
        grader_processing_class = processing_class

    # Stage -1: Prompt length filtering (cheap, no LLM calls)
    solver_prompt_length_limit = getattr(args, "solver_prompt_length_limit", 0)
    if solver_prompt_length_limit > 0:
        df = filter_by_prompt_length(
            df, processing_class, solver_prompt_length_limit,
            tool_schemas=tool_schemas,
        )
        print(f"After prompt length filtering: {len(df)} tasks remain")
        if len(df) == 0:
            return None

    # Stage 0: Quality filtering (before rubric generation to save compute)
    if (
        getattr(args, "quality_enabled", False)
        and quality_templates
    ):
        df = filter_by_quality(
            df, quality_templates, rollout, args.grader_base_url,
            processing_class, grader_model_name=grader_model_name,
            grader_processing_class=grader_processing_class,
            max_tokens=getattr(args, "quality_max_tokens", 256),
            required_sum=getattr(args, "quality_required_sum", len(quality_templates)),
            quality_gate_names=quality_gate_names,
            quality_retrieval_multi_template=quality_retrieval_multi_template,
        )
        print(f"After quality filtering: {len(df)} tasks remain")
        if len(df) == 0:
            return None

    # Stage A: Rubric generation
    needs_rubrics = any(
        _row_needs_rubrics(row) for _, row in df.iterrows()
    )
    if needs_rubrics:
        df = generate_rubrics(
            df, rubric_template, rollout, args.grader_base_url,
            processing_class, rubric_format=args.rubric_format,
            grader_model_name=grader_model_name,
            grader_processing_class=grader_processing_class,
            rubric_mode=getattr(args, "rubric_mode", "single"),
            initial_template=getattr(args, "_rubric_initial_template", None),
            turn_template=getattr(args, "_rubric_turn_template", None),
            synthesis_template=getattr(args, "_rubric_synthesis_template", None),
        )
        print(f"After rubric generation: {len(df)} tasks remain")

    if len(df) == 0:
        print("No tasks remaining after rubric generation.")
        return None

    # Stage B: Solver rollouts
    solver_histories = run_solver_rollouts(
        df, solver_template, rollout, K,
        max_search_turns=args.max_search_turns,
    )
    return df, solver_histories


def _run_grader_stages(
    df: pd.DataFrame,
    solver_histories: list[list[list[dict[str, str]]]],
    args: argparse.Namespace,
    grader_template: str,
    rollout: BatchMultiTurnRollout,
    processing_class: Any,
    grader_format: str,
    grader_model_name: str | None = None,
    grader_processing_class: Any = None,
) -> pd.DataFrame:
    """Run pipeline stages C–D: grade solver rollouts and filter by difficulty.

    Designed to run in a background thread while the next batch's solver
    stages run in the main thread.

    Args:
        df: DataFrame of tasks (post quality/rubric filtering).
        solver_histories: Solver rollout histories from
            :func:`_run_solver_stages`.
        args: Parsed CLI arguments.
        grader_template: Grader prompt template string.
        rollout: BatchMultiTurnRollout instance.
        processing_class: Tokenizer instance.
        grader_format: Grader output format (``"xml"``, ``"json"``,
            ``"v2"``).
        grader_model_name: Model name for grader server.
        grader_processing_class: Tokenizer for the grader model.

    Returns:
        pd.DataFrame: Difficulty-filtered DataFrame.
    """
    if grader_processing_class is None:
        grader_processing_class = processing_class

    # Stage C: Grading
    grader_per_rubric = getattr(args, "grader_per_rubric", False)
    if not grader_per_rubric and "grader_per_rubric" in (args.grader_template_path or ""):
        grader_per_rubric = True
    grader_outputs = grade_rollouts(
        df, solver_histories, grader_template, rollout,
        args.grader_base_url, processing_class,
        grader_format=grader_format,
        grader_retry=args.grader_retry,
        grader_model_name=grader_model_name,
        grader_processing_class=grader_processing_class,
        grader_per_rubric=grader_per_rubric,
    )

    # Stage D: Difficulty filtering
    difficulty_scores, difficulty_details = compute_difficulty_scores(
        df, grader_outputs,
        solver_histories=solver_histories,
        grader_format=grader_format,
        grader_min_coverage=args.grader_min_coverage,
        grader_pad_value=args.grader_pad_value,
        difficulty_fn=args.difficulty_fn,
        difficulty_target=args.difficulty_target,
    )

    print(f"\n=== Difficulty scores ===")
    if difficulty_scores:
        avg_diff = sum(difficulty_scores) / len(difficulty_scores)
        n_in_range = sum(
            1 for s in difficulty_scores
            if args.difficulty_min <= s <= args.difficulty_max
        )
        print(
            f"Mean: {avg_diff:.4f}, "
            f"In range [{args.difficulty_min}, {args.difficulty_max}]: "
            f"{n_in_range}/{len(difficulty_scores)}"
        )

    return filter_by_difficulty(
        df, difficulty_scores,
        difficulty_min=args.difficulty_min,
        difficulty_max=args.difficulty_max,
        difficulty_details=difficulty_details,
    )


def _run_pipeline_stages(
    df: pd.DataFrame,
    args: argparse.Namespace,
    rollout: BatchMultiTurnRollout,
    rubric_template: str,
    solver_template: str,
    grader_template: str,
    processing_class: Any,
    grader_format: str,
    K: int,
    grader_model_name: str | None = None,
    grader_processing_class: Any = None,
    quality_templates: list[str] | None = None,
    quality_gate_names: list[str] | None = None,
    quality_retrieval_multi_template: str | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Run pipeline stages 0 through D on a DataFrame chunk (sequential).

    Thin wrapper that calls :func:`_run_solver_stages` followed by
    :func:`_run_grader_stages`. Used by the non-batched code path.

    Args:
        df: Input DataFrame chunk to process.
        args: Parsed CLI arguments.
        rollout: BatchMultiTurnRollout instance.
        rubric_template: Rubric generation template string.
        solver_template: Solver prompt template string.
        grader_template: Grader prompt template string.
        processing_class: Tokenizer instance.
        grader_format: Grader output format (``"xml"``, ``"json"``, ``"v2"``).
        K: Number of solver rollouts per task.
        grader_model_name: Model name for grader server. When ``None``,
            ``grade_batch`` uses the rollout's default model.
        grader_processing_class: Tokenizer for the grader model. When
            ``None``, falls back to ``processing_class``.
        quality_templates: List of quality grading template strings
            (one per active gate). When ``None`` or empty, quality
            filtering is skipped.
        quality_gate_names: Names of active quality gates, matching
            order of ``quality_templates``.
        quality_retrieval_multi_template: Multi-retrieval quality
            template string.

    Returns:
        pd.DataFrame: Filtered DataFrame after difficulty filtering.
    """
    result = _run_solver_stages(
        df, args, rollout, rubric_template, solver_template,
        processing_class, K,
        grader_model_name=grader_model_name,
        grader_processing_class=grader_processing_class,
        quality_templates=quality_templates,
        quality_gate_names=quality_gate_names,
        quality_retrieval_multi_template=quality_retrieval_multi_template,
        tool_schemas=tool_schemas,
    )
    if result is None:
        return df.iloc[:0].reset_index(drop=True)

    prepared_df, solver_histories = result
    return _run_grader_stages(
        prepared_df, solver_histories, args, grader_template, rollout,
        processing_class, grader_format,
        grader_model_name=grader_model_name,
        grader_processing_class=grader_processing_class,
    )


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    """Run the full evaluation pipeline.

    Supports optional early-stop: when ``--target_filtered`` and
    ``--batch_size`` are both set, tasks are processed in pipeline batches
    and processing stops once enough filtered results accumulate.

    Args:
        args: Parsed command-line arguments.

    Returns:
        pd.DataFrame: Filtered output DataFrame.
    """
    from transformers import AutoTokenizer

    # Load data — use pyarrow directly to handle both legacy (nested Python
    # objects) and new (JSON-serialized) parquet files.  combine_chunks()
    # resolves the ArrowNotImplementedError for chunked arrays with nested types.
    import pyarrow.parquet as pq

    table = pq.read_table(args.input_path)
    table = table.combine_chunks()
    df = table.to_pandas()
    print(f"Loaded {len(df)} tasks from {args.input_path}")

    if args.max_items and args.max_items > 0:
        df = df.head(args.max_items).reset_index(drop=True)
        print(f"Truncated to {len(df)} tasks (--max_items={args.max_items})")

    # Resolve batch_size default: use target_filtered if set, else 5000
    if args.batch_size is None:
        if args.target_filtered > 0:
            args.batch_size = args.target_filtered
        else:
            args.batch_size = 5000

    # Resolve grader model default
    if args.grader_model_name is None:
        args.grader_model_name = args.model_name

    # Load tokenizer
    processing_class = AutoTokenizer.from_pretrained(args.model_name)

    # Load grader tokenizer (separate when grader model differs from solver)
    if args.grader_model_name != args.model_name:
        grader_processing_class = AutoTokenizer.from_pretrained(args.grader_model_name)
    else:
        grader_processing_class = processing_class

    # Load templates
    rubric_template = Path(args.rubric_template_path).read_text()
    solver_template = Path(args.solver_template_path).read_text()
    grader_template = Path(args.grader_template_path).read_text()

    # V19 per-turn rubric templates (stored on args for _run_pipeline_stages)
    rubric_mode = getattr(args, "rubric_mode", "single")
    args._rubric_initial_template = None
    args._rubric_turn_template = None
    args._rubric_synthesis_template = None
    if rubric_mode == "v19_perturn":
        _rip = getattr(args, "rubric_initial_template_path", "")
        _rtp = getattr(args, "rubric_turn_template_path", "")
        _rsp = getattr(args, "rubric_synthesis_template_path", "")
        if _rip and _rtp and _rsp:
            args._rubric_initial_template = Path(_rip).read_text()
            args._rubric_turn_template = Path(_rtp).read_text()
            args._rubric_synthesis_template = Path(_rsp).read_text()
            print(f"[V19] Loaded per-turn rubric templates: {_rip}, {_rtp}, {_rsp}")
        else:
            print("[V19] Warning: rubric_mode=v19_perturn but template paths missing, falling back to single mode")
            args.rubric_mode = "single"

    # Resolve active quality gates
    _qg_str = getattr(args, "quality_gates", "").strip()
    if _qg_str:
        quality_gate_names = parse_quality_gates(_qg_str)
    elif getattr(args, "quality_enabled", False):
        quality_gate_names = ["entity", "no_leakage", "retrieval"]
    else:
        quality_gate_names = []

    # Override required_sum default to match number of active gates
    if getattr(args, "quality_required_sum", -1) < 0:
        args.quality_required_sum = len(quality_gate_names)
    # Ensure quality_enabled is consistent for downstream checks
    args.quality_enabled = len(quality_gate_names) > 0

    # Load quality templates (only for active gates)
    _gate_template_attr = {
        "entity": "quality_entity_template_path",
        "no_leakage": "quality_no_leakage_template_path",
        "retrieval": "quality_retrieval_template_path",
        "source_relevance": "quality_source_relevance_template_path",
    }
    quality_templates: list[str] | None = None
    quality_retrieval_multi_template: str | None = None
    if quality_gate_names:
        quality_paths = [
            getattr(args, _gate_template_attr[g], "")
            for g in quality_gate_names
        ]
        if all(p for p in quality_paths):
            quality_templates = [Path(p).read_text() for p in quality_paths]
            print(f"Loaded {len(quality_templates)} quality templates for gates: {quality_gate_names}")
        else:
            print(f"Warning: quality gates {quality_gate_names} but not all template paths set; skipping quality filter")

        # Routed retrieval gate: separate template for multi-search tasks (2+ turns)
        _retrieval_multi_path = getattr(args, "quality_retrieval_multi_template_path", "")
        if _retrieval_multi_path:
            if Path(_retrieval_multi_path).is_file():
                quality_retrieval_multi_template = Path(_retrieval_multi_path).read_text()
                print(f"Loaded retrieval multi-search template: {_retrieval_multi_path}")
            else:
                print(f"WARNING: quality_retrieval_multi_template_path set to {_retrieval_multi_path!r} but file not found")

    # Detect per-rubric grading mode early (needed for format auto-detection)
    grader_per_rubric = getattr(args, "grader_per_rubric", False)
    if not grader_per_rubric and "grader_per_rubric" in (args.grader_template_path or ""):
        grader_per_rubric = True

    # Auto-detect grader format from template content
    grader_format = args.grader_format
    if grader_format == "xml" and grader_template:
        if "<scores>" in grader_template:
            grader_format = "v2"
    # Per-rubric mode reconstructs synthetic <scores> strings → always v2
    if grader_per_rubric:
        grader_format = "v2"

    # Load tool schemas (skip for search_r1 format)
    tool_schemas: list[dict[str, Any]] | None = None
    if args.tool_config_path and args.solver_format != "search_r1":
        tool_schemas = _load_tool_schemas(args.tool_config_path)

    # Parse solver stop tokens
    solver_stop: list[str] | None = None
    if args.solver_stop_tokens:
        solver_stop = json.loads(args.solver_stop_tokens)

    K = args.reward_rollout_n

    with BatchMultiTurnRollout(
        solver_base_url=args.solver_base_url,
        retrieval_url=args.retrieval_url,
        processing_class=processing_class,
        model_name=args.model_name,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        max_prompt_length=args.max_prompt_length,
        tool_schemas=tool_schemas,
        solver_format=args.solver_format,
        solver_retry=args.solver_retry,
        solver_stop=solver_stop,
        timeout=args.timeout,
        max_retries=args.max_retries,
        grader_batch_size=getattr(args, "grader_batch_size", 10000),
        tool_response_truncation_unit="token",  # match training config (search_multiturn_grpo.yaml)
    ) as rollout:
        # Per-prompt cap tracking (shared across batches)
        prompt_counts: dict[int, int] = {}
        max_per_prompt = args.max_tasks_per_prompt

        if args.target_filtered > 0 and args.batch_size > 0:
            # Early-stop batch loop with solver/grader overlap:
            # Grader for batch N runs in a background thread while
            # solver for batch N+1 runs in the main thread.
            accumulated_dfs: list[pd.DataFrame] = []
            kept_total = 0
            pending_future: Future[pd.DataFrame] | None = None

            def _collect_grader_result() -> bool:
                """Collect grader result from previous batch.

                Returns:
                    True if early-stop target reached.
                """
                nonlocal pending_future, kept_total
                if pending_future is None:
                    return False

                filtered_batch = pending_future.result()
                pending_future = None

                if max_per_prompt > 0:
                    nonlocal prompt_counts
                    filtered_batch, prompt_counts = cap_tasks_per_prompt(
                        filtered_batch, max_per_prompt, prompt_counts,
                    )

                remaining = args.target_filtered - kept_total
                if len(filtered_batch) > remaining:
                    print(
                        "Trimming filtered batch: "
                        f"{len(filtered_batch)} -> {remaining} "
                        f"to reach exact target {args.target_filtered}"
                    )
                    filtered_batch = filtered_batch.head(
                        remaining,
                    ).reset_index(drop=True)

                accumulated_dfs.append(filtered_batch)
                kept_total += len(filtered_batch)
                print(
                    f"Cumulative filtered: {kept_total}/{args.target_filtered} target"
                )
                if kept_total >= args.target_filtered:
                    print(
                        f"Early stop: {kept_total} >= {args.target_filtered}"
                    )
                    return True
                return False

            with ThreadPoolExecutor(max_workers=1) as grader_pool:
                for batch_start in range(0, len(df), args.batch_size):
                    if kept_total >= args.target_filtered:
                        break

                    batch_df = df.iloc[
                        batch_start:batch_start + args.batch_size
                    ].reset_index(drop=True)
                    batch_num = batch_start // args.batch_size + 1
                    print(
                        f"\n=== Pipeline batch {batch_num}:"
                        f" rows {batch_start}-{batch_start + len(batch_df) - 1}"
                        f" ({len(batch_df)} tasks) ==="
                    )

                    # Stage 0+A+B: solver in main thread.
                    # Grader for previous batch runs concurrently
                    # in background.
                    solver_result = _run_solver_stages(
                        batch_df, args, rollout,
                        rubric_template, solver_template,
                        processing_class, K,
                        grader_model_name=args.grader_model_name,
                        grader_processing_class=grader_processing_class,
                        quality_templates=quality_templates,
                        quality_gate_names=quality_gate_names,
                        quality_retrieval_multi_template=quality_retrieval_multi_template,
                        tool_schemas=tool_schemas,
                    )

                    # Collect grader result from previous batch (the
                    # grader has had the full solver-batch duration to
                    # run concurrently).
                    if _collect_grader_result():
                        break

                    # Submit Stage C+D (grader) for current batch to
                    # background thread.
                    if solver_result is not None:
                        prepared_df, solver_histories = solver_result
                        pending_future = grader_pool.submit(
                            _run_grader_stages,
                            prepared_df, solver_histories,
                            args, grader_template, rollout,
                            processing_class, grader_format,
                            grader_model_name=args.grader_model_name,
                            grader_processing_class=grader_processing_class,
                        )

                # Drain final batch's grader after the loop ends.
                _collect_grader_result()

            if accumulated_dfs:
                filtered_df = pd.concat(
                    accumulated_dfs, ignore_index=True,
                )
            else:
                filtered_df = pd.DataFrame(columns=df.columns)
        else:
            filtered_df = _run_pipeline_stages(
                df, args, rollout,
                rubric_template, solver_template, grader_template,
                processing_class, grader_format, K,
                grader_model_name=args.grader_model_name,
                grader_processing_class=grader_processing_class,
                quality_templates=quality_templates,
                quality_gate_names=quality_gate_names,
                quality_retrieval_multi_template=quality_retrieval_multi_template,
                tool_schemas=tool_schemas,
            )
            # Apply per-prompt cap if configured (non-batched path)
            if max_per_prompt > 0:
                filtered_df, prompt_counts = cap_tasks_per_prompt(
                    filtered_df, max_per_prompt, prompt_counts,
                )

    if args.target_filtered > 0 and len(filtered_df) > args.target_filtered:
        print(
            "Capping filtered output: "
            f"{len(filtered_df)} -> {args.target_filtered} "
            f"(--target_filtered)"
        )
        filtered_df = filtered_df.head(args.target_filtered).reset_index(drop=True)

    print(
        f"\n=== Filtering complete ===\n"
        f"Input: {len(df)} tasks\n"
        f"Output: {len(filtered_df)} tasks "
        f"({100 * len(filtered_df) / max(len(df), 1):.1f}%)\n"
        f"Difficulty range: [{args.difficulty_min}, {args.difficulty_max}]"
    )

    return filtered_df


def _row_needs_rubrics(row: Any) -> bool:
    """Check if a row needs rubric generation.

    Rows with a non-null ``ground_truth.target`` (e.g. short_form_qa with
    EM target) are considered complete and never need rubrics.

    Args:
        row: DataFrame row.

    Returns:
        bool: True if rubrics are empty or missing and no EM target is set.
    """
    rm = row.get("reward_model", {})
    if isinstance(rm, str):
        rm = json.loads(rm)
    gt = rm.get("ground_truth", {})
    # Skip rubric gen if EM target is already set (short_form_qa)
    if gt.get("target") is not None:
        return False
    rubrics = gt.get("rubrics", [])
    return len(rubrics) == 0


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the evaluation CLI.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Long-form challenger evaluation with difficulty filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to intermediate parquet from main_generation_long.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to write filtered output parquet.",
    )
    parser.add_argument(
        "--max_items", type=int, default=0,
        help="Max items to process (0 = all). Useful for testing.",
    )

    # Model / servers
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name for inference servers.",
    )
    parser.add_argument(
        "--grader_model_name", type=str, default=None,
        help="Model name for grader server. Defaults to --model_name if not set.",
    )
    parser.add_argument(
        "--solver_base_url", type=str, default="http://127.0.0.1:8001",
        help="Base URL of the solver inference server.",
    )
    parser.add_argument(
        "--grader_base_url", type=str, default="http://127.0.0.1:8001",
        help="Base URL of the grader inference server.",
    )
    parser.add_argument(
        "--retrieval_url", type=str, default="http://127.0.0.1:8000/retrieve",
        help="URL of the retrieval server endpoint.",
    )

    # Templates
    parser.add_argument(
        "--rubric_template_path", type=str,
        default="scope/prompts/rubric.txt",
        help="Path to rubric generation template.",
    )
    parser.add_argument(
        "--rubric_mode", type=str, default="single",
        choices=["single", "v19_perturn"],
        help="Rubric generation mode: 'single' (legacy) or 'v19_perturn'.",
    )
    parser.add_argument(
        "--rubric_initial_template_path", type=str, default="",
        help="V19 per-turn: path to initial (source doc) rubric template.",
    )
    parser.add_argument(
        "--rubric_turn_template_path", type=str, default="",
        help="V19 per-turn: path to per-turn rubric template.",
    )
    parser.add_argument(
        "--rubric_synthesis_template_path", type=str, default="",
        help="V19 per-turn: path to synthesis rubric template.",
    )
    parser.add_argument(
        "--solver_template_path", type=str,
        default="scope/prompts/solver_search_r1.txt",
        help="Path to solver prompt template.",
    )
    parser.add_argument(
        "--grader_template_path", type=str,
        default="scope/prompts/grader_per_rubric.txt",
        help="Path to grader prompt template.",
    )
    parser.add_argument(
        "--tool_config_path", type=str, default=None,
        help="Path to tool config YAML for solver tool-use injection.",
    )

    # Rollout settings
    parser.add_argument(
        "--reward_rollout_n", type=int, default=8,
        help="Number of solver rollouts per task (K).",
    )
    parser.add_argument(
        "--max_turns", type=int, default=5,
        help="Max assistant turns per solver conversation.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048,
        help="Max tokens per solver response.",
    )
    parser.add_argument(
        "--max_prompt_length", type=int, default=131072,
        help="Max prompt token length for solver.",
    )
    parser.add_argument(
        "--solver_prompt_length_limit", type=int, default=0,
        help="Drop tasks whose rendered solver prompt exceeds this many tokens. "
             "0 = disabled. Set to solver training max_prompt_length to prevent "
             "truncation crashes.",
    )
    parser.add_argument(
        "--max_search_turns", type=int, default=4,
        help="Max search turns for solver template formatting.",
    )
    parser.add_argument(
        "--solver_retry", action="store_true", default=False,
        help="Enable solver retry for missing <answer> tags.",
    )
    parser.add_argument(
        "--solver_stop_tokens", type=str, default=None,
        help='JSON list of solver stop tokens, e.g. \'["</answer>"]\'.',
    )
    parser.add_argument(
        "--solver_format", type=str, default="tool_call",
        choices=["tool_call", "search_r1", "function_calls", "function_calls_xml"],
        help="Solver tool interaction format.",
    )

    # Grader settings
    parser.add_argument(
        "--grader_format", type=str, default="xml",
        choices=["xml", "json", "v2"],
        help="Grader output format (auto-detected from template if xml).",
    )
    parser.add_argument(
        "--grader_retry", action="store_true", default=False,
        help="Enable grader retry for failed parses.",
    )
    parser.add_argument(
        "--grader_min_coverage", type=float, default=0.5,
        help="Min rubric coverage for grader recovery.",
    )
    parser.add_argument(
        "--grader_pad_value", type=float, default=0.0,
        help="Pad value for missing rubric assessments.",
    )
    parser.add_argument(
        "--grader_per_rubric", action="store_true", default=False,
        help="Grade each rubric independently.",
    )

    # Rubric settings
    parser.add_argument(
        "--rubric_format", type=str, default="xml",
        choices=["xml", "json"],
        help="Rubric output format.",
    )

    # Batching / timeout / early-stop
    parser.add_argument(
        "--timeout", type=float, default=1800.0,
        help="HTTP timeout in seconds for API calls.",
    )
    parser.add_argument(
        "--max_retries", type=int, default=5,
        help="Max API retries for HTTP calls.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help=(
            "Batch size for pipeline iteration and grader API calls. "
            "Defaults to target_filtered when target_filtered is set, "
            "otherwise 5000. 0 = no batching (send all at once)."
        ),
    )
    parser.add_argument(
        "--target_filtered", type=int, default=0,
        help=(
            "Target number of filtered instances; 0 = process all. "
            "When set, process tasks in pipeline batches and stop early "
            "once enough filtered results accumulate."
        ),
    )

    # Per-prompt cap
    parser.add_argument(
        "--max_tasks_per_prompt", type=int, default=0,
        help=(
            "Maximum number of tasks to keep per source prompt. "
            "0 = no limit. Prevents over-representation of any single "
            "input context after difficulty filtering."
        ),
    )

    # Quality filtering
    parser.add_argument(
        "--quality_gates", type=str, default="",
        help=(
            "Comma-separated quality gate names to enable, e.g. "
            "'entity,no_leakage' or 'entity,no_leakage,retrieval'. "
            "Empty string disables quality filtering."
        ),
    )
    parser.add_argument(
        "--quality_entity_template_path", type=str, default="",
        help="Path to entity identifiability quality grader template.",
    )
    parser.add_argument(
        "--quality_no_leakage_template_path", type=str, default="",
        help="Path to no-answer-leakage quality grader template.",
    )
    parser.add_argument(
        "--quality_retrieval_template_path", type=str, default="",
        help="Path to requires-retrieval quality grader template.",
    )
    parser.add_argument(
        "--quality_retrieval_multi_template_path", type=str, default="",
        help="Path to retrieval quality grader template for multi-search tasks (2+ turns).",
    )
    parser.add_argument(
        "--quality_source_relevance_template_path", type=str, default="",
        help="Path to source-relevance quality grader template.",
    )
    parser.add_argument(
        "--quality_enabled", action="store_true", default=False,
        help="(Legacy) Enable all quality gates. Prefer --quality_gates.",
    )
    parser.add_argument(
        "--quality_required_sum", type=int, default=-1,
        help="Required quality score sum to pass the gate. Default: number of active gates.",
    )
    parser.add_argument(
        "--quality_max_tokens", type=int, default=256,
        help="Max tokens for quality grading LLM calls. Qwen3 needs >=1024.",
    )

    # Difficulty filtering
    parser.add_argument(
        "--difficulty_min", type=float, default=0.2,
        help="Minimum difficulty score (inclusive).",
    )
    parser.add_argument(
        "--difficulty_max", type=float, default=0.8,
        help="Maximum difficulty score (inclusive).",
    )
    parser.add_argument(
        "--difficulty_fn", type=str, default="normalized",
        choices=["tent", "continuous", "normalized"],
        help="Difficulty scoring function.",
    )
    parser.add_argument(
        "--difficulty_target", type=float, default=0.5,
        help="Target normalized score for tent difficulty function.",
    )

    return parser


def _json_default(obj: Any) -> Any:
    """JSON fallback encoder for numpy types produced by pyarrow deserialization.

    Args:
        obj: The object that the default JSON encoder cannot serialize.

    Returns:
        A JSON-serializable Python equivalent.

    Raises:
        TypeError: If *obj* is not a recognized numpy type.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    filtered_df = run_pipeline(args)

    # Save output using HuggingFace datasets for native Arrow nested type
    # support.  This avoids serializing prompt/reward_model/metadata to JSON
    # strings, which would break verl's rl_dataset.py (expects list/dict,
    # not str).
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = hf_datasets.Dataset.from_pandas(filtered_df)
    ds.to_parquet(args.output_path)
    print(f"Saved {len(filtered_df)} tasks to {args.output_path}")


if __name__ == "__main__":
    main()
