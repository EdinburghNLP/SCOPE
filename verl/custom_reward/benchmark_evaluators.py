"""Per-benchmark evaluation functions for long-form validation benchmarks.

Provides evaluator functions for HealthBench, ResearchQA, DRB-RACE,
SQA-CS-V2, and ResearchRubrics benchmarks. Each evaluator takes extracted
answer texts and ground truth dicts, calls an LLM judge via
``rollout.grade_batch()``, and returns per-item score dicts compatible
with the reward function's ``reward_extra_info`` tracking.

Usage::

    from verl.custom_reward.benchmark_evaluators import BENCHMARK_EVALUATORS

    evaluator_fn = BENCHMARK_EVALUATORS["eval_healthbench"]
    results = evaluator_fn(responses, ground_truths, rollout, model, temp)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from verl.custom_reward.batch_reward_rollout import (
    _build_grader_call_entry,
    _build_grader_calls_from_flat,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# HealthBench grader template (verbatim from healthbench_eval.py)
# ---------------------------------------------------------------------------

HEALTHBENCH_GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

# ---------------------------------------------------------------------------
# ResearchQA coverage prompt (verbatim from compute_coverage.py)
# ---------------------------------------------------------------------------

RESEARCHQA_COVERAGE_PROMPT_TEMPLATE = (
    "Please judge the following questions based on the response below.\n"
    "For each question, select one of the following ratings to indicate "
    "the extent to which the response addresses the question:\n"
    "Not at all, Barely, Moderately, Mostly, Completely\n\n"
    "Definitions:\n"
    "- Not at all: *totally uninferable*\n"
    "- Barely: *unmentioned but inferrable*\n"
    "- Moderately: *mentioned but misses important details*\n"
    "- Mostly: *mentioned but misses some details*\n"
    "- Completely: *mentioned with sufficient details*\n\n"
    "Only output one of the five phrases for each question, separated by "
    "newlines, and nothing else.\n\n"
    "Response: {response}\n"
    "Questions:\n{questions}\n\nOutput:"
)

_TEXT_TO_SCORE = {
    "Not at all": 1,
    "Barely": 2,
    "Moderately": 3,
    "Mostly": 4,
    "Completely": 5,
}

# ---------------------------------------------------------------------------
# DRB-RACE scoring prompt
# ---------------------------------------------------------------------------

DRB_RACE_SCORING_PROMPT_TEMPLATE = """You are evaluating a research report written in response to the following prompt:

PROMPT: {prompt}

REPORT:
{response}

Please rate the report on the following 5 dimensions, each on a 1-5 scale:

1. **Depth**: How deeply does the report explore the topic? (1=superficial, 5=comprehensive)
2. **Structure**: How well-organized is the report? (1=disorganized, 5=excellent structure)
3. **Relevance**: How relevant is the content to the prompt? (1=off-topic, 5=highly relevant)
4. **Accuracy**: How accurate are the claims and cited references? (1=many errors, 5=highly accurate)
5. **Completeness**: How completely does the report address all aspects of the prompt? (1=very incomplete, 5=fully complete)

Return a JSON object with these exact keys: "depth", "structure", "relevance", "accuracy", "completeness". Each value should be an integer from 1 to 5.

Return just the JSON in markdown format."""

# ---------------------------------------------------------------------------
# SQA-CS-V2 scoring prompt
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ResearchRubrics grader template (adapted from HealthBench pattern)
# ---------------------------------------------------------------------------

RESEARCHRUBRICS_GRADER_TEMPLATE = """Your job is to evaluate a research report against a single rubric criterion and determine whether the criterion is satisfied.

# Task Prompt
The following prompt was given to the model:
<<task_prompt>>

# Response
<<response>>

# Rubric Criterion
Axis: <<axis>>
Weight: <<weight>>
Criterion: <<criterion>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criterion.
- The "criteria_met" field should be a boolean indicating whether the response satisfies the criterion.
- Judge the criterion as objectively as possible based on the response content.
- Note: Some criteria have negative weights, meaning they describe undesirable qualities. Still report truthfully whether the criterion is met — the scoring system handles the weight direction.

# Example
If the criterion is "Provides a clear executive summary" and the response includes one, return:

```json
{
  "explanation": "The response includes a clear executive summary at the beginning.",
  "criteria_met": true
}
```

Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


# ---------------------------------------------------------------------------
# WildBench scoring prompt (adapted from eval_template.score.v2.md)
# ---------------------------------------------------------------------------

WILDBENCH_SCORING_PROMPT_TEMPLATE = """# Instruction

You are an expert evaluator. Your task is to evaluate the quality of the responses generated by AI models.
We will provide you with the user query and an AI-generated responses.
You should first read the user query and the conversation history carefully for analyzing the task, and then evaluate the quality of the responses based on and rules provided below.

# Conversation between User and AI

## History
<|begin_of_history|>

{history}

<|end_of_history|>

## Current User Query
<|begin_of_query|>

{user_query}

<|end_of_query|>

## AI Response
<|begin_of_response|>

{model_output}

<|end_of_response|>


# Evaluation

## Checklist

<|begin_of_checklist|>

{checklist}

<|end_of_checklist|>

Please use this checklist to guide your evaluation, but do not limit your assessment to the checklist.

## Rules

You should compare the above response based on your analysis of the user queries and the conversation history.
You should first write down your analysis and the checklist that you used for the evaluation, and then provide your assessment according to the checklist.
The scores are in the range of 1~10, where 1 means the response is very poor and 10 means the response is perfect.
Here are more detailed criteria for the scores:

- Score 1~2: The response is very poor and does not make sense at all.
- Score 3~4: The response is poor and does help user solve the problem in a meaningful way.
- Score 5~6: The response is fair but has some issues (e.g., factual errors, hallucinations, missing key information).
- Score 7~8: The response is good enough but could be improved in some ways.
- Score 9~10: The response is perfect and provides helpful information that can help user solve the problem.

## Output Format
First, please output your analysis for the model response, and then summarize your assessment to two aspects: "strengths" and "weaknesses"; Finally, please write down your rating for the assessment.

Please provide your evaluation results in the following json format by filling in the placeholders in []:
```
{{"strengths": "[analysis for the strengths of the response]", "weaknesses": "[analysis for the weaknesses of the response]", "score": "[1~10]"}}
```
""".strip()


SQA_SCORING_PROMPT_TEMPLATE = """You are evaluating a response to the following question:

QUESTION: {question}

RESPONSE:
{response}

Rate the overall quality of this response on one dimension:

1. **Content Quality**: How well does the response answer the question? Consider accuracy, completeness, and clarity. (1-5 scale)

Return a JSON object with key "content_quality", an integer 1-5.

Return just the JSON in markdown format."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_json_response(text: str) -> dict:
    """Extract JSON from a string that may contain markdown code blocks.

    Args:
        text: Raw text that may contain ````` ```json ... ``` ````` markers.

    Returns:
        dict: Parsed dictionary, or empty dict on failure.
    """
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError:
        # Try to find any JSON object in the text
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}


def _normalize_5scale(x: float) -> float:
    """Normalize a 1-5 scale score to [0, 1].

    Args:
        x: Score on a 1-5 scale.

    Returns:
        float: Normalized score in [0, 1].
    """
    return max(0.0, min(1.0, (x - 1) / 4))


def _normalize_10scale(x: float) -> float:
    """Normalize a 1-10 scale score to [0, 1].

    Args:
        x: Score on a 1-10 scale.

    Returns:
        float: Normalized score in [0, 1].
    """
    return max(0.0, min(1.0, (x - 1) / 9))


# ---------------------------------------------------------------------------
# HealthBench evaluator
# ---------------------------------------------------------------------------


def healthbench_evaluate_batch(
    responses: list[str],
    ground_truths: list[dict[str, Any]],
    rollout: Any,
    grader_model_name: str,
    temperature: float,
) -> list[dict[str, Any]]:
    """Evaluate a batch of responses against HealthBench rubrics.

    For each item, reconstructs the full conversation with the model's
    response appended, then grades each rubric item individually using the
    HealthBench GRADER_TEMPLATE. Score = achieved_points / total_possible_points.

    Args:
        responses: Extracted answer texts from solver outputs.
        ground_truths: Ground truth dicts, each containing
            ``healthbench_rubrics`` (list of ``{criterion, points, tags}``)
            and ``healthbench_conversation`` (list of message dicts).
        rollout: ``BatchMultiTurnRollout`` instance for ``grade_batch()``.
        grader_model_name: OpenAI model name for the judge.
        temperature: Sampling temperature for the judge.

    Returns:
        list[dict[str, float | str]]: Per-item dicts with ``"score"`` and
            ``"healthbench_score"`` keys.
    """
    # Flatten all (item, rubric) pairs for a single grade_batch call
    flat_prompts: list[str] = []
    flat_messages: list[list[dict[str, str]]] = []
    flat_map: list[tuple[int, int]] = []  # (item_idx, rubric_idx)
    rubric_data: list[list[dict]] = []  # per-item rubric items

    for i, (response, gt) in enumerate(zip(responses, ground_truths)):
        rubrics = gt.get("healthbench_rubrics", [])
        conversation = gt.get("healthbench_conversation", [])
        rubric_data.append(rubrics)

        # Reconstruct full conversation with model response
        convo_with_response = list(conversation) + [
            {"role": "assistant", "content": response}
        ]
        convo_str = "\n\n".join(
            f"{m['role']}: {m['content']}" for m in convo_with_response
        )

        for ri, rubric in enumerate(rubrics):
            points = rubric.get("points", 1.0)
            criterion = rubric.get("criterion", "")
            rubric_str = f"[{points}] {criterion}"

            grader_prompt = HEALTHBENCH_GRADER_TEMPLATE.replace(
                "<<conversation>>", convo_str
            ).replace("<<rubric_item>>", rubric_str)

            msg = [{"role": "user", "content": grader_prompt}]
            flat_prompts.append(grader_prompt)
            flat_messages.append(msg)
            flat_map.append((i, ri))

    # Batch grade all rubric items at once
    if flat_prompts:
        _usage_start = len(rollout.grader_usage_log)
        raw_outputs = rollout.grade_batch(
            flat_prompts,
            "",  # grader_base_url unused for OpenAI models
            grader_model_name=grader_model_name,
            grader_messages=flat_messages,
            temperature=temperature,
        )
        grader_calls_per_item = _build_grader_calls_from_flat(
            rollout, len(responses), flat_map, flat_prompts,
            raw_outputs, _usage_start,
        )
    else:
        raw_outputs = []
        grader_calls_per_item = [[] for _ in range(len(responses))]

    # Parse grading responses and compute per-item scores
    # grading_results[item_idx] = list of {criteria_met: bool} per rubric
    grading_results: dict[int, list[dict]] = {
        i: [{"criteria_met": False}] * len(rubric_data[i])
        for i in range(len(responses))
    }

    for flat_idx, (item_idx, rubric_idx) in enumerate(flat_map):
        if flat_idx < len(raw_outputs):
            parsed = _parse_json_response(raw_outputs[flat_idx])
            if "criteria_met" in parsed:
                grading_results[item_idx][rubric_idx] = parsed
            else:
                grading_results[item_idx][rubric_idx] = {
                    "criteria_met": False,
                    "explanation": "Parse failure",
                }

    # Collect per-item raw grader outputs by concatenating rubric-level outputs
    per_item_raw: dict[int, list[str]] = {i: [] for i in range(len(responses))}
    for flat_idx, (item_idx, _rubric_idx) in enumerate(flat_map):
        output_str = raw_outputs[flat_idx] if flat_idx < len(raw_outputs) else ""
        per_item_raw[item_idx].append(output_str)

    # Compute normalized scores
    results: list[dict[str, Any]] = []
    for i in range(len(responses)):
        rubrics = rubric_data[i]
        gradings = grading_results[i]

        total_possible = sum(
            r.get("points", 1.0) for r in rubrics if r.get("points", 1.0) > 0
        )
        grader_output = "\n".join(per_item_raw.get(i, []))
        if total_possible == 0:
            results.append({
                "score": 0.0,
                "healthbench_score": 0.0,
                "grader_output": grader_output,
                "grader_calls": grader_calls_per_item[i],
            })
            continue

        achieved = sum(
            r.get("points", 1.0)
            for r, g in zip(rubrics, gradings)
            if g.get("criteria_met", False)
        )
        score = max(0.0, min(1.0, achieved / total_possible))
        results.append({
            "score": score,
            "healthbench_score": score,
            "grader_output": grader_output,
            "grader_calls": grader_calls_per_item[i],
        })

    return results


# ---------------------------------------------------------------------------
# ResearchQA evaluator
# ---------------------------------------------------------------------------


def researchqa_evaluate_batch(
    responses: list[str],
    ground_truths: list[dict[str, Any]],
    rollout: Any,
    grader_model_name: str,
    temperature: float,
) -> list[dict[str, Any]]:
    """Evaluate a batch of responses against ResearchQA rubric items.

    Builds a coverage evaluation prompt per item (all rubric questions in one
    prompt), calls the judge, and parses line-by-line ratings. Normalizes
    the 1-5 scale to [0, 1].

    Args:
        responses: Extracted answer texts from solver outputs.
        ground_truths: Ground truth dicts, each containing
            ``researchqa_rubric_items`` (list of ``{rubric_item, type}``).
        rollout: ``BatchMultiTurnRollout`` instance for ``grade_batch()``.
        grader_model_name: OpenAI model name for the judge.
        temperature: Sampling temperature for the judge.

    Returns:
        list[dict[str, float | str]]: Per-item dicts with ``"score"`` and
            ``"researchqa_coverage"`` keys.
    """
    prompts: list[str] = []
    messages_list: list[list[dict[str, str]]] = []
    rubric_counts: list[int] = []

    for response, gt in zip(responses, ground_truths):
        rubric_items = gt.get("researchqa_rubric_items", [])
        questions = [r.get("rubric_item", "") for r in rubric_items]
        rubric_counts.append(len(questions))

        if not questions:
            prompts.append("")
            messages_list.append([])
            continue

        prompt = RESEARCHQA_COVERAGE_PROMPT_TEMPLATE.format(
            response=response,
            questions="\n".join(questions),
        )
        msg = [{"role": "user", "content": prompt}]
        prompts.append(prompt)
        messages_list.append(msg)

    # Filter out empty prompts for grading
    valid_indices = [i for i, p in enumerate(prompts) if p]
    valid_prompts = [prompts[i] for i in valid_indices]
    valid_messages = [messages_list[i] for i in valid_indices]

    if valid_prompts:
        _usage_start = len(rollout.grader_usage_log)
        raw_outputs = rollout.grade_batch(
            valid_prompts,
            "",
            grader_model_name=grader_model_name,
            grader_messages=valid_messages,
            temperature=temperature,
        )
        _usage_entries = rollout.grader_usage_log[_usage_start:]
    else:
        raw_outputs = []
        _usage_entries = []

    # Map outputs and usage back to original indices
    output_map: dict[int, str] = {}
    usage_map: dict[int, dict[str, Any] | None] = {}
    prompt_map: dict[int, str] = {}
    for vi, orig_idx in enumerate(valid_indices):
        if vi < len(raw_outputs):
            output_map[orig_idx] = raw_outputs[vi]
            prompt_map[orig_idx] = valid_prompts[vi]
            usage_map[orig_idx] = (
                _usage_entries[vi] if vi < len(_usage_entries) else None
            )

    # Parse ratings and compute coverage scores
    results: list[dict[str, Any]] = []
    for i in range(len(responses)):
        output = output_map.get(i, "")
        n_rubrics = rubric_counts[i]
        grader_calls_i: list[dict[str, Any]] = (
            [_build_grader_call_entry(
                prompt_map.get(i, ""), output, usage_map.get(i))]
            if i in output_map
            else []
        )

        if not output or n_rubrics == 0:
            results.append({
                "score": 0.0,
                "researchqa_coverage": 0.0,
                "grader_output": output,
                "grader_calls": grader_calls_i,
            })
            continue

        lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
        normalized_scores = []
        for line in lines:
            if line in _TEXT_TO_SCORE:
                numeric = _TEXT_TO_SCORE[line]
                normalized_scores.append(_normalize_5scale(numeric))

        if normalized_scores:
            coverage = sum(normalized_scores) / len(normalized_scores)
        else:
            coverage = 0.0

        results.append({
            "score": coverage,
            "researchqa_coverage": coverage,
            "grader_output": output,
            "grader_calls": grader_calls_i,
        })

    return results


# ---------------------------------------------------------------------------
# DRB-RACE evaluator
# ---------------------------------------------------------------------------


def drb_race_evaluate_batch(
    responses: list[str],
    ground_truths: list[dict[str, Any]],
    rollout: Any,
    grader_model_name: str,
    temperature: float,
) -> list[dict[str, Any]]:
    """Evaluate a batch of responses using DRB-RACE holistic 5-dimension scoring.

    Rates each response on depth, structure, relevance, accuracy, and
    completeness (1-5 scale), normalizes to [0, 1], and returns the average
    as the RACE score.

    Args:
        responses: Extracted answer texts from solver outputs.
        ground_truths: Ground truth dicts, each containing
            ``drb_prompt`` (the original research prompt text).
        rollout: ``BatchMultiTurnRollout`` instance for ``grade_batch()``.
        grader_model_name: OpenAI model name for the judge.
        temperature: Sampling temperature for the judge.

    Returns:
        list[dict[str, float | str]]: Per-item dicts with ``"score"``,
            ``"drb_depth"``, ``"drb_structure"``, ``"drb_relevance"``,
            ``"drb_accuracy"``, ``"drb_completeness"`` keys.
    """
    prompts: list[str] = []
    messages_list: list[list[dict[str, str]]] = []

    for response, gt in zip(responses, ground_truths):
        prompt_text = gt.get("drb_prompt", "")
        scoring_prompt = DRB_RACE_SCORING_PROMPT_TEMPLATE.format(
            prompt=prompt_text,
            response=response,
        )
        msg = [{"role": "user", "content": scoring_prompt}]
        prompts.append(scoring_prompt)
        messages_list.append(msg)

    if prompts:
        _usage_start = len(rollout.grader_usage_log)
        raw_outputs = rollout.grade_batch(
            prompts,
            "",
            grader_model_name=grader_model_name,
            grader_messages=messages_list,
            temperature=temperature,
        )
        _usage_entries = rollout.grader_usage_log[_usage_start:]
    else:
        raw_outputs = []
        _usage_entries = []

    dimensions = ["depth", "structure", "relevance", "accuracy", "completeness"]
    results: list[dict[str, Any]] = []

    for i in range(len(responses)):
        output = raw_outputs[i] if i < len(raw_outputs) else ""
        scores = _parse_json_response(output)
        usage = _usage_entries[i] if i < len(_usage_entries) else None

        dim_scores = {}
        for dim in dimensions:
            val = scores.get(dim, 3)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 3.0
            dim_scores[dim] = _normalize_5scale(val)

        race_score = sum(dim_scores.values()) / len(dimensions)

        result: dict[str, Any] = {
            "score": race_score,
            "grader_output": output,
            "grader_calls": [
                _build_grader_call_entry(prompts[i], output, usage)
            ] if i < len(prompts) else [],
        }
        for dim in dimensions:
            result[f"drb_{dim}"] = dim_scores[dim]
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# SQA-CS-V2 evaluator
# ---------------------------------------------------------------------------


def sqa_evaluate_batch(
    responses: list[str],
    ground_truths: list[dict[str, Any]],
    rollout: Any,
    grader_model_name: str,
    temperature: float,
) -> list[dict[str, Any]]:
    """Evaluate a batch of responses using SQA-CS-V2 content quality scoring.

    Rates each response on content quality (1-5 scale), normalizes to [0, 1].
    ASTA conversion and citation quality are skipped (requires external repo
    and the SCOPE retrieval pipeline returns plain text without snippet tags).

    Args:
        responses: Extracted answer texts from solver outputs.
        ground_truths: Ground truth dicts, each containing
            ``sqa_question`` (the original question text).
        rollout: ``BatchMultiTurnRollout`` instance for ``grade_batch()``.
        grader_model_name: OpenAI model name for the judge.
        temperature: Sampling temperature for the judge.

    Returns:
        list[dict[str, float | str]]: Per-item dicts with ``"score"`` and
            ``"sqa_content_quality"`` keys.
    """
    prompts: list[str] = []
    messages_list: list[list[dict[str, str]]] = []

    for response, gt in zip(responses, ground_truths):
        question = gt.get("sqa_question", "")
        scoring_prompt = SQA_SCORING_PROMPT_TEMPLATE.format(
            question=question,
            response=response,
        )
        msg = [{"role": "user", "content": scoring_prompt}]
        prompts.append(scoring_prompt)
        messages_list.append(msg)

    if prompts:
        _usage_start = len(rollout.grader_usage_log)
        raw_outputs = rollout.grade_batch(
            prompts,
            "",
            grader_model_name=grader_model_name,
            grader_messages=messages_list,
            temperature=temperature,
        )
        _usage_entries = rollout.grader_usage_log[_usage_start:]
    else:
        raw_outputs = []
        _usage_entries = []

    results: list[dict[str, Any]] = []
    for i in range(len(responses)):
        output = raw_outputs[i] if i < len(raw_outputs) else ""
        scores = _parse_json_response(output)
        usage = _usage_entries[i] if i < len(_usage_entries) else None

        content_q = scores.get("content_quality", 3)
        try:
            content_q = float(content_q)
        except (TypeError, ValueError):
            content_q = 3.0

        normalized = _normalize_5scale(content_q)
        results.append({
            "score": normalized,
            "sqa_content_quality": normalized,
            "grader_output": output,
            "grader_calls": [
                _build_grader_call_entry(prompts[i], output, usage)
            ] if i < len(prompts) else [],
        })

    return results


# ---------------------------------------------------------------------------
# ResearchRubrics evaluator
# ---------------------------------------------------------------------------


def researchrubrics_evaluate_batch(
    responses: list[str],
    ground_truths: list[dict[str, Any]],
    rollout: Any,
    grader_model_name: str,
    temperature: float,
) -> list[dict[str, Any]]:
    """Evaluate a batch of responses against ResearchRubrics per-criterion rubrics.

    For each item, grades every rubric criterion individually using the
    ``RESEARCHRUBRICS_GRADER_TEMPLATE``, then computes a weighted compliance
    score: ``achieved_weight / total_positive_weight``, clamped to [0, 1].

    Args:
        responses: Extracted answer texts from solver outputs.
        ground_truths: Ground truth dicts, each containing
            ``researchrubrics_rubrics`` (list of ``{criterion, weight, axis}``)
            and ``researchrubrics_prompt`` (the original task prompt).
        rollout: ``BatchMultiTurnRollout`` instance for ``grade_batch()``.
        grader_model_name: OpenAI model name for the judge.
        temperature: Sampling temperature for the judge.

    Returns:
        list[dict[str, float | str]]: Per-item dicts with ``"score"`` and
            ``"researchrubrics_compliance"`` keys.
    """
    # Flatten all (item, rubric) pairs for a single grade_batch call
    flat_prompts: list[str] = []
    flat_messages: list[list[dict[str, str]]] = []
    flat_map: list[tuple[int, int]] = []  # (item_idx, rubric_idx)
    rubric_data: list[list[dict]] = []  # per-item rubric items

    for i, (response, gt) in enumerate(zip(responses, ground_truths)):
        rubrics = gt.get("researchrubrics_rubrics", [])
        task_prompt = gt.get("researchrubrics_prompt", "")
        rubric_data.append(rubrics)

        for ri, rubric in enumerate(rubrics):
            criterion = rubric.get("criterion", "")
            weight = rubric.get("weight", 1.0)
            axis = rubric.get("axis", "")

            grader_prompt = (
                RESEARCHRUBRICS_GRADER_TEMPLATE
                .replace("<<task_prompt>>", task_prompt)
                .replace("<<response>>", response)
                .replace("<<axis>>", str(axis))
                .replace("<<weight>>", str(weight))
                .replace("<<criterion>>", criterion)
            )

            msg = [{"role": "user", "content": grader_prompt}]
            flat_prompts.append(grader_prompt)
            flat_messages.append(msg)
            flat_map.append((i, ri))

    # Batch grade all rubric items at once
    if flat_prompts:
        _usage_start = len(rollout.grader_usage_log)
        raw_outputs = rollout.grade_batch(
            flat_prompts,
            "",  # grader_base_url unused for OpenAI models
            grader_model_name=grader_model_name,
            grader_messages=flat_messages,
            temperature=temperature,
        )
        grader_calls_per_item = _build_grader_calls_from_flat(
            rollout, len(responses), flat_map, flat_prompts,
            raw_outputs, _usage_start,
        )
    else:
        raw_outputs = []
        grader_calls_per_item = [[] for _ in range(len(responses))]

    # Parse grading responses and compute per-item scores
    grading_results: dict[int, list[dict]] = {
        i: [{"criteria_met": False}] * len(rubric_data[i])
        for i in range(len(responses))
    }

    for flat_idx, (item_idx, rubric_idx) in enumerate(flat_map):
        if flat_idx < len(raw_outputs):
            parsed = _parse_json_response(raw_outputs[flat_idx])
            if "criteria_met" in parsed:
                grading_results[item_idx][rubric_idx] = parsed
            else:
                grading_results[item_idx][rubric_idx] = {
                    "criteria_met": False,
                    "explanation": "Parse failure",
                }

    # Collect per-item raw grader outputs by concatenating rubric-level outputs
    rr_per_item_raw: dict[int, list[str]] = {i: [] for i in range(len(responses))}
    for flat_idx, (item_idx, _rubric_idx) in enumerate(flat_map):
        output_str = raw_outputs[flat_idx] if flat_idx < len(raw_outputs) else ""
        rr_per_item_raw[item_idx].append(output_str)

    # Compute weighted compliance scores
    results: list[dict[str, Any]] = []
    for i in range(len(responses)):
        rubrics = rubric_data[i]
        gradings = grading_results[i]

        total_possible = sum(
            r.get("weight", 1.0) for r in rubrics if r.get("weight", 1.0) > 0
        )
        grader_output = "\n".join(rr_per_item_raw.get(i, []))
        if total_possible == 0:
            results.append({
                "score": 0.0,
                "researchrubrics_compliance": 0.0,
                "grader_output": grader_output,
                "grader_calls": grader_calls_per_item[i],
            })
            continue

        achieved = sum(
            r.get("weight", 1.0)
            for r, g in zip(rubrics, gradings)
            if g.get("criteria_met", False)
        )
        score = max(0.0, min(1.0, achieved / total_possible))
        results.append({
            "score": score,
            "researchrubrics_compliance": score,
            "grader_output": grader_output,
            "grader_calls": grader_calls_per_item[i],
        })

    return results


# ---------------------------------------------------------------------------
# Research Plan Gen evaluator
# ---------------------------------------------------------------------------


def research_plan_gen_evaluate_batch(
    responses: list[str],
    ground_truths: list[dict[str, Any]],
    rollout: Any,
    grader_model_name: str,
    temperature: float,
) -> list[dict[str, Any]]:
    """Evaluate a batch of responses against research-plan-gen per-criterion rubrics.

    For each item, grades every rubric criterion individually using the
    ``RESEARCHRUBRICS_GRADER_TEMPLATE``, then computes a compliance score:
    ``count(criteria_met) / total_rubrics``, since all rubrics have equal
    weight (1.0).

    Args:
        responses: Extracted answer texts from solver outputs.
        ground_truths: Ground truth dicts, each containing
            ``research_plan_gen_rubrics`` (list of ``{criterion, weight}``)
            and ``research_plan_gen_goal`` (the original goal text).
        rollout: ``BatchMultiTurnRollout`` instance for ``grade_batch()``.
        grader_model_name: OpenAI model name for the judge.
        temperature: Sampling temperature for the judge.

    Returns:
        list[dict[str, float | str]]: Per-item dicts with ``"score"`` and
            ``"research_plan_gen_compliance"`` keys.
    """
    # Flatten all (item, rubric) pairs for a single grade_batch call
    flat_prompts: list[str] = []
    flat_messages: list[list[dict[str, str]]] = []
    flat_map: list[tuple[int, int]] = []  # (item_idx, rubric_idx)
    rubric_data: list[list[dict]] = []  # per-item rubric items

    for i, (response, gt) in enumerate(zip(responses, ground_truths)):
        rubrics = gt.get("research_plan_gen_rubrics", [])
        task_prompt = gt.get("research_plan_gen_goal", "")
        rubric_data.append(rubrics)

        for ri, rubric in enumerate(rubrics):
            criterion = rubric.get("criterion", "")
            weight = rubric.get("weight", 1.0)

            grader_prompt = (
                RESEARCHRUBRICS_GRADER_TEMPLATE
                .replace("<<task_prompt>>", task_prompt)
                .replace("<<response>>", response)
                .replace("<<axis>>", "")
                .replace("<<weight>>", str(weight))
                .replace("<<criterion>>", criterion)
            )

            msg = [{"role": "user", "content": grader_prompt}]
            flat_prompts.append(grader_prompt)
            flat_messages.append(msg)
            flat_map.append((i, ri))

    # Batch grade all rubric items at once
    if flat_prompts:
        _usage_start = len(rollout.grader_usage_log)
        raw_outputs = rollout.grade_batch(
            flat_prompts,
            "",
            grader_model_name=grader_model_name,
            grader_messages=flat_messages,
            temperature=temperature,
        )
        grader_calls_per_item = _build_grader_calls_from_flat(
            rollout, len(responses), flat_map, flat_prompts,
            raw_outputs, _usage_start,
        )
    else:
        raw_outputs = []
        grader_calls_per_item = [[] for _ in range(len(responses))]

    # Parse grading responses and compute per-item scores
    grading_results: dict[int, list[dict]] = {
        i: [{"criteria_met": False}] * len(rubric_data[i])
        for i in range(len(responses))
    }

    for flat_idx, (item_idx, rubric_idx) in enumerate(flat_map):
        if flat_idx < len(raw_outputs):
            parsed = _parse_json_response(raw_outputs[flat_idx])
            if "criteria_met" in parsed:
                grading_results[item_idx][rubric_idx] = parsed
            else:
                grading_results[item_idx][rubric_idx] = {
                    "criteria_met": False,
                    "explanation": "Parse failure",
                }

    # Collect per-item raw grader outputs by concatenating rubric-level outputs
    rpg_per_item_raw: dict[int, list[str]] = {i: [] for i in range(len(responses))}
    for flat_idx, (item_idx, _rubric_idx) in enumerate(flat_map):
        output_str = raw_outputs[flat_idx] if flat_idx < len(raw_outputs) else ""
        rpg_per_item_raw[item_idx].append(output_str)

    # Compute compliance scores (equal weight per rubric)
    results: list[dict[str, Any]] = []
    for i in range(len(responses)):
        rubrics = rubric_data[i]
        gradings = grading_results[i]

        total = len(rubrics)
        grader_output = "\n".join(rpg_per_item_raw.get(i, []))
        if total == 0:
            results.append({
                "score": 0.0,
                "research_plan_gen_compliance": 0.0,
                "grader_output": grader_output,
                "grader_calls": grader_calls_per_item[i],
            })
            continue

        met_count = sum(
            1 for g in gradings if g.get("criteria_met", False)
        )
        score = met_count / total
        results.append({
            "score": score,
            "research_plan_gen_compliance": score,
            "grader_output": grader_output,
            "grader_calls": grader_calls_per_item[i],
        })

    return results


# ---------------------------------------------------------------------------
# Arena-Hard-v2.0 creative writing evaluator (pairwise comparison)
# ---------------------------------------------------------------------------

ARENA_HARD_CW_JUDGE_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses"
    " provided by two AI assistants to the user prompt displayed below. You"
    " will be given assistant A's answer and assistant B's answer. Your job is"
    " to evaluate which assistant's answer is better.\n\n"
    "When evaluating the assistants' answers, compare both assistants' answers."
    " You must identify and correct any mistakes or inaccurate information.\n\n"
    "Then consider if the assistant's answers are helpful, relevant, and"
    " concise. Helpful means the answer correctly responds to the prompt or"
    " follows the instructions. Note when user prompt has any ambiguity or more"
    " than one interpretation, it is more helpful and appropriate to ask for"
    " clarifications or more information from the user than providing an answer"
    " based on assumptions. Relevant means all parts of the response closely"
    " connect or are appropriate to what is being asked. Concise means the"
    " response is clear and not verbose or excessive.\n\n"
    "Then consider the creativity and novelty of the assistant's answers when"
    " needed. Finally, identify any missing important information in the"
    " assistants' answers that would be beneficial to include when responding"
    " to the user prompt.\n\n"
    "After providing your explanation, you must output only one of the"
    " following choices as your final verdict with a label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n"
    "2. Assistant A is slightly better: [[A>B]]\n"
    "3. Tie, relatively the same: [[A=B]]\n"
    "4. Assistant B is slightly better: [[B>A]]\n"
    "5. Assistant B is significantly better: [[B>>A]]\n\n"
    'Example output: "My final verdict is tie: [[A=B]]".'
)

ARENA_HARD_PROMPT_TEMPLATE = (
    "<|User Prompt|>\n{question}\n\n"
    "<|The Start of Assistant A's Answer|>\n{answer_a}\n"
    "<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n{answer_b}\n"
    "<|The End of Assistant B's Answer|>"
)

ARENA_HARD_VERDICT_PATTERNS = [
    re.compile(r"\[\[([AB<>=]+)\]\]"),
    re.compile(r"\[([AB<>=]+)\]"),
]

# Verdict-to-score mapping (faithful to Arena-Hard show_result.py).
# Labels are from the perspective of "does A win?".
_ARENA_HARD_LABEL_TO_SCORE: dict[str, list[float]] = {
    "A>>B": [1.0, 1.0, 1.0],
    "A>B": [1.0],
    "A=B": [0.5],
    "A<B": [0.0],
    "A<<B": [0.0, 0.0, 0.0],
    "B>>A": [0.0, 0.0, 0.0],
    "B>A": [0.0],
    "B=A": [0.5],
    "B<A": [1.0],
    "B<<A": [1.0, 1.0, 1.0],
}


def _extract_arena_hard_verdict(text: str) -> str | None:
    """Extract verdict from judge output using Arena-Hard regex patterns.

    Searches for patterns like ``[[A>>B]]`` or ``[A>B]`` and returns the
    last match found (uppercased), consistent with Arena-Hard's
    ``get_score()`` implementation.

    Args:
        text: Raw judge output text.

    Returns:
        str | None: Verdict string (e.g. ``"A>>B"``, ``"B>A"``) or ``None``
            if no valid verdict is found.
    """
    for pattern in ARENA_HARD_VERDICT_PATTERNS:
        matches = pattern.findall(text.upper())
        matches = [m for m in matches if m]
        if matches:
            return matches[-1].strip("\n")
    return None


def _arena_hard_per_item_score(
    round1_verdict: str | None,
    round2_verdict: str | None,
) -> float:
    """Compute per-item win-rate score from two pairwise rounds.

    Implements the exact scoring logic from Arena-Hard ``show_result.py``
    lines 50-51:

    - Round 1 (A=baseline, B=model): scores are inverted (``1 - s``)
    - Round 2 (A=model, B=baseline): scores are used directly
    - Both are concatenated and averaged for a weighted win-rate

    Args:
        round1_verdict: Verdict from round 1 (A=baseline, B=model), or
            ``None`` if parsing failed.
        round2_verdict: Verdict from round 2 (A=model, B=baseline), or
            ``None`` if parsing failed.

    Returns:
        float: Weighted win-rate in [0, 1]. Returns 0.5 (tie) for both
            ``None`` verdicts, or uses a single round if only one parsed.
    """
    round1_scores = _ARENA_HARD_LABEL_TO_SCORE.get(round1_verdict or "", None)
    round2_scores = _ARENA_HARD_LABEL_TO_SCORE.get(round2_verdict or "", None)

    all_scores: list[float] = []

    # Round 2 (A=model): use directly
    if round2_scores is not None:
        all_scores.extend(round2_scores)

    # Round 1 (A=baseline): invert
    if round1_scores is not None:
        all_scores.extend(1.0 - s for s in round1_scores)

    if not all_scores:
        return 0.5  # both rounds failed to parse

    return sum(all_scores) / len(all_scores)


def arena_hard_cw_evaluate_batch(
    responses: list[str],
    ground_truths: list[dict[str, Any]],
    rollout: Any,
    grader_model_name: str,
    temperature: float,
) -> list[dict[str, Any]]:
    """Evaluate a batch of responses via Arena-Hard-v2.0 pairwise comparison.

    For each item, constructs two judge calls with position-swapped pairwise
    prompts (Round 1: baseline=A, model=B; Round 2: model=A, baseline=B).
    Both rounds use the creative writing judge system prompt. Verdicts are
    extracted via regex and converted to a weighted win-rate score.

    Args:
        responses: Extracted answer texts from solver outputs.
        ground_truths: Ground truth dicts, each containing
            ``arena_hard_prompt`` (original question),
            ``arena_hard_baseline_answer`` (baseline response text), and
            ``arena_hard_baseline_model`` (baseline model name).
        rollout: ``BatchMultiTurnRollout`` instance for ``grade_batch()``.
        grader_model_name: OpenAI model name for the judge.
        temperature: Sampling temperature for the judge.

    Returns:
        list[dict[str, float | str | None]]: Per-item dicts with ``"score"``,
            ``"grader_output"``, ``"arena_hard_cw_winrate"``,
            ``"arena_hard_round1_verdict"``, and
            ``"arena_hard_round2_verdict"`` keys.
    """
    flat_prompts: list[str] = []
    flat_messages: list[list[dict[str, str]]] = []
    # flat_map: (item_idx, round_number) where round is 0 or 1
    flat_map: list[tuple[int, int]] = []

    for i, (response, gt) in enumerate(zip(responses, ground_truths)):
        question = gt.get("arena_hard_prompt", "")
        baseline_answer = gt.get("arena_hard_baseline_answer", "")

        # Round 1: A=baseline, B=model
        user_prompt_r1 = ARENA_HARD_PROMPT_TEMPLATE.format(
            question=question,
            answer_a=baseline_answer,
            answer_b=response,
        )
        msg_r1 = [
            {"role": "system", "content": ARENA_HARD_CW_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_r1},
        ]
        flat_prompts.append(user_prompt_r1)
        flat_messages.append(msg_r1)
        flat_map.append((i, 0))

        # Round 2: A=model, B=baseline
        user_prompt_r2 = ARENA_HARD_PROMPT_TEMPLATE.format(
            question=question,
            answer_a=response,
            answer_b=baseline_answer,
        )
        msg_r2 = [
            {"role": "system", "content": ARENA_HARD_CW_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_r2},
        ]
        flat_prompts.append(user_prompt_r2)
        flat_messages.append(msg_r2)
        flat_map.append((i, 1))

    # Batch grade all rounds at once
    if flat_prompts:
        _usage_start = len(rollout.grader_usage_log)
        raw_outputs = rollout.grade_batch(
            flat_prompts,
            "",  # grader_base_url unused for OpenAI models
            grader_model_name=grader_model_name,
            grader_messages=flat_messages,
            temperature=temperature,
        )
        grader_calls_per_item = _build_grader_calls_from_flat(
            rollout, len(responses), flat_map, flat_prompts,
            raw_outputs, _usage_start,
        )
    else:
        raw_outputs = []
        grader_calls_per_item = [[] for _ in range(len(responses))]

    # Collect per-item verdicts and raw outputs
    item_verdicts: dict[int, dict[int, str | None]] = {
        i: {0: None, 1: None} for i in range(len(responses))
    }
    item_raw: dict[int, list[str]] = {
        i: [] for i in range(len(responses))
    }

    for flat_idx, (item_idx, round_num) in enumerate(flat_map):
        output = raw_outputs[flat_idx] if flat_idx < len(raw_outputs) else ""
        item_raw[item_idx].append(output)
        verdict = _extract_arena_hard_verdict(output)
        item_verdicts[item_idx][round_num] = verdict

    # Compute per-item scores
    results: list[dict[str, Any]] = []
    for i in range(len(responses)):
        round1_verdict = item_verdicts[i][0]
        round2_verdict = item_verdicts[i][1]
        grader_output = "\n---\n".join(item_raw.get(i, []))

        score = _arena_hard_per_item_score(round1_verdict, round2_verdict)
        results.append({
            "score": score,
            "grader_output": grader_output,
            "arena_hard_cw_winrate": score,
            "arena_hard_round1_verdict": round1_verdict,
            "arena_hard_round2_verdict": round2_verdict,
            "grader_calls": grader_calls_per_item[i],
        })

    return results


# ---------------------------------------------------------------------------
# WildBench evaluator
# ---------------------------------------------------------------------------


def wildbench_evaluate_batch(
    responses: list[str],
    ground_truths: list[dict[str, Any]],
    rollout: Any,
    grader_model_name: str,
    temperature: float,
) -> list[dict[str, Any]]:
    """Evaluate a batch of responses using WildBench v2 checklist-guided scoring.

    For each item, constructs a holistic evaluation prompt embedding the
    WildBench checklist criteria and conversation history. The judge scores
    on a 1-10 scale, which is normalized to [0, 1] for SCOPE compatibility.

    Args:
        responses: Extracted answer texts from solver outputs.
        ground_truths: Ground truth dicts, each containing
            ``wildbench_conversation`` (list of message dicts),
            ``wildbench_checklist`` (list of evaluation criteria strings),
            and ``wildbench_primary_tag`` (task category string).
        rollout: ``BatchMultiTurnRollout`` instance for ``grade_batch()``.
        grader_model_name: OpenAI model name for the judge.
        temperature: Sampling temperature for the judge.

    Returns:
        list[dict[str, float | str]]: Per-item dicts with ``"score"``,
            ``"wildbench_raw_score"``, ``"wildbench_primary_tag"``, and
            ``"grader_output"`` keys.
    """
    prompts: list[str] = []
    messages_list: list[list[dict[str, str]]] = []

    for response, gt in zip(responses, ground_truths):
        conversation = gt.get("wildbench_conversation", [])
        checklist_items = gt.get("wildbench_checklist", [])

        # Build history from all turns except the last (which is the query)
        history = ""
        if len(conversation) > 1:
            for msg in conversation[:-1]:
                role_label = msg["role"].upper()
                history += f"{role_label}: {msg['content']}\n\n"

        # Last user query. Use ``len()`` for the guard since ``conversation``
        # may be a numpy array when loaded from parquet — bare truthiness on
        # ndarrays of len > 1 raises ``ValueError``.
        user_query = (
            conversation[-1]["content"] if len(conversation) > 0 else ""
        )

        # Render checklist as markdown bullets
        checklist_md = ""
        for item in checklist_items:
            checklist_md += f"- {item}\n"

        scoring_prompt = WILDBENCH_SCORING_PROMPT_TEMPLATE.format(
            history=history,
            user_query=user_query,
            model_output=response,
            checklist=checklist_md,
        )
        msg = [{"role": "user", "content": scoring_prompt}]
        prompts.append(scoring_prompt)
        messages_list.append(msg)

    if prompts:
        _usage_start = len(rollout.grader_usage_log)
        raw_outputs = rollout.grade_batch(
            prompts,
            "",
            grader_model_name=grader_model_name,
            grader_messages=messages_list,
            temperature=temperature,
        )
        _usage_entries = rollout.grader_usage_log[_usage_start:]
    else:
        raw_outputs = []
        _usage_entries = []

    results: list[dict[str, Any]] = []
    for i in range(len(responses)):
        output = raw_outputs[i] if i < len(raw_outputs) else ""
        scores = _parse_json_response(output)
        usage = _usage_entries[i] if i < len(_usage_entries) else None

        raw_score = scores.get("score", 1)
        try:
            raw_score = float(raw_score)
        except (TypeError, ValueError):
            raw_score = 1.0
        # Clamp to [1, 10]
        raw_score = max(1.0, min(10.0, raw_score))

        normalized = _normalize_10scale(raw_score)
        primary_tag = ground_truths[i].get("wildbench_primary_tag", "")

        results.append({
            "score": normalized,
            "wildbench_raw_score": raw_score,
            "wildbench_primary_tag": primary_tag,
            "grader_output": output,
            "grader_calls": [
                _build_grader_call_entry(prompts[i], output, usage)
            ] if i < len(prompts) else [],
        })

    return results


# ---------------------------------------------------------------------------
# Registry: data_source -> evaluator function
# ---------------------------------------------------------------------------

BENCHMARK_EVALUATORS: dict[str, Any] = {
    "eval_healthbench": healthbench_evaluate_batch,
    "eval_researchqa": researchqa_evaluate_batch,
    "eval_drb_race": drb_race_evaluate_batch,
    "eval_sqa_cs_v2": sqa_evaluate_batch,
    "eval_researchrubrics": researchrubrics_evaluate_batch,
    "eval_research_plan_gen": research_plan_gen_evaluate_batch,
    "eval_arena_hard_cw": arena_hard_cw_evaluate_batch,
    "eval_wildbench": wildbench_evaluate_batch,
}
