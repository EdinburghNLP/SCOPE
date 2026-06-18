#!/usr/bin/env python3
"""Validator for challenger prompt template outputs.

This module validates LLM outputs from the challenger prompt template,
using XML format with 3-10 rubrics directly under the <rubrics> tag.

Example usage:
    # Python API
    from scope.utils.parse_challenger import (
        validate_intermediate_turn,
        validate_final_turn,
        validate_record,
    )
    result = validate_final_turn(content)
    if not result.is_valid:
        print(result.errors)

    # CLI
    python -m scope.utils.parse_challenger \\
        --input-file outputs/challenger.jsonl \\
        --verbose
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Valid task types for final turn validation
VALID_TASK_TYPES = frozenset({
    "long_form_qa",
    "writing",
    "summarization",
    "planning",
    "short_form_qa",
})

# Rubric count constraints
MIN_RUBRICS = 3
MAX_RUBRICS = 10


@dataclass
class ValidationResult:
    """Result of validating a single turn.

    Args:
        is_valid: Whether the turn passed validation.
        errors: List of error messages describing validation failures.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class RecordValidationResult:
    """Result of validating a complete conversation record.

    Args:
        intermediate_results: List of validation results for intermediate turns.
        final_result: Validation result for the final turn, or None if no final turn.
        all_intermediate_valid: Whether all intermediate turns passed.
        final_valid: Whether the final turn passed.
        all_valid: Whether all turns passed validation.
    """

    intermediate_results: list[ValidationResult] = field(default_factory=list)
    final_result: ValidationResult | None = None
    all_intermediate_valid: bool = True
    final_valid: bool = True
    all_valid: bool = True


@dataclass
class ExtractedTask:
    """Extracted task content from validated challenger output.

    Args:
        task_prompt: The task prompt/question for the solver.
        rubrics: List of rubric texts (positive only in v4). Empty list in
            v5 mode where rubrics are generated separately.
        priorities: List of priority values corresponding to each rubric.
            Values: "critical", "important", "bonus", or None if not specified.
        task_type: The task type string (e.g., "long_form_qa", "writing").
            Empty string if not extracted.
    """

    task_prompt: str
    rubrics: list[str]
    priorities: list[str | None]
    task_type: str = ""
    reference_answer: str = ""


def _validate_intermediate_turn_shared(content: str) -> ValidationResult:
    """Validate an intermediate turn (assistant message followed by tool_response).

    Intermediate turns must contain a valid tool_call with a search function.
    The tool_call must contain valid JSON with 'name': 'search' and
    'arguments': {'query_list': [...]}.

    Args:
        content: The assistant message content to validate.

    Returns:
        ValidationResult with is_valid flag and list of any errors found.
    """
    errors = []

    # Check for tool_call tags
    tool_call_match = re.search(
        r"<tool_call>(.*?)</tool_call>", content, re.DOTALL
    )
    if not tool_call_match:
        # Check if opening tag exists without closing
        if "<tool_call>" in content:
            errors.append("Unclosed <tool_call> tag")
        else:
            errors.append("Missing <tool_call> tags")
        return ValidationResult(is_valid=False, errors=errors)

    # Parse JSON inside tool_call
    json_str = tool_call_match.group(1).strip()
    try:
        tool_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        errors.append(f"Malformed JSON in tool_call: {e}")
        return ValidationResult(is_valid=False, errors=errors)

    # Validate JSON structure
    if not isinstance(tool_data, dict):
        errors.append("Tool call JSON must be an object")
        return ValidationResult(is_valid=False, errors=errors)

    # Check for 'name' key with value 'search'
    if "name" not in tool_data:
        errors.append("Missing 'name' key in tool call")
    elif tool_data["name"] != "search":
        errors.append(
            f"Invalid function name: expected 'search', got '{tool_data['name']}'"
        )

    # Check for 'arguments' key with 'query_list'
    if "arguments" not in tool_data:
        errors.append("Missing 'arguments' key in tool call")
    elif not isinstance(tool_data["arguments"], dict):
        errors.append("'arguments' must be an object")
    elif "query_list" not in tool_data["arguments"]:
        errors.append("Missing 'query_list' in arguments")
    elif not isinstance(tool_data["arguments"]["query_list"], list):
        errors.append("'query_list' must be an array")
    elif len(tool_data["arguments"]["query_list"]) == 0:
        errors.append("'query_list' is empty")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def _validate_xml_final(
    content: str, require_rubrics: bool = True
) -> ValidationResult:
    """Validate a final turn in XML format.

    When ``require_rubrics=False`` (v7/v8 simplified format), the task prompt
    is placed directly inside ``<task>...</task>`` with no sub-tags.

    When ``require_rubrics=True`` (v4 legacy format), the ``<task>`` block
    must contain ``<task_type>``, ``<task_prompt>``, and ``<rubrics>``
    sub-tags.

    Args:
        content: The assistant message content to validate.
        require_rubrics: If True (default), require ``<task_type>``,
            ``<task_prompt>``, and ``<rubrics>`` sub-tags (v4 format).
            If False, expect task prompt directly inside ``<task>``
            (v7/v8 simplified format).

    Returns:
        ValidationResult with is_valid flag and list of any errors found.
    """
    errors = []

    # Check for task tags.  Use the LAST match: multi-turn decoded
    # responses may contain instructional <task>...</task> from dynamic
    # user messages before the model's actual task output.
    task_matches = re.findall(r"<task>(.*?)</task>", content, re.DOTALL)
    if not task_matches:
        if "<task>" in content:
            errors.append("Unclosed <task> tag")
        else:
            errors.append("Missing <task> tags")
        return ValidationResult(is_valid=False, errors=errors)

    task_content = task_matches[-1]

    # --- Simplified format (v7/v8): task prompt directly inside <task> ---
    if not require_rubrics:
        if not task_content.strip():
            errors.append("Empty <task> content")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    # --- Legacy v4 format: sub-tags required ---

    # Check for task_type
    task_type_match = re.search(
        r"<task_type>(.*?)</task_type>", task_content, re.DOTALL
    )
    if not task_type_match:
        errors.append("Missing <task_type> tag")
    else:
        task_type = task_type_match.group(1).strip()
        if task_type not in VALID_TASK_TYPES:
            errors.append(
                f"Invalid task_type: '{task_type}'. "
                f"Valid types: {sorted(VALID_TASK_TYPES)}"
            )

    # Check for task_prompt
    task_prompt_match = re.search(
        r"<task_prompt>(.*?)</task_prompt>", task_content, re.DOTALL
    )
    if not task_prompt_match:
        errors.append("Missing <task_prompt> tag")
    else:
        task_prompt = task_prompt_match.group(1).strip()
        if not task_prompt:
            errors.append("Empty <task_prompt>")

    # Check for rubrics section
    rubrics_match = re.search(
        r"<rubrics>(.*?)</rubrics>", task_content, re.DOTALL
    )
    if not rubrics_match:
        errors.append("Missing <rubrics> section")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    rubrics_content = rubrics_match.group(1)

    # v4: Rubrics are directly under <rubrics> (no positive/negative wrappers)
    # Extract all rubric elements
    rubrics = re.findall(
        r"<rubric[^>]*>(.*?)</rubric>", rubrics_content, re.DOTALL
    )

    # Validate rubric count (3-10)
    if len(rubrics) < MIN_RUBRICS:
        errors.append(
            f"Too few rubrics: {len(rubrics)} (minimum {MIN_RUBRICS})"
        )
    elif len(rubrics) > MAX_RUBRICS:
        errors.append(
            f"Too many rubrics: {len(rubrics)} (maximum {MAX_RUBRICS})"
        )

    # Validate that rubrics have content
    empty_rubric_count = sum(1 for r in rubrics if not r.strip())
    if empty_rubric_count > 0:
        errors.append(f"{empty_rubric_count} empty rubric(s) found")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


# Two-step attribute-tolerant parsing: captures all attributes, then
# extracts priority separately.  Handles tags like
# <rubric priority="critical" type="factual"> with attributes in any order.
_RUBRIC_TAG_PATTERN = re.compile(
    r'<rubric\b([^>]*)>(.*?)</rubric>',
    re.DOTALL,
)
_PRIORITY_ATTR_PATTERN = re.compile(
    r'\bpriority=["\']?([^"\'>\s]+)["\']?'
)


def _extract_rubrics_with_priorities(
    rubrics_content: str,
) -> tuple[list[str], list[str | None]]:
    """Extract rubric texts and their priority attributes from flat v4 structure.

    Args:
        rubrics_content: Content inside <rubrics>...</rubrics> tags.

    Returns:
        Tuple of (list of rubric texts, list of priorities).
        Priorities are "critical", "important", "bonus", or None if not specified.
    """
    rubrics: list[str] = []
    priorities: list[str | None] = []

    for match in _RUBRIC_TAG_PATTERN.finditer(rubrics_content):
        attrs = match.group(1)
        text = match.group(2).strip()
        if text:  # Skip empty rubrics
            priority_match = _PRIORITY_ATTR_PATTERN.search(attrs)
            priority = priority_match.group(1) if priority_match else None
            rubrics.append(text)
            priorities.append(priority)

    return rubrics, priorities


def _validate_intermediate_turn_function_calls(content: str) -> ValidationResult:
    """Validate an intermediate turn using ``<function_calls>search(query=...)`` format.

    For OLMo function-calling format, intermediate turns must contain a
    ``<function_calls>search(query="...")`` call.

    Args:
        content: str, the assistant message content to validate.

    Returns:
        ValidationResult: Validation result with is_valid flag and errors.
    """
    errors: list[str] = []
    fc_match = re.search(
        r'<function_calls>\s*search\s*\(\s*query\s*=\s*["\'](.+?)["\']\s*\)',
        content, re.DOTALL,
    )
    if not fc_match:
        if "<function_calls>" in content:
            errors.append("No valid search function call found in <function_calls>")
        else:
            errors.append("Missing <function_calls> tags")
        return ValidationResult(is_valid=False, errors=errors)

    if not fc_match.group(1).strip():
        errors.append("Empty search query")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def _validate_intermediate_turn_search_r1(content: str) -> ValidationResult:
    """Validate an intermediate turn using ``<search>query</search>`` format.

    For Search-R1 style format, intermediate turns must contain a
    ``<search>`` tag with non-empty query text instead of ``<tool_call>``
    JSON.

    Args:
        content: str, the assistant message content to validate.

    Returns:
        ValidationResult: Validation result with is_valid flag and errors.
    """
    errors: list[str] = []
    search_match = re.search(r"<search>(.*?)</search>", content, re.DOTALL)
    if not search_match:
        if "<search>" in content:
            errors.append("Unclosed <search> tag")
        else:
            errors.append("Missing <search> tags")
        return ValidationResult(is_valid=False, errors=errors)

    if not search_match.group(1).strip():
        errors.append("Empty search query")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_intermediate_turn(
    content: str, challenger_format: str = "tool_call"
) -> ValidationResult:
    """Validate an intermediate turn (assistant message followed by tool_response).

    Intermediate turns must contain a valid tool_call with a search function
    (default) or a ``<search>`` tag (search_r1 format).

    Args:
        content: str, the assistant message content to validate.
        challenger_format: str, the format to validate against. Either
            ``"tool_call"`` (default) or ``"search_r1"``.

    Returns:
        ValidationResult with is_valid flag and list of any errors found.
    """
    if challenger_format == "search_r1":
        return _validate_intermediate_turn_search_r1(content)
    if challenger_format in ("function_calls", "function_calls_xml"):
        return _validate_intermediate_turn_function_calls(content)
    return _validate_intermediate_turn_shared(content)


def _validate_function_calls_final(content: str) -> ValidationResult:
    """Validate a final turn in function_calls format.

    Accepts the simplified format ``task(task_prompt="...")`` (v7+) as well
    as the legacy format ``task(task_type="...", task_prompt="...")``.

    Args:
        content: The assistant message content to validate.

    Returns:
        ValidationResult with is_valid flag and list of any errors found.
    """
    errors: list[str] = []

    # Try simplified format first: task(task_prompt="...")
    match = re.search(
        r'<function_calls>\s*task\s*\(\s*task_prompt\s*=\s*["\'](.+?)["\']\s*\)',
        content, re.DOTALL,
    )
    if match:
        task_prompt = match.group(1).strip()
        if not task_prompt:
            errors.append("Empty task_prompt")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    # Try legacy format: task(task_type="...", task_prompt="...")
    match = re.search(
        r'<function_calls>\s*task\s*\('
        r'\s*task_type\s*=\s*["\'](.+?)["\']\s*'
        r',\s*task_prompt\s*=\s*["\'](.+?)["\']\s*\)',
        content, re.DOTALL,
    )
    if not match:
        # Try reversed argument order
        match = re.search(
            r'<function_calls>\s*task\s*\('
            r'\s*task_prompt\s*=\s*["\'](.+?)["\']\s*'
            r',\s*task_type\s*=\s*["\'](.+?)["\']\s*\)',
            content, re.DOTALL,
        )
    if not match:
        if "<function_calls>" in content and "task(" in content:
            errors.append("Malformed task function call arguments")
        else:
            errors.append("Missing task function call")
        return ValidationResult(is_valid=False, errors=errors)

    # Legacy format matched -- extract task_prompt depending on arg order.
    # First regex: task_type=group(1), task_prompt=group(2)
    # Reversed regex: task_prompt=group(1), task_type=group(2)
    # Check which regex matched by looking for task_type before task_prompt
    raw = match.group(0)
    tt_pos = raw.find("task_type")
    tp_pos = raw.find("task_prompt")
    if tt_pos < tp_pos:
        task_prompt = match.group(2).strip()
    else:
        task_prompt = match.group(1).strip()

    if not task_prompt:
        errors.append("Empty task_prompt")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_final_turn(
    content: str, require_rubrics: bool = True,
    challenger_format: str = "xml",
) -> ValidationResult:
    """Validate a final turn (last assistant message, no subsequent tool_response).

    Final turns must contain a complete task specification with task_type,
    task_prompt, and (when ``require_rubrics=True``) rubrics containing 3-10
    rubric elements directly under the ``<rubrics>`` tag.

    Args:
        content: The assistant message content to validate.
        require_rubrics: If True (default), require ``<rubrics>`` section.
            If False, skip rubric validation (for v5 pipeline where rubrics
            are generated separately).
        challenger_format: Format to validate against. ``"xml"`` (default)
            or ``"function_calls"`` for OLMo function-calling format.

    Returns:
        ValidationResult with is_valid flag and list of any errors found.
    """
    if challenger_format == "function_calls":
        return _validate_function_calls_final(content)
    return _validate_xml_final(content, require_rubrics=require_rubrics)


def extract_task_function_call(
    content: str,
) -> tuple[ExtractedTask | None, list[str]]:
    """Validate and extract task from a function_calls format final turn.

    Accepts the simplified ``task(task_prompt="...")`` format (v7+) as
    well as the legacy ``task(task_type="...", task_prompt="...")`` format.
    Returns an ``ExtractedTask`` with empty rubrics/priorities (rubrics
    are generated separately).

    Args:
        content: The challenger's final assistant response in function_calls format.

    Returns:
        Tuple of (ExtractedTask or None if failed, list of error messages).
    """
    validation = _validate_function_calls_final(content)
    if not validation.is_valid:
        return None, validation.errors

    # Try simplified format first: task(task_prompt="...")
    match = re.search(
        r'<function_calls>\s*task\s*\(\s*task_prompt\s*=\s*["\'](.+?)["\']\s*\)',
        content, re.DOTALL,
    )
    if match:
        return (
            ExtractedTask(
                task_prompt=match.group(1).strip(),
                rubrics=[],
                priorities=[],
            ),
            [],
        )

    # Try legacy format: task(task_type="...", task_prompt="...")
    match = re.search(
        r'<function_calls>\s*task\s*\('
        r'\s*task_type\s*=\s*["\'](.+?)["\']\s*'
        r',\s*task_prompt\s*=\s*["\'](.+?)["\']\s*\)',
        content, re.DOTALL,
    )
    if match:
        return (
            ExtractedTask(
                task_prompt=match.group(2).strip(),
                rubrics=[],
                priorities=[],
            ),
            [],
        )

    # Try reversed argument order
    match = re.search(
        r'<function_calls>\s*task\s*\('
        r'\s*task_prompt\s*=\s*["\'](.+?)["\']\s*'
        r',\s*task_type\s*=\s*["\'](.+?)["\']\s*\)',
        content, re.DOTALL,
    )
    if match:
        return (
            ExtractedTask(
                task_prompt=match.group(1).strip(),
                rubrics=[],
                priorities=[],
            ),
            [],
        )

    return None, ["Failed to extract task function call arguments"]


def _extract_submit_task_tool_call(content: str) -> ExtractedTask | None:
    """Extract task from a submit_task tool call in hermes format.

    Args:
        content: Assistant turn content potentially containing a
            ``<tool_call>{"name": "submit_task", ...}</tool_call>`` block.

    Returns:
        ExtractedTask if a valid submit_task call is found, None otherwise.
    """
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        try:
            tc = json.loads(match.group(1).strip())
            if (
                isinstance(tc, dict)
                and tc.get("name") == "submit_task"
                and isinstance(tc.get("arguments"), dict)
            ):
                task_prompt = tc["arguments"].get("task_prompt", "").strip()
                if task_prompt:
                    return ExtractedTask(
                        task_prompt=task_prompt,
                        rubrics=[],
                        priorities=[],
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return None


def extract_task(
    content: str, require_rubrics: bool = True,
    challenger_format: str = "xml",
) -> tuple[ExtractedTask | None, list[str]]:
    """Validate and extract task from challenger output.

    Combines validation and extraction into one convenient function for
    training.  When ``require_rubrics=False`` (v7/v8 simplified format),
    the task prompt is taken directly from ``<task>...</task>`` content
    without sub-tags.  When ``require_rubrics=True`` (v4 legacy), uses
    ``<task_prompt>`` sub-tag and extracts rubrics.

    Args:
        content: The challenger's final assistant response.
        require_rubrics: If True (default), require and extract rubrics
            (v4 behavior). If False, return empty rubrics/priorities
            (v7/v8 simplified format).
        challenger_format: Format to use. ``"xml"`` (default) for ``<task>``
            XML tags, ``"function_calls"`` for OLMo function-calling format,
            ``"function_calls_xml"`` for hybrid (FC search + XML task).

    Returns:
        Tuple of (ExtractedTask or None if failed, list of error messages).
    """
    if challenger_format == "function_calls":
        return extract_task_function_call(content)

    # Validate format first — pass challenger_format so that
    # function_calls_xml routes to the XML validator correctly.
    validation = validate_final_turn(
        content,
        require_rubrics=require_rubrics,
        challenger_format=challenger_format,
    )
    if not validation.is_valid:
        # Fallback: check for submit_task tool call (hermes/tool_call format)
        if challenger_format == "tool_call":
            result = _extract_submit_task_tool_call(content)
            if result is not None:
                return result, []
        return None, validation.errors

    # Extract task block (must exist if validation passed).
    # Use the LAST match: multi-turn decoded responses may contain
    # instructional <task>...</task> from dynamic user messages before
    # the model's actual task output.
    task_matches = re.findall(r"<task>(.*?)</task>", content, re.DOTALL)
    if not task_matches:
        return None, ["Failed to extract <task> content"]
    task_content = task_matches[-1]

    # --- Simplified format (v7/v8): task prompt is direct <task> content ---
    if not require_rubrics:
        task_prompt = task_content.strip()
        # Extract reference answer from <answer> tag if present (short_form_qa)
        answer_matches = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
        reference_answer = answer_matches[-1].strip() if answer_matches else ""
        return (
            ExtractedTask(
                task_prompt=task_prompt,
                rubrics=[],
                priorities=[],
                reference_answer=reference_answer,
            ),
            [],
        )

    # --- Legacy v4 format: extract from sub-tags ---

    # Extract task_prompt
    prompt_match = re.search(
        r"<task_prompt>(.*?)</task_prompt>", task_content, re.DOTALL
    )
    if not prompt_match:
        return None, ["Failed to extract <task_prompt>"]
    task_prompt = prompt_match.group(1).strip()

    # Extract task_type
    task_type_match = re.search(
        r"<task_type>(.*?)</task_type>", task_content, re.DOTALL
    )
    task_type = task_type_match.group(1).strip() if task_type_match else ""

    # Extract rubrics section
    rubrics_match = re.search(
        r"<rubrics>(.*?)</rubrics>", task_content, re.DOTALL
    )
    if not rubrics_match:
        return None, ["Failed to extract <rubrics>"]

    # Extract rubrics with priorities using helper
    rubrics, priorities = _extract_rubrics_with_priorities(rubrics_match.group(1))

    if len(rubrics) == 0:
        return None, ["No rubrics found"]

    return (
        ExtractedTask(
            task_prompt=task_prompt,
            rubrics=rubrics,
            priorities=priorities,
            task_type=task_type,
        ),
        [],
    )


def classify_turns(
    conversation: list[dict[str, str]],
) -> tuple[list[tuple[int, str]], tuple[int, str] | None]:
    """Classify assistant turns as intermediate or final.

    An intermediate turn is any assistant message before the last assistant message.
    The final turn is the last assistant message in the conversation.

    Args:
        conversation: List of message dicts with 'role' and 'content' keys.

    Returns:
        A tuple of:
            - List of (index, content) tuples for intermediate turns
            - (index, content) tuple for final turn, or None if no assistant turns
    """
    # Collect all assistant turns
    assistant_turns = [
        (i, msg.get("content", ""))
        for i, msg in enumerate(conversation)
        if msg.get("role") == "assistant"
    ]

    if not assistant_turns:
        return [], None

    # All but last are intermediate, last is final
    intermediate_turns = assistant_turns[:-1]
    final_turn = assistant_turns[-1]

    return intermediate_turns, final_turn


def validate_record(
    record: dict[str, Any],
    challenger_format: str = "tool_call",
) -> RecordValidationResult:
    """Validate a complete conversation record.

    Args:
        record: A record dict containing 'conversation' and other metadata.
        challenger_format: str, the format to validate against. Either
            ``"tool_call"`` (default) or ``"search_r1"``.

    Returns:
        RecordValidationResult with validation results for all turns.
    """
    result = RecordValidationResult()

    conversation = record.get("conversation", [])
    if not conversation:
        result.all_valid = False
        result.final_valid = False
        result.final_result = ValidationResult(
            is_valid=False, errors=["Empty conversation"]
        )
        return result

    intermediate_turns, final_turn = classify_turns(conversation)

    # Validate intermediate turns
    for idx, content in intermediate_turns:
        validation = validate_intermediate_turn(
            content, challenger_format=challenger_format
        )
        result.intermediate_results.append(validation)
        if not validation.is_valid:
            result.all_intermediate_valid = False

    # Validate final turn
    if final_turn:
        idx, content = final_turn
        # function_calls and function_calls_xml use simplified <task>prompt</task>
        # without sub-tags, so require_rubrics must be False.
        _require_rubrics = challenger_format not in (
            "function_calls", "function_calls_xml",
        )
        result.final_result = validate_final_turn(
            content,
            require_rubrics=_require_rubrics,
            challenger_format=challenger_format,
        )
        result.final_valid = result.final_result.is_valid
    else:
        result.final_result = ValidationResult(
            is_valid=False, errors=["No final assistant turn found"]
        )
        result.final_valid = False

    result.all_valid = result.all_intermediate_valid and result.final_valid

    return result


def aggregate_metrics(
    results: list[tuple[dict[str, Any], RecordValidationResult]],
) -> dict[str, Any]:
    """Aggregate validation metrics across all records.

    Args:
        results: List of (record, validation_result) tuples.

    Returns:
        Dict containing aggregated metrics for intermediate and final turns.
    """
    total_records = len(results)
    total_intermediate = 0
    intermediate_pass = 0
    intermediate_errors: Counter[str] = Counter()

    total_final = 0
    final_pass = 0
    final_errors: Counter[str] = Counter()

    records_all_pass = 0
    records_final_pass = 0

    for record, validation in results:
        # Intermediate turn metrics
        for int_result in validation.intermediate_results:
            total_intermediate += 1
            if int_result.is_valid:
                intermediate_pass += 1
            else:
                for error in int_result.errors:
                    intermediate_errors[error] += 1

        # Final turn metrics
        if validation.final_result:
            total_final += 1
            if validation.final_result.is_valid:
                final_pass += 1
                records_final_pass += 1
            else:
                for error in validation.final_result.errors:
                    final_errors[error] += 1

        # Overall metrics
        if validation.all_valid:
            records_all_pass += 1

    return {
        "total_records": total_records,
        "intermediate_turns": {
            "total": total_intermediate,
            "passed": intermediate_pass,
            "failed": total_intermediate - intermediate_pass,
            "pass_ratio": (
                intermediate_pass / total_intermediate
                if total_intermediate > 0
                else 0.0
            ),
            "common_errors": dict(intermediate_errors.most_common(10)),
        },
        "final_turns": {
            "total": total_final,
            "passed": final_pass,
            "failed": total_final - final_pass,
            "pass_ratio": final_pass / total_final if total_final > 0 else 0.0,
            "common_errors": dict(final_errors.most_common(10)),
        },
        "overall": {
            "records_all_valid": records_all_pass,
            "records_all_valid_ratio": (
                records_all_pass / total_records if total_records > 0 else 0.0
            ),
            "records_final_valid": records_final_pass,
            "records_final_valid_ratio": (
                records_final_pass / total_records if total_records > 0 else 0.0
            ),
        },
    }


def main() -> None:
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate outputs from multiturn_search_generation.py with "
            "challenger prompt for format compliance (XML only)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate challenger outputs
  python -m scope.utils.parse_challenger \\
      --input-file outputs/challenger.jsonl

  # Detailed output with per-record results
  python -m scope.utils.parse_challenger \\
      --input-file outputs/challenger.jsonl \\
      --output-file outputs/validation_results.jsonl \\
      --verbose
        """,
    )

    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to JSONL output from multiturn_search_generation.py",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to save detailed validation results as JSONL",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-record validation details",
    )

    args = parser.parse_args()

    # Load input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")

    print(f"Loaded {len(records)} records from {args.input_file}")
    print("Validating format: xml")

    # Validate all records
    results = []
    for idx, record in enumerate(records):
        validation = validate_record(record)
        results.append((record, validation))

        if args.verbose:
            print(f"\n--- Record {idx} ---")
            print(
                f"Intermediate turns: {len(validation.intermediate_results)} "
                f"(all valid: {validation.all_intermediate_valid})"
            )
            for i, int_res in enumerate(validation.intermediate_results):
                status = "PASS" if int_res.is_valid else "FAIL"
                print(f"  Turn {i}: {status}")
                for error in int_res.errors:
                    print(f"    - {error}")

            if validation.final_result:
                status = "PASS" if validation.final_result.is_valid else "FAIL"
                print(f"Final turn: {status}")
                for error in validation.final_result.errors:
                    print(f"  - {error}")

    # Aggregate and print metrics
    metrics = aggregate_metrics(results)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal records: {metrics['total_records']}")
    print("Format: xml")

    print("\n--- Intermediate Turns ---")
    int_metrics = metrics["intermediate_turns"]
    print(f"Total: {int_metrics['total']}")
    print(f"Passed: {int_metrics['passed']}")
    print(f"Failed: {int_metrics['failed']}")
    print(f"Pass ratio: {int_metrics['pass_ratio']:.2%}")
    if int_metrics["common_errors"]:
        print("Common errors:")
        for error, count in int_metrics["common_errors"].items():
            print(f"  - {error}: {count}")

    print("\n--- Final Turns ---")
    final_metrics = metrics["final_turns"]
    print(f"Total: {final_metrics['total']}")
    print(f"Passed: {final_metrics['passed']}")
    print(f"Failed: {final_metrics['failed']}")
    print(f"Pass ratio: {final_metrics['pass_ratio']:.2%}")
    if final_metrics["common_errors"]:
        print("Common errors:")
        for error, count in final_metrics["common_errors"].items():
            print(f"  - {error}: {count}")

    print("\n--- Overall ---")
    overall = metrics["overall"]
    print(
        f"Records with all turns valid: {overall['records_all_valid']} "
        f"({overall['records_all_valid_ratio']:.2%})"
    )
    print(
        f"Records with final turn valid: {overall['records_final_valid']} "
        f"({overall['records_final_valid_ratio']:.2%})"
    )

    # Save detailed results if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            # Write per-record results
            for idx, (record, validation) in enumerate(results):
                result_record = {
                    "record_idx": idx,
                    "prompt_idx": record.get("prompt_idx"),
                    "num_turns": record.get("num_turns"),
                    "num_searches": record.get("num_searches"),
                    "format": "xml",
                    "all_valid": validation.all_valid,
                    "all_intermediate_valid": validation.all_intermediate_valid,
                    "final_valid": validation.final_valid,
                    "intermediate_errors": [
                        err
                        for res in validation.intermediate_results
                        for err in res.errors
                    ],
                    "final_errors": (
                        validation.final_result.errors
                        if validation.final_result
                        else []
                    ),
                }
                f.write(json.dumps(result_record, ensure_ascii=False) + "\n")

            # Write summary as last line
            f.write(json.dumps({"summary": metrics}, ensure_ascii=False) + "\n")

        print(f"\nDetailed results saved to {args.output_file}")


if __name__ == "__main__":
    main()
