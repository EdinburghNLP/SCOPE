"""Parser for standalone rubric generation output.

Parses rubric generation LLM responses in either XML (``<rubrics>`` tags) or
JSON (``{"rubrics": [...]}`` object) format, returning structured rubric texts,
priorities, and validation results.

Example usage::

    from scope.utils.parse_rubric import parse_rubric_output

    result = parse_rubric_output(llm_output)
    if result.is_valid:
        print(result.rubrics, result.priorities)

    # JSON format
    result = parse_rubric_output(llm_output, output_format="json")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal


# Type for output format selection (mirrors parse_grader.OutputFormat)
OutputFormat = Literal["xml", "json"]

# Rubric count constraints
MIN_RUBRICS = 3
MAX_RUBRICS = 10

# Two-step attribute-tolerant parsing: captures all attributes, then
# extracts priority separately.  Handles rubric_v18b tags like
# <rubric priority="critical" type="factual"> with attributes in any order.
_RUBRIC_TAG_PATTERN = re.compile(
    r'<rubric\b([^>]*)>(.*?)</rubric>',
    re.DOTALL,
)
_PRIORITY_ATTR_PATTERN = re.compile(
    r'\bpriority=["\']?([^"\'>\s]+)["\']?'
)


@dataclass
class RubricParseResult:
    """Result of parsing standalone rubric generation output.

    Args:
        rubrics: List of rubric text strings.
        priorities: List of priority values corresponding to each rubric.
            Values: ``"critical"``, ``"important"``, ``"bonus"``, or ``None``
            if not specified.
        is_valid: Whether the output passed all validation checks.
        errors: List of error messages describing validation failures.
    """

    rubrics: list[str] = field(default_factory=list)
    priorities: list[str | None] = field(default_factory=list)
    is_valid: bool = False
    errors: list[str] = field(default_factory=list)


def _parse_xml_rubric(
    content: str,
    min_rubrics: int = MIN_RUBRICS,
    max_rubrics: int = MAX_RUBRICS,
) -> RubricParseResult:
    """Parse ``<rubrics>`` XML from rubric generation LLM output.

    Args:
        content: The rubric generation LLM response text.
        min_rubrics: Minimum number of rubrics required for validity.
            Defaults to ``MIN_RUBRICS`` (3). Set to 1 for per-turn parsing
            where individual outputs have fewer rubrics.
        max_rubrics: Maximum number of rubrics allowed for validity.
            Defaults to ``MAX_RUBRICS`` (10).

    Returns:
        RubricParseResult with rubrics, priorities, validity, and errors.
    """
    errors: list[str] = []

    # Find <rubrics> section
    rubrics_match = re.search(
        r"<rubrics>(.*?)</rubrics>", content, re.DOTALL
    )
    if not rubrics_match:
        if "<rubrics>" in content:
            errors.append("Unclosed <rubrics> tag")
        else:
            errors.append("Missing <rubrics> tags")
        return RubricParseResult(errors=errors)

    rubrics_content = rubrics_match.group(1)

    # Extract rubrics with priorities (attribute-tolerant two-step parsing)
    rubrics: list[str] = []
    priorities: list[str | None] = []

    for match in _RUBRIC_TAG_PATTERN.finditer(rubrics_content):
        attrs = match.group(1)
        text = match.group(2).strip()
        if text:
            priority_match = _PRIORITY_ATTR_PATTERN.search(attrs)
            priority = priority_match.group(1) if priority_match else None
            rubrics.append(text)
            priorities.append(priority)
        else:
            errors.append("Empty rubric text found")

    # Validate rubric count
    if len(rubrics) < min_rubrics:
        errors.append(
            f"Too few rubrics: {len(rubrics)} (minimum {min_rubrics})"
        )
    elif len(rubrics) > max_rubrics:
        errors.append(
            f"Too many rubrics: {len(rubrics)} (maximum {max_rubrics})"
        )

    is_valid = len(errors) == 0
    return RubricParseResult(
        rubrics=rubrics,
        priorities=priorities,
        is_valid=is_valid,
        errors=errors,
    )


def _parse_json_rubric(content: str) -> RubricParseResult:
    """Parse JSON rubric object from rubric generation LLM output.

    Expects a JSON object with structure::

        {"rubrics": [{"text": "...", "priority": "..."}, ...]}

    Items may omit ``priority`` (defaults to ``None``), and items may be
    plain strings instead of objects (priority defaults to ``None``).

    Args:
        content: The rubric generation LLM response text.

    Returns:
        RubricParseResult with rubrics, priorities, validity, and errors.
    """
    errors: list[str] = []
    stripped = content.strip()

    # Find JSON object via bracket matching (LLMs may add preamble text)
    start = stripped.find("{")
    if start == -1:
        errors.append("No JSON object found")
        return RubricParseResult(errors=errors)

    # Find matching closing brace
    depth = 0
    end = -1
    for i in range(start, len(stripped)):
        if stripped[i] == "{":
            depth += 1
        elif stripped[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        errors.append("No JSON object found")
        return RubricParseResult(errors=errors)

    json_str = stripped[start : end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        errors.append(f"JSON parse error: {e}")
        return RubricParseResult(errors=errors)

    if not isinstance(data, dict) or "rubrics" not in data:
        errors.append("Missing 'rubrics' key")
        return RubricParseResult(errors=errors)

    raw_rubrics = data["rubrics"]
    if not isinstance(raw_rubrics, list):
        errors.append("Missing 'rubrics' key")
        return RubricParseResult(errors=errors)

    rubrics: list[str] = []
    priorities: list[str | None] = []

    for item in raw_rubrics:
        if isinstance(item, str):
            text = item.strip()
            priority = None
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            priority = item.get("priority", None)
            if priority is not None:
                priority = str(priority).strip() or None
        else:
            errors.append("Empty rubric text")
            continue

        if text:
            rubrics.append(text)
            priorities.append(priority)
        else:
            errors.append("Empty rubric text")

    # Validate rubric count
    if len(rubrics) < MIN_RUBRICS:
        errors.append(
            f"Too few rubrics: {len(rubrics)} (minimum {MIN_RUBRICS})"
        )
    elif len(rubrics) > MAX_RUBRICS:
        errors.append(
            f"Too many rubrics: {len(rubrics)} (maximum {MAX_RUBRICS})"
        )

    is_valid = len(errors) == 0
    return RubricParseResult(
        rubrics=rubrics,
        priorities=priorities,
        is_valid=is_valid,
        errors=errors,
    )


def parse_rubric_output(
    content: str,
    output_format: OutputFormat = "xml",
    min_rubrics: int = MIN_RUBRICS,
    max_rubrics: int = MAX_RUBRICS,
) -> RubricParseResult:
    """Parse rubric generation LLM output in XML or JSON format.

    Extracts rubric texts and priorities from the output, validates rubric
    count, and checks for non-empty text.

    Args:
        content: The rubric generation LLM response text.
        output_format: Format of the output — ``"xml"`` for ``<rubrics>``
            XML tags, ``"json"`` for ``{"rubrics": [...]}`` JSON object.
        min_rubrics: Minimum number of rubrics required for validity.
            Defaults to ``MIN_RUBRICS`` (3). Use 1 for per-turn rubric
            parsing where individual outputs may contain fewer rubrics.
        max_rubrics: Maximum number of rubrics allowed for validity.
            Defaults to ``MAX_RUBRICS`` (10).

    Returns:
        RubricParseResult with rubrics, priorities, validity, and errors.

    Raises:
        ValueError: If ``output_format`` is not ``"xml"`` or ``"json"``.
    """
    if output_format == "xml":
        return _parse_xml_rubric(content, min_rubrics=min_rubrics, max_rubrics=max_rubrics)
    elif output_format == "json":
        return _parse_json_rubric(content)
    else:
        raise ValueError(
            f"Unsupported output_format: {output_format!r} "
            f"(expected 'xml' or 'json')"
        )
