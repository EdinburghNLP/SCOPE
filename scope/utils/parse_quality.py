"""Parsers for quality grading outputs.

Parses ``<score>0 or 1</score>`` formatted responses from binary quality
graders and computes a gate over multiple rubric scores.
"""

from __future__ import annotations

import re
from typing import Optional

# Canonical gate names and the order they appear in template lists.
VALID_QUALITY_GATES = ("entity", "no_leakage", "retrieval", "source_relevance")

# Matches <score>0</score> or <score>1</score> with optional whitespace.
_SCORE_RE = re.compile(r"<score>\s*(0|1)\s*</score>")


def parse_quality_score(text: str) -> Optional[int]:
    """Parse a binary quality score from grader output.

    Extracts the first ``<score>0 or 1</score>`` tag found in *text*.

    Args:
        text: Raw grader output string, typically containing
            ``<think>...</think><score>0 or 1</score>``.

    Returns:
        int | None: ``0`` or ``1`` if a valid score tag is found,
            ``None`` if the text is empty or contains no parseable score.
    """
    if not text:
        return None
    match = _SCORE_RE.search(text)
    if match is None:
        return None
    return int(match.group(1))


def compute_quality_gate(
    scores: list[Optional[int]],
    required: int = -1,
) -> bool:
    """Check whether enough quality rubric scores pass.

    Args:
        scores: List of individual quality scores (each ``0``, ``1``,
            or ``None`` for unparseable).
        required: Minimum sum of scores to pass the gate. Defaults
            to ``-1`` which means all gates must pass (i.e.
            ``required = len(scores)``).

    Returns:
        bool: ``True`` only if no score is ``None`` and the sum of
            scores is at least *required*.
    """
    if not scores:
        return False
    if required < 0:
        required = len(scores)
    if any(s is None for s in scores):
        return False
    return sum(scores) >= required  # type: ignore[arg-type]


def parse_quality_gates(gates_str: str) -> list[str]:
    """Parse a comma-separated string of quality gate names.

    Args:
        gates_str: Comma-separated gate names, e.g.
            ``"entity,no_leakage"`` or ``"entity,no_leakage,retrieval"``.
            Empty or whitespace-only strings return an empty list.

    Returns:
        list[str]: Ordered list of validated gate names.

    Raises:
        ValueError: If any gate name is not in
            :data:`VALID_QUALITY_GATES`.
    """
    if not gates_str or not gates_str.strip():
        return []
    gates = [g.strip() for g in gates_str.split(",") if g.strip()]
    for g in gates:
        if g not in VALID_QUALITY_GATES:
            raise ValueError(
                f"Unknown quality gate: {g!r}. "
                f"Valid gates: {VALID_QUALITY_GATES}"
            )
    return gates
