"""Parser for grader model outputs.

This module extracts structured evaluation information from grader LLM outputs,
supporting both XML and JSON output formats.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Literal

# Type for output format selection
OutputFormat = Literal["xml", "json", "v2", "per_rubric"]

# XML regex patterns
EVALUATION_PATTERN = re.compile(r"<evaluation>(.*?)</evaluation>", re.DOTALL)
RUBRIC_ASSESSMENT_PATTERN = re.compile(
    r"<rubric_assessment>(.*?)</rubric_assessment>", re.DOTALL
)
RUBRIC_PATTERN = re.compile(r"<rubric>(.*?)</rubric>", re.DOTALL)
REASONING_PATTERN = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
SCORE_PATTERN = re.compile(r"<score>\s*([\d.]+)\s*</score>", re.DOTALL)

# V2 regex patterns
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
SCORES_PATTERN = re.compile(r"<scores>(.*?)</scores>", re.DOTALL)

# Valid scores
VALID_SCORES = {0, 0.5, 1}


@dataclass
class RubricAssessment:
    """Assessment result for a single rubric.

    Args:
        rubric: Summary or text of the rubric being assessed.
        reasoning: Explanation for the score given.
        score: Float score (0=not met, 0.5=partially met, 1=fully met).
    """

    rubric: str
    reasoning: str
    score: float

    def __post_init__(self) -> None:
        """Validate score is in valid range."""
        if self.score not in VALID_SCORES:
            raise ValueError(f"Score must be 0, 0.5, or 1, got {self.score}")


@dataclass
class GraderParseResult:
    """Result of parsing a grader model output.

    Args:
        assessments: List of rubric assessments in order.
        is_valid: Whether parsing succeeded and results are usable.
        errors: List of error messages encountered during parsing.
        raw_response: The original unparsed response string.
    """

    assessments: List[RubricAssessment] = field(default_factory=list)
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    raw_response: str = ""

    @property
    def total_score(self) -> float:
        """Calculate total score across all assessments.

        Returns:
            Sum of all rubric scores.
        """
        return sum(a.score for a in self.assessments)

    @property
    def max_score(self) -> int:
        """Calculate maximum possible score.

        Returns:
            Number of assessments multiplied by max score per rubric (1).
        """
        return len(self.assessments) * 1

    @property
    def normalized_score(self) -> float:
        """Calculate normalized score as fraction of maximum.

        Returns:
            Score as float between 0.0 and 1.0, or 0.0 if no assessments.
        """
        if self.max_score == 0:
            return 0.0
        return self.total_score / self.max_score

    @property
    def num_assessments(self) -> int:
        """Get count of rubric assessments.

        Returns:
            Number of assessments parsed.
        """
        return len(self.assessments)


def _parse_xml_grader(content: str) -> GraderParseResult:
    """Parse grader output in XML format.

    Expected format:
    <evaluation>
        <rubric_assessment>
            <rubric>summary</rubric>
            <reasoning>explanation</reasoning>
            <score>0|1|2</score>
        </rubric_assessment>
        ...
    </evaluation>

    Args:
        content: The raw grader response text in XML format.

    Returns:
        GraderParseResult with extracted assessments and any errors.
    """
    errors = []
    assessments = []

    # Find evaluation block
    eval_match = EVALUATION_PATTERN.search(content)
    if not eval_match:
        if "<evaluation>" in content:
            errors.append("Unclosed <evaluation> tag")
        else:
            errors.append("Missing <evaluation> tags")
        return GraderParseResult(
            is_valid=False, errors=errors, raw_response=content
        )

    eval_content = eval_match.group(1)

    # Find all rubric assessments
    assessment_matches = RUBRIC_ASSESSMENT_PATTERN.findall(eval_content)
    if not assessment_matches:
        errors.append("No <rubric_assessment> blocks found")
        return GraderParseResult(
            is_valid=False, errors=errors, raw_response=content
        )

    # Parse each assessment
    for idx, assessment_content in enumerate(assessment_matches):
        assessment_errors = []

        # Extract rubric
        rubric_match = RUBRIC_PATTERN.search(assessment_content)
        if not rubric_match:
            assessment_errors.append(f"Assessment {idx + 1}: Missing <rubric> tag")
            rubric = ""
        else:
            rubric = rubric_match.group(1).strip()

        # Extract reasoning
        reasoning_match = REASONING_PATTERN.search(assessment_content)
        if not reasoning_match:
            assessment_errors.append(
                f"Assessment {idx + 1}: Missing <reasoning> tag"
            )
            reasoning = ""
        else:
            reasoning = reasoning_match.group(1).strip()

        # Extract score
        score_match = SCORE_PATTERN.search(assessment_content)
        if not score_match:
            assessment_errors.append(f"Assessment {idx + 1}: Missing <score> tag")
            errors.extend(assessment_errors)
            continue  # Skip this assessment if no score

        try:
            score = float(score_match.group(1))
            if score not in VALID_SCORES:
                assessment_errors.append(
                    f"Assessment {idx + 1}: Invalid score {score}, must be 0, 0.5, or 1"
                )
                errors.extend(assessment_errors)
                continue
        except ValueError:
            assessment_errors.append(
                f"Assessment {idx + 1}: Non-numeric score: {score_match.group(1)}"
            )
            errors.extend(assessment_errors)
            continue

        errors.extend(assessment_errors)
        assessments.append(
            RubricAssessment(rubric=rubric, reasoning=reasoning, score=score)
        )

    return GraderParseResult(
        assessments=assessments,
        is_valid=len(assessments) > 0 and len(errors) == 0,
        errors=errors,
        raw_response=content,
    )


def _parse_json_grader(content: str) -> GraderParseResult:
    """Parse grader output in JSON format.

    Expected format:
    [
        {"rubric": "...", "reasoning": "...", "score": 0|1|2},
        ...
    ]

    Args:
        content: The raw grader response text in JSON format.

    Returns:
        GraderParseResult with extracted assessments and any errors.
    """
    errors = []
    assessments = []

    # Try to find JSON array in content
    # Sometimes LLMs add explanation text around JSON
    content_stripped = content.strip()

    # Try to extract JSON array using bracket matching
    json_start = content_stripped.find("[")
    json_end = content_stripped.rfind("]")

    if json_start == -1 or json_end == -1 or json_end <= json_start:
        errors.append("No JSON array found in response")
        return GraderParseResult(
            is_valid=False, errors=errors, raw_response=content
        )

    json_str = content_stripped[json_start : json_end + 1]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        errors.append(f"JSON parse error: {e}")
        return GraderParseResult(
            is_valid=False, errors=errors, raw_response=content
        )

    if not isinstance(parsed, list):
        errors.append(f"Expected JSON array, got {type(parsed).__name__}")
        return GraderParseResult(
            is_valid=False, errors=errors, raw_response=content
        )

    # Parse each assessment object
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            errors.append(
                f"Assessment {idx + 1}: Expected object, got {type(item).__name__}"
            )
            continue

        # Extract fields with defaults
        rubric = item.get("rubric", "")
        if not isinstance(rubric, str):
            rubric = str(rubric)
        rubric = rubric.strip()

        reasoning = item.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)
        reasoning = reasoning.strip()

        score_raw = item.get("score")
        if score_raw is None:
            errors.append(f"Assessment {idx + 1}: Missing 'score' field")
            continue

        try:
            score = float(score_raw)
        except (ValueError, TypeError):
            errors.append(
                f"Assessment {idx + 1}: Invalid score type: {score_raw}"
            )
            continue

        if score not in VALID_SCORES:
            errors.append(
                f"Assessment {idx + 1}: Invalid score {score}, must be 0, 0.5, or 1"
            )
            continue

        assessments.append(
            RubricAssessment(rubric=rubric, reasoning=reasoning, score=score)
        )

    return GraderParseResult(
        assessments=assessments,
        is_valid=len(assessments) > 0 and len(errors) == 0,
        errors=errors,
        raw_response=content,
    )


def _try_parse_bare_scores(text: str) -> str | None:
    """Try to interpret text as bare comma-separated scores (no tags).

    Validates that every comma-separated token is a valid score (0, 0.5, or 1).
    Returns the validated text if successful, or None if the text does not
    represent valid bare scores.

    Args:
        text: Whitespace-stripped text with any <think> block already removed.

    Returns:
        The bare scores string if all tokens are valid scores, None otherwise.
    """
    if not text:
        return None
    tokens = [t.strip() for t in text.split(",")]
    for token in tokens:
        if not token:
            return None
        try:
            value = float(token)
        except ValueError:
            return None
        if value not in VALID_SCORES:
            return None
    return text


def _parse_v2_grader(content: str) -> GraderParseResult:
    """Parse grader output in v2 format (think + comma-separated scores).

    Expected format:
    <think>reasoning for all rubrics</think>
    <scores>1, 0.5, 0, ...</scores>

    Args:
        content: The raw grader response text in v2 format.

    Returns:
        GraderParseResult with extracted assessments and any errors.
    """
    errors = []
    assessments = []

    # Extract optional think block
    think_match = THINK_PATTERN.search(content)
    reasoning = think_match.group(1).strip() if think_match else ""

    # Extract scores: try <scores> tags first, then fall back to bare scores
    scores_match = SCORES_PATTERN.search(content)
    if scores_match:
        scores_text = scores_match.group(1).strip()
    else:
        # Fallback: try to parse bare comma-separated scores from the content
        # Strip out the <think> block if present, then check remaining text
        bare_text = THINK_PATTERN.sub("", content).strip()
        bare_scores_text = _try_parse_bare_scores(bare_text)
        if bare_scores_text is not None:
            scores_text = bare_scores_text
        else:
            if "<scores>" in content:
                errors.append("Unclosed <scores> tag")
            else:
                errors.append("Missing <scores> tags")
            return GraderParseResult(
                is_valid=False, errors=errors, raw_response=content
            )
    if not scores_text:
        errors.append("Empty <scores> block")
        return GraderParseResult(
            is_valid=False, errors=errors, raw_response=content
        )

    # Parse comma-separated scores
    raw_scores = [s.strip() for s in scores_text.split(",")]
    for idx, raw_score in enumerate(raw_scores):
        if not raw_score:
            errors.append(f"Score {idx + 1}: Empty value")
            continue

        try:
            score = float(raw_score)
        except ValueError:
            errors.append(f"Score {idx + 1}: Non-numeric value: {raw_score}")
            continue

        if score not in VALID_SCORES:
            errors.append(
                f"Score {idx + 1}: Invalid score {score}, must be 0, 0.5, or 1"
            )
            continue

        assessments.append(
            RubricAssessment(
                rubric=f"Rubric {idx + 1}",
                reasoning=reasoning,
                score=score,
            )
        )

    return GraderParseResult(
        assessments=assessments,
        is_valid=len(assessments) > 0 and len(errors) == 0,
        errors=errors,
        raw_response=content,
    )


def _parse_per_rubric_grader(content: str) -> GraderParseResult:
    """Parse grader output for a single-rubric per-rubric grading call.

    Expected format:
    <think>reasoning</think>
    <score>0</score>  or  <score>1</score>

    Falls back to ``<scores>`` tag (models may pluralize) and then bare
    score after stripping the think block.

    Args:
        content: The raw grader response text for a single rubric.

    Returns:
        GraderParseResult with exactly 1 assessment if valid.
    """
    errors: list[str] = []

    # Extract optional think block
    think_match = THINK_PATTERN.search(content)
    reasoning = think_match.group(1).strip() if think_match else ""

    # Try <score> tag first (singular, preferred)
    score_match = SCORE_PATTERN.search(content)
    if score_match:
        score_text = score_match.group(1).strip()
    else:
        # Fallback: try <scores> tag (models may pluralize)
        scores_match = SCORES_PATTERN.search(content)
        if scores_match:
            score_text = scores_match.group(1).strip()
        else:
            # Fallback: bare score after stripping think block
            bare_text = THINK_PATTERN.sub("", content).strip()
            bare_result = _try_parse_bare_scores(bare_text)
            if bare_result is not None:
                score_text = bare_result
            else:
                if "<score>" in content:
                    errors.append("Unclosed <score> tag")
                else:
                    errors.append("Missing <score> tag")
                return GraderParseResult(
                    is_valid=False, errors=errors, raw_response=content
                )

    if not score_text:
        errors.append("Empty score value")
        return GraderParseResult(
            is_valid=False, errors=errors, raw_response=content
        )

    try:
        score = float(score_text)
    except ValueError:
        errors.append(f"Non-numeric score: {score_text}")
        return GraderParseResult(
            is_valid=False, errors=errors, raw_response=content
        )

    if score not in VALID_SCORES:
        errors.append(
            f"Invalid score {score}, must be 0, 0.5, or 1"
        )
        return GraderParseResult(
            is_valid=False, errors=errors, raw_response=content
        )

    assessment = RubricAssessment(
        rubric="Rubric 1", reasoning=reasoning, score=score
    )
    return GraderParseResult(
        assessments=[assessment],
        is_valid=True,
        errors=[],
        raw_response=content,
    )


def aggregate_per_rubric_results(
    results: list[GraderParseResult],
    rubric_names: list[str] | None = None,
) -> GraderParseResult:
    """Combine multiple single-rubric GraderParseResults into one multi-assessment result.

    This enables ``compute_rubric_sum_with_recovery()`` to work unchanged
    with per-rubric grading outputs.

    Args:
        results: List of GraderParseResults, each with 0 or 1 assessments.
        rubric_names: Optional rubric name labels matching the order of
            ``results``. When ``None``, uses ``"Rubric 1"``, ``"Rubric 2"``, etc.

    Returns:
        GraderParseResult with all valid assessments combined. ``is_valid``
        is ``True`` when at least one assessment was successfully parsed
        and there are no errors.
    """
    all_assessments: list[RubricAssessment] = []
    all_errors: list[str] = []
    raw_parts: list[str] = []

    for i, result in enumerate(results):
        name = rubric_names[i] if rubric_names and i < len(rubric_names) else f"Rubric {i + 1}"
        raw_parts.append(result.raw_response)

        if result.is_valid and result.assessments:
            assessment = result.assessments[0]
            all_assessments.append(
                RubricAssessment(
                    rubric=name,
                    reasoning=assessment.reasoning,
                    score=assessment.score,
                )
            )
        else:
            all_errors.extend(
                f"Rubric {i + 1}: {e}" for e in result.errors
            )

    return GraderParseResult(
        assessments=all_assessments,
        is_valid=len(all_assessments) > 0 and len(all_errors) == 0,
        errors=all_errors,
        raw_response=" ||| ".join(raw_parts),
    )


def parse_grader_response(
    content: str, output_format: OutputFormat = "xml"
) -> GraderParseResult:
    """Parse a grader model response and extract assessment information.

    Supports XML, JSON, v2 (think + comma-separated scores), and per_rubric
    (single-rubric think + score) output formats from the grader model.

    Args:
        content: The raw grader response text.
        output_format: Format of the response (``"xml"``, ``"json"``,
            ``"v2"``, or ``"per_rubric"``).

    Returns:
        GraderParseResult containing extracted assessments and any errors.

    Raises:
        ValueError: If output_format is not a recognized format string.
    """
    if not content:
        return GraderParseResult(
            is_valid=False,
            errors=["Empty response"],
            raw_response="",
        )

    if output_format == "xml":
        return _parse_xml_grader(content)
    elif output_format == "json":
        return _parse_json_grader(content)
    elif output_format == "v2":
        return _parse_v2_grader(content)
    elif output_format == "per_rubric":
        return _parse_per_rubric_grader(content)
    else:
        raise ValueError(
            f"Invalid output_format: {output_format}. "
            f"Use 'xml', 'json', 'v2', or 'per_rubric'"
        )


def get_score_summary(result: GraderParseResult) -> dict:
    """Get a summary of scores from a grader result.

    Args:
        result: A parsed grader result.

    Returns:
        Dictionary with total_score, max_score, normalized_score, and
        per_rubric scores.
    """
    return {
        "total_score": result.total_score,
        "max_score": result.max_score,
        "normalized_score": result.normalized_score,
        "is_valid": result.is_valid,
        "rubric_scores": [
            {"rubric": a.rubric, "score": a.score} for a in result.assessments
        ],
    }
