# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset builder script for long-form challenger training data.

This script generates training data for the challenger pipeline by iterating
over corpus documents and formatting them with the challenger template.

Example:
    python scope/process_train_challenger.py --num_samples 100
    python scope/process_train_challenger.py --corpus_dir ./corpus/wiki-18.jsonl --num_samples 10000
    python scope/process_train_challenger.py \
        --template_file scope/prompts/challenger_search_r1.txt \
        --task_descriptions_dir scope/prompts/tasks \
        --num_search_turns "4:3:2" \
        --num_samples 15000 \
        --output_filename iter1_challenger.parquet

    python scope/process_train_challenger.py \
        --template_file scope/prompts/challenger_search_r1.txt \
        --task_descriptions_dir scope/prompts/tasks \
        --num_search_turns "4:3:2" \
        --num_samples 15000 \
        --output_filename iter1_challenger_wo_creative.parquet

    python scope/process_train_challenger.py \
        --template_file scope/prompts/challenger_search_r1.txt \
        --task_descriptions_dir scope/prompts/tasks \
        --num_search_turns 1 \
        --num_samples 15000 \
        --output_filename iter1_challenger_wo_creative_1turn.parquet
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from collections.abc import Iterator
from typing import Any

# Add project root to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from transformers import AutoTokenizer

from search.index_builder import load_corpus


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Constants
DATA_SOURCE = "challenger"
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
MAX_DOC_TOKENS = 256
PROGRESS_INTERVAL = 1000
ABILITY = "fact-reasoning"
REWARD_MODEL = {"ground_truth": {"target": None}, "style": "rule"}


def parse_search_turns_spec(spec: str) -> list[int]:
    """Parse a search-turns specification string into a list of weights.

    Supports two formats:
    - Fixed mode: a single integer like ``"3"`` → ``[3]`` (all samples get 3 turns).
    - Ratio mode: colon-separated weights like ``"4:3:2"`` → ``[4, 3, 2]``
      where position *i* (1-indexed) corresponds to *i* search turns and the
      value is the relative weight.

    Args:
        spec: Specification string. Either a plain integer or colon-separated
            non-negative integers with at least one non-zero weight.

    Returns:
        list[int]: Parsed weight list.

    Raises:
        ValueError: If the string is empty, contains non-integers, negative
            values, or all-zero weights.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("Search turns spec must not be empty")

    parts = spec.split(":")
    try:
        weights = [int(p) for p in parts]
    except ValueError:
        raise ValueError(
            f"Search turns spec must contain integers separated by ':', got '{spec}'"
        )

    if any(w < 0 for w in weights):
        raise ValueError(
            f"Search turns spec must not contain negative values, got '{spec}'"
        )
    if all(w == 0 for w in weights):
        raise ValueError(
            f"Search turns spec must have at least one non-zero weight, got '{spec}'"
        )

    return weights


def build_search_turns_schedule(
    num_samples: int, weights: list[int], seed: int = 42,
) -> list[int]:
    """Build a per-sample search-turns schedule from weights.

    In **fixed mode** (single weight), every sample gets that many turns.
    In **ratio mode** (multiple weights), samples are allocated via
    largest-remainder apportionment so exact counts sum to ``num_samples``,
    then the resulting list is shuffled deterministically.

    Args:
        num_samples: Total number of samples to generate.
        weights: Weight list from :func:`parse_search_turns_spec`.
            Length 1 = fixed mode; length > 1 = ratio mode where index *i*
            corresponds to *i + 1* search turns.
        seed: Random seed for shuffling (default 42).

    Returns:
        list[int]: List of length ``num_samples`` where each element is the
            number of search turns for that sample.
    """
    if len(weights) == 1:
        return [weights[0]] * num_samples

    # Largest-remainder (Hamilton) apportionment
    total_weight = sum(weights)
    quotas = [num_samples * w / total_weight for w in weights]
    floor_counts = [int(q) for q in quotas]
    remainders = [q - f for q, f in zip(quotas, floor_counts)]

    allocated = sum(floor_counts)
    shortfall = num_samples - allocated

    # Award remaining seats to positions with the largest remainders
    indices_by_remainder = sorted(
        range(len(remainders)), key=lambda i: remainders[i], reverse=True
    )
    for i in indices_by_remainder[:shortfall]:
        floor_counts[i] += 1

    # Build flat list: index i → (i+1) search turns
    schedule: list[int] = []
    for idx, count in enumerate(floor_counts):
        turns = idx + 1
        schedule.extend([turns] * count)

    rng = random.Random(seed)
    rng.shuffle(schedule)
    return schedule


def load_task_descriptions(task_dir: str) -> dict[str, str]:
    """Load task type descriptions from a directory of text files.

    Each ``.txt`` file in the directory is loaded and keyed by its stem
    (filename without extension). For example, ``long_form_qa.txt`` becomes
    key ``"long_form_qa"``.

    Args:
        task_dir: Path to the directory containing task description files.

    Returns:
        dict[str, str]: Mapping from task type name to description text.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no ``.txt`` files are found in the directory.
    """
    dir_path = Path(task_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Task descriptions directory not found: {task_dir}")

    descriptions: dict[str, str] = {}
    for txt_file in sorted(dir_path.glob("*.txt")):
        descriptions[txt_file.stem] = txt_file.read_text().strip()

    if not descriptions:
        raise ValueError(f"No .txt files found in {task_dir}")

    return descriptions


def filter_task_descriptions(
    descriptions: dict[str, str],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, str]:
    """Filter task descriptions by inclusion or exclusion lists.

    When neither ``include`` nor ``exclude`` is provided, returns the original
    dict unchanged.  Exactly one of ``include`` or ``exclude`` may be given.

    Args:
        descriptions: Mapping from task type name to description text, as
            returned by :func:`load_task_descriptions`.
        include: Optional whitelist of task type names to keep. Only these
            types will be present in the returned dict.
        exclude: Optional blacklist of task type names to remove. These
            types will be absent from the returned dict.

    Returns:
        dict[str, str]: Filtered mapping of task type names to descriptions.

    Raises:
        ValueError: If both ``include`` and ``exclude`` are provided, if any
            specified name does not exist in ``descriptions``, or if the
            resulting dict would be empty.
    """
    if include is not None and exclude is not None:
        raise ValueError(
            "Cannot specify both --task_types and --exclude_task_types"
        )

    if include is None and exclude is None:
        return descriptions

    # Validate that all specified names exist
    names = include if include is not None else exclude
    assert names is not None  # narrowing for type checker
    unknown = set(names) - set(descriptions)
    if unknown:
        raise ValueError(
            f"Unknown task type(s): {sorted(unknown)}. "
            f"Available types: {sorted(descriptions)}"
        )

    if include is not None:
        filtered = {k: v for k, v in descriptions.items() if k in include}
    else:
        assert exclude is not None  # narrowing for type checker
        filtered = {k: v for k, v in descriptions.items() if k not in exclude}

    if not filtered:
        raise ValueError(
            "Filtering would remove all task types, resulting in an empty set"
        )

    return filtered


def load_template(template_path: str) -> str:
    """Load challenger template from file.

    Args:
        template_path: Path to the template file containing placeholders
            {source_document} and {max_search_turns}.

    Returns:
        str: Template content as a string.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    path = Path(template_path)
    if not path.is_file():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    return path.read_text()


def preprocess_document(raw_doc: str) -> str:
    """Format and truncate document to MAX_DOC_TOKENS tokens.

    Extracts title from first line and text from second line of the document,
    formats them, then truncates to fit within the token limit.

    Args:
        raw_doc: Raw document string from corpus['contents'] field.
            Expected format: "Title\\nDocument text..."

    Returns:
        str: Formatted and truncated document string.
    """
    lines = raw_doc.split("\n")
    doc_title = lines[0] if lines else ""
    doc_text = lines[1] if len(lines) > 1 else ""

    raw_document = f"(Title: {doc_title})\n{doc_text}\n"
    encoded = TOKENIZER.encode(raw_document)[:MAX_DOC_TOKENS]
    decoded_document = TOKENIZER.decode(encoded)

    return decoded_document


def process_single_row(
    corpus_iter: Iterator[dict[str, Any]],
    template: str,
    max_search_turns: int,
    row_index: int,
    task_descriptions: dict[str, str] | None = None,
    num_search_turns: int | None = None,
    rng: random.Random | None = None,
    short_form_qa_ratio: float = 0.0,
) -> dict[str, Any]:
    """Process a single corpus document into challenger format.

    Supports both v4 (``max_search_turns`` placeholder) and v5
    (``num_search_turns``, ``task_type``, ``task_description`` placeholders)
    templates. When ``task_descriptions`` is provided, a random task type is
    selected per sample.

    Args:
        corpus_iter: Iterator over corpus documents. Each document should have
            a 'contents' key containing the document text.
        template: Template string with ``{source_document}`` and either
            ``{max_search_turns}`` (v4) or ``{num_search_turns}``,
            ``{task_type}``, ``{task_description}`` (v5) placeholders.
        max_search_turns: Value for the v4 ``{max_search_turns}`` placeholder.
        row_index: Index for tracking in extra_info.
        task_descriptions: Optional dict mapping task type names to their
            description text. When provided, enables v5 mode with random
            task type selection.
        num_search_turns: Exact number of search turns for v5 template.
            Replaces ``{num_search_turns}`` placeholder.
        rng: Optional ``random.Random`` instance for deterministic task type
            selection. When ``None``, falls back to the global ``random``
            module.
        short_form_qa_ratio: Fraction of samples assigned to short_form_qa
            task type (0.0-1.0). Remaining samples are distributed uniformly
            among other task types. Default 0.0 means uniform sampling.

    Returns:
        dict: Dictionary with keys:
            - data_source (str): Fixed value "challenger"
            - prompt (list[dict]): List with single user message containing filled template
            - ability (str): Fixed value "fact-reasoning"
            - reward_model (dict): Reward model configuration
            - extra_info (dict): Metadata including tools_kwargs
            - metadata (None): Placeholder for metadata

    Raises:
        StopIteration: If corpus iterator is exhausted.
    """
    # Get next document from corpus
    doc = next(corpus_iter)['contents']
    decoded_document = preprocess_document(doc)

    # Fill template with document and max_search_turns
    # Use simple string replacement instead of .format() to avoid issues
    # with JSON curly braces in the template (e.g., {"name": "search", ...})
    user_content = template.replace("{source_document}", decoded_document)
    user_content = user_content.replace("{max_search_turns}", str(max_search_turns))

    # v5 mode: fill task type, description, and num_search_turns
    if task_descriptions is not None:
        if (
            short_form_qa_ratio > 0
            and "short_form_qa" in task_descriptions
            and (rng or random).random() < short_form_qa_ratio
        ):
            task_type = "short_form_qa"
        else:
            other_types = [k for k in task_descriptions if k != "short_form_qa"]
            task_type = (rng or random).choice(other_types) if other_types else (rng or random).choice(list(task_descriptions.keys()))
        task_description = task_descriptions[task_type]
        user_content = user_content.replace("{task_type}", task_type)
        user_content = user_content.replace("{task_description}", task_description)

        # Set extra_output_format based on task type
        if task_type == "short_form_qa":
            extra_output_format = "<answer>Your reference answer.</answer>"
        else:
            extra_output_format = ""
        user_content = user_content.replace("{extra_output_format}", extra_output_format)

    # Unconditional fallback: remove {extra_output_format} if still present
    # (e.g. when using v8 template without task_descriptions in v4 mode).
    user_content = user_content.replace("{extra_output_format}", "")

    if num_search_turns is not None:
        user_content = user_content.replace(
            "{num_search_turns}", str(num_search_turns)
        )

    prompt = [{"role": "user", "content": user_content}]

    # Build tools_kwargs structure (matching process_train.py pattern)
    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": "",
                "question": "",
                "data_source": DATA_SOURCE
            }
        }
    }

    # Build extra_info structure
    extra_info: dict[str, Any] = {
        "index": row_index,
        "need_tools_kwargs": True,
        "split": "train",
        "tools_kwargs": tools_kwargs,
    }

    # Store source_document and task_type in extra_info for rubric generation
    # at reward time (task_type is no longer extracted from LLM output)
    if task_descriptions is not None:
        extra_info["source_document"] = decoded_document
        extra_info["task_type"] = task_type

    # Store per-sample search turns so reward function can read it
    if num_search_turns is not None:
        extra_info["num_search_turns"] = num_search_turns

    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": ABILITY,
        "reward_model": REWARD_MODEL,
        "extra_info": extra_info,
        "metadata": None,
    }


def main(args: argparse.Namespace) -> None:
    """Main function to generate challenger training data.

    Args:
        args: Parsed command-line arguments containing:
            - local_dir: Output directory for parquet file
            - corpus_dir: Path to corpus JSONL file
            - template_file: Path to challenger template
            - max_search_turns: Value for template placeholder
            - num_samples: Number of samples to generate
            - output_filename: Output parquet filename
            - task_descriptions_dir: Optional directory with task description files (v5)
            - task_types: Optional whitelist of task type names to include
            - exclude_task_types: Optional blacklist of task type names to exclude
            - num_search_turns: Optional exact search turns for v5 template
            - corpus_start_index: Starting index into shuffled corpus (default 0)
    """
    # Setup output directory
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    logger.info(f"Output directory: {local_save_dir}")

    # Load template
    template = load_template(args.template_file)
    logger.info(f"Loaded template from {args.template_file}")

    # Load task descriptions if provided (v5 mode)
    task_descriptions: dict[str, str] | None = None
    task_descriptions_dir = getattr(args, "task_descriptions_dir", None)
    if isinstance(task_descriptions_dir, str):
        task_descriptions = load_task_descriptions(task_descriptions_dir)
        logger.info(
            f"Loaded {len(task_descriptions)} task types from "
            f"{args.task_descriptions_dir}: {sorted(task_descriptions.keys())}"
        )

        # Apply task type filtering if specified
        include = getattr(args, "task_types", None)
        exclude = getattr(args, "exclude_task_types", None)
        if include is not None or exclude is not None:
            task_descriptions = filter_task_descriptions(
                task_descriptions, include=include, exclude=exclude,
            )
            logger.info(
                f"Filtered to {len(task_descriptions)} task types: "
                f"{sorted(task_descriptions.keys())}"
            )
    else:
        include = getattr(args, "task_types", None)
        exclude = getattr(args, "exclude_task_types", None)
        if include is not None or exclude is not None:
            logger.warning(
                "--task_types / --exclude_task_types ignored because "
                "--task_descriptions_dir was not set"
            )

    # Load and shuffle corpus
    seed = getattr(args, "seed", 42)
    logger.info(f"Loading corpus from {args.corpus_dir} (seed={seed})")
    corpus = load_corpus(args.corpus_dir).shuffle(seed=seed)

    # Apply corpus offset for iterative batching
    corpus_start_index = getattr(args, "corpus_start_index", None)
    if not isinstance(corpus_start_index, int):
        corpus_start_index = 0
    if corpus_start_index > 0:
        if corpus_start_index >= len(corpus):
            raise ValueError(
                f"corpus_start_index ({corpus_start_index}) >= corpus size "
                f"({len(corpus)}). No documents available."
            )
        corpus = corpus.select(range(corpus_start_index, len(corpus)))
        logger.info(
            f"Applied corpus offset: start_index={corpus_start_index}, "
            f"remaining={len(corpus)} documents"
        )

    corpus_iter = iter(corpus)

    # Build per-sample search turns schedule
    turns_schedule: list[int] | None = None
    num_search_turns_spec = getattr(args, "num_search_turns", None)
    if isinstance(num_search_turns_spec, str):
        weights = parse_search_turns_spec(num_search_turns_spec)
        turns_schedule = build_search_turns_schedule(args.num_samples, weights, seed=seed)
        logger.info(
            f"Search turns schedule: spec='{num_search_turns_spec}', "
            f"weights={weights}, sample distribution="
            f"{ {t: turns_schedule.count(t) for t in sorted(set(turns_schedule))} }"
        )

    # Generate samples
    rng = random.Random(seed)
    rows = []
    logger.info(f"Generating {args.num_samples} samples...")

    for i in range(args.num_samples):
        try:
            num_search_turns_val = turns_schedule[i] if turns_schedule is not None else None
            row = process_single_row(
                corpus_iter, template, args.max_search_turns, i,
                task_descriptions=task_descriptions,
                num_search_turns=num_search_turns_val,
                rng=rng,
                short_form_qa_ratio=getattr(args, "short_form_qa_ratio", 0.0),
            )
            rows.append(row)

            if (i + 1) % PROGRESS_INTERVAL == 0:
                logger.info(f"Generated {i + 1}/{args.num_samples} samples")

        except StopIteration:
            logger.error(
                f"Corpus exhausted at sample {i}. "
                f"Requested {args.num_samples} samples but corpus has fewer documents."
            )
            raise

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    output_path = os.path.join(local_save_dir, args.output_filename)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")

    # Print example for verification
    logger.info("Example prompt (first sample):")
    print(df.prompt[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate challenger training data by filling template with corpus documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./data",
        help="Local directory to save the processed Parquet file."
    )
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default="./corpus/wiki-18.jsonl",
        help="Path to Wiki corpus JSONL file."
    )
    parser.add_argument(
        "--template_file",
        type=str,
        default="scope/prompts/challenger_search_r1.txt",
        help="Path to challenger template file."
    )
    parser.add_argument(
        "--max_search_turns",
        type=int,
        default=3,
        help="Value for max_search_turns template placeholder."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=15000,
        help="Number of samples to generate."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="iter1_challenger.parquet",
        help="Output parquet filename."
    )
    parser.add_argument(
        "--task_descriptions_dir",
        type=str,
        default=None,
        help=(
            "Directory with task description .txt files (v5 mode). "
            "Each file name (without .txt) becomes a task type. "
            "When set, a random task type is sampled per sample."
        ),
    )
    parser.add_argument(
        "--task_types",
        nargs="+",
        default=None,
        help=(
            "Whitelist of task type names to include (e.g. long_form_qa writing). "
            "Only these types will be used for sampling. "
            "Mutually exclusive with --exclude_task_types."
        ),
    )
    parser.add_argument(
        "--exclude_task_types",
        nargs="+",
        default=None,
        help=(
            "Blacklist of task type names to exclude "
            "All types except these will be used for sampling. "
            "Mutually exclusive with --task_types."
        ),
    )
    parser.add_argument(
        "--num_search_turns",
        type=str,
        default=None,
        help=(
            "Search turns specification for v5 template. "
            "A plain integer (e.g. '3') gives all samples that many turns. "
            "A colon-separated ratio (e.g. '4:3:2') distributes samples: "
            "position i (1-indexed) = i search turns, value = relative weight. "
            "Leave unset for v4 behavior."
        ),
    )
    parser.add_argument(
        "--corpus_start_index",
        type=int,
        default=0,
        help=(
            "Starting index into the shuffled corpus for iterative batching. "
            "Use to skip already-consumed documents across loop iterations. "
            "Default 0 means start from the beginning."
        ),
    )
    parser.add_argument(
        "--short_form_qa_ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of samples assigned to short_form_qa task type "
            "(0.0-1.0). Remaining samples are distributed uniformly among "
            "other task types. Default 0.0 excludes short_form_qa entirely."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for corpus shuffling, search turns schedule, "
            "and task type selection."
        ),
    )
    args = parser.parse_args()

    main(args)
