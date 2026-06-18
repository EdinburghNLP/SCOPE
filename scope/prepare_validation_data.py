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
"""Build the SCOPE validation parquet.

The output combines Search-R1 short-form QA rows with the open-ended benchmark
rows used for SCOPE evaluation. The benchmark mix and output path are fixed on
purpose; the only exposed option is the solver prompt format.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PROMPTS_PATH = REPO_ROOT / "verl" / "prompts.py"
_PROMPTS_SPEC = importlib.util.spec_from_file_location(
    "_scope_verl_prompts", _PROMPTS_PATH
)
if _PROMPTS_SPEC is None or _PROMPTS_SPEC.loader is None:
    raise ImportError(f"Could not load prompt templates from {_PROMPTS_PATH}")
_PROMPTS = importlib.util.module_from_spec(_PROMPTS_SPEC)
_PROMPTS_SPEC.loader.exec_module(_PROMPTS)

DEFAULT_SOLVER_PREFIX_SEARCH_R1 = _PROMPTS.DEFAULT_SOLVER_PREFIX_SEARCH_R1
DEFAULT_SOLVER_PREFIX_SEARCH_R1_QWEN3 = (
    _PROMPTS.DEFAULT_SOLVER_PREFIX_SEARCH_R1_QWEN3
)
DEFAULT_SOLVER_PREFIX_SEARCH_R1_OLMO3 = (
    _PROMPTS.DEFAULT_SOLVER_PREFIX_SEARCH_R1_OLMO3
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

HF_REPO_ID = "PeterJinGo/nq_hotpotqa_train"
DATA_DIR = "./data"
OUTPUT_FILENAMES = {
    "qwen2.5": "validation_qwen25.parquet",
    "qwen3": "validation_qwen3.parquet",
    "olmo3": "validation_olmo3.parquet",
}
MAX_SEARCH_TURNS = 5
SEED = 42
BENCHMARK = "all"
MAX_PER_SOURCE = 0

SEMANTIC_EQUIVALENCE_RUBRIC_TEMPLATE = (
    "The model's answer is semantically equivalent to one of the following"
    " gold answers: {gold_answers}. Award 1 if the answer refers to the same"
    " entity or value as a gold answer (allowing for aliases, abbreviations,"
    " minor formatting/casing/punctuation differences, and equivalent numeric"
    " or date representations). Award 0 if the answer is a different"
    " entity/value, is ambiguous, or contradicts the gold answers."
)


def convert_target_to_rubric(ground_truth: dict) -> dict:
    """Convert an EM-style ground_truth dict to rubric format for LLM judging.

    Takes a ground_truth dict with a ``"target"`` key containing gold answer(s)
    and converts it to rubric format with a single semantic-equivalence rubric.
    If ``target`` is already ``None`` (i.e., already in rubric format), the dict
    is returned unchanged.

    Args:
        ground_truth (dict): Ground truth dict. Expected to have a ``"target"``
            key with a list of gold answers (``list[str]``), a single string,
            or a numpy array. If ``target`` is ``None``, the dict is already in
            rubric format and is returned as-is.

    Returns:
        dict: Ground truth in rubric format with keys ``"target"`` (``None``),
            ``"rubrics"`` (``list[str]``), and ``"priorities"`` (``list[str]``).
    """
    target = ground_truth.get("target")
    if target is None:
        return ground_truth

    # Handle numpy arrays
    if hasattr(target, "tolist"):
        target = target.tolist()

    # Ensure target is a list
    if isinstance(target, str):
        target = [target]

    # Format gold answers as quoted, semicolon-separated string
    gold_answers = "; ".join(f'"{a}"' for a in target)
    rubric_text = SEMANTIC_EQUIVALENCE_RUBRIC_TEMPLATE.format(
        gold_answers=gold_answers
    )

    return {
        "target": None,
        "rubrics": [rubric_text],
        "priorities": ["critical"],
    }


def process_single_row(row, current_split_name, row_index):
    """Process a single row of data for SearchR1-like format.

    Args:
        row (pd.Series): DataFrame row containing the original data.
        current_split_name (str): Name of the current split (train/test).
        row_index (int): Index of the row in the DataFrame.
    Returns:
        pd.Series: Processed row data in the required format.
    """
    import pandas as pd

    question = row.get("question", "")

    # Build prompt structure
    user_content = user_content_prefix.format(question=question.strip())
    prompt = [{"role": "user", "content": user_content}]

    # Extract ground truth from reward_model or fallback to golden_answers
    reward_model_data = row.get("reward_model")
    if (
        isinstance(reward_model_data, dict)
        and "ground_truth" in reward_model_data
    ):
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    # Short-form QA is evaluated by the LLM judge, so always convert exact
    # gold answers into a semantic-equivalence rubric.
    if not isinstance(ground_truth, dict):
        ground_truth = {"target": ground_truth}
    ground_truth = convert_target_to_rubric(ground_truth)
    if isinstance(reward_model_data, dict):
        reward_model_data = {
            **reward_model_data,
            "ground_truth": ground_truth,
        }
    else:
        reward_model_data = {
            "style": "rule",
            "ground_truth": ground_truth,
        }

    # Process data source
    data_source_tagged = "searchR1_" + str(row.get("data_source", ""))

    # Build tools kwargs structure
    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": ground_truth,
                "question": question,
                "data_source": data_source_tagged,
            }
        }
    }

    # Build complete extra_info structure
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": row.get("ability"),
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
        }
    )


# ---------------------------------------------------------------------------
# Benchmark data sources
# ---------------------------------------------------------------------------

HEALTHBENCH_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/"
    "healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
)

SEARCH_R1_PREAMBLE_TEMPLATE = (
    "You must conduct reasoning inside <think> and </think> first every time"
    " you get new information. After reasoning, if you find you lack some"
    " knowledge, you can call a search engine by <search> query </search> and"
    " it will return the top searched results between <information> and"
    " </information>. You can search up to {max_search_turns} times. Provide"
    " your final response inside <answer> and </answer>.\n\n"
)

SEARCH_R1_PREAMBLE_TEMPLATE_QWEN3 = (
    "You must conduct reasoning inside <think> and </think> first every time"
    " you get new information. After reasoning, if you find you lack some"
    " knowledge, you can call the search function and it will return relevant"
    " results. You can search up to {max_search_turns} times. Provide your"
    " final response inside <answer> and </answer>.\n\n"
)

SEARCH_R1_PREAMBLE_TEMPLATE_OLMO3 = (
    "You must conduct reasoning inside <think> and </think> first every time"
    " you get new information. After reasoning, if you find you lack some"
    ' knowledge, you can search by calling <function_calls>search(query="your'
    ' query")</function_calls> and it will return relevant results. You can'
    " search up to {max_search_turns} times. Provide your final response"
    " inside <answer> and </answer>.\n\n"
)

# Default for backward compatibility
SEARCH_R1_PREAMBLE = SEARCH_R1_PREAMBLE_TEMPLATE.format(max_search_turns=5)

# Benchmark processors read these module-level templates.
solver_prefix = DEFAULT_SOLVER_PREFIX_SEARCH_R1
preamble_template = SEARCH_R1_PREAMBLE_TEMPLATE
user_content_prefix = DEFAULT_SOLVER_PREFIX_SEARCH_R1.format(
    max_search_turns=MAX_SEARCH_TURNS, question="{question}"
)


def configure_prompt_format(prompt_format: str) -> None:
    """Set prompt templates for the requested model family."""
    global solver_prefix, preamble_template, user_content_prefix

    prefix_map = {
        "qwen2.5": DEFAULT_SOLVER_PREFIX_SEARCH_R1,
        "qwen3": DEFAULT_SOLVER_PREFIX_SEARCH_R1_QWEN3,
        "olmo3": DEFAULT_SOLVER_PREFIX_SEARCH_R1_OLMO3,
    }
    preamble_map = {
        "qwen2.5": SEARCH_R1_PREAMBLE_TEMPLATE,
        "qwen3": SEARCH_R1_PREAMBLE_TEMPLATE_QWEN3,
        "olmo3": SEARCH_R1_PREAMBLE_TEMPLATE_OLMO3,
    }

    solver_prefix = prefix_map[prompt_format]
    preamble_template = preamble_map[prompt_format]
    user_content_prefix = solver_prefix.format(
        max_search_turns=MAX_SEARCH_TURNS, question="{question}"
    )


def _make_benchmark_row(
    data_source: str,
    prompt: list[dict],
    ability: str,
    ground_truth: dict,
    question: str,
    row_index: int,
) -> dict:
    """Build a single benchmark row in the standard parquet schema.

    Args:
        data_source: Benchmark data source identifier (e.g. ``"eval_healthbench"``).
        prompt: List of message dicts for the model.
        ability: Ability tag for the benchmark.
        ground_truth: Ground truth payload with ``eval_type`` and benchmark data.
        question: Original question text.
        row_index: Row index for tracking.

    Returns:
        dict: Row data matching the 6-column parquet schema.
    """
    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": ground_truth,
                "question": question,
                "data_source": data_source,
            }
        }
    }
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": "test",
        "tools_kwargs": tools_kwargs,
    }
    reward_model = {"ground_truth": ground_truth}
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": ability,
        "reward_model": reward_model,
        "extra_info": extra_info,
        # Use ``None`` (not ``{}``) so pyarrow can serialize a benchmark-
        # only DataFrame to parquet. An empty dict becomes an Arrow struct
        # with no child fields, which fails to write. ``None`` becomes a
        # nullable column and stays compatible if a later row populates it.
        "metadata": None,
    }


def _load_healthbench() -> list[dict]:
    """Download and load HealthBench JSONL data.

    Returns:
        list[dict]: Raw HealthBench examples.
    """
    local_path = os.path.join(
        tempfile.gettempdir(), "healthbench_oss_eval.jsonl"
    )
    if not os.path.exists(local_path):
        logger.info(f"Downloading HealthBench from {HEALTHBENCH_URL}...")
        with urllib.request.urlopen(HEALTHBENCH_URL) as response:
            data = response.read()
        with open(local_path, "wb") as f:
            f.write(data)

    with open(local_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def process_healthbench(max_search_turns: int = 5) -> list[dict]:
    """Process HealthBench examples into SCOPE parquet schema.

    Each example preserves its multi-turn conversation structure. The
    search_r1 preamble is prepended to the first user message so the
    model knows it can use search tools.

    Args:
        max_search_turns: Maximum number of search turns (for preamble text).

    Returns:
        list[dict]: Processed rows in the standard 6-column schema.
    """
    examples = _load_healthbench()
    logger.info(f"Loaded {len(examples)} HealthBench examples")

    rows = []
    for i, ex in enumerate(examples):
        prompt_messages = ex.get("prompt", [])
        rubrics = ex.get("rubrics", [])

        if not prompt_messages:
            continue

        # Prepend search_r1 preamble to the first user message
        preamble = preamble_template.format(
            max_search_turns=max_search_turns
        )
        messages = list(prompt_messages)
        messages[0] = {
            "role": messages[0]["role"],
            "content": preamble + messages[0]["content"],
        }

        # Store original conversation (without preamble) for judge
        ground_truth = {
            "target": None,
            "rubrics": None,
            "eval_type": "healthbench",
            "healthbench_rubrics": [
                {
                    "criterion": r["criterion"],
                    "points": r["points"],
                    "tags": r.get("tags", []),
                }
                for r in rubrics
            ],
            "healthbench_conversation": prompt_messages,
        }

        question = prompt_messages[-1].get("content", "")[:200]
        rows.append(
            _make_benchmark_row(
                data_source="eval_healthbench",
                prompt=messages,
                ability="medical-qa",
                ground_truth=ground_truth,
                question=question,
                row_index=i,
            )
        )

    logger.info(f"Processed {len(rows)} HealthBench rows")
    return rows


def process_researchqa(max_search_turns: int = 5) -> list[dict]:
    """Process ResearchQA test set into SCOPE parquet schema.

    Downloads from HuggingFace ``realliyifei/ResearchQA`` and converts
    each item's query into a single-turn search_r1 prompt.

    Args:
        max_search_turns: Maximum number of search turns for the prompt.

    Returns:
        list[dict]: Processed rows in the standard 6-column schema.
    """
    from huggingface_hub import hf_hub_download

    data_path = os.path.join(tempfile.gettempdir(), "researchqa_test.json")
    if not os.path.exists(data_path):
        logger.info("Downloading ResearchQA test set...")
        file_path = hf_hub_download(
            repo_id="realliyifei/ResearchQA",
            filename="test.json",
            repo_type="dataset",
            revision="87cdd81df0c5ea96de293859233e8e64dac3d168",
        )
        shutil.copy(file_path, data_path)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} ResearchQA items")

    rows = []
    for i, item in enumerate(data):
        query = item.get("query", "")
        rubric_items = item.get("rubric", [])

        prompt_text = solver_prefix.format(
            max_search_turns=max_search_turns, question=query.strip()
        )
        prompt = [{"role": "user", "content": prompt_text}]

        ground_truth = {
            "target": None,
            "rubrics": None,
            "eval_type": "researchqa",
            "researchqa_rubric_items": rubric_items,
        }

        rows.append(
            _make_benchmark_row(
                data_source="eval_researchqa",
                prompt=prompt,
                ability="research-qa",
                ground_truth=ground_truth,
                question=query,
                row_index=i,
            )
        )

    logger.info(f"Processed {len(rows)} ResearchQA rows")
    return rows


def process_drb_race(max_search_turns: int = 5) -> list[dict]:
    """Process DRB-RACE test set into SCOPE parquet schema.

    Downloads from HuggingFace ``rl-research/deep_research_bench_eval``
    and converts each example into a single-turn search_r1 prompt.

    Args:
        max_search_turns: Maximum number of search turns for the prompt.

    Returns:
        list[dict]: Processed rows in the standard 6-column schema.
    """
    import datasets as hf_datasets

    logger.info("Loading DRB-RACE from HuggingFace...")
    ds = hf_datasets.load_dataset(
        "rl-research/deep_research_bench_eval", split="test"
    )
    logger.info(f"Loaded {len(ds)} DRB-RACE examples")

    rows = []
    for i, ex in enumerate(ds):
        prompt_text_raw = ex.get("prompt", ex.get("topic", ""))

        prompt_text = solver_prefix.format(
            max_search_turns=max_search_turns, question=prompt_text_raw.strip()
        )
        prompt = [{"role": "user", "content": prompt_text}]

        ground_truth = {
            "target": None,
            "rubrics": None,
            "eval_type": "drb_race",
            "drb_prompt": prompt_text_raw,
        }

        rows.append(
            _make_benchmark_row(
                data_source="eval_drb_race",
                prompt=prompt,
                ability="deep-research",
                ground_truth=ground_truth,
                question=prompt_text_raw,
                row_index=i,
            )
        )

    logger.info(f"Processed {len(rows)} DRB-RACE rows")
    return rows


def process_sqa_cs_v2(max_search_turns: int = 5) -> list[dict]:
    """Process SQA-CS-V2 dataset into SCOPE parquet schema.

    Downloads from HuggingFace ``allenai/asta-bench`` rubrics_v2 split
    and converts each example into a single-turn search_r1 prompt.

    Args:
        max_search_turns: Maximum number of search turns for the prompt.

    Returns:
        list[dict]: Processed rows in the standard 6-column schema.
    """
    import datasets as hf_datasets

    logger.info("Loading SQA-CS-V2 from HuggingFace (allenai/asta-bench)...")
    ds = hf_datasets.load_dataset(
        "allenai/asta-bench",
        data_files="tasks/sqa/rubrics_v2_recomputed.json",
        split="train",
    )
    logger.info(f"Loaded {len(ds)} SQA-CS-V2 examples")

    rows = []
    for i, ex in enumerate(ds):
        question = ex.get("question", "")

        prompt_text = solver_prefix.format(
            max_search_turns=max_search_turns, question=question.strip()
        )
        prompt = [{"role": "user", "content": prompt_text}]

        ground_truth = {
            "target": None,
            "rubrics": None,
            "eval_type": "sqa_cs_v2",
            "sqa_question": question,
        }

        rows.append(
            _make_benchmark_row(
                data_source="eval_sqa_cs_v2",
                prompt=prompt,
                ability="scientific-qa",
                ground_truth=ground_truth,
                question=question,
                row_index=i,
            )
        )

    logger.info(f"Processed {len(rows)} SQA-CS-V2 rows")
    return rows


def process_researchrubrics(max_search_turns: int = 5) -> list[dict]:
    """Process ResearchRubrics dataset into SCOPE parquet schema.

    Downloads from HuggingFace ``ScaleAI/researchrubrics`` and converts
    each item into a single-turn search_r1 prompt with per-criterion
    weighted rubrics for binary grading.

    Args:
        max_search_turns: Maximum number of search turns for the prompt.

    Returns:
        list[dict]: Processed rows in the standard 6-column schema.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    data_path = os.path.join(
        tempfile.gettempdir(), "researchrubrics_processed_data.jsonl"
    )
    if not os.path.exists(data_path):
        logger.info("Downloading ResearchRubrics dataset...")
        try:
            file_path = hf_hub_download(
                repo_id="ScaleAI/researchrubrics",
                filename="processed_data.jsonl",
                repo_type="dataset",
            )
            shutil.copy(file_path, data_path)
        except EntryNotFoundError:
            logger.error(
                "processed_data.jsonl not found in ScaleAI/researchrubrics"
            )
            return []

    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(data)} ResearchRubrics items")

    rows = []
    for i, item in enumerate(data):
        prompt_text = solver_prefix.format(
            max_search_turns=max_search_turns, question=item["prompt"].strip()
        )
        prompt = [{"role": "user", "content": prompt_text}]

        ground_truth = {
            "target": None,
            "rubrics": None,
            "eval_type": "researchrubrics",
            "researchrubrics_rubrics": [
                {
                    "criterion": r["criterion"],
                    "weight": r["weight"],
                    "axis": r["axis"],
                }
                for r in item["rubrics"]
            ],
            "researchrubrics_prompt": item["prompt"],
        }

        rows.append(
            _make_benchmark_row(
                data_source="eval_researchrubrics",
                prompt=prompt,
                ability="deep-research",
                ground_truth=ground_truth,
                question=item["prompt"],
                row_index=i,
            )
        )

    logger.info(f"Processed {len(rows)} ResearchRubrics rows")
    return rows


RESEARCH_PLAN_GEN_PREFIX = (
    "I will provide you a research scenario. You have to provide me "
    "a concise yet thoughtful research plan with all details needed "
    "to execute it.\n\n"
)


def process_research_plan_gen(max_search_turns: int = 5) -> list[dict]:
    """Process facebook/research-plan-gen dataset into SCOPE parquet schema.

    Downloads all 3 subsets (ML, ArXiv, PubMed) from HuggingFace and converts
    each item into a single-turn search_r1 prompt with per-criterion binary
    rubrics for grading.

    Args:
        max_search_turns: Maximum number of search turns for the prompt.

    Returns:
        list[dict]: Processed rows in the standard 6-column schema.
    """
    import datasets as hf_datasets

    subsets = ["ml", "arxiv", "pubmed"]
    rows: list[dict] = []
    row_index = 0

    for subset in subsets:
        logger.info(
            f"Loading research-plan-gen subset '{subset}' from HuggingFace..."
        )
        ds = hf_datasets.load_dataset(
            "facebook/research-plan-gen", subset, split="test"
        )
        logger.info(f"Loaded {len(ds)} examples from subset '{subset}'")

        for ex in ds:
            goal = ex.get("Goal", "")
            rubric_list = ex.get("Rubric", [])

            question = RESEARCH_PLAN_GEN_PREFIX + goal.strip()
            prompt_text = solver_prefix.format(
                max_search_turns=max_search_turns, question=question
            )
            prompt = [{"role": "user", "content": prompt_text}]

            ground_truth = {
                "target": None,
                "rubrics": None,
                "eval_type": "research_plan_gen",
                "research_plan_gen_rubrics": [
                    {"criterion": rubric_text, "weight": 1.0}
                    for rubric_text in rubric_list
                ],
                "research_plan_gen_goal": goal,
            }

            rows.append(
                _make_benchmark_row(
                    data_source="eval_research_plan_gen",
                    prompt=prompt,
                    ability="research-planning",
                    ground_truth=ground_truth,
                    question=goal,
                    row_index=row_index,
                )
            )
            row_index += 1

    logger.info(
        f"Processed {len(rows)} research-plan-gen rows across"
        f" {len(subsets)} subsets"
    )
    return rows


ARENA_HARD_QUESTION_URL = (
    "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
    "data/arena-hard-v2.0/question.jsonl"
)
ARENA_HARD_BASELINE_URL = (
    "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
    "data/arena-hard-v2.0/model_answer/gemini-2.0-flash-001.jsonl"
)


def _load_arena_hard_questions() -> list[dict]:
    """Download and load Arena-Hard-v2.0 question JSONL data.

    Returns:
        list[dict]: Raw question dicts with ``uid``, ``category``,
            ``subcategory``, and ``prompt`` fields.
    """
    local_path = os.path.join(
        tempfile.gettempdir(), "arena_hard_v2_question.jsonl"
    )
    if not os.path.exists(local_path):
        logger.info(
            f"Downloading Arena-Hard-v2.0 questions from {ARENA_HARD_QUESTION_URL}..."
        )
        with urllib.request.urlopen(ARENA_HARD_QUESTION_URL) as response:
            data = response.read()
        with open(local_path, "wb") as f:
            f.write(data)

    with open(local_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_arena_hard_baseline() -> dict[str, str]:
    """Download and load Arena-Hard-v2.0 baseline answers for gemini-2.0-flash-001.

    Returns:
        dict[str, str]: Mapping of question ``uid`` to baseline answer text.
    """
    local_path = os.path.join(
        tempfile.gettempdir(), "arena_hard_v2_gemini_flash_baseline.jsonl"
    )
    if not os.path.exists(local_path):
        logger.info(
            f"Downloading Arena-Hard-v2.0 baseline answers from "
            f"{ARENA_HARD_BASELINE_URL}..."
        )
        with urllib.request.urlopen(ARENA_HARD_BASELINE_URL) as response:
            data = response.read()
        with open(local_path, "wb") as f:
            f.write(data)

    uid_to_answer: dict[str, str] = {}
    with open(local_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            uid = entry["uid"]
            answer_text = entry["messages"][-1]["content"]["answer"]
            uid_to_answer[uid] = answer_text

    return uid_to_answer


def process_arena_hard_creative_writing(
    max_search_turns: int = 5,
) -> list[dict]:
    """Process Arena-Hard-v2.0 creative writing subset into SCOPE parquet schema.

    Downloads questions and pre-generated baseline answers (gemini-2.0-flash-001)
    from the Arena-Hard-Auto GitHub repository. Filters to the ``creative_writing``
    category (250 items) and stores baseline answers in the ground truth for
    pairwise comparison during evaluation.

    Creative writing uses **single-turn raw prompts** (no solver prefix, no
    search instructions) matching the arena-hard-auto approach.  The model
    responds directly; the pairwise judge compares against the baseline.

    Args:
        max_search_turns: Unused — kept for signature compatibility with
            other benchmark processors.

    Returns:
        list[dict]: Processed rows in the standard 6-column schema.
    """
    questions = _load_arena_hard_questions()
    baseline_answers = _load_arena_hard_baseline()
    logger.info(
        f"Loaded {len(questions)} Arena-Hard questions, "
        f"{len(baseline_answers)} baseline answers"
    )

    # Filter to creative_writing category
    cw_questions = [
        q for q in questions if q.get("category") == "creative_writing"
    ]
    logger.info(f"Filtered to {len(cw_questions)} creative writing questions")

    rows = []
    for i, q in enumerate(cw_questions):
        uid = q["uid"]
        prompt_text_raw = q["prompt"]

        baseline_answer = baseline_answers.get(uid)
        if baseline_answer is None:
            logger.warning(
                f"No baseline answer for uid={uid}, skipping"
            )
            continue

        prompt = [{"role": "user", "content": prompt_text_raw.strip()}]

        ground_truth = {
            "target": None,
            "rubrics": None,
            "eval_type": "arena_hard_cw",
            "arena_hard_uid": uid,
            "arena_hard_prompt": prompt_text_raw,
            "arena_hard_baseline_answer": baseline_answer,
            "arena_hard_baseline_model": "gemini-2.0-flash-001",
        }

        rows.append(
            _make_benchmark_row(
                data_source="eval_arena_hard_cw",
                prompt=prompt,
                ability="creative-writing",
                ground_truth=ground_truth,
                question=prompt_text_raw,
                row_index=i,
            )
        )

    logger.info(f"Processed {len(rows)} Arena-Hard creative writing rows")
    return rows


def process_wildbench(max_search_turns: int = 5) -> list[dict]:
    """Process WildBench v2 dataset into SCOPE parquet schema.

    Downloads the full WildBench v2 set (1024 examples) from HuggingFace
    ``allenai/WildBench`` config ``v2``. Supports both single-turn and
    multi-turn conversations. Multi-turn examples follow the HealthBench
    pattern: preamble is prepended to the first user message, and the full
    original conversation is preserved in ground truth for the judge.

    Args:
        max_search_turns: Maximum number of search turns for the prompt.

    Returns:
        list[dict]: Processed rows in the standard 6-column schema.
    """
    import datasets as hf_datasets

    logger.info("Loading WildBench v2 from HuggingFace (allenai/WildBench)...")
    ds = hf_datasets.load_dataset("allenai/WildBench", "v2", split="test")
    logger.info(f"Loaded {len(ds)} WildBench v2 examples")

    preamble = preamble_template.format(
        max_search_turns=max_search_turns
    )

    rows = []
    for i, ex in enumerate(ds):
        conversation_input = ex["conversation_input"]
        checklist = ex.get("checklist", [])
        primary_tag = ex.get("primary_tag", "")

        ground_truth = {
            "target": None,
            "rubrics": None,
            "eval_type": "wildbench",
            "wildbench_conversation": conversation_input,
            "wildbench_checklist": checklist,
            "wildbench_primary_tag": primary_tag,
        }

        # Build solver prompt: prepend preamble to first user message
        # following HealthBench multi-turn pattern
        messages = list(conversation_input)
        messages[0] = {
            "role": messages[0]["role"],
            "content": preamble + messages[0]["content"],
        }

        question = conversation_input[-1].get("content", "")[:200]

        rows.append(
            _make_benchmark_row(
                data_source="eval_wildbench",
                prompt=messages,
                ability="instruction-following",
                ground_truth=ground_truth,
                question=question,
                row_index=i,
            )
        )

    logger.info(f"Processed {len(rows)} WildBench v2 rows")
    return rows


BENCHMARK_PROCESSORS = {
    "healthbench": process_healthbench,
    "researchqa": process_researchqa,
    "drb_race": process_drb_race,
    "sqa_cs_v2": process_sqa_cs_v2,
    "researchrubrics": process_researchrubrics,
    "research_plan_gen": process_research_plan_gen,
    "arena_hard_cw": process_arena_hard_creative_writing,
    "wildbench": process_wildbench,
}


def _cap_benchmark_rows(
    rows: list[dict],
    max_n: int,
    seed: int,
) -> list[dict]:
    """Deterministically cap a benchmark's rows at ``max_n`` preserving order.

    Samples ``max_n`` unique indices using a fresh ``random.Random(seed)``
    instance, sorts them, and returns ``rows`` indexed by the sorted
    sample. Preserving original order keeps parquet diffs stable across
    runs. If ``max_n <= 0`` or ``len(rows) <= max_n``, ``rows`` is
    returned unchanged.

    Args:
        rows (list[dict]): Processed benchmark rows.
        max_n (int): Maximum rows to keep. ``0`` (or any non-positive
            value) disables capping.
        seed (int): Deterministic RNG seed.

    Returns:
        list[dict]: The capped rows in their original order.
    """
    if max_n <= 0 or len(rows) <= max_n:
        return rows
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(rows)), max_n))
    return [rows[i] for i in chosen]


def _cap_df_per_source(
    df: pd.DataFrame,
    max_n: int,
    seed: int,
) -> pd.DataFrame:
    """Cap each ``data_source`` group in ``df`` at ``max_n`` rows.

    Groups stay in their original DataFrame order; within each group,
    ``max_n`` positional indices are sampled with ``random.Random(seed)``,
    sorted, and used to index the group. When ``max_n <= 0`` or a group
    is already under the cap, that group is returned unchanged.

    Args:
        df (pd.DataFrame): DataFrame with a ``data_source`` column.
        max_n (int): Maximum rows to keep per data_source. ``0``
            disables capping.
        seed (int): Deterministic RNG seed reused for each group.

    Returns:
        pd.DataFrame: Capped DataFrame with rows in their original order.
    """
    import pandas as pd

    if max_n <= 0 or "data_source" not in df.columns:
        return df
    kept_parts: list[pd.DataFrame] = []
    for source, group in df.groupby("data_source", sort=False):
        orig_n = len(group)
        if orig_n > max_n:
            rng = random.Random(seed)
            positions = sorted(rng.sample(range(orig_n), max_n))
            kept = group.iloc[positions]
            logger.info(
                f"Capped {source}: kept {max_n} of {orig_n} rows "
                f"(seed={seed})"
            )
        else:
            kept = group
        kept_parts.append(kept)
    return pd.concat(kept_parts).sort_index()


def _load_benchmark_rows(
    benchmark: str,
    max_search_turns: int,
    max_per_benchmark: int = 0,
    seed: int = 42,
) -> list[dict]:
    """Load and process benchmark rows for the given benchmark(s).

    Args:
        benchmark (str): Benchmark name or ``"all"`` for all benchmarks.
        max_search_turns (int): Maximum search turns for prompt construction.
        max_per_benchmark (int, optional): If ``> 0``, deterministically
            cap each benchmark's contribution at this many rows. Order
            within each benchmark is preserved. Defaults to ``0``
            (no cap — backward-compatible).
        seed (int, optional): RNG seed used when capping. Defaults to ``42``.

    Returns:
        list[dict]: Processed rows in the standard 6-column schema.
    """
    if benchmark == "all":
        benchmarks = list(BENCHMARK_PROCESSORS.keys())
    else:
        benchmarks = [benchmark]

    all_rows: list[dict] = []
    for bm in benchmarks:
        processor = BENCHMARK_PROCESSORS[bm]
        rows = processor(max_search_turns=max_search_turns)
        orig_n = len(rows)
        rows = _cap_benchmark_rows(rows, max_per_benchmark, seed)
        if len(rows) < orig_n:
            logger.info(
                f"Capped {bm}: kept {len(rows)} of {orig_n} rows "
                f"(seed={seed})"
            )
        all_rows.extend(rows)

    return all_rows


def build_validation_data(prompt_format: str) -> str:
    """Build the validation parquet and return its path."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    configure_prompt_format(prompt_format)

    local_save_dir = os.path.expanduser(DATA_DIR)
    os.makedirs(local_save_dir, exist_ok=True)
    output_file_path = os.path.join(
        local_save_dir, OUTPUT_FILENAMES[prompt_format]
    )

    with tempfile.TemporaryDirectory() as tmp_download_dir:
        logger.info("Downloading Search-R1 QA test split from %s", HF_REPO_ID)
        try:
            local_parquet_filepath = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="test.parquet",
                repo_type="dataset",
                local_dir=tmp_download_dir,
                local_dir_use_symlinks=False,
            )
        except EntryNotFoundError as exc:
            raise RuntimeError(
                f"test.parquet not found in HuggingFace dataset {HF_REPO_ID}"
            ) from exc

        df_raw = pd.read_parquet(local_parquet_filepath)
        logger.info("Loaded %d Search-R1 QA rows", len(df_raw))

        def apply_process_row(row):
            return process_single_row(
                row,
                current_split_name="test",
                row_index=row.name,
            )

        df_processed = df_raw.apply(apply_process_row, axis=1)
        assert isinstance(df_processed, pd.DataFrame)

        df_processed = _cap_df_per_source(
            df_processed,
            max_n=MAX_PER_SOURCE,
            seed=SEED,
        )

        benchmark_rows = _load_benchmark_rows(
            BENCHMARK,
            MAX_SEARCH_TURNS,
            max_per_benchmark=MAX_PER_SOURCE,
            seed=SEED,
        )
        df_bench = pd.DataFrame(benchmark_rows)
        df_processed = pd.concat(
            [df_processed, df_bench], ignore_index=True
        )

    df_processed.to_parquet(output_file_path, index=False)
    logger.info(
        "Saved %d validation rows to %s",
        len(df_processed),
        output_file_path,
    )
    logger.info(
        "Rows per data_source:\n%s",
        df_processed["data_source"].value_counts(),
    )
    return output_file_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the model-specific validation parquet from Search-R1 QA "
            "and the fixed SCOPE long-form benchmark suite."
        )
    )
    parser.add_argument(
        "--format",
        default="qwen2.5",
        choices=["qwen2.5", "qwen3", "olmo3"],
        help=(
            "Solver prompt format. qwen2.5 uses <search>/<information>, "
            "qwen3 uses native search-tool wording, and olmo3 uses "
            "function_calls XML."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_validation_data(args.format)
