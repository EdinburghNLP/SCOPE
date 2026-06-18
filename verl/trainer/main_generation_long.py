# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Post-processing utilities for long-form challenger rollout outputs."""

import json
import math
from collections import defaultdict

import pandas as pd

from verl.custom_reward.long_reward_function import (
    compute_long_challenger_format_scores_detailed,
    FormatScoreComponents,
)
from scope.utils.parse_challenger import extract_task


def save_trajectories_jsonl(
    output_path: str,
    component_scores: list[dict],
    all_raw_messages: list[list[str]],
    all_responses: list[list[str]],
) -> str:
    """Save all rollout trajectories with component scores to a JSONL file."""
    jsonl_path = output_path.replace(".parquet", "_trajectories.jsonl")
    with open(jsonl_path, "w") as f:
        for comp in component_scores:
            item_idx = comp["item_idx"]
            sample_idx = comp["sample_idx"]
            record = {
                "item_idx": item_idx,
                "sample_idx": sample_idx,
                "format_score": comp["format_score"],
                "think_reward": comp["think_reward"],
                "tool_reward": comp["tool_reward"],
                "structure_reward": comp["structure_reward"],
                "raw_message": all_raw_messages[item_idx][sample_idx],
                "response": all_responses[item_idx][sample_idx],
            }
            f.write(json.dumps(record) + "\n")
    return jsonl_path


def process_long_form_outputs(
    raw_messages: list[list[str]],
    responses: list[list[str]],
    dataset: pd.DataFrame,
    solver_template: str,
    require_rubrics: bool = True,
    max_search_turns: int = 4,
    challenger_format: str = "tool_call",
    budget_instructions: str = "",
) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Post-process rollout outputs into solver training data."""
    total_prompts = len(raw_messages)
    total_rollouts = sum(len(r) for r in responses)
    valid_format_count = 0
    extract_fail_count = 0
    items_with_rows: set[int] = set()

    output_rows: list[dict] = []
    extracted_tasks_out: dict[int, object] = {}
    selected_contexts_out: dict[int, str] = {}
    all_components: list[dict] = []

    for item_idx, (item_raw_msgs, item_responses) in enumerate(
        zip(raw_messages, responses)
    ):
        row_ei = dataset.iloc[item_idx].get("extra_info") if "extra_info" in dataset.columns else None
        if isinstance(row_ei, str):
            try:
                row_ei = json.loads(row_ei)
            except (ValueError, TypeError):
                row_ei = None
        item_num_search_turns = row_ei.get("num_search_turns", None) if isinstance(row_ei, dict) else None

        detailed_scores = compute_long_challenger_format_scores_detailed(
            item_raw_msgs, item_responses,
            num_search_turns=item_num_search_turns,
            require_rubrics=require_rubrics,
            challenger_format=challenger_format,
        )

        for sample_idx, comp in enumerate(detailed_scores):
            all_components.append({
                "item_idx": item_idx,
                "sample_idx": sample_idx,
                "format_score": comp.format_score,
                "think_reward": comp.think_reward,
                "tool_reward": comp.tool_reward,
                "structure_reward": comp.structure_reward,
            })

            if not (math.isclose(comp.tool_reward, 1.0, abs_tol=1e-9) and
                    math.isclose(comp.structure_reward, 1.0, abs_tol=1e-9)):
                continue
            valid_format_count += 1

            extracted, _errors = extract_task(
                item_responses[sample_idx], require_rubrics=require_rubrics,
                challenger_format=challenger_format,
            )
            if extracted is None or (require_rubrics and not extracted.rubrics):
                extract_fail_count += 1
                continue

            task_prompt = extracted.task_prompt.strip()
            orig_row = dataset.iloc[item_idx].to_dict()
            orig_row["prompt"] = [
                {
                    "role": "user",
                    "content": solver_template.format_map(
                        defaultdict(str, question=task_prompt,
                                    max_search_turns=str(max_search_turns),
                                    budget_instructions=budget_instructions)
                    ),
                }
            ]
            row_extra_info = orig_row.get("extra_info") or {}
            if isinstance(row_extra_info, str):
                try:
                    row_extra_info = json.loads(row_extra_info)
                except (ValueError, TypeError):
                    row_extra_info = {}
            task_type = row_extra_info.get("task_type", "long_form_qa")

            if task_type == "short_form_qa":
                if not extracted.reference_answer:
                    extract_fail_count += 1
                    continue
                orig_row["reward_model"] = {
                    "ground_truth": {
                        "target": extracted.reference_answer,
                        "rubrics": [],
                        "priorities": [],
                    },
                    "style": "rule",
                }
            else:
                orig_row["reward_model"] = {
                    "ground_truth": {
                        "target": None,
                        "rubrics": extracted.rubrics,
                        "priorities": extracted.priorities,
                    },
                    "style": "rule",
                }
            orig_row["metadata"] = {
                "raw_context": item_raw_msgs[sample_idx],
                "num_rubrics": len(extracted.rubrics),
                "task_type": task_type,
                "format_score": comp.format_score,
                "source_item_idx": item_idx,
                "source_rollout_idx": sample_idx,
            }

            out_idx = len(output_rows)
            extracted_tasks_out[out_idx] = extracted
            selected_contexts_out[out_idx] = item_raw_msgs[sample_idx]
            output_rows.append(orig_row)
            items_with_rows.add(item_idx)

    output_dataset = pd.DataFrame(output_rows)

    component_summary = {}
    if all_components:
        n = len(all_components)
        for key in ("think_reward", "tool_reward", "structure_reward"):
            values = [c[key] for c in all_components]
            failures = sum(1 for v in values if not math.isclose(v, 1.0, abs_tol=1e-9))
            component_summary[key] = {
                "mean": sum(values) / n,
                "failures": failures,
                "total": n,
            }

    stats = {
        "total_prompts": total_prompts,
        "total_rollouts": total_rollouts,
        "perfect_format": valid_format_count,
        "extract_failures": extract_fail_count,
        "valid": len(output_rows),
        "total": total_prompts,
        "filtered": total_prompts - len(items_with_rows),
        "component_scores": all_components,
        "component_summary": component_summary,
    }
    return output_dataset, stats, extracted_tasks_out, selected_contexts_out
