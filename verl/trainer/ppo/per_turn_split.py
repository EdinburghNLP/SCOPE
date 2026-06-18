"""Per-turn datum splitting for thinking-hidden multi-turn RL training.

When a model's chat template strips content from previous assistant turns
(e.g., Qwen3 strips <think> blocks), the extension property breaks: the
training datum's token sequence no longer matches what the model saw at
inference. This module splits a trajectory-level batch into per-turn datums
where each datum's prompt is re-rendered with the default template (thinking
stripped from previous turns) and the response preserves thinking for the
current turn.

Architecture:
    1. Rollout produces single-datum trajectories (unchanged)
    2. Rewards and GRPO advantages are computed at trajectory level (unchanged)
    3. This module splits into per-turn datums AFTER advantage computation
    4. old_log_probs and ref_log_probs are computed on per-turn datums
    5. Actor update trains on per-turn datums

Usage:
    Called from ray_trainer.py after compute_advantage() when
    ``config.actor_rollout_ref.rollout.multi_turn.per_turn_datums`` is True.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from verl.protocol import DataProto

logger = logging.getLogger(__name__)


def load_tool_schemas_for_split(tool_config_path: str) -> list[dict[str, Any]]:
    """Load tool schemas from a YAML tool config for template rendering.

    Args:
        tool_config_path: Path to the tool config YAML file.

    Returns:
        List of OpenAI-format tool schema dicts.
    """
    import yaml

    with open(tool_config_path, "r") as f:
        config = yaml.safe_load(f)
    return [tool["tool_schema"] for tool in config.get("tools", []) if "tool_schema" in tool]


def _messages_to_dicts(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert Message objects (Pydantic) to plain dicts for apply_chat_template.

    Args:
        messages: List of Message objects or dicts.

    Returns:
        List of plain dicts with ``exclude_none=True`` semantics.
    """
    result = []
    for m in messages:
        if hasattr(m, "model_dump"):
            result.append(m.model_dump(exclude_none=True))
        elif isinstance(m, dict):
            result.append(m)
        else:
            result.append(dict(m))
    return result


def split_to_per_turn_datums(
    batch: DataProto,
    processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast,
    tool_schemas: list[dict[str, Any]] | None = None,
) -> DataProto:
    """Split trajectory-level batch into per-turn training datums.

    For each trajectory with *T* assistant turns, produces *T* datums.
    Each datum has:

    * **prompt** – re-rendered with the default chat template so that
      ``<think>`` blocks from previous assistant turns are stripped
      (matching what the model saw at inference).
    * **response** – the current turn's generation with thinking preserved
      (the model's actual output, including ``<think>`` blocks).
    * **advantages** – scalar advantage inherited from the trajectory,
      broadcast to all response tokens.
    * **response_mask** – all ones (entire response is model generation;
      tool responses belong to the *next* turn's prompt).

    Args:
        batch: Trajectory-level ``DataProto`` with advantages already
            computed (i.e., after ``compute_advantage``).
        processing_class: Tokenizer used to encode prompts and responses.
        tool_schemas: Optional list of OpenAI-format tool schema dicts.
            Required for ``hermes`` format to inject tool descriptions
            into the system prompt during template rendering.  ``None``
            for ``search_r1`` format (tools described in prompt text).

    Returns:
        New ``DataProto`` containing per-turn datums with the same tensor
        and non-tensor fields as the input batch.

    Raises:
        ValueError: If no per-turn datums could be produced.
    """
    messages_arr = batch.non_tensor_batch["messages"]
    advantages = batch.batch["advantages"]
    response_mask = batch.batch["response_mask"]

    pad_id = processing_class.pad_token_id
    if pad_id is None:
        pad_id = processing_class.eos_token_id

    # Accumulators for per-turn datums
    all_prompt_ids: list[torch.Tensor] = []
    all_response_ids: list[torch.Tensor] = []
    all_prompt_lens: list[int] = []
    all_response_lens: list[int] = []
    all_adv_scalars: list[float] = []
    all_reward_scalars: list[float] = []

    # Mirror every non-tensor key so the output batch stays consistent
    non_tensor_keys = list(batch.non_tensor_batch.keys())
    all_non_tensor: dict[str, list[Any]] = {k: [] for k in non_tensor_keys}

    n_trajectories = len(messages_arr)
    n_datums = 0

    for i in range(n_trajectories):
        msgs_entry = messages_arr[i]
        # msgs_entry is {"messages": [...]} or a list directly
        if isinstance(msgs_entry, dict) and "messages" in msgs_entry:
            msgs_raw = msgs_entry["messages"]
        else:
            msgs_raw = msgs_entry
        msgs = _messages_to_dicts(msgs_raw)

        # --- Extract scalar advantage (GRPO broadcasts same value everywhere) ---
        valid = response_mask[i].bool()
        scalar_adv = advantages[i][valid][0].item() if valid.any() else 0.0

        # --- Extract scalar reward ---
        reward_key = "token_level_rewards" if "token_level_rewards" in batch.batch.keys() else "token_level_scores"
        scalar_reward = batch.batch[reward_key][i].sum().item() if reward_key in batch.batch.keys() else 0.0

        # --- Identify assistant turns ---
        assistant_indices = [j for j, m in enumerate(msgs) if m.get("role") == "assistant"]
        if not assistant_indices:
            logger.warning("Trajectory %d has no assistant turns — skipping", i)
            continue

        for msg_idx in assistant_indices:
            # Prompt: messages before this assistant turn, thinking stripped
            prompt_msgs = msgs[:msg_idx]
            prompt_str = processing_class.apply_chat_template(
                prompt_msgs,
                tools=tool_schemas,
                add_generation_prompt=True,
                tokenize=False,
            )

            # Full conversation through current turn.
            # The current assistant turn is the LAST one in the slice,
            # so the default template preserves its thinking.
            full_msgs = msgs[: msg_idx + 1]
            full_str = processing_class.apply_chat_template(
                full_msgs,
                tools=tool_schemas,
                add_generation_prompt=False,
                tokenize=False,
            )

            # Response = full_str minus the prompt prefix
            response_str = full_str[len(prompt_str) :]
            if not response_str.strip():
                logger.warning(
                    "Empty response for trajectory %d, assistant turn at msg_idx %d — skipping",
                    i,
                    msg_idx,
                )
                continue

            # Tokenize
            prompt_ids = processing_class.encode(prompt_str, add_special_tokens=False)
            response_ids = processing_class.encode(response_str, add_special_tokens=False)

            all_prompt_ids.append(torch.tensor(prompt_ids, dtype=torch.long))
            all_response_ids.append(torch.tensor(response_ids, dtype=torch.long))
            all_prompt_lens.append(len(prompt_ids))
            all_response_lens.append(len(response_ids))
            all_adv_scalars.append(scalar_adv)
            all_reward_scalars.append(scalar_reward)

            # Copy all non-tensor fields from the original trajectory
            for k in non_tensor_keys:
                all_non_tensor[k].append(batch.non_tensor_batch[k][i])

            n_datums += 1

    if n_datums == 0:
        raise ValueError(
            "split_to_per_turn_datums produced 0 datums from "
            f"{n_trajectories} trajectories — check message format"
        )

    logger.info(
        "Per-turn split: %d trajectories → %d datums (%.1f avg turns/traj)",
        n_trajectories,
        n_datums,
        n_datums / max(n_trajectories, 1),
    )

    # --- Pad and assemble tensors ---
    prompt_ids_padded = pad_sequence(
        all_prompt_ids, batch_first=True, padding_value=pad_id, padding_side="left"
    )
    response_ids_padded = pad_sequence(
        all_response_ids, batch_first=True, padding_value=pad_id
    )

    input_ids = torch.cat([prompt_ids_padded, response_ids_padded], dim=-1)

    # Build masks from lengths, NOT value equality (pad_id == eos_token_id is common,
    # so (input_ids != pad_id) would incorrectly mask valid EOS tokens in responses).
    prompt_lens_t = torch.tensor(all_prompt_lens)
    response_lens_t = torch.tensor(all_response_lens)
    max_prompt_len = prompt_ids_padded.shape[1]
    max_response_len = response_ids_padded.shape[1]

    # Left-padded prompts: valid positions are at the RIGHT end
    prompt_mask = torch.arange(max_prompt_len).unsqueeze(0) >= (max_prompt_len - prompt_lens_t.unsqueeze(1))
    # Right-padded responses: valid positions are at the LEFT end
    resp_mask = (torch.arange(max_response_len).unsqueeze(0) < response_lens_t.unsqueeze(1)).long()

    attention_mask = torch.cat([prompt_mask.long(), resp_mask], dim=-1)
    position_ids = (attention_mask.cumsum(dim=-1) - 1) * attention_mask

    # Broadcast scalar advantages → token-level (matching GRPO convention)
    adv_tensor = torch.tensor(all_adv_scalars, dtype=torch.float32)
    token_advantages = adv_tensor.unsqueeze(-1) * resp_mask.float()

    # Place trajectory reward at last valid response token (standard convention)
    reward_tensor = torch.tensor(all_reward_scalars, dtype=torch.float32)
    token_level_scores = torch.zeros(n_datums, max_response_len, dtype=torch.float32)
    for idx in range(n_datums):
        last_valid = all_response_lens[idx] - 1
        if last_valid >= 0:
            token_level_scores[idx, last_valid] = reward_tensor[idx]
    token_level_rewards = token_level_scores.clone()

    new_batch = TensorDict(
        {
            "prompts": prompt_ids_padded,
            "responses": response_ids_padded,
            "response_mask": resp_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "advantages": token_advantages,
            "token_level_scores": token_level_scores,
            "token_level_rewards": token_level_rewards,
            "returns": token_advantages.clone(),
        },
        batch_size=n_datums,
    )

    new_non_tensor = {}
    for k, v in all_non_tensor.items():
        try:
            new_non_tensor[k] = np.array(v)
        except ValueError:
            new_non_tensor[k] = np.array(v, dtype=object)

    return DataProto(
        batch=new_batch,
        non_tensor_batch=new_non_tensor,
        meta_info=batch.meta_info.copy() if batch.meta_info else {},
    )
