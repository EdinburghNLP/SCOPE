# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_llds_loss,
    compute_policy_loss,
    compute_self_distillation_loss,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False,
        distill_topk=None, topk_indices=None, module=None,
    ):
        """Forward pass over a micro-batch, computing log probs and optionally top-K logits.

        Args:
            micro_batch: Dict with keys ``responses``, ``input_ids``,
                ``attention_mask``, ``position_ids``.
            temperature: Temperature for logit scaling.
            calculate_entropy: Whether to compute token-level entropy.
            distill_topk: When set (e.g. 100), also compute top-K logits
                and their indices for distillation.
            topk_indices: When provided (teacher path), gather logits at
                these specific indices instead of computing new top-K.
            module: When provided, use this model instead of
                ``self.actor_module``.

        Returns:
            When ``distill_topk`` is None and ``topk_indices`` is None
            (default): tuple ``(entropy, log_probs)`` — unchanged behavior.

            When ``distill_topk`` is not None or ``topk_indices`` is not None:
            dict with keys ``log_probs``, ``entropy`` (or None),
            ``topk_logps``, ``topk_indices``.
        """
        use_topk = distill_topk is not None or topk_indices is not None
        model = module or self.actor_module
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                    # top-K logits for distillation (rmpad branch)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            # Teacher path: gather at student's top-K indices
                            full_topk_indices = torch.zeros(
                                batch_size, seqlen, topk_indices.size(-1),
                                device=topk_indices.device, dtype=topk_indices.dtype,
                            )
                            full_topk_indices[:, -response_length - 1:-1, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        # Chunk logsumexp to avoid OOM on long sequences
                        _lse_chunk = 4096
                        if logits_rmpad.shape[0] > _lse_chunk:
                            logsumexp_rmpad = torch.cat([
                                torch.logsumexp(logits_rmpad[i:i + _lse_chunk], dim=-1, keepdim=True)
                                for i in range(0, logits_rmpad.shape[0], _lse_chunk)
                            ], dim=0)
                        else:
                            logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        del logits_rmpad  # free large tensor immediately
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if topk_indices is None:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if use_topk:
                    topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                    if topk_indices is None:
                        topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                    # top-K logits for distillation (non-rmpad branch)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits.size(-1))
                            topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        # Chunk logsumexp to avoid OOM on long sequences
                        _lse_chunk = 4096
                        if logits.shape[1] > _lse_chunk:
                            logsumexp = torch.cat([
                                torch.logsumexp(logits[:, i:i + _lse_chunk], dim=-1, keepdim=True)
                                for i in range(0, logits.shape[1], _lse_chunk)
                            ], dim=1)
                        else:
                            logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
                        del logits  # free large tensor immediately
                        topk_logps = topk_logits - logsumexp

            # Return dict when top-k is requested, else preserve original tuple interface
            if use_topk:
                return {
                    "log_probs": log_probs,
                    "entropy": entropy,
                    "topk_logps": topk_logps,
                    "topk_indices": topk_indices,
                }
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        # SDPO: include teacher inputs when self-distillation is enabled
        sdpo_config = self.config.get("self_distillation", None)
        sdpo_enabled = sdpo_config is not None and sdpo_config.get("enabled", False)
        if sdpo_enabled and "self_distillation_mask" in data.batch.keys():
            select_keys.extend([
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "self_distillation_mask",
            ])

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        model_updated = False
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                if self.config.use_llds_loss:
                    mb_llds_gated = 0.0
                    mb_llds_preserving = 0.0

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    # When SDPO is active with available feedback, compute top-K
                    # logits in the same forward pass to avoid a redundant second
                    # student forward later.
                    # NOTE: Always run the teacher forward when SDPO is enabled
                    # (do NOT gate on sdpo_mask.any()) — FSDP requires all ranks
                    # to execute the same forward calls; skipping the teacher
                    # forward on some micro-batches causes NCCL deadlocks.
                    sdpo_topk = None
                    if sdpo_enabled and "self_distillation_mask" in model_inputs:
                        sdpo_mask = model_inputs["self_distillation_mask"]
                        sdpo_topk = sdpo_config.get("distillation_topk", 100)

                    if sdpo_topk is not None:
                        student_out = self._forward_micro_batch(
                            model_inputs, temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            distill_topk=sdpo_topk,
                        )
                        entropy, log_prob = student_out["entropy"], student_out["log_probs"]
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            model_inputs, temperature=temperature,
                            calculate_entropy=calculate_entropy,
                        )

                    # When SDPO is enabled, it is the sole training objective.
                    # GRPO policy loss is skipped entirely.
                    if sdpo_topk is not None:
                        # Teacher forward (no grad, same weights, different prompt)
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                        }
                        with torch.no_grad():
                            teacher_out = self._forward_micro_batch(
                                teacher_inputs, temperature,
                                distill_topk=sdpo_topk,
                                topk_indices=student_out["topk_indices"],
                            )

                        sdpo_loss, sdpo_metrics = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_out["log_probs"],
                            response_mask=response_mask,
                            self_distillation_config=sdpo_config,
                            old_log_probs=old_log_prob,
                            student_topk_log_probs=student_out["topk_logps"],
                            teacher_topk_log_probs=teacher_out["topk_logps"],
                            self_distillation_mask=sdpo_mask,
                            loss_agg_mode=loss_agg_mode,
                        )

                        policy_loss = sdpo_loss

                        # Zero out GRPO metrics for logging
                        pg_loss = torch.tensor(0.0, device=log_prob.device)
                        pg_clipfrac = torch.tensor(0.0, device=log_prob.device)
                        ppo_kl = torch.tensor(0.0, device=log_prob.device)
                        pg_clipfrac_lower = torch.tensor(0.0, device=log_prob.device)

                        # Entropy bonus + KL-to-ref as regularizers
                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(
                                loss_mat=entropy, loss_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                            policy_loss = policy_loss - entropy_loss * entropy_coeff

                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob,
                                kl_penalty=self.config.kl_loss_type,
                            )
                            kl_loss = agg_loss(
                                loss_mat=kld, loss_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        micro_batch_metrics["sdpo/loss"] = sdpo_loss.detach().item()
                        micro_batch_metrics.update(sdpo_metrics)

                    else:
                        # GRPO path (used when SDPO is not enabled)
                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                        if self.config.policy_loss.loss_mode == "vanilla":
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )

                        else:
                            policy_loss_fn = get_policy_loss_fn(loss_mode)
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                            )

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            # compute policy loss
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            # compute kl loss
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        if self.config.use_llds_loss:
                            llds_loss, llds_gated_resp, llds_gated_tok, llds_preserving = compute_llds_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                            )
                            policy_loss = policy_loss + llds_loss * self.config.llds_loss_coef
                            if model_updated:  # Skip first mini-batch (always 0)
                                micro_batch_metrics["actor/llds_loss"] = llds_loss.detach().item()
                                micro_batch_metrics["actor/llds_coef"] = self.config.llds_loss_coef
                            mb_llds_gated += llds_gated_resp.detach().item()
                            mb_llds_preserving += llds_preserving.detach().item()

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}

                if self.config.use_llds_loss and model_updated:
                    mini_batch_metrics["actor/llds_gated_responses"] = mb_llds_gated
                    mini_batch_metrics["actor/llds_gated_ratio"] = (
                        mb_llds_gated / mb_llds_preserving if mb_llds_preserving > 0 else 0.0
                    )

                append_to_dict(metrics, mini_batch_metrics)
                model_updated = True
        self.actor_optimizer.zero_grad()
        return metrics
