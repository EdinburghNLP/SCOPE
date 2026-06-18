#!/usr/bin/env python3
"""Async pipelined orchestrator for iterative task generation.

Used by ``create_task.sh``. It overlaps StageA (Phase 0 + Phase 1) of
iteration N+1 with StageB (Phase 2) of iteration N
using ``ProcessPoolExecutor``.

Pipeline model::

    Sequential:  |-- StageA(1) --|-- StageB(1) --|-- StageA(2) --|-- StageB(2) --|
    Pipelined:   |-- StageA(1) --|-- StageB(1) --|-- StageB(2) --|
                                  |-- StageA(2) --|

StageA = Phase 0 (process_train_challenger.main) + Phase 1 (create_task.run_generation)
StageB = Phase 2 (long_challenger_eval.run_pipeline)

Features:
    - Sentinel-based resume: ``.stageA.done`` / ``.stageB.done`` per iteration
    - Atomic parquet writes via ``.tmp`` + ``os.rename()``
    - Yield estimation to avoid over-generation
    - Merge results capped at ``target_filtered``

Usage::

    python scope/pipeline_orchestrator.py \\
        --model checkpoints/my_model \\
        --run-dir ./data/iter_loop_run \\
        --target-filtered 15000 \\
        --prompt-batch-n 5000 \\
        --challenger-port 8001 \\
        --solver-port 8002
"""

import argparse
import atexit
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker functions (module-level for pickling with 'spawn')
# ---------------------------------------------------------------------------


def _run_stage_a(args_dict: dict) -> None:
    """Run StageA: Phase 0 (prompt creation) + Phase 1 (task generation).

    Imports phase modules inside the function body for clean child-process
    isolation.  Writes a ``.stageA.done`` sentinel on success.

    Args:
        args_dict: Dict with keys ``"phase0"``, ``"phase1"``, ``"iter_dir"``,
            and ``"phase1_output_path"`` containing the arguments for each
            phase and the iteration directory path.

    Raises:
        Exception: Any exception from Phase 0 or Phase 1 is propagated to the
            parent via ``future.result()``.
    """
    # Ensure project root is on sys.path for spawned child process
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    iter_dir = Path(args_dict["iter_dir"])
    sentinel = iter_dir / ".stageA.done"

    # Idempotent: skip if already done
    if sentinel.exists():
        logger.info("StageA already done for %s, skipping", iter_dir)
        return

    failed_sentinel = iter_dir / ".stageA.failed"
    failed_sentinel.unlink(missing_ok=True)

    try:
        # Phase 0: Create input prompts from corpus
        from scope.process_train_challenger import main as phase0_main

        phase0_args = Namespace(**args_dict["phase0"])
        phase0_main(phase0_args)

        # Phase 1: Generate tasks via challenger
        from scope.create_task import run_generation

        phase1_args = Namespace(**args_dict["phase1"])
        final_path = Path(args_dict["phase1_output_path"])
        tmp_path = final_path.with_suffix(".parquet.tmp")

        # Write to temp path first, then atomic rename
        phase1_args.output_path = str(tmp_path)
        run_generation(phase1_args)

        if tmp_path.exists():
            os.rename(str(tmp_path), str(final_path))

        # Write sentinel with manifest
        row_count = len(pd.read_parquet(str(final_path)))
        sentinel.write_text(json.dumps({"rows": row_count, "path": str(final_path)}))
        logger.info("StageA complete for %s: %d rows", iter_dir, row_count)

    except Exception as e:
        failed_sentinel.write_text(str(e))
        raise


def _run_stage_b(args_dict: dict) -> int:
    """Run StageB: Phase 2 (evaluate difficulty + filter).

    Runs in the main process (no spawn needed).  Writes a ``.stageB.done``
    sentinel on success.

    Args:
        args_dict: Dict with keys ``"phase2"``, ``"iter_dir"``, and
            ``"phase2_output_path"`` containing the Phase 2 arguments and the
            iteration directory path.

    Returns:
        int: Number of filtered rows in the output parquet.

    Raises:
        Exception: Any exception from Phase 2 is propagated.
    """
    # Ensure project root is on sys.path for imports
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import datasets as hf_datasets

    from scope.long_challenger_eval import run_pipeline

    iter_dir = Path(args_dict["iter_dir"])
    sentinel = iter_dir / ".stageB.done"

    # Idempotent: skip if already done
    if sentinel.exists():
        manifest = json.loads(sentinel.read_text())
        logger.info("StageB already done for %s: %d rows", iter_dir, manifest["rows"])
        return manifest["rows"]

    failed_sentinel = iter_dir / ".stageB.failed"
    failed_sentinel.unlink(missing_ok=True)

    try:
        phase2_args = Namespace(**args_dict["phase2"])
        final_path = Path(args_dict["phase2_output_path"])
        tmp_path = final_path.with_suffix(".parquet.tmp")

        phase2_args.output_path = str(tmp_path)
        filtered_df = run_pipeline(phase2_args)

        # Save via HuggingFace datasets for native Arrow nested type support
        output_dir = tmp_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        ds = hf_datasets.Dataset.from_pandas(filtered_df)
        ds.to_parquet(str(tmp_path))

        # Atomic rename
        if tmp_path.exists():
            row_count = len(filtered_df)
            os.rename(str(tmp_path), str(final_path))
        else:
            row_count = 0

        sentinel.write_text(json.dumps({"rows": row_count, "path": str(final_path)}))
        logger.info("StageB complete for %s: %d rows", iter_dir, row_count)
        return row_count

    except Exception as e:
        failed_sentinel.write_text(str(e))
        raise


# ---------------------------------------------------------------------------
# PipelineOrchestrator
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    """Manages pipelined iterative task generation.

    Overlaps StageA (Phase 0 + 1) of iteration N+1 with StageB (Phase 2) of
    iteration N using a ``ProcessPoolExecutor`` with a single spawn worker.

    Args:
        config: Namespace with all pipeline configuration parameters.
    """

    def __init__(self, config: Namespace) -> None:
        self.config = config
        self.run_dir = Path(config.run_dir)
        self.accumulated = 0
        self.corpus_offset = config.corpus_offset
        self.current_iter = 0

        # Stratified sampling (opt-in via --stratified-mu / --stratified-sigma)
        self.stratified = (
            getattr(config, "stratified_mu", None) is not None
            and getattr(config, "stratified_sigma", None) is not None
        )
        self.bucket_quotas: dict[tuple[float, float], int] = {}
        self.bucket_counts: dict[tuple[float, float], int] = {}
        if self.stratified:
            from scope.stratified_sampler import compute_bucket_quotas

            self.bucket_quotas = compute_bucket_quotas(
                target_total=config.target_filtered,
                difficulty_min=config.difficulty_min,
                difficulty_max=config.difficulty_max,
                mu=config.stratified_mu,
                sigma=config.stratified_sigma,
            )
            self.bucket_counts = {b: 0 for b in self.bucket_quotas}

    def _iter_dir(self, iter_num: int) -> Path:
        """Return the directory for a given iteration number.

        Args:
            iter_num: int, 1-based iteration number.

        Returns:
            Path to ``run_dir/iter_XXXX``.
        """
        return self.run_dir / f"iter_{iter_num:04d}"

    def _resume(self) -> Optional[int]:
        """Scan existing iterations and resume from last complete state.

        Three-state detection per iteration:
            - ``.stageB.done`` exists: fully complete, add to ``accumulated``
            - ``.stageA.done`` exists (no ``.stageB.done``): ready for StageB
            - Neither: clean up failed sentinels, break

        Returns:
            Optional[int]: Iteration number ready for StageB, or None if no
                incomplete StageA is found.
        """
        stagea_ready = None
        iter_num = 0

        for iter_dir in sorted(self.run_dir.glob("iter_*")):
            if not iter_dir.is_dir():
                continue
            iter_num += 1

            stageb_sentinel = iter_dir / ".stageB.done"
            stagea_sentinel = iter_dir / ".stageA.done"

            if stageb_sentinel.exists():
                # Fully complete iteration
                manifest = json.loads(stageb_sentinel.read_text())
                self.accumulated += manifest["rows"]
                self.corpus_offset += self.config.prompt_batch_n
                self.current_iter = iter_num
                logger.info(
                    "Resume: %s complete with %d rows (total: %d)",
                    iter_dir.name,
                    manifest["rows"],
                    self.accumulated,
                )
            elif stagea_sentinel.exists():
                # StageA done but StageB not yet run
                self.corpus_offset += self.config.prompt_batch_n
                self.current_iter = iter_num
                stagea_ready = iter_num
                logger.info("Resume: %s has StageA done, StageB pending", iter_dir.name)
                break
            else:
                # Incomplete — clean up failed sentinels
                for f in iter_dir.glob(".stageA.failed"):
                    f.unlink(missing_ok=True)
                for f in iter_dir.glob(".stageB.failed"):
                    f.unlink(missing_ok=True)
                break

        if self.accumulated > 0:
            print(
                f"Resumed from {self.current_iter} iterations: "
                f"{self.accumulated}/{self.config.target_filtered} accumulated"
            )

        # Rebuild per-bucket counts from existing filtered parquets
        if self.stratified and self.accumulated > 0:
            from scope.stratified_sampler import (
                format_bucket_status,
                stratified_select,
            )

            parquet_files = sorted(self.run_dir.glob("iter_*/phase2_filtered.parquet"))
            if parquet_files:
                combined = pd.concat(
                    [pd.read_parquet(str(f)) for f in parquet_files],
                    ignore_index=True,
                )
                # Re-run stratified selection to get accurate counts
                _, self.bucket_counts = stratified_select(
                    combined, self.bucket_quotas, seed=self.config.seed,
                )
                self.accumulated = sum(self.bucket_counts.values())
                print("Stratified bucket status after resume:")
                print(format_bucket_status(self.bucket_quotas, self.bucket_counts))

        return stagea_ready

    def _should_continue(self) -> bool:
        """Check whether the generation loop should keep running.

        In stratified mode, checks per-bucket quotas.  Otherwise checks
        total accumulated vs target.

        Returns:
            bool: True if more tasks are needed.
        """
        if self.stratified:
            from scope.stratified_sampler import all_buckets_filled

            return not all_buckets_filled(self.bucket_quotas, self.bucket_counts)
        return self.accumulated < self.config.target_filtered

    def _update_bucket_counts_from_parquet(self, iter_dir: Path) -> None:
        """Read a stage B output parquet and update per-bucket counts.

        Args:
            iter_dir: Path to the iteration directory containing
                ``phase2_filtered.parquet``.
        """
        from scope.stratified_sampler import (
            count_per_bucket,
            format_bucket_status,
        )

        parquet_path = iter_dir / "phase2_filtered.parquet"
        if not parquet_path.exists():
            return

        df = pd.read_parquet(str(parquet_path))
        bucket_edges = sorted(self.bucket_quotas.keys())
        iter_counts = count_per_bucket(df, bucket_edges)

        for bucket in bucket_edges:
            remaining = self.bucket_quotas[bucket] - self.bucket_counts.get(bucket, 0)
            added = min(iter_counts.get(bucket, 0), max(remaining, 0))
            self.bucket_counts[bucket] = self.bucket_counts.get(bucket, 0) + added

        self.accumulated = sum(self.bucket_counts.values())
        print("  Stratified bucket status:")
        print(format_bucket_status(self.bucket_quotas, self.bucket_counts))

    def _should_pregenerate(self) -> bool:
        """Estimate whether another batch is needed based on rolling yield.

        In stratified mode the rolling-average yield estimate is unreliable
        because effective yield drops sharply as buckets fill.  We fall back
        to the simpler ``_should_continue`` check so that the next StageA is
        always pre-launched whenever any bucket still needs tasks.

        Returns:
            bool: True if pre-generation should be launched.
        """
        if self.current_iter == 0:
            return True
        if self.stratified:
            return self._should_continue()
        yield_per_batch = self.accumulated / self.current_iter
        return self.accumulated + yield_per_batch < self.config.target_filtered

    def _build_stage_a_args(self, _iter_num: int, iter_dir: Path) -> dict:
        """Build arguments dict for ``_run_stage_a``.

        Args:
            iter_num: int, 1-based iteration number.
            iter_dir: Path to the iteration directory.

        Returns:
            dict with keys ``"phase0"``, ``"phase1"``, ``"iter_dir"``,
            ``"phase1_output_path"``.
        """
        cfg = self.config
        phase0_output = str(iter_dir / "phase0_prompts.parquet")
        phase1_output = str(iter_dir / "phase1_intermediate.parquet")

        phase0_args = {
            "corpus_dir": cfg.corpus_path,
            "local_dir": str(iter_dir),
            "output_filename": "phase0_prompts.parquet",
            "template_file": cfg.challenger_template,
            "task_descriptions_dir": cfg.task_descriptions_dir,
            "num_samples": cfg.prompt_batch_n,
            "num_search_turns": cfg.input_search_turns,
            "corpus_start_index": self.corpus_offset,
            "seed": cfg.seed,
            "task_types": cfg.task_types,
            "exclude_task_types": cfg.exclude_task_types,
            "max_search_turns": 3,
        }

        phase1_args = {
            "model": cfg.model,
            "base_url": f"http://127.0.0.1:{cfg.challenger_port}",
            "model_name": cfg.model,
            "max_model_len": cfg.max_model_len,
            "input_path": phase0_output,
            "output_path": phase1_output,
            "solver_template_path": cfg.solver_template,
            "tool_config_path": cfg.tool_config,
            "format": cfg.format,
            "n_samples": cfg.n_rollouts,
            "batch_size": cfg.batch_size,
            "separate_rubric_generation": True,
            "verbose": True,
            "partition": cfg.partition,
            "retrieval_url": f"http://127.0.0.1:{cfg.retrieval_port}/retrieve",
            "temperature": 1.0,
            "top_p": 1.0,
            "response_length": 8192,
            "max_assistant_turns": cfg.max_assistant_turns,
            "dynamic_user_turns": cfg.dynamic_user_turns,
            "dynamic_style": cfg.dynamic_style,
            "request_batch_size": 1024,
            "timeout": 600.0,
            "max_retries": 5,
            "retrieval_topk": 3,
            "search_chunk_size": 2048,
            "max_tool_response_length": 500,
            "max_search_turns": 4,
            "prompt_key": "prompt",
        }

        return {
            "phase0": phase0_args,
            "phase1": phase1_args,
            "iter_dir": str(iter_dir),
            "phase1_output_path": phase1_output,
        }

    def _build_stage_b_args(self, _iter_num: int, iter_dir: Path, remaining: int) -> dict:
        """Build arguments dict for ``_run_stage_b``.

        Args:
            iter_num: int, 1-based iteration number.
            iter_dir: Path to the iteration directory.
            remaining: int, number of filtered tasks still needed.

        Returns:
            dict with keys ``"phase2"``, ``"iter_dir"``, ``"phase2_output_path"``.
        """
        cfg = self.config
        phase1_output = str(iter_dir / "phase1_intermediate.parquet")
        phase2_output = str(iter_dir / "phase2_filtered.parquet")

        phase2_args = {
            "input_path": phase1_output,
            "output_path": phase2_output,
            "model_name": cfg.solver_model,
            "grader_model_name": cfg.grader_model,
            "solver_base_url": f"http://127.0.0.1:{cfg.solver_port}",
            "grader_base_url": f"http://127.0.0.1:{cfg.grader_port}",
            "retrieval_url": f"http://127.0.0.1:{cfg.retrieval_port}/retrieve",
            "rubric_template_path": cfg.rubric_template,
            "solver_template_path": cfg.solver_template,
            "grader_template_path": cfg.grader_template,
            "tool_config_path": cfg.tool_config,
            "solver_format": cfg.format,
            "reward_rollout_n": cfg.reward_rollout_n,
            "max_search_turns": 4,
            "solver_retry": True,
            "solver_stop_tokens": cfg.solver_stop_tokens,
            "grader_retry": True,
            "difficulty_min": cfg.difficulty_min,
            "difficulty_max": cfg.difficulty_max,
            "difficulty_fn": "normalized",
            # Use a long timeout for StageB: the solver processes all
            # rollouts in one HTTP call, which can take several hours for
            # large iterations.  max_retries=0 prevents duplicate batch
            # accumulation in the solver queue on timeout retries.
            "timeout": 28800.0,
            "max_retries": 0,
            "batch_size": cfg.batch_size,
            # In stratified mode, process all tasks (no early stop) so the
            # orchestrator can select across the full difficulty range.
            "target_filtered": 0 if self.stratified else remaining,
            "max_tasks_per_prompt": cfg.max_tasks_per_prompt,
            # Defaults for remaining args
            "max_items": 0,
            "max_turns": 5,
            "max_tokens": 2048,
            "max_prompt_length": 131072,
            "grader_format": "xml",
            "grader_min_coverage": 0.5,
            "grader_pad_value": 0.0,
            "rubric_format": "xml",
            "difficulty_target": 0.5,
            # Quality filtering (opt-in via quality_gates string)
            "quality_gates": getattr(cfg, "quality_gates", ""),
            "quality_enabled": getattr(cfg, "quality_enabled", False),
            "quality_entity_template_path": getattr(cfg, "quality_entity_template_path", ""),
            "quality_no_leakage_template_path": getattr(cfg, "quality_no_leakage_template_path", ""),
            "quality_retrieval_template_path": getattr(cfg, "quality_retrieval_template_path", ""),
            "quality_retrieval_multi_template_path": getattr(cfg, "quality_retrieval_multi_template_path", ""),
            "quality_source_relevance_template_path": getattr(cfg, "quality_source_relevance_template_path", ""),
            "quality_required_sum": getattr(cfg, "quality_required_sum", -1),
            "quality_max_tokens": getattr(cfg, "quality_max_tokens", 256),
            "solver_prompt_length_limit": getattr(cfg, "solver_prompt_length_limit", 0),
            # Per-rubric grading
            "grader_per_rubric": getattr(cfg, "grader_per_rubric", False),
            # V19 per-turn rubric generation
            "rubric_mode": getattr(cfg, "rubric_mode", "single"),
            "rubric_initial_template_path": getattr(cfg, "rubric_initial_template_path", ""),
            "rubric_turn_template_path": getattr(cfg, "rubric_turn_template_path", ""),
            "rubric_synthesis_template_path": getattr(cfg, "rubric_synthesis_template_path", ""),
        }

        return {
            "phase2": phase2_args,
            "iter_dir": str(iter_dir),
            "phase2_output_path": phase2_output,
        }

    def _merge_results(self) -> Path:
        """Merge all per-iteration filtered parquets, capped at target.

        In stratified mode, applies per-bucket selection to enforce the
        target difficulty distribution.

        Returns:
            Path: Path to the merged final parquet file.
        """
        parquet_files = sorted(self.run_dir.glob("iter_*/phase2_filtered.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No filtered parquets found in {self.run_dir}/iter_*/"
            )

        dfs = [pd.read_parquet(str(f)) for f in parquet_files]
        combined = pd.concat(dfs, ignore_index=True)

        if self.stratified:
            from scope.stratified_sampler import (
                format_bucket_status,
                stratified_select,
            )

            combined, final_counts = stratified_select(
                combined, self.bucket_quotas, seed=self.config.seed,
            )
            print("Final stratified bucket distribution:")
            print(format_bucket_status(self.bucket_quotas, final_counts))
        else:
            combined = combined.head(self.config.target_filtered)

        final_path = Path(self.config.final_parquet)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(str(final_path), index=False)
        print(
            f"Merged {len(combined)} tasks from {len(parquet_files)} "
            f"iterations to {final_path}"
        )
        return final_path

    def run(self) -> Path:
        """Execute the pipelined iteration loop.

        Returns:
            Path: Path to the merged final parquet file.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        stagea_ready = self._resume()

        if self.accumulated >= self.config.target_filtered:
            print(
                f"Target already reached ({self.accumulated} >= "
                f"{self.config.target_filtered}), merging only."
            )
            return self._merge_results()

        ctx = mp.get_context("spawn")
        # Allow 1 concurrent StageA worker — there is only one challenger
        # server, so running multiple StageA jobs simultaneously causes
        # contention and timeouts.  Pre-submitting 2 ahead still keeps
        # the challenger busy (the second job queues behind the first).
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            # Priming: if no StageA-ready iteration, run first one
            if stagea_ready is None:
                self.current_iter += 1
                iter_dir = self._iter_dir(self.current_iter)
                iter_dir.mkdir(parents=True, exist_ok=True)
                args = self._build_stage_a_args(self.current_iter, iter_dir)
                print(
                    f"\n{'='*60}\n"
                    f"  Iteration {self.current_iter}: Priming StageA\n"
                    f"{'='*60}"
                )
                # Submit to executor so the worker process is warmed up.
                future = executor.submit(_run_stage_a, args)
                future.result()
                self.corpus_offset += self.config.prompt_batch_n

            # Track pre-submitted StageA futures by iteration number.
            # With max_workers=1, futures queue behind each other, keeping
            # the challenger server continuously busy.
            stagea_futures: dict = {}

            def _submit_stagea(iter_num: int) -> None:
                """Submit StageA for iter_num if not already done or queued."""
                d = self._iter_dir(iter_num)
                if (
                    iter_num in stagea_futures
                    or (d / ".stageA.done").exists()
                    or iter_num > self.config.max_iters
                ):
                    return
                d.mkdir(parents=True, exist_ok=True)
                sa_args = self._build_stage_a_args(iter_num, d)
                print(f"  Submitting StageA(iter {iter_num}) to executor...")
                stagea_futures[iter_num] = executor.submit(
                    _run_stage_a, sa_args
                )
                self.corpus_offset += self.config.prompt_batch_n

            while (
                self._should_continue()
                and self.current_iter <= self.config.max_iters
            ):
                iter_dir = self._iter_dir(self.current_iter)
                remaining = self.config.target_filtered - self.accumulated

                print(
                    f"\n{'='*60}\n"
                    f"  Iteration {self.current_iter}: "
                    f"{self.accumulated}/{self.config.target_filtered} accumulated, "
                    f"{remaining} remaining\n"
                    f"{'='*60}"
                )

                # Pre-submit StageA up to 2 iterations ahead. This keeps
                # the challenger server busy without wasting corpus documents
                # on iterations that may never be needed.
                max_ahead = min(
                    self.current_iter + 2,
                    self.config.max_iters,
                )
                for ahead in range(
                    self.current_iter + 1,
                    max_ahead + 1,
                ):
                    _submit_stagea(ahead)

                # Wait for StageA(current) if it was pre-submitted
                if self.current_iter in stagea_futures:
                    stagea_futures.pop(self.current_iter).result()

                # Fallback: if StageA(current) was never pre-generated
                # (e.g. _should_pregenerate returned False in a prior
                # iteration), run it now to avoid missing input data.
                if not (iter_dir / ".stageA.done").exists():
                    print(
                        f"  StageA(iter {self.current_iter}) not "
                        f"pre-generated, running synchronously..."
                    )
                    iter_dir.mkdir(parents=True, exist_ok=True)
                    sa_args = self._build_stage_a_args(
                        self.current_iter, iter_dir
                    )
                    _run_stage_a(sa_args)
                    self.corpus_offset += self.config.prompt_batch_n

                # Run StageB(current) in foreground
                print(f"  Running StageB(iter {self.current_iter}) in foreground...")
                sb_args = self._build_stage_b_args(
                    self.current_iter, iter_dir, remaining
                )
                count = _run_stage_b(sb_args)

                # In stratified mode, update per-bucket counts from this
                # iteration's output; accumulated = sum of bucket counts.
                if self.stratified:
                    self._update_bucket_counts_from_parquet(iter_dir)
                else:
                    self.accumulated += count

                print(
                    f"  StageB result: {count} filtered tasks, "
                    f"{self.accumulated}/{self.config.target_filtered} total"
                )

                self.current_iter += 1

                # Check if target reached after incrementing
                if not self._should_continue():
                    break

            # Cancel queued futures that are no longer needed
            for fut_iter, fut in list(stagea_futures.items()):
                if fut.cancel():
                    print(f"  Cancelled queued StageA(iter {fut_iter})")
                    del stagea_futures[fut_iter]
            # Wait for any already-running future to finish
            for fut_iter, fut in stagea_futures.items():
                try:
                    fut.result()
                except Exception as e:
                    print(
                        f"  Warning: StageA(iter {fut_iter}) failed "
                        f"during cleanup: {e}"
                    )

        return self._merge_results()


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

_executor_ref: Optional[ProcessPoolExecutor] = None


def _cleanup_handler(signum: Optional[int] = None, _frame: object = None) -> None:
    """Handle SIGTERM/SIGINT by shutting down the executor.

    Args:
        signum: Optional[int], signal number (None when called via atexit).
        frame: Stack frame (unused).
    """
    if mp.current_process().name != "MainProcess":
        return
    if _executor_ref is not None:
        logger.info("Shutting down executor due to signal %s", signum)
        _executor_ref.shutdown(wait=False, cancel_futures=True)
    if signum is not None:
        sys.exit(128 + signum)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the pipeline orchestrator.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Pipelined iterative task generation orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Basic usage with defaults:
    python scope/pipeline_orchestrator.py \\
        --model checkpoints/my_model --run-dir ./data/run

    # Custom target and batch sizes:
    python scope/pipeline_orchestrator.py \\
        --model checkpoints/my_model --run-dir ./data/run \\
        --target-filtered 20000 --prompt-batch-n 5000
""",
    )

    # Required
    parser.add_argument("--model", required=True, help="Challenger model path")
    parser.add_argument(
        "--run-dir", required=True, help="Per-iteration artifact directory"
    )

    # Target / batching
    parser.add_argument(
        "--target-filtered",
        type=int,
        default=15000,
        help="Target number of filtered tasks (default: 15000)",
    )
    parser.add_argument(
        "--prompt-batch-n",
        type=int,
        default=5000,
        help="Number of prompts per iteration (default: 5000)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=10,
        help="Maximum number of iterations (default: 10)",
    )

    # Server ports
    parser.add_argument(
        "--challenger-port",
        type=int,
        default=8001,
        help="Challenger server port (default: 8001)",
    )
    parser.add_argument(
        "--solver-port",
        type=int,
        default=8002,
        help="Solver server port (default: 8002)",
    )
    parser.add_argument(
        "--retrieval-port",
        type=int,
        default=8000,
        help="Retrieval server port (default: 8000)",
    )

    # Model config
    parser.add_argument(
        "--solver-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Solver/grader model (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--grader-model",
        default=None,
        help="Grader model name. Defaults to --solver-model if not set.",
    )
    parser.add_argument(
        "--grader-port",
        type=int,
        default=None,
        help="Grader server port. Defaults to --solver-port if not set.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Max context length for token budget check (default: 32768)",
    )

    # Paths
    parser.add_argument(
        "--corpus-path",
        default="./corpus/wiki-18.jsonl",
        help="Path to corpus JSONL",
    )
    parser.add_argument(
        "--challenger-template",
        default="./scope/prompts/challenger_search_r1.txt",
        help="Path to challenger prompt template",
    )
    parser.add_argument(
        "--task-descriptions-dir",
        default="./scope/prompts/tasks",
        help="Directory with task type description .txt files",
    )
    parser.add_argument(
        "--solver-template",
        default="./scope/prompts/solver_search_r1.txt",
        help="Path to solver prompt template",
    )
    parser.add_argument(
        "--grader-template",
        default="./scope/prompts/grader_per_rubric.txt",
        help="Path to grader prompt template",
    )
    parser.add_argument(
        "--rubric-template",
        default="./scope/prompts/rubric.txt",
        help="Path to rubric generation template",
    )
    parser.add_argument(
        "--rubric-mode", default="single", choices=["single", "v19_perturn"],
        help="Rubric generation mode: 'single' or 'v19_perturn'",
    )
    parser.add_argument(
        "--rubric-initial-template", default="",
        help="V19: path to initial (source doc) rubric template",
    )
    parser.add_argument(
        "--rubric-turn-template", default="",
        help="V19: path to per-turn rubric template",
    )
    parser.add_argument(
        "--rubric-synthesis-template", default="",
        help="V19: path to synthesis rubric template",
    )
    parser.add_argument(
        "--tool-config",
        default="./config/search_tool_config.yaml",
        help="Path to tool config YAML",
    )

    # Dynamic user turns
    parser.add_argument(
        "--dynamic-user-turns",
        action="store_true",
        help="Inject chain2-style user messages between search turns",
    )
    parser.add_argument(
        "--dynamic-style",
        type=str,
        default="chain2",
        help="Dynamic message style key (default: chain2)",
    )
    parser.add_argument(
        "--max-assistant-turns",
        type=int,
        default=5,
        help="Max assistant turns per conversation (default: 5)",
    )

    # Generation config
    parser.add_argument(
        "--format",
        default="search_r1",
        help="Tool interaction format (default: search_r1)",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=4,
        help="Number of rollout samples per prompt (default: 4)",
    )
    parser.add_argument(
        "--reward-rollout-n",
        type=int,
        default=4,
        help="Number of solver rollouts per task (default: 4)",
    )
    parser.add_argument(
        "--difficulty-min",
        type=float,
        default=0.2,
        help="Minimum difficulty threshold (default: 0.2)",
    )
    parser.add_argument(
        "--difficulty-max",
        type=float,
        default=0.8,
        help="Maximum difficulty threshold (default: 0.8)",
    )
    parser.add_argument(
        "--solver-stop-tokens",
        default=None,
        help='JSON list of solver stop tokens (e.g., \'["</answer>"]\')',
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="HTTP timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for pipeline processing (default: 5000)",
    )
    parser.add_argument(
        "--max-tasks-per-prompt",
        type=int,
        default=0,
        help="Max tasks per source prompt, 0=no limit (default: 0)",
    )
    parser.add_argument(
        "--final-parquet",
        default="./data/final_filtered.parquet",
        help="Path for merged final output parquet",
    )
    parser.add_argument(
        "--input-search-turns",
        default="4:3:2",
        help="Search turns spec for prompt creation (default: 4:3:2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt creation (default: 42)",
    )
    parser.add_argument(
        "--corpus-offset",
        type=int,
        default=0,
        help="Starting corpus offset (default: 0)",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=None,
        help="Data partition index 1-5 (default: None)",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=None,
        help="Whitelist of task types to include",
    )
    parser.add_argument(
        "--exclude-task-types",
        nargs="+",
        default=None,
        help="Blacklist of task types to exclude",
    )

    # Quality filtering
    parser.add_argument(
        "--quality-gates",
        default="",
        help=(
            "Comma-separated quality gate names, e.g. 'entity,no_leakage'. "
            "Empty string disables quality filtering."
        ),
    )
    parser.add_argument(
        "--quality-entity-template-path",
        default="",
        help="Path to entity identifiability quality grader template",
    )
    parser.add_argument(
        "--quality-no-leakage-template-path",
        default="",
        help="Path to no-answer-leakage quality grader template",
    )
    parser.add_argument(
        "--quality-retrieval-template-path",
        default="",
        help="Path to requires-retrieval quality grader template",
    )
    parser.add_argument(
        "--quality-retrieval-multi-template-path",
        default="",
        help="Path to retrieval quality grader template for multi-search tasks (2+ turns)",
    )
    parser.add_argument(
        "--quality-source-relevance-template-path",
        default="",
        help="Path to source-relevance quality grader template",
    )
    parser.add_argument(
        "--quality-enabled",
        action="store_true",
        default=False,
        help="(Legacy) Enable all quality gates. Prefer --quality-gates.",
    )
    parser.add_argument(
        "--quality-required-sum",
        type=int,
        default=-1,
        help="Required quality score sum to pass. Default: number of active gates.",
    )
    parser.add_argument(
        "--quality-max-tokens",
        type=int,
        default=256,
        help="Max tokens for quality grading LLM calls. Qwen3 needs >=1024.",
    )
    parser.add_argument(
        "--solver-prompt-length-limit",
        type=int,
        default=0,
        help="Drop tasks whose rendered solver prompt exceeds this many tokens. "
             "0 = disabled.",
    )

    # Stratified difficulty sampling (opt-in)
    parser.add_argument(
        "--stratified-mu",
        type=float,
        default=None,
        help=(
            "Mean of truncated Gaussian for stratified difficulty sampling. "
            "When set together with --stratified-sigma, per-bucket quotas "
            "replace flat accept/reject filtering."
        ),
    )
    parser.add_argument(
        "--stratified-sigma",
        type=float,
        default=None,
        help=(
            "Std dev of truncated Gaussian for stratified difficulty sampling. "
            "Must be set together with --stratified-mu."
        ),
    )

    return parser


def main() -> None:
    """CLI entry point for the pipeline orchestrator."""
    # Ensure project root is on sys.path for imports
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    parser = build_parser()
    args = parser.parse_args()

    # Default grader model/port to solver model/port when not specified
    if args.grader_model is None:
        args.grader_model = args.solver_model
    if args.grader_port is None:
        args.grader_port = args.solver_port

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Register signal handlers (main process only)
    if mp.current_process().name == "MainProcess":
        signal.signal(signal.SIGTERM, _cleanup_handler)
        signal.signal(signal.SIGINT, _cleanup_handler)
        atexit.register(_cleanup_handler)

    stratified_info = ""
    if args.stratified_mu is not None and args.stratified_sigma is not None:
        stratified_info = (
            f"  Stratified:      mu={args.stratified_mu}, "
            f"sigma={args.stratified_sigma}\n"
        )

    print(
        f"\n{'='*60}\n"
        f"  Pipeline Orchestrator\n"
        f"{'='*60}\n"
        f"  Model:           {args.model}\n"
        f"  Run dir:         {args.run_dir}\n"
        f"  Target filtered: {args.target_filtered}\n"
        f"  Batch size:      {args.prompt_batch_n}\n"
        f"  Max iterations:  {args.max_iters}\n"
        f"  Challenger port: {args.challenger_port}\n"
        f"  Solver port:     {args.solver_port}\n"
        f"{stratified_info}"
        f"{'='*60}"
    )

    orchestrator = PipelineOrchestrator(args)
    start_time = time.time()
    final_path = orchestrator.run()
    elapsed = time.time() - start_time

    print(
        f"\n{'='*60}\n"
        f"  Pipeline complete\n"
        f"{'='*60}\n"
        f"  Total iterations: {orchestrator.current_iter}\n"
        f"  Total accumulated: {orchestrator.accumulated}\n"
        f"  Final output: {final_path}\n"
        f"  Elapsed time: {elapsed:.1f}s\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
