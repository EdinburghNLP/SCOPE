#!/usr/bin/env python3
"""Phase 1 task generation via SGLang server (OpenAI-compatible API).

Replaces the Phase 1 pipeline in ``long_iter1_gen_data.sh`` with a single-process
script that connects to an SGLang server for batch generation and an external
retrieval server for search. Reads input from parquet, runs adaptive multi-turn
generation with tool use (up to ``max_assistant_turns``), then feeds results
through ``process_long_form_outputs()`` for format scoring and task extraction.

Usage::

    python scope/create_task.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --base-url http://127.0.0.1:8001 \\
        --input-path ./data/challenger_generated.parquet \\
        --output-path ./data/long_gen_intermediate.parquet \\
        --separate-rubric-generation \\
        --n-samples 8 \\
        --batch-size 512 \\
        --verbose
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datasets as hf_datasets
import pandas as pd
import yaml
from openai import OpenAI
from tqdm import trange
from transformers import AutoTokenizer

from verl.tools.utils.search_r1_like_utils import (
    _passages2string,
    call_search_api,
)
from verl.trainer.main_generation_long import process_long_form_outputs

logger = logging.getLogger(__name__)

SEARCH_R1_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
FC_SEARCH_RE = re.compile(
    r'<function_calls>\s*search\s*\(\s*query\s*=\s*["\'](.+?)["\']\s*\)',
    re.DOTALL,
)

# Dynamic user turn message styles.
# Injected between search turns to guide chain-of-entity exploration.
# Format placeholders: {remaining} = searches left, {target} = total budget.
DYNAMIC_STYLES: dict[str, dict[str, str]] = {
    "chain2": {
        "after_search": (
            "You have {remaining} search turn(s) remaining. "
            "Search for a person, place, or organization mentioned in "
            "the results above that you did NOT know about before reading them."
        ),
        "no_result": (
            "Your previous search was not executed (invalid format). "
            "You have {remaining} search turn(s) remaining. "
            "Use the correct search format for your next query."
        ),
        "final": (
            "All {target} search turns are complete. "
            "Now produce your final task output."
        ),
    },
    "chain2_xml": {
        "after_search": (
            "You have {remaining} search turn(s) remaining. "
            "Search for a person, place, or organization mentioned in "
            "the results above that you did NOT know about before reading them."
        ),
        "no_result": (
            "Your previous search was not executed (invalid format). "
            "You have {remaining} search turn(s) remaining. "
            "Use the correct search format for your next query."
        ),
        "final": (
            "All {target} search turns are complete. "
            "Now produce your final task inside <task> and </task> tags."
        ),
    },
}

# Tool schemas for OLMo function_calls format (search + task)
_FC_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for relevant information based on the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": (
                "Submit the final task after completing all search turns. "
                "Call this exactly once as your final action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "description": "The task type (e.g. long_form_qa, writing, planning).",
                    },
                    "task_prompt": {
                        "type": "string",
                        "description": "The task prompt for the solver.",
                    },
                },
                "required": ["task_type", "task_prompt"],
            },
        },
    },
]

# Search-only schemas for function_calls_xml format (task uses XML, not FC)
_FC_SEARCH_ONLY_SCHEMAS: list[dict[str, Any]] = [
    s for s in _FC_TOOL_SCHEMAS if s["function"]["name"] == "search"
]


class FunctionCallsParser:
    """Parser for ``<function_calls>search(query="...")`` format.

    Used when ``format == "function_calls"`` for OLMo native tool-calling.
    Extracts search queries via regex, compatible with the pipeline interface.
    """

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains a ``<function_calls>`` search call.

        Args:
            text: str, raw assistant response text.

        Returns:
            bool: True if text contains a valid function_calls search pattern.
        """
        return bool(FC_SEARCH_RE.search(text))

    def parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract search queries from ``<function_calls>search(query=...)`` patterns.

        Args:
            text: str, raw assistant response text.

        Returns:
            list[dict[str, Any]]: Parsed tool call dicts with ``name`` and
                ``arguments`` keys.
        """
        results: list[dict[str, Any]] = []
        for match in FC_SEARCH_RE.finditer(text):
            query = match.group(1).strip()
            if query:
                results.append(
                    {"name": "search", "arguments": {"query_list": [query]}}
                )
        return results

    def get_search_queries(self, tool_call: dict[str, Any]) -> list[str]:
        """Extract query_list from a parsed search tool call.

        Args:
            tool_call: dict[str, Any], parsed tool call dict.

        Returns:
            list[str]: List of non-empty query strings.
        """
        if not isinstance(tool_call, dict):
            return []
        arguments = tool_call.get("arguments")
        if not isinstance(arguments, dict):
            return []
        query_list = arguments.get("query_list", [])
        if isinstance(query_list, str):
            stripped = query_list.strip()
            return [stripped] if stripped else []
        if isinstance(query_list, list):
            return [q.strip() for q in query_list if isinstance(q, str) and q.strip()]
        return []


class SearchR1Parser:
    """Parser for extracting search queries from ``<search>`` tag format.

    Used when ``format == "search_r1"`` as an alternative to ``ToolCallParser``.
    Parses ``<search>query</search>`` tags and returns tool call dicts
    compatible with the existing pipeline.

    Mirrors the ``_parse_search_r1_calls`` method from
    ``verl.custom_reward.batch_reward_rollout.BatchMultiTurnRollout``.
    """

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains ``<search>`` tags.

        Args:
            text: str, raw assistant response text.

        Returns:
            bool: True if text contains at least one ``<search>...</search>``
                tag pair.
        """
        return bool(SEARCH_R1_RE.search(text))

    def parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract search queries from ``<search>`` tags into tool call dicts.

        Args:
            text: str, raw assistant response text.

        Returns:
            list[dict[str, Any]]: Parsed tool call dicts with ``name`` and
                ``arguments`` keys, compatible with the existing pipeline.
        """
        results: list[dict[str, Any]] = []
        for match in SEARCH_R1_RE.finditer(text):
            query = match.group(1).strip()
            if query:
                results.append(
                    {"name": "search", "arguments": {"query_list": [query]}}
                )
        return results

    def get_search_queries(self, tool_call: dict[str, Any]) -> list[str]:
        """Extract query_list from a parsed search tool call.

        Args:
            tool_call: dict[str, Any], parsed tool call dict from
                ``parse_tool_calls``.

        Returns:
            list[str]: List of non-empty query strings.
        """
        if not isinstance(tool_call, dict):
            return []
        arguments = tool_call.get("arguments")
        if not isinstance(arguments, dict):
            return []
        query_list = arguments.get("query_list", [])
        if isinstance(query_list, str):
            stripped = query_list.strip()
            return [stripped] if stripped else []
        if isinstance(query_list, list):
            return [q.strip() for q in query_list if isinstance(q, str) and q.strip()]
        return []


class ToolCallParser:
    """Parser for Qwen/Hermes ``<tool_call>`` JSON blocks.

    This mirrors the original offline task-generation parser from Dr. Zero and
    exposes the same interface as the Search-R1 and function-calls parsers.
    """

    def __init__(self) -> None:
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )

    def has_tool_call(self, text: str) -> bool:
        """Check whether text contains a complete tool-call block."""
        return (
            self.tool_call_start_token in text
            and self.tool_call_end_token in text
        )

    def parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract JSON tool calls from ``<tool_call>`` blocks."""
        tool_calls: list[dict[str, Any]] = []
        for match in self.tool_call_regex.findall(text):
            try:
                parsed = json.loads(match)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                tool_calls.append(parsed)
        return tool_calls

    def get_content_without_tool_calls(self, text: str) -> str:
        """Remove tool-call blocks from text."""
        return self.tool_call_regex.sub("", text)

    def get_search_queries(self, tool_call: dict[str, Any]) -> list[str]:
        """Safely extract search queries from a parsed search tool call."""
        if not isinstance(tool_call, dict):
            return []

        arguments = tool_call.get("arguments")
        if arguments is None:
            return []

        if isinstance(arguments, list):
            return self._extract_strings_from_list(arguments)

        if isinstance(arguments, dict):
            query_list = arguments.get("query_list")
            if isinstance(query_list, str):
                stripped = query_list.strip()
                return [stripped] if stripped else []
            if isinstance(query_list, list):
                return self._extract_strings_from_list(query_list)

        return []

    def _extract_strings_from_list(self, items: list[Any]) -> list[str]:
        """Extract non-empty query strings from a possibly mixed list."""
        results: list[str] = []
        for item in items:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    results.append(stripped)
            elif isinstance(item, dict):
                query = item.get("query")
                if isinstance(query, str):
                    stripped = query.strip()
                    if stripped:
                        results.append(stripped)
        return results


def load_tool_schemas(config_path: str) -> list[dict]:
    """Load tool schemas from a YAML tool config file.

    Reads the tool config YAML and extracts the ``tool_schema`` from each
    tool entry.

    Args:
        config_path: str, path to the YAML tool config file.

    Returns:
        list[dict]: List of OpenAI-format tool schema dicts.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a tool entry is missing a ``tool_schema`` key.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return [tool["tool_schema"] for tool in config["tools"]]


def _format_search_result(
    results_per_query: list[list[dict[str, Any]]],
    max_tool_response_length: int,
    raw: bool = False,
) -> str:
    """Format retrieval results into a truncated response string.

    Converts raw per-query retrieval results into a human-readable passage
    string and truncates if needed. When ``raw=False`` (default), wraps in a
    JSON ``{"result": ...}`` envelope for ``tool_call`` format. When
    ``raw=True``, returns the plain passage text for ``search_r1`` format
    (the ``<information>`` wrapping is applied by the caller).

    Args:
        results_per_query: list[list[dict[str, Any]]], list of retrieval
            results, one sub-list per query. Each sub-list contains document
            dicts with ``{"document": {"contents": "..."}}`` (the format
            returned by the server when ``return_scores=True``).
        max_tool_response_length: int, maximum character length for the
            combined passage text. Longer text is truncated with an
            ``...(truncated)`` suffix.
        raw: bool, if True return plain text instead of JSON-wrapped.

    Returns:
        str: JSON string with ``{"result": "formatted_passages"}`` when
            ``raw=False``, or plain formatted passages when ``raw=True``.
    """
    formatted_parts: list[str] = []
    for single_query_results in results_per_query:
        formatted_parts.append(_passages2string(single_query_results))
    combined = "\n---\n".join(formatted_parts)
    if max_tool_response_length > 0 and len(combined) > max_tool_response_length:
        combined = combined[:max_tool_response_length] + "...(truncated)"
    if raw:
        return combined
    return json.dumps({"result": combined})


def batch_search(
    search_tasks: dict[int, list[str]],
    retrieval_url: str,
    topk: int,
    max_tool_response_length: int,
    search_chunk_size: int,
    verbose: bool = False,
    raw: bool = False,
) -> dict[int, str]:
    """Execute batched search for multiple conversations in one API call.

    Flattens all per-conversation queries into a single batch, makes chunked
    ``call_search_api()`` calls, then distributes formatted results back
    to each conversation index. Follows the ``_batch_retrieve()`` pattern
    from ``batch_reward_rollout.py``.

    Args:
        search_tasks: dict[int, list[str]], mapping from conversation index
            to the list of search queries for that conversation.
        retrieval_url: str, URL of the retrieval server endpoint.
        topk: int, number of documents to retrieve per query.
        max_tool_response_length: int, max character length for each
            conversation's formatted tool response.
        search_chunk_size: int, maximum queries per API call to avoid
            overwhelming the server.
        verbose: bool, if True print timing information.
        raw: bool, if True return plain text instead of JSON-wrapped results
            (for ``search_r1`` format).

    Returns:
        dict[int, str]: Mapping from conversation index to the formatted
            tool response string. On error, the response contains an
            error message.
    """
    if not search_tasks:
        return {}

    # Flatten all queries with offset tracking
    all_queries: list[str] = []
    offsets: list[tuple[int, int, int]] = []  # (task_idx, offset, count)
    for task_idx, queries in search_tasks.items():
        offsets.append((task_idx, len(all_queries), len(queries)))
        all_queries.extend(queries)

    total_queries = len(all_queries)
    logger.info(
        "batch_search: %d conversations, %d total queries, chunk_size=%d",
        len(search_tasks), total_queries, search_chunk_size,
    )
    t0 = time.time()

    # Chunked batch retrieval
    all_raw_results: list = []
    error_msg: str | None = None
    num_chunks = -(-total_queries // search_chunk_size)
    for chunk_idx, chunk_start in enumerate(
        range(0, total_queries, search_chunk_size)
    ):
        chunk_queries = all_queries[chunk_start:chunk_start + search_chunk_size]
        t_chunk = time.time()
        api_response, chunk_error = call_search_api(
            retrieval_service_url=retrieval_url,
            query_list=chunk_queries,
            topk=topk,
            return_scores=True,  # Required: server unpacks 2 values; _passages2string needs {"document": ...} wrapper
        )
        if api_response and "result" in api_response:
            chunk_results = api_response["result"]
            all_raw_results.extend(chunk_results)
            logger.info(
                "batch_search: chunk %d/%d returned %d results in %.1fs",
                chunk_idx + 1, num_chunks, len(chunk_results),
                time.time() - t_chunk,
            )
        else:
            error_msg = chunk_error
            logger.error(
                "batch_search: chunk %d/%d failed: %s",
                chunk_idx + 1, num_chunks, chunk_error,
            )
            break

    elapsed = time.time() - t0
    if verbose:
        per_query = (elapsed / total_queries * 1000) if total_queries > 0 else 0.0
        print(
            f"    Search: {total_queries} queries batched, "
            f"{elapsed:.1f}s ({per_query:.1f}ms/query)"
        )

    # Distribute results back to conversations
    result_map: dict[int, str] = {}
    if not error_msg and len(all_raw_results) == total_queries:
        for task_idx, offset, count in offsets:
            conv_results = all_raw_results[offset:offset + count]
            result_map[task_idx] = _format_search_result(
                conv_results, max_tool_response_length, raw=raw,
            )
        logger.info(
            "batch_search: complete, %d results to %d conversations in %.1fs",
            total_queries, len(search_tasks), elapsed,
        )
    else:
        # On error, provide error message so conversations can continue
        logger.warning(
            "batch_search: result count mismatch (got %d, expected %d), error: %s",
            len(all_raw_results), total_queries, error_msg,
        )
        error_content = json.dumps(
            {"result": f"Search error: {error_msg or 'unknown'}"}
        )
        for task_idx, _, _ in offsets:
            result_map[task_idx] = error_content

    return result_map


def run_generation(args: argparse.Namespace) -> None:
    """Run the multi-turn generation pipeline via SGLang server.

    Connects to an SGLang server via the OpenAI-compatible API, reads input
    data, runs adaptive multi-turn generation with tool use, then processes
    outputs through ``process_long_form_outputs`` and saves the result as
    parquet.

    Args:
        args: argparse.Namespace, parsed command-line arguments.
    """
    # --- Setup ---
    challenger_format = args.format
    is_search_r1 = challenger_format == "search_r1"
    is_function_calls = challenger_format in ("function_calls", "function_calls_xml")

    if is_search_r1:
        parser = SearchR1Parser()
        tool_schemas = None
    elif challenger_format == "function_calls_xml":
        parser = FunctionCallsParser()
        tool_schemas = _FC_SEARCH_ONLY_SCHEMAS
    elif challenger_format == "function_calls":
        parser = FunctionCallsParser()
        tool_schemas = _FC_TOOL_SCHEMAS
    else:
        tool_schemas = load_tool_schemas(args.tool_config_path)
        parser = ToolCallParser()

    # Stop tokens for challenger generation — stops output at the first
    # closing tag so the model doesn't waste tokens generating past
    # </search> or </task>.  Saves KV cache and compute.
    challenger_stop_tokens: list[str] | None = getattr(
        args, "challenger_stop_tokens", None
    )
    if challenger_stop_tokens is None:
        if is_search_r1:
            challenger_stop_tokens = ["</search>", "</task>"]
        elif is_function_calls:
            challenger_stop_tokens = ["</function_calls>", "</task>"]
        # tool_call format: no stop tokens (uses native tool call parsing)

    solver_template = Path(args.solver_template_path).read_text()

    if args.verbose:
        print("=" * 60)
        print("Phase 1: Task Generation (SGLang server)")
        print("=" * 60)
        print(f"Model: {args.model}")
        print(f"Base URL: {args.base_url}")
        print(f"Format: {challenger_format}")
        print(f"Input: {args.input_path}")
        print(f"Output: {args.output_path}")
        print(f"N samples: {args.n_samples}")
        print(f"Batch size: {args.batch_size}")
        print(f"Temperature: {args.temperature}")
        print(f"Top-p: {args.top_p}")
        print(f"Response length: {args.response_length}")
        print(f"Max assistant turns: {args.max_assistant_turns}")
        print(f"Max tool response length: {args.max_tool_response_length}")
        print(f"Max model len: {args.max_model_len}")
        print(f"Separate rubric generation: {args.separate_rubric_generation}")
        print("=" * 60)

    # --- Load tokenizer and OpenAI client ---
    if args.verbose:
        print("\nLoading tokenizer...")
    model_path = str(Path(args.model).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model_name = args.model_name or args.model
    client = OpenAI(
        base_url=f"{args.base_url}/v1",
        api_key="EMPTY",
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    max_model_len = args.max_model_len

    if args.verbose:
        print(f"Connected to SGLang server at {args.base_url}")
        print(f"Model name for API: {model_name}")

    # --- Load data ---
    if args.verbose:
        print(f"\nLoading data from {args.input_path}...")
    dataset = pd.read_parquet(args.input_path)

    if args.partition is not None:
        partition = int(args.partition)
        partition_length = len(dataset) // 5
        assert 0 < partition <= 5, f"Partition must be 1-5, got {partition}"
        start = partition_length * (partition - 1)
        end = partition_length * partition
        dataset = dataset.iloc[start:end]
        if args.verbose:
            print(f"Using partition {partition}/5, rows {start} to {end}")

    chat_list = dataset[args.prompt_key].tolist()
    chat_list = [chat.tolist() if hasattr(chat, "tolist") else chat for chat in chat_list]

    # Load per-row num_search_turns from extra_info for dynamic turn tracking
    num_search_turns_list: list[int] = []
    if args.dynamic_user_turns and "extra_info" in dataset.columns:
        for ei in dataset["extra_info"].tolist():
            if isinstance(ei, str):
                ei = json.loads(ei)
            nst = ei.get("num_search_turns") if isinstance(ei, dict) else None
            num_search_turns_list.append(
                int(nst) if nst is not None else args.max_search_turns
            )
    else:
        num_search_turns_list = [args.max_search_turns] * len(dataset)
        if args.dynamic_user_turns and "extra_info" not in dataset.columns:
            logger.warning(
                "--dynamic-user-turns enabled but dataset has no 'extra_info' "
                "column; falling back to --max-search-turns=%d for all rows",
                args.max_search_turns,
            )

    dynamic_msgs: dict[str, str] | None = (
        DYNAMIC_STYLES.get(args.dynamic_style) if args.dynamic_user_turns else None
    )

    total_samples = len(dataset)
    num_batches = -(-total_samples // args.batch_size)

    if args.verbose:
        print(f"Total prompts: {total_samples}, Batches: {num_batches}")
        if dynamic_msgs:
            print(f"Dynamic user turns: {args.dynamic_style}")

    # --- Multi-turn generation ---
    all_raw_messages: list[list[str]] = []
    all_responses: list[list[str]] = []

    for batch_idx in trange(num_batches, desc="Batches"):
        batch_start = batch_idx * args.batch_size
        batch_end = min((batch_idx + 1) * args.batch_size, total_samples)
        batch_chats = chat_list[batch_start:batch_end]

        # Create n_samples copies of each prompt
        conversations: list[dict[str, Any]] = []
        for item_idx, messages in enumerate(batch_chats):
            abs_row = batch_start + item_idx
            nst = num_search_turns_list[abs_row]
            for rollout_idx in range(args.n_samples):
                conversations.append({
                    "item_idx": item_idx,
                    "rollout_idx": rollout_idx,
                    "messages": list(messages),
                    "complete": False,
                    "assistant_contents": [],
                    "num_search_turns": nst,
                    "searches_done": 0,
                    "turn_number": 0,
                })

        # Multi-turn loop
        for turn in range(args.max_assistant_turns):
            active = [c for c in conversations if not c["complete"]]
            if not active:
                break

            if args.verbose:
                print(
                    f"\n  [Batch {batch_idx}] Turn {turn + 1}/{args.max_assistant_turns}: "
                    f"{len(active)} active conversations"
                )

            # Build text prompts and check token budget
            prompts_to_generate: list[str] = []
            active_after_budget: list[dict[str, Any]] = []

            if args.verbose:
                t_tok_start = time.time()

            # Step 1: Apply chat template (sequential, Jinja2 limitation)
            text_prompts_with_conv: list[tuple[dict, str]] = []
            for conv in active:
                text_prompt = tokenizer.apply_chat_template(
                    conv["messages"],
                    tools=tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                text_prompts_with_conv.append((conv, text_prompt))

            # Step 2: Batch encode all prompts at once (Rust tokenizer, parallel)
            budget_exceeded_count = 0
            if text_prompts_with_conv:
                all_text_prompts = [tp for _, tp in text_prompts_with_conv]
                encoded_batch = tokenizer(
                    all_text_prompts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_length=True,
                )

                # Step 3: Budget check with pre-computed lengths
                for (conv, text_prompt), length in zip(
                    text_prompts_with_conv, encoded_batch["length"]
                ):
                    if length + args.response_length > max_model_len:
                        budget_exceeded_count += 1
                        conv["complete"] = True
                        if args.verbose:
                            print(
                                f"    Token budget exceeded for conv "
                                f"({conv['item_idx']}, {conv['rollout_idx']}): "
                                f"{length} + {args.response_length} > {max_model_len}"
                            )
                        continue
                    prompts_to_generate.append(text_prompt)
                    active_after_budget.append(conv)

            if args.verbose:
                t_tok_end = time.time()
                print(f"    Tokenization: {t_tok_end - t_tok_start:.1f}s")

            if not prompts_to_generate:
                break

            # Batched OpenAI completions via SGLang server
            if args.verbose:
                t_gen_start = time.time()

            all_texts: list[str] = []
            for chunk_start in range(
                0, len(prompts_to_generate), args.request_batch_size
            ):
                chunk = prompts_to_generate[
                    chunk_start : chunk_start + args.request_batch_size
                ]
                create_kwargs: dict[str, Any] = {
                    "model": model_name,
                    "prompt": chunk,
                    "max_tokens": args.response_length,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
                if challenger_stop_tokens:
                    create_kwargs["stop"] = challenger_stop_tokens
                response = client.completions.create(**create_kwargs)
                choices = sorted(response.choices, key=lambda c: c.index)
                for choice in choices:
                    text = choice.text
                    # Re-append the stop token that was stripped by the API
                    if (
                        challenger_stop_tokens
                        and choice.finish_reason == "stop"
                    ):
                        for st in challenger_stop_tokens:
                            if not text.endswith(st) and st.startswith("</"):
                                # Check if the opening tag is present
                                open_tag = "<" + st[2:]  # </x> -> <x>
                                open_tag = open_tag[:-1]  # <x> -> <x
                                # More robust: just check opening tag name
                                tag_name = st[2:-1]  # </search> -> search
                                if f"<{tag_name}>" in text or f"<{tag_name}" in text:
                                    text = text + st
                                    break
                    all_texts.append(text)

            if args.verbose:
                t_gen_end = time.time()
                print(f"    Generation: {t_gen_end - t_gen_start:.1f}s")

            # Phase A: Parse outputs and collect search queries
            search_tasks: dict[int, list[str]] = {}
            for i, (conv, assistant_content) in enumerate(
                zip(active_after_budget, all_texts)
            ):
                # Strict fixed turns: skip search parsing on the task turn
                # (turn_number is 0-indexed before increment, so >= N means
                # we've exhausted all N search turns).
                on_task_turn = (
                    dynamic_msgs
                    and conv["turn_number"] >= conv["num_search_turns"]
                )
                found_search = False
                if not on_task_turn and parser.has_tool_call(assistant_content):
                    tool_calls = parser.parse_tool_calls(assistant_content)
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        if tc.get("name") == "search":
                            query_list = parser.get_search_queries(tc)
                            if query_list:
                                # Strip trailing text after last closing tag
                                # to remove fabricated tool responses
                                if is_search_r1:
                                    last_close = assistant_content.rfind("</search>")
                                    if last_close != -1:
                                        assistant_content = assistant_content[
                                            : last_close + len("</search>")
                                        ]
                                elif is_function_calls:
                                    last_close = assistant_content.rfind("</function_calls>")
                                    if last_close != -1:
                                        assistant_content = assistant_content[
                                            : last_close + len("</function_calls>")
                                        ]
                                search_tasks[i] = query_list
                                found_search = True
                                break  # One search per turn

                if found_search:
                    conv["searches_done"] += 1

                # Always append assistant response first (before any
                # user message) to maintain correct chronology.
                conv["messages"].append({
                    "role": "assistant",
                    "content": assistant_content,
                })
                conv["assistant_contents"].append(assistant_content)
                conv["turn_number"] += 1

                if not found_search:
                    # Check for early task completion (<task> tag present)
                    has_task = "<task>" in assistant_content
                    if has_task:
                        conv["complete"] = True
                    elif (
                        dynamic_msgs
                        and conv["turn_number"] <= conv["num_search_turns"]
                    ):
                        # Strict fixed turns: format mistake during search
                        # phase doesn't end conversation.  Send dynamic
                        # message so model can try again on the next turn.
                        remaining = (
                            conv["num_search_turns"] - conv["turn_number"]
                        )
                        if remaining > 0:
                            msg = dynamic_msgs["no_result"].format(
                                remaining=remaining,
                                target=conv["num_search_turns"],
                            )
                        else:
                            # Last search turn used — send final message
                            msg = dynamic_msgs["final"].format(
                                remaining=0,
                                target=conv["num_search_turns"],
                            )
                        conv["messages"].append({
                            "role": "user",
                            "content": msg,
                        })
                    else:
                        # Legacy mode (no dynamic) or past search phase
                        conv["complete"] = True

            # Per-turn verbose stats (solver-style logging)
            if args.verbose:
                n_with_search = len(search_tasks)
                n_completed_no_tool = sum(
                    1 for c in active_after_budget if c["complete"]
                ) - budget_exceeded_count
                total_search_queries = sum(
                    len(qs) for qs in search_tasks.values()
                )
                print(
                    f"    [Turn {turn + 1}] Tool calls: {n_with_search} search, "
                    f"{n_completed_no_tool} completed (no search), "
                    f"{budget_exceeded_count} budget exceeded"
                )
                if total_search_queries > 0:
                    print(
                        f"    [Turn {turn + 1}] Search queries: "
                        f"{total_search_queries} from {n_with_search} conversations"
                    )

            # Phase B: Batched retrieval
            if search_tasks:
                result_map = batch_search(
                    search_tasks=search_tasks,
                    retrieval_url=args.retrieval_url,
                    topk=args.retrieval_topk,
                    max_tool_response_length=args.max_tool_response_length,
                    search_chunk_size=args.search_chunk_size,
                    verbose=args.verbose,
                    raw=is_search_r1 or is_function_calls,
                )
                for task_idx, result_text in result_map.items():
                    if is_search_r1 or is_function_calls:
                        info_content = f"<information>{result_text}</information>"
                        # Merge dynamic user turn message with info.
                        # Use turn_number (not searches_done) so the count
                        # stays consistent with the task-turn gate after
                        # any format mistakes that advanced turn_number
                        # without incrementing searches_done.
                        if dynamic_msgs:
                            conv = active_after_budget[task_idx]
                            remaining = (
                                conv["num_search_turns"] - conv["turn_number"]
                            )
                            if remaining > 0:
                                dynamic_text = dynamic_msgs[
                                    "after_search"
                                ].format(
                                    remaining=remaining,
                                    target=conv["num_search_turns"],
                                )
                            else:
                                dynamic_text = dynamic_msgs["final"].format(
                                    remaining=0,
                                    target=conv["num_search_turns"],
                                )
                            info_content = f"{info_content}\n\n{dynamic_text}"
                        active_after_budget[task_idx]["messages"].append({
                            "role": "user",
                            "content": info_content,
                        })
                    else:
                        active_after_budget[task_idx]["messages"].append({
                            "role": "tool",
                            "content": result_text,
                        })

        # Per-batch summary (solver-style logging)
        if args.verbose:
            n_complete = sum(1 for c in conversations if c["complete"])
            n_with_content = sum(
                1 for c in conversations if c["assistant_contents"]
            )
            print(
                f"\n  [Batch {batch_idx}] Summary: {len(conversations)} conversations, "
                f"{n_with_content} generated, {n_complete} complete"
            )

        # Group results by item_idx
        if args.verbose:
            t_agg_start = time.time()

        items_in_batch = len(batch_chats)
        batch_raw_messages: list[list[str]] = [[] for _ in range(items_in_batch)]
        batch_responses: list[list[str]] = [[] for _ in range(items_in_batch)]

        # Batch apply_chat_template for all conversations
        all_conv_messages = [conv["messages"] for conv in conversations]
        all_raw_msgs = [
            tokenizer.apply_chat_template(m, tools=tool_schemas, tokenize=False)
            for m in all_conv_messages
        ]

        for conv, raw_msg in zip(conversations, all_raw_msgs):
            item_idx = conv["item_idx"]
            response = "".join(conv["assistant_contents"])
            batch_raw_messages[item_idx].append(raw_msg)
            batch_responses[item_idx].append(response)

        if args.verbose:
            t_agg_end = time.time()
            print(f"  Post-batch aggregation: {t_agg_end - t_agg_start:.1f}s")

        all_raw_messages.extend(batch_raw_messages)
        all_responses.extend(batch_responses)

    # --- Post-processing ---
    if args.verbose:
        print("\n" + "=" * 60)
        print("Post-processing outputs...")

    require_rubrics = not args.separate_rubric_generation

    output_dataset, stats, _extracted_tasks, _selected_contexts = (
        process_long_form_outputs(
            all_raw_messages,
            all_responses,
            dataset,
            solver_template,
            require_rubrics=require_rubrics,
            max_search_turns=args.max_search_turns,
            challenger_format=challenger_format,
        )
    )

    # --- Print stats ---
    print(f"\n=== Phase 1 generation complete ===")
    print(f"Total prompts: {stats['total_prompts']}")
    print(f"Total rollouts: {stats['total_rollouts']}")
    print(f"Perfect format (1.0): {stats['perfect_format']}")
    print(f"Extract failures: {stats['extract_failures']}")
    print(
        f"Valid tasks: {stats['valid']} "
        f"({stats['valid'] / max(stats['total_rollouts'], 1) * 100:.1f}%)"
    )
    print(
        f"Expansion: {stats['valid'] / max(stats['total_prompts'], 1):.2f}x "
        f"tasks/prompt"
    )

    # --- Save output ---
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write using HuggingFace datasets for native Arrow nested type support.
    # This avoids serializing prompt/reward_model/metadata to JSON strings,
    # which would break verl's rl_dataset.py (expects list/dict, not str).
    ds = hf_datasets.Dataset.from_pandas(output_dataset)
    ds.to_parquet(args.output_path)
    print(f"\nOutput saved to {args.output_path}")


def main() -> None:
    """Main entry point with argparse for Phase 1 task generation."""
    arg_parser = argparse.ArgumentParser(
        description=(
            "Phase 1 task generation via SGLang server (OpenAI-compatible API). "
            "Connects to an SGLang server for batch generation with adaptive "
            "multi-turn tool use."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with separate rubric generation
  python scope/create_task.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --base-url http://127.0.0.1:8001 \\
      --input-path ./data/challenger_generated.parquet \\
      --output-path ./data/long_gen_intermediate.parquet \\
      --separate-rubric-generation \\
      --n-samples 8 --verbose

  # Small test run
  python scope/create_task.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --base-url http://127.0.0.1:8001 \\
      --input-path ./data/challenger_generated.parquet \\
      --output-path ./data/test_intermediate.parquet \\
      --separate-rubric-generation \\
      --n-samples 2 --batch-size 4 --verbose

  # With partition for parallel runs
  python scope/create_task.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --base-url http://127.0.0.1:8001 \\
      --input-path ./data/challenger_generated.parquet \\
      --output-path ./data/intermediate_p1.parquet \\
      --separate-rubric-generation \\
      --partition 1 --verbose
        """,
    )

    # Required arguments
    arg_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path (HuggingFace name or local checkpoint for tokenizer)",
    )
    arg_parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="SGLang server URL, e.g. http://127.0.0.1:8001",
    )
    arg_parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Input parquet file path",
    )
    arg_parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output parquet file path",
    )

    # Server settings
    arg_parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model identifier for API calls (default: same as --model)",
    )
    arg_parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Max context length for token budget check (default: 32768)",
    )
    arg_parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds (default: 600.0)",
    )
    arg_parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for API calls (default: 5)",
    )
    arg_parser.add_argument(
        "--request-batch-size",
        type=int,
        default=1024,
        help="Max prompts per API request for internal chunking (default: 1024)",
    )

    # Template and config paths
    arg_parser.add_argument(
        "--solver-template-path",
        type=str,
        default="scope/prompts/solver_search_r1.txt",
        help="Solver template path (default: scope/prompts/solver_search_r1.txt)",
    )
    arg_parser.add_argument(
        "--tool-config-path",
        type=str,
        default="config/search_tool_config.yaml",
        help="Tool config YAML path (default: config/search_tool_config.yaml)",
    )

    # Format settings
    arg_parser.add_argument(
        "--format",
        type=str,
        default="tool_call",
        choices=["tool_call", "search_r1", "function_calls", "function_calls_xml"],
        help="Tool interaction format (default: tool_call).",
    )

    # Retrieval settings
    arg_parser.add_argument(
        "--retrieval-url",
        type=str,
        default="http://127.0.0.1:8000/retrieve",
        help="Retrieval server URL (default: http://127.0.0.1:8000/retrieve)",
    )
    arg_parser.add_argument(
        "--retrieval-topk",
        type=int,
        default=3,
        help="Number of documents per query (default: 3)",
    )
    arg_parser.add_argument(
        "--search-chunk-size",
        type=int,
        default=2048,
        help="Max queries per batched search API call (default: 2048)",
    )

    # Generation settings
    arg_parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Number of rollout samples per prompt (default: 8)",
    )
    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of prompts per batch (default: 5000)",
    )
    arg_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    arg_parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling (default: 1.0)",
    )
    arg_parser.add_argument(
        "--response-length",
        type=int,
        default=8192,
        help="Max response tokens per generation (default: 8192)",
    )

    # Multi-turn settings
    arg_parser.add_argument(
        "--max-assistant-turns",
        type=int,
        default=5,
        help="Max assistant turns per conversation (default: 5)",
    )
    arg_parser.add_argument(
        "--max-tool-response-length",
        type=int,
        default=500,
        help="Max tool response chars before truncation (default: 500)",
    )

    # Pipeline settings
    arg_parser.add_argument(
        "--separate-rubric-generation",
        action="store_true",
        help="V5 mode: tasks without rubrics (rubrics generated separately)",
    )
    arg_parser.add_argument(
        "--max-search-turns",
        type=int,
        default=4,
        help="Max search turns for solver template (default: 4)",
    )

    # Data settings
    arg_parser.add_argument(
        "--partition",
        type=int,
        default=None,
        help="Partition index (1-5) for data subsetting (default: None)",
    )
    arg_parser.add_argument(
        "--prompt-key",
        type=str,
        default="prompt",
        help="Column name for prompts in parquet (default: prompt)",
    )

    # Stop tokens
    arg_parser.add_argument(
        "--challenger-stop-tokens",
        type=str,
        nargs="*",
        default=None,
        help="Stop tokens for challenger generation (default: auto based on format)",
    )

    # Dynamic user turns
    arg_parser.add_argument(
        "--dynamic-user-turns",
        action="store_true",
        help="Inject chain2-style user messages between search turns",
    )
    arg_parser.add_argument(
        "--dynamic-style",
        type=str,
        default="chain2",
        choices=list(DYNAMIC_STYLES.keys()),
        help="Dynamic message style key (default: chain2)",
    )
    # Output settings
    arg_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress and debug information",
    )

    args = arg_parser.parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
