"""Synchronous batch multi-turn reward rollout.

Replaces per-request async HTTP calls with synchronized turn-by-turn batch
waves. Instead of N independent async conversations, processes ALL conversations
in lockstep using the OpenAI-compatible ``/v1/completions`` endpoint natively
supported by both SGLang and VLLM servers.

Typical usage::

    with BatchMultiTurnRollout(
        solver_base_url="http://127.0.0.1:8001",
        processing_class=tokenizer,
    ) as rollout:
        histories = rollout.generate_sequences_batch(initial_messages)
        grader_outputs = rollout.grade_batch(grader_prompts, "http://127.0.0.1:8002")
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from verl.tools.utils.search_r1_like_utils import _passages2string, call_search_api

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser as _FunctionCallParser
except ImportError:
    try:
        from sglang.srt.function_call_parser import FunctionCallParser as _FunctionCallParser
    except ImportError:
        _FunctionCallParser = None

try:
    from sglang.srt.entrypoints.openai.protocol import Tool as _SglTool
except ImportError:
    try:
        from sglang.srt.openai_api.protocol import Tool as _SglTool
    except ImportError:
        _SglTool = None

logger = logging.getLogger(__name__)


class SearchServerDownError(RuntimeError):
    """Raised when the retrieval server fails too many consecutive batches.

    Callers should catch this and abort before spending grader API credits
    on degraded results where the model cannot search.
    """


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
SEARCH_R1_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
FC_SEARCH_RE = re.compile(
    r'<function_calls>\s*search\s*\(\s*query\s*=\s*["\'](.+?)["\']\s*\)',
    re.DOTALL,
)
FC_ANSWER_RE = re.compile(
    r'<function_calls>\s*answer\s*\(\s*answer\s*=\s*["\'](.+?)["\']\s*\)',
    re.DOTALL,
)
# DR-Tulu <call_tool name="...">query</call_tool> format
CALL_TOOL_RE = re.compile(
    r"<call_tool\s+([^>]*?)>(.*?)(?:</call_tool>|</call>)",
    re.DOTALL,
)
_CALL_TOOL_NAME_RE = re.compile(r'name="([^"]+)"')
# WebThinker <|begin_search_query|>query<|end_search_query|> format
WEBTHINKER_SEARCH_RE = re.compile(
    r"<\|begin_search_query\|>(.*?)<\|end_search_query\|>",
    re.DOTALL,
)

# Position-based parsing regexes (match sglang_rollout.py)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)

# Error feedback message for malformed tags (matches sglang_rollout.py:365)
_INVALID_ACTION_MESSAGE = (
    "My previous action is invalid. If I want to search, I should put the query "
    "between <search> and </search>. If I want to give the final answer, I should "
    "put the answer between <answer> and </answer>. Let me try again."
)

# Known tool-call delimiter pairs for regex fallback detection
_KNOWN_TOOL_PATTERNS = [
    ("<tool_call>", "</tool_call>"),           # Qwen
    ("<function_calls>", "</function_calls>"), # OLMo
]

_PYTHON_SEARCH_CALL_RE = re.compile(
    r'search\s*\(\s*query_list\s*=\s*(\[.*?\])\s*\)',
    re.DOTALL,
)

_OPENAI_MODEL_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")

# sglang server reserves a few tokens internally relative to --context-length
# (e.g. 16384 → max input 16378). Use 16 to leave headroom across model
# templates and avoid 400 errors at the prompt-length boundary.
_SGLANG_SAFETY_MARGIN = 16


def _extract_usage_dict(response: Any, model: str) -> dict[str, Any] | None:
    """Normalize an OpenAI ``chat.completions`` usage object to a plain dict.

    Returns ``None`` if the response has no usage info (e.g. non-OpenAI
    backends). Only the fields we care about downstream (token counts +
    model name) are kept; cached-input and reasoning-token details are
    carried through when present so later consumers can use them without
    a second round of introspection.

    Args:
        response: OpenAI SDK chat completion response object.
        model: Model name (used for pricing lookup later).

    Returns:
        dict with ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
            ``model``, plus optional ``cached_input_tokens`` and
            ``reasoning_tokens``, or ``None`` if no usage is available.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    out: dict[str, Any] = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "model": model,
    }
    ptd = getattr(usage, "prompt_tokens_details", None)
    if ptd is not None:
        cached = getattr(ptd, "cached_tokens", None)
        if cached is not None:
            out["cached_input_tokens"] = int(cached)
    ctd = getattr(usage, "completion_tokens_details", None)
    if ctd is not None:
        reasoning = getattr(ctd, "reasoning_tokens", None)
        if reasoning is not None:
            out["reasoning_tokens"] = int(reasoning)
    return out


def _build_grader_call_entry(
    prompt: str,
    output: str,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a single ``grader_calls`` entry for the output schema.

    ``cost_usd`` is intentionally not filled here — the pricing table
    lives in the eval script; callers add it after receiving this dict.

    Args:
        prompt: Full grader prompt text for this call.
        output: Raw grader response text.
        usage: Usage dict as returned by ``_extract_usage_dict`` (or
            ``None`` if unavailable).

    Returns:
        Dict with ``prompt``, ``output``, ``input_tokens``,
        ``output_tokens``, ``model`` keys (plus optional
        ``cached_input_tokens`` / ``reasoning_tokens`` when present).
    """
    entry: dict[str, Any] = {
        "prompt": prompt,
        "output": output,
        "input_tokens": 0,
        "output_tokens": 0,
        "model": "",
    }
    if usage is not None:
        entry["input_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
        entry["output_tokens"] = int(usage.get("completion_tokens", 0) or 0)
        entry["model"] = usage.get("model", "")
        if "cached_input_tokens" in usage:
            entry["cached_input_tokens"] = usage["cached_input_tokens"]
        if "reasoning_tokens" in usage:
            entry["reasoning_tokens"] = usage["reasoning_tokens"]
        # Per-item failure marker propagated up by ``_call`` so the save
        # layer can write a stub judge entry and resume can retry it.
        if "error" in usage:
            entry["error"] = usage["error"]
        # Soft-skip marker (e.g. length-exhausted empty completion).
        # Does NOT trigger stub/resume — just annotates the call.
        if "skipped" in usage:
            entry["skipped"] = usage["skipped"]
    return entry


def _build_grader_calls_from_flat(
    rollout: Any,
    n_items: int,
    flat_map: list[tuple[int, Any]],
    flat_prompts: list[str],
    raw_outputs: list[str],
    usage_start: int,
) -> list[list[dict[str, Any]]]:
    """Build per-item ``grader_calls`` lists from parallel flat inputs.

    Each position in ``flat_map`` / ``flat_prompts`` / ``raw_outputs``
    corresponds to one API call. The usage entry for call ``fi`` is
    ``rollout.grader_usage_log[usage_start + fi]``.

    Args:
        rollout: The ``BatchMultiTurnRollout`` instance whose
            ``grader_usage_log`` contains the usage entries.
        n_items: Total number of (outer) items to distribute calls across.
        flat_map: List of ``(item_idx, _)`` tuples, one per flat call.
        flat_prompts: Prompts sent per flat call.
        raw_outputs: Output texts per flat call.
        usage_start: Index into ``rollout.grader_usage_log`` where this
            batch's entries start. Typically captured as
            ``len(rollout.grader_usage_log)`` right before the
            ``grade_batch`` call.

    Returns:
        list[list[dict[str, Any]]]: ``per_item[item_idx]`` is the list of
            grader call dicts for that item.
    """
    usage_entries = rollout.grader_usage_log[usage_start:]
    per_item: list[list[dict[str, Any]]] = [[] for _ in range(n_items)]
    for fi, mapping in enumerate(flat_map):
        item_idx = mapping[0]
        prompt = flat_prompts[fi] if fi < len(flat_prompts) else ""
        output = raw_outputs[fi] if fi < len(raw_outputs) else ""
        usage = usage_entries[fi] if fi < len(usage_entries) else None
        per_item[item_idx].append(
            _build_grader_call_entry(prompt, output, usage)
        )
    return per_item


def _is_openai_model(model_name: str) -> bool:
    """Check whether a model name refers to an OpenAI-hosted model.

    Detection is based on well-known model name prefixes (``gpt-``,
    ``o1-``, ``o3-``, ``o4-``).

    Args:
        model_name: The model identifier string.

    Returns:
        bool: ``True`` if the model name starts with a known OpenAI prefix.
    """
    return any(model_name.startswith(p) for p in _OPENAI_MODEL_PREFIXES)


def _parse_python_search_call(content: str) -> dict[str, Any] | None:
    """Parse Python-style ``search(query_list=[...])`` into a tool call dict.

    Uses ``ast.literal_eval`` for safe parsing (no code execution).

    Args:
        content: Raw text that may contain a Python-style search call.

    Returns:
        dict[str, Any] | None: A tool call dict with ``name`` and
            ``arguments`` keys, or ``None`` if parsing fails or the
            result is invalid.
    """
    m = _PYTHON_SEARCH_CALL_RE.search(content)
    if not m:
        return None
    try:
        query_list = ast.literal_eval(m.group(1))
        if (
            isinstance(query_list, list)
            and len(query_list) > 0
            and all(isinstance(q, str) for q in query_list)
        ):
            return {"name": "search", "arguments": {"query_list": query_list}}
    except (ValueError, SyntaxError):
        pass
    return None


@dataclass
class ConversationState:
    """Tracks the state of a single conversation within a batch rollout.

    Attributes:
        index: Position in the original batch.
        messages: Conversation history as list of role/content dicts.
        is_active: Whether this conversation still needs more turns.
        final_response: Last assistant content when finished.
        finish_reason: Reason the conversation ended
            (``"stop"`` | ``"length"`` | ``"max_turns"``).
        turns_completed: Number of assistant turns completed.
        initial_prompt_len: Tokenized length of the initial rendered
            prompt (before any assistant turn). Used to compute the
            cumulative post-prompt token count for ``max_response_length``
            enforcement.
    """

    index: int
    messages: list[dict[str, str]] = field(default_factory=list)
    is_active: bool = True
    final_response: str = ""
    finish_reason: str = ""
    turns_completed: int = 0
    initial_prompt_len: int = 0


class BatchMultiTurnRollout:
    """Synchronous batch multi-turn rollout using OpenAI-compatible endpoints.

    Processes all conversations in lockstep turn-by-turn waves, dramatically
    reducing the number of HTTP requests from O(N * turns) to O(turns).

    Class constants:
        _MAX_FORCE_ANSWER_RETRIES: Maximum force-answer retry rounds in
            ``_solver_retry_phase``. Matches verl's sglang_rollout
            ``_MAX_FORCE_ANSWER_RETRIES`` so standalone eval and verl-driven
            eval nudge the same number of times.

    Args:
        solver_base_url: Base URL of the solver server
            (e.g. ``"http://127.0.0.1:8001"``).
        retrieval_url: URL of the retrieval server endpoint.
        processing_class: Tokenizer instance (e.g. Qwen tokenizer) with
            ``apply_chat_template`` method.
        model_name: Model identifier for the solver server.
        max_turns: Maximum number of total assistant turns per conversation
            (including the final answer turn).  This matches the framework's
            ``max_assistant_turns`` semantics — the model must budget its
            own answer turn within this limit.
        max_tokens: Maximum tokens to generate per turn.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        retrieval_topk: Number of top retrieval results per query.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum number of retries for solver API calls.
        max_prompt_length: Maximum prompt token length. Prompts exceeding
            this are terminated with ``finish_reason="length"``.
        max_response_length: Maximum cumulative post-prompt token length
            across all turns. Matches verl's ``data.max_response_length``
            semantic: once ``rendered_prompt_tokens - initial_prompt_tokens
            >= max_response_length``, the conversation is terminated with
            ``finish_reason="length"``. ``None`` (default) disables the
            cap.
        max_tool_response_length: Maximum length of tool responses (in
            characters or tokens depending on ``tool_response_truncation_unit``).
            Responses exceeding this limit are truncated with a
            ``"...(truncated)"`` suffix. Set to ``0`` to disable truncation.
        tool_schemas: Optional list of OpenAI-format tool schema dicts.
            When provided, these are passed to ``apply_chat_template`` via
            ``tools=`` so the tokenizer injects the model-specific tool-use
            system prompt. Also used to initialize sglang's
            ``FunctionCallParser`` for model-agnostic tool call parsing.
        solver_retry: If ``True``, attempt one nudge retry for
            conversations that lack an ``<answer>`` tag after the main
            loop.  This covers both conversations ending on a tool/user
            message (model used all turns for search) and conversations
            ending on an assistant message without ``<answer>``.  A nudge
            user message is injected and one more turn is generated.
        solver_format: Tool interaction format for the solver.
            ``"tool_call"`` (default) uses ``<tool_call>`` JSON format
            with model-specific tool schema injection. ``"search_r1"``
            uses ``<search>query</search>`` / ``<information>`` tags and
            skips tool schema injection entirely.
        solver_stop: Optional list of stop token strings (e.g.
            ``["</answer>"]``) passed to the solver server. If
            ``"</answer>"`` is in the list and the response has an
            ``<answer>`` tag without a closing ``</answer>``, the closing
            tag is automatically appended.
        grader_batch_size: Maximum number of prompts per grader API call.
            ``0`` (default) sends all prompts in a single call.
    """

    _MAX_FORCE_ANSWER_RETRIES: int = 5

    def __init__(
        self,
        solver_base_url: str,
        retrieval_url: str = "http://127.0.0.1:8000/retrieve",
        processing_class: Any = None,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_turns: int = 5,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        retrieval_topk: int = 3,
        timeout: float = 86400.0,
        max_retries: int = 0,
        max_prompt_length: int = 4096,
        max_response_length: int | None = None,
        max_model_len: int | None = None,
        max_tool_response_length: int = 500,
        tool_response_truncation_unit: str = "char",
        tool_schemas: list[dict[str, Any]] | None = None,
        solver_format: str = "tool_call",
        solver_retry: bool = False,
        solver_stop: list[str] | None = None,
        grader_batch_size: int = 0,
        openai_concurrency: int = 16,
        tool_call_parser_type: str | None = None,
        grader_reasoning_effort: str | None = None,
        grader_max_input_chars: int = 0,
        max_consecutive_search_failures: int = 0,
        grader_api_mode: str = "auto",
        grader_api_base_url: str | None = None,
        grader_api_key_env: str = "OPENAI_API_KEY",
        grader_provider_order: list[str] | None = None,
        grader_allow_fallbacks: bool = True,
        grader_require_parameters: bool = False,
        grader_request_timeout: float | None = None,
        grader_app_max_retries: int | None = None,
        grader_thinking_budget: int | None = None,
    ) -> None:
        self.grader_batch_size = grader_batch_size
        self.openai_concurrency = openai_concurrency
        self.solver_format = solver_format
        # Optional OpenAI ``reasoning_effort`` value (``"minimal" | "low" |
        # "medium" | "high"``) forwarded to ``chat.completions.create`` for
        # gpt-5 family graders. ``None`` -> param not sent (model default).
        self.grader_reasoning_effort = grader_reasoning_effort
        # Grader routing:
        #   "auto" (default)           -> prefix-sniff via ``_is_openai_model``
        #   "local"                    -> force local /v1/completions path
        #   "openai_compatible"        -> force ``_grade_batch_openai`` path
        #                                 (OpenAI proper, OpenRouter, Fireworks,
        #                                 Together, etc.) using ``grader_api_base_url``
        self.grader_api_mode = grader_api_mode
        # Optional override for the OpenAI-compatible base URL. ``None`` leaves
        # the SDK default (``api.openai.com``). Set to
        # ``"https://openrouter.ai/api/v1"`` for OpenRouter.
        self.grader_api_base_url = grader_api_base_url
        # Name of the environment variable holding the API key. Defaults to
        # ``OPENAI_API_KEY``; override with ``OPENROUTER_API_KEY`` for OpenRouter.
        self.grader_api_key_env = grader_api_key_env
        # Optional OpenRouter provider pinning. When non-empty, sends
        # ``extra_body={"provider": {"order": [...], "allow_fallbacks": ...}}``.
        self.grader_provider_order = grader_provider_order
        self.grader_allow_fallbacks = grader_allow_fallbacks
        self.grader_require_parameters = grader_require_parameters
        # Remote-specific per-request timeout (seconds). Overrides the global
        # ``self.timeout`` when using the OpenAI-compatible path to avoid the
        # 24h default hanging remote coroutines.
        self.grader_request_timeout = grader_request_timeout
        # App-level retry count override for ``_grade_batch_openai``. ``None``
        # uses mode-based defaults (5 for ``openai_compatible``, 3 for ``auto``/
        # ``openai``).
        self.grader_app_max_retries = grader_app_max_retries
        # Optional Anthropic extended-thinking budget (tokens). When set,
        # ``_grade_batch_openai`` sends ``extra_body={"thinking": ...}`` for
        # Claude models via the Anthropic OpenAI-compatible endpoint.
        self.grader_thinking_budget = grader_thinking_budget
        # Pre-flight skip for grader inputs whose concatenated message
        # content exceeds this many characters. ``0`` disables the check.
        # Useful when the upstream API (OpenAI gateway) rejects oversized
        # payloads with opaque "could not parse JSON body" 400s.
        self.grader_max_input_chars = grader_max_input_chars
        # Circuit breaker: abort after N consecutive failed retrieval batches.
        # 0 disables (old behaviour).
        self._cb_max = max_consecutive_search_failures
        self._cb_consecutive = 0
        # Cumulative log of per-API-call usage dicts (one entry per
        # OpenAI grader request, appended in call-order). Populated by
        # ``_grade_batch_openai``. Read by callers (``eval_standalone`` +
        # benchmark evaluators) to attribute tokens/cost per item.
        self.grader_usage_log: list[dict[str, Any]] = []
        self.solver_client = OpenAI(
            base_url=f"{solver_base_url}/v1",
            api_key="EMPTY",
            timeout=timeout,
            max_retries=max_retries,
        )
        self.retrieval_url = retrieval_url
        self.processing_class = processing_class
        self.model_name = model_name
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.retrieval_topk = retrieval_topk
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_len = max_model_len
        self.max_tool_response_length = max_tool_response_length
        self.tool_response_truncation_unit = tool_response_truncation_unit
        self.solver_retry = solver_retry
        self.solver_stop = solver_stop
        self._explicit_parser_type = tool_call_parser_type

        # Tool parsing setup
        self._function_call_parser: Any = None
        self._tools_for_template: list[dict[str, Any]] | None = None
        self._fallback_patterns: list[re.Pattern[str]] = [TOOL_CALL_RE]

        if tool_schemas and self.solver_format in ("function_calls", "function_calls_xml"):
            # Inject tool schemas into template (for OLMo function-calling
            # system prompt) but skip FunctionCallParser (use regex instead).
            self._tools_for_template = tool_schemas
        elif tool_schemas and self.solver_format not in ("search_r1", "call_tool", "webthinker", "webexplorer"):
            self._tools_for_template = tool_schemas
            self._init_tool_parser(processing_class, tool_schemas)

    def reset_usage_log(self) -> None:
        """Clear ``grader_usage_log`` in preparation for a new grading phase."""
        self.grader_usage_log = []

    def __enter__(self) -> BatchMultiTurnRollout:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def _init_tool_parser(
        self,
        processing_class: Any,
        tool_schemas: list[dict[str, Any]],
    ) -> None:
        """Initialize model-agnostic tool call parser from tool schemas.

        Tries sglang's ``FunctionCallParser`` first (detects parser type from
        the tokenizer vocab). Falls back to regex-based parsing using known
        tool-call delimiter pairs discovered in the vocab.

        Args:
            processing_class: Tokenizer with ``get_vocab`` method.
            tool_schemas: List of OpenAI-format tool schema dicts.
        """
        # Obtain tokenizer vocab for parser detection
        vocab: dict[str, int] | None = None
        try:
            vocab = processing_class.get_vocab()
        except AttributeError:
            try:
                vocab = processing_class.tokenizer.get_vocab()
            except AttributeError:
                return

        # Path 1: sglang FunctionCallParser (model-agnostic)
        if _FunctionCallParser is not None and _SglTool is not None and vocab is not None:
            try:
                # Use explicit parser type if provided (bypasses auto-detection
                # which can pick wrong type, e.g. "glm" for Qwen3).
                if self._explicit_parser_type is not None:
                    sgl_tools = [_SglTool.model_validate(s) for s in tool_schemas]
                    self._function_call_parser = _FunctionCallParser(
                        sgl_tools, self._explicit_parser_type,
                    )
                    logger.info(
                        "Tool parser: FunctionCallParser type=%s (explicit)",
                        self._explicit_parser_type,
                    )
                    return

                for pt, parser_cls in _FunctionCallParser.ToolCallParserEnum.items():
                    p = parser_cls()
                    if p.bot_token.strip() in vocab and (
                        p.eot_token == "" or p.eot_token.strip() in vocab
                    ):
                        sgl_tools = [_SglTool.model_validate(s) for s in tool_schemas]
                        self._function_call_parser = _FunctionCallParser(sgl_tools, pt)
                        logger.info("Tool parser: FunctionCallParser type=%s", pt)
                        return
            except Exception as e:
                logger.debug("FunctionCallParser init failed: %s", e)

        # Path 2: Regex fallback — collect ALL matching patterns from vocab
        if vocab is not None:
            patterns: list[re.Pattern[str]] = []
            for bot, eot in _KNOWN_TOOL_PATTERNS:
                if bot in vocab:
                    patterns.append(
                        re.compile(
                            f"{re.escape(bot)}(.*?){re.escape(eot)}", re.DOTALL,
                        )
                    )
            if patterns:
                self._fallback_patterns = patterns
                logger.info(
                    "Tool parser: regex fallback with %d pattern(s)",
                    len(patterns),
                )

    def generate_sequences_batch(
        self, initial_messages: list[list[dict[str, str]]]
    ) -> list[list[dict[str, str]]]:
        """Run batch multi-turn rollout for all conversations in lockstep.

        Args:
            initial_messages: List of initial message histories, each a list
                of role/content dicts (e.g. ``[{"role": "user", "content": "..."}]``).

        Returns:
            list[list[dict[str, str]]]: Final message histories per conversation.
        """
        N = len(initial_messages)
        if N == 0:
            return []

        states = [
            ConversationState(index=i, messages=list(msgs))
            for i, msgs in enumerate(initial_messages)
        ]

        if self.max_response_length is not None:
            init_chat_kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": False,
            }
            if self._tools_for_template is not None:
                init_chat_kwargs["tools"] = self._tools_for_template
            for s in states:
                init_prompt = self.processing_class.apply_chat_template(
                    s.messages, **init_chat_kwargs,
                )
                s.initial_prompt_len = len(
                    self.processing_class(
                        init_prompt, add_special_tokens=False,
                    )["input_ids"]
                )

        print(
            f"[BatchRollout] Starting: {N} conversations, max_turns={self.max_turns}"
        )
        total_start = time.time()
        turns_completed = 0

        for turn in range(self.max_turns):
            active = [s for s in states if s.is_active]
            if not active:
                break

            turn_start = time.time()
            print(
                f"[BatchRollout] === Turn {turn + 1}/{self.max_turns}"
                f" | {len(active)}/{N} active ==="
            )

            # 1. Build prompts via apply_chat_template
            t0 = time.time()
            prompts = []
            chat_kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": False,
            }
            if self._tools_for_template is not None:
                chat_kwargs["tools"] = self._tools_for_template
            for s in active:
                prompt = self.processing_class.apply_chat_template(
                    s.messages, **chat_kwargs,
                )
                prompts.append(prompt)

            # 2. Filter prompts exceeding max length
            valid_active: list[ConversationState] = []
            valid_prompts: list[str] = []
            filtered_count = 0
            filtered_response_count = 0
            for s, prompt in zip(active, prompts):
                token_ids = self.processing_class(
                    prompt, add_special_tokens=False,
                )["input_ids"]
                # Hard cap: if even one completion token wouldn't fit, stop.
                # Matches verl's sglang_rollout.py:1886 LENGTH behavior.
                # sglang reserves ~6 tokens internally for special/chat
                # template tokens; use 16 for safety margin.
                model_cap = (
                    self.max_model_len - _SGLANG_SAFETY_MARGIN
                    if self.max_model_len is not None
                    else self.max_prompt_length
                )
                if (
                    len(token_ids) > self.max_prompt_length
                    or len(token_ids) >= model_cap
                ):
                    s.is_active = False
                    s.finish_reason = "length"
                    s.final_response = (
                        s.messages[-1]["content"]
                        if s.messages else ""
                    )
                    filtered_count += 1
                elif (
                    self.max_response_length is not None
                    and len(token_ids) - s.initial_prompt_len
                    >= self.max_response_length
                ):
                    # Post-prompt cumulative token budget exhausted. Matches
                    # verl's ``data.max_response_length`` semantic.
                    s.is_active = False
                    s.finish_reason = "length"
                    s.final_response = (
                        s.messages[-1]["content"]
                        if s.messages else ""
                    )
                    filtered_response_count += 1
                else:
                    valid_active.append(s)
                    valid_prompts.append(prompt)

            prompt_time = time.time() - t0
            filtered_parts: list[str] = []
            if filtered_count:
                filtered_parts.append(f"{filtered_count} filtered (prompt)")
            if filtered_response_count:
                filtered_parts.append(
                    f"{filtered_response_count} filtered (response)"
                )
            filtered_str = (
                " | " + ", ".join(filtered_parts) if filtered_parts else ""
            )
            print(
                f"[BatchRollout]   Prompt build: {prompt_time:.1f}s{filtered_str}"
            )

            if not valid_active:
                continue

            # 3. Batch solver call
            response = self._batch_solver_call(valid_prompts)
            choices = sorted(response.choices, key=lambda c: c.index)

            # 4. Process responses
            retrieval_tasks: dict[int, list[str]] = {}
            finished_stop = 0
            finished_length = 0
            for state, choice in zip(valid_active, choices):
                text = choice.text

                # search_r1: position-based parsing (matches sglang_rollout.py)
                if self.solver_format == "search_r1":
                    answer_match = _ANSWER_RE.search(text)
                    search_match = _SEARCH_RE.search(text)
                    valid_search = (
                        search_match
                        if (search_match and search_match.group(1).strip())
                        else None
                    )

                    answer_first = answer_match and (
                        not valid_search
                        or answer_match.start() <= valid_search.start()
                    )
                    search_first = valid_search and (
                        not answer_match
                        or valid_search.start() < answer_match.start()
                    )

                    if answer_first:
                        text = text[: answer_match.end()]
                        state.messages.append(
                            {"role": "assistant", "content": text}
                        )
                        state.turns_completed += 1
                        state.is_active = False
                        state.final_response = text
                        state.finish_reason = "stop"
                        finished_stop += 1
                    elif search_first:
                        text = text[: search_match.end()]
                        query = search_match.group(1).strip()
                        state.messages.append(
                            {"role": "assistant", "content": text}
                        )
                        state.turns_completed += 1
                        retrieval_tasks[state.index] = [query]
                    else:
                        # No valid answer or search found
                        state.messages.append(
                            {"role": "assistant", "content": text}
                        )
                        state.turns_completed += 1
                        # Error feedback: malformed <search> tag, give model
                        # another chance (matches sglang_rollout.py behavior)
                        if (
                            "<search>" in text
                            and (turn + 1) < self.max_turns
                        ):
                            state.messages.append(
                                {
                                    "role": "user",
                                    "content": _INVALID_ACTION_MESSAGE,
                                }
                            )
                        elif choice.finish_reason == "length":
                            state.is_active = False
                            state.final_response = text
                            state.finish_reason = "length"
                            finished_length += 1
                        else:
                            state.is_active = False
                            state.final_response = text
                            state.finish_reason = "stop"
                            finished_stop += 1
                    continue

                # call_tool: position-based parsing for DR-Tulu format
                if self.solver_format == "call_tool":
                    answer_match = _ANSWER_RE.search(text)
                    call_match = CALL_TOOL_RE.search(text)
                    valid_call = (
                        call_match
                        if (call_match and call_match.group(2).strip())
                        else None
                    )
                    answer_first = answer_match and (
                        not valid_call
                        or answer_match.start() <= valid_call.start()
                    )
                    call_first = valid_call and (
                        not answer_match
                        or valid_call.start() < answer_match.start()
                    )
                    if answer_first:
                        text = text[: answer_match.end()]
                        state.messages.append(
                            {"role": "assistant", "content": text}
                        )
                        state.turns_completed += 1
                        state.is_active = False
                        state.final_response = text
                        state.finish_reason = "stop"
                        finished_stop += 1
                    elif call_first:
                        text = text[: call_match.end()]
                        query = call_match.group(2).strip()
                        state.messages.append(
                            {"role": "assistant", "content": text}
                        )
                        state.turns_completed += 1
                        retrieval_tasks[state.index] = [query]
                    else:
                        state.messages.append(
                            {"role": "assistant", "content": text}
                        )
                        state.turns_completed += 1
                        if (
                            "<call_tool" in text
                            and (turn + 1) < self.max_turns
                        ):
                            state.messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "My previous action is invalid. "
                                        "To search, use <call_tool name=\"google_search\">query</call_tool>. "
                                        "To answer, use <answer>answer</answer>."
                                    ),
                                }
                            )
                        elif choice.finish_reason == "length":
                            state.is_active = False
                            state.final_response = text
                            state.finish_reason = "length"
                            finished_length += 1
                        else:
                            state.is_active = False
                            state.final_response = text
                            state.finish_reason = "stop"
                            finished_stop += 1
                    continue

                # webthinker: position-based parsing for WebThinker format
                if self.solver_format == "webthinker":
                    answer_match = _ANSWER_RE.search(text)
                    search_match = WEBTHINKER_SEARCH_RE.search(text)
                    valid_search = (
                        search_match
                        if (search_match and search_match.group(1).strip())
                        else None
                    )
                    answer_first = answer_match and (
                        not valid_search
                        or answer_match.start() <= valid_search.start()
                    )
                    search_first = valid_search and (
                        not answer_match
                        or valid_search.start() < answer_match.start()
                    )
                    if answer_first:
                        text = text[: answer_match.end()]
                        state.messages.append(
                            {"role": "assistant", "content": text}
                        )
                        state.turns_completed += 1
                        state.is_active = False
                        state.final_response = text
                        state.finish_reason = "stop"
                        finished_stop += 1
                    elif search_first:
                        text = text[: search_match.end()]
                        query = search_match.group(1).strip()
                        state.messages.append(
                            {"role": "assistant", "content": text}
                        )
                        state.turns_completed += 1
                        retrieval_tasks[state.index] = [query]
                    else:
                        state.messages.append(
                            {"role": "assistant", "content": text}
                        )
                        state.turns_completed += 1
                        if (
                            "<|begin_search_query|>" in text
                            and (turn + 1) < self.max_turns
                        ):
                            state.messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "My previous action is invalid. "
                                        "To search, use <|begin_search_query|>query<|end_search_query|>. "
                                        "When done, provide your final answer."
                                    ),
                                }
                            )
                        elif choice.finish_reason == "length":
                            state.is_active = False
                            state.final_response = text
                            state.finish_reason = "length"
                            finished_length += 1
                        else:
                            state.is_active = False
                            state.final_response = text
                            state.finish_reason = "stop"
                            finished_stop += 1
                    continue

                # webexplorer: position-based parsing for WebExplorer format
                if self.solver_format == "webexplorer":
                    answer_match = _ANSWER_RE.search(text)
                    tc_match = TOOL_CALL_RE.search(text)
                    # Validate the tool_call contains parseable JSON
                    valid_tc = None
                    if tc_match:
                        try:
                            parsed_json = json.loads(tc_match.group(1).strip())
                            if isinstance(parsed_json, dict):
                                valid_tc = tc_match
                        except (json.JSONDecodeError, TypeError):
                            pass
                    answer_first = answer_match and (
                        not valid_tc
                        or answer_match.start() <= valid_tc.start()
                    )
                    tc_first = valid_tc and (
                        not answer_match
                        or valid_tc.start() < answer_match.start()
                    )
                    if answer_first:
                        text = text[: answer_match.end()]
                        state.messages.append(
                            {"role": "assistant", "content": text}
                        )
                        state.turns_completed += 1
                        state.is_active = False
                        state.final_response = text
                        state.finish_reason = "stop"
                        finished_stop += 1
                    elif tc_first:
                        text = text[: tc_match.end()]
                        state.messages.append(
                            {"role": "assistant", "content": text}
                        )
                        state.turns_completed += 1
                        calls = self._parse_webexplorer_calls(text)
                        if calls:
                            queries = []
                            for tc in calls:
                                queries.extend(tc["arguments"]["query_list"])
                            retrieval_tasks[state.index] = queries
                        else:
                            # Parsed <tool_call> but couldn't extract queries
                            if (turn + 1) < self.max_turns:
                                state.messages.append(
                                    {
                                        "role": "user",
                                        "content": (
                                            "My previous action is invalid. "
                                            'To search, use <tool_call>{"name": "search", '
                                            '"arguments": {"queries": ["your query"]}}</tool_call>. '
                                            "To answer, use <answer>answer</answer>."
                                        ),
                                    }
                                )
                            else:
                                state.is_active = False
                                state.final_response = text
                                state.finish_reason = "stop"
                                finished_stop += 1
                    else:
                        state.messages.append(
                            {"role": "assistant", "content": text}
                        )
                        state.turns_completed += 1
                        if (
                            "<tool_call>" in text
                            and (turn + 1) < self.max_turns
                        ):
                            state.messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "My previous action is invalid. "
                                        'To search, use <tool_call>{"name": "search", '
                                        '"arguments": {"queries": ["your query"]}}</tool_call>. '
                                        "To answer, use <answer>answer</answer>."
                                    ),
                                }
                            )
                        elif choice.finish_reason == "length":
                            state.is_active = False
                            state.final_response = text
                            state.finish_reason = "length"
                            finished_length += 1
                        else:
                            state.is_active = False
                            state.final_response = text
                            state.finish_reason = "stop"
                            finished_stop += 1
                    continue

                # Non-search_r1 formats: existing tool call parsing
                # When the model generates <tool_call>...</tool_call> but
                # keeps going (fabricating a tool response), the server may
                # return finish_reason="length".  We still want to extract
                # the valid tool call and strip the fabricated tail so that
                # real retrieval can proceed.
                tool_calls = self._parse_tool_calls(text)

                if tool_calls:
                    # Strip any text after the last closing tag to remove
                    # fabricated tool responses the model may have appended.
                    if self.solver_format in ("function_calls", "function_calls_xml"):
                        last_close = text.rfind("</function_calls>")
                        if last_close != -1:
                            text = text[: last_close + len("</function_calls>")]
                    else:
                        last_close = text.rfind("</tool_call>")
                        if last_close != -1:
                            text = text[: last_close + len("</tool_call>")]

                    state.messages.append(
                        {"role": "assistant", "content": text}
                    )
                    state.turns_completed += 1
                    queries: list[str] = []
                    for tc in tool_calls:
                        queries.extend(tc["arguments"]["query_list"])
                    retrieval_tasks[state.index] = queries
                elif choice.finish_reason == "length":
                    state.messages.append(
                        {"role": "assistant", "content": text}
                    )
                    state.turns_completed += 1
                    state.is_active = False
                    state.final_response = text
                    state.finish_reason = "length"
                    finished_length += 1
                else:
                    state.messages.append(
                        {"role": "assistant", "content": text}
                    )
                    state.turns_completed += 1
                    state.is_active = False
                    state.final_response = text
                    state.finish_reason = "stop"
                    finished_stop += 1

            finished_this_turn = finished_stop + finished_length
            print(
                f"[BatchRollout]   Responses: {len(retrieval_tasks)} need retrieval,"
                f" {finished_this_turn} finished"
                f" ({finished_stop} stop, {finished_length} length)"
            )

            # 5. Batch retrieval
            if retrieval_tasks:
                self._batch_retrieve(states, retrieval_tasks)

            turn_time = time.time() - turn_start
            print(f"[BatchRollout]   Turn total: {turn_time:.1f}s")
            turns_completed = turn + 1

        # Mark still-active conversations as max_turns
        for s in states:
            if s.is_active:
                s.is_active = False
                s.finish_reason = "max_turns"
                s.final_response = (
                    s.messages[-1]["content"] if s.messages else ""
                )

        # --- Solver retry phase (one chance) ---
        # Handles both tool/user-ending conversations (model used all turns
        # for search) and assistant-ending without <answer>.  Injects a nudge
        # and generates one more turn.
        if self.solver_retry:
            self._solver_retry_phase(states)

        finish_counts: dict[str, int] = {"stop": 0, "length": 0, "max_turns": 0}
        for s in states:
            finish_counts[s.finish_reason] = (
                finish_counts.get(s.finish_reason, 0) + 1
            )

        total_time = time.time() - total_start
        avg_turns = (
            sum(s.turns_completed for s in states) / N if N > 0 else 0.0
        )
        print(
            f"[BatchRollout] Complete: {N} conversations,"
            f" {turns_completed} turns, {total_time:.1f}s total"
        )
        print(
            f"[BatchRollout]   Finish reasons:"
            f" {finish_counts.get('stop', 0)} stop,"
            f" {finish_counts.get('length', 0)} length,"
            f" {finish_counts.get('max_turns', 0)} max_turns"
        )
        print(f"[BatchRollout]   Avg turns/conv: {avg_turns:.1f}")

        return [s.messages for s in states]

    def _solver_retry_phase(self, states: list[ConversationState]) -> None:
        """Nudge + regenerate up to ``_MAX_FORCE_ANSWER_RETRIES`` times for
        conversations without ``<answer>``.

        Handles two cases after the main loop:

        1. **Tool/user-ending**: model used all ``max_turns`` for search and
           the conversation ends on a tool or user message.
        2. **No-answer assistant**: last message is assistant but lacks an
           ``<answer>`` tag.

        Each pending state gets one initial nudge. Up to
        ``_MAX_FORCE_ANSWER_RETRIES`` generation rounds follow; after each
        round, states that still lack ``<answer>`` are re-nudged and carried
        into the next round. Matches verl's sglang_rollout force-answer
        retry loop for eval parity.

        Args:
            states: All conversation states (already marked as finished).
        """
        _ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)
        if self.solver_format == "function_calls":
            _NUDGE_MSG = (
                "You have used all your search attempts. "
                "You must now provide your final answer. "
                "Call the answer function inside <function_calls> tags "
                "with your answer as the argument."
            )
        elif self.solver_format == "webexplorer":
            _NUDGE_MSG = (
                "\n--- Final Step Reached ---\n"
                "Now you reach the final step.\n"
                "You are forbidden to call any tools.\n"
                "You must offer your final answer now."
            )
        else:
            _NUDGE_MSG = (
                "You have used all your search attempts. "
                "You must now provide your final answer. "
                "Wrap it in <answer> and </answer> tags."
            )

        pending: list[ConversationState] = []
        for s in states:
            if not s.messages or s.turns_completed == 0:
                continue
            last_role = s.messages[-1]["role"]
            if last_role in ("tool", "user"):
                pending.append(s)
            elif last_role == "assistant":
                has_answer = _ANSWER_RE.search(s.final_response)
                if self.solver_format == "function_calls":
                    has_answer = has_answer or FC_ANSWER_RE.search(s.final_response)
                elif self.solver_format == "webexplorer":
                    has_answer = "<tool_call>" not in s.final_response
                if not has_answer:
                    pending.append(s)

        if not pending:
            return
        initial_count = len(pending)

        chat_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if self._tools_for_template is not None:
            chat_kwargs["tools"] = self._tools_for_template

        # Initial nudge
        for s in pending:
            s.messages.append({"role": "user", "content": _NUDGE_MSG})

        total_calls = 0
        # Same hard cap verl uses in sglang_rollout.py:1886 — skip the call
        # if there's no room for even one completion token. Includes the
        # sglang safety margin (server reserves ~6 tokens internally).
        hard_cap = (
            self.max_model_len - _SGLANG_SAFETY_MARGIN
            if self.max_model_len is not None
            else self.max_prompt_length
        )
        for attempt in range(self._MAX_FORCE_ANSWER_RETRIES):
            valid_states: list[ConversationState] = []
            valid_prompts: list[str] = []
            for s in pending:
                prompt = self.processing_class.apply_chat_template(
                    s.messages, **chat_kwargs,
                )
                token_ids = self.processing_class(
                    prompt, add_special_tokens=False,
                )["input_ids"]
                if (
                    len(token_ids) <= self.max_prompt_length
                    and len(token_ids) < hard_cap
                ):
                    valid_states.append(s)
                    valid_prompts.append(prompt)
                else:
                    # No room for any completion — mark LENGTH like verl does
                    s.finish_reason = "length"
                    s.final_response = (
                        s.messages[-1]["content"]
                        if s.messages else ""
                    )

            if not valid_prompts:
                break

            response = self._batch_solver_call(valid_prompts)
            total_calls += len(valid_prompts)
            choices = sorted(response.choices, key=lambda c: c.index)

            still_pending: list[ConversationState] = []
            for state, choice in zip(valid_states, choices):
                text = choice.text
                state.messages.append({"role": "assistant", "content": text})
                state.final_response = text
                state.turns_completed += 1
                has_answer = _ANSWER_RE.search(text)
                if self.solver_format == "function_calls":
                    has_answer = has_answer or FC_ANSWER_RE.search(text)
                elif self.solver_format == "webexplorer":
                    has_answer = "<tool_call>" not in text
                if not has_answer:
                    # Re-nudge for next attempt (matches verl's loop)
                    state.messages.append(
                        {"role": "user", "content": _NUDGE_MSG}
                    )
                    still_pending.append(state)

            pending = still_pending
            if not pending:
                break

        succeeded = initial_count - len(pending)
        print(
            f"[BatchRollout] Solver retry: {initial_count} no-answer, "
            f"{succeeded} now have <answer> "
            f"({total_calls} solver calls across up to "
            f"{self._MAX_FORCE_ANSWER_RETRIES} rounds)"
        )

    def _batch_solver_call(self, prompts: list[str]) -> Any:
        """Send all prompts in a single batch call to the solver server.

        Matches verl's ``_handle_engine_call`` (sglang_rollout.py:1942)
        by adaptively capping ``max_tokens`` so the longest prompt in
        the batch + completion fits inside the server's
        ``max_model_len``. Without this cap, sglang returns a 400 error
        when ``len(prompt_ids) + max_tokens > max_model_len``.

        Args:
            prompts: List of formatted prompt strings.

        Returns:
            OpenAI Completion response object with sorted choices.
        """
        n = len(prompts)
        t0 = time.time()
        effective_max_tokens = self.max_tokens
        if self.max_model_len is not None and prompts:
            longest_prompt_tokens = max(
                len(self.processing_class(
                    p, add_special_tokens=False,
                )["input_ids"])
                for p in prompts
            )
            effective_max_tokens = max(
                1,
                min(
                    self.max_tokens,
                    self.max_model_len
                    - longest_prompt_tokens
                    - _SGLANG_SAFETY_MARGIN,
                ),
            )
        create_kwargs: dict[str, Any] = dict(
            model=self.model_name,
            prompt=prompts,
            max_tokens=effective_max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if self.solver_stop:
            create_kwargs["stop"] = self.solver_stop
        response = self.solver_client.completions.create(**create_kwargs)

        # When a closing XML tag is used as a stop token, the server strips
        # it from the response text.  Re-append when the response contains
        # the opening tag without its closing counterpart.
        if self.solver_stop:
            _STOP_TAG_PAIRS = [
                ("</answer>", "<answer>"),
                ("</tool_call>", "<tool_call>"),
                ("</search>", "<search>"),
                ("</function_calls>", "<function_calls>"),
                ("</task>", "<task>"),
                ("</call_tool>", "<call_tool"),
                ("</call>", "<call_tool"),
                ("<|end_search_query|>", "<|begin_search_query|>"),
            ]
            for close_tag, open_tag in _STOP_TAG_PAIRS:
                if close_tag in self.solver_stop:
                    for choice in response.choices:
                        # Strip <think> blocks before checking for
                        # open/close tag pairs so that tags mentioned
                        # inside reasoning (e.g. "I'll use <answer>")
                        # don't trigger spurious close-tag re-appends.
                        stripped = re.sub(
                            r"<think>.*?</think>", "", choice.text,
                            flags=re.DOTALL,
                        )
                        stripped = re.sub(
                            r"<think>.*", "", stripped, flags=re.DOTALL,
                        )
                        if open_tag in stripped and close_tag not in stripped:
                            choice.text += close_tag

        elapsed = time.time() - t0
        per_prompt = (elapsed / n * 1000) if n > 0 else 0.0
        print(
            f"[BatchRollout]   Solver: {n} prompts,"
            f" {elapsed:.1f}s ({per_prompt:.1f}ms/prompt)"
        )
        return response

    def _parse_search_r1_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse ``<search>query</search>`` tags into tool call dicts.

        Used when ``solver_format == "search_r1"`` to extract search queries
        from the Search-R1 style response format.

        Args:
            text: Raw assistant response text.

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

    def _parse_function_calls_search(self, text: str) -> list[dict[str, Any]]:
        """Parse ``<function_calls>search(query="...")`` into tool call dicts.

        Used when ``solver_format == "function_calls"`` to extract search
        queries from the OLMo function-calling response format.

        Args:
            text: Raw assistant response text.

        Returns:
            list[dict[str, Any]]: Parsed tool call dicts with ``name`` and
                ``arguments`` keys, compatible with the existing pipeline.
        """
        results: list[dict[str, Any]] = []
        for match in FC_SEARCH_RE.finditer(text):
            query = match.group(1).strip()
            if query:
                results.append(
                    {"name": "search", "arguments": {"query_list": [query]}}
                )
        return results

    def _parse_call_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse ``<call_tool name="...">query</call_tool>`` into tool call dicts.

        Used when ``solver_format == "call_tool"`` (DR-Tulu format).
        All tool types (google_search, snippet_search, browse_webpage)
        map to a single search query.

        Args:
            text: Raw assistant response text.

        Returns:
            list[dict[str, Any]]: Parsed tool call dicts.
        """
        results: list[dict[str, Any]] = []
        for match in CALL_TOOL_RE.finditer(text):
            content = match.group(2).strip()
            if content:
                results.append(
                    {"name": "search", "arguments": {"query_list": [content]}}
                )
        return results

    def _parse_webthinker_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse ``<|begin_search_query|>query<|end_search_query|>`` into tool call dicts.

        Used when ``solver_format == "webthinker"`` (WebThinker format).

        Args:
            text: Raw assistant response text.

        Returns:
            list[dict[str, Any]]: Parsed tool call dicts.
        """
        results: list[dict[str, Any]] = []
        for match in WEBTHINKER_SEARCH_RE.finditer(text):
            query = match.group(1).strip()
            if query:
                results.append(
                    {"name": "search", "arguments": {"query_list": [query]}}
                )
        return results

    def _parse_webexplorer_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse ``<tool_call>JSON</tool_call>`` with WebExplorer's JSON schema.

        Handles two tool types:
        - ``search``: maps ``arguments.queries`` -> ``query_list``.
        - ``browse``: maps ``arguments.query`` -> single-element ``query_list``.

        Args:
            text: Raw assistant response text.

        Returns:
            list[dict[str, Any]]: Parsed tool call dicts.
        """
        results: list[dict[str, Any]] = []
        for match in TOOL_CALL_RE.finditer(text):
            raw = match.group(1).strip()
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(parsed, dict):
                continue
            name = parsed.get("name")
            args = parsed.get("arguments", {})
            if not isinstance(args, dict):
                continue
            if name == "search":
                queries = args.get("queries", [])
                if (
                    isinstance(queries, list)
                    and len(queries) > 0
                    and all(isinstance(q, str) for q in queries)
                ):
                    results.append(
                        {"name": "search", "arguments": {"query_list": queries}}
                    )
            elif name == "browse":
                query = args.get("query", "")
                if isinstance(query, str) and query.strip():
                    results.append(
                        {"name": "search", "arguments": {"query_list": [query.strip()]}}
                    )
        return results

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse tool calls from assistant text using FunctionCallParser or regex.

        Tries sglang's ``FunctionCallParser`` first (when available), then
        falls back to regex-based parsing. Only returns valid search tool
        calls with non-empty string query lists.

        When ``solver_format == "search_r1"``, delegates to
        ``_parse_search_r1_calls()`` instead.
        When ``solver_format == "function_calls"``, delegates to
        ``_parse_function_calls_search()`` instead.

        Args:
            text: Raw assistant response text.

        Returns:
            list[dict[str, Any]]: Parsed tool call dicts with ``name`` and
                ``arguments`` keys.
        """
        if self.solver_format == "search_r1":
            return self._parse_search_r1_calls(text)
        if self.solver_format in ("function_calls", "function_calls_xml"):
            return self._parse_function_calls_search(text)
        if self.solver_format == "call_tool":
            return self._parse_call_tool_calls(text)
        if self.solver_format == "webthinker":
            return self._parse_webthinker_calls(text)
        if self.solver_format == "webexplorer":
            return self._parse_webexplorer_calls(text)

        # Path 1: FunctionCallParser (model-agnostic)
        if self._function_call_parser is not None:
            try:
                if self._function_call_parser.has_tool_call(text):
                    _, tool_calls = self._function_call_parser.parse_non_stream(
                        text,
                    )
                    results: list[dict[str, Any]] = []
                    for tc in tool_calls:
                        try:
                            args = (
                                json.loads(tc.parameters)
                                if isinstance(tc.parameters, str)
                                else tc.parameters
                            )
                            if (
                                tc.name == "search"
                                and isinstance(args, dict)
                                and isinstance(args.get("query_list"), list)
                                and len(args["query_list"]) > 0
                                and all(
                                    isinstance(q, str)
                                    for q in args["query_list"]
                                )
                            ):
                                results.append(
                                    {"name": tc.name, "arguments": args}
                                )
                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue
                    if results:
                        return results
                    # FunctionCallParser detected tool calls but couldn't
                    # extract valid search calls; fall through to regex
            except Exception:
                pass  # Fall through to regex

        # Path 2: Regex fallback — try all known delimiter patterns
        results = []
        for pattern in self._fallback_patterns:
            for match in pattern.finditer(text):
                raw = match.group(1).strip()
                # Try JSON first (existing path)
                try:
                    parsed = json.loads(raw)
                    if (
                        isinstance(parsed, dict)
                        and parsed.get("name") == "search"
                        and isinstance(parsed.get("arguments"), dict)
                        and isinstance(
                            parsed["arguments"].get("query_list"), list
                        )
                        and len(parsed["arguments"]["query_list"]) > 0
                        and all(
                            isinstance(q, str)
                            for q in parsed["arguments"]["query_list"]
                        )
                    ):
                        results.append(parsed)
                        continue
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
                # Try Python-style function call (new path)
                parsed_py = _parse_python_search_call(raw)
                if parsed_py is not None:
                    results.append(parsed_py)
        return results

    def _batch_retrieve(
        self,
        states: list[ConversationState],
        retrieval_tasks: dict[int, list[str]],
    ) -> None:
        """Flatten all queries, make one retrieval call, distribute results.

        Args:
            states: All conversation states (indexed by ``state.index``).
            retrieval_tasks: Mapping from conversation index to query list.
        """
        # Build offset tracking
        all_queries: list[str] = []
        offsets: list[tuple[int, int, int]] = []  # (conv_index, offset, count)
        for conv_idx, queries in retrieval_tasks.items():
            offsets.append((conv_idx, len(all_queries), len(queries)))
            all_queries.extend(queries)

        total_queries = len(all_queries)
        t0 = time.time()
        api_response, error_msg = call_search_api(
            self.retrieval_url, all_queries, topk=self.retrieval_topk,
        )
        elapsed = time.time() - t0
        per_query = (elapsed / total_queries * 1000) if total_queries > 0 else 0.0
        print(
            f"[BatchRollout]   Retrieval: {total_queries} queries,"
            f" {elapsed:.1f}s ({per_query:.1f}ms/query)"
        )

        # --- Circuit breaker ---
        if error_msg:
            self._cb_consecutive += 1
            logger.warning(
                "[CircuitBreaker] Retrieval batch failed (%d consecutive): %s",
                self._cb_consecutive, error_msg[:200],
            )
            if self._cb_max > 0 and self._cb_consecutive >= self._cb_max:
                raise SearchServerDownError(
                    f"Retrieval server at {self.retrieval_url!r} failed "
                    f"{self._cb_consecutive} consecutive batches. "
                    f"Last error: {error_msg}"
                )
        else:
            self._cb_consecutive = 0

        # Distribute results back to conversations
        state_lookup = {s.index: s for s in states}
        _uses_info_tags = self.solver_format in ("search_r1", "function_calls", "function_calls_xml")
        _uses_tool_output_tags = self.solver_format == "call_tool"
        _uses_search_result_tags = self.solver_format == "webthinker"
        _uses_tool_response_tags_we = self.solver_format == "webexplorer"
        if api_response and "result" in api_response:
            raw_results = api_response["result"]
            for conv_idx, offset, count in offsets:
                conv_results = raw_results[offset : offset + count]
                tool_content = self._format_tool_response(conv_results)
                if _uses_tool_response_tags_we:
                    state_lookup[conv_idx].messages.append(
                        {"role": "user", "content": tool_content}
                    )
                elif _uses_search_result_tags:
                    state_lookup[conv_idx].messages.append(
                        {"role": "user", "content": tool_content}
                    )
                elif _uses_tool_output_tags:
                    state_lookup[conv_idx].messages.append(
                        {"role": "user", "content": tool_content}
                    )
                elif _uses_info_tags:
                    state_lookup[conv_idx].messages.append(
                        {"role": "user", "content": f"<information>{tool_content}</information>"}
                    )
                else:
                    state_lookup[conv_idx].messages.append(
                        {"role": "tool", "content": tool_content}
                    )
        else:
            # On error, still append an error tool message so conversation
            # can continue
            if _uses_tool_response_tags_we:
                error_content = f"<tool_response>\nSearch error: {error_msg or 'unknown'}\n</tool_response>"
                role = "user"
            elif _uses_search_result_tags:
                error_content = f"<|begin_search_result|>Search error: {error_msg or 'unknown'}<|end_search_result|>"
                role = "user"
            elif _uses_tool_output_tags:
                error_content = f"<tool_output>Search error: {error_msg or 'unknown'}</tool_output>"
                role = "user"
            elif _uses_info_tags:
                error_content = f"<information>Search error: {error_msg or 'unknown'}</information>"
                role = "user"
            else:
                error_content = json.dumps(
                    {"result": f"Search error: {error_msg or 'unknown'}"}
                )
                role = "tool"
            for conv_idx, _, _ in offsets:
                state_lookup[conv_idx].messages.append(
                    {"role": role, "content": error_content}
                )

    def _format_tool_response(
        self, results_per_query: list[list[dict[str, Any]]]
    ) -> str:
        """Format retrieval results for a conversation's tool response.

        When ``solver_format == "search_r1"``, returns the raw passage text
        (the ``<information>`` wrapping is applied in ``_batch_retrieve``).
        Otherwise returns a JSON string with ``{"result": "..."}`` format.

        Args:
            results_per_query: List of retrieval results, one per query.
                Each is a list of document dicts.

        Returns:
            str: Formatted passage text (plain for search_r1, JSON otherwise).
        """
        formatted_parts: list[str] = []
        for single_query_results in results_per_query:
            formatted_parts.append(_passages2string(single_query_results))
        combined = "\n---\n".join(formatted_parts)
        if self.max_tool_response_length > 0:
            unit = getattr(self, "tool_response_truncation_unit", "char")
            if unit == "token" and self.processing_class is not None:
                ids = self.processing_class.encode(combined, add_special_tokens=False)
                if len(ids) > self.max_tool_response_length:
                    combined = self.processing_class.decode(
                        ids[: self.max_tool_response_length], skip_special_tokens=False
                    ) + "...(truncated)"
            else:
                if len(combined) > self.max_tool_response_length:
                    combined = combined[: self.max_tool_response_length] + "...(truncated)"
        if self.solver_format == "webexplorer":
            from webexplorer.format_tool_response import format_webexplorer_tool_response
            return format_webexplorer_tool_response(
                results_per_query,
                max_length=self.max_tool_response_length,
                processing_class=self.processing_class,
                truncation_unit=getattr(self, "tool_response_truncation_unit", "char"),
            )
        if self.solver_format == "webthinker":
            from webthinker.format_tool_response import format_webthinker_tool_response
            return format_webthinker_tool_response(
                results_per_query,
                max_length=self.max_tool_response_length,
                processing_class=self.processing_class,
                truncation_unit=getattr(self, "tool_response_truncation_unit", "char"),
            )
        if self.solver_format == "call_tool":
            from drtulu.format_tool_response import format_drtulu_tool_response
            return format_drtulu_tool_response(
                results_per_query,
                max_length=self.max_tool_response_length,
                processing_class=self.processing_class,
                truncation_unit=getattr(self, "tool_response_truncation_unit", "char"),
            )
        if self.solver_format in ("search_r1", "function_calls", "function_calls_xml"):
            return combined
        return json.dumps({"result": combined})

    def grade_batch(
        self,
        prompts: list[str],
        grader_base_url: str,
        max_tokens: int = 16384,
        temperature: float = 0.0,
        grader_model_name: str | None = None,
        grader_messages: list[list[dict[str, Any]]] | None = None,
    ) -> list[str]:
        """Grade a batch of prompts using the grader server.

        Supports two modes based on the effective grader model name:

        - **Local server** (default): Uses ``/v1/completions`` with batched
          prompts against an sglang/vllm server at ``grader_base_url``.
        - **OpenAI API** (when model name starts with ``gpt-``, ``o1-``,
          ``o3-``, or ``o4-``): Uses ``chat.completions.create()`` with
          the ``grader_messages`` parameter. Requires ``OPENAI_API_KEY``
          in the environment.

        Chunks prompts into sub-batches when ``grader_batch_size > 0`` to
        avoid HTTP timeouts on very large prompt lists (local mode only).

        Args:
            prompts: List of formatted grader prompt strings (used for
                local server mode).
            grader_base_url: Base URL of the grader server
                (e.g. ``"http://127.0.0.1:8002"``). Ignored for OpenAI
                models.
            max_tokens: Maximum tokens for grader response.
            temperature: Sampling temperature for grader.
            grader_model_name: Optional model name override for the grader
                server. When ``None`` (default), uses ``self.model_name``.
            grader_messages: List of chat message lists (one per item),
                each a list of ``{"role": ..., "content": ...}`` dicts.
                Required when using an OpenAI model; ignored for local
                servers.

        Returns:
            list[str]: Grader response texts, one per prompt.

        Raises:
            ValueError: If an OpenAI model is used but ``grader_messages``
                is ``None``.
        """
        if not prompts:
            return []

        effective_model = grader_model_name if grader_model_name is not None else self.model_name

        use_openai_compatible = (
            self.grader_api_mode in ("openai", "openai_compatible")
            or (self.grader_api_mode == "auto" and _is_openai_model(effective_model))
        )
        if use_openai_compatible:
            return self._grade_batch_openai(
                effective_model, grader_messages, max_tokens, temperature,
            )

        grader_client = OpenAI(
            base_url=f"{grader_base_url}/v1",
            api_key="EMPTY",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        batch_size = self.grader_batch_size if self.grader_batch_size > 0 else len(prompts)
        all_texts: list[str] = []
        t0 = time.time()
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        avg_prompt_len = sum(len(p) for p in prompts) / len(prompts) if prompts else 0
        print(
            f"[BatchRollout] Grader starting: {len(prompts)} prompts, "
            f"{n_batches} chunks, avg_prompt_chars={avg_prompt_len:.0f}, "
            f"timeout={self.timeout}s, max_retries={self.max_retries}",
            flush=True,
        )
        for batch_i, start in enumerate(range(0, len(prompts), batch_size)):
            chunk = prompts[start:start + batch_size]
            t_chunk = time.time()
            print(
                f"[BatchRollout] Grader chunk {batch_i + 1}/{n_batches} "
                f"sending {len(chunk)} prompts...",
                flush=True,
            )
            response = grader_client.completions.create(
                model=effective_model,
                prompt=chunk,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            chunk_elapsed = time.time() - t_chunk
            choices = sorted(response.choices, key=lambda c: c.index)
            out_tokens = sum(c.logprobs is not None for c in choices)  # approx
            total_usage = getattr(response, "usage", None)
            usage_str = ""
            if total_usage:
                usage_str = (
                    f" input_tok={total_usage.prompt_tokens}"
                    f" output_tok={total_usage.completion_tokens}"
                )
            all_texts.extend(c.text for c in choices)
            print(
                f"[BatchRollout] Grader chunk {batch_i + 1}/{n_batches} done:"
                f" {len(chunk)} prompts, {chunk_elapsed:.1f}s"
                f" ({chunk_elapsed/len(chunk)*1000:.1f}ms/prompt){usage_str}",
                flush=True,
            )
        elapsed = time.time() - t0
        per_prompt = (elapsed / len(prompts) * 1000) if prompts else 0.0
        print(
            f"[BatchRollout] Grader total: {len(prompts)} prompts,"
            f" {elapsed:.1f}s ({per_prompt:.1f}ms/prompt)",
            flush=True,
        )
        return all_texts

    def _grade_batch_openai(
        self,
        model: str,
        grader_messages: list[list[dict[str, Any]]] | None,
        max_tokens: int,
        temperature: float,
    ) -> list[str]:
        """Grade via OpenAI chat completions API with concurrent requests.

        Uses ``AsyncOpenAI`` and ``asyncio.gather`` with a semaphore to
        limit concurrency to ``self.openai_concurrency`` parallel requests.

        Args:
            model: OpenAI model identifier (e.g. ``"gpt-5.2"``).
            grader_messages: List of chat message lists, one per item.
            max_tokens: Maximum tokens for grader response.
            temperature: Sampling temperature for grader.

        Returns:
            list[str]: Grader response texts, one per conversation.

        Raises:
            ValueError: If ``grader_messages`` is ``None``.
            ValueError: If ``OPENAI_API_KEY`` environment variable is not set.
            Exception: Propagates the last OpenAI exception when a single
                request exhausts ``_APP_MAX_RETRIES`` application-level
                retries. Callers relying on save-on-failure semantics
                must handle this.
        """
        if grader_messages is None:
            raise ValueError(
                f"grader_messages is required when using OpenAI model "
                f"'{model}', but got None."
            )

        is_remote = self.grader_api_mode == "openai_compatible"
        api_key_env = self.grader_api_key_env
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"{api_key_env} environment variable is required when "
                f"using OpenAI-compatible grader (model='{model}', "
                f"mode={self.grader_api_mode!r})."
            )

        # Remote calls use a shorter timeout + SDK-level retries. Local OpenAI
        # (``auto`` mode matching ``gpt-*``) keeps the historical behavior.
        effective_timeout = (
            self.grader_request_timeout
            if self.grader_request_timeout is not None
            else self.timeout
        )
        sdk_max_retries = 2 if is_remote else self.max_retries

        client_kwargs: dict[str, Any] = dict(
            api_key=api_key,
            timeout=effective_timeout,
            max_retries=sdk_max_retries,
        )
        if self.grader_api_base_url is not None:
            client_kwargs["base_url"] = self.grader_api_base_url
        client = AsyncOpenAI(**client_kwargs)

        # Build ``extra_body`` once per batch. Combines provider-pinning
        # (OpenRouter) and Anthropic extended-thinking when configured.
        extra_body: dict[str, Any] | None = None
        if self.grader_provider_order:
            provider_block: dict[str, Any] = {
                "order": list(self.grader_provider_order),
                "allow_fallbacks": self.grader_allow_fallbacks,
            }
            if self.grader_require_parameters:
                provider_block["require_parameters"] = True
            extra_body = {"provider": provider_block}
        if self.grader_thinking_budget is not None and self.grader_thinking_budget > 0:
            extra_body = extra_body or {}
            extra_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.grader_thinking_budget,
            }

        _sem_limit = self.openai_concurrency if self.openai_concurrency > 0 else len(grader_messages)
        semaphore = asyncio.Semaphore(_sem_limit)

        if self.grader_app_max_retries is not None:
            _APP_MAX_RETRIES = self.grader_app_max_retries
        else:
            _APP_MAX_RETRIES = 5 if is_remote else 3
        _RATE_LIMIT_MAX_RETRIES = max(_APP_MAX_RETRIES, 10)
        _APP_BASE_DELAY = 2.0  # seconds
        _RATE_LIMIT_MIN_DELAY = 30.0  # floor for TPM 429s

        # Shared rate-limit gate: when one request hits 429, all
        # in-flight requests pause before their next attempt.
        _rate_limit_until: float = 0.0  # monotonic timestamp
        _rl_lock = asyncio.Lock()

        # Typed-exception retry classification. Retry on transient
        # network / rate-limit / server errors; break early only on
        # clearly-terminal client-config errors. Keeps generic ``Exception``
        # retryable so flaky-provider wrapping errors don't short-circuit
        # the batch (regression guard for the old ``"invalid" in err_str``).
        _retryable_exc: tuple[type[BaseException], ...] = (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
        )
        _terminal_exc: tuple[type[BaseException], ...] = (
            AuthenticationError,
            PermissionDeniedError,
            NotFoundError,
        )
        _permanent_bad_request_markers = ("context_length", "max_tokens")

        async def _call(
            messages: list[dict[str, Any]],
        ) -> tuple[str, dict[str, Any] | None]:
            nonlocal _rate_limit_until
            # Pre-flight length check — skip oversized payloads instead
            # of burning 3 retries on a 400 we can predict. Uses the
            # soft-skip path (no stub, no resume retry).
            if self.grader_max_input_chars > 0:
                total_chars = sum(
                    len(m.get("content") or "") for m in messages
                )
                if total_chars > self.grader_max_input_chars:
                    return "", {
                        "model": model,
                        "skipped": (
                            f"input {total_chars} chars > "
                            f"grader_max_input_chars={self.grader_max_input_chars}"
                        ),
                    }
            async with semaphore:
                last_err: Exception | None = None
                _excluded_providers: set[str] = set()
                _effective_max = _APP_MAX_RETRIES
                for attempt in range(_RATE_LIMIT_MAX_RETRIES):
                    if attempt >= _effective_max:
                        break
                    # Respect shared rate-limit gate
                    now = asyncio.get_event_loop().time()
                    if _rate_limit_until > now:
                        await asyncio.sleep(_rate_limit_until - now)
                    try:
                        create_kwargs: dict[str, Any] = dict(
                            model=model,
                            messages=messages,
                            max_completion_tokens=max_tokens,
                            temperature=temperature,
                        )
                        if self.grader_reasoning_effort is not None:
                            create_kwargs["reasoning_effort"] = (
                                self.grader_reasoning_effort
                            )
                        if extra_body is not None:
                            if _excluded_providers and self.grader_provider_order:
                                remaining = [
                                    p for p in self.grader_provider_order
                                    if p not in _excluded_providers
                                ]
                                retry_body: dict[str, Any] = {}
                                if remaining:
                                    retry_body["provider"] = {
                                        "order": remaining,
                                        "allow_fallbacks": True,
                                    }
                                if "thinking" in extra_body:
                                    retry_body["thinking"] = extra_body["thinking"]
                                if retry_body:
                                    create_kwargs["extra_body"] = retry_body
                            else:
                                create_kwargs["extra_body"] = extra_body
                        response = await client.chat.completions.create(
                            **create_kwargs,
                        )
                        if not response.choices:
                            _resp_info = ""
                            try:
                                _resp_info = repr(response.model_dump())
                            except Exception:
                                _resp_info = repr(response)
                            raise RuntimeError(
                                f"Provider returned empty choices "
                                f"(choices={response.choices!r}): {_resp_info}"
                            )
                        choice = response.choices[0]
                        content = choice.message.content or ""
                        usage = _extract_usage_dict(response, model)
                        # Record the actual provider that served the request
                        # (OpenRouter top-level field). Essential for auditing
                        # Alibaba-vs-fallback provider mix post-run.
                        provider = getattr(response, "provider", None)
                        if provider is None:
                            model_extra = getattr(response, "model_extra", None) or {}
                            provider = model_extra.get("provider")
                        if provider is not None:
                            usage = {**(usage or {"model": model}), "provider": provider}
                        # Empty content with ``finish_reason="length"`` means
                        # the model burned all ``max_completion_tokens`` on
                        # hidden reasoning and produced no visible reply.
                        # Treat it the same as the explicit ``max_tokens``
                        # 400 error so the downstream stub/resume path
                        # picks it up instead of silently saving an empty
                        # judge entry.
                        finish_reason = getattr(choice, "finish_reason", None)
                        if not content and finish_reason == "length":
                            # Empty content + finish_reason=length means
                            # the model burned all ``max_completion_tokens``
                            # on hidden reasoning. Don't retry (same ceiling
                            # will hit again) and don't flag as ``error``
                            # (which would trigger stub/retry-on-resume);
                            # mark ``skipped`` so downstream treats it as
                            # an unscored call without bookkeeping churn.
                            reason = (
                                f"length exhaustion "
                                f"(reasoning_tokens={usage.get('reasoning_tokens') if usage else 'n/a'}, "
                                f"max_completion_tokens={max_tokens})"
                            )
                            skip_usage = dict(usage) if usage else {"model": model}
                            skip_usage["skipped"] = reason
                            return "", skip_usage
                        return content, usage
                    except Exception as e:
                        last_err = e
                        # Terminal: auth/permission/not-found never retry.
                        if isinstance(e, _terminal_exc):
                            break
                        # Permanent bad-request: oversize / unsupported params.
                        # Narrowly scoped markers — generic BadRequestError from
                        # a flaky provider is still retryable (e.g. OpenRouter's
                        # "provider returned invalid response").
                        if isinstance(e, BadRequestError):
                            err_str = str(e).lower()
                            if any(m in err_str for m in _permanent_bad_request_markers):
                                break
                        # On rate-limit, exclude the offending provider
                        # so retries route to a different one.
                        if isinstance(e, RateLimitError):
                            _effective_max = _RATE_LIMIT_MAX_RETRIES
                            import re as _re
                            _prov_match = _re.search(
                                r"'provider_name':\s*'([^']+)'", str(e),
                            )
                            if _prov_match:
                                _excluded_providers.add(_prov_match.group(1))
                            else:
                                _excluded_providers.update(
                                    self.grader_provider_order or [],
                                )
                            # Parse "retry after Xms" hint from 429 body
                            _retry_ms_match = _re.search(
                                r"try again in (\d+(?:\.\d+)?)\s*ms", str(e), _re.IGNORECASE,
                            )
                            _retry_s_match = _re.search(
                                r"try again in (\d+(?:\.\d+)?)\s*s(?:ec)?", str(e), _re.IGNORECASE,
                            )
                            if _retry_ms_match:
                                _hint_delay = float(_retry_ms_match.group(1)) / 1000.0
                            elif _retry_s_match:
                                _hint_delay = float(_retry_s_match.group(1))
                            else:
                                _hint_delay = 0.0
                            import random
                            _rl_delay = max(_RATE_LIMIT_MIN_DELAY, _hint_delay) + random.uniform(0, 5)
                            # Set shared gate so other in-flight tasks also pause
                            async with _rl_lock:
                                _rl_target = asyncio.get_event_loop().time() + _rl_delay
                                if _rl_target > _rate_limit_until:
                                    _rate_limit_until = _rl_target
                            logger.warning(
                                "Rate limit hit (attempt %d/%d), "
                                "backing off %.1fs (hint=%.1fs): %s",
                                attempt + 1, _effective_max, _rl_delay,
                                _hint_delay, e,
                            )
                            await asyncio.sleep(_rl_delay)
                            continue
                        # All other errors (retryable typed, or generic
                        # Exception from wrapping SDKs) get retried with
                        # exponential backoff + jitter.
                        if attempt < _effective_max - 1:
                            import random
                            delay = _APP_BASE_DELAY * (2 ** attempt)
                            jitter = random.uniform(0, delay * 0.5)
                            logger.warning(
                                "OpenAI-compatible request failed "
                                "(attempt %d/%d), retrying in %.1fs: %s",
                                attempt + 1, _effective_max,
                                delay + jitter, e,
                            )
                            await asyncio.sleep(delay + jitter)
                logger.error(
                    "OpenAI grader request failed after %d attempts: %s",
                    _effective_max, last_err,
                )
                err_msg = (
                    str(last_err) if last_err is not None
                    else f"failed after {_effective_max} attempts"
                )
                return "", {"model": model, "error": err_msg}

        concurrency = self.openai_concurrency
        n_items = len(grader_messages)

        _progress_log = "/tmp/grader_progress.log"
        def _log_progress(msg: str) -> None:
            with open(_progress_log, "a") as _f:
                _f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                _f.flush()

        _log_progress(
            f"OpenAI grader: {n_items} items, concurrency={concurrency}"
        )
        print(
            f"[BatchRollout] OpenAI grader: {n_items} items,"
            f" concurrency={concurrency}",
            flush=True,
        )

        t0 = time.perf_counter()
        completed = 0

        _progress_interval = max(1, n_items // 20)  # ~5% increments
        _errors_so_far = 0

        async def _call_with_progress(
            messages: list[dict[str, Any]],
        ) -> tuple[str, dict[str, Any] | None]:
            nonlocal completed, _errors_so_far
            result = await _call(messages)
            completed += 1
            if isinstance(result[1], dict) and "error" in result[1]:
                _errors_so_far += 1
            if completed % _progress_interval == 0 or completed == n_items:
                elapsed = time.perf_counter() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n_items - completed) / rate if rate > 0 else 0
                err_part = f", errors={_errors_so_far}" if _errors_so_far else ""
                msg = (
                    f"progress: {completed}/{n_items} "
                    f"({elapsed:.0f}s, ~{eta:.0f}s left{err_part})"
                )
                _log_progress(msg)
                print(
                    f"[BatchRollout] OpenAI grader progress: "
                    f"{completed}/{n_items} ({elapsed:.0f}s elapsed, "
                    f"~{eta:.0f}s remaining{err_part})",
                    flush=True,
                )
            return result

        async def _run() -> list[tuple[str, dict[str, Any] | None]]:
            # ``return_exceptions=True`` ensures one unexpected failure
            # doesn't cancel the rest. ``_call`` already returns an error
            # sentinel for known OpenAI failures; this guards against
            # bugs / cancellation propagation.
            raw = await asyncio.gather(
                *[_call_with_progress(m) for m in grader_messages],
                return_exceptions=True,
            )
            results: list[tuple[str, dict[str, Any] | None]] = []
            for r in raw:
                if isinstance(r, BaseException):
                    results.append(("", {"model": model, "error": str(r)}))
                else:
                    results.append(r)
            return results

        try:
            all_results = asyncio.run(_run())
        except RuntimeError:
            # Already inside an event loop (e.g. Jupyter) – use nest_asyncio
            # or fall back to running in a new thread.
            loop = asyncio.new_event_loop()
            try:
                all_results = loop.run_until_complete(_run())
            finally:
                loop.close()

        all_texts = [r[0] for r in all_results]
        # Append usage entries in call-order to the cumulative log. Callers
        # (eval_standalone + benchmark evaluators) read the tail of this
        # log to attribute tokens/cost per item.
        for _, usage in all_results:
            self.grader_usage_log.append(
                usage if usage is not None else {"model": model}
            )

        elapsed = time.perf_counter() - t0
        per_item = (elapsed / n_items * 1000) if n_items else 0.0
        _log_progress(
            f"DONE: {n_items} items, {elapsed:.1f}s ({per_item:.1f}ms/item)"
        )
        print(
            f"[BatchRollout] OpenAI grader total: {n_items} items,"
            f" {elapsed:.1f}s ({per_item:.1f}ms/item)",
            flush=True,
        )
        # Batch-level provider/error summary and fail-fast check.
        provider_counts: Counter[str] = Counter()
        error_count = 0
        first_error: str | None = None
        for _, u in all_results:
            if isinstance(u, dict):
                if "error" in u:
                    error_count += 1
                    if first_error is None:
                        first_error = u["error"]
                else:
                    provider_counts[u.get("provider") or "unknown"] += 1
        if is_remote or error_count > 0:
            print(
                f"[BatchRollout] grader providers: {dict(provider_counts)}, "
                f"errors: {error_count}/{n_items}",
                flush=True,
            )

        error_ratio = error_count / n_items if n_items > 0 else 0.0
        if error_ratio > 0.5:
            raise RuntimeError(
                f"[BatchRollout] Grader batch failed: {error_count}/{n_items} "
                f"calls errored ({error_ratio:.0%}). Stopping to avoid "
                f"training on bad rewards. First error: {first_error}"
            )

        return all_texts
