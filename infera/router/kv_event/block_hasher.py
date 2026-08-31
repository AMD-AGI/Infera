###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import logging
from typing import Any

from infera.common.worker_pool import EngineType
from infera.router.kv_event.hasher import hash_request

logger = logging.getLogger(__name__)

# "thinking" drifts one token against the engine when the last assistant turn
# carries a tool_call, so the router renders chat prompts in "chat" mode.
DSV4_THINKING_MODE = "chat"


class BlockHasher:
    """Tokenize requests with the *worker-matching* tokenizer and chain block hashes.

    The router's token ids must match the serving engine's byte-for-byte, or
    every block hash diverges and cache lookups always miss. Engines differ:
    SGLang and vLLM can pick different tokenizer implementations for the same
    model (e.g. DeepSeek loads as a slow ``LlamaTokenizer`` under SGLang but a
    fast tokenizer under transformers, and the two produce different ids). So
    we load the tokenizer the way the engine does, keyed by ``(engine, source)``.
    """

    def __init__(
        self, tokenizer_path: str | None = None, dsv4_thinking_mode: str = DSV4_THINKING_MODE
    ) -> None:
        # Operator-supplied local path, used in preference to the advertised
        # model id so the router reads the exact files the workers use. The
        # loader is still chosen by engine (the files alone don't decide
        # fast-vs-slow / special-token config).
        self._tokenizer_path = tokenizer_path
        self._dsv4_thinking_mode = dsv4_thinking_mode
        self._tokenizers: dict[tuple[Any, str], Any] = {}
        self._hf_configs: dict[str, Any] = {}

    def hash_for(
        self, body: dict, *, block_size: int, engine: EngineType | None = None
    ) -> list[int]:
        model_id = body.get("model")
        if not model_id or block_size <= 0:
            return []

        tokenizer = self._get_tokenizer(model_id, engine)
        if tokenizer is None:
            return []

        # Tokenisation failure (e.g. apply_chat_template on a base model
        # without a chat template, or encode on an unexpected body type)
        # must not 500 the request -- degrade to "no cache info" and let the
        # cost function fall back to load-only routing.
        encoder = "chat-template"
        try:
            token_ids = None
            if messages := body.get("messages"):
                token_ids = self._encode_via_sglang_dsv4(tokenizer, model_id, messages, body)
                if token_ids is not None:
                    encoder = "sglang-dsv4"
                else:
                    template_kwargs = self._template_kwargs(body)
                    if template_kwargs is None:
                        return []
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        **template_kwargs,
                    )
            elif (prompt := body.get("prompt")) is not None:
                encoder = "prompt"
                text = prompt
            else:
                return []
            if token_ids is None:
                # The chat template / prompt already carries any leading special
                # token as text, so don't let the tokenizer add another (matches
                # how the engines tokenize an already-templated string).
                token_ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception as exc:
            logger.warning("kv-aware: tokenisation failed for model=%s: %s", model_id, exc)
            return []

        # A prefix that disagrees with the engine's is silent: no error, just a
        # permanent 0% hit rate. TODO: compare this count against the engine's
        # reported usage.prompt_tokens and warn (rate-limited) on a mismatch,
        # once the response usage is plumbed back to the router.
        logger.debug(
            "kv-aware: model=%s encoder=%s prompt_tokens=%d", model_id, encoder, len(token_ids)
        )
        return hash_request(token_ids, block_size)

    def _template_kwargs(self, body: dict) -> dict | None:
        """Everything besides `messages` that the engine puts in scope.

        None means "this request cannot be reproduced" -- the caller must route
        it on load rather than render an approximation.

        `serving_chat` calls::

            apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                tools=tools, return_dict=False, **extra_template_kwargs)

        where `extra_template_kwargs` is `reasoning_effort` (when the request set
        one) updated with `chat_template_kwargs`. Passing only `messages` is not a
        small omission: a template reads whatever names it likes off the context,
        and the ones it cannot see are simply undefined -- no error, no warning,
        just a different prompt. GLM-5.3's opens with

            {%- set effective_reasoning_effort = reasoning_effort if reasoning_effort
               is defined and reasoning_effort in ['low','high'] else 'high' -%}
            <|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}

        so a `reasoning_effort: "low"` request diverges from the engine in the
        FIRST block of the prompt, and every block after it chains off that hash.
        Tools land there too, ahead of the conversation. That is a permanent 0%
        hit rate on a fleet that reports itself healthy.

        Not modelled: the engine's `--default-chat-template-kwargs`, which
        `serving_chat` merges server-side. The router is never told that flag, so
        setting it re-breaks kv-aware -- see the router's own hasher notes.
        """
        kwargs: dict = {}
        tools, ok = self._chat_tools(body)
        if not ok:
            return None
        if tools is not None:
            kwargs["tools"] = tools
        effort = body.get("reasoning_effort")
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        extra = body.get("chat_template_kwargs")
        if isinstance(extra, dict):
            kwargs.update(extra)
        return kwargs

    def _chat_tools(self, body: dict) -> tuple[list[dict] | None, bool]:
        """`serving_chat`'s `tools` argument, plus whether we could build it.

        Returns `(tools, True)` on success -- `tools` is None exactly when the
        engine would also pass none. Returns `(None, False)` when the request
        carries tools we cannot reproduce: for the same reason as the dsv4 path,
        a guessed tool shape shifts the prefix with no error, and rendering
        without the tools is just as wrong, so the caller routes on load.

        Mirrors `serving_chat`'s two gates: `tool_choice == "none"` suppresses
        tools entirely, and a `{"type": "function", "function": {"name": ...}}`
        choice narrows the list to that one function.
        """
        tools = body.get("tools")
        if not tools:
            return None, True
        if body.get("tool_choice") == "none":
            return None, True
        normalised = self._normalise_tools(tools)
        if normalised is None:
            return None, False
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            wanted = (choice.get("function") or {}).get("name")
            if wanted:
                normalised = [
                    t for t in normalised if (t.get("function") or {}).get("name") == wanted
                ]
        return normalised, True

    def _encode_via_sglang_dsv4(
        self, tokenizer: Any, model_id: str, messages: list, body: dict
    ) -> list[int] | None:
        """Render chat messages with sglang's native DeepSeek-V4 encoder.

        Models on that encoder ship no chat template, so ``apply_chat_template``
        raises and kv-aware routing degrades to load-only for every chat request.
        Returns ``None`` whenever this isn't a dsv4 model or anything about the
        engine's rendering can't be reproduced exactly, leaving the caller on the
        chat-template path.
        """
        try:
            from sglang.srt.entrypoints.openai.chat_encoding import resolve_chat_encoding_spec
            from sglang.srt.entrypoints.openai.encoding_dsv4 import encode_messages
        except Exception:  # sglang not installed, or too old for these encoders
            return None

        hf_config = self._get_hf_config(model_id)
        if hf_config is None:
            return None
        try:
            spec = resolve_chat_encoding_spec(hf_config=hf_config, tokenizer=tokenizer)
        except Exception as exc:  # e.g. a config without .architectures
            logger.debug("kv-aware: chat encoding spec undecidable for %s: %s", model_id, exc)
            return None
        # Strictly dsv4: the other specs (dsv32 / kimi_k3 / inkling) need their
        # own encoders, so they must stay on the apply_chat_template path.
        if spec != "dsv4":
            return None

        rendered = [dict(m) for m in messages]
        if rendered[0].get("role") != "system":
            # The engine prepends an empty system message and hangs the tools off
            # it (serving_chat); tools are never passed to the encoder directly.
            rendered.insert(0, {"role": "system", "content": ""})
        if tools := body.get("tools"):
            normalised = self._normalise_tools(tools)
            if normalised is None:
                return None
            rendered[0]["tools"] = normalised

        prompt = encode_messages(rendered, thinking_mode=self._dsv4_thinking_mode)
        # The encoder emits its own BOS, and the engine tokenizes the result with
        # a plain encode(); match it rather than the add_special_tokens=False above.
        return tokenizer.encode(prompt)

    @staticmethod
    def _normalise_tools(tools: list) -> list[dict] | None:
        """The engine renders tools from a full pydantic ``Tool.model_dump()``,
        which materialises defaults the client never sent (``strict``,
        ``defer_loading``); hashing the raw request dicts shifts the prefix by a
        few tokens with no error at all, so skip dsv4 rather than guess."""
        try:
            from sglang.srt.entrypoints.openai.protocol import Tool
        except Exception:
            return None
        try:
            return [Tool(**tool).model_dump() for tool in tools]
        except Exception as exc:
            logger.debug("kv-aware: tools not normalisable for the dsv4 encoder: %s", exc)
            return None

    def _get_hf_config(self, model_id: str) -> Any | None:
        source = self._source(model_id)
        if source in self._hf_configs:
            return self._hf_configs[source]
        cfg = self._load_hf_config(source)
        # Cached even when None: a source without a readable config.json won't
        # grow one, and this runs on the request path.
        self._hf_configs[source] = cfg
        return cfg

    @staticmethod
    def _load_hf_config(source: str) -> Any | None:
        try:
            from transformers import AutoConfig
        except Exception:
            return None
        try:
            return AutoConfig.from_pretrained(source, trust_remote_code=True)
        except Exception as exc:
            logger.debug("kv-aware: no hf config for %s: %s", source, exc)
            return None

    def _source(self, model_id: str) -> str:
        source = self._tokenizer_path or model_id
        if self._tokenizer_path and self._tokenizer_path.endswith(".json"):
            source = self._tokenizer_path.rsplit("/", 1)[0] if "/" in self._tokenizer_path else "."
        return source

    def _get_tokenizer(self, model_id: str, engine: EngineType | None) -> Any | None:
        source = self._source(model_id)
        key = (engine, source)
        if key in self._tokenizers:
            return self._tokenizers[key]
        tok = self._load(source, engine)
        if tok is not None:
            self._tokenizers[key] = tok
        return tok

    def _load(self, source: str, engine: EngineType | None) -> Any | None:
        """Load ``source`` the way the serving engine does, falling back to a
        plain ``AutoTokenizer`` if the engine's own loader isn't importable
        (e.g. a router-only host without that engine installed)."""
        if engine == EngineType.SGLANG:
            return self._load_via_sglang(source) or self._load_via_auto(source)
        if engine == EngineType.VLLM:
            return self._load_via_vllm(source) or self._load_via_auto(source)
        return self._load_via_auto(source)

    @staticmethod
    def _load_via_sglang(source: str) -> Any | None:
        try:
            from sglang.srt.utils.hf_transformers_utils import get_tokenizer
        except Exception:  # sglang not installed on this host
            return None
        try:
            tok = get_tokenizer(source)
        except Exception as exc:
            logger.warning("kv-aware: sglang.get_tokenizer(%s) failed: %s", source, exc)
            return None
        logger.info("kv-aware: loaded tokenizer for %s via sglang", source)
        return tok

    @staticmethod
    def _load_via_vllm(source: str) -> Any | None:
        # get_tokenizer moved from vllm.transformers_utils.tokenizer to
        # vllm.tokenizers; try the new location first, fall back to the old.
        get_tokenizer = None
        for module in ("vllm.tokenizers", "vllm.transformers_utils.tokenizer"):
            try:
                get_tokenizer = __import__(module, fromlist=["get_tokenizer"]).get_tokenizer
                break
            except Exception:  # not this location, or vllm not installed
                continue
        if get_tokenizer is None:
            return None
        try:
            tok = get_tokenizer(source, trust_remote_code=True)
        except Exception as exc:
            logger.warning("kv-aware: vllm.get_tokenizer(%s) failed: %s", source, exc)
            return None
        logger.info("kv-aware: loaded tokenizer for %s via vllm", source)
        return tok

    @staticmethod
    def _load_via_auto(source: str) -> Any | None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            logger.warning("transformers not installed; tokenization disabled: %s", exc)
            return None
        try:
            tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        except Exception as exc:
            logger.warning("kv-aware: AutoTokenizer.from_pretrained(%s) failed: %s", source, exc)
            return None
        logger.info("kv-aware: loaded tokenizer for %s via AutoTokenizer", source)
        return tok
