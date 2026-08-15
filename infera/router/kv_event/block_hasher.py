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
from infera.server import metrics

logger = logging.getLogger(__name__)


class BlockHasher:
    """Tokenize requests with the *worker-matching* tokenizer and chain block hashes.

    The router's token ids must match the serving engine's byte-for-byte, or
    every block hash diverges and cache lookups always miss. Engines differ:
    SGLang and vLLM can pick different tokenizer implementations for the same
    model (e.g. DeepSeek loads as a slow ``LlamaTokenizer`` under SGLang but a
    fast tokenizer under transformers, and the two produce different ids). So
    we load the tokenizer the way the engine does, keyed by ``(engine, source)``.
    """

    def __init__(self, tokenizer_path: str | None = None) -> None:
        # Operator-supplied local path, used in preference to the advertised
        # model id so the router reads the exact files the workers use. The
        # loader is still chosen by engine (the files alone don't decide
        # fast-vs-slow / special-token config).
        self._tokenizer_path = tokenizer_path
        self._tokenizers: dict[tuple[Any, str], Any] = {}
        # Keys we have already failed to load. Without this every request
        # retries the load and re-logs the failure -- one broken deployment
        # produced 240 identical WARNINGs in a 120-request run, which buries
        # the one line that matters rather than surfacing it.
        self._failed: set[tuple[Any, str]] = set()

    def hash_for(
        self, body: dict, *, block_size: int, engine: EngineType | None = None
    ) -> list[int]:
        model_id = body.get("model")
        if not model_id or block_size <= 0:
            return []

        tokenizer = self._get_tokenizer(model_id, engine)
        if tokenizer is None:
            # Every request lands here once the tokenizer fails to load, and
            # each one routes with no cache information at all -- kv-aware is
            # silently doing least-loaded. Nothing else says so: the server
            # starts, /health is green, requests succeed. Count it so the
            # degradation is visible on a dashboard rather than only in a
            # WARNING somebody has to already suspect to go looking for.
            metrics.cache_locality_skipped_total.labels(reason="no_tokenizer").inc()
            return []

        # Tokenisation failure (e.g. apply_chat_template on a base model
        # without a chat template, or encode on an unexpected body type)
        # must not 500 the request -- degrade to "no cache info" and let the
        # cost function fall back to load-only routing.
        try:
            if messages := body.get("messages"):
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            elif (prompt := body.get("prompt")) is not None:
                text = prompt
            else:
                return []
            # The chat template / prompt already carries any leading special
            # token as text, so don't let the tokenizer add another (matches
            # how the engines tokenize an already-templated string).
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception as exc:
            logger.warning("kv-aware: tokenisation failed for model=%s: %s", model_id, exc)
            return []

        return hash_request(token_ids, block_size)

    def _get_tokenizer(self, model_id: str, engine: EngineType | None) -> Any | None:
        source = self._tokenizer_path or model_id
        if self._tokenizer_path and self._tokenizer_path.endswith(".json"):
            source = self._tokenizer_path.rsplit("/", 1)[0] if "/" in self._tokenizer_path else "."
        key = (engine, source)
        if key in self._tokenizers:
            return self._tokenizers[key]
        if key in self._failed:
            return None
        tok = self._load(source, engine)
        if tok is None:
            self._failed.add(key)
            # Logged once per (engine, source), at ERROR: this is not a
            # per-request hiccup, it disables cache-aware routing outright for
            # the life of the process. The individual loader warnings above say
            # what failed; this says what it costs.
            logger.error(
                "kv-aware DEGRADED: no tokenizer for %s -- every request will "
                "route with zero cache information, i.e. least-loaded, and "
                "--kv-overlap-weight will have no effect. Point "
                "--router-tokenizer-path at the same files the workers load "
                "(the recipes mount the model volume into the server pod and "
                "pass a local path, not a hub id).",
                source,
            )
            return None
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
