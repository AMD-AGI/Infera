###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/router/kv_event/block_hasher.py.

Two concerns:
  - hash_for: how a request body (chat or completion) is rendered to tokens
    and chained into block hashes. Tested with stubbed tokenizers.
  - engine-aware loading: the router must tokenize the way the serving engine
    does (SGLang vs vLLM can pick different tokenizers for the same model), so
    the loader is chosen by EngineType and cached per (engine, source).
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

from infera.common.worker_pool import EngineType
from infera.router.kv_event.block_hasher import BlockHasher
from infera.router.kv_event.hasher import ROUTER_SEED, hash_chunk, hash_request


class _StubTokenizer:
    chat_template: Any = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)


class _BrokenTokenizer:
    def encode(self, *_, **__) -> list[int]:
        raise RuntimeError("boom")

    def apply_chat_template(self, *_, **__) -> str:
        raise RuntimeError("no chat template for this base model")


# ---- hash_for: request -> block hashes ------------------------------------


def test_hash_for_returns_empty_on_missing_model():
    assert BlockHasher().hash_for({}, block_size=4) == []


def test_hash_for_returns_empty_on_zero_block_size():
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _StubTokenizer()
    assert hasher.hash_for({"model": "m", "prompt": "abcd"}, block_size=0) == []


def test_hash_for_returns_empty_when_no_tokenizer_loadable():
    hasher = BlockHasher()
    with patch.object(BlockHasher, "_load_via_auto", staticmethod(lambda _s: None)):
        assert hasher.hash_for({"model": "unknown", "prompt": "abcd"}, block_size=4) == []


def test_hash_for_uses_prompt_field_for_completions():
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _StubTokenizer()
    out = hasher.hash_for({"model": "m", "prompt": "abcd"}, block_size=4)
    assert out == [hash_chunk(ROUTER_SEED, [97, 98, 99, 100])]


def test_hash_for_applies_chat_template_for_chat_completions():
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _StubTokenizer()
    out = hasher.hash_for(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, block_size=4
    )
    assert out == hash_request([ord(c) for c in "<user>hi</user><assistant>"], 4)


def test_hash_for_prefers_messages_over_prompt():
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _StubTokenizer()
    out = hasher.hash_for(
        {"model": "m", "messages": [{"role": "user", "content": "ok"}], "prompt": "ignored"},
        block_size=4,
    )
    assert out == hash_request([ord(c) for c in "<user>ok</user><assistant>"], 4)


def test_hash_for_degrades_to_empty_on_tokenizer_exception(caplog):
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _BrokenTokenizer()
    with caplog.at_level(logging.WARNING, logger="infera.router.kv_event.block_hasher"):
        out = hasher.hash_for({"model": "m", "prompt": "abc"}, block_size=4)
    assert out == []
    assert any("tokenisation failed" in r.message for r in caplog.records)


def test_hash_for_returns_empty_when_body_has_neither_prompt_nor_messages():
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _StubTokenizer()
    assert hasher.hash_for({"model": "m"}, block_size=4) == []


def test_hash_for_stable_across_invocations():
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _StubTokenizer()
    body = {"model": "m", "prompt": "abcdefgh"}
    assert hasher.hash_for(body, block_size=4) == hasher.hash_for(body, block_size=4)


# ---- engine-aware loading -------------------------------------------------


def test_sglang_engine_uses_sglang_loader():
    sentinel = _StubTokenizer()
    hasher = BlockHasher()
    with (
        patch.object(BlockHasher, "_load_via_sglang", staticmethod(lambda _s: sentinel)),
        patch.object(
            BlockHasher,
            "_load_via_vllm",
            staticmethod(
                lambda _s: (_ for _ in ()).throw(
                    AssertionError("vllm loader must not run for SGLANG")
                )
            ),
        ),
    ):
        assert hasher._get_tokenizer("m", EngineType.SGLANG) is sentinel


def test_vllm_engine_uses_vllm_loader():
    sentinel = _StubTokenizer()
    hasher = BlockHasher()
    with (
        patch.object(BlockHasher, "_load_via_vllm", staticmethod(lambda _s: sentinel)),
        patch.object(
            BlockHasher,
            "_load_via_sglang",
            staticmethod(
                lambda _s: (_ for _ in ()).throw(
                    AssertionError("sglang loader must not run for VLLM")
                )
            ),
        ),
    ):
        assert hasher._get_tokenizer("m", EngineType.VLLM) is sentinel


def test_engine_loader_falls_back_to_auto():
    """SGLang/vLLM loader unavailable (returns None) -> AutoTokenizer."""
    sentinel = _StubTokenizer()
    hasher = BlockHasher()
    with (
        patch.object(BlockHasher, "_load_via_sglang", staticmethod(lambda _s: None)),
        patch.object(BlockHasher, "_load_via_auto", staticmethod(lambda _s: sentinel)),
    ):
        assert hasher._get_tokenizer("m", EngineType.SGLANG) is sentinel


def test_unknown_engine_uses_auto():
    sentinel = _StubTokenizer()
    hasher = BlockHasher()
    with patch.object(BlockHasher, "_load_via_auto", staticmethod(lambda _s: sentinel)):
        assert hasher._get_tokenizer("m", None) is sentinel


def test_tokenizer_cached_per_engine_and_source():
    hasher = BlockHasher()
    calls = {"n": 0}

    def fake_auto(_s):
        calls["n"] += 1
        return _StubTokenizer()

    with patch.object(BlockHasher, "_load_via_auto", staticmethod(fake_auto)):
        hasher._get_tokenizer("m", None)
        hasher._get_tokenizer("m", None)
    assert calls["n"] == 1  # cached on second call


def test_explicit_path_overrides_model_id_as_source():
    """--router-tokenizer-path becomes the load source regardless of body model."""
    seen = {}

    def fake_auto(source):
        seen["source"] = source
        return _StubTokenizer()

    hasher = BlockHasher(tokenizer_path="/models/pinned")
    with patch.object(BlockHasher, "_load_via_auto", staticmethod(fake_auto)):
        hasher._get_tokenizer("whatever-the-body-said", None)
    assert seen["source"] == "/models/pinned"


def test_sglang_loader_returns_none_when_module_missing():
    """Router-only host without sglang: loader swallows ImportError -> None."""
    result = BlockHasher._load_via_sglang("nonexistent-model-id-12345")
    assert result is None or hasattr(result, "encode")
