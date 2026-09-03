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
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

from infera.common.worker_pool import EngineType
from infera.router.kv_event.block_hasher import BlockHasher
from infera.router.kv_event.hasher import ROUTER_SEED, hash_chunk, hash_request


class _StubTokenizer:
    chat_template: Any = None

    def __init__(self) -> None:
        # Whether each encode() saw an explicit add_special_tokens. A stub that
        # only defaulted the kwarg would accept either choice at the call site,
        # and the two paths need opposite ones.
        self.specials: list[bool] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.specials.append(add_special_tokens)
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


# ---- sglang native chat encoders (DeepSeek-V4) ----------------------------
#
# DeepSeek-V4 ships no chat template, so apply_chat_template raises and every
# chat request loses its cache info. The router mirrors sglang's own encoder
# instead; these tests stub sglang so no weights or engine install are needed.


class _RecordingEncoder:
    """Stand-in for sglang.srt.entrypoints.openai.encoding_dsv4."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], str]] = []

    def encode_messages(self, messages, thinking_mode):
        self.calls.append(([dict(m) for m in messages], thinking_mode))
        return "DSV4:" + "|".join(m["role"] for m in messages)


class _FakeTool:
    """Stand-in for the pydantic Tool model: model_dump() fills in defaults."""

    def __init__(self, **fields):
        self.fields = fields

    def model_dump(self):
        return {"strict": False, "defer_loading": None, **self.fields}


@contextmanager
def _fake_sglang(arch: str, encoder: _RecordingEncoder, *, with_protocol: bool = True):
    def resolve_chat_encoding_spec(*, hf_config, tokenizer, tool_call_parser=None):
        name = (hf_config.architectures or [""])[0]
        if "DeepseekV4" in name:
            return "dsv4"
        if "KimiK3" in name:
            return "kimi_k3"
        return None

    modules: dict[str, Any] = {}
    for name in ("sglang", "sglang.srt", "sglang.srt.entrypoints", "sglang.srt.entrypoints.openai"):
        modules[name] = ModuleType(name)
    chat_encoding = ModuleType("sglang.srt.entrypoints.openai.chat_encoding")
    chat_encoding.resolve_chat_encoding_spec = resolve_chat_encoding_spec
    modules["sglang.srt.entrypoints.openai.chat_encoding"] = chat_encoding
    dsv4 = ModuleType("sglang.srt.entrypoints.openai.encoding_dsv4")
    dsv4.encode_messages = encoder.encode_messages
    modules["sglang.srt.entrypoints.openai.encoding_dsv4"] = dsv4
    if with_protocol:
        protocol = ModuleType("sglang.srt.entrypoints.openai.protocol")
        protocol.Tool = _FakeTool
        modules["sglang.srt.entrypoints.openai.protocol"] = protocol
    else:
        # The engine-image test environment already imports the real protocol
        # module. Mask it explicitly so this case still exercises the intended
        # router-only deployment where sglang's Tool model is unavailable.
        modules["sglang.srt.entrypoints.openai.protocol"] = None

    config = SimpleNamespace(architectures=[arch])
    with (
        patch.dict(sys.modules, modules),
        patch.object(BlockHasher, "_load_hf_config", staticmethod(lambda _s: config)),
    ):
        yield


def test_dsv4_arch_uses_sglang_native_encoder():
    encoder = _RecordingEncoder()
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _StubTokenizer()  # no chat_template, like DSv4
    with _fake_sglang("DeepseekV4ForCausalLM", encoder):
        out = hasher.hash_for(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, block_size=4
        )
    messages, thinking_mode = encoder.calls[0]
    assert thinking_mode == "chat"
    # The engine prepends an empty system message before rendering.
    assert messages == [
        {"role": "system", "content": ""},
        {"role": "user", "content": "hi"},
    ]
    assert out == hash_request([ord(c) for c in "DSV4:system|user"], 4)


def test_dsv4_output_is_tokenized_with_specials_and_templates_without():
    """serving_chat tokenizes the dsv4 encoder's output with a plain
    ``tokenizer.encode(real_input)``, but passes ``add_special_tokens=False`` at
    the chat-template site. Getting this backwards doubles the BOS on any
    tokenizer with ``add_bos_token: true`` -- and that misses silently, which is
    the failure the dsv4 encoder exists to remove."""
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    tok = _StubTokenizer()
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = tok
    with _fake_sglang("DeepseekV4ForCausalLM", _RecordingEncoder()):
        hasher.hash_for(body, block_size=4)
    assert tok.specials == [True]

    # A non-dsv4 arch falls through to the chat template, which spells out its
    # own specials as text.
    tok = _StubTokenizer()
    tok.chat_template = "irrelevant, the stub renders it"
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = tok
    with _fake_sglang("Qwen3ForCausalLM", _RecordingEncoder()):
        hasher.hash_for(body, block_size=4)
    assert tok.specials == [False]


def test_dsv4_tools_are_normalised_through_the_tool_model():
    encoder = _RecordingEncoder()
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _StubTokenizer()
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
    }
    with _fake_sglang("DeepseekV4ForCausalLM", encoder):
        assert hasher.hash_for(body, block_size=4) != []
    messages, _ = encoder.calls[0]
    # Raw request dicts would drop the model defaults and desync the prefix.
    assert messages[0]["tools"] == [
        {
            "strict": False,
            "defer_loading": None,
            "type": "function",
            "function": {"name": "f"},
        }
    ]


def test_dsv4_path_skipped_for_tools_when_tool_model_unavailable():
    """No Tool model to normalise with -> fall back rather than hash a wrong prefix."""
    encoder = _RecordingEncoder()
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _StubTokenizer()
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
    }
    with _fake_sglang("DeepseekV4ForCausalLM", encoder, with_protocol=False):
        out = hasher.hash_for(body, block_size=4)
    assert encoder.calls == []
    assert out == hash_request([ord(c) for c in "<user>hi</user><assistant>"], 4)


def test_non_dsv4_model_still_uses_apply_chat_template():
    """A model with a chat template and another arch must not take the dsv4 path."""
    encoder = _RecordingEncoder()
    tokenizer = _StubTokenizer()
    tokenizer.chat_template = "{{ messages }}"
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = tokenizer
    with _fake_sglang("KimiK3ForCausalLM", encoder):
        out = hasher.hash_for(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, block_size=4
        )
    assert encoder.calls == []
    assert out == hash_request([ord(c) for c in "<user>hi</user><assistant>"], 4)


def test_dsv4_encoder_import_error_falls_back_cleanly():
    """Router-only host: sglang missing -> today's chat-template behaviour."""
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _StubTokenizer()
    config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    broken = {name: None for name in ("sglang", "sglang.srt.entrypoints.openai.chat_encoding")}
    with (
        patch.dict(sys.modules, broken),
        patch.object(BlockHasher, "_load_hf_config", staticmethod(lambda _s: config)),
    ):
        out = hasher.hash_for(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, block_size=4
        )
    assert out == hash_request([ord(c) for c in "<user>hi</user><assistant>"], 4)


def test_dsv4_path_skipped_when_hf_config_unavailable():
    """No config.json beside the tokenizer -> no spec to resolve, no crash."""
    encoder = _RecordingEncoder()
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _StubTokenizer()
    with _fake_sglang("DeepseekV4ForCausalLM", encoder):
        with patch.object(BlockHasher, "_load_hf_config", staticmethod(lambda _s: None)):
            out = hasher.hash_for(
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, block_size=4
            )
    assert encoder.calls == []
    assert out == hash_request([ord(c) for c in "<user>hi</user><assistant>"], 4)


def test_spec_resolution_error_falls_back_to_chat_template():
    """hf_config without .architectures makes resolve_chat_encoding_spec raise."""
    encoder = _RecordingEncoder()
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _StubTokenizer()
    with _fake_sglang("DeepseekV4ForCausalLM", encoder):
        with patch.object(
            BlockHasher, "_load_hf_config", staticmethod(lambda _s: SimpleNamespace())
        ):
            out = hasher.hash_for(
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, block_size=4
            )
    assert encoder.calls == []
    assert out == hash_request([ord(c) for c in "<user>hi</user><assistant>"], 4)


def test_hf_config_cached_per_source():
    calls = {"n": 0}

    def fake_config(_source):
        calls["n"] += 1
        return SimpleNamespace(architectures=["LlamaForCausalLM"])

    hasher = BlockHasher()
    with patch.object(BlockHasher, "_load_hf_config", staticmethod(fake_config)):
        hasher._get_hf_config("m")
        hasher._get_hf_config("m")
    assert calls["n"] == 1
