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

import pytest

from infera.common.worker_pool import EngineType
from infera.router.kv_event.block_hasher import BlockHasher, _normalise_history
from infera.router.kv_event.hasher import ROUTER_SEED, hash_chunk, hash_request


class _StubTokenizer:
    chat_template: Any = None

    def __init__(self) -> None:
        # Whether each encode() saw an explicit add_special_tokens. A stub that
        # only defaulted the kwarg would accept either choice at the call site,
        # and the two paths need opposite ones.
        self.specials: list[bool] = []
        self.template_kwargs: list[dict] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.specials.append(add_special_tokens)
        return [ord(c) for c in text]

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        # Recorded, not ignored: every name the engine puts in scope and the
        # router does not is a silent prefix divergence, so the tests below
        # assert on this rather than on the rendered string.
        self.template_kwargs.append(dict(kwargs))
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
    protocol_key = "sglang.srt.entrypoints.openai.protocol"
    if with_protocol:
        protocol = ModuleType(protocol_key)
        protocol.Tool = _FakeTool
        modules[protocol_key] = protocol
    else:
        # patch.dict only ADDS. Without this, a real sglang install -- or a fake
        # left in sys.modules by an earlier test in the same process -- keeps the
        # Tool model importable and this helper tests the opposite of its name.
        modules[protocol_key] = None

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


def test_unreproducible_tools_route_on_load():
    """No Tool model to normalise with -> hash nothing rather than a wrong prefix.

    Falling back to `apply_chat_template` without the tools is not a safe
    degradation: the tools render ahead of the conversation, so every block --
    including the first -- diverges from the engine, with no error anywhere. An
    empty hash costs the same load-only routing and is honest about it.
    """
    encoder = _RecordingEncoder()
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = _StubTokenizer()
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        # No `function.name`: `Tool` would reject this and so does the local
        # fallback, so there is no dump to reproduce either way.
        "tools": [{"type": "function", "function": {}}],
    }
    with _fake_sglang("DeepseekV4ForCausalLM", encoder, with_protocol=False):
        out = hasher.hash_for(body, block_size=4)
    assert encoder.calls == []
    assert out == []


def test_tools_still_render_on_a_router_host_without_sglang():
    """A router pod built from the slim image is a supported deployment, and
    this module already carries sglang-free copies of the two message rewrites.
    Without one for the tool dump, `_chat_tools` turned the missing model into
    "cannot reproduce" and every tools-carrying request -- the whole agentic
    workload -- hashed to nothing and routed on load, silently, while tool-free
    chat kept the hit-rate metric looking plausible."""
    encoder = _RecordingEncoder()
    hasher = BlockHasher()
    tokenizer = _StubTokenizer()
    tokenizer.chat_template = "{{ messages }}"
    hasher._tokenizers[(None, "m")] = tokenizer
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    }
    with _fake_sglang("SomeOtherForCausalLM", encoder, with_protocol=False):
        out = hasher.hash_for(body, block_size=4)
    assert out != [], "the fallback dump must keep this request hashable"
    assert tokenizer.template_kwargs[-1]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "f",
                "description": None,
                "parameters": {},
                "strict": False,
            },
            "defer_loading": None,
        }
    ], "and it must materialise exactly the defaults pydantic would"


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


# ---- template context: what the engine puts in scope besides `messages` ----
#
# `serving_chat` renders with
#   apply_chat_template(msgs, ..., tools=tools, **extra_template_kwargs)
# and a template reads whatever names it likes off that context. A name the
# router omits is undefined, not an error -- GLM-5.3's very first line branches
# on `reasoning_effort`, and tools render ahead of the conversation, so an
# omission moves block 0 and every block chained off it. Nothing logs.


def _hash_with(body: dict) -> tuple[_StubTokenizer, list[int]]:
    tokenizer = _StubTokenizer()
    tokenizer.chat_template = "{{ messages }}"
    hasher = BlockHasher()
    hasher._tokenizers[(None, "m")] = tokenizer
    with patch.object(BlockHasher, "_load_hf_config", staticmethod(lambda _s: None)):
        out = hasher.hash_for({"model": "m", **body}, block_size=4)
    return tokenizer, out


_MSGS = [{"role": "user", "content": "hi"}]
_TOOL = {"type": "function", "function": {"name": "f"}}


def test_reasoning_effort_reaches_the_template():
    tokenizer, _ = _hash_with({"messages": _MSGS, "reasoning_effort": "low"})
    assert tokenizer.template_kwargs == [{"reasoning_effort": "low"}]


def test_absent_reasoning_effort_is_not_invented():
    """The engine only forwards it when the request set one; a default here
    would render `Reasoning Effort: Low` against the engine's `High`."""
    tokenizer, _ = _hash_with({"messages": _MSGS})
    assert tokenizer.template_kwargs == [{}]


def test_chat_template_kwargs_are_spread_and_win():
    tokenizer, _ = _hash_with(
        {
            "messages": _MSGS,
            "reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": "high", "clear_thinking": True},
        }
    )
    # serving_chat seeds reasoning_effort then `.update(chat_template_kwargs)`.
    assert tokenizer.template_kwargs == [{"reasoning_effort": "high", "clear_thinking": True}]


def test_tools_are_passed_in_model_dump_shape():
    with _fake_sglang("SomeOtherForCausalLM", _RecordingEncoder()):
        tokenizer, _ = _hash_with({"messages": _MSGS, "tools": [_TOOL]})
    assert tokenizer.template_kwargs == [
        {"tools": [{"strict": False, "defer_loading": None, **_TOOL}]}
    ]


def test_tool_choice_none_suppresses_tools():
    with _fake_sglang("SomeOtherForCausalLM", _RecordingEncoder()):
        tokenizer, _ = _hash_with({"messages": _MSGS, "tools": [_TOOL], "tool_choice": "none"})
    assert tokenizer.template_kwargs == [{}]


def test_named_tool_choice_narrows_the_list():
    other = {"type": "function", "function": {"name": "g"}}
    with _fake_sglang("SomeOtherForCausalLM", _RecordingEncoder()):
        tokenizer, _ = _hash_with(
            {
                "messages": _MSGS,
                "tools": [_TOOL, other],
                "tool_choice": {"type": "function", "function": {"name": "g"}},
            }
        )
    assert tokenizer.template_kwargs == [
        {"tools": [{"strict": False, "defer_loading": None, **other}]}
    ]


def test_a_tokenizer_that_cannot_load_is_only_attempted_once(caplog):
    """Failures used to be left out of the cache, so `_load` re-ran on EVERY
    request: an import plus a `get_tokenizer` that may reach the filesystem or
    the HF hub, on the routing hot path, and a log line per request on top.

    The load result is a property of (engine, source), not of the request.
    """
    hasher = BlockHasher()
    attempts = []
    with patch.object(BlockHasher, "_load", side_effect=lambda src, eng: attempts.append(src)):
        with caplog.at_level(logging.ERROR):
            for _ in range(5):
                assert hasher._get_tokenizer("m", EngineType.SGLANG) is None

    assert len(attempts) == 1, f"one load per (engine, source), got {len(attempts)}"
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1, (
        "and it says so once, not once per request"
    )
    assert not hasher.can_render("m", EngineType.SGLANG)


def test_caching_a_failure_does_not_bleed_across_engines():
    """The two engines pick different tokenizer implementations for the same
    model, so a failure under one says nothing about the other."""
    hasher = BlockHasher()
    hasher._tokenizers[(EngineType.SGLANG, "m")] = None
    assert not hasher.can_render("m", EngineType.SGLANG)
    assert hasher.can_render("m", EngineType.VLLM)


# ---- the rewrites `serving_chat` makes before the template sees a message --
#
# All three are unconditional in the engine and all three are invisible when
# missed: the template either raises (and the whole prompt renders to nothing,
# routing on load) or renders different bytes at the front.


def test_tool_content_parts_are_flattened_to_one_string():
    """`normalize_tool_content` (serving_chat.py:1438) joins the text parts of a
    `tool` message with a single space.

    Codex and every other client that returns tool output as content parts
    sends exactly this shape. Left as a list, GLM-5.3's `content.strip()`
    raises and takes the whole conversation with it.
    """
    out = _normalise_history(
        [
            {
                "role": "tool",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            }
        ]
    )
    assert out[0]["content"] == "a b"


def test_a_bare_string_part_is_joined_too():
    out = _normalise_history([{"role": "tool", "content": ["a", {"type": "text", "text": "b"}]}])
    assert out[0]["content"] == "a b"


def test_a_text_part_with_no_text_still_takes_its_place_in_the_join():
    """`p.get("text", "")` -- the empty part is a separator the engine emits."""
    out = _normalise_history(
        [{"role": "tool", "content": [{"type": "text"}, {"type": "text", "text": "b"}]}]
    )
    assert out[0]["content"] == " b"


def test_a_list_with_a_non_text_part_is_left_alone():
    """ "preserve lists containing non-text-type items that some templates
    intentionally iterate over" -- flattening here would diverge the other way."""
    content = [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "u"}}]
    out = _normalise_history([{"role": "tool", "content": content}])
    assert isinstance(out[0]["content"], list), "a non-text part must survive as a list"


def test_only_tool_messages_are_flattened():
    """`if role != "tool": return content`. A user message's content parts go to
    `process_content_for_template_format`, which is a different rewrite."""
    content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    out = _normalise_history([{"role": "user", "content": content}])
    assert out == [{"role": "user", "content": content}]


def test_flattening_does_not_mutate_the_caller_s_messages():
    """The body is shared across every variant the policy hashes for."""
    messages = [{"role": "tool", "content": [{"type": "text", "text": "a"}]}]
    _normalise_history(messages)
    assert messages == [{"role": "tool", "content": [{"type": "text", "text": "a"}]}]


# ---- the effort_kwarg remap ------------------------------------------------


_EFFORT_TEMPLATE = "{{ low_effort }}{{ truncate_history_thinking }}{{ messages }}"


def test_low_effort_is_remapped_for_a_template_that_takes_the_boolean():
    """serving_chat.py:1456-1458. Reachable only on `reasoning_effort: "low"`,
    and its only symptom when skipped is a hit rate lower than it should be."""
    tokenizer = _StubTokenizer()
    tokenizer.chat_template = _EFFORT_TEMPLATE
    hasher = BlockHasher()
    hasher._tokenizers[(EngineType.SGLANG, "m")] = tokenizer
    with patch.object(BlockHasher, "_load_hf_config", staticmethod(lambda _s: None)):
        hasher.hash_for(
            {"model": "m", "messages": _MSGS, "reasoning_effort": "low"},
            block_size=4,
            engine=EngineType.SGLANG,
        )
    assert tokenizer.template_kwargs == [{"reasoning_effort": "low", "low_effort": True}]


def test_the_remap_reads_the_tokenizer_the_render_is_using():
    """Regression: the lookup this replaced keyed the cache `(None, source)`
    while `_get_tokenizer` fills it under `(engine, source)`.

    In production an engine is always known, so the lookup missed every time
    and fell through to reading `chat_template.jinja` off what is usually a
    bare HF model id rather than a path -- i.e. the remap never fired on a live
    request. A test that seeds `(None, ...)`, as the ones above do, cannot see
    that; this one seeds the key production actually uses.
    """
    tokenizer = _StubTokenizer()
    tokenizer.chat_template = _EFFORT_TEMPLATE
    hasher = BlockHasher()
    hasher._tokenizers[(EngineType.VLLM, "m")] = tokenizer
    assert hasher._effort_kwarg("m", tokenizer) == "low_effort"


def test_an_ordinary_template_gets_no_remap():
    hasher = BlockHasher()
    tokenizer = _StubTokenizer()
    tokenizer.chat_template = "{{ messages }}"
    assert hasher._effort_kwarg("m", tokenizer) is None


# ---- the pydantic projection the engine applies before any encoder ---------


def test_messages_are_projected_the_way_the_engine_dumps_them():
    """serving_chat.py:1322 parses every message and dumps it back with no
    `exclude_unset`/`exclude_none`, so the template sees fields the client
    never sent. A template testing presence rather than truthiness renders
    differently against the client dict."""
    pytest.importorskip("sglang.srt.entrypoints.openai.protocol")
    out = _normalise_history(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"a": 1}'},
                    }
                ],
            }
        ]
    )
    msg = out[0]
    assert msg["content"] == "", "the null-content rewrite still applies"
    assert msg["tool_calls"][0]["function"]["arguments"] == {"a": 1}, "and the args parse"
    for materialised in ("tool_call_id", "name", "reasoning_content"):
        assert materialised in msg, f"{materialised} must be present, as the engine dumps it"
    assert "index" in msg["tool_calls"][0], "ToolCall.index is materialised too"


def test_an_undeclared_key_is_dropped_as_the_engine_drops_it():
    pytest.importorskip("sglang.srt.entrypoints.openai.protocol")
    out = _normalise_history([{"role": "user", "content": "hi", "not_a_field": 1}])
    assert "not_a_field" not in out[0]


def test_a_body_the_engine_would_reject_renders_as_sent():
    """Refusing here would drop kv-aware for a shape sglang may still accept;
    the projection is best-effort and falls back to the client's dict."""
    out = _normalise_history([{"role": "nonsense-role", "content": "hi"}])
    assert out[0]["role"] == "nonsense-role"


# ---- tool-call arguments: all-or-nothing -----------------------------------


def test_a_half_bad_tool_call_list_is_left_entirely_alone():
    """sglang's function assigns in place per call and re-raises on the first
    bad one, so normalising the live message and swallowing the error appended
    call #1 as a dict and call #2 as a string -- a hybrid no client produces,
    which usually renders fine and hashes a wrong prefix."""
    out = _normalise_history(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                    {
                        "id": "b",
                        "type": "function",
                        "function": {"name": "g", "arguments": "[1,2]"},
                    },
                ],
            }
        ]
    )
    args = [tc["function"]["arguments"] for tc in out[0]["tool_calls"]]
    assert all(isinstance(a, str) for a in args), (
        "either both parse or neither does; a mixed message is a shape the "
        "engine never renders, and on kimi_k3 it serves the verbatim strings"
    )
