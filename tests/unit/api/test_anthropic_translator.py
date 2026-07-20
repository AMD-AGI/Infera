###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for the Anthropic Messages ↔ OpenAI Chat translator.

Three layers tested in this file:

  1. Request shape: anthropic_to_openai_request — translation,
     refused features (tools, multimodal).
  2. Non-streaming response: openai_to_anthropic_response — usage
     mapping, finish_reason mapping, content shape.
  3. Streaming response: openai_to_anthropic_sse — event ordering,
     malformed-chunk tolerance, [DONE] handling.

No FastAPI / network / engine touched here; this is pure data-shape.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from infera.api.anthropic import (
    AnthropicRequestRejected,
    _flatten_message_content,
    _flatten_system,
    anthropic_to_openai_request,
    openai_to_anthropic_response,
    openai_to_anthropic_sse,
)

# ----------------------------------------------------------------------
# anthropic_to_openai_request — basic translations
# ----------------------------------------------------------------------


def test_basic_text_only_message_translation():
    body = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }
    out = anthropic_to_openai_request(body)
    assert out["model"] == "claude-3-5-sonnet-20240620"
    assert out["max_tokens"] == 1024
    assert out["messages"] == [{"role": "user", "content": "hello"}]


def test_system_string_becomes_first_message():
    body = {
        "model": "m",
        "system": "you are helpful",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = anthropic_to_openai_request(body)
    assert out["messages"][0] == {"role": "system", "content": "you are helpful"}
    assert out["messages"][1] == {"role": "user", "content": "hi"}


def test_system_array_concatenates_text_blocks():
    body = {
        "model": "m",
        "system": [
            {"type": "text", "text": "block A"},
            {"type": "text", "text": "block B"},
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = anthropic_to_openai_request(body)
    assert out["messages"][0]["role"] == "system"
    assert "block A" in out["messages"][0]["content"]
    assert "block B" in out["messages"][0]["content"]


def test_no_system_block_omits_system_message():
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    out = anthropic_to_openai_request(body)
    assert out["messages"][0]["role"] == "user"


def test_array_content_blocks_concatenated_as_text():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part 1"},
                    {"type": "text", "text": "part 2"},
                ],
            }
        ],
    }
    out = anthropic_to_openai_request(body)
    assert "part 1" in out["messages"][0]["content"]
    assert "part 2" in out["messages"][0]["content"]


def test_stop_sequences_become_stop():
    body = {"model": "m", "messages": [], "stop_sequences": ["END", "\n\n"]}
    out = anthropic_to_openai_request(body)
    assert out["stop"] == ["END", "\n\n"]


def test_metadata_user_id_becomes_user():
    body = {"model": "m", "messages": [], "metadata": {"user_id": "u123"}}
    out = anthropic_to_openai_request(body)
    assert out["user"] == "u123"


def test_temperature_top_p_pass_through():
    body = {"model": "m", "messages": [], "temperature": 0.7, "top_p": 0.9, "top_k": 40}
    out = anthropic_to_openai_request(body)
    assert out["temperature"] == 0.7
    assert out["top_p"] == 0.9
    assert out["top_k"] == 40


def test_stream_flag_propagates():
    body = {"model": "m", "messages": [], "stream": True}
    out = anthropic_to_openai_request(body)
    assert out["stream"] is True


def test_cache_control_annotations_dropped_during_translation():
    """The cache_control fields stay parseable on the Anthropic side;
    once translated to OpenAI the engine doesn't need them, only the
    raw text. We don't strip cache_control explicitly — the flatten
    helpers ignore the field. Verify text round-trips."""
    body = {
        "model": "m",
        "system": [
            {
                "type": "text",
                "text": "stable system",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "the question", "cache_control": {"type": "ephemeral"}}
                ],
            }
        ],
    }
    out = anthropic_to_openai_request(body)
    # Translated body has only text — cache_control etc. dropped.
    assert out["messages"][0]["content"] == "stable system"
    assert out["messages"][1]["content"] == "the question"
    # Verify no nested dict structures snuck through.
    for msg in out["messages"]:
        assert isinstance(msg["content"], str)


# ----------------------------------------------------------------------
# anthropic_to_openai_request — refused features
# ----------------------------------------------------------------------


def test_empty_tools_list_does_not_add_tools_field():
    """An empty `tools` list shouldn't add a field to the OpenAI body."""
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": []}
    out = anthropic_to_openai_request(body)
    assert out["messages"][0]["content"] == "hi"
    assert "tools" not in out


def test_rejects_image_block():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this:"},
                    {"type": "image", "source": {"data": "..."}},
                ],
            }
        ],
    }
    with pytest.raises(AnthropicRequestRejected, match="image"):
        anthropic_to_openai_request(body)


def test_rejects_audio_block():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "audio", "source": {"data": "..."}}],
            }
        ],
    }
    with pytest.raises(AnthropicRequestRejected, match="audio"):
        anthropic_to_openai_request(body)


def test_rejects_non_dict_body():
    with pytest.raises(AnthropicRequestRejected):
        anthropic_to_openai_request("not a dict")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Tool calling — request translation
# ----------------------------------------------------------------------


def test_tools_translated_to_openai_function_shape():
    body = {
        "model": "m",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "look up weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    }
    out = anthropic_to_openai_request(body)
    assert out["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "look up weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def test_tools_input_schema_missing_becomes_empty_parameters():
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "x", "description": "y"}],
    }
    out = anthropic_to_openai_request(body)
    assert out["tools"][0]["function"]["parameters"] == {}


def test_tool_def_empty_description_omitted():
    """Empty description on the Anthropic side
    should NOT appear as `description: ""` on the OpenAI side —
    strict receivers surface it back to the model as noise."""
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "x", "description": "", "input_schema": {}}],
    }
    out = anthropic_to_openai_request(body)
    assert "description" not in out["tools"][0]["function"]


def test_tool_def_missing_description_omitted():
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "x", "input_schema": {}}],
    }
    out = anthropic_to_openai_request(body)
    assert "description" not in out["tools"][0]["function"]


def test_tool_choice_auto():
    body = {"model": "m", "messages": [], "tool_choice": {"type": "auto"}}
    assert anthropic_to_openai_request(body)["tool_choice"] == "auto"


def test_tool_choice_any_becomes_required():
    body = {"model": "m", "messages": [], "tool_choice": {"type": "any"}}
    assert anthropic_to_openai_request(body)["tool_choice"] == "required"


def test_tool_choice_none():
    body = {"model": "m", "messages": [], "tool_choice": {"type": "none"}}
    assert anthropic_to_openai_request(body)["tool_choice"] == "none"


def test_tool_choice_specific_tool():
    body = {
        "model": "m",
        "messages": [],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }
    out = anthropic_to_openai_request(body)
    assert out["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


def test_tool_choice_unknown_type_dropped_and_warns(caplog):
    """Unknown tool_choice types are dropped (not
    a 400, so future Anthropic types don't break clients) but logged
    so operators see the drift."""
    import logging

    body = {"model": "m", "messages": [], "tool_choice": {"type": "future-tbd"}}
    with caplog.at_level(logging.WARNING, logger="infera.api.anthropic"):
        out = anthropic_to_openai_request(body)
    assert "tool_choice" not in out
    assert any("future-tbd" in rec.message for rec in caplog.records)


def test_tool_choice_disable_parallel_lifts_to_parallel_tool_calls_false():
    """Anthropic's
    ``disable_parallel_tool_use: true`` (nested in tool_choice) maps
    to OpenAI's top-level ``parallel_tool_calls: false``."""
    body = {
        "model": "m",
        "messages": [],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    }
    out = anthropic_to_openai_request(body)
    assert out["tool_choice"] == "auto"
    assert out["parallel_tool_calls"] is False


def test_tool_choice_disable_parallel_false_not_hoisted():
    """Only `true` is meaningful — `false` is the default; don't add
    the OpenAI field for it."""
    body = {
        "model": "m",
        "messages": [],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": False},
    }
    out = anthropic_to_openai_request(body)
    assert "parallel_tool_calls" not in out


def test_assistant_tool_use_becomes_openai_tool_calls():
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    },
                ],
            },
        ],
    }
    out = anthropic_to_openai_request(body)
    assert out["messages"][0] == {"role": "user", "content": "what's the weather?"}
    asst = out["messages"][1]
    assert asst["role"] == "assistant"
    assert asst["content"] == "Let me check."
    assert asst["tool_calls"] == [
        {
            "id": "toolu_abc",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            },
        }
    ]


def test_assistant_tool_use_only_no_text_emits_null_content():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu1", "name": "x", "input": {}},
                ],
            }
        ],
    }
    out = anthropic_to_openai_request(body)
    asst = out["messages"][0]
    assert asst["content"] is None
    assert asst["tool_calls"][0]["function"]["name"] == "x"


def test_assistant_tool_use_missing_id_synthesizes_one():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "x", "input": {}}],
            }
        ],
    }
    out = anthropic_to_openai_request(body)
    call_id = out["messages"][0]["tool_calls"][0]["id"]
    assert call_id.startswith("call_")


def test_user_tool_result_becomes_openai_tool_message():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "72F sunny"}
                ],
            }
        ],
    }
    out = anthropic_to_openai_request(body)
    assert len(out["messages"]) == 1
    assert out["messages"][0] == {
        "role": "tool",
        "tool_call_id": "toolu_abc",
        "content": "72F sunny",
    }


def test_user_tool_result_array_content_concatenated():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": [
                            {"type": "text", "text": "line 1"},
                            {"type": "text", "text": "line 2"},
                        ],
                    }
                ],
            }
        ],
    }
    out = anthropic_to_openai_request(body)
    assert out["messages"][0]["content"] == "line 1\nline 2"


def test_user_mixed_text_and_tool_result_splits_into_two_messages():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Got it, also:"},
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "data"},
                ],
            }
        ],
    }
    out = anthropic_to_openai_request(body)
    assert out["messages"][0] == {"role": "user", "content": "Got it, also:"}
    assert out["messages"][1] == {"role": "tool", "tool_call_id": "tu1", "content": "data"}


def test_multiple_tool_results_each_get_own_message():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": "r1"},
                    {"type": "tool_result", "tool_use_id": "b", "content": "r2"},
                ],
            }
        ],
    }
    out = anthropic_to_openai_request(body)
    assert out["messages"][0] == {"role": "tool", "tool_call_id": "a", "content": "r1"}
    assert out["messages"][1] == {"role": "tool", "tool_call_id": "b", "content": "r2"}


def test_tool_result_missing_id_rejected():
    """Silently coercing tool_use_id to '' would
    mismatch the OpenAI tool message back to the wrong tool_call.
    Reject at the front door."""
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": "result"}],
            }
        ],
    }
    with pytest.raises(AnthropicRequestRejected, match="tool_use_id"):
        anthropic_to_openai_request(body)


def test_tool_result_empty_string_id_rejected():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "", "content": "result"}],
            }
        ],
    }
    with pytest.raises(AnthropicRequestRejected, match="tool_use_id"):
        anthropic_to_openai_request(body)


def test_image_inside_tool_result_content_rejected():
    """A tool_result whose own content carries an
    image block silently dropped that image pre-fix. Multimodal
    rejection must recurse one level into tool_result.content."""
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": [
                            {"type": "text", "text": "see:"},
                            {"type": "image", "source": {"data": "..."}},
                        ],
                    }
                ],
            }
        ],
    }
    with pytest.raises(AnthropicRequestRejected, match="image"):
        anthropic_to_openai_request(body)


def test_document_inside_tool_result_content_rejected():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": [{"type": "document", "source": {"data": "..."}}],
                    }
                ],
            }
        ],
    }
    with pytest.raises(AnthropicRequestRejected, match="document"):
        anthropic_to_openai_request(body)


def test_full_round_trip_request_with_tools_and_history():
    """End-to-end: multi-turn with system, tools, prior tool_use,
    prior tool_result, new user question."""
    body = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 1024,
        "system": "you are helpful",
        "tools": [
            {
                "name": "search",
                "description": "search the web",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ],
        "tool_choice": {"type": "auto"},
        "messages": [
            {"role": "user", "content": "find python tutorials"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Searching..."},
                    {
                        "type": "tool_use",
                        "id": "tu_search_1",
                        "name": "search",
                        "input": {"q": "python tutorials"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_search_1",
                        "content": "Found 100 results...",
                    }
                ],
            },
        ],
    }
    out = anthropic_to_openai_request(body)
    assert len(out["messages"]) == 4
    assert out["messages"][0]["role"] == "system"
    assert out["messages"][1]["role"] == "user"
    assert out["messages"][2]["role"] == "assistant"
    assert out["messages"][2]["tool_calls"][0]["function"]["name"] == "search"
    assert out["messages"][3]["role"] == "tool"
    assert out["messages"][3]["tool_call_id"] == "tu_search_1"
    assert out["tools"][0]["type"] == "function"
    assert out["tool_choice"] == "auto"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def test_flatten_system_string():
    assert _flatten_system("hello") == "hello"


def test_flatten_system_none():
    assert _flatten_system(None) == ""


def test_flatten_system_array():
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert _flatten_system(blocks) == "a\nb"


def test_flatten_message_content_string():
    assert _flatten_message_content("plain") == "plain"


def test_flatten_message_content_blocks():
    blocks = [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]
    assert _flatten_message_content(blocks) == "x\ny"


# ----------------------------------------------------------------------
# openai_to_anthropic_response — non-streaming
# ----------------------------------------------------------------------


def test_response_content_array_shape():
    openai_resp = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [
            {"message": {"role": "assistant", "content": "the answer"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 34},
        "model": "engine-name",
    }
    out = openai_to_anthropic_response(openai_resp, model="claude-3-5-sonnet")
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["content"] == [{"type": "text", "text": "the answer"}]
    assert out["model"] == "claude-3-5-sonnet"
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {
        "input_tokens": 12,
        "output_tokens": 34,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def test_response_id_is_anthropic_shaped():
    out = openai_to_anthropic_response(
        {"choices": [{"message": {"content": "."}, "finish_reason": "stop"}]}, model="m"
    )
    assert out["id"].startswith("msg_")


def test_response_finish_reason_length_maps_to_max_tokens():
    out = openai_to_anthropic_response(
        {"choices": [{"message": {"content": "."}, "finish_reason": "length"}], "usage": {}},
        model="m",
    )
    assert out["stop_reason"] == "max_tokens"


def test_response_empty_choices_returns_empty_text():
    out = openai_to_anthropic_response({"choices": []}, model="m")
    assert out["content"] == [{"type": "text", "text": ""}]


# ----------------------------------------------------------------------
# openai_to_anthropic_sse — streaming
# ----------------------------------------------------------------------


async def _collect(aiter):
    out = []
    async for chunk in aiter:
        out.append(chunk)
    return b"".join(out)


def _make_chunk_iter(chunks: list[bytes]):
    """Wrap a list of byte chunks as an async iterator."""

    async def _gen():
        for c in chunks:
            yield c

    return _gen()


def _make_openai_chunk(content: str | None = None, finish_reason: str | None = None) -> bytes:
    payload = {"choices": [{"delta": {}, "finish_reason": finish_reason}]}
    if content is not None:
        payload["choices"][0]["delta"]["content"] = content
    return f"data: {json.dumps(payload)}\n".encode()


def test_sse_emits_message_start_then_content_block_start():
    async def _run():
        chunks = [
            _make_openai_chunk("Hello"),
            _make_openai_chunk(" world"),
            _make_openai_chunk(finish_reason="stop"),
            b"data: [DONE]\n",
        ]
        out = await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))
        return out

    out = asyncio.run(_run())
    assert b"event: message_start" in out
    assert b"event: content_block_start" in out
    assert b"event: content_block_delta" in out
    assert b"event: content_block_stop" in out
    assert b"event: message_delta" in out
    assert b"event: message_stop" in out


def test_sse_event_order_is_consistent():
    """Anthropic clients require a specific event order. Verify the
    sequence is correct end-to-end."""

    async def _run():
        chunks = [
            _make_openai_chunk("a"),
            _make_openai_chunk("b"),
            _make_openai_chunk(finish_reason="stop"),
            b"data: [DONE]\n",
        ]
        out = await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))
        return out

    out = asyncio.run(_run())
    # Find the index of each event type.
    text = out.decode()
    pos_start = text.index("event: message_start")
    pos_block_start = text.index("event: content_block_start")
    pos_first_delta = text.index("event: content_block_delta")
    pos_block_stop = text.index("event: content_block_stop")
    pos_msg_delta = text.index("event: message_delta")
    pos_msg_stop = text.index("event: message_stop")
    assert (
        pos_start
        < pos_block_start
        < pos_first_delta
        < pos_block_stop
        < pos_msg_delta
        < pos_msg_stop
    )


def test_sse_tolerates_malformed_chunk():
    """A malformed JSON chunk should be skipped, not crash the stream."""

    async def _run():
        chunks = [
            _make_openai_chunk("ok1"),
            b"data: {not json\n",  # malformed
            _make_openai_chunk("ok2"),
            _make_openai_chunk(finish_reason="stop"),
            b"data: [DONE]\n",
        ]
        out = await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))
        return out

    out = asyncio.run(_run())
    # Both valid deltas pass through.
    assert b"ok1" in out
    assert b"ok2" in out
    assert b"event: message_stop" in out


def test_sse_finish_reason_maps_through():
    async def _run():
        chunks = [
            _make_openai_chunk("partial"),
            _make_openai_chunk(finish_reason="length"),
            b"data: [DONE]\n",
        ]
        out = await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))
        return out

    out = asyncio.run(_run())
    text = out.decode()
    # message_delta payload carries the mapped stop_reason.
    assert '"stop_reason": "max_tokens"' in text


def test_sse_handles_missing_done():
    """Engine closes the stream without [DONE]: emit closers anyway."""

    async def _run():
        chunks = [
            _make_openai_chunk("hello"),
            _make_openai_chunk(finish_reason="stop"),
            # NO [DONE] chunk
        ]
        out = await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))
        return out

    out = asyncio.run(_run())
    assert b"event: message_stop" in out
    assert b"event: content_block_stop" in out


def test_sse_keepalives_dropped():
    """SSE comment lines (starting with `:`) and blank lines are ignored."""

    async def _run():
        chunks = [
            b": keepalive\n\n",
            _make_openai_chunk("x"),
            b"\n",
            _make_openai_chunk(finish_reason="stop"),
            b"data: [DONE]\n",
        ]
        out = await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))
        return out

    out = asyncio.run(_run())
    assert b"event: content_block_delta" in out
    assert b'"text": "x"' in out


def test_sse_usage_chunk_captures_output_tokens():
    """Some engines emit a final chunk with usage only (no choices).
    We should pick up the completion_tokens count."""

    async def _run():
        chunks = [
            _make_openai_chunk("answer"),
            _make_openai_chunk(finish_reason="stop"),
            b'data: {"usage":{"completion_tokens":17}}\n',
            b"data: [DONE]\n",
        ]
        out = await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))
        return out

    out = asyncio.run(_run())
    assert b'"output_tokens": 17' in out


def test_sse_no_content_emits_no_block_events():
    """If the engine sends only a finish_reason chunk with no content,
    don't fabricate a content_block_start/_stop pair."""

    async def _run():
        chunks = [
            _make_openai_chunk(finish_reason="stop"),
            b"data: [DONE]\n",
        ]
        out = await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))
        return out

    out = asyncio.run(_run())
    # message_start should fire (engine sent a chunk with finish_reason);
    # but no content_block_start because there was no text.
    assert b"event: message_start" in out
    assert b"event: content_block_start" not in out
    assert b"event: message_stop" in out


# ----------------------------------------------------------------------
# Response + streaming edge cases
# ----------------------------------------------------------------------


def test_response_uses_msg_id_hint_when_supplied():
    """When caller provides `msg_id_hint`
    (typically the infera request id), it's used as the Anthropic
    message id for log correlation."""
    out = openai_to_anthropic_response(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {}},
        model="m",
        msg_id_hint="abc123",
    )
    assert out["id"] == "msg_abc123"


def test_response_falls_back_to_uuid_when_no_hint():
    out = openai_to_anthropic_response(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {}},
        model="m",
    )
    assert out["id"].startswith("msg_")
    assert len(out["id"]) == len("msg_") + 24


def test_response_none_completion_tokens_becomes_zero():
    """Some engines emit `completion_tokens: null`
    which would crash `max(int, None)` (streaming path) and now must
    coerce cleanly in the non-streaming response path too."""
    out = openai_to_anthropic_response(
        {
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": None},
        },
        model="m",
    )
    assert out["usage"]["output_tokens"] == 0


def test_finish_reason_unknown_falls_back_to_end_turn():
    """Unknown OpenAI finish_reason values
    (e.g. SGLang's `eos_token`, vLLM's `abort`) used to silently
    return None; now fall back to `end_turn`."""
    out = openai_to_anthropic_response(
        {"choices": [{"message": {"content": "x"}, "finish_reason": "eos_token"}], "usage": {}},
        model="m",
    )
    assert out["stop_reason"] == "end_turn"


def test_finish_reason_content_filter_maps_to_end_turn():
    """content_filter used to → stop_sequence
    which is wrong — Anthropic doesn't expose a moderation stop."""
    out = openai_to_anthropic_response(
        {
            "choices": [{"message": {"content": "x"}, "finish_reason": "content_filter"}],
            "usage": {},
        },
        model="m",
    )
    assert out["stop_reason"] == "end_turn"


def test_finish_reason_none_returns_none():
    """No finish_reason → stop_reason None (still going)."""
    out = openai_to_anthropic_response(
        {"choices": [{"message": {"content": "x"}, "finish_reason": None}], "usage": {}},
        model="m",
    )
    assert out["stop_reason"] is None


def test_sse_usage_only_first_chunk_then_done_emits_nothing():
    """If engine sends ONLY a usage-only chunk
    then `[DONE]`, we must NOT emit message_stop without
    message_start (Anthropic protocol violation)."""

    async def _run():
        chunks = [
            b'data: {"usage":{"completion_tokens":5}}\n',
            b"data: [DONE]\n",
        ]
        return await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))

    out = asyncio.run(_run())
    assert b"event: message_start" not in out
    assert b"event: message_stop" not in out


def test_sse_none_completion_tokens_does_not_crash():
    """max(int, None) would TypeError without coercion."""

    async def _run():
        chunks = [
            _make_openai_chunk("x"),
            b'data: {"usage":{"completion_tokens":null}}\n',
            _make_openai_chunk(finish_reason="stop"),
            b"data: [DONE]\n",
        ]
        return await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))

    out = asyncio.run(_run())
    assert b"event: message_stop" in out


def test_sse_msg_id_hint_threads_through_to_message_start():
    async def _run():
        chunks = [
            _make_openai_chunk("x"),
            _make_openai_chunk(finish_reason="stop"),
            b"data: [DONE]\n",
        ]
        return await _collect(
            openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m", msg_id_hint="req-xyz")
        )

    out = asyncio.run(_run())
    assert b'"id": "msg_req-xyz"' in out


# ----------------------------------------------------------------------
# Tool calling — non-streaming response translation
# ----------------------------------------------------------------------


def test_response_with_single_tool_call():
    openai_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_xyz",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "NYC"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 25},
    }
    out = openai_to_anthropic_response(openai_resp, model="m")
    assert out["content"] == [
        {
            "type": "tool_use",
            "id": "call_xyz",
            "name": "get_weather",
            "input": {"city": "NYC"},
        }
    ]
    assert out["stop_reason"] == "tool_use"
    assert out["usage"]["input_tokens"] == 100


def test_response_with_text_and_tool_call():
    """Mixed: assistant says something, THEN calls a tool."""
    openai_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Let me check that for you.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {},
    }
    out = openai_to_anthropic_response(openai_resp, model="m")
    assert len(out["content"]) == 2
    assert out["content"][0] == {"type": "text", "text": "Let me check that for you."}
    assert out["content"][1]["type"] == "tool_use"
    assert out["content"][1]["name"] == "lookup"
    assert out["content"][1]["input"] == {}


def test_response_with_multiple_tool_calls():
    """The engine emits parallel tool calls — each becomes a tool_use block."""
    openai_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "a",
                            "type": "function",
                            "function": {"name": "f1", "arguments": '{"x":1}'},
                        },
                        {
                            "id": "b",
                            "type": "function",
                            "function": {"name": "f2", "arguments": '{"y":2}'},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {},
    }
    out = openai_to_anthropic_response(openai_resp, model="m")
    assert len(out["content"]) == 2
    names = [b["name"] for b in out["content"]]
    assert names == ["f1", "f2"]


def test_response_malformed_arguments_json_becomes_empty_dict_and_warns(caplog):
    """Engine ships garbage in `function.arguments` — don't crash; emit
    empty input. A WARNING log keeps the engine
    bug visible instead of silently swallowed."""
    import logging

    openai_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{not json"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {},
    }
    with caplog.at_level(logging.WARNING, logger="infera.api.anthropic"):
        out = openai_to_anthropic_response(openai_resp, model="m")
    assert out["content"][0]["input"] == {}
    assert any("malformed" in rec.message for rec in caplog.records)


def test_response_non_object_arguments_becomes_empty_dict_and_warns(caplog):
    """Engine returned VALID JSON but not an object (e.g. a plain
    string or a number). Tool inputs MUST be objects — substitute
    {} and log."""
    import logging

    openai_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "f", "arguments": '"a string"'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {},
    }
    with caplog.at_level(logging.WARNING, logger="infera.api.anthropic"):
        out = openai_to_anthropic_response(openai_resp, model="m")
    assert out["content"][0]["input"] == {}
    assert any("non-object" in rec.message for rec in caplog.records)


def test_response_tool_call_missing_id_synthesizes_one():
    openai_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"type": "function", "function": {"name": "f", "arguments": "{}"}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {},
    }
    out = openai_to_anthropic_response(openai_resp, model="m")
    assert out["content"][0]["id"].startswith("toolu_")


def test_response_finish_reason_tool_calls_maps_to_tool_use():
    openai_resp = {
        "choices": [
            {
                "message": {"content": "ok"},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {},
    }
    out = openai_to_anthropic_response(openai_resp, model="m")
    assert out["stop_reason"] == "tool_use"


# ----------------------------------------------------------------------
# Tool calling — streaming SSE
# ----------------------------------------------------------------------


def _tool_call_chunk(
    index: int,
    *,
    id: str | None = None,
    name: str | None = None,
    arg_frag: str | None = None,
    finish: str | None = None,
) -> bytes:
    """Build an OpenAI streaming chunk for a tool call delta."""
    tc: dict = {"index": index, "function": {}}
    if id is not None:
        tc["id"] = id
        tc["type"] = "function"
    if name is not None:
        tc["function"]["name"] = name
    if arg_frag is not None:
        tc["function"]["arguments"] = arg_frag
    payload = {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": finish}]}
    return f"data: {json.dumps(payload)}\n".encode()


def test_sse_tool_call_streaming_basic():
    """Standard pattern: id+name first chunk, then arg fragments, then [DONE]."""

    async def _run():
        chunks = [
            _tool_call_chunk(0, id="call_1", name="get_weather"),
            _tool_call_chunk(0, arg_frag='{"city"'),
            _tool_call_chunk(0, arg_frag=': "NYC"'),
            _tool_call_chunk(0, arg_frag="}"),
            _make_openai_chunk(finish_reason="tool_calls"),
            b"data: [DONE]\n",
        ]
        return await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))

    out = asyncio.run(_run())
    text = out.decode()
    assert text.index("event: message_start") < text.index("event: content_block_start")
    assert "tool_use" in text
    assert '"name": "get_weather"' in text
    assert "input_json_delta" in text
    assert '"partial_json": "{\\"city\\""' in text
    assert b"event: message_stop" in out


def test_sse_text_then_tool_call():
    """Mixed: text first, then a tool call. FCFS — text at 0, tool at 1."""

    async def _run():
        chunks = [
            _make_openai_chunk("Let me check"),
            _tool_call_chunk(0, id="c1", name="lookup"),
            _tool_call_chunk(0, arg_frag="{}"),
            _make_openai_chunk(finish_reason="tool_calls"),
            b"data: [DONE]\n",
        ]
        return await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))

    out = asyncio.run(_run())
    text = out.decode()
    text_block_start_pos = text.index('"index": 0')
    tool_block_start_pos = text.index('"index": 1')
    assert text_block_start_pos < tool_block_start_pos
    assert "Let me check" in text
    assert "lookup" in text


def test_sse_tool_before_text_lands_tool_at_index_zero():
    """FCFS, not 'text always at 0'. If the
    engine emits a tool_call delta BEFORE any text, the tool block
    takes Anthropic index 0; any later text takes the next free
    index. Clients keyed on the index would mis-attribute the data
    if we forced text-first ordering."""

    async def _run():
        chunks = [
            _tool_call_chunk(0, id="c1", name="ping"),
            _tool_call_chunk(0, arg_frag="{}"),
            _make_openai_chunk("then text"),
            _make_openai_chunk(finish_reason="tool_calls"),
            b"data: [DONE]\n",
        ]
        return await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))

    out = asyncio.run(_run())
    text = out.decode()
    # Tool block opens at index 0.
    tool_open = text.index('"content_block": {"type": "tool_use"')
    # Text block opens later (at index 1).
    text_open = text.index('"content_block": {"type": "text"')
    assert tool_open < text_open
    # And it's specifically indexed at 0:
    snippet = text[tool_open - 40 : tool_open]
    assert '"index": 0' in snippet


def test_sse_parallel_tool_calls():
    """Two tool calls streamed in interleaved order — each becomes its
    own content block at a unique Anthropic index."""

    async def _run():
        chunks = [
            _tool_call_chunk(0, id="a", name="f1"),
            _tool_call_chunk(1, id="b", name="f2"),
            _tool_call_chunk(0, arg_frag='{"x":1}'),
            _tool_call_chunk(1, arg_frag='{"y":2}'),
            _make_openai_chunk(finish_reason="tool_calls"),
            b"data: [DONE]\n",
        ]
        return await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))

    out = asyncio.run(_run())
    text = out.decode()
    assert "f1" in text
    assert "f2" in text
    assert text.count("event: content_block_stop") == 2


def test_sse_tool_call_without_arg_fragments():
    """A tool call with no arguments at all — just id + name, then finish.
    Should emit content_block_start + content_block_stop, no deltas."""

    async def _run():
        chunks = [
            _tool_call_chunk(0, id="x", name="ping"),
            _make_openai_chunk(finish_reason="tool_calls"),
            b"data: [DONE]\n",
        ]
        return await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))

    out = asyncio.run(_run())
    text = out.decode()
    assert "ping" in text
    assert "event: content_block_start" in text
    assert "event: content_block_stop" in text
    assert "input_json_delta" not in text


def test_sse_tool_call_no_explicit_id_synthesizes_one():
    """If the engine doesn't include an `id` in the first tool_call
    chunk, the translator should synthesize a `toolu_*` id."""

    async def _run():
        chunks = [
            (
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "type": "function",
                                            "function": {"name": "ping"},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + "\n"
            ).encode(),
            _make_openai_chunk(finish_reason="tool_calls"),
            b"data: [DONE]\n",
        ]
        return await _collect(openai_to_anthropic_sse(_make_chunk_iter(chunks), model="m"))

    out = asyncio.run(_run())
    text = out.decode()
    assert '"id": "toolu_' in text
