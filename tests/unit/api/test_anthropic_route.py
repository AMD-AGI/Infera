###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Integration tests for the `POST /v1/messages` route.

Uses FastAPI TestClient with a fake router that captures the
translated body + returns canned engine responses. Verifies:

- 400 on tools / multimodal
- Translated body lands at the router with the right shape
- Non-streaming OpenAI response → Anthropic JSON
- Streaming OpenAI SSE → Anthropic SSE
- `anthropic-version` + auth headers accepted (not rejected)
- Request-id correlation header round-trips
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from infera.router.base import BaseRouter
from infera.server import app as app_module
from infera.server.app import init_app


class _FakeRouter(BaseRouter):
    """Captures the dispatched body and returns a canned response."""

    def __init__(
        self,
        *,
        canned_response: dict[str, Any] | None = None,
        stream_chunks: list[bytes] | None = None,
    ) -> None:
        # Skip parent __init__ — we don't have pool/policy here.
        self.last_body: dict | None = None
        self.last_path: str | None = None
        self.last_stream: bool | None = None
        self._canned = canned_response
        self._stream_chunks = stream_chunks

    async def aclose(self) -> None:
        return

    async def dispatch(self, body, *, stream, path):
        self.last_body = body
        self.last_path = path
        self.last_stream = stream
        if stream:

            async def _gen():
                for c in self._stream_chunks or []:
                    yield c

            return StreamingResponse(_gen(), media_type="text/event-stream")
        return JSONResponse(content=self._canned or {})


@pytest.fixture
def app_and_router():
    """Fresh app + fake router for each test."""
    router = _FakeRouter()
    # init_app mutates module-level globals; reset between tests.
    init_app(reg=None, rtr=router, kv=None)  # type: ignore[arg-type]
    yield app_module.app, router


# ----------------------------------------------------------------------
# Refused-feature paths
# ----------------------------------------------------------------------


def test_route_accepts_tools_now(app_and_router):
    """Tools used to be rejected with 400; now they're translated.
    Regression net for tools support."""
    app, router = app_and_router
    router._canned = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {},
    }
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        headers={"anthropic-version": "2023-06-01"},
        json={
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "x", "description": "foo", "input_schema": {"type": "object"}}],
        },
    )
    assert r.status_code == 200
    assert router.last_body is not None
    assert router.last_body["tools"][0]["type"] == "function"
    assert router.last_body["tools"][0]["function"]["name"] == "x"


def test_route_rejects_image_inside_tool_result_with_400(app_and_router):
    """tool_result.content can itself carry image
    blocks — those must also be rejected. Without this the outer
    accept would silently drop the screenshot in the tool reply."""
    app, _ = app_and_router
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu1",
                            "content": [
                                {"type": "text", "text": "see image:"},
                                {"type": "image", "source": {"data": "..."}},
                            ],
                        }
                    ],
                }
            ],
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert "image" in body["error"]["message"].lower()


def test_route_rejects_tool_result_missing_id_with_400(app_and_router):
    """tool_result must reference an earlier
    tool_use.id. Without it OpenAI's role:'tool' message can't be
    matched back."""
    app, _ = app_and_router
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "result text"}],
                }
            ],
        },
    )
    assert r.status_code == 400
    assert "tool_use_id" in r.json()["error"]["message"]


def test_route_rejects_image_block_with_400(app_and_router):
    app, _ = app_and_router
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe:"},
                        {"type": "image", "source": {"data": "..."}},
                    ],
                }
            ],
        },
    )
    assert r.status_code == 400


def test_route_rejects_malformed_json_with_400(app_and_router):
    app, _ = app_and_router
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        headers={"content-type": "application/json"},
        content=b"{not json",
    )
    assert r.status_code == 400


# ----------------------------------------------------------------------
# Non-streaming happy path
# ----------------------------------------------------------------------


def test_non_streaming_response_shape(app_and_router):
    app, router = app_and_router
    router._canned = {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "choices": [
            {"message": {"role": "assistant", "content": "the answer"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "model": "engine-name",
    }
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        headers={"anthropic-version": "2023-06-01", "x-api-key": "sk-anything"},
        json={
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "what is 2+2?"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "the answer"}]
    assert body["model"] == "claude-3-5-sonnet-20240620"  # echoed from request
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["input_tokens"] == 10
    assert body["usage"]["output_tokens"] == 5

    # Router got the translated body.
    assert router.last_path == "/v1/chat/completions"
    assert router.last_stream is False
    assert router.last_body is not None
    assert router.last_body["messages"][0]["role"] == "user"
    assert router.last_body["messages"][0]["content"] == "what is 2+2?"
    assert router.last_body["max_tokens"] == 16


def test_system_block_routed_as_system_message(app_and_router):
    app, router = app_and_router
    router._canned = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {},
    }
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 16,
            "system": [
                {
                    "type": "text",
                    "text": "you are helpful",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    # Translated body has system as first message.
    assert router.last_body["messages"][0] == {"role": "system", "content": "you are helpful"}


def test_request_id_round_trips(app_and_router):
    app, _ = app_and_router
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        headers={"X-Infera-Request-Id": "my-custom-id"},
        json={
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    # Empty canned response, but still 200; header should echo back.
    assert r.headers.get("X-Infera-Request-Id") == "my-custom-id"


def test_cache_hints_present_in_dispatched_body(app_and_router):
    """The Anthropic body has cache_control; the router-side
    parse_cache_hints should have stamped _infera_cache_hints onto
    the body that reaches the router.dispatch call."""
    app, router = app_and_router
    router._canned = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {},
    }
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 16,
            "system": [
                {
                    "type": "text",
                    "text": "stable",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            "messages": [{"role": "user", "content": "go"}],
        },
    )
    assert r.status_code == 200
    assert router.last_body is not None
    hints = router.last_body.get("_infera_cache_hints")
    assert hints is not None
    # ttl=1h → retention=long. The Retention enum compares equal to its value string.
    assert hints.retention.value == "long"


# ----------------------------------------------------------------------
# Streaming happy path
# ----------------------------------------------------------------------


def test_streaming_response_emits_anthropic_events(app_and_router):
    app, router = app_and_router
    router._stream_chunks = [
        b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n',
        b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        b"data: [DONE]\n",
    ]
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    raw = r.content
    # Check the canonical Anthropic event sequence appears in order.
    for tag in (
        b"event: message_start",
        b"event: content_block_start",
        b"event: content_block_delta",
        b"event: content_block_stop",
        b"event: message_delta",
        b"event: message_stop",
    ):
        assert tag in raw, f"missing event: {tag!r}"

    # Text payload arrived as a text_delta event.
    assert b'"text": "Hello"' in raw
    assert b'"text": " world"' in raw


def test_streaming_default_value(app_and_router):
    """stream=true is opt-in; without it the route must be non-streaming."""
    app, router = app_and_router
    router._canned = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {},
    }
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            # no stream field
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")


# ----------------------------------------------------------------------
# Headers
# ----------------------------------------------------------------------


def test_accepts_anthropic_version_unsupported_version_does_not_reject(app_and_router):
    """We log unsupported versions but don't 4xx them — that would
    break clients pinned to a newer beta we haven't certified yet."""
    app, router = app_and_router
    router._canned = {
        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
        "usage": {},
    }
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        headers={"anthropic-version": "2026-01-01"},  # made-up future version
        json={"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200


def test_accepts_auth_headers_without_validation(app_and_router):
    """v1 doesn't enforce auth; both x-api-key and Bearer are accepted."""
    app, router = app_and_router
    router._canned = {
        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
        "usage": {},
    }
    client = TestClient(app)
    for hdrs in (
        {"x-api-key": "sk-blah"},
        {"authorization": "Bearer tk-blah"},
    ):
        r = client.post(
            "/v1/messages",
            headers=hdrs,
            json={"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, hdrs


# ----------------------------------------------------------------------
# Engine error + streaming passthrough
# ----------------------------------------------------------------------


def test_route_forwards_engine_4xx_unchanged(app_and_router):
    """Forward engine 4xx/5xx verbatim instead
    of wrapping empty Anthropic content around them."""
    from fastapi.responses import JSONResponse as _JSONResponse

    app, _ = app_and_router

    class _ErrRouter(_FakeRouter):
        async def dispatch(self, body, *, stream, path):
            self.last_body = body
            return _JSONResponse(
                status_code=503,
                content={"error": "no active mixed worker for model='m'"},
            )

    err_router = _ErrRouter()
    init_app(reg=None, rtr=err_router, kv=None)  # type: ignore[arg-type]
    client = TestClient(app_module.app)
    r = client.post(
        "/v1/messages",
        json={"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503
    assert r.json() == {"error": "no active mixed worker for model='m'"}


def test_route_rejects_empty_model_with_400(app_and_router):
    app, _ = app_and_router
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={"model": "", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 400
    assert "model" in r.json()["error"]["message"].lower()


def test_route_rejects_missing_model_with_400(app_and_router):
    app, _ = app_and_router
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 400


def test_route_injects_stream_options_include_usage_for_streaming(app_and_router):
    """When stream=True the route must inject
    `stream_options: {include_usage: True}` so the engine emits a
    usage chunk we can fold into Anthropic's output_tokens."""
    app, router = app_and_router
    router._stream_chunks = [b"data: [DONE]\n"]
    client = TestClient(app)
    client.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert router.last_body["stream_options"]["include_usage"] is True


def test_route_request_id_used_for_msg_id(app_and_router):
    app, router = app_and_router
    router._canned = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {},
    }
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        headers={"X-Infera-Request-Id": "fixed-request-id"},
        json={"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.json()["id"] == "msg_fixed-request-id"


# ----------------------------------------------------------------------
# Tool calling — end-to-end through the route
# ----------------------------------------------------------------------


def test_route_tool_use_response_round_trip(app_and_router):
    """End-to-end: client asks (with tools), engine returns tool_calls,
    route translates back to Anthropic tool_use shape."""
    app, router = app_and_router
    router._canned = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
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
        "usage": {"prompt_tokens": 50, "completion_tokens": 12},
    }
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 100,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "look up weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            "messages": [{"role": "user", "content": "weather in NYC?"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "get_weather",
            "input": {"city": "NYC"},
        }
    ]


def test_route_tool_result_follow_up(app_and_router):
    """Client sends a tool_result; route translates to role:'tool'
    message; engine sees the expected OpenAI shape."""
    app, router = app_and_router
    router._canned = {
        "choices": [{"message": {"content": "It is 72°F."}, "finish_reason": "stop"}],
        "usage": {},
    }
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "72F"}],
                },
            ],
        },
    )
    assert r.status_code == 200
    assert router.last_body is not None
    messages = router.last_body["messages"]
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call_1"
    assert messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "72F"}


def test_route_streaming_tool_call(app_and_router):
    """End-to-end streaming: client gets Anthropic-shaped tool_use SSE
    when the engine streams OpenAI tool_call deltas."""
    app, router = app_and_router
    router._stream_chunks = [
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
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "get_weather", "arguments": ""},
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
                                        "function": {"arguments": '{"city":'},
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
                                        "function": {"arguments": '"NYC"}'},
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
        b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n',
        b"data: [DONE]\n",
    ]
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 100,
            "stream": True,
            "tools": [{"name": "get_weather", "input_schema": {}}],
            "messages": [{"role": "user", "content": "weather?"}],
        },
    )
    assert r.status_code == 200
    text = r.content.decode()
    for tag in (
        "event: message_start",
        "event: content_block_start",
        "event: content_block_delta",
        "event: content_block_stop",
        "event: message_delta",
        "event: message_stop",
    ):
        assert tag in text
    assert '"type": "tool_use"' in text
    assert '"name": "get_weather"' in text
    assert '"type": "input_json_delta"' in text
    assert '"partial_json": "{\\"city\\":"' in text
    assert '"stop_reason": "tool_use"' in text
