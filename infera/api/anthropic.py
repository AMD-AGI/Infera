###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Anthropic Messages API ↔ OpenAI Chat translation layer.

Infera's workers (vLLM, SGLang) only speak OpenAI Chat. Clients
like openclaw speak the Anthropic Messages API. This module bridges
the two so a request to `POST /v1/messages` can land on the same
worker that serves `/v1/chat/completions`.

## Scope (v1 + tools followup)

In scope:

- **Request translation**: Anthropic body → OpenAI body. Handles
  `messages` (text content), `system` (string or block array),
  `max_tokens`, `stop_sequences`, `temperature`, `top_p`,
  `metadata.user_id` → `user`, **tool definitions, tool_use,
  tool_result, and tool_choice**.
- **Response translation (streaming)**: OpenAI SSE chunks
  (`data: {choices:[{delta:{...}}]}`) → Anthropic events
  (`event: message_start | content_block_start |
   content_block_delta | content_block_stop | message_delta |
   message_stop`). Tool-call deltas are reassembled into Anthropic
  `tool_use` content blocks with `input_json_delta` fragments.
- **Response translation (non-streaming)**: OpenAI completion JSON
  → Anthropic Messages JSON. Tool calls become `tool_use` blocks.
- **Usage block translation**: `prompt_tokens` / `completion_tokens`
  → `input_tokens` / `output_tokens`. `cache_read_input_tokens` /
  `cache_creation_input_tokens` are left at 0 in v1 (need engine
  to emit per-request hit counts).
- `cache_control` annotations stay on the body — the existing
  router-side `parse_cache_hints` already understands them; the
  Anthropic-flavored bits are dropped before the OpenAI request
  is forwarded to the engine.

Out of scope (v1):

- **Multimodal content** (`image`/`audio`/`video`/`document` blocks)
  — explicit 400, including when nested inside `tool_result.content`.
  Follow-up.
- **Anthropic thinking blocks** (`anthropic-beta: interleaved-
  thinking-2025-05-14`). Will pass through as text in v1; the
  beta header is accepted but the rich `thinking` events aren't
  produced.

## Why a shim instead of a proxy

The router needs to see the request body to:
  1. Parse cache_control (already does, on Anthropic-shaped bodies).
  2. Apply KV-aware routing decisions.
  3. Inject engine priority.

So the translation has to happen at the router edge. We translate
INBOUND to OpenAI so engines accept it, and translate the engine's
OUTBOUND response back to Anthropic shape for the client.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from infera.common.logsafe import scrub

logger = logging.getLogger(__name__)


class AnthropicRequestRejected(Exception):
    """Raised when an Anthropic-shaped request uses a feature we don't
    yet support (multimodal) or violates a translation precondition
    (e.g. tool_result missing tool_use_id). Caller maps to HTTP 400
    with the message body."""


# ----------------------------------------------------------------------
# Request translation: Anthropic Messages → OpenAI Chat
# ----------------------------------------------------------------------


def anthropic_to_openai_request(body: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages request body into an OpenAI
    Chat request body suitable for vLLM / SGLang.

    Raises `AnthropicRequestRejected` for features not supported in
    v1 (multimodal content) or invariants violations (tool_result
    missing tool_use_id).

    Pass-through fields kept on the translated body so downstream
    machinery (kvd retention propagation, etc.) keeps working:
      - `metadata` (Anthropic carries user / session ids here)
      - `_infera_*` annotations (added by the route before calling)

    ## Tool calling

    Anthropic and OpenAI have different shapes; translation is
    deterministic:

    Tool definitions (request `tools[]`):
        Anthropic: ``{name, description, input_schema}``
        OpenAI:    ``{type: "function", function: {name, description, parameters}}``
        (``description`` is omitted on the OpenAI side when empty.)

    Tool choice (request `tool_choice`):
        ``{type:"auto"}``                → ``"auto"``
        ``{type:"any"}``                 → ``"required"`` (force a tool)
        ``{type:"tool", name:"X"}``      → ``{type:"function", function:{name:"X"}}``
        ``{type:"none"}``                → ``"none"``
        ``disable_parallel_tool_use:true`` → top-level
                                           ``parallel_tool_calls: false``

    Tool use in messages (assistant's prior `content` blocks):
        Anthropic:  ``content: [{type:"tool_use", id, name, input}]``
        OpenAI:     ``tool_calls: [{id, type:"function",
                                    function: {name, arguments: <json-str>}}]``

    Tool result in messages (user follow-up):
        Anthropic: ``role:"user", content:[{type:"tool_result",
                                            tool_use_id, content}]``
        OpenAI:    a new message with ``role:"tool",
                                       tool_call_id, content``
    """
    if not isinstance(body, dict):
        raise AnthropicRequestRejected("request body must be a JSON object")

    # Multimodal content (image/audio/video/document) stays rejected
    # in this PR — separate followup. Tools and tool_use/tool_result
    # are NOW supported, so they're explicitly removed from the
    # rejection set below. The guard ALSO recurses into
    # `tool_result.content` so tool-returned screenshots don't
    # silently degrade.
    _reject_if_multimodal(body)

    out: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": [],
    }

    # System block(s) — Anthropic accepts string OR array of content
    # blocks. OpenAI accepts a single system message at the start of
    # `messages`.
    system_text = _flatten_system(body.get("system"))
    if system_text:
        out["messages"].append({"role": "system", "content": system_text})

    # Messages — translate per-role. Tool content blocks make the
    # mapping non-trivial: an assistant message with `tool_use` becomes
    # an assistant message with `tool_calls` in OpenAI; a user message
    # with `tool_result` blocks becomes one or more `role:"tool"`
    # messages (OpenAI's tool result vehicle), plus any leftover text
    # as a regular `role:"user"` message.
    for msg in body.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            # Anthropic doesn't have a "system" role inside messages
            # (system lives at top level). Skip anything else.
            continue
        translated = _translate_anthropic_message(role, msg.get("content"))
        out["messages"].extend(translated)

    # Tool definitions
    tools_in = body.get("tools")
    if isinstance(tools_in, list) and len(tools_in) > 0:
        out["tools"] = [_translate_tool_def(t) for t in tools_in if isinstance(t, dict)]

    # Tool choice
    if "tool_choice" in body:
        tc = _translate_tool_choice(body["tool_choice"])
        if tc is not None:
            out["tool_choice"] = tc
        # Anthropic's tool_choice sometimes carries
        # ``disable_parallel_tool_use: true``. OpenAI expresses the
        # same intent via a TOP-LEVEL ``parallel_tool_calls: false``
        # field. Hoist it so engines that honor the OpenAI shape see it.
        if isinstance(body["tool_choice"], dict):
            if body["tool_choice"].get("disable_parallel_tool_use") is True:
                out["parallel_tool_calls"] = False

    # Direct field translation. Anthropic field on left, OpenAI on right:
    #   max_tokens          → max_tokens          (same name)
    #   stop_sequences      → stop
    #   temperature         → temperature
    #   top_p               → top_p
    #   top_k               → NOT in OpenAI chat (dropped — vLLM/SGLang
    #                         accept it as an extra so keep it; both
    #                         engines tolerate unknown fields gracefully)
    #   stream              → stream
    #   metadata.user_id    → user
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    if "stop_sequences" in body and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if "temperature" in body:
        out["temperature"] = body["temperature"]
    if "top_p" in body:
        out["top_p"] = body["top_p"]
    if "top_k" in body:
        out["top_k"] = body[
            "top_k"
        ]  # engine-extra; harmless on OpenAI receivers that ignore unknown
    if "stream" in body:
        out["stream"] = bool(body["stream"])
    md = body.get("metadata") or {}
    if isinstance(md, dict) and isinstance(md.get("user_id"), str):
        out["user"] = md["user_id"]

    return out


def _flatten_system(system: Any) -> str:
    """Anthropic system can be a string OR an array of `{type:"text",text:"..."}` blocks."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for entry in system:
            if isinstance(entry, dict) and entry.get("type") == "text":
                text = entry.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _flatten_message_content(content: Any) -> str:
    """Anthropic message content is either a string or an array of
    content blocks. v1 supports text blocks only — multimodal blocks
    trigger `AnthropicRequestRejected` upstream."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


_MM_TYPES = frozenset({"image", "audio", "video", "document"})


def _reject_if_multimodal(body: dict[str, Any]) -> None:
    """Raise if any message content block is non-text-or-tool (image,
    audio, video, document). Multimodal needs the MM silent-
    corruption guard wiring before we can route it correctly.

    Recurses into ``tool_result.content`` blocks: tool-returned
    screenshots/PDFs are also a silent-drop hazard if we accept
    the outer message but quietly discard the image children.
    Both layers need the same rejection.

    NOTE: tool_use / tool_result are SUPPORTED now (this PR) so they
    are NOT in the rejection set."""
    for msg in body.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in _MM_TYPES:
                raise AnthropicRequestRejected(
                    f"content type {block_type!r} not supported in this Infera build. "
                    "Multimodal blocks land in a followup PR. "
                    "Submit text or tool blocks to proceed."
                )
            # tool_result.content can itself be a list of blocks; the
            # Anthropic spec allows image/document there. Recurse one
            # level so a tool that returns a screenshot doesn't get
            # silently degraded to an empty tool response.
            if block_type == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    for sub in inner:
                        if not isinstance(sub, dict):
                            continue
                        sub_type = sub.get("type")
                        if sub_type in _MM_TYPES:
                            raise AnthropicRequestRejected(
                                f"tool_result returned a {sub_type!r} block, which is not "
                                "supported in this Infera build. Multimodal tool outputs "
                                "land in a followup PR. Convert tool output to text first."
                            )


# ----------------------------------------------------------------------
# Tool / tool_choice request translation (Anthropic → OpenAI)
# ----------------------------------------------------------------------


def _translate_tool_def(t: dict[str, Any]) -> dict[str, Any]:
    """Anthropic tool definition → OpenAI ``{type:"function", function:{...}}``.

    Omits ``description`` entirely when the source had no description
    (or an empty string). Some strict OpenAI receivers treat an empty
    description as a real field and surface it back to the model,
    which is noisier than just not declaring it.
    """
    fn: dict[str, Any] = {
        "name": t.get("name", ""),
        # KEY rename: Anthropic ``input_schema`` → OpenAI ``parameters``.
        "parameters": t.get("input_schema") or {},
    }
    desc = t.get("description")
    if isinstance(desc, str) and desc:
        fn["description"] = desc
    return {"type": "function", "function": fn}


def _translate_tool_choice(tc: Any) -> Any:
    """Anthropic tool_choice → OpenAI tool_choice.

    auto                    → "auto"
    any                     → "required"
    tool with name "X"      → {"type":"function","function":{"name":"X"}}
    none                    → "none"
    None / unknown          → None  (drop the field; engine picks).
                              Unknown types log a warning so operators
                              notice when a client uses a new shape.
    """
    if not isinstance(tc, dict):
        return None
    t = tc.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "none":
        return "none"
    if t == "tool":
        name = tc.get("name")
        if isinstance(name, str) and name:
            return {"type": "function", "function": {"name": name}}
        logger.warning(
            "anthropic tool_choice type=tool with empty/missing name — "
            "falling back to engine default"
        )
        return None
    # Anthropic may add new tool_choice types; rather than 400, drop
    # the field and let the engine pick. Log so operators can spot
    # a client using something we haven't translated yet.
    logger.warning(
        "anthropic tool_choice type=%s not recognized — falling back to engine default",
        scrub(t),
    )
    return None


def _translate_anthropic_message(role: str, content: Any) -> list[dict[str, Any]]:
    """Convert one Anthropic message → one or more OpenAI messages.

    Most messages become a single OpenAI message. The exception is a
    user message with multiple `tool_result` blocks: each tool_result
    becomes its own OpenAI ``role:"tool"`` message. Any leftover text
    in that user message becomes a separate ``role:"user"`` message.

    Returns the list of OpenAI-shape messages in order.
    """
    # Plain-string content fast path.
    if isinstance(content, str):
        return [{"role": role, "content": content}]

    if not isinstance(content, list):
        return [{"role": role, "content": ""}]

    if role == "assistant":
        # Collect text + tool_use into ONE assistant message with
        # `content` + `tool_calls` (OpenAI's combined shape).
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif bt == "tool_use":
                # Anthropic's `input` is a dict; OpenAI's `arguments` is a
                # json-string of that dict. The engine expects a string
                # so always serialize here, even if the dict is empty.
                tool_calls.append(
                    {
                        "id": block.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
        msg: dict[str, Any] = {"role": "assistant"}
        if text_parts:
            msg["content"] = "\n".join(text_parts)
        if tool_calls:
            msg["tool_calls"] = tool_calls
        # OpenAI requires `content` even if null when tool_calls present.
        if "content" not in msg:
            msg["content"] = None
        return [msg]

    # role == "user". Two flavors:
    # 1. Plain user message: text blocks (possibly mixed with cache_control)
    # 2. Tool result follow-up: tool_result blocks
    text_parts = []
    tool_messages: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt == "text":
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        elif bt == "tool_result":
            # OpenAI requires `tool_call_id` to link a tool message
            # back to the assistant's earlier tool_calls[].id. A
            # missing id would silently mismatch and the engine
            # would either error or apply the result to the wrong
            # call. Reject at the front door instead.
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                raise AnthropicRequestRejected(
                    "tool_result block missing required 'tool_use_id' — each tool_result "
                    "must reference the assistant's prior tool_use.id."
                )
            # Each tool_result becomes its own OpenAI tool message.
            # Anthropic's content can be a string OR an array of text blocks.
            tr_content = block.get("content")
            if isinstance(tr_content, list):
                tr_text = "\n".join(
                    b.get("text", "")
                    for b in tr_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            elif isinstance(tr_content, str):
                tr_text = tr_content
            else:
                tr_text = ""
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": tr_text,
                }
            )
    out: list[dict[str, Any]] = []
    if text_parts:
        out.append({"role": "user", "content": "\n".join(text_parts)})
    # OpenAI accepts tool messages between user/assistant turns. Order
    # matches Anthropic's content array order.
    out.extend(tool_messages)
    if not out:
        # Empty content — emit an empty user message so the turn
        # structure is preserved.
        out.append({"role": "user", "content": ""})
    return out


# ----------------------------------------------------------------------
# Response translation (non-streaming): OpenAI completion → Anthropic Messages
# ----------------------------------------------------------------------


def openai_to_anthropic_response(
    openai_response: dict[str, Any],
    *,
    model: str,
    msg_id_hint: str | None = None,
) -> dict[str, Any]:
    """Translate an OpenAI chat completion JSON response into an
    Anthropic Messages JSON response.

    `model` is taken from the request because OpenAI may normalize
    it in the response. `msg_id_hint` (the infera request id)
    becomes part of the Anthropic message id so server↔client logs
    correlate — falls back to a random uuid if not provided.

    Tool calls: an OpenAI assistant message with `tool_calls` becomes
    an Anthropic content array with `text` blocks (if any content
    text) followed by `tool_use` blocks (one per OpenAI tool call).
    `finish_reason == "tool_calls"` maps to `stop_reason == "tool_use"`.
    """
    choices = openai_response.get("choices") or []
    first = choices[0] if choices else {}
    msg = first.get("message") or {}
    content_text = msg.get("content") or ""
    finish_reason = first.get("finish_reason")
    usage = openai_response.get("usage") or {}

    msg_id = (
        f"msg_{msg_id_hint[:24]}"
        if isinstance(msg_id_hint, str) and msg_id_hint
        else f"msg_{uuid.uuid4().hex[:24]}"
    )

    # Build Anthropic content array. Text (if any) first, then any
    # tool_use blocks (matching Anthropic's response shape).
    anthropic_content: list[dict[str, Any]] = []
    if content_text:
        anthropic_content.append({"type": "text", "text": content_text})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        # OpenAI emits arguments as a JSON string; Anthropic emits
        # input as a dict. Parse defensively — engines occasionally
        # ship malformed JSON during partial decodes, so don't crash.
        args_str = fn.get("arguments") or "{}"
        try:
            args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
            if not isinstance(args_obj, dict):
                # Engine returned valid JSON that isn't an object (string,
                # number, array). Tool inputs are required to be objects;
                # log and substitute {} so the client sees a well-formed
                # tool_use block instead of a 500.
                logger.warning(
                    "tool_call %r returned non-object arguments (type=%s) — substituting {}",
                    fn.get("name", ""),
                    type(args_obj).__name__,
                )
                args_obj = {}
        except json.JSONDecodeError as exc:
            logger.warning(
                "tool_call %r returned malformed arguments JSON: %s — substituting {}",
                fn.get("name", ""),
                exc,
            )
            args_obj = {}
        anthropic_content.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:20]}",
                "name": fn.get("name", ""),
                "input": args_obj,
            }
        )
    # Empty response case (engine returned no content or tool_calls).
    if not anthropic_content:
        anthropic_content.append({"type": "text", "text": ""})

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": anthropic_content,
        "model": model,
        "stop_reason": _map_finish_reason(finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or 0,
            # v1: we don't propagate per-request cache hit counts yet —
            # engines don't expose them on the OpenAI usage block. Set
            # to 0 so Anthropic-savvy clients have the keys present.
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def _map_finish_reason(openai_reason: str | None) -> str | None:
    """OpenAI finish_reason → Anthropic stop_reason.

    Unknown OpenAI values fall back to `end_turn` with a log line
    so operators see drift (e.g. SGLang's `eos_token`, vLLM's
    `abort`). `content_filter` had been mapped to `stop_sequence`
    which is wrong — Anthropic doesn't expose a moderation stop;
    treat as `end_turn`. PR #13 review fix P1.
    """
    if not openai_reason:
        return None
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
        "tool_calls": "tool_use",
    }
    if openai_reason in mapping:
        return mapping[openai_reason]
    logger.info(
        "openai finish_reason=%r not in our mapping; falling back to end_turn",
        openai_reason,
    )
    return "end_turn"


# ----------------------------------------------------------------------
# Response translation (streaming): OpenAI SSE → Anthropic SSE
# ----------------------------------------------------------------------
#
# OpenAI streaming format:
#
#   data: {"id":"...","object":"chat.completion.chunk",
#          "choices":[{"delta":{"role":"assistant","content":""}}]}
#   data: {"choices":[{"delta":{"content":"Hello"}}]}
#   data: {"choices":[{"delta":{"content":" world"}}]}
#   data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
#   data: [DONE]
#
# Anthropic streaming format (each event is two SSE lines):
#
#   event: message_start
#   data: {"type":"message_start","message":{"id":"...","role":"assistant", "usage":{...}}}
#
#   event: content_block_start
#   data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
#
#   event: content_block_delta
#   data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}
#
#   event: content_block_stop
#   data: {"type":"content_block_stop","index":0}
#
#   event: message_delta
#   data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}
#
#   event: message_stop
#   data: {"type":"message_stop"}


async def openai_to_anthropic_sse(
    openai_chunks: AsyncIterable[bytes],
    *,
    model: str,
    msg_id_hint: str | None = None,
) -> AsyncIterator[bytes]:
    """Pump OpenAI SSE chunks → Anthropic SSE events.

    Caller passes the engine's response body iterator. Yields bytes
    suitable for a `StreamingResponse(media_type="text/event-stream")`.

    `msg_id_hint` (the infera request id) becomes the Anthropic
    message id so client/server logs correlate.

    Handles two interleaved kinds of OpenAI delta content:

      1. Text — `delta.content` is a string. Emitted as an Anthropic
         text content block (`{"type":"text"}`) with `content_block_
         delta` carrying `{"type":"text_delta","text":...}`.
      2. Tool calls — `delta.tool_calls[]` arrives in pieces: a first
         chunk with `id`, `type`, `function.name`, then arbitrary
         many chunks each appending to `function.arguments`. Each
         OpenAI tool_call index becomes its own Anthropic content
         block (`{"type":"tool_use"}`); argument fragments are
         emitted as `input_json_delta` deltas — the JSON is a
         valid prefix at each step, NOT a complete value.

    Block-index policy: FCFS. Whichever block type (text vs.
    tool_use) we observe first in the OpenAI delta stream takes
    Anthropic block index 0; the next distinct block takes 1, and
    so on. A request that emits text before tools yields text at 0
    and tool blocks at 1+. A request whose engine emits tool deltas
    first (no preceding text) yields the first tool at index 0 and
    any later text at the next free index. We do NOT reserve index
    0 for text — that would break clients that key on the index to
    map back to specific content blocks.

    Robustness:
      - tolerates blank lines, `:` keepalive comments, malformed JSON
        (skipped silently — the alternative is killing the stream).
      - emits closers (`content_block_stop` for every open block,
        plus `message_delta` + `message_stop`) on `[DONE]` or stream
        end-of-input.
      - gates the `[DONE]`/end-of-stream closers on `started`: if the
        engine sends ONLY a usage-only chunk followed by `[DONE]`
        (no choices ever), we don't emit `message_stop` without
        `message_start` — that would be an Anthropic protocol
        violation. PR #13 review fix P1.
    """
    msg_id = (
        f"msg_{msg_id_hint[:24]}"
        if isinstance(msg_id_hint, str) and msg_id_hint
        else f"msg_{uuid.uuid4().hex[:24]}"
    )
    started = False
    text_index: int | None = None  # Anthropic block index for the text block, if any
    # Map OpenAI delta.tool_calls[].index → Anthropic content block index.
    tool_index_map: dict[int, int] = {}
    next_anthropic_block_index = 0
    finish_reason: str | None = None
    output_tokens = 0  # tracked from streamed usage when present

    async def _yield_event(event_type: str, payload: dict[str, Any]) -> bytes:
        body = json.dumps({"type": event_type, **payload})
        return f"event: {event_type}\ndata: {body}\n\n".encode()

    async def _close_all_blocks() -> AsyncIterator[bytes]:
        """Emit content_block_stop for every block we opened."""
        # Use a stable order: text (if any) first, then tool blocks in
        # ascending Anthropic-index order. Order doesn't matter for
        # protocol correctness as long as every opened block gets a
        # matching stop, but keeping it deterministic helps tests.
        indices: list[int] = []
        if text_index is not None:
            indices.append(text_index)
        indices.extend(sorted(tool_index_map.values()))
        for idx in indices:
            yield await _yield_event("content_block_stop", {"index": idx})

    # Buffer for partial SSE lines split across chunks.
    buf = b""
    async for chunk in openai_chunks:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith(b":"):  # SSE comment / keepalive
                continue
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:") :].strip()
            if data == b"[DONE]":
                # Engine finished. Only emit closers if we actually
                # emitted `message_start` — otherwise we'd produce a
                # `message_stop` with no `message_start`, which
                # Anthropic clients reject as a protocol violation.
                if not started:
                    return
                async for ev in _close_all_blocks():
                    yield ev
                yield await _yield_event(
                    "message_delta",
                    {
                        "delta": {
                            "stop_reason": _map_finish_reason(finish_reason),
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": output_tokens},
                    },
                )
                yield await _yield_event("message_stop", {})
                return

            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                # Skip malformed chunks rather than killing the stream.
                continue

            choices = obj.get("choices") or []
            if not choices:
                # Some OpenAI implementations send a final chunk with
                # `usage` only and no choices — capture token count.
                # `completion_tokens` may legitimately be None mid-
                # stream; coerce to 0 before `max()` to avoid TypeError.
                u = obj.get("usage") or {}
                if isinstance(u, dict):
                    n = u.get("completion_tokens") or 0
                    output_tokens = max(output_tokens, n)
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            chunk_finish = choice.get("finish_reason")
            if chunk_finish:
                finish_reason = chunk_finish

            # First non-trivial chunk → emit message_start (Anthropic
            # requires it before any content_block_*).
            if not started:
                yield await _yield_event(
                    "message_start",
                    {
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        }
                    },
                )
                started = True

            # --- Text delta path ---
            text_delta = delta.get("content")
            if isinstance(text_delta, str) and text_delta:
                if text_index is None:
                    text_index = next_anthropic_block_index
                    next_anthropic_block_index += 1
                    yield await _yield_event(
                        "content_block_start",
                        {
                            "index": text_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                yield await _yield_event(
                    "content_block_delta",
                    {
                        "index": text_index,
                        "delta": {"type": "text_delta", "text": text_delta},
                    },
                )

            # --- Tool-call delta path ---
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    openai_idx = tc.get("index", 0)
                    fn = tc.get("function") or {}
                    # First chunk for this tool call carries id + name.
                    if openai_idx not in tool_index_map:
                        anthro_idx = next_anthropic_block_index
                        next_anthropic_block_index += 1
                        tool_index_map[openai_idx] = anthro_idx
                        yield await _yield_event(
                            "content_block_start",
                            {
                                "index": anthro_idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:20]}",
                                    "name": fn.get("name", ""),
                                    # Anthropic clients expect `input` to
                                    # always be present even if empty;
                                    # subsequent input_json_delta events
                                    # accumulate into the final value.
                                    "input": {},
                                },
                            },
                        )
                    # Argument fragment — emit only when non-empty.
                    args_frag = fn.get("arguments")
                    if isinstance(args_frag, str) and args_frag:
                        yield await _yield_event(
                            "content_block_delta",
                            {
                                "index": tool_index_map[openai_idx],
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": args_frag,
                                },
                            },
                        )

    # Engine closed the stream without [DONE]. Best effort: emit
    # closers so the client sees a clean termination.
    if started:
        async for ev in _close_all_blocks():
            yield ev
        yield await _yield_event(
            "message_delta",
            {
                "delta": {
                    "stop_reason": _map_finish_reason(finish_reason),
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": output_tokens},
            },
        )
        yield await _yield_event("message_stop", {})


def now_ts_iso() -> str:
    """Helper used in tests / logs."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
