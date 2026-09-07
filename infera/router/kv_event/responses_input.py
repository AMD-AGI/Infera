###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""`/v1/responses` bodies, normalised into the chat body the engine renders.

The Responses API carries its conversation in ``input``, not ``messages``.
Every renderer in ``block_hasher`` keys off ``messages``/``prompt``, so a
Responses body hashes to nothing and kv-aware silently degrades to load
balancing -- measured on a live Kimi-K3 PD fleet as ``request_blocks=0`` on 462
of 528 routing decisions, every one of them Codex orchestration traffic, while
``/v1/chat/completions`` through the same router scored up to 94/118 hits.

The Rust router hand-ports ``OpenAIServingResponses._make_request`` because it
cannot import Python. This one can, and does: the normalisation below is the
engine's own ``_construct_input_messages`` and ``_response_tools_to_chat_tools``,
called directly. That is the same call the rest of this package makes for
``normalize_assistant_tool_call_arguments`` and the dsv4 encoder, and for the
same reason -- a second implementation of a prefix transformation is a second
thing to keep byte-aligned, and nothing reports it when the two drift.

Both of those are effectively static: ``_construct_input_messages`` touches
``self`` only for two sibling helpers and for ``msg_store``, which is reached
only on the ``previous_response_id`` path this module refuses outright. So an
uninitialised instance is enough to borrow them from, and nothing here starts a
server.

Refuses (returns ``None``) rather than guessing, matching ``render_dsv4``: a
prefix a few tokens off the engine's does not error, it silently never hits the
cache again, and refusing costs only the load-based routing an empty render
already gives.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def is_responses_body(body: dict) -> bool:
    """True when this body is a Responses request rather than a chat one."""
    return body.get("messages") is None and body.get("input") is not None


def to_chat_body(body: dict) -> dict | None:
    """`_construct_input_messages` + `_make_request`, as a chat body.

    ``None`` means "cannot reproduce" -- the caller must fall back to load
    routing.
    """
    # Conversation history lives in the engine's in-process `msg_store`, keyed
    # by a response id the router never sees. Nothing here can reconstruct it,
    # and a body missing its history hashes a prefix that is real but wrong --
    # it would route to a worker holding a *different* conversation. Refuse.
    # (`store=true` is what makes a *later* turn carry previous_response_id;
    # this turn is still self-contained, so it stays reproducible.)
    if body.get("previous_response_id") is not None:
        return None

    try:
        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
        from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
    except Exception as exc:  # sglang not installed, or too old
        logger.debug("kv-aware: no sglang Responses support to borrow: %s", exc)
        return None

    try:
        request = ResponsesRequest(**body)
    except Exception as exc:
        # A body the engine would 422. Rendering it would cache a prefix for a
        # request that never runs.
        logger.debug("kv-aware: not a valid Responses body: %s", exc)
        return None

    # The harmony path (`_make_request_with_harmony`) is not modelled:
    # `use_harmony` is `hf_config.model_type == "gpt_oss"`, a model-level
    # property, and gpt-oss ships a Jinja chat template the ordinary path
    # handles.
    shim: Any = OpenAIServingResponses.__new__(OpenAIServingResponses)
    try:
        messages = shim._construct_input_messages(request, None)
        chat_tools = shim._response_tools_to_chat_tools(request)
    except Exception as exc:
        logger.debug("kv-aware: Responses input not reproducible: %s", exc)
        return None

    out: dict = {"messages": messages}
    if body.get("model") is not None:
        out["model"] = body["model"]
    # `tools=chat_tools or None` -- an all-builtin tool list yields no chat
    # tools. They arrive as pydantic `Tool`s; the chat path re-dumps them, so
    # hand on the dumps rather than the models.
    if chat_tools:
        out["tools"] = [t.model_dump() if hasattr(t, "model_dump") else t for t in chat_tools]
        out["tool_choice"] = _chat_tool_choice(_effective_tool_choice(request))
    else:
        # `tool_choice=... if chat_tools else "none"`.
        out["tool_choice"] = "none"
    # `reasoning_effort=request.reasoning.effort if request.reasoning else None`.
    effort = getattr(getattr(request, "reasoning", None), "effort", None)
    if effort is not None:
        out["reasoning_effort"] = effort
    # `chat_template_kwargs=request.chat_template_kwargs` (serving_responses.py:592).
    # Read off the *validated* request, not the raw body: `ResponsesRequest`'s
    # `normalize_reasoning_to_thinking` validator folds `reasoning.effort ==
    # "none"` into this dict, and dropping it renders a thinking preamble the
    # engine does not.
    ctk = getattr(request, "chat_template_kwargs", None)
    if isinstance(ctk, dict) and ctk:
        out["chat_template_kwargs"] = dict(ctk)
    return out


def normalised(body: dict) -> dict:
    """`body` in the shape every renderer downstream expects.

    A chat body passes through; a Responses body becomes the chat body
    `_make_request` builds. A Responses body we cannot reproduce comes back
    unchanged -- still a Responses body, so `token_ids_for` refuses it exactly
    as it would have.

    Exists so the callers that rewrite a body BEFORE hashing it -- the policy
    and the render probe, both applying a `RenderVariant` -- can run in the
    engine's order. The engine normalises first (`_make_request`) and merges
    `--default-chat-template-kwargs` second (`_process_messages`, which
    `OpenAIServingResponses` inherits and calls on this path too). A variant
    applied to a Responses body writes `chat_template_kwargs` and a promoted
    top-level `reasoning_effort` onto fields the normalisation then rebuilds
    from scratch, so the whole variant is dropped for `/v1/responses` and only
    for `/v1/responses` -- a fleet launched with `--default-chat-template-kwargs`
    hashes chat right and Responses wrong, in block 0.
    """
    if not is_responses_body(body):
        return body
    return to_chat_body(body) or body


def _effective_tool_choice(request: Any) -> Any:
    """`ResponsesRequest.effective_tool_choice()`.

    Reduces `tool_choice` to what the server can honour: of the object forms
    only a named `function` survives; web_search / mcp / ... become "auto".
    Falls back to the raw field on an sglang too old to have the method.
    """
    fn = getattr(request, "effective_tool_choice", None)
    if callable(fn):
        return _plain(fn())
    return _plain(getattr(request, "tool_choice", None))


def _chat_tool_choice(tool_choice: Any) -> Any:
    """`OpenAIServingResponses._chat_tool_choice` (serving_responses.py:988).

    Re-nests `{"type":"function","name":X}` into the chat spelling
    `{"type":"function","function":{"name":X}}`. Not cosmetic: `_chat_tools`
    narrows the tool list by reading `tool_choice["function"]["name"]`, so the
    flat form narrows nothing and the router renders every tool where the
    engine renders one.
    """
    if not isinstance(tool_choice, dict):
        return tool_choice
    name = tool_choice.get("name")
    if name is None:
        return tool_choice
    return {"type": "function", "function": {"name": name}}


def _plain(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value
