###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""`/v1/responses` bodies, normalised into the chat body the engine renders.

Every assertion here is about a field that reaches the *template*. A field this
module drops does not raise and does not log: the router renders a prefix the
engine never built, the block hashes stop matching, and kv-aware degrades to
load balancing while every health signal stays green. So these read as
"and this field survived", which is the whole contract.

The render-parity corpus cannot stand in for them. Its Responses goldens are
produced by a generator that imports `to_chat_body` itself, so a field this
function drops is dropped from the golden too and the comparison passes.
"""

from __future__ import annotations

import pytest

from infera.router.kv_event import responses_input
from infera.router.kv_event.render_variant import RenderVariant

pytest.importorskip("sglang.srt.entrypoints.openai.serving_responses")


def _chat(body: dict) -> dict:
    out = responses_input.to_chat_body(body)
    assert out is not None, "the engine renders this body; declining to is a 0% hit rate"
    return out


# ---- tool_choice ----------------------------------------------------------
#
# `_make_request` passes `self._chat_tool_choice(request.effective_tool_choice())`,
# and both renderers read the result: `_chat_tools` drops the tool block on
# "none" and narrows the list to one function on a named choice.


def test_a_named_tool_choice_is_nested_the_way_chat_spells_it():
    """`{"type":"function","name":X}` -> `{"type":"function","function":{"name":X}}`.

    Flat, `_chat_tools` reads `choice["function"]["name"]`, finds nothing, and
    narrows nothing -- so the router renders both tools where the engine
    renders one, and diverges inside the tool block at the front of the prompt.
    """
    out = _chat(
        {
            "input": "hi",
            "tools": [
                {"type": "function", "name": "a", "parameters": {"type": "object"}},
                {"type": "function", "name": "b", "parameters": {"type": "object"}},
            ],
            "tool_choice": {"type": "function", "name": "b"},
        }
    )
    assert out["tool_choice"] == {"type": "function", "function": {"name": "b"}}


def test_a_tool_choice_the_server_cannot_force_becomes_auto():
    """`effective_tool_choice`: of the object forms only a named `function`
    survives; web_search / mcp / ... cannot go through the tool-call parser."""
    out = _chat(
        {
            "input": "hi",
            "tools": [{"type": "function", "name": "a", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "mcp", "server_label": "s"},
        }
    )
    assert out["tool_choice"] == "auto"


def test_tool_choice_none_survives_and_suppresses_the_tool_block():
    out = _chat(
        {
            "input": "hi",
            "tools": [{"type": "function", "name": "a", "parameters": {"type": "object"}}],
            "tool_choice": "none",
        }
    )
    assert out["tool_choice"] == "none"


def test_a_body_with_no_chat_tools_forces_none():
    """`tool_choice=... if chat_tools else "none"` -- an all-builtin tool list
    yields no chat tools, so the engine sends "none" whatever the client asked."""
    out = _chat({"input": "hi", "tools": [{"type": "web_search"}], "tool_choice": "required"})
    assert out["tool_choice"] == "none"
    assert "tools" not in out


# ---- chat_template_kwargs -------------------------------------------------


def test_chat_template_kwargs_are_forwarded():
    """`chat_template_kwargs=request.chat_template_kwargs` (serving_responses.py:592).

    `template_context` spreads these into the Jinja scope, so dropping them
    renders a different preamble for every request that sets one.
    """
    out = _chat({"input": "hi", "chat_template_kwargs": {"enable_thinking": False}})
    assert out["chat_template_kwargs"] == {"enable_thinking": False}


def test_reasoning_effort_none_carries_the_thinking_toggle():
    """`ResponsesRequest.normalize_reasoning_to_thinking` folds `effort: "none"`
    into `chat_template_kwargs` before `_make_request` ever reads the field.
    Reading the raw body instead of the validated request misses it, and the
    router renders the thinking preamble the engine was just told to omit."""
    out = _chat({"input": "hi", "reasoning": {"effort": "none"}})
    ctk = out["chat_template_kwargs"]
    assert ctk["thinking"] is False
    assert ctk["enable_thinking"] is False


def test_an_explicit_kwarg_wins_over_the_thinking_toggle():
    """The validator spreads the client's dict last."""
    out = _chat(
        {
            "input": "hi",
            "reasoning": {"effort": "none"},
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )
    assert out["chat_template_kwargs"]["enable_thinking"] is True


def test_a_plain_body_carries_no_chat_template_kwargs_key():
    """Absent, not empty: `template_context` binds nothing, and an empty dict
    would still be a body shape the chat path never sees."""
    assert "chat_template_kwargs" not in _chat({"input": "hi"})


# ---- normalised(): the order the variant is applied in --------------------


def test_normalised_passes_a_chat_body_straight_through():
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert responses_input.normalised(body) is body


def test_normalised_hands_back_an_unreproducible_body_unchanged():
    """Still a Responses body, so `token_ids_for` refuses it exactly as before.
    Returning a half-built chat body instead would hash a prefix that is real
    but belongs to a *different* conversation."""
    body = {"model": "m", "input": "hi", "previous_response_id": "resp_1"}
    assert responses_input.normalised(body) is body


def test_the_variants_promoted_effort_survives_only_in_the_engines_order():
    """`--default-chat-template-kwargs` is merged by `_process_messages`, which
    `OpenAIServingResponses` inherits and reaches on this path too -- i.e. after
    `_make_request`, never before.

    `apply` promotes the merged effort onto the top-level `reasoning_effort`
    because that is where the `effort_kwarg` remap reads it. Run the variant
    first and `to_chat_body` rebuilds that field from `request.reasoning`
    alone, so the promotion is dropped and `reasoning_effort: "low"` never
    becomes `low_effort=True` -- for `/v1/responses` only, on a fleet whose
    chat traffic hashes correctly.
    """
    variant = RenderVariant({"reasoning_effort": "low"})
    body = {"model": "m", "input": "hi"}

    right = variant.apply(responses_input.normalised(body))
    assert right["reasoning_effort"] == "low"

    wrong = responses_input.normalised(variant.apply(body))
    assert "reasoning_effort" not in wrong


def test_the_client_kwarg_precedence_is_only_right_in_the_engines_order():
    """The sharper half of the same ordering.

    The engine applies its defaults with `setdefault` to a
    `chat_template_kwargs` the `ResponsesRequest` validator has *already*
    seeded from `reasoning.effort == "none"`, so `enable_thinking` stays False.
    Applied first, the variant's value goes in as if the client had sent it and
    the validator's seed loses to it -- the router renders thinking on where
    the engine renders it off.
    """
    variant = RenderVariant({"enable_thinking": True})
    body = {"model": "m", "input": "hi", "reasoning": {"effort": "none"}}

    right = variant.apply(responses_input.normalised(body))
    assert right["chat_template_kwargs"]["enable_thinking"] is False

    wrong = responses_input.normalised(variant.apply(body))
    assert wrong["chat_template_kwargs"]["enable_thinking"] is True


def test_the_client_still_wins_over_the_variant_on_this_path():
    """`setdefault`, in the engine and here: a Responses request that sets its
    own effort agrees with the worker and must keep hitting."""
    variant = RenderVariant({"reasoning_effort": "high"})
    out = variant.apply(responses_input.normalised({"model": "m", "input": "hi",
                                                    "reasoning": {"effort": "low"}}))
    assert out["reasoning_effort"] == "low"
    assert out["chat_template_kwargs"]["reasoning_effort"] == "high"
