###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Reusable e2e scenario bodies, shared across engines.

A "scenario" is the engine-agnostic flow + assertions of a test. It takes a
running ``server`` context, a ``spawn`` worker factory (bound to a specific
engine adapter by that engine's conftest), and the ``EngineParams`` under
test. The sglang/vllm/atom ``test_*.py`` files stay thin: they parametrize
and delegate here.
"""

from __future__ import annotations

import os
from typing import NamedTuple

from . import client, correctness
from .adapter import emit_reporter_line
from .params import EngineParams


def _short(text: str, limit: int = 400) -> str:
    """One-line, length-capped view of a model reply for log output."""
    s = " ".join(text.split())
    return s if len(s) <= limit else s[:limit] + "…"


# ----------------------------------------------------------------------
# Single-assertion helpers (worker assumed already registered)
# ----------------------------------------------------------------------


# Disable Qwen3-style "thinking" for the liveness checks: with a tiny
# max_tokens budget the whole reply can otherwise be spent inside a <think>
# block, leaving message.content empty (some engines, e.g. ATOM, split the
# reasoning into reasoning_content). Templates that don't declare the kwarg
# ignore it.
_NO_THINK = {"enable_thinking": False}

# ...and a budget that survives the models it cannot silence. gpt-oss's harmony
# template has no `enable_thinking`, so it always writes an analysis channel
# first; on client.chat's 20-token default the whole reply is preamble and
# `content` comes back empty. 64 is what the counting/capital probes already use.
_LIVENESS_MAX_TOKENS = 64


async def _chat_json_no_think(server_url: str, model: str, content: str, **kw) -> dict:
    """chat_json with thinking disabled, but tolerant of models/engines whose
    chat template rejects unknown kwargs. Most templates ignore an unknown
    ``enable_thinking``, but some (e.g. ATOM's DeepSeek-V4 ``encode_messages``)
    raise HTTP 500 on it — so on a non-200 we retry once without the kwarg."""
    r = await client.chat(server_url, model, content, chat_template_kwargs=_NO_THINK, **kw)
    if r.status_code != 200:
        r = await client.chat(server_url, model, content, **kw)
    assert r.status_code == 200, f"chat failed {r.status_code}: {r.text}"
    return r.json()


async def assert_chat_ok(server_url: str, model: str) -> None:
    body = await _chat_json_no_think(server_url, model, "Say hi.", max_tokens=_LIVENESS_MAX_TOKENS)
    assert body["model"] == model
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] > 0


async def assert_chat_streaming_ok(server_url: str, model: str) -> None:
    body = await client.chat_stream_body(server_url, model, "Say hi.")
    # OpenAI SSE: many `data: {...}` events, terminated by `data: [DONE]`.
    assert b"data: " in body
    assert b"[DONE]" in body


# Whether an ADVISORY probe may fail the case. Off by default, because what quicksort
# grades is whether this checkpoint can write code within its token budget — a property
# of the model, not of the kernels, so a wrong answer there does not implicate the build
# under test. Turn it on for a hardware bring-up run, where that is exactly the signal
# wanted and a human is reading the output anyway.
_DEPTH_STRICT = os.environ.get("INFERA_E2E_DEPTH_STRICT") == "1"


class _Probe(NamedTuple):
    """One probe's verdict.

    ``ran=False`` means this deployment could not carry the probe (no chat route,
    context window too small) — a harness limit, not a wrong answer. ``advisory``
    means a wrong answer is reported but does not fail the case.
    """

    name: str
    ok: bool
    detail: str
    ran: bool = True
    advisory: bool = False


async def _counting_probe(server_url: str, model: str) -> _Probe:
    """Liveness — /v1/completions counting continuation. Seed "...1,2,3,4,5," and the
    model simply continues; no chat template or thinking to derail tiny models."""
    content = await client.completion_text(
        server_url,
        model,
        correctness.COUNTING_PROMPT,
        max_tokens=correctness.COUNTING_MAX_TOKENS,
        temperature=0.0,
    )
    return _Probe("counting", correctness.is_counting_correct(content), _short(content))


async def _capital_probe(server_url: str, model: str) -> _Probe:
    """Liveness — one memorised fact through the chat template, thinking disabled so the
    small budget is not spent in <think>. Chat is optional: ATOM's PD serves completions only."""
    try:
        body = await _chat_json_no_think(
            server_url,
            model,
            correctness.CAPITAL_PROMPT,
            max_tokens=correctness.CAPITAL_MAX_TOKENS,
            temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001 - chat may be unsupported (completions-only PD)
        return _Probe("capital", False, f"chat unavailable: {type(e).__name__}: {e}", ran=False)
    content = body["choices"][0]["message"].get("content") or ""
    return _Probe("capital", correctness.is_capital_correct(content), _short(content))


async def _longctx_probe(server_url: str, model: str) -> _Probe:
    """Depth — retrieve a 4-digit code buried mid-ledger. Over /v1/completions, so the
    PD tier gets it too and its prefill→decode hop finally transfers a real KV cache."""
    prompt = correctness.build_longctx_prompt()
    r = await client.completion(
        server_url, model, prompt, max_tokens=correctness.LONGCTX_MAX_TOKENS, temperature=0.0
    )
    if r.status_code == 400:
        # Rejected, not answered wrong: this case's --max-model-len cannot hold the ledger.
        detail = f"{len(prompt)} chars rejected: {_short(r.text, 200)}"
        return _Probe("long-context", False, detail, ran=False)
    assert r.status_code == 200, f"long-context completion failed {r.status_code}: {r.text}"
    content = r.json()["choices"][0].get("text") or ""
    return _Probe("long-context", correctness.is_longctx_correct(content), _short(content))


async def _quicksort_probe(server_url: str, model: str) -> _Probe:
    """Depth, advisory — the model writes quicksort and we execute it against
    ``sorted()``. Chat, for the instruction following; optional for the same reason the
    capital probe is. See :data:`_DEPTH_STRICT` for why it does not fail a case."""
    advisory = not _DEPTH_STRICT
    try:
        body = await _chat_json_no_think(
            server_url,
            model,
            correctness.QUICKSORT_PROMPT,
            max_tokens=correctness.QUICKSORT_MAX_TOKENS,
            temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001 - chat may be unsupported (completions-only PD)
        return _Probe("quicksort", False, f"chat unavailable: {type(e).__name__}: {e}", ran=False)
    reply = body["choices"][0]["message"].get("content") or ""
    ok, detail = correctness.run_quicksort_code(correctness.extract_python_code(reply))
    return _Probe("quicksort", ok, detail, advisory=advisory)


async def assert_correctness(server_url: str, model: str) -> None:
    """Two gates over four probes: at least one LIVENESS probe (counting, capital) must
    pass, and every DEPTH probe (long-context, quicksort) that ran and is not advisory
    must pass."""
    liveness = [await _counting_probe(server_url, model), await _capital_probe(server_url, model)]
    depth = [await _longctx_probe(server_url, model), await _quicksort_probe(server_url, model)]

    # Every verdict and the model's actual reply, live in the run output
    # (capture-suspended), so a pass or a fail explains itself without a rerun. An
    # advisory miss reads "warn": visible, and deliberately not the reason for a fail.
    for probe in liveness + depth:
        state = (
            "ok" if probe.ok else "n/a" if not probe.ran else "warn" if probe.advisory else "FAILED"
        )
        emit_reporter_line(f"[e2e correctness] {probe.name:<12} {state:<6} {probe.detail!r}")

    alive = any(probe.ok for probe in liveness)
    wrong = [p for p in depth if p.ran and not p.ok and not p.advisory]
    emit_reporter_line(f"[e2e correctness] {'PASS' if alive and not wrong else 'FAIL'}")

    assert alive, "correctness failed: no liveness probe returned a correct reply.\n" + "\n".join(
        f"  {p.name}: ran={p.ran} {p.detail!r}" for p in liveness
    )
    assert not wrong, "correctness failed: " + "; ".join(f"{p.name} — {p.detail}" for p in wrong)


# ----------------------------------------------------------------------
# Composite scenarios (spawn worker + run the relevant assertions)
# ----------------------------------------------------------------------


async def run_mixed(server: dict, spawn, params: EngineParams) -> list:
    """Full mixed-worker (prefill-decode-mix, no PD) scenario: spawn one worker and
    verify chat liveness + streaming + the four correctness probes."""
    workers = [await spawn(server, params)]

    await assert_chat_ok(server["url"], params.model)
    await assert_chat_streaming_ok(server["url"], params.model)
    await assert_correctness(server["url"], params.model)

    return workers
