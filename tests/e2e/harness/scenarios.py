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
    body = await _chat_json_no_think(server_url, model, "Say hi.")
    assert body["model"] == model
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] > 0


async def assert_chat_streaming_ok(server_url: str, model: str) -> None:
    body = await client.chat_stream_body(server_url, model, "Say hi.")
    # OpenAI SSE: many `data: {...}` events, terminated by `data: [DONE]`.
    assert b"data: " in body
    assert b"[DONE]" in body


async def _counting_probe(server_url: str, model: str) -> tuple[bool, str]:
    """Probe 1 — /v1/completions counting continuation. Seed "...1,2,3,4,5,"
    and the model simply continues "6,7,8,..."; no chat template / thinking to
    derail tiny models. Returns (ok, reply)."""
    content = await client.completion_text(
        server_url,
        model,
        correctness.COUNTING_PROMPT,
        max_tokens=correctness.COUNTING_MAX_TOKENS,
        temperature=0.0,
    )
    return correctness.is_counting_correct(content), content


async def _capital_probe(server_url: str, model: str) -> tuple[bool, str]:
    """Probe 2 — /v1/chat/completions factual question ("capital of China");
    correct iff the reply mentions "Beijing" (case-insensitive) and isn't
    garbage. Thinking is disabled so the small budget isn't spent in <think>.
    Returns (ok, reply).

    Tolerant: some PD-disaggregated engines only serve /v1/completions (e.g.
    ATOM), so a failed chat call is reported as (False, ...) rather than raised —
    the counting probe (/v1/completions) then carries correctness."""
    try:
        body = await _chat_json_no_think(
            server_url,
            model,
            correctness.CAPITAL_PROMPT,
            max_tokens=correctness.CAPITAL_MAX_TOKENS,
            temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001 - chat may be unsupported (completions-only PD)
        return False, f"chat unavailable: {type(e).__name__}: {e}"
    content = body["choices"][0]["message"].get("content") or ""
    return correctness.is_capital_correct(content), content


async def assert_correctness(server_url: str, model: str) -> None:
    """Semantic correctness: run both probes (counting continuation + capital
    question). Each probe's "correct" already requires a non-garbage reply, and
    the case passes as long as at least ONE probe is correct."""
    count_ok, count_reply = await _counting_probe(server_url, model)
    cap_ok, cap_reply = await _capital_probe(server_url, model)
    # Surface both probes' verdict + the model's actual reply live in the run
    # output (capture-suspended), so a pass/fail is self-explanatory.
    emit_reporter_line(f"[e2e correctness] counting ok={count_ok} reply={_short(count_reply)!r}")
    emit_reporter_line(f"[e2e correctness] capital  ok={cap_ok} reply={_short(cap_reply)!r}")
    verdict = "PASS" if (count_ok or cap_ok) else "FAIL"
    emit_reporter_line(f"[e2e correctness] {verdict} (counting={count_ok}, capital={cap_ok})")
    assert count_ok or cap_ok, (
        "correctness failed: neither probe returned a non-garbage correct reply.\n"
        f"  counting (/v1/completions)      ok={count_ok} reply={count_reply!r}\n"
        f"  capital  (/v1/chat/completions) ok={cap_ok} reply={cap_reply!r}"
    )


# ----------------------------------------------------------------------
# Composite scenarios (spawn worker + run the relevant assertions)
# ----------------------------------------------------------------------


async def run_mixed(server: dict, spawn, params: EngineParams) -> list:
    """Full mixed-worker (prefill-decode-mix, no PD) scenario: spawn one worker
    and verify chat liveness + streaming + semantic correctness (counting or
    capital probe)."""
    workers = [await spawn(server, params)]

    await assert_chat_ok(server["url"], params.model)
    await assert_chat_streaming_ok(server["url"], params.model)
    await assert_correctness(server["url"], params.model)

    return workers
