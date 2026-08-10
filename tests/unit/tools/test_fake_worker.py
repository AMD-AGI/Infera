###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The fake worker has to be trustworthy, or every test built on it is too.

Two properties matter more than the rest. It must register through the *real*
contract, so a fleet of fakes exercises the same discovery path a real fleet
does -- that is checked by building an actual ``EngineConfig`` and running it
through ``build_worker_payload``, the same function every engine uses. And its
queue must be real, because ``num_requests_waiting`` is the metric the entire
industry autoscales on, and a fake that always reports zero would make every
scaling test vacuously pass.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from infera.common.discovery import worker_info_from_json
from infera.common.registration import build_worker_payload
from infera.common.worker_pool import DisaggMode, EngineType
from infera.tools.fakeworker.server import (
    Behaviour,
    State,
    build_app,
    build_config,
    deterministic_canary,
    parse_args,
)


def _args(*extra):
    return parse_args(["--model-name", "m", *extra])


def _stack(**behaviour_kw):
    args = _args()
    cfg = build_config(args)
    state = State(ready=True)
    state._sem = asyncio.Semaphore(behaviour_kw.get("max_concurrency", 8))
    b = Behaviour(**behaviour_kw)
    return cfg, b, state, build_app(cfg, b, state)


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fake")


# --- the registration contract ------------------------------------------------


def test_registers_through_the_real_payload_builder():
    """If this breaks, the fake has drifted from what a real worker registers --
    which is the one failure that would silently invalidate everything else."""
    cfg = build_config(_args("--engine", "vllm", "--disagg-mode", "prefill"))
    payload = build_worker_payload(cfg)
    assert payload["model_name"] == "m"
    assert payload["engine"] == EngineType.VLLM
    assert payload["disagg_mode"] == DisaggMode.PREFILL
    # And discovery must be able to parse it back: worker_info_from_json is the
    # single function every backend (etcd, kubernetes) uses on the wire record,
    # so a round-trip through it is the real contract, not an approximation.
    info = worker_info_from_json(payload)
    assert info.worker_id == payload["worker_id"]
    assert info.disagg_mode is DisaggMode.PREFILL


def test_kv_block_is_absent_unless_asked_for():
    """Without --kv there is no canary, so fakes can join any fleet. With it,
    canary verification applies and mixing with real workers breaks."""
    assert build_config(_args()).kv is None
    assert build_config(_args("--kv")).kv is not None


def test_fakes_for_one_model_agree_on_the_canary():
    """Disagreeing fakes would be silently dropped from the pool by
    CanaryVerifier -- a fleet that looks half its intended size for no visible
    reason."""
    assert deterministic_canary("llama") == deterministic_canary("llama")
    assert deterministic_canary("llama") != deterministic_canary("qwen")


# --- the serving surface ------------------------------------------------------


@pytest.mark.asyncio
async def test_unary_completion_shape():
    _, _, _, app = _stack(ttft_ms=0, itl_ms=0)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json={"model": "m", "max_tokens": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] == 5


@pytest.mark.asyncio
async def test_streaming_emits_sse_and_terminates():
    """The router's failover and circuit breaker are both first-byte-sensitive,
    so a fake that cannot stream cannot exercise either."""
    _, _, _, app = _stack(ttft_ms=0, itl_ms=0)
    async with _client(app) as c:
        r = await c.post(
            "/v1/chat/completions", json={"model": "m", "max_tokens": 3, "stream": True}
        )
        body = (await r.aread()).decode()
    assert r.headers["content-type"].startswith("text/event-stream")
    assert body.count("data: ") == 4  # 3 chunks + [DONE]
    assert body.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_health_is_503_until_ready():
    """--startup-delay-s exists to reproduce autoscaler overshoot, which only
    happens because an unready replica still counts in the fleet."""
    cfg, b, state, app = _stack()
    state.ready = False
    async with _client(app) as c:
        assert (await c.get("/health")).status_code == 503
        state.ready = True
        assert (await c.get("/health")).status_code == 200


# --- the metrics an autoscaler would read -------------------------------------


@pytest.mark.asyncio
async def test_queue_depth_is_real():
    """The whole point. Drive more concurrent requests than the worker admits
    and the waiting count must actually rise -- a fake that always reports 0
    would make every scaling test pass without testing anything."""
    _, _, state, app = _stack(max_concurrency=2, ttft_ms=200, itl_ms=0)
    async with _client(app) as c:
        tasks = [
            asyncio.create_task(c.post("/v1/completions", json={"model": "m", "max_tokens": 1}))
            for _ in range(6)
        ]
        await asyncio.sleep(0.05)
        peak_waiting = state.waiting
        peak_running = state.running
        await asyncio.gather(*tasks)

    assert peak_running == 2, f"admitted {peak_running}, expected the concurrency cap"
    assert peak_waiting == 4, f"queued {peak_waiting}, expected the other 4"
    assert state.waiting == 0 and state.running == 0, "must settle back to idle"


@pytest.mark.asyncio
async def test_metrics_use_engine_native_names():
    """A scaling rule written against fakes should transfer to a real fleet
    unchanged, so the names have to be the engine's, not ours."""
    for engine, expected in (
        ("vllm", "vllm:num_requests_waiting"),
        ("sglang", "sglang:num_queue_reqs"),
    ):
        args = _args("--engine", engine)
        cfg = build_config(args)
        state = State(ready=True)
        state._sem = asyncio.Semaphore(1)
        app = build_app(cfg, Behaviour(), state)
        async with _client(app) as c:
            text = (await c.get("/metrics")).text
        assert expected in text, f"{engine}: missing {expected}\n{text}"


@pytest.mark.asyncio
async def test_kv_usage_tracks_inflight():
    _, _, state, app = _stack(max_concurrency=4, ttft_ms=200, itl_ms=0)
    async with _client(app) as c:
        idle = (await c.get("/metrics")).text
        tasks = [
            asyncio.create_task(c.post("/v1/completions", json={"model": "m", "max_tokens": 1}))
            for _ in range(4)
        ]
        await asyncio.sleep(0.05)
        busy = (await c.get("/metrics")).text
        await asyncio.gather(*tasks)

    def usage(t):
        return float(
            [ln for ln in t.splitlines() if "cache_usage" in ln or "token_usage" in ln][0].split()[
                -1
            ]
        )

    assert usage(idle) == 0.0
    assert usage(busy) > 0.0


# --- failure injection, for the circuit breaker -------------------------------


@pytest.mark.asyncio
async def test_fail_first_then_recovers():
    """Mirrors the breaker's half-open probe: a worker that is broken, stays
    broken for a while, then comes back."""
    _, _, state, app = _stack(ttft_ms=0, itl_ms=0, fail_first=3)
    async with _client(app) as c:
        codes = [
            (await c.post("/v1/completions", json={"model": "m", "max_tokens": 1})).status_code
            for _ in range(5)
        ]
    assert codes[:3] == [503, 503, 503]
    assert codes[3:] == [200, 200]


@pytest.mark.asyncio
async def test_draining_refuses_new_work():
    """SIGTERM deregisters before draining; until the router notices, arriving
    requests must be refused rather than accepted and then cut."""
    _, _, state, app = _stack(ttft_ms=0, itl_ms=0)
    state.draining = True
    async with _client(app) as c:
        r = await c.post("/v1/completions", json={"model": "m", "max_tokens": 1})
        assert r.status_code == 503
        assert "infera_fake_worker_draining 1" in (await c.get("/metrics")).text


def test_a_rank_header_cannot_forge_metrics_or_grow_without_bound():
    """The DP-rank header is attacker-controlled and ends up as a Prometheus
    label and a map key. Unvalidated, a quote or newline breaks out of the label
    and forges series in whatever scrapes the endpoint, and every distinct value
    adds a permanent entry to a map that /debug/routing returns in full."""
    from infera.tools.fakeworker.server import _rank_label

    assert _rank_label(None, 4) == "-"
    assert _rank_label("2", 4) == "2"

    for hostile in ('x"} 1\nup{job="prod"} 0', "a\\b", "1\n2", "", "  "):
        got = _rank_label(hostile, 4)
        assert got == "invalid", f"{hostile!r} -> {got!r}"

    # Out of range is meaningless here, and unbounded if accepted.
    assert _rank_label("4", 4) == "invalid"
    assert _rank_label("-1", 4) == "invalid"
    assert _rank_label("999999", 4) == "invalid"


def test_it_refuses_to_start_without_an_explicit_opt_in(monkeypatch):
    """It registers into real service discovery and answers with fabricated
    text, and it ships in the same package as the server. Starting it should be
    a decision, not a default."""
    import pytest

    from infera.tools.fakeworker.server import ALLOW_ENV, main

    monkeypatch.delenv(ALLOW_ENV, raising=False)
    with pytest.raises(SystemExit) as exc:
        main([])
    assert ALLOW_ENV in str(exc.value), "the error must say how to enable it"

    # Enabled, it gets as far as parsing arguments.
    monkeypatch.setenv(ALLOW_ENV, "1")
    with pytest.raises(SystemExit) as exc:
        main(["--nonsense-flag"])
    assert ALLOW_ENV not in str(exc.value), "past the gate, argparse should be what refuses"
