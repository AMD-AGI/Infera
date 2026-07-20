###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""DP-rank routing through PD-disagg.

Two wire channels carry the router's per-leg DP-rank choice to SGLang:
``X-Data-Parallel-Rank`` (header, both legs) and ``disagg_prefill_dp_rank``
(decode body, names the prefill rank that holds the KV). Endpoint-addressed
targets (``dp_rank=None``) are untouched; the ``disagg_prefill_dp_rank`` body
field stays SGLang-only, while the header and room-residue also steer
vLLM-mooncake per-rank addressing (#119).
"""

from __future__ import annotations

import json

import httpx
import pytest

from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
from infera.router.disagg import DisaggRouter
from infera.router.dp_routing import (
    align_room_to_prefill_rank,
    dp_rank_header,
    inject_disagg_prefill_dp_rank,
)
from infera.router.policy.target import RouteTarget


def _w(wid, engine=EngineType.SGLANG, dp_size=None):
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=engine,
        dp_size=dp_size,
    )


# --- pure helpers ------------------------------------------------------------


def test_dp_rank_header_sglang_rank():
    assert dp_rank_header(RouteTarget(_w("p"), 2)) == {"X-Data-Parallel-Rank": "2"}


def test_dp_rank_header_none_for_endpoint_target():
    # dp_rank is None (rank-as-endpoint / single rank) -> nothing to pin.
    assert dp_rank_header(RouteTarget(_w("p"))) is None


def test_dp_rank_header_honored_for_vllm():
    # #119: vLLM-mooncake per-rank addressing also honours the DP-rank header.
    assert dp_rank_header(RouteTarget(_w("p", engine=EngineType.VLLM), 2)) == {
        "X-Data-Parallel-Rank": "2"
    }


def test_inject_disagg_prefill_dp_rank_sets_field():
    body = {"a": 1}
    out = inject_disagg_prefill_dp_rank(
        body, prefill_target=RouteTarget(_w("p"), 5), decode_engine=EngineType.SGLANG
    )
    assert out == {"a": 1, "disagg_prefill_dp_rank": 5}
    assert "disagg_prefill_dp_rank" not in body  # original untouched


def test_inject_disagg_prefill_dp_rank_noop_for_endpoint_prefill():
    body = {"a": 1}
    out = inject_disagg_prefill_dp_rank(
        body, prefill_target=RouteTarget(_w("p")), decode_engine=EngineType.SGLANG
    )
    assert out is body


def test_inject_disagg_prefill_dp_rank_noop_for_non_sglang_decode():
    body = {"a": 1}
    out = inject_disagg_prefill_dp_rank(
        body, prefill_target=RouteTarget(_w("p"), 5), decode_engine=EngineType.VLLM
    )
    assert out is body


def test_align_room_encodes_prefill_rank():
    # dp_size=4; whatever the random room, residue must equal the chosen rank.
    for room in (0, 1, 2, 3, 4, 5, 1234567, 6630595958492129781):
        for rank in range(4):
            out = align_room_to_prefill_rank(room, RouteTarget(_w("p", dp_size=4), rank))
            assert out % 4 == rank
            assert out >= 0


def test_align_room_noop_for_endpoint_target():
    assert align_room_to_prefill_rank(123, RouteTarget(_w("p", dp_size=4))) == 123


def test_align_room_encodes_prefill_rank_for_vllm():
    # #119: vLLM-mooncake reuses the room residue to address the prefill rank.
    t = RouteTarget(_w("p", engine=EngineType.VLLM, dp_size=4), 2)
    assert align_room_to_prefill_rank(123, t) % 4 == 2


# --- DisaggRouter HTTP dispatch (concurrent topology) ------------------------


def _pd_worker(wid, mode, *, dp_size=None, bootstrap=None):
    meta: dict = {"protocol": "sglang-bootstrap"}
    meta["params"] = {"bootstrap_addr": bootstrap} if bootstrap else {}
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=EngineType.SGLANG,
        disagg_mode=mode,
        disagg_meta=meta,
        dp_size=dp_size,
        request_transport="http",
    )


class _PDPool:
    def __init__(self, prefill, decode):
        self._by_mode = {DisaggMode.PREFILL: [prefill], DisaggMode.DECODE: [decode]}

    def list_active(self, model=None, mode=None):
        return list(self._by_mode.get(mode, []))


class _PDPolicy:
    """Returns preset prefill/decode targets and records bookkeeping keys."""

    def __init__(self, p_target, d_target):
        self._targets = {"prefill": p_target, "decode": d_target}
        self.started: list[str] = []
        self.finished: list[str] = []

    def pick(self, candidates, body, role_hint=None):
        return self._targets[role_hint], []

    def on_request_started(self, route_key, blocks):
        self.started.append(route_key)

    def on_request_finished(self, route_key, blocks):
        self.finished.append(route_key)


def _capture_router(p_target, d_target):
    """DisaggRouter whose HTTP client records each leg's request and replies
    200 JSON. Returns (router, captured) where captured[host] = (headers, body)."""
    captured: dict[str, tuple[dict, dict]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured[request.url.host] = (dict(request.headers), json.loads(request.content))
        return httpx.Response(200, json={"id": request.url.host})

    pool = _PDPool(p_target.worker, d_target.worker)
    policy = _PDPolicy(p_target, d_target)
    r = DisaggRouter(pool, policy)
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return r, policy, captured


@pytest.mark.asyncio
async def test_disagg_dp_rank_injected_per_leg():
    p = _pd_worker("p1", DisaggMode.PREFILL, dp_size=4, bootstrap="boot:8998")
    d = _pd_worker("d1", DisaggMode.DECODE, dp_size=4)
    p_target = RouteTarget(p, 1)
    d_target = RouteTarget(d, 3)
    r, policy, captured = _capture_router(p_target, d_target)

    resp = await r.dispatch({"model": "m"}, stream=False)
    assert resp.status_code == 200

    p_headers, p_body = captured["p1"]
    d_headers, d_body = captured["d1"]

    # Each leg is pinned to its own DP rank.
    assert p_headers["x-data-parallel-rank"] == "1"
    assert d_headers["x-data-parallel-rank"] == "3"
    # Decode is told which prefill rank holds the KV; prefill is not.
    assert d_body["disagg_prefill_dp_rank"] == 1
    assert "disagg_prefill_dp_rank" not in p_body
    # Protocol bootstrap fields still present on both legs.
    assert p_body["bootstrap_room"] == d_body["bootstrap_room"]
    # Bookkeeping is per rank, not per endpoint.
    assert set(policy.started) == {"p1#dp1", "d1#dp3"}
    assert set(policy.finished) == {"p1#dp1", "d1#dp3"}
    await r.aclose()


@pytest.mark.asyncio
async def test_disagg_endpoint_target_has_no_dp_wire_fields():
    """Regression: single-rank PD pairs (dp_rank=None) route exactly as before
    — no X-Data-Parallel-Rank header, no disagg_prefill_dp_rank, route_key is
    the bare worker_id."""
    p = _pd_worker("p1", DisaggMode.PREFILL, bootstrap="boot:8998")
    d = _pd_worker("d1", DisaggMode.DECODE)
    r, policy, captured = _capture_router(RouteTarget(p), RouteTarget(d))

    resp = await r.dispatch({"model": "m"}, stream=False)
    assert resp.status_code == 200

    p_headers, p_body = captured["p1"]
    d_headers, d_body = captured["d1"]
    assert "x-data-parallel-rank" not in p_headers
    assert "x-data-parallel-rank" not in d_headers
    assert "disagg_prefill_dp_rank" not in d_body
    assert set(policy.started) == {"p1", "d1"}
    await r.aclose()
