###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for session affinity layered on a routing policy."""

from __future__ import annotations

import asyncio

import pytest

from infera.common.nats_request import TYPE_DATA, TYPE_DONE, TYPE_ERROR
from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
from infera.router.mixed import MixedRouter
from infera.router.policy import session_affinity
from infera.router.policy.factory import build_policy
from infera.router.policy.round_robin import RoundRobinPolicy
from infera.router.policy.session_affinity import SessionAffinityPolicy
from infera.server import metrics
from infera.server.args import parse_server_args


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _SpyPolicy(RoundRobinPolicy):
    """Round-robin that records the targets offered and the lifecycle hooks."""

    def __init__(self) -> None:
        super().__init__()
        self.offered: list[list[str]] = []
        self.started: list[str] = []

    def pick_targets(self, targets, request, *, role_hint=None):
        self.offered.append([t.route_key for t in targets])
        return super().pick_targets(targets, request, role_hint=role_hint)

    def on_request_started(self, route_key, blocks=None):
        self.started.append(route_key)


def _worker(wid: str, dp_size: int | None = None) -> WorkerInfo:
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        disagg_mode=DisaggMode.PREFILL,
        dp_size=dp_size,
    )


def _req(session: str | None = None, model: str = "m") -> dict:
    headers = {"x-dynamo-session-id": session} if session else {}
    return {"model": model, "_infera_session_headers": headers}


def _pick(policy, workers, request, role_hint="prefill") -> str:
    return policy.pick(workers, request, role_hint=role_hint)[0].route_key


W = [_worker("w0"), _worker("w1")]


def test_cli_wraps_the_router_policy():
    base = ["--router-tokenizer-path", "t", "--router-policy", "round-robin"]
    args = parse_server_args(base)
    assert (args.session_affinity, args.session_affinity_ttl_seconds) == ("off", 3600)
    assert isinstance(build_policy("round-robin"), RoundRobinPolicy)

    args = parse_server_args([*base, "--session-affinity", "prefill"])
    policy = build_policy(
        args.router_policy,
        session_affinity=args.session_affinity,
        session_affinity_ttl_seconds=args.session_affinity_ttl_seconds,
    )
    assert isinstance(policy, SessionAffinityPolicy)
    assert isinstance(policy.inner, RoundRobinPolicy)


@pytest.mark.parametrize("ttl", ["0", "86401", "abc"])
def test_cli_rejects_bad_ttl(ttl):
    with pytest.raises(SystemExit):
        parse_server_args(["--router-tokenizer-path", "t", "--session-affinity-ttl-seconds", ttl])


@pytest.mark.parametrize("kwargs", [{"mode": "off"}, {"mode": "Both"}, {"ttl_seconds": 0}])
def test_rejects_bad_config(kwargs):
    with pytest.raises(ValueError):
        SessionAffinityPolicy(RoundRobinPolicy(), **{"mode": "both", **kwargs})


def test_session_stays_on_its_target_and_others_use_the_inner_policy():
    policy = SessionAffinityPolicy(RoundRobinPolicy(), mode="prefill")
    first = _pick(policy, W, _req("a"))
    assert [_pick(policy, W, _req("a")) for _ in range(3)] == [first] * 3
    # No session id: plain round-robin.
    assert {_pick(policy, W, _req()), _pick(policy, W, _req())} == {"w0", "w1"}


def test_pinned_pick_goes_through_the_inner_policy_on_the_same_rank():
    inner = _SpyPolicy()
    policy = SessionAffinityPolicy(inner, mode="prefill")
    workers = [_worker("w", dp_size=4)]
    first = _pick(policy, workers, _req("a"))
    assert "#dp" in first
    assert _pick(policy, workers, _req("a")) == first
    assert inner.offered[-1] == [first]  # only the pinned target is offered
    policy.on_request_started(first)
    assert inner.started == [first]  # accounting stays with the inner policy


def test_prefill_mode_leaves_decode_alone_and_both_pins_each_role():
    prefill_only = SessionAffinityPolicy(RoundRobinPolicy(), mode="prefill")
    decode = {_pick(prefill_only, W, _req("a"), role_hint="decode") for _ in range(2)}
    assert decode == {"w0", "w1"}

    both = SessionAffinityPolicy(RoundRobinPolicy(), mode="both")
    for role in ("prefill", "decode", None):
        first = _pick(both, W, _req("a"), role_hint=role)
        assert _pick(both, W, _req("a"), role_hint=role) == first
    assert {key[2] for key in both._bindings} == {"prefill", "decode", "mixed"}


def test_bindings_are_scoped_by_model():
    policy = SessionAffinityPolicy(RoundRobinPolicy(), mode="prefill")
    _pick(policy, W, _req("a", model="m1"))
    _pick(policy, W, _req("a", model="m2"))
    assert len(policy._bindings) == 2


def test_missing_target_is_repicked():
    policy = SessionAffinityPolicy(RoundRobinPolicy(), mode="prefill")
    first = _pick(policy, W, _req("a"))
    survivors = [w for w in W if w.worker_id != first]
    moved = _pick(policy, survivors, _req("a"))
    assert moved != first
    assert _pick(policy, W, _req("a")) == moved  # the new target is the binding


def test_idle_ttl_drops_the_binding():
    clock = _Clock()
    inner = _SpyPolicy()
    policy = SessionAffinityPolicy(inner, mode="prefill", ttl_seconds=10, clock=clock)
    _pick(policy, W, _req("a"))
    clock.now = 9
    _pick(policy, W, _req("a"))  # refreshes last use
    clock.now = 18
    _pick(policy, W, _req("a"))
    assert inner.offered[-1] != ["w0", "w1"]  # still pinned
    clock.now = 28
    _pick(policy, W, _req("a"))
    assert inner.offered[-1] == ["w0", "w1"]  # expired: picked afresh


def test_at_capacity_new_sessions_are_not_bound(monkeypatch):
    monkeypatch.setattr(session_affinity, "MAX_SESSIONS", 1)
    policy = SessionAffinityPolicy(RoundRobinPolicy(), mode="prefill")
    first = _pick(policy, W, _req("a"))
    _pick(policy, W, _req("b"))
    assert [key[1] for key in policy._bindings] == ["a"]
    assert _pick(policy, W, _req("a")) == first


def test_failure_drops_the_binding_only_while_it_points_there():
    policy = SessionAffinityPolicy(RoundRobinPolicy(), mode="prefill")
    first = _pick(policy, W, _req("a"))
    other = "w1" if first == "w0" else "w0"
    policy.on_request_failed(other, _req("a"), role_hint="prefill")  # stale: kept
    policy.on_request_failed(first, _req("a"), role_hint="decode")  # other role: kept
    assert _pick(policy, W, _req("a")) == first
    policy.on_request_failed(first, _req("a"), role_hint="prefill")
    assert not policy._bindings


class _Pool:
    def __init__(self, workers):
        self._workers = workers

    def list_active(self, model=None, mode=None):
        return list(self._workers)


class _Nats:
    def __init__(self, scripts):
        self.scripts = scripts

    async def admit(self, worker_id):
        return True

    async def stream(self, worker_id, payload):
        for event in self.scripts[worker_id]:
            yield event


def _nats_worker(wid: str) -> WorkerInfo:
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=EngineType.SGLANG,
        request_transport="nats",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", [False, True])
async def test_router_drops_the_binding_of_a_failed_request(broken):
    ok = [(TYPE_DATA, None, b"data: x\n\n"), (TYPE_DONE, 200, b"")]
    failed = [(TYPE_DATA, None, b"data: x\n\n"), (TYPE_ERROR, None, b"crash")]
    policy = SessionAffinityPolicy(RoundRobinPolicy(), mode="both")
    nats = _Nats({"w0": failed if broken else ok, "w1": ok})
    router = MixedRouter(_Pool([_nats_worker("w0"), _nats_worker("w1")]), policy, nats_client=nats)

    resp = await router.dispatch(_req("a"), stream=True)
    async for _ in resp.body_iterator:
        pass

    assert bool(policy._bindings) is not broken
    await router.aclose()


def test_a_cancelled_request_is_not_a_failure():
    calls = []
    with pytest.raises(asyncio.CancelledError), metrics.track_request(router="mixed") as obs:
        obs.on_failure(lambda: calls.append("cancelled"))
        raise asyncio.CancelledError
    with metrics.track_request(router="mixed") as obs:
        obs.on_failure(lambda: calls.append("failed"))  # outcome never set to "ok"
    with metrics.track_request(router="mixed") as obs:
        obs.on_failure(lambda: calls.append("ok"))
        obs["outcome"] = "ok"
    assert calls == ["failed"]
