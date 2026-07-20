###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""RouteTarget / expand_targets + KvEventClient per-DP-rank cache views."""

from __future__ import annotations

from infera.common.worker_pool import WorkerInfo
from infera.router.kv_event.client import KvEventClient, WorkerSubscription, _offset_endpoint
from infera.router.kv_event.events import BlockStored
from infera.router.policy.target import RouteTarget, expand_targets


def _worker(wid: str, *, dp_size: int | None = None, dp_rank: int | None = None) -> WorkerInfo:
    return WorkerInfo(
        worker_id=wid, url=f"http://{wid}", model_name="m", dp_size=dp_size, dp_rank=dp_rank
    )


def test_route_key_single_rank_is_worker_id():
    assert RouteTarget(_worker("w1")).route_key == "w1"
    assert RouteTarget(_worker("w1"), 2).route_key == "w1#dp2"


def test_expand_single_and_external_lb_are_one_target():
    # Plain worker and a vLLM external-LB rank (own dp_rank) each map to one
    # endpoint target with dp_rank=None.
    plain = _worker("w1")
    ext_lb = _worker("w2", dp_size=2, dp_rank=1)
    targets = expand_targets([plain, ext_lb])
    assert [(t.worker.worker_id, t.dp_rank) for t in targets] == [("w1", None), ("w2", None)]


def test_expand_rank_multiplexed_fans_out():
    # SGLang native DP: one endpoint, dp_size ranks, no dp_rank.
    targets = expand_targets([_worker("s", dp_size=3)])
    assert [(t.worker.worker_id, t.dp_rank) for t in targets] == [
        ("s", 0),
        ("s", 1),
        ("s", 2),
    ]


def _stored(hashes: list[int], toks: list[int], parent: int | None = None) -> BlockStored:
    return BlockStored(
        block_hashes=hashes,
        parent_block_hash=parent,
        token_ids=toks,
        block_size=2,
        lora_id=None,
    )


def test_offset_endpoint_per_rank_port():
    # SGLang publishes rank r's kv-events on base_port + r.
    assert _offset_endpoint("tcp://h:5557", 0) == "tcp://h:5557"
    assert _offset_endpoint("tcp://h:5557", 2) == "tcp://h:5559"


def test_client_keeps_separate_view_per_rank():
    client = KvEventClient()
    sub = WorkerSubscription("s", "tcp://s:1", block_size=2, multiplexed=True)
    client._subs["s"] = sub

    # Each rank's socket loop feeds its pinned rank; different tokens ->
    # different chained hashes -> distinct per-rank views.
    client._handle_event(sub, _stored([100], [1, 2]), rank=0)
    client._handle_event(sub, _stored([200], [3, 4]), rank=1)

    v0 = client.cache_view("s", 0)
    v1 = client.cache_view("s", 1)
    assert v0 and v1 and v0 != v1
    assert client.cache_view("s", 2) == set()
    # dp_rank=None resolves to rank 0 (single-rank callers unchanged).
    assert client.cache_view("s") == v0
