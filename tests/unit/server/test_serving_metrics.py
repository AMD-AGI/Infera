###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Serving metrics exposed on GET /metrics for Prometheus."""

from __future__ import annotations

import asyncio

from infera.common.worker_pool import WorkerInfo
from infera.server import app as server_app
from infera.server import metrics


def test_request_tracker_records_ttft_itl_and_token_counters():
    metrics.set_sla_metrics_enabled(True)
    before = metrics.render_metrics()[0].decode()

    with metrics.track_request(router="mixed", model="m") as obs:
        obs.claim_stream()
        obs.set_workers(prefill_worker="w-p", decode_worker="w-d")
        obs.set_input_tokens(128)
        obs.observe_stream_chunk(b'data: {"choices":[{"delta":{"content":"a"}}]}\n')
        obs.observe_stream_chunk(b'data: {"choices":[{"delta":{"content":"b"}}]}\n')
        obs.observe_stream_chunk(
            b'data: {"choices":[],"usage":{"prompt_tokens":128,"completion_tokens":2}}\n'
        )
        obs["outcome"] = "ok"
        obs.close()

    body = metrics.render_metrics()[0].decode()
    assert "infera_time_to_first_token_seconds_count" in body
    assert "infera_inter_token_latency_seconds_count" in body
    assert "infera_prompt_tokens_total" in body
    assert "infera_generation_tokens_total" in body
    assert "infera_requests_total" in body
    assert 'prefill_worker="w-p"' in body
    assert 'decode_worker="w-d"' in body
    assert body != before


def test_record_pick_updates_prefix_cache_counters():
    metrics.set_sla_metrics_enabled(True)
    metrics.record_pick(role="prefill", worker_id="w1", cache_hits=3, request_blocks=10)
    body = metrics.render_metrics()[0].decode()
    assert "infera_prefix_cache_blocks_hit_total" in body
    assert "infera_prefix_cache_blocks_total" in body


def _observe_pair(prefill: str, decode: str) -> None:
    with metrics.track_request(router="mixed", model="card-model") as obs:
        obs.claim_stream()
        obs.set_workers(prefill_worker=prefill, decode_worker=decode)
        obs.set_input_tokens(8)
        obs.observe_stream_chunk(b'data: {"choices":[{"delta":{"content":"a"}}]}\n')
        obs.observe_stream_chunk(b'data: {"choices":[{"delta":{"content":"b"}}]}\n')
        obs["outcome"] = "ok"
        obs.close()


def test_ttft_series_follow_the_prefill_worker_not_the_pair():
    for prefill in ("p-card-1", "p-card-2"):
        for decode in ("d-card-1", "d-card-2"):
            _observe_pair(prefill, decode)
    body = metrics.render_metrics()[0].decode()
    lines = [
        line
        for line in body.splitlines()
        if line.startswith("infera_time_to_first_token_seconds_count{") and "card-model" in line
    ]
    assert len(lines) == 2
    assert all("decode_worker" not in line for line in lines)
    assert any('prefill_worker="p-card-1"' in line for line in lines)
    assert any('prefill_worker="p-card-2"' in line for line in lines)


def test_departed_worker_sla_series_are_dropped():
    _observe_pair("p-gone", "d-keep")
    _observe_pair("p-keep", "d-keep")
    metrics.record_pick(role="prefill", worker_id="p-gone", cache_hits=1, request_blocks=2)
    metrics.record_pick(role="prefill", worker_id="p-keep", cache_hits=1, request_blocks=2)
    body = metrics.render_metrics()[0].decode()
    active = {"p-keep", "d-keep"}
    for line in body.splitlines():
        for key in ('prefill_worker="', 'decode_worker="', 'worker_id="'):
            if key in line:
                value = line.split(key, 1)[1].split('"', 1)[0]
                if value:
                    active.add(value)
    active.discard("p-gone")
    metrics.prune_departed_workers(active)
    body = metrics.render_metrics()[0].decode()
    assert 'prefill_worker="p-gone"' not in body
    assert 'prefill_worker="p-keep"' in body
    assert 'worker_id="p-gone"' not in body
    assert 'worker_id="p-keep"' in body


def test_outcome_label_buckets_http_status():
    assert metrics.outcome_label(200) == "ok"
    assert metrics.outcome_label(429) == "4xx"
    assert metrics.outcome_label(502) == "5xx"
    assert metrics.outcome_label(None) == "error"


def test_prune_keeps_dp_route_key_picks_for_active_workers():
    metrics.record_pick(role="prefill", worker_id="w-live#dp0", cache_hits=1, request_blocks=2)
    metrics.record_pick(role="prefill", worker_id="w-live#dp1", cache_hits=1, request_blocks=2)
    metrics.record_pick(role="prefill", worker_id="w-gone#dp0", cache_hits=1, request_blocks=2)
    metrics.prune_departed_workers({"w-live"})
    body = metrics.render_metrics()[0].decode()
    assert 'worker_id="w-live#dp0"' in body
    assert 'worker_id="w-live#dp1"' in body
    assert 'worker_id="w-gone#dp0"' not in body


def test_apply_engine_scrape_sets_kv_usage_gauge():
    text = (
        "# HELP sglang:token_usage The token usage\n"
        "# TYPE sglang:token_usage gauge\n"
        'sglang:token_usage{model_name="m",tp_rank="0"} 0.42\n'
        'sglang:cache_hit_rate{model_name="m"} 0.75\n'
        'sglang:num_decode_transfer_queue_reqs{model_name="m"} 2\n'
    )
    metrics.apply_engine_scrape(
        worker_id="w1",
        engine="sglang",
        disagg_mode="decode",
        text=text,
    )
    body = metrics.render_metrics()[0].decode()
    assert "infera_engine_kv_cache_usage" in body
    assert "infera_engine_prefix_cache_hit_rate" in body
    assert "infera_engine_kv_transfer_queue_reqs" in body


def test_overlapping_metrics_scrapes_do_not_interleave(monkeypatch):
    """Gauge reset and the worker fetch share one lock, so two scrapes cannot nest."""

    class _Pool:
        def list_all(self):
            return [
                WorkerInfo(
                    worker_id="w-lock",
                    url="http://127.0.0.1:9",
                    model_name="m",
                )
            ]

    saved_registry = server_app.registry
    saved_kv = server_app.kv_client
    server_app.registry = _Pool()
    server_app.kv_client = None
    order: list[str] = []

    async def _scrape():
        order.append("enter")
        await asyncio.sleep(0.02)
        order.append("leave")
        return b""

    async def _run():
        monkeypatch.setattr(server_app, "_scrape_engine_metrics", _scrape)
        await asyncio.gather(
            server_app.prometheus_metrics(),
            server_app.prometheus_metrics(),
        )

    try:
        asyncio.run(_run())
    finally:
        server_app.registry = saved_registry
        server_app.kv_client = saved_kv

    assert order == ["enter", "leave", "enter", "leave"]
