###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Serving metrics exposed on GET /metrics for Prometheus."""

from __future__ import annotations

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
