###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Allowlisted engine-metric federation for frontend /metrics."""

from __future__ import annotations

from infera.common.engine_metrics import federate_engine_metrics
from infera.common.worker_pool import EngineType


def test_federate_vllm_injects_worker_id_and_keeps_buckets():
    text = (
        "# HELP vllm:e2e_request_latency_seconds Histogram of end to end request latency\n"
        "# TYPE vllm:e2e_request_latency_seconds histogram\n"
        'vllm:e2e_request_latency_seconds_bucket{le="0.5",model_name="m"} 1\n'
        'vllm:e2e_request_latency_seconds_bucket{le="+Inf",model_name="m"} 2\n'
        'vllm:e2e_request_latency_seconds_sum{model_name="m"} 1.5\n'
        'vllm:e2e_request_latency_seconds_count{model_name="m"} 2\n'
        'vllm:num_requests_running{model_name="m"} 3\n'
        'vllm:num_requests_waiting{model_name="m"} 1\n'
        'vllm:num_requests_swapped{model_name="m"} 0\n'
        'vllm:kv_cache_usage_perc{model_name="m"} 0.4\n'
        'vllm:cpu_cache_usage_perc{model_name="m"} 0.1\n'
        'vllm:request_success_total{finished_reason="stop",model_name="m"} 5\n'
        "# HELP process_cpu_seconds_total Total user and system CPU time\n"
        "# TYPE process_cpu_seconds_total counter\n"
        "process_cpu_seconds_total 9\n"
    )
    out = federate_engine_metrics(text, worker_id="w1", engine=EngineType.VLLM)
    assert "process_cpu_seconds_total" not in out
    assert 'worker_id="w1"' in out
    assert 'engine="vllm"' in out
    assert "vllm:e2e_request_latency_seconds_bucket" in out
    assert "vllm:num_requests_swapped" in out
    assert "vllm:cpu_cache_usage_perc" in out
    assert "vllm:request_success_total" in out
    assert "# HELP vllm:e2e_request_latency_seconds" in out


def test_federate_does_not_duplicate_existing_worker_id():
    text = 'vllm:num_requests_running{model_name="m",worker_id="keep"} 1\n'
    out = federate_engine_metrics(text, worker_id="new", engine="vllm")
    assert 'worker_id="keep"' in out
    assert 'worker_id="new"' not in out
    assert 'engine="vllm"' in out


def test_federate_sglang_scheduler_and_kv():
    text = (
        'sglang:num_running_reqs{tp_rank="0"} 2\n'
        'sglang:num_queue_reqs{tp_rank="0"} 4\n'
        'sglang:token_usage{tp_rank="0"} 0.55\n'
        'sglang:cache_hit_rate{model_name="m"} 0.8\n'
        "other_metric 1\n"
    )
    out = federate_engine_metrics(text, worker_id="p0", engine="sglang")
    assert "other_metric" not in out
    assert 'worker_id="p0"' in out
    assert "sglang:token_usage" in out
    assert "sglang:num_queue_reqs" in out


def test_federate_unknown_engine_is_empty():
    assert (
        federate_engine_metrics("vllm:num_requests_running 1\n", worker_id="w", engine="atom") == ""
    )
