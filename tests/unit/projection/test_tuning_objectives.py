###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A serving tuner can only optimize what it can read off a projection.

The projector reports twenty-odd serving metrics; the tuner parsed nine of them
and could rank on six. Everything else was invisible, so asking for the best
configuration by total tokens per GPU -- the figure InferenceX actually leads
with -- was not a question the tuner could be asked, and the one it answered
instead (decode-only throughput) has a different winner whenever the prompt
dominates the generation.
"""

from __future__ import annotations

from infera.projection.agents.tuning_agent.inference_tuning import (
    INFERENCEX_OBJECTIVES,
    objective_is_minimize,
    parse_inference_metrics,
    resolve_objective,
    score_result,
)

# A verbatim serving projection: DeepSeek-V4-Pro, TP8 with attention DP 8, at
# 130k/900 on MI355X.
PROJECTION = """
  Max sustainable concurrency: 360  (HBM=288 GB via --hbm-capacity-gb)
  Concurrency used: 32
  TTFT (time to first token):      2401.06 ms
  ITL / TPOT (per token):          30.76 ms
  Interactivity (per user):        32.5 tok/s/user
  Decode step latency (pure):      22.17 ms  | mixed: 984.25 ms
    Mixed-step fraction:           0.89%  -> TPOT pollution: 28.5%
  End-to-end request latency:      30057.02 ms
  Per-request decode throughput:   32.5 tok/s
  Aggregate decode throughput:     1041.4 tok/s
  Decode throughput / GPU:         130.2 tok/s/gpu
  Total throughput / GPU:          18932.6 tok/s/gpu   (in+out over 8 GPU)
  Prefill throughput:              856141.8 tok/s
  Replica GPUs (TP x PP):          8
  Projected Total Memory:   112.0271 GB
  Weights (fp4):           93.7605 GB
  KV cache (fp8):         17.1337 GB
  Activation working set:   1.1328 GB
  Max concurrent sequences: 360
    prefill:  TP-AR 293.35 | EP-A2A 0.00 | PP-P2P 0.00 | total 293.35
    decode:   TP-AR 0.24 | EP-A2A 0.00 | PP-P2P 0.00 | total 0.24
"""


def test_the_headline_inferencex_metric_is_read():
    """Total tokens per GPU, prompt plus generation. Previously unparsed."""
    m = parse_inference_metrics(PROJECTION)
    assert m["total_throughput_tps_per_gpu"] == 18932.6


def test_total_throughput_is_not_decode_throughput():
    """They rank configurations differently, so conflating them picks wrong.

    Here they differ by 145x because the prompt is 144x the generation.
    """
    m = parse_inference_metrics(PROJECTION)
    assert m["total_throughput_tps_per_gpu"] > 100 * m["decode_throughput_tps_per_gpu"]


def test_the_prompt_side_is_reported_per_gpu_as_well_as_whole():
    """Ranking on a fleet total just rewards buying more GPUs."""
    m = parse_inference_metrics(PROJECTION)
    assert m["prefill_throughput_tps"] == 856141.8
    assert m["prefill_throughput_tps_per_gpu"] == 856141.8 / 8
    assert m["total_throughput_tps"] == 18932.6 * 8


def test_what_one_user_feels_is_separate_from_what_the_fleet_delivers():
    m = parse_inference_metrics(PROJECTION)
    assert m["interactivity_tok_s_per_user"] == 32.5
    assert m["per_request_decode_tps"] == 32.5


def test_scheduler_interference_is_measurable():
    """Prefill chunks landing in decode steps is a tunable cost, not a given."""
    m = parse_inference_metrics(PROJECTION)
    assert m["mixed_step_fraction_pct"] == 0.89
    assert m["tpot_pollution_pct"] == 28.5
    assert m["decode_step_ms_pure"] == 22.17


def test_every_catalogued_objective_resolves_to_something_real():
    """An objective the projection never reports scores None on every trial,
    which the search then reports as "no legal serving config found" -- which is
    indistinguishable from a genuinely infeasible model. Four shipped that way.
    """
    m = parse_inference_metrics(PROJECTION)
    for obj in INFERENCEX_OBJECTIVES:
        assert resolve_objective(obj) == obj, f"{obj} is not canonical"
        assert obj in m, f"{obj} is catalogued but never populated"


def test_direction_is_right_for_each_objective():
    """Scoring is signed so higher always wins; a sign slip inverts the search."""
    for obj in (
        "ttft_ms",
        "itl_ms",
        "request_latency_ms",
        "memory_per_gpu_gb",
        "kv_cache_gb",
        "tpot_pollution_pct",
        "decode_step_ms_pure",
    ):
        assert objective_is_minimize(obj), f"{obj} should be minimized"
    for obj in (
        "total_throughput_tps_per_gpu",
        "decode_throughput_tps_per_gpu",
        "prefill_throughput_tps_per_gpu",
        "interactivity_tok_s_per_user",
        "max_concurrent_sequences",
        "mfu",
    ):
        assert not objective_is_minimize(obj), f"{obj} should be maximized"
    # Lower latency must score higher.
    assert score_result({"ttft_ms": 100.0}, "ttft_ms") > score_result({"ttft_ms": 900.0}, "ttft_ms")


def test_the_names_people_actually_use_reach_the_right_metric():
    assert resolve_objective("tput_per_gpu") == "total_throughput_tps_per_gpu"
    assert resolve_objective("output_tput_per_gpu") == "decode_throughput_tps_per_gpu"
    assert resolve_objective("input_tput_per_gpu") == "prefill_throughput_tps_per_gpu"
    assert resolve_objective("intvty") == "interactivity_tok_s_per_user"
    assert resolve_objective("min_tpot") == "itl_ms"
    assert resolve_objective("e2el") == "request_latency_ms"


def test_what_cannot_be_projected_is_not_offered():
    """InferenceX reports power and joules/token; the projector models neither.
    MFU and TFLOP/s are training-mode figures the serving path never emits.

    An objective that silently scores None would look like a failed search, and
    one that guessed would be worse. They are simply not offered.
    """
    from infera.projection.agents.tuning_agent.inference_tuning import (
        UNSUPPORTED_OBJECTIVES,
    )

    for absent in UNSUPPORTED_OBJECTIVES:
        assert absent not in INFERENCEX_OBJECTIVES
    assert {"avg_power_w", "mfu", "tflops_per_s_per_gpu"} <= UNSUPPORTED_OBJECTIVES


def test_the_step_that_absorbs_a_prefill_chunk_is_measurable():
    m = parse_inference_metrics(PROJECTION)
    assert m["decode_step_ms_pure"] == 22.17
    assert m["decode_step_ms_mixed"] == 984.25


def test_collective_time_is_split_by_phase():
    """Prefill and decode pay wildly different collective costs, and a single
    blended number hides which one a parallelism choice is hurting."""
    m = parse_inference_metrics(PROJECTION)
    assert m["prefill_comm_ms"] == 293.35
    assert m["decode_comm_ms"] == 0.24
