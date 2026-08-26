###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Minimal command-line surface for ``python -m infera.planner``."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

DEFAULT_TTFT_MS = 500.0
DEFAULT_ITL_MS = 50.0
DEFAULT_ADJUSTMENT_INTERVAL = 180.0
DEFAULT_METRICS_URL = "http://127.0.0.1:8000/metrics"


@dataclass
class PlannerArgs:
    """Configuration for the offline-profile-backed decision loop."""

    # SLA targets, in milliseconds to match the profiling data.
    ttft_ms: float = DEFAULT_TTFT_MS
    itl_ms: float = DEFAULT_ITL_MS

    # How long each observation window is, and therefore how often a decision
    # is produced.
    adjustment_interval: float = DEFAULT_ADJUSTMENT_INTERVAL

    profile_results: str = ""
    model: str = ""
    metrics_urls: list[str] = field(default_factory=list)

    # Override the GPUs-per-replica recorded in the profiling data. None keeps
    # the profiled value, which is what the throughput numbers were measured at.
    prefill_engine_num_gpu: int | None = None
    decode_engine_num_gpu: int | None = None

    log_level: str = "INFO"


def parse_planner_args(argv: list[str] | None = None) -> PlannerArgs:
    parser = argparse.ArgumentParser(
        prog="python -m infera.planner",
        description="SLA-based planner: decide how many prefill/decode replicas "
        "meet the TTFT/ITL targets.",
    )

    sla = parser.add_argument_group("SLA targets")
    sla.add_argument(
        "--ttft",
        type=float,
        default=DEFAULT_TTFT_MS,
        metavar="MS",
        help="Time-to-first-token target in milliseconds. Sizes the prefill pool "
        "whenever queueing puts TTFT over it (default: %(default)s).",
    )
    sla.add_argument(
        "--itl",
        type=float,
        default=DEFAULT_ITL_MS,
        metavar="MS",
        help="Inter-token-latency target in milliseconds. Sets the operating point "
        "the decode pool is sized to hold (default: %(default)s).",
    )

    obs = parser.add_argument_group("observation")
    obs.add_argument(
        "--metrics-url",
        action="append",
        default=None,
        metavar="URL",
        help="An Infera server /metrics endpoint to scrape. Repeat once per "
        f"server replica; per-endpoint window deltas are then summed. "
        f"(default: {DEFAULT_METRICS_URL})",
    )
    obs.add_argument(
        "--adjustment-interval",
        type=float,
        default=DEFAULT_ADJUSTMENT_INTERVAL,
        metavar="SECONDS",
        help="Length of each observation window, and how often a decision is "
        "produced (default: %(default)s).",
    )
    model = parser.add_argument_group("performance model")
    model.add_argument(
        "--model",
        required=True,
        help="Served model whose disaggregated metrics and profile should be used.",
    )
    model.add_argument(
        "--profile-results",
        required=True,
        metavar="PATH",
        help="Offline profiling results (JSON), as produced by `python -m infera.planner.profile`.",
    )
    model.add_argument(
        "--prefill-engine-num-gpu",
        type=int,
        default=None,
        metavar="N",
        help="GPUs per prefill replica. Defaults to the value in the profiling data.",
    )
    model.add_argument(
        "--decode-engine-num-gpu",
        type=int,
        default=None,
        metavar="N",
        help="GPUs per decode replica. Defaults to the value in the profiling data.",
    )

    misc = parser.add_argument_group("misc")
    misc.add_argument("--log-level", default="INFO")

    ns = parser.parse_args(argv)

    if ns.adjustment_interval <= 0:
        parser.error("--adjustment-interval must be positive")
    if ns.ttft <= 0 or ns.itl <= 0:
        parser.error("--ttft and --itl must be positive")
    if ns.prefill_engine_num_gpu is not None and ns.prefill_engine_num_gpu <= 0:
        parser.error("--prefill-engine-num-gpu must be positive")
    if ns.decode_engine_num_gpu is not None and ns.decode_engine_num_gpu <= 0:
        parser.error("--decode-engine-num-gpu must be positive")

    return PlannerArgs(
        ttft_ms=ns.ttft,
        itl_ms=ns.itl,
        adjustment_interval=ns.adjustment_interval,
        profile_results=ns.profile_results,
        model=ns.model,
        metrics_urls=ns.metrics_url or [DEFAULT_METRICS_URL],
        prefill_engine_num_gpu=ns.prefill_engine_num_gpu,
        decode_engine_num_gpu=ns.decode_engine_num_gpu,
        log_level=ns.log_level,
    )
