###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Command-line surface for ``python -m infera.planner``."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from infera.planner.predictor import PREDICTORS

DEFAULT_TTFT_MS = 500.0
DEFAULT_ITL_MS = 50.0
DEFAULT_ADJUSTMENT_INTERVAL = 180.0


@dataclass
class PlannerArgs:
    """Resolved planner configuration.

    A plain dataclass rather than an ``argparse.Namespace`` so the scaling core
    can be constructed directly in tests without going through the parser.
    """

    # SLA targets, in milliseconds to match the profiling data.
    ttft_ms: float = DEFAULT_TTFT_MS
    itl_ms: float = DEFAULT_ITL_MS

    # How long each observation window is, and therefore how often the
    # deployment is resized. Must comfortably exceed the time it takes an
    # engine replica to become ready, or decisions pile up on each other.
    adjustment_interval: float = DEFAULT_ADJUSTMENT_INTERVAL

    profile_results: str = ""
    metrics_urls: list[str] = field(default_factory=list)

    # Override the GPUs-per-replica recorded in the profiling data. None keeps
    # the profiled value, which is what the throughput numbers were measured at.
    prefill_engine_num_gpu: int | None = None
    decode_engine_num_gpu: int | None = None

    min_endpoint: int = 1
    max_gpu_budget: int = 8

    load_predictor: str = "constant"
    predictor_window: int = 10

    connector: str = "virtual"
    no_operation: bool = False

    # virtual connector
    etcd_endpoint: str = "127.0.0.1:2379"
    etcd_prefix: str = "/infera/workers"

    # kubernetes connector
    deployment_name: str = ""
    k8s_namespace: str | None = None
    prefill_service: str = "prefill"
    decode_service: str = "decode"

    metrics_port: int = 9085
    log_level: str = "INFO"


def parse_planner_args(argv: list[str] | None = None) -> PlannerArgs:
    parser = argparse.ArgumentParser(
        prog="python -m infera.planner",
        description="SLA-based planner: size the prefill/decode pools to meet TTFT/ITL targets.",
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
        "server replica; the samples are summed before windowing. "
        "(default: http://127.0.0.1:8000/metrics)",
    )
    obs.add_argument(
        "--adjustment-interval",
        type=float,
        default=DEFAULT_ADJUSTMENT_INTERVAL,
        metavar="SECONDS",
        help="Length of each observation window, and how often the deployment is "
        "resized. Keep it well above engine startup time (default: %(default)s).",
    )
    obs.add_argument(
        "--load-predictor",
        choices=sorted(PREDICTORS),
        default="constant",
        help="How to forecast the next interval's load (default: %(default)s).",
    )
    obs.add_argument(
        "--predictor-window",
        type=int,
        default=10,
        metavar="N",
        help="How many intervals of history the predictor keeps (default: %(default)s).",
    )

    model = parser.add_argument_group("performance model")
    model.add_argument(
        "--profile-results",
        required=True,
        metavar="PATH",
        help="Pre-deployment profiling results (JSON). See the SLA planner guide for the schema.",
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

    limits = parser.add_argument_group("limits")
    limits.add_argument(
        "--min-endpoint",
        type=int,
        default=1,
        metavar="N",
        help="Never scale either pool below this many replicas (default: %(default)s).",
    )
    limits.add_argument(
        "--max-gpu-budget",
        type=int,
        default=8,
        metavar="N",
        help="Total GPUs the planner may allocate across both pools (default: %(default)s).",
    )

    act = parser.add_argument_group("actuation")
    act.add_argument(
        "--connector",
        choices=("virtual", "kubernetes"),
        default="virtual",
        help="virtual: publish the decision to etcd for an external executor. "
        "kubernetes: patch the InferaDeployment directly (default: %(default)s).",
    )
    act.add_argument(
        "--no-operation",
        action="store_true",
        help="Observe and log decisions without applying them. Use this to "
        "validate profiling data against a live workload before handing over "
        "control of the deployment.",
    )
    act.add_argument(
        "--etcd-endpoint",
        default="127.0.0.1:2379",
        metavar="HOST:PORT",
        help="etcd endpoint for the virtual connector (default: %(default)s).",
    )
    act.add_argument(
        "--etcd-prefix",
        default="/infera/workers",
        metavar="PREFIX",
        help="etcd key prefix; the decision is written under <prefix>/planner/decision "
        "(default: %(default)s).",
    )
    act.add_argument(
        "--deployment-name",
        default="",
        metavar="NAME",
        help="InferaDeployment to patch (required for --connector kubernetes).",
    )
    act.add_argument(
        "--k8s-namespace",
        default=None,
        metavar="NS",
        help="Namespace of the InferaDeployment. Defaults to the planner pod's own.",
    )
    act.add_argument(
        "--prefill-service",
        default="prefill",
        metavar="NAME",
        help="Key under spec.services holding the prefill pool (default: %(default)s).",
    )
    act.add_argument(
        "--decode-service",
        default="decode",
        metavar="NAME",
        help="Key under spec.services holding the decode pool (default: %(default)s).",
    )

    misc = parser.add_argument_group("misc")
    misc.add_argument(
        "--metrics-port",
        type=int,
        default=9085,
        metavar="PORT",
        help="Port for the planner's own /metrics endpoint; 0 disables it (default: %(default)s).",
    )
    misc.add_argument("--log-level", default="INFO")

    ns = parser.parse_args(argv)

    if ns.connector == "kubernetes" and not ns.deployment_name:
        parser.error("--connector kubernetes requires --deployment-name")
    if ns.adjustment_interval <= 0:
        parser.error("--adjustment-interval must be positive")
    if ns.ttft <= 0 or ns.itl <= 0:
        parser.error("--ttft and --itl must be positive")
    if ns.min_endpoint < 0:
        parser.error("--min-endpoint cannot be negative")
    if ns.max_gpu_budget < 1:
        parser.error("--max-gpu-budget must be at least 1")

    return PlannerArgs(
        ttft_ms=ns.ttft,
        itl_ms=ns.itl,
        adjustment_interval=ns.adjustment_interval,
        profile_results=ns.profile_results,
        metrics_urls=ns.metrics_url or ["http://127.0.0.1:8000/metrics"],
        prefill_engine_num_gpu=ns.prefill_engine_num_gpu,
        decode_engine_num_gpu=ns.decode_engine_num_gpu,
        min_endpoint=ns.min_endpoint,
        max_gpu_budget=ns.max_gpu_budget,
        load_predictor=ns.load_predictor,
        predictor_window=ns.predictor_window,
        connector=ns.connector,
        no_operation=ns.no_operation,
        etcd_endpoint=ns.etcd_endpoint,
        etcd_prefix=ns.etcd_prefix,
        deployment_name=ns.deployment_name,
        k8s_namespace=ns.k8s_namespace,
        prefill_service=ns.prefill_service,
        decode_service=ns.decode_service,
        metrics_port=ns.metrics_port,
        log_level=ns.log_level,
    )
