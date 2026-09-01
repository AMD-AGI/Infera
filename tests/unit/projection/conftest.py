###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Run the inference projector in-process and hand back a flat metrics dict.

The projector's entry point is its CLI, so these tests drive it the way a user
does -- through ``build_parser`` + ``launch_projection_from_cli`` -- rather than
assembling config dataclasses by hand, which would let a regression in argument
plumbing slip through.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile

import pytest

# Mirrors Hyperloom's InferaSim bridge template: the model preset and parallelism
# arrive as env / CLI overrides, so one workload file drives every case.
_WORKLOAD_YAML = """\
work_group: infera
user_name: unit-test
exp_name: projection-regression
workspace: {workspace}

modules:
  pre_trainer:
    framework: megatron
    config: pre_trainer.yaml
    model: ${{INFERASIM_MODEL:gpt_oss_120B}}.yaml
    overrides:
      seq_length: 4096
      max_position_embeddings: 4096
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      mock_data: true
      train_data_path: null
      valid_data_path: null
      test_data_path: null
"""

_workload_path = None


def _workload_file():
    global _workload_path
    if _workload_path is None:
        tmp = tempfile.mkdtemp(prefix="inferasim-test-")
        _workload_path = os.path.join(tmp, "workload.yaml")
        with open(_workload_path, "w") as fh:
            fh.write(_WORKLOAD_YAML.format(workspace=os.path.join(tmp, "output")))
    return _workload_path


DEFAULTS = {
    "model": "gpt_oss_120B",
    "tp": 8,
    "ep": 1,
    "pp": 1,
    "concurrency": 64,
    "input_len": 1024,
    "output_len": 128,
    "weight_dtype": "mxfp4",
    "kv_cache_dtype": "bf16",
    "gpu_arch": "mi355x",
    "hbm_gb": 288.0,
}


def project_spec(**overrides):
    """Project one workload; returns ``{metric: value}``."""
    yaml = pytest.importorskip("yaml")  # noqa: F841 - projection extra
    from infera.projection.cli import build_parser
    from infera.projection.core.projection.inference_projection import (
        launch_projection_from_cli,
    )

    spec = {**DEFAULTS, **overrides}
    argv = [
        "inference",
        "--config", _workload_file(),
        "--inference-mode", "both",
        "--profiling-mode", "simulate",
        "--input-len", str(spec["input_len"]),
        "--output-len", str(spec["output_len"]),
        "--inference-batch-size", str(spec["concurrency"]),
        "--max-concurrency", str(spec["concurrency"]),
        "--weight-dtype", spec["weight_dtype"],
        "--kv-cache-dtype", spec["kv_cache_dtype"],
        "--gpu-arch", spec["gpu_arch"],
        "--hbm-capacity-gb", str(spec["hbm_gb"]),
    ]
    for flag, key in (
        ("--sliding-window", "sliding_window"),
        ("--max-num-batched-tokens", "max_num_batched_tokens"),
        ("--stream-interval", "stream_interval"),
        ("--decode-admission-steps", "decode_admission_steps"),
        ("--decode-kernel-occupancy-us", "decode_kernel_occupancy_us"),
        ("--gpu-cost-per-hour", "gpu_cost_per_hour"),
        ("--attention-dp-size", "attn_dp"),
        ("--sparse-attention-topk", "sparse_attention_topk"),
    ):
        if spec.get(key) is not None:
            argv += [flag, str(spec[key])]
    if spec.get("enable_deepep"):
        argv += ["--enable-deepep"]
    argv += [
        f"tensor_model_parallel_size={spec['tp']}",
        f"expert_model_parallel_size={spec['ep']}",
        f"pipeline_model_parallel_size={spec['pp']}",
    ]

    prev = os.environ.get("INFERASIM_MODEL")
    os.environ["INFERASIM_MODEL"] = spec["model"]
    try:
        args, extra = build_parser().parse_known_args(argv)
        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            results = launch_projection_from_cli(args, extra)
    finally:
        if prev is None:
            os.environ.pop("INFERASIM_MODEL", None)
        else:
            os.environ["INFERASIM_MODEL"] = prev

    perf, mem = results.get("performance"), results.get("memory")
    cfg = results.get("config")
    mc = getattr(cfg, "model_config", None)
    req = getattr(cfg, "request_config", None)
    extras = dict(getattr(perf, "extras", {}) or {})
    gib = 1024.0 ** 3
    return {
        "sliding_window": req.resolved_sliding_window(getattr(mc, "sink_sliding_window", 0)),
        "shared_expert_size": getattr(mc, "moe_shared_expert_intermediate_size", None),
        "ttft_ms": getattr(perf, "ttft_ms", None),
        "tpot_ms": getattr(perf, "itl_ms", None),
        "decode_tps": getattr(perf, "decode_throughput_tps", None),
        "e2el_ms": getattr(perf, "request_latency_ms", None),
        "memory_gb": float(getattr(mem, "total_bytes", 0) or 0) / gib,
        "kv_cache_gb": float(getattr(mem, "kv_cache_bytes", 0) or 0) / gib,
        "sustainable_concurrency": extras.get("sustainable_concurrency", 0),
        # The pure (unmixed) decode step. Distinct from ``tpot_ms``, which under
        # continuous batching blends in the more expensive mixed prefill+decode
        # steps; scheduler waits are counted in whole decode steps, not TPOT.
        "decode_step_ms": extras.get(
            "pure_step_latency_ms", getattr(perf, "decode_step_latency_ms", None)
        ),
        "replica_gpus": getattr(perf, "replica_gpus", None),
        "max_concurrent_sequences": getattr(mem, "max_concurrent_sequences", None),
        "comm_decode_tp_allreduce_ms": extras.get("comm_decode_tp_allreduce_ms"),
        "comm_decode_ep_a2a_ms": extras.get("comm_decode_ep_a2a_ms"),
        # What the run printed, for the parts of the report that are computed
        # nowhere else.
        "report": report.getvalue(),
    }
