###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Per-rank weight memory under attention-DP, and DeepSeek-V4's attention weights.

Anchors are vLLM's ``Model loading took`` on MI325X (rocm/vllm 0.29.1) serving
deepseek-ai/DeepSeek-V4-Pro: tp4 201.79 GiB, tp4ep4dp4 220.31, tp8ep8 103.02,
tp8ep8dp8 124.65.
"""

from __future__ import annotations

import contextlib
import io
import os

import pytest

from .conftest import _workload_file

_GIB = 1024.0**3


def _weights_gib(tp: int, attn_dp: int = 1, engine: str = "vllm") -> float:
    return _project(tp, attn_dp, engine)["memory"].weight_bytes / _GIB


def _project(tp: int, attn_dp: int = 1, engine: str = "vllm") -> dict:
    pytest.importorskip("yaml")
    from infera.projection.cli import build_parser
    from infera.projection.core.projection.inference_projection import (
        launch_projection_from_cli,
    )

    argv = [
        "inference",
        "--config",
        _workload_file(),
        "--inference-mode",
        "memory",
        "--profiling-mode",
        "simulate",
        "--serving-engine",
        engine,
        "--input-len",
        "8192",
        "--output-len",
        "1024",
        "--inference-batch-size",
        "64",
        "--max-concurrency",
        "64",
        "--weight-dtype",
        "fp8",
        "--moe-expert-dtype",
        "mxfp4",
        "--linear-weight-dtype",
        "fp8",
        "--kv-cache-dtype",
        "fp8",
        "--gpu-arch",
        "mi325x",
        "--hbm-capacity-gb",
        "256",
    ]
    if attn_dp > 1:
        argv += ["--attention-dp-size", str(attn_dp)]
    argv += [
        f"tensor_model_parallel_size={tp}",
        f"expert_model_parallel_size={tp}",
        "pipeline_model_parallel_size=1",
    ]
    prev = os.environ.get("INFERASIM_MODEL")
    os.environ["INFERASIM_MODEL"] = "deepseek_v4_pro"
    try:
        args, extra = build_parser().parse_known_args(argv)
        with contextlib.redirect_stdout(io.StringIO()):
            res = launch_projection_from_cli(args, extra)
    finally:
        if prev is None:
            os.environ.pop("INFERASIM_MODEL", None)
        else:
            os.environ["INFERASIM_MODEL"] = prev
    return res


@pytest.mark.parametrize(
    "tp, attn_dp, measured",
    [(4, 1, 201.79), (4, 4, 220.31), (8, 1, 103.02), (8, 8, 124.65)],
)
def test_dsv4_pro_weights_track_what_vllm_loads(tp, attn_dp, measured):
    assert _weights_gib(tp, attn_dp) == pytest.approx(measured, rel=0.05)


def test_vllm_attention_dp_replicates_every_non_expert_weight():
    """A vLLM DP rank is a whole engine: what tp splits, tp/dp splits instead.

    Measured: +18.5 GiB going tp4 -> tp4ep4dp4 and +21.6 going tp8ep8 ->
    tp8ep8dp8, i.e. (1 - 1/dp) of one ~25 GiB non-expert block both times.
    """
    dp4 = _weights_gib(4, 4) - _weights_gib(4)
    dp8 = _weights_gib(8, 8) - _weights_gib(8)
    assert dp4 == pytest.approx(18.5, rel=0.15)
    assert dp8 == pytest.approx(21.6, rel=0.15)
    assert dp4 / 3 * 4 == pytest.approx(dp8 / 7 * 8, rel=0.02)


def test_sglang_attention_dp_replicates_attention_only():
    vllm = _weights_gib(8, 8) - _weights_gib(8)
    sglang = _weights_gib(8, 8, "sglang") - _weights_gib(8, 1, "sglang")
    assert 0 < sglang < vllm


def test_v4_attention_is_counted_from_its_own_shapes():
    """The checkpoint holds 19.8B attention parameters across 61 layers.

    Dense 128-head Q/K/V/O at a 512 head width would be 57.8B; V4 has one KV
    head and a grouped low-rank output instead.
    """
    from infera.projection.core.projection.inference_projection import memory

    cfg = _project(8)["config"]
    sharded, replicated = memory._v4_attention_params(cfg, range(cfg.model_config.num_layers))
    assert (sharded + replicated) / 1e9 == pytest.approx(19.8, rel=0.1)
    assert replicated < sharded / 5
