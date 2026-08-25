###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Measuring is the expensive step, so what one run brings back decides its worth.

An anchor is harvested once and then answers questions for every projection that
reuses it, which means it has to cover the axes it will be *asked* about rather
than only the point it was taken at: the projector transports a measured curve
and cannot invent one from a single point.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from infera.projection.core.projection.inference_projection import benchmark, benchmark_vllm


def _inference_config(concurrency=32, input_len=1024, output_len=128, tp=8, ep=1):
    return SimpleNamespace(
        model_parallel_config=SimpleNamespace(
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=1,
            expert_model_parallel_size=ep,
        ),
        request_config=SimpleNamespace(
            input_seq_len=input_len,
            output_seq_len=output_len,
            max_concurrency=concurrency,
            batch_size=concurrency,
        ),
    )


def _spawn_argv(monkeypatch, tmp_path, **cfg):
    """Run the anchor spawn with the harness stubbed; return the argv it asked for."""
    seen = {}
    save = tmp_path / "anchor.json"

    def fake_main(argv):
        seen["argv"] = argv
        save.write_text(json.dumps({"backend": "vllm", "sweep": []}))

    monkeypatch.setattr(benchmark_vllm, "main", fake_main)
    args = SimpleNamespace(
        bench_model="openai/gpt-oss-120b",
        save_benchmark=str(save),
        benchmark_gpus=None,
    )
    benchmark.spawn_inference_benchmark(args, _inference_config(**cfg))
    return seen["argv"]


def _argv_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


# --- the anchor has to cover the axes it will be asked about -----------------

def test_benchmark_mode_asks_for_a_swept_anchor_not_a_single_batch(monkeypatch, tmp_path):
    """A one-batch anchor is not a curve, and the projector cannot make it one.

    Batch transport interpolates measured points and deliberately never falls
    back to the analytical model, so a lone point is held flat across every
    batch. That is silently wrong for exactly the questions benchmark mode is
    run to answer -- a load sweep, or a search that varies concurrency -- and it
    is wrong while wearing a measurement's credibility.
    """
    argv = _spawn_argv(monkeypatch, tmp_path, concurrency=32)
    assert _argv_value(argv, "--concurrency") == "32"
    assert "--batch" not in argv


def test_the_anchor_is_taken_at_the_recipe_being_projected(monkeypatch, tmp_path):
    """Measuring some other shape than the one asked for is worse than not
    measuring: the regime check would still accept it."""
    argv = _spawn_argv(monkeypatch, tmp_path, concurrency=8, input_len=4096, output_len=256)
    assert _argv_value(argv, "--input-len") == "4096"
    assert _argv_value(argv, "--output-len") == "256"
    assert _argv_value(argv, "--tp") == "8"
    assert "--enable-expert-parallel" not in argv


def test_expert_parallelism_reaches_the_engine_it_is_measured_on(monkeypatch, tmp_path):
    """A dense engine is not the engine an MoE recipe runs on.

    Expert parallelism changes which kernels execute -- a per-rank expert slice
    and an all-to-all, rather than a tensor-sliced expert -- so measuring
    without it anchors the wrong engine. A serving engine has no separate
    expert axis, so it arrives as a flag on a TP group wide enough to hold the
    experts.
    """
    argv = _spawn_argv(monkeypatch, tmp_path, tp=1, ep=8)
    assert "--enable-expert-parallel" in argv
    # EP == TP on a serving engine, so the group has to be wide enough.
    assert _argv_value(argv, "--tp") == "8"
