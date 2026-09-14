###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A default is a claim, so it should be the mode whose numbers are defensible.

Only the calibrated path has an established correlation against real serving.
For as long as ``inference`` defaulted to ``simulate``, saying nothing got you
the uncalibrated projection and the tool's most-used answer was its least
supportable one. Measuring is the default now, which moves the burden: the
uncalibrated path is still there, but it has to be asked for.

The cost of that flip is that two things must stay explicit, or the search
paths silently start spawning serving engines per candidate.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

# --- the default itself ------------------------------------------------------


def test_inference_defaults_to_measuring():
    """The subparser must not override the flag back to ``simulate``."""
    from infera.projection.cli import build_parser

    args, _ = build_parser().parse_known_args(["inference", "--config", "x.yaml"])
    assert args.profiling_mode == "benchmark"


def test_the_uncalibrated_path_is_still_reachable():
    """Not-the-default is not the same as gone."""
    from infera.projection.cli import build_parser

    for mode in ("simulate", "benchmark", "both"):
        args, _ = build_parser().parse_known_args(
            ["inference", "--config", "x.yaml", "--profiling-mode", mode]
        )
        assert args.profiling_mode == mode


# --- what a laptop sees now --------------------------------------------------


def test_no_accelerator_names_every_way_out(monkeypatch):
    """Measuring by default means an ordinary laptop run lands on this error.

    It is the first thing a new user sees, so it has to be an instruction
    rather than a complaint: an anchor file, an anchor store, or an explicit
    request for the analytical projection.
    """
    from infera.projection.core.projection.inference_projection import benchmark

    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    with pytest.raises(RuntimeError) as exc:
        benchmark.assert_measurable()

    msg = str(exc.value)
    assert "--load-benchmark" in msg
    assert "--anchor-store" in msg
    assert "--profiling-mode simulate" in msg


# --- the search paths must not inherit the new default -----------------------


def _cfg():
    return SimpleNamespace(
        input_len=1024,
        output_len=1024,
        batch_size=32,
        max_concurrency=32,
        weight_dtype="fp8",
        kv_cache_dtype="fp8",
        chunked_prefill_size=0,
        speculative_num_tokens=0,
        ep_load_balance=1.0,
    )


def _agent_cfg():
    return SimpleNamespace(
        target_cluster=SimpleNamespace(gpu_arch="mi355x", gpu_clock_mhz=None, num_nodes=1),
        optimization=SimpleNamespace(hbm_capacity_gb=288.0),
    )


def test_the_tuning_agent_asks_for_simulate_out_loud(tmp_path: Path):
    """A search scores thousands of candidates on the cost model.

    It used to get that by omitting the flag and inheriting the CLI default.
    With the default flipped, omission would spawn a serving engine per
    candidate, so the intent has to be stated rather than assumed.
    """
    from infera.projection.agents.tuning_agent.evaluator import _build_inference_cmd

    cmd = _build_inference_cmd(
        tmp_path / "w.yaml",
        _cfg(),
        _agent_cfg(),
        tmp_path,
    )
    assert "--profiling-mode" in cmd
    assert cmd[cmd.index("--profiling-mode") + 1] == "simulate"


def test_an_explicit_measurement_request_still_measures(tmp_path: Path):
    from infera.projection.agents.tuning_agent.evaluator import _build_inference_cmd

    cmd = _build_inference_cmd(
        tmp_path / "w.yaml",
        _cfg(),
        _agent_cfg(),
        tmp_path,
        profiling_mode="benchmark",
        bench_gpus=4,
    )
    assert cmd[cmd.index("--profiling-mode") + 1] == "benchmark"


def test_a_loaded_anchor_needs_no_mode_at_all(tmp_path: Path):
    """``--load-benchmark`` already carries the measurement, so neither
    measuring again nor declaring the analytical path is right."""
    from infera.projection.agents.tuning_agent.evaluator import _build_inference_cmd

    cmd = _build_inference_cmd(
        tmp_path / "w.yaml",
        _cfg(),
        _agent_cfg(),
        tmp_path,
        load_benchmark=tmp_path / "anchor.json",
    )
    assert "--load-benchmark" in cmd
    assert "--profiling-mode" not in cmd


def test_sweeps_still_force_the_no_gpu_path():
    """A sweep that needed a GPU per point would not be a sweep."""
    src = Path("infera/projection/core/projection/inference_projection/sweep.py").read_text()
    # Read as a token sequence rather than one string, because a formatter is
    # free to put each list element on its own line and did.
    args = re.findall(r'"(--[a-z-]+|simulate)"', src)
    assert "--profiling-mode" in args
    assert args[args.index("--profiling-mode") + 1] == "simulate"
