###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A disaggregated deployment is two experiments, so it needs two anchors.

Prefill and decode are separated precisely because they are not the same
computation. One runs a long prompt through a compute-bound forward pass at
whatever width keeps TTFT inside the SLO; the other streams single tokens
against a memory-bound weight read, at its own parallelism, its own attention
layout and its own batch composition. That is the entire motivation for
splitting the pools.

The projector already modelled the two pools separately -- ``prefill_parallel``
and ``decode_parallel`` give each its own TP, EP and attention-DP -- but it
calibrated both from one artifact, because ``--load-benchmark`` takes one path
and both pool projectors were handed it. A colocated measurement is a
compromise between the two phases, so it describes neither pool exactly, and
the error does not announce itself: both pools return confident numbers.

These tests pin the routing and the refusals around it. They deliberately do
not assert a magnitude, because the magnitude is whatever the two measurements
happen to disagree by -- the invariant is that each pool is priced from the run
that measured it.
"""

from __future__ import annotations

import json

import pytest

from infera.projection.core.projection.inference_projection.performance import (
    InferencePerformanceProjector,
)

COLOCATED = {"meta": {"model": "m", "tp": 8}, "measured": {"model": 9.0}}
PREFILL_POOL = {"meta": {"model": "m", "tp": 4}, "measured": {"model": 5.0}}
DECODE_POOL = {"meta": {"model": "m", "tp": 16}, "measured": {"model": 3.0}}


class _Pools:
    """``_pool_anchor``'s choice, over just the state it reads.

    Bound per-call rather than at class creation so the rest of this module
    still collects against a build that does not have the method yet.
    """

    def __init__(self, *, shared=None, per_pool=None):
        self._bench_measured = shared
        self._pool_benchmarks = dict(per_pool or {})

    def _pool_anchor(self, pool):
        return InferencePerformanceProjector._pool_anchor(self, pool)


def test_each_pool_is_priced_from_the_run_that_measured_it():
    """The gap, stated as the routing it needs.

    Both pools used to receive ``self._bench_measured`` -- the same object --
    so there was no way to express a prefill measured at TP4 beside a decode
    measured at TP16, which is how these deployments are actually configured.
    """
    p = _Pools(
        shared=COLOCATED,
        per_pool={"prefill": PREFILL_POOL, "decode": DECODE_POOL},
    )
    assert p._pool_anchor("prefill") is PREFILL_POOL
    assert p._pool_anchor("decode") is DECODE_POOL


def test_one_pool_may_be_measured_without_the_other():
    """Partial calibration is the common case, not an error.

    Decode is the pool worth measuring first -- it sets ITL and the token
    throughput the deployment is sold on -- and a prefill harvest at the
    prefill pool's width may not exist yet. The unmeasured pool falls back to
    the shared anchor rather than refusing the whole projection.
    """
    p = _Pools(shared=COLOCATED, per_pool={"decode": DECODE_POOL})
    assert p._pool_anchor("decode") is DECODE_POOL
    assert p._pool_anchor("prefill") is COLOCATED


def test_a_run_without_the_per_pool_flags_is_unchanged():
    """The fallback has to be exactly the old behaviour.

    Both pools reading the shared anchor is what every existing disaggregated
    projection does, so this is the regression guard on the change itself.
    """
    p = _Pools(shared=COLOCATED)
    assert p._pool_anchor("prefill") is COLOCATED
    assert p._pool_anchor("decode") is COLOCATED


def test_no_anchor_at_all_stays_no_anchor():
    """An absent measurement must not become a truthy empty one: the projector
    reads this to decide whether it is calibrated or simulating."""
    p = _Pools()
    assert not p._pool_anchor("prefill")
    assert not p._pool_anchor("decode")


# --------------------------------------------------------------------------
# The flags that reach it
# --------------------------------------------------------------------------


def test_the_flags_exist_and_are_parsed():
    """The routing is unreachable if the arguments are not plumbed."""
    pytest.importorskip("yaml")
    from infera.projection.cli import build_parser

    args = build_parser().parse_args(
        [
            "inference",
            "--config",
            "unused.yaml",
            "--prefill-benchmark",
            "/tmp/p.json",
            "--decode-benchmark",
            "/tmp/d.json",
        ]
    )
    assert args.prefill_benchmark == "/tmp/p.json"
    assert args.decode_benchmark == "/tmp/d.json"


def test_per_pool_anchors_without_disaggregation_are_refused(tmp_path, monkeypatch):
    """Ignoring them would report a colocated projection as a calibrated split.

    The flags only mean something when there are two pools. Accepting them
    silently on a colocated run would answer a question about a deployment the
    user did not describe, and the reported numbers would look exactly as
    confident either way.
    """
    from infera.projection.core.projection.inference_projection import launcher

    path = tmp_path / "decode.json"
    path.write_text(json.dumps(DECODE_POOL))

    monkeypatch.setattr(launcher, "_assert_anchor_is_this_model", lambda *a, **k: None)
    with pytest.raises(ValueError, match="disaggregat"):
        launcher._load_pool_benchmarks(
            _Args(decode_benchmark=str(path)), disaggregation_enabled=False
        )


def test_a_pool_anchor_is_checked_against_the_target_model(tmp_path):
    """The same guard the shared anchor gets.

    A foreign anchor does not fail loudly when it is applied -- it prices one
    architecture's kernels onto another -- and routing it to a pool does not
    make it any safer. This path skips the store, so the check has to be here.
    """
    from infera.projection.core.projection.inference_projection import launcher

    path = tmp_path / "prefill.json"
    path.write_text(json.dumps({"meta": {"model": "openai/gpt-oss-120b"}}))

    with pytest.raises(ValueError, match="was measured on"):
        launcher._load_pool_benchmarks(
            _Args(prefill_benchmark=str(path), bench_model="deepseek-ai/DeepSeek-R1"),
            disaggregation_enabled=True,
        )


class _Args:
    def __init__(self, **kw):
        self.prefill_benchmark = None
        self.decode_benchmark = None
        self.bench_model = None
        for k, v in kw.items():
            setattr(self, k, v)
