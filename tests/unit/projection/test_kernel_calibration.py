###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""One profile of a serving stack corrects the kernel report for its other deployments.

The report times kernels as efficient implementations would run them, so a
stack's slow small kernels rank last and its own kernels (a pad's fill) aren't
listed. A calibration multiplies the slow ones' times and lists the engine's
own kernels as a share of each step; a speedup then buys what the kernel's
real share of the step allows.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infera.projection.core.projection.inference_projection import kernel_calibration as kc
from infera.projection.core.projection.inference_projection import kernel_speedups as ksp
from infera.projection.core.projection.inference_projection import kernel_tasks as kt
from infera.projection.core.projection.inference_projection import step_mix as sm

from .test_des_step_mix import _cfg, _mix, _replay, _Timed

CAL = {
    "schema": kc.SCHEMA,
    "version": 1,
    "factors": [
        {"op": "moe_topk", "factor": 1000.0},
        {"op": "attn_qkv", "factor": 2.0},
        {"op": "attn_qkv", "dims": {"m": 1}, "factor": 3.0},
    ],
    "extras": [{"op": "engine_fill", "share": 0.5, "kernels": ["FillFunctor"]}],
}


def _by_op(rows):
    out = {}
    for r in rows:
        out.setdefault(r["op"], []).append(r)
    return out


def test_parse_refuses_another_schema_and_non_positive_values():
    with pytest.raises(ValueError):
        kc.Calibration.parse({"schema": ksp.SCHEMA})
    for bad in (
        {"factors": [{"op": "x", "factor": 0}]},
        {"extras": [{"op": "x", "share": -0.1}]},
    ):
        with pytest.raises(ValueError):
            kc.Calibration.parse({"schema": kc.SCHEMA, **bad})
    cal = kc.Calibration.parse(CAL)
    assert cal.factor({"op": "attn_qkv", "m": 1}) == 3.0  # the most specific wins
    assert cal.factor({"op": "attn_qkv", "m": 8}) == 2.0
    assert cal.factor({"op": "mlp_down"}) is None


def test_factors_scale_the_report_and_extras_take_a_share_of_each_pool():
    p = _Timed(_cfg())
    trace, _ = _replay(p)
    plain = _by_op(trace.kernels())
    trace = _replay(_Timed(_cfg()))[0]
    trace.calibration = kc.Calibration.parse(CAL)
    rows = trace.kernels()
    cal = _by_op(rows)
    before = {r["tokens"]: r["est_us"] for r in plain["moe_topk"]}
    for b in cal["moe_topk"]:
        assert b["est_source"] == "roofline+calibrated"
        if before[b["tokens"]] >= 0.01:  # rounded to 3 places
            assert b["est_us"] == pytest.approx(1000.0 * before[b["tokens"]], rel=0.05)
    assert {r["est_source"] for r in cal["rmsnorm"]} == {"roofline"}
    [fill] = cal["engine_fill"]
    kernels_ms = sum(r["est_total_ms"] or 0.0 for r in rows if r["kind"] != "engine")
    assert fill["kind"] == "engine" and fill["calls"] == trace.num_steps
    assert fill["est_total_ms"] == pytest.approx(0.5 * kernels_ms, rel=1e-3)
    assert rows[0]["op"] in ("engine_fill", "moe_topk")  # the slow ones now rank first


def test_a_speedup_buys_the_kernel_s_calibrated_share_of_the_step():
    p = _Timed(_cfg())
    trace, base = _replay(p)
    mix = _mix(trace, p, base)
    topk = ksp.Speedups.parse([{"op": "moe_topk", "speedup": 2.0}])

    def ratio(cal):
        t = sm.to_trace(mix, {"engine": p})
        t.calibration = cal
        return sm.reprice(mix, t, {"engine": p}, topk).pools["engine"]["ratio"]

    assert ratio(None) == pytest.approx(1.0, abs=1e-3)  # a roofline top-k is nothing
    calibrated = ratio(kc.Calibration.parse(CAL))
    assert calibrated < 0.9
    no_extra = ratio(kc.Calibration.parse({**CAL, "extras": []}))
    assert no_extra < calibrated  # the engine's fill doesn't get faster


def test_engine_kernels_become_tasks():
    p = _Timed(_cfg())
    trace, _ = _replay(p)
    trace.calibration = kc.Calibration.parse(CAL)
    doc = kt.build(trace.kernels(), p.cfg, timer=SimpleNamespace(bw=8000.0, peak=2500.0))
    fill = next(t for t in doc["tasks"] if t["op"] == "engine_fill")
    assert fill["share_pct"] > 0
