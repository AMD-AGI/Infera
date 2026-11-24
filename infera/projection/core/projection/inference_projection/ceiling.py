###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The ceiling bound, and the headroom between it and the expected projection.

An expected projection prices the stack as it runs today, which is what a
calibration fits. A better kernel, a fused one or proper overlap then beats it,
and a projection that a measurement beats cannot say what is left to win. The
ceiling prices the same deployment with each lever at its physical best:

  * ``overlap`` -- communication hides behind the layer's compute, up to that
    compute; the expert A2A and all-reduce share it.
  * ``fusion``  -- norms, residual adds, casts, the expert permute and the
    activation fold into the GEMM or attention kernel beside them, so their
    HBM round trips go.
  * ``launch``  -- no per-step host overhead, per-kernel occupancy, launch
    floor or mixed-step penalty: graph-captured, persistent-kernel behaviour.
  * ``kernel``  -- a measured prefill rate slower than the simulator does not
    slow the ceiling down.

GEMM and attention kernels stay Origami's and the SDPA simulator's in both
bounds. A measurement above the ceiling is a modelling error to fix, not a
better kernel. The headroom, split by lever, is what a tuning loop has left to
chase, and where.
"""

from __future__ import annotations

LEVERS = ("overlap", "fusion", "launch", "kernel")
BOUNDS = ("expected", "ceiling")


def parse_levers(text) -> tuple:
    names = [t.strip() for t in str(text or "").split(",") if t.strip()]
    bad = [n for n in names if n not in LEVERS]
    if bad or not names:
        raise ValueError(f"ceiling levers {bad or names!r}: choose from {', '.join(LEVERS)}")
    return tuple(dict.fromkeys(names))


def check_bound(req, *, benchmark: bool) -> None:
    bound = getattr(req, "projection_bound", "expected")
    if bound not in BOUNDS:
        raise ValueError(f"projection bound {bound!r}: choose from {', '.join(BOUNDS)}")
    bad = [n for n in getattr(req, "ceiling_levers", ()) if n not in LEVERS]
    if bad:
        raise ValueError(f"ceiling levers {bad!r}: choose from {', '.join(LEVERS)}")
    if bound == "ceiling" and benchmark:
        raise ValueError(
            "the ceiling bound is priced from the simulator; a measured anchor "
            "already carries today's overheads, so project it in simulate mode"
        )


def _metrics(perf) -> dict:
    return {
        "output_tput_per_gpu": round(float(perf.decode_throughput_tps_per_gpu), 2),
        "tpot_ms": round(float(perf.itl_ms), 4),
        "ttft_ms": round(float(perf.ttft_ms), 3),
        "decode_step_ms": round(float(perf.decode_step_latency_ms), 4),
    }


def _gain_pct(new: float, base: float) -> float:
    return round(100.0 * (new / base - 1.0), 2) if base > 0 else 0.0


def headroom(projector, measured_tput_per_gpu: float | None = None) -> dict:
    """Expected and ceiling output throughput per GPU, what each lever buys on
    its own, and a waterfall adding them biggest first. Levers interact (less
    compute hides less communication), so the waterfall's steps, not the
    standalone gains, sum to the total."""
    if projector.is_benchmark_calibrated:
        raise ValueError("headroom is priced from the simulator; project in simulate mode")
    with projector.bound("expected"):
        exp = _metrics(projector.project())
    alone = {}
    for lever in LEVERS:
        with projector.bound("ceiling", (lever,)):
            alone[lever] = _metrics(projector.project())
    with projector.bound("ceiling", LEVERS):
        ceil = _metrics(projector.project())

    base = exp["output_tput_per_gpu"]
    order = sorted(LEVERS, key=lambda n: -alone[n]["output_tput_per_gpu"])
    waterfall, applied, prev = [], [], base
    for lever in order:
        applied.append(lever)
        with projector.bound("ceiling", applied):
            tput = float(projector.project().decode_throughput_tps_per_gpu)
        waterfall.append(
            {
                "lever": lever,
                "output_tput_per_gpu": round(tput, 2),
                "step_pct": _gain_pct(tput, prev),
            }
        )
        prev = tput

    rep = {
        "expected": exp,
        "ceiling": ceil,
        "headroom_pct": _gain_pct(ceil["output_tput_per_gpu"], base),
        "levers": [
            {"lever": n, **alone[n], "gain_pct": _gain_pct(alone[n]["output_tput_per_gpu"], base)}
            for n in order
        ],
        "waterfall": waterfall,
    }
    if measured_tput_per_gpu:
        m = float(measured_tput_per_gpu)
        top = ceil["output_tput_per_gpu"]
        rep["measured"] = {
            "output_tput_per_gpu": m,
            "of_expected": round(m / base, 4) if base > 0 else None,
            "of_ceiling": round(m / top, 4) if top > 0 else None,
            "above_ceiling": bool(m > top),
        }
    return rep
