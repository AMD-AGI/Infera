###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Validate an inference op model before it is registered.

An op model is accepted only when its step cost is physically possible and
agrees with the math it came from, in both phases. On the workload's own
shape and on scaled variants -- prefill at 2x batch and 2x prompt, decode at 2x
batch and 2x context -- it must:

* be structurally complete: name, valid slot, references, the gap's keys
  consumed or explicitly ignored;
* actually match the workload and run on a sane share of layers;
* return finite, positive time, FLOPs and bytes, with a breakdown that sums
  to the total;
* never beat the roofline ``max(FLOPs / peak, bytes / HBM BW)``;
* agree with :meth:`OpModel.reference_flops`, the closed form from the paper;
* not get cheaper with more tokens, more sequences or more context.

A step far above its roofline is only a warning: small decode steps are
latency-bound and legitimately sit there.

    python -m infera.projection.core.projection.op_models.validate \\
        --config <workload.yaml> --op-file <op.py> [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from .base import SLOTS, OpContext, OpCost, OpModel

ROOFLINE_SLACK = 0.97
ROOFLINE_WARN_RATIO = 50.0
REFERENCE_FLOPS_TOL = 0.25
BREAKDOWN_TOL = 0.02

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class ValidationReport:
    op_name: str
    checks: list[Check] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != FAIL for c in self.checks)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    def as_dict(self) -> dict[str, Any]:
        return {
            "op": self.op_name,
            "ok": self.ok,
            "checks": [c.__dict__ for c in self.checks],
            "cases": self.cases,
        }

    def feedback(self) -> str:
        lines = [f"Validation of op model '{self.op_name}': {'PASSED' if self.ok else 'FAILED'}"]
        lines += [
            f"- [{c.status.upper()}] {c.name}: {c.detail}" for c in self.checks if c.status != PASS
        ]
        return "\n".join(lines)

    def text(self) -> str:
        out = [f"Op model '{self.op_name}': {'OK' if self.ok else 'FAILED'}"]
        for c in self.checks:
            out.append(
                f"  {c.status.upper():4s} {c.name}" + (f" -- {c.detail}" if c.detail else "")
            )
        for case in self.cases:
            out.append(
                f"  case {case['label']:>15s}: {case['ms']:.4f} ms  (roofline {case['roofline_ms']:.4f} ms, "
                f"{case['flops']:.3e} FLOPs, {case['bytes']:.3e} B)"
            )
        return "\n".join(out)


@dataclass
class Workload:
    training_config: Any  # TrainingConfig view of the InferenceConfig
    gemm_backend: Any
    sdpa_backend: Any
    batch: int
    input_len: int
    output_len: int
    inference_config: Any = None


def load_workload(
    config_path: str,
    *,
    gpu_arch: str | None = None,
    batch: int | None = None,
    input_len: int | None = None,
    output_len: int | None = None,
    simulate_backends: bool = True,
) -> Workload:
    """InferenceConfig and simulation backends, as ``inferasim inference`` builds them."""
    from infera.projection.core.launcher.parser import load_config
    from infera.projection.core.projection.training_config import (
        convert_config_to_inference_config,
    )

    config, _ = load_config(SimpleNamespace(config=str(config_path)), [])
    overrides = {
        k: v
        for k, v in (
            ("batch_size", batch),
            ("input_seq_len", input_len),
            ("output_seq_len", output_len),
        )
        if v is not None
    }
    ic = convert_config_to_inference_config(config, inference_overrides=overrides)
    req = ic.request_config
    b = int(req.batch_size or 1)
    lin = int(req.input_seq_len or 1024)
    view = ic.as_training_config(batch_size=b, seq_len=lin)
    gemm = sdpa = None
    if simulate_backends:
        from infera.projection.core.projection.simulation_backends.factory import (
            get_gemm_simulation_backend,
            get_sdpa_simulation_backend,
        )

        try:
            gemm = get_gemm_simulation_backend(
                backend_name=None, gpu_arch=gpu_arch, gpu_clock_mhz=None
            )
            sdpa = get_sdpa_simulation_backend(gpu_arch=gpu_arch, gpu_clock_mhz=None)
        except Exception as exc:
            print(
                f"[InferaSim:OpValidate] simulation backends unavailable ({exc}); using roofline primitives"
            )
    return Workload(view, gemm, sdpa, b, lin, int(req.output_seq_len or 1), ic)


def load_op_class(path: str) -> type[OpModel]:
    from .registry import load_op_model_file, registered_op_models

    registered_op_models()
    found = load_op_model_file(path)
    if not found:
        raise ValueError(
            f"{path} registers no OpModel (decorate the class with @register_op_model)"
        )
    if len(found) > 1:
        raise ValueError(f"{path} registers {len(found)} op models; put one op per file")
    return found[0]


def _finite_pos(x: Any) -> bool:
    try:
        return math.isfinite(float(x)) and float(x) > 0.0
    except (TypeError, ValueError):
        return False


def _cases(wl: Workload) -> list[tuple[str, str, int, int, int]]:
    """``(label, phase, batch, q_len, kv_len)``."""
    b, s, ctx = wl.batch, wl.input_len, wl.input_len + wl.output_len // 2
    return [
        ("prefill", "prefill", b, s, s),
        ("prefill_2x_batch", "prefill", 2 * b, s, s),
        ("prefill_2x_len", "prefill", b, 2 * s, 2 * s),
        ("decode", "decode", b, 1, ctx),
        ("decode_2x_batch", "decode", 2 * b, 1, ctx),
        ("decode_2x_ctx", "decode", b, 1, 2 * ctx),
    ]


def validate_op_model(
    op: OpModel | type[OpModel], wl: Workload, *, gap_keys: Sequence[str] = ()
) -> ValidationReport:
    inst = op() if isinstance(op, type) else op
    cls = type(inst)
    rep = ValidationReport(getattr(cls, "name", "") or cls.__name__)
    mc = wl.training_config.model_config

    problems = []
    if not _NAME_RE.match(cls.name or ""):
        problems.append(f"name {cls.name!r} is not snake_case")
    if cls.slot not in SLOTS:
        problems.append(f"slot {cls.slot!r} not in {SLOTS}")
    if not cls.claims:
        problems.append("claims is empty: the coverage check cannot tell this op is modeled")
    if not cls.references:
        problems.append(
            "references is empty: cite the paper / model card / kernel the equations come from"
        )
    rep.add("structure", FAIL if problems else PASS, "; ".join(problems))

    covered = set(cls.claims) | set(cls.consumes) | set(getattr(cls, "ignores", {}) or {})
    missing = sorted(k for k in gap_keys if k not in covered)
    rep.add(
        "gap_keys_covered",
        FAIL if missing else PASS,
        f"keys {missing} are neither consumed nor listed in `ignores` with a reason"
        if missing
        else "",
    )

    try:
        matched = bool(cls.matches(mc))
        rep.add(
            "matches",
            PASS if matched else FAIL,
            "" if matched else "matches() is False for this workload",
        )
    except Exception as exc:
        rep.add("matches", FAIL, f"matches() raised {exc!r}")
    try:
        frac = float(inst.layer_fraction(mc))
        ok = 0.0 < frac <= 1.0
        rep.add(
            "layer_fraction",
            PASS if ok else FAIL,
            f"{frac:.4f}" + ("" if ok else " (must be in (0, 1])"),
        )
    except Exception as exc:
        rep.add("layer_fraction", FAIL, f"layer_fraction() raised {exc!r}")

    by_label: dict[str, OpCost] = {}
    for label, phase, b, q, kv in _cases(wl):
        ctx = OpContext(wl.training_config, b, q, kv, phase, wl.gemm_backend, wl.sdpa_backend)
        try:
            c = inst.cost(ctx)
        except Exception as exc:
            rep.add(f"cost[{label}]", FAIL, f"cost() raised {type(exc).__name__}: {exc}")
            continue
        if not isinstance(c, OpCost):
            rep.add(f"cost[{label}]", FAIL, f"cost() returned {type(c).__name__}, expected OpCost")
            continue
        bad = [n for n in ("ms", "flops", "bytes") if not _finite_pos(getattr(c, n))]
        rep.add(
            f"finite_positive[{label}]",
            FAIL if bad else PASS,
            f"{bad} must be finite and > 0" if bad else "",
        )
        if bad:
            continue
        by_label[label] = c
        roof = ctx.roofline_ms(c.flops, c.bytes)
        rep.cases.append(
            {
                "label": label,
                "phase": phase,
                "batch": b,
                "q_len": q,
                "kv_len": kv,
                "ms": float(c.ms),
                "flops": float(c.flops),
                "bytes": float(c.bytes),
                "roofline_ms": roof,
                "breakdown": dict(c.breakdown),
            }
        )
        if c.ms < ROOFLINE_SLACK * roof:
            rep.add(
                f"roofline[{label}]",
                FAIL,
                f"{c.ms:.4f} ms beats the roofline {roof:.4f} ms (FLOPs or bytes overstated, or time too low)",
            )
        elif roof > 0 and c.ms > ROOFLINE_WARN_RATIO * roof:
            rep.add(
                f"roofline[{label}]",
                WARN,
                f"{c.ms:.4f} ms is {c.ms / roof:.0f}x the roofline {roof:.4f} ms",
            )
        else:
            rep.add(f"roofline[{label}]", PASS)
        if c.breakdown:
            s = sum(c.breakdown.values())
            off = abs(s - c.ms) / c.ms
            rep.add(
                f"breakdown_sums[{label}]",
                PASS if off <= BREAKDOWN_TOL else FAIL,
                f"breakdown sums to {s:.4f} ms vs total {c.ms:.4f} ms"
                if off > BREAKDOWN_TOL
                else "",
            )
        try:
            ref = inst.reference_flops(ctx)
        except Exception as exc:
            rep.add(f"reference_flops[{label}]", FAIL, f"reference_flops() raised {exc!r}")
            continue
        if ref is None:
            rep.add(
                f"reference_flops[{label}]", WARN, "no closed form given; FLOPs not cross-checked"
            )
        else:
            err = (c.flops - float(ref)) / float(ref) if ref else 0.0
            rep.add(
                f"reference_flops[{label}]",
                PASS if abs(err) <= REFERENCE_FLOPS_TOL else FAIL,
                f"{c.flops:.3e} vs closed form {float(ref):.3e} ({err:+.0%})",
            )

    for base, more in (
        ("prefill", "prefill_2x_batch"),
        ("prefill", "prefill_2x_len"),
        ("decode", "decode_2x_batch"),
        ("decode", "decode_2x_ctx"),
    ):
        if base in by_label and more in by_label:
            a, b_ = by_label[base].ms, by_label[more].ms
            ok = b_ >= a * 0.999
            rep.add(
                f"monotonic[{more}]",
                PASS if ok else FAIL,
                ""
                if ok
                else f"{b_:.4f} ms < {a:.4f} ms at {base}: cost must not fall with more work",
            )
    return rep


def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--op-file", required=True)
    p.add_argument("--gpu-arch", default=os.getenv("INFERASIM_GPU_ARCH", "mi355x"))
    p.add_argument("--gap-keys", default="")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--input-len", type=int, default=None)
    p.add_argument("--output-len", type=int, default=None)
    p.add_argument("--no-backends", action="store_true")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    os.environ.setdefault("INFERASIM_OP_COVERAGE", "0")
    try:
        op_cls = load_op_class(args.op_file)
        wl = load_workload(
            args.config,
            gpu_arch=args.gpu_arch,
            batch=args.batch,
            input_len=args.input_len,
            output_len=args.output_len,
            simulate_backends=not args.no_backends,
        )
        rep = validate_op_model(op_cls, wl, gap_keys=[k for k in args.gap_keys.split(",") if k])
    except Exception as exc:
        rep = ValidationReport(os.path.basename(args.op_file))
        rep.add("load", FAIL, f"{type(exc).__name__}: {exc}")
    print(json.dumps(rep.as_dict(), indent=2, default=str) if args.json else rep.text())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
