#!/usr/bin/env python3
"""Op-agnostic optimize-loop scaffold (issue #40).

The point of this directory is the *scaffold*, not any one kernel: a uniform
**measure → profile → tune → inject** loop that works for ANY op behind the
Infera vLLM op-injection plugin. Adding an op is writing one :class:`OpSpec` and
registering it (see ``ops/``); `loop.py` then drives measure/profile/tune for it
by name. The kernels (e.g. ``infera_decode``) are just candidates plugged in.

An OpSpec tells the loop how to exercise one op:
  * ``make_inputs(dims, device)``     → the tensors the op takes,
  * ``baseline(*inputs)``             → the engine's built-in op (the A in A/B),
  * ``candidate(*inputs)``            → the plugin's op (selected variant; the B),
  * ``reference(*inputs)`` (optional) → a correctness oracle,
  * ``traffic_bytes(dims)`` (optional)→ bytes moved, for the roofline profile,
  * ``tune_env`` / ``tune_grid`` / ``inject`` (optional) → the tune ring.
Anything an op doesn't provide, the loop simply skips for that op.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch

# --- registry --------------------------------------------------------------

_REGISTRY: dict[str, OpSpec] = {}


def register_op(spec: OpSpec) -> OpSpec:
    _REGISTRY[spec.name] = spec
    return spec


def get_op(name: str) -> OpSpec:
    if name not in _REGISTRY:
        # ops self-register on import; import the module named after the op.
        importlib.import_module(f"ops.{name}")
    return _REGISTRY[name]


def list_ops() -> list[str]:
    return sorted(_REGISTRY)


@dataclass
class OpSpec:
    name: str
    default_dims: dict
    make_inputs: Callable  # (dims, device) -> tuple of tensors
    baseline: Callable  # (*inputs) -> Tensor
    candidate: Callable  # (*inputs) -> Tensor
    reference: Callable | None = None  # (*inputs) -> Tensor
    traffic_bytes: Callable | None = None  # (dims) -> int
    tune_env: tuple[str, ...] = ()  # env keys the tuner sweeps
    tune_grid: Callable | None = None  # (dims) -> list[tuple] of full configs
    inject: Callable | None = None  # (config) -> None  (bake into the plugin)
    peak_gbps: float = 8000.0  # HBM peak for the roofline (MI355X ~8 TB/s)


# --- shared helpers --------------------------------------------------------


def timed(fn, iters=30, warmup=8) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def rel_err(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def _cast(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return v


def _dims(spec: OpSpec, overrides: dict) -> dict:
    d = dict(spec.default_dims)
    d.update({k: _cast(v) for k, v in overrides.items()})
    return d


# --- the three rings, op-agnostic -----------------------------------------


def measure(spec: OpSpec, overrides: dict, dev="cuda"):
    dims = _dims(spec, overrides)
    inp = spec.make_inputs(dims, dev)
    base = spec.baseline(*inp)
    cand = spec.candidate(*inp)
    print(f"op={spec.name}  dims={dims}")
    rows = [("baseline (built-in)", timed(lambda: spec.baseline(*inp)), 0.0)]
    if spec.reference is not None:
        ref = spec.reference(*inp)
        rows.insert(
            0, ("reference (oracle)", timed(lambda: spec.reference(*inp), 5, 1), rel_err(ref, base))
        )
    rows.append(("candidate (plugin op)", timed(lambda: spec.candidate(*inp)), rel_err(cand, base)))
    base_t = next(t for n, t, _ in rows if n.startswith("baseline"))
    print(f"  {'impl':26s} {'median ms':>10s} {'rel vs base':>12s} {'speedup':>9s}")
    for n, t, r in rows:
        print(f"  {n:26s} {t:10.4f} {r:12.2e} {base_t / t:8.2f}x")


def profile(spec: OpSpec, overrides: dict, kernels=False, dev="cuda"):
    dims = _dims(spec, overrides)
    inp = spec.make_inputs(dims, dev)
    spec.candidate(*inp)  # warm
    ms = timed(lambda: spec.candidate(*inp))
    print(f"op={spec.name}  dims={dims}\n  latency        : {ms:.4f} ms")
    if spec.traffic_bytes is not None:
        wb = spec.traffic_bytes(dims)
        bw = (wb / 1e9) / (ms / 1e3)  # GB/s
        pct = 100 * bw / spec.peak_gbps
        verdict = (
            "bandwidth-bound — near peak; win by moving less traffic"
            if pct >= 55
            else "launch/occupancy-bound — headroom; tune tiling/warps"
        )
        print(f"  traffic        : {wb / 1e6:.1f} MB")
        print(
            f"  achieved BW    : {bw / 1e3:.2f} TB/s  ({pct:.0f}% of {spec.peak_gbps / 1e3:.1f} TB/s peak)"
        )
        print(f"  bottleneck     : {verdict}")
    if kernels:
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as tprofile

        for _ in range(5):
            spec.candidate(*inp)
        torch.cuda.synchronize()
        with tprofile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                spec.candidate(*inp)
            torch.cuda.synchronize()
        print("\n  per-kernel device time (top 6):")
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=6))


def tune(spec: OpSpec, overrides: dict, inject=False, dev="cuda"):
    if spec.tune_grid is None or not spec.tune_env:
        print(f"op={spec.name}: not tunable (no tune_grid/tune_env)")
        return
    dims = _dims(spec, overrides)
    inp = spec.make_inputs(dims, dev)
    ref = spec.baseline(*inp).float()
    base_t = timed(lambda: spec.baseline(*inp))
    best, best_t = None, float("inf")
    for cfg in spec.tune_grid(dims):
        for k, v in zip(spec.tune_env, cfg):
            os.environ[k] = str(v)
        try:
            out = spec.candidate(*inp)
            if rel_err(out, ref) > 1e-2:
                continue
            t = timed(lambda: spec.candidate(*inp))
        except Exception:  # noqa: BLE001
            continue
        if t < best_t:
            best, best_t = cfg, t
    print(f"op={spec.name}  dims={dims}  baseline {base_t:.4f} ms")
    if best is None:
        print("  no correct config found")
        return
    print(f"  best {best} -> {best_t:.4f} ms ({base_t / best_t:.2f}x vs baseline)")
    if inject and spec.inject is not None:
        spec.inject(best)
        print(f"  injected {best} into the plugin")
