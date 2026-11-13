###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Anchored projection: measured refinement steps, transported analytically.

The autoregressive path's anchors (``manual/simulation/anchors.md``) carry a
measurement to a configuration nobody measured. This is the same idea for a
refinement step. An anchor is a handful of measured steps -- a full forward
over ``q`` positions and the confidence sampler that ranks them -- on one
model, precision and layout. A projected step is the nearest measured point,
scaled by what the simulator says changes between the two::

    step(target) = measured(point) x sim(target) / sim(point on the anchor's layout)

Both simulations price compute and collectives the same way, so what carries
over from the measurement is everything the simulator does not model -- the
framework, the sampler's real cost, the kernels a model's implementation
actually picks -- and what the ratio supplies is how the step scales with
batch, sequence, cache mode and parallel layout.

An anchor only transports within its regime: the model, its weight,
activation and KV-cache precision, the attention backend and the graph mode.
A projection asked to cross one is refused with what differed, never silently
rescaled.

The file is engine neutral JSON::

    {"kind": "dllm", "backend": "hf", "gpu_arch": "mi355x",
     "regime": {"model": "llada_8B", "weight_dtype": "bf16", "act_dtype": "bf16",
                "kv_cache_dtype": "bf16", "attention_backend": "sdpa",
                "graph_mode": "eager"},
     "parallel": {"tp": 1, "pp": 1, "ep": 1},
     "points": [{"batch": 1, "q": 1024, "kv": 1024, "logits": 1024, "step_ms": 41.2}]}
"""

from __future__ import annotations

import glob
import json
import math
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any

from .comm import CommConfig
from .contention import ContentionConfig
from .hardware import DLLMHardware
from .model import DLLMModelSpec, Parallel, SeqShape
from .runtime import RuntimeConfig, get_runtime
from .step_model import DLLMStepModel, StepBreakdown

KIND = "dllm"
STORE_ENV = "INFERASIM_ANCHOR_STORE"
# The regime keys every anchor carries; a target that states one must match it.
REGIME_KEYS = (
    "model",
    "weight_dtype",
    "act_dtype",
    "kv_cache_dtype",
    "attention_backend",
    "graph_mode",
    "runtime",
)


class AnchorMismatch(ValueError):
    """An anchor was asked to transport outside its regime."""


@dataclass
class AnchorPoint:
    batch: int
    q: int
    kv: int
    logits: int
    step_ms: float

    def seqs(self) -> list[SeqShape]:
        return [SeqShape(self.q, self.kv, self.logits)] * self.batch


@dataclass
class DLLMAnchor:
    regime: dict[str, str]
    points: list[AnchorPoint]
    parallel: Parallel = field(default_factory=Parallel)
    gpu_arch: str = "mi355x"
    backend: str = "hf"
    source: str = ""

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("[dllm-sim] an anchor needs at least one measured point")
        missing = [k for k in ("model", "weight_dtype", "act_dtype") if not self.regime.get(k)]
        if missing:
            raise ValueError(f"[dllm-sim] anchor regime is missing {', '.join(missing)}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": KIND,
            "backend": self.backend,
            "gpu_arch": self.gpu_arch,
            "regime": dict(self.regime),
            "parallel": {"tp": self.parallel.tp, "pp": self.parallel.pp, "ep": self.parallel.ep},
            "points": [p.__dict__ for p in self.points],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], source: str = "") -> DLLMAnchor:
        if d.get("kind") != KIND:
            raise ValueError(f"[dllm-sim] {source or 'anchor'} is not a dLLM anchor")
        return cls(
            regime={k: str(v) for k, v in (d.get("regime") or {}).items()},
            points=[AnchorPoint(**p) for p in d.get("points") or []],
            parallel=Parallel(**(d.get("parallel") or {})),
            gpu_arch=str(d.get("gpu_arch", "mi355x")),
            backend=str(d.get("backend", "hf")),
            source=source,
        )

    def mismatches(self, target: dict[str, str]) -> list[str]:
        """Regime keys both sides state and disagree on."""
        out = []
        for k in REGIME_KEYS:
            a, t = self.regime.get(k), target.get(k)
            if a and t and str(a).lower() != str(t).lower():
                out.append(f"{k}: anchor {a} != target {t}")
        return out

    def nearest(self, seqs: list[SeqShape]) -> AnchorPoint:
        """The measured point closest in log tokens, context and batch."""
        tokens = sum(s.q for s in seqs)
        kv = sum(s.kv for s in seqs) / len(seqs)
        n = len(seqs)

        def dist(p: AnchorPoint) -> float:
            return (
                abs(math.log(tokens / (p.batch * p.q)))
                + abs(math.log(kv / p.kv))
                + 0.5 * abs(math.log(n / p.batch))
            )

        return min(self.points, key=dist)


def save_anchor(anchor: DLLMAnchor, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(anchor.as_dict(), f, indent=2)


def load_anchor(path: str) -> DLLMAnchor:
    with open(path, encoding="utf-8") as f:
        return DLLMAnchor.from_dict(json.load(f), source=path)


def target_regime(spec: DLLMModelSpec, **extra: str | None) -> dict[str, str]:
    reg = {
        "model": spec.name,
        "weight_dtype": spec.weight_dtype,
        "act_dtype": spec.act_dtype,
        "kv_cache_dtype": spec.kv_cache_dtype,
    }
    reg.update({k: str(v) for k, v in extra.items() if v})
    return reg


def find_anchor(store: str, regime: dict[str, str], gpu_arch: str | None = None) -> DLLMAnchor:
    """The store's anchor for ``regime``, preferring ``gpu_arch``.

    Raises ``AnchorMismatch`` listing every candidate and why it was refused.
    """
    rejected: list[str] = []
    ok: list[DLLMAnchor] = []
    for path in sorted(glob.glob(os.path.join(store, "**", "*.json"), recursive=True)):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or d.get("kind") != KIND:
            continue
        a = DLLMAnchor.from_dict(d, source=path)
        bad = a.mismatches(regime)
        if bad:
            rejected.append(f"{path}: {'; '.join(bad)}")
        else:
            ok.append(a)
    if not ok:
        why = "\n  ".join(rejected) or "no dLLM anchors in the store"
        raise AnchorMismatch(f"[dllm-sim] no anchor in {store} matches {regime}:\n  {why}")
    same = [a for a in ok if gpu_arch and a.gpu_arch.lower() == gpu_arch.lower()]
    return (same or ok)[0]


class AnchoredStepModel:
    """A ``DLLMStepModel`` whose steps are measured, then transported.

    Exposes ``spec``, ``par``, ``hw``, ``comm`` and ``step`` like the step
    model it wraps, so both simulators run on it unchanged.
    """

    def __init__(
        self,
        anchor: DLLMAnchor,
        spec: DLLMModelSpec,
        par: Parallel,
        hw: DLLMHardware,
        comm: CommConfig | None = None,
        contention: ContentionConfig | None = None,
        *,
        regime: dict[str, str] | None = None,
        runtime: RuntimeConfig | str | None = None,
    ) -> None:
        rt = get_runtime(runtime)
        bad = anchor.mismatches(regime or target_regime(spec, runtime=rt.name))
        if bad:
            raise AnchorMismatch(
                f"[dllm-sim] anchor {anchor.source or ''} cannot transport here: " + "; ".join(bad)
            )
        self.anchor = anchor
        self.target = DLLMStepModel(spec, par, hw, comm, contention, runtime=rt)
        same_layout = (anchor.parallel.tp, anchor.parallel.pp, anchor.parallel.ep) == (
            par.tp,
            par.pp,
            par.ep,
        )
        if anchor.gpu_arch.lower() == str(hw.name).lower() and same_layout:
            self.reference = self.target
        else:
            same_part = anchor.gpu_arch.lower() == str(hw.name).lower()
            a_hw = DLLMHardware.from_arch(
                anchor.gpu_arch,
                backend=hw.backend,
                gpu_clock_mhz=hw.gpu_clock_mhz if same_part else None,
            )
            self.reference = DLLMStepModel(
                spec, anchor.parallel, a_hw, comm, contention, runtime=rt
            )
        self.spec, self.par, self.hw = spec, par, hw
        self.comm = self.target.comm
        self._cache: dict[tuple, StepBreakdown] = {}

    def scale(self, seqs: list[SeqShape]) -> tuple[float, AnchorPoint]:
        p = self.anchor.nearest(seqs)
        ref = self.reference.step(p.seqs()).total_ms
        return (p.step_ms / ref if ref > 0 else 1.0), p

    def step(self, seqs: list[SeqShape]) -> StepBreakdown:
        key = tuple(sorted((s.q, s.kv, s.logits) for s in seqs))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        sim = self.target.step(seqs)
        k, _ = self.scale(seqs)
        out = replace(sim, total_ms=sim.total_ms * k, anchor_scale=k)
        self._cache[key] = out
        return out


# -- harvesting ---------------------------------------------------------------


def _sync(device: Any) -> None:
    import torch

    if getattr(device, "type", str(device)).startswith("cuda"):
        torch.cuda.synchronize(device)


def _step_once(model: Any, ids: Any) -> Any:
    """One refinement step: a full forward and the confidence sampler."""
    import torch

    out = model(ids)
    logits = out.logits if hasattr(out, "logits") else out[0] if isinstance(out, tuple) else out
    conf, pred = torch.softmax(logits.float(), dim=-1).max(dim=-1)
    k = max(1, conf.shape[-1] // 8)
    return conf.topk(k, dim=-1).indices, pred


def harvest(
    model: Any,
    shapes: list[tuple[int, int]],
    *,
    regime: dict[str, str],
    vocab_size: int,
    gpu_arch: str = "mi355x",
    parallel: Parallel | None = None,
    backend: str = "hf",
    warmup: int = 2,
    iters: int = 5,
    device: Any = None,
) -> DLLMAnchor:
    """Time ``model`` at each ``(batch, seq)`` and return the anchor.

    Each point is a step without a cache: ``seq`` positions recomputed and
    predicted, attending to each other. GPU steps are timed by CUDA events,
    CPU steps by the wall clock.
    """
    import torch

    dev = torch.device(device) if device is not None else next(model.parameters()).device
    gpu = dev.type == "cuda"
    points = []
    with torch.inference_mode():
        for batch, seq in shapes:
            ids = torch.randint(0, vocab_size, (batch, seq), device=dev)
            for _ in range(warmup):
                _step_once(model, ids)
            _sync(dev)
            if gpu:
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(iters):
                    _step_once(model, ids)
                end.record()
                torch.cuda.synchronize(dev)
                ms = start.elapsed_time(end) / iters
            else:
                t0 = time.perf_counter()
                for _ in range(iters):
                    _step_once(model, ids)
                ms = (time.perf_counter() - t0) * 1e3 / iters
            points.append(AnchorPoint(int(batch), int(seq), int(seq), int(seq), float(ms)))
    return DLLMAnchor(
        regime=dict(regime),
        points=points,
        parallel=parallel or Parallel(),
        gpu_arch=gpu_arch,
        backend=backend,
    )
