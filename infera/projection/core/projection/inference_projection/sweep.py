###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Project a whole configuration space in one process, and rank it.

A search like Hyperloom's spends its GPU time discovering that most candidate
deployments are bad, or do not fit in memory at all. Each of those answers costs
a model load and a benchmark -- minutes on a full node -- and the search only
needs the ranking, not the measurement.

This projects every point in a config space analytically so the search can spend
real GPU time on the finalists. One point costs ~29 ms against ~29 s to measure
it, and the fixed setup (~0.7 s, mostly the Origami import) is paid once for the
whole sweep rather than per config, so a subprocess-per-config caller gets a
further ~25x on top.

Feasibility is the part worth having even when the ranking is uncertain: a
config whose weights and KV cache do not fit is unrunnable for a reason that
needs no measurement to establish, and pruning those first is exact rather than
approximate.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence


@dataclass
class SweepPoint:
    """One projected configuration."""

    tp: int
    ep: int
    pp: int
    concurrency: int
    isl: int
    osl: int

    feasible: bool = True
    reason: str = ""

    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    decode_tps: float = 0.0
    decode_tps_per_gpu: float = 0.0
    memory_per_gpu_gb: float = 0.0

    def gpus(self) -> int:
        return max(1, self.tp * self.pp)


@dataclass
class SweepResult:
    """Everything a sweep produced, plus what it cost to produce."""

    points: list[SweepPoint] = field(default_factory=list)
    elapsed_s: float = 0.0
    n_projected: int = 0

    @property
    def feasible(self) -> list[SweepPoint]:
        return [p for p in self.points if p.feasible]

    def ranked(self, objective: str = "decode_tps_per_gpu",
               maximize: bool = True) -> list[SweepPoint]:
        """Feasible points ordered by ``objective``."""
        return sorted(self.feasible, key=lambda p: getattr(p, objective),
                      reverse=maximize)

    def shortlist(self, n: int = 5, objective: str = "decode_tps_per_gpu",
                  maximize: bool = True) -> list[SweepPoint]:
        return self.ranked(objective, maximize)[:n]


def _project_one(
    model: str, tp: int, ep: int, pp: int, conc: int, isl: int, osl: int,
    *, gpu_arch: str, hbm_gb: float, weight_dtype: str, kv_dtype: str,
    workload: str, attention_backend: str,
) -> tuple[Any, Any]:
    """Run one projection, returning ``(performance, memory)``."""
    os.environ["INFERASIM_MODEL"] = model
    from infera.projection.cli import build_parser
    from infera.projection.core.projection.inference_projection import (
        launch_projection_from_cli,
    )

    argv = [
        "inference",
        "--config", workload,
        "--inference-mode", "both", "--profiling-mode", "simulate",
        "--serving-model", "static",
        "--input-len", str(isl), "--output-len", str(osl),
        "--inference-batch-size", str(conc), "--max-concurrency", str(conc),
        "--weight-dtype", weight_dtype, "--kv-cache-dtype", kv_dtype,
        "--attention-backend", attention_backend,
        "--gpu-arch", gpu_arch, "--hbm-capacity-gb", str(hbm_gb),
        f"tensor_model_parallel_size={tp}",
        f"expert_model_parallel_size={ep}",
        f"pipeline_model_parallel_size={pp}",
    ]
    parsed, overrides = build_parser().parse_known_args(argv)
    # The projection prints a human-readable report; a sweep wants the numbers.
    with contextlib.redirect_stdout(io.StringIO()):
        res = launch_projection_from_cli(parsed, overrides)
    return res.get("performance"), res.get("memory")


def sweep(
    model: str,
    *,
    tp: Sequence[int] = (1, 2, 4, 8),
    ep: Sequence[int] = (1,),
    pp: Sequence[int] = (1,),
    concurrency: Sequence[int] = (1, 8, 32, 128, 256),
    isl: int = 1024,
    osl: int = 1024,
    gpu_arch: str = "mi355x",
    hbm_gb: float = 288.0,
    weight_dtype: str = "mxfp4",
    kv_dtype: str = "bf16",
    attention_backend: str = "aiter",
    workload: str | None = None,
    valid: Callable[[int, int, int], bool] | None = None,
    progress: bool = False,
) -> SweepResult:
    """Project every point of a config space.

    ``valid`` filters combinations before projecting them, which is where a
    caller expresses rules the cost model should not have to know -- expert
    parallelism not exceeding tensor parallelism, say, or a fixed GPU budget.
    Points that fail to project are kept and marked infeasible rather than
    dropped, so a search can tell "does not fit" from "was never tried".
    """
    if workload is None:
        # .../infera/projection/core/projection/inference_projection -> repo root
        workload = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "..", "..", "bench", "hyperloom_validation",
            "inferasim_workload.yaml",
        )
        workload = os.path.normpath(workload)

    combos: Iterable[tuple[int, int, int, int]] = itertools.product(
        tp, ep, pp, concurrency
    )
    out = SweepResult()
    t0 = time.perf_counter()
    for _tp, _ep, _pp, _conc in combos:
        if valid is not None and not valid(_tp, _ep, _pp):
            continue
        pt = SweepPoint(tp=_tp, ep=_ep, pp=_pp, concurrency=_conc, isl=isl, osl=osl)
        try:
            perf, mem = _project_one(
                model, _tp, _ep, _pp, _conc, isl, osl,
                gpu_arch=gpu_arch, hbm_gb=hbm_gb, weight_dtype=weight_dtype,
                kv_dtype=kv_dtype, workload=workload,
                attention_backend=attention_backend,
            )
            pt.ttft_ms = float(getattr(perf, "ttft_ms", 0.0) or 0.0)
            # Inter-token latency is what a serving SLO is written against.
            pt.tpot_ms = float(getattr(perf, "itl_ms", 0.0) or 0.0)
            pt.decode_tps = float(getattr(perf, "decode_throughput_tps", 0.0) or 0.0)
            # Prefer the engine's own per-GPU figure: it divides by the replica's
            # real GPU count, which is not always tp*pp.
            pt.decode_tps_per_gpu = float(
                getattr(perf, "decode_throughput_tps_per_gpu", 0.0) or 0.0
            ) or (pt.decode_tps / pt.gpus())
            if mem is not None:
                total = float(getattr(mem, "total_bytes", 0) or 0)
                pt.memory_per_gpu_gb = total / (1024.0 ** 3)
                cap = float(getattr(mem, "hbm_capacity_bytes", 0) or 0)
                if getattr(mem, "fits", True) is False:
                    pt.feasible = False
                    pt.reason = (
                        f"needs {pt.memory_per_gpu_gb:.0f} GB/GPU, device has "
                        f"{cap / (1024.0 ** 3):.0f} GB"
                    )
        except Exception as exc:  # noqa: BLE001 - a bad point must not stop a sweep
            pt.feasible = False
            pt.reason = f"{type(exc).__name__}: {exc}"
        out.points.append(pt)
        out.n_projected += 1
        if progress:
            print(f"  tp{_tp} ep{_ep} pp{_pp} c{_conc:<4d} "
                  f"{'ok ' if pt.feasible else 'X  '}"
                  f"tpot={pt.tpot_ms:7.2f} ms  tps/gpu={pt.decode_tps_per_gpu:8.1f}"
                  f"{'  ' + pt.reason if pt.reason else ''}")
    out.elapsed_s = time.perf_counter() - t0
    return out


def to_json(result: SweepResult) -> dict[str, Any]:
    """Serialise a sweep, keeping the cost of producing it alongside."""
    return {
        "n_projected": result.n_projected,
        "elapsed_s": result.elapsed_s,
        "per_config_ms": (result.elapsed_s / result.n_projected * 1e3)
        if result.n_projected else 0.0,
        "points": [asdict(p) for p in result.points],
    }
