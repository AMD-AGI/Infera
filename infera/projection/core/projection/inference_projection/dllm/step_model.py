###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The time of one refinement step on one replica.

A stage's step is its compute, its collectives at the precision they are sent
at, the pack kernels that precision costs, and whatever part of the
collectives the engine overlaps with compute, less the contention that overlap
pays. A pipelined step cuts the batch into micro-batches: the first one crosses
every stage, each later one adds the slowest stage, and the unmasked tokens
return to the first stage before the next step can start.

Compute comes from a backend. The default prices the op graph on the roofline,
which needs nothing but this package. ``InferaSimBackend`` asks InferaSim's
autoregressive projector for a ``refine``-phase forward instead -- the same
measured GEMM and attention models the serving simulator trusts -- on a ROCm
host where Origami is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .comm import CommConfig, DLLMCommModel
from .contention import ContentionConfig, overlap
from .hardware import DLLMHardware
from .model import DLLMModelSpec, Parallel, SeqShape, StageGraph, build_stage_graph
from .ops import memory_bound_fraction, summarize, total_ms


class ComputeBackend(Protocol):
    def stage_compute_ms(
        self, graph: StageGraph, seqs: list[SeqShape], *, first: bool, last: bool
    ) -> float: ...


class RooflineBackend:
    def __init__(self, hw: DLLMHardware) -> None:
        self.hw = hw

    def stage_compute_ms(
        self, graph: StageGraph, seqs: list[SeqShape], *, first: bool, last: bool
    ) -> float:
        return total_ms(graph.ops, self.hw)


class InferaSimBackend:
    """Compute from ``InferencePerformanceProjector._forward_times``.

    The projector prices a uniform batch, so a ragged step is priced at its
    mean shape; the projector already prices its own rank's share of the
    pipeline, embedding and head included, so every stage gets the same
    time. Its communication is dropped -- the dLLM comm model prices that, with
    the precision and overlap this package adds.
    """

    def __init__(self, projector: Any) -> None:
        self.projector = projector

    def stage_compute_ms(
        self, graph: StageGraph, seqs: list[SeqShape], *, first: bool, last: bool
    ) -> float:
        n = len(seqs)
        q = max(1, round(sum(s.q for s in seqs) / n))
        kv = max(q, round(sum(s.kv for s in seqs) / n))
        ft = self.projector._forward_times(n, q, "refine", kv)
        comm = getattr(getattr(ft, "comm", None), "total_ms", 0.0) or 0.0
        return max(0.0, float(ft.total_ms) - float(comm))


@dataclass
class StageTime:
    total_ms: float = 0.0
    compute_ms: float = 0.0
    comm_wire_ms: float = 0.0
    quant_ms: float = 0.0
    exposed_comm_ms: float = 0.0
    contention_ms: float = 0.0
    hop_ms: float = 0.0
    by_kind: dict[str, float] = field(default_factory=dict)
    mem_bound_fraction: float = 0.0


@dataclass
class StepBreakdown:
    total_ms: float
    microbatches: int
    stages: list[StageTime]
    token_return_ms: float = 0.0

    def _sum(self, attr: str) -> float:
        return sum(getattr(s, attr) for s in self.stages)

    def as_dict(self) -> dict[str, Any]:
        """Totals over the first micro-batch's pass through every stage."""
        kinds: dict[str, float] = {}
        for s in self.stages:
            for k, v in s.by_kind.items():
                kinds[k] = kinds.get(k, 0.0) + v
        return {
            "step_ms": self.total_ms,
            "microbatches": self.microbatches,
            "compute_ms": self._sum("compute_ms"),
            "comm_wire_ms": self._sum("comm_wire_ms"),
            "quant_ms": self._sum("quant_ms"),
            "exposed_comm_ms": self._sum("exposed_comm_ms"),
            "contention_ms": self._sum("contention_ms"),
            "pp_hop_ms": self._sum("hop_ms") + self.token_return_ms,
            "by_kind_ms": kinds,
        }


class DLLMStepModel:
    def __init__(
        self,
        spec: DLLMModelSpec,
        par: Parallel,
        hw: DLLMHardware,
        comm: CommConfig | None = None,
        contention: ContentionConfig | None = None,
        backend: ComputeBackend | None = None,
    ) -> None:
        self.spec, self.par, self.hw = spec, par, hw
        self.comm_cfg = comm or CommConfig()
        self.contention = contention or ContentionConfig()
        self.comm = DLLMCommModel(spec, par, hw, self.comm_cfg)
        self.backend = backend or RooflineBackend(hw)
        self._boundaries = self.comm.pp_boundaries()
        self._cache: dict[tuple, StepBreakdown] = {}

    def _stage(self, seqs: list[SeqShape], s: int) -> StageTime:
        pp = self.par.pp
        first, last = s == 0, s == pp - 1
        g = build_stage_graph(self.spec, seqs, self.par, first=first, last=last)
        compute = self.backend.stage_compute_ms(g, seqs, first=first, last=last)
        wire = quant = wire_bytes = 0.0
        for need in g.comm:
            c = self.comm.collective(need.kind, need.tokens)
            wire += c.wire_ms * need.count
            quant += c.quant_ms * need.count
            wire_bytes += c.wire_bytes * need.count
        mem_frac = memory_bound_fraction(g.ops, self.hw)
        ov = overlap(
            compute,
            wire,
            overlap_fraction=self.comm_cfg.overlap,
            wire_bytes=wire_bytes,
            mem_fraction=mem_frac,
            hw=self.hw,
            cfg=self.contention,
        )
        hop = 0.0
        if not last:
            h = self.comm.pp_hop(sum(x.q for x in seqs), cross_node=self._boundaries[s])
            hop = h.total_ms
            quant += h.quant_ms
        return StageTime(
            total_ms=ov.total_ms + quant + hop,
            compute_ms=compute,
            comm_wire_ms=wire,
            quant_ms=quant,
            exposed_comm_ms=ov.exposed_comm_ms,
            contention_ms=ov.contention_ms,
            hop_ms=hop,
            by_kind=summarize(g.ops, self.hw),
            mem_bound_fraction=mem_frac,
        )

    def step(self, seqs: list[SeqShape]) -> StepBreakdown:
        if not seqs:
            raise ValueError("[dllm-sim] a step needs at least one sequence")
        key = tuple(sorted((s.q, s.kv, s.logits) for s in seqs))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        pp = self.par.pp
        m = self.par.microbatches(len(seqs))
        ordered = sorted(seqs, key=lambda x: (x.q, x.kv), reverse=True)
        groups = [ordered[i::m] for i in range(m)]
        first_pass = [self._stage(groups[0], s) for s in range(pp)]
        total = sum(st.total_ms for st in first_pass)
        for grp in groups[1:]:
            if len(grp) == len(groups[0]) and grp == groups[0]:
                total += max(st.total_ms for st in first_pass)
            else:
                total += max(self._stage(grp, s).total_ms for s in range(pp))
        ret = self.comm.token_return_ms(sum(x.logits for x in seqs))
        out = StepBreakdown(total + ret, m, first_pass, ret)
        self._cache[key] = out
        return out
