###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The time of one refinement step on one replica.

A stage is its layers and, on the first and last stage, the embedding and the
head. Each layer is charged the way the autoregressive projector charges a
forward: its compute, plus its collectives exposed by ``overlap_keep`` against
that same layer type's compute -- a dense layer's all-reduces, a MoE layer's
all-reduce and all-to-all each at their own exposure. Any pack kernels a
quantized collective needs, and the opt-in contention penalty on the part that
was hidden, come on top.

A pipelined step cuts the batch into micro-batches: the first one crosses
every stage, each later one adds the slowest stage, and the forward's
``pp_p2p_ms`` -- every stage boundary and the sampled tokens' return -- is
charged once at the pipeline's exposure, as the projector charges it. The
later micro-batches' sends run on the communication stream behind the stages
still computing.

Compute comes from a backend. The default prices this package's op graph with
the hardware's pricer: Origami's GEMM and FAv3 attention models, as the
projector uses, unless the roofline was explicitly asked for.
``InferaSimBackend`` asks the projector itself for a ``refine``-phase forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .comm import CommConfig, CommCost, DLLMCommModel
from .contention import ContentionConfig, penalty_ms
from .hardware import DLLMHardware
from .model import DLLMModelSpec, Parallel, SeqShape, StageGraph, build_stage_graph
from .ops import ATTN_BLOCK, DENSE_MLP, MOE_MLP, ONCE, group_ms, memory_bound_fraction, summarize


class ComputeBackend(Protocol):
    def stage_compute(
        self, graph: StageGraph, seqs: list[SeqShape], *, first: bool, last: bool
    ) -> dict[str, float]:
        """Compute per group (``ops.ATTN_BLOCK`` ... ``ops.ONCE``), all layers."""
        ...


class OpGraphBackend:
    """The stage's op graph, priced by ``hw.pricer`` (Origami) or the roofline."""

    def __init__(self, hw: DLLMHardware) -> None:
        self.hw = hw

    def stage_compute(
        self, graph: StageGraph, seqs: list[SeqShape], *, first: bool, last: bool
    ) -> dict[str, float]:
        return group_ms(graph.ops, self.hw)


class InferaSimBackend:
    """Compute from ``InferencePerformanceProjector._forward_times``.

    The projector prices a uniform batch, so a ragged step is priced at its
    mean shape, and its own rank's share of the pipeline, so every stage gets
    the same time. Its collectives are taken back out -- this package charges
    them with the same model -- and what is left of its layer time is split
    between dense and MoE layers in the proportion the projector priced them.
    """

    def __init__(self, projector: Any) -> None:
        self.projector = projector

    def stage_compute(
        self, graph: StageGraph, seqs: list[SeqShape], *, first: bool, last: bool
    ) -> dict[str, float]:
        n = len(seqs)
        q = max(1, round(sum(s.q for s in seqs) / n))
        kv = max(q, round(sum(s.kv for s in seqs) / n))
        ft = self.projector._forward_times(n, q, "refine", kv)
        c = ft.comm
        layer_comm = c.tp_allreduce_ms + c.ep_a2a_ms + c.dcp_ms
        layers = max(0.0, float(ft.layers_ms) - layer_comm)
        wd = graph.n_dense * float(ft.dense_layer_ms)
        wm = graph.n_moe * float(ft.moe_layer_ms)
        share = wd / (wd + wm) if wd + wm > 0 else 1.0
        return {
            ATTN_BLOCK: 0.0,
            DENSE_MLP: layers * share,
            MOE_MLP: layers * (1.0 - share),
            ONCE: max(0.0, float(ft.total_ms) - float(ft.layers_ms) - c.pp_p2p_ms),
        }


@dataclass
class StageTime:
    total_ms: float = 0.0
    compute_ms: float = 0.0
    comm_wire_ms: float = 0.0
    quant_ms: float = 0.0
    exposed_comm_ms: float = 0.0
    contention_ms: float = 0.0
    # Exposure of a dense layer's and a MoE layer's all-reduce, which the
    # pipeline's sends inherit.
    keep_dense: float = 1.0
    keep_moe: float = 1.0
    by_kind: dict[str, float] = field(default_factory=dict)
    mem_bound_fraction: float = 0.0


@dataclass
class StepBreakdown:
    total_ms: float
    microbatches: int
    stages: list[StageTime]
    pp_p2p_ms: float = 0.0
    # measured / simulated at the nearest anchor point; 1 when not anchored.
    # ``total_ms`` carries it, the component times are the simulation's.
    anchor_scale: float = 1.0

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
            "pp_p2p_ms": self.pp_p2p_ms,
            "anchor_scale": self.anchor_scale,
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
        self.backend = backend or OpGraphBackend(hw)
        self._has_moe = spec.num_moe_layers() > 0
        self._cache: dict[tuple, StepBreakdown] = {}

    def _penalty(self, c: CommCost, wire_ms: float, keep: float, mem_frac: float) -> float:
        share = wire_ms / c.wire_ms if c.wire_ms > 0 else 0.0
        return penalty_ms(
            wire_ms,
            keep,
            wire_bytes=c.wire_bytes * share,
            mem_fraction=mem_frac,
            hw=self.hw,
            cfg=self.contention,
        )

    def _stage(self, seqs: list[SeqShape], s: int) -> StageTime:
        pp = self.par.pp
        first, last = s == 0, s == pp - 1
        g = build_stage_graph(self.spec, seqs, self.par, first=first, last=last)
        parts = self.backend.stage_compute(g, seqs, first=first, last=last)
        compute = sum(parts.values())
        attn = parts.get(ATTN_BLOCK, 0.0) / g.layers if g.layers else 0.0
        mem_frac = memory_bound_fraction(g.ops, self.hw)

        wire = quant = exposed = contention = 0.0
        keep_dense = keep_moe = 1.0
        if g.n_dense:
            layer = attn + parts.get(DENSE_MLP, 0.0) / g.n_dense
            c = self.comm.layer("dense", g.tokens)
            keep_dense = self.comm.keep(c.wire_ms, layer)
            wire += g.n_dense * c.wire_ms
            quant += g.n_dense * c.quant_ms
            exposed += g.n_dense * c.wire_ms * keep_dense
            contention += g.n_dense * self._penalty(c, c.wire_ms, keep_dense, mem_frac)
        if g.n_moe:
            layer = attn + parts.get(MOE_MLP, 0.0) / g.n_moe
            c = self.comm.layer("moe", g.tokens)
            ar = c.wire_ms - c.a2a_ms
            keep_moe = self.comm.keep(ar, layer)
            keep_a2a = self.comm.keep(c.a2a_ms, layer)
            wire += g.n_moe * c.wire_ms
            quant += g.n_moe * c.quant_ms
            exposed += g.n_moe * (ar * keep_moe + c.a2a_ms * keep_a2a)
            contention += g.n_moe * (
                self._penalty(c, ar, keep_moe, mem_frac)
                + self._penalty(c, c.a2a_ms, keep_a2a, mem_frac)
            )
        return StageTime(
            total_ms=compute + exposed + quant + contention,
            compute_ms=compute,
            comm_wire_ms=wire,
            quant_ms=quant,
            exposed_comm_ms=exposed,
            contention_ms=contention,
            keep_dense=keep_dense,
            keep_moe=keep_moe,
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
            if grp == groups[0]:
                total += max(st.total_ms for st in first_pass)
            else:
                total += max(self._stage(grp, s).total_ms for s in range(pp))
        p2p = 0.0
        if pp > 1:
            st0 = first_pass[0]
            pp_keep = st0.keep_moe if self._has_moe else st0.keep_dense
            c = self.comm.pipeline(sum(x.q for x in groups[0]))
            p2p = c.wire_ms * pp_keep + c.quant_ms
        out = StepBreakdown(total + p2p, m, first_pass, p2p)
        self._cache[key] = out
        return out
