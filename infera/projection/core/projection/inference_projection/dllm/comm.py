###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Collectives of a refinement step, at the precision they cross the wire at.

Every collective is priced by InferaSim's own ``InferenceCollectiveModel``, so
the measured intra-node all-reduce and all-to-all fits, the NIC warm-up and the
multi-node overhead stripping all carry over unchanged. What this layer adds is
the one decision a non-autoregressive step makes that a decode step does not
have to: whether to quantize an activation before it is sent.

A decode step sends a few kilobytes per layer, so nothing it could save on the
wire pays for a pack kernel. A refinement step sends every position's hidden
state -- ``D = B x S x H x Pb`` bytes, the proposal's payload -- and when that
crosses a 50 GB/s NIC instead of the node fabric the wire time can dominate the
step. Quantizing trades a memory-bound pack and unpack, priced at HBM
bandwidth, for fewer bytes on the link, and whether it wins depends on which
link it is. ``tradeoff`` answers that per collective.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from infera.projection.core.projection.training_config import (
    InferenceCollectiveConfig,
    ModelConfig,
    ModelParallelConfig,
)

from ..collectives import InferenceCollectiveModel
from .hardware import DLLMHardware, dtype_bytes
from .model import DLLMModelSpec, Parallel

SCOPES = ("none", "inter_node", "all")

# Pack/unpack rounds per collective. A quantized all-reduce reduces in high
# precision, so it unpacks and repacks between its reduce-scatter and
# all-gather halves; a send is packed once and unpacked once; an all-to-all is
# a dispatch and a combine, each packed and unpacked.
_ROUNDS = {"tp_allreduce": 2, "ep_alltoall": 2, "pp_hop": 1}


@dataclass
class CommConfig:
    # Precision a quantized collective sends at, and which collectives do.
    payload_dtype: str = "bf16"
    quant_scope: str = "none"
    # Fraction of a layer's collectives the engine can issue concurrently with
    # compute -- the same ceiling as ``--prefill-comm-overlap``. ``0`` is a
    # synchronous collective, which is what serving engines issue today.
    overlap: float = 0.0
    tp_allreduce_algo: str = "auto"

    def __post_init__(self) -> None:
        if self.quant_scope not in SCOPES:
            raise ValueError(f"[dllm-sim] quant_scope must be one of {SCOPES}")
        if not 0.0 <= self.overlap <= 1.0:
            raise ValueError("[dllm-sim] overlap must be in [0, 1]")


@dataclass
class CommCost:
    wire_ms: float = 0.0
    quant_ms: float = 0.0
    wire_bytes: float = 0.0
    cross_node: bool = False

    @property
    def total_ms(self) -> float:
        return self.wire_ms + self.quant_ms


class DLLMCommModel:
    """Prices a stage's collectives on one replica's placement."""

    def __init__(
        self,
        spec: DLLMModelSpec,
        par: Parallel,
        hw: DLLMHardware,
        cfg: CommConfig | None = None,
    ) -> None:
        self.spec, self.par, self.hw = spec, par, hw
        self.cfg = cfg or CommConfig()
        gpn = hw.gpus_per_node
        self.nodes = max(1, -(-par.gpus_per_replica // gpn))
        mc = ModelConfig(
            hidden_size=spec.hidden_size,
            num_layers=spec.num_layers,
            num_attention_heads=spec.num_attention_heads,
            kv_channels=spec.head_dim,
            moe_router_topk=spec.moe_router_topk,
        )
        mp = ModelParallelConfig(
            tensor_model_parallel_size=par.tp,
            pipeline_model_parallel_size=par.pp,
            expert_model_parallel_size=par.ep,
        )
        cc = InferenceCollectiveConfig(
            hardware_config=hw.collective_config(),
            tp_allreduce_algo=self.cfg.tp_allreduce_algo,
        )
        self._icm = InferenceCollectiveModel(mc, mp, cc, num_nodes=self.nodes, gpus_per_node=gpn)
        self._act_bytes = dtype_bytes(spec.act_dtype)

    # -- placement -----------------------------------------------------------

    @property
    def tp_crosses_node(self) -> bool:
        return self.par.tp > self.hw.gpus_per_node

    @property
    def ep_crosses_node(self) -> bool:
        return self.par.ep > self.hw.gpus_per_node

    def pp_boundaries(self) -> list[bool]:
        """For each of the ``pp - 1`` stage boundaries, whether it leaves a node.

        Stages are placed contiguously, ``tp`` GPUs each, so a boundary crosses
        exactly when the first GPU of the next stage starts a new node.
        """
        gpn, tp = self.hw.gpus_per_node, self.par.tp
        return [((s + 1) * tp) // gpn != ((s + 1) * tp - 1) // gpn for s in range(self.par.pp - 1)]

    # -- precision -----------------------------------------------------------

    def _payload(self, cross_node: bool) -> str | None:
        """The dtype the config sends this collective at; None = unquantized."""
        scope = self.cfg.quant_scope
        if scope == "none" or dtype_bytes(self.cfg.payload_dtype) >= self._act_bytes:
            return None
        return self.cfg.payload_dtype if scope == "all" or cross_node else None

    def _pack_ms(self, elements: float, rounds: int, payload: str) -> float:
        """Quantize then dequantize ``elements``, ``rounds`` times.

        Both directions are memory-bound passes: the pack reads the activation
        and writes the narrow copy and its scales, the unpack the reverse. Each
        is its own kernel.
        """
        traffic = elements * (self._act_bytes + dtype_bytes(payload)) * 2 * rounds
        launches = 2 * rounds
        return traffic / self.hw.mem_bytes_per_s() * 1e3 + launches * self.hw.kernel_launch_us / 1e3

    # -- collectives -----------------------------------------------------------

    @functools.lru_cache(maxsize=4096)  # noqa: B019 - one model per step model
    def _cost(self, kind: str, tokens: int, payload: str | None = "") -> CommCost:
        """``payload`` "" follows the config; None sends at activation precision."""
        if tokens <= 0:
            return CommCost()
        h = self.spec.hidden_size
        if kind == "tp_allreduce":
            cross = self.tp_crosses_node
        elif kind == "ep_alltoall":
            cross = self.ep_crosses_node
        elif kind in ("pp_hop_intra", "pp_hop_cross"):
            cross = kind == "pp_hop_cross"
        else:
            raise ValueError(f"[dllm-sim] unknown collective {kind!r}")
        if payload == "":
            payload = self._payload(cross)
        bpe = dtype_bytes(payload) if payload else self._act_bytes
        elements = float(tokens) * h
        if kind == "tp_allreduce":
            # The model prices both of a layer's all-reduces; this is one.
            wire = self._icm.tp_allreduce_ms(1, tokens, bpe) / 2.0
            rounds = _ROUNDS["tp_allreduce"]
        elif kind == "ep_alltoall":
            wire = self._icm.ep_a2a_ms(1, tokens, bpe)
            elements *= self.spec.moe_router_topk
            rounds = _ROUNDS["ep_alltoall"]
        else:
            wire = self._icm.pp_hop_ms(elements * bpe, cross_node=cross)
            rounds = _ROUNDS["pp_hop"]
        return CommCost(
            wire_ms=float(wire),
            quant_ms=self._pack_ms(elements, rounds, payload) if payload else 0.0,
            wire_bytes=elements * bpe,
            cross_node=cross,
        )

    def collective(self, kind: str, tokens: int) -> CommCost:
        return self._cost(kind, int(tokens))

    def pp_hop(self, tokens: int, *, cross_node: bool) -> CommCost:
        return self._cost("pp_hop_cross" if cross_node else "pp_hop_intra", int(tokens))

    def token_return_ms(self, positions: int) -> float:
        """The last stage's chosen tokens back to the first stage.

        The first stage cannot start the next refinement step until it knows
        which positions were unmasked, so this is on the critical path once per
        step. It is a few bytes a position, never worth quantizing.
        """
        if self.par.pp <= 1 or positions <= 0:
            return 0.0
        cross = self.nodes > 1
        return float(self._icm.pp_hop_ms(positions * 8.0, cross_node=cross))

    def tradeoff(self, kind: str, tokens: int, payload: str = "fp8") -> dict[str, float]:
        """The same collective at activation precision and sent at ``payload``."""
        plain = self._cost(kind, int(tokens), None)
        quant = self._cost(kind, int(tokens), payload)
        return {
            "plain_ms": plain.total_ms,
            "quantized_wire_ms": quant.wire_ms,
            "quantized_pack_ms": quant.quant_ms,
            "quantized_ms": quant.total_ms,
            "saving_ms": plain.total_ms - quant.total_ms,
            "cross_node": float(plain.cross_node),
        }
