###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A refinement step's collectives, priced by the autoregressive path's model.

Nothing here is a second communication model. The collective model is
InferaSim's ``InferenceCollectiveModel``, built the way the projector builds
it -- the collective model's own link defaults, the architecture's datasheet
interconnect when one is known, a calibrated ``INFERASIM_COLL_HW`` profile if
set -- and a forward's collectives are charged by the projector's rules:

* a dense layer pays ``tp_allreduce_ms`` (its attention and MLP reductions);
* a MoE layer pays that scaled by ``_moe_tp_allreduce_count /
  _dense_tp_allreduce_count`` -- one reduction once experts are fully expert
  parallel, since the all-to-all combines them -- plus ``ep_a2a_ms``;
* a pipelined forward pays ``pp_p2p_ms``: its stage boundaries, and the
  sampled tokens' return to the first stage;
* each is exposed by ``overlap_keep`` against the same layer type's compute,
  up to the ``refine`` phase's ceiling, ``prefill_overlap``.

What this layer adds is opt-in and off by default, so a default run is the
projector's communication exactly. A refinement step sends every position's
hidden state -- ``D = B x S x H x Pb`` bytes per collective -- and quantizing
that payload to FP8 or MXFP4 trades a memory-bound pack and unpack for fewer
bytes on the wire. ``quant_scope`` sends the narrower ``payload_dtype``, the
collective model is priced at that width, and the pack kernels are added.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from types import SimpleNamespace

from infera.projection.core.projection.module_profilers.transformer_layer import (
    _dense_tp_allreduce_count,
    _moe_tp_allreduce_count,
)
from infera.projection.core.projection.training_config import (
    INTERCONNECT_PROFILES,
    InferenceCollectiveConfig,
    ModelConfig,
    ModelParallelConfig,
)

from ..collectives import InferenceCollectiveModel
from ..performance import overlap_keep
from .hardware import DLLMHardware, dtype_bytes
from .model import DLLMModelSpec, Parallel

SCOPES = ("none", "inter_node", "all")

# Pack/unpack rounds per collective. A quantized all-reduce reduces in high
# precision, so it unpacks and repacks between its reduce-scatter and
# all-gather halves; an all-to-all is a dispatch and a combine; a pipeline
# send is packed once and unpacked once.
_ROUNDS_PER_ALLREDUCE = 2
_ROUNDS_PER_ALLTOALL = 2
_ROUNDS_PER_HOP = 1

# How the pack and unpack run. ``fused`` is one kernel per direction moving the
# activation and its narrow copy once. ``eager`` is the same arithmetic as
# separate framework ops -- absmax, scale, cast, concatenate the scales, and
# the reverse -- measured on MI325X at 7.5x the fused traffic and 10 kernels
# per direction (an FP8 two-shot all-reduce over 8 GPUs: 5.8 ms of pack
# kernels at 512 MB, 0.18 ms at 256 KB).
PACKS = {"fused": (1.0, 1), "eager": (7.5, 10)}


@dataclass
class CommConfig:
    # Precision a quantized collective sends at, and which collectives do.
    payload_dtype: str = "bf16"
    quant_scope: str = "none"
    # The overlap ceiling of the ``refine`` phase: the projector's
    # ``prefill_overlap``. 0 = synchronous collectives, the projector's default.
    overlap: float = 0.0
    tp_allreduce_algo: str = "auto"
    pack: str = "fused"

    def __post_init__(self) -> None:
        if self.quant_scope not in SCOPES:
            raise ValueError(f"[dllm-sim] quant_scope must be one of {SCOPES}")
        if self.pack not in PACKS:
            raise ValueError(f"[dllm-sim] pack must be one of {tuple(PACKS)}")
        if not 0.0 <= self.overlap <= 1.0:
            raise ValueError("[dllm-sim] overlap must be in [0, 1]")


@dataclass
class CommCost:
    wire_ms: float = 0.0
    quant_ms: float = 0.0
    wire_bytes: float = 0.0
    # The all-to-all's share of ``wire_ms``: the projector exposes it on its own.
    a2a_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.wire_ms + self.quant_ms


class DLLMCommModel:
    """One replica's collectives, per layer type and per forward."""

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
            prefill_overlap=self.cfg.overlap,
        )
        if cc.interconnect is None:
            cc.interconnect = INTERCONNECT_PROFILES.get(str(hw.name).lower())
        self.collective_config = cc
        self._icm = InferenceCollectiveModel(mc, mp, cc, num_nodes=self.nodes, gpus_per_node=gpn)
        view = SimpleNamespace(model_parallel_config=mp)
        self._dense_ars = _dense_tp_allreduce_count(view)
        self._moe_ars = _moe_tp_allreduce_count(view)
        self._act_bytes = dtype_bytes(spec.act_dtype)

    # -- placement and precision -------------------------------------------

    @property
    def tp_crosses_node(self) -> bool:
        return self.par.tp > self.hw.gpus_per_node

    @property
    def ep_crosses_node(self) -> bool:
        return self.par.ep > self.hw.gpus_per_node

    @property
    def pp_crosses_node(self) -> bool:
        return self.par.pp > 1 and self._icm.pp_nodes() > 1

    def _payload(self, cross_node: bool) -> str | None:
        """The dtype the config sends a collective at; None = unquantized."""
        scope = self.cfg.quant_scope
        if scope == "none" or dtype_bytes(self.cfg.payload_dtype) >= self._act_bytes:
            return None
        return self.cfg.payload_dtype if scope == "all" or cross_node else None

    def _pack_ms(self, elements: float, rounds: int, payload: str | None) -> float:
        """Quantize then dequantize ``elements``, ``rounds`` times.

        Both directions are memory-bound passes: the pack reads the activation
        and writes the narrow copy and its scales, the unpack the reverse. Each
        is its own kernel.
        """
        if not payload or rounds <= 0:
            return 0.0
        scale, kernels = PACKS[self.cfg.pack]
        traffic = elements * (self._act_bytes + dtype_bytes(payload)) * 2 * rounds * scale
        launches = 2 * rounds * kernels
        return traffic / self.hw.mem_bytes_per_s() * 1e3 + launches * self.hw.kernel_launch_us / 1e3

    def _bpe(self, payload: str | None) -> float:
        return dtype_bytes(payload) if payload else self._act_bytes

    # -- the projector's per-forward collectives ----------------------------

    @functools.lru_cache(maxsize=4096)  # noqa: B019 - one model per step model
    def layer(self, kind: str, tokens: int, payload: str | None = "") -> CommCost:
        """One ``dense`` or ``moe`` layer's collectives over ``tokens`` rows.

        ``payload`` "" follows the config; None sends at activation precision.
        """
        if tokens <= 0:
            return CommCost()
        h = self.spec.hidden_size
        rows = float(tokens) * h
        follow = payload == ""
        a2a_payload = self._payload(self.ep_crosses_node) if follow else payload
        if follow:
            payload = self._payload(self.tp_crosses_node)
        bpe = self._bpe(payload)
        tp_ar = float(self._icm.tp_allreduce_ms(1, tokens, bpe))
        if kind == "dense":
            ars = self._dense_ars
            wire = tp_ar
            quant = self._pack_ms(rows * ars, _ROUNDS_PER_ALLREDUCE, payload)
            return CommCost(wire, quant, rows * bpe * ars)
        if kind != "moe":
            raise ValueError(f"[dllm-sim] unknown layer kind {kind!r}")
        ars = self._moe_ars
        wire = tp_ar * (ars / self._dense_ars) if self._dense_ars else 0.0
        quant = self._pack_ms(rows * ars, _ROUNDS_PER_ALLREDUCE, payload)
        nbytes = rows * bpe * ars
        a2a = 0.0
        if self.par.ep > 1:
            a2a_bpe = self._bpe(a2a_payload)
            a2a = float(self._icm.ep_a2a_ms(1, tokens, a2a_bpe))
            a2a_rows = rows * self.spec.moe_router_topk
            quant += self._pack_ms(a2a_rows, _ROUNDS_PER_ALLTOALL, a2a_payload)
            nbytes += 2 * a2a_rows * a2a_bpe
        return CommCost(wire + a2a, quant, nbytes, a2a)

    @functools.lru_cache(maxsize=4096)  # noqa: B019 - one model per step model
    def pipeline(self, tokens: int, payload: str | None = "") -> CommCost:
        """``pp_p2p_ms`` for one forward of ``tokens`` rows, plus pack kernels."""
        if self.par.pp <= 1 or tokens <= 0:
            return CommCost()
        if payload == "":
            payload = self._payload(self.pp_crosses_node)
        bpe = self._bpe(payload)
        rows = float(tokens) * self.spec.hidden_size
        hops = self.par.pp - 1
        return CommCost(
            wire_ms=float(self._icm.pp_p2p_ms(1, tokens, bpe)),
            quant_ms=self._pack_ms(rows * hops, _ROUNDS_PER_HOP, payload),
            wire_bytes=rows * bpe * hops,
        )

    def keep(self, comm_ms: float, compute_ms: float | None) -> float:
        """The projector's exposed fraction at the ``refine`` ceiling."""
        return overlap_keep(self.cfg.overlap, comm_ms, compute_ms)

    def tradeoff(self, kind: str, tokens: int, payload: str = "fp8") -> dict[str, float]:
        """One ``dense``/``moe`` layer's or one ``pipeline`` forward's
        collectives at activation precision and sent at ``payload``."""
        if kind == "pipeline":
            plain, quant = self.pipeline(int(tokens), None), self.pipeline(int(tokens), payload)
        else:
            plain, quant = (
                self.layer(kind, int(tokens), None),
                self.layer(kind, int(tokens), payload),
            )
        return {
            "plain_ms": plain.total_ms,
            "quantized_wire_ms": quant.wire_ms,
            "quantized_pack_ms": quant.quant_ms,
            "quantized_ms": quant.total_ms,
            "saving_ms": plain.total_ms - quant.total_ms,
        }
