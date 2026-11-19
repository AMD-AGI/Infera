###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Where registered op models enter the inference projection.

A forward step is priced on one representative layer per kind (dense / MoE) and
multiplied by the layer counts, so an op that runs on a fraction ``f`` of the
layers enters as a per-layer average: ``(1 - f) * built_in + f * op`` for a
replacing slot and ``+ f * op`` for an additive one -- the same blend
``_forward_times`` applies to the built-in linear-attention layers.
"""

from __future__ import annotations

import os

from .base import (
    SLOT_ATTENTION,
    SLOT_ATTENTION_EXTRA,
    SLOT_LAYER_EXTRA,
    SLOT_MLP,
    OpContext,
    OpModel,
)
from .registry import active_op_models

_ANNOUNCED: set = set()


def _fractioned(ops: list[OpModel], model_config) -> list[tuple[OpModel, float]]:
    out = []
    for op in ops:
        f = max(0.0, min(1.0, float(op.layer_fraction(model_config))))
        if f > 0.0:
            out.append((op, f))
    return out


def _announce(op: OpModel, frac: float, ctx: OpContext, ms: float, breakdown: dict) -> None:
    key = (op.name, ctx.phase)
    if key in _ANNOUNCED and not os.getenv("INFERASIM_DEBUG_BREAKDOWN"):
        return
    _ANNOUNCED.add(key)
    parts = " ".join(f"{k}={v:.4f}" for k, v in breakdown.items())
    print(
        f"[InferaSim:OpModel] {op.name} ({op.slot}) on {frac:.0%} of layers, {ctx.phase} "
        f"(batch={ctx.batch}, q_len={ctx.q_len}, kv_len={ctx.kv_len}): "
        f"{ms:.4f} ms/layer" + (f"  [{parts}]" if parts else "")
    )


class LayerOpModels:
    """The op models of one model, split by slot, ready to price a step."""

    def __init__(self, model_config):
        self.mc = model_config
        ops = active_op_models(model_config)
        self.attention = _fractioned([o for o in ops if o.slot == SLOT_ATTENTION], model_config)
        self.attention_extra = _fractioned(
            [o for o in ops if o.slot == SLOT_ATTENTION_EXTRA], model_config
        )
        self.mlp = _fractioned([o for o in ops if o.slot == SLOT_MLP], model_config)
        self.layer_extra = _fractioned([o for o in ops if o.slot == SLOT_LAYER_EXTRA], model_config)

    def __bool__(self) -> bool:
        return bool(self.attention or self.attention_extra or self.mlp or self.layer_extra)

    @property
    def touches_attention(self) -> bool:
        return bool(self.attention or self.attention_extra)

    @staticmethod
    def _sum(pairs, ctx: OpContext) -> float:
        total = 0.0
        for op, f in pairs:
            c = op.cost(ctx)
            _announce(op, f, ctx, c.ms, c.breakdown)
            total += f * c.ms
        return total

    def attention_ms(self, ctx: OpContext, built_in_ms: float, other_share: float = 0.0) -> float:
        """Attention block per layer: built-in on what no op (or ``other_share``) replaces."""
        f_rep = min(1.0 - other_share, sum(f for _, f in self.attention))
        return (
            (1.0 - other_share - f_rep) * built_in_ms
            + self._sum(self.attention, ctx)
            + self._sum(self.attention_extra, ctx)
        )

    def mlp_ms(self, ctx: OpContext, built_in_ms: float) -> float:
        f_rep = min(1.0, sum(f for _, f in self.mlp))
        return (1.0 - f_rep) * built_in_ms + self._sum(self.mlp, ctx)

    def layer_extra_ms(self, ctx: OpContext) -> float:
        return self._sum(self.layer_extra, ctx)


def layer_op_models(model_config) -> LayerOpModels | None:
    lom = LayerOpModels(model_config)
    return lom if lom else None


def kv_cache_op_terms(
    inference_config, per_token_per_layer: float, layers_on_rank: int
) -> tuple[float, float]:
    """``(delta_bytes_per_token, state_bytes_per_sequence)`` the op models add on one rank.

    A replacing attention op that reports ``kv_bytes_per_token`` swaps the
    built-in per-token entry on its share of the layers (0 for a layer with
    no cache); any op's ``state_bytes_per_seq`` is fixed-size state a resident
    sequence holds on those layers.
    """
    mc = inference_config.model_config
    lom = layer_op_models(mc)
    if lom is None:
        return 0.0, 0.0
    view = inference_config.as_training_config(batch_size=1, seq_len=1)
    ctx = OpContext(view, 1, 1, 1, "decode")
    layers = max(1, int(layers_on_rank))
    d_token = state = 0.0
    for pairs, replaces in (
        (lom.attention, True),
        (lom.attention_extra, False),
        (lom.mlp, False),
        (lom.layer_extra, False),
    ):
        for op, f in pairs:
            c = op.cost(ctx)
            if c.kv_bytes_per_token is not None:
                d_token += (
                    f * layers * (c.kv_bytes_per_token - (per_token_per_layer if replaces else 0.0))
                )
            if c.state_bytes_per_seq:
                state += f * layers * c.state_bytes_per_seq
    return d_token, state
