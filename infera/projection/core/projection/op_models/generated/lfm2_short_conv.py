###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""LFM2 gated short convolution (Liquid AI LFM2 ``conv`` layers), inference.

The conv operator replaces attention on the layers ``lfm_layer_types`` marks
``conv`` (HF ``Lfm2ShortConv``):

    B, C, x = split(x W_in, 3)            W_in: hidden -> 3 * hidden
    y = C * causal_dwconv_L(B * x)        depthwise, kernel L = conv_L_cache
    out = y W_out                         W_out: hidden -> hidden

The two projections are GEMMs; the gates and the depthwise conv (L = 3 taps)
are memory bound and priced as HBM streams.  Channels shard over TP like the
attention heads they replace.

Serving keeps no KV cache on these layers: a sequence carries only the last
``L - 1`` inputs of ``B * x`` per channel (the conv state vLLM / SGLang keep in
their Mamba-style state cache), read and rewritten every decode step.
"""

from __future__ import annotations

from infera.projection.core.projection.op_models.base import (
    SLOT_ATTENTION,
    OpContext,
    OpCost,
    OpModel,
    cfg_value,
)
from infera.projection.core.projection.op_models.registry import register_op_model

_ACT = 2.0


@register_op_model
class Lfm2ShortConv(OpModel):
    name = "lfm2_short_conv"
    slot = SLOT_ATTENTION
    claims = {"lfm_layer_types": ("conv",)}
    consumes = ("lfm_layer_types", "conv_L_cache", "conv_bias")
    references = (
        "Liquid AI. LFM2 Technical Report (gated short convolution blocks). arXiv:2511.23404",
        "HF transformers Lfm2ShortConv (in_proj -> B, C, x; conv1d groups=hidden; out_proj)",
        "vLLM ShortConv layer (conv state cache of conv_L_cache - 1 inputs per channel)",
    )

    @classmethod
    def matches(cls, model_config) -> bool:
        types = cfg_value(model_config, "lfm_layer_types") or []
        return any(str(t).lower() == "conv" for t in types)

    def layer_fraction(self, model_config) -> float:
        types = cfg_value(model_config, "lfm_layer_types") or []
        return sum(str(t).lower() == "conv" for t in types) / len(types) if types else 0.0

    def cost(self, ctx: OpContext) -> OpCost:
        h = int(ctx.mc.hidden_size)
        L = int(ctx.value("conv_L_cache", 3) or 3)
        m = ctx.tokens
        ch_r = ctx.per_rank(h)
        seqs = ctx.batch_per_rank

        parts, flops, nbytes = {}, 0.0, 0.0
        for tag, n, k, shard in (("in_proj", 3 * h, h, "column"), ("out_proj", h, h, "row")):
            parts[tag] = ctx.linear(n, k, shard=shard)
            flops += (
                2.0
                * m
                * ctx.per_rank(n if shard == "column" else k)
                * (k if shard == "column" else n)
            )
            nbytes += ctx.linear_bytes(n, k, shard=shard)

        # B * x (read B, x; write), depthwise conv (read, write), C * y (read C, y; write),
        # plus the per-sequence conv window: read (decode) and rewrite.
        state_bytes = (L - 1) * ch_r * _ACT
        elem_bytes = (3 + 2 + 3) * m * ch_r * _ACT + seqs * state_bytes * (
            2 if ctx.is_decode else 1
        )
        parts["gates_conv"] = ctx.stream_ms(elem_bytes)
        flops += m * ch_r * (2.0 * L + 2.0)
        nbytes += elem_bytes

        return OpCost(
            ms=sum(parts.values()),
            flops=flops,
            bytes=nbytes,
            breakdown=parts,
            kv_bytes_per_token=0.0,
            state_bytes_per_seq=state_bytes,
            num_params=4 * h * h + h * L,
        )

    def reference_flops(self, ctx: OpContext):
        h, L = int(ctx.mc.hidden_size), int(ctx.value("conv_L_cache", 3) or 3)
        m, ch_r = ctx.tokens, ctx.per_rank(h)
        return 2.0 * m * 4 * h * h / ctx.tp + m * ch_r * (2.0 * L + 2.0)
