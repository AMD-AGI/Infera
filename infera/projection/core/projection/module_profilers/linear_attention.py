###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Time of one linear-attention layer (KDA / gated delta net), inference-only.

A linear layer keeps a fixed-size recurrent state per head instead of a KV
cache, and its cost is nothing like an attention layer reading ``d`` keys:

  * **Projections.** KDA (Kimi-K3) projects q, k, v and a full output gate,
    each ``hidden x heads*d``, plus a low-rank per-channel decay gate and a
    per-head beta. GDN (Qwen3.5) fuses q, k, v and its gate into one
    projection and its two per-head scalars into another. At decode these
    weight reads are most of the layer: for Kimi-K3 they are ~440M parameters
    a layer, about twice one of its gated MLA layers (~230M).
  * **Short convolution.** A causal depthwise conv over q, k and v. At decode
    it reads and rewrites a ``kernel - 1`` row window per sequence.
  * **Recurrence.** Decode reads and rewrites the ``dk x dv`` state of every
    value head of every sequence, every step -- the term that grows with
    batch. Prefill runs the chunked form, whose arithmetic is batched
    chunk-sized matmuls.
  * **Gated norm** on the output, then the output projection.

Heads shard over tensor parallelism like GQA; attention-DP keeps them whole
and splits the requests instead, as the attention profiler does.
"""

from __future__ import annotations

import math

from .moe_mlp import _ACTIVATION_BW_FRACTION, _FALLBACK_HBM_BW_GBPS

# The chunk the chunked prefill kernels (fla ``chunk_kda`` / ``chunk_gated_delta_rule``)
# run at.
_CHUNK = 64
_ACT_BYTES = 2.0  # bf16 activations and stored state, as the memory model sizes it


class LinearAttentionProfiler:
    def __init__(self, config, gemm_backend, *, hbm_bandwidth_gbps: float | None = None):
        self.config = config
        self._gemm = gemm_backend
        self._bw = float(hbm_bandwidth_gbps or _FALLBACK_HBM_BW_GBPS) * _ACTIVATION_BW_FRACTION

    def _stream_ms(self, nbytes: float) -> float:
        return nbytes / (self._bw * 1e6)

    def _gemm_ms(self, m: int, n: int, k: int, dtype: str, batch: int = 1) -> float:
        return self._gemm.simulate_gemm(
            max(1, m), max(1, n), max(1, k), dtype, batch=batch
        ).forward_time_ms

    def layer_ms(self, batch: int, q_len: int, phase: str) -> float:
        """One linear layer's attention block on one rank, in ms."""
        mc = self.config.model_config
        mp = self.config.model_parallel_config
        tp = max(1, mp.tensor_model_parallel_size)
        cp = max(1, mp.context_model_parallel_size)
        dp = max(1, getattr(mp, "attention_data_parallel_size", 1) or 1)
        if dp > 1:
            tp = max(1, tp // dp)
            batch = max(1, -(-batch // dp))
        hk, hv, dk, dv = mc.linear_attention_geometry()
        hk, hv = max(1, hk // tp), max(1, hv // tp)
        pk, pv = hk * dk, hv * dv
        hidden = mc.hidden_size
        tokens = max(1, batch * q_len // cp)
        w = getattr(mc, "linear_weight_dtype", None) or (
            "fp8" if getattr(mc, "fp8", None) else "bf16"
        )
        kda = str(mc.linear_attention_kind).lower() == "kda"

        ms = 0.0
        if kda:
            ms += self._gemm_ms(tokens, pk, hidden, w)  # q
            ms += self._gemm_ms(tokens, pk, hidden, w)  # k
            ms += self._gemm_ms(tokens, pv, hidden, w)  # v
            ms += self._gemm_ms(tokens, dk, hidden, w)  # decay gate down, replicated
            ms += self._gemm_ms(tokens, pv, dk, w)  # decay gate up
            ms += self._gemm_ms(tokens, hv, hidden, w)  # beta
            if mc.linear_attention_full_rank_gate:
                ms += self._gemm_ms(tokens, pv, hidden, w)
            else:
                ms += self._gemm_ms(tokens, dv, hidden, w) + self._gemm_ms(tokens, pv, dv, w)
        else:
            ms += self._gemm_ms(tokens, 2 * pk + 2 * pv, hidden, w)  # q, k, v, z
            ms += self._gemm_ms(tokens, 2 * hv, hidden, w)  # a, b
        ms += self._gemm_ms(tokens, hidden, pv, w)  # output projection

        conv_ch = 2 * pk + pv
        kernel = max(1, int(mc.linear_attention_conv_kernel or 4))
        conv_bytes = 2 * tokens * conv_ch * _ACT_BYTES + conv_ch * kernel * _ACT_BYTES
        conv_bytes += 2 * batch * conv_ch * (kernel - 1) * _ACT_BYTES  # window read + write
        ms += self._stream_ms(conv_bytes)

        state = batch * hv * dk * dv * _ACT_BYTES
        io = tokens * (2 * pk + 3 * pv) * _ACT_BYTES  # q, k, v, decay gate in; output out
        ms += self._stream_ms(2 * state + io)
        if phase != "decode":
            # Chunked form: per chunk of every value head, the intra-chunk
            # scores and their application (two C x C x d products), the
            # WY correction (two more), and the state's read and update.
            chunks = batch * hv * max(1, math.ceil(q_len / cp / _CHUNK))
            c = _CHUNK
            ms += self._gemm_ms(c, c, dk, "bf16", batch=chunks)
            ms += self._gemm_ms(c, dv, c, "bf16", batch=chunks)
            ms += self._gemm_ms(c, c, dk, "bf16", batch=chunks)
            ms += self._gemm_ms(c, dv, c, "bf16", batch=chunks)
            ms += self._gemm_ms(c, dv, dk, "bf16", batch=chunks)
            ms += self._gemm_ms(dk, dv, c, "bf16", batch=chunks)

        ms += self._stream_ms(3 * tokens * pv * _ACT_BYTES)  # gated norm: o, gate in; out
        return ms
