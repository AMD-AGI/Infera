###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Runtime activation quantization / cast cost model (inference-only).

Low-precision serving (fp8, mxfp4) keeps the *weights* pre-quantized offline,
but the *activations* feeding each GEMM are produced in bf16 and must be cast
at runtime: read the bf16 activation, compute an amax (per-token or per-tensor
for fp8; per-32-element block for mxfp4), and write the packed low-precision
tensor plus its scale(s).  These cast kernels are pure memory-bandwidth ops the
GEMM simulator does not see, and they are not part of the (bf16) training
forward we ported.  We price them per transformer layer so a low-precision
step's cost includes the quantization tail (which matters at decode, where the
GEMMs themselves are tiny).

Per element the cast streams ``read(bf16) + write(packed) + scale`` bytes at
the sequential HBM-BW fraction.  A layer casts a handful of activation buffers:

  * **dense** — attention input, MLP input (both ``hidden``-wide) and the MLP
    down-projection input (``ffn``-wide).
  * **MoE**  — attention input (``hidden``-wide over the local tokens) plus the
    routed expert input (``hidden``-wide) and expert down-projection input
    (``moe_ffn``-wide), both over the ``topk``-expanded token set.
"""

from __future__ import annotations

# Activation casts stream contiguous buffers → same sequential HBM efficiency
# as the other element-wise ops.
_CAST_BW_FRACTION = 0.566
_FALLBACK_HBM_BW_GBPS = 5300.0
_READ_BYTES_BF16 = 2.0  # activations arrive in bf16

# Bytes written per element for the packed low-precision tensor, including the
# amortized scale metadata (mxfp4 stores one shared exponent per 32-element
# block → +1/32 byte; fp8 uses per-token/per-tensor scales → negligible).
_WRITE_BYTES = {
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "mxfp4": 0.5 + 1.0 / 32.0,
}


class QuantCastProfiler:
    """Analytical cost of runtime activation quantization for low-precision GEMMs."""

    def __init__(self, config, *, hbm_bandwidth_gbps: float | None = None, dtype: str = "fp8"):
        self.config = config
        self._hbm_bw = float(hbm_bandwidth_gbps or _FALLBACK_HBM_BW_GBPS)
        self.dtype = str(dtype).lower()
        self._write_bytes = _WRITE_BYTES.get(self.dtype, 1.0)

    def _per_token_rank(self, batch: int, q_len: int) -> int:
        mp = self.config.model_parallel_config
        tp = max(1, mp.tensor_model_parallel_size)
        cp = max(1, mp.context_model_parallel_size)
        return max(1, batch * q_len // tp // cp)

    def _cast_ms(self, tokens: int, width: int) -> float:
        eff_bw = self._hbm_bw * _CAST_BW_FRACTION  # GB/s
        tensor_bytes = tokens * max(1, width) * (_READ_BYTES_BF16 + self._write_bytes)
        return tensor_bytes / (eff_bw * 1e6)

    def dense_layer_ms(self, batch: int, q_len: int) -> float:
        """Activation-cast cost for one dense transformer layer."""
        mc = self.config.model_config
        tokens = self._per_token_rank(batch, q_len)
        hidden = mc.hidden_size
        ffn = mc.ffn_hidden_size
        # attention input + MLP input (hidden-wide) + MLP down input (ffn-wide)
        return 2 * self._cast_ms(tokens, hidden) + self._cast_ms(tokens, ffn)

    def moe_layer_ms(self, batch: int, q_len: int) -> float:
        """Activation-cast cost for one MoE transformer layer."""
        mc = self.config.model_config
        tokens = self._per_token_rank(batch, q_len)
        hidden = mc.hidden_size
        moe_ffn = mc.moe_ffn_hidden_size or mc.ffn_hidden_size
        topk = max(1, getattr(mc, "moe_router_topk", 1) or 1)
        topk_tokens = tokens * topk
        # attention input (hidden) + routed expert input (hidden) + expert
        # down-projection input (moe_ffn), the latter two over topk-expanded tokens
        return (
            self._cast_ms(tokens, hidden)
            + self._cast_ms(topk_tokens, hidden)
            + self._cast_ms(topk_tokens, moe_ffn)
        )
