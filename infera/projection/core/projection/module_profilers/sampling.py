###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Token-sampling / logits post-processing cost model (inference-only).

After the LM head produces logits ``[sampled_tokens, vocab]`` the serving
engine turns them into the next token(s).  This is a memory-bandwidth-bound
reduction over the vocabulary dimension that the GEMM/SDPA simulators do not
cover, and it is *not* part of training (which computes a cross-entropy loss
instead).  We model it explicitly so a decode step's cost includes the
sampling tail, which is non-negligible for large-vocab models at low batch.

The op is priced as ``n_passes`` streaming reads/writes of the per-rank logits
tensor at a fraction of peak HBM bandwidth:

  * **greedy / argmax** — read logits + max-reduce ≈ 2 passes.
  * **temperature scaling** — one extra fused scale pass when ``temperature``
    differs from 1.0.
  * **top-k / top-p (nucleus)** — a partial sort / threshold + renormalize
    pass on top of the softmax read ≈ +1 pass.

Logits are read in the LM-head output dtype (bf16, 2 bytes); the softmax /
reduction accumulates in registers, so only the vocab read dominates.
"""

from __future__ import annotations

# Sampling streams the (contiguous) logits row per token, so it achieves the
# same sequential-streaming HBM efficiency as other element-wise ops.
_SAMPLING_BW_FRACTION = 0.566
_FALLBACK_HBM_BW_GBPS = 5300.0  # MI300X default when the backend can't report BW
_LOGITS_BYTES_PER_EL = 2  # bf16 logits out of the LM head


class SamplingProfiler:
    """Analytical cost of turning logits into sampled tokens (forward-only)."""

    def __init__(
        self,
        vocab_size: int,
        *,
        hbm_bandwidth_gbps: float | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        self.vocab_size = max(1, int(vocab_size))
        self._hbm_bw = float(hbm_bandwidth_gbps or _FALLBACK_HBM_BW_GBPS)
        self.top_k = max(0, int(top_k))
        self.top_p = float(top_p)
        self.temperature = float(temperature)

    def _num_passes(self) -> int:
        # Baseline greedy/argmax: read logits + reduce.
        passes = 2
        if self.temperature not in (0.0, 1.0):
            passes += 1  # fused temperature scale
        # Nucleus / top-k need a partial sort/threshold + renormalize pass.
        if self.top_k > 0 or (0.0 < self.top_p < 1.0):
            passes += 1
        return passes

    def forward_time_ms(self, num_sampled_tokens: int) -> float:
        """Sampling latency for ``num_sampled_tokens`` sampled this forward.

        ``num_sampled_tokens`` is ``batch * tokens_per_sequence`` (1 per
        sequence for prefill/decode; ``q_len`` for speculative verification).
        """
        tokens = max(1, int(num_sampled_tokens))
        eff_bw = self._hbm_bw * _SAMPLING_BW_FRACTION  # GB/s
        tensor_bytes = tokens * self.vocab_size * _LOGITS_BYTES_PER_EL
        return self._num_passes() * tensor_bytes / (eff_bw * 1e6)
