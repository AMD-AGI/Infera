"""Sampler vocabulary softmax — the kernel under optimization.

Traced hot kernel from an sglang v0.5.14 decode step on AMD MI300X (gfx942),
model Qwen/Qwen3-0.6B. sglang's sampler does, per decode step:

    # python/sglang/srt/layers/sampler.py:183
    logits[:] = torch.softmax(logits, dim=-1)

with ``logits`` of shape ``[running_requests, vocab_size]`` in float32. With 8
concurrent requests that is ``[8, 151936]`` fp32, and ATen dispatches
``cunn_SoftMaxForwardGmem<4, float, float, float, ...>`` — the global-memory
fallback used when the reduced dimension is too large for the shared-memory
path.

Measured baseline (torch.profiler, DECODE stage, batch 8):
    55.59 us/call, 14.50% of all decode GPU time.
    Traffic 8*151936*4 B read + same written = 9.72 MB -> ~175 GB/s achieved,
    against ~5.3 TB/s HBM peak on MI300X. Memory-bound with large headroom.

This seed is the measured baseline itself, so forge-loop's auto-measured
baseline equals the production number it must beat.
"""

from __future__ import annotations

import torch


def sampler_softmax(logits: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the vocabulary dimension.

    Args:
        logits: ``[batch, vocab]`` float32 tensor of raw logits.
        out:    ``[batch, vocab]`` float32 tensor receiving the probabilities.
                Pre-allocated by the caller so the benchmark can be graph-captured.

    Returns:
        ``out``.
    """
    torch.softmax(logits, dim=-1, out=out)
    return out
