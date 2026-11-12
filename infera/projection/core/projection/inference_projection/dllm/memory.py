###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Whether a batch of refinement steps fits on a rank.

Three things occupy HBM. Weights, sharded over the replica's ``tp x pp``
ranks. The KV cache, which only a cached decoder keeps, for the whole canvas
of every running request. And the step's activations -- for an
autoregressive decode a rounding error, for a refinement step the largest
term: the head writes ``positions x vocabulary`` logits, and the released
samplers upcast them to FP32 for the softmax and the Gumbel noise before
ranking confidences, so a 1,024-position canvas at a 126k vocabulary is
~0.5 GB per sequence before a single weight is counted.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import DLLMHardware, dtype_bytes
from .model import DLLMModelSpec, Parallel
from .workload import DLLMWorkload

# HBM the runtime, the allocator's slack and the communication buffers keep.
_USABLE_FRACTION = 0.90


@dataclass
class MemoryReport:
    weights_gb: float
    kv_cache_gb: float
    activations_gb: float
    capacity_gb: float

    @property
    def total_gb(self) -> float:
        return self.weights_gb + self.kv_cache_gb + self.activations_gb

    @property
    def fits(self) -> bool:
        return self.total_gb <= self.capacity_gb

    def as_dict(self) -> dict[str, float]:
        return {
            "weights_gb": self.weights_gb,
            "kv_cache_gb": self.kv_cache_gb,
            "activations_gb": self.activations_gb,
            "total_gb": self.total_gb,
            "capacity_gb": self.capacity_gb,
            "fits": float(self.fits),
        }


def weights_bytes_per_rank(spec: DLLMModelSpec, par: Parallel) -> float:
    p = spec.weight_params()
    wb = dtype_bytes(spec.weight_dtype)
    shard = par.tp * par.pp
    body = (p["attention"] + p["dense_mlp"] + p["experts"] + p["shared"]) * wb / shard
    # Embedding and head stay BF16 and live on the first and last stages, split
    # over the tensor-parallel group; the busiest rank is the one holding both
    # when there is one stage.
    ends = (p["embedding"] + p["head"]) * 2.0 / par.tp
    if par.pp > 1:
        ends = max(p["embedding"], p["head"]) * 2.0 / par.tp
    return body + ends


def kv_bytes_per_request(spec: DLLMModelSpec, par: Parallel, wl: DLLMWorkload) -> float:
    per_token = spec.kv_bytes_per_token(par.tp) / par.pp
    return wl.kv_cached_tokens() * per_token * wl.seqs_per_request


def activation_bytes(spec: DLLMModelSpec, par: Parallel, wl: DLLMWorkload, batch: int) -> float:
    """Peak transient memory of the costliest step.

    The peak is whichever is larger: the logits (BF16 out of the head, the
    FP32 copy the sampler works on and the FP32 probabilities it ranks), or the
    widest layer intermediate (the gated MLP's up projection) alongside the
    residual stream.
    """
    seqs = batch * wl.seqs_per_request
    steps = wl.schedule()
    q = max(s.shape.q for s in steps)
    logits = max(s.shape.logits for s in steps)
    tp = par.tp
    ab = dtype_bytes(spec.act_dtype)
    out_w = spec.output_width() / (tp if spec.vocab_head else 1)
    head = seqs * logits * out_w * (2.0 + 4.0 + 4.0)
    mlp_mult = 2 if spec.gated_mlp else 1
    ffn = max(spec.ffn_hidden_size, spec.moe_ffn_hidden_size * max(1, spec.moe_router_topk))
    layer = seqs * q * (mlp_mult * ffn / tp + 3 * spec.hidden_size) * ab
    return max(head, layer)


def memory_report(
    spec: DLLMModelSpec, par: Parallel, hw: DLLMHardware, wl: DLLMWorkload, batch: int
) -> MemoryReport:
    gb = 1e9
    return MemoryReport(
        weights_gb=weights_bytes_per_rank(spec, par) / gb,
        kv_cache_gb=kv_bytes_per_request(spec, par, wl) * batch / gb,
        activations_gb=activation_bytes(spec, par, wl, batch) / gb,
        capacity_gb=hw.hbm_gb * _USABLE_FRACTION,
    )


def max_batch(
    spec: DLLMModelSpec, par: Parallel, hw: DLLMHardware, wl: DLLMWorkload, limit: int = 4096
) -> int:
    """The largest batch of concurrent requests a replica's ranks can hold."""
    if not memory_report(spec, par, hw, wl, 1).fits:
        return 0
    lo, hi = 1, limit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if memory_report(spec, par, hw, wl, mid).fits:
            lo = mid
        else:
            hi = mid - 1
    return lo
