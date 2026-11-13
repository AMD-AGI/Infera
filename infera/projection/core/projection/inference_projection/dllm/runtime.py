###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""How the framework running a refinement step turns the model into kernels.

The op graph describes the arithmetic a step must do. The same arithmetic
costs very different time depending on what runs it, and for a dLLM today the
difference is the size of the effect being modelled: a serving engine fuses
q/k/v into one GEMM and gate/up into another, runs each norm and RoPE as one
kernel, and samples with a fused kernel over the logits; the Hugging Face
reference implementations every dLLM ships as run each projection
separately, compute norms and RoPE in FP32 as chains of element-wise kernels,
and sample with an FP32 softmax, a max and a top-k.

``engine`` is the first; ``hf_eager`` is the second, calibrated on MI325X
from a kernel census of LLaDA-8B (``manual/simulation/dllm.md``):

* ``fused_projections=False``: q, k, v, gate and up are separate GEMMs.
* ``elementwise_traffic``: HBM bytes the element-wise chain moves per byte the
  fused op would, from the slope of element-wise time against tokens.
* ``elementwise_kernels_per_layer`` and ``kernel_floor_us``: the kernels that
  chain launches per layer, and what each small one costs regardless of size
  (the intercept of the same fit).
* ``sampler="eager"``: a bf16-to-fp32 cast, softmax and max over the logits,
  18 bytes per logit, instead of two bf16 passes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .ops import GEMM, MEMORY, Op

SAMPLERS = ("fused", "eager")
# Bytes per logit. The graph's sampler is two bf16 passes (confidence, then
# argmax); the eager one reads the bf16 logits, writes and reads an fp32 copy,
# writes and reads the softmax and reads it again for the max.
_SAMPLER_BYTES = {"fused": 4.0, "eager": 2.0 + 4.0 + 4.0 + 4.0 + 4.0}
_SAMPLER_KERNELS = 5
# Ops the element-wise chain multiplies; the sampler, casts and the embedding
# gather are priced on their own terms.
_CHAIN = {
    "attn_norm",
    "rope",
    "attn_residual",
    "mlp_norm",
    "mlp_residual",
    "mlp_act",
    "expert_act",
    "final_norm",
}


@dataclass(frozen=True)
class RuntimeConfig:
    name: str = "engine"
    fused_projections: bool = True
    elementwise_traffic: float = 1.0
    # 0 = one kernel per element-wise op in the graph.
    elementwise_kernels_per_layer: float = 0.0
    # Per-kernel cost of the element-wise chain; None = the hardware's launch.
    kernel_floor_us: float | None = None
    sampler: str = "fused"

    def __post_init__(self) -> None:
        if self.sampler not in SAMPLERS:
            raise ValueError(f"[dllm-sim] sampler must be one of {SAMPLERS}")
        if self.elementwise_traffic <= 0:
            raise ValueError("[dllm-sim] elementwise_traffic must be > 0")


RUNTIMES = {
    "engine": RuntimeConfig(),
    "hf_eager": RuntimeConfig(
        name="hf_eager",
        fused_projections=False,
        elementwise_traffic=4.85,
        elementwise_kernels_per_layer=34.3,
        kernel_floor_us=4.4,
        sampler="eager",
    ),
}


def get_runtime(r: str | RuntimeConfig | dict | None) -> RuntimeConfig:
    if r is None:
        return RUNTIMES["engine"]
    if isinstance(r, RuntimeConfig):
        return r
    if isinstance(r, dict):
        d = dict(r)
        base = RUNTIMES.get(str(d.pop("preset", d.get("name", "engine"))), RuntimeConfig())
        return replace(base, **d)
    if r not in RUNTIMES:
        raise ValueError(f"[dllm-sim] unknown runtime {r!r}; known: {', '.join(RUNTIMES)}")
    return RUNTIMES[r]


def _split(op: Op, parts: list[tuple[str, float]]) -> list[Op]:
    """``op`` as separate GEMMs along N, in the given shares of its width."""
    m, n, k, b = op.gemm
    out = []
    for name, share in parts:
        ni = max(1, round(n * share))
        f = ni / n
        out.append(
            replace(
                op,
                name=name,
                flops=op.flops * f,
                # Each reads the whole activation and its slice of the weight.
                bytes_read=(op.bytes_read - m * k * 2.0) * f + m * k * 2.0,
                bytes_written=op.bytes_written * f,
                gemm=(m, ni, k, b),
            )
        )
    return out


def apply_runtime(
    ops: list[Op], rt: RuntimeConfig, *, heads: int, kv_heads: int, layers: int
) -> list[Op]:
    if rt == RUNTIMES["engine"]:
        return ops
    out: list[Op] = []
    for op in ops:
        if not rt.fused_projections and op.kind == GEMM and op.gemm is not None:
            if op.name == "qkv_proj":
                tot = heads + 2 * kv_heads
                out += _split(
                    op,
                    [
                        ("q_proj", heads / tot),
                        ("k_proj", kv_heads / tot),
                        ("v_proj", kv_heads / tot),
                    ],
                )
                continue
            if op.name in ("up_gate_proj", "shared_up_gate"):
                out += _split(op, [(op.name + ".gate", 0.5), (op.name + ".up", 0.5)])
                continue
        out.append(op)
    chain = [o for o in out if o.kind == MEMORY and o.name in _CHAIN]
    per_layer = sum(o.count for o in chain) / max(1, layers)
    kernels = (
        rt.elementwise_kernels_per_layer / per_layer
        if rt.elementwise_kernels_per_layer and per_layer
        else 1.0
    )
    for o in chain:
        o.bytes_read *= rt.elementwise_traffic
        o.bytes_written *= rt.elementwise_traffic
        if rt.kernel_floor_us is not None:
            o.overhead_us = rt.kernel_floor_us * kernels
    for o in out:
        if o.name == "unmask_select":
            o.bytes_read *= _SAMPLER_BYTES[rt.sampler] / _SAMPLER_BYTES["fused"]
            if rt.sampler == "eager" and rt.kernel_floor_us is not None:
                o.overhead_us = rt.kernel_floor_us * _SAMPLER_KERNELS
    return out
