###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Operations of one refinement step, and what each costs on its own.

Both frontends -- the configuration-driven graph builder and the
``torch.compile`` trace -- reduce a model to a list of ``Op`` records, and the
step model prices that list. Keeping the record this small is deliberate: an op
is its arithmetic and the bytes it moves, so a traced op and a constructed one
are priced by exactly the same rule.

A GEMM op carries the ``(M, N, K, batch)`` problem it runs as and an attention
op its ``(batch, heads, q, kv, head_dim, kv_heads, causal)`` shape; with
Origami attached (the default) they are priced by InferaSim's GEMM and FAv3
attention models, exactly as the autoregressive projector prices its own. That
is also how a convolution, lowered to its implicit GEMM, gets the same kernel
model as a linear layer.

Everything else -- and everything, under the ``roofline`` backend -- costs
``max(flops / rate, bytes / bandwidth)`` plus one launch. The proposal's
blueprint divided a GEMM's *bytes* by the FLOP rate and summed every op's
bound; taking the larger of the two bounds per op is what a roofline is.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import DLLMHardware

GEMM = "gemm"
ATTENTION = "attention"
MEMORY = "memory"
KINDS = (GEMM, ATTENTION, MEMORY)
# Where an op sits in the model, which decides the compute a layer's
# collectives can overlap with: every layer's attention block, the dense or the
# MoE MLP, or once per step (embedding, head, sampling).
ATTN_BLOCK, DENSE_MLP, MOE_MLP, ONCE = "attn", "dense", "moe", "once"


@dataclass
class Op:
    """One kernel's work on one rank."""

    name: str
    kind: str
    flops: float = 0.0
    bytes_read: float = 0.0
    bytes_written: float = 0.0
    # Operand precision, which sets the matrix-core rate for GEMM and attention.
    dtype: str = "bf16"
    # How many times the op runs per step (once per layer, say).
    count: int = 1
    # The GEMM problem the op runs as, ``(M, N, K, batch)``; GEMM ops only.
    gemm: tuple[int, int, int, int] | None = None
    # ``(batch, heads, q, kv, head_dim, kv_heads, causal)``; attention ops only.
    attn: tuple | None = None
    # KV bytes per token on this rank, for the attention model's HBM bound.
    kv_bytes_per_token: float | None = None
    group: str = ONCE
    # Fixed cost per execution (us) in place of the hardware's kernel launch:
    # a framework that runs this op as a chain of small kernels pays each one.
    overhead_us: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"[dllm-sim] op {self.name!r}: unknown kind {self.kind!r}")
        if self.gemm is not None and self.kind != GEMM:
            raise ValueError(f"[dllm-sim] op {self.name!r}: only a GEMM has a GEMM problem")
        if self.attn is not None and self.kind != ATTENTION:
            raise ValueError(f"[dllm-sim] op {self.name!r}: only attention has an attention shape")

    @property
    def bytes_moved(self) -> float:
        return self.bytes_read + self.bytes_written

    def compute_ms(self, hw: DLLMHardware) -> float:
        if self.kind == MEMORY or self.flops <= 0:
            return 0.0
        return self.flops / hw.peak_flops(self.dtype) * 1e3

    def memory_ms(self, hw: DLLMHardware) -> float:
        return self.bytes_moved / hw.mem_bytes_per_s() * 1e3

    def time_ms(self, hw: DLLMHardware, *, launch: bool = True) -> float:
        """One execution: Origami's time for a GEMM or attention problem, else
        the binding roofline bound; plus its launch."""
        p = hw.pricer
        if p is not None and self.gemm is not None:
            t = p.gemm_ms(self.gemm, self.dtype)
        elif p is not None and self.attn is not None:
            t = p.attention_ms(self.attn, self.dtype, self.kv_bytes_per_token)
        else:
            t = max(self.compute_ms(hw), self.memory_ms(hw))
        if launch:
            t += (hw.kernel_launch_us if self.overhead_us is None else self.overhead_us) / 1e3
        return t

    def memory_bound(self, hw: DLLMHardware) -> bool:
        return self.memory_ms(hw) >= self.compute_ms(hw)


def total_ms(ops: list[Op], hw: DLLMHardware, *, launch: bool = True) -> float:
    return sum(op.time_ms(hw, launch=launch) * op.count for op in ops)


def memory_bound_fraction(ops: list[Op], hw: DLLMHardware) -> float:
    """Share of the ops' time spent in memory-bound kernels.

    Contention needs it: a collective running alongside steals HBM bandwidth
    from a memory-bound kernel and compute units from a compute-bound one, so
    which resource the step leans on decides how much it is slowed.
    """
    total = 0.0
    mem = 0.0
    for op in ops:
        t = op.time_ms(hw, launch=False) * op.count
        total += t
        if op.memory_bound(hw):
            mem += t
    return mem / total if total > 0 else 0.0


def group_ms(ops: list[Op], hw: DLLMHardware) -> dict[str, float]:
    """Time per model group (ms), launches included, every execution counted."""
    out = {g: 0.0 for g in (ATTN_BLOCK, DENSE_MLP, MOE_MLP, ONCE)}
    for op in ops:
        out[op.group] = out.get(op.group, 0.0) + op.time_ms(hw) * op.count
    return out


def summarize(ops: list[Op], hw: DLLMHardware) -> dict[str, float]:
    """Time per op kind (ms), launches included."""
    out = {k: 0.0 for k in KINDS}
    for op in ops:
        out[op.kind] += op.time_ms(hw) * op.count
    return out
