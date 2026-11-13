###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Configuration-driven frontend: a model's dimensions to one rank's ops.

A refinement step of a masked-diffusion language model runs the same
transformer an autoregressive model does. What differs is the shape it is run
at and what it is not allowed to skip:

* every position in the canvas is a query, not one token, so the GEMMs are
  prefill-shaped on every step, not only the first;
* attention is bidirectional, so a step pays the full ``q x kv`` product rather
  than the causal triangle;
* the LM head runs over every position the step predicts, which at a
  126k-entry vocabulary is one of the largest GEMMs in the step and the largest
  activation it writes.

The same dimensions describe the existing autoregressive presets, so any of
them can be loaded as a non-autoregressive model, and the diffusion presets in
``presets/`` use the same keys.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass, field, fields, replace
from typing import Any

from .hardware import dtype_bytes
from .ops import ATTENTION, ATTN_BLOCK, DENSE_MLP, GEMM, MEMORY, MOE_MLP, ONCE, Op
from .runtime import RuntimeConfig, apply_runtime

_PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")
# The vocabulary InferaSim assumes when a preset does not give one.
_DEFAULT_VOCAB = 100352


@dataclass
class DLLMModelSpec:
    """The dimensions a refinement step's cost depends on."""

    name: str = "custom"
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_query_groups: int = 0  # 0 = multi-head attention
    head_dim: int = 0  # 0 = hidden_size / num_attention_heads
    vocab_size: int = _DEFAULT_VOCAB
    gated_mlp: bool = True
    # Mixture of experts; ``num_experts == 0`` is a dense model.
    num_experts: int = 0
    moe_router_topk: int = 2
    moe_ffn_hidden_size: int = 0
    moe_shared_ffn_hidden_size: int = 0
    # Every ``moe_layer_freq``-th layer is MoE; the rest are dense.
    moe_layer_freq: int = 1
    # Busiest EP rank's expert tokens over the mean (1.0 = balanced).
    moe_imbalance: float = 1.0
    # Precisions: weights the GEMMs stream and compute at, the activations
    # between kernels, and the KV cache a cached mode keeps.
    weight_dtype: str = "bf16"
    act_dtype: str = "bf16"
    kv_cache_dtype: str = "bf16"
    # ``False`` for a continuous (flow-matching) head that projects to a latent
    # rather than to a vocabulary; ``head_size`` is then the latent width.
    vocab_head: bool = True
    head_size: int = 0

    def __post_init__(self) -> None:
        if self.num_query_groups <= 0:
            self.num_query_groups = self.num_attention_heads
        if self.head_dim <= 0:
            self.head_dim = self.hidden_size // max(1, self.num_attention_heads)
        if self.num_experts and self.moe_ffn_hidden_size <= 0:
            self.moe_ffn_hidden_size = self.ffn_hidden_size

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    def num_moe_layers(self) -> int:
        if not self.is_moe:
            return 0
        return self.num_layers // max(1, self.moe_layer_freq)

    def output_width(self) -> int:
        if self.vocab_head:
            return self.vocab_size
        return self.head_size or self.hidden_size

    def kv_bytes_per_token(self, tp: int = 1) -> float:
        """K and V for every layer, per token, on one TP rank."""
        kv_heads = max(1, self.num_query_groups // max(1, tp))
        return 2 * kv_heads * self.head_dim * dtype_bytes(self.kv_cache_dtype) * self.num_layers

    def weight_params(self) -> dict[str, float]:
        """Parameter counts by where they are sharded."""
        h, hd = self.hidden_size, self.head_dim
        attn = h * (self.num_attention_heads + 2 * self.num_query_groups) * hd
        attn += self.num_attention_heads * hd * h
        mlp_mult = 3 if self.gated_mlp else 2
        dense_mlp = mlp_mult * h * self.ffn_hidden_size
        n_moe = self.num_moe_layers()
        n_dense = self.num_layers - n_moe
        experts = n_moe * self.num_experts * mlp_mult * h * self.moe_ffn_hidden_size
        shared = n_moe * (mlp_mult * h * self.moe_shared_ffn_hidden_size + h * self.num_experts)
        return {
            "attention": self.num_layers * attn,
            "dense_mlp": n_dense * dense_mlp,
            "experts": experts,
            "shared": shared,
            "embedding": h * self.vocab_size,
            "head": h * self.output_width(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], name: str = "custom") -> DLLMModelSpec:
        """Read either this class's field names or the Megatron preset keys."""
        alias = {
            "padded_vocab_size": "vocab_size",
            "kv_channels": "head_dim",
            "swiglu": "gated_mlp",
            "moe_shared_expert_intermediate_size": "moe_shared_ffn_hidden_size",
        }
        known = {f.name for f in fields(cls)}
        out: dict[str, Any] = {"name": name}
        for key, value in d.items():
            key = alias.get(key, key)
            if key in known and value is not None:
                out[key] = value
        if not d.get("group_query_attention", True) and "num_query_groups" not in d:
            out.pop("num_query_groups", None)
        freq = out.get("moe_layer_freq")
        if isinstance(freq, str):
            out["moe_layer_freq"] = int(freq) if freq.strip().isdigit() else 1
        for key in ("num_experts", "moe_ffn_hidden_size", "moe_shared_ffn_hidden_size"):
            if out.get(key) is None:
                out.pop(key, None)
        return cls(**out)


def load_model(name: str, overrides: dict[str, Any] | None = None) -> DLLMModelSpec:
    """A dLLM preset, or an InferaSim (Megatron) preset, plus overrides."""
    from infera.projection.core.config.yaml_loader import parse_yaml

    path = os.path.join(_PRESET_DIR, f"{name}.yaml")
    if os.path.isfile(path):
        raw = parse_yaml(path)
    else:
        from infera.projection.core.config.preset_loader import PresetLoader

        try:
            raw = PresetLoader.load(name, framework="megatron")
        except FileNotFoundError as exc:
            raise ValueError(
                f"[dllm-sim] no model preset {name!r} in {_PRESET_DIR} or the Megatron presets"
            ) from exc
    raw = {**raw, **(overrides or {})}
    return DLLMModelSpec.from_dict(raw, name=name)


def dllm_presets() -> list[str]:
    if not os.path.isdir(_PRESET_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(_PRESET_DIR) if f.endswith(".yaml"))


@dataclass(frozen=True)
class SeqShape:
    """What one sequence asks of one step.

    ``q``: positions the step recomputes. ``kv``: positions they attend to,
    themselves included. ``logits``: positions the head predicts.
    """

    q: int
    kv: int
    logits: int

    def __post_init__(self) -> None:
        if self.q <= 0 or self.kv < self.q or self.logits < 0 or self.logits > self.q:
            raise ValueError(f"[dllm-sim] inconsistent sequence shape {self}")


@dataclass
class Parallel:
    tp: int = 1
    pp: int = 1
    ep: int = 1
    replicas: int = 1
    # Micro-batches a step is cut into across pipeline stages; 0 = min(batch, pp).
    pp_microbatches: int = 0

    def __post_init__(self) -> None:
        for name in ("tp", "pp", "ep", "replicas"):
            if getattr(self, name) < 1:
                raise ValueError(f"[dllm-sim] {name} must be >= 1")
        # Experts are spread inside the tensor-parallel group, as the serving
        # engines do without attention-DP; a wider EP would need DP attention.
        if self.tp % self.ep:
            raise ValueError("[dllm-sim] ep must divide tp")

    @property
    def gpus_per_replica(self) -> int:
        return self.tp * self.pp

    @property
    def gpus(self) -> int:
        return self.gpus_per_replica * self.replicas

    def microbatches(self, batch: int) -> int:
        if self.pp <= 1:
            return 1
        m = self.pp_microbatches or self.pp
        return max(1, min(m, batch))


@dataclass
class StageGraph:
    """One rank's ops for one pipeline stage, and what its collectives carry.

    The collectives are not listed op by op: the comm model prices them per
    layer type from ``tokens``, ``n_dense`` and ``n_moe`` with the same rules
    the autoregressive projector applies to a forward.
    """

    ops: list[Op] = field(default_factory=list)
    tokens: int = 0
    n_dense: int = 0
    n_moe: int = 0

    @property
    def layers(self) -> int:
        return self.n_dense + self.n_moe


def _gemm(
    name: str,
    m: float,
    n: float,
    k: float,
    w_dtype: str,
    a_dtype: str,
    count: int,
    group: str,
) -> Op:
    wb, ab = dtype_bytes(w_dtype), dtype_bytes(a_dtype)
    return Op(
        name,
        GEMM,
        flops=2.0 * m * n * k,
        bytes_read=n * k * wb + m * k * ab,
        bytes_written=m * n * 2.0,
        dtype=w_dtype,
        count=count,
        gemm=(math.ceil(m), math.ceil(n), math.ceil(k), 1) if m > 0 else None,
        group=group,
    )


def _elementwise(
    name: str, elems_in: float, elems_out: float, count: int, group: str, eb: float = 2.0
) -> Op:
    return Op(
        name,
        MEMORY,
        bytes_read=elems_in * eb,
        bytes_written=elems_out * eb,
        count=count,
        group=group,
    )


def _expected_experts_touched(experts: int, topk: int, tokens: float) -> float:
    """Distinct experts ``tokens`` independent top-k choices reach."""
    if experts <= 0 or tokens <= 0:
        return 0.0
    return experts * (1.0 - (1.0 - min(1.0, topk / experts)) ** tokens)


def build_stage_graph(
    spec: DLLMModelSpec,
    seqs: list[SeqShape],
    par: Parallel,
    *,
    first: bool,
    last: bool,
    runtime: RuntimeConfig | None = None,
) -> StageGraph:
    """One rank's ops for one pipeline stage of one refinement step, as the
    ``runtime`` (default: a serving engine) launches them."""
    tp = par.tp
    T = float(sum(s.q for s in seqs))
    logits = float(sum(s.logits for s in seqs))
    h, hd = spec.hidden_size, spec.head_dim
    heads = max(1, spec.num_attention_heads // tp)
    kv_heads = max(1, spec.num_query_groups // tp)
    wd, ad = spec.weight_dtype, spec.act_dtype
    quantized = dtype_bytes(wd) < dtype_bytes(ad)

    n_layers = math.ceil(spec.num_layers / par.pp)
    n_moe = round(spec.num_moe_layers() * n_layers / spec.num_layers)
    n_dense = n_layers - n_moe
    g = StageGraph(tokens=int(T), n_dense=n_dense, n_moe=n_moe)
    L = n_layers

    def gemm(name: str, m: float, n: float, k: float, count: int, group: str) -> Op:
        # A low-precision GEMM needs its activation cast first: a memory-bound
        # pass over the input, which is where W8A8 pays for its faster core.
        if quantized:
            g.ops.append(
                Op(
                    f"{name}_act_quant",
                    MEMORY,
                    bytes_read=m * k * dtype_bytes(ad),
                    bytes_written=m * k * dtype_bytes(wd),
                    count=count,
                    group=group,
                )
            )
        op = _gemm(name, m, n, k, wd, wd if quantized else ad, count, group)
        g.ops.append(op)
        return op

    if first:
        g.ops.append(_elementwise("embedding", T * h, T * h, 1, ONCE))

    # Attention, every layer: one kernel per distinct sequence shape, which is
    # what a varlen flash kernel's cost depends on.
    A = ATTN_BLOCK
    g.ops.append(_elementwise("attn_norm", T * h, T * h, L, A))
    gemm("qkv_proj", T, (heads + 2 * kv_heads) * hd, h, L, A)
    g.ops.append(
        _elementwise("rope", T * (heads + kv_heads) * hd, T * (heads + kv_heads) * hd, L, A)
    )
    kv_bpe = dtype_bytes(spec.kv_cache_dtype)
    kv_bpt = kv_heads * hd * 2 * kv_bpe
    for (q, kv), n in sorted(Counter((s.q, s.kv) for s in seqs).items()):
        g.ops.append(
            Op(
                "attention" if len(seqs) == n else f"attention[q={q},kv={kv}]",
                ATTENTION,
                flops=4.0 * n * q * kv * heads * hd,
                bytes_read=n * (q * heads * hd * 2 + kv * kv_bpt),
                bytes_written=n * q * heads * hd * 2,
                dtype="bf16",
                count=L,
                attn=(n, heads, q, kv, hd, kv_heads, False),
                kv_bytes_per_token=kv_bpt,
                group=A,
            )
        )
    gemm("o_proj", T, h, heads * hd, L, A)
    g.ops.append(_elementwise("attn_residual", 2 * T * h, T * h, L, A))
    g.ops.append(_elementwise("mlp_norm", T * h, T * h, L, A))
    g.ops.append(_elementwise("mlp_residual", 2 * T * h, T * h, L, A))

    mlp_mult = 2 if spec.gated_mlp else 1
    if n_dense:
        D = DENSE_MLP
        ffn = spec.ffn_hidden_size / tp
        gemm("up_gate_proj", T, mlp_mult * ffn, h, n_dense, D)
        g.ops.append(_elementwise("mlp_act", mlp_mult * T * ffn, T * ffn, n_dense, D))
        gemm("down_proj", T, h, ffn, n_dense, D)
    if n_moe:
        M = MOE_MLP
        ep = par.ep
        expert_tp = max(1, tp // ep)
        local = spec.num_experts / ep
        routed = T * spec.moe_router_topk / ep * spec.moe_imbalance
        moe_ffn = spec.moe_ffn_hidden_size / expert_tp
        touched = _expected_experts_touched(spec.num_experts, spec.moe_router_topk, T) / ep
        touched = min(local, touched)
        g.ops.append(_gemm("router", T, spec.num_experts, h, "bf16", ad, n_moe, M))
        for name, n_out, k_in in (
            ("expert_up_gate", mlp_mult * moe_ffn, h),
            ("expert_down", h, moe_ffn),
        ):
            op = gemm(name, routed, n_out, k_in, n_moe, M)
            # Grouped GEMM: only the experts some token reached are streamed.
            op.bytes_read = touched * n_out * k_in * dtype_bytes(wd) + routed * k_in * dtype_bytes(
                wd if quantized else ad
            )
            # As a batch of the touched experts at the mean tokens per expert,
            # the approximation InferaSim's MoE profiler hands Origami too.
            if touched >= 1:
                op.gemm = (
                    math.ceil(routed / touched),
                    math.ceil(n_out),
                    math.ceil(k_in),
                    math.ceil(touched),
                )
            else:
                op.gemm = None
        g.ops.append(
            _elementwise("expert_act", mlp_mult * routed * moe_ffn, routed * moe_ffn, n_moe, M)
        )
        g.ops.append(_elementwise("moe_permute", 2 * routed * h, 2 * routed * h, n_moe, M))
        if spec.moe_shared_ffn_hidden_size:
            sffn = spec.moe_shared_ffn_hidden_size / tp
            gemm("shared_up_gate", T, mlp_mult * sffn, h, n_moe, M)
            gemm("shared_down", T, h, sffn, n_moe, M)

    if last:
        g.ops.append(_elementwise("final_norm", T * h, T * h, 1, ONCE))
        if logits > 0:
            out = spec.output_width() / (tp if spec.vocab_head else 1)
            g.ops.append(_gemm("lm_head", logits, out, h, "bf16", ad, 1, ONCE))
            if spec.vocab_head:
                # Softmax confidence and the argmax the remasking rule ranks by:
                # two passes over the logits.
                g.ops.append(_elementwise("unmask_select", 2 * logits * out, logits, 1, ONCE))
    if runtime is not None:
        moe = {"experts": local, "touched": touched} if n_moe else {}
        g.ops = apply_runtime(g.ops, runtime, heads=heads, kv_heads=kv_heads, layers=L, **moe)
    return g


def with_dtypes(spec: DLLMModelSpec, **dtypes: str) -> DLLMModelSpec:
    return replace(spec, **{k: v for k, v in dtypes.items() if v})
