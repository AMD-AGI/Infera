###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Every kernel a replayed trace runs, with the shapes the kernels see.

:mod:`gemm_shapes` reports the linear layers. This extends the same recording
to the rest of a forward pass, so a serving stack can tune or pre-select every
kernel for the shapes a real trace produces:

* ``gemm``: the linear layers, exactly as :func:`gemm_shapes.step_gemms`
  reports them.
* ``attention``: prefill (extend) and decode attention, one row per step
  composition: sequences, query tokens, the longest and mean KV span, heads and
  head dims on one rank, the KV-cache dtype and the sliding window. Windowed
  layers are their own rows; their prefill spans are the sequences' own (the
  kernel masks outside the window), and their decode spans are capped at the
  window only when the engine keeps just the window's blocks
  (``sink_window_decode_capped``, SGLang's hybrid SWA pool). MLA prefill is the
  per-head attention it runs and MLA decode the absorbed form over the latent.
  DeepSeek-V4's layers attend over their local window plus a compressed pool
  (``pool_kv`` entries, all of them on HCA layers, the indexer's top-k on CSA
  ones). A top-k sparse model's attention carries its ``topk``. Split-KV
  decode (every engine's GQA decode past :data:`DECODE_PARTITION` tokens) adds
  the kernel that merges its partitions.
* ``indexer``: the sparse indexer's scoring over its index cache, its top-k
  selection and the index-cache write.
* ``linear_attention``: GDN / KDA chunked prefill, recurrent decode, causal
  conv1d and the gated output norm.
* ``moe``: the router's top-k, the token-to-expert alignment, the activation
  between the expert GEMMs and the top-k reduction (and a latent MoE's norm).
  Where the engine pads expert weights (:func:`gemm_shapes.moe_pad`), the expert
  GEMMs run at the padded sizes and the input's pad is its own kernel.
* ``norm``, ``elementwise``: RMSNorm (fused residual add), QK-norm, MLA's
  latent norms, the MLP activation, RoPE, the KV-cache write, the embedding,
  DeepSeek-V4's mHC Sinkhorn and Kimi-K3's attention-residual mix.
* ``quant``: the activation cast in front of each low-precision GEMM, when the
  deployment quantizes activations.
* ``collective``: the messages a step sends, in bytes, over the group that
  sends them. These follow the communication model in :mod:`collectives`.
* ``sampling``: the sampler over the full vocabulary, for each sampled row.

Counts are per rank. Under pipeline parallelism each row names its ``stage``
and counts only that stage's layers; the embedding is on the first stage and
the LM head and sampler on the last. A speculating replay also reports its
drafter's forward passes (pool ``<pool>/draft``), priced the way the
projection prices them.

Every row carries a time estimate and its source: Origami for GEMMs, the SDPA
simulator for attention, the collective model for communication, and a
roofline over the GPU's HBM bandwidth and peak FLOPs for the rest.

Spans are bucketed (KV to :data:`KV_BUCKET` tokens, prefill query lengths to
:data:`Q_BUCKET`) so a trace gives a tunable set rather than one row per step;
every other dimension is exact.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import json
import math
from collections import Counter
from dataclasses import dataclass

from . import gemm_shapes
from . import layer_plan as lp

KV_BUCKET = 256
Q_BUCKET = 64
_ACT_BYTES = 2
# Achievable fractions of peak for a roofline estimate, as the profilers use.
_BW_FRACTION = 0.566
_COMPUTE_EFF = 0.7
_SAMPLING_PASSES = 2
_LINEAR_CHUNK = 64


@dataclass(frozen=True)
class Kernel:
    kind: str
    op: str
    dtype: str
    dims: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Seqs:
    """One phase of one step: its sequences, bucketed."""

    n: int
    q: int  # query tokens across the sequences
    max_q: int
    max_kv: int  # longest span attended, this step's tokens included
    mean_kv: int


def _up(x: float, b: int) -> int:
    return max(b, int(math.ceil(x / b)) * b)


def seqs_of(pairs, window: int = 0) -> Seqs | None:
    """``pairs`` are ``(query tokens, KV already computed)`` per sequence."""
    if not pairs:
        return None
    spans = [kv + q for q, kv in pairs]
    if window:
        spans = [min(s, window) for s in spans]
    qs = [q for q, _ in pairs]
    return Seqs(
        n=len(pairs),
        q=sum(qs),
        max_q=max(qs),
        max_kv=_up(max(spans), KV_BUCKET),
        mean_kv=_up(sum(spans) / len(spans), KV_BUCKET),
    )


@dataclass(frozen=True)
class Step:
    tokens: int
    sampled_rows: int
    decode: Seqs | None = None
    prefill: Seqs | None = None
    decode_window: Seqs | None = None


def _bucket_q(s: Seqs | None) -> Seqs | None:
    if s is None:
        return None
    return Seqs(s.n, _up(s.q, Q_BUCKET), _up(s.max_q, Q_BUCKET), s.max_kv, s.mean_kv)


def make_step(cfg, tokens, sampled_rows, decode=None, prefill=None, q_len=1) -> Step:
    """A step from its per-sequence ``(query tokens, KV computed)`` pairs."""
    dec = [(q_len, kv) for kv in decode] if decode else []
    pre = list(prefill) if prefill else []
    w, frac = lp.window_of(cfg)
    capped = w and frac > 0 and getattr(cfg.model_config, "sink_window_decode_capped", False)
    return Step(
        int(tokens),
        int(sampled_rows),
        seqs_of(dec),
        _bucket_q(seqs_of(pre)),
        seqs_of(dec, w) if capped and dec else None,
    )


def _k(kind: str, op: str, dtype: str, **dims) -> Kernel:
    return Kernel(kind, op, dtype, tuple(dims.items()))


def _ranks(mp) -> tuple[int, int, int, int, int]:
    """``(tp, attention tp, attention dp, ep, dcp)``."""
    tp = max(1, int(mp.tensor_model_parallel_size or 1))
    dp = max(1, int(getattr(mp, "attention_data_parallel_size", 1) or 1))
    atp = max(1, tp // dp) if dp > 1 else tp
    ep = max(1, int(getattr(mp, "expert_model_parallel_size", 1) or 1))
    dcp = max(1, int(getattr(mp, "decode_context_parallel_size", 1) or 1))
    return tp, atp, dp, ep, dcp


def _per_rank(s: Seqs, dp: int) -> Seqs:
    """Attention-DP: a rank attends over its share of the step's sequences."""
    if dp <= 1:
        return s
    n = max(1, math.ceil(s.n / dp))
    return Seqs(n, max(1, math.ceil(s.q / dp)), s.max_q, s.max_kv, s.mean_kv)


def _kv_dtype(cfg) -> str:
    return str(
        getattr(getattr(cfg, "request_config", None), "kv_cache_dtype", None) or "bf16"
    ).lower()


def _sparse_topk(cfg) -> int:
    req = getattr(cfg, "request_config", None)
    return int(getattr(req, "sparse_attention_topk", 0) or 0) or int(
        getattr(cfg.model_config, "index_topk", 0) or 0
    )


def _geometry(cfg) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """``(prefill, decode)`` attention geometry on a rank: heads, KV heads, QK and V dims."""
    mc = cfg.model_config
    _, atp, _, _, _ = _ranks(cfg.model_parallel_config)
    heads = max(1, int(mc.num_attention_heads) // atp)
    if getattr(mc, "multi_latent_attention", False):
        nope, rope, latent = int(mc.qk_head_dim), int(mc.qk_pos_emb_head_dim), int(mc.kv_lora_rank)
        return (heads, heads, nope + rope, int(mc.v_head_dim)), (heads, 1, latent + rope, latent)
    d = int(mc.kv_channels)
    groups = int(mc.num_query_groups) if mc.group_query_attention and mc.num_query_groups else 0
    kv_heads = max(1, (groups or int(mc.num_attention_heads)) // atp)
    return (heads, kv_heads, d, d), (heads, kv_heads, d, d)


def _attention(cfg, step: Step, c: lp.Counts) -> list[tuple[Kernel, int]]:
    mc = cfg.model_config
    _, _, dp, _, dcp = _ranks(cfg.model_parallel_config)
    kv_dt = _kv_dtype(cfg)
    pre_geo, dec_geo = _geometry(cfg)
    out: list[tuple[Kernel, int]] = []

    def row(op, s, geo, layers, **extra):
        h, hk, dqk, dv = geo
        dims = dict(batch=s.n, q=s.q if op == "attn_prefill" else s.q // max(1, s.n))
        if op == "attn_prefill":
            dims["max_q"] = s.max_q
        dims.update(max_kv=s.max_kv, mean_kv=s.mean_kv, heads=h, kv_heads=hk, qk_dim=dqk, v_dim=dv)
        dims.update(extra)
        out.append((_k("attention", op, kv_dt, **dims), layers))

    if gemm_shapes.is_v4(mc):
        w = int(getattr(mc, "attn_sliding_window", 0) or 0)
        topk = int(getattr(mc, "index_topk", 0) or 0)
        for kind in (lp.V4_SWA, lp.V4_CSA, lp.V4_HCA):
            layers = c.n(kind)
            if not layers:
                continue
            cr = lp.V4_RATIO[kind]
            for op, s, geo in (
                ("attn_prefill", step.prefill, pre_geo),
                ("attn_decode", step.decode, dec_geo),
            ):
                if s is None:
                    continue
                s = _per_rank(s, dp)
                local = Seqs(
                    s.n,
                    s.q,
                    s.max_q,
                    min(s.max_kv, _up(w, KV_BUCKET)) if w else s.max_kv,
                    min(s.mean_kv, _up(w, KV_BUCKET)) if w else s.mean_kv,
                )
                pool = math.ceil(s.max_kv / cr) if cr else 0
                mean_pool = math.ceil(s.mean_kv / cr) if cr else 0
                if kind == lp.V4_CSA and topk:
                    pool, mean_pool = min(pool, topk), min(mean_pool, topk)
                row(
                    op,
                    local,
                    geo,
                    layers,
                    window=w,
                    compress=cr,
                    pool_kv=pool,
                    mean_pool_kv=mean_pool,
                )
        return out

    w, _ = lp.window_of(cfg)
    topk = _sparse_topk(cfg) if int(getattr(mc, "sparse_index_n_heads", 0) or 0) else 0
    extra = {"topk": topk} if topk else {}
    part = 0 if getattr(mc, "multi_latent_attention", False) else decode_partition(cfg)
    for layers, window, dec, pre in (
        (c.n(lp.FULL), 0, step.decode, step.prefill),
        (c.n(lp.WINDOW), w, step.decode_window or step.decode, step.prefill),
    ):
        if not layers:
            continue
        if pre is not None:
            row("attn_prefill", _per_rank(pre, dp), pre_geo, layers, window=window, **extra)
        if dec is not None:
            s = _per_rank(dec, dp)
            h, hk, dqk, dv = dec_geo
            if dcp > 1:
                # A rank attends over its slice of every context, for the heads
                # its DCP group gathered.
                s = Seqs(
                    s.n,
                    s.q,
                    s.max_q,
                    _up(s.max_kv / dcp, KV_BUCKET),
                    _up(s.mean_kv / dcp, KV_BUCKET),
                )
                h = min(int(mc.num_attention_heads), h * dcp)
            row("attn_decode", s, (h, hk, dqk, dv), layers, window=window, **extra)
            parts = math.ceil(s.max_kv / part) if part else 0
            if parts > 1:
                out.append(
                    (
                        _k(
                            "attention",
                            "attn_decode_reduce",
                            "bf16",
                            batch=s.n * max(1, s.q // max(1, s.n)),
                            heads=h,
                            partitions=parts,
                            v_dim=dv,
                        ),
                        layers,
                    )
                )
    return out


# Split-KV decode attention (vLLM's paged attention, the Triton decode kernels)
# splits each sequence's KV into partitions of this many tokens and merges the
# partial outputs in a second kernel.
DECODE_PARTITION = 256


def decode_partition(cfg) -> int:
    """KV tokens per split-KV decode partition; ``attn_decode_partition=0`` for a one-pass kernel."""
    v = getattr(getattr(cfg, "request_config", None), "attn_decode_partition", None)
    return DECODE_PARTITION if v is None else max(0, int(v))


def _indexer(cfg, step: Step, c: lp.Counts) -> list[tuple[Kernel, int]]:
    """Scoring over the index cache, the top-k pick and the index-cache write."""
    mc = cfg.model_config
    _, _, dp, _, _ = _ranks(cfg.model_parallel_config)
    if gemm_shapes.is_v4(mc):
        layers = c.n(lp.V4_CSA)
        heads, dim = int(mc.index_n_heads or 0), int(mc.index_head_dim or 0)
        topk, ratio = int(mc.index_topk or 0), 4
    else:
        layers = c.indexed
        heads = int(getattr(mc, "sparse_index_n_heads", 0) or 0)
        dim = int(getattr(mc, "sparse_index_head_dim", 0) or 0)
        topk, ratio = _sparse_topk(cfg), 1
    if not (layers and heads and dim):
        return []
    dt = str(getattr(mc, "sparse_index_dtype", None) or "bf16").lower()
    out: list[tuple[Kernel, int]] = []
    for phase, s in (("prefill", step.prefill), ("decode", step.decode)):
        if s is None:
            continue
        s = _per_rank(s, dp)
        cand, mean = math.ceil(s.max_kv / ratio), math.ceil(s.mean_kv / ratio)
        q = dict(q=s.q, max_q=s.max_q) if phase == "prefill" else dict(q=s.q // max(1, s.n))
        out.append(
            (
                _k(
                    "indexer",
                    f"indexer_{phase}",
                    dt,
                    batch=s.n,
                    **q,
                    max_kv=cand,
                    mean_kv=mean,
                    heads=heads,
                    dim=dim,
                ),
                layers,
            )
        )
        if topk:
            out.append(
                (
                    _k(
                        "indexer",
                        "indexer_topk",
                        "fp32",
                        rows=s.q,
                        candidates=cand,
                        k=min(topk, cand),
                    ),
                    layers,
                )
            )
    t = max(1, math.ceil(step.tokens / dp))
    out.append(
        (
            _k("indexer", "index_cache_write", dt, tokens=max(1, math.ceil(t / ratio)), dim=dim),
            layers,
        )
    )
    return out


def _attention_side(cfg, tokens: int, c: lp.Counts) -> list[tuple[Kernel, int]]:
    """Norms, RoPE and the KV-cache write around each softmax-attention layer."""
    mc = cfg.model_config
    kv_dt = _kv_dtype(cfg)
    _, atp, dp, _, _ = _ranks(cfg.model_parallel_config)
    n_attn = c.softmax
    if not n_attn:
        return []
    t = max(1, math.ceil(tokens / dp))
    heads = max(1, int(mc.num_attention_heads) // atp)
    out: list[tuple[Kernel, int]] = []
    if getattr(mc, "multi_latent_attention", False):
        rope, latent = int(mc.qk_pos_emb_head_dim), int(mc.kv_lora_rank)
        if mc.q_lora_rank:
            out.append((_k("norm", "q_a_rmsnorm", "bf16", tokens=t, width=int(mc.q_lora_rank)), 1))
        out.append((_k("norm", "kv_a_rmsnorm", "bf16", tokens=t, width=latent), 1))
        out.append(
            (_k("elementwise", "rope", "bf16", tokens=t, heads=heads, kv_heads=1, dim=rope), 1)
        )
        out.append(
            (_k("elementwise", "kv_cache_write", kv_dt, tokens=t, kv_heads=1, dim=latent + rope), 1)
        )
    elif gemm_shapes.is_v4(mc):
        d, rope = int(mc.kv_channels), int(mc.qk_pos_emb_head_dim or 0)
        out.append((_k("norm", "q_a_rmsnorm", "bf16", tokens=t, width=int(mc.q_lora_rank)), 1))
        out.append((_k("norm", "kv_rmsnorm", "bf16", tokens=t, width=d), 1))
        out.append(
            (_k("elementwise", "rope", "bf16", tokens=t, heads=heads, kv_heads=1, dim=rope or d), 1)
        )
        out.append((_k("elementwise", "kv_cache_write", kv_dt, tokens=t, kv_heads=1, dim=d), 1))
    else:
        d = int(mc.kv_channels)
        groups = int(mc.num_query_groups) if mc.group_query_attention and mc.num_query_groups else 0
        kv_heads = max(1, (groups or int(mc.num_attention_heads)) // atp)
        if getattr(mc, "qk_layernorm", False):
            out.append(
                (_k("norm", "qk_rmsnorm", "bf16", tokens=t, heads=heads + kv_heads, width=d), 1)
            )
        out.append(
            (_k("elementwise", "rope", "bf16", tokens=t, heads=heads, kv_heads=kv_heads, dim=d), 1)
        )
        out.append(
            (_k("elementwise", "kv_cache_write", kv_dt, tokens=t, kv_heads=kv_heads, dim=2 * d), 1)
        )
    rows = [(k, n * n_attn) for k, n in out]
    if gemm_shapes.is_v4(mc):
        d = int(mc.kv_channels)
        for kind in (lp.V4_CSA, lp.V4_HCA):
            if c.n(kind):
                entries = max(1, math.ceil(t / lp.V4_RATIO[kind]))
                rows.append(
                    (
                        _k(
                            "elementwise",
                            "pool_cache_write",
                            kv_dt,
                            tokens=entries,
                            kv_heads=1,
                            dim=d,
                        ),
                        c.n(kind),
                    )
                )
    gate = gemm_shapes._gate_width(
        mc, heads, heads * int(getattr(mc, "v_head_dim", 0) or mc.kv_channels)
    )
    if gate:
        rows.append((_k("elementwise", "attn_gate_mul", "bf16", tokens=t, width=gate), n_attn))
    return rows


def _linear_attention(cfg, step: Step, c: lp.Counts) -> list[tuple[Kernel, int]]:
    mc = cfg.model_config
    n_lin = c.n(lp.LINEAR)
    if not n_lin:
        return []
    _, atp, dp, _, _ = _ranks(cfg.model_parallel_config)
    hk, hv, dk, dv = mc.linear_attention_geometry()
    hk, hv = max(1, hk // atp), max(1, hv // atp)
    kind = str(getattr(mc, "linear_attention_kind", "") or "linear").lower()
    geo = {"key_heads": hk, "value_heads": hv, "dk": dk, "dv": dv}
    out: list[tuple[Kernel, int]] = []
    if step.prefill is not None:
        s = _per_rank(step.prefill, dp)
        out.append(
            (
                _k(
                    "linear_attention",
                    f"{kind}_prefill",
                    "bf16",
                    batch=s.n,
                    q=s.q,
                    max_q=s.max_q,
                    **geo,
                ),
                n_lin,
            )
        )
    if step.decode is not None:
        s = _per_rank(step.decode, dp)
        out.append(
            (
                _k(
                    "linear_attention",
                    f"{kind}_decode",
                    "bf16",
                    batch=s.n,
                    q=s.q // max(1, s.n),
                    **geo,
                ),
                n_lin,
            )
        )
    t = max(1, math.ceil(step.tokens / dp))
    conv = int(getattr(mc, "linear_attention_conv_kernel", 0) or 0)
    if conv:
        out.append(
            (
                _k(
                    "linear_attention",
                    "causal_conv1d",
                    "bf16",
                    tokens=t,
                    channels=2 * hk * dk + hv * dv,
                    kernel=conv,
                ),
                n_lin,
            )
        )
    out.append((_k("norm", "gated_rmsnorm", "bf16", tokens=t, heads=hv, width=dv), n_lin))
    return out


def _layer_side(cfg, tokens: int, c: lp.Counts) -> list[tuple[Kernel, int]]:
    """Per-layer framing: DeepSeek-V4's mHC and Kimi-K3's attention residuals."""
    mc = cfg.model_config
    hidden = int(mc.hidden_size)
    out: list[tuple[Kernel, int]] = []
    hc = int(getattr(mc, "hc_mult", 0) or 0)
    if hc > 1:
        iters = int(getattr(mc, "hc_sinkhorn_iters", 0) or 20)
        out.append(
            (
                _k("elementwise", "hc_sinkhorn", "fp32", tokens=tokens, streams=hc, iters=iters),
                2 * c.layers,
            )
        )
        out.append(
            (
                _k("elementwise", "hc_apply", "bf16", tokens=tokens, streams=hc, width=hidden),
                2 * c.layers,
            )
        )
    block = int(getattr(mc, "attn_res_block_size", 0) or 0)
    if block > 0 and c.layers:
        blocks = max(1, math.ceil(int(mc.num_layers or c.layers) / block / 2))
        out.append(
            (
                _k(
                    "elementwise",
                    "attn_res_mix",
                    "bf16",
                    tokens=tokens,
                    width=hidden,
                    blocks=blocks,
                ),
                2 * c.layers,
            )
        )
    return out


def _mlp_side(cfg, tokens: int, c: lp.Counts) -> list[tuple[Kernel, int]]:
    """Activations and the MoE's routing and reduction kernels."""
    mc = cfg.model_config
    tp, _, _, ep, _ = _ranks(cfg.model_parallel_config)
    act = "silu_and_mul" if mc.swiglu else "gelu"
    out: list[tuple[Kernel, int]] = []
    if c.dense:
        ffn = max(1, int(mc.ffn_hidden_size) // tp)
        out.append((_k("elementwise", act, "bf16", tokens=tokens, width=ffn), c.dense))
    if c.moe:
        experts, topk = int(mc.num_experts), int(mc.moe_router_topk or 1)
        local = max(1, experts // ep)
        pad = gemm_shapes.moe_pad(cfg)
        ffn = gemm_shapes._pad(
            max(1, int(mc.moe_ffn_hidden_size or mc.ffn_hidden_size) // max(1, tp // ep)), pad
        )
        rows = max(1, math.ceil(tokens * topk * local / experts))
        width = int(getattr(mc, "moe_latent_hidden_size", 0) or 0) or int(mc.hidden_size)
        padded = gemm_shapes._pad(width, pad)
        if padded != width:
            out.append(
                (_k("moe", "moe_pad", "bf16", tokens=tokens, width=width, padded=padded), c.moe)
            )
        out += [
            (_k("moe", "moe_topk", "fp32", tokens=tokens, experts=experts, topk=topk), c.moe),
            (_k("moe", "moe_align", "int32", tokens=tokens, topk=topk, local_experts=local), c.moe),
            (_k("moe", f"moe_{act}", "bf16", rows=rows, width=ffn), c.moe),
            (_k("moe", "moe_sum", "bf16", tokens=tokens, topk=topk, width=width), c.moe),
        ]
        if getattr(mc, "moe_latent_hidden_size", 0):
            out.append(
                (_k("norm", "moe_latent_rmsnorm", "bf16", tokens=tokens, width=width), c.moe)
            )
        shared = int(getattr(mc, "moe_shared_expert_intermediate_size", 0) or 0)
        if shared:
            out.append(
                (
                    _k(
                        "elementwise",
                        f"shared_{act}",
                        "bf16",
                        tokens=tokens,
                        width=max(1, shared // tp),
                    ),
                    c.moe,
                )
            )
    return out


def _act_quant_dtype(cfg) -> str | None:
    req = getattr(cfg, "request_config", None)
    if hasattr(req, "resolved_act_quant_dtype"):
        q = req.resolved_act_quant_dtype(getattr(cfg.model_config, "fp8", None))
        return str(q).lower() if q else None
    return None


def _quant(cfg, gemms: list[tuple[gemm_shapes.GemmShape, int]]) -> list[tuple[Kernel, int]]:
    """One activation cast in front of each low-precision GEMM."""
    dt = _act_quant_dtype(cfg)
    if not dt:
        return []
    out = []
    for g, calls in gemms:
        if g.dtype in ("bf16", "fp16", "fp32") or g.op in ("moe_router", "lm_head"):
            continue
        rows = g.rows_per_expert * g.groups if g.groups > 1 and g.op == "moe_down" else g.m
        out.append((_k("quant", f"{g.op}_act_quant", dt, rows=rows, width=g.k), calls))
    return out


def _collectives(cfg, step: Step, c: lp.Counts) -> list[tuple[Kernel, int]]:
    mc = cfg.model_config
    tp, _, dp, ep, dcp = _ranks(cfg.model_parallel_config)
    hidden = int(mc.hidden_size)
    tokens = step.tokens
    msg = tokens * hidden * _ACT_BYTES
    out: list[tuple[Kernel, int]] = []
    if tp > 1 and c.layers:
        if dp > 1:
            # A rank owns whole requests: it gathers the tokens the TP MLP needs
            # and scatters the result back.
            out.append((_k("collective", "tp_allgather", "bf16", ranks=tp, bytes=msg), c.layers))
            out.append(
                (_k("collective", "tp_reduce_scatter", "bf16", ranks=tp, bytes=msg), c.layers)
            )
        else:
            out.append(
                (_k("collective", "tp_allreduce", "bf16", ranks=tp, bytes=msg), 2 * c.layers)
            )
    if tp > 1 and c.first:
        out.append((_k("collective", "embed_allreduce", "bf16", ranks=tp, bytes=msg), 1))
    if tp > 1 and c.last:
        vocab = int(mc.padded_vocab_size)
        out.append(
            (
                _k(
                    "collective",
                    "logits_allgather",
                    "bf16",
                    ranks=tp,
                    bytes=step.sampled_rows * vocab * _ACT_BYTES,
                ),
                1,
            )
        )
    if ep > 1 and c.moe:
        topk = int(mc.moe_router_topk or 1)
        a2a = max(1, math.ceil(tokens / dp)) * hidden * topk * _ACT_BYTES
        out.append((_k("collective", "ep_dispatch", "bf16", ranks=ep, bytes=a2a), c.moe))
        out.append((_k("collective", "ep_combine", "bf16", ranks=ep, bytes=a2a), c.moe))
    if dcp > 1 and step.decode is not None and c.softmax:
        heads = int(mc.num_attention_heads)
        width = int(getattr(mc, "kv_lora_rank", 0) or 0) or int(mc.kv_channels)
        m = step.decode.q * heads * width * _ACT_BYTES
        out.append((_k("collective", "dcp_allgather", "bf16", ranks=dcp, bytes=m), c.softmax))
        out.append((_k("collective", "dcp_reduce_scatter", "bf16", ranks=dcp, bytes=m), c.softmax))
    if not c.last:
        out.append((_k("collective", "pp_send_recv", "bf16", ranks=2, bytes=msg), 1))
    return out


capture_sizes = gemm_shapes.capture_sizes


def graph_tokens(cfg, step: Step) -> int:
    """The tokens a step's non-attention kernels launch over (see :func:`gemm_shapes.graph_tokens`)."""
    return gemm_shapes.graph_tokens(
        cfg, step.tokens, step.prefill is None and step.decode is not None
    )


def _absorb_mla(cfg, step: Step, gemms):
    """MLA decode runs absorbed: two per-head batched GEMMs instead of ``kv_b``.

    The query's no-RoPE part is multiplied into the latent (``W_UK``) and the
    latent attention output back out of it (``W_UV``), one batch per local
    head. ``kv_b`` is left over the prefill's keys and values: its new tokens
    and the cached context it up-projects.
    """
    mc = cfg.model_config
    if not getattr(mc, "multi_latent_attention", False) or step.decode is None:
        return gemms
    atp, _ = gemm_shapes.attention_ranks(cfg.model_parallel_config, 1)
    _, _, dp, _, _ = _ranks(cfg.model_parallel_config)
    heads = max(1, int(mc.num_attention_heads) // atp)
    nope, v, latent = int(mc.qk_head_dim), int(mc.v_head_dim), int(mc.kv_lora_rank)
    # A decode-only step's tokens are its graph's padded batch, which the BMMs replay at.
    t_dec = max(1, math.ceil((step.decode.q if step.prefill else step.tokens) / dp))
    out = []
    for g, calls in gemms:
        if g.op != "attn_kv_b":
            out.append((g, calls))
            continue
        if step.prefill is not None:
            m = max(1, math.ceil(step.prefill.n * step.prefill.mean_kv / dp))
            out.append((dataclasses.replace(g, m=m), calls))
        out.append(
            (
                gemm_shapes.GemmShape(
                    "attn_q_absorb", t_dec, latent, nope, "bf16", heads, 0, t_dec
                ),
                calls,
            )
        )
        out.append(
            (
                gemm_shapes.GemmShape("attn_o_absorb", t_dec, v, latent, "bf16", heads, 0, t_dec),
                calls,
            )
        )
    return out


def step_kernels(cfg, step: Step, counts: lp.Counts | None = None) -> list[tuple[Kernel, int]]:
    """Every kernel one forward pass launches on a rank, with how many times.

    ``counts`` restricts it to some layers (one pipeline stage); by default the
    whole model, as one rank holding every layer.
    """
    mc, mp = cfg.model_config, cfg.model_parallel_config
    c = counts or lp.counts(cfg)
    tp = max(1, int(mp.tensor_model_parallel_size or 1))
    step = dataclasses.replace(step, tokens=graph_tokens(cfg, step))
    tokens = max(1, step.tokens)
    hidden = int(mc.hidden_size)
    vocab = int(mc.padded_vocab_size)
    gemms = _absorb_mla(cfg, step, gemm_shapes.step_gemms(cfg, tokens, step.sampled_rows, c))
    out = [
        (
            _k(
                "gemm",
                g.op,
                g.dtype,
                m=g.m,
                n=g.n,
                k=g.k,
                groups=g.groups,
                topk=g.topk,
                rows_per_expert=g.rows_per_expert,
            ),
            calls,
        )
        for g, calls in gemms
    ]
    out += _quant(cfg, gemms)
    if c.first:
        out.append(
            (
                _k(
                    "elementwise",
                    "embedding",
                    "bf16",
                    tokens=tokens,
                    vocab=max(1, vocab // tp),
                    width=hidden,
                ),
                1,
            )
        )
    norms = 2 * c.layers + (1 if c.last else 0)
    if norms:
        out.append((_k("norm", "rmsnorm", "bf16", tokens=tokens, width=hidden), norms))
    out += _attention(cfg, step, c)
    out += _indexer(cfg, step, c)
    out += _attention_side(cfg, tokens, c)
    out += _linear_attention(cfg, step, c)
    out += _layer_side(cfg, tokens, c)
    out += _mlp_side(cfg, tokens, c)
    out += _collectives(cfg, step, c)
    if c.last:
        out.append(
            (
                _k(
                    "sampling",
                    "sampling",
                    "fp32",
                    rows=max(1, step.sampled_rows),
                    vocab=max(1, vocab),
                ),
                1,
            )
        )
    return out


# ── the drafter ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Draft:
    """A speculating pool's drafter, resolved once from its projector."""

    cfg: object
    k: int
    first_q: int
    parallel: bool
    serial_head_rank: int
    vocab: int
    method: str
    fused_layers: int


def resolve_draft(projector) -> Draft | None:
    """The drafter the projector prices, as a config of its own layers."""
    try:
        d = projector._drafter()
        k = int(projector._spec_k())
    except Exception:
        return None
    if d is None or k <= 0 or d.layers <= 0:
        return None
    cfg = projector.cfg
    mc = cfg.model_config
    over = dict(
        num_layers=d.layers,
        moe_pattern=[1 if d.moe else 0] * d.layers,
        linear_attention_layers=0,
        linear_attention_freq=0,
        linear_attention_kind="",
        sparse_index_n_heads=0,
        padded_vocab_size=max(1, int(round(int(mc.padded_vocab_size) * d.vocab_fraction))),
    )
    if gemm_shapes.is_v4(mc):
        over["compress_ratios"] = [0] * d.layers
    elif d.window:
        over.update(
            sink_sliding_window=d.window,
            sink_window_layer_fraction=1.0,
            sink_window_decode_capped=True,
        )
    if d.ffn_hidden_size:
        over["ffn_hidden_size"] = d.ffn_hidden_size
        over["moe_pattern"] = [0] * d.layers
    for field, value in (
        ("num_attention_heads", d.heads),
        ("num_query_groups", d.kv_heads),
        ("kv_channels", d.head_dim),
    ):
        if value:
            over[field] = value
    if d.kv_heads:
        over["group_query_attention"] = True
    dcfg = copy.copy(cfg)
    dcfg.model_config = (
        dataclasses.replace(mc, **over) if dataclasses.is_dataclass(mc) else _ns_replace(mc, over)
    )
    dcfg.model_parallel_config = _single_stage(cfg.model_parallel_config)
    first_q = 1
    if not d.parallel:
        try:
            from .speculative import acceptance_length

            first_q = max(1, int(round(acceptance_length(projector._spec_rates(k) or []))))
        except Exception:
            first_q = 1
    return Draft(
        dcfg,
        k,
        first_q,
        d.parallel,
        int(d.serial_head_rank),
        int(mc.padded_vocab_size),
        d.method,
        int(d.fused_layers),
    )


def _ns_replace(obj, over: dict):
    new = copy.copy(obj)
    for key, value in over.items():
        setattr(new, key, value)
    return new


def _single_stage(mp):
    if dataclasses.is_dataclass(mp):
        return dataclasses.replace(mp, pipeline_model_parallel_size=1)
    return _ns_replace(mp, {"pipeline_model_parallel_size": 1})


def draft_passes(draft: Draft, dec: Seqs) -> list[tuple[Step, int, list[tuple[Kernel, int]]]]:
    """The drafter's forward passes for one verify step's decodes, ``dec``.

    An autoregressive drafter extends over the tokens the last verify committed
    and then runs one single-token pass per further draft; a parallel (block)
    drafter runs once over the block, then its serial head once per position.
    The drafter attends over the same contexts as the target.
    """
    b = dec.n
    if not b:
        return []
    mc = draft.cfg.model_config
    hidden = int(mc.hidden_size)
    w, frac = lp.window_of(draft.cfg)
    capped = w and frac > 0 and getattr(mc, "sink_window_decode_capped", False)

    def step(tokens: int, rows: int, q: int) -> Step:
        s = Seqs(b, b * q, q, dec.max_kv, dec.mean_kv)
        win = (
            Seqs(
                b, b * q, q, min(dec.max_kv, _up(w, KV_BUCKET)), min(dec.mean_kv, _up(w, KV_BUCKET))
            )
            if capped
            else None
        )
        return Step(tokens, rows, s, None, win)

    def fuse(tokens: int) -> list[tuple[Kernel, int]]:
        if draft.method == "mtp":
            return [
                (
                    _k(
                        "gemm",
                        "draft_eh_proj",
                        "bf16",
                        m=tokens,
                        n=hidden,
                        k=2 * hidden,
                        groups=1,
                        topk=0,
                        rows_per_expert=0,
                    ),
                    draft.cfg.model_config.num_layers,
                )
            ]
        fused = draft.fused_layers or (3 if draft.method == "eagle3" else 0)
        if fused:
            return [
                (
                    _k(
                        "gemm",
                        "draft_fc",
                        "bf16",
                        m=tokens,
                        n=hidden,
                        k=fused * hidden,
                        groups=1,
                        topk=0,
                        rows_per_expert=0,
                    ),
                    1,
                )
            ]
        return []

    if draft.parallel:
        q = draft.k + 1
        extra = fuse(b * q)
        if draft.serial_head_rank:
            tp = max(1, int(draft.cfg.model_parallel_config.tensor_model_parallel_size or 1))
            extra.append(
                (
                    _k(
                        "gemm",
                        "draft_serial_head",
                        "bf16",
                        m=b,
                        n=max(1, draft.vocab // tp),
                        k=draft.serial_head_rank,
                        groups=1,
                        topk=0,
                        rows_per_expert=0,
                    ),
                    draft.k,
                )
            )
        return [(step(b * q, b * draft.k, q), 1, extra)]
    out = [(step(b * draft.first_q, b, draft.first_q), 1, fuse(b * draft.first_q))]
    if draft.k > 1:
        out.append((step(b, b, 1), draft.k - 1, fuse(b)))
    return out


# ── timing ──────────────────────────────────────────────────────────────────


class Timer:
    """Per-kernel time estimates from the projector's own simulators."""

    def __init__(self, projector) -> None:
        self.gemm = getattr(projector, "_gemm", None)
        self.sdpa = getattr(projector, "_sdpa", None)
        self.comm = getattr(projector, "_comm", None)
        self.bw = float(getattr(self.gemm, "hbm_bandwidth_gbps", None) or 5300.0)
        self.peak = float(getattr(self.gemm, "peak_tflops_bf16", None) or 1307.0)
        mp = getattr(getattr(projector, "cfg", None), "model_parallel_config", None)
        self.dp = max(1, int(getattr(mp, "attention_data_parallel_size", 1) or 1))

    def us(self, k: Kernel) -> tuple[float | None, str | None]:
        d = dict(k.dims)
        try:
            if k.kind == "gemm":
                g = gemm_shapes.GemmShape(
                    k.op,
                    d["m"],
                    d["n"],
                    d["k"],
                    k.dtype,
                    d["groups"],
                    d["topk"],
                    d["rows_per_expert"],
                )
                us = gemm_shapes._estimate_us(self.gemm, g)
                return (us, "origami") if us is not None else self._roofline(k, d)
            if (
                k.kind == "attention"
                and k.op != "attn_decode_reduce"
                and self.sdpa is not None
                and self.sdpa.is_available()
            ):
                return self._sdpa_us(k, d), "sdpa"
            if k.kind == "collective" and self.comm is not None:
                us = self._comm_us(k, d)
                if us is not None:
                    return us, "collectives"
        except Exception:
            pass
        return self._roofline(k, d)

    def _sdpa_us(self, k: Kernel, d: dict) -> float:
        prefill = k.op == "attn_prefill"
        span = d["mean_kv"] + d.get("mean_pool_kv", 0)
        if d.get("topk"):
            span = min(span, d["topk"])
        q = max(1, d["q"] // max(1, d["batch"])) if prefill else d["q"]
        r = self.sdpa.simulate_sdpa(
            batch_size=d["batch"],
            num_heads=d["heads"],
            seq_len=q,
            head_dim=d["qk_dim"],
            causal=prefill,
            dtype="bf16",
            seq_len_kv=max(q, span),
            num_heads_kv=d["kv_heads"],
            head_dim_v=d["v_dim"],
        )
        return 1000.0 * float(r.forward_time_ms)

    def _comm_us(self, k: Kernel, d: dict) -> float | None:
        from infera.projection.core.projection.module_profilers import collective_model as cm

        from .collectives import _min_over_protocols

        c, msg = self.comm, int(d["bytes"])
        hidden = max(1, int(getattr(c, "hidden", 0) or 1))
        tokens = max(1, msg // (hidden * _ACT_BYTES))
        if k.op in ("tp_allreduce", "embed_allreduce"):
            per = 1.0 if getattr(c, "attn_dp", 1) > 1 else 2.0
            return 1000.0 * c.tp_allreduce_ms(tokens, 1) / per
        if k.op in ("tp_allgather", "tp_reduce_scatter"):
            return 1000.0 * c.tp_allreduce_ms(tokens, 1) / 2.0
        if k.op in ("ep_dispatch", "ep_combine"):
            topk = max(1, int(getattr(c, "topk", 1) or 1))
            return 1000.0 * c.ep_a2a_ms(max(1, tokens // topk) * self.dp, 1) / 2.0
        if k.op in ("dcp_allgather", "dcp_reduce_scatter"):
            return None
        if k.op == "pp_send_recv":
            return float(cm.sendrecv(c._args, msg))
        if k.op == "logits_allgather":
            return float(_min_over_protocols(cm.run_allgather, c._args, msg, d["ranks"], ["tp"]))
        return None

    def _roofline(self, k: Kernel, d: dict) -> tuple[float | None, str | None]:
        flops, nbytes = _traffic(k, d)
        if not flops and not nbytes:
            return None, None
        t_mem = nbytes / (self.bw * 1e9 * _BW_FRACTION)
        t_cmp = flops / (self.peak * 1e12 * _COMPUTE_EFF)
        return 1e6 * max(t_mem, t_cmp), "roofline"


def _bytes(dtype: str) -> float:
    from infera.projection.core.projection.training_config import dtype_num_bytes

    return float(dtype_num_bytes(dtype))


def _traffic(k: Kernel, d: dict) -> tuple[float, float]:
    """``(FLOPs, bytes)`` one launch of ``k`` moves, for a roofline estimate."""
    a, op = _ACT_BYTES, k.op
    if k.kind == "gemm":
        m = d["rows_per_expert"] * d["groups"] if d["groups"] > 1 else d["m"]
        w = d["n"] * d["k"] * d["groups"] * _bytes(k.dtype)
        return 2.0 * m * d["n"] * d["k"], w + m * (d["k"] + d["n"]) * a
    if op == "attn_decode_reduce":
        # Each partition's output plus its max and sum, merged into one row per head.
        partial = d["batch"] * d["heads"] * d["partitions"] * (d["v_dim"] * a + 8.0)
        return 2.0 * d["batch"] * d["heads"] * d["partitions"] * d["v_dim"], partial + d[
            "batch"
        ] * d["heads"] * d["v_dim"] * a
    if k.kind == "attention":
        span = d["mean_kv"] + d.get("mean_pool_kv", 0)
        if d.get("topk"):
            span = min(span, d["topk"])
        q = d["q"] if op == "attn_prefill" else d["q"] * d["batch"]
        flops = 2.0 * q * d["heads"] * (d["qk_dim"] + d["v_dim"]) * span
        if op == "attn_prefill":
            flops /= 2.0
        kv = d["batch"] * span * d["kv_heads"] * (d["qk_dim"] + d["v_dim"]) * _bytes(k.dtype)
        return flops, kv + q * d["heads"] * (d["qk_dim"] + d["v_dim"]) * a
    if op.startswith("indexer_") and op != "indexer_topk":
        q = d["q"] if op == "indexer_prefill" else d["q"] * d["batch"]
        return 2.0 * q * d["heads"] * d["dim"] * d["mean_kv"], d["batch"] * d["mean_kv"] * d[
            "dim"
        ] * _bytes(k.dtype)
    if op == "indexer_topk":
        return 0.0, d["rows"] * d["candidates"] * 4.0
    if k.kind == "linear_attention":
        if op == "causal_conv1d":
            return 2.0 * d["tokens"] * d["channels"] * d["kernel"], d["tokens"] * d[
                "channels"
            ] * 2 * a
        hv, dk, dv = d["value_heads"], d["dk"], d["dv"]
        state = d["batch"] * hv * dk * dv * 4.0
        if op.endswith("_decode"):
            tok = d["batch"] * d["q"]
            return 4.0 * tok * hv * dk * dv, 2 * state + tok * hv * (2 * dk + 2 * dv) * a
        tok = d["q"]
        flops = 2.0 * tok * hv * (2 * dk * dv + _LINEAR_CHUNK * (dk + dv))
        return flops, 2 * state + tok * hv * (2 * dk + 2 * dv) * a
    if op == "moe_pad":
        return 0.0, d["tokens"] * (d["width"] + d["padded"]) * a
    if op == "sampling":
        return 0.0, _SAMPLING_PASSES * d["rows"] * d["vocab"] * 4.0
    if op == "moe_topk":
        return 0.0, d["tokens"] * d["experts"] * 4.0
    if op == "moe_align":
        return 0.0, d["tokens"] * d["topk"] * 4.0 * 2
    if op == "moe_sum":
        return 0.0, d["tokens"] * (d["topk"] + 1) * d["width"] * a
    if k.kind == "quant":
        return 0.0, d["rows"] * d["width"] * (a + _bytes(k.dtype))
    if op in ("kv_cache_write", "pool_cache_write"):
        return 0.0, d["tokens"] * d["kv_heads"] * d["dim"] * (a + _bytes(k.dtype))
    if op == "index_cache_write":
        return 0.0, d["tokens"] * d["dim"] * (a + _bytes(k.dtype))
    if op == "rope":
        return 0.0, 2 * d["tokens"] * (d["heads"] + d["kv_heads"]) * d["dim"] * a
    if op == "hc_sinkhorn":
        return 0.0, 2 * d["tokens"] * d["streams"] ** 2 * d["iters"] * 4.0
    if op == "hc_apply":
        return 0.0, 2 * d["tokens"] * d["streams"] * d["width"] * a
    if op == "attn_res_mix":
        return 0.0, d["tokens"] * d["width"] * (d["blocks"] + 1) * a
    if op == "embedding":
        return 0.0, 2 * d["tokens"] * d["width"] * a
    if k.kind == "collective":
        return 0.0, 0.0
    width = d.get("width", 0) * max(1, d.get("heads", 1))
    rows = d.get("tokens", d.get("rows", 0))
    if k.kind == "norm":
        return 0.0, 3 * rows * width * a
    # activations: read gate and up, write one
    return 0.0, 3 * rows * width * a


# ── the trace ───────────────────────────────────────────────────────────────


class KernelTrace(gemm_shapes.GemmTrace):
    """A :class:`GemmTrace` that also keeps each step's sequence composition."""

    def __init__(self) -> None:
        super().__init__()
        self._compositions: dict[str, Counter] = {}
        self._timers: dict[str, Timer] = {}
        self._drafts: dict[str, Draft | None] = {}
        # (step, cost-kernel key) per pool: what the DES charged each step as.
        self._priced: dict[str, Counter] = {}
        # Per pool: the cost kernel's speculation length and attention-DP spread.
        self._pool_meta: dict[str, dict] = {}
        self.speedups = None
        self.calibration = None

    def record(
        self,
        projector,
        pool,
        tokens,
        sampled_rows,
        decode=None,
        prefill=None,
        q_len=1,
        cost=None,
        dpa_spread=False,
    ):
        super().record(projector, pool, tokens, sampled_rows, decode=decode, prefill=prefill)
        if pool not in self._timers:
            self._timers[pool] = Timer(projector)
            self._drafts[pool] = resolve_draft(projector) if q_len > 1 else None
            self._pool_meta[pool] = {"q_len": int(q_len), "dpa_spread": bool(dpa_spread)}
        step = make_step(self._cfgs[pool], tokens, sampled_rows, decode, prefill, q_len)
        self.add_step(pool, step, cost)

    def add_step(self, pool: str, step: Step, cost=None, n: int = 1) -> None:
        self._compositions.setdefault(pool, Counter())[step] += n
        self._priced.setdefault(pool, Counter())[(step, tuple(cost) if cost else None)] += n

    def bind(self, pool: str, projector, q_len: int = 1, dpa_spread: bool = False) -> None:
        """Price and expand ``pool``'s steps with ``projector`` (its config, simulators, drafter)."""
        self._cfgs[pool] = projector.cfg
        self._backends[pool] = getattr(projector, "_gemm", None)
        self._steps.setdefault(pool, Counter())
        self._timers[pool] = Timer(projector)
        self._drafts[pool] = resolve_draft(projector) if q_len > 1 else None
        self._pool_meta[pool] = {"q_len": int(q_len), "dpa_spread": bool(dpa_spread)}

    def _draft(self, pool: str) -> tuple[Counter, Counter]:
        """The drafter's passes over every verify step ``pool`` ran, and its own GEMMs."""
        steps: Counter = Counter()
        extra: Counter = Counter()
        draft = self._drafts.get(pool)
        if draft is None:
            return steps, extra
        for step, n in self._compositions.get(pool, {}).items():
            if step.decode is None:
                continue
            for s, m, ks in draft_passes(draft, step.decode):
                steps[s] += n * m
                for k, c in ks:
                    extra[k] += n * m * c
        return steps, extra

    def pools(self) -> dict[str, tuple[object, Counter, Counter]]:
        """Per reported pool: its config, its step compositions and its extra kernels."""
        out = {}
        for pool, steps in self._compositions.items():
            out[pool] = (self._cfgs[pool], steps, Counter())
            draft = self._drafts.get(pool)
            if draft is not None:
                out[f"{pool}/draft"] = (draft.cfg, *self._draft(pool))
        return out

    def absorb(self, other: gemm_shapes.GemmTrace) -> None:
        super().absorb(other)
        for pool, steps in getattr(other, "_compositions", {}).items():
            self._compositions.setdefault(pool, Counter()).update(steps)
            if pool not in self._cfgs and pool in other._cfgs:
                self._cfgs[pool] = other._cfgs[pool]
        for pool, priced in getattr(other, "_priced", {}).items():
            self._priced.setdefault(pool, Counter()).update(priced)
        for pool, meta in getattr(other, "_pool_meta", {}).items():
            self._pool_meta.setdefault(pool, meta)
        for pool, timer in getattr(other, "_timers", {}).items():
            self._timers.setdefault(pool, timer)
        for pool, draft in getattr(other, "_drafts", {}).items():
            self._drafts.setdefault(pool, draft)

    def kernels(self, estimate: bool = True) -> list[dict]:
        """One row per distinct (pool, stage, kernel), heaviest first."""
        calls: Counter = Counter()
        for pool, (cfg, steps, extra) in self.pools().items():
            # A drafter runs beside the target's last stage.
            offset = self._draft_stage(pool)
            stages = lp.stage_counts(cfg)
            for step, n in steps.items():
                for stage, c in enumerate(stages):
                    for k, m in step_kernels(cfg, step, c):
                        calls[(pool, stage + offset, k)] += n * m
            for k, m in extra.items():
                calls[(pool, offset, k)] += m
        out = []
        busy: Counter = Counter()
        for (pool, stage, k), n in calls.items():
            us, src = self.time(pool, k) if estimate else (None, None)
            row = {
                "pool": pool,
                "stage": stage,
                "kind": k.kind,
                "op": k.op,
                "dtype": k.dtype,
                **dict(k.dims),
                "calls": n,
            }
            row["est_us"] = None if us is None else round(us, 3)
            row["est_total_ms"] = None if us is None else round(us * n / 1000.0, 3)
            row["est_source"] = src
            busy[pool] += (us or 0.0) * n
            out.append(row)
        out += self._extra_rows(busy)
        if self.speedups is not None:
            for row in out:
                self.speedups.annotate(row)
        out.sort(
            key=lambda r: (
                -(r["est_total_ms"] or 0.0),
                -r["calls"],
                r["pool"],
                r["stage"],
                r["kind"],
                r["op"],
            )
        )
        return out

    def time(self, pool: str, k: Kernel) -> tuple[float | None, str | None]:
        """One call of ``k`` on ``pool``, memoised, calibrated, before any speedup."""
        memo = self.__dict__.setdefault("_timed", {})
        key = (pool, k)
        if key not in memo:
            timer = self._timers.get(pool) or self._timers.get(pool.split("/", 1)[0])
            us, src = timer.us(k) if timer is not None else (None, None)
            if us is not None and self.calibration is not None:
                f = self.calibration.factor(
                    {"pool": pool, "kind": k.kind, "op": k.op, "dtype": k.dtype, **dict(k.dims)}
                )
                if f is not None:
                    us, src = us * f, f"{src}+calibrated"
            memo[key] = (us, src)
        return memo[key]

    def _extra_rows(self, busy: Counter) -> list[dict]:
        """The calibration's engine kernels: one row per pool, a share of its kernels' time."""
        if self.calibration is None:
            return []
        rows = []
        for pool, us in busy.items():
            steps = sum(self._compositions.get(pool.split("/", 1)[0], {}).values()) or 1
            for e in self.calibration.extras_for(pool):
                total = e.share * us
                rows.append(
                    {
                        "pool": pool,
                        "stage": 0,
                        "kind": "engine",
                        "op": e.op,
                        "dtype": "",
                        "calls": steps,
                        "est_us": round(total / steps, 3),
                        "est_total_ms": round(total / 1000.0, 3),
                        "est_source": "calibrated",
                    }
                )
        return rows

    def _draft_stage(self, pool: str) -> int:
        base, _, rest = pool.partition("/")
        if rest != "draft" or base not in self._cfgs:
            return 0
        return len(lp.stages(self._cfgs[base])) - 1

    def write_kernels(self, path: str, estimate: bool = True) -> list[dict]:
        rows = self.kernels(estimate=estimate)
        if path.endswith(".json"):
            with open(path, "w") as fh:
                json.dump(
                    {"steps": self.num_steps, "kernels": rows, "by_op": _by_op(rows)}, fh, indent=1
                )
            return rows
        head = ["pool", "stage", "kind", "op", "dtype"]
        tail = ["calls", "est_us", "est_total_ms", "est_source"]
        dims = sorted({c for r in rows for c in r} - set(head) - set(tail))
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=head + dims + tail, restval="")
            w.writeheader()
            w.writerows(rows)
        return rows


def _by_op(rows: list[dict]) -> list[dict]:
    """Per (pool, stage, kind, op): calls, distinct shapes and estimated time."""
    acc: dict[tuple, dict] = {}
    for r in rows:
        a = acc.setdefault(
            (r["pool"], r["stage"], r["kind"], r["op"]),
            {
                "pool": r["pool"],
                "stage": r["stage"],
                "kind": r["kind"],
                "op": r["op"],
                "calls": 0,
                "distinct_shapes": 0,
                "est_total_ms": 0.0,
            },
        )
        a["calls"] += r["calls"]
        a["distinct_shapes"] += 1
        a["est_total_ms"] = round(a["est_total_ms"] + (r["est_total_ms"] or 0.0), 3)
    return sorted(acc.values(), key=lambda a: -a["est_total_ms"])
