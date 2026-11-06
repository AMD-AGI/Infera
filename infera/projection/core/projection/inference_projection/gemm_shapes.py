###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The GEMM shapes a replayed trace actually runs, for kernel tuning.

A DES replay already decides, step by step, how many tokens each forward pass
carries. Those token counts are the GEMM ``M``, and under continuous batching
they vary every step, which is why a fixed sweep of ``M`` misses the shapes a
production trace spends its time in. :class:`GemmTrace` records each executed
step's token count while the DES runs, and :func:`step_gemms` expands one
step into the linear-layer GEMMs it launches on one GPU.

The shapes here are the kernels' shapes, not the cost model's. The layer
profilers size some GEMMs FLOP-equivalently (dividing ``M`` by TP rather than
sharding ``N``), which prices them correctly but describes a kernel that does
not exist. Here weights are sharded the way vLLM and SGLang shard them, and the
projections those engines fuse are fused:

* attention: fused QKV (column-parallel over heads, KV heads replicated when
  fewer than TP) and the row-parallel output projection; MLA as the fused
  ``q_a``/``kv_a`` down-projection (replicated), ``q_b``, ``kv_b`` and ``o``.
  Under attention-DP a rank sees ``1/dp`` of the tokens over ``tp/dp`` of the
  heads.
* dense MLP and shared experts: fused gate/up and down, sharded over TP.
* routed experts: one grouped GEMM per projection over the rank's local
  experts; ``m`` is the tokens entering the fused-MoE op (the tuning key vLLM
  and AITER use) and ``rows_per_expert`` the mean rows each expert sees.
* router and LM head: the LM head only over the rows that are sampled.

A mixed step runs its prefill chunks and decodes through the linear layers as
one batch, so ``M`` is their sum; attention itself (SDPA) is not a GEMM here.
Linear-attention layers list their own projections (GDN fused, KDA separate),
DeepSeek-V4 its Q LoRA, shared KV head, grouped output LoRA, compressors,
indexer and mHC mixers, a sparse-attention model its indexer's projections, and
an attention output gate is listed fused into QKV (or on its own beside MLA).
MLA is listed unabsorbed. Counts are per tensor-parallel rank, summed over every
layer of the model (all pipeline stages).
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass

_ACTIVE: GemmTrace | None = None


@dataclass(frozen=True)
class GemmShape:
    op: str
    m: int
    n: int
    k: int
    dtype: str
    groups: int = 1
    topk: int = 0
    rows_per_expert: int = 0


def _weight_dtype(cfg, field: str) -> str:
    mc = cfg.model_config
    declared = getattr(getattr(cfg, "request_config", None), field, None) or getattr(
        mc, field, None
    )
    return str(declared or ("fp8" if getattr(mc, "fp8", None) else "bf16")).lower()


def attention_ranks(mp, tokens: int) -> tuple[int, int]:
    """``(TP ranks attention heads are split over, tokens one rank attends)``."""
    tp = max(1, int(mp.tensor_model_parallel_size or 1))
    dp = max(1, int(getattr(mp, "attention_data_parallel_size", 1) or 1))
    if dp > 1:
        return max(1, tp // dp), max(1, math.ceil(tokens / dp))
    return tp, tokens


def is_v4(mc) -> bool:
    """DeepSeek-V4's attention: a Q LoRA, one shared KV head and a grouped output LoRA."""
    return bool(getattr(mc, "hybrid_attention_enabled", False)) and bool(
        int(getattr(mc, "o_lora_rank", 0) or 0)
    )


def _gate_width(mc, heads: int, width: int) -> int:
    """Output columns of the attention output gate (0 without one)."""
    if not getattr(mc, "attention_output_gate", False):
        return 0
    return heads if getattr(mc, "attention_output_gate_per_head", False) else width


def _attention(mc, mp, tokens: int, dt: str) -> list[GemmShape]:
    tp, tokens = attention_ranks(mp, tokens)
    hidden = int(mc.hidden_size)
    heads = max(1, int(mc.num_attention_heads) // tp)
    if getattr(mc, "multi_latent_attention", False):
        nope, rope, v = int(mc.qk_head_dim), int(mc.qk_pos_emb_head_dim), int(mc.v_head_dim)
        q_lora, kv_lora = int(mc.q_lora_rank or 0), int(mc.kv_lora_rank)
        out = []
        if q_lora:
            out.append(GemmShape("attn_qkv_a", tokens, q_lora + kv_lora + rope, hidden, dt))
            out.append(GemmShape("attn_q_b", tokens, heads * (nope + rope), q_lora, dt))
        else:
            out.append(GemmShape("attn_q", tokens, heads * (nope + rope), hidden, dt))
            out.append(GemmShape("attn_kv_a", tokens, kv_lora + rope, hidden, dt))
        out.append(GemmShape("attn_kv_b", tokens, heads * (nope + v), kv_lora, dt))
        out.append(GemmShape("attn_o", tokens, hidden, heads * v, dt))
        gate = _gate_width(mc, heads, heads * v)
        if gate:
            out.append(GemmShape("attn_gate", tokens, gate, hidden, dt))
        return out
    d = int(mc.kv_channels)
    if is_v4(mc):
        q_lora, o_lora = int(mc.q_lora_rank), int(mc.o_lora_rank)
        o_groups = max(1, int(getattr(mc, "o_groups", 0) or 1))
        groups = max(1, o_groups // tp)
        per_group = int(mc.num_attention_heads) * d // o_groups
        return [
            GemmShape("attn_q_a", tokens, q_lora, hidden, dt),
            GemmShape("attn_q_b", tokens, heads * d, q_lora, dt),
            GemmShape("attn_kv", tokens, d, hidden, dt),
            GemmShape("attn_o_a", tokens, o_lora, per_group, dt, groups, 0, tokens),
            GemmShape("attn_o_b", tokens, hidden, groups * o_lora, dt),
        ]
    groups = (
        int(mc.num_query_groups)
        if mc.group_query_attention and mc.num_query_groups
        else int(mc.num_attention_heads)
    )
    kv_heads = max(1, groups // tp)
    # An output gate is fused into the Q projection (Qwen3-Next, Qwen3.5).
    gate = _gate_width(mc, heads, heads * d)
    return [
        GemmShape("attn_qkv", tokens, (heads + 2 * kv_heads) * d + gate, hidden, dt),
        GemmShape("attn_o", tokens, hidden, heads * d, dt),
    ]


def _v4_extras(mc, mp, tokens: int, dt: str, c) -> list[tuple[GemmShape, int]]:
    """DeepSeek-V4's compressors, CSA indexer and mHC mixers.

    Laid out as the V4 reference and Primus' closed-form FLOPs count them: a
    compressed layer projects each token to its pool entry and gate
    (``2 * d`` wide on CSA's overlapping pool, ``d`` on HCA's), a CSA layer's
    indexer has its own low-rank query, per-head weights and mini-compressor,
    and every layer mixes its ``hc_mult`` residual streams before attention and
    before the MLP. All of these are replicated over TP.
    """
    from . import layer_plan as lp

    _, t = attention_ranks(mp, tokens)
    hidden, d = int(mc.hidden_size), int(mc.kv_channels)
    out: list[tuple[GemmShape, int]] = []
    csa, hca = c.n(lp.V4_CSA), c.n(lp.V4_HCA)
    if csa:
        out.append((GemmShape("compressor_kv_gate", t, 2 * 2 * d, hidden, dt), csa))
        ih, idim = int(mc.index_n_heads or 0), int(mc.index_head_dim or 0)
        if ih and idim:
            out += [
                (GemmShape("idx_q_a", t, idim, hidden, dt), csa),
                (GemmShape("idx_q_b", t, ih * idim, idim, dt), csa),
                (GemmShape("idx_weights", t, ih, hidden, dt), csa),
                (GemmShape("idx_compressor", t, 2 * 2 * idim, hidden, dt), csa),
            ]
    if hca:
        out.append((GemmShape("compressor_kv_gate", t, 2 * d, hidden, dt), hca))
    hc = int(getattr(mc, "hc_mult", 0) or 0)
    if hc > 1:
        out.append(
            (GemmShape("hc_mixer", tokens, (2 + hc) * hc, hc * hidden, "fp32"), 2 * c.layers)
        )
        if c.last:
            out.append((GemmShape("hc_head", tokens, hc, hc * hidden, "fp32"), 1))
    return out


def _sparse_indexer(mc, mp, tokens: int, dt: str, n: int) -> list[tuple[GemmShape, int]]:
    """A uniform top-k indexer's projections (DeepSeek-V3.2 / GLM-5 layout), replicated."""
    ih = int(getattr(mc, "sparse_index_n_heads", 0) or 0)
    idim = int(getattr(mc, "sparse_index_head_dim", 0) or 0)
    if not (ih and idim and n):
        return []
    _, t = attention_ranks(mp, tokens)
    hidden = int(mc.hidden_size)
    q_in = int(mc.q_lora_rank or 0) or hidden
    return [
        (GemmShape("idx_q", t, ih * idim, q_in, dt), n),
        (GemmShape("idx_k", t, idim, hidden, dt), n),
        (GemmShape("idx_weights", t, ih, hidden, dt), n),
    ]


def _mlp(op: str, tokens: int, hidden: int, ffn: int, swiglu: bool, dt: str) -> list[GemmShape]:
    return [
        GemmShape(
            f"{op}_gate_up" if swiglu else f"{op}_up",
            tokens,
            (2 if swiglu else 1) * ffn,
            hidden,
            dt,
        ),
        GemmShape(f"{op}_down", tokens, hidden, ffn, dt),
    ]


def _moe(mc, mp, tokens: int, dt: str, linear_dt: str) -> list[GemmShape]:
    tp = max(1, int(mp.tensor_model_parallel_size or 1))
    ep = max(1, int(mp.expert_model_parallel_size or 1))
    hidden = int(mc.hidden_size)
    experts = int(mc.num_experts)
    topk = int(mc.moe_router_topk or 1)
    ffn = max(1, int(mc.moe_ffn_hidden_size or mc.ffn_hidden_size) // max(1, tp // ep))
    local = max(1, experts // ep)
    rows = max(1, math.ceil(tokens * topk / experts))
    gate_up = 2 * ffn if mc.swiglu else ffn
    # Latent MoE (Kimi-K3): the experts run on a narrower projection of the residual.
    latent = int(getattr(mc, "moe_latent_hidden_size", 0) or 0)
    width = latent or hidden
    out = [GemmShape("moe_router", tokens, experts, hidden, "bf16")]
    if latent:
        out.append(GemmShape("moe_latent_down", tokens, latent, hidden, linear_dt))
    out += [
        GemmShape(
            "moe_gate_up" if mc.swiglu else "moe_up", tokens, gate_up, width, dt, local, topk, rows
        ),
        GemmShape("moe_down", tokens, width, ffn, dt, local, topk, rows),
    ]
    if latent:
        out.append(GemmShape("moe_latent_up", tokens, hidden, latent, linear_dt))
    shared = int(getattr(mc, "moe_shared_expert_intermediate_size", 0) or 0)
    if shared:
        out += _mlp("shared", tokens, hidden, max(1, shared // tp), mc.swiglu, linear_dt)
    return out


def _linear_attention(mc, mp, tokens: int, dt: str, n_lin: int) -> list[tuple[GemmShape, int]]:
    """A linear layer's projections, sharded over heads.

    GDN in the Qwen3-Next layout: fused ``q, k, v, z`` and fused ``a, b``. KDA
    as the linear-attention profiler prices it: separate q, k and v, a low-rank
    decay gate whose down-projection is replicated, a per-head beta and a full-
    or low-rank output gate.
    """
    kind = str(getattr(mc, "linear_attention_kind", "") or "").lower()
    if kind not in ("gdn", "kda") or not n_lin:
        return []
    tp, t = attention_ranks(mp, tokens)
    hk, hv, dk, dv = mc.linear_attention_geometry()
    hk, hv = max(1, hk // tp), max(1, hv // tp)
    pk, pv = hk * dk, hv * dv
    hidden = int(mc.hidden_size)
    if kind == "gdn":
        shapes = [
            GemmShape("gdn_in_proj_qkvz", t, 2 * pk + 2 * pv, hidden, dt),
            GemmShape("gdn_in_proj_ba", t, 2 * hv, hidden, dt),
            GemmShape("gdn_out_proj", t, hidden, pv, dt),
        ]
    else:
        shapes = [
            GemmShape("kda_q", t, pk, hidden, dt),
            GemmShape("kda_k", t, pk, hidden, dt),
            GemmShape("kda_v", t, pv, hidden, dt),
            GemmShape("kda_decay_a", t, dk, hidden, dt),
            GemmShape("kda_decay_b", t, pv, dk, dt),
            GemmShape("kda_beta", t, hv, hidden, dt),
        ]
        if getattr(mc, "linear_attention_full_rank_gate", False):
            shapes.append(GemmShape("kda_gate", t, pv, hidden, dt))
        else:
            shapes += [
                GemmShape("kda_gate_a", t, dv, hidden, dt),
                GemmShape("kda_gate_b", t, pv, dv, dt),
            ]
        shapes.append(GemmShape("kda_out_proj", t, hidden, pv, dt))
    return [(g, n_lin) for g in shapes]


def step_gemms(cfg, tokens: int, sampled_rows: int, counts=None) -> list[tuple[GemmShape, int]]:
    """The GEMMs one forward pass of ``tokens`` tokens launches on a TP rank.

    Returns ``(shape, calls)`` pairs, ``calls`` being how many layers launch it.
    ``counts`` (:class:`layer_plan.Counts`) restricts them to some of the
    layers, such as one pipeline stage's; by default the whole model's.
    """
    from . import layer_plan as lp

    mc, mp = cfg.model_config, cfg.model_parallel_config
    c = counts or lp.counts(cfg)
    tokens = max(1, int(tokens))
    tp = max(1, int(mp.tensor_model_parallel_size or 1))
    linear_dt = _weight_dtype(cfg, "linear_weight_dtype")
    out: list[tuple[GemmShape, int]] = []
    if c.softmax:
        out += [(g, c.softmax) for g in _attention(mc, mp, tokens, linear_dt)]
    if is_v4(mc):
        out += _v4_extras(mc, mp, tokens, linear_dt, c)
    else:
        out += _sparse_indexer(mc, mp, tokens, linear_dt, c.indexed)
    out += _linear_attention(mc, mp, tokens, linear_dt, c.n(lp.LINEAR))
    if c.dense:
        ffn = max(1, int(mc.ffn_hidden_size) // tp)
        out += [
            (g, c.dense)
            for g in _mlp("mlp", tokens, int(mc.hidden_size), ffn, mc.swiglu, linear_dt)
        ]
    if c.moe:
        out += [
            (g, c.moe)
            for g in _moe(mc, mp, tokens, _weight_dtype(cfg, "moe_expert_dtype"), linear_dt)
        ]
    if c.last:
        vocab = max(1, int(mc.padded_vocab_size) // tp)
        out.append(
            (GemmShape("lm_head", max(1, int(sampled_rows)), vocab, int(mc.hidden_size), "bf16"), 1)
        )
    return out


class GemmTrace:
    """Token counts of every executed DES step, expanded into GEMM shapes."""

    def __init__(self) -> None:
        self._steps: dict[str, Counter] = {}
        self._cfgs: dict[str, object] = {}
        self._backends: dict[str, object] = {}

    def record(self, projector, pool: str, tokens: int, sampled_rows: int, **_composition) -> None:
        if pool not in self._cfgs:
            self._cfgs[pool] = projector.cfg
            self._backends[pool] = getattr(projector, "_gemm", None)
            self._steps[pool] = Counter()
        self._steps[pool][(int(tokens), int(sampled_rows))] += 1

    def absorb(self, other: GemmTrace) -> None:
        for pool, steps in other._steps.items():
            if pool not in self._cfgs:
                self._cfgs[pool] = other._cfgs[pool]
                self._backends[pool] = other._backends[pool]
                self._steps[pool] = Counter()
            self._steps[pool].update(steps)

    @property
    def num_steps(self) -> int:
        return sum(sum(c.values()) for c in self._steps.values())

    def shapes(self, estimate: bool = True) -> list[dict]:
        """One row per distinct (pool, shape), heaviest first when timed."""
        calls: Counter = Counter()
        for pool, steps in self._steps.items():
            cfg = self._cfgs[pool]
            for (tokens, rows), n in steps.items():
                for g, layers in step_gemms(cfg, tokens, rows):
                    calls[(pool, g)] += n * layers
        timed: dict[tuple, float | None] = {}
        out = []
        for (pool, g), n in calls.items():
            us = None
            if estimate:
                key = (pool, g)
                if key not in timed:
                    timed[key] = _estimate_us(self._backends.get(pool), g)
                us = timed[key]
            row = {"pool": pool, **asdict(g), "calls": n}
            row["est_us"] = None if us is None else round(us, 3)
            row["est_total_ms"] = None if us is None else round(us * n / 1000.0, 3)
            out.append(row)
        out.sort(
            key=lambda r: (-(r["est_total_ms"] or 0.0), -r["calls"], r["pool"], r["op"], r["m"])
        )
        return out

    def write(self, path: str, estimate: bool = True) -> list[dict]:
        rows = self.shapes(estimate=estimate)
        if path.endswith(".json"):
            with open(path, "w") as fh:
                json.dump(
                    {"steps": self.num_steps, "shapes": rows, "by_weight": _by_weight(rows)},
                    fh,
                    indent=1,
                )
        else:
            fields = (
                list(rows[0])
                if rows
                else ["pool", *GemmShape.__dataclass_fields__, "calls", "est_us", "est_total_ms"]
            )
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
        return rows


def _estimate_us(backend, g: GemmShape) -> float | None:
    if backend is None or not getattr(backend, "is_available", lambda: False)():
        return None
    m = g.rows_per_expert if g.groups > 1 else g.m
    try:
        return 1000.0 * float(
            backend.simulate_gemm(m, g.n, g.k, g.dtype, batch=g.groups).forward_time_ms
        )
    except Exception:
        return None


def _by_weight(rows: list[dict]) -> list[dict]:
    """Per weight matrix (op, n, k, dtype): how the trace spreads its ``M``."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["pool"], r["op"], r["n"], r["k"], r["dtype"], r["groups"]), []).append(
            r
        )
    out = []
    for (pool, op, n, k, dt, grp), rs in groups.items():
        ms = sorted((r["m"], r["calls"]) for r in rs)
        total = sum(c for _, c in ms)
        timed = [r["est_total_ms"] for r in rs if r["est_total_ms"] is not None]
        out.append(
            {
                "pool": pool,
                "op": op,
                "n": n,
                "k": k,
                "dtype": dt,
                "groups": grp,
                "calls": total,
                "distinct_m": len(ms),
                "m_p50": _weighted_pct(ms, total, 0.50),
                "m_p90": _weighted_pct(ms, total, 0.90),
                "m_p99": _weighted_pct(ms, total, 0.99),
                "m_max": ms[-1][0],
                "est_total_ms": round(sum(timed), 3) if timed else None,
            }
        )
    out.sort(key=lambda r: (-(r["est_total_ms"] or 0.0), -r["calls"]))
    return out


def _weighted_pct(ms: list[tuple[int, int]], total: int, p: float) -> int:
    target, seen = p * total, 0
    for m, c in ms:
        seen += c
        if seen >= target:
            return m
    return ms[-1][0]


def active() -> GemmTrace | None:
    return _ACTIVE


@contextlib.contextmanager
def recording(trace: GemmTrace) -> Iterator[GemmTrace]:
    """Record every DES step executed inside this block into ``trace``."""
    global _ACTIVE
    prev, _ACTIVE = _ACTIVE, trace
    try:
        yield trace
    finally:
        _ACTIVE = prev


class _Pass:
    def __init__(self, target: GemmTrace | None) -> None:
        self.target = target
        self.trace = type(target)() if target is not None else GemmTrace()

    def commit(self) -> None:
        """Keep this pass's steps; a pass that is not committed is discarded."""
        if self.target is not None:
            self.target.absorb(self.trace)


@contextlib.contextmanager
def pass_capture() -> Iterator[_Pass]:
    """Record a simulation pass on its own, for a caller that may rerun it."""
    global _ACTIVE
    p = _Pass(_ACTIVE)
    prev, _ACTIVE = _ACTIVE, (p.trace if _ACTIVE is not None else None)
    try:
        yield p
    finally:
        _ACTIVE = prev


@contextlib.contextmanager
def paused() -> Iterator[None]:
    """Steps executed inside this block are not recorded."""
    global _ACTIVE
    prev, _ACTIVE = _ACTIVE, None
    try:
        yield
    finally:
        _ACTIVE = prev
