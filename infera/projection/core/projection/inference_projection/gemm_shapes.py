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
Linear-attention (GDN/KDA) layers' own projections are not enumerated, and MLA
is listed unabsorbed. Counts are per tensor-parallel rank, summed over every
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


def _layer_counts(mc) -> tuple[int, int, int]:
    """(dense MLP layers, MoE layers, full-attention layers)."""
    n_layers = int(mc.num_layers or 0)
    pattern = mc.moe_pattern or [0] * n_layers
    n_moe = sum(1 for x in pattern if x) if getattr(mc, "num_experts", 0) else 0
    linear = mc.linear_attention_layer_count() if hasattr(mc, "linear_attention_layer_count") else 0
    return n_layers - n_moe, n_moe, max(0, n_layers - linear)


def _attention(mc, mp, tokens: int, dt: str) -> list[GemmShape]:
    tp = max(1, int(mp.tensor_model_parallel_size or 1))
    dp = max(1, int(getattr(mp, "attention_data_parallel_size", 1) or 1))
    if dp > 1:
        tp = max(1, tp // dp)
        tokens = max(1, math.ceil(tokens / dp))
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
        return out
    d = int(mc.kv_channels)
    groups = (
        int(mc.num_query_groups)
        if mc.group_query_attention and mc.num_query_groups
        else int(mc.num_attention_heads)
    )
    kv_heads = max(1, groups // tp)
    return [
        GemmShape("attn_qkv", tokens, (heads + 2 * kv_heads) * d, hidden, dt),
        GemmShape("attn_o", tokens, hidden, heads * d, dt),
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
    out = [
        GemmShape("moe_router", tokens, experts, hidden, "bf16"),
        GemmShape(
            "moe_gate_up" if mc.swiglu else "moe_up", tokens, gate_up, hidden, dt, local, topk, rows
        ),
        GemmShape("moe_down", tokens, hidden, ffn, dt, local, topk, rows),
    ]
    shared = int(getattr(mc, "moe_shared_expert_intermediate_size", 0) or 0)
    if shared:
        out += _mlp("shared", tokens, hidden, max(1, shared // tp), mc.swiglu, linear_dt)
    return out


def step_gemms(cfg, tokens: int, sampled_rows: int) -> list[tuple[GemmShape, int]]:
    """The GEMMs one forward pass of ``tokens`` tokens launches on a TP rank.

    Returns ``(shape, calls)`` pairs, ``calls`` being how many layers launch it.
    """
    mc, mp = cfg.model_config, cfg.model_parallel_config
    tokens = max(1, int(tokens))
    tp = max(1, int(mp.tensor_model_parallel_size or 1))
    n_dense, n_moe, n_attn = _layer_counts(mc)
    linear_dt = _weight_dtype(cfg, "linear_weight_dtype")
    out: list[tuple[GemmShape, int]] = []
    if n_attn:
        out += [(g, n_attn) for g in _attention(mc, mp, tokens, linear_dt)]
    if n_dense:
        ffn = max(1, int(mc.ffn_hidden_size) // tp)
        out += [
            (g, n_dense)
            for g in _mlp("mlp", tokens, int(mc.hidden_size), ffn, mc.swiglu, linear_dt)
        ]
    if n_moe:
        out += [
            (g, n_moe)
            for g in _moe(mc, mp, tokens, _weight_dtype(cfg, "moe_expert_dtype"), linear_dt)
        ]
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

    def record(self, projector, pool: str, tokens: int, sampled_rows: int) -> None:
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
        self.trace = GemmTrace()

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
