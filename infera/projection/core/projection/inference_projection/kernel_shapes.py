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
* ``attention``: prefill (extend) and decode attention over the full-attention
  layers, one row per step composition: sequences, query tokens, the longest
  and mean KV span, heads and head dims on one rank, the KV-cache dtype and the
  sliding window. Windowed layers are their own rows. Their prefill spans are
  the sequences' own (the kernel masks outside the window), and their decode
  spans are capped at the window only when the engine keeps just the window's
  blocks (``sink_window_decode_capped``, SGLang's hybrid SWA pool). MLA prefill is listed as the per-head attention it runs; MLA
  decode as the absorbed form over the shared latent (one KV head of width
  ``kv_lora_rank + rope``).
* ``linear_attention``: GDN / KDA layers' chunked prefill, recurrent decode and
  causal conv1d.
* ``moe``: the router's top-k, the token-to-expert alignment, the activation
  between the expert GEMMs and the top-k reduction.
* ``norm``, ``elementwise``: RMSNorm (fused residual add), QK-norm, MLA's
  latent norms, the MLP activation, RoPE, the KV-cache write and the embedding
  lookup.
* ``collective``: the messages a step sends, in bytes, over the group that
  sends them: TP all-reduce (or all-gather plus reduce-scatter under
  attention-DP), EP dispatch and combine, DCP's gather and scatter, PP
  send/recv and the LM head's logit gather. These follow the communication
  model in :mod:`collectives`.
* ``sampling``: the sampler over the full vocabulary, for each sampled row.

Counts are per rank, summed over every layer of the model. Spans are bucketed
(KV to :data:`KV_BUCKET` tokens, prefill query lengths to :data:`Q_BUCKET`) so a
trace gives a tunable set rather than one row per step; every other dimension
is exact. Only GEMMs carry a time estimate (Origami).

Not enumerated: DeepSeek-V4's compressed attention branches and sparse
indexers, KDA's own projections, attention output gates, activation
quantization kernels and a speculative drafter's own forward.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass

from . import gemm_shapes

KV_BUCKET = 256
Q_BUCKET = 64
_ACT_BYTES = 2


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


def _up(x: int, b: int) -> int:
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
        mean_kv=_up(round(sum(spans) / len(spans)), KV_BUCKET),
    )


@dataclass(frozen=True)
class Step:
    tokens: int
    sampled_rows: int
    decode: Seqs | None = None
    prefill: Seqs | None = None
    decode_window: Seqs | None = None


def window_of(cfg) -> tuple[int, int, int]:
    """``(window, windowed attention layers, full attention layers)``."""
    mc, req = cfg.model_config, getattr(cfg, "request_config", None)
    _, _, n_attn = gemm_shapes._layer_counts(mc)
    model_w = int(getattr(mc, "sink_sliding_window", 0) or 0)
    w = req.resolved_sliding_window(model_w) if hasattr(req, "resolved_sliding_window") else model_w
    if not w:
        return 0, 0, n_attn
    even = bool(getattr(mc, "sink_window_even_layers_only", False))
    frac = float(getattr(mc, "sink_window_layer_fraction", 0.0) or 0.0)
    if hasattr(req, "resolved_sliding_window_fraction"):
        f = req.resolved_sliding_window_fraction(even, frac)
    else:
        f = frac or (0.5 if even else 1.0)
    n_win = min(n_attn, round(n_attn * f))
    return w, n_win, n_attn - n_win


def make_step(cfg, tokens, sampled_rows, decode=None, prefill=None, q_len=1) -> Step:
    """A step from its per-sequence ``(query tokens, KV computed)`` pairs."""
    dec = [(q_len, kv) for kv in decode] if decode else []
    pre = list(prefill) if prefill else []
    w, n_win, _ = window_of(cfg)
    capped = n_win and getattr(cfg.model_config, "sink_window_decode_capped", False)
    return Step(
        int(tokens),
        int(sampled_rows),
        seqs_of(dec),
        _bucket_q(seqs_of(pre)),
        seqs_of(dec, w) if capped and dec else None,
    )


def _bucket_q(s: Seqs | None) -> Seqs | None:
    if s is None:
        return None
    return Seqs(s.n, _up(s.q, Q_BUCKET), _up(s.max_q, Q_BUCKET), s.max_kv, s.mean_kv)


def _k(kind: str, op: str, dtype: str, **dims) -> Kernel:
    return Kernel(kind, op, dtype, tuple(dims.items()))


def _ranks(mp) -> tuple[int, int, int, int, int, int]:
    """``(tp, attention tp, attention dp, ep, dcp, pp)``."""
    tp = max(1, int(mp.tensor_model_parallel_size or 1))
    dp = max(1, int(getattr(mp, "attention_data_parallel_size", 1) or 1))
    atp = max(1, tp // dp) if dp > 1 else tp
    ep = max(1, int(getattr(mp, "expert_model_parallel_size", 1) or 1))
    dcp = max(1, int(getattr(mp, "decode_context_parallel_size", 1) or 1))
    pp = max(1, int(getattr(mp, "pipeline_model_parallel_size", 1) or 1))
    return tp, atp, dp, ep, dcp, pp


def _per_rank(s: Seqs, dp: int) -> Seqs:
    """Attention-DP: a rank attends over its share of the step's sequences."""
    if dp <= 1:
        return s
    n = max(1, math.ceil(s.n / dp))
    return Seqs(n, max(1, math.ceil(s.q / dp)), s.max_q, s.max_kv, s.mean_kv)


def _attention(cfg, step: Step) -> list[tuple[Kernel, int]]:
    mc, mp = cfg.model_config, cfg.model_parallel_config
    req = getattr(cfg, "request_config", None)
    kv_dt = str(getattr(req, "kv_cache_dtype", None) or "bf16").lower()
    tp, atp, dp, _, dcp, _ = _ranks(mp)
    w, n_win, n_full = window_of(cfg)
    heads = max(1, int(mc.num_attention_heads) // atp)
    if getattr(mc, "multi_latent_attention", False):
        nope, rope = int(mc.qk_head_dim), int(mc.qk_pos_emb_head_dim)
        latent = int(mc.kv_lora_rank)
        prefill_geo = (heads, heads, nope + rope, int(mc.v_head_dim))
        decode_geo = (heads, 1, latent + rope, latent)
    else:
        d = int(mc.kv_channels)
        kv = int(mc.num_query_groups) if mc.group_query_attention and mc.num_query_groups else 0
        kv_heads = max(1, (kv or int(mc.num_attention_heads)) // atp)
        prefill_geo = decode_geo = (heads, kv_heads, d, d)
    out: list[tuple[Kernel, int]] = []
    for layers, window, dec, pre in (
        (n_full, 0, step.decode, step.prefill),
        (n_win, w, step.decode_window or step.decode, step.prefill),
    ):
        if not layers:
            continue
        if pre is not None:
            s = _per_rank(pre, dp)
            h, hk, dqk, dv = prefill_geo
            out.append(
                (
                    _k(
                        "attention",
                        "attn_prefill",
                        kv_dt,
                        batch=s.n,
                        q=s.q,
                        max_q=s.max_q,
                        max_kv=s.max_kv,
                        mean_kv=s.mean_kv,
                        heads=h,
                        kv_heads=hk,
                        qk_dim=dqk,
                        v_dim=dv,
                        window=window,
                    ),
                    layers,
                )
            )
        if dec is not None:
            s = _per_rank(dec, dp)
            h, hk, dqk, dv = decode_geo
            max_kv, mean_kv = s.max_kv, s.mean_kv
            if dcp > 1:
                # A rank attends over its slice of every context, for the heads
                # its DCP group gathered.
                max_kv = _up(math.ceil(max_kv / dcp), KV_BUCKET)
                mean_kv = _up(math.ceil(mean_kv / dcp), KV_BUCKET)
                h = min(int(mc.num_attention_heads), h * dcp)
            out.append(
                (
                    _k(
                        "attention",
                        "attn_decode",
                        kv_dt,
                        batch=s.n,
                        q=s.q // max(1, s.n),
                        max_kv=max_kv,
                        mean_kv=mean_kv,
                        heads=h,
                        kv_heads=hk,
                        qk_dim=dqk,
                        v_dim=dv,
                        window=window,
                    ),
                    layers,
                )
            )
    return out


def _attention_side(cfg, tokens: int) -> list[tuple[Kernel, int]]:
    """Norms, RoPE and the KV-cache write around each full-attention layer."""
    mc, mp = cfg.model_config, cfg.model_parallel_config
    req = getattr(cfg, "request_config", None)
    kv_dt = str(getattr(req, "kv_cache_dtype", None) or "bf16").lower()
    _, atp, dp, _, _, _ = _ranks(mp)
    _, _, n_attn = gemm_shapes._layer_counts(mc)
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
        return [(k, n * n_attn) for k, n in out]
    d = int(mc.kv_channels)
    groups = int(mc.num_query_groups) if mc.group_query_attention and mc.num_query_groups else 0
    kv_heads = max(1, (groups or int(mc.num_attention_heads)) // atp)
    if getattr(mc, "qk_layernorm", False):
        out.append((_k("norm", "qk_rmsnorm", "bf16", tokens=t, heads=heads + kv_heads, width=d), 1))
    out.append(
        (_k("elementwise", "rope", "bf16", tokens=t, heads=heads, kv_heads=kv_heads, dim=d), 1)
    )
    out.append(
        (_k("elementwise", "kv_cache_write", kv_dt, tokens=t, kv_heads=kv_heads, dim=2 * d), 1)
    )
    return [(k, n * n_attn) for k, n in out]


def _linear_attention(cfg, step: Step) -> list[tuple[Kernel, int]]:
    mc, mp = cfg.model_config, cfg.model_parallel_config
    n_lin = mc.linear_attention_layer_count() if hasattr(mc, "linear_attention_layer_count") else 0
    if not n_lin:
        return []
    _, atp, dp, _, _, _ = _ranks(mp)
    hk, hv, dk, dv = mc.linear_attention_geometry()
    hk, hv = max(1, hk // atp), max(1, hv // atp)
    kind = str(getattr(mc, "linear_attention_kind", "") or "linear").lower()
    out: list[tuple[Kernel, int]] = []
    geo = {"key_heads": hk, "value_heads": hv, "dk": dk, "dv": dv}
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
    conv = int(getattr(mc, "linear_attention_conv_kernel", 0) or 0)
    if conv:
        t = max(1, math.ceil(step.tokens / dp))
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
    return out


def _mlp_side(cfg, tokens: int) -> list[tuple[Kernel, int]]:
    """Activations and the MoE's routing and reduction kernels."""
    mc, mp = cfg.model_config, cfg.model_parallel_config
    tp, _, dp, ep, _, _ = _ranks(mp)
    n_dense, n_moe, _ = gemm_shapes._layer_counts(mc)
    act = "silu_and_mul" if mc.swiglu else "gelu"
    out: list[tuple[Kernel, int]] = []
    if n_dense:
        ffn = max(1, int(mc.ffn_hidden_size) // tp)
        out.append((_k("elementwise", act, "bf16", tokens=tokens, width=ffn), n_dense))
    if n_moe:
        experts, topk = int(mc.num_experts), int(mc.moe_router_topk or 1)
        local = max(1, experts // ep)
        ffn = max(1, int(mc.moe_ffn_hidden_size or mc.ffn_hidden_size) // max(1, tp // ep))
        rows = max(1, math.ceil(tokens * topk * local / experts))
        hidden = int(mc.hidden_size)
        out += [
            (_k("moe", "moe_topk", "fp32", tokens=tokens, experts=experts, topk=topk), n_moe),
            (_k("moe", "moe_align", "int32", tokens=tokens, topk=topk, local_experts=local), n_moe),
            (_k("moe", f"moe_{act}", "bf16", rows=rows, width=ffn), n_moe),
            (_k("moe", "moe_sum", "bf16", tokens=tokens, topk=topk, width=hidden), n_moe),
        ]
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
                    n_moe,
                )
            )
    return out


def _collectives(cfg, step: Step) -> list[tuple[Kernel, int]]:
    mc, mp = cfg.model_config, cfg.model_parallel_config
    tp, atp, dp, ep, dcp, pp = _ranks(mp)
    hidden = int(mc.hidden_size)
    layers = int(mc.num_layers or 0)
    _, n_moe, n_attn = gemm_shapes._layer_counts(mc)
    tokens = step.tokens
    msg = tokens * hidden * _ACT_BYTES
    out: list[tuple[Kernel, int]] = []
    if tp > 1:
        if dp > 1:
            # A rank owns whole requests: it gathers the tokens the TP MLP needs
            # and scatters the result back.
            out.append((_k("collective", "tp_allgather", "bf16", ranks=tp, bytes=msg), layers))
            out.append((_k("collective", "tp_reduce_scatter", "bf16", ranks=tp, bytes=msg), layers))
        else:
            out.append((_k("collective", "tp_allreduce", "bf16", ranks=tp, bytes=msg), 2 * layers))
        out.append((_k("collective", "embed_allreduce", "bf16", ranks=tp, bytes=msg), 1))
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
    if ep > 1 and n_moe:
        topk = int(mc.moe_router_topk or 1)
        a2a = max(1, math.ceil(tokens / dp)) * hidden * topk * _ACT_BYTES
        out.append((_k("collective", "ep_dispatch", "bf16", ranks=ep, bytes=a2a), n_moe))
        out.append((_k("collective", "ep_combine", "bf16", ranks=ep, bytes=a2a), n_moe))
    if dcp > 1 and step.decode is not None and n_attn:
        heads = int(mc.num_attention_heads)
        width = int(getattr(mc, "kv_lora_rank", 0) or 0) or int(mc.kv_channels)
        m = step.decode.q * heads * width * _ACT_BYTES
        out.append((_k("collective", "dcp_allgather", "bf16", ranks=dcp, bytes=m), n_attn))
        out.append((_k("collective", "dcp_reduce_scatter", "bf16", ranks=dcp, bytes=m), n_attn))
    if pp > 1:
        out.append((_k("collective", "pp_send_recv", "bf16", ranks=2, bytes=msg), pp - 1))
    return out


def step_kernels(cfg, step: Step) -> list[tuple[Kernel, int]]:
    """Every kernel one forward pass launches on a rank, with how many times."""
    mc, mp = cfg.model_config, cfg.model_parallel_config
    tp = max(1, int(mp.tensor_model_parallel_size or 1))
    tokens = max(1, step.tokens)
    hidden = int(mc.hidden_size)
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
        for g, calls in gemm_shapes.step_gemms(cfg, tokens, step.sampled_rows)
    ]
    vocab = int(mc.padded_vocab_size)
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
    out.append(
        (
            _k("norm", "rmsnorm", "bf16", tokens=tokens, width=hidden),
            2 * int(mc.num_layers or 0) + 1,
        )
    )
    out += _attention(cfg, step)
    out += _attention_side(cfg, tokens)
    out += _linear_attention(cfg, step)
    out += _mlp_side(cfg, tokens)
    out += _collectives(cfg, step)
    out.append(
        (_k("sampling", "sampling", "fp32", rows=max(1, step.sampled_rows), vocab=max(1, vocab)), 1)
    )
    return out


class KernelTrace(gemm_shapes.GemmTrace):
    """A :class:`GemmTrace` that also keeps each step's sequence composition."""

    def __init__(self) -> None:
        super().__init__()
        self._compositions: dict[str, Counter] = {}

    def record(self, projector, pool, tokens, sampled_rows, decode=None, prefill=None, q_len=1):
        super().record(projector, pool, tokens, sampled_rows)
        step = make_step(self._cfgs[pool], tokens, sampled_rows, decode, prefill, q_len)
        self._compositions.setdefault(pool, Counter())[step] += 1

    def absorb(self, other: gemm_shapes.GemmTrace) -> None:
        super().absorb(other)
        for pool, steps in getattr(other, "_compositions", {}).items():
            self._compositions.setdefault(pool, Counter()).update(steps)

    def kernels(self, estimate: bool = True) -> list[dict]:
        """One row per distinct (pool, kernel), heaviest first."""
        calls: Counter = Counter()
        for pool, steps in self._compositions.items():
            cfg = self._cfgs[pool]
            for step, n in steps.items():
                for k, c in step_kernels(cfg, step):
                    calls[(pool, k)] += n * c
        out = []
        for (pool, k), n in calls.items():
            us = None
            if estimate and k.kind == "gemm":
                d = dict(k.dims)
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
                us = gemm_shapes._estimate_us(self._backends.get(pool), g)
            row = {
                "pool": pool,
                "kind": k.kind,
                "op": k.op,
                "dtype": k.dtype,
                **dict(k.dims),
                "calls": n,
            }
            row["est_us"] = None if us is None else round(us, 3)
            row["est_total_ms"] = None if us is None else round(us * n / 1000.0, 3)
            out.append(row)
        out.sort(
            key=lambda r: (-(r["est_total_ms"] or 0.0), -r["calls"], r["pool"], r["kind"], r["op"])
        )
        return out

    def write_kernels(self, path: str, estimate: bool = True) -> list[dict]:
        rows = self.kernels(estimate=estimate)
        if path.endswith(".json"):
            with open(path, "w") as fh:
                json.dump(
                    {"steps": self.num_steps, "kernels": rows, "by_op": _by_op(rows)}, fh, indent=1
                )
            return rows
        head = ["pool", "kind", "op", "dtype"]
        tail = ["calls", "est_us", "est_total_ms"]
        dims = sorted({c for r in rows for c in r} - set(head) - set(tail))
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=head + dims + tail, restval="")
            w.writeheader()
            w.writerows(rows)
        return rows


def _by_op(rows: list[dict]) -> list[dict]:
    """Per (pool, kind, op): calls and how many distinct shapes it takes."""
    acc: dict[tuple, dict] = {}
    for r in rows:
        a = acc.setdefault(
            (r["pool"], r["kind"], r["op"]),
            {"pool": r["pool"], "kind": r["kind"], "op": r["op"], "calls": 0, "distinct_shapes": 0},
        )
        a["calls"] += r["calls"]
        a["distinct_shapes"] += 1
    return sorted(acc.values(), key=lambda a: (a["kind"], -a["calls"]))
