"""Naive PyTorch equivalents of the DSv4 unified_kv decode/prefill attention.

Study replacement (see infera.sglang.study/CLAUDE.md). These reproduce the exact
math of the vendored Triton kernels (`sparse_attn_v4_paged_decode` /
`sparse_attn_v4_paged_prefill`) using plain gather + dense softmax, so they are
trivially readable and serve as a correctness oracle.

Semantics recovered from paged_decode.py / paged_prefill.py:
  For each query token t and head h, over that token's ragged set of valid KV
  slots S_t (given by kv_indices[kv_indptr[t]:kv_indptr[t+1]]):
      k_j    = source[slot_j]                       # [D]  (D=512, page_size 1)
      s_j    = (q_{t,h} . k_j) * softmax_scale
      # attention sink = one virtual key with score = attn_sink[h] and V = 0,
      # contributing only to the softmax denominator.
      w      = softmax( [s_j for j] ++ [attn_sink[h]] )        # last entry = sink
      out    = sum_j w_j * k_j                       # V == K, full D dims
      # (sink weight multiplies V=0 -> adds nothing to the numerator)
  out: [T, H, D], same dtype as q. bf16 unified_kv (no fp8 in the R4 path).

Decode uses a single unified_kv source. Prefill concatenates two sources per
token: a "prefix" stream gathered from unified_kv and an "extend" stream gathered
from the current-chunk `kv` tensor; both share one softmax.

These are O(T * maxlen * D) dense and only meant for tiny batches (max-bs < 4).
"""
from __future__ import annotations

import torch


def _ragged_lengths(indptr: torch.Tensor) -> torch.Tensor:
    return (indptr[1:] - indptr[:-1]).to(torch.int64)


def _attend_one_source(
    q: torch.Tensor,            # [T, H, D] fp32
    source: torch.Tensor,       # [pages, D] fp32
    indices: torch.Tensor,      # [total] int  (flat per-token slot lists)
    indptr: torch.Tensor,       # [T+1] int
    softmax_scale: float,
):
    """Return per-token unnormalized (m, l, acc) online-softmax state for one
    KV source. m:[T,H] running max, l:[T,H] denom, acc:[T,H,D] numerator.
    Tokens with zero length yield m=-inf, l=0, acc=0."""
    T, H, D = q.shape
    device = q.device
    NEG = torch.finfo(torch.float32).min

    m = torch.full((T, H), NEG, dtype=torch.float32, device=device)
    l = torch.zeros((T, H), dtype=torch.float32, device=device)
    acc = torch.zeros((T, H, D), dtype=torch.float32, device=device)

    lengths = _ragged_lengths(indptr)
    for t in range(T):
        n = int(lengths[t].item())
        if n == 0:
            continue
        start = int(indptr[t].item())
        slots = indices[start : start + n].to(torch.int64)
        # Kernel skips slot < 0 (sentinels); ragged-packed indices are usually
        # all-valid, but guard anyway.
        valid = slots >= 0
        if not bool(valid.all()):
            slots = slots[valid]
            if slots.numel() == 0:
                continue
        k = source[slots]                       # [n, D] fp32
        qt = q[t]                               # [H, D]
        scores = (qt @ k.t()) * softmax_scale   # [H, n]
        m[t] = scores.max(dim=1).values         # [H]
        p = torch.exp(scores - m[t][:, None])   # [H, n]
        l[t] = p.sum(dim=1)                     # [H]
        acc[t] = p @ k                          # [H, D]
    return m, l, acc


def _finalize(m, l, acc, attn_sink, out_dtype):
    """Fold the attention sink (virtual K, V=0) into the denominator and
    normalize. m:[T,H], l:[T,H], acc:[T,H,D], attn_sink:[H]."""
    T, H, D = acc.shape
    sink = attn_sink[:H].to(torch.float32).to(acc.device)      # [H]
    m_final = torch.maximum(m, sink[None, :])                  # [T,H]
    alpha = torch.exp(m - m_final)                             # rescale prior state
    l_final = l * alpha + torch.exp(sink[None, :] - m_final)   # sink adds to denom
    denom = torch.clamp(l_final, min=1e-30)
    out = (acc * alpha[:, :, None]) / denom[:, :, None]
    # kernel writes 0 where l_final <= 0 (all-empty token)
    out = torch.where((l_final > 0.0)[:, :, None], out, torch.zeros_like(out))
    return out.to(out_dtype)


def naive_decode(
    *,
    q: torch.Tensor,            # [T, H, D]
    unified_kv: torch.Tensor,   # [pages, D]
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,    # [H]
    softmax_scale: float,
) -> torch.Tensor:
    out_dtype = q.dtype
    qf = q.to(torch.float32)
    src = unified_kv.to(torch.float32)
    m, l, acc = _attend_one_source(qf, src, kv_indices, kv_indptr, softmax_scale)
    return _finalize(m, l, acc, attn_sink, out_dtype)


def _merge(m1, l1, acc1, m2, l2, acc2):
    """Combine two online-softmax partial states into one (same-frame merge)."""
    m = torch.maximum(m1, m2)
    a1 = torch.exp(m1 - m)
    a2 = torch.exp(m2 - m)
    l = l1 * a1 + l2 * a2
    acc = acc1 * a1[:, :, None] + acc2 * a2[:, :, None]
    return m, l, acc


def naive_prefill(
    *,
    q: torch.Tensor,                 # [T, H, D]
    unified_kv: torch.Tensor,        # [pages, D]
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_extend: torch.Tensor,         # [total_tokens, D] current-chunk K
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,         # [H]
    softmax_scale: float,
) -> torch.Tensor:
    out_dtype = q.dtype
    qf = q.to(torch.float32)
    src_pre = unified_kv.to(torch.float32)
    src_ext = kv_extend.to(torch.float32)
    m1, l1, acc1 = _attend_one_source(
        qf, src_pre, kv_indices_prefix, kv_indptr_prefix, softmax_scale
    )
    m2, l2, acc2 = _attend_one_source(
        qf, src_ext, kv_indices_extend, kv_indptr_extend, softmax_scale
    )
    m, l, acc = _merge(m1, l1, acc1, m2, l2, acc2)
    return _finalize(m, l, acc, attn_sink, out_dtype)
