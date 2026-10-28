#!/usr/bin/env python3
"""Forge driver: GLM-5.2-MXFP4 DCP4 decode split indexer (one DCP rank) on MI355X.

Contract (KernelForge):
  * correctness  ``python driver.py``                -> ``case_snr: <id> <dB>`` lines, ``SNR: <min> dB``
  * benchmark    ``python driver.py --bench-mode``   -> ``wall_ms`` samples, one ``case_ms`` per case
  * profile      ``python driver.py --profile-run``  -> a few calls of the profile case

What it measures: ``dcp_indexer_topk`` = Indexer._dcp_split_decode_topk of sglang 93034eba91 on DCP rank 0 for the
TARGET_VERIFY rows of the DCP4 decode leg (ppdc config): logits over the owned pages (r, r+4, ...) of a replicated
index-K cache, local top-2048, pack, (all_gather: the other 3 ranks' packs are an input), merge + slot transform.
  T = 6 x bs_graph rows, q fp8 [T, 32, 128], weights fp32 [T, 32]; index-K pool = the whole decode cache
  (9.46 M tokens = 147,824 pages x 64 x (128 fp8 + 4 B scale), aiter preshuffled layout, production writer).
  Contexts: lognormal (median 89k, p90 283k tokens, AgentX ISL); row i of a request has length L + i (i = 1..6).
Oracle: the same pipeline in torch for all 4 ranks (aiter logits, torch.topk, torch pack/merge of
dsa_dcp_split_indexer.py) -> per-row global top-2048 slot sets. case_snr = 100 dB when every row's set is
identical, else -10*log10(1 - mean recall). Timing: N_LAYERS index pools per graph replay; case_ms = replay / N_LAYERS.

This file is the measurement oracle: forge must not edit it.
"""
from __future__ import annotations

import argparse
import math
import sys

import torch

from graph_harness import cuda_graph_bench
from dcp_indexer_kernel import dcp_indexer_topk

H, D, PS, TOPK, DCP = 32, 128, 64, 2048, 4
NUM_PAGES = 9_460_736 // PS
N_LAYERS = 4
CASES = {24: 0.046, 32: 0.160, 40: 0.262, 48: 0.220, 64: 0.214}   # same batch model as tasks/dcp_sparse
SCORED = {32, 40, 48, 64}
PROFILE_BS = 40
SEED = 20261023


def case_id(bs: int) -> str:
    return f"T{6 * bs}_bs{bs}"


def make_pools(n: int, device: str):
    from aiter.ops.cache import indexer_k_quant_and_cache   # production index-K writer
    g = torch.Generator(device=device).manual_seed(SEED)
    pools = []
    chunk = 1 << 20
    for _ in range(n):
        buf = torch.zeros(NUM_PAGES, PS * (D + 4), dtype=torch.uint8, device=device)
        view = buf.view(-1, PS, D + 4).view(torch.float8_e4m3fn)
        for s in range(0, NUM_PAGES * PS, chunk):
            e = min(s + chunk, NUM_PAGES * PS)
            k = torch.randn(e - s, D, device=device, generator=g, dtype=torch.float32).to(torch.bfloat16)
            slots = torch.arange(s, e, device=device, dtype=torch.int64)
            indexer_k_quant_and_cache(k, view, slots, D, "ue8m0", preshuffle=True)
        pools.append(buf)
    return pools


# ---- torch reference of dsa_dcp_split_indexer.py (93034eba91) ------------------------------------------------
def owned_page_table(bt, w, r):
    return bt[:, r::w].contiguous()


def owned_lens(lens, page, w, r):
    lens = lens.to(torch.int64).clamp(min=0)
    full, rem = lens // page, lens % page
    owned_full = ((full - r + w - 1) // w).clamp(min=0)
    tail = torch.where(full % w == r, rem, torch.zeros_like(rem))
    return (owned_full * page + tail).to(torch.int32)


def aiter_logits(q, kv, w, lens, tables, max_len):
    from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
    q4 = q.unsqueeze(1)
    out = torch.empty((q.shape[0], max_len), device=q.device, dtype=torch.float32)
    deepgemm_fp8_paged_mqa_logits(q4, kv, w, out, lens, tables, max_len, Preshuffle=True, KVBlockSize=PS)
    return out


def ref_pack(logits, lens, r):
    if logits.shape[1] < TOPK:
        logits = torch.nn.functional.pad(logits, (0, TOPK - logits.shape[1]), value=-float("inf"))
    col = torch.arange(logits.shape[1], device=logits.device)[None, :]
    masked = torch.where(col < lens[:, None].long(), logits, torch.full_like(logits, -float("inf")))
    score, idx = torch.topk(masked, TOPK, dim=1)
    valid = score > -float("inf")
    pos = ((idx // PS) * DCP + r) * PS + idx % PS
    pos = torch.where(valid, pos, torch.full_like(pos, -1))
    out = torch.empty((2, logits.shape[0], TOPK), dtype=torch.int32, device=logits.device)
    out[0] = torch.where(valid, score, torch.full_like(score, -float("inf"))).contiguous().view(torch.int32)
    out[1] = pos.to(torch.int32)
    return out


def ref_merge_slots(gathered, real_pt):
    packs = gathered.view(DCP, 2, gathered.shape[1], TOPK)
    scores = packs[:, 0].contiguous().view(torch.float32).permute(1, 0, 2).reshape(-1, DCP * TOPK)
    pos = packs[:, 1].permute(1, 0, 2).reshape(-1, DCP * TOPK)
    best, pick = torch.topk(scores, TOPK, dim=1)
    chosen = torch.where(best > -float("inf"), pos.gather(1, pick), torch.full_like(pick, -1).to(pos.dtype))
    p = chosen.to(torch.int64)
    safe = p.clamp(min=0)
    slots = real_pt.to(torch.int64).gather(1, safe // PS) * PS + safe % PS
    return torch.where(p >= 0, slots, torch.full_like(slots, -1)).to(torch.int32)


def make_case(bs: int, ctx: int, pool, device: str):
    t = 6 * bs
    g = torch.Generator(device="cpu").manual_seed(SEED + 100 * bs + ctx)
    L = torch.exp(math.log(89_000) + 0.90 * torch.randn(bs, generator=g)).clamp_(4096, 1_000_000).long()
    pages_per_req = ((L + 6 + PS - 1) // PS).tolist()
    P = max(pages_per_req)
    P = (P + DCP - 1) // DCP * DCP
    real_pt = torch.zeros(t, P, dtype=torch.int32)
    lens = torch.empty(t, dtype=torch.int32)
    for b in range(bs):
        pg = torch.randperm(NUM_PAGES, generator=g)[: pages_per_req[b]].to(torch.int32)
        for i in range(6):
            real_pt[6 * b + i, : pages_per_req[b]] = pg
            lens[6 * b + i] = int(L[b]) + i + 1
    real_pt, lens = real_pt.to(device), lens.to(device)
    gq = torch.Generator(device=device).manual_seed(SEED + 7 * bs + ctx)
    q = torch.randn(t, H, D, device=device, generator=gq).clamp_(-4, 4).to(torch.float8_e4m3fn)
    w = (torch.randn(t, H, device=device, generator=gq) * (H ** -0.5)).float().contiguous()
    kv = pool.view(NUM_PAGES, PS, 1, D + 4)
    packs = []
    for r in range(DCP):
        tab, ln = owned_page_table(real_pt, DCP, r), owned_lens(lens, PS, DCP, r)
        packs.append(ref_pack(aiter_logits(q, kv, w, ln, tab, tab.shape[1] * PS), ln, r))
    ref = ref_merge_slots(torch.cat(packs, 0), real_pt)
    c = dict(q=q, w=w, tab0=owned_page_table(real_pt, DCP, 0), len0=owned_lens(lens, PS, DCP, 0), real_pt=real_pt,
             other=torch.cat(packs[1:], 0).contiguous())
    return c, ref


def run_kernel(c, pool):
    kv = pool.view(NUM_PAGES, PS, 1, D + 4)
    return dcp_indexer_topk(c["q"], kv, c["w"], c["tab0"], c["len0"], c["real_pt"], c["other"], TOPK)


def recall_db(out, ref):
    same, rec = 0, 0.0
    for r in range(ref.shape[0]):
        a = set(out[r][out[r] >= 0].tolist())
        b = set(ref[r][ref[r] >= 0].tolist())
        same += int(a == b)
        rec += len(a & b) / max(1, len(b))
    rec /= ref.shape[0]
    return (100.0 if same == ref.shape[0] else -10 * math.log10(max(1e-10, 1 - rec))), same, rec


def correctness(device: str) -> int:
    pool = make_pools(1, device)[0]
    snrs = []
    for bs in CASES:
        for ctx in (0, 1):
            c, ref = make_case(bs, ctx, pool, device)
            s, same, rec = recall_db(run_kernel(c, pool), ref)
            snrs.append(s)
            print(f"case_snr: {case_id(bs)}_ctx{ctx} {s:.2f}  (identical rows {same}/{ref.shape[0]}, recall {rec:.6f})")
    print(f"SNR: {min(snrs):.2f} dB")
    print(f"allclose: {min(snrs) >= 60.0}")
    return 0


def bench(warmup: int, iters: int, device: str) -> int:
    pools = make_pools(N_LAYERS, device)
    weighted = 0.0
    for bs, p in CASES.items():
        c, _ = make_case(bs, 0, pools[0], device)
        ref_out = run_kernel(c, pools[-1]).clone()
        captured: dict = {}

        def step() -> None:
            for pool in pools:
                captured["out"] = run_kernel(c, pool)

        res = cuda_graph_bench(step, warmup=warmup, iters=iters,
                               dirty=lambda: captured["out"].zero_(),
                               verify=lambda: torch.equal(torch.sort(captured["out"], 1).values,
                                                          torch.sort(ref_out, 1).values))
        times = sorted(t / N_LAYERS for t in res["times_ms"])
        print(f"# {case_id(bs)} mode: {res['mode']}")
        for t in times:
            print(f"wall_ms: {t:.6f}")
        med = times[len(times) // 2]
        weighted += p * med
        tag = "" if bs in SCORED else " unscored"
        print(f"case_ms: {case_id(bs)} {med:.6f}{tag}")
    print(f"median_ms: {weighted:.6f}")
    return 0


def profile(device: str) -> int:
    pool = make_pools(1, device)[0]
    c, _ = make_case(PROFILE_BS, 0, pool, device)
    for _ in range(6):
        run_kernel(c, pool)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GLM-5.2 DCP4 split indexer forge driver")
    ap.add_argument("--bench-mode", action="store_true")
    ap.add_argument("--profile-run", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args, _ = ap.parse_known_args()
    if not torch.cuda.is_available():
        print("error: no GPU available")
        return 1
    device = "cuda"
    if args.profile_run:
        return profile(device)
    if args.bench_mode:
        return bench(args.warmup, args.iters, device)
    return correctness(device)


if __name__ == "__main__":
    sys.exit(main())
