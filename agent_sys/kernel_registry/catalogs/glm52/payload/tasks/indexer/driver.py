"""Measurement driver: GLM-5.2-MXFP4 decode-leg DSA top-k indexer (TARGET_VERIFY shape).

forge-loop contract (stdout only):
  * correctness  ``python driver.py``                -> case SNRs + ``SNR: <min> dB``
  * benchmark    ``python driver.py --bench-mode``   -> ``wall_ms`` samples, one ``case_ms`` per case
                                                        (bs8/bs14 ``unscored``), ``median_ms`` aggregate
  * profiling    ``python driver.py --profile-run``  -> only the target op, bs12 case

Production call (sglang v0.5.21 Indexer._get_topk_paged, HIP + aiter preshuffle + sgl-kernel top-k,
SGLANG_OPT_USE_TOPK_V2=false):
  logits = aiter_paged_mqa_logits(q_fp8, kv_cache, weights, seqlens_expanded, real_page_table,
                                  max_seq_len, preshuffle=True, kv_block_size=64)
  topk   = fast_topk_transform_fused(logits, seqlens_expanded, page_table_1, cu_seqlens_q, 2048)
Shapes per attention-DP rank: rows = 6 * bs_graph (one query token per row, causal lengths
L+1..L+6), 32 index heads x 128, index-K pool 34,787 pages x 64 tokens x 132 B (fp8 K in the aiter
16x16 preshuffle layout + fp32 ue8m0 scales), page tables as wide as the decode CUDA graph's
(req_to_token width 1,048,651 = context 1,048,576 + 75 -> 16,386 pages -> logits [rows, 1,048,704]).

Correctness oracle (top-k index sets are not comparable by SNR): fp32 reference scores
  score[t] = sum_h relu(q[h] . K[t]) * w[h]   over the dequantized cache (workset reference),
then, per row: the returned slots must be valid (exactly min(L, 2048) non-negative entries, no
duplicate, every slot one of the row's first L tokens), and the reference scores AT the returned
tokens, sorted descending, are compared with the reference top-2048 scores sorted descending.
Case SNR = 10 log10(sum ref^2 / sum (got - ref)^2) over all rows; any invalid row -> -100 dB.

Timing: one step runs the op over N_LAYERS distinct index-K pools (as the 21 indexer layers of one
decode iteration do); every pool (294 MB) alone exceeds the 256 MB MALL. case_ms = step / N_LAYERS.

This file is the measurement oracle: forge must not edit it.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
from safetensors.torch import load_file

from graph_harness import cuda_graph_bench
from dsa_indexer_kernel import dsa_indexer_topk_paged

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
H, D, PS, TOPK = 32, 128, 64, 2048
NUM_PAGES = 34_787                         # per-rank index-K pool (workset num_pages)
W1 = 1_048_576 + 75                        # req_to_token width: context_len + get_req_to_token_extra_context_len()
W64 = (W1 + PS - 1) // PS                  # 16,386 pages -> max_seq_len 1,048,704
N_LAYERS = 4
# bs_graph -> P(bs_graph) (GLM workset batch model, renormalized over the exported points).
CASES = {8: 0.019908, 10: 0.423407, 12: 0.487302, 14: 0.069383}
SCORED = {10, 12}            # 91 % of the probability mass; 8 and 14 are reported unscored
PROFILE_BS = 12
SEED = 20261020


def case_id(bs: int) -> str:
    return f"T{6 * bs}_bs{bs}"


def _quant_and_cache():
    from aiter.ops.cache import indexer_k_quant_and_cache  # production index-K writer
    return indexer_k_quant_and_cache


def make_pools(n: int, device: str, with_plain: bool = False):
    """n index-K pools written by the production writer (aiter indexer_k_quant_and_cache,
    block 128, ue8m0 scales, preshuffle=True). With with_plain, also the same K written with
    preshuffle=False (the workset reference layout) for the oracle."""
    write = _quant_and_cache()
    g = torch.Generator(device=device).manual_seed(SEED)
    ntok = NUM_PAGES * PS
    slots = torch.arange(ntok, device=device, dtype=torch.int64)
    pools, plains = [], []
    for _ in range(n):
        k = torch.randn(ntok, D, device=device, generator=g, dtype=torch.float32).to(torch.bfloat16)
        buf = torch.zeros(NUM_PAGES, PS * (D + 4), dtype=torch.uint8, device=device)
        write(k, buf.view(-1, PS, D + 4).view(torch.float8_e4m3fn), slots, D, "ue8m0", preshuffle=True)
        pools.append(buf)
        if with_plain:
            plain = torch.zeros_like(buf)
            write(k, plain.view(-1, PS, D + 4).view(torch.float8_e4m3fn), slots, D, "ue8m0", preshuffle=False)
            plains.append(plain)
        del k
    return pools, plains


def make_case(bs: int, ctx: int, device: str):
    rows = 6 * bs
    d = load_file(os.path.join(DATA, f"dsa_topk_indexer_fp8_h32_d128_topk2048_ps64_bs{bs}_ctx{ctx}.safetensors"))
    seq_lens = d["seq_lens"].to(device=device, dtype=torch.int32).contiguous()
    bt = d["block_table"].to(device=device, dtype=torch.int32)
    assert seq_lens.shape == (rows,) and bt.shape[0] == rows, (seq_lens.shape, bt.shape)
    real_page_table = torch.zeros(rows, W64, dtype=torch.int32, device=device)
    real_page_table[:, : bt.shape[1]] = bt
    page_table_1 = torch.zeros(rows, W1, dtype=torch.int32, device=device)
    w = bt.shape[1] * PS
    page_table_1[:, :w] = (bt.unsqueeze(-1) * PS
                           + torch.arange(PS, device=device, dtype=torch.int32)).reshape(rows, w)
    cu_seqlens_q = torch.arange(rows + 1, dtype=torch.int32, device=device)
    g = torch.Generator(device=device).manual_seed(SEED + bs * 10 + ctx)
    q = torch.randn(rows, H, D, device=device, generator=g).clamp_(-4, 4).to(torch.float8_e4m3fn)
    weights = (torch.randn(rows, H, device=device, generator=g) * (H ** -0.5)).float().contiguous()
    return dict(q=q, weights=weights, seq_lens=seq_lens, real_page_table=real_page_table,
                page_table_1=page_table_1, cu_seqlens_q=cu_seqlens_q, block_table=bt)


def run_kernel(c, pool):
    return dsa_indexer_topk_paged(c["q"], pool, c["weights"], c["seq_lens"], c["real_page_table"],
                                  c["page_table_1"], c["cu_seqlens_q"], TOPK)


def dequant_plain(plain: torch.Tensor) -> torch.Tensor:
    """Workset reference dequant (non-shuffled layout): [NUM_PAGES, PS, D] fp32."""
    flat = plain.view(NUM_PAGES, PS * (D + 4))
    vals = flat[:, : PS * D].contiguous().view(torch.float8_e4m3fn).view(NUM_PAGES, PS, D).float()
    scale = flat[:, PS * D:].contiguous().view(torch.float32).view(NUM_PAGES, PS, 1)
    return vals * scale


def check_case(c, out: torch.Tensor, k_all: torch.Tensor):
    """Returns (snr_db, n_invalid_rows, mean_recall)."""
    rows = out.shape[0]
    q = c["q"].float()
    num, den, n_bad, recall = 0.0, 0.0, 0, 0.0
    page_pos = torch.full((NUM_PAGES,), -1, dtype=torch.int64, device=out.device)
    for r in range(rows):
        L = int(c["seq_lens"][r])
        npg = (L + PS - 1) // PS
        pages = c["block_table"][r, :npg].long()
        K = k_all[pages].reshape(-1, D)[:L]
        score = (torch.relu(q[r] @ K.t()) * c["weights"][r][:, None]).sum(0)
        k = min(TOPK, L)
        ref_vals, ref_idx = torch.topk(score, k)
        o = out[r].long()
        sel = o[o >= 0]
        page_pos.fill_(-1)
        page_pos[pages] = torch.arange(npg, device=out.device)
        pos = page_pos[(sel // PS).clamp(0, NUM_PAGES - 1)]
        t = pos * PS + sel % PS
        ok = (sel.numel() == k and int((o == -1).sum()) == TOPK - k
              and sel.unique().numel() == sel.numel() and bool((pos >= 0).all()) and bool((t < L).all()))
        if not ok:
            n_bad += 1
            continue
        got = torch.sort(score[t], descending=True).values
        num += torch.sum(ref_vals.double() ** 2).item()
        den += torch.sum((got.double() - ref_vals.double()) ** 2).item()
        recall += torch.isin(t, ref_idx).float().mean().item()
    if n_bad:
        return -100.0, n_bad, recall / max(1, rows - n_bad)
    snr = 100.0 if den <= 0.0 else 10.0 * math.log10(num / den)
    return snr, 0, recall / rows


def correctness(device: str) -> int:
    pools, plains = make_pools(1, device, with_plain=True)
    k_all = dequant_plain(plains[0])
    del plains
    snrs = []
    for bs in CASES:
        for ctx in (0, 1):
            c = make_case(bs, ctx, device)
            out = run_kernel(c, pools[0])
            torch.cuda.synchronize()
            s, bad, rec = check_case(c, out, k_all)
            snrs.append(s)
            print(f"case_snr: {case_id(bs)}_ctx{ctx} {s:.2f}")
            print(f"# {case_id(bs)}_ctx{ctx} invalid_rows={bad} topk_set_recall={rec:.6f}")
            del c, out
    print(f"SNR: {min(snrs):.2f} dB")
    print(f"allclose: {min(snrs) >= 30.0}")
    return 0


def bench(warmup: int, iters: int, device: str) -> int:
    pools, _ = make_pools(N_LAYERS, device)
    weighted = 0.0
    for bs, p in CASES.items():
        c = make_case(bs, 0, device)
        ref_out = torch.sort(run_kernel(c, pools[-1]), dim=1).values
        captured: dict = {}

        def step() -> None:
            for pool in pools:                         # one call per "layer"
                captured["out"] = run_kernel(c, pool)

        res = cuda_graph_bench(step, warmup=warmup, iters=iters,
                               dirty=lambda: captured["out"].zero_(),
                               verify=lambda: torch.equal(torch.sort(captured["out"], dim=1).values, ref_out))
        times = sorted(t / N_LAYERS for t in res["times_ms"])
        print(f"# {case_id(bs)} mode: {res['mode']}")
        for t in times:
            print(f"wall_ms: {t:.6f}")
        med = times[len(times) // 2]
        weighted += p * med
        tag = "" if bs in SCORED else " unscored"
        print(f"case_ms: {case_id(bs)} {med:.6f}{tag}")
        del c, captured, res
        torch.cuda.empty_cache()
    print(f"median_ms: {weighted:.6f}")
    return 0


def profile(device: str) -> int:
    pools, _ = make_pools(1, device)
    c = make_case(PROFILE_BS, 0, device)
    for _ in range(6):
        run_kernel(c, pools[0])
    torch.cuda.synchronize()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GLM-5.2 DSA top-k indexer forge driver")
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
