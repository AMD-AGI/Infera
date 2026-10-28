#!/usr/bin/env python3
"""Forge driver: GLM-5.2 gfx950 FUSED decode DSA indexer (TARGET_VERIFY rows 6..48) on MI355X.

Contract (KernelForge):
  * correctness  ``python driver.py``                -> ``case_snr: <id> <dB>`` lines, ``SNR: <min> dB``
  * benchmark    ``python driver.py --bench-mode``   -> ``wall_ms`` samples, one ``case_ms`` per case
  * profile      ``python driver.py --profile-run``  -> a few calls of the profile case

What it measures: ``fused_decode_indexer`` (= sglang Gfx950FusedIndexer.run, 4 kernels) for the TARGET_VERIFY rows
of a DPA decode rank holding bs = 1..8 requests (rows = 6 x bs <= 48: every such call goes to this path, not to the
standard indexer). Per request: context L ~ lognormal (median 180k tokens, sigma 0.9, clipped to [4096, 600k]);
its 6 verify rows have positions L + i and lengths L + i + 1 (i = 0..5), and write their K at those positions.
Index-K pool per rank: 34,787 pages x 64 tokens x (128 fp8 + 4 B scale), written by the production writer.
Oracle: the image's own production kernels (sglang kernels/ops/attention/dsa/hip_gfx950/loader.py) run on the same
inputs: every row's top-2048 slot SET must be identical, and so must the bytes of the K slots written this step.
case_snr = 100 dB if all rows and the written K match, else -10*log10(1 - mean recall) (K mismatch -> -100 dB).
Timing: N_LAYERS index pools per CUDA-graph replay; case_ms = replay / N_LAYERS.

This file is the measurement oracle: forge must not edit it.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

# sglang's JIT cache publishes builds with rename(), which the NFS home refuses here: keep it node-local.
os.environ.setdefault("SGLANG_JIT_CACHE_DIR", "REDACTED_LOCAL_PATH")

import torch  # noqa: E402

from graph_harness import cuda_graph_bench  # noqa: E402
from fused_indexer_kernel import Workspace, fused_decode_indexer  # noqa: E402

H, D, PS, TOPK = 32, 128, 64, 2048
NUM_PAGES = 34_787
N_LAYERS = 4
MAX_POS = 1 << 20
# bs -> P (per-rank decode batch of the same-node dpax1 runs, A2 windows, bs 1..8 renormalised, pairs merged).
CASES = {2: 0.24, 4: 0.31, 6: 0.29, 8: 0.16}
SCORED = {2, 4, 6, 8}
PROFILE_BS = 6
SEED = 20261024
EPS = 1e-6
WSCALE = (32 ** -0.5) * (128 ** -0.5)


def case_id(bs: int) -> str:
    return f"T{6 * bs}_bs{bs}"


@torch.no_grad()
def make_static(device: str):
    g = torch.Generator(device=device).manual_seed(SEED)
    w_q_b = (torch.randn(4096, 2048, device=device, generator=g) * 0.02).to(torch.bfloat16).contiguous()
    w_kw = (torch.randn(160, 6144, device=device, generator=g) * 0.02).to(torch.bfloat16).contiguous()
    norm_w = (1 + 0.1 * torch.randn(128, device=device, generator=g)).float().contiguous()
    norm_b = (0.05 * torch.randn(128, device=device, generator=g)).float().contiguous()
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64))
    ang = torch.arange(MAX_POS, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    cos2d, sin2d = ang.cos().to(torch.bfloat16).contiguous(), ang.sin().to(torch.bfloat16).contiguous()
    return dict(w_q_b=w_q_b, w_kw=w_kw, norm_w=norm_w, norm_b=norm_b, cos2d=cos2d, sin2d=sin2d)


@torch.no_grad()
def make_pools(n: int, device: str):
    from aiter.ops.cache import indexer_k_quant_and_cache
    g = torch.Generator(device=device).manual_seed(SEED + 1)
    pools, chunk = [], 1 << 20
    for _ in range(n):
        buf = torch.zeros(NUM_PAGES, PS * (D + 4), dtype=torch.uint8, device=device)
        view = buf.view(-1, PS, D + 4).view(torch.float8_e4m3fn)
        for s in range(0, NUM_PAGES * PS, chunk):
            e = min(s + chunk, NUM_PAGES * PS)
            k = torch.randn(e - s, D, device=device, generator=g, dtype=torch.float32).to(torch.bfloat16)
            indexer_k_quant_and_cache(k, view, torch.arange(s, e, device=device, dtype=torch.int64), D, "ue8m0",
                                      preshuffle=True)
        pools.append(buf)
    return pools


def make_case(bs: int, ctx: int, device: str):
    rows = 6 * bs
    g = torch.Generator(device="cpu").manual_seed(SEED + 100 * bs + ctx)
    L = torch.exp(math.log(180_000) + 0.90 * torch.randn(bs, generator=g)).clamp_(4096, 600_000).long()
    npg = ((L + 6 + PS - 1) // PS).tolist()
    W = (max(npg) + 63) // 64 * 64
    perm = torch.randperm(NUM_PAGES, generator=g)
    pt = torch.zeros(rows, W, dtype=torch.int32)
    pos = torch.empty(rows, dtype=torch.int64)
    loc = torch.empty(rows, dtype=torch.int64)
    lens = torch.empty(rows, dtype=torch.int32)
    off = 0
    for b in range(bs):
        pages = perm[off: off + npg[b]].to(torch.int32)
        off += npg[b]
        for i in range(6):
            r = 6 * b + i
            pt[r, : npg[b]] = pages
            p = int(L[b]) + i
            pos[r], lens[r] = p, p + 1
            loc[r] = int(pages[p // PS]) * PS + p % PS
    assert off <= NUM_PAGES, "page budget"
    gd = torch.Generator(device=device).manual_seed(SEED + 7 * bs + ctx)
    q_lora = torch.randn(rows, 2048, device=device, generator=gd).to(torch.bfloat16)
    x = torch.randn(rows, 6144, device=device, generator=gd).to(torch.bfloat16)
    return dict(q_lora=q_lora, x=x, positions=pos.to(device), loc=loc.to(device), lens=lens.to(device),
                pt=pt.to(device).contiguous(), W=W)


def run_with(fn, ws, st, c, pool):
    kv = pool.view(NUM_PAGES, PS, D + 4).view(torch.float8_e4m3fn)            # write view (qk kernel)
    kv_read = pool.view(NUM_PAGES, PS * (D + 4)).view(torch.float8_e4m3fn)    # read view (logits kernel)
    return fn(ws=ws, q_lora=c["q_lora"], x_bf16=c["x"], w_q_b=st["w_q_b"], w_kw=st["w_kw"], norm_w=st["norm_w"],
              norm_b=st["norm_b"], cos2d=st["cos2d"], sin2d=st["sin2d"], positions=c["positions"],
              out_cache_loc=c["loc"], kv_cache=kv, kv_cache_read=kv_read, seqlens=c["lens"], page_table_64=c["pt"],
              eps=EPS, weights_scale=WSCALE)


def production(*, ws, q_lora, x_bf16, w_q_b, w_kw, norm_w, norm_b, cos2d, sin2d, positions, out_cache_loc,
               kv_cache, kv_cache_read, seqlens, page_table_64, eps, weights_scale):
    """The image's own kernels (sglang hip_gfx950/loader.py) with Gfx950FusedIndexer.run's marshalling."""
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import loader
    gemv, qk, logits_mod, topk_mod = loader.modules()
    rows = q_lora.shape[0]
    cols = page_table_64.shape[1] * PS
    q_fp8, head_gate, logits = ws.q_fp8[:rows], ws.head_gate[:rows], ws.logits[:rows, :cols]
    q_proj, kw = ws.q_proj[:rows], ws.kw[:rows]
    gemv.dual_gemv_bf16(q_lora, w_q_b, q_proj, x_bf16, w_kw, kw)
    qk.indexer_qk_rope_hadamard_quant_and_cache(q_proj.view(rows, H, D), q_fp8, kw[:, D:], head_gate, kw[:, :D],
                                                kv_cache, out_cache_loc, norm_w, norm_b, positions, cos2d, sin2d,
                                                float(eps), float(weights_scale))
    out = torch.empty((rows, TOPK), dtype=torch.int32, device=logits.device)
    logits_mod.logits_hist(q_fp8, kv_cache_read, head_gate, seqlens, page_table_64, logits, ws.ghist[:rows],
                           loader.LOGITS_BLOCKS_PER_ROW)
    topk_mod.topk_transform(logits, seqlens, page_table_64, out, ws.ghist[:rows], ws.cursor[:rows],
                            ws.cand_idx[:rows], ws.cand_val[:rows], loader.TOPK_G, PS)
    return out


def compare(out, ref, pool_kf, pool_ref, loc):
    same, rec = 0, 0.0
    for r in range(ref.shape[0]):
        a, b = set(out[r][out[r] >= 0].tolist()), set(ref[r][ref[r] >= 0].tolist())
        same += int(a == b)
        rec += len(a & b) / max(1, len(b))
    rec /= ref.shape[0]
    flat_kf = pool_kf.view(NUM_PAGES * PS, D + 4)[loc]
    flat_ref = pool_ref.view(NUM_PAGES * PS, D + 4)[loc]
    k_ok = bool(torch.equal(flat_kf, flat_ref))
    if not k_ok:
        return -100.0, same, rec, k_ok
    return (100.0 if same == ref.shape[0] else -10 * math.log10(max(1e-10, 1 - rec))), same, rec, k_ok


def correctness(device: str) -> int:
    st = make_static(device)
    base = make_pools(1, device)[0]
    snrs = []
    for bs in CASES:
        for ctx in (0, 1):
            c = make_case(bs, ctx, device)
            p_kf, p_ref = base.clone(), base.clone()
            out = run_with(fused_decode_indexer, Workspace(device=device, max_cols=c["W"] * PS), st, c, p_kf)
            ref = run_with(production, Workspace(device=device, max_cols=c["W"] * PS), st, c, p_ref)
            torch.cuda.synchronize()
            s, same, rec, k_ok = compare(out, ref, p_kf, p_ref, c["loc"])
            snrs.append(s)
            print(f"case_snr: {case_id(bs)}_ctx{ctx} {s:.2f}  (identical rows {same}/{ref.shape[0]}, recall {rec:.6f},"
                  f" K write {'ok' if k_ok else 'MISMATCH'}, max ctx {int(c['lens'].max())})")
            del p_kf, p_ref
    print(f"SNR: {min(snrs):.2f} dB")
    print(f"allclose: {min(snrs) >= 60.0}")
    return 0


def bench(warmup: int, iters: int, device: str) -> int:
    st = make_static(device)
    pools = make_pools(N_LAYERS, device)
    weighted = 0.0
    for bs, p in CASES.items():
        c = make_case(bs, 0, device)
        ws = Workspace(device=device, max_cols=c["W"] * PS)
        ref_out = run_with(fused_decode_indexer, ws, st, c, pools[-1]).clone()
        captured: dict = {}

        def step() -> None:
            for pool in pools:
                captured["out"] = run_with(fused_decode_indexer, ws, st, c, pool)

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
    st = make_static(device)
    pool = make_pools(1, device)[0]
    c = make_case(PROFILE_BS, 0, device)
    ws = Workspace(device=device, max_cols=c["W"] * PS)
    for _ in range(6):
        run_with(fused_decode_indexer, ws, st, c, pool)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GLM-5.2 gfx950 fused decode indexer forge driver")
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
