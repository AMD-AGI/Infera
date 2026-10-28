#!/usr/bin/env python3
"""Forge driver: GLM-5.2-MXFP4 DCP4 decode sparse MLA (split-K, owned slots + LSE) on MI355X.

Contract (KernelForge):
  * correctness  ``python driver.py``                -> ``case_snr: <id> <dB>`` lines, ``SNR: <min> dB``
  * benchmark    ``python driver.py --bench-mode``   -> ``wall_ms`` samples, one ``case_ms`` per case
  * profile      ``python driver.py --profile-run``  -> a few calls of the profile case

What it measures: ``dcp_sparse_mla_decode`` = DeepseekSparseAttnBackend._forward_triton_dcp's kernel call
(sglang 93034eba91, dsa_backend.py:3632) for the TARGET_VERIFY rows of the DCP4 decode leg (TP4 EP4 DP1 dcp 4, the
ppdc config of historical-e2e-workspace). One DCP rank (rank 0) is modelled:
  T = 6 x bs_graph query rows, 64 heads (16 local x 4 all-gathered), q fp8 [T, 64, 512] + [T, 64, 64];
  the global top-2048 slots of a row are localized (transform_index_page_table_dcp): slots with slot % 4 == 0 are
  owned, stored as slot // 4 packed first, then -1; lengths = owned count (~512); KV = this rank's shard
  [2,365,184 local slots, 1, 576] fp8 (9.46 M tokens / 4, ppdc decode max_total_num_tokens).
Output: bf16 [1, T, 64, 512] + fp32 base-2 LSE [T, 64] (merged across ranks by the caller).
Rows' contexts: lognormal lengths (median 89k, p90 283k tokens, AgentX ISL), 2048 distinct random slots per row.
Timing: one step runs the kernel over N_LAYERS distinct KV shards (78 layers per decode iteration in the model);
case_ms = step / N_LAYERS. Oracle: fp32 torch over the owned slots (output SNR and LSE error).

This file is the measurement oracle: forge must not edit it.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

from graph_harness import cuda_graph_bench
from dcp_sparse_kernel import dcp_sparse_mla_decode

H, D_V, D_ROPE, TOPK, DCP = 64, 512, 64, 2048, 4
GLOBAL_SLOTS = 9_460_736
LOCAL_SLOTS = GLOBAL_SLOTS // DCP
SM_SCALE = 0.0625
N_LAYERS = 8
# bs_graph -> P (DCP decode running requests snapped to the capture list, ppdc1 C56 A1 + ppdc2 C72 A2, 50/50).
CASES = {24: 0.046, 32: 0.160, 40: 0.262, 48: 0.220, 64: 0.214}
SCORED = {32, 40, 48, 64}
PROFILE_BS = 40
SEED = 20261023
LSE_TOL = 0.05            # |lse - ref| in log2 units (= 3.5 % in probability mass)


def case_id(bs: int) -> str:
    return f"T{6 * bs}_bs{bs}"


def make_kv(n: int, device: str) -> list[torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(SEED)
    pools = []
    for _ in range(n):
        x = torch.randn(LOCAL_SLOTS, 1, D_V + D_ROPE, device=device, generator=g, dtype=torch.bfloat16)
        pools.append(x.clamp_(-4, 4).to(torch.float8_e4m3fn))
    return pools


def make_case(bs: int, ctx: int, device: str):
    t = 6 * bs
    g = torch.Generator(device="cpu").manual_seed(SEED + 100 * bs + ctx)
    # per request (6 verify rows share a context of length L + i): lognormal L, median 89k, sigma 0.90
    lens = torch.exp(math.log(89_000) + 0.90 * torch.randn(bs, generator=g)).clamp_(4096, 1_000_000).long()
    rows = []
    for b in range(bs):
        start = int(torch.randint(0, GLOBAL_SLOTS - int(lens[b]), (1,), generator=g))
        for _ in range(6):
            rows.append(start + torch.randperm(int(lens[b]), generator=g)[:TOPK])
    glob = torch.stack(rows).to(torch.int64)                       # [T, 2048] global slots
    owned = glob % DCP == 0
    counts = owned.sum(1).to(torch.int32)
    local = torch.full((t, TOPK), -1, dtype=torch.int32)
    for r in range(t):
        s = (glob[r][owned[r]] // DCP).to(torch.int32)
        local[r, : s.numel()] = s
    gq = torch.Generator(device=device).manual_seed(SEED + 7 * bs + ctx)
    q_nope = torch.randn(t, H, D_V, device=device, generator=gq).clamp_(-4, 4).to(torch.float8_e4m3fn)
    q_rope = torch.randn(t, H, D_ROPE, device=device, generator=gq).clamp_(-4, 4).to(torch.float8_e4m3fn)
    return q_nope, q_rope, local.to(device).unsqueeze(1).contiguous(), counts.to(device)


def run_kernel(q_nope, q_rope, kv, indices, lengths, workspace=None):
    return dcp_sparse_mla_decode(q_nope, q_rope, kv, indices, lengths, SM_SCALE, d_v=D_V, workspace=workspace)


def reference(q_nope, q_rope, kv, indices, lengths):
    """fp32: softmax over the owned slots only; LSE base 2 of the scaled scores (empty row -> -2**30)."""
    t = indices.shape[0]
    out = torch.zeros(t, H, D_V, device=kv.device, dtype=torch.float32)
    lse = torch.full((t, H), -1073741824.0, device=kv.device, dtype=torch.float32)
    kv2 = kv.squeeze(1)
    for r in range(t):
        n = int(lengths[r])
        if n == 0:
            continue
        k = kv2[indices[r, 0, :n].long()].float()
        q = torch.cat([q_nope[r].float(), q_rope[r].float()], dim=-1)
        s = (q @ k.t()) * SM_SCALE
        out[r] = torch.softmax(s, dim=-1) @ k[:, :D_V]
        lse[r] = torch.logsumexp(s, dim=-1) / math.log(2.0)
    return out, lse


def snr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    ref, test = ref.float(), test.float()
    noise = torch.mean((test - ref) ** 2).item()
    signal = torch.mean(ref ** 2).item()
    if not math.isfinite(noise):
        return -100.0
    if noise <= 0.0:
        return 100.0
    return 10.0 * math.log10(signal / noise) if signal > 0 else 0.0


def correctness(device: str) -> int:
    kv = make_kv(1, device)[0]
    snrs, lse_ok = [], True
    for bs in CASES:
        for ctx in (0, 1):
            qn, qr, idx, ln = make_case(bs, ctx, device)
            out, lse = run_kernel(qn, qr, kv, idx, ln)
            ro, rl = reference(qn, qr, kv, idx, ln)
            s = snr_db(ro, out.reshape(-1, H, D_V))
            err = (lse.float() - rl).abs().max().item()
            lse_ok &= err <= LSE_TOL
            snrs.append(s if err <= LSE_TOL else -100.0)
            print(f"case_snr: {case_id(bs)}_ctx{ctx} {snrs[-1]:.2f}  (lse max err {err:.4f}, owned mean "
                  f"{ln.float().mean().item():.0f})")
    print(f"SNR: {min(snrs):.2f} dB")
    print(f"allclose: {lse_ok}")
    return 0


def bench(warmup: int, iters: int, device: str) -> int:
    pools = make_kv(N_LAYERS, device)
    weighted = 0.0
    ws: list = []
    for bs, p in CASES.items():
        qn, qr, idx, ln = make_case(bs, 0, device)
        ref_out = run_kernel(qn, qr, pools[-1], idx, ln, ws)[0].clone()
        captured: dict = {}

        def step() -> None:
            for kv in pools:
                captured["out"] = run_kernel(qn, qr, kv, idx, ln, ws)[0]

        res = cuda_graph_bench(step, warmup=warmup, iters=iters,
                               dirty=lambda: captured["out"].zero_(),
                               verify=lambda: torch.equal(captured["out"], ref_out))
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
    kv = make_kv(1, device)[0]
    qn, qr, idx, ln = make_case(PROFILE_BS, 0, device)
    for _ in range(6):
        run_kernel(qn, qr, kv, idx, ln)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GLM-5.2 DCP4 sparse MLA decode forge driver")
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
