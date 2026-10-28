"""Measurement driver: GLM-5.2-MXFP4 decode-leg sparse MLA (TARGET_VERIFY / DRAFT_EXTEND shape).

forge-loop contract (stdout only):
  * correctness  ``python driver.py``                -> case SNRs + ``SNR: <min> dB``
  * benchmark    ``python driver.py --bench-mode``   -> ``wall_ms`` samples, one ``case_ms`` per case
                                                        (bs8/bs14 ``unscored``), ``median_ms`` aggregate
  * profiling    ``python driver.py --profile-run``  -> only the target kernel, bs12 case

Production call (sglang v0.5.21 dsa_backend, dsa_impl == "triton"):
  triton_sparse_mla_fwd(q_nope, q_rope, kv=kv_cache, indices=page_table_1.unsqueeze(1),
                        sm_scale=layer.scaling, d_v=512, topk_length=None,
                        max_topk_length=min(max_seq_len_k, 2048))
Shapes per attention-DP rank: T = 6 * bs_graph query rows (verify / draft-extend), 64 heads,
q_nope fp8 [T,64,512], q_rope fp8 [T,64,64], KV pool fp8 [2,226,368, 1, 576] (34,787 pages x 64,
the deployment's per-rank max_total_num_tokens), indices int32 [T,1,2048] global token slots
(from the GLM workset blobs, contexts > 2048 so no -1 padding), sm_scale 0.0625.

Timing: one step runs the kernel over N_LAYERS distinct KV pools (as the 78 layers of one decode
iteration do), so the working set exceeds the 256 MB MALL; case_ms = step time / N_LAYERS.
Oracle: fp32 torch reference (the workset definition's reference semantics).

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
from sparse_mla_kernel import triton_sparse_mla_fwd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
H, D_V, D_ROPE, TOPK = 64, 512, 64, 2048
NUM_TOKENS_POOL = 34_787 * 64
SM_SCALE = 0.0625
N_LAYERS = 8
# bs_graph -> P(bs_graph) (GLM workset batch model, renormalized over the exported points).
CASES = {8: 0.019908, 10: 0.423407, 12: 0.487302, 14: 0.069383}
SCORED = {10, 12}            # 91 % of the probability mass; 8 and 14 are reported unscored
PROFILE_BS = 12
SEED = 20261020


def case_id(bs: int) -> str:
    return f"T{6 * bs}_bs{bs}"


def make_kv(n: int, device: str) -> list[torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(SEED)
    pools = []
    for _ in range(n):
        x = torch.randn(NUM_TOKENS_POOL, 1, D_V + D_ROPE, device=device, generator=g, dtype=torch.bfloat16)
        pools.append(x.clamp_(-4, 4).to(torch.float8_e4m3fn))
    return pools


def make_case(bs: int, ctx: int, device: str):
    t = 6 * bs
    idx = load_file(os.path.join(DATA, f"dsa_sparse_mla_fp8_h64_ckv512_kpe64_topk2048_ps64_bs{bs}_ctx{ctx}.safetensors"))
    indices = idx["sparse_indices"].to(device=device, dtype=torch.int32)
    assert indices.shape == (t, TOPK), indices.shape
    g = torch.Generator(device=device).manual_seed(SEED + bs * 10 + ctx)
    q_nope = torch.randn(t, H, D_V, device=device, generator=g).clamp_(-4, 4).to(torch.float8_e4m3fn)
    q_rope = torch.randn(t, H, D_ROPE, device=device, generator=g).clamp_(-4, 4).to(torch.float8_e4m3fn)
    return q_nope, q_rope, indices.unsqueeze(1).contiguous()


def run_kernel(q_nope, q_rope, kv, indices):
    return triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, SM_SCALE, d_v=D_V,
                                 topk_length=None, max_topk_length=TOPK)


def reference(q_nope, q_rope, kv, indices) -> torch.Tensor:
    """fp32 sparse MLA: softmax over the 2048 selected tokens, V = latent ckv (first 512 dims)."""
    idx = indices.squeeze(1).long()
    out = torch.empty(idx.shape[0], H, D_V, device=kv.device, dtype=torch.float32)
    kv2 = kv.squeeze(1)
    for r in range(idx.shape[0]):                      # row loop keeps memory small
        k = kv2[idx[r]].float()                        # [TOPK, 576]
        q = torch.cat([q_nope[r].float(), q_rope[r].float()], dim=-1)   # [H, 576]
        p = torch.softmax((q @ k.t()) * SM_SCALE, dim=-1)
        out[r] = p @ k[:, :D_V]
    return out


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
    snrs = []
    for bs in CASES:
        for ctx in (0, 1):
            q_nope, q_rope, indices = make_case(bs, ctx, device)
            out = run_kernel(q_nope, q_rope, kv, indices)
            out = out.reshape(-1, H, D_V)
            s = snr_db(reference(q_nope, q_rope, kv, indices), out)
            snrs.append(s)
            print(f"case_snr: {case_id(bs)}_ctx{ctx} {s:.2f}")
    print(f"SNR: {min(snrs):.2f} dB")
    print(f"allclose: {min(snrs) >= 30.0}")
    return 0


def bench(warmup: int, iters: int, device: str) -> int:
    pools = make_kv(N_LAYERS, device)
    weighted = 0.0
    for bs, p in CASES.items():
        q_nope, q_rope, indices = make_case(bs, 0, device)
        ref_out = run_kernel(q_nope, q_rope, pools[-1], indices).clone()
        captured: dict = {}

        def step() -> None:
            for kv in pools:                           # one call per "layer"
                captured["out"] = run_kernel(q_nope, q_rope, kv, indices)

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
    q_nope, q_rope, indices = make_case(PROFILE_BS, 0, device)
    for _ in range(6):
        run_kernel(q_nope, q_rope, kv, indices)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GLM-5.2 sparse MLA forge driver")
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
