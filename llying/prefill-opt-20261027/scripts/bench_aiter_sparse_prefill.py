"""DSA sparse-MLA prefill at GLM-5.2 TP1 prefill shapes: the production Triton kernel (--dsa-prefill-backend triton) vs
the aiter path (--dsa-prefill-backend aiter = DSABackend._forward_aiter_extend: every prefill token is a decode query of
the asm mla_decode_fwd over its own top-k rows, persistent FP8 scheduling from get_mla_metadata_v1). Same q (bf16) and
FP8 KV pool for both; reports the per-call time of each (aiter with and without its per-layer index / metadata prep)
and the output difference.
  python3 bench_aiter_sparse_prefill.py [--seq 8192] [--pool-gb 6.2] [--iters 10]"""

import argparse

import torch
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd

from sglang.kernels.ops.attention.dsa import triton_sparse_mla as tsm
from sglang.kernels.ops.attention.dsa.triton_kernel import get_valid_kv_indices

ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, default=8192)
ap.add_argument("--reqs", type=int, default=4)
ap.add_argument("--ctx", type=int, default=131072)
ap.add_argument("--heads", type=int, default=64)
ap.add_argument("--topk", type=int, default=2048)
ap.add_argument("--overlap", type=float, default=0.7)
ap.add_argument("--pool-gb", type=float, default=6.2)
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--splits", type=int, default=64)
a = ap.parse_args()

torch.manual_seed(0)
dev = "cuda"
fp8 = torch.float8_e4m3fnuz if tsm._IS_FNUZ else torch.float8_e4m3fn
S, H, DV, DT, DIM, K = a.seq, a.heads, 512, 64, 576, a.topk
rows = max(a.reqs * a.ctx, int(a.pool_gb * 1e9 / DIM))
kv = torch.empty(rows, 1, DIM, device=dev, dtype=fp8)
kv.view(torch.uint8).random_(0, 0x40)
q = (torch.randn(S, H, DIM, device=dev) * 0.5).to(fp8).to(torch.bfloat16)
per_req = S // a.reqs
idx = torch.empty(S, K, dtype=torch.int32, device=dev)
phys = torch.randperm(rows, device=dev)[: a.reqs * a.ctx].to(torch.int32)  # request tokens scattered over the pool
for r in range(a.reqs):
    rnd = torch.randint(0, a.ctx, (per_req, K), device=dev, dtype=torch.int32)
    hot = torch.randperm(a.ctx, device=dev)[:K].to(torch.int32)
    rnd[:, : int(a.overlap * K)] = hot[: int(a.overlap * K)]
    idx[r * per_req:(r + 1) * per_req] = phys[(r * a.ctx + rnd).long()]
scale = DIM ** -0.5


def bench(fn):
    for _ in range(2):
        out = fn()
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    for _ in range(a.iters):
        out = fn()
    ev[1].record()
    torch.cuda.synchronize()
    return ev[0].elapsed_time(ev[1]) / a.iters, out


def triton_prod():
    return tsm.triton_sparse_mla_fwd(q[..., :DV], q[..., DV:], kv, idx.unsqueeze(1), scale, DV,
                                     topk_length=None, max_topk_length=K).view(S, H, DV)


meta_sizes = get_mla_metadata_info_v1(S, 1, H, torch.bfloat16, fp8, is_sparse=True, fast_mode=False,
                                      num_kv_splits=a.splits, intra_batch_mode=True)
wm, wi, wis, ri, rfm, rpm = [torch.zeros(sz, dtype=ty, device=dev) for sz, ty in meta_sizes]
print("metadata buffer sizes:", [(tuple(sz) if hasattr(sz, "__len__") else sz) for sz, _ in meta_sizes], flush=True)
cu_q = torch.arange(0, S + 1, dtype=torch.int32, device=dev)
kv_last = torch.ones(S, dtype=torch.int32, device=dev)
kv_scale = torch.ones((), dtype=torch.float32, device=dev)
kv_indptr = torch.zeros(S + 1, dtype=torch.int32, device=dev)
kv_indices = torch.zeros(S * K, dtype=torch.int32, device=dev)
out_a = torch.empty(S, H, DV, dtype=torch.bfloat16, device=dev)


def aiter_prep():
    kv_indptr[1:] = torch.cumsum((idx != -1).sum(dim=1), dim=0)
    get_valid_kv_indices(idx, kv_indptr, kv_indices, S)
    get_mla_metadata_v1(cu_q, kv_indptr, kv_last, H, 1, False, wm, wis, wi, ri, rfm, rpm,
                        page_size=1, kv_granularity=16, max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=False, topk=K,
                        max_split_per_batch=a.splits, intra_batch_mode=True, dtype_q=torch.bfloat16, dtype_kv=fp8)


def aiter_kernel():
    mla_decode_fwd(q, kv.view(-1, 1, 1, DIM), out_a, cu_q, kv_indptr, kv_indices, kv_last, 1, sm_scale=scale,
                   logit_cap=0.0, q_scale=None, kv_scale=kv_scale, work_meta_data=wm, work_indptr=wi,
                   work_info_set=wis, reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
                   intra_batch_mode=True, num_kv_splits=a.splits)
    return out_a


def aiter_full():
    aiter_prep()
    return aiter_kernel()


t_t, ref = bench(triton_prod)
ref = ref.float()
print(f"seq {S} H {H} topk {K} pool {rows * DIM / 1e9:.1f} GB (request tokens scattered), overlap {a.overlap}")
print(f"triton (production)    {t_t:7.3f} ms")
t_p, _ = bench(aiter_prep)
print(f"aiter index + metadata {t_p:7.3f} ms")
aiter_prep()
t_k, out = bench(aiter_kernel)
d = (out.float() - ref).abs()
print(f"aiter mla_decode_fwd   {t_k:7.3f} ms  x{t_t / t_k:.2f}  max|diff| {d.max().item():.4f}  mean|diff| {d.mean().item():.5f}  "
      f">2% of max: {(d > 0.02 * ref.abs().max()).float().mean().item():.2e}")
t_f, _ = bench(aiter_full)
print(f"aiter total            {t_f:7.3f} ms  x{t_t / t_f:.2f}")
