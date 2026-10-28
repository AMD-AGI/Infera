"""Small-shape check of the aiter DSA prefill path (asm mla_decode_fwd per token over its top-k rows) against a torch
reference and the production Triton kernel, to find why the full-size run returned NaN in 0.14 ms.
  python3 debug_aiter_sparse_prefill.py [--seq 64] [--topk 2048] [--rows 200000]"""

import argparse

import torch
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd

from sglang.kernels.ops.attention.dsa import triton_sparse_mla as tsm
from sglang.kernels.ops.attention.dsa.triton_kernel import get_valid_kv_indices

ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, default=64)
ap.add_argument("--heads", type=int, default=64)
ap.add_argument("--topk", type=int, default=2048)
ap.add_argument("--rows", type=int, default=200000)
ap.add_argument("--splits", type=int, default=64)
a = ap.parse_args()
torch.manual_seed(0)
dev = "cuda"
fp8 = torch.float8_e4m3fnuz if tsm._IS_FNUZ else torch.float8_e4m3fn
S, H, DV, DIM, K = a.seq, a.heads, 512, 576, a.topk
kv = torch.empty(a.rows, 1, DIM, device=dev, dtype=fp8)
kv.view(torch.uint8).random_(0, 0x40)
q = (torch.randn(S, H, DIM, device=dev) * 0.5).to(fp8).to(torch.bfloat16)
idx = torch.stack([torch.randperm(a.rows, device=dev)[:K] for _ in range(S)]).to(torch.int32)
scale = DIM ** -0.5

# torch reference
kvf = kv.view(a.rows, DIM).float()
ref = torch.empty(S, H, DV, device=dev)
for s in range(S):
    k = kvf[idx[s].long()]                       # [K, 576]
    att = (q[s].float() @ k.T) * scale           # [H, K]
    ref[s] = torch.softmax(att, -1) @ k[:, :DV]
tri = tsm.triton_sparse_mla_fwd(q[..., :DV], q[..., DV:], kv, idx.unsqueeze(1), scale, DV,
                                topk_length=None, max_topk_length=K).view(S, H, DV).float()
print(f"triton vs ref: max|diff| {(tri - ref).abs().max().item():.4f}")

sizes = get_mla_metadata_info_v1(S, 1, H, torch.bfloat16, fp8, is_sparse=True, fast_mode=False,
                                 num_kv_splits=a.splits, intra_batch_mode=True)
print("metadata buffer sizes:", [(tuple(sz) if hasattr(sz, '__len__') else sz, str(ty)) for sz, ty in sizes])
wm, wi, wis, ri, rfm, rpm = [torch.zeros(sz, dtype=ty, device=dev) for sz, ty in sizes]
cu_q = torch.arange(0, S + 1, dtype=torch.int32, device=dev)
kv_last = torch.ones(S, dtype=torch.int32, device=dev)
kv_indptr = torch.zeros(S + 1, dtype=torch.int32, device=dev)
kv_indptr[1:] = torch.cumsum((idx != -1).sum(dim=1), dim=0)
kv_indices = torch.zeros(S * K, dtype=torch.int32, device=dev)
get_valid_kv_indices(idx, kv_indptr, kv_indices, S)
print("kv_indices match:", torch.equal(kv_indices.view(S, K), idx))
get_mla_metadata_v1(cu_q, kv_indptr, kv_last, H, 1, False, wm, wis, wi, ri, rfm, rpm,
                    page_size=1, kv_granularity=16, max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=False, topk=K,
                    max_split_per_batch=a.splits, intra_batch_mode=True, dtype_q=torch.bfloat16, dtype_kv=fp8)
torch.cuda.synchronize()
print("work_indptr[:8]", wi.flatten()[:8].tolist(), "work_meta_data[:4]", wm.flatten()[:4].tolist())
out = torch.full((S, H, DV), float("nan"), dtype=torch.bfloat16, device=dev)
for kw in ({"kv_scale": torch.ones((), dtype=torch.float32, device=dev)},):
    out.fill_(float("nan"))
    try:
        mla_decode_fwd(q, kv.view(-1, 1, 1, DIM), out, cu_q, kv_indptr, kv_indices, kv_last, 1, sm_scale=scale,
                       logit_cap=0.0, q_scale=None, work_meta_data=wm, work_indptr=wi, work_info_set=wis,
                       reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm, intra_batch_mode=True,
                       num_kv_splits=a.splits, **kw)
        torch.cuda.synchronize()
        o = out.float()
        print(f"aiter kw={list(kw)}: nan frac {torch.isnan(o).float().mean().item():.3f}  "
              f"max|diff vs ref| {(o - ref).nan_to_num(1e9).abs().max().item():.4f}")
    except Exception as e:
        print(f"aiter kw={list(kw)} failed: {type(e).__name__}: {e}")
