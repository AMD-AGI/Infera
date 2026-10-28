"""Compile the bucketed sparse-MLA kernel (and the production split-dim kernel) on a small problem over a 6.2 GB pool and
count buffer vs global loads in the AMDGCN, to see whether the per-tile branch kept the 32-bit buffer path."""

import re

import torch

import sparse_mla_bucketed as smb
from sglang.kernels.ops.attention.dsa import triton_sparse_mla as tsm

dev = "cuda"
fp8 = torch.float8_e4m3fnuz if tsm._IS_FNUZ else torch.float8_e4m3fn
S, H, DV, DT, DIM, K = 64, 64, 512, 64, 576, 2048
rows = int(6.2e9 / DIM)
kv = torch.empty(rows, 1, DIM, device=dev, dtype=fp8)
kv.view(torch.uint8).random_(0, 0x40)
q = (torch.randn(S, H, DV + DT, device=dev) * 0.5).to(fp8)
idx = torch.randint(0, rows, (S, K), device=dev, dtype=torch.int32)
for bn, st in ((64, 2), (64, 1)):
    smb.sparse_mla_fwd_bucketed(q[..., :DV], q[..., DV:], kv, idx, 0.1, tsm._FP8_MAX, block_n=bn, num_stages=st)
torch.cuda.synchronize()


def counts(asm):
    return {k: len(re.findall(k, asm)) for k in ("buffer_load_dword", "global_load_dword", "s_cbranch", "v_mfma")}


for fn, name in ((smb.sparse_mla_fwd_bucketed_kernel, "bucketed"),):
    for cache in fn.device_caches.values():
        for key, k in cache[0].items():
            asm = k.asm["amdgcn"]
            print(name, k.metadata.num_stages if hasattr(k.metadata, "num_stages") else "", counts(asm),
                  "vgpr", re.findall(r"\.vgpr_count:\s+(\d+)", asm)[:1], "spill", re.findall(r"\.vgpr_spill_count:\s+(\d+)", asm)[:1])
out = tsm.triton_sparse_mla_fwd(q[..., :DV], q[..., DV:], kv, idx.unsqueeze(1), 0.1, DV, topk_length=None, max_topk_length=K)
torch.cuda.synchronize()
for cache in tsm._sparse_mla_fwd_split_dim_kernel.fn.device_caches.values():
    for key, k in cache[0].items():
        asm = k.asm["amdgcn"]
        print("prod split_dim", counts(asm), "vgpr", re.findall(r"\.vgpr_count:\s+(\d+)", asm)[:1])
