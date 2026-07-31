"""Minimal repro of the aiter DSA decode kernel at GLM-5.2 shapes.

Mirrors sglang dsa_backend._forward_aiter exactly (incl. the head 8->16 pad and the
persistent metadata buffers sized ONCE at max_bs), then sweeps bs to see which one
faults. Single GPU, a few hundred MB. ~1 min instead of a 10-min server cold start.

GLM-5.2: 64 heads / TP8 = 8 q-heads, kv_lora_rank=512, qk_rope=64 -> head_dim 576,
v_head_dim 512, index_topk 2048, kv dtype fp8_e4m3, q dtype bf16.
"""
import os, sys, traceback
import torch

MAX_BS = int(os.environ.get("MAX_BS", 64))
BS_LIST = [int(x) for x in os.environ.get("BS_LIST", "64,56,48,40,32,24,16,12,8,4,2,1").split(",")]
TOPK = int(os.environ.get("TOPK", 2048))
SEQ = int(os.environ.get("SEQ", 4096))
NUM_Q_HEADS = int(os.environ.get("NUM_Q_HEADS", 8))
KV_LORA, QK_ROPE = 512, 64
HEAD_DIM = KV_LORA + QK_ROPE     # 576
V_HEAD_DIM = KV_LORA             # 512
MAX_SPLIT = int(os.environ.get("MAX_SPLIT", 64))

fp8 = torch.float8_e4m3fn
dev = "cuda"

from aiter.mla import mla_decode_fwd
from aiter.ops.attention import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.jit.utils.chip_info import get_gfx, get_cu_num

need_pad = NUM_Q_HEADS < 16
rf = 16 // NUM_Q_HEADS if need_pad else 1
NHEAD_PAD = NUM_Q_HEADS * rf

print(f"gfx={get_gfx()} cu={get_cu_num()}")
print(f"num_q_heads={NUM_Q_HEADS} need_pad={need_pad} rf={rf} padded={NHEAD_PAD}")
print(f"head_dim={HEAD_DIM} v_head_dim={V_HEAD_DIM} topk={TOPK} max_bs={MAX_BS} max_split={MAX_SPLIT}")

# ---- KV pool (raw MLA layout, fp8, page_size=1) -------------------------------
NUM_TOK = MAX_BS * SEQ + 1024
kv_cache = torch.zeros((NUM_TOK, 1, HEAD_DIM), dtype=fp8, device=dev)
print(f"kv_cache {tuple(kv_cache.shape)} {kv_cache.dtype} "
      f"{kv_cache.numel()*kv_cache.element_size()/2**30:.2f} GiB")

# ---- persistent metadata buffers, sized ONCE at MAX_BS (as sglang does) -------
def make_meta(batch_size, max_seqlen_q=1):
    sizes = get_mla_metadata_info_v1(
        batch_size, max_seqlen_q, NHEAD_PAD, torch.bfloat16, fp8,
        is_sparse=True, fast_mode=False,
        num_kv_splits=MAX_SPLIT, intra_batch_mode=True,
    )
    return [torch.empty(s, dtype=t, device=dev) for (s, t) in sizes]

meta = make_meta(MAX_BS)
(work_metadata, work_indptr, work_info_set,
 reduce_indptr, reduce_final_map, reduce_partial_map) = meta
print("metadata buffer shapes @max_bs:",
      [tuple(m.shape) if m.dim() else (m.numel(),) for m in meta])

kv_last_page_lens_buf = torch.ones((MAX_BS,), dtype=torch.int32, device=dev)

def run_one(bs):
    torch.cuda.synchronize()
    q = torch.randn((bs, NHEAD_PAD, HEAD_DIM), dtype=torch.bfloat16, device=dev)
    o = torch.empty((bs, NHEAD_PAD, V_HEAD_DIM), dtype=torch.bfloat16, device=dev)

    cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device=dev)
    # each request selected exactly TOPK tokens (dense worst case)
    kv_indptr = torch.arange(0, (bs + 1) * TOPK, TOPK, dtype=torch.int32, device=dev)
    kv_indices = torch.randint(0, NUM_TOK - 1, (bs * TOPK,), dtype=torch.int32, device=dev)

    kv_last_page_lens_buf[:bs].fill_(1)
    klpl = kv_last_page_lens_buf[:bs]

    get_mla_metadata_v1(
        cu_seqlens_q, kv_indptr, klpl,
        NHEAD_PAD, 1, False,
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=1, kv_granularity=16,
        max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=False,
        topk=TOPK, max_split_per_batch=MAX_SPLIT, intra_batch_mode=True,
        dtype_q=torch.bfloat16, dtype_kv=fp8,
    )
    torch.cuda.synchronize()

    kv_scale = torch.ones((), dtype=torch.float32, device=dev)
    mla_decode_fwd(
        q, kv_cache.view(-1, 1, 1, HEAD_DIM), o,
        cu_seqlens_q, kv_indptr, kv_indices, klpl,
        1,  # max_seqlen_q
        sm_scale=1.0 / (HEAD_DIM ** 0.5), logit_cap=0.0,
        q_scale=None, kv_scale=kv_scale,
        work_meta_data=work_metadata, work_indptr=work_indptr,
        work_info_set=work_info_set, reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map, reduce_partial_map=reduce_partial_map,
        intra_batch_mode=True, num_kv_splits=MAX_SPLIT,
    )
    torch.cuda.synchronize()
    bad = (~torch.isfinite(o)).sum().item()
    return o.float().abs().mean().item(), bad

fails = []
for bs in BS_LIST:
    try:
        m, bad = run_one(bs)
        print(f"  bs={bs:4d}  OK   mean|o|={m:.5f}  nonfinite={bad}", flush=True)
        if bad:
            fails.append((bs, f"nonfinite={bad}"))
    except Exception as e:
        print(f"  bs={bs:4d}  FAIL {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        fails.append((bs, repr(e)))
        break

print("\nRESULT:", "ALL_OK" if not fails else f"FAILURES {fails}")
