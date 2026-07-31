"""Repro v2: real (nonzero) KV + sglang's real index path (page_table with -1 padding
-> get_valid_kv_indices), sweeping bs AND short seqlens. v1 (dense topk, zero KV) passed
everything, so the trigger must be in the sparse/ragged index structure.

Key difference from v1: sglang builds kv_indptr from `page_table_1 != -1` counts, so
early in decode a request has FEWER than topk valid entries -> ragged kv_indptr, and
kv_indices is a fixed max-size buffer only partially filled.
"""
import os, traceback
import torch

MAX_BS   = int(os.environ.get("MAX_BS", 64))
BS_LIST  = [int(x) for x in os.environ.get("BS_LIST", "64,56,48,40,32,16,8,1").split(",")]
TOPK     = int(os.environ.get("TOPK", 2048))
# seqlen per request; when < TOPK the page_table is -1 padded (the realistic warmup case)
SEQ_LIST = [int(x) for x in os.environ.get("SEQ_LIST", "1,7,64,513,2048,4096").split(",")]
NUM_Q_HEADS = int(os.environ.get("NUM_Q_HEADS", 8))
KV_LORA, QK_ROPE = 512, 64
HEAD_DIM, V_HEAD_DIM = KV_LORA + QK_ROPE, KV_LORA
MAX_SPLIT = int(os.environ.get("MAX_SPLIT", 64))

fp8, dev = torch.float8_e4m3fn, "cuda"
from aiter.mla import mla_decode_fwd
from aiter.ops.attention import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.jit.utils.chip_info import get_gfx
from sglang.kernels.ops.attention.dsa.triton_kernel import get_valid_kv_indices

need_pad = NUM_Q_HEADS < 16
rf = 16 // NUM_Q_HEADS if need_pad else 1
NHEAD_PAD = NUM_Q_HEADS * rf
print(f"gfx={get_gfx()} nq={NUM_Q_HEADS} padded={NHEAD_PAD} topk={TOPK} max_bs={MAX_BS}")

NUM_TOK = MAX_BS * max(SEQ_LIST) + 4096
kv_cache = torch.randn((NUM_TOK, 1, HEAD_DIM), dtype=torch.bfloat16, device=dev).to(fp8)
print(f"kv_cache {tuple(kv_cache.shape)} nonzero-init "
      f"{kv_cache.numel()*1/2**30:.2f} GiB")

sizes = get_mla_metadata_info_v1(MAX_BS, 1, NHEAD_PAD, torch.bfloat16, fp8,
                                 is_sparse=True, fast_mode=False,
                                 num_kv_splits=MAX_SPLIT, intra_batch_mode=True)
(work_metadata, work_indptr, work_info_set,
 reduce_indptr, reduce_final_map, reduce_partial_map) = [
     torch.empty(s, dtype=t, device=dev) for (s, t) in sizes]
print("meta shapes:", [tuple(x.shape) for x in
      (work_metadata, work_indptr, work_info_set, reduce_indptr,
       reduce_final_map, reduce_partial_map)])

# sglang allocates these ONCE at max_bs (dsa_backend __init__)
kv_indptr_buf  = torch.zeros((MAX_BS + 1,), dtype=torch.int32, device=dev)
kv_indices_buf = torch.zeros(MAX_BS * TOPK, dtype=torch.int32, device=dev)
klpl_buf       = torch.ones((MAX_BS,), dtype=torch.int32, device=dev)

REBUILD = os.environ.get("REBUILD", "0") == "1"

def _mk(bs):
    sz = get_mla_metadata_info_v1(bs, 1, NHEAD_PAD, torch.bfloat16, fp8,
                                  is_sparse=True, fast_mode=False,
                                  num_kv_splits=MAX_SPLIT, intra_batch_mode=True)
    return [torch.empty(s_, dtype=t_, device=dev) for (s_, t_) in sz]


def run_one(bs, seq):
    global work_metadata, work_indptr, work_info_set
    global reduce_indptr, reduce_final_map, reduce_partial_map
    if REBUILD:
        (work_metadata, work_indptr, work_info_set,
         reduce_indptr, reduce_final_map, reduce_partial_map) = _mk(bs)
    q = torch.randn((bs, NHEAD_PAD, HEAD_DIM), dtype=torch.bfloat16, device=dev)
    o = torch.empty((bs, NHEAD_PAD, V_HEAD_DIM), dtype=torch.bfloat16, device=dev)

    # page_table_1: [bs, topk], -1 where the request has fewer than topk tokens
    valid = min(seq, TOPK)
    page_table = torch.full((bs, TOPK), -1, dtype=torch.int32, device=dev)
    for i in range(bs):
        page_table[i, :valid] = torch.randint(0, NUM_TOK - 1, (valid,),
                                              dtype=torch.int32, device=dev)

    # exactly what _forward_aiter does
    if os.environ.get("ZERO_INDPTR", "0") == "1":
        kv_indptr_buf.zero_()
    if os.environ.get("ZERO_IDX", "0") == "1":
        kv_indices_buf.zero_()
    if os.environ.get("EXACT_IDX", "0") == "1":
        # allocate kv_indices at EXACTLY the needed size, like a fresh buffer
        pass
    non_minus1 = (page_table != -1).sum(dim=1)
    kv_indptr_buf[0] = 0
    kv_indptr_buf[1:bs + 1] = torch.cumsum(non_minus1, dim=0)
    get_valid_kv_indices(page_table, kv_indptr_buf, kv_indices_buf, bs)

    if os.environ.get("OVERSIZED_QO", "0") == "1":
        # emulate sglang: metadata.cu_seqlens_q is a preallocated max_bs+1 buffer,
        # only [0:bs+1] is meaningful -> aiter infers bs = shape[0]-1 = MAX_BS
        cu_seqlens_q = torch.zeros(MAX_BS + 1, dtype=torch.int32, device=dev)
        cu_seqlens_q[: bs + 1] = torch.arange(0, bs + 1, dtype=torch.int32, device=dev)
        cu_seqlens_q[bs + 1 :] = bs
    else:
        cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device=dev)
    klpl_buf[:bs].fill_(1)
    klpl = klpl_buf[:bs]

    get_mla_metadata_v1(
        cu_seqlens_q, kv_indptr_buf, klpl, NHEAD_PAD, 1, False,
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=1, kv_granularity=16, max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, topk=TOPK, max_split_per_batch=MAX_SPLIT,
        intra_batch_mode=True, dtype_q=torch.bfloat16, dtype_kv=fp8)
    torch.cuda.synchronize()

    mla_decode_fwd(
        q, kv_cache.view(-1, 1, 1, HEAD_DIM), o,
        cu_seqlens_q, kv_indptr_buf, kv_indices_buf, klpl, 1,
        sm_scale=1.0 / (HEAD_DIM ** 0.5), logit_cap=0.0,
        q_scale=None, kv_scale=torch.ones((), dtype=torch.float32, device=dev),
        work_meta_data=work_metadata, work_indptr=work_indptr,
        work_info_set=work_info_set, reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map, reduce_partial_map=reduce_partial_map,
        intra_batch_mode=True, num_kv_splits=MAX_SPLIT)
    torch.cuda.synchronize()
    of = o.float()
    return of.abs().mean().item(), (~torch.isfinite(of)).sum().item()

fails = []
for seq in SEQ_LIST:
    for bs in BS_LIST:
        try:
            m, bad = run_one(bs, seq)
            flag = "OK  " if (bad == 0 and m > 0) else "SUSPECT"
            print(f"  seq={seq:5d} bs={bs:4d}  {flag} mean|o|={m:.5f} nonfinite={bad}",
                  flush=True)
            if bad:
                fails.append((seq, bs, f"nonfinite={bad}"))
        except Exception as e:
            print(f"  seq={seq:5d} bs={bs:4d}  FAIL {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            fails.append((seq, bs, repr(e)))
            raise SystemExit(1)

print("\nRESULT:", "ALL_OK" if not fails else f"FAILURES {fails}")
