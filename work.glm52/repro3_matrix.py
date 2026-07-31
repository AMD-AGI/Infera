"""Orthogonal scan. Each run = ONE (meta_bs, call_bs) pair in a FRESH process, so no
cross-round state can confound. Prints a single line the driver can grep.

meta_bs  = batch_size used to SIZE the aiter metadata buffers (get_mla_metadata_info_v1)
call_bs  = batch_size actually passed to get_mla_metadata_v1 / mla_decode_fwd

sglang's behavior == (meta_bs = max_bs, call_bs = current capture bs).
"""
import os, sys, traceback
import torch

META_BS = int(os.environ["META_BS"])
CALL_BS = int(os.environ["CALL_BS"])
SEQ     = int(os.environ.get("SEQ", 1))
TOPK    = int(os.environ.get("TOPK", 2048))
NQ      = int(os.environ.get("NUM_Q_HEADS", 8))
MAX_SPLIT = int(os.environ.get("MAX_SPLIT", 64))
KV_LORA, QK_ROPE = 512, 64
HEAD_DIM, V_HEAD_DIM = KV_LORA + QK_ROPE, KV_LORA
fp8, dev = torch.float8_e4m3fn, "cuda"

from aiter.mla import mla_decode_fwd
from aiter.ops.attention import get_mla_metadata_info_v1, get_mla_metadata_v1
from sglang.kernels.ops.attention.dsa.triton_kernel import get_valid_kv_indices

rf = 16 // NQ if NQ < 16 else 1
NHEAD_PAD = NQ * rf
NUM_TOK = max(META_BS, CALL_BS) * max(SEQ, 1) + 8192

kv_cache = torch.randn((NUM_TOK, 1, HEAD_DIM), dtype=torch.bfloat16, device=dev).to(fp8)

sizes = get_mla_metadata_info_v1(META_BS, 1, NHEAD_PAD, torch.bfloat16, fp8,
                                 is_sparse=True, fast_mode=False,
                                 num_kv_splits=MAX_SPLIT, intra_batch_mode=True)
(wm, wi, ws, ri, rf_map, rp) = [torch.empty(s, dtype=t, device=dev) for (s, t) in sizes]

bs = CALL_BS
valid = min(SEQ, TOPK)
page_table = torch.full((bs, TOPK), -1, dtype=torch.int32, device=dev)
page_table[:, :valid] = torch.randint(0, NUM_TOK - 1, (bs, valid),
                                      dtype=torch.int32, device=dev)

kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
kv_indptr[1:] = torch.cumsum((page_table != -1).sum(dim=1), dim=0)
kv_indices = torch.zeros(bs * TOPK, dtype=torch.int32, device=dev)
get_valid_kv_indices(page_table, kv_indptr, kv_indices, bs)

cu_q = torch.arange(0, bs + 1, dtype=torch.int32, device=dev)
klpl = torch.ones((bs,), dtype=torch.int32, device=dev)
q = torch.randn((bs, NHEAD_PAD, HEAD_DIM), dtype=torch.bfloat16, device=dev)
o = torch.empty((bs, NHEAD_PAD, V_HEAD_DIM), dtype=torch.bfloat16, device=dev)

try:
    get_mla_metadata_v1(cu_q, kv_indptr, klpl, NHEAD_PAD, 1, False,
                        wm, ws, wi, ri, rf_map, rp,
                        page_size=1, kv_granularity=16, max_seqlen_qo=1,
                        uni_seqlen_qo=1, fast_mode=False, topk=TOPK,
                        max_split_per_batch=MAX_SPLIT, intra_batch_mode=True,
                        dtype_q=torch.bfloat16, dtype_kv=fp8)
    torch.cuda.synchronize()
    mla_decode_fwd(q, kv_cache.view(-1, 1, 1, HEAD_DIM), o,
                   cu_q, kv_indptr, kv_indices, klpl, 1,
                   sm_scale=1.0 / (HEAD_DIM ** 0.5), logit_cap=0.0,
                   q_scale=None,
                   kv_scale=torch.ones((), dtype=torch.float32, device=dev),
                   work_meta_data=wm, work_indptr=wi, work_info_set=ws,
                   reduce_indptr=ri, reduce_final_map=rf_map,
                   reduce_partial_map=rp,
                   intra_batch_mode=True, num_kv_splits=MAX_SPLIT)
    torch.cuda.synchronize()
    of = o.float()
    bad = (~torch.isfinite(of)).sum().item()
    m = of.abs().mean().item()
    print(f"VERDICT meta_bs={META_BS} call_bs={CALL_BS} seq={SEQ} "
          f"{'OK' if (bad == 0 and m > 0) else 'BADVAL'} mean={m:.5f} nonfinite={bad} "
          f"rp_len={rp.numel()} ws={tuple(ws.shape)}", flush=True)
except Exception as e:
    print(f"VERDICT meta_bs={META_BS} call_bs={CALL_BS} seq={SEQ} EXC {type(e).__name__}: {e}",
          flush=True)
    traceback.print_exc()
    sys.exit(1)
