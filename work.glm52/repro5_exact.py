"""Exact replay of the real failing call, shapes taken from __DSA_TRACE__ on chi2835:

  prep bs=48 maxq=1 cap=64 qo=(49,) kvindptr=(65,) nhead_pad=16 topk=2048
  dec  bs=48 q=(48,16,576) o=(48,16,512) kv=(3552256,1,576) head_dim=576 v=512
       cu_q=(49,) kvindptr=(65,) kvidx=(131072,) maxq=1 pt=(48,2048)

The bit my earlier repros got wrong: kv_indptr is the OVERSIZED persistent buffer of
length max_bs+1 = 65 (sglang's self.kv_indptr), while cu_seqlens_q is a bs+1 = 49 slice.
kv_indices is likewise the full max_bs*topk = 131072 buffer.

KV_MODE selects what lives in kv_indptr[bs+1:]:
  stale   - leftover from the previous, larger bs (what sglang actually leaves there)
  zero    - zeroed tail
  clamped - tail held at kv_indptr[bs]
"""
import os, sys, traceback
import torch

MAX_BS  = int(os.environ.get("MAX_BS", 64))
BS      = int(os.environ.get("BS", 48))
TOPK    = int(os.environ.get("TOPK", 2048))
SEQ     = int(os.environ.get("SEQ", 1))
NQ      = int(os.environ.get("NUM_Q_HEADS", 8))
MAX_SPLIT = int(os.environ.get("MAX_SPLIT", 64))
KV_MODE = os.environ.get("KV_MODE", "stale")
NUM_TOK = int(os.environ.get("NUM_TOK", 3552256))

KV_LORA, QK_ROPE = 512, 64
HEAD_DIM, V_HEAD_DIM = KV_LORA + QK_ROPE, KV_LORA
fp8, dev = torch.float8_e4m3fn, "cuda"

from aiter.mla import mla_decode_fwd
from aiter.ops.attention import get_mla_metadata_info_v1, get_mla_metadata_v1
from sglang.kernels.ops.attention.dsa.triton_kernel import get_valid_kv_indices

rf = 16 // NQ if NQ < 16 else 1
NHEAD_PAD = NQ * rf

kv_cache = torch.zeros((NUM_TOK, 1, HEAD_DIM), dtype=fp8, device=dev)
print(f"kv_cache {tuple(kv_cache.shape)} "
      f"{kv_cache.numel()/2**30:.2f} GiB  mode={KV_MODE} max_bs={MAX_BS} bs={BS}",
      flush=True)

sizes = get_mla_metadata_info_v1(MAX_BS, 1, NHEAD_PAD, torch.bfloat16, fp8,
                                 is_sparse=True, fast_mode=False,
                                 num_kv_splits=MAX_SPLIT, intra_batch_mode=True)
(wm, wi, ws, ri, rfm, rp) = [torch.empty(s, dtype=t, device=dev) for (s, t) in sizes]

# sglang's persistent, max_bs-sized buffers
kv_indptr_buf  = torch.zeros((MAX_BS + 1,), dtype=torch.int32, device=dev)
kv_indices_buf = torch.zeros(MAX_BS * TOPK, dtype=torch.int32, device=dev)
klpl_buf       = torch.ones((MAX_BS,), dtype=torch.int32, device=dev)


def one(bs):
    valid = min(SEQ, TOPK)
    pt = torch.full((bs, TOPK), -1, dtype=torch.int32, device=dev)
    pt[:, :valid] = torch.randint(0, NUM_TOK - 1, (bs, valid),
                                  dtype=torch.int32, device=dev)

    # exactly _forward_aiter: only [1:bs+1] is written, the tail keeps whatever
    # the previous (larger) batch left behind
    kv_indptr_buf[1:bs + 1] = torch.cumsum((pt != -1).sum(dim=1), dim=0)
    if KV_MODE == "zero":
        kv_indptr_buf[bs + 1:] = 0
    elif KV_MODE == "clamped":
        kv_indptr_buf[bs + 1:] = kv_indptr_buf[bs]
    # "stale": leave as is

    get_valid_kv_indices(pt, kv_indptr_buf, kv_indices_buf, bs)

    cu_q = torch.arange(0, bs + 1, dtype=torch.int32, device=dev)
    klpl_buf[:bs].fill_(1)
    klpl = klpl_buf[:bs]
    q = torch.randn((bs, NHEAD_PAD, HEAD_DIM), dtype=torch.bfloat16, device=dev)
    o = torch.empty((bs, NHEAD_PAD, V_HEAD_DIM), dtype=torch.bfloat16, device=dev)

    get_mla_metadata_v1(cu_q, kv_indptr_buf, klpl, NHEAD_PAD, 1, False,
                        wm, ws, wi, ri, rfm, rp,
                        page_size=1, kv_granularity=16, max_seqlen_qo=1,
                        uni_seqlen_qo=1, fast_mode=False, topk=TOPK,
                        max_split_per_batch=MAX_SPLIT, intra_batch_mode=True,
                        dtype_q=torch.bfloat16, dtype_kv=fp8)
    torch.cuda.synchronize()
    mla_decode_fwd(q, kv_cache.view(-1, 1, 1, HEAD_DIM), o,
                   cu_q, kv_indptr_buf, kv_indices_buf, klpl, 1,
                   sm_scale=1.0 / (HEAD_DIM ** 0.5), logit_cap=0.0,
                   q_scale=None,
                   kv_scale=torch.ones((), dtype=torch.float32, device=dev),
                   work_meta_data=wm, work_indptr=wi, work_info_set=ws,
                   reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rp,
                   intra_batch_mode=True, num_kv_splits=MAX_SPLIT)
    torch.cuda.synchronize()
    return o


SEQUENCE = [int(x) for x in os.environ.get("SEQUENCE", f"{MAX_BS},{BS}").split(",")]
try:
    for b in SEQUENCE:
        o = one(b)
        print(f"VERDICT bs={b} OK mean={o.float().abs().mean().item():.5f}", flush=True)
    print("ALL_DONE", flush=True)
except Exception as e:
    print(f"VERDICT EXC {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
