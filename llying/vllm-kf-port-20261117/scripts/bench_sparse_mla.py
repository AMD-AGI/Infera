"""GLM-5.2 sparse-MLA decode microbenchmark at vLLM's DP-attention decode shapes (64 heads per rank, top-k 2048,
per-tensor fp8 KV [slots, 576], fp8 q).

Candidates:
  asm      vLLM default: aiter asm mla_decode_fwd with persistent work metadata, built as
           ROCMAiterMLASparseMetadataBuilder does (page 1, kv_granularity 16, fast_mode, max_split 16)
  aiter_tr VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1: aiter Triton sparse_mla_fwd, fp8 dots
  sgl47278 sglang PR #47278 triton_sparse_mla_fwd (gfx950 64-head fp8 split-K decode plan)
  kf       KernelOptForGLM52 campaign-sparse best (kf_sparse_mla.py)

Every candidate is checked against a float32 torch reference (SNR in dB). Timing cycles through NPOOL distinct
KV pools / index sets so the gathered rows do not stay resident in the 256 MB MALL, as with 78 distinct layers.
Usage: python bench_sparse_mla.py [rows ...]   env POOL (slots per pool, 2e6), NPOOL (8), ITERS (40), ONLY
"""

import math
import os
import sys

import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "shim"))
sys.path.insert(0, here)

H, D, DV, TOPK = 64, 576, 512, 2048
FP8 = torch.float8_e4m3fn
POOL = int(float(os.environ.get("POOL", 2e6)))
NPOOL = int(os.environ.get("NPOOL", 8))
ITERS = int(os.environ.get("ITERS", 40))
ONLY = set(os.environ.get("ONLY", "").split(",")) - {""}
SM_SCALE = 1.0 / math.sqrt(256)  # GLM-5.2 qk_head_dim 256; the value does not change the cost
dev = torch.device("cuda")


def snr_db(ref: torch.Tensor, out: torch.Tensor) -> float:
    ref = ref.float()
    err = (out.float() - ref).pow(2).mean()
    return float(10 * torch.log10(ref.pow(2).mean() / err.clamp_min(1e-30)))


def reference(q8, kv8, idx):
    outs = []
    for r0 in range(0, q8.shape[0], 16):
        q = q8[r0 : r0 + 16].float()
        k = kv8[idx[r0 : r0 + 16].long()].float()  # [r, topk, D]
        s = torch.einsum("rhd,rkd->rhk", q, k) * SM_SCALE
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("rhk,rkd->rhd", p, k[..., :DV]))
    return torch.cat(outs)


def make_asm(rows):
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
    from aiter.mla import mla_decode_fwd

    qo_indptr = torch.arange(rows + 1, dtype=torch.int32, device=dev)
    kv_indptr = torch.arange(rows + 1, dtype=torch.int32, device=dev) * TOPK
    last_page = torch.ones(rows, dtype=torch.int32, device=dev)
    info = get_mla_metadata_info_v1(rows, 1, H, FP8, FP8, is_sparse=True, fast_mode=True)
    bufs = [torch.empty(size, dtype=dtype, device=dev) for size, dtype in info]
    wmd, wind, wis, rind, rfm, rpm = bufs
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    get_mla_metadata_v1(
        qo_indptr, kv_indptr, last_page, H, 1, True, wmd, wis, wind, rind, rfm, rpm,
        page_size=1, kv_granularity=16, max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=True,
        max_split_per_batch=min(16, num_cu), dtype_q=FP8, dtype_kv=FP8,
    )
    one = torch.ones(1, dtype=torch.float32, device=dev)
    o = torch.empty(rows, H, DV, dtype=torch.bfloat16, device=dev)

    def run(q8, kv8, idx):
        mla_decode_fwd(
            q8, kv8.view(-1, 1, 1, D), o, qo_indptr, kv_indptr, idx.reshape(-1), last_page, 1,
            sm_scale=SM_SCALE, logit_cap=0.0, q_scale=one, kv_scale=one, causal=True,
            work_meta_data=wmd, work_indptr=wind, work_info_set=wis,
            reduce_indptr=rind, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
        return o

    return run


def make_aiter_triton(rows):
    from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd

    kv_indptr = torch.arange(rows + 1, dtype=torch.int32, device=dev) * TOPK
    one = torch.ones(1, dtype=torch.float32, device=dev)
    o = torch.empty(rows, H, DV, dtype=torch.bfloat16, device=dev)

    def run(q8, kv8, idx):
        sparse_mla_fwd(
            q8, kv8.view(-1, 1, 1, D), kv_indptr, idx.reshape(-1), SM_SCALE, kv_scale=one,
            kv_lora_rank=DV, qk_rope_head_dim=D - DV, has_invalid=False, dot_precision="fp8",
            q_scale=one, out=o,
        )
        return o

    return run


def make_sgl(module_name):
    mod = __import__(module_name)

    def factory(rows):
        def run(q8, kv8, idx):
            out = mod.triton_sparse_mla_fwd(
                q8[..., :DV], q8[..., DV:], kv8.view(-1, 1, D), idx.view(-1, 1, TOPK), SM_SCALE, d_v=DV
            )
            return out.view(rows, H, DV)

        return run

    return factory


CANDIDATES = {
    "asm": make_asm,
    "aiter_tr": make_aiter_triton,
    "sgl47278": make_sgl("sgl47278_triton_sparse_mla"),
    "kf": make_sgl("kf_campaign_sparse_mla"),
}


def main():
    rows_list = [int(a) for a in sys.argv[1:]] or [16, 32, 64, 104, 128, 160, 208]
    props = torch.cuda.get_device_properties(0)
    print(f"device {props.name} {props.gcnArchName} CUs {props.multi_processor_count}; "
          f"pool {POOL} slots x {NPOOL}, H {H}, top-k {TOPK}, iters {ITERS}", flush=True)
    torch.manual_seed(0)
    pools = [(torch.randn(POOL, D, device=dev) * 0.5).to(FP8) for _ in range(NPOOL)]
    names = [n for n in CANDIDATES if not ONLY or n in ONLY]
    print("rows | " + " | ".join(f"{n} us (SNR dB)" for n in names) + " | best vs asm", flush=True)
    for rows in rows_list:
        qs = [(torch.randn(rows, H, D, device=dev) * 0.5).to(FP8) for _ in range(NPOOL)]
        idxs = [torch.randint(0, POOL, (rows, TOPK), dtype=torch.int32, device=dev) for _ in range(NPOOL)]
        ref = reference(qs[0], pools[0], idxs[0])
        cells, times = [], {}
        for name in names:
            try:
                run = CANDIDATES[name](rows)
                snr = snr_db(ref, run(qs[0], pools[0], idxs[0]))
                for i in range(NPOOL):
                    run(qs[i], pools[i], idxs[i])
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for it in range(ITERS):
                    i = it % NPOOL
                    run(qs[i], pools[i], idxs[i])
                end.record()
                torch.cuda.synchronize()
                us = start.elapsed_time(end) * 1000 / ITERS
                times[name] = us
                cells.append(f"{us:.1f} ({snr:.1f})")
            except Exception as e:  # report and keep going with the other candidates
                cells.append(f"error: {type(e).__name__}: {str(e)[:80]}")
        best = min((t, n) for n, t in times.items() if n != "asm") if len(times) > 1 else None
        rel = f"{best[1]} {times['asm'] / best[0]:.2f}x" if best and "asm" in times else "-"
        print(f"{rows} | " + " | ".join(cells) + f" | {rel}", flush=True)


if __name__ == "__main__":
    main()
