#!/usr/bin/env python3
"""U12 diagnosis: where and how much the gfx950 gluon logits differ from aiter's, per input setting.
For each (max_model_len, context range, batch) it prints the largest |diff|, the logit magnitude, the row / column of
the largest diff (and its distance from the row's last visible column and from a 64-token page boundary), how many
visible elements differ by > 1e-5 relative, and which kernels ran (from the info_once logs).
Run in the vLLM image with the U12 branch files mounted.
"""

import json

import torch

from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
from vllm.v1.worker.workspace import init_workspace_manager

HEADS, DIM, PAGE, NEXT_N = 32, 128, 64, 4
FP8 = torch.float8_e4m3fn
init_workspace_manager(torch.device("cuda"))


def inputs(batch, seed, ctx_min, ctx_max, max_len):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ctx = torch.randint(ctx_min, ctx_max + 1, (batch,), generator=g, dtype=torch.int32)
    blocks = (ctx + PAGE - 1) // PAGE
    total = int(blocks.sum())
    perm = torch.randperm(total, generator=g).to(torch.int32)
    table = torch.zeros(batch, (max_len + PAGE - 1) // PAGE, dtype=torch.int32)
    off = 0
    for b, n in enumerate(blocks.tolist()):
        table[b, :n] = perm[off : off + n]
        off += n
    kv = torch.empty(total, PAGE * (DIM + 4), dtype=torch.uint8, device="cuda")
    kv[:, : PAGE * DIM] = (torch.randn(total, PAGE * DIM, device="cuda") * 0.5).to(FP8).view(torch.uint8)
    kv[:, PAGE * DIM :] = torch.ones(total, PAGE, device="cuda").view(torch.uint8)
    q = (torch.randn(batch, NEXT_N, HEADS, DIM, device="cuda") * 0.5).to(FP8)
    w = torch.rand(batch * NEXT_N, HEADS, device="cuda")
    return ctx.cuda(), table.cuda(), kv.view(total, PAGE, 1, DIM + 4), q, w


def run(on, q, kv, w, lens, table, max_len, group):
    import os

    os.environ["VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS"] = "1" if on else "0"
    ops._gluon_paged_mqa_logits_enabled.cache_clear()
    return ops.rocm_fp8_paged_mqa_logits(q, kv, w, lens, table, None, max_len, flat_group_rows=group).clone()


for max_len, lo, hi in ((202752, 40000, 200000), (65536, 40000, 65536), (65536, 4096, 65536), (202752, 4096, 65536)):
    for batch in (1, 4, 32):
        ctx, table, kv, q, w = inputs(batch, 7 * batch, lo, hi, max_len)
        row_lens = (ctx[:, None] - (NEXT_N - 1) + torch.arange(NEXT_N, device="cuda", dtype=torch.int32)).reshape(-1)
        ref = run(False, q, kv, w, ctx, table, max_len, 0)
        qf = q.reshape(-1, 1, HEADS, DIM)
        out = run(True, qf, kv, w, row_lens, table.repeat_interleave(NEXT_N, dim=0), max_len, NEXT_N)
        vis = torch.arange(max_len, device="cuda")[None] < row_lens[:, None]
        diff = torch.where(vis, (ref - out).abs(), torch.zeros_like(ref))
        mag = float(torch.where(vis, ref.abs(), torch.zeros_like(ref)).max())
        idx = int(diff.argmax())
        r, c = divmod(idx, max_len)
        big = int((diff > 1e-5 * mag).sum())
        print(json.dumps({
            "max_model_len": max_len, "ctx": [lo, hi], "batch": batch, "max_abs_diff": float(diff.max()),
            "logit_max": mag, "rel": float(diff.max()) / mag, "row": r, "col": c,
            "from_row_end": int(row_lens[r]) - 1 - c, "col_mod_page": c % PAGE,
            "n_over_1e-5_rel": big, "n_visible": int(vis.sum()),
            "ref_at": float(ref[r, c]), "gluon_at": float(out[r, c]),
        }), flush=True)
