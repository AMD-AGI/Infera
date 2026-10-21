#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""GPU check of the DCP merge kernels of patch_aiter_custom_alltoall.py.

The strict anchors only prove that the patched source text did not drift. Run
this in the image whenever ATOM_BASE_IMAGE moves, on 2, 4 or 8 GPUs of a node
(see README.md):

    torchrun --nproc-per-node 4 check_dcp_merge.py

It runs each path on aiter's custom all-reduce communicator eagerly and
replayed from a CUDA graph (the registered-buffer path of the decode), and
compares it with RCCL:

- ``custom_all_to_all`` equals ``all_to_all_single`` bit for bit;
- ATOM's ``cp_lse_a2a`` with the per-token FP8 combine, at the GLM-5.2 decode
  shape (64 heads, head_dim 256) with empty ranks (LSE -inf, NaN partials) and
  rows whose owned count is 0, equals the RCCL path bit for bit over the custom
  all-to-all, and up to FP8 rounding over the one-kernel merge
  (``custom_a2a_combine_quant``; a few codes in a million differ).

A case fails if it fails on any rank. Rank 0 prints one line per case, with its
own numbers; the exit code is 1 if any case failed.
"""

import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
dev = torch.device(f"cuda:{rank}")

from aiter import dtypes  # noqa: E402
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce  # noqa: E402
from atom.model_ops.dcp_ops import cp_lse_a2a  # noqa: E402

ca = CustomAllreduce(group=dist.new_group(backend="gloo"), device=dev)
HEADS, HEAD_DIM = 64, 256
# Merged rows (4 x graph size in the MTP verify): graph 40, 36, a draft, 128.
ROWS = (160, 144, 40, 512)
# One-kernel merge vs RCCL: FP8 codes that may differ, relative scale error.
MAX_CODE_DIFF, MAX_SCALE_REL = 1e-4, 1e-5


def dcp_group(comm):
    """The parts of ATOM's DCP group coordinator that cp_lse_a2a reads."""
    return SimpleNamespace(
        world_size=world,
        device_group=dist.group.WORLD,
        device_communicator=SimpleNamespace(ca_comm=comm),
    )


PATHS = {
    "rccl": dcp_group(None),
    # Without supports_a2a_combine_quant: pack -> custom all-to-all -> unpack.
    "a2a": dcp_group(SimpleNamespace(disabled=False, custom_all_to_all=ca.custom_all_to_all)),
    "merge": dcp_group(ca),
}


def run(fn, graph):
    """fn(), eagerly or captured in a CUDA graph and replayed once."""
    if not graph:
        return fn()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with ca.capture(), torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g, stream=s):
            out = fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g.replay()
    torch.cuda.synchronize()
    return out


failed = []


def report(case, ok, detail=""):
    """Record a case that passed on every rank, or failed on any."""
    bad = torch.tensor([0 if ok else 1], device=dev)
    dist.all_reduce(bad)
    if bad.item():
        failed.append(case)
    if rank == 0:
        print(
            f"[check_dcp_merge] {'FAIL' if bad.item() else 'ok'} {case} {detail}".rstrip(),
            flush=True,
        )


def merge_inputs(rows):
    g = torch.Generator(device=dev).manual_seed(rows * 100 + rank)
    out = torch.randn(rows, HEADS, HEAD_DIM, generator=g, device=dev).to(torch.bfloat16)
    lse = torch.rand(rows, HEADS, generator=g, device=dev) * 8
    # No KV on this rank for some rows: LSE -inf and NaN partials, as aiter returns.
    empty = torch.rand(rows, generator=g, device=dev) < 0.05
    lse[empty] = float("-inf")
    out[empty] = float("nan")
    owned = torch.randint(1, 100, (rows,), generator=g, device=dev, dtype=torch.int32)
    owned[torch.rand(rows, generator=g, device=dev) < 0.05] = 0
    return out, lse, owned


# The merge's all-to-all at graph 40 and at a draft: 5.3 / 1.3 MB per rank.
for mb in (5.28, 1.32):
    n = int(mb * 1e6) // 2 // (8 * world) * (8 * world)
    send = torch.randn(n, dtype=torch.bfloat16, device=dev)
    ref = torch.empty_like(send)
    dist.all_to_all_single(ref, send)
    for graph in (False, True):
        got = run(lambda send=send: ca.custom_all_to_all(send), graph)
        ok = got is not None and torch.equal(got, ref)
        report(f"custom_all_to_all {mb} MB/rank {'graph' if graph else 'eager'}", ok)

h_local = HEADS // world
for rows in ROWS:
    x = merge_inputs(rows)
    probe = torch.empty((world * rows * h_local, HEAD_DIM + 2), dtype=torch.bfloat16, device=dev)
    # Otherwise the "merge" path would fall back to RCCL and pass trivially.
    report(
        f"merge applies rows={rows}",
        ca.supports_a2a_combine_quant(probe, h_local, HEAD_DIM, dtypes.fp8),
    )
    ref_q, ref_s = cp_lse_a2a(x[0], x[1], PATHS["rccl"], owned_counts=x[2], quant_dtype=dtypes.fp8)
    ref_codes = ref_q.view(torch.uint8)
    for path in ("a2a", "merge"):
        for graph in (False, True):
            q, s = run(
                lambda path=path, x=x: cp_lse_a2a(
                    x[0], x[1], PATHS[path], owned_counts=x[2], quant_dtype=dtypes.fp8
                ),
                graph,
            )
            code_diff = (q.view(torch.uint8) != ref_codes).float().mean().item()
            scale_rel = ((s - ref_s).abs() / ref_s.abs().clamp_min(1e-30)).max().item()
            finite = bool(torch.isfinite(q.float() * s.view(-1, 1, 1)).all())
            if path == "a2a":
                ok = code_diff == 0 and torch.equal(s, ref_s)
            else:
                ok = finite and code_diff <= MAX_CODE_DIFF and scale_rel <= MAX_SCALE_REL
            detail = f"code diff {code_diff:.2e}, scale rel {scale_rel:.2e}, finite {finite}"
            report(f"cp_lse_a2a {path} rows={rows} {'graph' if graph else 'eager'}", ok, detail)

# Free the IPC buffers now; left to interpreter exit, their __del__ fails noisily.
ca.close()
dist.destroy_process_group()
if rank == 0 and failed:
    print(f"[check_dcp_merge] FAILED: {len(failed)} case(s)", flush=True)
sys.exit(1 if failed else 0)
