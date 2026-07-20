###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Concurrent multi-worker load microbench: hipFile device-read + Triton scatter.

This is the connector's load-side hot path (``_load_chunk_packed`` device
branch) made self-contained and driven at N concurrent workers, to measure
the parallel storage→GPU read + scatter FAN-OUT that the single-chunk
connector round-trip bench (``infera-kvd-l3-bench``, which absorbed the
retired ``bench_packed_v2``) cannot show.

Each worker thread does, per chunk:
    hipFile.read(chunk file) -> registered device buffer (P2PDMA / gpu-direct)
    -> kv_chunk_scatter() into a paged KV cache (the real Triton kernel)

The blocking ``HipFile.read`` releases the GIL, so N Python threads truly
fan out the reads; the Triton scatter runs async on the default stream.

Why this bench exists (and the raw read-only variant was dropped): the
interesting quantity is the connector's REAL load path — read AND the Triton
scatter — at concurrency. A read-only variant (no scatter) only measures raw
fabric BW and is already covered by infera.kvd.bench.

Usage:
  python3 -u -m bench.kvcache.hipfile.bench_concurrent_load \\
      --root /mnt/store/probe --chunk-mib 128 --num-chunks 32 \\
      --layers 61 --hidden-dim 64 --workers 1 4 8 16

Typical result (MI300X): on NFS-RDMA read+scatter scales ~4 GB/s (N=1) →
~22 GB/s (N=8); a single NVMe device saturates ~6.5 GB/s regardless of N.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import time

import torch

from infera.engine.sglang.hipfile_shim import HipFile, HipFileDriver, RegisteredBuffer
from infera.engine.vllm.triton_kv_gather import kv_chunk_scatter

PAGE = 4096
BLOCK = 64  # paged-KV page size in tokens


def _seed_files(root: str, num_chunks: int, payload_bytes: int) -> list[str]:
    """Pre-write each chunk as a 4 KiB-aligned plain file (POSIX write),
    with a per-chunk leading byte so a scatter round-trip is verifiable."""
    os.makedirs(root, exist_ok=True)
    paths = []
    base = (torch.arange(payload_bytes, dtype=torch.uint8) % 251).numpy().tobytes()
    for c in range(num_chunks):
        p = os.path.join(root, f"chunk_{c:04d}.bin")
        if not (os.path.exists(p) and os.path.getsize(p) == payload_bytes):
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.write(fd, bytes([(c + 1) % 251]) + base[1:])
            os.close(fd)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", required=True, help="dir for the bench's chunk files (place on the target FS)"
    )
    ap.add_argument("--chunk-mib", type=int, default=128)
    ap.add_argument("--num-chunks", type=int, default=32)
    ap.add_argument("--layers", type=int, default=61)
    ap.add_argument(
        "--hidden-dim", type=int, default=64, help="per-token KV elements (num_kv_heads × head_dim)"
    )
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA/ROCm device required")

    dev = torch.device(args.device)
    dtype = torch.bfloat16
    HipFileDriver().ensure_open()

    payload_bytes = args.chunk_mib * (1 << 20)
    L, H = args.layers, args.hidden_dim
    # staging [2, L, CT, H]; payload = 2·L·CT·H·2(bf16). Derive CT, round to pages.
    CT = (payload_bytes // (2 * L * H * 2) // BLOCK) * BLOCK
    ppc = CT // BLOCK
    payload_bytes = 2 * L * CT * H * 2
    num_blocks = args.num_chunks * ppc + 4
    print(
        f"chunk={payload_bytes / (1 << 20):.1f}MiB layers={L} hidden={H} CT={CT} "
        f"pages/chunk={ppc} num_chunks={args.num_chunks} num_blocks={num_blocks} root={args.root}"
    )

    layers = [torch.zeros((2, num_blocks, BLOCK, H), dtype=dtype, device=dev) for _ in range(L)]
    layer_to_group = [0] * L
    paths = _seed_files(args.root, args.num_chunks, payload_bytes)

    def load_chunk(c: int) -> None:
        raw = torch.empty(payload_bytes + 2 * PAGE, dtype=torch.uint8, device=dev)
        base = int(raw.data_ptr())
        pre = (-base) % PAGE
        with RegisteredBuffer(base + pre, (payload_bytes + PAGE - 1) & ~(PAGE - 1)) as reg:
            with HipFile(paths[c], "r") as fh:
                n = fh.read(reg.handle, payload_bytes, 0, 0)
                assert int(n) == payload_bytes, f"short read {n}/{payload_bytes}"
        staging = raw[pre : pre + payload_bytes].view(dtype).reshape(2, L, CT, H)
        per_page = tuple((c * ppc + p,) for p in range(ppc))
        kv_chunk_scatter(staging, layers, per_page, layer_to_group, BLOCK, use_triton=True)

    # Correctness: single-thread load chunk 0, confirm a scattered slot is non-zero.
    for t in layers:
        t.zero_()
    load_chunk(0)
    torch.cuda.synchronize()
    print(f"correctness: scattered-nonzero={bool(layers[0][:, 0:ppc].any().item())}")

    print(f"{'N':>4} {'GB/s':>8} {'ms':>9}")
    for n_workers in args.workers:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:  # warmup
            list(ex.map(load_chunk, range(args.num_chunks)))
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(args.reps):
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
                list(ex.map(load_chunk, range(args.num_chunks)))
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        gbs = (args.num_chunks * payload_bytes) / best / (1 << 30)
        print(f"{n_workers:>4} {gbs:>8.2f} {best * 1000:>9.1f}")


if __name__ == "__main__":
    main()
