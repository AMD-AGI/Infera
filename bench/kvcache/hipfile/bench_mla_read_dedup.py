###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Microbenchmark: MLA TP read-dedup NFS-bandwidth saving (#64).

Under MLA the cached latent is byte-identical on every TP rank. Today each
of the ``tp_size`` ranks reads its OWN copy of a chunk from L3, so the
storage tier must move ``tp_size × chunk_bytes`` per chunk. With the dedup
(one shared file + rank-0-reads, broadcast to the rest over the GPU
interconnect) the tier moves ``1 × chunk_bytes``.

This bench quantifies that on the *real* storage path (POSIX read, cold via
``posix_fadvise(DONTNEED)`` so it actually hits NFS, not page cache):

    A. "per-rank (today)"  : N threads each read a DISTINCT cold file
                             -> storage moves N × chunk_bytes
    B. "dedup (proposed)"  : 1 thread reads ONE cold file (rank 0)
                             -> storage moves 1 × chunk_bytes
                             (the broadcast to the other N-1 ranks is over
                              XGMI/IF at ~TB/s, negligible vs NFS — NOT
                              measured here; this isolates the NFS saving)

It reports wall-clock, aggregate GB/s, and the projected MLA decode-prefix
load ceiling (tok/s) for each, using the model's KV bytes/token.

Usage:
  python3 -m bench.kvcache.hipfile.bench_mla_read_dedup \\
      --root /mnt/store/probe-dedup --chunk-mib 128 --ranks 8 \\
      --kv-bytes-per-token 68608   # MLA bf16: (512+64)*2*61
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import time

_ALIGN = 4096


def _seed(path: str, nbytes: int) -> None:
    if os.path.exists(path) and os.path.getsize(path) == nbytes:
        return
    buf = b"\xa5" * (1 << 20)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        written = 0
        while written < nbytes:
            n = os.write(fd, buf[: min(len(buf), nbytes - written)])
            written += n
        os.fsync(fd)
    finally:
        os.close(fd)


def _evict(path: str) -> None:
    """Drop this file from the page cache so the next read hits storage."""
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    finally:
        os.close(fd)


def _read_whole(path: str, chunk: int = 8 << 20) -> int:
    fd = os.open(path, os.O_RDONLY)
    total = 0
    try:
        while True:
            b = os.read(fd, chunk)
            if not b:
                break
            total += len(b)
    finally:
        os.close(fd)
    return total


def _timed_concurrent(paths: list[str], workers: int) -> tuple[float, int]:
    for p in paths:
        _evict(p)
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        got = list(ex.map(_read_whole, paths))
    dt = time.perf_counter() - t0
    return dt, sum(got)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="dir on the target storage tier (e.g. NFS)")
    ap.add_argument("--chunk-mib", type=int, default=128)
    ap.add_argument("--ranks", type=int, default=8, help="tp_size (per-rank copies today)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--kv-bytes-per-token",
        type=int,
        default=(512 + 64) * 2 * 61,
        help="KV bytes/token to project tok/s (default = Kimi MLA bf16: 576*2*61)",
    )
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    size = args.chunk_mib * (1 << 20)
    paths = [os.path.join(args.root, f"rank{r}.bin") for r in range(args.ranks)]
    for p in paths:
        _seed(p, size)

    print(
        f"chunk={args.chunk_mib} MiB  ranks(tp_size)={args.ranks}  "
        f"root={args.root}  kv_bytes/token={args.kv_bytes_per_token}"
    )

    def best(paths_, workers):
        b = 1e9
        bytes_ = 0
        for _ in range(args.reps):
            dt, n = _timed_concurrent(paths_, workers)
            if dt < b:
                b, bytes_ = dt, n
        return b, bytes_

    # A: today — N ranks each read a DISTINCT file (storage moves N×size)
    a_dt, a_bytes = best(paths, args.ranks)
    a_gbs = a_bytes / a_dt / (1 << 30)

    # B: dedup — rank 0 reads ONE file; broadcast to the rest is off-NFS
    b_dt, b_bytes = best(paths[:1], 1)
    b_gbs = b_bytes / b_dt / (1 << 30)

    # Projected MLA load ceiling: storage_GBs / (kv_bytes/token × copies)
    def tok_s(gbs, copies):
        return gbs * (1 << 30) / (args.kv_bytes_per_token * copies)

    print(f"{'scenario':<22}{'wall_ms':>9}{'NFS_GB/s':>10}{'NFS_bytes':>14}{'proj_tok/s':>12}")
    print(
        f"{'A per-rank (today)':<22}{a_dt * 1e3:>9.1f}{a_gbs:>10.2f}"
        f"{a_bytes:>14}{tok_s(a_gbs, args.ranks):>12,.0f}"
    )
    print(
        f"{'B dedup (rank0 only)':<22}{b_dt * 1e3:>9.1f}{b_gbs:>10.2f}"
        f"{b_bytes:>14}{tok_s(b_gbs, 1):>12,.0f}"
    )
    # The headline: NFS *traffic* per chunk drops from N× to 1×.
    print(
        f"\nNFS traffic per chunk: A={args.ranks}x vs B=1x  -> {args.ranks}x reduction; "
        f"wall A/B = {a_dt / b_dt:.1f}x"
    )
    print(
        "(Broadcast of the shared latent to the other ranks is over "
        "XGMI/Infinity-Fabric ~TB/s, negligible vs NFS — not included.)"
    )


if __name__ == "__main__":
    main()
