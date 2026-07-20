#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""L3 bench — one run reports THROUGHPUT (write+read GB/s), per-chunk LATENCY
(save/load p50/p95 ms), and round-trip CORRECTNESS across KV layouts, all
measured THROUGH the real engine connector, not a hand-written IO loop.

This drives the production ``InferaKvdConnector`` save/load path: it starts a
real kvd daemon, allocates GPU KV-cache tensors, builds the connector against
``--dir``, and times ``wait_for_save()`` (write) and a cold ``start_load_kv()``
(read). So the number reflects the engine's actual chunked-fusion pipeline —
per-chunk staging rings, per-layer H2D overlap, Triton scatter/gather,
CUDA-event pipelining, GPU-direct-vs-POSIX transport and worker fan-out — the
same machinery production uses, rather than a synthetic preadv/pwrite ceiling.

The per-chunk latency + round-trip correctness across KV layouts (regular / mla
/ mla-aiter) were folded in from the retired ``bench_packed_v2``.

Each CLI knob maps to a real connector tunable, set as env BEFORE the connector
is built (it reads them at __init__):

  --transport     posix / gpu-direct / auto  -> INFERA_KVD_GPU_DIRECT (0/1/unset)
  --chunk-tokens  chunked-fusion window      -> INFERA_KVD_CHUNK_TOKENS
  --workers       save+load fan-out          -> INFERA_KVD_SAVE_WORKERS / _LOAD_WORKERS

The connector resolves the rest (O_DIRECT, layerwise mode, the P2PDMA clamp)
itself; the bench prints the RESOLVED path so the number is unambiguous. READ
is COLD: the KV tensors are zeroed before the load loop, forcing a real reload
from the file tier rather than a page-cache echo.

Usage:
  infera-kvd-l3-bench --dir /mnt/nvme8/l3-bench --total-gb 16
  infera-kvd-l3-bench --dir /data/l3 --transport gpu-direct --chunk-tokens 512 --workers 8
  # equivalently: python -m infera.kvd.bench.l3_bench --dir ...
Run inside the engine image (needs torch + a visible GPU; gpu-direct also needs
the hipFile binding + a P2PDMA-capable driver).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def _detect_p2pdma() -> tuple[bool, str]:
    """The verdict the connector uses: ais-check 'Kernel P2PDMA support: True'."""
    try:
        out = subprocess.run(["ais-check"], capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:
        return False, f"ais-check unavailable ({exc}) -> assume NO P2PDMA"
    ok = bool(re.search(r"Kernel P2PDMA support\s*:\s*True", out))
    line = next((ln.strip() for ln in out.splitlines() if "P2PDMA" in ln), "")
    return ok, line or "ais-check ran"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="L3 throughput bench — write+read GB/s through the REAL connector path",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dir", required=True, help="L3 directory to benchmark")
    ap.add_argument("--total-gb", type=float, default=16.0, help="data volume per side")
    ap.add_argument("--mode", choices=["read", "write", "both"], default="both")
    ap.add_argument(
        "--transport",
        choices=["auto", "posix", "gpu-direct"],
        default="auto",
        help="connector transport -> INFERA_KVD_GPU_DIRECT (auto=unset/auto-detect, "
        "posix=0, gpu-direct=1). The bench prints the RESOLVED path.",
    )
    ap.add_argument(
        "--chunk-tokens",
        default="auto",
        help="chunk window in tokens; 'auto' sizes to --chunk-target-mib, matching "
        "production (INFERA_KVD_CHUNK_TOKENS=auto). A fixed small value under-reports "
        "load — parallel fan-out needs >=128 MiB chunks.",
    )
    ap.add_argument(
        "--chunk-target-mib",
        type=int,
        default=128,
        help="target per-chunk MiB when --chunk-tokens=auto (matches "
        "INFERA_KVD_CHUNK_TARGET_MIB=128)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="pin BOTH save+load workers (INFERA_KVD_*_WORKERS); 0 = connector-resolved",
    )
    ap.add_argument(
        "--layout",
        choices=["regular", "mla", "mla-aiter"],
        default="mla",
        help="KV-cache tensor layout: 'regular' (K/V split, channels=2), 'mla' "
        "(combined latent, channels=1), 'mla-aiter' (ROCM_AITER_MLA fold, "
        "middle dim 1 with logical block_size > 1). Default mla.",
    )
    ap.add_argument(
        "--hidden-dim",
        type=int,
        default=576,
        help="per-token KV element count (num_kv_heads_per_rank × head_dim); "
        "MLA latent = 576 (default), gpt-oss TP=8 = 64",
    )
    ap.add_argument("--device", default="cuda:0", help="GPU the KV cache lives on")
    ap.add_argument("--json", default="", help="write results JSON here")
    args = ap.parse_args()

    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)

    p2p, p2p_detail = _detect_p2pdma()

    print(f"=== L3 throughput bench (real connector path): {d} ===")
    print(f"  transport req : {args.transport}")
    print(f"  P2PDMA        : {'YES' if p2p else 'NO'}   ({p2p_detail})")
    print(f"  chunk tokens  : {args.chunk_tokens}   workers: {args.workers or 'auto'}")
    print(f"  layout        : {args.layout}   hidden_dim: {args.hidden_dim}")
    print(f"  volume        : {args.total_gb:g} GB/side   mode: {args.mode}", flush=True)

    from infera.kvd.bench._connector_bench import run_connector_bench

    # The connector must SAVE before it can LOAD, so a real cold read requires a
    # preceding write regardless of --mode; the helper always round-trips and we
    # print only the requested side(s).
    res = run_connector_bench(
        str(d),
        total_gb=args.total_gb,
        transport=args.transport,
        chunk_tokens=args.chunk_tokens,
        chunk_target_mib=args.chunk_target_mib,
        workers=args.workers,
        layout=args.layout,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )

    rv = res["resolved"]
    load_path = (
        "gpu-direct (S2D)"
        if rv.get("gpu_direct") and not rv.get("force_serial")
        else "POSIX mmap+H2D"
    )
    save_path = "gpu-direct (D2S)" if rv.get("save_gpu_direct") else "POSIX D2H+write"
    print("")
    print(
        f"  resolved      : gpu_direct={rv.get('gpu_direct')} "
        f"save_gpu_direct={rv.get('save_gpu_direct')} "
        f"force_serial={rv.get('force_serial')} layerwise={rv.get('layerwise')}"
    )
    print(
        f"  per-chunk     : {res['per_chunk_mib']:.2f} MiB × {res['num_chunks']} chunks "
        f"(layout={res['resolved'].get('layout')}, "
        f"chunk_tokens={res['resolved'].get('chunk_tokens')})"
    )
    print("  THROUGHPUT (batched fan-out):")
    if args.mode in ("write", "both") and res["save_gbps"] is not None:
        print(f"    WRITE (save): {res['save_gbps']:.2f} GB/s   ({save_path})", flush=True)
    if args.mode in ("read", "both") and res["load_gbps"] is not None:
        print(
            f"    READ  (load): {res['load_gbps']:.2f} GB/s   ({load_path}, cold reload)",
            flush=True,
        )

    def _ms(v):
        return f"{v:.2f}" if v is not None else "n/a"

    print("  LATENCY (per-chunk):")
    if args.mode in ("write", "both"):
        print(
            f"    save        : p50={_ms(res['save_ms_p50'])} ms  p95={_ms(res['save_ms_p95'])} ms"
        )
    if args.mode in ("read", "both"):
        print(
            f"    load        : p50={_ms(res['load_ms_p50'])} ms  p95={_ms(res['load_ms_p95'])} ms"
        )
    verdict = "PASS" if res["correct"] else "FAIL"
    detail = f"  ({res['correctness_detail']})" if res["correctness_detail"] else ""
    print(f"  CORRECTNESS   : {verdict}{detail}", flush=True)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "dir": str(d),
                    "total_gb": args.total_gb,
                    "mode": args.mode,
                    "write_gbps": res["save_gbps"],
                    "read_gbps": res["load_gbps"],
                    "save_ms_p50": res["save_ms_p50"],
                    "save_ms_p95": res["save_ms_p95"],
                    "load_ms_p50": res["load_ms_p50"],
                    "load_ms_p95": res["load_ms_p95"],
                    "correct": res["correct"],
                    "correctness_detail": res["correctness_detail"],
                    "per_chunk_mib": res["per_chunk_mib"],
                    "num_chunks": res["num_chunks"],
                    "p2pdma": p2p,
                    "p2pdma_detail": p2p_detail,
                    **res["resolved"],
                },
                indent=2,
            )
        )
        print(f"  json          : {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
