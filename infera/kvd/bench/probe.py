#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""kvd storage probe — is this mount fast enough for kvd L3, measured the way
kvd actually drives it?

This drives the **REAL engine connector** (``InferaKvdConnector`` save/load),
not a hand-written preadv/pwrite loop. It starts a real kvd daemon, allocates
GPU KV-cache tensors, builds the connector against ``--dir``, then times the
production ``wait_for_save()`` / ``start_load_kv()`` path — the same per-chunk
staging rings, per-layer H2D overlap, Triton scatter/gather, CUDA-event
pipelining, GPU-direct-vs-POSIX transport and worker fan-out production uses.
The GB/s it prints is therefore what the engine's L3 tier would actually get
on this mount, not a synthetic IO ceiling.

It also prints the SAME P2PDMA judgment (ais-check "Kernel P2PDMA support")
that decides whether L3 loads run GPU-direct+parallel or fall back to POSIX —
a verdict that is frequently a container false-negative (missing /boot/config-*).

Usage:
    infera-kvd-probe --dir /mnt/aic-nfs/kvd-probe --size-gb 16
    # equivalently: python -m infera.kvd.bench.probe --dir ... --size-gb 16
Run inside the engine image so `infera`, `torch`, `ais-check`, and INFERA_KVD_*
all match production (needs a visible GPU — the connector lands KV in HBM).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def detect_p2pdma() -> tuple[bool, str]:
    """Same verdict kvd's connector uses: ais-check reporting
    'Kernel P2PDMA support: True'. Returns (ok, detail)."""
    try:
        out = subprocess.run(["ais-check"], capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:
        return False, f"ais-check unavailable ({exc}) -> kvd would assume NO P2PDMA"
    ok = bool(re.search(r"Kernel P2PDMA support\s*:\s*True", out))
    line = next((ln.strip() for ln in out.splitlines() if "P2PDMA" in ln), "")
    return ok, line or "ais-check ran"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="kvd storage probe — R/W GB/s through the REAL connector save/load path"
    )
    ap.add_argument("--dir", required=True, help="directory to probe (kvd L3 candidate mount)")
    ap.add_argument("--size-gb", type=float, default=16.0, help="data volume driven per side")
    ap.add_argument(
        "--transport",
        choices=["auto", "posix", "gpu-direct"],
        default="auto",
        help="connector transport: auto (env/ais-check auto-detect), posix "
        "(INFERA_KVD_GPU_DIRECT=0), gpu-direct (INFERA_KVD_GPU_DIRECT=1). "
        "The probe prints the RESOLVED path.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="pin BOTH save+load workers (INFERA_KVD_*_WORKERS); 0 = connector-resolved",
    )
    ap.add_argument(
        "--chunk-tokens",
        default="auto",
        help="chunk window in tokens; 'auto' sizes to --chunk-target-mib, matching "
        "production (INFERA_KVD_CHUNK_TOKENS=auto). A fixed small value under-reports load.",
    )
    ap.add_argument(
        "--chunk-target-mib",
        type=int,
        default=128,
        help="target per-chunk MiB when --chunk-tokens=auto (INFERA_KVD_CHUNK_TARGET_MIB=128)",
    )
    ap.add_argument("--device", default="cuda:0", help="GPU the KV cache lives on")
    ap.add_argument("--json", default="", help="write results JSON here")
    args = ap.parse_args()

    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)

    p2p, p2p_detail = detect_p2pdma()

    print(f"=== kvd storage probe (real connector path): {d} ===")
    print(f"  transport req : {args.transport}")
    print(f"  P2PDMA        : {'YES' if p2p else 'NO'}   ({p2p_detail})")
    print(f"  chunk tokens  : {args.chunk_tokens}   workers: {args.workers or 'auto'}")
    print(f"  volume        : {args.size_gb:g} GB/side", flush=True)

    from infera.kvd.bench._connector_bench import run_connector_bench

    res = run_connector_bench(
        str(d),
        total_gb=args.size_gb,
        transport=args.transport,
        chunk_tokens=args.chunk_tokens,
        chunk_target_mib=args.chunk_target_mib,
        workers=args.workers,
        device=args.device,
        verbose=True,  # probe is a diagnostic — log daemon/chunk/config steps
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
        f"(chunk_tokens={res['resolved'].get('chunk_tokens')})"
    )
    print(f"  LOAD path     : {load_path}")
    print(f"  SAVE path     : {save_path}")
    if res["save_gbps"] is not None:
        print(f"  SAVE (write)  : {res['save_gbps']:.2f} GB/s", flush=True)
    if res["load_gbps"] is not None:
        print(f"  LOAD (read)   : {res['load_gbps']:.2f} GB/s   (cold reload)", flush=True)
    # A mount that corrupts KV is a hard fail worth surfacing even in the probe.
    verdict = "PASS" if res["correct"] else "FAIL"
    detail = f"  ({res['correctness_detail']})" if res["correctness_detail"] else ""
    print(f"  CORRECTNESS   : {verdict}{detail}", flush=True)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "dir": str(d),
                    "size_gb": args.size_gb,
                    "save_gbps": res["save_gbps"],
                    "load_gbps": res["load_gbps"],
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
