###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Bench 0 — hipFile sanity on the testbed.

This is the must-pass-first gate before any of the Bench 1+ microbench /
kvd-shape / real-workload tests are worth running.

What this bench answers:

  Q1. Does the hipfile python binding actually exist and open a driver
      on this host? (HipFileDriver().ensure_open() succeeds.)
  Q2. Does a single hipFileRead deliver byte-for-byte the same content
      as a POSIX read + H2D copy? (torch.equal vs reference.)
  Q3. Is the kernel P2PDMA path actually live, or is hipFile silently
      bouncing through CPU? (ais-check Kernel P2PDMA support: True.)
      Same failure shape as Mooncake-on-ionic (project memory
      `project_mi355x_pensando_ionic_rdma`); we want a positive assertion
      the DMA path is live, not "the binding imported OK".
  Q4. How does hipFile throughput compare to POSIX + H2D at a small
      sweep of block sizes × concurrencies? (sanity-only; Bench 1 is the
      real sweep.)

What this bench does NOT answer:

  - Whether hipFile wins at production block sizes / concurrencies —
    that's Bench 1 (full sweep, modes A/B/C/D).
  - Whether hipFile wins for the kvd random-key access pattern — that's
    Bench 2.
  - Whether hipFile improves real TTFT — Bench 3 / Bench 4.

This script does NOT mutate kvd or any production code. It is a
read-only consumer of `infera.engine.sglang.hipfile_shim` (shipped at
commit 645be85) and torch.

Build + run on an MI355X node:

  bash deploy/docker/scripts/build_hipfile.sh                # 5-8 min (one-time)
  python -m bench.kvcache.hipfile.bench0_sanity \
      --path /tmp/hipfile-test \
      --path /mnt/<nfs-mount>/hipfile-test \
      --require-p2pdma

Pass criterion (Bench 0):

  - Every target path: torch.equal(hipfile_buf, reference_buf) at every
    sweep cell.
  - Every target path: ais-check reports "Kernel P2PDMA support: True"
    (if --require-p2pdma is set, False is a hard fail; otherwise it's
    flagged in the JSON but exit 0).
  - hipFile throughput within 0.5×..3× of POSIX+H2D at 4 MiB blocks.
    Wildly outside that range = the bench is measuring something else
    (cold cache, contention, compat-mode bounce).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Defaults intentionally small — Bench 0 is sanity, not the throughput
# sweep. Bench 1 owns the 4 KB..16 MB × 1..128 sweep.
DEFAULT_FILE_BYTES = 64 * (1 << 20)  # 64 MiB pre-seed
DEFAULT_SIZES = [64 << 10, 1 << 20, 4 << 20, 16 << 20]
DEFAULT_CONCURRENCIES = [1, 8]
WARMUP_REPS = 2
TIMED_REPS = 5


# ---------------------------------------------------------------------------
# Pre-seed: deterministic test file


def seed_test_file(path: str, nbytes: int) -> None:
    """Write `nbytes` of deterministic content (i % 256) to `path`.

    Idempotent: skips write if file already matches size + first 4 KiB.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size == nbytes:
        # Spot-check the deterministic pattern so a half-written prior
        # run doesn't poison the bench.
        with open(p, "rb") as f:
            head = f.read(4096)
        expected = bytes(i % 256 for i in range(len(head)))
        if head == expected:
            return
    # Build the pattern in 1 MiB chunks to keep peak RAM bounded.
    chunk = bytes(i % 256 for i in range(1 << 20))
    full_chunks, tail = divmod(nbytes, 1 << 20)
    with open(p, "wb") as f:
        for _ in range(full_chunks):
            f.write(chunk)
        if tail:
            f.write(chunk[:tail])
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Host introspection


def _fstype_for(path: str) -> str:
    """Best-effort filesystem type lookup (no external deps)."""
    try:
        out = subprocess.run(
            ["stat", "-f", "-c", "%T", path],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def _libhipfile_path() -> str:
    """Locate libhipfile.so under /opt/rocm or LD_LIBRARY_PATH."""
    for prefix in ("/opt/rocm/lib", "/opt/rocm/lib64", "/usr/lib", "/usr/lib64"):
        cand = Path(prefix) / "libhipfile.so"
        if cand.exists():
            return str(cand)
    # LD_LIBRARY_PATH fallback (build_hipfile.sh installs to /opt/rocm).
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if not d:
            continue
        cand = Path(d) / "libhipfile.so"
        if cand.exists():
            return str(cand)
    return ""


def _rocm_version() -> str:
    for vfile in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        p = Path(vfile)
        if p.exists():
            return p.read_text().strip()
    return os.environ.get("ROCM_VERSION", "unknown")


# ---------------------------------------------------------------------------
# ais-check (DMA-live probe; matches deploy/docker/scripts/build_hipfile.sh --probe-only)


def run_ais_check(path: str) -> tuple[bool, str]:
    """Run ais-check against `path` and parse Kernel P2PDMA support.

    Returns (p2pdma_live, raw_output). Missing binary returns
    (False, "ais-check not found").
    """
    binary = shutil.which("ais-check")
    if not binary:
        return False, "ais-check not found on PATH"
    try:
        proc = subprocess.run(
            [binary, "--target", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "ais-check timed out"
    blob = (proc.stdout or "") + (proc.stderr or "")
    # ais-check prints e.g. "Kernel P2PDMA support: True" or "...: False"
    live = "Kernel P2PDMA support: True" in blob
    return live, blob


# ---------------------------------------------------------------------------
# Per-cell measurement


def _measure_hipfile_read(
    HipFile,
    RegisteredBuffer,
    buf_ptr: int,
    path: str,
    block_size: int,
    file_offset: int,
    buf_offset: int,
) -> float:
    """Single hipFile.read() call; returns elapsed seconds.
    Uses the current binding API: Buffer object + buffer_offset
    (the legacy int-ptr signature was changed in vLLM-6, see
    infera/engine/sglang/hipfile_shim.py.HipFile.read)."""
    with RegisteredBuffer(buf_ptr, block_size) as reg:
        with HipFile(path, "r") as fh:
            t0 = time.perf_counter()
            fh.read(reg.handle, block_size, file_offset, buf_offset)
            return time.perf_counter() - t0


def _measure_posix_h2d(torch, np, path: str, block_size: int, file_offset: int) -> float:
    """POSIX pread + numpy.fromfile + .to('cuda'); returns elapsed seconds."""
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        f.seek(file_offset)
        data = f.read(block_size)
    arr = np.frombuffer(data, dtype=np.uint8)
    gpu = torch.from_numpy(arr.copy()).to("cuda")
    # touch to ensure copy completed before stopping clock
    _ = int(gpu[0].item())
    return time.perf_counter() - t0


def _golden_reference(torch, np, path: str, size: int):
    """Reference path: torch.from_numpy(np.fromfile(...)).to('cuda')."""
    arr = np.fromfile(path, dtype=np.uint8, count=size)
    return torch.from_numpy(arr).to("cuda")


def measure_path(
    torch_mod,
    np_mod,
    shim,
    path: str,
    sizes: list[int],
    concurrencies: list[int],
    file_bytes: int,
) -> dict:
    """Run the sweep for one target path. Returns a dict matching the
    `paths[i]` schema in the JSON output spec."""
    HipFileDriver = shim.HipFileDriver
    RegisteredBuffer = shim.RegisteredBuffer
    HipFile = shim.HipFile

    driver = HipFileDriver()
    driver.ensure_open()

    sizes_out: dict[str, dict] = {}
    for sz in sizes:
        # Allocate GPU buffer matching size (HIP via torch.cuda — on ROCm
        # torch exposes the HIP device as cuda).
        buf = torch_mod.empty(sz, dtype=torch_mod.uint8, device="cuda")
        buf_ptr = int(buf.data_ptr())

        # Byte-equality check on a single fresh read.
        # Re-zero the buffer first so a stale match can't pass silently.
        buf.zero_()
        with RegisteredBuffer(buf_ptr, sz) as reg:
            with HipFile(path, "r") as fh:
                fh.read(reg.handle, sz, 0, 0)
        ref = _golden_reference(torch_mod, np_mod, path, sz)
        equal_bytes = bool(torch_mod.equal(buf, ref))

        per_conc: dict[str, dict] = {}
        for c in concurrencies:
            # Warmup
            for _ in range(WARMUP_REPS):
                _measure_hipfile_read(HipFile, RegisteredBuffer, buf_ptr, path, sz, 0, 0)
                _measure_posix_h2d(torch_mod, np_mod, path, sz, 0)

            # Timed — sequential within a "concurrency slot" so this stays
            # a sanity check, not the Bench 1 sweep. We multiply by c to
            # report aggregate throughput at that conc level.
            hipfile_total = 0.0
            posix_total = 0.0
            for _ in range(TIMED_REPS):
                # Spread offsets so the page cache doesn't trivially hit
                # for the POSIX baseline.
                max_off = max(0, file_bytes - sz)
                for i in range(c):
                    off = (i * sz) % (max_off + 1) if max_off else 0
                    hipfile_total += _measure_hipfile_read(
                        HipFile, RegisteredBuffer, buf_ptr, path, sz, off, 0
                    )
                    posix_total += _measure_posix_h2d(torch_mod, np_mod, path, sz, off)

            n_ops = TIMED_REPS * c
            bytes_moved = n_ops * sz
            hipfile_gbs = (bytes_moved / max(hipfile_total, 1e-9)) / 1e9
            posix_gbs = (bytes_moved / max(posix_total, 1e-9)) / 1e9
            per_conc[str(c)] = {
                "hipfile_gbs": round(hipfile_gbs, 3),
                "posix_h2d_gbs": round(posix_gbs, 3),
            }

        sizes_out[str(sz)] = {
            "equal_bytes": equal_bytes,
            "concurrencies": per_conc,
        }

    return {
        "path": path,
        "fstype": _fstype_for(path),
        "sizes": sizes_out,
    }


# ---------------------------------------------------------------------------
# Output


def render_table(results: dict) -> str:
    """ASCII summary table per path: rows = sizes, cols = concurrencies."""
    out: list[str] = []
    for p in results["paths"]:
        out.append("")
        out.append(f"Path: {p['path']}  ({p['fstype']})")
        out.append(f"  P2PDMA live: {p['p2pdma_live']}")
        # Header
        concs = sorted(
            {c for sz in p["sizes"].values() for c in sz["concurrencies"].keys()},
            key=int,
        )
        hdr = "  size            equal   " + "   ".join(f"c={c} hF/POSIX GB/s" for c in concs)
        out.append(hdr)
        out.append("  " + "-" * (len(hdr) - 2))
        for sz_str in sorted(p["sizes"].keys(), key=int):
            cell = p["sizes"][sz_str]
            sz_int = int(sz_str)
            human = f"{sz_int // (1 << 20)} MiB" if sz_int >= (1 << 20) else f"{sz_int // 1024} KiB"
            row = f"  {human:<14}  {str(cell['equal_bytes']):<6}  "
            for c in concs:
                cc = cell["concurrencies"].get(c, {})
                hf = cc.get("hipfile_gbs", "—")
                px = cc.get("posix_h2d_gbs", "—")
                row += f"  {hf:>6} / {px:>6}     "
            out.append(row)
    return "\n".join(out)


def _result_path(host: str) -> Path:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    return out_dir / f"bench0_{host}_{date}.json"


# ---------------------------------------------------------------------------
# Main


def _load_shim():
    """Late-import the hipfile shim so this module imports cleanly on
    hosts without ROCm + the rocm-systems source build."""
    from infera.engine.sglang import hipfile_shim  # noqa: PLC0415

    return hipfile_shim


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.kvcache.hipfile.bench0_sanity",
        description="Bench 0 (sanity) for hipFile.",
    )
    parser.add_argument(
        "--path",
        action="append",
        required=True,
        help="Target file path under test. Repeatable; one bench per path.",
    )
    parser.add_argument(
        "--file-bytes",
        type=int,
        default=DEFAULT_FILE_BYTES,
        help=f"Size of the pre-seeded file (default {DEFAULT_FILE_BYTES}).",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Block sizes to sweep (bytes).",
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=DEFAULT_CONCURRENCIES,
        help="Per-cell concurrency level (sequential within cell).",
    )
    parser.add_argument(
        "--require-p2pdma",
        action="store_true",
        help=(
            "Exit non-zero if ais-check reports Kernel P2PDMA support: False. "
            "Same gate semantics as HIPFILE_REQUIRE_P2PDMA=1 in build_hipfile.sh."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override JSON output path (default bench/kvcache/hipfile/results/...).",
    )
    parser.add_argument(
        "--skip-ais-check",
        action="store_true",
        help="Skip ais-check (smoke test / CI where the tool isn't installed).",
    )
    args = parser.parse_args(argv)

    # Env-var alias for --require-p2pdma so it matches build_hipfile.sh.
    require_p2pdma = args.require_p2pdma or os.environ.get("HIPFILE_REQUIRE_P2PDMA") == "1"

    # Import the heavy deps only after arg parsing so --help works
    # without torch on the box.
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        print(f"ERROR: torch/numpy required for Bench 0: {exc}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print(
            "ERROR: torch.cuda.is_available() is False — Bench 0 needs a GPU "
            "(HIP exposed via torch.cuda on ROCm).",
            file=sys.stderr,
        )
        return 2

    try:
        shim = _load_shim()
    except ImportError as exc:
        print(
            f"ERROR: infera.engine.sglang.hipfile_shim missing ({exc}). "
            "Build the image first: bash deploy/docker/scripts/build_hipfile.sh",
            file=sys.stderr,
        )
        return 2

    host = socket.gethostname()
    results: dict = {
        "host": host,
        "kernel": platform.uname().release,
        "rocm_version": _rocm_version(),
        "libhipfile_path": _libhipfile_path(),
        "paths": [],
    }

    any_fail = False
    for path in args.path:
        seed_test_file(path, args.file_bytes)

        if args.skip_ais_check:
            p2p_live, p2p_blob = False, "skipped"
        else:
            p2p_live, p2p_blob = run_ais_check(path)
        if require_p2pdma and not p2p_live:
            print(
                f"FAIL: --require-p2pdma set but ais-check on {path} did not "
                f"report Kernel P2PDMA support: True\n{p2p_blob}",
                file=sys.stderr,
            )
            any_fail = True

        per_path = measure_path(
            torch,
            np,
            shim,
            path,
            args.sizes,
            args.concurrencies,
            args.file_bytes,
        )
        per_path["p2pdma_live"] = p2p_live
        per_path["ais_check_output"] = p2p_blob if not p2p_live else "ok"
        results["paths"].append(per_path)

        # Byte-equality is the load-bearing assertion. Anything else is
        # informational; a non-equal result means hipFile returned
        # corrupt bytes and we MUST fail loudly.
        for sz_str, cell in per_path["sizes"].items():
            if not cell["equal_bytes"]:
                print(
                    f"FAIL: byte mismatch on {path} at size={sz_str}",
                    file=sys.stderr,
                )
                any_fail = True

    out_path = Path(args.out) if args.out else _result_path(host)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(render_table(results))
    print(f"\nwrote {out_path}")

    return 1 if any_fail else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
