###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""GPU perf: per-GPU bf16 GEMM TFLOPS and HBM copy bandwidth.

Needs torch on GPU (present in the engine image). Best-effort: if torch or a GPU
is unavailable each check records an info finding and skips, so a host-only run
still works. A card is flagged when it falls well below the node median (relative
outlier — identical GPUs perform within a few percent of each other).
"""

from __future__ import annotations

import statistics
import time

from ..finding import Finding

# GEMM: square bf16 matmul, rotating through a ~1 GiB input pool so the L2 cache
# can't re-serve the same operands and inflate the number.
_SIZE = 8192
_WARMUP = 100
_ITERS = 1000
_BUF_BYTES = 1 << 30

# HBM copy: a ~1 GiB random buffer (>> L2, so every access hits HBM).
_HBM_BYTES = 1 << 30
_HBM_WARMUP = 20
_HBM_ITERS = 100

# P2P: copy a ~512 MiB buffer between every GPU pair (large enough to saturate xGMI).
_P2P_BYTES = 1 << 29
_P2P_WARMUP = 3
_P2P_ITERS = 10

# Flag a GPU/link below this fraction of the node median.
_SLOW_FRAC = 0.9

# MI355X (CDNA4, gfx950) absolute floors — only applied on MI355X, since other
# hardware has different theoretical numbers.
_MI355X_GFX = "gfx950"
_MI355X_GEMM_MIN_TFLOPS = 1300.0  # bf16, 8192 square GEMM
_MI355X_P2P_MIN_GBPS = 50.0  # per xGMI link (spec 76.8 GB/s/direction)


def _torch_gpu():
    """Return (torch, device_count) or (None, reason)."""
    try:
        import torch
    except Exception as e:  # noqa: BLE001 - torch is only in the engine image
        return None, f"torch unavailable: {e}"
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None, "no visible GPU"
    return torch, torch.cuda.device_count()


def _is_mi355x(torch, device: int) -> bool:
    try:
        return torch.cuda.get_device_properties(device).gcnArchName.startswith(_MI355X_GFX)
    except Exception:  # noqa: BLE001
        return False


def _gemm_tflops(torch, device: int) -> float:
    dev = f"cuda:{device}"
    bytes_per_matrix = _SIZE * _SIZE * 2  # bf16 = 2 bytes/element
    nbuf = max(1, _BUF_BYTES // (2 * bytes_per_matrix))  # a pair (A + B) per buffer
    a = [torch.randn((_SIZE, _SIZE), device=dev, dtype=torch.bfloat16) for _ in range(nbuf)]
    b = [torch.randn((_SIZE, _SIZE), device=dev, dtype=torch.bfloat16) for _ in range(nbuf)]

    torch.cuda.synchronize(device)
    for i in range(_WARMUP):
        torch.matmul(a[i % nbuf], b[i % nbuf])
    torch.cuda.synchronize(device)
    start = time.time()
    for i in range(_ITERS):
        torch.matmul(a[i % nbuf], b[i % nbuf])
    torch.cuda.synchronize(device)

    dt = (time.time() - start) / _ITERS
    return 2 * _SIZE**3 / dt / 1e12


def _hbm_gbps(torch, device: int) -> float:
    dev = f"cuda:{device}"
    src = torch.randn(_HBM_BYTES // 2, device=dev, dtype=torch.bfloat16)  # bf16 = 2 bytes
    dst = torch.empty_like(src)

    torch.cuda.synchronize(device)
    for _ in range(_HBM_WARMUP):
        dst.copy_(src)
    torch.cuda.synchronize(device)
    start = time.time()
    for _ in range(_HBM_ITERS):
        dst.copy_(src)
    torch.cuda.synchronize(device)

    dt = (time.time() - start) / _HBM_ITERS
    return 2 * _HBM_BYTES / dt / 1e9  # read src + write dst


def _outliers(vals: dict[int, float], label: str, unit: str) -> list[Finding]:
    if len(vals) < 3:
        return []
    med = statistics.median(vals.values())
    return [
        Finding("warn", f"gpu{i} {label} outlier", {unit: v, "node_median": round(med, 1)})
        for i, v in sorted(vals.items())
        if v < med * _SLOW_FRAC
    ]


def collect_gemm() -> list[Finding]:
    torch, info = _torch_gpu()
    if torch is None:
        return [Finding("info", f"GPU compute check skipped ({info})", {})]

    findings: list[Finding] = []
    tflops: dict[int, float] = {}
    for i in range(info):
        try:
            tflops[i] = round(_gemm_tflops(torch, i), 1)
            findings.append(
                Finding(
                    "info",
                    f"gpu{i} GEMM (bf16)",
                    {"tflops": tflops[i], "dtype": "bf16", "size": _SIZE},
                )
            )
        except Exception as e:  # noqa: BLE001
            findings.append(Finding("warn", f"gpu{i} GEMM (bf16) failed", {"error": str(e)}))
    findings += _outliers(tflops, "compute", "tflops")
    for i, tf in sorted(tflops.items()):
        if tf < _MI355X_GEMM_MIN_TFLOPS and _is_mi355x(torch, i):
            findings.append(
                Finding(
                    "warn",
                    f"gpu{i} below MI355X bf16 GEMM floor",
                    {"tflops": tf, "floor": _MI355X_GEMM_MIN_TFLOPS},
                )
            )
    return findings


def collect_hbm() -> list[Finding]:
    torch, info = _torch_gpu()
    if torch is None:
        return [Finding("info", f"HBM bandwidth check skipped ({info})", {})]

    findings: list[Finding] = []
    gbps: dict[int, float] = {}
    buf_gib = round(_HBM_BYTES / 1024**3, 1)
    for i in range(info):
        try:
            gbps[i] = round(_hbm_gbps(torch, i), 1)
            findings.append(
                Finding("info", f"gpu{i} HBM copy", {"gb_s": gbps[i], "buf_gib": buf_gib})
            )
        except Exception as e:  # noqa: BLE001
            findings.append(Finding("warn", f"gpu{i} HBM copy failed", {"error": str(e)}))
    return findings + _outliers(gbps, "bandwidth", "gb_s")


def _p2p_gbps(torch, src, dst, di: int, dj: int) -> float:
    def sync():
        torch.cuda.synchronize(di)
        torch.cuda.synchronize(dj)

    sync()
    for _ in range(_P2P_WARMUP):
        dst.copy_(src)
    sync()
    start = time.time()
    for _ in range(_P2P_ITERS):
        dst.copy_(src)
    sync()
    dt = (time.time() - start) / _P2P_ITERS
    return _P2P_BYTES / dt / 1e9


def collect_p2p() -> list[Finding]:
    torch, info = _torch_gpu()
    if torch is None:
        return [Finding("info", f"P2P bandwidth check skipped ({info})", {})]
    if info < 2:
        return [Finding("info", "P2P check needs >= 2 GPUs; skipped", {"gpu_count": info})]

    bufs = [
        torch.empty(_P2P_BYTES // 2, device=f"cuda:{d}", dtype=torch.bfloat16) for d in range(info)
    ]
    matrix: list[list[float | None]] = [[None] * info for _ in range(info)]
    offdiag: list[tuple[int, int, float]] = []
    findings: list[Finding] = []
    for i in range(info):
        for j in range(info):
            if i == j:
                continue
            try:
                bw = round(_p2p_gbps(torch, bufs[i], bufs[j], i, j), 1)
                matrix[i][j] = bw
                offdiag.append((i, j, bw))
            except Exception as e:  # noqa: BLE001
                findings.append(Finding("warn", f"gpu{i}->{j} P2P failed", {"error": str(e)}))

    detail = {"matrix": matrix, "unit": "GB/s", "buf_mib": _P2P_BYTES // 1024 // 1024}
    findings.insert(0, Finding("info", "P2P bandwidth matrix", detail))

    vals = [bw for _, _, bw in offdiag]
    if len(vals) >= 3:
        med = statistics.median(vals)
        for i, j, bw in offdiag:
            if bw < med * _SLOW_FRAC:
                findings.append(
                    Finding("warn", f"gpu{i}->{j} link slow", {"gb_s": bw, "median": round(med, 1)})
                )

    mi355x = {d for d in range(info) if _is_mi355x(torch, d)}
    for i, j, bw in offdiag:
        if bw < _MI355X_P2P_MIN_GBPS and i in mi355x and j in mi355x:
            findings.append(
                Finding(
                    "warn",
                    f"gpu{i}->{j} below MI355X xGMI floor",
                    {"gb_s": bw, "floor": _MI355X_P2P_MIN_GBPS},
                )
            )
    return findings
