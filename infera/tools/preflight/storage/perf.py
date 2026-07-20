###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Storage throughput probe. Two measurements on a local NVMe mount:

  - NVMe↔DRAM: multi-threaded chunked pwrite/pread under the io_mode + worker
    count kvd itself resolves for this mount (via ``infera.kvd.storage_classify``),
    so the number is what kvd's L3 actually gets — not a generic ceiling. Falls
    back to O_DIRECT-when-supported + cpu-count workers if storage_classify is
    unavailable.
  - NVMe↔HBM: single-stream, staged through a pinned host bounce buffer
    (pread→H2D / D2H→pwrite) — the CPU-bounce KV load/save path kvd uses without
    P2PDMA; a lower bound (P2PDMA-direct hipFile DMA is faster). Needs torch + a
    GPU, so it runs in the engine container and is skipped otherwise.

Standalone (no infera.kvd import). The target not being local NVMe is a
deliberate FAIL (the check's whole point: KV cache must land on fast local NVMe);
unexpected errors degrade to a warn and never raise into the run.

Target dir: ``INFERA_PREFLIGHT_STORAGE_PATH`` if set, else the largest writable
local NVMe-backed mount (auto-detected via lsblk's device tree, so an md-raid of
NVMe or an NVMe partition both resolve to the physical ``nvme`` transport).
Volume: ``INFERA_PREFLIGHT_STORAGE_GB`` GB (default 4; 0 skips). Temp files under
the target are removed afterwards.
"""

from __future__ import annotations

import json
import mmap
import os
import shutil
import tempfile
import threading
import time

from ..finding import Finding
from ..util import run

_ALIGN = 4096  # O_DIRECT buffer/offset alignment
_CHUNK = 128 << 20  # 128 MiB per request (a KV-cache-shaped chunk)
_WORKERS = min(8, os.cpu_count() or 8)
_DEFAULT_GB = 4.0


def _size_gb_env() -> float:
    try:
        return float(os.environ.get("INFERA_PREFLIGHT_STORAGE_GB", _DEFAULT_GB))
    except ValueError:
        return _DEFAULT_GB


def _kvd_io_config(target: str) -> tuple[bool | None, int | None, str]:
    """The io_mode + workers kvd resolves for this mount, via the same
    ``storage_classify`` functions the daemon calls at startup. Returns
    ``(o_direct, workers, rationale)``; ``(None, None, "")`` if unavailable so
    the caller falls back to the standalone O_DIRECT-probe + cpu-count default."""
    try:
        from pathlib import Path

        from infera.kvd.storage_classify import pick_io_mode, pick_workers_per_shard

        o_direct, io_rat = pick_io_mode(Path(target))
        workers, w_rat = pick_workers_per_shard(Path(target))
        return bool(o_direct), int(workers), f"kvd-resolved: {io_rat}; workers={workers} ({w_rat})"
    except Exception:  # noqa: BLE001 - best-effort; fall back to standalone defaults
        return None, None, ""


def _nvme_backed_mounts() -> dict[str, dict]:
    """``{mountpoint: {"dev": name}}`` for every mount whose backing device — or
    an ancestor of it (md-raid member, parent disk of a partition) — is NVMe.
    Walks lsblk's JSON tree so md/dm/partition layers resolve to the physical
    ``nvme`` transport; falls back to the ``nvme`` name prefix for containers
    whose /sys doesn't populate the TRAN column."""
    rc, out = run(["lsblk", "-o", "NAME,TRAN,MOUNTPOINT", "--json"], merge_stderr=False)
    if rc != 0:
        return {}
    try:
        tree = json.loads(out).get("blockdevices", [])
    except (ValueError, TypeError, AttributeError):
        return {}

    found: dict[str, dict] = {}

    def walk(node: dict, anc_is_nvme: bool) -> None:
        name = (node.get("name") or "").strip()
        tran = (node.get("tran") or "").strip().lower()
        is_nvme = anc_is_nvme or tran == "nvme" or name.startswith("nvme")
        mp = (node.get("mountpoint") or "").strip()
        if mp and mp != "[SWAP]" and is_nvme:
            found.setdefault(mp, {"dev": name})
        for child in node.get("children") or []:
            walk(child, is_nvme)

    for dev in tree:
        walk(dev, False)
    return found


def _pick_target() -> tuple[str | None, str]:
    """``(dir, how)`` — the dir to test and a human-readable rationale. Honors
    ``INFERA_PREFLIGHT_STORAGE_PATH``; else the largest writable NVMe-backed
    mount. ``dir`` is None when nothing suitable is found."""
    override = os.environ.get("INFERA_PREFLIGHT_STORAGE_PATH", "").strip()
    if override:
        return override, f"explicit INFERA_PREFLIGHT_STORAGE_PATH={override}"
    ranked: list[tuple[int, str, dict]] = []
    for mp, meta in _nvme_backed_mounts().items():
        try:
            ranked.append((shutil.disk_usage(mp).free, mp, meta))
        except OSError:
            continue
    ranked.sort(reverse=True)
    for free, mp, meta in ranked:
        # Must be a writable *directory*: Docker bind-mounts /etc/resolv.conf etc.
        # as files, so a node whose docker/root disk is NVMe shows a phantom nvme
        # "mount" at a file path — writable but not a usable KV dir.
        if os.path.isdir(mp) and os.access(mp, os.W_OK):
            return (
                mp,
                f"auto: largest local NVMe mount (dev={meta['dev']}, {free / 1e9:.0f}GB free)",
            )
    return None, "no writable local NVMe-backed mount found"


def _nvme_backing(path: str) -> str | None:
    """The NVMe device backing ``path``, or None if the mount covering it isn't
    local NVMe. ``findmnt -T`` resolves the covering mount (so a subdir of an NVMe
    mount still counts), which we then look up in the NVMe-backed mount set — the
    deterministic 'is KV cache actually on local NVMe?' check."""
    # -f: one line even when the device is mounted at several points (else the
    # multi-line output wouldn't match a mountpoint key).
    rc, out = run(["findmnt", "-f", "-T", path, "-no", "TARGET"], merge_stderr=False)
    if rc != 0 or not out.strip():
        return None
    mp = out.strip().splitlines()[0]
    return (_nvme_backed_mounts().get(mp) or {}).get("dev")


def _odirect_ok(d: str) -> bool:
    """True if O_DIRECT writes work under ``d`` (xfs/ext4/NVMe: yes; tmpfs and
    some overlay/NFS: no). Probes with one aligned 4K write, then cleans up."""
    if not getattr(os, "O_DIRECT", 0):
        return False
    p = os.path.join(d, ".odtest")
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
        b = mmap.mmap(-1, _ALIGN)
        os.pwrite(fd, memoryview(b)[:_ALIGN], 0)
        os.close(fd)
        return True
    except OSError:
        return False
    finally:
        try:
            os.unlink(p)
        except OSError:
            pass


def _run_threads(fn, workers: int) -> float:
    """Run ``fn(i)`` on ``workers`` threads; return the max per-worker wall time
    (the phase isn't done until the slowest worker finishes)."""
    res = [0.0] * workers
    ths = []
    for i in range(workers):
        t = threading.Thread(target=lambda i=i: res.__setitem__(i, fn(i)))
        t.start()
        ths.append(t)
    for t in ths:
        t.join()
    return max(res) or 1e-9


def _measure(target_dir: str, size_gb: float, workers: int, prefer_direct: bool) -> dict:
    per = max(1, (int(size_gb * 1e9) // workers) // _CHUNK)  # chunks per worker file
    nbytes = per * workers * _CHUNK
    d = tempfile.mkdtemp(prefix=".preflight_stperf_", dir=target_dir)
    files = [os.path.join(d, f"f{i}.bin") for i in range(workers)]
    buf = mmap.mmap(-1, _CHUNK)  # anonymous mmap is page-aligned → O_DIRECT-safe
    buf.write(os.urandom(1 << 20).ljust(_CHUNK, b"\0"))  # 1 MiB random head, rest zeros
    view = memoryview(buf)
    # Honor kvd's resolved io_mode, but only when the FS actually supports
    # O_DIRECT (tmpfs/some NFS reject it); else buffered.
    odf = os.O_DIRECT if (prefer_direct and _odirect_ok(d)) else 0

    def w(i: int) -> float:
        fd = os.open(files[i], os.O_WRONLY | os.O_CREAT | os.O_TRUNC | odf, 0o644)
        t0 = time.monotonic()
        for c in range(per):
            os.pwrite(fd, view, c * _CHUNK)
        os.fsync(fd)
        os.close(fd)
        return time.monotonic() - t0

    def r(i: int) -> float:
        fd = os.open(files[i], os.O_RDONLY | odf)
        t0 = time.monotonic()
        off, end = 0, per * _CHUNK
        while off < end:
            n = os.preadv(fd, [view], off)
            if n <= 0:
                break
            off += n
        os.close(fd)
        return time.monotonic() - t0

    try:
        w_gbps = nbytes / _run_threads(w, workers) / 1e9
        r_gbps = nbytes / _run_threads(r, workers) / 1e9
    finally:
        for f in files:
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            os.rmdir(d)
        except OSError:
            pass
    return {
        "write_gbps": round(w_gbps, 1),
        "read_gbps": round(r_gbps, 1),
        "io_mode": "O_DIRECT" if odf else "buffered",
        "workers": workers,
        "chunk_mib": _CHUNK >> 20,
        "vol_gb": round(nbytes / 1e9, 1),
    }


def _nvme_hbm_staged(target_dir: str, size_gb: float, torch) -> dict:
    """End-to-end NVMe↔HBM through a pinned host bounce buffer — the CPU-bounce
    path kvd falls back to without P2PDMA. store: D2H → host → pwrite; load:
    pread → host → H2D. Single stream, no read/copy overlap, so it's a
    conservative lower bound; on P2PDMA hosts the direct hipFile DMA is faster.
    O_DIRECT (when supported) keeps the read off the page cache so it hits NVMe."""
    n = max(1, int(size_gb * 1e9) // _CHUNK)
    nbytes = n * _CHUNK
    d = tempfile.mkdtemp(prefix=".preflight_hbm_", dir=target_dir)
    path = os.path.join(d, "hbm.bin")
    host = torch.empty(_CHUNK, dtype=torch.uint8, pin_memory=True)  # page-aligned pinned bounce
    gpu = torch.empty(_CHUNK, dtype=torch.uint8, device="cuda:0")
    mv = memoryview(host.numpy())
    odf = os.O_DIRECT if _odirect_ok(d) else 0
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | odf, 0o644)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for c in range(n):
            host.copy_(gpu)  # D2H
            torch.cuda.synchronize()
            os.pwrite(fd, mv, c * _CHUNK)  # host → NVMe
        os.fsync(fd)
        os.close(fd)
        store = nbytes / (time.monotonic() - t0) / 1e9

        fd = os.open(path, os.O_RDONLY | odf)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for c in range(n):
            os.preadv(fd, [mv], c * _CHUNK)  # NVMe → host
            gpu.copy_(host)  # H2D
            torch.cuda.synchronize()
        load = nbytes / (time.monotonic() - t0) / 1e9
        os.close(fd)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
        try:
            os.rmdir(d)
        except OSError:
            pass
    return {
        "load_gbps": round(load, 1),
        "store_gbps": round(store, 1),
        "io_mode": "O_DIRECT" if odf else "buffered",
        "vol_gb": round(nbytes / 1e9, 1),
    }


def _hbm_finding(target: str, gb: float) -> Finding:
    """Measure NVMe↔HBM if torch + a GPU are present (engine container); otherwise
    record a skip. Best-effort — a failure is a warn, never fatal."""
    try:
        from ..gpu.perf import _torch_gpu

        torch, info = _torch_gpu()
        if torch is None:
            return Finding("info", f"NVMe↔HBM test skipped ({info})", {})
        h = _nvme_hbm_staged(target, gb, torch)
    except Exception as e:  # noqa: BLE001 - best-effort probe
        return Finding("warn", "NVMe↔HBM test failed", {"target": target, "err": str(e)})
    h["note"] = "CPU-bounce via host; GPU-direct/P2PDMA (see Firmware section) is faster"
    return Finding(
        "info",
        f"NVMe↔HBM (single-stream, staged): load {h['load_gbps']} GB/s, "
        f"store {h['store_gbps']} GB/s",
        h,
    )


def collect() -> list[Finding]:
    gb = _size_gb_env()
    if gb <= 0:
        return [Finding("info", "storage throughput test skipped (STORAGE_GB=0)", {})]
    target, how = _pick_target()
    # The point of the test is confirming KV cache lands on local NVMe. If the
    # target dir isn't NVMe-backed (no NVMe mount, or an explicit path that fell
    # back to the overlay fs), that's a real misconfig — KV cache would be slow —
    # so FAIL deterministically rather than reporting a misleadingly-OK number.
    if not target or not os.path.isdir(target):
        return [Finding("fail", "no local NVMe storage for KV cache", {"reason": how})]
    dev = _nvme_backing(target)
    if not dev:
        return [
            Finding(
                "fail",
                "KV storage target is not local NVMe (KV cache would be slow)",
                {"target": target, "how": how},
            )
        ]

    findings: list[Finding] = []
    # NVMe ↔ host DRAM under the io_mode + workers kvd resolves for this mount
    # (falls back to O_DIRECT-probe + cpu-count when storage_classify is absent).
    kvd_direct, kvd_workers, kvd_rat = _kvd_io_config(target)
    workers = kvd_workers or _WORKERS
    prefer_direct = kvd_direct if kvd_direct is not None else True
    try:
        m = _measure(target, gb, workers, prefer_direct)
        m["target"], m["how"], m["dev"] = target, how, dev
        if kvd_rat:
            m["config_source"] = kvd_rat
        suffix = "  (kvd-resolved io_mode/workers)" if kvd_rat else ""
        findings.append(
            Finding(
                "info",
                f"NVMe↔DRAM throughput: write {m['write_gbps']} GB/s, "
                f"read {m['read_gbps']} GB/s{suffix}",
                m,
            )
        )
    except Exception as e:  # noqa: BLE001 - best-effort probe
        findings.append(
            Finding("warn", "NVMe↔DRAM throughput test failed", {"target": target, "err": str(e)})
        )
    # NVMe ↔ HBM: the real KV load/save path (needs a GPU; runs in-container).
    findings.append(_hbm_finding(target, gb))
    return findings
