###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Startup L3 storage self-check for the kvd daemon.

Writes then reads a few kvd-shaped chunks under the **same io_mode / workers /
chunk size kvd resolves for the long-path** (via ``storage_classify``) and logs
the measured write + read GB/s. Operators then see, right in the daemon boot log,
whether the configured L3 storage is fast enough — instead of discovering a slow
mount only via degraded TTFT later.

Best-effort: never raises into daemon startup. On by default; disable with
``INFERA_KVD_STORAGE_SELFCHECK=0``; size via
``INFERA_KVD_SELFCHECK_GB`` (default 2). The standalone, more featureful
version (P2PDMA verdict, read/write worker split, --force-parallel) lives in
``infera/kvd/bench/probe.py``.
"""

from __future__ import annotations

import logging
import mmap
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger("infera.kvd.storage_selfcheck")
_ALIGN = 4096


def _enabled() -> bool:
    # On by default — operators should see L3 throughput at boot. Disable with
    # INFERA_KVD_STORAGE_SELFCHECK=0 (e.g. tiny/ephemeral test mounts).
    return os.environ.get("INFERA_KVD_STORAGE_SELFCHECK", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _abuf(n: int) -> mmap.mmap:
    return mmap.mmap(-1, (n + _ALIGN - 1) // _ALIGN * _ALIGN)


def _resolve(path: Path) -> tuple[bool, int, int, str]:
    """(o_direct, workers, chunk_bytes, rationale) — exactly how kvd drives it."""
    o_direct, workers, why = False, 8, "defaults"
    try:
        from infera.kvd.storage_classify import pick_io_mode, pick_workers_per_shard

        o_direct, io_why = pick_io_mode(path)
        workers, w_why = pick_workers_per_shard(path, n_shards=1)
        why = f"{io_why}; {w_why}"
    except Exception as exc:  # never block startup
        why = f"storage_classify unavailable ({exc}); conservative defaults"
    chunk = max(1, int(os.environ.get("INFERA_KVD_CHUNK_TARGET_MIB", "128"))) << 20
    return bool(o_direct), int(workers or 8), chunk, why


def _run(workers: int, fn) -> float:
    res, ths = [0.0] * workers, []
    for i in range(workers):
        t = threading.Thread(target=lambda i=i: res.__setitem__(i, fn(i)), daemon=True)
        t.start()
        ths.append(t)
    for t in ths:
        t.join()
    return max(res) or 1e-9


def run_storage_selfcheck(
    long_path: str,
    *,
    size_gb: float | None = None,
    force: bool = False,
    write_workers: int | None = None,
    read_workers: int | None = None,
    o_direct: bool | None = None,
    chunk_bytes: int | None = None,
    label: str = "L3",
    extra: str = "",
) -> dict | None:
    """Measure + log write/read GB/s for ``long_path``. Returns the metrics
    dict (or None if skipped/failed). Safe to call at daemon/connector startup.

    By default the io_mode / workers / chunk are resolved via
    ``storage_classify`` (the daemon-tablespace POSIX path). The kvd
    **connector** (aggregated or PD) resolves its OWN write/load worker counts
    (load is clamped to 1 without P2PDMA) and gpu_direct decision, so it passes
    them explicitly via ``write_workers`` / ``read_workers`` / ``o_direct`` and a
    descriptive ``label`` (e.g. ``"L3 connector"``) + ``extra`` note (e.g.
    ``"gpu_direct=on p2pdma=True"``). The probed bytes still flow over POSIX so
    the number is the storage substrate's throughput at the connector's
    resolved parallelism — the GPU-direct DMA leg is reported in ``extra`` but
    not exercised at startup (would need a live GPU buffer)."""
    if not (force or _enabled()):
        return None
    try:
        d = Path(long_path) / ".kvd_selfcheck"
        d.mkdir(parents=True, exist_ok=True)
        r_od, r_workers, r_chunk, why = _resolve(Path(long_path))
        o_direct = r_od if o_direct is None else bool(o_direct)
        chunk = r_chunk if chunk_bytes is None else int(chunk_bytes)
        w_workers = int(write_workers) if write_workers else r_workers
        rd_workers = (
            int(read_workers) if read_workers else (w_workers if write_workers else r_workers)
        )
        if write_workers or read_workers or chunk_bytes is not None:
            why = (extra + "; " if extra else "") + (
                f"connector-resolved w={w_workers} r={rd_workers}"
            )
        elif extra:
            why = f"{extra}; {why}"
        total = int((size_gb or float(os.environ.get("INFERA_KVD_SELFCHECK_GB", "2"))) * 1e9)
        # write phase drives w_workers; read phase drives rd_workers (may differ:
        # the connector clamps load to 1 without P2PDMA). Stage one file per
        # read worker so the read phase has real per-worker parallelism.
        workers = max(w_workers, rd_workers)
        per = max(1, (total // workers) // chunk)
        files = [str(d / f"sc_{i}.bin") for i in range(workers)]
        buf = _abuf(chunk)
        buf.write(os.urandom(min(chunk, 1 << 20)).ljust(chunk, b"\0")[:chunk])
        view = memoryview(buf)[:chunk]

        # Re-shard the files across whatever worker count each phase drives, so
        # write (w_workers) and read (rd_workers) can differ — the connector's
        # no-P2PDMA load clamp (rd_workers=1) is then measured honestly.
        w_assign = [files[i::w_workers] for i in range(w_workers)]
        r_assign = [files[i::rd_workers] for i in range(rd_workers)]

        def w(i):
            fl = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_TRUNC
                | (getattr(os, "O_DIRECT", 0) if o_direct else 0)
            )
            t0 = time.monotonic()
            for f in w_assign[i]:
                fd = os.open(f, fl, 0o600)
                for c in range(per):
                    os.pwrite(fd, view, c * chunk)
                os.fsync(fd)
                os.close(fd)
            return time.monotonic() - t0

        def r(i):
            t0 = time.monotonic()
            for f in r_assign[i]:
                fd = os.open(f, os.O_RDONLY | (getattr(os, "O_DIRECT", 0) if o_direct else 0))
                off, n = 0, per * chunk
                while off < n:
                    got = os.preadv(fd, [view], off)
                    if got <= 0:
                        break
                    off += got
                os.close(fd)
            return time.monotonic() - t0

        nbytes = per * workers * chunk
        w_gbps = nbytes / _run(w_workers, w) / 1e9
        r_gbps = nbytes / _run(rd_workers, r) / 1e9
        for f in files:
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            d.rmdir()
        except OSError:
            pass
        wlabel = (
            f"workers={w_workers}"
            if w_workers == rd_workers
            else f"write_workers={w_workers} read_workers={rd_workers}"
        )
        logger.info(
            "[kvd] %s storage self-check (%s): WRITE %.2f GB/s  READ %.2f GB/s  "
            "[io_mode=%s %s chunk=%dMiB vol=%.1fGB] (%s)",
            label,
            long_path,
            w_gbps,
            r_gbps,
            "O_DIRECT" if o_direct else "buffered",
            wlabel,
            chunk >> 20,
            nbytes / 1e9,
            why,
        )
        return {
            "write_gbps": w_gbps,
            "read_gbps": r_gbps,
            "o_direct": o_direct,
            "write_workers": w_workers,
            "read_workers": rd_workers,
            "chunk_mib": chunk >> 20,
            "label": label,
        }
    except Exception as exc:  # pragma: no cover — best-effort
        logger.warning("[kvd] L3 storage self-check failed (non-fatal): %s", exc)
        return None
