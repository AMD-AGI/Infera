###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Storage-aware io_mode + workers_per_shard selection for Infera-kvd's L3 read path.

Bench data on the MI355X testbed showed that the unconditional ``O_DIRECT`` we
adopted for the tablespace long region is the wrong default on a slow
local SATA SSD and on NFS-low-nconnect — both cases lose to buffered
IO, where the kernel page cache + readahead is doing the work that
``O_DIRECT`` deliberately bypasses. The winning configuration depends
on the underlying transport, not on the filesystem alone:

  - **NVMe SSD**         → O_DIRECT (random pread storm is bound by
                           device queue depth, no readahead win)
  - **SAS enterprise SSD** → O_DIRECT (same shape as NVMe at the
                           kernel block layer)
  - **SATA SSD**         → buffered (slow random-read latency means
                           readahead is the only thing keeping
                           throughput up)
  - **Rotational HDD**   → buffered (seek-bound; readahead mandatory)
  - **iSCSI / FC SAN**   → buffered (network-bound, behaves like NFS)
  - **NFS / NFS4**       → buffered (kernel readahead is the only
                           knob that gets you anywhere near nconnect's
                           theoretical multiplexing limit)
  - **tmpfs / ramfs**    → buffered (O_DIRECT unsupported)
  - **unknown**          → buffered (conservative)

The same probe also feeds the ``workers_per_shard`` decision for the
striped long region's intra-shard pread fan-out. Per-device queue depth
caps the throughput-per-shard at very different points depending on the
underlying transport:

  - **NVMe / SAS SSD**   → 8  (per-device queue saturates ~8 in-flight)
  - **SATA SSD / RAID1** → 4  (single SATA drive queue depth saturates)
  - **HDD**              → 2  (seek penalty caps concurrency)
  - **tmpfs / ramfs**    → 8  (RAM-backed, no queue limit)
  - **NFS nconnect≥32**  → min(8, nconnect/4) (cap by network slots)
  - **NFS nconnect 16-31** → 4 (mid-tier)
  - **NFS nconnect≤15**  → 2 (low — quick contention point)
  - **iSCSI / FC SAN**   → 4 (conservative SAN block)
  - **unknown**          → 4 (conservative)

A CPU-count guardrail caps the final pick at
``max(2, cpu_count() // n_shards)`` so 8-shard configs on small boxes
never spin up more threads than the box can productively schedule.

The probe is a chain of cheap subprocess calls — ``findmnt`` and
``lsblk`` — both shipped by util-linux on every modern distro. If
either binary is missing (containers, exotic systems) we fall back to
the conservative "buffered" + workers=4 default and log a WARN.

The chain handles md-raid, LVM, dm-crypt, and bind mounts transparently
by walking ``lsblk -no NAME,TRAN,ROTA`` recursively. For mixed-device
arrays (e.g. NVMe + SATA in the same md0) the worst-case device wins
— a single SATA member in an mdraid pulls the whole array into the
"buffered" bucket.

Public API:

  - ``classify_storage(path)``       → ``StorageInfo``
  - ``pick_io_mode(path)``           → ``(o_direct: bool, rationale: str)``
  - ``pick_workers_per_shard(path, *, n_shards=1)``
                                     → ``(workers: int, rationale: str)``

Both pickers are pure-Python (stdlib + ``subprocess.run``), no external
deps, and MUST NOT raise on weird devices — callers depend on the
"conservative buffered / workers=4" fallback for unknown shapes.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Transports that act like a network (latency dominated, readahead
# essential). Always buffered. NVMe/SAS get O_DIRECT (handled inline
# in pick_io_mode); SATA/USB/unknown fall through to buffered.
_NETWORK_TRANSPORTS = ("iscsi", "fc", "fcoe")
# Filesystems where O_DIRECT is straight-up unsupported.
_NO_DIRECT_FSTYPES = ("tmpfs", "ramfs")
# Network filesystems — buffered regardless of underlying transport.
_NETWORK_FSTYPES = ("nfs", "nfs4", "cifs", "smb", "smb3")


@dataclass
class DeviceInfo:
    """One underlying physical device behind a mount point."""

    dev: str  # e.g. "sda", "nvme0n1" — bare name, no /dev/ prefix
    transport: str  # "nvme" / "sata" / "sas" / "iscsi" / "fc" / "usb" / ""
    rotational: bool  # True for HDD, False for SSD/NVMe


@dataclass
class StorageInfo:
    """Result of probing a path's underlying storage."""

    path: Path
    mount_source: str  # e.g. "/dev/md0", "vast-server:/export"
    fs_type: str  # "ext4" / "xfs" / "nfs" / "nfs4" / "tmpfs" / ...
    nconnect: int | None  # NFS only, parsed from /proc/mounts options
    # NFS only: maximum per-RPC payload size in bytes. Parsed from
    # rsize=/wsize= mount options. Vast servers support
    # up to 16 MB; kernel-default 1 MB fragments each 4 MB Infera slot
    # write into 4 RPCs. The pick_io_mode result includes a warning when
    # wsize < 8 MB on Vast-class NFS.
    rsize_bytes: int | None = None
    wsize_bytes: int | None = None
    devices: list[DeviceInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Subprocess helpers
# ----------------------------------------------------------------------


def _run(cmd: list[str], timeout: float = 2.0) -> subprocess.CompletedProcess[str] | None:
    """Run ``cmd``; return the CompletedProcess on success, ``None`` on
    any failure (binary missing, timeout, non-zero exit, OSError). All
    failures are logged at DEBUG — callers fall back to conservative
    defaults silently."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("storage_classify: %s failed: %s", cmd[0], exc)
        return None


def _findmnt(path: Path) -> tuple[str, str] | None:
    """Return ``(source, fstype)`` for the mount containing ``path``.

    Uses ``findmnt --target`` so bind mounts and nested mounts resolve
    correctly. Returns ``None`` if findmnt is missing or fails.
    """
    r = _run(["findmnt", "--target", str(path), "--noheadings", "--output", "SOURCE,FSTYPE"])
    if r is None or r.returncode != 0:
        return None
    line = r.stdout.strip()
    if not line:
        return None
    # findmnt prints aligned columns separated by whitespace. The source
    # field can contain spaces in exotic cases (it shouldn't on normal
    # mounts), but for our purposes the last token is fstype.
    parts = line.split()
    if len(parts) < 2:
        return None
    fstype = parts[-1]
    source = " ".join(parts[:-1])
    return source, fstype


def _parse_lsblk_devices(stdout: str) -> list[tuple[str, str, bool]]:
    """Parse ``lsblk -no NAME,TRAN,ROTA`` output into
    ``[(dev, transport, rotational)]``.

    lsblk pads columns with leading tree-drawing characters for child
    devices (``├─sda1``, ``└─sda2``). We strip those.

    Only **leaf** devices are returned (partitions/lvs/etc that have no
    children in the output). For a partition table we return the
    partitions, not the whole disk; for an mdraid we return the
    member partitions, not md0 itself. The caller treats every leaf as
    an underlying device.

    For lsblk's TRAN column: it's populated on the **disk** row, not
    on partition rows. lsblk inherits the disk's transport for its
    partitions via the recursive walk we trigger with the disk name as
    target. To handle this robustly we propagate the most recent
    non-empty transport seen as we walk the tree — partitions inherit
    the parent disk's transport.
    """
    leaves: list[tuple[str, str, bool]] = []
    rows: list[tuple[int, str, str, str]] = []  # (depth, name, tran, rota)
    for line in stdout.splitlines():
        if not line.strip():
            continue
        # lsblk uses unicode box-drawing for the tree. Each nesting
        # level is 2 columns wide; the branch char (``├`` / ``└``) sits
        # at column ``2 * (depth - 1)`` for depth >= 1, and the root has
        # no branch char at all. Vertical bars (``│``) and spaces in the
        # prefix mark "parent at this level is still going on" without
        # changing the current row's depth — so depth = floor(prefix_width
        # / 2) + 1 if a branch char is present, else 0 for the root.
        prefix_len = 0
        for ch in line:
            if ch in (" ", "│", "├", "└", "─"):
                prefix_len += 1
            else:
                break
        prefix = line[:prefix_len]
        if "├" in prefix or "└" in prefix:
            depth = prefix_len // 2
        else:
            depth = 0
        stripped = line[prefix_len:]
        parts = stripped.split()
        if not parts:
            continue
        name = parts[0]
        # TRAN and ROTA are the remaining columns. lsblk left-pads with
        # blanks → the split eats them; but if TRAN is empty we'd
        # collapse the columns. To keep things simple we treat
        # everything we got: parts[1] is TRAN if present, parts[2] is
        # ROTA if present. Real lsblk always emits ROTA (0 or 1).
        tran = parts[1] if len(parts) >= 2 and parts[1] not in ("0", "1") else ""
        # ROTA is the last 0/1 token.
        rota_token = parts[-1] if parts[-1] in ("0", "1") else ""
        rows.append((depth, name, tran, rota_token))

    # Identify leaves: a row is a leaf if no row immediately after it
    # has greater depth.
    for i, (depth, name, tran, _rota) in enumerate(rows):
        is_leaf = True
        for j in range(i + 1, len(rows)):
            next_depth = rows[j][0]
            if next_depth > depth:
                is_leaf = False
                break
            if next_depth <= depth:
                break
        if not is_leaf:
            continue
        # Walk back to find the most recent ancestor with a non-empty
        # transport — partitions inherit from the parent disk.
        leaf_tran = tran
        if not leaf_tran:
            for j in range(i - 1, -1, -1):
                prev_depth, _pn, prev_tran, _pr = rows[j]
                if prev_depth < depth and prev_tran:
                    leaf_tran = prev_tran
                    break
        # Same for rotational — but ROTA is always present per row in
        # real lsblk output, so we keep the row's own value.
        leaf_rota = rows[i][3] == "1"
        leaves.append((name, leaf_tran, leaf_rota))
    return leaves


def _lsblk_for_source(source: str) -> list[DeviceInfo]:
    """Walk ``source`` (e.g. ``/dev/md0``) into underlying leaf devices.

    Returns an empty list if lsblk is missing or the source isn't a
    block device (e.g. NFS ``server:/export``). Errors are non-fatal —
    callers fall back to "unknown transport, buffered".

    For a partition source (e.g. ``/dev/sda2``), the forward-walk
    returns just the partition itself with an empty TRAN column — lsblk
    only populates TRAN on the parent disk row. We follow up with an
    ``--inverse`` walk for any leaf that came back without a transport,
    and inherit the parent disk's TRAN/ROTA. This matters in practice
    because findmnt usually returns ``SOURCE=/dev/sdX2`` for a /
    partition, not the bare disk.
    """
    # Only attempt lsblk on block device sources. NFS sources look
    # like ``server:/export``; tmpfs is just ``tmpfs``.
    if not source.startswith("/"):
        return []
    r = _run(["lsblk", "-no", "NAME,TRAN,ROTA", source])
    if r is None or r.returncode != 0:
        return []
    leaves = _parse_lsblk_devices(r.stdout)
    resolved: list[DeviceInfo] = []
    for name, tran, rotational in leaves:
        if not tran:
            # Walk up via lsblk --inverse to find the parent disk's
            # transport. Cheap (one subprocess) and only triggers on
            # the leaf path.
            parent_tran, parent_rota = _lsblk_inverse_parent(f"/dev/{name}")
            if parent_tran:
                tran = parent_tran
            # If the leaf says rotational=0 but the parent says 1, the
            # truth is the underlying disk — but lsblk's ROTA on a
            # partition reflects the disk anyway, so this rarely fires.
            if parent_rota is not None and not rotational:
                rotational = parent_rota
        resolved.append(DeviceInfo(dev=name, transport=tran, rotational=rotational))
    return resolved


def _lsblk_inverse_parent(devpath: str) -> tuple[str, bool | None]:
    """Walk ``lsblk --inverse <devpath>`` to find the parent disk's
    TRAN + ROTA. Returns ``("", None)`` on failure.

    Output shape:

        sda2          0
        └─sda  sata   0     ← the parent we want

    The bottom-most row (greatest depth) is the parent disk.
    """
    r = _run(["lsblk", "--inverse", "-no", "NAME,TRAN,ROTA", devpath])
    if r is None or r.returncode != 0:
        return "", None
    parents = _parse_lsblk_devices(r.stdout)
    # In an --inverse walk, the parent disk is the deepest row with a
    # non-empty transport. _parse_lsblk_devices only returns leaves of
    # the forward walk — for --inverse, the "leaf" IS the parent disk.
    for _name, tran, rota in parents:
        if tran:
            return tran, rota
    return "", None


def _nfs_mount_opts(source: str, path: Path) -> str:
    """Return the raw mount-options string for the NFS mount covering
    ``path``. Empty string if /proc/mounts isn't readable or no NFS
    mount matches. Used by both the nconnect and rsize/wsize pickers
    to avoid re-walking /proc/mounts."""
    try:
        with open("/proc/mounts") as f:
            mounts = f.read()
    except OSError:
        return ""
    target = str(path)
    best_mp = ""
    best_opts = ""
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        src, mp, fs, opts = parts[0], parts[1], parts[2], parts[3]
        if fs not in _NETWORK_FSTYPES:
            continue
        if src == source or target == mp or target.startswith(mp.rstrip("/") + "/"):
            if len(mp) > len(best_mp):
                best_mp, best_opts = mp, opts
    return best_opts


def _nconnect_for_nfs(source: str, path: Path) -> int | None:
    """Parse ``nconnect=N`` from /proc/mounts for an NFS mount.

    Returns ``None`` if /proc/mounts isn't readable, the mount isn't
    found, or nconnect isn't set (kernel default = 1, but we report
    None to signal "operator left it unset").
    """
    opts = _nfs_mount_opts(source, path)
    if not opts:
        return None
    m = re.search(r"nconnect=(\d+)", opts)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _rsize_wsize_for_nfs(source: str, path: Path) -> tuple[int | None, int | None]:
    """Parse ``rsize=N`` and ``wsize=N`` from /proc/mounts for an NFS
    mount. Values are in bytes. Returns (None, None) when mount or
    options aren't present.

    Kernel-default rsize=wsize=1MB on Linux fragments
    each 4 MB Infera L3 slot write into 4 RPCs. Vast servers support
    up to 16 MB. Bumping the mount to wsize=16MB cuts that to 1 RPC
    per slot — a near-free +25-40% on NFS write throughput.
    """
    opts = _nfs_mount_opts(source, path)
    if not opts:
        return None, None
    r = re.search(r"\brsize=(\d+)", opts)
    w = re.search(r"\bwsize=(\d+)", opts)
    rsize = int(r.group(1)) if r else None
    wsize = int(w.group(1)) if w else None
    return rsize, wsize


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def classify_storage(path: Path) -> StorageInfo:
    """Probe ``path`` and return a populated ``StorageInfo``.

    Never raises — exotic mounts, missing binaries, weird containers
    all collapse to a ``StorageInfo`` with ``fs_type='unknown'`` and an
    empty ``devices`` list. Callers (``pick_io_mode``) then fall back
    to conservative buffered IO.
    """
    info = StorageInfo(path=path, mount_source="", fs_type="unknown", nconnect=None)
    fm = _findmnt(path)
    if fm is None:
        info.warnings.append("findmnt unavailable or failed; defaulting to buffered")
        return info
    source, fstype = fm
    info.mount_source = source
    info.fs_type = fstype
    if fstype in _NETWORK_FSTYPES:
        info.nconnect = _nconnect_for_nfs(source, path)
        info.rsize_bytes, info.wsize_bytes = _rsize_wsize_for_nfs(source, path)
        # Warn loudly when wsize is the kernel default
        # (1 MB), which fragments each 4 MB slot write into 4 RPCs.
        # Vast / WekaFS / EFS all support 16 MB; remounting with
        # wsize=16777216 is a near-free +25-40% on NFS write throughput.
        WSIZE_LOW_THRESHOLD = 8 * 1024 * 1024
        if info.wsize_bytes is not None and info.wsize_bytes < WSIZE_LOW_THRESHOLD:
            info.warnings.append(
                f"NFS wsize={info.wsize_bytes // 1024}KB is below the recommended "
                f"{WSIZE_LOW_THRESHOLD // (1024 * 1024)}MB. With 4 MB slot writes "
                f"this fragments each write into multiple RPCs. Remount with "
                f"wsize=16777216,rsize=16777216 for +25-40% write throughput on "
                f"Vast/WekaFS-class servers."
            )
        return info
    if fstype in _NO_DIRECT_FSTYPES:
        return info
    devices = _lsblk_for_source(source)
    if not devices:
        info.warnings.append(
            f"lsblk returned no devices for source={source!r}; defaulting to buffered"
        )
    info.devices = devices
    return info


def _device_severity(d: DeviceInfo) -> int:
    """Rank a device by how badly O_DIRECT would hurt it.

    Higher = worse for O_DIRECT. We pick the max across devices in a
    multi-device array so a single SATA member in an mdraid pulls the
    whole array into "buffered".

      0  NVMe SSD              (O_DIRECT ideal)
      1  SAS SSD               (O_DIRECT fine)
      2  SATA SSD              (buffered better — readahead win)
      3  Unknown SSD           (conservative)
      4  Network SAN (iscsi/fc) (always buffered)
      5  USB                   (slow, buffered)
      6  Rotational HDD        (mandatory buffered)
    """
    if d.rotational:
        return 6
    if d.transport == "usb":
        return 5
    if d.transport in _NETWORK_TRANSPORTS:
        return 4
    if d.transport == "nvme":
        return 0
    if d.transport == "sas":
        return 1
    if d.transport == "sata":
        return 2
    return 3  # unknown SSD


def pick_io_mode(path: Path) -> tuple[bool, str]:
    """Decide O_DIRECT vs buffered for ``path``.

    Returns ``(o_direct, rationale)`` — the rationale is a short
    human-readable string for the startup log. Never raises.
    """
    try:
        info = classify_storage(path)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "storage_classify: classify_storage(%s) raised %s; falling back to buffered",
            path,
            exc,
        )
        return False, f"classify_storage error ({exc}); conservative buffered"

    if info.fs_type in _NETWORK_FSTYPES:
        nc = info.nconnect if info.nconnect is not None else "?"
        return False, f"NFS-buffered (nconnect={nc})"
    if info.fs_type in _NO_DIRECT_FSTYPES:
        return False, f"{info.fs_type} (O_DIRECT unsupported)"
    if not info.devices:
        for w in info.warnings:
            logger.warning("storage_classify: %s", w)
        return False, "unknown device, conservative buffered"

    worst = max(info.devices, key=_device_severity)
    if worst.transport == "nvme" and not worst.rotational:
        return True, f"nvme-ssd ({worst.dev})"
    if worst.transport == "sas" and not worst.rotational:
        return True, f"sas-ssd ({worst.dev})"
    if worst.transport == "sata" and not worst.rotational:
        return False, f"sata-ssd ({worst.dev}) → buffered for cold-read readahead"
    if worst.rotational:
        return False, f"rotational HDD ({worst.dev}) → buffered required for readahead"
    if worst.transport in _NETWORK_TRANSPORTS:
        return False, f"SAN {worst.transport} ({worst.dev}) → buffered (network like NFS)"
    if worst.transport == "usb":
        return False, f"usb ({worst.dev}) → buffered"
    logger.warning(
        "storage_classify: unknown transport %r on %s; conservative buffered",
        worst.transport,
        worst.dev,
    )
    return False, f"unknown transport '{worst.transport}' ({worst.dev}), conservative buffered"


# ----------------------------------------------------------------------
# ad-hoc CLI (`infera-kvd-classify <path>` via __main__)
# ----------------------------------------------------------------------


def format_decision(path: Path) -> str:
    """Render a multi-line summary suitable for the startup log /
    inspection CLI. Used by both ``server._main_async`` and
    ``infera.kvd.__main__``.
    """
    info = classify_storage(path)
    o_direct, rationale = pick_io_mode(path)
    devs = (
        ", ".join(
            f"{d.dev} ({d.transport or '?'}, {'hdd' if d.rotational else 'ssd'})"
            for d in info.devices
        )
        or "(none)"
    )
    mode = "DIRECT" if o_direct else "BUFFERED"
    lines = [
        f"L3 io_mode: {mode} (auto)",
        f"  path     = {info.path}",
        f"  mount    = {info.mount_source or '?'} ({info.fs_type})",
        f"  devices  = [{devs}]",
    ]
    if info.fs_type in _NETWORK_FSTYPES:
        lines.append(f"  nconnect = {info.nconnect if info.nconnect is not None else '?'}")
        if info.rsize_bytes is not None or info.wsize_bytes is not None:
            r_str = f"{info.rsize_bytes // 1024}KB" if info.rsize_bytes else "?"
            w_str = f"{info.wsize_bytes // 1024}KB" if info.wsize_bytes else "?"
            lines.append(f"  rsize    = {r_str}, wsize = {w_str}")
    lines.append(f"  rationale: {rationale}")
    lines.append("  override: --io-mode {direct,buffered} or INFERA_KVD_IO_MODE=direct")
    # Surface NFS-mount warnings (wsize < 8MB on Vast/Weka)
    # at the decision-block level so the startup log shows them inline.
    for w in info.warnings:
        lines.append(f"  WARN: {w}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# workers_per_shard auto-pick (task #258 — companion of pick_io_mode)
# ----------------------------------------------------------------------


def pick_workers_per_shard(path: Path, *, n_shards: int = 1) -> tuple[int, str]:
    """Decide intra-shard pread fan-out for ``path``.

    Returns ``(workers_per_shard, rationale)``. The rationale is a short
    human-readable string for the startup log. Never raises — exotic
    devices collapse to the conservative ``4`` default.

    Decision matrix (see module docstring):

      - NVMe / SAS SSD    → 8 (per-device queue depth)
      - SATA SSD / RAID1  → 4
      - HDD               → 2 (seek penalty)
      - tmpfs / ramfs     → 8 (RAM-backed)
      - NFS nconnect≥32   → min(8, max(2, nconnect//4))
      - NFS nconnect 16-31 → 4
      - NFS nconnect≤15   → 2
      - iSCSI / FC SAN    → 4 (conservative)
      - unknown           → 4 (conservative, log WARN)

    Plus a CPU-count guardrail: never exceed
    ``max(2, cpu_count() // n_shards)`` so 8-shard configs on tiny
    boxes don't thrash. With ``n_shards=0`` the guardrail is skipped.
    """
    try:
        info = classify_storage(path)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "storage_classify: classify_storage(%s) raised %s; falling back to workers=4",
            path,
            exc,
        )
        return 4, f"classify_storage error ({exc}); conservative default"

    if info.fs_type in _NETWORK_FSTYPES:
        nconn = info.nconnect if info.nconnect is not None else 8
        if nconn >= 32:
            # Cap at 8; bottom-out at 2 so a freakishly low nconnect//4
            # never gets clamped to 1 (which would defeat the fan-out).
            picked = min(8, max(2, nconn // 4))
            why = f"NFS nconnect={nconn} → min(8, nconnect/4) = {picked}"
        elif nconn >= 16:
            picked = 4
            why = f"NFS nconnect={nconn} (mid-tier)"
        else:
            picked = 2
            why = f"NFS nconnect={nconn} (low — quick contention point)"
    elif info.fs_type in _NO_DIRECT_FSTYPES:
        picked = 8
        why = f"{info.fs_type} (RAM-backed, no I/O queue limit)"
    elif info.devices:
        worst = max(info.devices, key=_device_severity)
        if worst.rotational:
            picked = 2
            why = f"rotational HDD ({worst.dev}) — seek penalty caps concurrency"
        elif worst.transport == "nvme":
            picked = 8
            why = f"nvme-ssd ({worst.dev}) — high queue depth"
        elif worst.transport == "sas":
            picked = 8
            why = f"sas-ssd ({worst.dev}) — high queue depth"
        elif worst.transport == "sata":
            picked = 4
            why = f"sata-ssd ({worst.dev}) — single-drive queue depth"
        elif worst.transport in _NETWORK_TRANSPORTS:
            picked = 4
            why = f"SAN {worst.transport} ({worst.dev}) — conservative network"
        else:
            logger.warning(
                "storage_classify: unknown transport %r on %s; "
                "workers_per_shard=4 conservative default",
                worst.transport,
                worst.dev,
            )
            picked = 4
            why = f"unknown transport '{worst.transport}' ({worst.dev}), conservative default"
    else:
        for w in info.warnings:
            logger.warning("storage_classify: %s", w)
        picked = 4
        why = "no device info, conservative default"

    # CPU guardrail. We only apply it when n_shards > 0 — n_shards=0
    # is a degenerate caller intent (no striping, picker would have
    # been called with n_shards=1 in normal use). The floor is 2 so we
    # never collapse to a single worker (which defeats the per-shard
    # fan-out entirely).
    if n_shards > 0:
        cpu = os.cpu_count() or 4
        cap = max(2, cpu // n_shards)
        if picked > cap:
            picked = cap
            why += f" (capped at {cap} by cpu={cpu} / n_shards={n_shards})"

    return picked, why


def format_workers_decision(
    info: StorageInfo,
    workers: int,
    rationale: str,
    *,
    n_shards: int = 1,
) -> str:
    """Render a multi-line summary of the workers_per_shard decision,
    parallel in shape to ``format_decision`` for io_mode. Used by both
    ``server._main_async`` and ``infera.kvd.__main__``.
    """
    devs = (
        ", ".join(
            f"{d.dev} ({d.transport or '?'}, {'hdd' if d.rotational else 'ssd'})"
            for d in info.devices
        )
        or "(none)"
    )
    lines = [
        f"L3 workers_per_shard: {workers} (auto)",
        f"  path     = {info.path}",
        f"  mount    = {info.mount_source or '?'} ({info.fs_type})",
        f"  devices  = [{devs}]",
        f"  fs_type  = {info.fs_type}",
        f"  cpu      = {os.cpu_count() or '?'}",
        f"  n_shards = {n_shards}",
    ]
    if info.fs_type in _NETWORK_FSTYPES:
        lines.append(f"  nconnect = {info.nconnect if info.nconnect is not None else '?'}")
    lines.append(f"  rationale: {rationale}")
    lines.append("  override: --workers-per-shard N or INFERA_KVD_WORKERS_PER_SHARD=N")
    return "\n".join(lines)


def format_workers_decision_for_path(path: Path, *, n_shards: int = 1) -> str:
    """Convenience wrapper — does the classify + pick + format in one
    call. Used by the ``classify`` subcommand for ad-hoc inspection."""
    info = classify_storage(path)
    workers, rationale = pick_workers_per_shard(path, n_shards=n_shards)
    return format_workers_decision(info, workers, rationale, n_shards=n_shards)
