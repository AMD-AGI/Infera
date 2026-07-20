###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tablespace-pattern SSD region for infera-kvd (Phase B).

The Phase 4.0 `LongStorageRegion` uses one file per cached block. That
works up to ~100K entries; beyond that the filesystem metadata layer
(inode lookup, dentry cache, directory tree) becomes the bottleneck.

This module implements the alternative: **pre-allocated container
files with an internal bitset allocator** — same pattern 3FS's
chunk_engine uses, and the standard approach for storage engines
(PostgreSQL heap files, RocksDB SSTs, MySQL tablespaces).

## Layout

```
<region_dir>/
    containers/
        0000.bin         ← pre-allocated, fixed size (e.g. 1 GB each)
        0001.bin
        ...
    index.log            ← append-only ops journal
    index.snapshot.json  ← periodic compacted index
    index.snapshot.json.bak
```

- File count is a **small constant** (= max_bytes / container_bytes).
  64 GB region with 1 GB containers = 64 files total, regardless of
  how many blocks live inside.
- Each container is divided into fixed `slot_bytes` slots (default
  64 KB, matching the typical packed-KV block size on TP=2 models).
- A bitset in RAM tracks which slots are free. Allocation is
  in-memory; persistence is the journal.

## Crash safety

The journal is append-only and fsynced after every PUT. Sequence:

  1. Reserve a slot in the in-memory allocator (RAM only)
  2. `pwrite` the value to the container file's slot offset, then `fsync`
  3. Append the PUT entry to the journal, then `fsync`
  4. (Optionally) Update the in-memory index entry

If we crash between (2) and (3), the slot is allocated on disk but
not recorded → on restart we replay the journal, see no PUT for that
slot, treat it as free → safe (we'll overwrite it later). If we
crash between (3) and (4)... no, the index update is in-memory, so
(3) and (4) are atomic from the perspective of a restart.

Snapshot rotation: on graceful `shutdown()` (or periodic compaction
in Phase B.2), we write the current index to `index.snapshot.json`
atomically (temp + rename + checksum + `.bak` rotation, same shape
as the file-per-block manifest), then truncate the journal. On
startup we load the snapshot first, then replay any remaining
journal entries.

## What's deferred to Phase B.2+

- **`O_DIRECT`** — skip page cache for reads/writes. Requires
  4-KB-aligned buffers; doable but complicates the byte path.
- **`io_uring`** — batched async IO. The current `pwrite`/`pread`
  path is sync (fine for correctness validation).
- **Multi-pool slot sizes** — 3FS has 64K / 512K / 4M pools.
  Our first cut is a single fixed slot size; values larger than
  that get rejected (caller falls back to file-per-block region
  or recomputes).
- **Background compaction** — for now we only snapshot on
  `shutdown()`. Periodic compaction (when journal exceeds N MB)
  comes after we have profile data showing it's needed.

## Why not RocksDB

3FS uses RocksDB for the index. We considered it but for B.1:

- Adds a heavy C++ dependency (rocksdb crate ~50 MB)
- Our index ops are ~thousands/sec, not millions — JSON journal
  with periodic snapshot is plenty
- Easier to debug — operators can `cat index.log` to see history
- Easier to recover from corruption — JSON parse failures are
  obvious; RocksDB internal corruption is opaque

When/if we need >10K ops/sec persistent index updates, swap the
`TablespaceJournal` class for an LMDB or RocksDB implementation
behind the same interface.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from infera.kvd.wire import RETENTION_LONG, validate_retention

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Filesystem-aware defaults (auto-detect at start)
# ----------------------------------------------------------------------
#
# The right `o_direct` and `flush_interval_ms` settings vary by
# backend:
#
#   - ext4/xfs/btrfs:  O_DIRECT on, inline fsync (cheap)
#   - nfs/nfs4:        O_DIRECT on, batched fsync 20 ms
#   - wekafs:          O_DIRECT OFF (client writecache does
#                      RDMA coalescing; bypassing it costs 5–11×
#                      throughput), batched fsync 20 ms
#   - unknown:         safe conservative default — buffered, inline
#
# Operators can override by passing explicit `o_direct=` and
# `flush_interval_ms=` to the constructor. Auto-detect kicks in only
# when the constructor receives `None` for that field.
_FS_DEFAULTS: dict[str, dict[str, Any]] = {
    "ext4": {"o_direct": True, "flush_interval_ms": 0},
    "xfs": {"o_direct": True, "flush_interval_ms": 0},
    "btrfs": {"o_direct": True, "flush_interval_ms": 0},
    "nfs": {"o_direct": True, "flush_interval_ms": 20},
    "nfs4": {"o_direct": True, "flush_interval_ms": 20},
    "wekafs": {"o_direct": False, "flush_interval_ms": 20},
    "tmpfs": {"o_direct": False, "flush_interval_ms": 0},
    "overlay": {"o_direct": False, "flush_interval_ms": 0},
}
_FS_DEFAULTS_FALLBACK: dict[str, Any] = {"o_direct": False, "flush_interval_ms": 0}


def _detect_fstype(path: Path) -> str:
    """Return the fstype at ``path``, or 'unknown' if we can't tell.

    Try ``findmnt -T`` first (cleanest — handles nested mounts), then
    fall back to longest-prefix match in /proc/mounts."""
    # findmnt is in util-linux; present on every modern distro but
    # absent in some minimal containers.
    try:
        r = subprocess.run(
            ["findmnt", "-T", str(path), "-o", "FSTYPE", "-n"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if r.returncode == 0:
            fstype = r.stdout.strip()
            if fstype:
                return fstype
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # /proc/mounts fallback — choose the longest mount-point prefix
    # that contains the target. resolve() handles symlinks. Wrap the
    # whole read in a thread with a 2 s timeout: some container
    # runtimes can wedge /proc reads when the kernel is stuck. We
    # don't want kvd startup to hang indefinitely on a misbehaving
    # procfs (PR #9 review fix P1).
    import concurrent.futures

    def _read_proc_mounts() -> str:
        target = str(path.resolve())
        best_mp, best_fs = "", ""
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mp, fs = parts[1], parts[2]
                if target == mp or target.startswith(mp.rstrip("/") + "/"):
                    if len(mp) > len(best_mp):
                        best_mp, best_fs = mp, fs
        return best_fs

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_read_proc_mounts)
            best_fs = future.result(timeout=2.0)
        if best_fs:
            return best_fs
    except (OSError, concurrent.futures.TimeoutError):
        pass
    return "unknown"


def detect_fs_defaults(path: str | Path) -> dict[str, Any]:
    """Return per-backend defaults for the path's filesystem.

    Result has keys: ``fstype`` (str), ``o_direct`` (bool),
    ``flush_interval_ms`` (int). Unknown filesystems get the
    conservative fallback (buffered IO, inline fsync) — safe
    everywhere; operator can opt into faster modes with explicit
    constructor args."""
    fstype = _detect_fstype(Path(path))
    defaults = _FS_DEFAULTS.get(fstype, _FS_DEFAULTS_FALLBACK)
    return {"fstype": fstype, **defaults}


_DEFAULT_SLOT_BYTES = 64 * 1024  # 64 KB — observed avg block size on TP=2 MiniMax-M2.5
_DEFAULT_CONTAINER_BYTES = 1024 * 1024 * 1024  # 1 GB per container file
_JOURNAL_FILENAME = "index.log"
_SNAPSHOT_FILENAME = "index.snapshot.json"
_SNAPSHOT_BAK_FILENAME = "index.snapshot.json.bak"
_SNAPSHOT_TMP_FILENAME = "index.snapshot.json.tmp"

# Direct-IO alignment. Linux requires reads/writes with O_DIRECT to be
# aligned to the filesystem's logical block size — virtually always
# 512 or 4096 bytes. 4096 covers both safely. We pick 4096 statically
# rather than probing per-filesystem because:
# - Probing via `BLKSSZGET` ioctl needs root; we can't do it from a
#   userspace tablespace constructor reliably.
# - Over-aligning (4K when the FS only needs 512) costs nothing.
# - Our slot_bytes default 64 KB is already a 16× multiple of 4096.
_DIRECT_IO_ALIGN_BYTES = 4096


@dataclass
class TablespaceEntry:
    """Per-block metadata for a tablespace region.

    Slot location encoded as ``(container_idx, slot_idx)`` — the byte
    offset is `slot_idx * slot_bytes`. `size_bytes` is the actual
    value size (may be less than `slot_bytes`; the rest of the slot
    is unused dead space until LRU reclaims it).
    """

    key: bytes
    container_idx: int
    slot_idx: int
    size_bytes: int
    retention: str
    model: str = ""
    compat_key: str = ""
    last_access: float = field(default_factory=time.monotonic)
    metadata: dict = field(default_factory=dict)

    def slot_global(self, slots_per_container: int) -> int:
        return self.container_idx * slots_per_container + self.slot_idx


# ----------------------------------------------------------------------
# Bitset allocator
# ----------------------------------------------------------------------


class BitsetAllocator:
    """In-memory bitset tracking which slots in the tablespace are free.

    One bit per slot: 0 = free, 1 = allocated. Internally a bytearray
    so it's compact (1M slots = 128 KB of bitset state).

    ## Allocation strategy: hash-distributed across containers

    Slots are partitioned into ``num_containers`` contiguous ranges of
    ``slots_per_container`` slots each. ``alloc(key_hash=H)`` first
    tries the container ``H % num_containers``; on a full target it
    probes the remaining containers in linear-then-wrap order until it
    finds one with room.

    Each container gets its own cursor + allocated-count so the scan
    inside a container starts from its last free position. A single
    flat bitmap keeps ``free`` / ``mark_used`` / ``is_set`` byte math
    unchanged from the legacy layout — restart replay sets bits by
    slot_global and never has to know which container any slot belongs
    to.

    ## Why this layout (vs sequential-fill)

    The legacy single-cursor allocator picked the lowest free bit, so
    consecutive PUTs clustered into the same container file. After the
    striped sub-pool wired N concurrent workers per shard, all N still
    hit the SAME inode (= same container file), and the kernel
    serialized them at the inode lock — per-device throughput stalled
    around 1.25 GB/s vs ~6 GB/s/device fio peak. Hash-distributing
    across containers makes consecutive PUTs land in DIFFERENT inodes,
    so reads (and writes) fan across all containers from the start.

    The hash is the caller's responsibility — ``alloc()`` only needs
    an integer mod the container count, deterministic across restarts.
    See ``TablespaceLongRegion.put`` for the composite-key blake2b hash
    that's used in practice.

    Thread-safety: the allocator does NOT take its own lock —
    `TablespaceLongRegion` holds its `_lock` across the calls.
    """

    def __init__(
        self,
        total_slots: int,
        *,
        num_containers: int = 1,
        slots_per_container: int | None = None,
    ) -> None:
        if total_slots <= 0:
            raise ValueError(f"total_slots must be positive, got {total_slots}")
        if num_containers < 1:
            raise ValueError(f"num_containers must be >= 1, got {num_containers}")
        # Derive slots_per_container if not given. The default = total_slots
        # (one logical container) keeps the legacy single-container shape
        # for callers that don't care about hash distribution.
        if slots_per_container is None:
            slots_per_container = total_slots // num_containers
        if slots_per_container < 1:
            raise ValueError(
                f"slots_per_container must be >= 1 (total_slots={total_slots}, "
                f"num_containers={num_containers})"
            )
        if num_containers * slots_per_container < total_slots:
            raise ValueError(
                f"num_containers*slots_per_container ({num_containers}*{slots_per_container}"
                f"={num_containers * slots_per_container}) must cover total_slots ({total_slots})"
            )

        self._total_slots = total_slots
        self._num_containers = int(num_containers)
        self._slots_per_container = int(slots_per_container)
        n_bytes = (total_slots + 7) // 8
        self._bits = bytearray(n_bytes)
        self._allocated = 0
        # Per-container state. Cursor is a slot index RELATIVE TO THE
        # CONTAINER (0..slots_per_container-1) where the next intra-
        # container scan starts. Speeds up the common case where the
        # last alloc was at slot N and the next one wants N+1.
        self._per_container_cursor: list[int] = [0] * self._num_containers
        self._per_container_allocated: list[int] = [0] * self._num_containers
        # Sticky flag set the first time the hash-target container is
        # full and we have to fall back. Surfaces "hash distribution
        # failed — region is probably saturated" to operators without
        # spamming the log on every subsequent PUT.
        self._fallback_warning_logged = False

    @property
    def total_slots(self) -> int:
        return self._total_slots

    @property
    def allocated(self) -> int:
        return self._allocated

    @property
    def num_free(self) -> int:
        return self._total_slots - self._allocated

    @property
    def num_containers(self) -> int:
        return self._num_containers

    @property
    def slots_per_container(self) -> int:
        return self._slots_per_container

    def per_container_allocated(self, container_idx: int) -> int:
        """Slots currently allocated inside ``container_idx``. Mostly for
        tests / stats — not on any hot path."""
        if container_idx < 0 or container_idx >= self._num_containers:
            raise ValueError(
                f"container_idx {container_idx} out of range [0, {self._num_containers})"
            )
        return self._per_container_allocated[container_idx]

    def is_set(self, slot: int) -> bool:
        if slot < 0 or slot >= self._total_slots:
            return False
        return bool(self._bits[slot // 8] & (1 << (slot % 8)))

    def _alloc_in_container(self, container_idx: int) -> int | None:
        """Find a free slot inside ``container_idx`` only. Returns the
        global slot index, or None if the container has no free slots.
        Updates per-container cursor + count and the global ``_allocated``.

        Scans starting from the container's cursor (byte-granular), one
        byte at a time, low-to-high bit inside each byte. Wraps once
        within the container. This matches the legacy ``BitsetAllocator``
        semantics (which used a byte-granular cursor): consecutive
        SETs hit consecutive slots; an immediate ``free(s)`` then
        ``alloc(same_target)`` reuses ``s`` because the cursor still
        points at its byte.
        """
        if self._per_container_allocated[container_idx] >= self._slots_per_container:
            return None
        spc = self._slots_per_container
        base = container_idx * self._slots_per_container
        # Cursor is a LOCAL byte index inside the container's slot range.
        # Containers always start on a byte boundary (we own the bitmap
        # range [base, base+spc) and base/spc are typically multiples of
        # 8); when they don't (tail container with spc not divisible by
        # 8) we still iterate by local slot index so out-of-range bits
        # at the tail of the last byte are simply skipped.
        local_cursor = self._per_container_cursor[container_idx]
        # Bytes-per-container, rounded up so the final byte (with maybe
        # only a few valid bits) still gets scanned.
        n_local_bytes = (spc + 7) // 8
        # Convert local byte_idx + local bit -> local slot index, then
        # +base -> global slot.
        for byte_offset in range(n_local_bytes):
            local_byte = (local_cursor + byte_offset) % n_local_bytes
            # Global byte index for this local byte. local_byte * 8 is
            # the local slot index of the first bit in this byte.
            local_slot_base = local_byte * 8
            # Read the relevant slice of the shared bitmap. Slots
            # [base + local_slot_base, base + local_slot_base + 8) live
            # entirely inside a single global byte iff base is byte-
            # aligned — which it always is in our shape (slots_per_container
            # is set from container_bytes // slot_bytes, and the only
            # caller below uses values that make base byte-aligned).
            # For safety we still go bit-by-bit and recompute the
            # global byte_idx per-bit.
            for bit_offset in range(8):
                local_slot = local_slot_base + bit_offset
                if local_slot >= spc:
                    break  # past the end of this container's valid slots
                slot_global = base + local_slot
                if slot_global >= self._total_slots:
                    break
                gbyte = slot_global // 8
                gbit = slot_global % 8
                mask = 1 << gbit
                if self._bits[gbyte] & mask:
                    continue
                self._bits[gbyte] = self._bits[gbyte] | mask
                self._allocated += 1
                self._per_container_allocated[container_idx] += 1
                # Park cursor at the LOCAL byte of the slot we just
                # took. Next call starts scanning at the same byte —
                # finds the taken bit, walks to the next bit (still
                # free if any) → consecutive SETs hit consecutive slots.
                self._per_container_cursor[container_idx] = local_byte
                return slot_global
        return None

    def alloc(self, key_hash: int | None = None) -> int | None:
        """Return the index of a newly-allocated slot, or None if the
        region is full. The slot is marked as allocated.

        When ``key_hash`` is given, the target container is
        ``key_hash % num_containers`` — keys hashing to different
        containers land in different inodes, which is the entire point
        of the refactor (see class docstring). A full target falls back
        to the next container with wrap-around; the first time that
        happens we log one WARN so operators see "hash distribution
        failed — region is saturated" without subsequent PUTs spamming
        the log on every miss.

        When ``key_hash is None`` we keep the legacy sequential-fill
        behavior — container 0 first, then container 1, etc. Callers
        without a stable key (compaction, replay-time placement) use
        this path; HostStore PUTs always supply a hash.
        """
        if self._allocated >= self._total_slots:
            return None

        if key_hash is None:
            # Legacy sequential fill: container 0 → 1 → ... → N-1.
            for ci in range(self._num_containers):
                slot = self._alloc_in_container(ci)
                if slot is not None:
                    return slot
            return None

        # Hash-targeted path with fallback probing.
        target = key_hash % self._num_containers
        slot = self._alloc_in_container(target)
        if slot is not None:
            return slot

        # Target is full. Probe the remaining containers in order
        # starting at target+1, wrapping. (The probe is deterministic;
        # we don't randomize because deterministic fallback is easier
        # to reason about under load — same input shape always picks
        # the same fallback container.)
        if not self._fallback_warning_logged:
            logger.warning(
                "tablespace allocator: hash-target container %d is full; "
                "falling back to next available. This usually means the "
                "region is at/near capacity — subsequent fallbacks will "
                "not be logged (one-shot warning).",
                target,
            )
            self._fallback_warning_logged = True

        for offset in range(1, self._num_containers):
            ci = (target + offset) % self._num_containers
            slot = self._alloc_in_container(ci)
            if slot is not None:
                return slot
        return None

    def free(self, slot: int) -> None:
        """Mark a slot as free. No-op if already free or out of range."""
        if slot < 0 or slot >= self._total_slots:
            return
        byte_idx = slot // 8
        bit = slot % 8
        mask = 1 << bit
        if self._bits[byte_idx] & mask:
            self._bits[byte_idx] &= ~mask
            self._allocated -= 1
            ci = slot // self._slots_per_container
            if 0 <= ci < self._num_containers:
                self._per_container_allocated[ci] = max(0, self._per_container_allocated[ci] - 1)

    def mark_used(self, slot: int) -> bool:
        """For restart replay: assert that ``slot`` is allocated WITHOUT
        going through `alloc()`. Returns True if the bit was newly set,
        False if it was already set (e.g. duplicate journal entry — safe
        to ignore).

        Per-container allocated count is bumped in lock-step. Cursor is
        NOT updated here — the restart finalizer (`recompute_state`)
        resets cursors after all marks land, which keeps replay order
        independent of the cursor state."""
        if slot < 0 or slot >= self._total_slots:
            raise ValueError(f"slot {slot} out of range [0, {self._total_slots})")
        byte_idx = slot // 8
        bit = slot % 8
        mask = 1 << bit
        if self._bits[byte_idx] & mask:
            return False
        self._bits[byte_idx] = self._bits[byte_idx] | mask
        self._allocated += 1
        ci = slot // self._slots_per_container
        if 0 <= ci < self._num_containers:
            self._per_container_allocated[ci] += 1
        return True

    def recompute_state(self) -> None:
        """Recompute per-container allocated counts and cursors from the
        raw bitmap. Called after a bulk replay path that uses
        ``mark_used`` — those calls keep counts in sync as a side
        effect, but the cursors aren't, and a region restarted with
        every slot taken would otherwise scan from cursor=0 for every
        alloc. This rebuilds both authoritatively, so any external
        tampering with the bitmap (tests, recovery tools) is also
        absorbed cleanly.

        Cursor placement: for each container, point the cursor (= LOCAL
        byte index inside the container) at the first byte that
        contains a FREE slot. The next alloc inside that container
        starts scanning there. For full containers the cursor is left
        at 0 — they'll be skipped by the alloc fast path on
        ``per_container_allocated == slots_per_container`` anyway.
        """
        spc = self._slots_per_container
        n_local_bytes = (spc + 7) // 8
        total_allocated = 0
        for ci in range(self._num_containers):
            base = ci * spc
            count = 0
            first_free_local_byte = 0
            found_free = False
            for local in range(spc):
                slot_global = base + local
                if slot_global >= self._total_slots:
                    break
                byte_idx = slot_global // 8
                bit = slot_global % 8
                if self._bits[byte_idx] & (1 << bit):
                    count += 1
                elif not found_free:
                    # local byte INDEX inside the container (slot //8).
                    first_free_local_byte = local // 8
                    found_free = True
            self._per_container_allocated[ci] = count
            self._per_container_cursor[ci] = first_free_local_byte if found_free else 0
            # Defensive clamp on cursor when there are no free slots.
            if not found_free:
                # leave at 0; alloc fast path returns None up front
                self._per_container_cursor[ci] = 0
            else:
                # n_local_bytes is the valid range
                self._per_container_cursor[ci] = min(
                    self._per_container_cursor[ci], n_local_bytes - 1
                )
            total_allocated += count
        self._allocated = total_allocated


# ----------------------------------------------------------------------
# Append-only journal
# ----------------------------------------------------------------------


class TablespaceJournal:
    """Append-only newline-delimited JSON log of index ops.

    Each line is one JSON object:
        {"op": "PUT", "key_hex": "...", "container": 0, "slot": 12,
         "size": 65536, "retention": "long", "model": "...",
         "compat_key": "...", "metadata": {}, "ts": 1779348567.89}
        {"op": "DEL", "key_hex": "...", "model": "...", "compat_key": "..."}

    Writes are fsynced after each append (configurable via constructor
    for tests). Reads tolerate a truncated last line (crash mid-write)
    by stopping replay at the corruption point — anything BEFORE the
    corruption is still valid.
    """

    def __init__(self, path: Path, *, sync_writes: bool = True) -> None:
        self._path = path
        self._sync_writes = sync_writes
        self._fp = None  # type: ignore[var-annotated]
        self._lock = threading.Lock()

    def open(self) -> None:
        """Open for append. Idempotent."""
        if self._fp is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Buffering=0 → unbuffered; we still flush + fsync after each
        # entry, but unbuffered means a partial write is bounded by
        # the kernel's write semantics rather than Python's buffer.
        self._fp = open(self._path, "ab", buffering=0)

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.close()
                except OSError:
                    pass
                self._fp = None

    def append(self, entry: dict[str, Any]) -> None:
        """Append one entry. Caller must have called `open()`."""
        line = json.dumps(entry, separators=(",", ":")).encode("utf-8") + b"\n"
        with self._lock:
            if self._fp is None:
                raise RuntimeError("journal not open")
            self._fp.write(line)
            if self._sync_writes:
                os.fsync(self._fp.fileno())

    def read_all(self) -> list[dict[str, Any]]:
        """Read every parseable entry from the journal. If the file
        doesn't exist, return []. A truncated last line (crash
        mid-write) is dropped silently; entries before it are returned."""
        if not self._path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self._path.open("rb") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entries.append(json.loads(stripped))
                except json.JSONDecodeError:
                    logger.warning("tablespace journal %s: dropped truncated tail line", self._path)
                    break
        return entries

    def truncate(self) -> None:
        """Drop the journal file. Caller should have flushed state to a
        snapshot first, otherwise data is lost."""
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.close()
                except OSError:
                    pass
                self._fp = None
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass
        self.open()

    @property
    def path(self) -> Path:
        return self._path


# ----------------------------------------------------------------------
# Snapshot read/write — mirrors `manifest.py` shape
# ----------------------------------------------------------------------


def _write_snapshot(
    snapshot_path: Path,
    bak_path: Path,
    tmp_path: Path,
    entries: list[dict[str, Any]],
    *,
    slot_bytes: int,
    container_bytes: int,
) -> None:
    """Atomic snapshot write: tmp + fsync + atomic primary swap +
    bak rotation. Includes a sha256 over the sorted-JSON entries.

    Rotation order is *primary-first* (PR #9 review fix P1):

      1. Write tmp + fsync.
      2. If primary exists, hard-link it (or fall back to copy) into
         `bak_path.tmp` — a staging copy that has the OLD primary's
         bytes, accessible under its own name.
      3. `os.replace(tmp_path, snapshot_path)` — atomically replaces
         primary with the new content.
      4. `os.replace(bak_path.tmp, bak_path)` — atomically promotes
         the staged old primary to the bak slot.

    At no intermediate point does `snapshot_path` cease to exist —
    readers ALWAYS see a valid primary (old or new), and they NEVER
    fall back to a `bak` that hasn't been fully transferred. The
    previous order (`primary→bak` then `tmp→primary`) had a small
    window where primary was missing and readers would hit a bak
    that held the previous primary — recoverable, but not as clean.

    Why include slot_bytes + container_bytes in the snapshot: the
    region's geometry (and therefore the meaning of `(container, slot)`)
    depends on those config values. If an operator changes the config,
    we don't want to silently mis-decode old snapshots.
    """
    entries_json = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    checksum = hashlib.sha256(entries_json).hexdigest()
    payload = {
        "version": 1,
        "geometry": {
            "slot_bytes": slot_bytes,
            "container_bytes": container_bytes,
        },
        "checksum": f"sha256:{checksum}",
        "entries": entries,
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("wb") as f:
        f.write(payload_bytes)
        f.flush()
        os.fsync(f.fileno())

    # Stage the old primary as bak.tmp BEFORE we replace primary. Try
    # hard-link first (cheap, atomic on POSIX); fall back to a full
    # copy on filesystems that disallow cross-name links (some NFS
    # configs). Failing this stage is non-fatal — we still get the
    # new snapshot in primary; we just lose the bak rotation.
    bak_tmp_path = bak_path.with_suffix(bak_path.suffix + ".tmp")
    staged_bak = False
    if snapshot_path.exists():
        try:
            if bak_tmp_path.exists():
                bak_tmp_path.unlink()
            os.link(snapshot_path, bak_tmp_path)
            staged_bak = True
        except OSError:
            # Hard-link unsupported → fall back to copy.
            try:
                import shutil as _shutil

                _shutil.copy2(snapshot_path, bak_tmp_path)
                staged_bak = True
            except OSError:
                logger.warning(
                    "tablespace snapshot: failed to stage primary→bak.tmp; "
                    "skipping bak rotation this cycle"
                )

    # Atomic primary swap — readers see either old or new, never gone.
    os.replace(tmp_path, snapshot_path)

    # Promote the staged old primary to bak. If this fails, bak holds
    # the previous-previous snapshot, which is still recoverable —
    # not a hard failure.
    if staged_bak:
        try:
            os.replace(bak_tmp_path, bak_path)
        except OSError:
            logger.warning("tablespace snapshot: failed to promote bak.tmp → bak")


def _read_snapshot(
    snapshot_path: Path,
    bak_path: Path,
    *,
    expected_slot_bytes: int,
    expected_container_bytes: int,
) -> list[dict[str, Any]]:
    """Read a snapshot. Tries primary then `.bak`. Returns [] if both
    fail or if the geometry doesn't match — geometry mismatch means
    the operator changed config and the old snapshot can't be safely
    interpreted; we fall back to journal-only replay (which is
    geometry-agnostic since journal entries are self-describing)."""
    for path in (snapshot_path, bak_path):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_bytes().decode("utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("tablespace snapshot %s: unreadable (%s)", path, exc)
            continue
        if not isinstance(data, dict) or "entries" not in data:
            logger.warning("tablespace snapshot %s: malformed", path)
            continue
        entries = data.get("entries", [])
        # Verify checksum.
        entries_json = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
        expected_checksum = data.get("checksum", "")
        computed = "sha256:" + hashlib.sha256(entries_json).hexdigest()
        if expected_checksum != computed:
            logger.warning(
                "tablespace snapshot %s: checksum mismatch (have %r, expected %r)",
                path,
                expected_checksum,
                computed,
            )
            continue
        # Verify geometry.
        geom = data.get("geometry", {})
        if (
            geom.get("slot_bytes") != expected_slot_bytes
            or geom.get("container_bytes") != expected_container_bytes
        ):
            logger.warning(
                "tablespace snapshot %s: geometry mismatch "
                "(have slot=%s container=%s, expected slot=%d container=%d) — ignoring",
                path,
                geom.get("slot_bytes"),
                geom.get("container_bytes"),
                expected_slot_bytes,
                expected_container_bytes,
            )
            continue
        return entries
    return []


# ----------------------------------------------------------------------
# TablespaceLongRegion
# ----------------------------------------------------------------------


class TablespaceLongRegion:
    """Long-retention SSD region using the tablespace pattern.

    Implements the same public surface as `LongStorageRegion`:
    `start()`, `shutdown()`, `get_bytes()`, `exists()`, `get_entry()`,
    `put()`, `remove()`, `clear()`, `used_bytes`, `entries_count`,
    `max_bytes`. `HostStore` should drop it in transparently — we
    don't inherit from `SsdRegion` because the on-disk layout is
    materially different (no `<dir>/blocks/<path>.kv` files).

    Construction:
        region = TablespaceLongRegion(
            path="/var/lib/kvd-long",
            max_bytes=64 * 1024**3,        # 64 GB
            slot_bytes=64 * 1024,          # 64 KB per slot
            container_bytes=1024**3,       # 1 GB per container file
        )
        region.start()

    Note: `slot_bytes` is a hard cap on stored value size. Values
    larger than `slot_bytes` are rejected (the caller — typically
    `HostStore` — will see `accepted=False` and treat the SET as a
    no-op for this region). Multi-pool support (smaller + larger
    slot pools) is Phase B.2.
    """

    def __init__(
        self,
        path: str | Path,
        max_bytes: int,
        *,
        slot_bytes: int = _DEFAULT_SLOT_BYTES,
        container_bytes: int = _DEFAULT_CONTAINER_BYTES,
        sync_writes: bool = True,
        o_direct: bool | None = True,
        flush_interval_ms: int | None = 0,
    ) -> None:
        """Default is **O_DIRECT on** — page cache is double-bookkeeping
        for a daemon that owns its own cache policy via HostStore.
        Bench data on AMD MI355X + Samsung MZ1L2960 NVMe (ext4, 256 MB
        workload):

          - Page cache footprint:  262 MB → 460 kB  (−99.8%)
          - Write throughput:      332  → 422 MB/s (+27%)
          - Write p50 latency:     179  → 143 µs   (−20%)
          - Cold read p50:         99   → 106 µs   (≈ same; NVMe bound)
          - Warm read p50:         6.7  → 105 µs   (cache hit lost)

        The warm-cache speedup we give up rarely fires for kvd: most
        reads are engine-restart cold reloads, gap of minutes-to-hours,
        kernel has evicted the page cache by then anyway.

        Pass ``o_direct=False`` to opt back into buffered IO. Reasons
        to want that:
          - Filesystem doesn't support O_DIRECT (tmpfs, some NFS
            configs). Startup fail-fast tells you which flag to flip.
          - Dev/test deployments with tiny long regions (< 1 GB) where
            RAM cost is negligible and warm-cache benefit real.

        ``slot_bytes`` MUST be a multiple of 4096 when O_DIRECT is on
        (offset/length alignment). The default 64 KB is already a 16×
        multiple, so the constraint usually doesn't bite.

        ``flush_interval_ms``: group-commit / batch-fsync window. When
        ``> 0``, each ``put`` calls ``pwrite`` and journal-append but
        does NOT fsync; a background thread fsyncs every
        ``flush_interval_ms`` milliseconds. Trades a bounded durability
        window (last <flush_interval_ms> of writes lost on power loss)
        for throughput.

        **Auto-detect**: passing ``o_direct=None`` or
        ``flush_interval_ms=None`` triggers ``detect_fs_defaults(path)``
        at ``start()`` time, which probes the filesystem via
        ``findmnt`` and picks per-backend defaults (see ``_FS_DEFAULTS``).
        Mixed mode
        is supported: passing one explicit value and one ``None``
        auto-detects only the ``None`` field. The selected value is
        logged at INFO so operators can verify what was chosen.

        Sizing the window — backend matters:

          - Local NVMe (fsync ~50 µs): leave at 0. The fsync is
            cheap; batching adds latency for no real gain.
          - Vast NFS / NAS (fsync ~1.2 ms): set to 10-50 ms. With
            10 ms window and 10 PUTs/window, fsync amortizes from
            1.2 ms each to 0.12 ms each — 10× write speedup at the
            cost of losing ≤10 ms of writes on crash.
          - 3FS over RDMA: similar logic; tune to measured fsync
            latency × expected concurrent PUT rate.

        For L3 (cross-node best-effort cache) the durability window
        is fine — uncommitted writes lost on crash are simply
        recomputed by the engine, no correctness issue. For L2
        (local persistence), most operators leave this at 0.
        """
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        if slot_bytes <= 0:
            raise ValueError(f"slot_bytes must be positive, got {slot_bytes}")
        if container_bytes < slot_bytes:
            raise ValueError(
                f"container_bytes ({container_bytes}) must be >= slot_bytes ({slot_bytes})"
            )
        # Alignment check only fires if the caller passed an explicit
        # True — when o_direct is None we defer the check to start()
        # after auto-detect runs.
        if o_direct is True and slot_bytes % _DIRECT_IO_ALIGN_BYTES != 0:
            raise ValueError(
                f"o_direct=True requires slot_bytes ({slot_bytes}) to be a multiple of "
                f"{_DIRECT_IO_ALIGN_BYTES} for alignment"
            )

        self._path = Path(path)
        self._max_bytes = max_bytes
        self._slot_bytes = slot_bytes
        self._container_bytes = container_bytes
        self._slots_per_container = container_bytes // slot_bytes
        # Container count rounded down — we never overshoot max_bytes.
        self._num_containers = max(1, max_bytes // container_bytes)
        self._total_slots = self._num_containers * self._slots_per_container
        # `_o_direct` may be None here (auto-detect sentinel); resolved
        # in start() before any IO. Same for `_flush_interval_ms`.
        self._o_direct = o_direct

        self._allocator = BitsetAllocator(
            self._total_slots,
            num_containers=self._num_containers,
            slots_per_container=self._slots_per_container,
        )
        self._entries: dict[tuple[str, str, bytes], TablespaceEntry] = {}
        self._used_bytes = 0
        self._lock = threading.Lock()

        self._containers_dir = self._path / "containers"
        self._journal_path = self._path / _JOURNAL_FILENAME
        self._snapshot_path = self._path / _SNAPSHOT_FILENAME
        self._snapshot_bak_path = self._path / _SNAPSHOT_BAK_FILENAME
        self._snapshot_tmp_path = self._path / _SNAPSHOT_TMP_FILENAME
        self._journal = TablespaceJournal(self._journal_path, sync_writes=sync_writes)

        self._container_fds: list[int] = []
        self._started = False

        # Group-commit state. When `flush_interval_ms > 0`, `put`
        # marks `_dirty_containers` instead of fsyncing inline; a
        # background thread periodically fsyncs and clears the set.
        # None = auto-detect at start(); int = explicit (clamped).
        self._flush_interval_ms: int | None = (
            None if flush_interval_ms is None else max(0, flush_interval_ms)
        )
        self._dirty_containers: set[int] = set()
        self._flush_lock = threading.Lock()
        # Last fsync error seen by the background flusher, if any.
        # Sticky: cleared only when a subsequent fsync succeeds. PUT
        # checks this and refuses new writes when the flusher is
        # failing — without this, a full disk / unplugged device
        # silently logs WARN forever while PUTs return success and
        # data goes nowhere (PR #9 review fix P1).
        self._flush_error: OSError | None = None
        self._flush_stop = threading.Event()
        self._flush_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "tablespace_long"

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def slot_bytes(self) -> int:
        return self._slot_bytes

    @property
    def container_bytes(self) -> int:
        return self._container_bytes

    @property
    def num_containers(self) -> int:
        return self._num_containers

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    @property
    def entries_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open + preallocate container files; restore index from
        snapshot + journal."""
        if self._started:
            return
        self._path.mkdir(parents=True, exist_ok=True)
        self._containers_dir.mkdir(parents=True, exist_ok=True)

        # Resolve auto-detect sentinels BEFORE any IO. detect_fs_defaults
        # probes the path's filesystem via findmnt; the path needs to
        # exist (we just created it above). Mixed mode is supported —
        # auto-detect only fields that were left as None.
        if self._o_direct is None or self._flush_interval_ms is None:
            detected = detect_fs_defaults(self._path)
            fstype = detected["fstype"]
            if self._o_direct is None:
                self._o_direct = bool(detected["o_direct"])
                logger.info(
                    "tablespace: auto-detected fstype=%s at %s → o_direct=%s "
                    "(pass explicit o_direct= to override)",
                    fstype,
                    self._path,
                    self._o_direct,
                )
            if self._flush_interval_ms is None:
                self._flush_interval_ms = int(detected["flush_interval_ms"])
                logger.info(
                    "tablespace: auto-detected fstype=%s at %s → flush_interval_ms=%d "
                    "(pass explicit flush_interval_ms= to override)",
                    fstype,
                    self._path,
                    self._flush_interval_ms,
                )

        # Re-validate alignment now that o_direct has a concrete value.
        # This catches the case where auto-detect chose o_direct=True
        # but slot_bytes isn't a 4 KB multiple.
        if self._o_direct and self._slot_bytes % _DIRECT_IO_ALIGN_BYTES != 0:
            raise ValueError(
                f"o_direct=True (detected for fstype) requires slot_bytes "
                f"({self._slot_bytes}) to be a multiple of "
                f"{_DIRECT_IO_ALIGN_BYTES}; pass o_direct=False to override"
            )

        # Open / preallocate containers.
        # When o_direct is on, we use a TWO-PHASE open: first open
        # buffered (so we can fallocate cleanly — fallocate works on
        # O_DIRECT fds too but reports inconsistent results on some
        # ext4 versions), then close and reopen with O_DIRECT for the
        # IO path.
        self._container_fds.clear()
        open_flags = os.O_RDWR | os.O_CREAT
        for i in range(self._num_containers):
            container_path = self._containers_dir / f"{i:04d}.bin"
            fd = os.open(container_path, open_flags, 0o600)
            current_size = os.fstat(fd).st_size
            if current_size < self._container_bytes:
                # Preallocate. fallocate is the fast path on ext4/xfs; if
                # the filesystem doesn't support it (Lustre, NFS) fall
                # back to ftruncate — slower first-write since blocks
                # are sparse, but correct.
                try:
                    os.posix_fallocate(fd, 0, self._container_bytes)
                except OSError as exc:
                    logger.warning(
                        "tablespace: posix_fallocate failed on %s (%s); using ftruncate",
                        container_path,
                        exc,
                    )
                    os.ftruncate(fd, self._container_bytes)
            # If O_DIRECT was requested, reopen the container with the
            # flag now that preallocation is done. Some kernels (and
            # NFS clients) reject O_DIRECT at open time on freshly-
            # created sparse files — preallocating first sidesteps that.
            if self._o_direct:
                os.close(fd)
                try:
                    fd = os.open(container_path, open_flags | os.O_DIRECT, 0o600)
                except OSError as exc:
                    # Filesystem doesn't support O_DIRECT (e.g. tmpfs,
                    # certain NFS configs, some overlay FSes). We don't
                    # silently downgrade — the daemon's default is
                    # O_DIRECT for a reason (avoid page-cache double
                    # bookkeeping); silently flipping to buffered would
                    # quietly re-introduce 2× RAM use.
                    #
                    # Fail with a message that names the exact escape
                    # hatch the operator should reach for:
                    #   * Daemon CLI flag: `--tablespace-buffered-io`
                    #   * Library API:     `o_direct=False`
                    # plus a mount-check hint so they can verify
                    # filesystem capability before deciding.
                    raise OSError(
                        f"tablespace: O_DIRECT open of {container_path} failed ({exc}).\n"
                        f"\n"
                        f"The underlying filesystem at {self._containers_dir} does not "
                        f"support O_DIRECT. Common causes:\n"
                        f"  - tmpfs (e.g. some CI runners mount /tmp as tmpfs)\n"
                        f"  - certain NFS configurations (NFSv3 without nolock, NFSv4 with "
                        f"odd flags)\n"
                        f"  - overlay/squashfs mounts in some container runtimes\n"
                        f"\n"
                        f"Two ways forward:\n"
                        f"  1. Re-run the daemon with `--tablespace-buffered-io` (or pass "
                        f"`o_direct=False` to the constructor). Trades −99% page-cache RAM "
                        f"and +27%% write throughput for compatibility with this FS.\n"
                        f"  2. Move --long-path onto ext4/xfs/btrfs (which DO support "
                        f"O_DIRECT) and re-launch with the default."
                    ) from exc
            self._container_fds.append(fd)

        # Aligned-buffer allocation lives per-IO in
        # `_aligned_io_read` / `_aligned_io_write`. A per-region
        # shared buffer would force serialization across threads;
        # mmap.mmap(-1, slot_bytes) returns page-aligned anonymous
        # memory and costs ~1 µs (one mmap + one munmap syscall) —
        # cheap relative to a 100 µs disk read, and thread-safe.

        # Recover index.
        self._restore_index_from_snapshot_and_journal()

        # Open journal for append (idempotent if already open).
        self._journal.open()

        # Start the background flusher if group-commit is enabled.
        if self._flush_interval_ms > 0:
            self._flush_stop.clear()
            self._flush_thread = threading.Thread(
                target=self._flush_loop,
                name=f"tablespace-flusher-{self._path.name}",
                daemon=True,
            )
            self._flush_thread.start()

        self._started = True
        logger.info(
            "tablespace_long region started at %s "
            "(max=%d bytes, slot=%d, container=%d × %d containers, "
            "flush_interval_ms=%d; recovered %d entries / %d bytes)",
            self._path,
            self._max_bytes,
            self._slot_bytes,
            self._container_bytes,
            self._num_containers,
            self._flush_interval_ms,
            len(self._entries),
            self._used_bytes,
        )

    def shutdown(self) -> None:
        """Persist a snapshot, truncate journal, close fds.

        Stops the flusher and forces a final fsync of all dirty
        containers before tearing down — graceful shutdown must
        commit pending writes (uncommitted writes would be lost,
        but the manifest in snapshot wouldn't reflect them either,
        so the consumer wouldn't be able to find them anyway —
        we'd just be leaving slot bytes orphaned)."""
        if not self._started:
            return
        # Stop the background flusher and force a final flush.
        if self._flush_thread is not None:
            self._flush_stop.set()
            self._flush_thread.join(timeout=5.0)
            self._flush_thread = None
        self._flush_dirty_containers()  # idempotent if already clean
        # Write a fresh snapshot reflecting the current index.
        try:
            self._snapshot_now()
        except OSError as exc:
            logger.error("tablespace shutdown: snapshot write failed (%s)", exc)
        self._journal.close()
        for fd in self._container_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._container_fds.clear()
        self._started = False

    def flush(self) -> None:
        """Force an immediate fsync of all dirty containers + journal.

        Public method for tests and explicit checkpoints (e.g. before
        a planned restart). The background flusher runs this on its
        own schedule; callers shouldn't need it in normal operation."""
        if not self._started:
            return
        self._flush_dirty_containers()

    def _flush_dirty_containers(self) -> None:
        """Fsync all containers in `_dirty_containers` and clear the
        set. Called from the background thread + from shutdown().

        On persistent failure (full disk, device disconnected) we set
        `_flush_error` — PUT path reads it and refuses new writes
        until the flusher recovers. Logs at ERROR so this surfaces in
        operator dashboards instead of disappearing into a noisy
        WARN stream (PR #9 review fix P1)."""
        with self._flush_lock:
            to_flush = list(self._dirty_containers)
            self._dirty_containers.clear()
        any_failure: OSError | None = None
        for container_idx in to_flush:
            if container_idx >= len(self._container_fds):
                continue
            try:
                # Container files are posix_fallocate'd to full size at
                # startup; steady-state writes never grow the file or
                # change inode metadata. fdatasync skips the metadata
                # flush — saves one inode-update RPC on NFS (~1 ms
                # roundtrip on Vast).
                os.fdatasync(self._container_fds[container_idx])
            except OSError as exc:
                any_failure = exc
                logger.error(
                    "tablespace: deferred fsync failed on container %d (%s); "
                    "data may be lost on crash. Marking flusher as failing — "
                    "future PUTs will be rejected until the underlying issue "
                    "is resolved (full disk / device disconnected / fd revoked).",
                    container_idx,
                    exc,
                )
        # Sticky: only clear on a clean sweep. A flusher tick that
        # processed nothing (empty `to_flush`) shouldn't clear the
        # error flag — there might still be a pending PUT that hasn't
        # observed the flusher recovery yet.
        if to_flush and any_failure is None:
            self._flush_error = None
        elif any_failure is not None:
            self._flush_error = any_failure

    def _flush_loop(self) -> None:
        """Background thread: every flush_interval_ms, fsync dirty
        containers. Honors `_flush_stop` for graceful shutdown."""
        interval_s = self._flush_interval_ms / 1000.0
        while not self._flush_stop.wait(interval_s):
            self._flush_dirty_containers()

    # ------------------------------------------------------------------
    # Index restore (from snapshot + journal)
    # ------------------------------------------------------------------

    def _restore_index_from_snapshot_and_journal(self) -> None:
        """Snapshot-first replay. Journal entries are layered ON TOP of
        the snapshot's index — they reflect ops that happened AFTER the
        last snapshot. PUT overwrites, DEL removes."""
        snapshot_entries = _read_snapshot(
            self._snapshot_path,
            self._snapshot_bak_path,
            expected_slot_bytes=self._slot_bytes,
            expected_container_bytes=self._container_bytes,
        )
        # Materialize an index keyed by composite (model, compat_key, key).
        index: dict[tuple[str, str, bytes], dict[str, Any]] = {}
        for raw in snapshot_entries:
            composite = (
                raw.get("model", ""),
                raw.get("compat_key", ""),
                bytes.fromhex(raw.get("key_hex", "")),
            )
            index[composite] = raw

        # Replay journal.
        for op in self._journal.read_all():
            kind = op.get("op")
            if kind == "PUT":
                composite = (
                    op.get("model", ""),
                    op.get("compat_key", ""),
                    bytes.fromhex(op.get("key_hex", "")),
                )
                index[composite] = op
            elif kind == "DEL":
                composite = (
                    op.get("model", ""),
                    op.get("compat_key", ""),
                    bytes.fromhex(op.get("key_hex", "")),
                )
                index.pop(composite, None)

        # Populate the in-memory state.
        with self._lock:
            self._entries.clear()
            self._used_bytes = 0
            self._allocator = BitsetAllocator(
                self._total_slots,
                num_containers=self._num_containers,
                slots_per_container=self._slots_per_container,
            )
            for composite, raw in index.items():
                container_idx = int(raw.get("container", 0))
                slot_idx = int(raw.get("slot", 0))
                size = int(raw.get("size", 0))
                if (
                    container_idx < 0
                    or container_idx >= self._num_containers
                    or slot_idx < 0
                    or slot_idx >= self._slots_per_container
                    # PUT rejects size==0 (empty_value) and size>slot_bytes;
                    # restore enforces the same to keep on-disk and runtime
                    # invariants in lock-step.
                    or size <= 0
                    or size > self._slot_bytes
                ):
                    logger.warning(
                        "tablespace: dropping malformed entry at restore "
                        "(container=%d, slot=%d, size=%d)",
                        container_idx,
                        slot_idx,
                        size,
                    )
                    continue
                slot_global = container_idx * self._slots_per_container + slot_idx
                if self._allocator.is_set(slot_global):
                    # Duplicate — last PUT wins (and we're iterating dict
                    # in insertion order so this shouldn't fire unless
                    # the snapshot itself has dupes). Free old, take new.
                    logger.debug(
                        "tablespace: duplicate slot during restore (%d, %d)",
                        container_idx,
                        slot_idx,
                    )
                self._allocator.mark_used(slot_global)
                entry = TablespaceEntry(
                    key=composite[2],
                    container_idx=container_idx,
                    slot_idx=slot_idx,
                    size_bytes=size,
                    retention=raw.get("retention", RETENTION_LONG),
                    model=raw.get("model", ""),
                    compat_key=raw.get("compat_key", ""),
                    last_access=time.monotonic(),
                    metadata=raw.get("metadata", {}) or {},
                )
                self._entries[composite] = entry
                self._used_bytes += size
            # Restore is done. Recompute per-container cursors from the
            # bitmap so the first post-restart alloc starts scanning at
            # the right free slot in each container. mark_used kept the
            # per-container counts honest; cursors need this finalize
            # step because they're not reconstructible from individual
            # marks (we don't know the order they were called in).
            self._allocator.recompute_state()

    def _snapshot_now(self) -> None:
        """Write current index to snapshot.json, rotate previous, then
        truncate journal. Caller must NOT hold ``self._lock``; we acquire
        briefly to grab a consistent snapshot of the entries."""
        with self._lock:
            snapshot_entries = [
                {
                    "key_hex": e.key.hex(),
                    "container": e.container_idx,
                    "slot": e.slot_idx,
                    "size": e.size_bytes,
                    "retention": e.retention,
                    "model": e.model,
                    "compat_key": e.compat_key,
                    "metadata": e.metadata,
                }
                for e in self._entries.values()
            ]
        _write_snapshot(
            self._snapshot_path,
            self._snapshot_bak_path,
            self._snapshot_tmp_path,
            snapshot_entries,
            slot_bytes=self._slot_bytes,
            container_bytes=self._container_bytes,
        )
        # Journal is now redundant; truncate.
        self._journal.truncate()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_bytes(self, key: bytes, *, model: str = "", compat_key: str = "") -> bytes | None:
        composite = (model, compat_key, key)
        with self._lock:
            entry = self._entries.get(composite)
            if entry is None:
                return None
            entry.last_access = time.monotonic()
            container_idx = entry.container_idx
            slot_idx = entry.slot_idx
            size = entry.size_bytes

        # Defend against a corrupted index entry sliding size out of
        # range — without this, `_aligned_read(buf[:size])` would
        # silently return zeros or stale slot padding. PUT rejects
        # `size == 0` and `size > slot_bytes`, so any entry violating
        # those bounds is malformed; treat as miss + log.
        if size <= 0 or size > self._slot_bytes:
            logger.warning(
                "tablespace: index entry has out-of-range size_bytes=%d "
                "(slot_bytes=%d) for container=%d slot=%d; treating as miss",
                size,
                self._slot_bytes,
                container_idx,
                slot_idx,
            )
            return None

        if container_idx >= len(self._container_fds):
            logger.warning("tablespace: index points at non-existent container %d", container_idx)
            return None
        fd = self._container_fds[container_idx]
        offset = slot_idx * self._slot_bytes
        try:
            if self._o_direct:
                return self._aligned_read(fd, size, offset)
            return os.pread(fd, size, offset)
        except OSError as exc:
            logger.warning(
                "tablespace: pread failed for (container=%d, slot=%d): %s",
                container_idx,
                slot_idx,
                exc,
            )
            return None

    def exists(self, keys: list[bytes], *, model: str = "", compat_key: str = "") -> list[bool]:
        with self._lock:
            return [(model, compat_key, k) in self._entries for k in keys]

    def lookup_read_targets(
        self,
        keys: list[bytes],
        *,
        model: str = "",
        compat_key: str = "",
    ) -> list[tuple[int, int, int] | None]:
        """Batched (fd, byte_offset, size_bytes) lookup for the native
        io_uring worker.

        Returns one entry per input key:
          - ``(fd, offset, size)`` for a hit — caller submits a read
            against this triple,
          - ``None`` for a miss / corrupted entry / pointer to a
            non-existent container.

        Resolves everything under a SINGLE lock acquisition. The Python
        side of the native hot path needs this — a per-key
        ``get_entry`` would burn the dict lock 32-256 times per batch
        (the dispatch overhead PR #247 was pushing against). The fd is
        stable for the lifetime of the started region, so caching the
        returned fd in the caller is safe within one batch.
        """
        if not keys:
            return []
        out: list[tuple[int, int, int] | None] = []
        with self._lock:
            now = time.monotonic()
            num_fds = len(self._container_fds)
            for k in keys:
                entry = self._entries.get((model, compat_key, k))
                if entry is None:
                    out.append(None)
                    continue
                entry.last_access = now
                container_idx = entry.container_idx
                size = entry.size_bytes
                slot_idx = entry.slot_idx
                if size <= 0 or size > self._slot_bytes or container_idx >= num_fds:
                    out.append(None)
                    continue
                fd = self._container_fds[container_idx]
                out.append((fd, slot_idx * self._slot_bytes, size))
        return out

    def get_entry(
        self, key: bytes, *, model: str = "", compat_key: str = ""
    ) -> TablespaceEntry | None:
        with self._lock:
            return self._entries.get((model, compat_key, key))

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def put(
        self,
        key: bytes,
        value: bytes,
        *,
        retention: str,
        model: str = "",
        compat_key: str = "",
        metadata: dict | None = None,
    ) -> tuple[bool, str | None]:
        """Insert ``key`` with inline ``value`` bytes. Source bytes are
        pwrite'd (or ``_aligned_write`` under O_DIRECT) from the caller's
        buffer."""
        validate_retention(retention)
        if retention != RETENTION_LONG:
            return False, "tablespace_long_only_accepts_long_retention"
        if not self._started:
            return False, "region_not_started"

        # Refuse writes while the background flusher is failing — see
        # `_flush_dirty_containers`. Without this, PUT would return
        # success and the journal would record an entry whose container
        # bytes can't be durably persisted. The error stays sticky
        # until the flusher next succeeds (PR #9 review fix P1).
        if self._flush_error is not None:
            return False, f"flusher_failing: {self._flush_error}"

        size = len(value)
        if size == 0:
            return False, "empty_value"
        if size > self._slot_bytes:
            return False, f"value_exceeds_slot_bytes ({size} > {self._slot_bytes})"

        composite = (model, compat_key, key)
        # Hash of the composite (model, compat_key, key) bytes drives
        # the allocator's container choice for new slots. Same blake2b
        # digest_size=8 used by ``StripedLongRegion._pick`` so a key's
        # container assignment is restart-stable AND consistent in
        # spirit with the shard-routing layer above us. The hash is
        # only consulted for NEW slots — same-key updates reuse the
        # existing slot regardless.
        key_hash = int.from_bytes(
            hashlib.blake2b(
                model.encode("utf-8") + b"\x00" + compat_key.encode("utf-8") + b"\x00" + key,
                digest_size=8,
            ).digest(),
            "big",
        )

        with self._lock:
            existing = self._entries.get(composite)
            if existing is not None:
                # Reuse same slot — in-place update.
                container_idx = existing.container_idx
                slot_idx = existing.slot_idx
                old_size = existing.size_bytes
                existing.size_bytes = size
                existing.last_access = time.monotonic()
                existing.metadata = metadata or {}
                self._used_bytes += size - old_size
                rollback_old_size = old_size  # for write failure
                slot_was_new = False
            else:
                # Evict until we have BOTH a free slot AND enough byte budget.
                # Bitset capacity is the dominant constraint with fixed slots,
                # but used_bytes drifts as slot-bytes overhead accumulates.
                while self._allocator.num_free == 0 or self._used_bytes + size > self._max_bytes:
                    victim = self._pick_lru_victim_locked()
                    if victim is None:
                        return False, "tablespace_full_no_victim"
                    self._evict_locked(victim)

                slot_global = self._allocator.alloc(key_hash)
                if slot_global is None:
                    # Shouldn't happen — we just confirmed num_free > 0.
                    return False, "tablespace_alloc_failed"
                container_idx = slot_global // self._slots_per_container
                slot_idx = slot_global % self._slots_per_container

                entry = TablespaceEntry(
                    key=key,
                    container_idx=container_idx,
                    slot_idx=slot_idx,
                    size_bytes=size,
                    retention=retention,
                    model=model,
                    compat_key=compat_key,
                    last_access=time.monotonic(),
                    metadata=metadata or {},
                )
                self._entries[composite] = entry
                self._used_bytes += size
                rollback_old_size = 0
                slot_was_new = True

        # Disk write outside the lock.
        fd = self._container_fds[container_idx]
        offset = slot_idx * self._slot_bytes
        try:
            if self._o_direct:
                self._aligned_write(fd, value, offset)
            else:
                written = os.pwrite(fd, value, offset)
                if written != size:
                    raise OSError(f"short write: {written} != {size}")
            # Durability: inline fdatasync (default) vs deferred (group commit).
            # In group-commit mode we mark the container as dirty; the
            # background flusher fdatasyncs it on the next tick. PUT returns
            # success based on pwrite success — durability window is
            # bounded by flush_interval_ms.
            # fdatasync over fsync: containers are preallocated at startup
            # so inode metadata (size) doesn't change on writes; skipping
            # the metadata flush saves one inode-update RPC on NFS.
            if self._flush_interval_ms > 0:
                with self._flush_lock:
                    self._dirty_containers.add(container_idx)
            else:
                os.fdatasync(fd)
        except OSError as exc:
            # Roll back the index change.
            with self._lock:
                rolled = self._entries.pop(composite, None)
                if rolled is not None:
                    if slot_was_new:
                        slot_global = (
                            rolled.container_idx * self._slots_per_container + rolled.slot_idx
                        )
                        self._allocator.free(slot_global)
                        self._used_bytes = max(0, self._used_bytes - rolled.size_bytes)
                    else:
                        # We mutated an existing entry's size; restore old size in-memory.
                        self._used_bytes = max(
                            0, self._used_bytes - (rolled.size_bytes - rollback_old_size)
                        )
            return False, f"disk_write_failed: {exc}"

        # Disk write succeeded → append PUT to journal.
        self._journal.append(
            {
                "op": "PUT",
                "key_hex": key.hex(),
                "container": container_idx,
                "slot": slot_idx,
                "size": size,
                "retention": retention,
                "model": model,
                "compat_key": compat_key,
                "metadata": metadata or {},
                "ts": time.time(),
            }
        )
        return True, None

    # ------------------------------------------------------------------
    # Remove / clear
    # ------------------------------------------------------------------

    def remove(self, key: bytes, *, model: str = "", compat_key: str = "") -> bool:
        composite = (model, compat_key, key)
        with self._lock:
            entry = self._entries.pop(composite, None)
            if entry is None:
                return False
            slot_global = entry.container_idx * self._slots_per_container + entry.slot_idx
            self._allocator.free(slot_global)
            self._used_bytes = max(0, self._used_bytes - entry.size_bytes)
        # Journal append outside lock.
        self._journal.append(
            {
                "op": "DEL",
                "key_hex": key.hex(),
                "model": model,
                "compat_key": compat_key,
            }
        )
        return True

    def clear(self) -> int:
        """Drop every block. Bytes on disk become dead slots (the
        allocator marks them free; next PUT will overwrite). We DON'T
        zero the slots — same content-addressed reasoning as
        LongStorageRegion: the bytes are inaccessible once the index
        loses their key, so leaving them is fine.

        Snapshot is rewritten as empty. Journal truncated."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._used_bytes = 0
            self._allocator = BitsetAllocator(
                self._total_slots,
                num_containers=self._num_containers,
                slots_per_container=self._slots_per_container,
            )
        # Persist the empty state so a restart sees it.
        try:
            self._snapshot_now()
        except OSError as exc:
            logger.warning("tablespace: snapshot after clear failed (%s)", exc)
        return count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pick_lru_victim_locked(self) -> tuple[str, str, bytes] | None:
        """Plain LRU. Caller holds self._lock."""
        if not self._entries:
            return None
        best_key: tuple[str, str, bytes] | None = None
        best_t = float("inf")
        for k, entry in self._entries.items():
            if entry.last_access < best_t:
                best_t = entry.last_access
                best_key = k
        return best_key

    def _evict_locked(self, composite: tuple[str, str, bytes]) -> None:
        """Drop ``composite`` from in-memory state. Caller holds self._lock.
        The journal DEL is appended later (outside the lock) by the
        caller — for the PUT-eviction path this means the new PUT's
        journal entry implicitly supersedes the evicted entry once
        snapshot rolls forward."""
        entry = self._entries.pop(composite)
        slot_global = entry.container_idx * self._slots_per_container + entry.slot_idx
        self._allocator.free(slot_global)
        self._used_bytes = max(0, self._used_bytes - entry.size_bytes)

    # ------------------------------------------------------------------
    # O_DIRECT aligned-IO helpers
    # ------------------------------------------------------------------

    def _aligned_read(self, fd: int, size: int, offset: int) -> bytes:
        """O_DIRECT read into an mmap-backed aligned buffer, then slice
        the actual `size` bytes out.

        Why mmap.mmap(-1, slot_bytes):
        - The kernel hands out anonymous mmap regions page-aligned —
          satisfies O_DIRECT's buffer-alignment requirement for free.
        - mmap memory is direct-mappable for DMA on Linux (the kernel
          handles pinning during the syscall).
        - Per-call alloc costs ~1 µs; trivial vs a 100 µs NVMe read.

        Why not a pre-allocated per-region buffer:
        - That would force serialization across concurrent calls.
          One mmap per call is thread-safe and the alloc is cheap.
        """
        import mmap

        # Read the full slot to keep IO sizes a multiple of slot_bytes
        # (and 4 KB). Returning only `size` bytes is a memoryview slice.
        buf = mmap.mmap(-1, self._slot_bytes)
        try:
            n = os.preadv(fd, [buf], offset)
            if n != self._slot_bytes:
                # Partial read with O_DIRECT is rare but possible on
                # short files. Treat as failure — caller falls back.
                raise OSError(f"short O_DIRECT read: got {n}, expected {self._slot_bytes}")
            # Copy out the actual value bytes. Can't return memoryview
            # because the mmap will be unmapped when we close it; we
            # want a stable bytes object to give the caller.
            return bytes(buf[:size])
        finally:
            buf.close()

    def _aligned_write(self, fd: int, value: bytes, offset: int) -> None:
        """O_DIRECT write of `value` (padded to slot_bytes for alignment)
        into the container at `offset`. The dead bytes between
        len(value) and slot_bytes are zeros — the index records the
        actual size separately, so the padding is invisible to readers."""
        import mmap

        if len(value) > self._slot_bytes:
            raise ValueError(f"value larger than slot: {len(value)} > {self._slot_bytes}")
        buf = mmap.mmap(-1, self._slot_bytes)
        try:
            buf[: len(value)] = value
            # Tail bytes are already zero from anonymous mmap.
            n = os.pwritev(fd, [buf], offset)
            if n != self._slot_bytes:
                raise OSError(f"short O_DIRECT write: wrote {n}, expected {self._slot_bytes}")
        finally:
            buf.close()
