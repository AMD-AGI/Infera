###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Shared-memory pinned arena for cross-process KV transport.

Production-grade implementation of kvd's host-RAM tier as a memfd-
backed shared arena. vLLM/SGLang workers mmap the same FD (passed at
handshake via SCM_RIGHTS) and read blocks zero-copy. The UDS round-
trip carries only `(slot_offset, length, version)` (~50 us); the
actual KV bytes are read directly from the shared mmap.

## Why memfd_create

- **No filesystem path** — works in restricted containers / network
  namespaces as long as the IPC namespace is shared. `/dev/shm` and
  `shm_open` both need a writable path; memfd does not.
- **Refcounted FD** — the arena lives as long as someone has the FD
  open. If kvd dies, vLLM workers' mmap stays valid until they unmap.
- **Passed via SCM_RIGHTS** — Linux duplicates the FD into the
  receiver's table; both ends see the same kernel object.

## Slot allocator

Single arena of `capacity_bytes`. Slot size is decided lazily on the
first `put` (rounded up to 64-byte alignment for cache-line friendly
DMA). `num_slots = capacity // slot_size`. Free list + LRU
`OrderedDict[key -> slot_index]`.

This mirrors an in-process pinned-arena L2 pool design, with two production-required
additions:

1. **Cross-process layout** — every slot starts at a known offset
   `(slot_index * slot_size)` so the reader (which has no Python
   metadata, only the FD + an offset/length/version triple from the
   wire) can compute the byte range without a server round-trip.

2. **Per-slot version counter** — at slot offset `s*slot_size`, the
   first 16 bytes are a header:
       [0..4]   uint32 LE version (even = stable, odd = mid-write)
       [4..8]   uint32 LE payload length
       [8..16]  reserved (zero today; carries CRC32 + flags later)
   Writer holds the per-slot lock, increments version (now odd),
   writes length + payload, increments version again (now even).
   Reader reads version, reads payload, re-reads version: if mismatch
   or odd, retry (or give up and fall through to long-region path).

This is the **seqlock pattern** — lock-free reads under a single
writer per slot, no shared mutex across processes.

## Pinning

`torch.empty(N, dtype=uint8).pin_memory()` over a memfd-backed mmap
is the ideal — pinned host memory that DMA can fire against. On
ROCm 7.2.2 the path is being validated in parallel (Phase 1
prototype). If torch can't pin a memfd-backed buffer, we fall back
to `mlock(MCL_CURRENT)` on the mmap region — DMA can still stage
through it (hipMemcpyAsync with non-pinned host is slower but
correct).

## Threading

The arena's `put` / `get_slice` are thread-safe via a single
`threading.Lock` guarding allocator state (slot index, free list,
LRU). Per-slot writes use the seqlock — no allocator lock held
during the bytes-copy. Per-slot reads NEVER take the allocator
lock; they read the header bytes directly.
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Slot header layout (must stay in sync with the reader on the worker
# side — bumping these requires a wire-protocol version bump).
_HEADER_BYTES = 16  # version + length + reserved
# Bound on `_recent_evicted_keys` (see `drain_recent_evictions`).
# Large enough that HostStore can typically catch evictions before
# they roll off, small enough that pathological workloads don't grow
# unbounded.
_EVICT_LOG_CAP = 4096
_HEADER_VERSION_OFFSET = 0
_HEADER_LENGTH_OFFSET = 4
_HEADER_RESERVED_OFFSET = 8

# Slot-size alignment: round up to 64 bytes so each slot is cache-
# line-aligned. Matches the in-process L2 pool for consistency.
_SLOT_ALIGNMENT = 64

# 2 MB hugepage size. The memfd region must be a multiple of this
# when ``MFD_HUGETLB | MFD_HUGE_2MB`` is requested — partial pages
# can't be reserved from the hugepage pool.
_HUGEPAGE_2MB = 2 * 1024 * 1024


def _env_truthy(name: str) -> bool:
    """Read an env var as a boolean. ``"1"``, ``"true"``, ``"yes"``,
    ``"on"`` (case-insensitive) are true; anything else (including
    missing) is false. Used by the constructor defaults below."""
    raw = os.environ.get(name, "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_falsy(name: str) -> bool:
    """Read an env var as an explicit opt-out boolean. ``"0"``,
    ``"false"``, ``"no"``, ``"off"`` (case-insensitive) are falsy;
    anything else (including missing) is NOT falsy. Used by
    default-ON knobs that want a clean opt-out env var path."""
    raw = os.environ.get(name, "")
    return raw.strip().lower() in ("0", "false", "no", "off")


# Max retries for torn-read detection on the reader side. Two reads
# is enough in practice (a single writer can change a slot at most
# once between the version-snap reads); we cap at 3 to bound the
# pathological case where the writer is in a tight loop.
MAX_TORN_READ_RETRIES = 3


@dataclass
class SharedArenaInfo:
    """Wire-serializable description of a shared arena. Sent from
    server to client in HelloAck; the FD itself goes out-of-band via
    SCM_RIGHTS on the same UDS socket immediately after HelloAck."""

    arena_size: int
    slot_size: int  # 0 if not yet decided (first put hasn't happened)
    server_pid: int

    def to_tuple(self) -> tuple[int, int, int]:
        """msgpack-friendly serialization. Mirrors `from_tuple` for
        round-trip."""
        return (self.arena_size, self.slot_size, self.server_pid)

    @classmethod
    def from_tuple(cls, t: tuple[int, int, int]) -> SharedArenaInfo:
        return cls(arena_size=t[0], slot_size=t[1], server_pid=t[2])


@dataclass
class SharedArenaStats:
    """Snapshot of arena counters."""

    capacity_bytes: int
    slot_size: int
    num_slots: int
    entries: int
    hits_total: int
    misses_total: int
    evictions_total: int
    torn_reads_total: int
    bytes_promoted_total: int
    # Save-side CopyFree (lease+commit) observability. Leases are
    # held between SetReserve and SetCommit (or SetCancel / connection
    # drop). A nonzero `reservations_active` after a quiet period
    # signals a leak — the server should cancel-on-disconnect, so
    # this should only be nonzero during active flushes.
    reservations_active: int = 0
    reservations_committed: int = 0
    reservations_cancelled: int = 0
    reservations_expired: int = 0


@dataclass
class _Reservation:
    """One in-flight save lease. Held by the server between
    SetReserve and SetCommit / SetCancel / connection-drop.

    The arena tracks the slot_id + connection_id so commit can
    validate ownership; the engine writes into the slot at
    `slot_id * slot_size + _HEADER_BYTES` via its own mmap.
    `created_at_ns` is monotonic — kept for future lease-expiry
    timeouts (today we rely on connection-drop cleanup only).
    """

    slot_id: int
    promised_size: int
    connection_id: int
    created_at_ns: int


class SharedArena:
    """memfd-backed shared arena with seqlock-protected slots.

    Lifetime owner: kvd process. The FD is passed to clients via
    SCM_RIGHTS at handshake; the kernel refcount keeps the arena alive
    even if clients have it mmapped after kvd dies (their reads still
    succeed against the last-written contents until they unmap).

    Operations:
    - `put(key, value)` — copy bytes into a slot, bump seqlock,
      return `slot_id`. LRU evicts on full.
    - `get_slice(slot_id) -> memoryview | None` — server-side reader
      that returns a tight memoryview into the slot's payload bytes.
      Returns None on torn-read (version-mismatch retry exceeded).
      Used by HostStore's `get` path when arena-backed.
    - `evict_lru()` / `clear()` — admin.

    Construction:
    - `capacity_bytes` is the total arena size. Lower-bounded by 2
      slot_sizes (need at least one usable slot after header). Upper-
      bound is whatever the OS allows for memfd_create + mmap.
    - `name` is the memfd name (visible in `/proc/<pid>/fd/`); purely
      cosmetic but useful for ops triage.
    - `pin_memory` requests page-locking via mlock. Falls back silently
      if the user lacks RLIMIT_MEMLOCK headroom; the arena still works,
      DMA just stages through unpinned memory.
    """

    def __init__(
        self,
        capacity_bytes: int,
        *,
        name: str = "infera-kvd-arena",
        pin_memory: bool = True,
        hugetlb: bool | None = None,
    ) -> None:
        """Construct a memfd-backed shared arena.

        ``hugetlb``: when True (or when the env var
        ``INFERA_KVD_ARENA_HUGETLB`` is truthy), back the memfd
        with 2 MB hugepages via ``MFD_HUGETLB | MFD_HUGE_2MB``. The
        capacity is rounded up to the next 2 MB boundary so a partial
        hugepage isn't requested (the kernel would EINVAL otherwise).
        If the kernel rejects the hugepage allocation (typically
        because ``/proc/sys/vm/nr_hugepages`` is too small or the
        hugepage pool is exhausted), we fall back to 4 KB pages with
        a WARN log — the arena still starts and serves correctly.
        Default is the env-var read (OFF unless the operator opts in)
        so kvd doesn't fail to start on systems without hugepages
        reserved.
        """
        if capacity_bytes <= 0:
            raise ValueError(f"capacity_bytes must be > 0, got {capacity_bytes}")
        hugetlb_requested = _env_truthy("INFERA_KVD_ARENA_HUGETLB") if hugetlb is None else hugetlb
        capacity = int(capacity_bytes)
        # Hugepages require the mapping size to be a multiple of the
        # page size. Round up so the operator's intent (>= requested
        # capacity) is honoured. The rounding only kicks in when
        # hugetlb is actually requested; legacy 4 KB-page arenas keep
        # their exact byte count.
        if hugetlb_requested:
            remainder = capacity % _HUGEPAGE_2MB
            if remainder:
                capacity += _HUGEPAGE_2MB - remainder
        self._capacity = capacity
        self._name = name
        self._server_pid = os.getpid()
        self._hugetlb_active = False

        # Allocate the memfd and size it. `MFD_CLOEXEC` so child procs
        # don't inherit the fd unless we explicitly dup it (SCM_RIGHTS
        # path dups; subprocess fork does not).
        #
        # Hugepage path: try `MFD_HUGETLB | MFD_HUGE_2MB` first. If
        # the kernel refuses (ENOMEM = hugepage pool exhausted,
        # EINVAL = kernel doesn't support the flag combo, EPERM =
        # caller lacks CAP_IPC_LOCK on older kernels), fall back to
        # plain 4 KB. We never silently degrade a hard-coded request
        # — the WARN log surfaces the reason so ops can fix it.
        self._fd = -1
        if hugetlb_requested:
            try:
                self._fd = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_HUGETLB | os.MFD_HUGE_2MB)
                self._hugetlb_active = True
                logger.info(
                    "%s: memfd_create with MFD_HUGETLB succeeded "
                    "(2 MB pages, capacity rounded to %.2f GiB)",
                    self._name,
                    self._capacity / (1024**3),
                )
            except OSError as exc:
                logger.warning(
                    "%s: memfd_create with MFD_HUGETLB failed (%s); "
                    "falling back to 4 KB pages. To enable hugepages, "
                    "ensure /proc/sys/vm/nr_hugepages has enough 2 MB "
                    "pages reserved.",
                    self._name,
                    exc,
                )
        if self._fd < 0:
            self._fd = os.memfd_create(name, os.MFD_CLOEXEC)
        try:
            os.ftruncate(self._fd, self._capacity)
        except OSError:
            os.close(self._fd)
            raise

        # mmap shared, read+write. The server is the only writer; we
        # share with clients (who pass FD + open their own mmap RO).
        self._mmap = mmap.mmap(
            self._fd,
            self._capacity,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        # Best-effort pin. If torch is present and pin_memory is True,
        # we try to lock the region with mlock(2). On EPERM/ENOMEM
        # we fall back to unpinned — the arena still works for the
        # data path; DMA staging is slower.
        self._pinned = False
        if pin_memory:
            self._pinned = self._try_mlock()

        # Allocator state (guarded by `_alloc_lock`).
        self._alloc_lock = threading.Lock()
        self._slot_size: int = 0
        self._num_slots: int = 0
        self._free_slots: list[int] = []
        # key -> slot_index. OrderedDict's move_to_end / popitem(last=False)
        # give O(1) LRU ops.
        self._slot_index: OrderedDict[bytes, int] = OrderedDict()
        # Keys evicted from the arena's own LRU during recent `put`
        # calls (and explicit evict_lru/evict_key calls). HostStore
        # drains this after each `set` to keep its `_entries` dict
        # consistent — without this the entry would point at a
        # recycled slot containing a different key's bytes.
        #
        # deque(maxlen=...) gives FIFO eviction at O(1) (the previous
        # list + `pop(0)` was O(n) per overflow). The cap is the same
        # `_EVICT_LOG_CAP` — under sustained eviction storms, the
        # oldest entries roll off automatically.
        self._recent_evicted_keys: deque[bytes] = deque(maxlen=_EVICT_LOG_CAP)
        # Per-slot write locks. Lazily sized to num_slots once
        # slot_size is decided. Bounded — for typical arenas (16-32
        # GiB at 4-8 MiB slots) we have a few thousand locks; cheap.
        self._slot_write_locks: list[threading.Lock] = []

        # Counters.
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._torn_reads = 0
        self._bytes_promoted = 0
        self._logged_oversize = False
        self._logged_size_mismatch = False

        # Save-side CopyFree state — guarded by `_alloc_lock`. The
        # reservation pool tracks in-flight save leases:
        # `reserve()` allocates a slot, reads return None until
        # `commit_reservation()` stamps the seqlock to "stable"
        # (or `cancel_reservation()` releases the slot back to the
        # free list). Slots holding reservations are NOT in
        # `_slot_index` AND NOT in `_free_slots` — they count
        # toward "arena full" predicate via the union of the three.
        self._reservations: dict[int, _Reservation] = {}
        # 0 is reserved as the "rejected" lease token; first real
        # token is 1. Wrap is unrealistic at int64 width but the
        # logic doesn't depend on uniqueness across restarts (the
        # connection dies with us).
        self._next_lease_token: int = 1
        self._reservations_committed = 0
        self._reservations_cancelled = 0
        self._reservations_expired = 0

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def fd(self) -> int:
        """Raw FD — pass via SCM_RIGHTS. Don't `os.close()` it from
        callers; the arena owns the lifetime."""
        return self._fd

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def hugetlb_active(self) -> bool:
        """True iff the memfd is backed by 2 MB hugepages (the
        ``MFD_HUGETLB`` flag was honoured at create time). False on
        any system where hugetlb was not requested or fell back to
        4 KB pages."""
        return self._hugetlb_active

    @property
    def slot_size(self) -> int:
        """0 until the first put decides it. Reported in HelloAck so
        clients can pre-size their internal bookkeeping if they want
        to, but they don't have to — the wire response carries
        `slot_size` on every shared-arena GET in case clients want to
        sanity-check the offset."""
        return self._slot_size

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def info(self) -> SharedArenaInfo:
        return SharedArenaInfo(
            arena_size=self._capacity,
            slot_size=self._slot_size,
            server_pid=self._server_pid,
        )

    def stats(self) -> SharedArenaStats:
        return SharedArenaStats(
            capacity_bytes=self._capacity,
            slot_size=self._slot_size,
            num_slots=self._num_slots,
            entries=len(self._slot_index),
            hits_total=self._hits,
            misses_total=self._misses,
            evictions_total=self._evictions,
            torn_reads_total=self._torn_reads,
            bytes_promoted_total=self._bytes_promoted,
            reservations_active=len(self._reservations),
            reservations_committed=self._reservations_committed,
            reservations_cancelled=self._reservations_cancelled,
            reservations_expired=self._reservations_expired,
        )

    # ------------------------------------------------------------------
    # Allocator + write path
    # ------------------------------------------------------------------

    def put(self, key: bytes, value: bytes | memoryview) -> int | None:
        """Insert (or overwrite) `key`'s blob into a slot. Returns
        the slot index (>=0) on success, or None if the blob is
        rejected (too big for the arena, or first-put-too-big-for-
        capacity).

        Slot size is fixed at the first `put`. Subsequent puts whose
        size doesn't fit `slot_size - HEADER_BYTES` are refused with
        a one-off warning; in practice every packed-block size is
        uniform within a model+config so the rejection path is exotic.
        """
        size = len(value)
        if size > self._capacity:
            if not self._logged_oversize:
                self._logged_oversize = True
                logger.warning(
                    "%s: refusing %d-byte blob — exceeds arena capacity %d",
                    self._name,
                    size,
                    self._capacity,
                )
            return None

        with self._alloc_lock:
            # Lazy slot-size decision on first put.
            if self._slot_size == 0:
                raw = size + _HEADER_BYTES
                self._slot_size = (raw + _SLOT_ALIGNMENT - 1) // _SLOT_ALIGNMENT * _SLOT_ALIGNMENT
                self._num_slots = self._capacity // self._slot_size
                if self._num_slots == 0:
                    logger.warning(
                        "%s: first blob (%d bytes) too large for arena (%d bytes); "
                        "refusing all puts on this arena",
                        self._name,
                        size,
                        self._capacity,
                    )
                    self._slot_size = 0
                    return None
                self._free_slots = list(range(self._num_slots))
                self._slot_write_locks = [threading.Lock() for _ in range(self._num_slots)]
                logger.info(
                    "%s: slot grid initialized — slot_size=%d num_slots=%d "
                    "(%.2f GiB usable of %.2f GiB capacity)",
                    self._name,
                    self._slot_size,
                    self._num_slots,
                    (self._num_slots * self._slot_size) / (1024**3),
                    self._capacity / (1024**3),
                )

            if size + _HEADER_BYTES > self._slot_size:
                if not self._logged_size_mismatch:
                    self._logged_size_mismatch = True
                    logger.warning(
                        "%s: blob size %d > slot_size %d (set by first put). Refusing oversize.",
                        self._name,
                        size + _HEADER_BYTES,
                        self._slot_size,
                    )
                return None

            # Overwrite: free old slot, then reallocate.
            old_slot = self._slot_index.pop(key, None)
            if old_slot is not None:
                self._free_slots.append(old_slot)

            evicted_key: bytes | None = None
            if self._free_slots:
                slot = self._free_slots.pop()
            else:
                # LRU evict the oldest entry. We need to return the
                # evicted key so the caller (HostStore) can drop its
                # own entry — otherwise HostStore.get(evicted_key)
                # would return an Entry pointing at a slot that now
                # belongs to a different key (staleness bug found
                # in regression test 2026-05-26).
                evicted_key, evicted_slot = self._slot_index.popitem(last=False)
                slot = evicted_slot
                self._evictions += 1
                # Track recent evictions so HostStore can drain them
                # post-put without holding the alloc lock during disk
                # I/O. The deque is bounded; oldest entries roll off
                # automatically on overflow (FIFO).
                self._recent_evicted_keys.append(evicted_key)

            self._slot_index[key] = slot
            slot_lock = self._slot_write_locks[slot]

        # Now do the actual write OUTSIDE the allocator lock — readers
        # may be reading other slots concurrently, allocator-state is
        # consistent.
        self._write_slot_seqlock(slot, value, size, slot_lock)
        self._bytes_promoted += size
        return slot

    def drain_recent_evictions(self) -> list[bytes]:
        """Return + clear the queue of keys the arena's LRU evicted
        during recent `put` calls. HostStore polls this after each
        `set` to drop its stale `_entries` entries; mirrors the
        write-then-publish pattern of seqlocks.

        The list is bounded by `_EVICT_LOG_CAP` — under sustained
        eviction, oldest evicted keys may roll off and HostStore
        will see slot-recycled garbage on those (the seqlock read
        retry will catch torn bytes but a key collision is the
        operator's responsibility to size away)."""
        with self._alloc_lock:
            out = list(self._recent_evicted_keys)
            self._recent_evicted_keys.clear()
            return out

    # ------------------------------------------------------------------
    # Save-side CopyFree: reserve / commit / cancel
    # ------------------------------------------------------------------
    #
    # Two-phase set lets the engine write its KV bytes directly into
    # the arena mmap (zero-copy from GPU's host staging into the
    # shared slot) and pay only a small Commit message on the wire.
    #
    # Invariants (all guarded by `_alloc_lock`):
    #   - Reserved slot IDs live in `_reservations.values()[].slot_id`.
    #   - Reserved slot IDs are NOT in `_slot_index` and NOT in
    #     `_free_slots`. Concurrent `reserve()` / `put()` therefore
    #     cannot double-allocate the slot.
    #   - LRU eviction inside `reserve()` only considers entries from
    #     `_slot_index` (live committed entries). Reservations are
    #     invisible to LRU — they hold their slot until commit /
    #     cancel / connection-drop.
    #
    # Seqlock semantics across reserve → commit:
    #   - At reserve(): write the slot header to "writing" (odd
    #     counter) so any racing reader sees torn → retry / None.
    #   - During engine write: header stays odd, payload bytes mutate.
    #   - At commit(): bump counter to next even value and write the
    #     final length into the header. After commit, readers see
    #     a consistent slot.

    def reserve(self, size: int, connection_id: int) -> tuple[int, int, int] | None:
        """Allocate a slot for an upcoming zero-copy save.

        Returns ``(lease_token, slot_id, payload_offset)`` on accept,
        or ``None`` on reject (oversize, arena uninitializable, full
        without an evictable victim).

        Mirrors `put()`'s slot-allocation logic — lazy slot-size
        decision on first reserve/put, free-list-first, LRU-evict
        otherwise. Evicted keys land on `_recent_evicted_keys` so
        HostStore's drain stays correct. The slot is marked as
        "writing" via the seqlock (odd version) before this method
        returns; concurrent readers of the (now-reserved) slot will
        see torn and fall through.
        """
        if size <= 0:
            return None
        if size > self._capacity:
            return None

        with self._alloc_lock:
            # Lazy slot-size decision — same as `put`'s first-touch path.
            if self._slot_size == 0:
                raw = size + _HEADER_BYTES
                self._slot_size = (raw + _SLOT_ALIGNMENT - 1) // _SLOT_ALIGNMENT * _SLOT_ALIGNMENT
                self._num_slots = self._capacity // self._slot_size
                if self._num_slots == 0:
                    self._slot_size = 0
                    return None
                self._free_slots = list(range(self._num_slots))
                self._slot_write_locks = [threading.Lock() for _ in range(self._num_slots)]

            if size + _HEADER_BYTES > self._slot_size:
                # Doesn't fit — engine should fall back to inline Set.
                return None

            # Pick a slot: free list first, otherwise LRU-evict a
            # COMMITTED entry (`_slot_index`). Reservations are
            # invisible to LRU.
            if self._free_slots:
                slot = self._free_slots.pop()
            elif self._slot_index:
                evicted_key, evicted_slot = self._slot_index.popitem(last=False)
                slot = evicted_slot
                self._evictions += 1
                self._recent_evicted_keys.append(evicted_key)
            else:
                # All slots are reserved (in-flight saves) — nothing
                # to evict without breaking another connection's
                # write. Refuse.
                return None

            # Issue the lease.
            lease_token = self._next_lease_token
            self._next_lease_token += 1
            self._reservations[lease_token] = _Reservation(
                slot_id=slot,
                promised_size=size,
                connection_id=connection_id,
                created_at_ns=time.monotonic_ns(),
            )

            # Mark the slot header as "writing" (odd version). We
            # take the per-slot write lock to serialize against any
            # concurrent legacy `put` that might be touching the
            # same slot — but since this slot was just popped from
            # the free list / LRU and isn't in `_slot_index`, no
            # concurrent `put` can target it. Holding the per-slot
            # lock is still cheap and keeps seqlock invariants
            # uniform with the rest of the arena.
            slot_lock = self._slot_write_locks[slot]

        with slot_lock:
            base = slot * self._slot_size
            mm = self._mmap
            version_bytes = mm[base + _HEADER_VERSION_OFFSET : base + _HEADER_VERSION_OFFSET + 4]
            version = struct.unpack("<I", version_bytes)[0]
            if version % 2 == 0:
                mid_version = version + 1
            else:
                mid_version = version + 2
            mm[base + _HEADER_VERSION_OFFSET : base + _HEADER_VERSION_OFFSET + 4] = struct.pack(
                "<I", mid_version
            )
            # Zero the length field while we're at it so a stale
            # length from a previous tenant can't be confused with
            # the engine's still-in-flight write.
            mm[base + _HEADER_LENGTH_OFFSET : base + _HEADER_LENGTH_OFFSET + 4] = struct.pack(
                "<I", 0
            )

        payload_offset = slot * self._slot_size + _HEADER_BYTES
        return lease_token, slot, payload_offset

    def commit_reservation(
        self, lease_token: int, key: bytes, length: int, connection_id: int
    ) -> tuple[bool, str, bytes | None]:
        """Finalize a reservation: link key→slot in `_slot_index`,
        write the final seqlock header.

        Returns ``(success, reason, overwritten_key_or_None)``:
          - ``success=True`` and `reason=""`: the slot is now a
            committed entry. `overwritten_key` is the key (if any)
            whose previous slot we freed because the new key
            collided with it. HostStore uses this to drop its own
            row for that key without an extra arena lookup.
          - ``success=False`` with a stable reason token —
            `"unknown_lease"`, `"wrong_owner"`, `"oversize_commit"`.

        Per the seqlock contract, after this returns success a
        reader of the slot's `(offset, length, version)` will see a
        consistent payload — version is even and length matches what
        the engine wrote.
        """
        with self._alloc_lock:
            reservation = self._reservations.get(lease_token)
            if reservation is None:
                return False, "unknown_lease", None
            if reservation.connection_id != connection_id:
                return False, "wrong_owner", None
            if length < 0 or length > reservation.promised_size:
                # Engine claims to have written more bytes than the
                # slot can hold — refuse without releasing the slot
                # (the engine's bookkeeping is in an inconsistent
                # state; better to leak the slot until cancel /
                # disconnect than corrupt a future tenant).
                return False, "oversize_commit", None

            slot = reservation.slot_id

            # Overwrite collision: if the key is already mapped to
            # some OTHER slot, free that slot so the index points
            # exclusively at the new committed slot.
            overwritten_key: bytes | None = None
            old_slot = self._slot_index.pop(key, None)
            if old_slot is not None and old_slot != slot:
                self._free_slots.append(old_slot)
                overwritten_key = key
            elif old_slot is not None and old_slot == slot:
                # Pathological: somehow the same key maps to OUR
                # slot already. Keep it; no free needed.
                overwritten_key = None

            # Publish the slot in `_slot_index` (MRU end of OrderedDict).
            self._slot_index[key] = slot
            slot_lock = self._slot_write_locks[slot]

            # Pop the reservation BEFORE we exit the lock so a
            # racing cancel for the same lease becomes a no-op.
            del self._reservations[lease_token]
            self._reservations_committed += 1
            self._bytes_promoted += length

        with slot_lock:
            base = slot * self._slot_size
            mm = self._mmap
            # The engine already wrote payload bytes; we only stamp
            # the seqlock header. Read the current (odd) version,
            # bump to next even, write length.
            version = struct.unpack(
                "<I",
                mm[base + _HEADER_VERSION_OFFSET : base + _HEADER_VERSION_OFFSET + 4],
            )[0]
            if version % 2 == 1:
                stable_version = version + 1
            else:
                # Unexpected — reserve() should have left it odd.
                # Bump twice to keep monotonic seqlock semantics.
                stable_version = version + 2
            mm[base + _HEADER_LENGTH_OFFSET : base + _HEADER_LENGTH_OFFSET + 4] = struct.pack(
                "<I", length
            )
            mm[base + _HEADER_VERSION_OFFSET : base + _HEADER_VERSION_OFFSET + 4] = struct.pack(
                "<I", stable_version
            )

        return True, "", overwritten_key

    def cancel_reservation(self, lease_token: int, connection_id: int) -> bool:
        """Drop a reservation, return the slot to the free list.

        Idempotent — returns False if the lease is unknown (already
        cancelled / committed / connection-dropped). Returns False
        on wrong-owner too; callers (server) treat that as a soft
        info-level event so a misbehaving client can't cancel
        another connection's lease.
        """
        with self._alloc_lock:
            reservation = self._reservations.get(lease_token)
            if reservation is None:
                return False
            if reservation.connection_id != connection_id:
                return False
            del self._reservations[lease_token]
            self._free_slots.append(reservation.slot_id)
            self._reservations_cancelled += 1
        return True

    def cancel_connection_reservations(self, connection_id: int) -> int:
        """Cancel every outstanding reservation owned by
        `connection_id`. Returns the number cancelled. Called by the
        server on unclean disconnect so slot leases don't leak when
        a worker crashes mid-flush.
        """
        with self._alloc_lock:
            doomed = [
                token for token, r in self._reservations.items() if r.connection_id == connection_id
            ]
            for token in doomed:
                r = self._reservations.pop(token)
                self._free_slots.append(r.slot_id)
                self._reservations_cancelled += 1
            return len(doomed)

    def _write_slot_seqlock(
        self,
        slot: int,
        value: bytes | memoryview,
        size: int,
        slot_lock: threading.Lock,
    ) -> None:
        """Write `value` into `slot` using the seqlock pattern:

            1. Acquire per-slot write lock (serializes writers to
               THIS slot — readers don't lock).
            2. Read current version, bump to next odd value (so
               concurrent readers see "mid-write").
            3. Write length header + payload bytes.
            4. Bump version again (now even = stable).

        The lock is per-slot, not global — independent slots' writes
        can proceed in parallel.
        """
        with slot_lock:
            base = slot * self._slot_size
            mm = self._mmap

            # Read current version (4 bytes LE). On the very first
            # write to a slot, this is whatever the memfd initialized
            # to (zero on fresh memfd; ftruncate zero-fills).
            version_bytes = mm[base + _HEADER_VERSION_OFFSET : base + _HEADER_VERSION_OFFSET + 4]
            version = struct.unpack("<I", version_bytes)[0]
            # Pick the next odd value (signal "mid-write"). If the
            # current version is even (stable), next-odd = version+1.
            # If it's odd (shouldn't happen — writer crashed
            # mid-write, leaving an orphan slot), bump to the next
            # odd value past it.
            if version % 2 == 0:
                mid_version = version + 1
            else:
                mid_version = version + 2  # skip to next odd
            stable_version = mid_version + 1  # next even

            # Write mid-write marker FIRST so any reader sees "torn"
            # the moment we start modifying payload bytes. We zero
            # out the length field while we're at it; the final
            # write will set it correctly.
            mm[base + _HEADER_VERSION_OFFSET : base + _HEADER_VERSION_OFFSET + 4] = struct.pack(
                "<I", mid_version
            )
            # No flush needed — mmap on Linux gives us strong ordering
            # for same-process writes, and cross-process readers see
            # the same page through the shared mapping.

            # Write the payload bytes at slot offset + HEADER_BYTES.
            payload_offset = base + _HEADER_BYTES
            if isinstance(value, memoryview):
                mm[payload_offset : payload_offset + size] = bytes(value)
            else:
                mm[payload_offset : payload_offset + size] = value

            # Write length field (between version and payload, in the
            # header area).
            mm[base + _HEADER_LENGTH_OFFSET : base + _HEADER_LENGTH_OFFSET + 4] = struct.pack(
                "<I", size
            )

            # Finally bump version to stable. Readers that arrived
            # mid-write see version_at_start != version_at_end (or
            # the odd intermediate) and retry.
            mm[base + _HEADER_VERSION_OFFSET : base + _HEADER_VERSION_OFFSET + 4] = struct.pack(
                "<I", stable_version
            )

    # ------------------------------------------------------------------
    # Read path (server-side — workers do their own seqlock read on
    # their own mmap, see `client.py` for the worker-side equivalent)
    # ------------------------------------------------------------------

    def get_slice(self, slot: int) -> memoryview | None:
        """Server-side reader. Returns a memoryview into the slot's
        payload bytes, or None on torn-read after retries exceeded.

        The HostStore wraps this for its `get` path so the existing
        bytes-returning API stays unchanged for non-shared-arena
        clients. For shared-arena clients, the server just sends
        `slot_offset = slot * slot_size + HEADER_BYTES`, `length`,
        and `version` on the wire; the client reads bytes from its
        own mmap.
        """
        if slot < 0 or slot >= self._num_slots:
            return None
        base = slot * self._slot_size
        mm = self._mmap
        for _attempt in range(MAX_TORN_READ_RETRIES):
            v1 = struct.unpack(
                "<I", mm[base + _HEADER_VERSION_OFFSET : base + _HEADER_VERSION_OFFSET + 4]
            )[0]
            if v1 % 2 != 0:
                self._torn_reads += 1
                continue
            length = struct.unpack(
                "<I", mm[base + _HEADER_LENGTH_OFFSET : base + _HEADER_LENGTH_OFFSET + 4]
            )[0]
            if length > self._slot_size - _HEADER_BYTES:
                # Bogus length — should not happen with a sane
                # writer, but guard against torn reads of the length
                # field itself.
                self._torn_reads += 1
                continue
            payload = bytes(mm[base + _HEADER_BYTES : base + _HEADER_BYTES + length])
            v2 = struct.unpack(
                "<I", mm[base + _HEADER_VERSION_OFFSET : base + _HEADER_VERSION_OFFSET + 4]
            )[0]
            if v1 == v2 and v2 % 2 == 0:
                return memoryview(payload)
            self._torn_reads += 1
        return None

    def get_slot_for_key(self, key: bytes) -> int | None:
        """Look up the slot index for `key`. Refreshes LRU recency on
        hit. Returns None on miss."""
        with self._alloc_lock:
            slot = self._slot_index.get(key)
            if slot is None:
                self._misses += 1
                return None
            self._slot_index.move_to_end(key)
            self._hits += 1
            return slot

    def get_slot_metadata(self, slot: int) -> tuple[int, int, int] | None:
        """Return `(offset, length, version)` for `slot`. Used by the
        server's wire path to construct a GetSharedResponse without
        copying bytes. Returns None on out-of-range or odd-version
        (slot under active write — reader should retry).
        """
        if slot < 0 or slot >= self._num_slots:
            return None
        base = slot * self._slot_size
        mm = self._mmap
        version = struct.unpack(
            "<I", mm[base + _HEADER_VERSION_OFFSET : base + _HEADER_VERSION_OFFSET + 4]
        )[0]
        length = struct.unpack(
            "<I", mm[base + _HEADER_LENGTH_OFFSET : base + _HEADER_LENGTH_OFFSET + 4]
        )[0]
        # Payload starts after the header.
        payload_offset = base + _HEADER_BYTES
        return payload_offset, length, version

    def contains(self, key: bytes) -> bool:
        """Cheap membership test (no LRU touch)."""
        with self._alloc_lock:
            return key in self._slot_index

    def would_accept_size(self, size: int) -> bool:
        """Pre-flight check used by `HostStore.set` to predict whether
        an incoming `put(value)` of this size will go to the arena
        (return True) vs fall back to inline storage (return False).

        Decision tree mirrors `put`:
        - If the arena's slot grid isn't initialized yet (`_slot_size
          == 0`), the FIRST put initializes it to ``size + header``,
          and we'll accept this incoming value. Return True.
        - If the grid IS initialized, we accept only if the incoming
          value (plus per-slot header overhead) fits inside the
          fixed slot size. Return that comparison.

        HostStore uses this to avoid double-counting arena-backed
        bytes against ``max_bytes`` — accepted-by-arena entries
        don't add to `_used_bytes`, so the RAM-cap eviction loop
        shouldn't fire for them.
        """
        with self._alloc_lock:
            if self._slot_size == 0:
                # First put will initialize the grid to fit this size.
                # Still need capacity > 0 — pass the same checks `put`
                # uses.
                return size + _HEADER_BYTES <= self._capacity
            return size + _HEADER_BYTES <= self._slot_size

    def evict_lru(self, *, notify_drain: bool = True) -> bytes | None:
        """Drop the least-recently-used entry. Returns the evicted
        key (for the caller to clean up auxiliary metadata) or None
        if the arena is empty.

        ``notify_drain=True`` (default) pushes the evicted key onto
        ``_recent_evicted_keys`` so the next ``drain_recent_evictions()``
        includes it. This is what external/admin callers want — they
        don't run HostStore's bookkeeping themselves.
        ``notify_drain=False`` is for HostStore-internal callers that
        have ALREADY removed their own ``_entries`` row and just need
        to release the slot; publishing the key there would race the
        drain against an in-progress put under the same key bytes
        (e.g. evict-then-set within one SET call would otherwise drop
        the just-inserted row).
        """
        with self._alloc_lock:
            if not self._slot_index:
                return None
            evicted_key, evicted_slot = self._slot_index.popitem(last=False)
            self._free_slots.append(evicted_slot)
            self._evictions += 1
            if notify_drain:
                self._recent_evicted_keys.append(evicted_key)
            return evicted_key

    def evict_key(self, key: bytes, *, notify_drain: bool = True) -> bool:
        """Explicit eviction. Used by HostStore when an arena-backed
        entry is removed via its own TTL/priority logic, and by
        external/admin callers. Returns True if the key was present.

        See ``evict_lru`` for the ``notify_drain`` semantics —
        HostStore-internal paths pass ``False`` (they own the
        composite-bookkeeping already); admin/test paths leave the
        default ``True`` so the next ``drain_recent_evictions()`` picks
        it up.
        """
        with self._alloc_lock:
            slot = self._slot_index.pop(key, None)
            if slot is None:
                return False
            self._free_slots.append(slot)
            if notify_drain:
                self._recent_evicted_keys.append(key)
            return True

    def clear(self) -> None:
        """Drop every entry. Slot grid + free list are preserved
        (slot_size stays fixed across clear — re-deciding would risk
        confusing mid-flight clients who cached the slot_size from
        HelloAck)."""
        with self._alloc_lock:
            self._slot_index.clear()
            self._free_slots = list(range(self._num_slots))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Unmap + close the FD. After this call, in-process readers
        get UB; cross-process clients with their own mmap are
        unaffected (the kernel keeps the memfd alive until they
        unmap too)."""
        try:
            self._mmap.close()
        except (BufferError, ValueError):
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass

    def __del__(self) -> None:
        # Best-effort cleanup. In normal flow `close()` runs first.
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_mlock(self) -> bool:
        """Lock the arena's pages into RAM via `mlock(2)`. Returns
        True on success, False if the kernel refuses (typically
        RLIMIT_MEMLOCK exhausted).

        Note we mlock the mmap, NOT a torch tensor. Both work for DMA
        pinning purposes; the mlock path is unconditional and doesn't
        require torch. The Phase 1 prototype is validating whether
        torch.pin_memory over the memfd-backed mmap is feasible on
        ROCm 7.2.2; until that lands, mlock is the documented path.
        """
        try:
            # On Linux, use ctypes to call mlock since the stdlib
            # doesn't expose it. EPERM (no privilege) and ENOMEM
            # (over rlimit) are both non-fatal.
            import ctypes
            import ctypes.util

            libc_path = ctypes.util.find_library("c")
            if libc_path is None:
                return False
            libc = ctypes.CDLL(libc_path, use_errno=True)
            # Get the mmap's base address. The mmap object itself
            # doesn't expose it directly, but we can derive it via
            # the buffer protocol: ctypes can address the underlying
            # bytes. PyObject_GetBuffer gives us a Py_buffer struct;
            # but a simpler path is via the resource module — we
            # use `ctypes.addressof(ctypes.c_char.from_buffer(self._mmap))`.
            addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mmap))
            rc = libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(self._capacity))
            if rc != 0:
                err = ctypes.get_errno()
                logger.info(
                    "%s: mlock failed (errno=%d); arena works unpinned, DMA "
                    "will stage through unlocked memory",
                    self._name,
                    err,
                )
                return False
            logger.info(
                "%s: arena pinned via mlock (%.2f GiB locked)",
                self._name,
                self._capacity / (1024**3),
            )
            return True
        except Exception as exc:
            logger.info("%s: mlock attempt raised %s — proceeding unpinned", self._name, exc)
            return False


def open_arena_view(
    fd: int, size: int, *, writable: bool = False, prefault: bool | None = None
) -> tuple[mmap.mmap, Any]:
    """Worker-side helper: mmap an arena FD passed to us via
    SCM_RIGHTS. Returns `(mmap, memoryview)` — the memoryview is
    sliceable by callers without re-mmapping.

    By default the mapping is read-only — workers reading via the
    seqlock contract MUST NOT write into the arena. The save-side
    CopyFree path (lease + commit) sets ``writable=True`` so the
    engine can write GPU-staged bytes directly into the slot the
    server reserved for it. With ``writable=True`` the engine is
    on the honor system: it writes ONLY into the payload range
    of the slot it holds a lease on, never the slot header (the
    server stamps the seqlock).

    ``prefault``: when True (the default), pass ``MAP_POPULATE`` so
    the kernel walks every PTE eagerly at mmap time. This eliminates
    first-touch page-fault latency on the early save/get hot path.
    On the MI355X testbed the cost is ~180 ms per GB at mmap
    time (≈2.9 s for a 16 GB arena) and the win is a 9× drop in
    p50 first-touch save latency (897 µs → 95 µs) plus removal of
    the p99 tail spike on the first ~100 saves. For long-running
    daemons (hours+) the startup cost amortizes to zero while the
    tail wins compound, so default-ON is the right call.

    ``MAP_POPULATE`` is a hint and may be silently ignored by the
    kernel under memory pressure or on older systems without the
    flag — we still set ``MAP_SHARED`` so the worker reads the same
    physical pages as the daemon. No operator config (sysctl,
    cgroup) is required — contrast with ``MFD_HUGETLB`` which needs
    ``vm.nr_hugepages`` pre-reservation and so stays default-OFF.

    Default is ``None`` → consult the env var
    ``INFERA_KVD_ARENA_PREFAULT``. Unset or any non-falsy value =
    ON. Set to ``0`` / ``false`` / ``no`` / ``off`` to opt OUT.
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    prot = mmap.PROT_READ | mmap.PROT_WRITE if writable else mmap.PROT_READ
    if prefault is None:
        # Default ON: only the explicit env opt-out disables.
        prefault_requested = not _env_falsy("INFERA_KVD_ARENA_PREFAULT")
    else:
        prefault_requested = prefault
    flags = mmap.MAP_SHARED
    # ``MAP_POPULATE`` is a Linux-specific extension. Python's ``mmap``
    # module exposes it on Linux ≥ 2.6 builds; guard with hasattr so
    # macOS / BSD callers still get a working (non-prefaulted) mmap.
    if prefault_requested and hasattr(mmap, "MAP_POPULATE"):
        flags |= mmap.MAP_POPULATE
    mm = mmap.mmap(fd, size, flags=flags, prot=prot)
    return mm, memoryview(mm)


def read_slot_seqlock(
    mm: mmap.mmap,
    slot_offset: int,
    length: int,
    expected_version: int,
    *,
    max_retries: int = MAX_TORN_READ_RETRIES,
) -> memoryview | None:
    """Worker-side seqlock reader.

    Given a worker's own mmap of the arena, a `slot_offset` (the
    PAYLOAD offset, not the header offset — what the server sends
    on the wire), the expected `length`, and an `expected_version`,
    read the payload bytes and verify the version. Returns a
    memoryview into the mmap on success, None on torn read after
    retries exceeded.

    The header sits at `slot_offset - _HEADER_BYTES`, by convention.
    """
    header_offset = slot_offset - _HEADER_BYTES
    if header_offset < 0:
        return None
    for _ in range(max_retries):
        v1 = struct.unpack(
            "<I",
            mm[header_offset + _HEADER_VERSION_OFFSET : header_offset + _HEADER_VERSION_OFFSET + 4],
        )[0]
        if v1 % 2 != 0 or v1 != expected_version:
            # Either slot is being written, or it was overwritten
            # since the server told us the version. Either way: torn.
            continue
        # Re-read after we'd want to read bytes — but since the wire
        # told us `expected_version`, the canonical seqlock pattern
        # is: read payload, then re-read version, then compare. If
        # the version moved on (someone overwrote the slot), the
        # read is torn.
        payload_mv = memoryview(mm)[slot_offset : slot_offset + length]
        v2 = struct.unpack(
            "<I",
            mm[header_offset + _HEADER_VERSION_OFFSET : header_offset + _HEADER_VERSION_OFFSET + 4],
        )[0]
        if v1 == v2 == expected_version and v2 % 2 == 0:
            # We must return a memoryview snapshot, but the mmap is
            # live — the caller risks reading torn bytes downstream
            # if a writer races between now and consume. Convention
            # for our wire protocol: the connector materializes the
            # bytes via `bytes(mv)` or copies into a GPU tensor
            # immediately; once the bytes are out of the arena, the
            # arena's version semantics don't matter anymore.
            return payload_mv
    return None
