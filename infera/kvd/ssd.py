###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SSD storage regions for infera-kvd.

Two regions with distinct policies, fed from `HostStore`:

| Region | Retention served | Write policy | Survives restart? | Eviction |
| ------ | ---------------- | ------------ | ----------------- | -------- |
| **spillover** | `short` (evicted from RAM) | lazy write_back | no — wiped on start | LRU when full |
| **long** | `long` (proactive write_through) | eager write_through + fsync | **yes** — sidecar-recovered | LRU when full |

The two regions are **disjoint** on disk. A `short` block can never
end up in the long region; a `long` block can never end up in
spillover. Routing decisions happen in `HostStore` at write time
based on the block's retention.

Why disjoint instead of one merged region with priority-aware LRU?
Two reasons:
1. **Resource isolation**. A burst of short-retention traffic can
   evict spillover blocks freely without ever touching long-retention
   bytes — operator can set the spillover region small (10-20% of
   total disk budget) without affecting persistence guarantees.
2. **Write-policy mismatch**. Spillover wants write_back (lazy,
   batch-friendly for IO throughput). Long wants write_through +
   fsync per SET (durability — must be on disk before we ack). One
   merged region with two policies is a code complexity multiplier.

## On-disk layout (since 2026-06 — Task #15 sharded refactor)

Each block lives at:

    {root}/{hash[:2]}/{hash[2:4]}/<urlencoded(composite_key)>.kvcache
    {root}/{hash[:2]}/{hash[2:4]}/<urlencoded(composite_key)>.kvcache.metadata

where ``composite_key = f"{model}|{compat_key}|{b64url(key)}"`` and
``hash = sha256(composite_key)``. The first 4 hex chars of the hash
are split 2/2 to spread entries across a 256×256 directory tree, so
no leaf shard holds more than ``N/65536`` entries — at one million
blocks that's ~15 per dir, well within NFS metadata budgets.

The ``.kvcache.metadata`` sidecar is a 4 KiB fixed-size file carrying
``{version, retention, size_bytes, last_access, model, compat_key,
value_sha256}``. On startup the long region rebuilds its in-memory
index by parallel scandir over each ``{hash[:2]}`` shard, reading
sidecars only (NOT the data files — sidecar is small). This subsumes
the legacy ``manifest.json``.

Atomic writes order: **data first, then sidecar**. On crash between
the two renames, the next startup scan finds a ``.kvcache`` with no
matching ``.kvcache.metadata`` and treats it as an orphan (unlinked).
This is safer than the inverse — a dangling sidecar pointing at no
data would force a phantom-entry recovery.

## Migration from the legacy ``{root}/blocks/<path>.kv`` layout

No in-place migration. If a region's root contains both ``blocks/``
and the new sharded shape, ``start()`` raises — ambiguous state. If
only ``blocks/`` is present, we log a manual-migration WARN and
ignore it (recovery proceeds with the new layout, empty). Operators
should ``rm -rf blocks/`` (or move it aside) after confirming the
data is no longer needed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import shutil
import struct
import threading
import time
import urllib.parse
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from infera.kvd.wire import RETENTION_LONG, RETENTION_SHORT, validate_retention

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Filesystem-type detection (engine-side hipFile auto-disable on tmpfs)
# ----------------------------------------------------------------------

_HIPFILE_COMPAT_FSTYPES = frozenset({"tmpfs", "overlay", "overlayfs"})


def _read_mounts() -> list[tuple[str, str]]:
    """Parse /proc/mounts → [(mount_point, fstype), ...].

    Mount points are octal-escaped (``\\040`` etc.) for whitespace;
    decode them so prefix matching works for paths like
    ``/mnt/my dir/foo``.
    """
    mounts: list[tuple[str, str]] = []
    with open("/proc/mounts", encoding="utf-8") as f:
        for raw in f:
            parts = raw.split()
            if len(parts) < 3:
                continue
            mounts.append((_decode_mount_field(parts[1]), parts[2]))
    return mounts


def _decode_mount_field(s: str) -> str:
    """Decode /proc/mounts octal escapes (``\\040`` etc.). Stdlib only."""
    if "\\" not in s:
        return s
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 3 < len(s) and s[i + 1 : i + 4].isdigit():
            try:
                out.append(chr(int(s[i + 1 : i + 4], 8)))
                i += 4
                continue
            except ValueError:
                pass
        out.append(s[i])
        i += 1
    return "".join(out)


def _is_path_prefix(mount_point: str, target: str) -> bool:
    """Path-segment-aware prefix check. ``/var`` does NOT match
    ``/var-old`` — only ``/var`` and ``/var/...``."""
    if mount_point == target:
        return True
    if mount_point == "/":
        return target.startswith("/")
    return target.startswith(mount_point + "/")


def get_fstype(path: str | Path) -> str:
    """Return the filesystem type of the mount backing ``path``.

    The path is resolved (symlinks + relative parts collapsed) before
    matching, so bind-mounts and symlinked dirs land on the right
    underlying mount. Longest-prefix wins.

    Raises ``RuntimeError`` if no mount in /proc/mounts is a prefix —
    means /proc/mounts is broken or we're in an exotic container.
    """
    resolved_str = str(Path(path).resolve())
    best_mount: str | None = None
    best_fstype: str | None = None
    for mount_point, fstype in _read_mounts():
        if not _is_path_prefix(mount_point, resolved_str):
            continue
        if best_mount is None or len(mount_point) > len(best_mount):
            best_mount = mount_point
            best_fstype = fstype
    if best_fstype is None:
        raise RuntimeError(f"Unable to detect fstype for {path}")
    return best_fstype


def hipfile_friendly_fstype(fstype: str) -> bool:
    """False for tmpfs/overlay/overlayfs — hipFile direct DMA silently
    compat-falls-back on those, so the GDS path is pure overhead.
    True for everything else; let hipFile decide at runtime."""
    return fstype not in _HIPFILE_COMPAT_FSTYPES


# ----------------------------------------------------------------------
# Sidecar format
# ----------------------------------------------------------------------

_SIDECAR_VERSION = 1
_SIDECAR_HEADER_LEN_BYTES = 8  # uint64 LE = length of JSON payload
_SIDECAR_TOTAL_BYTES = 4096

# On-disk extensions.
_DATA_EXT = ".kvcache"
_METADATA_EXT = ".kvcache.metadata"


class SidecarError(Exception):
    """Raised when a sidecar file is missing, malformed, or refuses to
    parse. The caller decides whether to drop, repair, or fail."""


def _encode_sidecar(payload: dict) -> bytes:
    """Encode a metadata dict into a fixed 4 KiB sidecar buffer.

    Layout:
        [0..8)            uint64 LE  payload length (N)
        [8..8+N)          JSON payload (utf-8)
        [8+N..4096)       zero padding

    Fixed-size buffers + a length prefix let us add fields later
    without invalidating older readers (they just stop at N).
    """
    json_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(json_bytes) > _SIDECAR_TOTAL_BYTES - _SIDECAR_HEADER_LEN_BYTES:
        raise SidecarError(
            f"sidecar payload too large: {len(json_bytes)} > "
            f"{_SIDECAR_TOTAL_BYTES - _SIDECAR_HEADER_LEN_BYTES} bytes"
        )
    header = struct.pack("<Q", len(json_bytes))
    pad = b"\0" * (_SIDECAR_TOTAL_BYTES - _SIDECAR_HEADER_LEN_BYTES - len(json_bytes))
    return header + json_bytes + pad


def _decode_sidecar(blob: bytes) -> dict:
    """Inverse of `_encode_sidecar`. Raises SidecarError on any
    malformed input (wrong length, bad header, garbled JSON)."""
    if len(blob) != _SIDECAR_TOTAL_BYTES:
        raise SidecarError(f"sidecar wrong size: got {len(blob)}, expected {_SIDECAR_TOTAL_BYTES}")
    (n,) = struct.unpack("<Q", blob[:_SIDECAR_HEADER_LEN_BYTES])
    if n > _SIDECAR_TOTAL_BYTES - _SIDECAR_HEADER_LEN_BYTES:
        raise SidecarError(f"sidecar header length {n} exceeds buffer")
    try:
        return json.loads(blob[_SIDECAR_HEADER_LEN_BYTES : _SIDECAR_HEADER_LEN_BYTES + n])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SidecarError(f"sidecar JSON malformed: {exc}") from exc


# ----------------------------------------------------------------------
# Composite key encoding (reversible, filesystem-safe)
# ----------------------------------------------------------------------


def _encode_composite(model: str, compat_key: str, key: bytes) -> str:
    """Build the unique-on-disk identifier for a block.

    Format: ``f"{model}|{compat_key}|{b64url_no_pad(key)}"``

    - ``|`` separator: not legal in model names per HuggingFace
      convention; if it ever shows up we still get a unique string
      because b64 is bounded on the right by the EOL.
    - Key is base64 (urlsafe, no padding) so arbitrary binary keys
      round-trip without losing bytes.

    The string itself is then URL-encoded before use as a filename so
    the on-disk name is always POSIX-safe.
    """
    key_b64 = base64.urlsafe_b64encode(key).rstrip(b"=").decode("ascii")
    return f"{model}|{compat_key}|{key_b64}"


def _decode_composite(composite: str) -> tuple[str, str, bytes]:
    """Inverse of `_encode_composite`. Raises ValueError on malformed."""
    parts = composite.split("|", 2)
    if len(parts) != 3:
        raise ValueError(f"composite key missing fields: {composite!r}")
    model, compat_key, key_b64 = parts
    # Re-add padding for urlsafe_b64decode.
    pad = "=" * (-len(key_b64) % 4)
    try:
        key = base64.urlsafe_b64decode(key_b64 + pad)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError(f"composite key b64 decode failed: {exc}") from exc
    return model, compat_key, key


def _filename_for_composite(composite: str) -> str:
    """URL-encode the composite for use as a filename. ``safe=""``
    encodes literally everything except unreserved chars, so the
    result is a flat POSIX-safe string with no path separators."""
    return urllib.parse.quote(composite, safe="")


def _composite_hash(composite: str) -> str:
    """sha256 hex; first 4 chars used for 2/2 sharding."""
    return hashlib.sha256(composite.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# In-memory entry
# ----------------------------------------------------------------------


@dataclass
class SsdEntry:
    """Per-block metadata kept in-memory.

    Bytes live in
    ``{root}/{hash[:2]}/{hash[2:4]}/<urlencoded(composite)>.kvcache``
    where composite = encode_composite(model, compat_key, key) and
    hash = sha256(composite).
    """

    key: bytes
    size_bytes: int
    retention: str
    model: str = ""
    compat_key: str = ""
    inserted_at: float = field(default_factory=time.monotonic)
    last_access: float = field(default_factory=time.monotonic)
    metadata: dict = field(default_factory=dict)

    @property
    def composite(self) -> str:
        return _encode_composite(self.model, self.compat_key, self.key)


def _to_wall(monotonic_t: float) -> float:
    """Convert a monotonic timestamp to wall-clock time for persistence.
    The conversion is approximate (we just store the wall-clock now if
    the monotonic delta is fresh) but good enough for "is this older
    than that" comparisons."""
    return time.time() - (time.monotonic() - monotonic_t)


def _from_wall(wall_t: float) -> float:
    """Inverse of _to_wall: reconstruct a monotonic-ish reference. After
    restart the monotonic clock has reset, so this is approximate. Used
    only as an ordering hint for LRU."""
    return time.monotonic() - (time.time() - wall_t)


# ----------------------------------------------------------------------
# Region base class
# ----------------------------------------------------------------------


class SsdRegion(ABC):
    """Shared shape for spillover + long regions.

    Sub-classes differ in:
      - Write policy (lazy vs eager + fsync)
      - Whether to wipe / recover on startup
    """

    def __init__(self, path: str | Path, max_bytes: int, name: str) -> None:
        if max_bytes <= 0:
            raise ValueError(f"{name}: max_bytes must be positive, got {max_bytes}")
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._name = name
        self._entries: dict[tuple[str, str, bytes], SsdEntry] = {}
        self._used_bytes = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        return self._path

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    @property
    def entries_count(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Lifecycle — subclasses override
    # ------------------------------------------------------------------

    @abstractmethod
    def start(self) -> None:
        """Prepare the region for use. Spillover: clean slate. Long:
        recover from sidecars."""

    def shutdown(self) -> None:  # noqa: B027 — intentional no-op hook for subclasses
        """Optional override for graceful shutdown work."""

    def _log_fstype_if_compat(self) -> None:
        """Emit one INFO line at start() if this region sits on a
        filesystem where hipFile / GDS would compat-fallback.

        kvd's own POSIX writes work fine on tmpfs/overlay — this log
        is purely informational for operators who expect engine-side
        file-tier reads to use direct DMA. The engine (F1) probes
        fstype itself before opening hipFile; this log makes the
        situation greppable from kvd's startup output.
        """
        try:
            fstype = get_fstype(self._path)
        except (OSError, RuntimeError) as exc:
            # Don't fail region startup over a detection edge case.
            logger.debug("%s: fstype detection skipped (%s)", self._name, exc)
            return
        if not hipfile_friendly_fstype(fstype):
            logger.info(
                "ssd region %s on %s — file-tier reads would compat-fallback "
                "under hipFile; sharded layout still functional for the "
                "kvd-internal write path",
                self._name,
                fstype,
            )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_bytes(self, key: bytes, *, model: str = "", compat_key: str = "") -> bytes | None:
        """Read a block's bytes from disk. Returns None on miss. Refreshes
        `last_access` on hit so LRU stays accurate."""
        composite = (model, compat_key, key)
        with self._lock:
            entry = self._entries.get(composite)
            if entry is None:
                return None
            entry.last_access = time.monotonic()
            data_path, _ = self._paths_for(entry)

        # Disk read outside the lock — bytes can be many MB.
        try:
            return data_path.read_bytes()
        except OSError as exc:
            logger.warning("%s: failed to read %s: %s", self._name, data_path, exc)
            with self._lock:
                # File missing or unreadable — drop from index so future
                # lookups don't keep retrying.
                self._entries.pop(composite, None)
                self._used_bytes = max(0, self._used_bytes - entry.size_bytes)
            return None

    def exists(self, keys: list[bytes], *, model: str = "", compat_key: str = "") -> list[bool]:
        with self._lock:
            return [(model, compat_key, k) in self._entries for k in keys]

    def get_entry(self, key: bytes, *, model: str = "", compat_key: str = "") -> SsdEntry | None:
        """Index-only lookup (no disk read). Returns the entry metadata
        or None. Used by the host store to decide whether to populate
        the host RAM cache from this region."""
        with self._lock:
            return self._entries.get((model, compat_key, key))

    # ------------------------------------------------------------------
    # Write — subclasses override write policy
    # ------------------------------------------------------------------

    @abstractmethod
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
        """Insert a block into this region. Returns (accepted, reason)
        the same shape as `HostStore.set`."""

    def insert_metadata_only(
        self,
        key: bytes,
        *,
        path: str,
        size: int,
        retention: str,
        model: str = "",
        compat_key: str = "",
        version: int = 0,
    ) -> tuple[bool, str | None]:
        """Record metadata for an entry whose BYTES were written by the
        engine via hipFileWrite — kvd never opens or stats the file.

        Used by the RegisterFileEntry wire op: engine owns L3 bytes, kvd owns
        the metadata index. ``path`` lives in ``metadata['path']``
        and ``version`` in ``metadata['version']`` so LookupTier can
        answer without disk IO.

        Retention routing (Spillover-only-short, Long-only-long) is
        enforced at the server level (``_handle_register_file_entry``),
        not here. Only validates: ``retention`` is one of
        ``none|short|long`` and ``size`` > 0.

        On region-full → returns ``(False, "region_full")``. Eviction
        uses the same LRU policy as ``put``. NOTE: when an evicted entry
        is hipfile-tier, ``_delete_block_files_locked`` no-ops on the
        kvd-owned sharded paths (FileNotFoundError swallowed). kvd does
        NOT touch the engine-owned file at ``metadata['path']``.

        Restart NOTE: hipfile-tier entries are in-memory only — they do
        NOT survive kvd daemon restart. Engine must re-issue
        RegisterFileEntry after a restart. Persistent sidecars for
        hipfile entries are a follow-up.
        """
        validate_retention(retention)
        if size <= 0:
            return False, "bad_size"

        composite = (model, compat_key, key)
        meta = {"path": str(path), "version": int(version), "_hipfile": True}
        with self._lock:
            existing = self._entries.get(composite)
            if existing is not None:
                old_size = existing.size_bytes
                existing.size_bytes = size
                existing.last_access = time.monotonic()
                existing.metadata = meta
                existing.retention = retention
                self._used_bytes += size - old_size
                return True, None

            while self._used_bytes + size > self._max_bytes:
                victim = self._pick_lru_victim_locked()
                if victim is None:
                    return False, "region_full"
                self._evict_locked(victim)

            entry = SsdEntry(
                key=key,
                size_bytes=size,
                retention=retention,
                model=model,
                compat_key=compat_key,
                metadata=meta,
            )
            self._entries[composite] = entry
            self._used_bytes += size
            return True, None

    # ------------------------------------------------------------------
    # Eviction (LRU) and removal
    # ------------------------------------------------------------------

    def remove(self, key: bytes, *, model: str = "", compat_key: str = "") -> bool:
        """Drop a block from the region. Returns True if it was present."""
        composite = (model, compat_key, key)
        with self._lock:
            entry = self._entries.pop(composite, None)
            if entry is None:
                return False
            self._used_bytes = max(0, self._used_bytes - entry.size_bytes)
            self._delete_block_files_locked(entry)
            return True

    def clear(self) -> int:
        """Drop everything in the region. Returns number of entries removed."""
        with self._lock:
            count = len(self._entries)
            for entry in self._entries.values():
                self._delete_block_files_locked(entry)
            self._entries.clear()
            self._used_bytes = 0
        return count

    # ------------------------------------------------------------------
    # Internal helpers — must be called under self._lock unless noted
    # ------------------------------------------------------------------

    def _paths_for(self, entry: SsdEntry) -> tuple[Path, Path]:
        """Return (data_path, metadata_path) for an entry.

        Computes the 2/2 sharded directory and the urlencoded composite
        filename. Pure function of the entry's (model, compat_key, key).
        """
        return self._paths_for_composite(entry.composite)

    def _paths_for_composite(self, composite: str) -> tuple[Path, Path]:
        h = _composite_hash(composite)
        shard_dir = self._path / h[:2] / h[2:4]
        fname = _filename_for_composite(composite)
        return shard_dir / (fname + _DATA_EXT), shard_dir / (fname + _METADATA_EXT)

    def _delete_block_files_locked(self, entry: SsdEntry) -> None:
        """Best-effort unlink of both data and sidecar."""
        data_path, meta_path = self._paths_for(entry)
        for p in (data_path, meta_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("%s: failed to unlink %s: %s", self._name, p, exc)

    def _pick_lru_victim_locked(self) -> tuple[str, str, bytes] | None:
        """Plain LRU within the region — no priority class. Returns
        the composite key of the oldest entry, or None if empty."""
        if not self._entries:
            return None
        best_key = None
        best_t = float("inf")
        for k, entry in self._entries.items():
            if entry.last_access < best_t:
                best_t = entry.last_access
                best_key = k
        return best_key

    def _evict_locked(self, composite: tuple[str, str, bytes]) -> None:
        entry = self._entries.pop(composite)
        self._used_bytes = max(0, self._used_bytes - entry.size_bytes)
        self._delete_block_files_locked(entry)

    def _write_block_file_unlocked(self, entry: SsdEntry, value: bytes, *, fsync: bool) -> None:
        """Write data + sidecar atomically. NOT under self._lock —
        disk I/O is slow; the caller has already reserved capacity in
        `_used_bytes`.

        Order:
          1. Write data file via ``.tmp.<rand>`` + ``os.replace``.
          2. Write sidecar via ``.tmp.<rand>`` + ``os.replace``.

        If we crash between (1) and (2), the next startup scan sees a
        data file with no sidecar and unlinks it (orphan reclaim). If
        we crashed in the middle of (1), the .tmp file gets unlinked
        on next scan. The sidecar is the authoritative "this entry
        committed" marker.
        """
        data_path, meta_path = self._paths_for(entry)
        shard_dir = data_path.parent
        shard_dir.mkdir(parents=True, exist_ok=True)

        rand_suffix = os.urandom(8).hex()
        data_tmp = data_path.with_name(data_path.name + f".tmp.{rand_suffix}")
        meta_tmp = meta_path.with_name(meta_path.name + f".tmp.{rand_suffix}")

        # ---- Data ----
        try:
            with data_tmp.open("wb") as f:
                f.write(value)
                if fsync:
                    f.flush()
                    os.fsync(f.fileno())
            os.replace(data_tmp, data_path)
        except OSError:
            try:
                data_tmp.unlink()
            except OSError:
                pass
            raise

        # ---- Sidecar ----
        sidecar_payload = {
            "version": _SIDECAR_VERSION,
            "retention": entry.retention,
            "size_bytes": entry.size_bytes,
            "inserted_at": _to_wall(entry.inserted_at),
            "last_access": _to_wall(entry.last_access),
            "model": entry.model,
            "compat_key": entry.compat_key,
            # value_sha256 is optional / informational; included so
            # operator tools can fsck disk vs sidecar without reading
            # the whole file (just over the data once at write time).
            "value_sha256": hashlib.sha256(value).hexdigest(),
            "metadata": entry.metadata,
        }
        try:
            sidecar_blob = _encode_sidecar(sidecar_payload)
            with meta_tmp.open("wb") as f:
                f.write(sidecar_blob)
                if fsync:
                    f.flush()
                    os.fsync(f.fileno())
            os.replace(meta_tmp, meta_path)
        except OSError:
            # Data is on disk but sidecar failed — leave the data
            # file; next startup scan will treat as orphan and clean.
            try:
                meta_tmp.unlink()
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Startup helpers
    # ------------------------------------------------------------------

    def _assert_no_legacy_blocks_dir(self) -> None:
        """Refuse to start with ambiguous on-disk shape.

        Old layout: ``{root}/blocks/<path>.kv``.
        New layout: ``{root}/{XX}/{YY}/...``.

        If both are present we don't know which to trust — fail fast.
        If only the old one is present we WARN and proceed with an
        empty index (operator decides if they need the legacy bytes).
        """
        legacy = self._path / "blocks"
        if not legacy.exists():
            return

        # Detect at least one new-style shard dir at the root.
        has_new_shape = False
        if self._path.exists():
            for child in self._path.iterdir():
                # 2-hex shard names: lowercase hex chars, exactly 2.
                if child.is_dir() and len(child.name) == 2 and _is_lower_hex(child.name):
                    has_new_shape = True
                    break

        if has_new_shape:
            raise RuntimeError(
                f"{self._name}: ambiguous on-disk state at {self._path}: both "
                f"legacy 'blocks/' and new sharded layout exist. Resolve "
                f"manually (move 'blocks/' aside) before restarting."
            )

        logger.warning(
            "%s: legacy 'blocks/' directory at %s is ignored. The on-disk "
            "format changed in 2026-06 to 2/2-sharded "
            "<XX>/<YY>/<urlencoded(key)>.kvcache. Manual migration "
            "instructions: stop kvd, copy any blocks you want to preserve "
            "into a separate dir for offline reprocessing, then rm -rf "
            "%s. Recovery is proceeding with an empty index.",
            self._name,
            legacy,
            legacy,
        )

    def _scan_metadata_dir(self, shard_dir: Path) -> list[tuple[SsdEntry, int]]:
        """Scan a single ``{root}/{XX}`` shard dir for sidecars and
        rebuild a list of ``(entry, size_bytes)`` tuples.

        Also performs orphan reclaim:

          - ``.kvcache`` with no matching ``.kvcache.metadata`` → unlink
            (crash between data and sidecar write).
          - ``.kvcache.metadata`` with no matching ``.kvcache`` → unlink
            both (sidecar dangling without data).
          - Leftover ``.tmp.*`` files → unlink.
          - Sidecars that fail to decode or claim a size that doesn't
            match the data file → unlink both.

        Returns whatever survived. May be called concurrently from a
        threadpool; doesn't touch ``self._entries``.
        """
        out: list[tuple[SsdEntry, int]] = []
        if not shard_dir.exists():
            return out

        for leaf in shard_dir.iterdir():
            if not leaf.is_dir():
                continue
            if len(leaf.name) != 2 or not _is_lower_hex(leaf.name):
                continue
            out.extend(self._scan_leaf_dir(leaf))
        return out

    def _scan_leaf_dir(self, leaf: Path) -> list[tuple[SsdEntry, int]]:
        """Scan one ``{XX}/{YY}`` directory."""
        # First pass: collect data and metadata stems separately.
        data_stems: dict[str, Path] = {}
        meta_stems: dict[str, Path] = {}
        tmp_files: list[Path] = []
        try:
            children = list(leaf.iterdir())
        except OSError as exc:
            logger.warning("%s: scan failed at %s: %s", self._name, leaf, exc)
            return []

        for f in children:
            name = f.name
            if ".tmp." in name:
                tmp_files.append(f)
            elif name.endswith(_METADATA_EXT):
                meta_stems[name[: -len(_METADATA_EXT)]] = f
            elif name.endswith(_DATA_EXT):
                data_stems[name[: -len(_DATA_EXT)]] = f
            # Ignore anything else — leave for operators.

        # Reclaim orphans.
        for f in tmp_files:
            try:
                f.unlink()
            except OSError:
                pass

        # Data without sidecar → orphan (crash mid-write).
        for stem, p in data_stems.items():
            if stem not in meta_stems:
                logger.info("%s: orphan data file %s (no sidecar) — unlinking", self._name, p)
                try:
                    p.unlink()
                except OSError:
                    pass

        out: list[tuple[SsdEntry, int]] = []
        for stem, meta_path in meta_stems.items():
            data_path = data_stems.get(stem)
            if data_path is None:
                logger.warning(
                    "%s: sidecar %s has no data file — unlinking sidecar",
                    self._name,
                    meta_path,
                )
                try:
                    meta_path.unlink()
                except OSError:
                    pass
                continue

            try:
                blob = meta_path.read_bytes()
                payload = _decode_sidecar(blob)
            except (OSError, SidecarError) as exc:
                logger.warning(
                    "%s: sidecar %s unreadable (%s) — dropping pair",
                    self._name,
                    meta_path,
                    exc,
                )
                _unlink_pair(data_path, meta_path)
                continue

            # Validate version.
            if payload.get("version") != _SIDECAR_VERSION:
                logger.warning(
                    "%s: sidecar %s version %r unsupported — dropping pair",
                    self._name,
                    meta_path,
                    payload.get("version"),
                )
                _unlink_pair(data_path, meta_path)
                continue

            # Decode composite from filename (URL-decode then split).
            try:
                composite = urllib.parse.unquote(stem)
                model, compat_key, key = _decode_composite(composite)
            except ValueError as exc:
                logger.warning(
                    "%s: filename %s not decodable as composite (%s) — dropping pair",
                    self._name,
                    stem,
                    exc,
                )
                _unlink_pair(data_path, meta_path)
                continue

            # Cross-check sidecar's claimed (model, compat_key) against
            # filename's decoded composite — mismatches mean tampering.
            if payload.get("model") != model or payload.get("compat_key") != compat_key:
                logger.warning(
                    "%s: sidecar %s identity mismatch (filename=%r/%r, "
                    "sidecar=%r/%r) — dropping pair",
                    self._name,
                    meta_path,
                    model,
                    compat_key,
                    payload.get("model"),
                    payload.get("compat_key"),
                )
                _unlink_pair(data_path, meta_path)
                continue

            # Cross-check size against on-disk data.
            try:
                actual_size = data_path.stat().st_size
            except OSError as exc:
                logger.warning(
                    "%s: data file %s stat failed (%s) — dropping pair",
                    self._name,
                    data_path,
                    exc,
                )
                _unlink_pair(data_path, meta_path)
                continue
            declared_size = int(payload.get("size_bytes", -1))
            if actual_size != declared_size:
                logger.warning(
                    "%s: %s size mismatch (sidecar=%d, disk=%d) — dropping pair",
                    self._name,
                    data_path,
                    declared_size,
                    actual_size,
                )
                _unlink_pair(data_path, meta_path)
                continue

            entry = SsdEntry(
                key=key,
                size_bytes=actual_size,
                retention=str(payload.get("retention", RETENTION_LONG)),
                model=model,
                compat_key=compat_key,
                inserted_at=_from_wall(float(payload.get("inserted_at", time.time()))),
                last_access=_from_wall(float(payload.get("last_access", time.time()))),
                metadata=dict(payload.get("metadata") or {}),
            )
            out.append((entry, actual_size))
        return out


def _unlink_pair(*paths: Path) -> None:
    for p in paths:
        try:
            p.unlink()
        except OSError:
            pass


def _is_lower_hex(s: str) -> bool:
    if not s:
        return False
    return all(c in "0123456789abcdef" for c in s)


# ----------------------------------------------------------------------
# Spillover region — short retention, lazy write, wiped on restart
# ----------------------------------------------------------------------


class SpilloverRegion(SsdRegion):
    """For `short`-retention blocks evicted from host RAM. Lazy
    write_back (no fsync per block). LRU eviction within the region.
    Wiped on every startup — there's no persistence guarantee.

    The contract with `HostStore`: when the store evicts a short-
    retention block from RAM, it calls `spillover.put(...)`. On a
    cache GET, the store consults spillover after the long region.
    """

    def __init__(self, path: str | Path, max_bytes: int) -> None:
        super().__init__(path, max_bytes, name="spillover")

    def start(self) -> None:
        """Wipe any leftover blocks from a prior process. By spec, the
        spillover region is non-persistent — restart drops everything."""
        # Refuse ambiguous state (old + new on disk simultaneously).
        self._assert_no_legacy_blocks_dir()

        if self._path.exists():
            # Remove every 2-hex shard dir at the root. We don't blow
            # away the whole region root because operators may have
            # mounted it specifically; the legacy 'blocks/' WARN was
            # already emitted by _assert_no_legacy_blocks_dir().
            for child in self._path.iterdir():
                if child.is_dir() and len(child.name) == 2 and _is_lower_hex(child.name):
                    shutil.rmtree(child, ignore_errors=True)
        self._path.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._entries.clear()
            self._used_bytes = 0
        logger.info(
            "spillover region initialized at %s (max=%d bytes)", self._path, self._max_bytes
        )
        self._log_fstype_if_compat()

    def put(
        self,
        key: bytes,
        value: bytes,
        *,
        retention: str = RETENTION_SHORT,
        model: str = "",
        compat_key: str = "",
        metadata: dict | None = None,
    ) -> tuple[bool, str | None]:
        """Insert a short-retention block. Refuses to accept long-retention
        (those go to the long region)."""
        validate_retention(retention)
        if retention != RETENTION_SHORT:
            return False, "spillover_only_accepts_short_retention"

        size = len(value)
        if size > self._max_bytes:
            return False, "value_larger_than_region"

        composite = (model, compat_key, key)
        with self._lock:
            existing = self._entries.get(composite)
            if existing is not None:
                # Update in place — adjust used bytes for size delta.
                old_size = existing.size_bytes
                existing.size_bytes = size
                existing.last_access = time.monotonic()
                existing.metadata = metadata or {}
                self._used_bytes += size - old_size
                entry_to_write = existing
            else:
                # Evict LRU until we have room.
                while self._used_bytes + size > self._max_bytes:
                    victim = self._pick_lru_victim_locked()
                    if victim is None:
                        return False, "spillover_full_no_victim"
                    self._evict_locked(victim)

                entry_to_write = SsdEntry(
                    key=key,
                    size_bytes=size,
                    retention=retention,
                    model=model,
                    compat_key=compat_key,
                    metadata=metadata or {},
                )
                self._entries[composite] = entry_to_write
                self._used_bytes += size

        # Disk write outside the lock. write_back semantics — no fsync.
        try:
            self._write_block_file_unlocked(entry_to_write, value, fsync=False)
        except (OSError, SidecarError) as exc:
            # Roll back the index entry on disk failure.
            with self._lock:
                rolled = self._entries.pop(composite, None)
                if rolled is not None:
                    self._used_bytes = max(0, self._used_bytes - rolled.size_bytes)
            return False, f"disk_write_failed: {exc}"
        return True, None


# ----------------------------------------------------------------------
# Long region — long retention, write-through with fsync, persistent
# ----------------------------------------------------------------------


class LongStorageRegion(SsdRegion):
    """For `long`-retention blocks. Write_through; fsync optional.

    Recovery on startup is sidecar-driven: each block has a 4 KiB
    ``.kvcache.metadata`` next to it, and the index is reconstructed
    by parallel scandir over the ``{root}/{XX}`` shard dirs at start
    time. The legacy ``manifest.json`` was removed in the 2026-06
    refactor — the sidecar format subsumes it.

    The contract with `HostStore`: when the store handles a
    ``set(retention=long, ...)``, it calls ``long_region.put(...)``
    BEFORE acknowledging the client. On a cache GET, the store
    consults long before spillover.

    Durability vs throughput knob:
      ``INFERA_KVD_LONG_FSYNC=1`` → fsync per SET. Block is durable
        before ack; kvd crash never loses an acknowledged Set.
      ``INFERA_KVD_LONG_FSYNC=0`` (default) → write_through to page
        cache, no per-SET fsync. Block is visible to GET immediately
        (page-cache hit) and survives kvd daemon restart via sidecar
        scan on the OS-flushed file. On UNCLEAN host crash within a
        few seconds of the Set, the block may be lost — those keys
        then read as cache miss, NOT corruption.

      The default is OFF because this is an L3 cache, not a database:
      losing a few seconds of writes on host crash is acceptable, and
      the per-Set fsync cost dominates save wall time under heavy
      agentic load (bench: c=32 dataset-replay shows 30%+ TTFT drop
      after switching default-retention to long with fsync=on, no
      throughput benefit on the MI355X testbed where L3 reload bandwidth is
      already CPU-bounce-limited).
    """

    # Cap thread pool used for parallel startup scan. The scan is
    # mostly metadata IO; small thread count keeps NFS / fuse mounts
    # from getting swamped. Operators can override via env if needed.
    _SCAN_WORKERS = int(os.environ.get("INFERA_KVD_SCAN_WORKERS", "16"))

    # Per-SET fsync — default off (cache durability is best-effort).
    _FSYNC_PER_SET = os.environ.get("INFERA_KVD_LONG_FSYNC", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    def __init__(self, path: str | Path, max_bytes: int) -> None:
        super().__init__(path, max_bytes, name="long")

    def start(self) -> None:
        """Recover index by parallel-scanning sidecar files. Bytes are
        NOT loaded; each block is loaded lazily on first ``get_bytes``."""
        # Refuse ambiguous state.
        self._assert_no_legacy_blocks_dir()
        self._path.mkdir(parents=True, exist_ok=True)

        entries = self._recover_from_sidecars()

        with self._lock:
            self._entries.clear()
            self._used_bytes = 0
            for entry in entries:
                self._entries[(entry.model, entry.compat_key, entry.key)] = entry
                self._used_bytes += entry.size_bytes

        logger.info(
            "long region recovered from %s: %d entries, %d bytes",
            self._path,
            len(self._entries),
            self._used_bytes,
        )
        self._log_fstype_if_compat()

    def _recover_from_sidecars(self) -> list[SsdEntry]:
        """Parallel scan over ``{root}/{XX}`` shard dirs.

        Each worker handles one L1 shard (which fans out to L2 leaves
        internally). With 16 workers and 256 L1 shards we get ~16x
        scandir parallelism, enough to amortize NFS latency without
        thrashing the metadata server.
        """
        if not self._path.exists():
            return []

        shard_dirs = [
            d
            for d in self._path.iterdir()
            if d.is_dir() and len(d.name) == 2 and _is_lower_hex(d.name)
        ]
        if not shard_dirs:
            return []

        all_entries: list[SsdEntry] = []
        # ThreadPoolExecutor: small pool, mostly metadata-bound IO.
        with ThreadPoolExecutor(max_workers=self._SCAN_WORKERS) as pool:
            for shard_results in pool.map(self._scan_metadata_dir, shard_dirs):
                for entry, _size in shard_results:
                    all_entries.append(entry)
        return all_entries

    def put(
        self,
        key: bytes,
        value: bytes,
        *,
        retention: str = RETENTION_LONG,
        model: str = "",
        compat_key: str = "",
        metadata: dict | None = None,
    ) -> tuple[bool, str | None]:
        """Insert a long-retention block. Refuses other retentions."""
        validate_retention(retention)
        if retention != RETENTION_LONG:
            return False, "long_region_only_accepts_long_retention"

        size = len(value)
        if size > self._max_bytes:
            return False, "value_larger_than_region"

        composite = (model, compat_key, key)
        with self._lock:
            existing = self._entries.get(composite)
            if existing is not None:
                old_size = existing.size_bytes
                existing.size_bytes = size
                existing.last_access = time.monotonic()
                existing.metadata = metadata or {}
                self._used_bytes += size - old_size
                entry_to_write = existing
            else:
                # Evict LRU until we have room.
                while self._used_bytes + size > self._max_bytes:
                    victim = self._pick_lru_victim_locked()
                    if victim is None:
                        return False, "long_region_full_no_victim"
                    self._evict_locked(victim)

                entry_to_write = SsdEntry(
                    key=key,
                    size_bytes=size,
                    retention=retention,
                    model=model,
                    compat_key=compat_key,
                    metadata=metadata or {},
                )
                self._entries[composite] = entry_to_write
                self._used_bytes += size

        try:
            # write_through to page cache. fsync controlled by env
            # flag (see class docstring): OFF by default so save cost
            # tracks the spillover path; ON when the operator wants
            # strict per-Set durability. Sidecar is part of the
            # atomic commit (data first, then sidecar) and is also
            # subject to the same fsync flag.
            self._write_block_file_unlocked(
                entry_to_write,
                value,
                fsync=self._FSYNC_PER_SET,
            )
        except (OSError, SidecarError) as exc:
            with self._lock:
                rolled = self._entries.pop(composite, None)
                if rolled is not None:
                    self._used_bytes = max(0, self._used_bytes - rolled.size_bytes)
            return False, f"disk_write_failed: {exc}"

        return True, None
