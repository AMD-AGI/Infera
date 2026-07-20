###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SGLang HiCacheStorage backend that proxies to infera-kvd.

Phase 4.5 (skeleton). Implements the methods SGLang's hicache layer
calls when blocks are evicted from device HBM and need to spill to
the next tier:

    HBM (SGLang radix tree, with PriorityStrategy eviction)
        ↓ evict
    Host RAM (SGLang's HostKVCache, mirrors radix tree)
        ↓ evict (this is where we intercept)
    infera-kvd's HostStore  →  Phase 4 also writes SSD long region

Why proxy instead of inheriting SGLang's `HiCacheFile`:
- We want one node-wide store across multiple worker processes.
- We want retention-aware partitioning that the file backend lacks.
- We want restart recovery (Phase 4) that the file backend lacks.

## Retention propagation status (Phase A.2)

- The adapter implements `batch_set_v1(extra_info=...)` and reads
  `extra_info.extra_info["infera_retention"]` when present —
  this is the in-band path that needs upstream SGLang to populate
  the dict from request priority. Tracking PR upstream; until it
  lands, batch_set_v1 falls back to `batch_set` semantics.
- The deployment-wide retention default comes from
  `INFERA_KVD_RETENTION_DEFAULT` (`none`/`short`/`long`).
  Operators running a kvd dedicated to long-retention KV (e.g. a
  single tenant with stable system prompts) set this to `long`.
- Stats hookup into the server's Prometheus surface — the daemon
  exposes its own `Stats` op; we'll wire it in Phase 4.
- Zero-copy GPU→host transfers. We round-trip bytes through Python.
  Acceptable for skeleton; replace with shared memory in Phase 4.

Sync↔async bridge:
SGLang calls `HiCacheStorage.get / set / ...` from its
`CacheController` worker thread, which is **synchronous** Python.
Our `KvdClient` is async. We spawn a single-thread event loop in
the background on adapter init, and dispatch from the sync methods
via `asyncio.run_coroutine_threadsafe(...).result()`. Same pattern
the LMCache vLLM adapter uses; well-trodden.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from infera.kvd.client import KvdClient, KvdConnectionError, KvdProtocolError
from infera.kvd.ssd import (
    _composite_hash,
    _encode_composite,
    _filename_for_composite,
)
from infera.kvd.wire import RETENTION_LONG, RETENTION_SHORT

# main removed the file-tier response markers (TIER_*) from the wire when the
# file-tier RPCs were dropped; tolerate their absence with sentinels, exactly
# as infera/engine/vllm/kvd_connector.py does. `getattr(resp, "tier",
# TIER_MISS)` then yields the sentinel and the tier dispatch degrades to the
# default path (RAM/file resolved by the daemon, not by a wire marker).
try:
    from infera.kvd.wire import TIER_FILE, TIER_MISS, TIER_RAM
except ImportError:
    TIER_FILE, TIER_MISS, TIER_RAM = object(), object(), object()  # sentinels

if TYPE_CHECKING:
    import torch  # type-only — actual torch import is lazy inside methods

# Lazy: SGLang and torch both deferred so this module imports cleanly
# on a router-only host or in unit tests. When the engine subprocess
# is actually running, both packages ARE available at import time.
try:
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig

    _SGLANG_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only without sglang installed

    class HiCacheStorage:  # type: ignore[no-redef]
        """Stand-in so this module imports without sglang. Real type
        kicks in when sglang is present at engine runtime."""

    class HiCacheStorageConfig:  # type: ignore[no-redef]
        model_name: str | None = None
        tp_rank: int = 0
        tp_size: int = 1
        pp_rank: int = 0
        pp_size: int = 1

    _SGLANG_AVAILABLE = False


def _torch():
    """Lazy torch handle. Raises a clean error if called when torch
    isn't installed (i.e. on a non-engine host). Adapter construction
    works without torch — only the tensor-touching methods need it."""
    import torch as _t

    return _t


logger = logging.getLogger(__name__)


_DEFAULT_SOCKET_PATH = "/var/run/infera-kvd.sock"


# ----------------------------------------------------------------------
# Per-request retention override (PR #9 review fix P0-3/4)
# ----------------------------------------------------------------------
#
# The SGLang HiCacheStorage ABC's `set`/`batch_set` don't carry per-
# request retention. Without an in-band channel, every adapter write
# inherits the deployment-wide default (`INFERA_KVD_RETENTION_DEFAULT`
# env or constructor `retention_default`). That conflicts with the
# Anthropic-style `cache_control: {"ttl": "1h"}` promise on chat APIs.
#
# This ContextVar provides a per-request override that any request-
# handler shim (router-side or upstream patch) can populate. Adapter
# `set` and `batch_set` consult it before falling back to the default.
#
# ContextVar (vs threading.local) is right because SGLang's async path
# may suspend across awaits between the hook setting and the cache
# spill landing in `set()`.
_request_retention_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "infera_kvd_request_retention", default=None
)


def set_request_retention_hint(retention: str | None) -> contextvars.Token | None:
    """Set the per-request retention override (or clear with None).

    Returns the ContextVar token so the caller can reset it after the
    request completes (`_request_retention_override.reset(token)`).
    Invalid values are logged at WARN and ignored.
    """
    if retention is not None and retention not in ("none", "short", "long"):
        logger.warning(
            "set_request_retention_hint: ignoring invalid retention=%r "
            "(must be one of none/short/long)",
            retention,
        )
        return None
    return _request_retention_override.set(retention)


def _resolve_retention(default: str) -> str:
    """Resolve the retention to use for this `set()` call. ContextVar
    override wins; default is the constructor/env value."""
    override = _request_retention_override.get()
    if override is not None:
        return override
    return default


class InferaKvdBackend(HiCacheStorage):  # type: ignore[misc]
    """Drop-in `HiCacheStorage` impl that talks to a infera-kvd daemon.

    Constructed by SGLang's `StorageBackendFactory` when the worker is
    started with `--hicache-storage-backend infera-kvd`. Configuration
    flows through SGLang's `extra_config` dict (the backend factory
    accepts arbitrary key-values when the backend is registered as
    'dynamic'; for 'infera-kvd' we read env vars and the daemon
    socket from a known location).

    Public methods match SGLang's `HiCacheStorage` ABC:
      get, batch_get, set, batch_set, exists, batch_exists, clear,
      get_stats.

    Per-instance state is **one persistent connection** to kvd plus
    one background event loop thread. The connection is shared across
    SGLang's worker threads via the loop's executor — fine because
    `KvdClient` itself holds a write lock that serializes the wire
    frames.
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig | None = None,
        mem_pool_host: Any = None,
        *,
        socket_path: str | None = None,
        retention_default: str | None = None,
        client_id: str | None = None,
        gpu_direct: bool | None = None,
        hipfile_roots: dict[str, str] | None = None,
    ) -> None:
        """Args:
        storage_config: SGLang's standard config object; we read
          ``model_name`` and TP/PP rank metadata so multi-rank
          workers don't collide on the same kvd namespace.
        mem_pool_host: ignored — SGLang passes the registered
          host KV pool here but we don't need it (kvd owns its
          own bytes).
        socket_path: kvd UDS path. Defaults to
          ``INFERA_KVD_SOCKET`` env var, then ``/var/run/infera-kvd.sock``.
        retention_default: retention to use for SETs when the
          incoming call carries no per-request retention hint
          (most calls today — see "SGLang upstream gap" below).
          Resolution order:
            1. constructor arg (this parameter, if given)
            2. ``INFERA_KVD_RETENTION_DEFAULT`` env var
               ∈ {``none``, ``short``, ``long``}
            3. fallback ``long`` (the L3 tier's whole purpose is a
               durable, cross-restart / cross-engine content-addressed
               pool; ``long`` is what actually write-throughs blocks to
               the on-disk long region. With ``short``, evicted blocks
               are DROPPED — no spillover region is configured — so the
               L3 disk tier silently never populates. fsync stays OFF by
               default, so the write is lazy page-cache writeback, not a
               per-SET fsync; see ``LongStorageRegion``.)
          Set the env var to ``short`` only for an ephemeral, RAM-only
          deployment that intentionally has no durable L3.

        ## SGLang upstream gap

        SGLang's ``HiCacheStorage.set / batch_set`` SPI doesn't carry
        per-request retention/priority. The ``batch_set_v1`` SPI
        adds an ``extra_info: HiCacheStorageExtraInfo`` parameter
        which has a generic ``extra_info: dict`` slot, but upstream
        ``cache_controller._generic_page_set`` only populates
        ``prefix_keys`` — request priority dies at the block
        aggregation boundary.

        We implement ``batch_set_v1`` so a small upstream patch
        (``extra_info.extra_info["infera_retention"] = req.priority``)
        will Just Work without further changes here. Until then,
        per-deployment retention via env var is the practical
        workaround.
        """
        super().__init__()
        if storage_config is None:
            storage_config = HiCacheStorageConfig()  # type: ignore[call-arg]
        self._storage_config = storage_config
        self._socket_path = (
            socket_path or os.environ.get("INFERA_KVD_SOCKET") or _DEFAULT_SOCKET_PATH
        )
        env_retention = os.environ.get("INFERA_KVD_RETENTION_DEFAULT")
        retention_source = (
            "constructor"
            if retention_default is not None
            else ("env" if env_retention is not None else "fallback")
        )
        self._retention_default = retention_default or env_retention or RETENTION_LONG
        if self._retention_default not in ("none", "short", "long"):
            raise ValueError(
                f"retention_default must be one of (none, short, long); "
                f"got {self._retention_default!r}"
            )
        # Log at INFO so operators see the asymmetry between vLLM
        # (per-request via kv_transfer_params) and SGLang (deployment-
        # wide via env, until upstream wires extra_info). Warn when the
        # resolved default is ``short``: it disables the on-disk L3 tier
        # (evicted blocks are dropped, not spilled), which is a silent
        # surprise for anyone expecting a durable/shared pool. See PR #9
        # review fix P0-3/4.
        log_level = logging.WARNING if self._retention_default == RETENTION_SHORT else logging.INFO
        logger.log(
            log_level,
            "InferaKvdBackend: kvd retention default = %s (source: %s). "
            "Per-request override via set_request_retention_hint() or "
            "batch_set_v1 extra_info[infera_retention]; both fall back "
            "to this value.",
            self._retention_default,
            retention_source,
        )
        self._model = (storage_config.model_name or "").strip()
        self._compat_key = self._derive_compat_key(storage_config)
        self._client_id = client_id or f"sglang-tp{storage_config.tp_rank}-pid{os.getpid()}"

        # Background loop for async kvd ops.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._client: KvdClient | None = None
        self._closed = False

        # ----- GPU-direct (hipFile) state -----
        # cudaHostUnregister callbacks indexed by pool name. Populated when
        # register_mem_host_pool_v2 succeeds at registering the pool's
        # host buffer; drained in close() in reverse order vs
        # _hipfile_pool_state's RegisteredBuffer.__exit__.
        self._hipfile_host_unregister_cbs: dict[Any, Any] = {}

        # Per-pool state: name -> (RegisteredBuffer, registered_base, page_stride_bytes, prefix)
        # Populated when register_mem_host_pool_v2 fires and gpu_direct is on.
        #
        # `prefix` is the number of bytes between the page-aligned
        # `registered_base` we pass to hipFile and the actual pool tensor
        # base_ptr that SGLang owns. hipFileBufRegister requires a 4 KiB
        # page-aligned ptr; torch tensors typically land on a 64-byte
        # boundary, so we round DOWN to the previous page and store the
        # delta. read/write paths add `prefix + page_offset * page_stride`
        # as the dest/src offset within the registered region.
        self._hipfile_pool_state: dict[Any, tuple[Any, int, int, int]] = {}
        self._gpu_direct = self._resolve_gpu_direct_flag(gpu_direct)
        self._hipfile_roots = self._resolve_hipfile_roots(hipfile_roots)
        # WARN-once latches for hipfile fallback paths — without these the
        # adapter would log a line per failed page on a sticky failure.
        self._hipfile_read_warned: set[Any] = set()
        self._hipfile_write_warned: set[Any] = set()

        self._start_background_loop()
        self._connect_or_raise()

        # Probe hipFile shim only AFTER the kvd connection is up — failure
        # here disables gpu_direct silently (we keep the UDS path) rather
        # than killing the whole adapter: opt-in, never crash on missing binding.
        if self._gpu_direct:
            self._gpu_direct = self._probe_hipfile_or_disable()

    # ------------------------------------------------------------------
    # SGLang HiCacheStorage ABC implementation
    # ------------------------------------------------------------------

    def get(
        self,
        key: str,
        target_location: Any = None,
        target_sizes: Any = None,
    ) -> torch.Tensor | None:
        """Read one block. Writes into `target_location` (a torch.Tensor
        pre-allocated by SGLang's host pool) on hit; returns None on miss.
        """
        if target_location is None:
            # Without a target, SGLang's contract is ambiguous; the file
            # backend allocates one internally. We require it for now to
            # avoid bytes-to-tensor guesswork.
            raise ValueError("InferaKvdBackend.get requires target_location")
        kvd_key = _encode_key(key)
        value = self._run_async(
            self._client.get(kvd_key, model=self._model, compat_key=self._compat_key)
        )
        if value is None:
            return None
        return _bytes_into_tensor(value, target_location)

    def batch_get(
        self,
        keys: list[str],
        target_locations: Any = None,
        target_sizes: Any = None,
    ) -> list[torch.Tensor | None]:
        """Sequential dispatch — SGLang ABC's v1 `batch_get` is just an
        ergonomic wrapper. Real batching happens in v2 (PoolTransfer)
        which we don't implement yet. Pipelining N gets across one
        connection IS the throughput path; we'll add it when measured."""
        if target_locations is None or len(target_locations) != len(keys):
            raise ValueError("InferaKvdBackend.batch_get requires aligned target_locations")
        return [self.get(k, t) for k, t in zip(keys, target_locations, strict=True)]

    def set(
        self,
        key: str,
        value: Any = None,
        target_location: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        """Store one block. `value` is a torch.Tensor (already on host).
        Returns True on success, False if kvd rejected (e.g. would
        displace higher-priority entry).
        """
        if value is None:
            raise ValueError("InferaKvdBackend.set requires value")
        kvd_key = _encode_key(key)
        payload = _tensor_to_bytes(value)
        # ContextVar override → constructor default. Set the override
        # via `set_request_retention_hint(...)` from a request shim
        # to propagate per-request `cache_control` into kvd writes.
        retention = _resolve_retention(self._retention_default)
        accepted, reason = self._run_async(
            self._client.set(
                kvd_key,
                payload,
                retention=retention,
                model=self._model,
                compat_key=self._compat_key,
            )
        )
        if not accepted:
            logger.debug(
                "kvd refused set key=%s reason=%s retention=%s",
                key,
                reason,
                retention,
            )
        return accepted

    def batch_set(
        self,
        keys: list[str],
        values: Any = None,
        target_locations: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        """All-or-something semantics (matches SGLang's v1): return True
        iff every set succeeded. Sequential for v1; future v2 with
        PoolTransfer can pipeline."""
        if values is None or len(values) != len(keys):
            raise ValueError("InferaKvdBackend.batch_set requires aligned values")
        success = True
        for k, v in zip(keys, values, strict=True):
            if not self.set(k, v):
                success = False
        return success

    # ------------------------------------------------------------------
    # v2 SPI — pool-aware batched IO used by GLM-style DSA/hybrid controllers
    # ------------------------------------------------------------------
    #
    # `batch_set_v2`/`batch_get_v2` are the entrypoints SGLang's hybrid-
    # cache controller calls (see `hybrid_cache_controller._page_backup`
    # and `_page_transfer`). They take a list of `PoolTransfer` records
    # — one per pool (KV, INDEXER, MAMBA, …) — and return a
    # `{pool_name: [success, …]}` dict. Each transfer carries:
    #   * `name`          : `PoolName` enum identifying the registered host pool
    #   * `keys`          : per-page storage keys
    #   * `host_indices`  : torch tensor mapping page → host pool slot
    #                       (length = `len(keys) * host_pool.page_size`)
    #
    # The base ABC's `register_mem_host_pool_v2` populates
    # `self.registered_pools`; we look up the host pool by name to read
    # /write the actual bytes per page via the standard
    # `get_data_page`/`set_from_flat_data_page` interface — same pattern
    # used by `HiCacheFile._batch_io_v2`.
    #
    # Without these methods, GLM's prefetch_thread / backup_thread call
    # the base `raise NotImplementedError`, crash, and the engine silently
    # stops touching kvd (sets/gets stay 0 under any load). See
    # `project_sglang_kvd_backend_not_in_spawn` memory + the HANDOFF.

    def batch_exists_v2(
        self,
        keys: list[str],
        pool_transfers: Any = None,
        extra_info: Any = None,
    ) -> Any:
        """Per-pool prefix-existence check used by the hybrid controller.

        Returns a `PoolTransferResult(kv_hit_pages, extra_pool_hit_pages)`
        where `kv_hit_pages` is the longest contiguous KV prefix present
        in kvd, intersected with each extra pool's hit policy
        (ALL_PAGES or TRAILING_PAGES). Mirrors `HiCacheFile.batch_exists_v2`.

        Without this, GLM's prefetch_thread hits the base ABC's
        `raise NotImplementedError`, crashes silently, and no v2 reads
        ever fire — exactly the symptom HANDOFF Bug #2 describes.
        """
        # PoolName/PoolHitPolicy/PoolTransferResult only exist when sglang
        # is installed (engine subprocess), not in router-only test envs.
        try:
            from sglang.srt.mem_cache.hicache_storage import (
                PoolHitPolicy,
                PoolName,
                PoolTransferResult,
            )
        except ImportError:
            return None

        if not keys:
            return PoolTransferResult.empty()

        # KV presence over the full key list.
        kv_present = self._exists_for_pool(None, keys)
        kv_pages = next((i for i, ok in enumerate(kv_present) if not ok), len(kv_present))

        hit_count: dict = {}
        if kv_pages:
            hit_count[PoolName.KV] = kv_pages
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = transfer.name
            policy = getattr(transfer, "hit_policy", PoolHitPolicy.ALL_PAGES)
            if policy == PoolHitPolicy.ALL_PAGES:
                # Need every page in [0, kv_pages) present for this pool.
                pool_present = self._exists_for_pool(name, keys[:kv_pages])
                boundary = next(
                    (i for i, ok in enumerate(pool_present) if not ok), len(pool_present)
                )
            else:  # TRAILING_PAGES
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                pool_present = self._exists_for_pool(name, keys[:kv_pages])
                boundary = 0
                # largest prefix_len such that the trailing N pages all exist
                for prefix_len in range(kv_pages, 0, -1):
                    lo = max(0, prefix_len - trailing)
                    if all(pool_present[i] for i in range(lo, prefix_len)):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _exists_for_pool(self, pool_name: Any, keys: list[str]) -> list[bool]:
        """Bulk-check kvd presence for one pool's per-page keys.

        Suffixes each key per ``_pool_storage_key`` so KV and INDEXER
        pages addressed at the same page hash don't alias.
        """
        if not keys:
            return []
        encoded = [_encode_key(self._pool_storage_key(pool_name, k)) for k in keys]
        try:
            present = self._run_async(
                self._client.exists(encoded, model=self._model, compat_key=self._compat_key)
            )
        except (KvdConnectionError, KvdProtocolError):
            logger.exception("kvd v2 exists failed for pool=%s (%d keys)", pool_name, len(keys))
            return [False] * len(keys)
        return [bool(x) for x in present]

    def batch_set_v2(
        self,
        transfers: Any,
        extra_info: Any = None,
    ) -> dict:
        """Write per-pool host pages to kvd. Returns per-pool success lists."""
        return self._batch_io_v2(transfers, write=True, extra_info=extra_info)

    def batch_get_v2(
        self,
        transfers: Any,
        extra_info: Any = None,
    ) -> dict:
        """Read per-pool pages from kvd into host pool slots. Returns per-pool success lists."""
        return self._batch_io_v2(transfers, write=False, extra_info=extra_info)

    def _batch_io_v2(self, transfers: Any, *, write: bool, extra_info: Any) -> dict:
        registered = getattr(self, "registered_pools", None) or {}
        results: dict = {}
        if not transfers:
            return results
        retention = (
            self._resolve_retention_from_extra_info(extra_info)
            if write
            else self._retention_default
        )
        for transfer in transfers:
            name = transfer.name
            keys = list(transfer.keys or [])
            host_pool = registered.get(name)
            if host_pool is None:
                # Engine never registered this pool with us — treat as
                # all-miss on read and all-fail on write. Log loudly because
                # this means our wiring missed a pool the controller knows
                # about (would silently halve the cache otherwise).
                logger.error(
                    "kvd v2 %s: pool %r not in registered_pools=%s",
                    "set" if write else "get",
                    name,
                    list(registered.keys()),
                )
                results[name] = [False] * len(keys)
                continue
            page_size = int(getattr(host_pool, "page_size", 1) or 1)
            host_indices = transfer.host_indices
            expected = len(keys) * page_size
            host_count = int(host_indices.numel()) if host_indices is not None else 0
            if host_indices is None or host_count != expected:
                logger.error(
                    "kvd v2 %s: indices/keys mismatch for pool %r (expected %d host slots, got %d)",
                    "set" if write else "get",
                    name,
                    expected,
                    host_count,
                )
                results[name] = [False] * len(keys)
                continue
            per_key: list[bool] = []
            for i, key in enumerate(keys):
                page_offset = int(host_indices[i * page_size].item())
                if write:
                    per_key.append(
                        self._v2_write_page(name, key, host_pool, page_offset, retention)
                    )
                else:
                    per_key.append(self._v2_read_page(name, key, host_pool, page_offset))
            results[name] = per_key
        return results

    def _pool_storage_key(self, pool_name: Any, key: str) -> str:
        """Mirror HiCacheFile's `_log_key`: the KV pool uses the bare key;
        other pools (INDEXER, MAMBA, …) get a `.{pool}` suffix so writes
        for different pools at the same logical position don't collide."""
        if pool_name is None:
            return key
        name = pool_name.value if hasattr(pool_name, "value") else str(pool_name)
        if name in ("", "kv", "__default__"):
            return key
        return f"{key}.{name}"

    def _v2_write_page(
        self,
        pool_name: Any,
        key: str,
        host_pool: Any,
        page_offset: int,
        retention: str,
    ) -> bool:
        storage_key = self._pool_storage_key(pool_name, key)

        # GPU-direct write path: if a retention root is configured AND
        # this pool has a hipfile RegisteredBuffer, write the page bytes
        # directly to disk via hipFileWrite (no host memcpy), then tell
        # kvd via RegisterFileEntry. On any failure (write OR
        # accept=False), fall through to the legacy UDS Set so the
        # write is never lost.
        pool_state = self._hipfile_pool_state.get(pool_name) if self._gpu_direct else None
        if pool_state is not None and retention in ("short", "long"):
            handled, ok = self._try_hipfile_write(
                pool_name, storage_key, page_offset, retention, pool_state
            )
            if handled:
                return ok

        data_page = host_pool.get_data_page(page_offset, flat=True)
        payload = _tensor_to_bytes(data_page)
        accepted, reason = self._run_async(
            self._client.set(
                _encode_key(storage_key),
                payload,
                retention=retention,
                model=self._model,
                compat_key=self._compat_key,
            )
        )
        if not accepted:
            logger.debug(
                "kvd v2 set refused pool=%s key=%s reason=%s retention=%s",
                pool_name,
                key,
                reason,
                retention,
            )
        return accepted

    def _try_hipfile_write(
        self,
        pool_name: Any,
        storage_key: str,
        page_offset: int,
        retention: str,
        pool_state: tuple[Any, int, int, int],
    ) -> tuple[bool, bool]:
        """Attempt a GPU-direct write via hipFile + RegisterFileEntry.

        Returns ``(handled, ok)`` — same shape as ``_try_hipfile_read``.
        ``handled=False`` means caller falls through to UDS Set so the
        write isn't lost.
        """
        path = self._hipfile_path_for(retention, storage_key)
        if path is None:
            # No root configured for this retention — UDS path handles it.
            return False, False
        _reg, registered_base, page_stride, prefix = pool_state
        size = int(page_stride)
        # Offset into the registered region: skip the alignment prefix,
        # then jump to the requested page.
        src_offset = int(prefix) + int(page_offset) * int(page_stride)

        # mkdir -p the shard subdir; EEXIST is fine (parallel writers).
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            if pool_name not in self._hipfile_write_warned:
                self._hipfile_write_warned.add(pool_name)
                logger.warning(
                    "hipFile shard mkdir failed for pool=%s path=%s (%s) "
                    "— falling back to UDS for this and future pages on "
                    "this pool (warn-once)",
                    pool_name,
                    path.parent,
                    exc,
                )
            return False, False

        try:
            from infera.engine.sglang.hipfile_shim import HipFile

            with HipFile(str(path), "w") as hf:
                hf.write(registered_base, size, 0, src_offset)
        except Exception as exc:
            if pool_name not in self._hipfile_write_warned:
                self._hipfile_write_warned.add(pool_name)
                logger.warning(
                    "hipFile write failed for pool=%s key=%s path=%s "
                    "(%s) — falling back to UDS for this and future pages "
                    "on this pool (warn-once)",
                    pool_name,
                    storage_key,
                    path,
                    exc,
                )
            return False, False

        try:
            accepted, reason = self._run_async(
                self._client.register_file_entry(
                    _encode_key(storage_key),
                    path=str(path),
                    file_offset=0,
                    size=size,
                    version=0,
                    retention=retention,
                    model=self._model,
                    compat_key=self._compat_key,
                )
            )
        except (KvdConnectionError, KvdProtocolError) as exc:
            logger.warning(
                "kvd register_file_entry raised for pool=%s key=%s (%s) "
                "— falling back to UDS Set so the write is not lost",
                pool_name,
                storage_key,
                exc,
            )
            return False, False
        if not accepted:
            logger.debug(
                "kvd register_file_entry refused pool=%s key=%s reason=%s "
                "retention=%s — falling back to UDS Set",
                pool_name,
                storage_key,
                reason,
                retention,
            )
            return False, False
        return True, True

    def _v2_read_page(
        self,
        pool_name: Any,
        key: str,
        host_pool: Any,
        page_offset: int,
    ) -> bool:
        storage_key = self._pool_storage_key(pool_name, key)

        # GPU-direct read path: ask kvd where the bytes live; if
        # tier=="file", read directly via hipFile into the registered
        # pool buffer (zero bytes hop). tier=="ram" falls through to the
        # legacy UDS Get; tier=="miss" returns False immediately.
        pool_state = self._hipfile_pool_state.get(pool_name) if self._gpu_direct else None
        if pool_state is not None:
            handled, ok = self._try_hipfile_read(pool_name, storage_key, page_offset, pool_state)
            if handled:
                return ok

        target = host_pool.get_dummy_flat_data_page()
        value = self._run_async(
            self._client.get(
                _encode_key(storage_key),
                model=self._model,
                compat_key=self._compat_key,
            )
        )
        if value is None:
            return False
        filled = _bytes_into_tensor(value, target)
        if filled is None:
            return False
        host_pool.set_from_flat_data_page(page_offset, filled)
        return True

    def _try_hipfile_read(
        self,
        pool_name: Any,
        storage_key: str,
        page_offset: int,
        pool_state: tuple[Any, int, int, int],
    ) -> tuple[bool, bool]:
        """Attempt a GPU-direct read via hipFile.

        Returns ``(handled, ok)``. ``handled=False`` means the caller
        should fall through to the UDS Get path (typical on tier=="ram"
        or hipFile error). ``handled=True`` means we owned the outcome
        — either the page was filled (``ok=True``) or kvd reported a
        miss across all tiers (``ok=False``).
        """
        try:
            resp = self._run_async(
                self._client.lookup_tier(
                    _encode_key(storage_key),
                    model=self._model,
                    compat_key=self._compat_key,
                )
            )
        except (KvdConnectionError, KvdProtocolError) as exc:
            logger.debug(
                "kvd lookup_tier raised (%s) — falling back to UDS Get for pool=%s key=%s",
                exc,
                pool_name,
                storage_key,
            )
            return False, False

        tier = getattr(resp, "tier", TIER_MISS)
        if tier == TIER_RAM:
            return False, False  # legacy UDS path handles it
        if tier == TIER_MISS:
            return True, False  # no bytes anywhere — return miss

        if tier != TIER_FILE:
            logger.debug(
                "kvd lookup_tier returned unknown tier=%r — falling back to UDS",
                tier,
            )
            return False, False

        path = getattr(resp, "path", None)
        size = int(getattr(resp, "size", 0) or 0)
        file_offset = int(getattr(resp, "file_offset", 0) or 0)
        if not path or size <= 0:
            logger.warning(
                "kvd lookup_tier tier=file but malformed payload "
                "(path=%r size=%d) — falling back to UDS",
                path,
                size,
            )
            return False, False

        _reg, registered_base, page_stride, prefix = pool_state
        dest_offset = int(prefix) + int(page_offset) * int(page_stride)
        try:
            from infera.engine.sglang.hipfile_shim import HipFile

            with HipFile(path, "r") as hf:
                hf.read(registered_base, size, file_offset, dest_offset)
        except Exception as exc:
            if pool_name not in self._hipfile_read_warned:
                self._hipfile_read_warned.add(pool_name)
                logger.warning(
                    "hipFile read failed for pool=%s key=%s path=%s "
                    "(%s) — falling back to UDS for this and future pages "
                    "on this pool (warn-once)",
                    pool_name,
                    storage_key,
                    path,
                    exc,
                )
            return False, False
        return True, True

    def batch_set_v1(
        self,
        keys: list[str],
        host_indices: Any,
        extra_info: Any = None,
    ) -> list[bool]:
        """SGLang's `batch_set_v1` SPI — accepts the
        ``HiCacheStorageExtraInfo`` dict which carries
        per-request hints when upstream populates it.

        We honor ``extra_info.extra_info["infera_retention"]`` if
        present; otherwise fall back to ``self._retention_default``.
        See the class docstring for the upstream gap that keeps
        ``extra_info.extra_info`` from being populated today.

        ``host_indices`` is SGLang's torch tensor of host-pool block
        indices. We read each block's bytes via the registered host
        pool. For now we don't have a clean handle to the pool
        (mem_pool_host is ignored at construction), so we fall back
        to ``batch_set`` semantics by REJECTING this entry point if
        host_indices is the only data we have. Operators should pipe
        sglang's call through ``batch_set`` (the legacy SPI) until
        the v1 path is fully wired in a follow-up.
        """
        retention = self._resolve_retention_from_extra_info(extra_info)
        # Without mem_pool_host wired we can't pull bytes from
        # host_indices. Report all-fail so SGLang's controller falls
        # back to its legacy path (batch_set). Logging is verbose-
        # debug because a misconfigured controller would hammer it.
        logger.debug(
            "InferaKvdBackend.batch_set_v1 fallback path: "
            "retention=%s, %d keys, host_indices_shape=%s",
            retention,
            len(keys),
            getattr(host_indices, "shape", None),
        )
        return [False] * len(keys)

    def _resolve_retention_from_extra_info(self, extra_info: Any) -> str:
        """Read per-request retention from SGLang's
        ``HiCacheStorageExtraInfo.extra_info`` dict, with a three-step
        fallback chain:

          1. `extra_info.extra_info["infera_retention"]` — what
             upstream SGLang would carry once the in-band channel is
             wired (see class docstring §"SGLang upstream gap").
          2. ContextVar override set via
             `set_request_retention_hint(...)` — covers the today-state
             where operators bridge the gap with a request-handler shim.
          3. `self._retention_default` — deployment-wide fallback.
        """
        if extra_info is not None:
            inner = getattr(extra_info, "extra_info", None)
            if isinstance(inner, dict):
                candidate = inner.get("infera_retention")
                if isinstance(candidate, str) and candidate in ("none", "short", "long"):
                    return candidate
        return _resolve_retention(self._retention_default)

    def exists(self, key: str) -> bool:
        """Single-key membership check. Falls through to a batch op."""
        result = self._run_async(
            self._client.exists([_encode_key(key)], model=self._model, compat_key=self._compat_key)
        )
        return bool(result and result[0])

    def batch_exists(
        self,
        keys: list[str],
        extra_info: Any = None,
    ) -> int:
        """Override the default sequential-exists implementation with a
        single bulk wire op. Returns the count of consecutive existing
        keys from the start of the list — matches SGLang's longest-prefix
        semantics."""
        if not keys:
            return 0
        result = self._run_async(
            self._client.exists(
                [_encode_key(k) for k in keys],
                model=self._model,
                compat_key=self._compat_key,
            )
        )
        # Count leading True values; first False ends the run.
        for i, present in enumerate(result):
            if not present:
                return i
        return len(result)

    def clear(self) -> None:
        """Drop our entire (model, compat_key) namespace. Other models
        on the same kvd are untouched."""
        self._run_async(self._client.clear(model=self._model, compat_key=self._compat_key))

    def get_stats(self) -> dict | None:
        """Daemon-side counters. SGLang's hicache observability picks
        this up; we surface the same numbers to Prometheus on the server
        via `/v1/kv-stats`."""
        try:
            stats = self._run_async(self._client.stats())
        except (KvdConnectionError, KvdProtocolError):
            return None
        return {
            "entries": stats.entries,
            "host_bytes": stats.host_bytes,
            "hits_total": stats.hits_total,
            "misses_total": stats.misses_total,
            "evictions_total": stats.evictions_total,
        }

    def close(self) -> None:
        """Tear down the kvd connection and background loop. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Drop hipfile RegisteredBuffers first so the underlying device
        # mappings are released before the engine tears down its pool.
        if self._hipfile_pool_state:
            for name, (reg, _ptr, _stride, _prefix) in list(self._hipfile_pool_state.items()):
                try:
                    reg.__exit__(None, None, None)
                except Exception:  # pragma: no cover — defensive
                    logger.debug(
                        "hipfile RegisteredBuffer teardown raised for pool=%s; ignoring",
                        name,
                        exc_info=True,
                    )
            self._hipfile_pool_state.clear()
        # cudaHostUnregister AFTER the hipFile BufDeregister so we don't
        # yank HIP's view of the buffer while hipFile still tracks it.
        if self._hipfile_host_unregister_cbs:
            for name, cb in list(self._hipfile_host_unregister_cbs.items()):
                try:
                    cb()
                except Exception:  # pragma: no cover — defensive
                    logger.debug(
                        "cudaHostUnregister failed for pool=%s; ignoring",
                        name,
                        exc_info=True,
                    )
            self._hipfile_host_unregister_cbs.clear()
        if self._client is not None and self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._client.close(), self._loop).result(
                    timeout=5.0
                )
            except (Exception, TimeoutError):
                pass
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)

    def __del__(self) -> None:  # pragma: no cover — interpreter teardown timing
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # GPU-direct (hipFile) helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_gpu_direct_flag(explicit: bool | None) -> bool:
        """Resolution order: explicit kwarg → env INFERA_KVD_AIS
        (legacy INFERA_KVD_GPU_DIRECT still honored, deprecated) → False."""
        if explicit is not None:
            return bool(explicit)
        env = os.environ.get("INFERA_KVD_AIS")
        if env is None:
            env = os.environ.get("INFERA_KVD_GPU_DIRECT")
            if env is not None:
                logger.warning("INFERA_KVD_GPU_DIRECT is deprecated; use INFERA_KVD_AIS")
        env = (env or "").strip()
        return env in ("1", "true", "True", "yes", "on")

    @staticmethod
    def _resolve_hipfile_roots(explicit: dict[str, str] | None) -> dict[str, str]:
        """Resolution order: explicit kwarg → env INFERA_KVD_HIPFILE_ROOTS → {}.

        Env format: ``"short=/nvme/short,long=/mnt/long"`` — comma-
        separated ``retention=path`` pairs. Unknown retention keys are
        kept verbatim; the write path filters them later.
        """
        if explicit is not None:
            return dict(explicit)
        env = os.environ.get("INFERA_KVD_HIPFILE_ROOTS", "").strip()
        if not env:
            return {}
        roots: dict[str, str] = {}
        for pair in env.split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            key, _, value = pair.partition("=")
            key = key.strip()
            value = value.strip()
            if key and value:
                roots[key] = value
        return roots

    def _probe_hipfile_or_disable(self) -> bool:
        """Probe the hipfile shim's is_available(). On False, log a WARN
        with the failure mode and turn gpu_direct off for THIS adapter
        instance. The UDS path remains the active code path.

        We catch ImportError defensively too — the shim itself is
        committed and importable, but in pathological environments
        (broken install of `infera` partial) we'd rather degrade than
        crash the engine.
        """
        try:
            from infera.engine.sglang import hipfile_shim
        except ImportError as exc:  # pragma: no cover — shim ships with package
            logger.warning(
                "kvd adapter: gpu_direct requested but hipfile_shim import "
                "failed (%s); disabling for this instance, falling back to UDS",
                exc,
            )
            return False
        try:
            available = hipfile_shim.is_available()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "kvd adapter: gpu_direct probe raised (%s); disabling for "
                "this instance, falling back to UDS",
                exc,
            )
            return False
        if not available:
            logger.warning(
                "kvd adapter: gpu_direct requested but hipfile_shim.is_available() "
                "returned False (see prior WARN from hipfile_shim for reason); "
                "disabling for this instance, falling back to UDS"
            )
            return False
        logger.info(
            "kvd adapter: gpu_direct enabled (roots=%s); per-pool RegisteredBuffer "
            "will attach on register_mem_host_pool_v2()",
            self._hipfile_roots or "<unset; writes will fall back to UDS>",
        )
        return True

    @staticmethod
    def _get_pool_buffer_info(host_pool: Any) -> tuple[int, int, int] | None:
        """Inspect ``host_pool`` and return ``(base_ptr, total_bytes,
        page_stride_bytes)`` or ``None`` for unknown pool shapes.

        Recognized pool shapes (matches GLM-5.1 A3 research):

        - ``MLATokenToKVPoolHost``: detected by ``hasattr(host_pool,
          "kv_buffer")``. The buffer is one pinned CPU torch tensor;
          ``page_stride_bytes = page_size * kv_cache_dim * dtype.itemsize``.
        - ``NSAIndexerPoolHost``: detected by
          ``hasattr(host_pool, "index_k_with_scale_buffer")``. Same
          shape but the per-page stride is
          ``layer_num * indexer_page_stride_size``.

        Detection is duck-typed — we never import sglang types here, so
        the adapter loads cleanly in unit-test envs without sglang.

        Returns ``None`` for any pool whose shape we don't recognize;
        the caller (``register_mem_host_pool_v2`` override) then skips
        the RegisteredBuffer for that pool, and the v2 read/write paths
        fall back to UDS for transfers against it.
        """
        # MLATokenToKVPoolHost shape
        if hasattr(host_pool, "kv_buffer"):
            buf = host_pool.kv_buffer
            try:
                base_ptr = int(buf.data_ptr())
                total_bytes = int(buf.numel()) * int(buf.element_size())
                page_size = int(getattr(host_pool, "page_size", 1) or 1)
                kv_cache_dim = int(getattr(host_pool, "kv_cache_dim", 0) or 0)
                dtype = getattr(host_pool, "dtype", None) or buf.dtype
                itemsize = int(dtype.itemsize)
                page_stride_bytes = page_size * kv_cache_dim * itemsize
            except (AttributeError, TypeError, ValueError):
                return None
            if page_stride_bytes <= 0:
                return None
            return base_ptr, total_bytes, page_stride_bytes

        # NSAIndexerPoolHost shape
        if hasattr(host_pool, "index_k_with_scale_buffer"):
            buf = host_pool.index_k_with_scale_buffer
            try:
                base_ptr = int(buf.data_ptr())
                total_bytes = int(buf.numel()) * int(buf.element_size())
                layer_num = int(getattr(host_pool, "layer_num", 0) or 0)
                indexer_stride = int(getattr(host_pool, "indexer_page_stride_size", 0) or 0)
                page_stride_bytes = layer_num * indexer_stride
            except (AttributeError, TypeError, ValueError):
                return None
            if page_stride_bytes <= 0:
                return None
            return base_ptr, total_bytes, page_stride_bytes

        return None

    def register_mem_host_pool_v2(self, host_pool: Any, name: Any) -> None:
        """Override the base ABC's pool registration so we also attach a
        hipFile ``RegisteredBuffer`` over the pool's pinned buffer when
        gpu_direct is on. Falls back to base behavior for pools we don't
        recognize (``_get_pool_buffer_info`` returns None) — those still
        get registered as UDS-path pools, just not hipFile-pinned.
        """
        # Always defer to the base method (when sglang is present) so
        # ``self.registered_pools[name] = host_pool`` happens. On test
        # envs without sglang our stub base lacks this method; populate
        # the dict ourselves so the v2 SPI keeps working.
        super_method = getattr(super(), "register_mem_host_pool_v2", None)
        if callable(super_method):
            super_method(host_pool, name)
        else:
            registered = getattr(self, "registered_pools", None)
            if registered is None:
                registered = {}
                self.registered_pools = registered  # type: ignore[attr-defined]
            registered[name] = host_pool

        if not self._gpu_direct:
            return
        info = self._get_pool_buffer_info(host_pool)
        if info is None:
            logger.info(
                "kvd adapter: gpu_direct pool %r has unrecognized shape "
                "(no kv_buffer / index_k_with_scale_buffer attr); transfers "
                "for this pool will use the UDS path",
                name,
            )
            return
        base_ptr, total_bytes, page_stride = info
        # hipFileBufRegister requires page-aligned base. torch tensors
        # typically land on a 64-byte boundary; round DOWN to the page
        # and extend the size to cover the prefix. The aligned-down
        # bytes belong to torch's allocator arena (page-sized chunks),
        # so the address is mapped — we just won't write through it.
        _PAGE = 4096
        prefix = base_ptr & (_PAGE - 1)
        registered_base = base_ptr - prefix
        registered_size = (total_bytes + prefix + _PAGE - 1) & ~(_PAGE - 1)
        # SGLang's MLATokenToKVPoolHost.kv_buffer is torch.empty(pin_memory=True)
        # — that maps to libc malloc + mlock, NOT to a HIP-runtime-registered
        # pinned host region. hipFile's BufRegister calls hipPointerGetAttributes
        # and rejects with hipFileHipMemoryTypeInvalid (5013) on those tensors.
        # We explicitly cudaHostRegister the page-aligned range here so the HIP
        # runtime knows about it, THEN hand it to hipFile. Unregister on close.
        # Flags: 1=Portable + 2=Mapped → 3 (visible to all CUDA contexts +
        # mapped into device address space — matches what cuFile docs require).
        import ctypes as _ct

        _cudart = None
        try:
            import torch as _t

            _cudart = _t.cuda.cudart()
        except Exception as exc:
            # Test env without GPU — proceed without cudaHostRegister and let
            # the (mocked) hipFile shim decide. In prod torch.cuda is always
            # importable; if it's not, our register-buffer call below will
            # still surface a clear error from hipFile itself.
            logger.debug(
                "kvd adapter: torch.cuda.cudart() unavailable (%s) — skipping "
                "cudaHostRegister for pool %r (likely a non-GPU test env)",
                exc,
                name,
            )
        if _cudart is not None:
            try:
                # torch's pybind11 binding expects plain int / SupportsInt,
                # NOT ctypes. flags=3 → Portable(1)|Mapped(2).
                rc_reg = int(
                    _cudart.cudaHostRegister(
                        int(registered_base),
                        int(registered_size),
                        3,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "kvd adapter: cudaHostRegister raised for pool %r "
                    "(aligned_base=0x%x aligned_size=%d): %s — UDS path will be used",
                    name,
                    registered_base,
                    registered_size,
                    exc,
                )
                return
            if rc_reg != 0:
                logger.warning(
                    "kvd adapter: cudaHostRegister failed (rc=%d) for pool %r "
                    "(aligned_base=0x%x aligned_size=%d) — UDS path will be used",
                    rc_reg,
                    name,
                    registered_base,
                    registered_size,
                )
                return
        try:
            from infera.engine.sglang import hipfile_shim

            reg = hipfile_shim.RegisteredBuffer(registered_base, registered_size)
            reg.__enter__()
        except Exception as exc:
            # Roll back the cudaHostRegister we just did.
            try:
                _cudart.cudaHostUnregister(_ct.c_void_p(registered_base))
            except Exception:
                pass
            logger.warning(
                "kvd adapter: RegisteredBuffer attach failed for pool %r "
                "(base_ptr=0x%x size=%d aligned_base=0x%x aligned_size=%d): "
                "%s — UDS path will be used for this pool",
                name,
                base_ptr,
                total_bytes,
                registered_base,
                registered_size,
                exc,
            )
            return
        # Stash the cudaHostUnregister callback alongside the RegisteredBuffer
        # so close() can tear everything down in reverse order. Only stash if
        # we actually called Register (test env may have skipped it).
        if _cudart is not None:
            self._hipfile_host_unregister_cbs[name] = lambda b=registered_base: (
                _cudart.cudaHostUnregister(int(b))
            )
        self._hipfile_pool_state[name] = (reg, registered_base, page_stride, prefix)
        logger.info(
            "kvd adapter: hipFile RegisteredBuffer attached for pool %r "
            "(base_ptr=0x%x size=%d aligned_base=0x%x aligned_size=%d "
            "page_stride=%d prefix=%d)",
            name,
            base_ptr,
            total_bytes,
            registered_base,
            registered_size,
            page_stride,
            prefix,
        )

    def _hipfile_path_for(self, retention: str, storage_key: str) -> Path | None:
        """Compute the on-disk hipFile path for ``(retention, storage_key)``
        from the configured ``hipfile_roots``. Returns ``None`` if the
        retention has no configured root — caller then falls back to UDS.

        Layout matches kvd's sharded SsdRegion shape (``ssd.py``
        §"On-disk layout"): ``{root}/{hash[:2]}/{hash[2:4]}/<fname>.kvcache``
        where ``fname = urlencode(composite)`` and
        ``composite = f"{model}|{compat_key}|{b64url(key)}"`` and
        ``hash = sha256(composite)``. This is the contract that lets a
        future kvd-side reader find the file by re-deriving the same
        path from the (model, compat_key, key) tuple alone — see
        ``test_engine_path_matches_kvd_sharded_layout``.
        """
        root_str = self._hipfile_roots.get(retention)
        if not root_str:
            return None
        composite = _encode_composite(self._model, self._compat_key, storage_key.encode("utf-8"))
        h = _composite_hash(composite)
        shard = Path(root_str) / h[:2] / h[2:4]
        fname = _filename_for_composite(composite)
        return shard / (fname + ".kvcache")

    def _derive_compat_key(self, config: HiCacheStorageConfig) -> str:
        """Distinct compat_keys for TP/PP variants of the same model
        prevent cross-rank pollution. We don't include the model_name
        (that's the top-level namespace) — just the rank topology.
        MLA models share KV across TP ranks, so we collapse those.
        """
        is_mla = bool(getattr(config, "is_mla_model", False))
        tp_rank = int(getattr(config, "tp_rank", 0))
        tp_size = int(getattr(config, "tp_size", 1))
        pp_rank = int(getattr(config, "pp_rank", 0))
        pp_size = int(getattr(config, "pp_size", 1))
        if is_mla:
            return f"pp{pp_rank}of{pp_size}"
        return f"tp{tp_rank}of{tp_size}_pp{pp_rank}of{pp_size}"

    def _start_background_loop(self) -> None:
        loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_forever()
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        thread = threading.Thread(target=_runner, name=f"kvd-loop-{self._client_id}", daemon=True)
        thread.start()
        self._loop = loop
        self._loop_thread = thread

    def _connect_or_raise(self) -> None:
        client = KvdClient(self._socket_path, client_id=self._client_id)
        try:
            self._run_async_with_client(client, client.connect())
        except Exception:
            # Tear down the loop before re-raising so we don't leak.
            self.close()
            raise
        self._client = client
        logger.info(
            "infera-kvd adapter connected to %s (model=%s, compat_key=%s)",
            self._socket_path,
            self._model or "<empty>",
            self._compat_key,
        )

    def _run_async(self, coro):
        if self._loop is None or self._client is None:
            raise RuntimeError("adapter is closed or not yet initialized")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _run_async_with_client(self, _client: KvdClient, coro):
        # Variant used during init when self._client isn't set yet.
        if self._loop is None:
            raise RuntimeError("background loop missing")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


# ----------------------------------------------------------------------
# Helpers (module-level so they can be unit-tested in isolation)
# ----------------------------------------------------------------------


def _encode_key(key: str) -> bytes:
    """SGLang uses string keys (with model+tp suffix). Kvd takes bytes.
    UTF-8 is fine for the SGLang key shape (printable ASCII + digits)."""
    return key.encode("utf-8")


def _tensor_to_bytes(value: torch.Tensor) -> bytes:
    """Tensor → raw bytes. The tensor must be on CPU and contiguous;
    we move + clone if needed. Layout is whatever the tensor's strides
    define — we don't normalize, so the read side must use the same
    dtype and shape (typically guaranteed by SGLang's host pool layout).
    """
    t = _torch()
    if value.device.type != "cpu":
        value = value.cpu()
    if not value.is_contiguous():
        value = value.contiguous()
    return value.view(t.uint8).numpy().tobytes()


def _bytes_into_tensor(payload: bytes, target: torch.Tensor) -> torch.Tensor | None:
    """Copy bytes into a pre-allocated target tensor. Returns the target
    on success, None if size mismatch OR non-contiguous target
    (treat as cache miss — caller should not error out on this; it
    means the cache is stale or the layout is wrong).

    PR #9 review fix B: non-contiguous targets are rejected loudly
    rather than silently corrupting. The `.view(uint8).copy_()`
    pattern only works when the target is contiguous (`.view()`
    requires compatible strides). Production SGLang flows always
    pre-allocate contiguous host pool tensors; this guard catches
    upstream regressions before they silently lose data.
    """
    t = _torch()
    expected = target.numel() * target.element_size()
    if len(payload) != expected:
        logger.warning(
            "kvd value size mismatch (expected %d bytes, got %d) — treating as miss",
            expected,
            len(payload),
        )
        return None
    if not target.is_contiguous():
        logger.warning(
            "kvd: _bytes_into_tensor received non-contiguous target "
            "(shape=%s strides=%s) — treating as miss to avoid silent "
            "corruption. Investigate the SGLang host pool layout.",
            tuple(target.shape),
            target.stride(),
        )
        return None
    src = t.frombuffer(bytearray(payload), dtype=t.uint8)
    target.view(t.uint8).copy_(src.view_as(target.view(t.uint8)))
    return target


# ----------------------------------------------------------------------
# Registration helper — called by the SGLang worker entrypoint
# ----------------------------------------------------------------------


def register_kvd_backend_with_sglang() -> None:
    """Tell SGLang's `StorageBackendFactory` about our backend. Called
    once at worker startup, before launch_server, so SGLang can resolve
    `--hicache-storage-backend infera-kvd` to our class.

    Safe to call multiple times — the factory rejects duplicates with
    a clear error, which we silence to make idempotent."""
    if not _SGLANG_AVAILABLE:
        raise RuntimeError("SGLang not importable; cannot register backend")
    try:
        from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory

        StorageBackendFactory.register_backend(
            "infera-kvd",
            "infera.engine.sglang.kvd_adapter",
            "InferaKvdBackend",
        )
        logger.info("registered infera-kvd backend with SGLang factory")
    except ValueError as exc:
        # `register_backend` raises if already present. Idempotent.
        if "already registered" not in str(exc):
            raise
