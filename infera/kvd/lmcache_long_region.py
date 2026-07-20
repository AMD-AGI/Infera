###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""LMCache remote-backend adapter for kvd's L4 (distributed long region).

Wraps LMCache's ``RemoteConnector`` so one integration gives Infera
access to LMCache's sub-backend roster: Redis, Valkey, S3, NIXL, NFS,
WEKA.

## Validated against real Redis (2026-05-29)

The adapter was built by probing the **actual** LMCache v0.x API against
a live Redis container, not from docs. Key findings that shaped it:

  - LMCache's storage stack is **tensor-shaped, not opaque-bytes**. The
    RemoteConnector wire protocol (`lmcache/v1/protocol.py`) serializes
    per-tensor ``(shape, dtype)`` pairs and rejects ``BINARY`` /
    ``BINARY_BUFFER`` MemoryObjs (their dtype is None → the protocol's
    ``zip(shapes, dtypes, strict=True)`` raises).
  - To store opaque KV blocks we therefore encode the bytes as a
    ``KV_2TD``-format MemoryObj with shape ``(2, ceil(m/2), 1)`` uint8,
    where ``m = 8-byte length header + payload``. The length header lets
    us recover the exact byte count on read (the shape pads to even).
  - The connector returned by ``CreateConnector`` is an
    ``InstrumentedRemoteConnector`` whose ``put`` / ``get`` are
    **coroutines** — bridged sync via a background asyncio loop.
  - Construction needs ``LMCacheEngineConfig`` + ``LMCacheMetadata`` +
    a ``LocalCPUBackend`` (for the MemoryObj allocator) + an event loop.

This is heavier than a generic K/V client because LMCache was designed
to live *inside* vLLM/SGLang with full tensor context. We adapt it to
opaque-bytes L4 use; a native Infera L4 would be leaner (future work).

Defensive import: the module loads without ``lmcache`` installed; the
import error surfaces from ``start()``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import struct
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from infera.kvd.long_region_proto import (
    REASON_EMPTY_VALUE,
    REASON_NOT_STARTED,
    REASON_OK,
    REASON_OVERSIZE,
    REASON_RPC_FAILED,
    REASON_TIMEOUT,
    REASON_WRONG_RETENTION,
)

logger = logging.getLogger(__name__)

# 8-byte little-endian length header prepended to every payload so the
# exact original byte count survives the KV_2TD even-padding.
_LEN_HEADER = struct.Struct("<Q")

_COMPAT_RE = re.compile(r"tp(?P<tp>\d+).*?pp(?P<pp>\d+)", re.IGNORECASE)


def _parse_compat_key(compat_key: str) -> tuple[int, int]:
    """Best-effort (world_size, worker_id) from a compat_key fingerprint.
    Falls back to (1, 0); the caller keeps the full compat_key in the
    model_name so distinct fingerprints never collide."""
    m = _COMPAT_RE.search(compat_key or "")
    if not m:
        return 1, 0
    try:
        return int(m.group("tp")), int(m.group("pp"))
    except (TypeError, ValueError):
        return 1, 0


def _chunk_hash_int(key: bytes) -> int:
    """CacheEngineKey wants an int chunk_hash. Use a stable 63-bit
    blake2b digest of the raw key so it's restart-stable + collision-safe."""
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") >> 1


@dataclass
class LMCacheRemoteConfig:
    remote_url: str  # "redis://host:6379" / "s3://bucket/pfx" / ...
    prefix: str  # namespace / tenant boundary (→ model_name prefix)
    serde: str = "naive"  # kept for compat; LMCache config handles serde
    connector_extra: dict = field(default_factory=dict)
    max_value_bytes: int = 64 * 1024 * 1024
    rpc_timeout_s: float = 30.0


class _LMCacheBinding:
    """Owns the real LMCache objects: config, metadata, cpu_backend,
    connector, plus the background event loop the connector's async
    methods run on. Isolated so unit tests can subclass + stub it."""

    def __init__(self) -> None:
        self._connector: Any = None
        self._cpu: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._CacheEngineKey: Any = None
        self._MemoryFormat: Any = None
        self._torch: Any = None
        self._timeout_s: float = 30.0

    def connect(self, config: LMCacheRemoteConfig) -> None:
        try:
            import torch
            from lmcache.utils import CacheEngineKey
            from lmcache.v1.config import LMCacheEngineConfig
            from lmcache.v1.memory_management import MemoryFormat
            from lmcache.v1.metadata import LMCacheMetadata
            from lmcache.v1.storage_backend.connector import CreateConnector
            from lmcache.v1.storage_backend.local_cpu_backend import (
                LocalCPUBackend,
            )
        except ImportError as exc:
            raise RuntimeError(
                "LMCacheRemoteLongRegion requires the lmcache package + the "
                "chosen sub-backend's deps (redis-py / boto3 / nixl). "
                f"Install via `pip install lmcache`. Error: {exc}"
            ) from exc

        self._torch = torch
        self._CacheEngineKey = CacheEngineKey
        self._MemoryFormat = MemoryFormat
        self._timeout_s = config.rpc_timeout_s

        # Background asyncio loop for the connector's coroutine put/get.
        ready = threading.Event()

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=_run, name="lmcache-l4-loop", daemon=True)
        self._loop_thread.start()
        if not ready.wait(timeout=5.0) or self._loop is None:
            raise RuntimeError("LMCache adapter: event loop failed to start")

        lm_cfg = LMCacheEngineConfig.from_defaults(
            remote_url=config.remote_url, **config.connector_extra
        )
        # Minimal metadata — opaque-bytes use, so kv_shape/dtype are
        # placeholders. world_size/worker_id default; per-key routing
        # happens in the CacheEngineKey, not here.
        meta = LMCacheMetadata(
            model_name=config.prefix,
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.uint8,
            kv_shape=(1, 1, 1, 1, 1),
        )
        self._cpu = LocalCPUBackend(lm_cfg, meta, dst_device="cpu")
        self._connector = CreateConnector(config.remote_url, self._loop, self._cpu, lm_cfg, meta)

    # -- async bridge --
    def _run(self, coro_or_val: Any) -> Any:
        if not asyncio.iscoroutine(coro_or_val):
            return coro_or_val
        assert self._loop is not None
        fut: Future = asyncio.run_coroutine_threadsafe(coro_or_val, self._loop)
        return fut.result(timeout=self._timeout_s)

    # -- key + memory-object helpers --
    def make_key(self, model_name: str, world_size: int, worker_id: int, chunk_hash: int) -> Any:
        return self._CacheEngineKey(
            model_name, world_size, worker_id, chunk_hash, self._torch.uint8
        )

    def encode(self, payload: bytes) -> Any:
        """bytes → KV_2TD MemoryObj with an 8-byte length header.

        Fill is done through the buffer protocol (numpy view → memoryview
        slice), NOT ``torch.Tensor.copy_``. Measured on the MI355X testbed: torch's
        uint8 ``copy_`` of a 256 KB block takes ~6.5 ms (~40 MB/s — a
        per-element fallback), while the memoryview slice is ~2 µs
        (a real memcpy). That's a ~3400× difference and was the entire
        L4-put overhead before this fix."""
        torch = self._torch
        body = _LEN_HEADER.pack(len(payload)) + payload
        m = len(body)
        half = math.ceil(m / 2)
        mo = self._cpu.allocate(
            torch.Size([2, half, 1]), torch.uint8, fmt=self._MemoryFormat.KV_2TD
        )
        if mo is None:
            raise RuntimeError("LMCache cpu allocator returned None (out of memory?)")
        # numpy() is a zero-copy view over the (cpu, contiguous) tensor's
        # buffer; the memoryview slice assignment is a single memcpy.
        mv = memoryview(mo.get_tensor(0).flatten().numpy())
        mv[:m] = body
        return mo

    @staticmethod
    def decode(mo: Any) -> bytes | None:
        if mo is None:
            return None
        # tobytes() over the numpy view is a single copy out; reading the
        # 8-byte length header lets us slice off the even-pad.
        raw = mo.get_tensor(0).flatten().numpy().tobytes()
        if len(raw) < _LEN_HEADER.size:
            return None
        (length,) = _LEN_HEADER.unpack(raw[: _LEN_HEADER.size])
        start = _LEN_HEADER.size
        return raw[start : start + length]

    def put(self, key: Any, mo: Any) -> None:
        self._run(self._connector.put(key, mo))

    def get(self, key: Any) -> Any:
        return self._run(self._connector.get(key))

    def contains(self, key: Any) -> bool:
        try:
            return bool(self._connector.exists_sync(key))
        except Exception:
            return bool(self._run(self._connector.exists(key)))

    def close(self) -> None:
        if self._connector is not None:
            try:
                self._run(self._connector.close())
            except Exception:  # pragma: no cover
                pass
        self._connector = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
        self._loop = None
        self._loop_thread = None


@dataclass
class LMCacheEntryShim:
    key: bytes
    size_bytes: int
    retention: str
    model: str
    compat_key: str
    metadata: dict
    last_access: float


class LMCacheRemoteLongRegion:
    """Distributed L4 backend proxying to an LMCache remote sub-backend.
    Implements ``LongRegionLike`` (infera/kvd/long_region_proto.py)."""

    RETENTION_LONG = "long"

    def __init__(
        self,
        config: LMCacheRemoteConfig,
        *,
        binding: _LMCacheBinding | None = None,
    ) -> None:
        self._config = config
        self._binding = binding if binding is not None else _LMCacheBinding()
        self._started = False
        self._lock = threading.Lock()
        self._puts = 0
        self._put_failures = 0
        self._gets = 0
        self._get_misses = 0
        self._used_bytes = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._binding.connect(self._config)
        self._started = True
        logger.info(
            "LMCacheRemoteLongRegion: connected url=%s prefix=%s",
            self._config.remote_url,
            self._config.prefix,
        )

    def shutdown(self) -> None:
        if not self._started:
            return
        self._binding.close()
        self._started = False
        logger.info("LMCacheRemoteLongRegion: closed")

    # ------------------------------------------------------------------
    # Key mapping
    # ------------------------------------------------------------------

    def _key(self, key: bytes, model: str, compat_key: str) -> Any:
        ws, wid = _parse_compat_key(compat_key)
        model_name = (
            f"{self._config.prefix}/{model}"
            if (ws, wid) != (1, 0)
            else f"{self._config.prefix}/{model}/{compat_key}"
        )
        return self._binding.make_key(model_name, ws, wid, _chunk_hash_int(key))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def put(
        self,
        key: bytes,
        value: bytes,
        *,
        retention: str,
        model: str,
        compat_key: str,
        metadata: dict,
    ) -> tuple[bool, str | None]:
        if not self._started:
            return False, REASON_NOT_STARTED
        if retention != self.RETENTION_LONG:
            return False, REASON_WRONG_RETENTION
        size = len(value)
        if size == 0:
            return False, REASON_EMPTY_VALUE
        if size > self._config.max_value_bytes:
            return False, REASON_OVERSIZE
        try:
            ck = self._key(key, model, compat_key)
            mo = self._binding.encode(value)
            self._binding.put(ck, mo)
        except TimeoutError:
            with self._lock:
                self._put_failures += 1
            return False, REASON_TIMEOUT
        except Exception as exc:
            logger.warning(
                "LMCacheRemoteLongRegion.put failed key=%s err=%s",
                key.hex()[:16],
                exc,
            )
            with self._lock:
                self._put_failures += 1
            return False, REASON_RPC_FAILED
        with self._lock:
            self._puts += 1
            self._used_bytes += size
        return True, REASON_OK

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_bytes(self, key: bytes, *, model: str, compat_key: str) -> bytes | None:
        if not self._started:
            return None
        try:
            ck = self._key(key, model, compat_key)
            mo = self._binding.get(ck)
        except Exception:
            logger.exception("LMCacheRemoteLongRegion.get_bytes failed")
            return None
        out = self._binding.decode(mo)
        with self._lock:
            self._gets += 1
            if out is None:
                self._get_misses += 1
        return out

    def get_bytes_batch(
        self, keys: list[bytes], *, model: str, compat_key: str
    ) -> list[bytes | None]:
        # The InstrumentedRemoteConnector doesn't expose a stable public
        # batched coroutine across versions; loop single gets (correct,
        # and the connector pipelines on its end). A batched-RPC fast
        # path is a follow-up once we pin an LMCache version.
        return [self.get_bytes(k, model=model, compat_key=compat_key) for k in keys]

    def exists(self, keys: list[bytes], *, model: str, compat_key: str) -> list[bool]:
        if not self._started or not keys:
            return [False] * len(keys)
        out = []
        for k in keys:
            try:
                out.append(self._binding.contains(self._key(k, model, compat_key)))
            except Exception:
                out.append(False)
        return out

    def get_entry(self, key: bytes, *, model: str, compat_key: str) -> LMCacheEntryShim | None:
        v = self.get_bytes(key, model=model, compat_key=compat_key)
        if v is None:
            return None
        import time

        return LMCacheEntryShim(
            key=key,
            size_bytes=len(v),
            retention=self.RETENTION_LONG,
            model=model,
            compat_key=compat_key,
            metadata={},
            last_access=time.monotonic(),
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        scheme = (
            self._config.remote_url.split("://", 1)[0] if "://" in self._config.remote_url else "?"
        )
        with self._lock:
            return {
                "used_bytes": self._used_bytes,
                "max_bytes": -1,
                "entries_count": self._puts,
                "backend_name": f"lmcache_remote:{scheme}",
                "puts_total": self._puts,
                "put_failures_total": self._put_failures,
                "gets_total": self._gets,
                "get_misses_total": self._get_misses,
                "remote_url": self._config.remote_url,
                "prefix": self._config.prefix,
            }

    def clear(self) -> int:
        logger.warning(
            "LMCacheRemoteLongRegion.clear() is a no-op — clear via the "
            "sub-backend's tools scoped to prefix=%s (redis FLUSHDB / "
            "aws s3 rm).",
            self._config.prefix,
        )
        return 0
