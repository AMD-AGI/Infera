###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Mooncake Store backend for kvd's L4 (distributed long region).

Wraps ``MooncakeDistributedStore`` (kvcache-ai/Mooncake) behind the
``LongRegionLike`` shape. It is the top L4 candidate — best
ROCm story (native HIP transport via ``-DUSE_HIP=ON``), genuine
opaque-bytes object store (unlike LMCache, which is tensor-coupled — see
``project_lmcache_is_tensor_coupled`` memory).

## API matched to the real binding (2026-05-29)

The adapter's binding shim mirrors the **actual** Mooncake Python API,
read from the installed ``mooncake`` package source (``store.so`` +
``mooncake_store_service.py``):

    store = MooncakeDistributedStore()          # no-arg constructor
    store.setup(local_hostname, metadata_server, global_segment_size,
                local_buffer_size, protocol, device_name,
                master_server_address)          # returns 0 on success
    store.put(key: str, value: bytes) -> int    # 0 on success
    store.get(key: str) -> bytes                 # b"" / None on miss
    store.is_exist(key: str) -> int              # 1 present, 0 absent
    store.remove(key: str, force=False) -> int
    store.remove_all()
    store.close()

Batch: the HIP-built binding exposes ``get_batch(list[str])->list[bytes]``,
``batch_is_exist(list[str])->list[int]`` and ``batch_remove`` — one RPC for
N keys instead of N round trips (a big win cross-node). The adapter uses
them for ``get_bytes_batch`` / ``exists`` / ``clear`` and falls back to
per-key calls if a batch call raises or returns a mismatched length.

## Runtime validation status (2026-05-29 — built from source on ROCm)

The PyPI ``mooncake`` wheel is CUDA-linked and won't import on a ROCm
host. We **built Mooncake from source with ``-DUSE_HIP=ON``** on
an MI355X node (ROCm 7.1.1 / gfx950) and validated:

  - `store.cpython-312.so` compiles clean under HIP (gfx950) and imports
    on AMD with no CUDA libs (libamdhip64-linked). The `nvlink-allocator`
    target fails under HIP (NVIDIA NVLink fabric handles) but is NOT
    needed — build the `store.cpython-*.so` target directly to skip it.
  - `mooncake_master` + `mooncake_http_metadata_server` run on ROCm.
  - **Every method this adapter's binding shim calls** — setup / put /
    get / get_batch / is_exist / batch_is_exist / remove / batch_remove /
    close — exists on the real HIP-built `MooncakeDistributedStore`.

  CROSS-NODE DATA PLANE PROVEN (2026-05-29, two MI355X nodes, TCP over
  the AINIC): real put/get round-trips, peak ~3.8 GB/s put / 4.2 GB/s get
  at 4 MB blocks. Single-host loopback still fails ("Local segment
  descriptor not found" / backend_unavailable) — the supported topology
  is ≥2 nodes (one segment provider + one client). Two operational
  must-knows: (1) the ``mooncake_master`` binary MUST be built from the
  same source tree as ``store.so`` or every put fails with
  ``invalid rpc arg``; (2) ``local_hostname`` MUST be a routable
  ``host:port`` (in k8s, derive from ``POD_IP`` — ``gethostname()``
  returns the non-routable pod name).

Key operational note: ``local_hostname`` MUST include a port
(``host:port``, e.g. ``localhost:12355``) — the transfer engine binds
the local segment there. Bare host fails with "Local segment descriptor
not found".

Defensive import: module loads without the binding; the import error
surfaces from ``start()``.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

from infera.kvd.long_region_proto import (
    REASON_BACKEND_UNAVAILABLE,
    REASON_EMPTY_VALUE,
    REASON_NOT_STARTED,
    REASON_OK,
    REASON_OVERSIZE,
    REASON_RPC_FAILED,
    REASON_WRONG_RETENTION,
)

logger = logging.getLogger(__name__)


def _object_key(prefix: str, model: str, compat_key: str, key: bytes) -> str:
    """Flat string key (Mooncake keys are str). Prefix is the tenant /
    cluster namespace; key.hex() keeps it ASCII + CLI-inspectable."""
    return f"{prefix}:{model}\x00{compat_key}\x00{key.hex()}"


_ALLOW_TCP_FALLBACK_ENV = "INFERA_L4_ALLOW_TCP_FALLBACK"


def _verify_rdma_or_raise(config: MooncakeStoreConfig) -> None:
    """Refuse to start if ``protocol=rdma`` was requested but RDMA can't be
    used — Mooncake silently degrades to TCP, which the operator asked NOT
    to happen. Mirrors the PD "refuse TCP at worker startup" rule
    (``feedback_pd_refuse_tcp``). Pre-flight only: confirms a usable RDMA
    device is configured + present in ``/sys/class/infiniband``. It can NOT
    detect a device that exists but whose driver is broken at the verbs
    layer (e.g. the MI355X ionic ABI issue) — that still falls back to TCP
    silently. Hidden override: set ``INFERA_L4_ALLOW_TCP_FALLBACK=1``."""
    # TODO(rdma-validation): the actual RDMA data plane is UNVALIDATED. The
    # MI355X testbed can't test it — ionic RDMA is ABI-broken (CQE
    # retry_exceeded) and the fabric NICs (enp25s0/...) are unaddressed.
    # Validate end-to-end on a Mellanox host (or fixed ionic + IP'd fabric
    # NICs) before claiming RDMA throughput. TCP cross-node is proven; the
    # RDMA build link + config plumbing + this guard are unit-tested.
    if config.protocol.strip().lower() != "rdma":
        return
    if os.environ.get(_ALLOW_TCP_FALLBACK_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        logger.warning(
            "protocol=rdma but %s is set — skipping RDMA pre-flight; "
            "Mooncake may silently fall back to TCP",
            _ALLOW_TCP_FALLBACK_ENV,
        )
        return
    if not config.device_name.strip():
        raise RuntimeError(
            "protocol=rdma requires device_name (e.g. mlx5_0 / ionic_0); "
            "without it Mooncake silently uses TCP. Set device_name, or "
            f"set {_ALLOW_TCP_FALLBACK_ENV}=1 to allow TCP."
        )
    ib_dir = "/sys/class/infiniband"
    available = set(os.listdir(ib_dir)) if os.path.isdir(ib_dir) else set()
    if config.device_name not in available:
        raise RuntimeError(
            f"protocol=rdma device_name={config.device_name!r} not found in "
            f"{ib_dir} (available: {sorted(available) or 'none'}). Refusing to "
            f"start to avoid a silent TCP fallback. Fix the device, or set "
            f"{_ALLOW_TCP_FALLBACK_ENV}=1 to allow TCP."
        )


@dataclass
class MooncakeStoreConfig:
    """Operator-facing config. Maps to MooncakeDistributedStore.setup()."""

    master_address: str  # master_server_address, e.g. "10.0.0.1:50051"
    cluster_id: str  # our tenant/namespace prefix on keys
    metadata_server: str = ""  # etcd/http metadata URL (transfer engine)
    protocol: str = "tcp"  # "tcp" | "rdma" (rdma needs working fabric)
    device_name: str = ""  # RDMA device; "" for tcp
    local_hostname: str = ""  # auto-detected if empty
    global_segment_size: int = 1 << 30  # DRAM this node donates as a segment (1 GiB)
    local_buffer_size: int = 1 << 28  # local transfer staging buffer (256 MiB)
    max_value_bytes: int = 64 * 1024 * 1024


class _MooncakeBinding:
    """Resolves + drives the real MooncakeDistributedStore. Held by the
    region so a missing/CUDA-only binding fails at ``start()`` with a
    clear message. Subclass + stub in unit tests."""

    def __init__(self) -> None:
        self._store: Any = None

    def connect(self, config: MooncakeStoreConfig) -> None:
        try:
            mod = __import__("mooncake.store", fromlist=["MooncakeDistributedStore"])
        except ImportError as exc:
            raise RuntimeError(
                "MooncakeStoreLongRegion requires the Mooncake Python "
                "bindings. On AMD/ROCm the PyPI wheel is CUDA-linked and "
                "won't import — build kvcache-ai/Mooncake from source with "
                "-DUSE_HIP=ON -DBUILD_PYTHON=ON. "
                f"Original error: {exc}"
            ) from exc
        cls = getattr(mod, "MooncakeDistributedStore", None)
        if cls is None:
            raise RuntimeError("mooncake.store imported but has no MooncakeDistributedStore")
        _verify_rdma_or_raise(config)
        store = cls()
        local_host = config.local_hostname or socket.gethostname()
        ret = store.setup(
            local_host,
            config.metadata_server,
            config.global_segment_size,
            config.local_buffer_size,
            config.protocol,
            config.device_name,
            config.master_address,
        )
        if ret != 0:
            raise RuntimeError(f"MooncakeDistributedStore.setup returned {ret} (!= 0)")
        self._store = store

    # -- data plane (str keys, bytes values) --
    def put(self, key: str, value: bytes) -> int:
        return int(self._store.put(key, value))

    def get(self, key: str) -> bytes | None:
        v = self._store.get(key)
        if v is None:
            return None
        b = bytes(v)
        return b if len(b) > 0 else None

    def is_exist(self, key: str) -> bool:
        return bool(self._store.is_exist(key))

    def get_batch(self, keys: list[str]) -> list[bytes | None]:
        vals = self._store.get_batch(list(keys))
        out: list[bytes | None] = []
        for v in vals:
            if v is None:
                out.append(None)
            else:
                b = bytes(v)
                out.append(b if len(b) > 0 else None)
        return out

    def is_exist_batch(self, keys: list[str]) -> list[bool]:
        return [bool(x) for x in self._store.batch_is_exist(list(keys))]

    def remove_batch(self, keys: list[str]) -> list[int]:
        # force=True: clear() removes keys we own; without it Mooncake refuses
        # lease-protected objects with rc -706 (validated on the live cluster).
        return [int(x) for x in self._store.batch_remove(list(keys), True)]

    def remove(self, key: str) -> int:
        return int(self._store.remove(key, True))

    def close(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            except Exception:  # pragma: no cover
                logger.exception("Mooncake store close() raised; ignoring")
        self._store = None


@dataclass
class MooncakeEntryShim:
    key: bytes
    size_bytes: int
    retention: str
    model: str
    compat_key: str
    metadata: dict
    last_access: float


class MooncakeStoreLongRegion:
    """Distributed L4 backend proxying to a Mooncake cluster. Implements
    ``LongRegionLike`` (infera/kvd/long_region_proto.py)."""

    RETENTION_LONG = "long"

    def __init__(
        self,
        config: MooncakeStoreConfig,
        *,
        binding: _MooncakeBinding | None = None,
    ) -> None:
        self._config = config
        self._binding = binding if binding is not None else _MooncakeBinding()
        self._started = False
        self._lock = threading.Lock()
        self._puts = 0
        self._put_failures = 0
        self._gets = 0
        self._get_misses = 0
        self._used_bytes = 0
        # Track keys we've written so clear() can scope to our namespace
        # (Mooncake's remove_all() is cluster-wide → unsafe for multi-tenant).
        self._written: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._binding.connect(self._config)
        self._started = True
        logger.info(
            "MooncakeStoreLongRegion: connected master=%s cluster=%s protocol=%s",
            self._config.master_address,
            self._config.cluster_id,
            self._config.protocol,
        )

    def shutdown(self) -> None:
        if not self._started:
            return
        self._binding.close()
        self._started = False
        logger.info("MooncakeStoreLongRegion: closed")

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
        ok_key = _object_key(self._config.cluster_id, model, compat_key, key)
        try:
            ret = self._binding.put(ok_key, value)
        except Exception as exc:
            logger.warning(
                "MooncakeStoreLongRegion.put failed key=%s err=%s",
                key.hex()[:16],
                exc,
            )
            with self._lock:
                self._put_failures += 1
            return False, REASON_RPC_FAILED
        if ret != 0:
            with self._lock:
                self._put_failures += 1
            return False, REASON_BACKEND_UNAVAILABLE
        with self._lock:
            self._puts += 1
            self._used_bytes += size
            self._written.add(ok_key)
        return True, REASON_OK

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_bytes(self, key: bytes, *, model: str, compat_key: str) -> bytes | None:
        if not self._started:
            return None
        ok_key = _object_key(self._config.cluster_id, model, compat_key, key)
        try:
            v = self._binding.get(ok_key)
        except Exception:
            logger.exception("MooncakeStoreLongRegion.get_bytes failed")
            return None
        with self._lock:
            self._gets += 1
            if v is None:
                self._get_misses += 1
        return v

    def get_bytes_batch(
        self, keys: list[bytes], *, model: str, compat_key: str
    ) -> list[bytes | None]:
        if not self._started or not keys:
            return [None] * len(keys)
        ok_keys = [_object_key(self._config.cluster_id, model, compat_key, k) for k in keys]
        vals: list[bytes | None] | None = None
        try:
            got = self._binding.get_batch(ok_keys)
            if len(got) == len(keys):
                vals = got
        except Exception:
            logger.exception("get_bytes_batch: batch get failed; per-key fallback")
        if vals is None:
            return [self.get_bytes(k, model=model, compat_key=compat_key) for k in keys]
        with self._lock:
            self._gets += len(keys)
            self._get_misses += sum(1 for v in vals if v is None)
        return vals

    def exists(self, keys: list[bytes], *, model: str, compat_key: str) -> list[bool]:
        if not self._started or not keys:
            return [False] * len(keys)
        ok_keys = [_object_key(self._config.cluster_id, model, compat_key, k) for k in keys]
        try:
            res = self._binding.is_exist_batch(ok_keys)
            if len(res) == len(keys):
                return res
        except Exception:
            logger.exception("exists: batch is_exist failed; per-key fallback")
        out = []
        for ok_key in ok_keys:
            try:
                out.append(self._binding.is_exist(ok_key))
            except Exception:
                out.append(False)
        return out

    def get_entry(self, key: bytes, *, model: str, compat_key: str) -> MooncakeEntryShim | None:
        v = self.get_bytes(key, model=model, compat_key=compat_key)
        if v is None:
            return None
        return MooncakeEntryShim(
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
        with self._lock:
            return {
                "used_bytes": self._used_bytes,
                "max_bytes": -1,
                "entries_count": self._puts,
                "backend_name": "mooncake_store",
                "puts_total": self._puts,
                "put_failures_total": self._put_failures,
                "gets_total": self._gets,
                "get_misses_total": self._get_misses,
                "master_address": self._config.master_address,
                "cluster_id": self._config.cluster_id,
                "protocol": self._config.protocol,
            }

    def clear(self) -> int:
        """Remove this daemon's keys only. Mooncake's remove_all() is
        cluster-wide (unsafe for multi-tenant), so we remove the keys we
        tracked writing. Best-effort — keys written by a prior process
        instance aren't tracked and won't be cleared here."""
        if not self._started:
            return 0
        with self._lock:
            to_remove = list(self._written)
        removed = 0
        if to_remove:
            try:
                rets = self._binding.remove_batch(to_remove)
                removed = sum(1 for r in rets if r == 0)
            except Exception:
                logger.exception("clear: batch_remove failed; per-key fallback")
                for ok_key in to_remove:
                    try:
                        if self._binding.remove(ok_key) == 0:
                            removed += 1
                    except Exception:
                        logger.exception("clear: remove failed")
        with self._lock:
            self._written.clear()
        return removed
