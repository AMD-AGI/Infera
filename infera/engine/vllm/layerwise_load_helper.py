###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""DRAFT — helpers for Variant B layerwise pipelined load. NOT WIRED IN.

Hosts the three pieces both Variant B and any future prefetch-worker
variant share:

* :func:`_ensure_load_copy_stream` — lazy-init a dedicated CUDA copy
  stream on the connector (mirror of ``_ensure_save_stream`` at
  kvd_connector.py:1193). H2D for per-layer slices runs here so the
  default stream (attention + Triton scatter) is never blocked by a
  copy.
* :class:`_PinnedHostBufferPool` — a tiny size-keyed allocator for
  pinned-host staging tensors. We pre-allocate one slot per
  (chunk × prefetch_depth) in start_load_kv and return them to the pool
  when the chunk retires. Pinned-host is mandatory for
  ``copy_(non_blocking=True)`` on the copy stream to actually overlap
  with default-stream work; pageable host makes the call serialise.
* :class:`_LayerEventRegistry` — per-(chunk, layer) ``cuda.Event``
  registry. The pipelining model issues H2D on the copy stream and
  records an event; :func:`wait_for_layer_load` looks up that event
  in O(1) and calls ``default_stream.wait_event(event)`` so the
  subsequent Triton scatter (default stream) sees the staging-tensor
  writes.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Dedicated H2D copy stream
# ----------------------------------------------------------------------


def _ensure_load_copy_stream(connector: Any, torch_mod: Any) -> Any | None:
    """Lazy-init ``connector._load_copy_stream``. Mirrors
    :meth:`InferaKvdConnector._ensure_save_stream` (kvd_connector.py:
    1193) — same construction pattern, separate stream so save D2H and
    load H2D never queue behind each other. Returns ``None`` if CUDA
    isn't available or no KV cache is registered yet (caller falls back
    to synchronous in-line copies on the default stream — equivalent
    to today's :func:`_load_chunk_packed`).

    IMPLEMENTER: add ``self._load_copy_stream: Any | None = None`` to
    ``InferaKvdConnector.__init__`` next to ``self._save_stream``.
    """
    stream = getattr(connector, "_load_copy_stream", None)
    if stream is not None:
        return stream
    if not torch_mod.cuda.is_available():
        connector._load_copy_stream = None
        return None
    if not connector._kv_caches:
        return None
    sample = next(iter(connector._kv_caches.values()))
    device = sample.device
    if device.type != "cuda":
        connector._load_copy_stream = None
        return None
    try:
        connector._load_copy_stream = torch_mod.cuda.Stream(device=device)
    except Exception:
        logger.exception(
            "_ensure_load_copy_stream: torch.cuda.Stream construction failed"
            " — falling back to default-stream sync copies"
        )
        connector._load_copy_stream = None
    return connector._load_copy_stream


# ----------------------------------------------------------------------
# Pinned-host buffer pool
# ----------------------------------------------------------------------


@dataclass
class _PinnedSlot:
    """One reusable pinned-host staging tensor. ``shape`` is the
    largest shape ever requested at this size; the caller passes in a
    sub-view via narrow/reshape rather than reallocating per layer."""

    tensor: Any  # torch.Tensor on CPU, pinned=True
    nbytes: int


class _PinnedHostBufferPool:
    """Size-keyed pool of pinned-host staging tensors.

    We do NOT try to be clever about exact shape — every slot is a
    flat ``uint8`` pinned tensor of size ``slot_nbytes``; the caller
    reinterprets it with ``.view(dtype).reshape(...)`` for each layer
    slice. Per-chunk Kimi K2.5 MLA per-layer slice is ~590 KiB; a
    prefetch_depth=2 fleet at concurrency 24 × 35 chunks needs
    24 × 35 × 2 = 1,680 slots ≈ 970 MiB if we pre-allocated everything.

    In practice we allocate **per-chunk × prefetch_depth** slots
    LAZILY on first acquire and return them when the chunk retires.
    Cap pool size via ``INFERA_KVD_LAYERWISE_PINNED_CAP_MIB`` (default
    256 MiB). Above the cap, fall back to a fresh pageable allocation
    (which loses overlap but doesn't OOM).

    Pool is process-global; access from the model thread only — no
    locking needed in the steady-state path. The ``_lock`` only guards
    the cap accounting + fallback log so a stray ``shutdown`` path
    doesn't race.
    """

    def __init__(self, cap_mib: int = 256) -> None:
        self._cap_bytes = int(cap_mib) * 1024 * 1024
        self._free: dict[int, list[_PinnedSlot]] = {}
        self._allocated_bytes = 0
        self._lock = threading.Lock()
        self._fallback_warned = False

    def acquire(self, nbytes: int, torch_mod: Any) -> _PinnedSlot | None:
        """Return a pinned slot with at least ``nbytes`` bytes, or
        ``None`` if cap exceeded (caller falls back to pageable +
        blocking copy)."""
        bucket = self._free.get(nbytes)
        if bucket:
            return bucket.pop()
        with self._lock:
            if self._allocated_bytes + nbytes > self._cap_bytes:
                if not self._fallback_warned:
                    logger.warning(
                        "_PinnedHostBufferPool: cap %d MiB exceeded — falling "
                        "back to pageable host (overlap lost); raise "
                        "INFERA_KVD_LAYERWISE_PINNED_CAP_MIB",
                        self._cap_bytes // (1024 * 1024),
                    )
                    self._fallback_warned = True
                return None
            self._allocated_bytes += nbytes
        try:
            tensor = torch_mod.empty(nbytes, dtype=torch_mod.uint8, pin_memory=True)
        except (RuntimeError, MemoryError):
            logger.exception(
                "_PinnedHostBufferPool: pin_memory alloc failed (%d B)",
                nbytes,
            )
            with self._lock:
                self._allocated_bytes -= nbytes
            return None
        return _PinnedSlot(tensor=tensor, nbytes=nbytes)

    def release(self, slot: _PinnedSlot | None) -> None:
        if slot is None:
            return
        self._free.setdefault(slot.nbytes, []).append(slot)

    def drain(self) -> None:
        """Drop all cached slots — call on connector shutdown."""
        self._free.clear()
        with self._lock:
            self._allocated_bytes = 0
            self._fallback_warned = False


# ----------------------------------------------------------------------
# Per-(chunk, layer) CUDA event registry
# ----------------------------------------------------------------------


@dataclass
class _LayerEventRegistry:
    """Per-chunk dense ``list[cuda.Event | None]`` indexed by local
    layer index. ``record(layer_idx, event)`` and ``wait(layer_idx,
    target_stream)`` are O(1). ``None`` for layers whose H2D hasn't
    been kicked off yet, also for layers whose copy already had the
    target stream wait once (avoids redundant wait_event calls in the
    case where wait_for_layer_load fires multiple times for the same
    name — vLLM doesn't today, but the registry is defensive)."""

    num_layers: int
    events: list[Any] = field(default_factory=list)
    waited: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.events = [None] * self.num_layers
        self.waited = [False] * self.num_layers

    def record(self, layer_idx: int, event: Any) -> None:
        self.events[layer_idx] = event

    def wait(self, layer_idx: int, target_stream: Any) -> None:
        """Have ``target_stream`` block on this layer's H2D event. If
        no event was recorded (layer never prefetched — e.g. fell off
        the prefetch_depth window during failure handling), fall back
        to a synchronous device sync (slow path; logs once)."""
        if self.waited[layer_idx]:
            return
        ev = self.events[layer_idx]
        if ev is None:
            # Fallback: the prefetcher never reached this layer. Caller
            # will then do a synchronous H2D itself; the missing wait is
            # not a correctness problem because Variant B's caller does
            # the H2D + scatter inline when there's no event waiting.
            return
        target_stream.wait_event(ev)
        self.waited[layer_idx] = True

    def has_event(self, layer_idx: int) -> bool:
        return self.events[layer_idx] is not None


# ----------------------------------------------------------------------
# Module-level singletons (the connector owns them)
# ----------------------------------------------------------------------


def get_or_create_pinned_pool(connector: Any) -> _PinnedHostBufferPool:
    """Attach a single ``_PinnedHostBufferPool`` to the connector on
    first use. Cap is read from ``INFERA_KVD_LAYERWISE_PINNED_CAP_MIB``
    (default 256). IMPLEMENTER: drain on ``InferaKvdConnector.close``.
    """
    pool = getattr(connector, "_layerwise_pinned_pool", None)
    if pool is not None:
        return pool
    import os as _os

    try:
        cap = int(_os.environ.get("INFERA_KVD_LAYERWISE_PINNED_CAP_MIB", "256"))
    except ValueError:
        cap = 256
    pool = _PinnedHostBufferPool(cap_mib=cap)
    connector._layerwise_pinned_pool = pool
    return pool


def get_prefetch_depth() -> int:
    """``INFERA_KVD_LAYERWISE_PREFETCH_DEPTH`` (default 2). 2 means
    layer N+2's H2D is kicked off when we ``wait_for_layer_load(N)``;
    layer N's attention covers layer N+1's H2D. Higher = more pinned
    buffer, more potential overlap; bounded by per-chunk per-layer
    nbytes × concurrency × num_chunks (see design doc Variant B
    steady-state pipeline math)."""
    import os as _os

    try:
        d = int(_os.environ.get("INFERA_KVD_LAYERWISE_PREFETCH_DEPTH", "2"))
    except ValueError:
        d = 2
    return max(1, d)
