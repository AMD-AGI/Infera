###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Plug-in shape for the long-retention / spillover region — the L4 boundary.

Implementations may be local (``TablespaceLongRegion`` on striped NVMe; PR #29)
or distributed (``MooncakeStoreLongRegion``, ``LMCacheRemoteLongRegion`` ...).
Exactly one is active in a given kvd process; the choice is controlled by the
``--long-backend`` CLI flag at startup.

``HostStore`` already consumes the long region duck-typed — this module just
documents the contract so backend authors have a single source of truth.
Existing ``TablespaceLongRegion`` and ``StripedLongRegion`` (PR #29) conform
without modification.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LongRegionLike(Protocol):
    """The duck-typed surface every long-region backend must expose."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open / recover the region. Distributed backends may connect to
        a Master / cluster here. MUST be idempotent."""
        ...

    def shutdown(self) -> None:
        """Optional. Flush, disconnect. Called on graceful kvd
        termination. Idempotent."""
        ...

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def put(
        self,
        key: bytes,
        value: bytes,
        *,
        retention: str,  # "short" | "long"
        model: str,
        compat_key: str,
        metadata: dict,
    ) -> tuple[bool, str | None]:
        """Write a block. Returns ``(accepted, reason)``. ``accepted`` is
        True on success; ``reason`` carries a short error code on
        rejection (matches ``HostStore.set`` return contract)."""
        ...

    # ------------------------------------------------------------------
    # Read — three flavors covering kvd's existing call sites
    # ------------------------------------------------------------------
    def get_entry(self, key: bytes, *, model: str, compat_key: str) -> Any | None:
        """Index-only lookup. Returns the in-memory metadata (an SsdEntry-
        shaped object) or None. Used by ``HostStore`` to decide whether to
        populate L1 from this region without paying the read cost."""
        ...

    def get_bytes(self, key: bytes, *, model: str, compat_key: str) -> bytes | None:
        """Single-key read. None on miss."""
        ...

    def get_bytes_batch(
        self, keys: list[bytes], *, model: str, compat_key: str
    ) -> list[bytes | None]:
        """Vectorized read. Order preserves ``keys``. None entries for
        misses. Distributed backends SHOULD batch the underlying RPC."""
        ...

    def exists(self, keys: list[bytes], *, model: str, compat_key: str) -> list[bool]:
        """Vectorized presence probe — no bytes returned. Distributed
        backends SHOULD batch the underlying RPC."""
        ...

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Region statistics. MUST include at minimum
        ``used_bytes``, ``max_bytes``, ``entries_count``, ``backend_name``;
        may include backend-specific gauges (``replication_lag``,
        ``p50_latency_us``, ``rdma_queue_depth`` ...)."""
        ...

    def clear(self) -> int:
        """Drop everything in this daemon's namespace. Returns count
        removed. Distributed backends MUST scope to this daemon — never
        cross-tenant."""
        ...


# ----------------------------------------------------------------------
# Shared error codes (so adapters return consistent reasons)
# ----------------------------------------------------------------------

REASON_OK: str | None = None
REASON_NOT_STARTED = "region_not_started"
REASON_BACKEND_UNAVAILABLE = "backend_unavailable"
REASON_OVER_QUOTA = "over_quota"
REASON_OVERSIZE = "value_exceeds_slot_bytes"
REASON_EMPTY_VALUE = "empty_value"
REASON_RPC_FAILED = "rpc_failed"
REASON_TIMEOUT = "rpc_timeout"
REASON_WRONG_RETENTION = "wrong_retention"
