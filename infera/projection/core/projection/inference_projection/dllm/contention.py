###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""What overlapping a collective with compute actually saves.

The ideal answer, ``max(compute, comm)``, assumes the two share nothing. On a
GPU they share two things. RCCL runs its channels as kernels, so the compute
kernels running beside it get fewer CUs; and every byte a collective moves is
read from or written to the same HBM the compute kernels stream their weights
and activations from. A memory-bound kernel slows by the HBM share the
collective takes, a compute-bound one by the CU share.

Over the window ``W = overlap x comm`` in which they run together, compute
progresses at rate::

    r = 1 - [f_mem x hbm_share + (1 - f_mem) x cu_share]

where ``f_mem`` is the fraction of the compute time spent in memory-bound
kernels. The step then takes ``comm + max(0, compute - r x W)``: ``max(compute,
comm)`` at ``r = 1`` and full overlap, ``compute + comm`` with no overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import DLLMHardware


@dataclass
class ContentionConfig:
    enabled: bool = True
    # CUs RCCL's channels occupy while a collective runs (16 of 256 on MI355X).
    comm_cu_fraction: float = 16.0 / 256.0
    # HBM bytes a collective moves per wire byte: the read of the send buffer
    # and the write of the receive buffer, plus the reduction's read.
    hbm_traffic_factor: float = 3.0
    # Floor on the compute rate under contention.
    min_rate: float = 0.05


@dataclass
class Overlapped:
    total_ms: float
    exposed_comm_ms: float
    contention_ms: float
    rate: float


def compute_rate(
    comm_ms: float,
    wire_bytes: float,
    mem_fraction: float,
    hw: DLLMHardware,
    cfg: ContentionConfig,
) -> float:
    if not cfg.enabled or comm_ms <= 0:
        return 1.0
    hbm_rate = cfg.hbm_traffic_factor * wire_bytes / (comm_ms / 1e3)
    hbm_share = min(1.0, hbm_rate / (hw.hbm_gbps * 1e9))
    cu_share = min(1.0, max(0.0, cfg.comm_cu_fraction))
    f = min(1.0, max(0.0, mem_fraction))
    return max(cfg.min_rate, 1.0 - (f * hbm_share + (1.0 - f) * cu_share))


def overlap(
    compute_ms: float,
    comm_ms: float,
    *,
    overlap_fraction: float,
    wire_bytes: float,
    mem_fraction: float,
    hw: DLLMHardware,
    cfg: ContentionConfig,
) -> Overlapped:
    """Combine one span of compute with the collectives issued alongside it."""
    o = min(1.0, max(0.0, overlap_fraction))
    if comm_ms <= 0 or o <= 0:
        return Overlapped(compute_ms + comm_ms, comm_ms, 0.0, 1.0)
    window = o * comm_ms
    r = compute_rate(comm_ms, wire_bytes, mem_fraction, hw, cfg)
    total = comm_ms + max(0.0, compute_ms - r * window)
    ideal = comm_ms + max(0.0, compute_ms - window)
    return Overlapped(total, total - compute_ms, total - ideal, r)
