###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""What hiding a collective behind compute costs the compute.

The projector's overlap model hides up to ``ceiling x comm`` of a layer's
collective behind that layer's compute for free. On a GPU the two share two
things. RCCL runs its channels as kernels, so the compute kernels beside it get
fewer CUs; and every byte a collective moves is read from or written to the
same HBM the compute kernels stream from. A memory-bound kernel slows by the
HBM share the collective takes, a compute-bound one by the CU share. Over the
hidden window compute progresses at rate::

    r = 1 - [f_mem x hbm_share + (1 - f_mem) x cu_share]

where ``f_mem`` is the fraction of compute time in memory-bound kernels, so the
hidden part costs ``(1 - r) x hidden`` of extra compute.

Off by default: a default run charges exactly what the projector charges, and
with the projector's default ceiling of 0 nothing is hidden to contend.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import DLLMHardware


@dataclass
class ContentionConfig:
    enabled: bool = False
    # CUs RCCL's channels occupy while a collective runs (16 of 256 on MI355X).
    comm_cu_fraction: float = 16.0 / 256.0
    # HBM bytes a collective moves per wire byte: the read of the send buffer
    # and the write of the receive buffer, plus the reduction's read.
    hbm_traffic_factor: float = 3.0
    # Floor on the compute rate under contention.
    min_rate: float = 0.05


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


def penalty_ms(
    comm_ms: float,
    keep: float,
    *,
    wire_bytes: float,
    mem_fraction: float,
    hw: DLLMHardware,
    cfg: ContentionConfig,
) -> float:
    """Extra compute time from hiding ``(1 - keep) x comm_ms`` of a collective."""
    hidden = comm_ms * (1.0 - min(1.0, max(0.0, keep)))
    if hidden <= 0:
        return 0.0
    r = compute_rate(comm_ms, wire_bytes, mem_fraction, hw, cfg)
    return (1.0 - r) * hidden
