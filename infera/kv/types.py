###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    """Storage tier for a cached KV block."""

    DEVICE = "device"
    HOST = "host"
    DISK = "disk"
    FABRIC = "fabric"


# Per-tier default credits used by PrefixCachePolicy. See 13-tier-and-pool.md
# § "Per-tier credit rationale" for why these values.
DEFAULT_TIER_CREDIT: dict[Tier, float] = {
    Tier.DEVICE: 1.0,
    Tier.HOST: 0.75,
    Tier.DISK: 0.25,
    Tier.FABRIC: 0.15,
}


class AttentionRole(str, Enum):
    """How a layer group behaves for prefix-cache routing.

    See 11-attention-kinds.md. Five roles cover the algorithm zoo
    (FullAttention, MLA, SinkFull, sliding window, Mamba, DeltaNet, RWKV,
    chunked-local, encoder-only, cross-attention) without enum churn when
    a new attention variant lands upstream.
    """

    INDEXABLE = "indexable"
    SLIDING = "sliding"
    RECURRENT = "recurrent"
    ENCODER_ONLY = "encoder_only"
    CROSS = "cross"
    UNKNOWN = "unknown"

    def is_indexable(self) -> bool:
        return self in {AttentionRole.INDEXABLE, AttentionRole.SLIDING}

    def emits_events_to_main_index(self) -> bool:
        return self == AttentionRole.INDEXABLE or self == AttentionRole.SLIDING


@dataclass(frozen=True)
class AttentionGroup:
    """One layer group's structural signature for KV routing.

    `family` is the canonical model family name (see families.py).
    `window` and `sinks` are populated only for SLIDING role.
    """

    family: str
    role: AttentionRole
    layer_indices: tuple[int, ...] | None = None  # None means "all layers"
    window: int | None = None
    sinks: int | None = None

    def __post_init__(self) -> None:
        if self.role == AttentionRole.SLIDING and self.window is None:
            raise ValueError(f"window required for role=SLIDING, got family={self.family}")


@dataclass(frozen=True)
class BlockKey:
    """One block in a sequence-hash chain. See hashing.py."""

    sequence_hash: int  # u64
    block_hash: int  # u64
    parent_sequence_hash: int | None  # None at chain head


@dataclass
class OverlapBlocks:
    """Per-tier overlap count for one worker against a query chain."""

    device: int = 0
    host: int = 0
    disk: int = 0
    fabric: int = 0

    def total(self) -> int:
        return self.device + self.host + self.disk + self.fabric

    def best_tier_for(self, position: int) -> Tier | None:
        """Highest-credit tier covering this block index, or None if uncovered."""
        if position < self.device:
            return Tier.DEVICE
        if position < self.host:
            return Tier.HOST
        if position < self.disk:
            return Tier.DISK
        if position < self.fabric:
            return Tier.FABRIC
        return None


@dataclass(frozen=True)
class MmRun:
    """A contiguous run of placeholder tokens belonging to one multimodal object.

    See 12-multimodal.md. The hasher uses (mm_hash, start, end) to apply
    Dynamo's offset-relative encoding so the same image at different
    prompt positions hashes to the same per-block contribution.
    """

    mm_hash: int  # u64; content-stable hash of the MM object
    start: int  # inclusive global token index where the run begins
    end: int  # exclusive global token index where the run ends

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"MmRun end ({self.end}) must be > start ({self.start})")

    def __len__(self) -> int:
        return self.end - self.start


# ExecutionPlan variants used by Policy.decide(). Kept lightweight; the routers
# pattern-match on `kind`. See 10-composability.md.


@dataclass(frozen=True)
class Execution:
    """Concrete plan returned by Policy.decide()."""

    kind: str  # "mixed" | "disagg" | "local"
    worker_id: str | None = None  # mixed and local
    p_worker_id: str | None = None  # disagg
    d_worker_id: str | None = None  # disagg

    @classmethod
    def mixed(cls, worker_id: str) -> Execution:
        return cls(kind="mixed", worker_id=worker_id)

    @classmethod
    def disagg(cls, p_worker_id: str, d_worker_id: str) -> Execution:
        return cls(kind="disagg", p_worker_id=p_worker_id, d_worker_id=d_worker_id)

    @classmethod
    def local(cls, d_worker_id: str) -> Execution:
        return cls(kind="local", worker_id=d_worker_id, d_worker_id=d_worker_id)

    def is_mixed(self) -> bool:
        return self.kind == "mixed"

    def is_disagg(self) -> bool:
        return self.kind == "disagg"

    def is_local(self) -> bool:
        return self.kind == "local"
