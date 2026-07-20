###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Wire schema for KV cache events: msgpack-framed batches over ZMQ.

Topic framing: ZMQ multipart message is `[topic_bytes, payload_bytes]`
where `topic_bytes = model_name.encode("utf-8")` and `payload_bytes` is
msgpack-encoded EventBatch.

Schema versioning: every batch carries `event_version` (currently 1).
Subscribers refuse to consume future versions they don't understand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import msgpack

# Current wire protocol version. Bump when the schema changes in a
# backwards-incompatible way (see 19-trust-and-deployment.md § "Forward-
# compat reminders"). Subscribers reject batches whose event_version is
# beyond what they support.
EVENT_VERSION = 1


class EventType(str, Enum):
    STORED = "stored"
    REMOVED = "removed"
    CLEARED = "cleared"


@dataclass(frozen=True)
class NormalizedEvent:
    """One event in a batch. Field set varies by `type` — fields not relevant
    to the event type are None and not emitted on the wire.

    See 03-data-model.md § "Wire format" for the full schema.
    """

    type: EventType
    # Common to stored / removed:
    sequence_hash: int | None = None
    tier: str | None = None
    role: str | None = None
    group_idx: int | None = None
    # Stored-only:
    block_hash: int | None = None
    parent_sequence_hash: int | None = None
    mm_extra: list | None = None
    # Cleared-only:
    scope: str | None = None  # "all" | "model:<name>" | "tier:<name>" | "compat_key:<hex>"
    # Pool-tagged (fabric tier / infera-kvd disk tier):
    pool_id: str | None = None
    pool_type: str | None = None
    # Observation time on the publisher (not used for ordering — batch_id is).
    ts_ms: int | None = None

    def to_msgpack_dict(self) -> dict:
        """Compact dict for msgpack encoding — omits None fields."""
        d: dict = {"type": self.type.value}
        for fname in (
            "sequence_hash",
            "tier",
            "role",
            "group_idx",
            "block_hash",
            "parent_sequence_hash",
            "mm_extra",
            "scope",
            "pool_id",
            "pool_type",
            "ts_ms",
        ):
            v = getattr(self, fname)
            if v is not None:
                d[fname] = v
        return d

    @classmethod
    def from_msgpack_dict(cls, d: dict) -> NormalizedEvent:
        type_str = d.get("type")
        if type_str not in {t.value for t in EventType}:
            raise ValueError(f"unknown event type: {type_str!r}")
        return cls(
            type=EventType(type_str),
            sequence_hash=d.get("sequence_hash"),
            tier=d.get("tier"),
            role=d.get("role"),
            group_idx=d.get("group_idx"),
            block_hash=d.get("block_hash"),
            parent_sequence_hash=d.get("parent_sequence_hash"),
            mm_extra=d.get("mm_extra"),
            scope=d.get("scope"),
            pool_id=d.get("pool_id"),
            pool_type=d.get("pool_type"),
            ts_ms=d.get("ts_ms"),
        )


@dataclass(frozen=True)
class EventBatch:
    """One ZMQ message payload."""

    publisher_id: str
    publisher_type: str  # "worker" | "pool"
    model_name: str
    compat_key: str
    index_block_size: int
    batch_id: int  # monotonic per (publisher_id, model_name, compat_key) stream
    events: tuple[NormalizedEvent, ...] = field(default_factory=tuple)
    event_version: int = EVENT_VERSION


# ----------------------------------------------------------------------
# Encode / decode
# ----------------------------------------------------------------------


def encode_topic(model_name: str) -> bytes:
    """ZMQ topic frame. Subscribers filter by topic, so this is the most
    selective key we can give them."""
    return model_name.encode("utf-8")


def decode_topic(topic: bytes) -> str:
    return topic.decode("utf-8")


def encode_batch(batch: EventBatch) -> bytes:
    """msgpack-encoded payload."""
    payload = {
        "v": batch.event_version,
        "publisher_id": batch.publisher_id,
        "publisher_type": batch.publisher_type,
        "model_name": batch.model_name,
        "compat_key": batch.compat_key,
        "index_block_size": batch.index_block_size,
        "batch_id": batch.batch_id,
        "events": [e.to_msgpack_dict() for e in batch.events],
    }
    return msgpack.packb(payload, use_bin_type=True)


def decode_batch(data: bytes) -> EventBatch:
    """Decode a msgpack-framed batch. Raises on malformed input or
    incompatible event_version.
    """
    obj = msgpack.unpackb(data, raw=False)
    if not isinstance(obj, dict):
        raise ValueError(f"expected dict at root, got {type(obj).__name__}")
    version = obj.get("v")
    if version is None:
        raise ValueError("missing 'v' (event_version) field")
    if not isinstance(version, int):
        raise ValueError(f"'v' must be int, got {type(version).__name__}")
    if version > EVENT_VERSION:
        raise ValueError(
            f"event_version {version} is newer than supported ({EVENT_VERSION}); refusing to decode"
        )
    events_raw = obj.get("events") or []
    events = tuple(NormalizedEvent.from_msgpack_dict(e) for e in events_raw)
    return EventBatch(
        publisher_id=obj["publisher_id"],
        publisher_type=obj["publisher_type"],
        model_name=obj["model_name"],
        compat_key=obj["compat_key"],
        index_block_size=obj["index_block_size"],
        batch_id=obj["batch_id"],
        events=events,
        event_version=version,
    )


# Convenience factory helpers used by probes / daemons.


def make_stored_event(
    *,
    sequence_hash: int,
    block_hash: int,
    parent_sequence_hash: int | None,
    tier: str,
    role: str = "indexable",
    group_idx: int = 0,
    mm_extra: list | None = None,
    pool_id: str | None = None,
    pool_type: str | None = None,
    ts_ms: int | None = None,
) -> NormalizedEvent:
    return NormalizedEvent(
        type=EventType.STORED,
        sequence_hash=sequence_hash,
        block_hash=block_hash,
        parent_sequence_hash=parent_sequence_hash,
        tier=tier,
        role=role,
        group_idx=group_idx,
        mm_extra=mm_extra,
        pool_id=pool_id,
        pool_type=pool_type,
        ts_ms=ts_ms,
    )


def make_removed_event(
    *,
    sequence_hash: int,
    tier: str,
    role: str = "indexable",
    group_idx: int = 0,
    pool_id: str | None = None,
    pool_type: str | None = None,
    ts_ms: int | None = None,
) -> NormalizedEvent:
    return NormalizedEvent(
        type=EventType.REMOVED,
        sequence_hash=sequence_hash,
        tier=tier,
        role=role,
        group_idx=group_idx,
        pool_id=pool_id,
        pool_type=pool_type,
        ts_ms=ts_ms,
    )


def make_cleared_event(*, scope: str, ts_ms: int | None = None) -> NormalizedEvent:
    return NormalizedEvent(type=EventType.CLEARED, scope=scope, ts_ms=ts_ms)


# ----------------------------------------------------------------------
# Snapshot wire format (HTTP / JSON, not msgpack)
# ----------------------------------------------------------------------
#
# Snapshots travel over HTTP (worker GET /v1/kv-snapshot) at low frequency
# (~30 s reconcile interval + on-demand on event-gap detection). JSON is
# easier to debug than msgpack here and the volume is small.


@dataclass(frozen=True)
class SnapshotBlock:
    """One cached block in a snapshot. Equivalent to one apply_stored
    event but expressed in a form suitable for bulk reconstruction.
    """

    sequence_hash: int
    parent_sequence_hash: int | None
    block_hash: int
    tiers: tuple[str, ...]  # tier names currently resident


@dataclass(frozen=True)
class Snapshot:
    """Point-in-time dump of one publisher's cache state for one
    (model_name, compat_key)."""

    publisher_id: str
    publisher_type: str
    model_name: str
    compat_key: str
    index_block_size: int
    batch_id: int  # highest batch_id reflected in this snapshot
    blocks: tuple[SnapshotBlock, ...]
    event_version: int = EVENT_VERSION


def snapshot_to_json(snapshot: Snapshot) -> dict:
    """Serialize a Snapshot to a JSON-safe dict (large integers are
    represented as hex strings to avoid JSON-int precision issues
    in clients that aren't Python — JS/Go/etc. cap at 53 bits).
    """
    return {
        "v": snapshot.event_version,
        "publisher_id": snapshot.publisher_id,
        "publisher_type": snapshot.publisher_type,
        "model_name": snapshot.model_name,
        "compat_key": snapshot.compat_key,
        "index_block_size": snapshot.index_block_size,
        "batch_id": snapshot.batch_id,
        "blocks": [
            {
                "sequence_hash": f"0x{b.sequence_hash:016x}",
                "parent_sequence_hash": (
                    f"0x{b.parent_sequence_hash:016x}"
                    if b.parent_sequence_hash is not None
                    else None
                ),
                "block_hash": f"0x{b.block_hash:016x}",
                "tiers": list(b.tiers),
            }
            for b in snapshot.blocks
        ],
    }


def snapshot_from_json(obj: dict) -> Snapshot:
    """Inverse of snapshot_to_json. Accepts integers or 0x-prefixed hex strings
    for the hash fields (tolerant to either encoding)."""
    version = obj.get("v")
    if version is None:
        raise ValueError("missing 'v' (event_version) field")
    if version > EVENT_VERSION:
        raise ValueError(f"event_version {version} is newer than supported ({EVENT_VERSION})")

    def _hash(value) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value, 16)
        raise ValueError(f"bad hash field: {value!r}")

    blocks_raw = obj.get("blocks") or []
    blocks = tuple(
        SnapshotBlock(
            sequence_hash=_hash(b["sequence_hash"]),
            parent_sequence_hash=(
                _hash(b["parent_sequence_hash"])
                if b.get("parent_sequence_hash") is not None
                else None
            ),
            block_hash=_hash(b["block_hash"]),
            tiers=tuple(b.get("tiers") or ()),
        )
        for b in blocks_raw
    )
    return Snapshot(
        publisher_id=obj["publisher_id"],
        publisher_type=obj["publisher_type"],
        model_name=obj["model_name"],
        compat_key=obj["compat_key"],
        index_block_size=obj["index_block_size"],
        batch_id=obj["batch_id"],
        blocks=blocks,
        event_version=version,
    )
