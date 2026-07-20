###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""IPC wire format for infera-kvd.

Length-prefixed msgpack frames over a stream socket (Unix domain
socket today; same shape works for TCP / TLS later).

Each frame::

    [4 bytes BE u32 length][msgpack-encoded request or response]

Request types: GET, SET, EXISTS, CLEAR, STATS, HELLO.
Response types: GET_RESPONSE, SET_ACK, EXISTS_RESPONSE,
CLEAR_ACK, STATS_RESPONSE, HELLO_ACK, ERROR.

Design notes:

- We carry a `retention` field on every SET so the future SSD policy
  (Phase 3.5) can partition writes into spillover vs long regions
  without protocol changes. Today (Phase 3.0) the host RAM store
  uses it only as the priority key in eviction.

- We carry a `model` + `compat_key` namespace so a single daemon can
  serve multiple models / quantization variants without key
  collisions. Both default to empty string (single-tenant deployments
  can ignore them).

- Block bytes are carried inline. For 1-2 GB device→host transfers,
  the eventual production path will use shared memory or pinned-buffer
  ring + DMA. Phase 3.0 inlines bytes because it keeps the skeleton
  simple and lets us validate the protocol semantics first.

- Frames are msgpack to be **shape-compatible with LMCache's
  control protocol** — we pick the same encoder so a future swap or
  protocol bridge has minimum impedance mismatch.
"""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass, field
from typing import Any

import msgpack

# Length prefix: 4 bytes, big-endian, unsigned 32-bit. Max frame ~4 GiB
# — we don't enforce a smaller cap at the wire layer because individual
# Set frames legitimately carry whole blocks (hundreds of MB).
LENGTH_PREFIX_BYTES = 4
LENGTH_BYTEORDER = "big"


class OpCode(str, enum.Enum):
    """Top-level message type. String-valued so msgpack-encoded frames
    are debuggable without a schema lookup."""

    # Client → server
    HELLO = "hello"
    GET = "get"
    BATCH_GET = "batch_get"
    SET = "set"
    BATCH_SET = "batch_set"
    SET_RESERVE = "set_reserve"
    SET_COMMIT = "set_commit"
    SET_CANCEL = "set_cancel"
    PREFETCH_HINT = "prefetch_hint"
    EXISTS = "exists"
    CLEAR = "clear"
    STATS = "stats"

    # Server → client
    HELLO_ACK = "hello_ack"
    GET_RESPONSE = "get_response"
    GET_SHARED_RESPONSE = "get_shared_response"
    BATCH_GET_RESPONSE = "batch_get_response"
    BATCH_GET_SHARED_RESPONSE = "batch_get_shared_response"
    SET_ACK = "set_ack"
    BATCH_SET_ACK = "batch_set_ack"
    SET_RESERVE_RESPONSE = "set_reserve_response"
    SET_COMMIT_RESPONSE = "set_commit_response"
    SET_CANCEL_RESPONSE = "set_cancel_response"
    EXISTS_RESPONSE = "exists_response"
    CLEAR_ACK = "clear_ack"
    STATS_RESPONSE = "stats_response"
    ERROR = "error"


# Retention values match `infera.router.cache_control.Retention`.
# Repeated here as plain strings so this module doesn't import the
# router (the daemon must be runnable without router code present).
#
# Ordering: `ephemeral < short < long`. The store evicts strictly in
# that order (ephemeral entries leave first under capacity pressure,
# long entries survive longest). Callers can drive this via the
# request-level ``kv_transfer_params.infera_retention`` knob.
#
# `ephemeral` is the new (2026-05-25, issue #20) class for blocks
# with near-zero reuse value: reasoning/thinking tokens, intra-
# session scratch, etc. Treatment differs from `short` in two ways:
# (1) the vLLM connector SKIPS L2 write-back for ephemeral blocks
# (no point caching in pinned host RAM if we'll never look it up
# again), and (2) the daemon's eviction policy treats ephemeral as
# strictly lower priority than `short`. Use it when the caller knows
# the bytes are scratch.
RETENTION_NONE = "none"
RETENTION_EPHEMERAL = "ephemeral"
RETENTION_SHORT = "short"
RETENTION_LONG = "long"
_VALID_RETENTIONS = frozenset(
    {RETENTION_NONE, RETENTION_EPHEMERAL, RETENTION_SHORT, RETENTION_LONG}
)

# Storage-tier discriminator returned by a LookupTier query: which tier a
# key currently resolves from, WITHOUT transferring its bytes. Lets an
# engine decide between a UDS Get (RAM) and a direct hipFile read (file).
TIER_MISS = "miss"  # not present in any tier
TIER_RAM = "ram"  # in the host RAM store — serve via UDS Get
TIER_FILE = "file"  # on the file tier — engine may hipFile-read directly


# ----------------------------------------------------------------------
# Message dataclasses (msgpack-encoded as dicts with top-level `op` key)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Hello:
    """First frame from client. Lets server reject incompatible clients
    early (currently a no-op, but the version field gates future schema
    evolution).

    ``prefers_shared_arena``: when True and the server has a SharedArena
    wired, the server sends back HelloAck with a non-None ``shared_arena``
    field, then immediately sends the arena FD via SCM_RIGHTS on the
    same socket. The client mmaps the FD and uses GetShared / BatchGetShared
    responses (offsets, not bytes). Default False for backward compat —
    existing clients connect unchanged.
    """

    client_id: str
    protocol_version: int = 1
    prefers_shared_arena: bool = False


@dataclass(frozen=True)
class HelloAck:
    """Server's reply to Hello.

    ``shared_arena``: when present (the client asked AND the server can
    provide), describes the shared arena. The actual FD follows
    immediately as an SCM_RIGHTS ancillary message on the same socket.
    Encoded on the wire as a 3-tuple ``(arena_size, slot_size,
    server_pid)`` — see ``infera.kvd.shared_arena.SharedArenaInfo``.
    None means "fall back to inline-bytes responses."
    """

    server_id: str
    protocol_version: int = 1
    # 3-tuple form on the wire: (arena_size, slot_size, server_pid).
    # Stored as the tuple so msgpack can round-trip it without a
    # custom encoder. Clients reconstruct ``SharedArenaInfo`` via
    # ``SharedArenaInfo.from_tuple``.
    shared_arena: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class Get:
    """Look up one block by content hash.

    `key` is the engine-side sequence hash (XXH3-64) of the block,
    encoded as 8 bytes. Multi-block lookups are issued as multiple
    GETs in the same connection — batching is a future optimization.
    """

    key: bytes
    model: str = ""
    compat_key: str = ""


@dataclass(frozen=True)
class GetResponse:
    """Cache hit returns `value` (block bytes); miss returns None."""

    value: bytes | None
    # Round-trip the metadata so the engine can verify retention or layer count.
    retention: str | None = None


@dataclass(frozen=True)
class GetSharedResponse:
    """Shared-arena GET response — carries an offset into the arena
    instead of the bytes themselves. The client (which mmapped the
    arena at handshake) reads bytes directly via seqlock.

    Fields:
    - ``slot_offset``: byte offset of the payload within the arena.
      Already past the slot header — client just reads `length`
      bytes from `mm[slot_offset:slot_offset+length]`.
    - ``length``: payload size in bytes.
    - ``version``: expected seqlock version. Client reads version
      from the slot header (at `slot_offset - HEADER_BYTES`),
      reads payload, re-reads version; on mismatch, retries
      internally up to a small bound, then returns None (caller
      falls through to the long-region path).
    - ``retention``: same metadata the existing `GetResponse`
      carries (so the connector can promote / refresh).
    - ``slot_size``: arena slot_size at the time of this response.
      Carried for client-side sanity (matches HelloAck) — bumping
      arenas without re-handshake would corrupt the offset math,
      so the server's contract is "slot_size never changes after
      the first put."

    Cache miss: client receives a regular ``GetResponse(value=None)``,
    not a GetSharedResponse with length=0. This keeps the miss path
    identical for shared and inline clients.
    """

    slot_offset: int
    length: int
    version: int
    retention: str | None = None
    slot_size: int = 0


@dataclass(frozen=True)
class BatchGetSharedResponse:
    """Batch variant of ``GetSharedResponse``. Parallel arrays:
    `offsets[i]`, `lengths[i]`, `versions[i]`, `retentions[i]`
    describe the i-th block. Missing entries carry `lengths[i] == 0`
    AND `offsets[i] == -1` to disambiguate miss-on-batch from
    zero-byte hit. Clients treat `lengths[i] == 0 or offsets[i] < 0`
    as miss and return None for that position."""

    offsets: list[int]
    lengths: list[int]
    versions: list[int]
    retentions: list[str | None]
    slot_size: int = 0


@dataclass(frozen=True)
class BatchGet:
    """Look up N blocks in one round-trip. Equivalent to N serial
    ``Get`` calls but with the connection lock held once and a
    single response frame. Used by the connector's ``start_load_kv``
    to coalesce per-block GETs for a request.

    `keys` is a list of 8-byte content hashes. Order matters —
    response values come back in the same order.

    Frame-size note: the response carries every hit's block bytes
    inline. For ~4 MB packed multi-layer blobs, the practical batch
    cap is ~250 keys per request (1 GB response frame). The
    connector enforces this cap before calling; the daemon will
    process whatever it's asked.
    """

    keys: list[bytes]
    model: str = ""
    compat_key: str = ""


@dataclass(frozen=True)
class BatchGetResponse:
    """Per-key result, same length and order as the request's `keys`.
    `values[i] is None` means the i-th key missed.
    `retentions[i]` carries the retention level so the connector
    can promote/refresh entries; mirrors `GetResponse`."""

    values: list[bytes | None]
    retentions: list[str | None]


@dataclass(frozen=True)
class Set:
    """Insert a block.

    `retention` is the policy hint that future SSD partitioning will
    use to decide spillover-vs-long region (Phase 3.5+). Today the host
    RAM store treats it as the priority key in eviction.

    `metadata` is engine-specific opaque key-value pairs (e.g.
    `layer_count`, `dtype`, `mm_offsets`) the daemon stores but doesn't
    interpret. Bound by msgpack encoding size limits but practically
    small.
    """

    key: bytes
    value: bytes
    retention: str = RETENTION_SHORT
    model: str = ""
    compat_key: str = ""
    metadata: dict = field(default_factory=dict)
    # Time-to-live in seconds. None means "no TTL — eviction governed
    # by retention level + capacity pressure only." A finite value
    # makes the entry expire at SET time + ttl_seconds, regardless of
    # retention. Lazy expiration — checked on `get`, not via sweeper —
    # so a stale entry costs nothing if nobody asks for it. Issue #20
    # item 1: matches Anthropic's prompt-cache `cache_control` /
    # Dynamo's `nvext.cache_control.ttl` semantics.
    ttl_seconds: float | None = None


@dataclass(frozen=True)
class SetAck:
    """Confirms the block was accepted. `accepted=False` means the
    daemon rejected (e.g., would-be-evicted has higher priority than
    the incoming block, or the store is full of pinned entries). The
    client decides whether to retry, downgrade retention, or drop."""

    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class BatchSet:
    """Insert N blocks in one round-trip. Equivalent to N serial
    ``Set`` calls but with the connection lock held once and a single
    response frame. Used by the connector's save flush path to
    coalesce per-block SETs after a request's prefill completes.

    Parallel arrays — `keys[i]`, `values[i]`, `retentions[i]`,
    `metadatas[i]` together describe one block. All four must have
    the same length. We use parallel arrays (rather than a list of
    tuples) so msgpack semantics stay unambiguous round-trip.

    `model` and `compat_key` apply to ALL items in the batch — every
    block in a single connector flush belongs to the same engine and
    model.

    Frame-size note: the REQUEST carries every block's value inline,
    so the cap is symmetric with `BatchGet`'s response — ~250 entries
    at 4 MB packed blobs (1 GB request frame). The connector should
    chunk above that. The daemon processes whatever it's asked.
    """

    keys: list[bytes]
    values: list[bytes]
    retentions: list[str]
    metadatas: list[dict]
    model: str = ""
    compat_key: str = ""
    # Parallel list — one TTL per item (None == no TTL). Optional;
    # missing field on the wire defaults to no-TTL for all items.
    # See `Set.ttl_seconds` for the eviction contract.
    ttls_seconds: list[float | None] | None = None


@dataclass(frozen=True)
class BatchSetAck:
    """Per-item result, same length and order as the request's
    `items`. `accepted[i]=False` plus `reasons[i]=<code>` mirrors the
    single-`SetAck` semantics — caller decides per item whether to
    retry, downgrade retention, or drop."""

    accepted: list[bool]
    reasons: list[str | None]


@dataclass(frozen=True)
class SetReserve:
    """Request the daemon to reserve an arena slot for a forthcoming
    zero-copy write (save-side CopyFree, two-phase set).

    The engine intends to write ``size`` bytes directly into the
    shared arena slot via its own mmap, then send a SetCommit naming
    the actual key. No key is carried in the reserve frame — that
    lets the engine reuse a reserved slot for whatever key it
    settles on (the GPU→host copy may race with the engine's own
    bookkeeping that decides the key).

    The lease is identified by an opaque ``lease_token`` issued by
    the server on accept. The server tracks reservations per
    connection so an unclean disconnect can drop them automatically.
    """

    size: int


@dataclass(frozen=True)
class SetReserveResponse:
    """Server's reply to a SetReserve.

    ``lease_token``: nonzero on accept (opaque to the client; only
      meaningful as the lookup key for SetCommit / SetCancel).
      ``0`` = rejected; the client should fall back to the legacy
      inline-bytes Set path.
    ``slot_id``: the arena slot index allocated to this lease.
      ``-1`` on reject. Carried for debugging / observability.
    ``payload_offset``: byte offset of the payload inside the shared
      arena mmap, already past the slot header — the engine writes
      its bytes at ``arena_mv[payload_offset:payload_offset+length]``.
    ``payload_max_size``: ``slot_size - HEADER_BYTES`` (upper bound
      on bytes the engine may write into this slot).
    ``reason``: empty on accept; on reject, a short stable token
      such as ``"no_arena"``, ``"oversize"``,
      ``"arena_full_no_evictable"``.
    """

    lease_token: int
    slot_id: int
    payload_offset: int
    payload_max_size: int
    reason: str = ""


@dataclass(frozen=True)
class SetCommit:
    """Finalize a reservation: name the key, declare how many bytes
    were written, and hand off retention/TTL/namespace metadata.

    After the server processes this:
      - The store's index points the composite (model, compat_key,
        key) at the arena slot.
      - The seqlock header is bumped to "stable" so readers see a
        consistent view.
      - The lease is dropped from the server's reservation table.

    ``length`` must be ``<= payload_max_size`` from the reserve
    response — the server enforces and rejects oversize commits.
    """

    lease_token: int
    key: bytes
    length: int
    model: str = ""
    compat_key: str = ""
    # Retention level — same set as `Set.retention`. "default" means
    # "use the server's default" (currently treated as short by the
    # commit handler).
    retention: str = "default"
    # TTL in seconds. 0 = no TTL (retention-only eviction). Mirrors
    # `Set.ttl_seconds=None` but with an int wire type since 0 is
    # the natural "no TTL" sentinel for the lease path.
    ttl_seconds: int = 0


@dataclass(frozen=True)
class SetCommitResponse:
    """Server's reply to SetCommit. ``accepted=False`` + ``reason``
    means the lease was unknown, expired, or oversize-on-commit;
    the engine should NOT retry under the same lease."""

    accepted: bool
    reason: str = ""


@dataclass(frozen=True)
class SetCancel:
    """Drop a reservation without committing. Used when the engine's
    GPU→host copy fails or the surrounding flush is aborted.

    Idempotent at the wire level — the server returns success even
    for unknown lease tokens (the engine's retry safety net should
    not have to distinguish). Mismatched connection ownership is
    swallowed at info-level on the server."""

    lease_token: int


@dataclass(frozen=True)
class SetCancelResponse:
    """Empty ack so the client's round-trip semantics stay uniform."""


@dataclass(frozen=True)
class PrefetchHint:
    """Fire-and-forget: pull these block hashes from the long /
    spillover tier into the host RAM tier so the engine's next
    `get` sees a fast hit instead of a cold L3 miss.

    No response — the daemon dispatches into an internal async
    worker and returns control to the read loop immediately. The
    caller is the router (issue #20 item 3 — speculative L3 prefetch).

    `deadline_ms` is the router's TTFT budget for this hint. The
    daemon uses it as the TTL on warmed entries so a hint nobody
    follows through on doesn't pin host RAM indefinitely.
    Keys already in host RAM are filtered out by the worker; the
    hint is idempotent and over-eager hints are cheap (filter +
    no-op).

    PD design doc §6.2 + Phase 3 spec this; PR closing issue #20
    item 3 implements it.
    """

    keys: list[bytes]
    model: str = ""
    compat_key: str = ""
    deadline_ms: int = 1000


@dataclass(frozen=True)
class Exists:
    """Membership test — same as GET but doesn't return the bytes.
    Used by routers (cache_view) and by engines pre-flight before
    allocating a load slot."""

    keys: list[bytes]
    model: str = ""
    compat_key: str = ""


@dataclass(frozen=True)
class ExistsResponse:
    """One bool per key in the request, same order."""

    present: list[bool]


@dataclass(frozen=True)
class LookupTier:
    """Ask which storage tier holds ``key`` (RAM / file / miss) WITHOUT
    transferring the bytes. The response (`LookupTierResponse`) lets the
    caller decide between a UDS Get (RAM tier) and a direct hipFile read
    (file tier), avoiding a redundant RAM round-trip for on-disk chunks."""

    key: bytes
    model: str = ""
    compat_key: str = ""


@dataclass(frozen=True)
class LookupTierResponse:
    """Where ``key`` resolved. ``tier`` is one of TIER_MISS / TIER_RAM /
    TIER_FILE. For ``tier == TIER_FILE`` the (``path``, ``file_offset``,
    ``size``) triple locates the payload for a direct hipFile read;
    ``version`` and ``retention`` carry the entry's metadata."""

    tier: str
    path: str | None = None
    file_offset: int = 0
    size: int = 0
    version: int = 0
    retention: str | None = None


@dataclass(frozen=True)
class RegisterFileEntry:
    """Register an already-on-disk chunk with the daemon so a later
    ``LookupTier`` resolves ``key`` to (``path``, ``file_offset``,
    ``size``) for a direct hipFile read — the engine writes the file
    itself and only registers the metadata, so no bytes flow through the
    UDS Set path."""

    key: bytes
    path: str
    file_offset: int = 0
    size: int = 0
    version: int = 0
    retention: str = RETENTION_SHORT
    model: str = ""
    compat_key: str = ""


@dataclass(frozen=True)
class RegisterFileEntryAck:
    """Daemon's acknowledgement of a ``RegisterFileEntry``."""

    ok: bool = True


@dataclass(frozen=True)
class Clear:
    """Drop everything in a namespace. Used by tests + by operators
    via a CLI. Setting `model="" compat_key=""` (default) clears the
    whole store."""

    model: str = ""
    compat_key: str = ""


@dataclass(frozen=True)
class ClearAck:
    cleared_entries: int


@dataclass(frozen=True)
class Stats:
    """No-arg query — daemon returns current state for /metrics
    and ops debugging."""


@dataclass(frozen=True)
class StatsResponse:
    """Per-tier byte counts + counters. Phase 3.0 only fills `host_bytes`
    and the counters; other tiers stay zero until SSD lands."""

    entries: int
    host_bytes: int
    spillover_bytes: int
    long_bytes: int
    gets_total: int
    sets_total: int
    hits_total: int
    misses_total: int
    evictions_total: int


@dataclass(frozen=True)
class ErrorMessage:
    """Response to any client message that the daemon refuses to
    service. `code` is a stable short token (e.g. `bad_op`,
    `unknown_op`, `internal_error`) suitable for client switching."""

    code: str
    message: str


# Union type for static checkers. Use a tuple of classes for runtime checks.
Request = (
    Hello
    | Get
    | BatchGet
    | Set
    | BatchSet
    | SetReserve
    | SetCommit
    | SetCancel
    | PrefetchHint
    | Exists
    | Clear
    | Stats
)
Response = (
    HelloAck
    | GetResponse
    | GetSharedResponse
    | BatchGetResponse
    | BatchGetSharedResponse
    | SetAck
    | BatchSetAck
    | SetReserveResponse
    | SetCommitResponse
    | SetCancelResponse
    | ExistsResponse
    | ClearAck
    | StatsResponse
    | ErrorMessage
)


# ----------------------------------------------------------------------
# Encoding / decoding
# ----------------------------------------------------------------------


_OP_TO_CLS: dict[str, type] = {
    OpCode.HELLO.value: Hello,
    OpCode.GET.value: Get,
    OpCode.BATCH_GET.value: BatchGet,
    OpCode.SET.value: Set,
    OpCode.BATCH_SET.value: BatchSet,
    OpCode.SET_RESERVE.value: SetReserve,
    OpCode.SET_COMMIT.value: SetCommit,
    OpCode.SET_CANCEL.value: SetCancel,
    OpCode.PREFETCH_HINT.value: PrefetchHint,
    OpCode.EXISTS.value: Exists,
    OpCode.CLEAR.value: Clear,
    OpCode.STATS.value: Stats,
    OpCode.HELLO_ACK.value: HelloAck,
    OpCode.GET_RESPONSE.value: GetResponse,
    OpCode.GET_SHARED_RESPONSE.value: GetSharedResponse,
    OpCode.BATCH_GET_RESPONSE.value: BatchGetResponse,
    OpCode.BATCH_GET_SHARED_RESPONSE.value: BatchGetSharedResponse,
    OpCode.SET_ACK.value: SetAck,
    OpCode.BATCH_SET_ACK.value: BatchSetAck,
    OpCode.SET_RESERVE_RESPONSE.value: SetReserveResponse,
    OpCode.SET_COMMIT_RESPONSE.value: SetCommitResponse,
    OpCode.SET_CANCEL_RESPONSE.value: SetCancelResponse,
    OpCode.EXISTS_RESPONSE.value: ExistsResponse,
    OpCode.CLEAR_ACK.value: ClearAck,
    OpCode.STATS_RESPONSE.value: StatsResponse,
    OpCode.ERROR.value: ErrorMessage,
}


def encode(msg: Any) -> bytes:
    """Serialize a message dataclass to a length-prefixed msgpack frame."""
    op = _opcode_for(msg)
    payload = {"op": op, **_to_dict(msg)}
    body = msgpack.packb(payload, use_bin_type=True)
    return len(body).to_bytes(LENGTH_PREFIX_BYTES, LENGTH_BYTEORDER) + body


def decode(body: bytes) -> Any:
    """Parse a single msgpack frame body (no length prefix) into the
    matching dataclass. Raises ValueError on unknown ops or schema
    mismatch."""
    obj = msgpack.unpackb(body, raw=False)
    if not isinstance(obj, dict) or "op" not in obj:
        raise ValueError(f"frame is not an op dict: {obj!r}")
    op = obj.pop("op")
    cls = _OP_TO_CLS.get(op)
    if cls is None:
        raise ValueError(f"unknown op: {op!r}")
    if cls is Stats:
        return Stats()
    # msgpack decodes tuples as lists; for fields we documented as
    # tuple-typed (HelloAck.shared_arena = (arena_size, slot_size,
    # server_pid)) we re-tupleize so equality / type checks downstream
    # behave consistently.
    if cls is HelloAck and obj.get("shared_arena") is not None:
        obj["shared_arena"] = tuple(obj["shared_arena"])
    try:
        return cls(**obj)
    except TypeError as exc:
        raise ValueError(f"bad payload for {op!r}: {exc}") from exc


async def read_frame(reader: asyncio.StreamReader) -> Any:
    """Read one length-prefixed frame from a stream and decode it.
    Raises asyncio.IncompleteReadError on connection close, ValueError
    on schema violations.
    """
    header = await reader.readexactly(LENGTH_PREFIX_BYTES)
    length = int.from_bytes(header, LENGTH_BYTEORDER)
    if length == 0:
        raise ValueError("empty frame body")
    body = await reader.readexactly(length)
    return decode(body)


async def write_frame(writer: asyncio.StreamWriter, msg: Any) -> None:
    """Encode + write + drain. Caller handles ConnectionError."""
    writer.write(encode(msg))
    await writer.drain()


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _opcode_for(msg: Any) -> str:
    """Map a dataclass instance to its OpCode value."""
    if isinstance(msg, Hello):
        return OpCode.HELLO.value
    if isinstance(msg, HelloAck):
        return OpCode.HELLO_ACK.value
    if isinstance(msg, Get):
        return OpCode.GET.value
    if isinstance(msg, GetResponse):
        return OpCode.GET_RESPONSE.value
    if isinstance(msg, GetSharedResponse):
        return OpCode.GET_SHARED_RESPONSE.value
    if isinstance(msg, BatchGet):
        return OpCode.BATCH_GET.value
    if isinstance(msg, BatchGetResponse):
        return OpCode.BATCH_GET_RESPONSE.value
    if isinstance(msg, BatchGetSharedResponse):
        return OpCode.BATCH_GET_SHARED_RESPONSE.value
    if isinstance(msg, Set):
        return OpCode.SET.value
    if isinstance(msg, SetAck):
        return OpCode.SET_ACK.value
    if isinstance(msg, BatchSet):
        return OpCode.BATCH_SET.value
    if isinstance(msg, BatchSetAck):
        return OpCode.BATCH_SET_ACK.value
    if isinstance(msg, SetReserve):
        return OpCode.SET_RESERVE.value
    if isinstance(msg, SetReserveResponse):
        return OpCode.SET_RESERVE_RESPONSE.value
    if isinstance(msg, SetCommit):
        return OpCode.SET_COMMIT.value
    if isinstance(msg, SetCommitResponse):
        return OpCode.SET_COMMIT_RESPONSE.value
    if isinstance(msg, SetCancel):
        return OpCode.SET_CANCEL.value
    if isinstance(msg, SetCancelResponse):
        return OpCode.SET_CANCEL_RESPONSE.value
    if isinstance(msg, PrefetchHint):
        return OpCode.PREFETCH_HINT.value
    if isinstance(msg, Exists):
        return OpCode.EXISTS.value
    if isinstance(msg, ExistsResponse):
        return OpCode.EXISTS_RESPONSE.value
    if isinstance(msg, Clear):
        return OpCode.CLEAR.value
    if isinstance(msg, ClearAck):
        return OpCode.CLEAR_ACK.value
    if isinstance(msg, Stats):
        return OpCode.STATS.value
    if isinstance(msg, StatsResponse):
        return OpCode.STATS_RESPONSE.value
    if isinstance(msg, ErrorMessage):
        return OpCode.ERROR.value
    raise TypeError(f"not a kvd wire message: {type(msg).__name__}")


def _to_dict(msg: Any) -> dict[str, Any]:
    """Shallow asdict that preserves bytes as bytes (dataclasses.asdict
    deep-copies and serializes through reprs which mangles bytes)."""
    import dataclasses

    return {f.name: getattr(msg, f.name) for f in dataclasses.fields(msg)}


def validate_retention(value: str) -> str:
    """Raise ValueError if `value` isn't a known retention level.
    Callers may also pass-through unknown values if they want forward
    compatibility, but the daemon enforces the closed set."""
    if value not in _VALID_RETENTIONS:
        raise ValueError(f"unknown retention: {value!r} (expected one of {_VALID_RETENTIONS})")
    return value
