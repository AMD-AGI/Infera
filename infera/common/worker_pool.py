###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WorkerStatus(str, Enum):
    ACTIVE = "active"
    DRAINING = "draining"
    DEAD = "dead"


class DisaggMode(str, Enum):
    MIXED = "mixed"
    PREFILL = "prefill"
    DECODE = "decode"


class EngineType(str, Enum):
    SGLANG = "sglang"
    VLLM = "vllm"
    ATOM = "atom"


@dataclass
class KvRegistrationMetadata:
    """The `kv` block in the worker / pool registration payload.

    Workers without these fields predate Phase 1 KV management; they're
    still routable (round-robin) but can't contribute to the prefix
    cache index.
    """

    engine_block_size: int
    index_block_size: int
    tokenizer: str
    tokenizer_digest: str
    tokenizer_canary: list[int]
    supports_events: bool = True
    event_version: int = 1
    events_endpoint: str | None = None  # ZMQ PUB tcp:// address
    snapshot_endpoint: str | None = None  # HTTP base URL for /v1/kv-snapshot;
    #                                       defaults to WorkerInfo.url server-side
    tiers: list[str] = field(default_factory=lambda: ["device"])

    def to_dict(self) -> dict:
        return {
            "engine_block_size": self.engine_block_size,
            "index_block_size": self.index_block_size,
            "tokenizer": self.tokenizer,
            "tokenizer_digest": self.tokenizer_digest,
            "tokenizer_canary": list(self.tokenizer_canary),
            "supports_events": self.supports_events,
            "event_version": self.event_version,
            "events_endpoint": self.events_endpoint,
            "snapshot_endpoint": self.snapshot_endpoint,
            "tiers": list(self.tiers),
        }

    @classmethod
    def from_dict(cls, d: dict) -> KvRegistrationMetadata:
        return cls(
            engine_block_size=int(d["engine_block_size"]),
            index_block_size=int(d["index_block_size"]),
            tokenizer=str(d["tokenizer"]),
            tokenizer_digest=str(d["tokenizer_digest"]),
            tokenizer_canary=list(d["tokenizer_canary"]),
            supports_events=bool(d.get("supports_events", True)),
            event_version=int(d.get("event_version", 1)),
            events_endpoint=d.get("events_endpoint"),
            snapshot_endpoint=d.get("snapshot_endpoint"),
            tiers=list(d.get("tiers") or ["device"]),
        )


@dataclass
class WorkerInfo:
    worker_id: str
    url: str
    model_name: str
    engine: EngineType = EngineType.SGLANG
    status: WorkerStatus = WorkerStatus.ACTIVE
    disagg_mode: DisaggMode = DisaggMode.MIXED

    # PD coordination data. Only populated when disagg_mode != MIXED.
    # Schema is owned by `infera.router.disagg_protocols`:
    #   { "protocol": "<name>", "params": { ...protocol-specific... } }
    # The router resolves `protocol` against the registry; `params` is
    # opaque to the router and consumed by the protocol's annotate_* /
    # extract_handoff methods.
    disagg_meta: dict[str, Any] = field(default_factory=dict)

    # KV-aware routing flat fields (from PR #10's KvEventClient/Policy).
    # Populated when worker enables KV cache event publishing.
    kv_events_endpoint: str | None = None  # e.g. "tcp://host:5557"
    kv_block_size: int | None = None  # SGLang's page_size, vLLM's block_size

    # Data parallel fields. Populated when the engine process runs dp_size > 1 ranks.
    dp_rank: int | None = None
    dp_size: int | None = None

    # Request transport the router must use to reach this worker:
    #   "http" (default) -> direct httpx POST to ``url``
    #   "nats"           -> publish to the worker's NATS request subject
    # Lets a fleet mix transports; the router picks per-worker.
    request_transport: str = "http"

    # Phase 1 nested KV management metadata (digest, canary, tiers, snapshot
    # endpoint). None for workers that didn't register the nested kv block;
    # they may still participate in PR #10's kv-aware routing via the flat
    # `kv_events_endpoint` / `kv_block_size` fields above.
    kv: KvRegistrationMetadata | None = None


class CanaryMismatch(Exception):
    """Raised by CanaryVerifier when a registering worker's tokenizer
    canary differs from the first-registered worker's for the same
    model_name. See 03-data-model.md § "Cross-worker tokenizer
    consistency (canary verification)"."""

    def __init__(
        self,
        model_name: str,
        first_worker_id: str,
        new_worker_id: str,
        expected: tuple[int, ...],
        got: tuple[int, ...],
    ) -> None:
        self.model_name = model_name
        self.first_worker_id = first_worker_id
        self.new_worker_id = new_worker_id
        self.expected = expected
        self.got = got
        # Show short prefix to keep log message tractable for long tokenizations.
        exp_prefix = list(expected[:6]) + (["..."] if len(expected) > 6 else [])
        got_prefix = list(got[:6]) + (["..."] if len(got) > 6 else [])
        super().__init__(
            f"tokenizer canary mismatch for model {model_name!r}: "
            f"worker {new_worker_id!r} produced {got_prefix} "
            f"vs first-registered worker {first_worker_id!r} {exp_prefix}"
        )


class CanaryVerifier:
    """Per-model_name canary registry.

    The first worker to register for a model_name sets the reference
    canary; subsequent workers must match. This catches operator
    misconfigurations (mismatched tokenizer versions across workers
    claiming the same model_name) before any request reaches them.

    The verifier is intentionally NOT shared with `compat_key`: workers
    serving distinct compat_keys for the same model_name (e.g., one in
    fp16 and one in fp8) still share a canary because they share the
    tokenizer.
    """

    def __init__(self) -> None:
        # model_name → (canary_tuple, first_worker_id)
        self._canary: dict[str, tuple[tuple[int, ...], str]] = {}

    def verify(
        self,
        *,
        model_name: str,
        worker_id: str,
        canary: list[int],
    ) -> None:
        """Record (if first) or verify against the reference. Raises
        CanaryMismatch on conflict."""
        key = tuple(canary)
        existing = self._canary.get(model_name)
        if existing is None:
            self._canary[model_name] = (key, worker_id)
            return
        if existing[0] == key:
            return
        raise CanaryMismatch(
            model_name=model_name,
            first_worker_id=existing[1],
            new_worker_id=worker_id,
            expected=existing[0],
            got=key,
        )

    def forget(self, model_name: str) -> None:
        """Drop the reference for `model_name` — used when no workers remain
        and we want a fresh canary on next registration."""
        self._canary.pop(model_name, None)

    def reference(self, model_name: str) -> tuple[int, ...] | None:
        """Return the currently-recorded canary for a model, or None."""
        entry = self._canary.get(model_name)
        return entry[0] if entry is not None else None


class WorkerPool:
    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}

    def add(self, worker: WorkerInfo) -> None:
        self._workers[worker.worker_id] = worker

    def remove(self, worker_id: str) -> None:
        self._workers.pop(worker_id, None)

    def get(self, worker_id: str) -> WorkerInfo | None:
        return self._workers.get(worker_id)

    def list_active(
        self,
        model: str | None = None,
        mode: DisaggMode | None = None,
    ) -> list[WorkerInfo]:
        workers = [w for w in self._workers.values() if w.status == WorkerStatus.ACTIVE]
        if model:
            workers = [w for w in workers if w.model_name == model]
        if mode:
            workers = [w for w in workers if w.disagg_mode == mode]
        return workers

    def list_all(self) -> list[WorkerInfo]:
        return list(self._workers.values())

    def __len__(self) -> int:
        return len(self._workers)
