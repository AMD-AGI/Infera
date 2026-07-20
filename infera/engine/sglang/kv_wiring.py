###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SGLang worker-side KV management wiring.

Builds and starts the per-worker KV pipeline:
  - SnapshotProducer (mirrors emitted events)
  - KvEventPublisher (ZMQ PUB bound at events_endpoint)
  - KvEventProbe (engine-agnostic coalescer)
  - small FastAPI snapshot server (mounts make_snapshot_router)
  - attach_to_radix_cache on a *running* SGLang RadixCache (best-effort)

Also computes the `KvRegistrationMetadata` that goes in the worker's
etcd registration: tokenizer_digest, tokenizer_canary, index_block_size,
endpoints, etc.

Failure modes:
  - Tokenizer / digest computation fails (model files not where we
    expected): worker still starts, just without a kv block (no
    prefix-cache participation, but routing works).
  - RadixCache attach hook fails (SGLang version skew): logs an error
    and continues; the events plane will be empty but the worker is
    functional. Reconciler will pull empty snapshots; cache hit rate
    will be 0%.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import uvicorn

from infera.common.worker_pool import KvRegistrationMetadata
from infera.kv.api import make_snapshot_router
from infera.kv.compat_key import compute_tokenizer_digest, tokenize_canary
from infera.kv.probe import KvEventProbe
from infera.kv.publisher import KvEventPublisher
from infera.kv.snapshot import SnapshotProducer
from infera.kv.types import Tier

logger = logging.getLogger(__name__)


@dataclass
class SglangKvWiring:
    """Bundle of the worker-side components, returned by `build_and_start`."""

    producer: SnapshotProducer
    publisher: KvEventPublisher
    probe: KvEventProbe
    snapshot_task: asyncio.Task
    snapshot_originals: dict[str, Any]  # for detach_from_radix_cache
    radix_cache: Any | None  # the cache we attached to, or None
    metadata: KvRegistrationMetadata

    async def stop(self) -> None:
        from infera.engine.sglang.kv_probe import detach_from_radix_cache

        if self.snapshot_originals and self.radix_cache is not None:
            try:
                detach_from_radix_cache(self.snapshot_originals, self.radix_cache)
            except Exception:
                logger.debug("detach_from_radix_cache during shutdown failed (probably OK)")
        try:
            await self.publisher.stop()
        except Exception:
            logger.exception("publisher stop failed")
        if not self.snapshot_task.done():
            self.snapshot_task.cancel()
            try:
                await self.snapshot_task
            except (asyncio.CancelledError, Exception):
                pass


def _resolve_tokenizer_path(model_path_or_id: str, *, trust_remote_code: bool = False) -> str:
    """Find the file or directory whose hash defines `tokenizer_digest`.

    Cheap heuristic: if `model_path_or_id` is a local path, hash the
    `tokenizer.json` file inside (or the whole tokenizer subdir as a
    fallback). If it's a HF id, we hash the cached copy by loading it
    through `transformers` and finding its file. Anything fancier
    (e.g. multi-file tokenizer with shared resources) is documented in
    15-infera-kvd.md and can be added when needed.
    """
    import os

    if os.path.isdir(model_path_or_id):
        # Prefer the canonical tokenizer.json if present; fall back to
        # hashing the directory (compute_tokenizer_digest handles both).
        tokfile = os.path.join(model_path_or_id, "tokenizer.json")
        return tokfile if os.path.isfile(tokfile) else model_path_or_id
    # HF id: load the tokenizer to populate the cache, then find its
    # tokenizer.json. transformers stores them under
    # ~/.cache/huggingface/hub.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path_or_id, trust_remote_code=trust_remote_code)
    tokfile = getattr(tok, "vocab_file", None) or getattr(tok, "name_or_path", None)
    # Walk up to the snapshot dir and look for tokenizer.json
    if isinstance(tokfile, str) and os.path.isfile(tokfile):
        candidate = os.path.join(os.path.dirname(tokfile), "tokenizer.json")
        if os.path.isfile(candidate):
            return candidate
        return tokfile
    # Last resort: hash a deterministic representation of the tokenizer
    # state itself. This is less stable but better than failing.
    return model_path_or_id


def compute_kv_metadata(
    *,
    model_id: str,
    tokenizer_path: str | None = None,
    engine_block_size: int,
    index_block_size: int,
    events_endpoint: str,
    snapshot_endpoint: str,
    tiers: list[str] | None = None,
    trust_remote_code: bool = False,
) -> KvRegistrationMetadata:
    """Build the kv block. Loads the tokenizer once to compute digest +
    canary. Raises if tokenizer / digest can't be computed — caller
    decides whether to skip the kv block or fail-fast.

    `model_id` is the display/routing name (may be a served-model-name
    alias); digest + canary load from `tokenizer_path` (the real local
    path) so they match the router's. Falls back to `model_id`.

    `trust_remote_code` must match the engine's setting — models with a
    custom tokenizer (e.g. tiktoken via `auto_map`) won't load otherwise.
    """
    from transformers import AutoTokenizer

    tok_source = tokenizer_path or model_id
    digest_path = _resolve_tokenizer_path(tok_source, trust_remote_code=trust_remote_code)
    digest = compute_tokenizer_digest(digest_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=trust_remote_code)
    canary = tokenize_canary(tokenizer)
    return KvRegistrationMetadata(
        engine_block_size=engine_block_size,
        index_block_size=index_block_size,
        tokenizer=model_id,
        tokenizer_digest=digest,
        tokenizer_canary=list(canary),
        supports_events=True,
        event_version=1,
        events_endpoint=events_endpoint,
        snapshot_endpoint=snapshot_endpoint,
        tiers=list(tiers or ["device"]),
    )


async def _start_snapshot_http(
    producer: SnapshotProducer,
    host: str,
    port: int,
) -> asyncio.Task:
    """Run a small FastAPI app on (host, port) serving /v1/kv-snapshot.

    Returns the asyncio.Task running uvicorn; caller cancels on shutdown.
    """
    from fastapi import FastAPI

    app = FastAPI(title="infera-sglang-snapshot")
    app.include_router(make_snapshot_router(producer))
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    return asyncio.create_task(server.serve(), name="infera-sglang-snapshot-http")


def resolve_advertise_endpoint(bind: str, host: str, scheme: str = "tcp") -> str:
    """Convert a 0.0.0.0 bind into something a peer can connect to.

    bind = "tcp://0.0.0.0:5557", host = "10.0.0.5" -> "tcp://10.0.0.5:5557"
    bind = "tcp://10.0.0.5:5557", host = "10.99.99.99" -> "tcp://10.0.0.5:5557"
    Only the host portion is rewritten; scheme and port preserved.
    """
    if "://" not in bind:
        bind = f"{scheme}://{bind}"
    head, rest = bind.split("://", 1)
    if ":" in rest:
        host_part, port_part = rest.rsplit(":", 1)
    else:
        host_part, port_part = rest, ""
    if host_part in ("0.0.0.0", "*", "", "::"):
        host_part = host
    return f"{head}://{host_part}:{port_part}" if port_part else f"{head}://{host_part}"


def _find_radix_cache(engine: Any) -> Any | None:
    """Try a few known attribute paths to locate a running SGLang
    RadixCache instance. Returns None if we can't find it; caller logs
    and continues without the probe attach.

    SGLang versions don't agree on where the RadixCache lives. This
    helper is the one place we centralize the engine-internals
    introspection.
    """
    candidates = (
        "tree_cache",  # SGLang scheduler
        "radix_cache",
        "_radix_cache",
    )
    for attr in candidates:
        rc = getattr(engine, attr, None)
        if rc is not None:
            return rc
    return None


async def build_and_start(
    *,
    model_id: str,
    tokenizer_path: str | None = None,
    engine_block_size: int,
    index_block_size: int,
    publisher_id: str,
    events_bind: str,
    events_advertise: str,
    snapshot_host: str,
    snapshot_port: int,
    snapshot_advertise: str,
    radix_cache: Any | None = None,
    trust_remote_code: bool = False,
) -> SglangKvWiring:
    """Assemble the worker-side KV plane and start it.

    `events_bind`     — what the publisher binds to (e.g. tcp://0.0.0.0:5557)
    `events_advertise`— what we register so servers can connect (e.g. tcp://<host>:5557)
    `snapshot_host`/`snapshot_port` — what the snapshot uvicorn binds to
    `snapshot_advertise` — full base URL the server should pull from
    `radix_cache` — the SGLang RadixCache instance (or None to skip attach)
    """
    metadata = compute_kv_metadata(
        model_id=model_id,
        tokenizer_path=tokenizer_path,
        engine_block_size=engine_block_size,
        index_block_size=index_block_size,
        events_endpoint=events_advertise,
        snapshot_endpoint=snapshot_advertise,
        trust_remote_code=trust_remote_code,
    )

    producer = SnapshotProducer(
        publisher_id=publisher_id,
        publisher_type="worker",
        index_block_size=index_block_size,
    )
    publisher = KvEventPublisher(
        bind_endpoint=events_bind,
        publisher_id=publisher_id,
        publisher_type="worker",
        index_block_size=index_block_size,
    )
    await publisher.start()

    probe = KvEventProbe(
        publisher=publisher,
        snapshot_producer=producer,
        model_name=model_id,
        compat_key=metadata.tokenizer_digest,
        index_block_size=index_block_size,
        tier=Tier.DEVICE,
    )

    # Attach to SGLang's RadixCache. If we couldn't locate one, log and
    # continue — events plane will be empty but worker is functional.
    originals: dict[str, Any] = {}
    if radix_cache is not None:
        from infera.engine.sglang.kv_probe import attach_to_radix_cache

        try:
            originals = attach_to_radix_cache(probe, radix_cache)
            logger.info("attached KvEventProbe to SGLang RadixCache")
        except Exception:
            logger.exception(
                "failed to attach probe to RadixCache; events plane will be empty for this worker"
            )
    else:
        logger.warning(
            "no RadixCache instance available; probe will not emit device-tier events. "
            "Snapshot reconciliation still works once the engine emits events on its own."
        )

    snapshot_task = await _start_snapshot_http(producer, snapshot_host, snapshot_port)

    return SglangKvWiring(
        producer=producer,
        publisher=publisher,
        probe=probe,
        snapshot_task=snapshot_task,
        snapshot_originals=originals,
        radix_cache=radix_cache,
        metadata=metadata,
    )
