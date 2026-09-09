###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Thin async NATS connection helper shared by the worker-side relay and
the router-side KV-event feed.

This is the transport substrate for migrating the KV-event plane off the
direct ZMQ pub/sub mesh (one SUB socket per server x worker) onto a NATS
broker (every producer publishes to a subject; any number of consumers
subscribe once). See docs discussion: the broker collapses the N x M mesh
to O(N+M) and lets multiple consumers (router, planner FPM, metrics) share
one stream.

Kept deliberately small: connect with sane reconnect defaults, publish raw
bytes, subscribe with an async callback. Higher-level framing (the msgpack
``KVEventBatch`` wire) is unchanged and travels in the NATS message payload.
"""

from __future__ import annotations

import base64
import logging
import os
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# Subject namespace for per-worker KV events: ``infera.kv.events.<worker_id>``.
# The worker_id (host:port) is appended so a single ``infera.kv.events.>``
# subscription on the router demultiplexes by publisher.
KV_EVENTS_SUBJECT_PREFIX = "infera.kv.events"

# JetStream stream capturing the live KV-event deltas for durable delivery.
KV_EVENTS_STREAM = "INFERA_KV_EVENTS"

DEFAULT_NATS_URL = "nats://127.0.0.1:4222"


def resolve_nats_url(explicit: str | None = None) -> str:
    """Pick the NATS URL: explicit arg > ``NATS_SERVER`` env > default.

    Mirrors dynamo's ``NATS_SERVER`` convention so operators can point both
    at the same broker.
    """
    return explicit or os.environ.get("NATS_SERVER") or DEFAULT_NATS_URL


def _token(worker_id: str) -> str:
    """Reversible, subject-/KV-safe base64url encoding of a worker_id
    (host:port has dots and a colon that collide with NATS dot tokens)."""
    return base64.urlsafe_b64encode(worker_id.encode()).decode().rstrip("=")


def subject_for_worker(worker_id: str, rank: int = 0) -> str:
    """``infera.kv.events.<token>.<rank>``. ``rank`` is the DP rank
    (0 for single-rank workers); for SGLang ``--dp-size`` each rank's events
    travel on a distinct subject so the router can score them separately."""
    return f"{KV_EVENTS_SUBJECT_PREFIX}.{_token(worker_id)}.{rank}"


def parse_kv_subject(subject: str) -> tuple[str, int] | None:
    """Inverse of :func:`subject_for_worker`: ``(worker_id, rank)`` or None."""
    prefix = KV_EVENTS_SUBJECT_PREFIX + "."
    if not subject.startswith(prefix):
        return None
    rest = subject[len(prefix) :]
    token, _, rank_s = rest.rpartition(".")
    if not token:
        return None
    worker_id = _decode_token(token)
    if worker_id is None:
        return None
    try:
        return worker_id, int(rank_s)
    except ValueError:
        return None


# KV bucket holding each worker's current router-side cache view (the set of
# chained block hashes). A cold-starting / reconnecting router reads this for
# an instant bootstrap instead of pulling an HTTP snapshot. JetStream-backed.
KV_VIEW_BUCKET = "infera_kv_view"


def js_store_failed(exc: BaseException) -> bool:
    """True when JetStream cannot open its on-disk msg blocks.

    nats-server reports this as JSStreamStoreFailedF (10077), often with an
    empty msg-block path when store_dir is unset or the FILE stream is
    corrupt. Live routing still works; FILE-backed KV/stream writes do not.
    """
    text = str(exc).lower()
    return (
        "10077" in text
        or "jsstreamstorefailed" in text.replace(" ", "")
        or "msg block file" in text
        or "opening msg block file" in text
    )


def kv_key_for_worker(worker_id: str, rank: int = 0) -> str:
    """KV bucket key ``<token>.<rank>`` (NATS KV keys allow ``.``)."""
    return f"{_token(worker_id)}.{rank}"


def parse_kv_key(key: str) -> tuple[str, int] | None:
    """Inverse of :func:`kv_key_for_worker`: ``(worker_id, rank)`` or None."""
    token, _, rank_s = key.rpartition(".")
    if not token:
        return None
    worker_id = _decode_token(token)
    if worker_id is None:
        return None
    try:
        return worker_id, int(rank_s)
    except ValueError:
        return None


def _decode_token(token: str) -> str | None:
    pad = "=" * (-len(token) % 4)
    try:
        return base64.urlsafe_b64decode(token + pad).decode()
    except (ValueError, UnicodeDecodeError):
        return None


class NatsBus:
    """Owns one NATS connection. Both the relay (publisher) and the router
    feed (subscriber) use an instance.

    ``nats-py`` is imported lazily so that importing this module (and thus
    infera) never hard-requires NATS when the operator stays on ZMQ.
    """

    def __init__(self, url: str | None = None) -> None:
        self._url = resolve_nats_url(url)
        self._nc = None  # nats.aio.client.Client

    @property
    def url(self) -> str:
        return self._url

    async def connect(self) -> None:
        if self._nc is not None:
            return
        try:
            import nats
        except ImportError as exc:  # pragma: no cover - dep guard
            raise RuntimeError(
                "nats-py is required for --kv-event-transport=nats; "
                "install it (pip install nats-py) or use --kv-event-transport=zmq"
            ) from exc
        # Reconnect forever; the KV plane tolerates gaps (snapshot reconcile
        # on the nested plane, and routing degrades gracefully to load-only).
        self._nc = await nats.connect(
            self._url,
            max_reconnect_attempts=-1,
            reconnect_time_wait=1.0,
            name="infera-kv",
        )
        logger.info("NATS connected: %s", self._url)

    async def publish(self, subject: str, payload: bytes) -> None:
        if self._nc is None:
            return
        await self._nc.publish(subject, payload)

    async def subscribe(self, subject: str, cb: Callable[[str, bytes], Awaitable[None]]):
        """Subscribe to ``subject`` (may be a wildcard like
        ``infera.kv.events.>``). ``cb(subject, payload)`` is awaited per
        message. Returns the subscription handle."""
        if self._nc is None:
            raise RuntimeError("NatsBus.subscribe called before connect()")

        async def _handler(msg) -> None:
            try:
                await cb(msg.subject, msg.data)
            except Exception:  # keep the subscription alive on handler errors
                logger.exception("NATS KV handler error on %s", msg.subject)

        return await self._nc.subscribe(subject, cb=_handler)

    def _event_stream_config(self, storage):
        from nats.js.api import DiscardPolicy, RetentionPolicy, StreamConfig

        return StreamConfig(
            name=KV_EVENTS_STREAM,
            subjects=[f"{KV_EVENTS_SUBJECT_PREFIX}.>"],
            retention=RetentionPolicy.LIMITS,
            storage=storage,
            discard=DiscardPolicy.OLD,
            max_bytes=256 * 1024 * 1024,
            max_msgs=1_000_000,
        )

    async def _install_event_stream(self, js, cfg) -> Exception | None:
        try:
            await js.add_stream(cfg)
            return None
        except Exception as add_exc:
            try:
                await js.update_stream(cfg)
                return None
            except Exception as update_exc:
                return update_exc if js_store_failed(update_exc) else add_exc

    async def ensure_event_stream(self) -> None:
        """Idempotently create the JetStream stream that captures live KV-event
        deltas (infera.kv.events.>). Bounded by bytes/msgs so it can't grow
        unbounded; OLD messages are discarded first. Both relay (publisher) and
        router (subscriber) call this so neither races ahead of the other.

        FILE is preferred. A 10077 / empty msg-block failure means the
        broker's filestore is unusable, so the stream is deleted and recreated
        in MEMORY rather than leaving kv-aware with a stream that cannot
        accept publishes.
        """
        if self._nc is None:
            return
        from nats.js.api import StorageType

        js = self._nc.jetstream()
        err = await self._install_event_stream(js, self._event_stream_config(StorageType.FILE))
        if err is None or not js_store_failed(err):
            return
        logger.warning(
            "KV event stream FILE store failed (%s); recreating %s in MEMORY",
            err,
            KV_EVENTS_STREAM,
        )
        try:
            await js.delete_stream(KV_EVENTS_STREAM)
        except Exception:
            pass
        mem_err = await self._install_event_stream(
            js, self._event_stream_config(StorageType.MEMORY)
        )
        if mem_err is not None:
            logger.warning("KV event stream MEMORY recreate failed: %s", mem_err)

    async def js_publish(self, subject: str, payload: bytes) -> None:
        """Persistent publish onto the JetStream stream (acked, flow-controlled,
        no silent drop on slow consumers — unlike core NATS publish)."""
        if self._nc is None:
            return
        await self._nc.jetstream().publish(subject, payload)

    async def js_subscribe(self, subject: str, cb: Callable[[str, bytes], Awaitable[None]]):
        """Ephemeral JetStream push subscription replaying the FULL retained
        stream from the beginning, then live deltas.

        DeliverPolicy.ALL (not NEW) is required for KV-aware correctness: a
        ``BlockStored`` event references its ``parent_block_hash``, so the
        router must see a prefix's root block before any child or the whole
        chain is dropped (``_on_block_stored`` skips events with an unknown
        parent). With NEW, a router that subscribes after a worker has already
        cached prefixes never sees those roots and builds an empty view. ALL
        replays the durable history in order so the radix view is rebuilt
        exactly, and re-delivers on reconnect (self-healing). Ephemeral +
        per-process => every router replica gets its own full replay."""
        if self._nc is None:
            raise RuntimeError("NatsBus.js_subscribe called before connect()")
        from nats.js.api import ConsumerConfig, DeliverPolicy

        js = self._nc.jetstream()

        async def _handler(msg) -> None:
            try:
                await cb(msg.subject, msg.data)
            except Exception:
                logger.exception("JetStream KV handler error on %s", msg.subject)

        return await js.subscribe(
            subject,
            cb=_handler,
            config=ConsumerConfig(deliver_policy=DeliverPolicy.ALL),
        )

    async def _create_kv_view(self, js, storage):
        from nats.js.api import KeyValueConfig

        return await js.create_key_value(KeyValueConfig(bucket=KV_VIEW_BUCKET, storage=storage))

    async def kv_view_store(self):
        """Return the JetStream KV bucket holding per-worker cache views,
        creating it on first use. Returns None if NATS isn't connected."""
        if self._nc is None:
            return None
        js = self._nc.jetstream()
        try:
            return await js.key_value(KV_VIEW_BUCKET)
        except Exception:
            return await self._create_kv_view_with_fallback(js)

    async def _create_kv_view_with_fallback(self, js):
        from nats.js.api import StorageType

        try:
            return await self._create_kv_view(js, StorageType.FILE)
        except Exception as exc:
            if js_store_failed(exc):
                logger.warning("KV view bucket FILE create failed (%s); retrying in MEMORY", exc)
                try:
                    await js.delete_key_value(KV_VIEW_BUCKET)
                except Exception:
                    pass
                return await self._create_kv_view(js, StorageType.MEMORY)
            try:
                return await js.key_value(KV_VIEW_BUCKET)
            except Exception:
                return await self._create_kv_view(js, StorageType.MEMORY)

    async def rebuild_kv_view_store(self):
        """Drop a filestore-broken KV bucket and recreate it.

        Called after a put returns 10077. FILE is retried first; MEMORY is
        the fallback so a cold router can still seed from live workers.
        """
        if self._nc is None:
            return None
        js = self._nc.jetstream()
        try:
            await js.delete_key_value(KV_VIEW_BUCKET)
        except Exception as exc:
            logger.warning("KV view bucket delete failed (continuing recreate): %s", exc)
        return await self._create_kv_view_with_fallback(js)

    async def close(self) -> None:
        if self._nc is not None:
            try:
                await self._nc.drain()
            except Exception:
                pass
            self._nc = None
