###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""NATS per-instance request transport (Model B) for the infera request path.

This is the dynamo-style alternative to the default direct-HTTP forwarding in
``infera.router.mixed``: the router still *selects* a worker with the unchanged
policy (round-robin / kv-aware), then instead of ``httpx.POST {worker.url}`` it
publishes the request onto that specific worker's NATS subject and streams the
reply back over a reply inbox. Selection/scoring is transport-agnostic, so
kv-aware routing works identically — NATS is only the transport.

Wire protocol (one request -> N reply messages on a fresh inbox):

  request  (server -> ``infera.req.<token(worker_id)>``, reply=<inbox>):
      JSON  {"path": str, "stream": bool, "headers": {..}|null, "body": {..},
             "migratable": bool}

  reply    (worker -> <inbox>), framed by the ``rs-type`` header:
      data : payload = raw response bytes (an SSE chunk, or the full JSON body)
      done : payload = b"" , header ``rs-status`` = HTTP status code
      error: payload = utf-8 error text  (transport/proxy failure)

``migratable`` says the router can continue this generation on another worker
(see infera.router.migration), which is what lets a draining worker hand it back
early instead of holding the shutdown open. It is a promise about the *router*,
so the worker must not assume it: without it, a request is drained the slow way
and severing it early would just be an error the client did not have to see.

The worker side proxies to its own local engine HTTP (127.0.0.1:<port>), so the
engine itself is unchanged; the consumer is a thin task started alongside it
(like the KV relay).

Optional admission throttle (default OFF): set ``INFERA_NATS_REQ_MAX_PENDING``
> 0 on both server and worker pods to make the request path JetStream-backed
(WorkQueue stream ``INFERA_REQUESTS``, one durable consumer per worker). The
router then queries that worker's consumer backlog (``num_pending`` +
``num_ack_pending``) before dispatching and returns HTTP 429 when it has
reached the limit. Unset / 0 keeps the pure core-NATS path unchanged.

Two request timeouts (either expiring => 504 + the request's inbox is published
to ``infera.cancel.<worker>`` (resent a few times); the worker cancels the
matching in-flight proxy task, tearing down the engine connection so it stops
generating; the same cancel fires on client disconnect):

  - IDLE (inactivity / stall): ``INFERA_NATS_REQ_IDLE_TIMEOUT`` (seconds,
    default 900) bounds the wait for the *next* reply chunk. Reset on every
    chunk, so a steadily-streaming long request never trips it; only a stall
    does. NOT an overall deadline. 0 = off.

  - TOTAL (overall): ``INFERA_NATS_REQ_MAX_DURATION`` (seconds, default 0 =
    OFF) caps the whole request's wall-clock regardless of token flow — for
    runaway / very long requests. Enforced router-side (interrupt + 504 + no
    result) and worker-side (hard local abort, also a backstop for a lost
    cancel signal).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import httpx

from infera.kv.nats_bus import _token, resolve_nats_url

logger = logging.getLogger(__name__)

REQUEST_SUBJECT_PREFIX = "infera.req"

# JetStream stream backing the request path when admission throttling is on.
# WorkQueue retention => a request is removed once its worker acks it, so the
# stream's per-worker consumer pending count is a live backlog gauge.
REQUEST_STREAM = "INFERA_REQUESTS"

# How often a draining worker re-checks its JetStream backlog. Short, because
# the messages are already accepted and every poll is one round-trip.
_QUEUED_POLL_INTERVAL_S = 0.2

# Reply framing headers.
HDR_TYPE = "rs-type"
HDR_STATUS = "rs-status"
# Reply inbox carried as a header in JetStream mode (a JS-delivered message's
# ``reply`` field is the ack subject, not the publisher's inbox).
HDR_INBOX = "rs-inbox"
TYPE_DATA = "data"
TYPE_DONE = "done"
TYPE_ERROR = "error"

# The error payload a worker sends when it hands a generation back at the start
# of a drain rather than holding the shutdown open for it. The router reads it
# to tell "this worker is leaving on purpose" apart from a worker that broke,
# which are the same event to the client but not to an operator reading metrics.
DRAINING_NOTICE = b"infera: worker draining"

# Throttle knob (single variable, default OFF). When > 0, the per-instance
# request path is JetStream-backed and the router refuses to dispatch to a
# worker whose backlog (num_pending + num_ack_pending on its request consumer)
# has reached this many messages. 0 / unset => disabled: the transport stays
# pure core-NATS exactly as before.
MAX_PENDING_ENV = "INFERA_NATS_REQ_MAX_PENDING"

# Timeout #1 — IDLE (inactivity / stall) timeout: max seconds the router waits
# for the *next* reply chunk (covers first-byte / TTFT and inter-chunk stalls;
# reset on every chunk, so a steadily-streaming long request never trips it —
# this is NOT an overall request deadline, see MAX_DURATION_ENV for that). On
# expiry the router returns 504 and signals the worker to abort. Default 900s
# (15 min); set 0 to disable (wait forever).
IDLE_TIMEOUT_ENV = "INFERA_NATS_REQ_IDLE_TIMEOUT"
DEFAULT_IDLE_TIMEOUT_S = 900

# Subject a worker listens on for "abort this in-flight request" signals; the
# payload is the request's reply inbox (already unique per request).
CANCEL_SUBJECT_PREFIX = "infera.cancel"

# Timeout #2 — TOTAL (overall) request timeout: hard wall-clock cap (s) for the
# whole request measured from dispatch, regardless of token flow. For runaway /
# very long requests: on expiry the router interrupts the token stream, returns
# no result (504) and cancels the worker; the worker also enforces the same cap
# locally as a backstop if the cancel signal is lost. Default 0 = OFF; set > 0
# to enable.
MAX_DURATION_ENV = "INFERA_NATS_REQ_MAX_DURATION"

# Server resends the cancel signal a few times (best-effort) so a worker that is
# briefly reconnecting at timeout/disconnect still gets it.
_CANCEL_RESEND = 3
_CANCEL_RESEND_GAP_S = 1.0

# Long ack_wait so a slow generation still counts as in-flight (num_ack_pending)
# instead of silently aging out of the backlog gauge.
_REQUEST_ACK_WAIT_S = 3600


def max_pending_limit() -> int:
    """Per-worker in-NATS request backlog limit; 0/invalid => throttling off."""
    try:
        return max(0, int(os.environ.get(MAX_PENDING_ENV, "0") or 0))
    except ValueError:
        return 0


def request_idle_timeout() -> float:
    """Idle (inactivity) timeout (s). Default 900; 0/invalid => off (wait forever)."""
    raw = os.environ.get(IDLE_TIMEOUT_ENV)
    if raw is None:
        return float(DEFAULT_IDLE_TIMEOUT_S)
    try:
        t = float(raw)
    except ValueError:
        return float(DEFAULT_IDLE_TIMEOUT_S)
    return t if t > 0 else 0.0


def request_max_duration() -> float:
    """Total (overall) request timeout (s), enforced both router- and worker-side.
    0/invalid => off."""
    try:
        d = float(os.environ.get(MAX_DURATION_ENV, "0") or 0)
    except ValueError:
        return 0.0
    return d if d > 0 else 0.0


def cancel_subject(worker_id: str) -> str:
    """``infera.cancel.<token(worker_id)>`` — abort signals for a worker."""
    return f"{CANCEL_SUBJECT_PREFIX}.{_token(worker_id)}"


def _resolve_pending(flag: int | None) -> int:
    """flag (CLI) > env > built-in default. Backlog limit; <=0 disables."""
    if flag is None:
        return max_pending_limit()
    try:
        return max(0, int(flag))
    except (TypeError, ValueError):
        return max_pending_limit()


def _resolve_idle_timeout(flag: float | None) -> float:
    if flag is None:
        return request_idle_timeout()
    try:
        f = float(flag)
    except (TypeError, ValueError):
        return request_idle_timeout()
    return f if f > 0 else 0.0


def _resolve_max_duration(flag: float | None) -> float:
    if flag is None:
        return request_max_duration()
    try:
        f = float(flag)
    except (TypeError, ValueError):
        return request_max_duration()
    return f if f > 0 else 0.0


def request_subject(worker_id: str) -> str:
    """Per-instance request subject ``infera.req.<token(worker_id)>``.

    Per-instance (not a shared queue group) so the router keeps full control of
    which worker handles each request — exactly what kv-aware needs.
    """
    return f"{REQUEST_SUBJECT_PREFIX}.{_token(worker_id)}"


def request_durable(worker_id: str) -> str:
    """Durable JetStream consumer name for a worker's request subject."""
    return _token(worker_id)


async def _ensure_request_stream(js) -> None:
    """Idempotently create the WorkQueue stream covering ``infera.req.>``.
    Memory-backed: requests are ephemeral, so losing them on a NATS restart is
    acceptable (the client gets an error and retries)."""
    from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

    cfg = StreamConfig(
        name=REQUEST_STREAM,
        subjects=[f"{REQUEST_SUBJECT_PREFIX}.>"],
        retention=RetentionPolicy.WORK_QUEUE,
        storage=StorageType.MEMORY,
        discard=DiscardPolicy.OLD,
        max_msgs=1_000_000,
    )
    try:
        await js.add_stream(cfg)
    except Exception:
        try:
            await js.update_stream(cfg)
        except Exception:
            pass


async def _connect(url: str | None, name: str):
    try:
        import nats
    except ImportError as exc:  # pragma: no cover - dep guard
        raise RuntimeError(
            "nats-py is required for --request-transport=nats (pip install nats-py)"
        ) from exc
    return await nats.connect(
        resolve_nats_url(url),
        max_reconnect_attempts=-1,
        reconnect_time_wait=1.0,
        name=name,
    )


class NatsRequestClient:
    """Server-side: send a selected worker a request over NATS and stream back
    its reply. One long-lived connection shared across requests."""

    def __init__(
        self,
        url: str | None = None,
        *,
        max_pending: int | None = None,
        idle_timeout: float | None = None,
        max_duration: float | None = None,
    ) -> None:
        self._url = url
        self._nc = None
        self._js = None
        # flag (entry-point CLI) > env > built-in default.
        self._max_pending = _resolve_pending(max_pending)
        self._idle_timeout = _resolve_idle_timeout(idle_timeout)
        self._max_duration = _resolve_max_duration(max_duration)
        # Short TTL cache so a burst of requests doesn't issue a consumer_info
        # round-trip each: worker_id -> (monotonic_ts, backlog).
        self._pending_cache: dict[str, tuple[float, int]] = {}
        self._cache_ttl_s = 0.2

    @property
    def throttle_enabled(self) -> bool:
        return self._max_pending > 0

    async def start(self) -> None:
        if self._nc is None:
            self._nc = await _connect(self._url, "infera-router-req")
            if self._max_pending > 0:
                self._js = self._nc.jetstream()
                await _ensure_request_stream(self._js)
                logger.info(
                    "NATS request transport connected (JetStream throttle on): %s max_pending=%d",
                    resolve_nats_url(self._url),
                    self._max_pending,
                )
            else:
                logger.info("NATS request transport connected: %s", resolve_nats_url(self._url))

    async def aclose(self) -> None:
        if self._nc is not None:
            try:
                await self._nc.drain()
            except Exception:
                pass
            self._nc = None

    async def admit(self, worker_id: str) -> bool:
        """Admission check for the per-instance request throttle.

        Returns True (admit) when throttling is off, when the backlog can't be
        measured (fail-open: the worker may not have a JS consumer yet), or when
        the worker's in-NATS backlog is below the limit; False to refuse."""
        if self._max_pending <= 0 or self._js is None:
            return True
        backlog = await self._pending(worker_id)
        if backlog is None:
            return True
        return backlog < self._max_pending

    async def _pending(self, worker_id: str) -> int | None:
        now = time.monotonic()
        cached = self._pending_cache.get(worker_id)
        if cached is not None and (now - cached[0]) < self._cache_ttl_s:
            return cached[1]
        try:
            info = await self._js.consumer_info(REQUEST_STREAM, request_durable(worker_id))
        except Exception:
            return None  # no consumer yet / transient -> fail-open
        backlog = int(getattr(info, "num_pending", 0) or 0) + int(
            getattr(info, "num_ack_pending", 0) or 0
        )
        self._pending_cache[worker_id] = (now, backlog)
        return backlog

    async def stream(self, worker_id: str, payload: dict):
        """Publish the request to the worker and async-yield (kind, status, data)
        tuples: ("data", None, bytes) per chunk, then ("done", status, b"") or
        ("error", None, bytes). The caller turns this into an HTTP response."""
        if self._nc is None:
            raise RuntimeError("NatsRequestClient.stream called before start()")
        inbox = self._nc.new_inbox()
        queue: asyncio.Queue = asyncio.Queue()
        sub = await self._nc.subscribe(inbox)

        async def pump():
            try:
                async for msg in sub.messages:
                    hdrs = msg.headers or {}
                    rtype = hdrs.get(HDR_TYPE, TYPE_DATA)
                    if rtype == TYPE_DONE:
                        status = int(hdrs.get(HDR_STATUS, "200") or "200")
                        await queue.put((TYPE_DONE, status, b""))
                        return
                    if rtype == TYPE_ERROR:
                        await queue.put((TYPE_ERROR, None, msg.data))
                        return
                    await queue.put((TYPE_DATA, None, msg.data))
            except Exception as exc:  # subscription torn down
                await queue.put((TYPE_ERROR, None, str(exc).encode()))

        pump_task = asyncio.create_task(pump(), name="nats-reply-pump")
        done_seen = False
        try:
            body = json.dumps(payload).encode()
            if self._js is not None:
                # JetStream mode: the consumer can't read msg.reply (that's the
                # ack subject), so carry the inbox in a header. The reply itself
                # still flows back over the core inbox subscription above.
                await self._js.publish(request_subject(worker_id), body, headers={HDR_INBOX: inbox})
            else:
                await self._nc.publish(request_subject(worker_id), body, reply=inbox)
            start = time.monotonic()
            while True:
                # Two independent deadlines:
                #  - idle (self._idle_timeout): max gap to the NEXT chunk;
                #  - total (self._max_duration): hard cap on the whole request.
                # Each wait_for budget is the smaller of (idle, remaining total).
                # Either expiring => 504 + finally cancels the worker.
                waits: list[float] = []
                if self._idle_timeout > 0:
                    waits.append(self._idle_timeout)
                total_left = None
                if self._max_duration > 0:
                    total_left = self._max_duration - (time.monotonic() - start)
                    if total_left <= 0:
                        yield (
                            TYPE_ERROR,
                            504,
                            f"nats request exceeded total timeout "
                            f"{self._max_duration:.0f}s".encode(),
                        )
                        return
                    waits.append(total_left)
                budget = min(waits) if waits else None
                try:
                    item = (
                        await asyncio.wait_for(queue.get(), budget)
                        if budget is not None
                        else await queue.get()
                    )
                except asyncio.TimeoutError:
                    # Which deadline fired? total if we've reached the overall cap.
                    if self._max_duration > 0 and (time.monotonic() - start) >= self._max_duration:
                        msg_txt = f"nats request exceeded total timeout {self._max_duration:.0f}s"
                    else:
                        msg_txt = (
                            f"nats request stalled: no reply within "
                            f"{self._idle_timeout:.0f}s (idle timeout)"
                        )
                    yield (TYPE_ERROR, 504, msg_txt.encode())
                    return
                yield item
                if item[0] in (TYPE_DONE, TYPE_ERROR):
                    done_seen = True
                    return
        finally:
            pump_task.cancel()
            # If the request didn't finish cleanly (timeout or client disconnect),
            # tell the worker to abort so it stops burning GPU on a reply nobody
            # is reading. Keyed by the reply inbox (unique per request). Resent a
            # few times in the background to survive a worker briefly reconnecting.
            if not done_seen:
                asyncio.create_task(self._send_cancel(worker_id, inbox), name="nats-req-cancel")
            try:
                await sub.unsubscribe()
            except Exception:
                pass

    async def _send_cancel(self, worker_id: str, inbox: str) -> None:
        subject = cancel_subject(worker_id)
        for i in range(_CANCEL_RESEND):
            try:
                if self._nc is not None:
                    await self._nc.publish(subject, inbox.encode())
            except Exception:
                pass
            if i + 1 < _CANCEL_RESEND:
                await asyncio.sleep(_CANCEL_RESEND_GAP_S)


class NatsRequestServer:
    """Worker-side: subscribe to this worker's request subject, proxy each
    request to the local engine HTTP, and stream the reply back to the inbox."""

    def __init__(
        self,
        worker_id: str,
        local_port: int,
        url: str | None = None,
        local_host: str = "127.0.0.1",
        *,
        max_pending: int | None = None,
        idle_timeout: float | None = None,
        max_duration: float | None = None,
    ) -> None:
        self._worker_id = worker_id
        self._base = f"http://{local_host}:{local_port}"
        self._url = url
        self._nc = None
        self._sub = None
        self._js = None  # set only under the throttle (JetStream-backed path)
        self._cancel_sub = None
        self._http: httpx.AsyncClient | None = None
        # flag (entry-point CLI) > env > built-in default.
        self._max_pending = _resolve_pending(max_pending)
        self._idle_timeout = _resolve_idle_timeout(idle_timeout)
        self._max_duration = _resolve_max_duration(max_duration)
        # In-flight proxy tasks keyed by reply inbox, so a cancel signal can
        # abort the matching request's engine call.
        self._inflight: dict[str, asyncio.Task] = {}
        # Inboxes whose router can continue the generation on another worker.
        self._migratable: set[str] = set()
        # Inboxes already told the worker is draining, so the cancellation that
        # follows does not report itself a second time.
        self._handed_back: set[str] = set()

    async def start(self) -> None:
        self._nc = await _connect(self._url, "infera-worker-req")
        # Read timeout mirrors the router's idle budget so a stalled engine
        # is abandoned locally too (worker then replies error + acks).
        read_to = self._idle_timeout if self._idle_timeout > 0 else None
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(read_to, connect=30.0))
        # Listen for abort signals (payload = the request's reply inbox).
        self._cancel_sub = await self._nc.subscribe(
            cancel_subject(self._worker_id), cb=self._on_cancel
        )
        subject = request_subject(self._worker_id)
        if self._max_pending > 0:
            # JetStream-backed: a durable WorkQueue consumer on this worker's
            # subject so the router can query the backlog (num_pending +
            # num_ack_pending) before dispatching. Explicit ack on done, no
            # redelivery (max_deliver=1) so a crash mid-request can't re-run a
            # generation; the in-flight count drops when the worker acks.
            from nats.js.api import AckPolicy, ConsumerConfig

            js = self._nc.jetstream()
            self._js = js
            await _ensure_request_stream(js)
            self._sub = await js.subscribe(
                subject,
                durable=request_durable(self._worker_id),
                cb=self._on_request,
                manual_ack=True,
                config=ConsumerConfig(
                    ack_policy=AckPolicy.EXPLICIT,
                    ack_wait=_REQUEST_ACK_WAIT_S,
                    max_deliver=1,
                ),
            )
            logger.info(
                "NATS request consumer up (JetStream, max_pending=%d): subject=%s -> %s",
                self._max_pending,
                subject,
                self._base,
            )
        else:
            self._sub = await self._nc.subscribe(subject, cb=self._on_request)
            logger.info(
                "NATS request consumer up: subject=%s -> %s",
                subject,
                self._base,
            )

    async def stop(self, *, drain: bool = False, drain_timeout: float = 0.0) -> None:
        """Tear down the consumer. With ``drain=True`` (graceful shutdown for a
        rolling upgrade) stop accepting NEW requests first, then let in-flight
        generations finish for up to ``drain_timeout`` seconds before cancelling
        any leftovers, so a worker being rolled does not sever active streams.
        With ``drain=False`` (default) in-flight tasks are cancelled at once."""
        deadline = time.monotonic() + drain_timeout if (drain and drain_timeout > 0) else None

        # 0. Under the throttle, requests land in a WorkQueue stream and are
        # pulled from it -- so one can be accepted, queued, and invisible to
        # `_inflight`, which only tracks what has been *delivered* here.
        # `unsubscribe()` discards remaining messages ("remaining messages will
        # be discarded", nats-py), so closing the door first would strand work
        # the router already handed us: the client waits out the full idle
        # timeout (900s by default) for a reply nobody will ever send.
        if deadline is not None:
            await self._await_queued(deadline)

        # 1. Stop accepting NEW requests immediately so nothing new lands while
        # we drain (unsubscribe the request subject first).
        if self._sub is not None:
            try:
                await self._sub.unsubscribe()
            except Exception:
                pass
            self._sub = None
        # 2. Optionally let in-flight requests finish (bounded by drain_timeout).
        if deadline is not None:
            inflight = [t for t in self._inflight.values() if not t.done()]
            if inflight:
                remaining = max(0.0, deadline - time.monotonic())
                logger.info(
                    "draining %d in-flight NATS request(s), up to %.0fs",
                    len(inflight),
                    remaining,
                )
                _done, pending = await asyncio.wait(inflight, timeout=remaining)
                if pending:
                    logger.warning(
                        "drain timeout; %d request(s) did not finish in time", len(pending)
                    )
        # 2b. Whatever is still running was about to be cancelled. Hand the
        # resumable part back to the router instead: migration is the
        # alternative to cutting these, not a shortcut past the wait above.
        # Moving a generation costs the next worker a re-read of everything
        # produced so far, so it is worth doing only once the request has been
        # given the window it was promised -- or, with no window configured,
        # once it is clear there will not be one.
        if drain:
            await self._hand_back_migratable()
        # 3. Cancel whatever is left (all of it on the non-drain path).
        for task in list(self._inflight.values()):
            if not task.done():
                task.cancel()
        self._inflight.clear()
        # 4. Delete this worker's durable consumer. `unsubscribe()` only tears
        # down the local subscription -- a durable survives on the server by
        # definition, and its name is derived from worker_id, which a rebuilt
        # Pod never reuses (the IP changes). Left behind, every rollout adds an
        # orphan holding WorkQueue quota that nothing will ever consume.
        if self._js is not None:
            try:
                await self._js.delete_consumer(REQUEST_STREAM, request_durable(self._worker_id))
            except Exception as exc:  # noqa: BLE001 - shutdown must continue
                logger.debug("could not delete request consumer: %s", exc)
            self._js = None
        # 5. Drop the cancel listener and the connection.
        if self._cancel_sub is not None:
            try:
                await self._cancel_sub.unsubscribe()
            except Exception:
                pass
            self._cancel_sub = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        if self._nc is not None:
            try:
                await self._nc.drain()
            except Exception:
                pass
            self._nc = None

    async def _await_queued(self, deadline: float) -> None:
        """Let JetStream hand over everything already queued for this worker.

        Only ``num_pending`` (accepted, not yet delivered) is waited on:
        ``num_ack_pending`` is work already delivered, which is exactly what
        ``_inflight`` tracks and step 2 waits for. Counting both would double
        the wait for the same requests.

        Never raises, and gives up rather than hanging when the consumer cannot
        be read -- a shutdown that stalls on a broker hiccup is worse than one
        that drops a queued request, and the caller's deadline is shared with
        the in-flight wait that follows.
        """
        if self._js is None:
            return
        durable = request_durable(self._worker_id)
        while True:
            try:
                info = await self._js.consumer_info(REQUEST_STREAM, durable)
            except Exception as exc:  # noqa: BLE001 - shutdown must continue
                logger.debug("drain: cannot read consumer backlog (%s); not waiting", exc)
                return
            queued = int(getattr(info, "num_pending", 0) or 0)
            if queued <= 0:
                return
            if time.monotonic() >= deadline:
                logger.warning(
                    "drain timeout with %d request(s) still queued in JetStream; "
                    "they will not be served",
                    queued,
                )
                return
            logger.info("drain: waiting for %d queued request(s) to be delivered", queued)
            await asyncio.sleep(min(_QUEUED_POLL_INTERVAL_S, max(0.0, deadline - time.monotonic())))

    async def _hand_back_migratable(self) -> None:
        """Give the router back what it can finish elsewhere, instead of cutting it.

        Called once the drain window has run out, on the requests that outlived
        it. Everything here was going to be cancelled a moment later, so the
        choice this makes is not whether these generations survive the drain --
        it is whether the client learns that they did not.

        The notice goes out *before* the task is cancelled: cancellation also
        sends an error, but a generic one, and the router would then treat a
        planned handover as a worker that broke.
        """
        handed = [i for i in self._migratable if not self._done(i)]
        if not handed:
            return
        logger.info("drain: handing %d resumable request(s) back to the router", len(handed))
        for inbox in handed:
            self._handed_back.add(inbox)
            try:
                await self._reply(inbox, TYPE_ERROR, DRAINING_NOTICE)
            except Exception as exc:  # noqa: BLE001 - shutdown must continue
                logger.debug("could not hand back %s: %s", inbox[-12:], exc)
            task = self._inflight.get(inbox)
            if task is not None and not task.done():
                # Stops the engine generating for a stream nobody reads now that
                # the router has been told to take it elsewhere.
                task.cancel()
        self._migratable.clear()

    def _done(self, inbox: str) -> bool:
        task = self._inflight.get(inbox)
        return task is None or task.done()

    async def _reply(
        self, inbox: str, rtype: str, data: bytes = b"", status: int | None = None
    ) -> None:
        headers = {HDR_TYPE: rtype}
        if status is not None:
            headers[HDR_STATUS] = str(status)
        await self._nc.publish(inbox, data, headers=headers)

    async def _on_request(self, msg) -> None:
        # JetStream mode carries the reply inbox in a header (msg.reply is the
        # ack subject there); core mode uses the NATS reply field.
        inbox = (msg.headers or {}).get(HDR_INBOX) or msg.reply
        if not inbox:
            logger.warning("NATS request without reply inbox; dropping")
            await self._ack(msg)
            return
        # Run the proxy in a task keyed by inbox so a cancel signal can abort it
        # (which tears down the engine connection -> engine stops generating).
        task = asyncio.create_task(self._proxy(inbox, msg), name=f"nats-req-{inbox[-12:]}")
        self._inflight[inbox] = task
        # Remembered for drain: only a request the router said it can continue
        # elsewhere may be handed back early.
        try:
            if json.loads(msg.data).get("migratable"):
                self._migratable.add(inbox)
        except Exception:  # noqa: BLE001 - a body _proxy will reject anyway
            pass

    async def _on_cancel(self, msg) -> None:
        # nats-py requires subscription callbacks to be coroutines.
        inbox = msg.data.decode(errors="replace")
        task = self._inflight.get(inbox)
        if task is not None and not task.done():
            logger.info("NATS request cancel: aborting in-flight request %s", inbox[-12:])
            task.cancel()

    async def _proxy(self, inbox: str, msg) -> None:
        try:
            if self._max_duration > 0:
                # Backstop: force-abort after a hard wall-clock cap even if the
                # engine is still emitting bytes (covers a lost cancel signal).
                await asyncio.wait_for(self._do_proxy(inbox, msg), self._max_duration)
            else:
                await self._do_proxy(inbox, msg)
        except asyncio.TimeoutError:
            logger.warning(
                "NATS request exceeded max duration %.0fs; aborting %s",
                self._max_duration,
                inbox[-12:],
            )
            try:
                await self._reply(inbox, TYPE_ERROR, b"request exceeded max duration")
            except Exception:
                pass
        except asyncio.CancelledError:
            # Router gave up (timeout / client disconnect). The engine connection
            # is torn down by exiting the stream context; best-effort error reply
            # (the router may have already unsubscribed). A request handed back
            # for drain was cancelled by us and has already been told why, so it
            # must not get a second, contradictory error frame.
            if inbox not in self._handed_back:
                logger.info("NATS request aborted (cancelled): %s", inbox[-12:])
                try:
                    await self._reply(inbox, TYPE_ERROR, b"request cancelled")
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("NATS request proxy failed: %s", exc)
            try:
                await self._reply(inbox, TYPE_ERROR, str(exc).encode())
            except Exception:
                pass
        finally:
            self._inflight.pop(inbox, None)
            self._migratable.discard(inbox)
            self._handed_back.discard(inbox)
            # Ack only after the request is fully proxied so the backlog gauge
            # (num_ack_pending) reflects genuinely in-flight work.
            await self._ack(msg)

    async def _do_proxy(self, inbox: str, msg) -> None:
        """Proxy one request to the local engine and stream the reply back."""
        req = json.loads(msg.data)
        path = req.get("path", "/v1/chat/completions")
        body = req.get("body", {})
        stream = bool(req.get("stream", False))
        headers = req.get("headers") or None
        url = f"{self._base}{path}"
        if stream:
            async with self._http.stream("POST", url, json=body, headers=headers) as resp:
                async for chunk in resp.aiter_raw():
                    if chunk:
                        await self._reply(inbox, TYPE_DATA, chunk)
                await self._reply(inbox, TYPE_DONE, status=resp.status_code)
        else:
            resp = await self._http.post(url, json=body, headers=headers)
            await self._reply(inbox, TYPE_DATA, resp.content)
            await self._reply(inbox, TYPE_DONE, status=resp.status_code)

    @staticmethod
    async def _ack(msg) -> None:
        ack = getattr(msg, "ack", None)
        if ack is None:
            return  # core-NATS message: nothing to ack
        try:
            await ack()
        except Exception:
            pass
