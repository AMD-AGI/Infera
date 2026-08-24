###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Flush an engine's prefix cache so its KV-event chain re-anchors.

The router's kv-aware index is a chain: every ``BlockStored`` event names the
``parent_block_hash`` it hangs off, and only the radix root reports ``None``.
The index can therefore only be built *forward* from that one rooted event, and
an event whose parent was never seen is dropped -- along with its own hash,
which orphans everything downstream of it in turn. Losing the anchor is an
absorbing state: no amount of later traffic recovers it, and nothing errors.
kv-aware simply degrades to load-only routing behind a green ``/health``.

Losing it is the default, not the exception. The engine binds its KV-event
publisher at launch and its startup warmup writes the radix tree immediately, so
the anchor is broadcast before any subscriber exists -- and neither a ZMQ PUB
nor the worker-side NATS relay retains what it never received.

Flushing the cache is the repair, because ``AllBlocksCleared`` is the one event
that rebuilds an anchor from nothing: it puts engine and router back on the only
state they can agree on, and the next request re-roots the chain. This module is
that call, deliberately small and unable to raise -- it runs on the startup path
next to registration, where an exception would cost a worker its registration to
fix a routing optimisation.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from infera.common.worker_pool import EngineType

logger = logging.getLogger(__name__)

#: Per-engine cache-flush endpoint. SGLang clears its radix tree and emits
#: ``AllBlocksCleared``; vLLM's twin is ``/reset_prefix_cache`` (see
#: ``infera/engine/vllm/kvd_connector.py``).
_FLUSH_PATHS = {
    EngineType.SGLANG: "/flush_cache",
    EngineType.VLLM: "/reset_prefix_cache",
    EngineType.ATOM: "/flush_cache",
}


async def flush_engine_prefix_cache(
    *,
    host: str,
    port: int,
    engine: EngineType,
    timeout: float = 10.0,
) -> bool:
    """POST the engine's cache-flush endpoint. Returns True if it flushed.

    Never raises. A False return distinguishes nothing -- callers that care why
    should read the log line -- but the common retryable case is worth naming:
    SGLang refuses with **HTTP 400** while its scheduler is not idle, so a
    rejection is visible rather than a silent no-op, and retrying is meaningful.
    """
    path = _FLUSH_PATHS.get(engine)
    if path is None:
        logger.info("kv flush: no cache-flush endpoint known for %s; skipping", engine)
        return False

    # The engine binds the advertised port, but 0.0.0.0 is not a destination.
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    url = f"http://{probe_host}:{port}{path}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url)
    except httpx.HTTPError as exc:
        logger.warning("kv flush: %s unreachable (%s)", url, exc)
        return False
    except Exception as exc:  # noqa: BLE001 - startup must continue regardless
        logger.warning("kv flush: aborted (%s: %s)", type(exc).__name__, exc)
        return False

    if resp.status_code == 200:
        return True
    if resp.status_code == 400:
        # Not idle. Worth saying out loud: at the point this runs the worker is
        # not registered yet, so anything in flight is the engine's own warmup
        # and the condition should clear on its own within a retry or two.
        logger.info("kv flush: engine not idle yet (HTTP 400)")
        return False
    logger.warning("kv flush: %s returned HTTP %d", url, resp.status_code)
    return False


async def anchor_kv_chain(
    *,
    host: str,
    port: int,
    engine: EngineType,
    observed: asyncio.Event,
    attempts: int = 5,
    settle: float = 2.0,
    post_timeout: float = 3.0,
    deadline: float = 15.0,
) -> bool:
    """Flush until a subscriber confirms it saw the resulting clear event.

    ``observed`` is set by whoever is tailing the engine's KV events (the NATS
    relay). Waiting on it is the point: ZMQ ``connect()`` is asynchronous, so a
    relay whose ``start()`` has returned may not have attached its subscription
    yet, and a flush issued into that gap is lost exactly the way the original
    anchor was -- leaving a worker that looks freshly repaired and is not. A
    fixed sleep would only move the guess around; this observes the outcome.

    ``deadline`` bounds the whole loop, not each attempt. Callers run this
    immediately before registering, so every second here is a second the worker
    exists and cannot be routed to; ``attempts`` alone would let an engine that
    answers slowly turn a routing optimisation into a startup stall of
    ``attempts * (post_timeout + settle)``. The POST timeout is short for the
    same reason -- this is a loopback call to a process that is up enough to
    have bound its port, so anything slower is a wedged engine, and the retry is
    a better answer to that than waiting.

    Returns True once the clear is seen. Never raises, and a False return is
    logged, not fatal: kv-aware degrades, the worker still serves.
    """
    if observed.is_set():
        return True

    expiry = asyncio.get_running_loop().time() + deadline
    attempt = 0
    for attempt in range(1, attempts + 1):
        await flush_engine_prefix_cache(
            host=host, port=port, engine=engine, timeout=post_timeout
        )
        left = expiry - asyncio.get_running_loop().time()
        if left <= 0:
            break
        try:
            await asyncio.wait_for(observed.wait(), timeout=min(settle, left))
        except (TimeoutError, asyncio.TimeoutError):
            logger.info(
                "kv flush: no clear event observed after attempt %d/%d; retrying",
                attempt,
                attempts,
            )
            continue
        except Exception as exc:  # noqa: BLE001 - startup must continue regardless
            logger.warning("kv flush: wait aborted (%s: %s)", type(exc).__name__, exc)
            return False
        logger.info("kv events: cache flushed and the chain re-anchored (attempt %d)", attempt)
        return True

    logger.warning(
        "kv events: flushed %d time(s) in %.0fs but never saw the resulting clear event "
        "-- the router's chain has no anchor, so kv-aware will route this worker on load "
        "alone. Registration continues regardless.",
        attempt,
        deadline,
    )
    return False
