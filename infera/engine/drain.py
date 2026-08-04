###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Let in-flight generations finish before the engine is stopped.

On the NATS transport infera owns the request path, so it knows exactly what is
in flight and ``NatsRequestServer.stop(drain=True)`` waits for it. On HTTP the
router talks straight to the engine's own server: infera never sees the request,
cannot count it, and so — until this — did not wait for it. Shutdown went
``deregister()`` then ``engine.stop()``, cutting every active generation.

The way out is to ask the engine, which does know. It publishes its running and
queued request counts on ``/metrics``; poll until both reach zero or the timeout
expires. That is a poll rather than a signal, so it is bounded by
``poll_interval`` rather than exact — acceptable, because the alternative is not
draining at all.

Two behaviours are deliberate:

* **Deregister first, then drain.** Ordering is the whole point. Draining while
  still registered just means more work arrives; the router has to stop choosing
  this worker before waiting for the work it already has.
* **An unreadable metric does not block shutdown.** If the engine's in-flight
  count cannot be determined — an unknown engine, a renamed series, a dead HTTP
  server — this logs loudly and returns rather than hanging until the timeout.
  A rolling update that stalls on a parse failure is a worse outcome than one
  that cuts a request, and a silent full-timeout wait would look identical to a
  genuinely busy worker.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from infera.common.engine_metrics import inflight_from_metrics
from infera.common.worker_pool import EngineType

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 0.5


async def drain_engine_inflight(
    *,
    host: str,
    port: int,
    engine: EngineType,
    timeout: float,
    poll_interval: float = _POLL_INTERVAL_S,
) -> bool:
    """Wait until the engine reports no in-flight work, bounded by ``timeout``.

    Returns True if it drained, False if it timed out or could not be measured.
    Never raises: this runs on the shutdown path, where an exception would skip
    the engine teardown that follows.
    """
    if timeout <= 0:
        return False

    # The engine binds the advertised port, but 0.0.0.0 is not a destination.
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    url = f"http://{probe_host}:{port}/metrics"
    deadline = time.monotonic() + timeout
    peak = 0.0

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                try:
                    resp = await client.get(url)
                    inflight = (
                        inflight_from_metrics(resp.text, engine)
                        if resp.status_code == 200
                        else None
                    )
                except httpx.HTTPError as exc:
                    logger.info("drain: engine metrics unreachable (%s); not waiting", exc)
                    return False

                if inflight is None:
                    logger.warning(
                        "drain: cannot read in-flight count for %s from %s -- shutting down "
                        "WITHOUT draining. In-flight generations will be cut.",
                        engine.value,
                        url,
                    )
                    return False

                peak = max(peak, inflight)
                if inflight <= 0:
                    if peak > 0:
                        logger.info("drain: engine idle, %.0f request(s) completed", peak)
                    return True

                if time.monotonic() >= deadline:
                    logger.warning(
                        "drain: timeout after %.0fs with %.0f request(s) still in flight; "
                        "they will be cut",
                        timeout,
                        inflight,
                    )
                    return False

                await asyncio.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
    except Exception as exc:  # noqa: BLE001 - shutdown must continue regardless
        logger.warning("drain: aborted (%s: %s); not waiting", type(exc).__name__, exc)
        return False
