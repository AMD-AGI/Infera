###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Abort a stuck PD engine request so it cannot occupy an inflight slot forever.

SGLang's Mooncake path keeps a prefill ``/generate`` inflight until KV
handoff completes. A dropped client or a hung transfer leaves that slot
busy (the 64-token / 4.27 tok/s log line) while ``/health`` still returns
200. The router posts SGLang's ``/abort_request`` so the engine drops it.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

ABORT_PATH = "/abort_request"
TIMEOUT_ENV = "INFERA_PD_PREFILL_DRAIN_TIMEOUT"
DEFAULT_TIMEOUT_S = 300.0


def rid_for_room(room_id: int) -> str:
    """Stable SGLang ``rid`` shared by both PD legs of one bootstrap room."""
    return f"infera-{int(room_id)}"


def prefill_drain_timeout_s() -> float:
    """Seconds to wait for a detached prefill POST before aborting it.

    0 disables the wall-clock cap (client-disconnect abort still runs).
    """
    raw = os.environ.get(TIMEOUT_ENV)
    if raw is None or raw == "":
        return DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_S
    return value if value > 0 else 0.0


def abort_url(worker_url: str) -> str:
    return f"{worker_url.rstrip('/')}{ABORT_PATH}"


async def abort_engine_request(
    http: httpx.AsyncClient,
    worker_url: str,
    rid: str | None,
) -> None:
    """POST ``/abort_request``; errors are logged and swallowed."""
    if not rid:
        return
    url = abort_url(worker_url)
    try:
        resp = await http.post(url, json={"rid": rid}, timeout=5.0)
        if resp.status_code >= 400:
            logger.warning(
                "PD abort %s rid=%s returned %s", url, rid, resp.status_code
            )
        else:
            logger.info("PD abort %s rid=%s", url, rid)
    except Exception as exc:
        logger.warning("PD abort %s rid=%s failed: %s", url, rid, exc)
