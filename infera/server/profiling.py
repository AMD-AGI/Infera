###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unified torch-profiler control plane for the Infera frontend.

The frontend does not run a torch profiler itself; each worker runs the
engine's native HTTP server (SGLang / vLLM) which exposes ``/start_profile``
and ``/stop_profile``. This module resolves a set of target workers from the
registry and fans the profile control request out to their engine endpoints,
so an operator has a single entry point instead of curling each worker port.

Targeting mirrors the registry view: by default every ACTIVE worker, with
optional ``worker_id`` / ``model`` / ``role`` (mixed|prefill|decode) filters.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from infera.common.worker_pool import DisaggMode, WorkerInfo, WorkerStatus

logger = logging.getLogger(__name__)

# Valid profile actions. The engine path is ``/{action}_profile`` and is the
# same for SGLang and vLLM.
PROFILE_ACTIONS = ("start", "stop")


def _normalize_role(role: str | None) -> DisaggMode | None:
    """Map a ``role`` query value to a DisaggMode, or None for 'no filter'.

    Raises ValueError on an unknown role so the caller can return 400.
    """
    if role is None or role == "":
        return None
    try:
        return DisaggMode(role.lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in DisaggMode)
        raise ValueError(f"invalid role {role!r}; expected one of: {valid}") from exc


def select_targets(
    workers: list[WorkerInfo],
    *,
    worker_id: str | None = None,
    model: str | None = None,
    role: str | None = None,
) -> list[WorkerInfo]:
    """Filter ACTIVE workers by the optional worker_id / model / role.

    No filter selects every ACTIVE worker (broadcast). Role is matched
    against ``WorkerInfo.disagg_mode``.
    """
    mode = _normalize_role(role)
    out = []
    for w in workers:
        if w.status != WorkerStatus.ACTIVE:
            continue
        if worker_id is not None and w.worker_id != worker_id:
            continue
        if model is not None and w.model_name != model:
            continue
        if mode is not None and w.disagg_mode != mode:
            continue
        out.append(w)
    return out


async def _profile_one(
    client: httpx.AsyncClient,
    worker: WorkerInfo,
    action: str,
    body: dict[str, Any] | None,
) -> dict[str, Any]:
    """POST ``/{action}_profile`` to a single worker's engine endpoint.

    Never raises: transport / HTTP errors are captured into the result dict
    so one unreachable worker does not fail the whole fan-out.
    """
    url = f"{worker.url}/{action}_profile"
    try:
        resp = await client.post(url, json=body or {})
        ok = resp.status_code < 400
        detail: Any
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        return {
            "worker_id": worker.worker_id,
            "url": url,
            "engine": worker.engine.value
            if hasattr(worker.engine, "value")
            else str(worker.engine),
            "status_code": resp.status_code,
            "ok": ok,
            "detail": detail,
        }
    except httpx.HTTPError as exc:
        logger.warning("profile %s failed for %s: %s", action, url, exc)
        return {
            "worker_id": worker.worker_id,
            "url": url,
            "engine": worker.engine.value
            if hasattr(worker.engine, "value")
            else str(worker.engine),
            "status_code": None,
            "ok": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }


async def fan_out_profile(
    client: httpx.AsyncClient,
    workers: list[WorkerInfo],
    action: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fan a profile ``action`` (start|stop) out to every target worker.

    Returns an aggregate ``{ "action", "requested", "succeeded", "results" }``
    where ``results`` is the per-worker outcome list. Workers are hit
    concurrently; individual failures are reported, not raised.
    """
    if action not in PROFILE_ACTIONS:
        raise ValueError(f"invalid action {action!r}; expected one of: {PROFILE_ACTIONS}")

    results = await asyncio.gather(*(_profile_one(client, w, action, body) for w in workers))
    succeeded = sum(1 for r in results if r["ok"])
    return {
        "action": action,
        "requested": len(results),
        "succeeded": succeeded,
        "results": list(results),
    }
