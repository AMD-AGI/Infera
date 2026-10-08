# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Best-effort, allowlisted deployment usage reporting (never request data)."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import time
import uuid
from contextlib import asynccontextmanager, suppress
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

# Set only after an Infera-owned collector is provisioned. Never reuse another
# project's collector or guess a domain. Deployments can configure their own.
DEFAULT_ENDPOINT = ""
HEARTBEAT_SECONDS = 600
SEND_TIMEOUT_SECONDS = 2
_TRUE = {"1", "true", "yes", "on"}
_CI_VARS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL", "PYTEST_CURRENT_TEST")


def usage_enabled() -> bool:
    """Opt-out takes precedence; re-evaluated before every report."""
    for key in ("INFERA_NO_USAGE_STATS", "DO_NOT_TRACK", "TELEMETRY_DISABLED"):
        if os.environ.get(key, "").strip().lower() in _TRUE:
            return False
    if any(os.environ.get(key, "").strip().lower() not in ("", "0", "false") for key in _CI_VARS):
        return False
    try:
        config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        return not (config_home / "infera" / "do_not_track").exists()
    except (OSError, RuntimeError, ValueError):
        # If the privacy preference cannot be read, do not collect.
        return False


def usage_endpoint() -> str:
    endpoint = os.environ.get("INFERA_USAGE_STATS_SERVER", DEFAULT_ENDPOINT).strip()
    try:
        parts = urlsplit(endpoint)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
        ):
            return ""
        _ = parts.port  # Reject malformed ports before starting the reporter.
    except ValueError:
        return ""
    return endpoint


def _category(value: str, allowed: tuple[str, ...]) -> str:
    return next((item for item in allowed if value == item), "unknown")


def _metadata(component: str, engine: str, mode: str, transport: str) -> dict:
    try:
        package_version = version("amd-infera")
    except PackageNotFoundError:
        package_version = "unknown"
    # Strip local build suffixes: they can contain internal branch names.
    if match := re.match(r"^\d+\.\d+\.\d+(?=$|[.+a-z-])", package_version):
        package_version = match.group()
    else:
        package_version = "unknown"
    return {
        "infera_version": package_version,
        "python_version": platform.python_version(),
        "os": _category(platform.system(), ("Linux", "Windows", "Darwin")),
        "cpu_architecture": _category(platform.machine(), ("x86_64", "aarch64", "AMD64", "arm64")),
        "component": _category(component, ("server", "worker")),
        "engine": _category(engine, ("vllm", "sglang", "atom", "none")),
        "disagg_mode": _category(mode, ("mixed", "prefill", "decode", "none")),
        "request_transport": _category(transport, ("http", "nats")),
    }


async def _report(endpoint: str, metadata: dict) -> None:
    try:
        session_id = uuid.uuid4().hex
        started = time.monotonic()
        # No redirects, response-body buffering, persistent identifiers, or
        # on-disk spool. A failure drops this event; the next heartbeat may try.
        async with httpx.AsyncClient(
            timeout=SEND_TIMEOUT_SECONDS, follow_redirects=False
        ) as client:
            sequence = 0
            while usage_enabled():
                payload = {
                    "schema_version": 1,
                    "event": "session_start" if sequence == 0 else "heartbeat",
                    "session_id": session_id,
                    "sequence": sequence,
                    "elapsed_seconds": int(time.monotonic() - started),
                    **metadata,
                }

                async def send(payload=payload) -> None:
                    async with client.stream("POST", endpoint, json=payload):
                        pass

                try:
                    await asyncio.wait_for(send(), timeout=SEND_TIMEOUT_SECONDS)
                except Exception:
                    # Do not log exceptions: URLs/proxy credentials can appear
                    # in transport errors. Telemetry must never affect serving.
                    pass
                sequence += 1
                await asyncio.sleep(HEARTBEAT_SECONDS)
    except Exception:
        pass


@asynccontextmanager
async def usage_session(
    component: str, *, engine: str = "none", mode: str = "none", transport: str = "http"
):
    """Run reporting only for the lifetime of a serving session.

    Imports, argument parsing and --help never start reporting. The task is
    cancelled at shutdown; no exit request delays engine draining.
    """
    task = None
    try:
        if usage_enabled():
            endpoint = usage_endpoint()
            if endpoint:
                metadata = _metadata(component, engine, mode, transport)
                task = asyncio.create_task(_report(endpoint, metadata), name="infera-usage")
                logger.info(
                    "Infera usage reporting enabled. Set INFERA_NO_USAGE_STATS=1 to opt out; "
                    "see manual/reference/usage-stats.md for collected fields."
                )
            else:
                logger.info("Infera usage reporting inactive: no valid HTTPS collector configured")
    except Exception:
        pass
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
