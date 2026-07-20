###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""infera-kvd ↔ SGLang HiCacheStorage wiring (called from
`infera.engine.sglang.__main__`).

Split into its own module so it imports cleanly without sglang on the
PYTHONPATH — useful for unit tests on router-only hosts. The actual
SGLang factory registration is itself lazy (inside
`register_kvd_backend_with_sglang`), so this module is a thin
orchestrator: probe the daemon, register the backend, mutate
``server_args`` to enable hicache.

Why a separate module from `args.py`:
- ``args.py`` does ``from sglang.srt.server_args import ServerArgs`` at
  import time. That's load-bearing for parsing the CLI but makes
  unit-testing any helper that lives in the same module impossible
  without sglang installed.
- This module avoids the sglang import; the test suite can exercise
  the orchestration logic without GPU dependencies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _has_cli_flag(argv: list[str], flag: str) -> bool:
    return flag in argv or any(item.startswith(f"{flag}=") for item in argv)


def _append_sglang_hicache_argv(args: Any) -> None:
    """Append child-process flags needed to load the kvd backend.

    Mutating ``ServerArgs`` is useful for metadata, but the actual
    ``sglang.launch_server`` subprocess reparses ``args.sglang_argv``.
    Therefore the wrapper must also patch the argv forwarded to the child.
    """
    argv = getattr(args, "sglang_argv", None)
    if argv is None:
        return

    if not _has_cli_flag(argv, "--enable-hierarchical-cache"):
        argv.append("--enable-hierarchical-cache")
        logger.info("--infera-kvd-socket appends --enable-hierarchical-cache")

    if not _has_cli_flag(argv, "--hicache-storage-backend"):
        # Use SGLang's dynamic backend loader in the child process. Registering
        # the backend in the wrapper process does not cross the subprocess
        # boundary, while dynamic imports the adapter in the serving process.
        argv += ["--hicache-storage-backend", "dynamic"]
        logger.info("--infera-kvd-socket appends --hicache-storage-backend dynamic")

    if not _has_cli_flag(argv, "--hicache-storage-backend-extra-config"):
        cfg = {
            "backend_name": "infera-kvd",
            "module_path": "infera.engine.sglang.kvd_adapter",
            "class_name": "InferaKvdBackend",
            "prefetch_threshold": 64,
        }
        argv += ["--hicache-storage-backend-extra-config", json.dumps(cfg)]
        logger.info("--infera-kvd-socket appends dynamic backend extra config")


async def _probe_kvd(socket_path: str) -> None:
    """Probe kvd reachability before SGLang spawns subprocesses."""
    from infera.kvd.client import KvdClient

    client = KvdClient(socket_path, client_id="sglang-startup-probe")
    # Bound each phase: a wedged daemon (accepts UDS but never
    # answers an RPC) would otherwise hang startup forever. 5 s
    # is more than enough for a healthy local daemon (typical
    # response: <1 ms) and tight enough that operators see the
    # symptom quickly (PR #9 review fix P1).
    await asyncio.wait_for(client.connect(), timeout=5.0)
    await asyncio.wait_for(client.stats(), timeout=5.0)
    await asyncio.wait_for(client.close(), timeout=5.0)


def _finish_wiring(args: Any, socket_path: str) -> None:
    """Wire the SGLang HiCacheStorage backend after kvd is confirmed live.

    Division of labour (important — these are NOT redundant):

    - ``_append_sglang_hicache_argv`` (called at the end) is the
      *load-bearing* path: SGLang runs in a ``sglang.launch_server``
      **subprocess** that re-parses ``args.sglang_argv`` from scratch and
      ignores this process's in-memory ``ServerArgs``. So the flags that
      actually select our backend must be appended to the forwarded argv.
    - ``register_kvd_backend_with_sglang()`` here registers the backend in
      THIS (wrapper) process. The subprocess does NOT inherit that registry
      (spawn start-method → fresh interpreter); it gets its own
      registration via SGLang's built-in ``dynamic`` backend loader (the
      backend class path is forwarded on the subprocess argv). We still call
      it here as an **early smoke-check**:
      it fails fast if sglang or the adapter module can't be imported
      (a genuinely-broken install), rather than surfacing that only once
      the subprocess is already mid-startup.
    - The ``server_args`` mutations below are **metadata / back-compat
      only** — they don't reach the subprocess. They're kept so callers
      that introspect ``args.server_args`` (and any future in-process use)
      see a coherent view.
    """
    from infera.engine.sglang.kvd_adapter import register_kvd_backend_with_sglang

    register_kvd_backend_with_sglang()

    # Metadata-only mutation (see docstring): defensively set the fields if
    # they exist AND the operator didn't already set them on the CLI. The
    # subprocess re-parses argv, so this does not affect the running engine;
    # `_append_sglang_hicache_argv` is what actually selects the backend.
    sa = args.server_args
    if hasattr(sa, "enable_hierarchical_cache") and not sa.enable_hierarchical_cache:
        sa.enable_hierarchical_cache = True
        logger.info("--infera-kvd-socket implies --enable-hierarchical-cache")
    if hasattr(sa, "hicache_storage_backend") and not sa.hicache_storage_backend:
        sa.hicache_storage_backend = "infera-kvd"
        logger.info("--infera-kvd-socket implies --hicache-storage-backend infera-kvd")
    # PR #9 review fix P1 (prefetch_threshold silent perf failure):
    # SGLang's default prefetch_threshold is 256 tokens. The runbook on
    # MI355X documents that 64 is needed for cache_control workloads
    # where the cacheable system prompt may be <256 tokens. Without
    # the override, prompts under 256 tokens never trigger L3 prefetch
    # → `gets_total = 0` and operators see no cache reuse.
    #
    # Try common field names across SGLang versions and lower the
    # default; if none match (older SGLang), log a clear WARNING with
    # the override the operator should pass via extra-config.
    _PREFETCH_FIELDS = (
        "hicache_storage_prefetch_threshold",
        "hicache_prefetch_threshold",
        "prefetch_threshold",
    )
    overridden = False
    for field in _PREFETCH_FIELDS:
        if hasattr(sa, field):
            current = getattr(sa, field)
            if current is None or current == 256:
                setattr(sa, field, 64)
                logger.info(
                    "--infera-kvd-socket lowered SGLang.%s to 64 "
                    "(cache_control workloads on short prompts)",
                    field,
                )
            else:
                logger.info(
                    "SGLang.%s already set to %s — leaving operator value in place",
                    field,
                    current,
                )
            overridden = True
            break
    if not overridden:
        logger.warning(
            "SGLang version has no recognized prefetch_threshold field "
            "(tried %s). Prompts under 256 tokens may not trigger L3 "
            "prefetch. Pass an explicit override via --hicache-storage-"
            'config / extra-config JSON: {"prefetch_threshold":64}.',
            ", ".join(_PREFETCH_FIELDS),
        )

    _append_sglang_hicache_argv(args)
    logger.info("infera-kvd HiCacheStorage backend ready (socket=%s)", socket_path)


async def awire_infera_kvd_backend(args: Any) -> None:
    """Async variant for callers already inside an event loop."""
    socket_path = args.infera_kvd_socket
    if not socket_path:
        return

    # The adapter (constructed later by SGLang) reads this env var;
    # this is the only seam we have because SGLang's factory
    # instantiates the backend without a way to pass kwargs through
    # the CLI.
    os.environ["INFERA_KVD_SOCKET"] = socket_path

    try:
        await _probe_kvd(socket_path)
    except Exception:
        logger.exception(
            "infera-kvd unreachable at %s; refusing to start engine. "
            "Start the kvd daemon first (e.g. `make up-kvd-aware` or "
            "`python -m infera.kvd --socket %s`), then restart the engine. "
            "If the daemon is up but unresponsive (TimeoutError from this "
            "probe), the daemon is wedged — restart it.",
            socket_path,
            socket_path,
        )
        raise

    _finish_wiring(args, socket_path)


def wire_infera_kvd_backend(args: Any) -> None:
    """If ``args.infera_kvd_socket`` is set, probe the kvd daemon,
    register the backend with SGLang's storage factory, and inject the
    forwarded ``sglang.launch_server`` argv flags (``--hicache-storage-backend dynamic`` + extra-config).

    Fails fast on:

    - kvd socket missing or unreachable → caller's startup aborts. We
      don't want SGLang to come up with a silently-broken cache
      backend; operators want the error in the engine startup log,
      not as opaque misses in production traffic.
    - infera.engine.sglang.kvd_adapter import error → also fatal
      (something is genuinely wrong with the install).

    Args:
        args: SglangWorkerArgs-shaped object — needs
            ``args.infera_kvd_socket`` and ``args.server_args``.
            We accept a SimpleNamespace stand-in in tests to avoid
            the sglang dependency.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(awire_infera_kvd_backend(args))
        return
    raise RuntimeError(
        "wire_infera_kvd_backend() cannot be called from a running event loop; "
        "use `await awire_infera_kvd_backend(args)` instead"
    )
