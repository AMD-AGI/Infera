###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Entrypoint for ``python -m infera.engine.atom``.

Spawns ATOM's OpenAI server (``atom.entrypoints.openai_server``) as a
subprocess, then self-registers with etcd — same launcher protocol as the
SGLang / vLLM workers.

Mixed (prefill+decode) example::

    python -m infera.engine.atom \\
        --model Qwen/Qwen3-0.6B --server-port 30000 --host 0.0.0.0 \\
        -tp 1 --etcd-endpoint 127.0.0.1:2379

PD disaggregation is selected via ATOM's own ``--kv-transfer-config``
(``kv_role`` = ``kv_producer`` / ``kv_consumer``); see the project README.

KV-aware routing is **off by default** — ATOM has no native KV-event stream,
so infera must monkey-patch the engine to emit events. Pass
``--enable-kv-events`` to opt in: the launcher then allocates a ZMQ port and
the ATOM subprocess publishes its BlockManager prefix-cache index in the
router's event wire format (see :mod:`infera.engine.atom.hooks.kv_events`).
Without it the worker registers no ``kv_events_endpoint`` and the router
routes it round-robin.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from infera.common.net import free_tcp_port
from infera.common.registration import RegistrationClient
from infera.engine.atom.args import parse_atom_args
from infera.engine.atom.worker import AtomEngine
from infera.engine.base import watch_engine_death

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # silence etcd keepalive spam
logger = logging.getLogger(__name__)


def _kill_process_group_safely() -> None:
    # TERM → 0.5s → KILL. Avoids killing the parent shell that didn't `setsid`.
    try:
        pgid = os.getpgrp()
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    import time as _time

    _time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def main() -> None:
    args = parse_atom_args()
    logger.info(
        "parsed args: model=%s host=%s server_port=%d tp=%d disagg=%s disagg_meta=%s",
        args.model,
        args.host,
        args.server_port,
        args.tensor_parallel_size,
        args.disagg_mode,
        args.disagg_meta or "{}",
    )

    # KV-aware routing: allocate a ZMQ port for the BlockManager event
    # publisher. ``bind`` is what the ATOM subprocess binds (tcp://*:port);
    # ``advertise`` is the reachable address the router subscribes to.
    kv_events_bind: str | None = None
    kv_events_endpoint: str | None = None
    kv_block_size: int | None = None
    if args.enable_kv_events:
        advertise_host = args.advertise_host or args.host
        if advertise_host in ("0.0.0.0", ""):
            logger.warning(
                "--enable-kv-events with host=0.0.0.0 and no --advertise-host; "
                "the KV events endpoint won't be reachable by a remote router"
            )
        port = free_tcp_port()
        kv_events_bind = f"tcp://*:{port}"
        kv_events_endpoint = f"tcp://{advertise_host}:{port}"
        kv_block_size = args.kv_block_size

    # MI325X (gfx942) + DeepSeek-V4: enforce the support matrix (raise on an
    # unsupported quant/engine combo) and apply the fp8 env defaults (Flash also
    # gets MTP CLI). No-op on other arches / non-dsv4 models. Must run before the
    # ATOM subprocess is spawned so the env is inherited and injected CLI reaches it.
    from infera.engine.dsv4_gfx942 import apply_gfx942_dsv4

    args.atom_argv = apply_gfx942_dsv4(args.model, engine="atom", argv=args.atom_argv)

    engine = AtomEngine(
        atom_argv=args.atom_argv,
        model_name=args.model,
        host=args.host,
        port=args.server_port,
        advertise_host=args.advertise_host,
        kv_events_endpoint=kv_events_endpoint,
        kv_events_bind=kv_events_bind,
        kv_block_size=kv_block_size,
        disagg_mode=args.disagg_mode,
        disagg_meta=args.disagg_meta,
    )

    try:
        config = await engine.start()
    except Exception:
        logger.exception("atom engine failed to start; tearing down")
        try:
            await engine.stop()
        finally:
            _kill_process_group_safely()
        raise
    logger.info(
        "worker ready: model=%s url=http://%s:%d engine=%s disagg=%s meta=%s",
        config.model_name,
        config.host,
        config.port,
        config.engine,
        config.disagg_mode,
        config.disagg_meta,
    )

    logger.info("registering with etcd: %s prefix=%s", args.etcd_endpoint, args.etcd_prefix)
    reg_client = RegistrationClient(endpoint=args.etcd_endpoint, prefix=args.etcd_prefix)
    await reg_client.register(config)
    hb_task = asyncio.create_task(reg_client.heartbeat_loop(), name="worker-heartbeat")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    death_task = watch_engine_death(engine, stop)
    await stop.wait()

    death_task.cancel()
    try:
        await death_task
    except asyncio.CancelledError:
        pass

    hb_task.cancel()
    try:
        await hb_task
    except asyncio.CancelledError:
        pass

    await reg_client.deregister()
    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
