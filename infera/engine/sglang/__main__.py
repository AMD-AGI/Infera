###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""python -m infera.engine.sglang --model-path Qwen/Qwen3-0.6B --port 30000 \\
    --etcd-endpoint host:2379

For PD disaggregation, pass SGLang's own --disaggregation-mode:
    --disaggregation-mode prefill --disaggregation-bootstrap-port 8998
    --disaggregation-mode decode

KV management:
    --kv-events auto         # auto (default) | on | off
    --kv-events-bind tcp://0.0.0.0:5557
    --kv-snapshot-port 8801
    --index-block-size 64
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from infera.common.disagg_preflight import (
    validate_advertise_host,
    validate_sglang_transport,
)
from infera.common.registration import RegistrationClient
from infera.engine.base import watch_engine_death
from infera.engine.sglang.args import SglangWorkerArgs, parse_sglang_args
from infera.engine.sglang.kv_wiring import (
    SglangKvWiring,
    build_and_start,
    resolve_advertise_endpoint,
)
from infera.engine.sglang.kvd_wiring import awire_infera_kvd_backend
from infera.engine.sglang.worker import SglangEngine


def _kill_process_group_safely() -> None:
    """Tear down the process group with SIGTERM → wait → SIGKILL.

    A naive `os.killpg(getpgrp(), SIGKILL)` also kills the parent shell when
    the worker wasn't started via `setsid` (most dev invocations). SIGTERM
    first lets the SGLang children clean up; the brief sleep may take us
    (and them) down before the SIGKILL, which is the intent.
    """
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


logging.basicConfig(level=logging.INFO)
# httpx logs every etcd lease keepalive (~ every ttl/3 seconds). That's
# pure noise in the worker log; surface only warnings/errors.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _wire_mori_dispatch_buffer(server_args) -> None:
    """Size the mori-MoE expert-dispatch buffer from the documented chunked-prefill knob.

    SGLang's mori MoE all-to-all preallocates a per-rank dispatch buffer of
    ``SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK`` tokens (default 4096) and, on the
    PREFILL engine, asserts ``chunked_prefill_size <= that`` (server_args.py; the check
    is skipped for ``disaggregation-mode=decode``). That env var is undocumented, so a
    user who legitimately raises the documented ``--chunked-prefill-size`` gets a cryptic
    assert and has to discover a second, hidden knob. Instead: when mori-MoE is active and
    the operator hasn't pinned the env explicitly, derive the buffer from the chunked-prefill
    size they chose — one documented knob, no surprise assert. The env is read by the
    ``sglang.launch_server`` subprocess, which inherits this process's environment.
    """
    if getattr(server_args, "moe_a2a_backend", None) != "mori":
        return  # mori MoE off -> no buffer, no assert
    if os.environ.get("SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK"):
        return  # explicit operator override wins
    cps = getattr(server_args, "chunked_prefill_size", None)
    if not cps or cps <= 0:  # chunked prefill disabled -> assert skipped
        return
    os.environ["SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = str(int(cps))
    logger.info(
        "mori-MoE: auto-set SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=%d to match "
        "--chunked-prefill-size (satisfies sglang's prefill dispatch-buffer assert)",
        int(cps),
    )


async def _maybe_start_kv_plane(
    args: SglangWorkerArgs,
    engine: SglangEngine,
    *,
    model_name: str,
) -> SglangKvWiring | None:
    """Build the KV plane unless `--kv-events off`.

    Returns the SglangKvWiring (caller stops it on shutdown) or None if
    KV was skipped or failed to come up under --kv-events auto.
    """
    if args.kv_events == "off":
        logger.info("--kv-events off; worker will register with kv=None")
        return None

    # 0.0.0.0 bind is fine inside the worker but useless on the wire;
    # the registered endpoint must be reachable by the server.
    advertise_host = args.server_args.host
    events_advertise = args.kv_events_advertise or resolve_advertise_endpoint(
        args.kv_events_bind, advertise_host
    )
    snapshot_advertise = args.kv_snapshot_advertise or (
        f"http://{advertise_host}:{args.kv_snapshot_port}"
    )

    # SGLang's KV page size lives on server_args; defaults to 1 on ROCm
    # AITER and we honor whatever the operator chose.
    engine_block_size = int(getattr(args.server_args, "page_size", 1) or 1)

    # Locate the RadixCache. SGLang exposes it on its scheduler, which
    # the launch_server thread keeps alive — but it isn't reliably
    # reachable from outside the worker process, so we attach only if we
    # can grab it. Otherwise the plane still runs (events empty, the
    # reconciler will catch up via empty snapshots until SGLang grows
    # an out-of-process event hook).
    try:
        from infera.engine.sglang.kv_wiring import _find_radix_cache

        radix_cache = _find_radix_cache(engine)
    except Exception:
        radix_cache = None

    publisher_id = f"{args.server_args.host}:{args.server_args.port}"
    try:
        wiring = await build_and_start(
            model_id=model_name,
            tokenizer_path=args.server_args.tokenizer_path or args.server_args.model_path,
            engine_block_size=engine_block_size,
            index_block_size=args.index_block_size,
            publisher_id=publisher_id,
            events_bind=args.kv_events_bind,
            events_advertise=events_advertise,
            snapshot_host=args.kv_snapshot_host,
            snapshot_port=args.kv_snapshot_port,
            snapshot_advertise=snapshot_advertise,
            radix_cache=radix_cache,
            trust_remote_code=bool(getattr(args.server_args, "trust_remote_code", False)),
        )
    except Exception:
        # `auto` swallows; `on` fails fast.
        if args.kv_events == "on":
            raise
        logger.exception(
            "--kv-events auto: KV plane failed to start; worker will register with kv=None"
        )
        return None

    logger.info(
        "KV plane up: events_bind=%s events_advertise=%s snapshot=%s "
        "engine_block_size=%d index_block_size=%d",
        args.kv_events_bind,
        events_advertise,
        snapshot_advertise,
        engine_block_size,
        args.index_block_size,
    )
    return wiring


async def main() -> None:
    args = parse_sglang_args()

    # Kubernetes discovery: advertise the routable Pod IP (downward API) rather
    # than the 0.0.0.0 bind host, so the server/peers can reach this worker
    # (the registered url/worker_id and kv endpoints all derive from this).
    if args.discovery_backend == "kubernetes" and not args.advertise_host:
        pod_ip = os.environ.get("POD_IP")
        if pod_ip:
            args.advertise_host = pod_ip
            logger.info("k8s discovery: advertising Pod IP %s", pod_ip)

    # Fail fast on disagg configs that silently break across nodes: a
    # non-routable advertise host, or no explicit RDMA transfer backend
    # (which risks a silent TCP fallback). No-op for mixed workers.
    is_disagg = args.server_args.disaggregation_mode in ("prefill", "decode")
    validate_advertise_host(args.advertise_host or args.server_args.host, is_disagg=is_disagg)
    validate_sglang_transport(
        getattr(args.server_args, "disaggregation_transfer_backend", None),
        is_disagg=is_disagg,
        allow_tcp=args.disaggregation_allow_tcp,
    )

    # Optional infera-kvd HiCacheStorage wiring. Done BEFORE engine.start()
    # because SGLang's launch_server reads server_args once at startup, and
    # we need to register the backend with the storage factory before any
    # subprocess opens it.
    await awire_infera_kvd_backend(args)

    # Ionic RoCE-v2 RDMA env defaults for the Mooncake/MoRI transfer engines
    # (set-if-unset; an operator/launcher still overrides via env). Must run
    # BEFORE engine.start() spawns the sglang subprocess so it's inherited.
    # Without these the transfer engine silently falls back to TCP / hangs.
    from infera.engine.rocm_rdma_env import (
        apply_dsv4_gfx942_env_defaults,
        apply_kv_host_ip_default,
        apply_rocm_rdma_env_defaults,
    )

    apply_rocm_rdma_env_defaults()
    # Pin the KV host IP to the RDMA rail (else get_ip() picks the public NIC and
    # KV transfer targets the wrong interface). Must follow the GID default above.
    apply_kv_host_ip_default()
    # MI325X (gfx942) + DeepSeek-V4-Pro (FP4 experts): enable the FP4->FP8 MoE
    # dequant path + gfx942 MLA/GEMM defaults (set-if-unset). Triple-gated on
    # arch + model, so it's a no-op on MI355X (gfx950) and every other model.
    apply_dsv4_gfx942_env_defaults(args.server_args.model_path, engine="sglang")

    # Auto-size the mori-MoE dispatch buffer from --chunked-prefill-size so operators
    # only set the one documented knob (else sglang asserts on the prefill engine).
    _wire_mori_dispatch_buffer(args.server_args)

    engine = SglangEngine(
        args.server_args,
        sglang_argv=args.sglang_argv,
        advertise_host=args.advertise_host,
        enable_kv_events=args.enable_kv_events,
    )

    try:
        config = await engine.start()
    except Exception:
        logger.exception("engine failed to start; tearing down")
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

    # Everything past engine.start() must tear the engine down on failure;
    # otherwise an exception (e.g. --kv-events on failing the KV plane) escapes
    # without reaping the sglang subprocess tree, orphaning it (holds ports/GPUs).
    try:
        await _run_after_start(args, engine, config)
    except Exception:
        logger.exception("worker failed after engine start; tearing down")
        try:
            await engine.stop()
        finally:
            _kill_process_group_safely()
        raise


async def _run_after_start(args: SglangWorkerArgs, engine: SglangEngine, config) -> None:
    """Worker lifecycle once the sglang subprocess is up: KV plane,
    registration, then serve until shutdown. Raises on any setup failure so
    ``main`` can tear the engine down (avoids orphaning the subprocess tree).
    """
    # --- Multinode follower gate ---
    # In a multi-node (LeaderWorkerSet) TP group only node-rank 0 runs the
    # serving HTTP API and should register; ranks > 0 are pure TP workers (their
    # sglang answers /health but cannot serve /v1/*). Registering them would let
    # the router proxy requests to a non-serving endpoint (404). So a follower
    # skips the KV plane + registration and just keeps its sglang subprocess
    # alive until shutdown. node-rank comes from sglang ServerArgs (set via the
    # injected --node-rank $LWS_WORKER_INDEX), with the LWS env as a fallback.
    node_rank = int(getattr(args.server_args, "node_rank", 0) or 0)
    if node_rank <= 0:
        try:
            node_rank = int(os.environ.get("LWS_WORKER_INDEX", "0") or "0")
        except ValueError:
            node_rank = 0
    if node_rank > 0:
        logger.info(
            "multinode follower (node-rank %d): TP worker only; skipping KV plane "
            "+ registration (node-rank 0 serves and registers).",
            node_rank,
        )
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        await stop.wait()
        await engine.stop()
        return

    # --- KV plane (best-effort under auto, fatal under on, skipped under off) ---
    kv_wiring = await _maybe_start_kv_plane(args, engine, model_name=config.model_name)
    if kv_wiring is not None:
        config.kv = kv_wiring.metadata

    # --- KV-event NATS relay (opt-in) ---
    # In NATS mode, forward this worker's engine KV events onto the broker so
    # the router subscribes once instead of dialing our kv_events_endpoint.
    # No-op without --enable-kv-events (nothing to relay).
    kv_relay = None
    if args.kv_event_transport == "nats" and config.kv_events_endpoint:
        from infera.kv.nats_relay import KvEventNatsRelay

        # SGLang --dp-size multiplexes DP ranks on base_port + r; relay tails
        # each. Single-rank (dp_size None/1) stays rank 0.
        _dp = config.dp_size or 1
        kv_relay = KvEventNatsRelay(
            worker_id=f"{config.host}:{config.port}",
            engine_zmq_endpoint=config.kv_events_endpoint,
            block_size=config.kv_block_size or 1,
            dp_size=_dp,
            multiplexed=_dp > 1,
            nats_url=args.nats_server,
        )
        try:
            await kv_relay.start()
        except Exception:
            logger.exception("KV NATS relay failed to start; continuing without it")
            kv_relay = None

    # --- Optional NATS request transport: run a consumer that proxies requests
    # from this worker's per-instance subject to the local engine HTTP. Advertise
    # request_transport=nats so the router publishes here instead of HTTP. ---
    config.request_transport = args.request_transport
    nats_req_server = None
    if args.request_transport == "nats":
        from infera.common.nats_request import NatsRequestServer

        worker_id = f"{config.host}:{config.port}"
        nats_req_server = NatsRequestServer(
            worker_id,
            config.port,
            url=args.nats_server,
            max_pending=args.nats_req_max_pending,
            idle_timeout=args.nats_req_idle_timeout,
            max_duration=args.nats_req_max_duration,
        )
        try:
            await nats_req_server.start()
        except Exception:
            logger.exception("NATS request consumer failed to start; registering as http instead")
            config.request_transport = "http"
            nats_req_server = None

    # --- Auto-registration (etcd or kubernetes) ---
    if args.discovery_backend == "kubernetes":
        from infera.common.registration_k8s import K8sRegistrationClient

        logger.info("using kubernetes registration: namespace=%s", args.k8s_namespace or "<pod>")
        reg_client = K8sRegistrationClient(namespace=args.k8s_namespace)
    else:
        if not args.etcd_endpoint:
            raise SystemExit("--discovery-backend=etcd requires --etcd-endpoint")
        logger.info(
            "using etcd registration: endpoint=%s prefix=%s",
            args.etcd_endpoint,
            args.etcd_prefix,
        )
        reg_client = RegistrationClient(
            endpoint=args.etcd_endpoint,
            prefix=args.etcd_prefix,
        )
    await reg_client.register(config)
    hb_task = asyncio.create_task(reg_client.heartbeat_loop(), name="worker-heartbeat")

    # --- Wait for shutdown signal (or engine subprocess death) ---
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    death_task = watch_engine_death(engine, stop)

    await stop.wait()

    # --- Graceful shutdown ---
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
    if nats_req_server is not None:
        await nats_req_server.stop(drain=True, drain_timeout=args.drain_timeout)

    if kv_relay is not None:
        await kv_relay.stop()

    if kv_wiring is not None:
        await kv_wiring.stop()

    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
