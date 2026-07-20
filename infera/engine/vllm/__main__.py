###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Entrypoint for `python -m infera.engine.vllm`. Spawns `vllm serve`
as a subprocess, optionally wires KV events for KV-aware routing, then
self-registers with etcd."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import signal

from infera.common.disagg_preflight import (
    validate_advertise_host,
    validate_vllm_transport,
)
from infera.common.net import free_tcp_port
from infera.common.registration import RegistrationClient
from infera.common.worker_pool import DisaggMode, KvRegistrationMetadata
from infera.engine.base import watch_engine_death
from infera.engine.vllm.args import VllmWorkerArgs, parse_vllm_args
from infera.engine.vllm.worker import VllmEngine

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # silence etcd keepalive spam
logger = logging.getLogger(__name__)


def _inject_kv_events_config(args: VllmWorkerArgs) -> str | None:
    """Inject --kv-events-config into vLLM argv. Returns the advertise URL or None."""
    if not args.enable_kv_events:
        return None

    if not os.environ.get("PYTHONHASHSEED"):
        # vLLM's NONE_HASH defaults to os.urandom(32); without a fixed seed,
        # block hashes shuffle every restart. kv-aware re-hashes on the
        # router side so it's fine, but hash-based PD connectors break.
        logger.warning(
            "PYTHONHASHSEED is not set; set it (e.g. 0) for cross-restart "
            "determinism if you use hash-based PD connectors."
        )
    # infera's msgspec schema decodes block hashes as list[int]; tell
    # vLLM to emit 8-byte ints rather than 32-byte sha256 digests.
    os.environ["VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES"] = "1"

    port = free_tcp_port()
    # vLLM's ZmqEventPublisher decides bind-vs-connect by a string heuristic:
    # `*` or `::` means bind, anything else means connect (see
    # vllm/distributed/kv_events.py _socket_setup). `tcp://*:<port>` is the
    # only form that gets the PUB socket to actually bind.
    bind = f"tcp://*:{port}"
    advertise = f"tcp://{args.advertise_host or args.host}:{port}"
    if args.advertise_host is None and args.host in ("0.0.0.0", ""):
        logger.warning(
            "--enable-kv-events with host=0.0.0.0 and no --advertise-host; "
            "KV events endpoint %s won't be reachable by remote subscribers",
            advertise,
        )

    cfg = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": bind,
        # vLLM defaults to "" (empty topic) which our KvEventClient won't match.
        "topic": "kv-events",
    }
    args.vllm_argv += ["--kv-events-config", json.dumps(cfg)]
    logger.info("--enable-kv-events ⇒ --kv-events-config %s; advertise=%s", cfg, advertise)
    return advertise


def _compute_kv_metadata(
    args: VllmWorkerArgs,
    *,
    model_name: str,
    events_endpoint: str | None,
) -> KvRegistrationMetadata | None:
    """tokenizer digest + canary (`auto` swallows, `on` fails fast, `off` skips)."""
    if args.kv_events == "off":
        logger.info("--kv-events off; worker will register with kv=None")
        return None

    try:
        from infera.engine.sglang.kv_wiring import compute_kv_metadata

        metadata = compute_kv_metadata(
            model_id=args.model,
            # vLLM's --block-size defaults to None (engine auto-selects);
            # fall back to vLLM's ROCm default of 16 so we never register
            # engine_block_size=None, which the server's from_dict rejects
            # (int(None)) and would drop the whole worker. Mirrors the
            # SGLang launcher's `page_size or 1`.
            engine_block_size=args.block_size or 16,
            index_block_size=args.index_block_size,
            events_endpoint=events_endpoint or "",
            snapshot_endpoint=f"http://{args.advertise_host or args.host}:{args.port}",
            trust_remote_code=args.trust_remote_code,
        )
        # Publisher disabled but digest still wanted: clear endpoint so
        # the router doesn't try to subscribe.
        if not events_endpoint:
            metadata = dataclasses.replace(metadata, events_endpoint=None, supports_events=False)
        logger.info(
            "kv metadata: model=%s digest=%s canary_len=%d engine_block=%d index_block=%d",
            model_name,
            metadata.tokenizer_digest,
            len(metadata.tokenizer_canary),
            metadata.engine_block_size,
            metadata.index_block_size,
        )
        return metadata
    except Exception:
        if args.kv_events == "on":
            raise
        logger.exception("--kv-events auto: metadata computation failed; registering with kv=None")
        return None


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
    args = parse_vllm_args()

    # Ionic RoCE-v2 RDMA env defaults for the Mooncake/MoRIIO connectors
    # (set-if-unset; override via env). Must run before the engine starts so
    # the transfer connector picks up the GID/HIP-transport settings instead of
    # silently falling back to TCP.
    from infera.engine.rocm_rdma_env import (
        apply_kv_host_ip_default,
        apply_mooncake_topology_default,
        apply_rocm_rdma_env_defaults,
        apply_vllm_aiter_default,
    )

    apply_rocm_rdma_env_defaults()
    # Default AITER on (set-if-unset): MXFP4 MoE models (MiniMax/Kimi) otherwise
    # fail at load with "No MXFP4 MoE backend". Override with VLLM_ROCM_USE_AITER=0.
    apply_vllm_aiter_default()
    # Pin VLLM_HOST_IP to the RDMA rail so the Mooncake/MoRIIO connector advertises
    # its ionic address (not the public NIC vLLM's get_ip() would pick) and moves KV
    # over RDMA rather than the wrong interface. Must follow the GID default above.
    apply_kv_host_ip_default()
    # Pin Mooncake's per-GPU HCA to one NIC when RoCE NICs span multiple subnets,
    # so prefill/decode don't land on cross-subnet NICs (QP-RTR [110] storm).
    apply_mooncake_topology_default()

    # Kubernetes discovery: advertise the routable Pod IP (downward API) rather
    # than the 0.0.0.0 bind host so the server/peers can reach this worker.
    if args.discovery_backend == "kubernetes" and not args.advertise_host:
        pod_ip = os.environ.get("POD_IP")
        if pod_ip:
            args.advertise_host = pod_ip
            logger.info("k8s discovery: advertising Pod IP %s", pod_ip)

    logger.info(
        "parsed args: model=%s host=%s port=%d disagg=%s disagg_meta=%s enable_kv_events=%s",
        args.model,
        args.host,
        args.port,
        args.disagg_mode,
        # Empty for MIXED workers; {"protocol": ..., "params": ...} otherwise.
        args.disagg_meta or "{}",
        args.enable_kv_events,
    )

    # Fail fast on disagg configs that silently break across nodes: a
    # non-routable advertise host, or no recognized RDMA KV connector.
    is_disagg = args.disagg_mode != DisaggMode.MIXED
    validate_advertise_host(args.advertise_host or args.host, is_disagg=is_disagg)
    validate_vllm_transport(
        args.disagg_meta, is_disagg=is_disagg, allow_tcp=args.disaggregation_allow_tcp
    )

    kv_events_endpoint = _inject_kv_events_config(args)

    model_name = args.served_model_name or args.model

    engine = VllmEngine(
        vllm_argv=args.vllm_argv,
        model_name=model_name,
        host=args.host,
        port=args.port,
        advertise_host=args.advertise_host,
        kv_events_endpoint=kv_events_endpoint,
        kv_block_size=args.block_size,
        disagg_mode=args.disagg_mode,
        disagg_meta=args.disagg_meta,
        dp_rank=args.dp_rank,
        dp_size=args.dp_size,
    )

    try:
        config = await engine.start()
    except Exception:
        logger.exception("vllm engine failed to start; tearing down")
        try:
            await engine.stop()
        finally:
            _kill_process_group_safely()
        raise
    logger.info(
        "worker ready: model=%s url=http://%s:%d engine=%s disagg=%s kv_events=%s",
        config.model_name,
        config.host,
        config.port,
        config.engine,
        config.disagg_mode,
        kv_events_endpoint,
    )

    config.kv = _compute_kv_metadata(
        args, model_name=model_name, events_endpoint=kv_events_endpoint
    )

    # KV-event NATS relay (opt-in). vLLM DP is external-LB (each rank is its
    # own worker/port), so the relay is single-rank per worker.
    kv_relay = None
    if args.kv_event_transport == "nats" and kv_events_endpoint:
        from infera.kv.nats_relay import KvEventNatsRelay

        kv_relay = KvEventNatsRelay(
            worker_id=f"{config.host}:{config.port}",
            engine_zmq_endpoint=kv_events_endpoint,
            block_size=args.block_size or 1,
            nats_url=args.nats_server,
        )
        try:
            await kv_relay.start()
        except Exception:
            logger.exception("KV NATS relay failed to start; continuing without it")
            kv_relay = None

    # Multinode follower gate: only node-rank 0 serves + registers; ranks > 0
    # are pure TP workers (registering them would route requests to a
    # non-serving endpoint). Use the LWS worker index as the rank signal.
    try:
        _node_rank = int(os.environ.get("LWS_WORKER_INDEX", "0") or "0")
    except ValueError:
        _node_rank = 0
    if _node_rank > 0:
        logger.info(
            "multinode follower (LWS_WORKER_INDEX=%d): TP worker only; skipping "
            "registration (node-rank 0 serves and registers).",
            _node_rank,
        )
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        await stop.wait()
        await engine.stop()
        return

    # Optional NATS request transport: proxy this worker's per-instance subject
    # to the local engine HTTP and advertise request_transport=nats.
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
            logger.exception("NATS request consumer failed to start; registering as http")
            config.request_transport = "http"
            nats_req_server = None

    if args.discovery_backend == "kubernetes":
        from infera.common.registration_k8s import K8sRegistrationClient

        logger.info(
            "registering via k8s Pod annotation: namespace=%s", args.k8s_namespace or "<pod>"
        )
        reg_client = K8sRegistrationClient(namespace=args.k8s_namespace)
    else:
        if not args.etcd_endpoint:
            raise SystemExit("--discovery-backend=etcd requires --etcd-endpoint")
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
    if nats_req_server is not None:
        await nats_req_server.stop(drain=True, drain_timeout=args.drain_timeout)
    if kv_relay is not None:
        await kv_relay.stop()
    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
