###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import asyncio
import logging

import httpx
import uvicorn

from infera.common.discovery import Registry
from infera.common.discovery_k8s import KubernetesRegistry
from infera.common.tokenizer import resolve_tokenizer_path
from infera.common.worker_pool import WorkerInfo
from infera.kv.api import make_http_snapshot_puller, make_stats_router
from infera.kv.index import KVIndex
from infera.kv.snapshot import SnapshotReconciler
from infera.kv.subscriber import KvEventSubscriberPool
from infera.kv.writer import KvIndexWriter
from infera.router.auto import AutoRouter
from infera.router.direct import DirectRouter
from infera.router.policy.factory import build_policy
from infera.server.app import init_app
from infera.server.args import parse_server_args

logging.basicConfig(level=logging.INFO)
# httpx logs every outbound request (etcd keepalives every ~10s, and one
# per inference request when proxying). At benchmark concurrency that
# floods the log; surface only warnings/errors.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def main(args) -> None:
    tokenizer_path = resolve_tokenizer_path(args.router_tokenizer_path)

    # --- Policy (kv-aware or round-robin) ---
    policy = build_policy(
        args.router_policy,
        overlap_weight=args.kv_overlap_weight,
        prefill_overlap_weight=args.kv_prefill_overlap_weight,
        decode_overlap_weight=args.kv_decode_overlap_weight,
        tokenizer_path=tokenizer_path,
        kv_event_transport=args.kv_event_transport,
        nats_server=args.nats_server,
    )
    if args.router_policy == "kv-aware":
        logger.info(
            "router-policy=%s overlap_weight=%g prefill=%s decode=%s",
            args.router_policy,
            args.kv_overlap_weight,
            args.kv_prefill_overlap_weight
            if args.kv_prefill_overlap_weight is not None
            else f"(default {args.kv_overlap_weight})",
            args.kv_decode_overlap_weight
            if args.kv_decode_overlap_weight is not None
            else f"(default {args.kv_overlap_weight})",
        )
    else:
        logger.info("router-policy=%s", args.router_policy)

    # --- Phase 1 KV data plane: index + queue + writer + subscribers + reconciler ---
    # Active only for workers that registered the nested `kv` block (which
    # carries tokenizer_digest, canary, snapshot_endpoint, etc.). Workers
    # that only set the flat `kv_events_endpoint` / `kv_block_size` go
    # through PR #10's KvEventClient via the policy callbacks below.
    index = KVIndex()
    event_queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)
    writer = KvIndexWriter(index=index, queue=event_queue)
    subscribers = KvEventSubscriberPool(output_queue=event_queue)
    # HTTP snapshot reconciliation is RETIRED under the NATS transport: the
    # NatsKvEventClient bootstraps and self-heals the routing view from the
    # JetStream KV bucket instead of pulling each worker's /v1/kv-snapshot.
    # Only the ZMQ transport still uses the HTTP SnapshotReconciler.
    use_http_snapshot = args.kv_event_transport != "nats"
    http_client: httpx.AsyncClient | None = None
    reconciler: SnapshotReconciler | None = None
    if use_http_snapshot:
        http_client = httpx.AsyncClient(timeout=10.0)
        snapshot_pull = make_http_snapshot_puller(client=http_client)
        reconciler = SnapshotReconciler(
            index=index,
            writer=writer,
            pull_fn=snapshot_pull,
            interval_s=30.0,
        )
        # Close the loop: writer reports each applied batch_id back to the
        # reconciler so its 30 s staleness check correctly skips snapshots
        # older than what's already in the index.
        writer.set_reconciler(reconciler)

    # Per-worker snapshot of fields we need at removal time. The Registry
    # has already evicted the WorkerInfo from its pool by the time
    # on_worker_removed fires, so we stash the kv block at registration.
    # 4-tuple: (events_endpoint, snapshot_endpoint, compat_key, model_name)
    # — model_name is needed to unregister the reconciler target by the
    # SAME (publisher, endpoint, model, compat_key) key it was
    # registered under. PR #9 review fix P1: previously this was a
    # 3-tuple and unregister passed model="" → pop missed → orphan
    # target lingered and reconcile spent cycles pulling from a dead
    # worker.
    kv_snapshots: dict[str, tuple[str, str, str, str]] = {}

    # --- Worker lifecycle callbacks ---
    # Both the policy (PR #10's kv-aware) and the Phase 1 reconciler want
    # to react to fleet changes. We fan out one Registry callback into
    # both consumers; either may run even when the other has no work.
    def on_worker_added(info: WorkerInfo) -> None:
        # PR #10 kv-aware: subscribe via the flat endpoint.
        try:
            policy.on_worker_added(info)
        except Exception:
            logger.exception("policy.on_worker_added failed for %s", info.worker_id)

        # Phase 1 nested kv block: ZMQ subscribe + reconciler register.
        if info.kv is None or not info.kv.supports_events:
            return
        ep = info.kv.events_endpoint
        if ep:
            logger.info(
                "subscribing to %s @ %s for %s",
                info.worker_id,
                ep,
                info.model_name,
            )
            asyncio.create_task(subscribers.add(ep), name=f"sub-add-{info.worker_id}")
        # Snapshot endpoint defaults to the worker's main URL if the kv
        # block didn't specify a dedicated one.
        snapshot_endpoint = info.kv.snapshot_endpoint or info.url
        compat_key = info.kv.tokenizer_digest
        if reconciler is not None:
            reconciler.register_target(
                publisher_id=info.worker_id,
                endpoint=snapshot_endpoint,
                model=info.model_name,
                compat_key=compat_key,
            )
        kv_snapshots[info.worker_id] = (
            ep or "",
            snapshot_endpoint,
            compat_key,
            info.model_name,
        )

    def on_worker_removed(worker_id: str) -> None:
        # PR #10 kv-aware.
        try:
            policy.on_worker_removed(worker_id)
        except Exception:
            logger.exception("policy.on_worker_removed failed for %s", worker_id)

        # Phase 1 reconciler/subscriber cleanup.
        snap = kv_snapshots.pop(worker_id, None)
        if snap is None:
            return
        events_endpoint, snapshot_endpoint, compat_key, model_name = snap
        if events_endpoint:
            asyncio.create_task(subscribers.remove(events_endpoint), name=f"sub-rm-{worker_id}")
            # Drop the publisher's tree from the index so old entries don't pile up.
            index.drop_publisher(worker_id)
        if reconciler is not None:
            reconciler.unregister_target(
                publisher_id=worker_id,
                endpoint=snapshot_endpoint,
                model=model_name,
                compat_key=compat_key,
            )

    # --- Worker discovery (etcd or kubernetes) ---
    if args.discovery_backend == "kubernetes":
        if not args.k8s_label_selector:
            raise SystemExit(
                "--discovery-backend=kubernetes requires --k8s-label-selector "
                "(e.g. 'infera.amd.com/deployment=<name>')"
            )
        logger.info(
            "using kubernetes discovery: selector=%r namespace=%s",
            args.k8s_label_selector,
            args.k8s_namespace or "<pod-namespace>",
        )
        registry = KubernetesRegistry(
            label_selector=args.k8s_label_selector,
            namespace=args.k8s_namespace,
            on_worker_added=on_worker_added,
            on_worker_removed=on_worker_removed,
        )
    else:
        if not args.etcd_endpoint:
            raise SystemExit("--discovery-backend=etcd requires --etcd-endpoint")
        logger.info(
            "using etcd discovery: endpoint=%s prefix=%s",
            args.etcd_endpoint,
            args.etcd_prefix,
        )
        registry = Registry(
            endpoint=args.etcd_endpoint,
            prefix=args.etcd_prefix,
            on_worker_added=on_worker_added,
            on_worker_removed=on_worker_removed,
        )

    # --- Start data plane BEFORE registry so callbacks land in working tasks ---
    await writer.start()
    if reconciler is not None:
        await reconciler.start()
    await registry.start()

    # --- Optional NATS request transport (dynamo-style per-instance routing) ---
    nats_request_client = None
    if args.request_transport == "nats":
        from infera.common.nats_request import NatsRequestClient

        nats_request_client = NatsRequestClient(
            args.nats_server,
            max_pending=args.nats_req_max_pending,
            idle_timeout=args.nats_req_idle_timeout,
            max_duration=args.nats_req_max_duration,
        )
        await nats_request_client.start()
        logger.info("request transport: nats (per-instance subjects)")

    # --- Router + FastAPI app ---
    # router-mode=direct trusts an upstream GAIE EPP's per-request worker
    # selection (x-worker-instance-id header); auto selects in-process.
    if args.router_mode == "direct":
        router = DirectRouter(
            registry.pool,
            policy,
            nats_client=nats_request_client,
            request_max_retries=args.request_max_retries,
        )
        logger.info("router-mode=direct (honouring GAIE EPP x-worker-instance-id)")
    else:
        router = AutoRouter(
            registry.pool,
            policy,
            nats_client=nats_request_client,
            request_max_retries=args.request_max_retries,
        )
    app = init_app(
        registry,
        router,
        kv=policy.kv_client,
        kvd_socket_path=args.kvd_socket_path,
        enable_profiling=args.enable_profiling,
    )
    app.include_router(
        make_stats_router(
            index=index,
            writer=writer,
            subscribers=subscribers,
            reconciler=reconciler,
        )
    )

    config = uvicorn.Config(app, host=args.host, port=args.port)
    server = uvicorn.Server(config)

    logger.info("starting Infera on %s:%d", args.host, args.port)
    try:
        await server.serve()
    finally:
        await registry.stop()
        if reconciler is not None:
            await reconciler.stop()
        await subscribers.stop_all()
        await writer.stop()
        if http_client is not None:
            await http_client.aclose()
        await router.aclose()
        await policy.aclose()
        if nats_request_client is not None:
            await nats_request_client.aclose()


if __name__ == "__main__":
    _args = parse_server_args()
    # `--router-backend rust` replaces this process with the Rust router binary.
    if _args.router_backend == "rust":
        from infera.server.launch_rust import exec_rust

        exec_rust(_args)  # never returns
    asyncio.run(main(_args))
