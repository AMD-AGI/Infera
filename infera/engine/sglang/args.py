###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field

from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SglangWorkerArgs:
    server_args: ServerArgs
    discovery_backend: str  # "etcd" | "kubernetes"
    etcd_endpoint: str | None
    etcd_prefix: str
    k8s_namespace: str | None
    request_transport: str  # "http" | "nats"
    # NATS request timeouts / admission (None => env $INFERA_NATS_REQ_* or the
    # built-in defaults in infera.common.nats_request).
    nats_req_idle_timeout: float | None = None
    nats_req_max_duration: float | None = None
    nats_req_max_pending: int | None = None
    drain_timeout: float = 30.0
    advertise_host: str | None
    # Benchmark-only escape hatch: skip the disagg transport preflight that
    # rejects configs prone to a silent TCP fallback. Default False.
    disaggregation_allow_tcp: bool
    # PR #10: SGLang-native KV event publisher for KvEventClient.
    enable_kv_events: bool
    # Original argv (everything we didn't consume) forwarded verbatim to the
    # `sglang.launch_server` subprocess. Re-parsing ServerArgs would lose
    # multi-value list flags (`--cuda-graph-bs 1 2 3 ...`) and other quirks.
    sglang_argv: list[str] = field(default_factory=list)

    # Phase 1: KvEventProbe + snapshot server. `auto` enables iff
    # tokenizer loads cleanly. `off` skips the whole worker-side KV
    # plane; the worker still registers but with `kv=None`.
    kv_events: str  # "on" | "off" | "auto"
    kv_events_bind: str  # tcp://0.0.0.0:<port> — what the publisher binds to
    kv_events_advertise: str | None  # what to register; default = derived from bind + host
    kv_snapshot_host: str  # bind for the snapshot HTTP server
    kv_snapshot_port: int  # bind for the snapshot HTTP server
    kv_snapshot_advertise: str | None  # base URL the server pulls from
    index_block_size: int

    # KV-event transport for KV-aware routing: "zmq" (router connects to
    # this worker's kv_events_endpoint directly) or "nats" (this worker
    # relays its engine events onto a NATS broker). Default "nats".
    kv_event_transport: str
    nats_server: str | None

    # Phase 4.5+: infera-kvd HiCacheStorage backend.
    # When set, register `infera-kvd` with SGLang's StorageBackendFactory
    # before launch_server, and (if not already set on server_args) inject
    # the SGLang flags that select our backend.
    infera_kvd_socket: str | None  # UDS path the kvd daemon listens on


def parse_sglang_args(argv: list[str] | None = None) -> SglangWorkerArgs:
    parser = argparse.ArgumentParser(add_help=True)

    # Infera-specific args (not forwarded to SGLang).
    # Disagg role is read directly from SGLang's --disaggregation-mode.
    parser.add_argument(
        "--discovery-backend",
        choices=("etcd", "kubernetes"),
        default="kubernetes",
        help="Self-registration transport. 'kubernetes' (default) writes the "
        "worker record into this worker's own Pod annotation (no etcd; requires "
        "POD_NAME/POD_NAMESPACE downward API + RBAC to patch its own Pod). 'etcd' "
        "takes an etcd lease and PUTs the worker record (--etcd-endpoint).",
    )
    parser.add_argument(
        "--etcd-endpoint",
        default=None,
        help="Etcd endpoint (host:port, host, or http(s)://...) for "
        "lease-based self-registration. Required for --discovery-backend=etcd. "
        "All Infera servers watching the same --etcd-prefix see this worker.",
    )
    parser.add_argument(
        "--etcd-prefix",
        default="/infera/workers/",
        help="Etcd key prefix (default: /infera/workers/)",
    )
    parser.add_argument(
        "--k8s-namespace",
        default=None,
        help="Namespace of this worker's Pod for --discovery-backend=kubernetes "
        "(default: POD_NAMESPACE env / the Pod's mounted ServiceAccount namespace).",
    )
    parser.add_argument(
        "--request-transport",
        choices=("http", "nats"),
        default="nats",
        help="How the router reaches this worker. 'nats' (default) runs a NATS "
        "consumer that proxies requests from this worker's per-instance subject to "
        "the local engine HTTP (worker advertises request_transport=nats so the "
        "router uses NATS; requires a reachable --nats-server / $NATS_SERVER). "
        "'http' serves the engine HTTP directly.",
    )
    parser.add_argument(
        "--nats-req-idle-timeout",
        type=float,
        default=None,
        help="NATS request idle (inactivity) timeout (s) for the worker's local "
        "read timeout. None => $INFERA_NATS_REQ_IDLE_TIMEOUT or default 900.",
    )
    parser.add_argument(
        "--nats-req-max-duration",
        type=float,
        default=None,
        help="NATS request total (overall) timeout (s); the worker hard-aborts a "
        "request at this wall-clock cap. None => $INFERA_NATS_REQ_MAX_DURATION "
        "or built-in default 0 (off).",
    )
    parser.add_argument(
        "--nats-req-max-pending",
        type=int,
        default=None,
        help="NATS request admission limit; >0 makes this worker consume requests "
        "via a JetStream consumer so the router can throttle by backlog. None => "
        "$INFERA_NATS_REQ_MAX_PENDING or built-in default 0 (off).",
    )
    parser.add_argument(
        "--drain-timeout",
        type=float,
        default=float(__import__("os").environ.get("INFERA_DRAIN_TIMEOUT", "30") or 30),
        help="Graceful shutdown: on SIGTERM the worker stops accepting new NATS "
        "requests and lets in-flight generations finish for up to this many "
        "seconds before cancelling leftovers (rolling-upgrade drain). Default 30; "
        "0 = cancel in-flight immediately. Overrides $INFERA_DRAIN_TIMEOUT.",
    )
    parser.add_argument(
        "--advertise-host",
        default=None,
        help="Host/IP to publish to etcd (worker_id, url, bootstrap_addr). "
        "Use this when the engine binds on 0.0.0.0 but peers need to "
        "reach it via a routable address. Defaults to --host.",
    )
    parser.add_argument(
        "--enable-kv-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish KV cache events on a ZMQ socket for KV-aware routing "
        "(default on; pass --no-enable-kv-events to disable). A free port is "
        "allocated automatically and reported to the router via "
        "WorkerInfo.kv_events_endpoint.",
    )
    parser.add_argument(
        "--disaggregation-allow-tcp",
        action="store_true",
        default=False,
        help="Benchmark-only: skip the disagg transport preflight that "
        "rejects configs prone to a silent TCP fallback (no explicit RDMA "
        "--disaggregation-transfer-backend). Do NOT use in production.",
    )

    # KV management.
    parser.add_argument(
        "--kv-events",
        choices=("on", "off", "auto"),
        default="auto",
        help="Enable per-worker KV event publishing (ZMQ PUB). 'auto' "
        "(default) turns it on iff tokenizer loads cleanly. 'off' "
        "skips it: the worker still registers but the router falls "
        "back to round-robin for it.",
    )
    parser.add_argument(
        "--kv-events-bind",
        default="tcp://0.0.0.0:5557",
        help="ZMQ PUB endpoint to bind. Default tcp://0.0.0.0:5557.",
    )
    parser.add_argument(
        "--kv-events-advertise",
        default=None,
        help="ZMQ endpoint to advertise to the server (etcd kv block). "
        "Default: derived from --kv-events-bind, with host filled from "
        "SGLang --host if the bind uses 0.0.0.0.",
    )
    parser.add_argument(
        "--kv-snapshot-host",
        default="0.0.0.0",
        help="Bind host for the per-worker snapshot HTTP server. Default 0.0.0.0.",
    )
    parser.add_argument(
        "--kv-snapshot-port",
        type=int,
        default=8801,
        help="Bind port for the per-worker snapshot HTTP server. Default 8801.",
    )
    parser.add_argument(
        "--kv-snapshot-advertise",
        default=None,
        help="Base URL the server should pull /v1/kv-snapshot from. "
        "Default: derived from --kv-snapshot-port + SGLang --host.",
    )
    parser.add_argument(
        "--index-block-size",
        type=int,
        default=64,
        help="Block size used for index_block hashing. This is the "
        "router's coalescing unit, NOT the engine's KV page size. "
        "Default 64.",
    )

    parser.add_argument(
        "--kv-event-transport",
        choices=("zmq", "nats"),
        default="nats",
        help="How this worker exposes KV events to the router. 'nats' "
        "(default): relay engine events onto a NATS broker so the router can "
        "subscribe once (infera.kv.events.>). 'zmq': engine publishes on "
        "kv_events_endpoint, router connects directly. Requires --enable-kv-events.",
    )
    parser.add_argument(
        "--nats-server",
        default=None,
        help="NATS server URL for --kv-event-transport=nats (default: "
        "$NATS_SERVER or nats://127.0.0.1:4222).",
    )

    # Phase 4.5+: infera-kvd HiCacheStorage adapter wiring.
    parser.add_argument(
        "--infera-kvd-socket",
        default=None,
        help="UDS path the infera-kvd daemon listens on. When set, the "
        "infera-kvd backend is registered with SGLang's storage factory "
        "and selected via --hicache-storage-backend infera-kvd. The "
        "socket must be reachable BEFORE the engine starts (the worker "
        "probes it and aborts on failure rather than running with a "
        "silently-broken cache backend). Also sets INFERA_KVD_SOCKET "
        "in the environment for child processes.",
    )

    # Split our args from SGLang's args
    known, remaining = parser.parse_known_args(argv)

    # Parse the rest as SGLang ServerArgs
    sglang_parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(sglang_parser)
    sglang_parsed = sglang_parser.parse_args(remaining)

    # When KV events are on, enable the decode-side prefix radix cache so the
    # router can steer repeats to the rank holding the prefix and prefill only
    # transfers the delta. SGLang's flag defaults off; append it to the forwarded
    # argv so the launch_server subprocess (which is what re-parses these) gets it.
    # SGLang only accepts this flag with the mooncake transfer backend; with mori
    # (or nixl in our stack) it aborts, so gate the append on the backend.
    if (
        known.enable_kv_events
        and sglang_parsed.disaggregation_mode == "decode"
        and getattr(sglang_parsed, "disaggregation_transfer_backend", None) == "mooncake"
        and "--disaggregation-decode-enable-radix-cache" not in remaining
    ):
        remaining.append("--disaggregation-decode-enable-radix-cache")

    server_args = ServerArgs.from_cli_args(sglang_parsed)

    # infera product default: fp8 KV cache (fp8_e4m3) unless the operator passed
    # --kv-cache-dtype explicitly. fp8 halves the KV footprint -> ~2x the KV that
    # fits in VRAM and halves PD KV-transfer + RDMA memory-registration volume
    # (bf16 hit ionic ibv_reg_mr ENOMEM at high concurrency / long inputs). Small
    # accuracy cost; opt out with --kv-cache-dtype auto|bf16 or INFERA_DEFAULT_KV_FP8=0.
    if os.environ.get("INFERA_DEFAULT_KV_FP8", "1") != "0" and not any(
        t == "--kv-cache-dtype" or t.startswith("--kv-cache-dtype=") for t in remaining
    ):
        server_args.kv_cache_dtype = "fp8_e4m3"
        logger.info(
            "infera default: kv_cache_dtype=fp8_e4m3 "
            "(override with --kv-cache-dtype or INFERA_DEFAULT_KV_FP8=0)"
        )

    from infera.engine.sglang.hicache_validate import warn_if_hicache_prefetch_disabled

    warn_if_hicache_prefetch_disabled(server_args)

    return SglangWorkerArgs(
        server_args=server_args,
        discovery_backend=known.discovery_backend,
        etcd_endpoint=known.etcd_endpoint,
        etcd_prefix=known.etcd_prefix,
        k8s_namespace=known.k8s_namespace,
        request_transport=known.request_transport,
        nats_req_idle_timeout=known.nats_req_idle_timeout,
        nats_req_max_duration=known.nats_req_max_duration,
        nats_req_max_pending=known.nats_req_max_pending,
        drain_timeout=known.drain_timeout,
        advertise_host=known.advertise_host,
        disaggregation_allow_tcp=known.disaggregation_allow_tcp,
        enable_kv_events=known.enable_kv_events,
        sglang_argv=list(remaining),
        kv_events=known.kv_events,
        kv_events_bind=known.kv_events_bind,
        kv_events_advertise=known.kv_events_advertise,
        kv_snapshot_host=known.kv_snapshot_host,
        kv_snapshot_port=known.kv_snapshot_port,
        kv_snapshot_advertise=known.kv_snapshot_advertise,
        index_block_size=known.index_block_size,
        kv_event_transport=known.kv_event_transport,
        nats_server=known.nats_server,
        infera_kvd_socket=known.infera_kvd_socket,
    )


# `warn_if_hicache_prefetch_disabled` lives in
# `infera.engine.sglang.hicache_validate` — kept separate so the
# unit tests don't have to import sglang.srt.server_args.
