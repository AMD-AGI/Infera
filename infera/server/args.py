###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import argparse
import os

from infera.router.policy.factory import POLICY_NAMES


def parse_server_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infera server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--discovery-backend",
        choices=("etcd", "kubernetes"),
        default="kubernetes",
        help="Worker discovery transport. 'kubernetes' (default) watches worker "
        "Pods via the in-cluster API server (no etcd; workers self-register into "
        "their Pod annotation; requires --k8s-label-selector / "
        "$INFERA_K8S_LABEL_SELECTOR). 'etcd' watches an external etcd "
        "(--etcd-endpoint). 'kubernetes' is the infera analogue of dynamo's "
        "discoveryBackend=kubernetes.",
    )
    parser.add_argument(
        "--etcd-endpoint",
        default=None,
        help="Etcd endpoint (host:port, host, or http(s)://...). Required for "
        "--discovery-backend=etcd. The server watches --etcd-prefix for worker "
        "register / lease-expire events. Multiple servers can share one etcd.",
    )
    parser.add_argument(
        "--etcd-prefix",
        default="/infera/workers/",
        help="Etcd key prefix for worker records (default: /infera/workers/)",
    )
    parser.add_argument(
        "--k8s-label-selector",
        default=os.environ.get("INFERA_K8S_LABEL_SELECTOR"),
        help="Label selector for worker Pods with --discovery-backend=kubernetes "
        "(e.g. 'infera.amd.com/deployment=my-idep'). Required for that backend; "
        "scope it to this deployment's workers. Defaults to "
        "$INFERA_K8S_LABEL_SELECTOR (injected by the operator).",
    )
    parser.add_argument(
        "--k8s-namespace",
        default=None,
        help="Namespace to watch with --discovery-backend=kubernetes (default: "
        "the Pod's own namespace from the mounted ServiceAccount).",
    )
    parser.add_argument(
        "--request-transport",
        choices=("http", "nats"),
        default="nats",
        help="How the router reaches the selected worker. 'nats' (default) "
        "publishes to the worker's per-instance NATS subject and streams the reply "
        "over a reply inbox (dynamo-style; worker selection / kv-aware are "
        "unchanged; requires the workers to run request-transport=nats and a "
        "reachable --nats-server). 'http' forwards directly to the worker URL.",
    )
    parser.add_argument(
        "--router-backend",
        choices=("python", "rust"),
        default=os.environ.get("INFERA_ROUTER_BACKEND", "python"),
        help="Router data-plane implementation. 'python' (default) runs the "
        "in-process asyncio router. 'rust' execs the infera-router binary "
        "(multi-core data plane) for the hot path; it supports only "
        "--router-policy round-robin, --discovery-backend etcd, "
        "--request-transport http, --router-mode auto. Overrides "
        "$INFERA_ROUTER_BACKEND.",
    )
    parser.add_argument(
        "--router-mode",
        choices=("auto", "direct"),
        default=os.environ.get("INFERA_ROUTER_MODE", "auto"),
        help="Worker selection mode. 'auto' (default) selects the worker "
        "in-process via --router-policy (kv-aware / round-robin) and PD-preferred "
        "AutoRouter. 'direct' trusts an upstream GAIE Inference Gateway EPP: it "
        "dispatches to the worker named by the 'x-worker-instance-id' request "
        "header and skips in-process selection (falls back to policy pick when the "
        "header is absent). Overrides $INFERA_ROUTER_MODE.",
    )
    parser.add_argument(
        "--router-policy",
        choices=POLICY_NAMES,
        default="kv-aware",
        help="Worker selection strategy (used when --router-mode=auto). 'kv-aware' "
        "(default) subscribes to worker kv events and routes by prefix-cache "
        "locality (workers must publish kv events). 'round-robin' for a stateless "
        "spread.",
    )
    parser.add_argument(
        "--kv-event-transport",
        choices=("zmq", "nats"),
        default="nats",
        help="Transport for the kv-aware policy's per-worker event feed. "
        "'nats' (default) subscribes once to a NATS broker (infera.kv.events.>); "
        "workers must run a relay (kv-event-transport=nats) to forward their engine "
        "events. 'zmq' opens one SUB socket per worker directly to its "
        "kv_events_endpoint.",
    )
    parser.add_argument(
        "--nats-server",
        default=None,
        help="NATS server URL for --kv-event-transport=nats (default: "
        "$NATS_SERVER or nats://127.0.0.1:4222).",
    )
    parser.add_argument(
        "--nats-req-idle-timeout",
        type=float,
        default=None,
        help="NATS request IDLE (inactivity / stall) timeout in seconds: max wait "
        "for the NEXT reply chunk (reset per chunk; NOT an overall deadline). On "
        "expiry the router returns 504 and aborts the worker. Built-in default 900 "
        "(15 min); 0 = wait forever. Overrides $INFERA_NATS_REQ_IDLE_TIMEOUT.",
    )
    parser.add_argument(
        "--nats-req-max-duration",
        type=float,
        default=None,
        help="NATS request TOTAL (overall) timeout in seconds: hard wall-clock "
        "cap on the whole request regardless of token flow. On expiry the router "
        "interrupts the stream, returns 504 and aborts the worker. Built-in "
        "default 0 = OFF. Overrides $INFERA_NATS_REQ_MAX_DURATION when given.",
    )
    parser.add_argument(
        "--nats-req-max-pending",
        type=int,
        default=None,
        help="NATS request per-worker admission limit: max in-NATS backlog "
        "(num_pending + num_ack_pending) before the router returns 429. >0 makes "
        "the request path JetStream-backed. Built-in default 0 = OFF. Overrides "
        "$INFERA_NATS_REQ_MAX_PENDING when given.",
    )
    parser.add_argument(
        "--router-tokenizer-path",
        required=True,
        help="HuggingFace model id (e.g. Qwen/Qwen3-0.6B) or local path to a "
        "tokenizer.json file / directory containing one. Must match the "
        "tokenizer the workers use.",
    )
    parser.add_argument(
        "--kv-overlap-weight",
        type=float,
        default=1.0,
        help="kv-aware only: weight on the cache-locality term in "
        "cost = w * (request_blocks - hits) + active_blocks. Larger values "
        "favour cache reuse over load balance (default: 1.0). Used for "
        "mixed-pool routing and as the fallback when the prefill/decode "
        "weights below are unset.",
    )
    parser.add_argument(
        "--kv-prefill-overlap-weight",
        type=float,
        default=None,
        help="kv-aware + PD only: overlap weight when picking a PREFILL "
        "worker. Prefill is compute-bound and benefits enormously from "
        "cache hits (a hit skips the entire prefill pass), so this is "
        "set high. Defaults to --kv-overlap-weight if unset; typical "
        "production value: 20.0.",
    )
    parser.add_argument(
        "--kv-decode-overlap-weight",
        type=float,
        default=None,
        help="kv-aware + PD only: overlap weight when picking a DECODE "
        "worker. Decode is memory-bound on KV transfer and doesn't get "
        "much from a prefill-time cache hit, so this is low — routing "
        "by load is more important. Defaults to --kv-overlap-weight if "
        "unset; typical production value: 2.0.",
    )
    # Issue #20 item 3 / PD §6.2 — speculative L3 prefetch endpoint.
    parser.add_argument(
        "--kvd-socket-path",
        default=None,
        help="UDS path for the kvd daemon. When set, exposes "
        "POST /v1/cache/prewarm so agentic harnesses can fire "
        "PrefetchHint frames at kvd (warms L3 → host RAM ahead of "
        "the next inference call). Without this flag the endpoint "
        "returns 503. Mirrors Anthropic's `cache_control` + NVIDIA "
        "Dynamo's `nvext.cache_control`; see PD §6.2.",
    )
    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        default=os.environ.get("INFERA_ENABLE_PROFILING", "").lower() in ("1", "true", "yes"),
        help="Enable the unified torch-profiler control plane: "
        "POST /v1/admin/profile/start|stop forwards to each worker's native "
        "engine /start_profile|/stop_profile (SGLang/vLLM). Default OFF "
        "(returns 403), mirroring dynamo's system status server being disabled "
        "unless DYN_SYSTEM_PORT is set. Also enabled via "
        "$INFERA_ENABLE_PROFILING=1.",
    )
    parser.add_argument(
        "--request-max-retries",
        type=int,
        default=int(os.environ.get("INFERA_REQUEST_MAX_RETRIES", "1") or 1),
        help="Bounded failover: number of ALTERNATE mixed workers to try if a "
        "dispatch fails BEFORE any response data has reached the client "
        "(unreachable / NATS error / idle-timeout-before-first-token / 429 "
        "backlog). Mid-stream failures are never retried. Default 1; 0 disables. "
        "Overrides $INFERA_REQUEST_MAX_RETRIES.",
    )
    return parser.parse_args(argv)
