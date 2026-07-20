###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import argparse
import os

from infera.router.policy.factory import POLICY_NAMES


def parse_epp_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI for the GAIE Endpoint Picker (ext_proc) server. Flag names mirror
    ``infera.server.args`` where the knob is shared (discovery, policy, kv,
    tokenizer) so an operator configures both the same way."""
    parser = argparse.ArgumentParser(description="Infera GAIE Endpoint Picker (ext_proc)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=int(os.environ.get("INFERA_EPP_GRPC_PORT", "9002") or 9002),
        help="ext_proc gRPC port the gateway connects to (default 9002).",
    )
    parser.add_argument(
        "--grpc-health-port",
        type=int,
        default=int(os.environ.get("INFERA_EPP_GRPC_HEALTH_PORT", "9003") or 9003),
        help="gRPC health-check port (default 9003). Probed by the k8s "
        "readiness/liveness grpc probe with service 'inference-extension'.",
    )
    parser.add_argument(
        "--destination-port",
        type=int,
        default=int(os.environ.get("INFERA_EPP_DEST_PORT", "8000") or 8000),
        help="Port of the gateway-routable worker endpoint (the direct-mode "
        "frontend sidecar = the InferencePool target port, default 8000). The "
        "picker emits <pod-ip>:<this> in x-gateway-destination-endpoint; the "
        "worker's own URL carries the engine port, which is not a pool member.",
    )
    parser.add_argument(
        "--secure-serving",
        dest="secure_serving",
        action=argparse.BooleanOptionalAction,
        default=(os.environ.get("INFERA_EPP_SECURE", "true").lower() != "false"),
        help="Serve the ext_proc gRPC port over TLS with a self-signed cert "
        "(default true). Inference Gateways (kgateway) connect to the EndpointPicker "
        "over TLS with verification skipped; plaintext serving fails the TLS handshake.",
    )

    # --- discovery (same semantics as the server) ---
    parser.add_argument(
        "--discovery-backend",
        choices=("etcd", "kubernetes"),
        default="kubernetes",
        help="Worker discovery transport (default kubernetes).",
    )
    parser.add_argument("--etcd-endpoint", default=None)
    parser.add_argument("--etcd-prefix", default="/infera/workers/")
    parser.add_argument(
        "--k8s-label-selector",
        default=os.environ.get("INFERA_K8S_LABEL_SELECTOR"),
        help="Label selector for worker Pods with --discovery-backend=kubernetes.",
    )
    parser.add_argument("--k8s-namespace", default=None)

    # --- policy / kv (same as the server) ---
    parser.add_argument("--router-policy", choices=POLICY_NAMES, default="kv-aware")
    parser.add_argument("--kv-event-transport", choices=("zmq", "nats"), default="nats")
    parser.add_argument(
        "--nats-server",
        default=None,
        help="NATS server URL for --kv-event-transport=nats (default $NATS_SERVER "
        "or nats://127.0.0.1:4222).",
    )
    parser.add_argument(
        "--router-tokenizer-path",
        required=True,
        help="HuggingFace model id or local tokenizer path; must match the workers'.",
    )
    parser.add_argument("--kv-overlap-weight", type=float, default=1.0)
    parser.add_argument("--kv-prefill-overlap-weight", type=float, default=None)
    parser.add_argument("--kv-decode-overlap-weight", type=float, default=None)
    return parser.parse_args(argv)
