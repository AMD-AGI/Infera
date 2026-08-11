###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Hand off to the Rust router binary (``--router-backend rust``).

Validates the requested config against the Rust MVP's supported subset,
translates the server flags to the binary's CLI, and ``os.execvp``s it — the
Python process is replaced, so there's no supervisor or extra hop. Anything the
Rust backend doesn't cover yet errors with a pointer to ``--router-backend
python``.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# (arg attr, required value, human name for the error message)
_REQUIRED = [
    ("router_mode", "auto", "GAIE direct mode"),
]

# Discovery backends the Rust binary implements.
_SUPPORTED_DISCOVERY = ("etcd", "kubernetes")

# Request transports the Rust binary implements.
_SUPPORTED_TRANSPORT = ("http", "nats")

# kv-event sources the Rust binary implements.
_SUPPORTED_KV_TRANSPORT = ("zmq", "nats")

# Routing policies the Rust backend implements.
_SUPPORTED_POLICIES = ("round-robin", "kv-aware")


def _find_binary() -> str:
    if env := os.environ.get("INFERA_ROUTER_BIN"):
        return env
    if found := shutil.which("infera-router"):
        return found
    repo_root = Path(__file__).resolve().parents[2]  # infera/server/ -> repo
    for profile in ("release", "debug"):
        cand = repo_root / "rust" / "target" / profile / "infera-router"
        if cand.exists():
            return str(cand)
    raise SystemExit(
        "infera-router binary not found. Build it (`cd rust && cargo build "
        "--release`) or set INFERA_ROUTER_BIN=/path/to/infera-router."
    )


def exec_rust(args: argparse.Namespace) -> None:
    """Validate + exec the Rust router. Never returns (replaces the process)."""
    unsupported = [hint for attr, want, hint in _REQUIRED if getattr(args, attr, None) != want]
    if args.enable_profiling:
        unsupported.append("the profiling control plane")
    if args.router_policy not in _SUPPORTED_POLICIES:
        unsupported.append(f"--router-policy {args.router_policy}")
    if args.discovery_backend not in _SUPPORTED_DISCOVERY:
        unsupported.append(f"--discovery-backend {args.discovery_backend}")
    if args.request_transport not in _SUPPORTED_TRANSPORT:
        unsupported.append(f"--request-transport {args.request_transport}")
    if args.kv_event_transport not in _SUPPORTED_KV_TRANSPORT:
        unsupported.append(f"--kv-event-transport {args.kv_event_transport}")
    if unsupported:
        raise SystemExit(
            "--router-backend rust does not support " + ", ".join(unsupported) + ".\n"
            "Use --router-backend python for these."
        )
    if args.discovery_backend == "etcd" and not args.etcd_endpoint:
        raise SystemExit("--router-backend rust requires --etcd-endpoint")
    if args.discovery_backend == "kubernetes" and not args.k8s_label_selector:
        raise SystemExit(
            "--router-backend rust with kubernetes discovery requires "
            "--k8s-label-selector (or INFERA_K8S_LABEL_SELECTOR)"
        )

    binary = _find_binary()
    argv = [
        binary,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--etcd-endpoint",
        args.etcd_endpoint,
        "--etcd-prefix",
        args.etcd_prefix,
        "--router-policy",
        args.router_policy,
        "--discovery-backend",
        args.discovery_backend,
        "--request-transport",
        args.request_transport,
        "--request-max-retries",
        str(args.request_max_retries),
        "--breaker-failure-threshold",
        str(args.breaker_failure_threshold),
        "--breaker-cooldown-s",
        str(args.breaker_cooldown_s),
        "--breaker-max-cooldown-s",
        str(args.breaker_max_cooldown_s),
    ]
    if args.discovery_backend == "kubernetes":
        argv += ["--k8s-label-selector", args.k8s_label_selector]
        # Left unset, the Rust side reads the Pod's own namespace, which is
        # what the Python backend does too.
        if args.k8s_namespace:
            argv += ["--k8s-namespace", args.k8s_namespace]
    argv += ["--kv-event-transport", args.kv_event_transport]
    if args.nats_server:
        argv += ["--nats-server", args.nats_server]
    # kv-aware needs the tokenizer + overlap weights, or it degrades to
    # load-only routing (no cache locality). Resolve HF ids to a local path
    # (same as the Python router) since the Rust binary loads from disk.
    if args.router_policy == "kv-aware":
        if args.router_tokenizer_path:
            from infera.common.tokenizer import resolve_tokenizer_path

            argv += ["--kv-tokenizer-path", resolve_tokenizer_path(args.router_tokenizer_path)]
        argv += ["--kv-overlap-weight", str(args.kv_overlap_weight)]
        if args.kv_prefill_overlap_weight is not None:
            argv += ["--kv-prefill-overlap-weight", str(args.kv_prefill_overlap_weight)]
        if args.kv_decode_overlap_weight is not None:
            argv += ["--kv-decode-overlap-weight", str(args.kv_decode_overlap_weight)]
    print(f"[infera] --router-backend rust: exec {binary}", flush=True)
    os.execvp(binary, argv)
