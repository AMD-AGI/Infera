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
    ("discovery_backend", "etcd", "kubernetes discovery"),
    ("request_transport", "http", "the NATS request transport"),
    ("router_mode", "auto", "GAIE direct mode"),
]

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
    if unsupported:
        raise SystemExit(
            "--router-backend rust does not support " + ", ".join(unsupported) + ".\n"
            "Use --router-backend python for these."
        )
    if not args.etcd_endpoint:
        raise SystemExit("--router-backend rust requires --etcd-endpoint")

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
    ]
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
