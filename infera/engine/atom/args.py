###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""CLI parsing for ``python -m infera.engine.atom``.

Mirrors ``infera.engine.vllm.args`` / ``infera.engine.sglang.args``:
parse the handful of infera-specific flags, then hand the rest to ATOM's
own ``EngineArgs`` parser so we can read out the fields infera needs
(model, host, http port, tensor-parallel size, kv-transfer-config) while
forwarding the original argv verbatim to the spawned subprocess.

ATOM's OpenAI server (``atom.entrypoints.openai_server``) splits ports two
ways:

* ``--server-port`` — the OpenAI HTTP port infera registers + routes to.
* ``--port`` — an *internal* engine port used as ``MASTER_PORT`` for the
  torch-distributed group. Two ATOM instances on one host (e.g. a prefill
  and a decode worker) must not share it, so the launcher auto-allocates a
  free one when the operator didn't pin it.

PD disaggregation is driven by ATOM's ``--kv-transfer-config`` JSON. The
``kv_role`` field selects prefill (``kv_producer``) / decode
(``kv_consumer``) / mixed (absent or ``kv_both``); infera resolves the
pair through the ``atom-mooncake`` DisaggProtocol, which replays the body
shaping ATOM's reference proxy does.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from infera.common.net import free_tcp_port
from infera.common.worker_pool import DisaggMode

logger = logging.getLogger(__name__)

# ATOM mooncake connector default for the bootstrap side-channel port
# (see atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py).
_DEFAULT_HANDSHAKE_PORT = 6301

_KV_ROLE_TO_DISAGG: dict[str, DisaggMode] = {
    "kv_producer": DisaggMode.PREFILL,
    "kv_consumer": DisaggMode.DECODE,
    "kv_both": DisaggMode.MIXED,
}

# ATOM kv_connector → infera DisaggProtocol identity. Only the mooncake
# connector is wired through infera's DisaggRouter (it's ATOM's
# RDMA-validated PD transport).
_CONNECTOR_TO_PROTOCOL: dict[str, str] = {
    "mooncake": "atom-mooncake",
}


@dataclass(kw_only=True)
class AtomWorkerArgs:
    # Full argv forwarded verbatim to `python -m atom.entrypoints.openai_server`.
    atom_argv: list[str] = field(default_factory=list)

    model: str
    host: str
    server_port: int
    tensor_parallel_size: int

    etcd_endpoint: str
    etcd_prefix: str
    advertise_host: str | None

    # KV-aware routing. When enabled the launcher injects a ZMQ-bind endpoint
    # into the ATOM subprocess (via env) so its BlockManager publishes KV
    # cache events; ``kv_block_size`` is ATOM's paged-KV block size, which the
    # router needs to re-derive block hashes.
    enable_kv_events: bool
    kv_block_size: int

    disagg_mode: DisaggMode
    # ``{"protocol": "atom-mooncake", "params": {...}}`` for PD workers,
    # ``{}`` for mixed. Consumed by the DisaggProtocol on the router side.
    disagg_meta: dict[str, Any] = field(default_factory=dict)


def _disagg_mode_from_kv_transfer(cfg: dict[str, Any]) -> DisaggMode:
    role = cfg.get("kv_role")
    if not role:
        return DisaggMode.MIXED
    mode = _KV_ROLE_TO_DISAGG.get(role)
    if mode is None:
        logger.warning("unknown kv_role=%r; registering as MIXED", role)
        return DisaggMode.MIXED
    return mode


def _compute_disagg_meta(
    cfg: dict[str, Any],
    *,
    advertise_host: str,
    server_port: int,
    tensor_parallel_size: int,
) -> dict[str, Any]:
    """Build the ``disagg_meta`` payload written to etcd for a PD worker.

    Mixed workers (no ``kv_role`` / ``kv_both``) get ``{}``. Producer and
    decode workers both advertise the addressing the ``atom-mooncake``
    protocol needs to replay ATOM's proxy hand-off: the routable host, the
    OpenAI HTTP port, the mooncake bootstrap (handshake) port, and TP/DP
    sizes.
    """
    role = cfg.get("kv_role")
    if not role or role == "kv_both":
        return {}

    # infera drives ATOM PD over the mooncake connector; require it
    # explicitly rather than inheriting ATOM's internal "moriio" default.
    connector = cfg.get("kv_connector", "mooncake")
    protocol = _CONNECTOR_TO_PROTOCOL.get(connector)
    if protocol is None:
        logger.warning(
            "kv_connector=%r has no infera ATOM protocol (known: %s); "
            "registering without disagg_meta",
            connector,
            sorted(_CONNECTOR_TO_PROTOCOL),
        )
        return {}

    handshake_port = int(cfg.get("handshake_port", _DEFAULT_HANDSHAKE_PORT))
    dp_size = int(cfg.get("dp_size", 1) or 1)

    params: dict[str, Any] = {
        "host": advertise_host,
        "http_port": server_port,
        "handshake_port": handshake_port,
        "tp_size": tensor_parallel_size,
        "dp_size": dp_size,
    }
    return {"protocol": protocol, "params": params}


def parse_atom_args(argv: list[str] | None = None) -> AtomWorkerArgs:
    # allow_abbrev=False: ATOM flags are forwarded verbatim; prefix matching
    # would let our flags accidentally swallow theirs.
    parser = argparse.ArgumentParser(
        prog="python -m infera.engine.atom",
        description="Infera ATOM worker launcher (spawns `atom.entrypoints.openai_server`).",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--etcd-endpoint",
        required=True,
        help="Etcd endpoint (host:port, host, or http(s)://...) for lease-based self-registration.",
    )
    parser.add_argument(
        "--etcd-prefix",
        default="/infera/workers/",
        help="Etcd key prefix (default: /infera/workers/)",
    )
    parser.add_argument(
        "--advertise-host",
        default=None,
        help="Host/IP to publish to etcd. Use when ATOM binds on 0.0.0.0 "
        "but peers need a routable address. Defaults to --host.",
    )
    parser.add_argument(
        "--enable-kv-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Publish ATOM BlockManager KV cache events on a ZMQ socket for "
        "KV-aware routing. Default OFF (ATOM has no native event stream, so "
        "infera must monkey-patch the engine to emit them — opt in here). "
        "Pass --enable-kv-events to turn on.",
    )
    known, remaining = parser.parse_known_args(argv)

    # Parse ATOM's own CLI to extract the fields infera needs. The full
    # `remaining` argv is forwarded to the subprocess untouched (apart from
    # an auto-injected internal --port when absent).
    from atom.model_engine.arg_utils import EngineArgs

    atom_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    EngineArgs.add_cli_args(atom_parser)
    # These live on the server entrypoint, not EngineArgs.
    atom_parser.add_argument("--host", type=str, default="0.0.0.0")
    atom_parser.add_argument("--server-port", type=int, default=8000)
    atom_parser.add_argument("--default-chat-template-kwargs", type=str, default=None)
    atom_parser.add_argument("--request-log", type=str, default=None)
    atom_parsed, _ = atom_parser.parse_known_args(remaining)

    # ATOM's paged-KV block size (EngineArgs.block_size, default 16). The
    # router needs it to chunk a request's tokens into the same blocks the
    # engine hashes. Fall back to 16 (ATOM's default) if absent.
    kv_block_size = int(getattr(atom_parsed, "block_size", 16) or 16)

    try:
        kv_cfg = json.loads(atom_parsed.kv_transfer_config or "{}")
        if not isinstance(kv_cfg, dict):
            kv_cfg = {}
    except (TypeError, ValueError):
        logger.warning("could not parse --kv-transfer-config as JSON; treating as mixed")
        kv_cfg = {}

    advertise_host = known.advertise_host or atom_parsed.host
    disagg_mode = _disagg_mode_from_kv_transfer(kv_cfg)
    disagg_meta = _compute_disagg_meta(
        kv_cfg,
        advertise_host=advertise_host,
        server_port=atom_parsed.server_port,
        tensor_parallel_size=atom_parsed.tensor_parallel_size,
    )

    # Auto-allocate the internal engine MASTER_PORT when the operator didn't
    # pin one, so prefill + decode on the same host don't collide.
    atom_argv = list(remaining)
    if not any(t == "--port" or t.startswith("--port=") for t in atom_argv):
        atom_argv += ["--port", str(free_tcp_port())]

    return AtomWorkerArgs(
        atom_argv=atom_argv,
        model=atom_parsed.model,
        host=atom_parsed.host,
        server_port=atom_parsed.server_port,
        tensor_parallel_size=atom_parsed.tensor_parallel_size,
        etcd_endpoint=known.etcd_endpoint,
        etcd_prefix=known.etcd_prefix,
        advertise_host=known.advertise_host,
        enable_kv_events=known.enable_kv_events,
        kv_block_size=kv_block_size,
        disagg_mode=disagg_mode,
        disagg_meta=disagg_meta,
    )
