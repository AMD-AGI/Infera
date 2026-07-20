###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""CLI parsing for `python -m infera.engine.vllm`. Mirrors sglang/args.py."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from infera.common.worker_pool import DisaggMode

logger = logging.getLogger(__name__)

_KV_ROLE_TO_DISAGG: dict[str, DisaggMode] = {
    "kv_producer": DisaggMode.PREFILL,
    "kv_consumer": DisaggMode.DECODE,
    "kv_both": DisaggMode.MIXED,
}

# vLLM kv_connector → infera DisaggProtocol identity. Pinned against
# vLLM 0.21. MoRIIO defaults to write; the env var below flips to read.
_CONNECTOR_TO_PROTOCOL: dict[str, str] = {
    "MooncakeConnector": "vllm-mooncake",
    "MooncakeStoreConnector": "vllm-mooncake",
    "MoRIIOConnector": "vllm-mori-write",
    "NixlConnector": "vllm-nixl",
}


@dataclass(kw_only=True)
class VllmWorkerArgs:
    vllm_argv: list[str] = field(default_factory=list)

    model: str
    served_model_name: str | None
    host: str
    port: int
    block_size: int
    # Mirrors vLLM's --trust-remote-code; threaded into the tokenizer load
    # that computes the KV digest/canary (custom tokenizers won't load otherwise).
    trust_remote_code: bool = False

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

    enable_kv_events: bool
    kv_events: str  # "on" | "off" | "auto"
    index_block_size: int

    # KV-event transport for KV-aware routing: "zmq" (router connects to this
    # worker's kv_events_endpoint) or "nats" (relay engine events onto NATS).
    kv_event_transport: str
    nats_server: str | None

    disagg_mode: DisaggMode
    # ``{"protocol": ..., "params": {...}}`` for PD workers, ``{}`` for
    # MIXED or unmapped connectors. Consumed by DisaggProtocol classes.
    disagg_meta: dict[str, Any] = field(default_factory=dict)

    # Data-parallel identity, recorded for observability + DP-aware routing.
    # Populated only for external-LB DP (one launcher process per rank, each
    # its own --port → distinct worker_id); ``None`` for single-rank workers.
    dp_rank: int | None = None
    dp_size: int | None = None


def _disagg_mode_from_kv_transfer_config(cfg) -> DisaggMode:
    if cfg is None:
        return DisaggMode.MIXED
    role = getattr(cfg, "kv_role", None)
    if role is None:
        return DisaggMode.MIXED
    mode = _KV_ROLE_TO_DISAGG.get(role)
    if mode is None:
        logger.warning("unknown kv_role=%r; registering as MIXED", role)
        return DisaggMode.MIXED
    return mode


def _compute_disagg_meta(
    cfg,
    *,
    advertise_host: str | None = None,
) -> dict[str, Any]:
    """``disagg_meta`` payload written to etcd. ``{}`` for MIXED or
    unmapped connectors; ``{"protocol": ..., "params": ...}`` otherwise.
    ``params`` is a passthrough — each ``Vllm*Protocol`` is the only
    consumer."""
    if cfg is None:
        return {}
    role = getattr(cfg, "kv_role", None)
    if role is None or role == "kv_both":
        return {}

    connector = getattr(cfg, "kv_connector", None)
    if connector is None:
        logger.warning("kv_role=%r but kv_connector unset; no disagg_meta", role)
        return {}

    # MultiConnector wraps several connectors (e.g. a kvd L3 cache + an RDMA
    # transport). The disagg transport (Mooncake/MoRIIO/Nixl) is nested under
    # kv_connector_extra_config["connectors"]; surface it so we register its
    # protocol + bootstrap params instead of bailing out on "MultiConnector".
    # (vLLM forwards the top-level engine_id to each child, so we must NOT pin
    # engine_id into the nested dicts — see _pin_engine_id_in_argv.)
    nested_extra: dict | None = None
    if connector == "MultiConnector":
        children = (getattr(cfg, "kv_connector_extra_config", None) or {}).get("connectors") or []
        for child in children:
            cand = (child or {}).get("kv_connector")
            if cand in _CONNECTOR_TO_PROTOCOL:
                connector = cand
                nested_extra = (child or {}).get("kv_connector_extra_config") or {}
                break

    protocol = _CONNECTOR_TO_PROTOCOL.get(connector)
    if protocol is None:
        logger.warning(
            "kv_connector=%r has no infera protocol (known: %s)",
            connector,
            sorted(_CONNECTOR_TO_PROTOCOL),
        )
        return {}

    # MoRIIO advertises read vs write via env var (mirrors vLLM's own check).
    if connector == "MoRIIOConnector" and os.environ.get("VLLM_MORIIO_CONNECTOR_READ_MODE"):
        protocol = "vllm-mori-read"

    params: dict[str, Any] = {
        "engine_id": getattr(cfg, "engine_id", None),
        "kv_ip": getattr(cfg, "kv_ip", None),
        "kv_port": getattr(cfg, "kv_port", None),
        "kv_connector_extra_config": dict(
            nested_extra
            if nested_extra is not None
            else (getattr(cfg, "kv_connector_extra_config", None) or {})
        ),
    }
    if protocol == "vllm-mooncake":
        # Mooncake's BootstrapServer binds to data_parallel_master_ip
        # (defaults to 127.0.0.1, wrong for cross-node). Use the same
        # advertise_host the router will POST to; single-DP simplification.
        port = int(os.environ.get("VLLM_MOONCAKE_BOOTSTRAP_PORT", "8998"))
        params["bootstrap_addr"] = f"http://{advertise_host or '127.0.0.1'}:{port}"

    return {"protocol": protocol, "params": params}


def _pin_engine_id_in_argv(argv: list[str], engine_id: str | None) -> list[str]:
    """Rewrite ``--kv-transfer-config`` JSON in argv so the spawned vLLM
    subprocess uses ``engine_id`` instead of regenerating its own.

    Without this, parsing happens twice (here + in the subprocess) and
    ``KVTransferConfig.engine_id`` gets re-randomised, so the value we
    register in etcd disagrees with what Mooncake's bootstrap server
    publishes — every Mooncake D-leg POST would reference a non-existent
    engine_id and KV transfer would 4xx.

    Idempotent if engine_id is already set in the JSON; no-op for argvs
    without ``--kv-transfer-config`` (MIXED mode).
    """
    if not engine_id:
        return argv

    def _rewrite(blob: str) -> str:
        try:
            cfg = json.loads(blob)
        except (TypeError, ValueError):
            return blob
        if not isinstance(cfg, dict):
            return blob
        cfg["engine_id"] = engine_id
        return json.dumps(cfg)

    out = list(argv)
    i = 0
    while i < len(out):
        tok = out[i]
        if tok == "--kv-transfer-config" and i + 1 < len(out):
            out[i + 1] = _rewrite(out[i + 1])
            i += 2
            continue
        if tok.startswith("--kv-transfer-config="):
            out[i] = "--kv-transfer-config=" + _rewrite(tok.split("=", 1)[1])
        i += 1
    return out


def parse_vllm_args(argv: list[str] | None = None) -> VllmWorkerArgs:
    # allow_abbrev=False: vLLM flags are forwarded verbatim; prefix matching
    # would let our flags accidentally swallow theirs.
    parser = argparse.ArgumentParser(
        prog="python -m infera.engine.vllm",
        description="Infera vLLM worker launcher (spawns `vllm serve`).",
        allow_abbrev=False,
    )
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
        "lease-based self-registration. Required for --discovery-backend=etcd.",
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
        help="How the router reaches this worker. 'http' serves engine "
        "HTTP directly; 'nats' also runs a consumer proxying its per-instance "
        "NATS subject to the local engine HTTP. Requires --nats-server/$NATS_SERVER.",
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
        help="NATS request total (overall) timeout (s); worker hard-aborts at this "
        "cap. None => $INFERA_NATS_REQ_MAX_DURATION or built-in default 0 (off).",
    )
    parser.add_argument(
        "--nats-req-max-pending",
        type=int,
        default=None,
        help="NATS request admission limit; >0 makes this worker consume via a "
        "JetStream consumer. None => $INFERA_NATS_REQ_MAX_PENDING or default 0.",
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
        help="Host/IP to publish (worker_id, url, bootstrap_addr). Use when vLLM "
        "binds on 0.0.0.0 but peers need a routable address. Defaults to --host.",
    )
    parser.add_argument(
        "--enable-kv-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish KV cache events on a ZMQ socket for KV-aware routing "
        "(default on; pass --no-enable-kv-events to disable).",
    )
    parser.add_argument(
        "--disaggregation-allow-tcp",
        action="store_true",
        default=False,
        help="Benchmark-only: skip the disagg transport preflight that "
        "rejects configs without a recognized RDMA KV connector "
        "(Mooncake / MoRIIO / Nixl). Do NOT use in production.",
    )
    parser.add_argument(
        "--kv-events",
        choices=("on", "off", "auto"),
        default="auto",
        help="Compute KvRegistrationMetadata (tokenizer digest + canary). "
        "'auto' (default) turns it on iff tokenizer loads cleanly.",
    )
    parser.add_argument(
        "--index-block-size",
        type=int,
        default=64,
        help="Router's coalescing unit (NOT vLLM's --block-size). Default 64.",
    )
    parser.add_argument(
        "--kv-event-transport",
        choices=("zmq", "nats"),
        default="nats",
        help="How this worker exposes KV events to the router. 'nats' (default) "
        "relays engine events onto a NATS broker (infera.kv.events.>); 'zmq' "
        "publishes directly on kv_events_endpoint. Requires --enable-kv-events.",
    )
    parser.add_argument(
        "--nats-server",
        default=None,
        help="NATS server URL for --kv-event-transport=nats (default: "
        "$NATS_SERVER or nats://127.0.0.1:4222).",
    )
    known, remaining = parser.parse_known_args(argv)

    # infera product default: fp8 KV cache (fp8_e4m3) unless the operator set a
    # --kv-cache-dtype explicitly. fp8 halves the KV footprint -> ~2x the KV that
    # fits in VRAM (bigger batches / longer contexts) and halves PD KV-transfer +
    # RDMA memory-registration volume (bf16 hit the ionic HCA's ibv_reg_mr ENOMEM
    # at high concurrency / long inputs, stalling decode). Small accuracy cost; opt
    # out with an explicit --kv-cache-dtype auto|bf16 or INFERA_DEFAULT_KV_FP8=0.
    if os.environ.get("INFERA_DEFAULT_KV_FP8", "1") != "0" and not any(
        t == "--kv-cache-dtype" or t.startswith("--kv-cache-dtype=") for t in remaining
    ):
        remaining += ["--kv-cache-dtype", "fp8_e4m3"]
        logger.info(
            "infera default: --kv-cache-dtype fp8_e4m3 "
            "(override with --kv-cache-dtype or INFERA_DEFAULT_KV_FP8=0)"
        )

    from vllm.entrypoints.openai.cli_args import make_arg_parser

    vllm_args = make_arg_parser(
        argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ).parse_args(remaining)

    # vLLM 0.21's --model has a non-None default, so vllm_args.model can't
    # tell us whether the operator actually passed it. Scan argv ourselves.
    if not any(t == "--model" or t.startswith("--model=") for t in remaining):
        parser.error("vLLM --model is required.")

    # vLLM's --served-model-name is nargs='+' (multi-alias). Infera registers
    # a single primary name in etcd; vLLM still serves all aliases over HTTP.
    served = vllm_args.served_model_name
    if isinstance(served, list):
        if len(served) > 1:
            logger.warning(
                "--served-model-name has %d aliases; registering %r in etcd "
                "(other aliases still served by vLLM HTTP)",
                len(served),
                served[0],
            )
        served = served[0] if served else None

    disagg_meta = _compute_disagg_meta(
        vllm_args.kv_transfer_config,
        advertise_host=known.advertise_host or vllm_args.host,
    )

    # Only record DP identity for an actual DP deployment (dp_size>1). A
    # single-rank worker stays ``None`` so it registers like any other.
    dp_size_raw = int(getattr(vllm_args, "data_parallel_size", 1) or 1)
    dp_size = dp_size_raw if dp_size_raw > 1 else None
    dp_rank = getattr(vllm_args, "data_parallel_rank", None) if dp_size is not None else None

    return VllmWorkerArgs(
        vllm_argv=_pin_engine_id_in_argv(
            list(remaining),
            (disagg_meta.get("params") or {}).get("engine_id"),
        ),
        model=vllm_args.model,
        served_model_name=served,
        host=vllm_args.host,
        port=vllm_args.port,
        block_size=vllm_args.block_size,
        trust_remote_code=bool(getattr(vllm_args, "trust_remote_code", False)),
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
        kv_events=known.kv_events,
        index_block_size=known.index_block_size,
        kv_event_transport=known.kv_event_transport,
        nats_server=known.nats_server,
        disagg_mode=_disagg_mode_from_kv_transfer_config(vllm_args.kv_transfer_config),
        disagg_meta=disagg_meta,
        dp_rank=dp_rank,
        dp_size=dp_size,
    )
