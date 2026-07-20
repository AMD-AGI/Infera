###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM MoRIIO PD coordination protocols.

Pinned against vLLM 0.21. Both READ and WRITE modes parse peer
``host:handshake:notify`` out of the request_id (regex), so the router
forges a request_id sandwich that matches ``_PREFILL_ZMQ_RE`` /
``_DECODE_ZMQ_RE`` in ``vllm/.../moriio/moriio_common.py``.

The two modes share the prefill-leg body shape and differ only in how
D receives KV: READ pulls via D's scheduler after P responds (serial),
WRITE expects P to push and notify via zmq while D is already
in-flight (concurrent).
"""

from __future__ import annotations

import uuid
from typing import Literal
from urllib.parse import urlparse

from infera.common.worker_pool import WorkerInfo
from infera.router.disagg_protocols.base import (
    decode_remaining,
    prefill_single_token,
)


def _transfer_id_for_room(room_id: int) -> str:
    """Deterministic transfer_id from room_id — concurrent dispatch
    annotates P and D bodies independently, so they can't share a uuid;
    deriving from room_id keeps both legs in agreement. ``tx-`` matches
    MoRIIOConstants.TRANSFER_PREFIX."""
    return f"tx-{room_id:032x}"


def _zmq_address(w: WorkerInfo, side: str) -> str:
    host = urlparse(w.url).hostname
    extra = (w.disagg_meta.get("params") or {}).get("kv_connector_extra_config") or {}
    try:
        return f"host:{host},handshake:{int(extra['handshake_port'])},notify:{int(extra['notify_port'])}"
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(
            f"{side} worker {w.worker_id}: kv_connector_extra_config "
            f"missing/invalid handshake_port/notify_port ({extra!r})"
        ) from exc


def _forge_request_id(p: WorkerInfo, d: WorkerInfo) -> str:
    return (
        f"___prefill_addr_{_zmq_address(p, 'prefill')}"
        f"___decode_addr_{_zmq_address(d, 'decode')}"
        f"_{uuid.uuid4().hex}"
    )


def _tp_size_of(w: WorkerInfo) -> int:
    """WHAT: peer TP size for KV-shard offset math, read from advertised
    kv_connector_extra_config.tp_size (fallback 1). WHY: wrong value makes decode
    read the wrong rank's KV → garbage. Connector's ReqMeta keys on ``tp_size``
    (moriio_common.py), not ``remote_tp_size``; both are injected by callers."""
    extra = (w.disagg_meta.get("params") or {}).get("kv_connector_extra_config") or {}
    try:
        return int(extra["tp_size"])
    except (KeyError, ValueError, TypeError):
        return 1


def _annotate_prefill(body: dict, d: WorkerInfo, transfer_id: str) -> dict:
    """P-leg body shape, shared by READ and WRITE — matches
    toy_proxy ``send_request_to_prefill``."""
    out = prefill_single_token(body)
    out["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "transfer_id": transfer_id,
        "remote_dp_size": d.dp_size or 1,
        "tp_size": _tp_size_of(d),
        "remote_tp_size": _tp_size_of(d),
    }
    return out


class VllmMoRIIOReadProtocol:
    """Serial-pull: D pulls KV from P after P prefills.

    Wire shape matches vLLM's reference
    ``examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py``
    (READ branch).
    """

    name: str = "vllm-mori-read"
    topology: Literal["concurrent", "serial-pull"] = "serial-pull"

    def annotate_prefill(self, body: dict, p: WorkerInfo, d: WorkerInfo, room_id: int) -> dict:
        # Serial-pull: P echoes transfer_id in its response, so a fresh
        # uuid is enough — no need to agree with D up front.
        return _annotate_prefill(body, d, f"tx-{uuid.uuid4()}")

    def annotate_decode(
        self,
        body: dict,
        p: WorkerInfo,
        d: WorkerInfo,
        room_id: int,
        prefill_handoff: dict | None,
    ) -> dict:
        if not prefill_handoff:
            raise ValueError(f"{self.name}: serial-pull requires prefill handoff")
        out = decode_remaining(body)
        out["kv_transfer_params"] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "transfer_id": prefill_handoff["transfer_id"],
            "remote_engine_id": prefill_handoff["remote_engine_id"],
            "remote_block_ids": prefill_handoff["remote_block_ids"],
            "remote_dp_size": p.dp_size or 1,
            "tp_size": _tp_size_of(p),
            "remote_tp_size": _tp_size_of(p),
        }
        return out

    def extract_handoff(self, prefill_response_payload: dict) -> dict:
        kv = prefill_response_payload.get("kv_transfer_params")
        if not kv:
            raise KeyError(
                f"{self.name}: prefill response missing kv_transfer_params "
                f"(keys: {sorted(prefill_response_payload)})"
            )
        return dict(kv)

    def request_id_for(self, p: WorkerInfo, d: WorkerInfo, room_id: int) -> str | None:
        return _forge_request_id(p, d)


class VllmMoRIIOWriteProtocol:
    """Concurrent push: P pushes KV to D over RDMA while D is already
    in-flight; P signals completion to D's notify_port via zmq.

    Wire shape matches toy_proxy WRITE branch. P body is identical to
    READ; D body omits the handoff fields (remote_engine_id /
    remote_block_ids stay None — D doesn't pull, P pushes).
    """

    name: str = "vllm-mori-write"
    topology: Literal["concurrent", "serial-pull"] = "concurrent"

    def annotate_prefill(self, body: dict, p: WorkerInfo, d: WorkerInfo, room_id: int) -> dict:
        return _annotate_prefill(body, d, _transfer_id_for_room(room_id))

    def annotate_decode(
        self,
        body: dict,
        p: WorkerInfo,
        d: WorkerInfo,
        room_id: int,
        prefill_handoff: dict | None,
    ) -> dict:
        out = decode_remaining(body)
        out["kv_transfer_params"] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "transfer_id": _transfer_id_for_room(room_id),
            "remote_dp_size": p.dp_size or 1,
            "tp_size": _tp_size_of(p),
            "remote_tp_size": _tp_size_of(p),
        }
        return out

    def extract_handoff(self, prefill_response_payload: dict) -> dict:
        return {}  # concurrent — router never calls this

    def request_id_for(self, p: WorkerInfo, d: WorkerInfo, room_id: int) -> str | None:
        return _forge_request_id(p, d)
