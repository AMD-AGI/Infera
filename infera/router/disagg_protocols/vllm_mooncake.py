###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM Mooncake PD coordination protocol.

Pinned against vLLM 0.21. Mooncake's transport peers discover each
other via a HTTP bootstrap server colocated with the prefill worker;
the decode body just carries ``remote_bootstrap_addr`` + the prefill's
``remote_engine_id`` and D's connector does the rest.

Wire shape mirrors vLLM's reference
``examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py``.

Prefill DP>1: each rank registers its KV under ``engine_id_dp{rank}`` in
the bootstrap. vLLM's internal LB doesn't report which rank served a
request, so the router steers the prefill leg to a chosen rank K (header
+ ``align_room_to_prefill_rank``) and ``annotate_decode`` addresses that
rank's ``engine_id`` (recovered as ``room_id % dp_size``).
"""

from __future__ import annotations

from typing import Literal

from infera.common.worker_pool import WorkerInfo
from infera.router.disagg_protocols.base import (
    decode_remaining,
    prefill_single_token,
)


def _transfer_id_for_room(room_id: int) -> str:
    """Deterministic transfer_id from room_id so concurrent
    annotate_prefill / annotate_decode agree without explicit threading.
    Format mirrors the reference proxy (``xfer-<id>``)."""
    return f"xfer-{room_id:032x}"


def _required_param(w: WorkerInfo, key: str, side: str) -> str:
    val = (w.disagg_meta.get("params") or {}).get(key)
    if not val:
        raise ValueError(f"{side} worker {w.worker_id}: disagg_meta.params[{key!r}] missing")
    return val


class VllmMooncakeProtocol:
    """Concurrent push: P pushes KV to D over Mooncake transport;
    D discovers P through the bootstrap server."""

    name: str = "vllm-mooncake"
    topology: Literal["concurrent", "serial-pull"] = "concurrent"
    # Router steers the prefill leg to a specific DP rank (see module docstring).
    pin_prefill_dp_rank: bool = True

    def annotate_prefill(self, body: dict, p: WorkerInfo, d: WorkerInfo, room_id: int) -> dict:
        out = prefill_single_token(body)
        out["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "transfer_id": _transfer_id_for_room(room_id),
        }
        return out

    def annotate_decode(
        self,
        body: dict,
        p: WorkerInfo,
        d: WorkerInfo,
        room_id: int,
        prefill_handoff: dict | None,
    ) -> dict:
        out = decode_remaining(body)
        # Address the steered prefill rank K = room_id % dp_size (registered as
        # ``engine_id_dp{K}``); DP=1 keeps the base engine_id.
        engine_id = _required_param(p, "engine_id", "prefill")
        dp_size = p.dp_size or 1
        if dp_size > 1:
            engine_id = f"{engine_id}_dp{room_id % dp_size}"
        out["kv_transfer_params"] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_bootstrap_addr": _required_param(p, "bootstrap_addr", "prefill"),
            "remote_engine_id": engine_id,
            "transfer_id": _transfer_id_for_room(room_id),
        }
        return out

    def extract_handoff(self, prefill_response_payload: dict) -> dict:
        return {}  # concurrent — router never calls this

    def request_id_for(self, p: WorkerInfo, d: WorkerInfo, room_id: int) -> str | None:
        return None  # Mooncake discovers peers via bootstrap_addr, not request_id
