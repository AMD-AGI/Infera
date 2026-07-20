###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SGLang's bootstrap protocol — covers SGLang × every transport.

SGLang's PD coordination is engine-level: the transport choice
(``--disaggregation-transfer-backend {mooncake,mori,...}``) lives inside
the worker and is invisible to the router. From the router's
perspective every SGLang PD pair speaks the same wire shape: top-level
``bootstrap_host`` / ``bootstrap_port`` / ``bootstrap_room`` injected
into the request body, identical bodies POSTed concurrently to both
legs.
"""

from __future__ import annotations

from typing import Literal

from infera.common.worker_pool import WorkerInfo


class SglangBootstrapProtocol:
    name: str = "sglang-bootstrap"
    topology: Literal["concurrent", "serial-pull"] = "concurrent"

    def annotate_prefill(
        self,
        body: dict,
        p: WorkerInfo,
        d: WorkerInfo,
        room_id: int,
    ) -> dict:
        params = p.disagg_meta.get("params") or {}
        boot_addr = params.get("bootstrap_addr")
        if not boot_addr:
            raise ValueError(
                f"prefill worker {p.worker_id} missing disagg_meta['params']['bootstrap_addr']"
            )
        host, port_str = boot_addr.rsplit(":", 1)
        return {
            **body,
            "bootstrap_host": host,
            "bootstrap_port": int(port_str),
            "bootstrap_room": room_id,
        }

    def annotate_decode(
        self,
        body: dict,
        p: WorkerInfo,
        d: WorkerInfo,
        room_id: int,
        prefill_handoff: dict | None,
    ) -> dict:
        # SGLang's decode body is identical to prefill's (both legs see the
        # same bootstrap_*; handoff is concurrent, not pull-based).
        return self.annotate_prefill(body, p, d, room_id)

    def extract_handoff(self, prefill_response_payload: dict) -> dict:
        # Concurrent topology — router never calls this. Defensive return.
        return {}

    def request_id_for(self, p: WorkerInfo, d: WorkerInfo, room_id: int) -> str | None:
        return None
