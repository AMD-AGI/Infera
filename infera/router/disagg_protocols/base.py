###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""``DisaggProtocol`` interface (Protocol class).

Lives in its own module so concrete protocol implementations can
``from infera.router.disagg_protocols.base import DisaggProtocol``
without circular imports through the package ``__init__``.
"""

from __future__ import annotations

from typing import Literal, Protocol

from infera.common.worker_pool import WorkerInfo


class DisaggProtocol(Protocol):
    """Stateless coordinator for one (P, D) pair.

    Protocols are pure data + a handful of small functions that shape
    per-leg bodies, derive a request id (some connectors smuggle peer
    addressing through it), and extract handoff metadata. The router
    owns connection lifecycle, retries, and metrics — none of that
    varies per protocol. A single instance is shared across all
    requests; protocol methods must not hold per-request state.
    """

    name: str
    topology: Literal["concurrent", "serial-pull"]

    def annotate_prefill(
        self,
        body: dict,
        p: WorkerInfo,
        d: WorkerInfo,
        room_id: int,
    ) -> dict: ...

    def annotate_decode(
        self,
        body: dict,
        p: WorkerInfo,
        d: WorkerInfo,
        room_id: int,
        prefill_handoff: dict | None,
    ) -> dict: ...

    def extract_handoff(self, prefill_response_payload: dict) -> dict:
        """Pull connector-specific handoff fields from the prefill response.

        Only called when ``topology == "serial-pull"``. Concurrent
        protocols return ``{}``.
        """
        ...

    def request_id_for(
        self,
        p: WorkerInfo,
        d: WorkerInfo,
        room_id: int,
    ) -> str | None:
        """Forged ``X-Request-Id`` for both P and D POSTs, or ``None`` to
        let the engine assign its own. MoRIIO needs this because its
        connector parses peer addressing out of the request_id."""
        ...


# Shared body shaping for vLLM PD connectors: the prefill leg runs for a
# single token (just enough to populate the KV cache) and the decode leg
# produces the rest. Both return a shallow copy, leaving the input intact.


def prefill_single_token(body: dict) -> dict:
    out = dict(body)
    out["max_tokens"] = 1
    out["stream"] = False
    out.pop("stream_options", None)
    if "max_completion_tokens" in out:
        out["max_completion_tokens"] = 1
    return out


def decode_remaining(body: dict) -> dict:
    out = dict(body)
    for k in ("max_tokens", "max_completion_tokens"):
        if isinstance(out.get(k), int):
            out[k] = max(1, out[k] - 1)
    return out
