###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""PD dispatch protocols.

A ``DisaggProtocol`` describes how to coordinate one (prefill, decode)
pair for a specific (engine, kv_connector) combination: how to shape the
per-leg request bodies and what dispatch topology (concurrent push vs
serial pull) the connector wants.

Worker launchers write the protocol identity into
``WorkerInfo.disagg_meta["protocol"]`` (e.g. ``"sglang-bootstrap"``,
``"vllm-mori-read"``). The router resolves the protocol once per request
and delegates body shaping to it; connection management, retries, and
metrics stay in the router.
"""

from __future__ import annotations

from infera.common.worker_pool import WorkerInfo
from infera.router.disagg_protocols.atom_mooncake import AtomMooncakeProtocol
from infera.router.disagg_protocols.base import DisaggProtocol
from infera.router.disagg_protocols.sglang_bootstrap import SglangBootstrapProtocol
from infera.router.disagg_protocols.vllm_mooncake import VllmMooncakeProtocol
from infera.router.disagg_protocols.vllm_moriio import (
    VllmMoRIIOReadProtocol,
    VllmMoRIIOWriteProtocol,
)

_PROTOCOLS: dict[str, DisaggProtocol] = {
    SglangBootstrapProtocol.name: SglangBootstrapProtocol(),
    VllmMoRIIOReadProtocol.name: VllmMoRIIOReadProtocol(),
    VllmMoRIIOWriteProtocol.name: VllmMoRIIOWriteProtocol(),
    VllmMooncakeProtocol.name: VllmMooncakeProtocol(),
    AtomMooncakeProtocol.name: AtomMooncakeProtocol(),
}


class ProtocolMismatch(ValueError):
    """Prefill and decode workers advertise different protocols."""


class UnknownProtocol(KeyError):
    """Worker advertises a protocol not present in the registry."""


def resolve_protocol(p: WorkerInfo, d: WorkerInfo) -> DisaggProtocol:
    p_id = p.disagg_meta.get("protocol")
    d_id = d.disagg_meta.get("protocol")
    if p_id is None or d_id is None:
        raise ProtocolMismatch(
            f"workers missing disagg_meta['protocol']: "
            f"prefill={p.worker_id}({p_id!r}), decode={d.worker_id}({d_id!r})"
        )
    if p_id != d_id:
        raise ProtocolMismatch(
            f"protocol mismatch: prefill={p.worker_id}({p_id}) vs decode={d.worker_id}({d_id})"
        )
    try:
        return _PROTOCOLS[p_id]
    except KeyError:
        raise UnknownProtocol(
            f"no DisaggProtocol registered for {p_id!r} (known: {sorted(_PROTOCOLS)})"
        ) from None


__all__ = [
    "DisaggProtocol",
    "ProtocolMismatch",
    "UnknownProtocol",
    "resolve_protocol",
]
