###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Config-level preflight checks for disaggregated (PD) workers.

Cross-node prefill/decode has two silent-failure modes that turn into
opaque hangs or 5-20x slowdowns instead of clear errors:

  1. A worker advertises a non-routable host (``0.0.0.0`` / ``127.0.0.1``)
     to etcd, so the peer worker and router can't reach its bootstrap /
     KV endpoint across nodes.
  2. The transport silently falls back to TCP (e.g. Mooncake when RDMA
     setup is skipped), which is 5-20x slower than RDMA.

These checks run at launch, before the engine subprocess starts, and
fail fast with an actionable message. They are pure config validation
(no RDMA hardware probing) so they run anywhere, including CI.

Runtime RDMA-negotiation detection (hooking the transport to confirm it
actually used RDMA, not TCP) is intentionally out of scope here -- it is
coupled to the RDMA NIC bring-up and is handled separately.
"""

from __future__ import annotations

# Hosts that are valid to bind on but useless to advertise to remote peers.
_NON_ROUTABLE_HOSTS: frozenset[str] = frozenset(
    {"", "0.0.0.0", "127.0.0.1", "localhost", "::", "::1"}
)

# SGLang --disaggregation-transfer-backend values that use RDMA. Anything
# outside this set (or an empty value) risks a silent TCP fallback.
_SGLANG_RDMA_BACKENDS: frozenset[str] = frozenset({"mooncake", "mori", "nixl", "ascend"})


class DisaggPreflightError(RuntimeError):
    """Raised when a disaggregated worker's config would fail across nodes."""


def is_routable_host(host: str | None) -> bool:
    """True if ``host`` can be reached by a remote peer.

    Treats blank / wildcard / loopback addresses as non-routable.
    """
    if host is None:
        return False
    return host.strip().lower() not in _NON_ROUTABLE_HOSTS


def validate_advertise_host(advertise_host: str | None, *, is_disagg: bool) -> None:
    """Ensure a disagg worker advertises an address peers can reach.

    No-op for mixed (non-disagg) workers, which only need to be reachable
    by the local router on the same node.
    """
    if not is_disagg:
        return
    if not is_routable_host(advertise_host):
        raise DisaggPreflightError(
            f"disaggregated worker must advertise a routable host for cross-node "
            f"KV transfer, got {advertise_host!r}. Pass --advertise-host=<routable-ip> "
            f"(the address peers and the router use to reach this worker)."
        )


def validate_sglang_transport(backend: str | None, *, is_disagg: bool, allow_tcp: bool) -> None:
    """Require an explicit RDMA transfer backend for disagg SGLang workers.

    Relying on the default (or a non-RDMA backend) risks a silent TCP
    fallback that is 5-20x slower. ``allow_tcp`` is the benchmark-only
    escape hatch.
    """
    if not is_disagg or allow_tcp:
        return
    normalized = (backend or "").strip().lower()
    if not normalized:
        raise DisaggPreflightError(
            "disaggregated SGLang worker requires an explicit "
            "--disaggregation-transfer-backend (one of: "
            f"{', '.join(sorted(_SGLANG_RDMA_BACKENDS))}); relying on the default "
            "risks a silent TCP fallback. Pass --disaggregation-allow-tcp to "
            "override (benchmarks only)."
        )
    if normalized not in _SGLANG_RDMA_BACKENDS:
        raise DisaggPreflightError(
            f"disaggregated SGLang worker has non-RDMA transfer backend "
            f"{backend!r}; expected one of: {', '.join(sorted(_SGLANG_RDMA_BACKENDS))}. "
            f"Pass --disaggregation-allow-tcp to override (benchmarks only)."
        )


def validate_vllm_transport(disagg_meta: dict | None, *, is_disagg: bool, allow_tcp: bool) -> None:
    """Require a recognized RDMA KV connector for disagg vLLM workers.

    A disagg worker whose connector didn't map to a known infera
    protocol (Mooncake / MoRIIO / Nixl) won't transfer KV correctly
    across nodes. ``allow_tcp`` is the benchmark-only escape hatch.
    """
    if not is_disagg or allow_tcp:
        return
    protocol = (disagg_meta or {}).get("protocol")
    if not protocol:
        raise DisaggPreflightError(
            "disaggregated vLLM worker has no recognized RDMA KV connector "
            "(expected a Mooncake / MoRIIO / Nixl connector via "
            "--kv-transfer-config). Without one, cross-node KV transfer is "
            "not configured. Pass --disaggregation-allow-tcp to override "
            "(benchmarks only)."
        )
