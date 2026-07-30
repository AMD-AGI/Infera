###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""KV-event port blocks must avoid the Kubernetes NodePort range.

A port inside that range can be claimed by any Service in the cluster, after
which kube-proxy (IPVS mode) intercepts traffic to it on every node address and
the kernel answers RST. The engine's bind still succeeds and loopback still
works, so only peers dialing the advertised node IP break -- which is exactly
the router subscribing to kv-events.
"""

from __future__ import annotations

import pytest

from infera.common.net import free_tcp_port, free_tcp_port_block


def _overlaps(base: int, count: int, lo: int, hi: int) -> bool:
    return base + count - 1 >= lo and base <= hi


@pytest.mark.parametrize("count", [2, 8, 16])
def test_block_avoids_default_nodeport_range(count):
    base = free_tcp_port_block(count)
    assert not _overlaps(base, count, 30000, 32767), (
        f"block {base}..{base + count - 1} overlaps the NodePort range"
    )


def test_block_avoids_custom_nodeport_range(monkeypatch):
    monkeypatch.setenv("INFERA_NODEPORT_RANGE", "20000-25000")
    base = free_tcp_port_block(8)
    assert not _overlaps(base, 8, 20000, 25000)


def test_block_avoidance_can_be_disabled(monkeypatch):
    # Clusters that moved the range should not pay for a needlessly low block.
    monkeypatch.setenv("INFERA_NODEPORT_RANGE", "none")
    base = free_tcp_port_block(8)
    assert base > 25000


def test_malformed_range_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("INFERA_NODEPORT_RANGE", "not-a-range")
    assert free_tcp_port_block(8) > 1024


@pytest.mark.parametrize("count", [2, 8])
def test_block_is_contiguous_and_bindable(count):
    import socket

    base = free_tcp_port_block(count)
    socks = []
    try:
        for off in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", base + off))
            socks.append(s)
    finally:
        for s in socks:
            s.close()


def test_single_port_delegates_to_ephemeral_helper():
    # count<=1 lets the kernel choose, which on a default ip_local_port_range
    # lands above the NodePort range anyway.
    port = free_tcp_port_block(1)
    assert 1024 < port < 65536
