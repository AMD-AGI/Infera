###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Auto-allocated ports must avoid the Kubernetes NodePort range.

A port inside that range can be claimed by any Service in the cluster, after
which kube-proxy (IPVS mode) intercepts traffic to it on every node address and
the kernel answers RST. The engine's bind still succeeds and loopback still
works, so only peers dialing the advertised node IP break -- which is exactly
the router subscribing to kv-events.
"""

from __future__ import annotations

import socket

import pytest

from infera.common.net import (
    _reserved_nodeport_range,
    free_tcp_port,
    free_tcp_port_block,
)

_DEFAULT_LO, _DEFAULT_HI = 30000, 32767


def _overlaps(base: int, count: int, lo: int, hi: int) -> bool:
    return base + count - 1 >= lo and base <= hi


@pytest.mark.parametrize("count", [2, 8, 16])
def test_block_avoids_default_nodeport_range(count):
    base = free_tcp_port_block(count)
    assert not _overlaps(base, count, _DEFAULT_LO, _DEFAULT_HI), (
        f"block {base}..{base + count - 1} overlaps the NodePort range"
    )


def test_block_avoids_custom_nodeport_range(monkeypatch):
    monkeypatch.setenv("INFERA_NODEPORT_RANGE", "20000-25000")
    base = free_tcp_port_block(8)
    assert not _overlaps(base, 8, 20000, 25000)


def test_range_can_be_disabled(monkeypatch):
    # A cluster that moved the range off the default should not pay for a
    # needlessly low block.
    monkeypatch.setenv("INFERA_NODEPORT_RANGE", "none")
    assert _reserved_nodeport_range() is None


def test_malformed_range_keeps_the_default(monkeypatch):
    # A typo must not silently drop the guard.
    monkeypatch.setenv("INFERA_NODEPORT_RANGE", "not-a-range")
    assert _reserved_nodeport_range() == (_DEFAULT_LO, _DEFAULT_HI)


@pytest.mark.parametrize("spec", ["", "   "])
def test_empty_range_keeps_the_default(monkeypatch, spec):
    # A manifest rendering the var from something unset must not read as an
    # opt-out; that accident is the one the guard exists to survive.
    monkeypatch.setenv("INFERA_NODEPORT_RANGE", spec)
    assert _reserved_nodeport_range() == (_DEFAULT_LO, _DEFAULT_HI)


@pytest.mark.parametrize("count", [2, 8])
def test_block_is_contiguous_and_bindable(count):
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


def test_single_port_avoids_custom_nodeport_range(monkeypatch):
    # The single-port path feeds advertised endpoints too (single-DP kv events,
    # the ATOM rendezvous port), so it owes the same guarantee as the block.
    monkeypatch.setenv("INFERA_NODEPORT_RANGE", "20000-25000")
    assert not 20000 <= free_tcp_port() <= 25000


def test_single_port_falls_back_when_the_ranges_overlap(monkeypatch):
    # A range covering all of ip_local_port_range leaves the kernel no legal
    # port to hand out, so retrying cannot converge and the scan must take over.
    lo = int(open("/proc/sys/net/ipv4/ip_local_port_range").read().split()[0])
    monkeypatch.setenv("INFERA_NODEPORT_RANGE", f"{lo}-65535")
    port = free_tcp_port()
    assert 1024 < port < lo
