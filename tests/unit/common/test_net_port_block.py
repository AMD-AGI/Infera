###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Regression tests for ``infera.common.net.free_tcp_port_block``.

Two independent bugs, both of which make the *block* unusable while every
individual port passes a bind probe:

1. **Collision.** The scan used to start at a fixed
   ``ip_local_port_range.low - count`` and the probe released the block before
   returning, so two engines launched on the same host (the prefill and decode
   legs of a PD pair, each with ``dp_size > 1``) deterministically picked the
   *same* base. The second leg's sglang subprocess then died with
   ``ZMQError: Address already in use`` binding ``base + attn_dp_rank``.
2. **NodePort range.** A base inside Kubernetes' NodePort window binds here and
   answers on loopback, but once any Service claims that port kube-proxy (IPVS)
   intercepts it on every node address and the kernel answers RST — so only a
   *peer* dialing the advertised node IP breaks, which is exactly the router
   subscribing to a worker's kv-events.

Both guards live in one scan, so they are tested together: a base must be both
spread out and outside the reserved window.
"""

from __future__ import annotations

import socket

import pytest

from infera.common.net import _reserved_nodeport_range, free_tcp_port_block

_DEFAULT_LO, _DEFAULT_HI = 30000, 32767


def _overlaps(base: int, count: int, lo: int, hi: int) -> bool:
    return base + count - 1 >= lo and base <= hi


def _ephemeral_low() -> int:
    try:
        return int(open("/proc/sys/net/ipv4/ip_local_port_range").read().split()[0])
    except (OSError, ValueError, IndexError):
        return 32768


def test_block_ports_are_free_and_contiguous() -> None:
    count = 4
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


def test_block_sits_below_the_ephemeral_range() -> None:
    """A base inside the ephemeral range could be handed out by bind(("", 0))."""
    count = 4
    base = free_tcp_port_block(count)
    assert 1024 <= base
    assert base + count - 1 < _ephemeral_low()


def test_repeated_calls_do_not_all_collide() -> None:
    """Consecutive callers must not deterministically get the same base.

    With the old fixed scan start every call returned an identical base while
    nothing held the ports, which is exactly the PD two-leg crash. Randomising
    the scan start makes repeats spread out.
    """
    bases = [free_tcp_port_block(4) for _ in range(10)]
    assert len(set(bases)) > 1, f"all calls returned the same base: {bases[0]}"


def test_count_of_one_delegates_to_single_port() -> None:
    port = free_tcp_port_block(1)
    assert 1024 <= port <= 65535


@pytest.mark.parametrize("count", [2, 8, 16])
def test_block_avoids_default_nodeport_range(count: int) -> None:
    base = free_tcp_port_block(count)
    assert not _overlaps(base, count, _DEFAULT_LO, _DEFAULT_HI), (
        f"block {base}..{base + count - 1} overlaps the NodePort range"
    )


def test_block_avoids_custom_nodeport_range(monkeypatch) -> None:
    monkeypatch.setenv("INFERA_NODEPORT_RANGE", "20000-25000")
    base = free_tcp_port_block(8)
    assert not _overlaps(base, 8, 20000, 25000)


def test_range_can_be_disabled(monkeypatch) -> None:
    # A cluster that moved the range off the default should not pay for a
    # needlessly low block.
    monkeypatch.setenv("INFERA_NODEPORT_RANGE", "none")
    assert _reserved_nodeport_range() is None


def test_malformed_range_keeps_the_default(monkeypatch) -> None:
    # A typo must not silently drop the guard.
    monkeypatch.setenv("INFERA_NODEPORT_RANGE", "not-a-range")
    assert _reserved_nodeport_range() == (_DEFAULT_LO, _DEFAULT_HI)
