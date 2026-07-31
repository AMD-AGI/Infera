###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Regression tests for ``infera.common.net.free_tcp_port_block``.

The bug these guard against: the scan used to start at a fixed
``ip_local_port_range.low - count`` and the probe released the block before
returning, so two engines launched on the same host (the prefill and decode
legs of a PD pair, each with ``dp_size > 1``) deterministically picked the
*same* base. The second leg's sglang subprocess then died with
``ZMQError: Address already in use`` binding ``base + attn_dp_rank``.
"""

from __future__ import annotations

import socket

from infera.common.net import free_tcp_port_block


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
