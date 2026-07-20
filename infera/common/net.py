###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import socket


def free_tcp_port() -> int:
    """Bind to port 0 to let the kernel assign a free port, then release.

    There is a small race window between releasing and the caller re-binding;
    in practice this is acceptable for our use case (engine processes
    allocating ports for ZMQ event publishers, PD bootstrap sockets, etc.)
    and matches what SGLang itself does internally.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def free_tcp_port_block(count: int) -> int:
    """Lowest base such that ``base .. base+count-1`` are all free TCP ports.

    SGLang binds one KV-event publisher per DP rank at ``base + attn_dp_rank``,
    so a single free base is not enough. We scan downward from just below the OS
    ephemeral range: a base there won't be handed out to any ``bind(("", 0))``
    caller (ours or SGLang's internal sockets), so the whole block survives the
    window until the engine binds it.
    """
    if count <= 1:
        return free_tcp_port()
    try:
        low = int(open("/proc/sys/net/ipv4/ip_local_port_range").read().split()[0])
    except (OSError, ValueError, IndexError):
        low = 32768
    for base in range(low - count, 1024, -1):
        socks: list[socket.socket] = []
        try:
            for off in range(count):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("", base + off))
                socks.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError(f"could not find {count} contiguous free TCP ports")
