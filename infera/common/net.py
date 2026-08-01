###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import itertools
import logging
import os
import random
import socket

logger = logging.getLogger(__name__)

# How many randomly-chosen bases free_tcp_port_block tries before falling back
# to an exhaustive scan.
_PORT_BLOCK_TRIES = 64

# Kubernetes' default --service-node-port-range. A port in this window can be
# claimed cluster-wide by any Service at any time; kube-proxy in IPVS mode then
# creates an IPVS service for it on *every* node address, with no real server
# backing it on this node. Traffic to a node IP is then swallowed by IPVS before
# it reaches a local 0.0.0.0 listener and the kernel answers RST, so the port
# stays bindable and reachable over loopback while being "connection refused"
# from the node IP -- which is the address we advertise to peers.
_NODEPORT_RANGE_DEFAULT = "30000-32767"


def _parse_port_range(spec: str) -> tuple[int, int] | None:
    lo, _, hi = spec.partition("-")
    try:
        return int(lo), int(hi)
    except ValueError:
        return None


def _reserved_nodeport_range() -> tuple[int, int] | None:
    """The port window to keep out of, or None when the caller opted out.

    ``INFERA_NODEPORT_RANGE="lo-hi"`` for a cluster that moved the range,
    ``"none"`` to drop the guard. An unparseable value keeps the default
    rather than silently dropping the guard.
    """
    spec = os.environ.get("INFERA_NODEPORT_RANGE", _NODEPORT_RANGE_DEFAULT).strip()
    if spec.lower() in ("", "none", "off"):
        return None
    parsed = _parse_port_range(spec)
    if parsed is None:
        logger.warning(
            "INFERA_NODEPORT_RANGE=%r is not a 'lo-hi' range; using the default %s",
            spec,
            _NODEPORT_RANGE_DEFAULT,
        )
    return parsed or _parse_port_range(_NODEPORT_RANGE_DEFAULT)


def free_tcp_port() -> int:
    """Bind to port 0 to let the kernel assign a free port, then release.

    There is a small race window between releasing and the caller re-binding;
    in practice this is acceptable for our use case (engine processes
    allocating ports for ZMQ event publishers, PD bootstrap sockets, etc.)
    and matches what SGLang itself does internally.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind to loopback only: we just need a free port number, not a
        # publicly reachable listener. Binding to "" (all interfaces) would
        # briefly expose the probe socket on every NIC.
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def free_tcp_port_block(count: int) -> int:
    """Lowest base such that ``base .. base+count-1`` are all free TCP ports.

    SGLang binds one KV-event publisher per DP rank at ``base + attn_dp_rank``,
    so a single free base is not enough. We scan below the OS ephemeral range:
    a base there won't be handed out to any ``bind(("", 0))`` caller (ours or
    SGLang's internal sockets), so the whole block survives the window until
    the engine binds it.

    The scan start is **randomised**. The probe must release the block before
    returning (the caller's real listener binds ``0.0.0.0``, and a retained
    ``127.0.0.1`` reservation would shut its own child out), so the reservation
    cannot be exclusive. With a fixed start (``low - count``) that made
    collisions deterministic rather than merely possible: two engines launched
    on one host — e.g. the prefill and decode legs of a PD pair — both scanned
    from the same place, both saw the same free block, and the second died with
    ``ZMQError: Address already in use`` on ``base + rank``. Randomising spreads
    the picks so concurrent callers land on different blocks.

    The scan also skips the Kubernetes NodePort range, which a loopback bind
    probe cannot rule out on its own.
    """
    if count <= 1:
        return free_tcp_port()
    try:
        low = int(open("/proc/sys/net/ipv4/ip_local_port_range").read().split()[0])
    except (OSError, ValueError, IndexError):
        low = 32768
    reserved = _reserved_nodeport_range()
    # Candidate window: [1024, low - count]. Probe randomly-chosen bases first,
    # then fall back to an exhaustive downward scan so we never fail while a
    # block is still available.
    highest = low - count
    randomised = [random.randint(1024, highest) for _ in range(_PORT_BLOCK_TRIES)]
    for base in itertools.chain(randomised, range(highest, 1024, -1)):
        # A block inside the NodePort range binds and answers on loopback here,
        # so only skipping it up front keeps it out of the advertised endpoint.
        if reserved and base + count - 1 >= reserved[0] and base <= reserved[1]:
            continue
        socks: list[socket.socket] = []
        try:
            for off in range(count):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # Loopback-only reservation; see free_tcp_port() rationale.
                s.bind(("127.0.0.1", base + off))
                socks.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError(f"could not find {count} contiguous free TCP ports")
