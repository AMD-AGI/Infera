###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""RDMA fabric probe: per device, its RoCE v2 GID, its netdev's routable subnet,
and active MTU. Informational only -- whether cross-node RoCE actually works is
decided by the netperf section's live ib_write_bw measurement, not by a static
subnet comparison (which false-positives on link-local / rail-isolated fabrics).
"""

from __future__ import annotations

import ipaddress
import os
import re

from ..finding import Finding
from ..util import read_text, run

_IB = "/sys/class/infiniband"


def _rdma_devices() -> list[str]:
    try:
        return sorted(x for x in os.listdir(_IB) if x)
    except OSError:
        return []


def _netdev_map() -> dict[str, str]:
    """rdma device -> netdev, from `rdma link show`."""
    rc, out = run(["rdma", "link", "show"])
    mapping: dict[str, str] = {}
    if rc == 0:
        for line in out.splitlines():
            m = re.search(r"(\w+)/\d+\b.*\bnetdev\s+(\S+)", line)
            if m:
                mapping[m.group(1)] = m.group(2)
    return mapping


def _netdev_sysfs(dev: str) -> str | None:
    """rdma device -> netdev via sysfs. Fallback for when `rdma link show` is
    unavailable or reports no netdev (e.g. ionic inside a container), so the
    routable-address check still finds the NIC's IP."""
    try:
        nets = [n for n in os.listdir(f"{_IB}/{dev}/device/net") if n]
    except OSError:
        return None
    return sorted(nets)[0] if nets else None


def _roce_v2_gid(dev: str) -> str | None:
    """A RoCE v2 GID for the device, preferring a routable one (global IPv6 or an
    IPv4-mapped ``::ffff:a.b.c.d`` GID) over a link-local ``fe80::`` one. Only the
    unset all-zero slot is skipped -- an IPv4-mapped GID is literally
    ``0000:...:ffff:<ipv4>`` and must not be dropped by a bare ``0000:`` prefix."""
    base = f"{_IB}/{dev}/ports/1"
    fallback = None
    ipv4_mapped = None
    for i in range(16):
        gid = (read_text(f"{base}/gids/{i}") or "").strip()
        typ = (read_text(f"{base}/gid_attrs/types/{i}") or "").strip()
        if not gid or "RoCE v2" not in typ:
            continue
        if set(gid.replace(":", "")) <= {"0"}:  # unset / all-zero slot
            continue
        low = gid.lower()
        if low.startswith("fe80"):
            fallback = fallback or gid
        elif low.startswith("0000:0000:0000:0000:0000:ffff:"):  # IPv4-mapped RoCE v2
            # Match the full ::ffff: prefix, not a bare ":ffff:" substring -- a global
            # IPv6 GID may legitimately contain an ffff hextet mid-address and must
            # still be treated as global (returned immediately) below.
            ipv4_mapped = ipv4_mapped or gid
        else:  # global IPv6 RoCE v2
            return gid
    return ipv4_mapped or fallback


def _addrs_by_iface() -> dict[str, list[str]]:
    """iface -> routable CIDRs (IPv4 + IPv6), skipping loopback and link-local."""
    res: dict[str, list[str]] = {}
    for fam in ("-4", "-6"):
        rc, out = run(["ip", "-o", fam, "addr", "show"])
        if rc != 0:
            continue
        for line in out.splitlines():
            m = re.search(r"^\s*\d+:\s+(\S+)\s+inet6?\s+(\S+)", line)
            if not m:
                continue
            iface, cidr = m.group(1), m.group(2)
            if iface == "lo" or cidr.lower().startswith("fe80"):
                continue
            res.setdefault(iface, []).append(cidr)
    return res


def _subnet(cidr: str) -> str | None:
    try:
        return str(ipaddress.ip_network(cidr, strict=False))
    except ValueError:
        return None


def _active_mtu(dev: str) -> str | None:
    rc, out = run(["ibv_devinfo", "-d", dev])
    if rc == 0:
        m = re.search(r"active_mtu:\s*(\S+)", out)
        if m:
            return m.group(1)
    return None


def collect() -> list[Finding]:
    devs = _rdma_devices()
    if not devs:
        return [Finding("warn", "no RDMA devices found", {})]

    netdev = _netdev_map()
    addrs = _addrs_by_iface()
    table: dict[str, dict] = {}
    routable = 0
    for dev in devs:
        nd = netdev.get(dev) or _netdev_sysfs(dev) or "?"
        cidrs = addrs.get(nd, [])
        subnets = sorted({s for c in cidrs if (s := _subnet(c))})
        if cidrs:
            routable += 1
        table[dev] = {
            "netdev": nd,
            "gid_v2": _roce_v2_gid(dev),
            "subnets": subnets,
            "mtu": _active_mtu(dev),
        }

    findings = [Finding("info", "RDMA fabric", {"devices": table})]
    if routable == 0:
        findings.append(Finding("warn", "no RDMA NIC has a routable address (only link-local)", {}))
    return findings
