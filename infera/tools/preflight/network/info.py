###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Network probe: NICs + RDMA devices + ionic↔netdev map + link state + per-NIC IP.

Inventory only; the netperf section does the live cross-node RoCE bandwidth test.
"""

from __future__ import annotations

import os
import re

from ..finding import Finding
from ..util import read_text, run


def _list_dir(path: str) -> list[str]:
    try:
        return sorted(x for x in os.listdir(path) if x)
    except OSError:
        return []


def _ib_port_state(dev: str) -> str:
    # e.g. "4: ACTIVE"
    txt = read_text(f"/sys/class/infiniband/{dev}/ports/1/state") or ""
    return txt.strip().split(":")[-1].strip() or "?"


def _rdma_link_map() -> dict[str, str]:
    """ionic_X -> netdev, parsed from `rdma link show`."""
    rc, out = run(["rdma", "link", "show"])
    mapping: dict[str, str] = {}
    if rc == 0:
        for line in out.splitlines():
            m = re.search(r"(ionic_\d+)/\d+\b.*\bnetdev\s+(\S+)", line)
            if m:
                mapping[m.group(1)] = m.group(2)
    return mapping


def _ipv4_by_iface() -> dict[str, list[str]]:
    rc, out = run(["ip", "-o", "-4", "addr", "show"])
    res: dict[str, list[str]] = {}
    if rc == 0:
        for line in out.splitlines():
            m = re.search(r"^\s*\d+:\s+(\S+)\s+inet\s+(\d+\.\d+\.\d+\.\d+/\d+)", line)
            if m:
                res.setdefault(m.group(1), []).append(m.group(2))
    return res


def collect() -> list[Finding]:
    findings: list[Finding] = []

    nics = _list_dir("/sys/class/net")
    ib_devices = _list_dir("/sys/class/infiniband")
    ipv4 = _ipv4_by_iface()
    findings.append(
        Finding(
            "info",
            "NIC inventory",
            {"nics": nics, "rdma_devices": ib_devices, "ipv4": ipv4},
        )
    )

    if not ib_devices:
        findings.append(Finding("warn", "no RDMA devices found", {}))
        return findings

    link_map = _rdma_link_map()
    inactive = []
    dev_table = {}
    for dev in ib_devices:
        state = _ib_port_state(dev)
        netdev = link_map.get(dev, "?")
        dev_table[dev] = {"netdev": netdev, "state": state}
        if state.upper() != "ACTIVE":
            inactive.append(dev)

    findings.append(Finding("info", "RDMA device map", {"devices": dev_table}))
    if inactive:
        findings.append(Finding("warn", "some RDMA ports not ACTIVE", {"inactive": inactive}))

    return findings
