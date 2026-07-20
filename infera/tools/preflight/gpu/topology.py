###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""GPU topology: inter-GPU link type (xGMI vs PCIe) + GPU<->NUMA mapping.

Parses rocm-smi; no torch needed. Warns if any GPU-GPU link isn't xGMI (a
high-speed link is missing or degraded to PCIe) or if GPUs are split unevenly
across NUMA nodes.
"""

from __future__ import annotations

import re
from collections import Counter

from ..finding import Finding
from ..util import have, run


def _link_matrix() -> list[list[str | None]] | None:
    rc, out = run(["rocm-smi", "--showtopotype"])
    if rc != 0:
        return None
    rows: list[list[str]] = []
    for line in out.splitlines():
        m = re.match(r"^GPU(\d+)\s+(.*)", line)
        if m:
            rows.append(m.group(2).split())
    if not rows:
        return None
    n = len(rows)
    return [
        [None if i == j else (rows[i][j] if j < len(rows[i]) else "?") for j in range(n)]
        for i in range(n)
    ]


def _numa_map() -> dict[int, int]:
    rc, out = run(["rocm-smi", "--showtoponuma"])
    numa: dict[int, int] = {}
    if rc == 0:
        for m in re.finditer(r"GPU\[(\d+)\].*?Numa Node:\s*(\d+)", out):
            numa[int(m.group(1))] = int(m.group(2))
    return numa


def collect() -> list[Finding]:
    if not have("rocm-smi"):
        return [Finding("info", "rocm-smi not found; topology check skipped", {})]

    findings: list[Finding] = []

    matrix = _link_matrix()
    if matrix:
        findings.append(Finding("info", "GPU link topology", {"matrix": matrix, "unit": "link"}))
        for i, row in enumerate(matrix):
            for j, t in enumerate(row):
                # link type is symmetric; report each unordered pair once
                if j > i and t is not None and t.upper() != "XGMI":
                    findings.append(Finding("warn", f"gpu{i}-gpu{j} link not xGMI", {"type": t}))
    else:
        findings.append(Finding("warn", "GPU link topology unavailable", {}))

    numa = _numa_map()
    if numa:
        per_node = dict(Counter(numa.values()))
        findings.append(Finding("info", "GPU<->NUMA mapping", {"numa": numa, "per_node": per_node}))
        if len(set(per_node.values())) > 1:
            findings.append(
                Finding("warn", "GPU<->NUMA distribution imbalanced", {"per_node": per_node})
            )

    return findings
