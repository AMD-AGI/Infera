###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Host probe: CPU / memory / disk / NUMA / memlock ulimit.

Read-only. memlock must be unlimited for RDMA pinned memory — a low limit is
the classic cause of RDMA registration failures, so we flag it.
"""

from __future__ import annotations

import os
import re
import resource
import shutil

from ..finding import Finding
from ..util import read_text, run


def _cpu_summary() -> dict:
    rc, out = run(["lscpu"])
    info = {}
    if rc == 0:
        for key in ("Model name", "CPU(s)", "Socket(s)", "NUMA node(s)", "Thread(s) per core"):
            m = re.search(rf"^{re.escape(key)}:\s*(.+)$", out, re.MULTILINE)
            if m:
                info[key] = m.group(1).strip()
    return info


def _mem_gib() -> dict:
    txt = read_text("/proc/meminfo") or ""
    out = {}
    for key in ("MemTotal", "MemAvailable"):
        m = re.search(rf"^{key}:\s*(\d+)\s*kB", txt, re.MULTILINE)
        if m:
            out[key] = round(int(m.group(1)) / 1024 / 1024, 1)
    return out


def _memlock_limit() -> tuple:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        return soft, hard
    except Exception:  # noqa: BLE001
        return None, None


def collect() -> list[Finding]:
    findings: list[Finding] = []

    cpu = _cpu_summary()
    mem = _mem_gib()
    findings.append(
        Finding(
            "info",
            "Host CPU/memory",
            {
                "cpu_model": cpu.get("Model name"),
                "cpus": cpu.get("CPU(s)"),
                "sockets": cpu.get("Socket(s)"),
                "numa_nodes": cpu.get("NUMA node(s)"),
                "mem_total_gib": mem.get("MemTotal"),
                "mem_avail_gib": mem.get("MemAvailable"),
            },
        )
    )

    # Disk free on paths that matter for a run (root, scratch, model cache dir).
    disk = {}
    for path in ("/", "/tmp", os.environ.get("INFERA_E2E_MODEL_DIR") or ""):
        if path and os.path.isdir(path):
            try:
                usage = shutil.disk_usage(path)
                disk[path] = f"{usage.free / 1e9:.0f}GB free / {usage.total / 1e9:.0f}GB"
            except Exception:  # noqa: BLE001
                pass
    findings.append(Finding("info", "Disk free", {"paths": disk}))

    soft, hard = _memlock_limit()
    unlimited = soft == resource.RLIM_INFINITY
    detail = {"soft": "unlimited" if unlimited else soft, "hard": hard}
    if unlimited:
        findings.append(Finding("info", "memlock ulimit unlimited", detail))
    else:
        findings.append(
            Finding("warn", "memlock ulimit not unlimited (RDMA registration may fail)", detail)
        )

    return findings
