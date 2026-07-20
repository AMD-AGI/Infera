###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Storage probe: enumerate local NVMe drives (count / size / model).

Read-only, best-effort: a missing ``lsblk`` degrades to a warn, never raises.
A device counts as local NVMe when its transport is ``nvme`` OR its name starts
with ``nvme`` — the name is the reliable fallback when a container's /sys leaves
lsblk's TRAN column empty (the kernel always names NVMe namespaces ``nvmeXnY``).
"""

from __future__ import annotations

import json

from ..finding import Finding
from ..util import run


def _lsblk_disks() -> list[dict] | None:
    """Whole-disk block devices via lsblk JSON, or None if lsblk is unavailable
    / its output can't be parsed. ``-d`` skips partitions, ``-b`` gives SIZE in
    bytes so the GB conversion is exact."""
    rc, out = run(
        ["lsblk", "-d", "-b", "-o", "NAME,TRAN,SIZE,MODEL", "--json"],
        merge_stderr=False,
    )
    if rc != 0:
        return None
    try:
        return json.loads(out).get("blockdevices", [])
    except (ValueError, TypeError, AttributeError):
        return None


def _size_gb(v: object) -> float | None:
    # lsblk -b emits SIZE in bytes: int on new util-linux, str on old ones.
    try:
        return round(int(v) / 1e9, 1)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def collect() -> list[Finding]:
    disks = _lsblk_disks()
    if disks is None:
        return [Finding("warn", "cannot enumerate block devices (lsblk unavailable)", {})]

    nvme: list[dict] = []
    for d in disks:
        name = (d.get("name") or "").strip()
        tran = (d.get("tran") or "").strip().lower()
        if tran != "nvme" and not name.startswith("nvme"):
            continue
        nvme.append(
            {
                "name": name,
                "size_gb": _size_gb(d.get("size")),
                "model": (d.get("model") or "").strip() or None,
            }
        )

    if nvme:
        return [Finding("info", f"{len(nvme)} local NVMe device(s)", {"devices": nvme})]
    seen = sorted({f"{d.get('name') or '?'}({d.get('tran') or '?'})" for d in disks})
    return [Finding("warn", "no local NVMe detected", {"disks_seen": seen})]
