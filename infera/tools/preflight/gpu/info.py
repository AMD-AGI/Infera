###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""GPU probe: count / model / VRAM / temperature / driver.

Uses rocm-smi (AMD). Best-effort: if rocm-smi is absent we fall back to counting
render nodes so the report still says how many GPUs the kernel sees.
"""

from __future__ import annotations

import glob
import json
import statistics

from ..finding import Finding
from ..util import have, run

# A card hotter than the node median by this many °C is flagged as an outlier.
_TEMP_OUTLIER_C = 15.0


def _rocm_smi_json() -> dict:
    rc, out = run(
        ["rocm-smi", "--showproductname", "--showtemp", "--showmeminfo", "vram", "--json"],
        merge_stderr=False,
    )
    if rc != 0:
        return {}
    try:
        return json.loads(out)
    except (ValueError, TypeError):
        return {}


def _driver_version() -> str:
    rc, out = run(["rocm-smi", "--showdriverversion"])
    if rc == 0:
        for line in out.splitlines():
            if "Driver" in line and ":" in line:
                return line.split(":", 1)[1].strip()
    return ""


def _count_render_nodes() -> int:
    return len(glob.glob("/dev/dri/renderD*"))


def collect() -> list[Finding]:
    findings: list[Finding] = []

    if not have("rocm-smi"):
        n = _count_render_nodes()
        lvl = "warn" if n == 0 else "info"
        findings.append(Finding(lvl, "rocm-smi not found; counted render nodes", {"gpu_count": n}))
        return findings

    data = _rocm_smi_json()
    cards = {k: v for k, v in data.items() if k.lower().startswith("card")}
    models = sorted({(v.get("Card Series") or v.get("Card Model") or "?") for v in cards.values()})
    # GFX arch (e.g. gfx950 = MI355X) is stable across rocm-smi versions, unlike
    # the marketing name which some rocm-smi builds report as "AMD Radeon Graphics".
    gfx = sorted({v.get("GFX Version", "?") for v in cards.values()})
    findings.append(
        Finding(
            "info" if cards else "warn",
            "GPU inventory",
            {
                "gpu_count": len(cards),
                "model": models[0] if len(models) == 1 else models,
                "gfx": gfx[0] if len(gfx) == 1 else gfx,
                "driver": _driver_version(),
            },
        )
    )

    temps: dict[str, float] = {}
    vrams: set[float] = set()
    for card, v in sorted(cards.items()):
        name = v.get("Card Series") or v.get("Card Model") or v.get("Card SKU") or "?"
        # rocm-smi key names drift across versions; prefer junction (hotspot),
        # else fall back to any "Temperature ... (C)" field.
        temp = None
        for key, val in v.items():
            if "Temperature" in key and "junction" in key.lower():
                temp = val
                break
            if temp is None and "Temperature" in key:
                temp = val
        detail = {"card": card, "name": name, "gfx": v.get("GFX Version"), "temp_c": temp}
        vram = v.get("VRAM Total Memory (B)")
        if vram:
            try:
                gib = round(int(vram) / 1024**3, 1)
                detail["vram_gib"] = gib
                vrams.add(gib)
            except (TypeError, ValueError):
                pass
        try:
            temps[card] = float(temp)
        except (TypeError, ValueError):
            pass
        findings.append(Finding("info", f"{card} present", detail))

    # A node's cards should all report the same VRAM; a smaller one is degraded.
    if len(vrams) > 1:
        findings.append(
            Finding("warn", "GPU VRAM differs across cards", {"vram_gib": sorted(vrams)})
        )

    # Flag a card running much hotter than the node median. A relative outlier
    # beats a fixed ceiling: "all 50, one 70" is a real problem below any threshold.
    if len(temps) >= 3:
        med = statistics.median(temps.values())
        for card, t in sorted(temps.items()):
            if t >= med + _TEMP_OUTLIER_C:
                findings.append(
                    Finding(
                        "warn", f"{card} temperature outlier", {"temp_c": t, "node_median": med}
                    )
                )

    return findings
