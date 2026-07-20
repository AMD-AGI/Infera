###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Firmware/env probe: GPU firmware (MEC) + GPU-direct (ais-check / P2PDMA).

The MEC firmware version has no reliable absolute threshold (it varies by GPU
generation), so we only *report* it here; cross-node inconsistency is flagged
later at render time (a node with an odd firmware is the real signal).

ais-check is a real functional check: if it's False/absent, the kvd L3 connector
silently downgrades gpu-direct to a CPU-bounce path — perf loss with no error.
"""

from __future__ import annotations

import glob
import gzip
import os
import re

from ..finding import Finding
from ..util import have, read_text, run


def _min_mec_fw() -> int | None:
    rc, out = run(["rocm-smi", "--showfw"])
    if rc != 0:
        return None
    vals = []
    for line in out.splitlines():
        if "MEC" in line:
            m = re.search(r"(\d+)\s*$", line.strip())
            if m:
                vals.append(int(m.group(1)))
    return min(vals) if vals else None


def _pci_p2pdma() -> str | None:
    """Read CONFIG_PCI_P2PDMA from kernel config, if exposed."""
    for path in glob.glob("/boot/config-*"):
        txt = read_text(path)
        if txt:
            m = re.search(r"^CONFIG_PCI_P2PDMA=(\S+)", txt, re.MULTILINE)
            if m:
                return m.group(1)
    try:
        with gzip.open("/proc/config.gz", "rt") as f:
            for line in f:
                if line.startswith("CONFIG_PCI_P2PDMA="):
                    return line.strip().split("=", 1)[1]
    except Exception:  # noqa: BLE001
        pass
    return None


def collect() -> list[Finding]:
    findings: list[Finding] = []

    mec = _min_mec_fw()
    if mec is None:
        findings.append(Finding("info", "MEC firmware version unknown", {}))
    else:
        findings.append(Finding("info", "MEC firmware", {"mec": mec}))

    # ais-check decides gpu-direct. Mirror the connector's exact verdict
    # (kvd_connector._detect_p2pdma_support): engaged iff a line matches
    # "Kernel P2PDMA support: True" (padded, hence \s-tolerant). A loose "true"
    # substring would false-positive on any unrelated line; rc isn't the verdict.
    ais = "/opt/rocm/bin/ais-check"
    if os.path.exists(ais) or have("ais-check"):
        rc, out = run([ais if os.path.exists(ais) else "ais-check"])
        text = out.strip()
        p2p_line = next((ln.strip() for ln in text.splitlines() if "P2PDMA" in ln), "")
        ok = bool(re.search(r"Kernel\s+P2PDMA\s+support\s*:\s*True", text))
        lvl = "info" if ok else "warn"
        msg = (
            "GPU-direct (ais-check) engaged" if ok else "GPU-direct NOT engaged (silent CPU-bounce)"
        )
        detail = {"rc": rc, "p2pdma": p2p_line or "(none)", "out": text[:200]}
        findings.append(Finding(lvl, msg, detail))
    else:
        findings.append(
            Finding("warn", "ais-check not found (cannot confirm gpu-direct)", {"path": ais})
        )

    p2p = _pci_p2pdma()
    findings.append(Finding("info", "kernel CONFIG_PCI_P2PDMA", {"value": p2p or "unknown"}))

    return findings
