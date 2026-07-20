###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Finding: the single result unit every probe emits.

Mirrors the pattern used by Primus preflight — each check returns a list of
``Finding`` with a level (info/warn/fail), a short message, and a details dict.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Finding:
    level: str  # "info" | "warn" | "fail"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def status_from(findings: Iterable[Finding]) -> str:
    """Roll a list of findings up to a single OK / WARN / FAIL status."""
    levels = {f.level for f in findings}
    if "fail" in levels:
        return "FAIL"
    if "warn" in levels:
        return "WARN"
    return "OK"
