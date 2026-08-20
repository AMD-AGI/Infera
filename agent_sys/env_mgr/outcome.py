# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Outcome — the result of running one stage against one item."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Severity ascending; index = severity rank.
LEVELS = ("ok", "info", "warn", "fail")


@dataclass
class Outcome:
    level: str  # one of LEVELS
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def worst_level(outcomes: Iterable[Outcome]) -> str:
    """Highest-severity level among outcomes; 'ok' if empty."""
    worst = 0
    for o in outcomes:
        worst = max(worst, LEVELS.index(o.level))
    return LEVELS[worst]


def status_from(outcomes: Iterable[Outcome]) -> str:
    """Roll a list of outcomes up to OK / WARN / FAIL."""
    level = worst_level(outcomes)
    if level == "fail":
        return "FAIL"
    if level == "warn":
        return "WARN"
    return "OK"
