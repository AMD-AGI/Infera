# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Render Outcomes for humans and machines."""

from __future__ import annotations

import json

from .outcome import Outcome

_ICON = {"ok": "OK ", "info": "INFO", "warn": "WARN", "fail": "FAIL"}


def render_human(outcomes: list[Outcome], status: str) -> str:
    lines = [f"[{_ICON.get(o.level, o.level)}] {o.message}" for o in outcomes]
    lines.append("")
    lines.append(f"status: {status}")
    return "\n".join(lines)


def render_json(outcomes: list[Outcome], status: str) -> str:
    return json.dumps(
        {
            "status": status,
            "outcomes": [
                {"level": o.level, "message": o.message, "details": o.details} for o in outcomes
            ],
        },
        indent=2,
    )
