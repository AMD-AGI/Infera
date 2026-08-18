# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Five-layer model. Override runs left -> right (system most general)."""

from __future__ import annotations

LAYER_ORDER = ("system", "workspace", "project", "repo", "worktree")


def layer_index(name: str) -> int:
    try:
        return LAYER_ORDER.index(name)
    except ValueError as e:
        raise ValueError(f"unknown layer: {name!r} (expected one of {LAYER_ORDER})") from e
