###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Strict anchor-edit helper shared by the patch_*.py scripts in this directory.

A patch is ``{rel_path: (marker, [(old, new), ...])}``, paths relative to one
``atom`` package root (the importable one, else /app/ATOM/atom). A file whose
source already holds its marker counts as applied and is left alone. Otherwise
every ``old`` must occur exactly once (checked in order, each edit seeing the
previous ones) or the patch fails; ``(old, old)`` thus only requires ``old``.
Every file is checked before any is written, so a failure leaves ATOM untouched.
"""

import importlib.util
import os

FALLBACK_ROOT = "/app/ATOM/atom"


def _atom_root() -> str:
    """One root for every file, so a patch never spans two ATOM copies."""
    spec = importlib.util.find_spec("atom")
    locations = list(spec.submodule_search_locations or []) if spec else []
    return locations[0] if locations else FALLBACK_ROOT


def apply(tag: str, patch: dict[str, tuple[str, list[tuple[str, str]]]]) -> int:
    """Apply ``patch``; print one log line and return the process exit code."""
    pending: dict[str, str] = {}
    applied: list[str] = []
    root = _atom_root()
    for rel, (marker, edits) in patch.items():
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            print(f"[{tag}] FAILED: {rel} not found under {root}")
            return 1
        with open(path, encoding="utf-8", newline="") as f:
            src = f.read()
        if marker in src:
            applied.append(rel)
            continue
        for old, new in edits:
            n = src.count(old)
            if n != 1:
                anchor = old.strip().splitlines()[0]
                print(f"[{tag}] FAILED: {rel}: anchor {anchor!r} found {n}x, expected 1")
                return 1
            src = src.replace(old, new, 1)
        pending[path] = src

    for path, src in pending.items():
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(src)
    parts = []
    if pending:
        parts.append(f"patched {', '.join(pending)}")
    if applied:
        parts.append(f"already applied: {', '.join(applied)}")
    print(f"[{tag}] {'; '.join(parts)}")
    return 0
