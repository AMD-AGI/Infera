###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Strict anchor-edit helper for the patch_*.py scripts in this directory.

A patch is ``{rel_path: (marker, [(old, new), ...])}``, paths relative to the
root of the importable ``package``. A file whose source already holds its
marker counts as applied and is left alone. Otherwise every ``old`` must occur
exactly once (checked in order, each edit seeing the previous ones) or the
patch fails; ``(old, old)`` thus only requires ``old``. Every file is checked
before any is written, so a failure leaves the package untouched.
"""

import importlib.util
import os


def _package_root(package: str) -> str | None:
    """One root for every file, so a patch never spans two package copies."""
    spec = importlib.util.find_spec(package)
    locations = list(spec.submodule_search_locations or []) if spec else []
    return locations[0] if locations else None


def apply(
    tag: str,
    patch: dict[str, tuple[str, list[tuple[str, str]]]],
    package: str,
) -> int:
    """Apply ``patch``; print one log line and return the process exit code."""
    pending: dict[str, str] = {}
    applied: list[str] = []
    root = _package_root(package)
    if root is None:
        print(f"[{tag}] FAILED: package {package!r} not found")
        return 1
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
