# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Rendering for the CLI's `domain` and `zone` sub-commands. Design §12.2.

Above the wall. It is a separate module rather than more of `cli.py` because
`cli.py` is the one shipped file this work touches, and 65 tests are pointed at
it: the less of the new surface that lives there, the smaller the thing those
tests are exposed to.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from env_mgr.meta import Meta, configured_path, load

__all__ = ["render_domains", "render_zones"]


def _meta(args: argparse.Namespace) -> Meta:
    """`--meta`, then ``$ENV_MGR_META``, then ``~/.config/env_mgr/meta.json``.

    The order lives in `meta.configured_path` now that `cli/main.py` reads the
    same file to configure a run's sync mapping. This module is no longer its
    only reader, so it may no longer be its owner.
    """
    return load(configured_path(args.meta))


def _emit(rows: list[dict[str, Any]], args: argparse.Namespace, empty: str) -> str:
    if args.json:
        return json.dumps(rows, indent=2, sort_keys=True)
    if not rows:
        return empty
    return "\n".join("  ".join(f"{k}={v}" for k, v in row.items()) for row in rows)


def render_domains(args: argparse.Namespace) -> str:
    meta = _meta(args)
    rows = [
        {"name": name, "kind": kind, "root": root}
        for name, root, kind in meta.domains
        if args.name is None or name == args.name
    ]
    return _emit(rows, args, "no domains registered")


def render_zones(args: argparse.Namespace) -> str:
    """Every zone under the storage root, or one task's.

    A zone is a directory, so this reads the filesystem rather than a record:
    the layout *is* the index, which is the same property that lets a subtask be
    placed under a parent whose `Task` object is not in hand.
    """
    from env_mgr.fs.layout import _ZONE_PREFIX  # noqa: PLC0415 - one caller

    meta = _meta(args)
    try:
        base = meta.registry().storage_root()
    except ValueError as e:
        return f"no zone root: {e}"
    rows: list[dict[str, Any]] = []
    for dirpath, dirnames, _ in os.walk(base):
        for name in sorted(dirnames):
            if not name.startswith(_ZONE_PREFIX):
                continue
            parts = name.split(".")
            task_id, attempt = (parts[1], parts[-2]) if len(parts) >= 4 else (name, "?")
            if args.task_id is not None and task_id != args.task_id:
                continue
            rows.append(
                {
                    "task": task_id,
                    "attempt": attempt,
                    "root": os.path.join(dirpath, name),
                }
            )
    return _emit(rows, args, "no zones")
