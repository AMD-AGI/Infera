# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The one-shot sync job. Design §9.

Not a reconciliation loop: once, at task start, scoped to this task's subtree
and never the root. ``rsync`` is what spec §5.2 names and what §9.1 needs — a
one-shot copy, not a reconciler — and §9.3 adds the one thing it cannot do.
"""

from __future__ import annotations

import filecmp
import os
import shutil
import subprocess
from collections.abc import Mapping
from enum import Enum

from env_mgr.fs.domain import DomainKind, subdir_for
from env_mgr.fs.zone import Zone
from env_mgr.protocols import SyncReport

__all__ = ["Direction", "PLAYGROUND", "conflicts", "remote_root", "sync"]

PLAYGROUND = subdir_for(DomainKind.PLAYGROUND)


class Direction(str, Enum):
    """Required, never defaulted.

    Measured: ``rsync -a --delete`` makes two trees equal by **destroying**
    everything the destination had that the source did not, and there is no
    symmetric mode. Spec §5.3's "local and remote are made identical" is
    therefore implemented as "the destination is made identical to the source,
    and the caller names which is which".
    """

    LOCAL_TO_REMOTE = "local_to_remote"
    REMOTE_TO_LOCAL = "remote_to_local"


def remote_root(zone: Zone, mapping: Mapping[str, str]) -> str | None:
    """This zone's counterpart on the far side, or `None` if nothing maps it.

    **The mapping is this module's, so the walk over it is too.** It was inlined
    in `_ends`, which is the shape a caller needs for an rsync — a source and a
    destination ordered by direction. `paths.zone_env` needs the other shape:
    *"where does this zone live over there"*, with no direction and no copy. A
    second walk in that module would have been `engineer_principle.md` §3's
    symptom exactly — an outsider taking `Context.mapping` and computing with it
    — and would have given one fact two writers, so the next mapping rule
    (a trailing separator, a nested mapping, a longest-prefix tie) would land in
    one of them.

    **`None` rather than a raise**, because the two callers want different
    things from a miss. `_ends` cannot proceed and says so; an environment
    variable for a side that is not configured is simply *absent*, which is this
    module's established shape for an unresolvable path — `grants.output_paths`
    omits a slot with no pinned version rather than presenting it as empty.
    """
    for local_root, far_root in mapping.items():
        prefix = local_root.rstrip(os.sep)
        if zone.root == prefix or zone.root.startswith(prefix + os.sep):
            rel = os.path.relpath(zone.root, prefix)
            return os.path.join(far_root, rel) if rel != os.curdir else far_root
    return None


def _ends(zone: Zone, mapping: dict[str, str], direction: Direction) -> tuple[str, str]:
    """This zone's two sides. Per task, never the root (spec §5.3).

    Both reasons the spec gives were measured, and only one holds at the size
    measured: 20 tasks × 50 files is 108 ms for the whole root against 53 ms for
    one task, because rsync's fixed startup dominates. **The argument that holds
    here is correctness** — not touching another task's material.
    """
    remote = remote_root(zone, mapping)
    if remote is None:
        raise KeyError(
            f"no mapping covers zone {zone.root!r} (have {sorted(mapping)}); "
            f"sync is per task and needs the task's own mapping"
        )
    if direction is Direction.LOCAL_TO_REMOTE:
        return zone.root, remote
    return remote, zone.root


def conflicts(src: str, dst: str) -> tuple[str, ...]:
    """Paths that exist on **both** sides and differ, before anything is written.

    ``rsync`` cannot report this and no flag makes it: with both sides edited,
    ``-a`` and ``--checksum`` silently discard one and ``--update`` guesses by
    mtime. Detection is a pre-pass, and it is what converts silent data loss
    into a stopped task. That is not conflict *resolution*, which stays on the
    roadmap — it is the difference the open question is actually about.
    """
    if not (os.path.isdir(src) and os.path.isdir(dst)):
        return ()
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        if rel.split(os.sep)[0] == PLAYGROUND:
            dirnames[:] = []
            continue
        for name in filenames:
            a = os.path.join(dirpath, name)
            b = os.path.join(dst, os.path.relpath(a, src))
            if os.path.exists(b) and not filecmp.cmp(a, b, shallow=False):
                found.append(os.path.relpath(a, src))
    return tuple(sorted(found))


def sync(zone: Zone, mapping: dict[str, str], *, direction: Direction) -> SyncReport:
    """Once, at task start. Excludes the playground.

    ``--exclude playground/`` omits the contents but still creates the directory
    on the far side, empty — which is consistent with the remote having its own
    playground, and is what criterion 16's assertion must actually say.
    """
    src, dst = _ends(zone, mapping, direction)
    found = conflicts(src, dst)
    os.makedirs(dst, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync is None:
        raise RuntimeError("rsync is not installed; the one-shot sync has no mechanism")
    proc = subprocess.run(
        [
            rsync,
            "-a",
            "--delete",
            "--stats",
            f"--exclude={PLAYGROUND}/**",
            src.rstrip(os.sep) + os.sep,
            dst.rstrip(os.sep) + os.sep,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    os.makedirs(os.path.join(dst, PLAYGROUND), exist_ok=True)
    return SyncReport(
        sent=_stat(proc.stdout, "Number of regular files transferred"),
        received=_stat(proc.stdout, "Number of deleted files"),
        conflicts=found,
    )


def _stat(text: str, label: str) -> int:
    for line in text.splitlines():
        if line.startswith(label):
            digits = line.split(":", 1)[1].strip().replace(",", "")
            try:
                return int(digits)
            except ValueError:  # pragma: no cover - rsync formats vary
                return 0
    return 0
