# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`Zone` — one task **attempt's** region. Design §3.4."""

from __future__ import annotations

import hashlib
from typing import Any, NamedTuple

from env_mgr.fs.path import contained

__all__ = ["Zone", "zone_dirname", "validation_dirname"]

#: Readability and accident-avoidance only. Spec §4.1 settled that an
#: unguessable prefix is security-by-obscurity — it was recovered three ways by
#: the agent that holds it — so this buys no confidence and is not asked to.
_HASH_CHARS = 8


def _tag(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:_HASH_CHARS]


def zone_dirname(task_id: Any, attempt: int) -> str:
    """``task.<uuid>.<version>.<hash>`` — spec §5.1.

    The zone id is the runtime ``uuid.version``: the task's own identity, not a
    separate namespace. ``version`` is the attempt, because a zone belongs to an
    attempt (design §11.3).
    """
    uid = str(task_id)
    return f"task.{uid}.{attempt}.{_tag(uid, str(attempt))}"


def validation_dirname(task_id: Any, phase: str) -> str:
    """``validation.<uuid>.<phase>.<hash>`` — design §8.3, deviation D5.

    A **sibling** of the producing task's zone, never a descendant of it.
    Anything under the producing task's directory is inside its subtree and
    therefore reachable, which is exactly what criterion 13 forbids.
    """
    uid = str(task_id)
    return f"validation.{uid}.{phase}.{_tag(uid, phase)}"


class Zone(NamedTuple):
    """One task attempt's region.

    Not one task's. Grants resolve to ``<root>/<hid>/v<N>/`` and ``N`` lives on
    `Execution`, so a retry has a different granted set and rebuilds.
    """

    task_id: Any
    attempt: int
    root: str

    def contains(self, path: str) -> bool:
        """Because the layout nests and permissions cover the task's own subtree
        recursively, *"may this task reach that path"* is this one function."""
        return contained(path, self.root)
