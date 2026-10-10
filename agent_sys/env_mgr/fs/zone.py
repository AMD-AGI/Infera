# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`Zone` — one task **attempt's** region. Design §3.4."""

from __future__ import annotations

import hashlib
from typing import Any, NamedTuple

from env_mgr.fs.path import contained

__all__ = ["Zone", "slug", "zone_dirname", "validation_dirname"]

#: Readability and accident-avoidance only. Spec §4.1 settled that an
#: unguessable prefix is security-by-obscurity — it was recovered three ways by
#: the agent that holds it — so this buys no confidence and is not asked to.
_HASH_CHARS = 8

#: Long enough to recognise a closure name, short enough that a zone path still
#: fits in a terminal beside a full uuid. A truncated slug is a label, never a
#: key — nothing resolves a directory through it.
_SLUG_CHARS = 40


def _tag(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:_HASH_CHARS]


def slug(text: Any) -> str:
    """A directory-name-safe label, or ``""`` when there is nothing to say.

    **``.`` must not survive**, and that is the whole reason this exists rather
    than the name being interpolated raw: ``.`` is this module's field
    separator, so a closure called ``a.b`` would silently add a field and
    `find_zone_dir`'s ``parts[-2]`` would read the wrong one.
    """
    if text is None:
        return ""
    out: list[str] = []
    for char in str(text):
        keep = char if (char.isascii() and (char.isalnum() or char in "_-")) else "-"
        if keep == "-" and (not out or out[-1] == "-"):
            continue
        out.append(keep)
    return "".join(out).strip("-")[:_SLUG_CHARS].strip("-")


def zone_dirname(task_id: Any, attempt: int, name: Any = None) -> str:
    """``task.<name>.<uuid>.<version>.<hash>`` — spec §5.1, plus a label.

    The zone id is the runtime ``uuid.version``: the task's own identity, not a
    separate namespace. ``version`` is the attempt, because a zone belongs to an
    attempt (design §11.3).

    `name` is the task's closure — a label for whoever is reading the tree, and
    nothing else. It is **not** in `_tag`'s input and nothing resolves through
    it, so an unnamed task still gets today's exact name and a renamed closure
    does not move an existing zone. It sits *before* the uuid because that is
    what makes ``ls`` and tab-completion useful; the uuid stays whole and stays
    the field before the attempt, which is what every lookup keys on.
    """
    uid = str(task_id)
    label = slug(name)
    prefix = f"task.{label}." if label else "task."
    return f"{prefix}{uid}.{attempt}.{_tag(uid, str(attempt))}"


def validation_dirname(task_id: Any, phase: str, name: Any = None) -> str:
    """``validation.<name>.<uuid>.<phase>.<hash>`` — design §8.3, deviation D5.

    A **sibling** of the producing task's zone, never a descendant of it.
    Anything under the producing task's directory is inside its subtree and
    therefore reachable, which is exactly what criterion 13 forbids.

    `name` is a label on the same terms as `zone_dirname`'s.
    """
    uid = str(task_id)
    label = slug(name)
    prefix = f"validation.{label}." if label else "validation."
    return f"{prefix}{uid}.{phase}.{_tag(uid, phase)}"


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
