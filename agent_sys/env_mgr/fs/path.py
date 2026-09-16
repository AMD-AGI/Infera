# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Canonical containment, failing closed. Design §3.2 and §6.3.

The bottom of the import graph: this module imports only ``os`` and ``pathlib``,
and three others sit on it. That is what "the path is the fact" means as an
import edge.

**This is not the enforcement mechanism.** Measured (materials §1): the kernel
already denies all three documented ``startswith`` defeats with no userspace
check involved. This module has three real callers — policy construction, which
decides what is handed to the kernel; grant resolution, which refuses a grant
whose literal and canonical forms disagree; and the ``PreToolUse`` hook, which
attributes a denial to a tool call in a way the kernel cannot.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "canonical_here",
    "canonical_syntax",
    "contained",
    "contained_syntactically",
    "resolve_strict",
    "resolve_for_check",
]

#: Characters a grant path may not contain. ``*`` is the closed side of the
#: covering grammar (`closure` design §6.3): if the schema never admits a
#: wildcard, no component can be the one that gives it meaning.
_WILDCARDS = ("*", "?", "[", "]")


def resolve_strict(path: str | os.PathLike[str]) -> str | None:
    """Resolve `path`, or return ``None``. **Any** failure returns ``None``.

    ``os.path.realpath`` does not raise on a broken symlink or on a symlink
    loop — it returns a partly-resolved path — and neither does
    ``Path.resolve()`` at its default ``strict=False``. That default is the
    trap, because it is what an implementation reaches for.

    A NUL byte raises ``ValueError``, **not** ``OSError``, so a handler written
    the natural way (``except OSError``) makes the NUL rule dead code.
    """
    try:
        return os.fspath(Path(path).resolve(strict=True))
    except (OSError, ValueError, RuntimeError):
        return None


def resolve_for_check(path: str | os.PathLike[str]) -> str | None:
    """As `resolve_strict`, but a path that does not exist yet resolves through
    its **parent** — design §3.2 rule 3's last clause.

    The fallback is available only when the final component does **not exist at
    all**. A broken symlink and a symlink loop both exist and both fail to
    resolve, and those are exactly criterion 4's cases: falling back for them
    would resolve the loop to the zone it sits in and call it contained, which
    is the failure the criterion names.
    """
    try:
        p = Path(path)
        exists = os.path.lexists(p)
    except (TypeError, ValueError, OSError):
        return None
    resolved = resolve_strict(p)
    if resolved is not None:
        return resolved
    if exists:
        return None
    parent = resolve_strict(p.parent) if p.parent != p else None
    if parent is None:
        return None
    name = p.name
    if not name or name in (os.curdir, os.pardir) or "\x00" in name:
        return None
    return os.path.join(parent, name)


def contained(path: str | os.PathLike[str], zone: str | os.PathLike[str]) -> bool:
    """True iff `path` is `zone` or lies beneath it, on resolved paths.

    Four rules, each measured (design §3.2):

    1. Resolve first, both sides.
    2. The trailing separator is load-bearing — ``zone-EVIL/x`` passes a bare
       ``startswith`` and fails this. That is the CVE-2025-54794 defeat.
    3. Canonicalisation fails closed.
    4. Reject NUL bytes.

    Canonicalised per check, at use time. Resolving attacker-mutable components
    early is itself a TOCTOU bug.
    """
    p = resolve_for_check(path)
    # The zone must exist. A zone that cannot be resolved is not a zone, and
    # under principle 3 an undecidable comparison denies.
    z = resolve_strict(zone)
    if p is None or z is None:
        return False
    if z == os.sep:
        return p.startswith(os.sep)
    z = z.rstrip(os.sep)
    return p == z or p.startswith(z + os.sep)


def contained_syntactically(rel: str, root: str) -> str | None:
    """Join `rel` under `root` **without touching a filesystem**, or `None`.

    `contained` is the one to use whenever the path is on this machine: it
    resolves both sides, so it defeats a symlink out of the zone. It calls
    `resolve_strict(zone)`, which requires the zone to **exist** — so it cannot
    be used for a path on another host, where it would deny everything.

    This is the weaker check for exactly that case, and the weakness is
    specific and must not be forgotten: **a symlink on the far side defeats
    it.** `<remote zone>/link -> /etc` is refused by `contained` and admitted
    here. What it does stop is the whole class that does not need a filesystem
    — an absolute path, and any `..` that climbs out — which is what an agent
    reaches for by accident.

    That is consistent with what the module already says about the far side
    rather than a new hole: `remote/__init__.py` states outright that *the far
    side is less confined than the near side*, and design §10.4 records it as
    open. Closing it properly means asking the far side to resolve the path —
    `realpath` over the connection — and comparing there, which is a round trip
    per call and a TOCTOU window of its own. Not done; recorded.

    Returns the joined path, or `None` when `rel` escapes. `None` rather than a
    raise because the caller phrases the refusal for its own audience.
    """
    if not rel or "\x00" in rel or "\x00" in root:
        return None
    if os.path.isabs(rel):
        return None
    norm = os.path.normpath(rel)
    if norm == os.pardir or norm.startswith(os.pardir + os.sep):
        return None
    if norm == os.curdir:
        return root.rstrip(os.sep) or os.sep
    return os.path.join(root.rstrip(os.sep), norm)


def canonical_syntax(path: str) -> bool:
    """Load-time, **no filesystem**: is this string already its own canonical form?

    Absolute; no ``.`` or ``..`` segment; no trailing separator; no repeated
    separator; no NUL; no wildcard character. Purely syntactic, so it is
    checkable when no zone exists yet — which is the half of design D4 that
    belongs in the schema admitting grant paths.
    """
    if not isinstance(path, str) or not path:
        return False
    if "\x00" in path or any(c in path for c in _WILDCARDS):
        return False
    if not path.startswith(os.sep):
        return False
    if path == os.sep:
        return True
    if path.endswith(os.sep):
        return False
    segments = path.split(os.sep)[1:]
    return all(s and s not in (os.curdir, os.pardir) for s in segments)


def canonical_here(path: str) -> bool:
    """Zone-build time: does `path` resolve to itself? Fails closed.

    The realpath half of design D4. It cannot run at load, because the symlinks
    it would follow do not exist until the layout does.
    """
    if not canonical_syntax(path):
        return False
    return resolve_strict(path) == path
