"""The in-process containment check, and an honest statement of its worth.

Spec §6.1 makes reach a matter of containment; `env_mgr` owns the OS
enforcement. What lives here is the check that runs in-process, and
`design.md` §7.1 is the sentence that must not be lost: a validated path
*string* proves filesystem containment **at an instant**, never enforced
containment. The kernel layer is the boundary.

What the string check still buys, and the kernel layer cannot: an early
attributable error naming the offending task, a check on paths that **do not
exist yet**, and coverage before any syscall.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePath

from handoff.errors import NotContained

__all__ = ["check_contained"]


def check_contained(candidate: Path, zone: Path) -> None:
    """Raise `NotContained` unless `candidate` resolves inside `zone`.

    Both naive checks are wrong, measured against zone `…/a/b` and candidate
    `…/a/bc/x`: `str.startswith(zone)` says True — that is the live bug behind
    CVE-2026-5422, CVE-2026-40256 and CVE-2026-48544 — and
    `Path.is_relative_to` alone says True for `…/a/b/../bc/x`, because it is
    purely lexical and does not collapse `..`. So `..` is rejected **by policy
    before resolving**, which also means a rejected path is reported as written
    rather than as resolved.

    **Fails closed: unresolvable means denied.** `validator`'s separation check
    needs the opposite direction — there, unresolvable must be treated as
    *inside*, because containment means reject — so importing this and negating
    at the call site would accept a dangling validator symlink. Two uses of one
    idea, two failure directions, and they are not one function.

    A zone equal to the candidate is contained: a task reaches its own subtree.
    """
    if any(part == os.pardir for part in PurePath(candidate).parts):
        raise NotContained(
            f"{candidate} is rejected by policy: a path containing '..' is not "
            f"checked against {zone}, it is refused as written"
        )
    try:
        resolved = Path(candidate).resolve()
        base = Path(zone).resolve()
    except OSError as exc:
        raise NotContained(f"{candidate} cannot be resolved against {zone}: {exc}") from exc

    if not resolved.is_relative_to(base):
        raise NotContained(
            f"{candidate} resolves to {resolved}, which is outside {base}. "
            f"Permission is containment (handoff spec §6.1)"
        )
