"""Criterion 11 — the producer may not legislate its own standard.

The principle is a separation of legislative and executive power. A producing
task has, and must have, the power to *execute* code in its own zone. What it may
not have is the power to *write the rule it will be judged by*: the candidate
writing the exam, the answer key, and grading it.

Two declarations, compared at load. `env_mgr` builds a sandbox at task start and
that is irrelevant here — the sandbox *executes* the declaration; this reads it.

Two things about the comparison, and the second is the trap.

**Resolution is mandatory, not prudent.** Spec §9.1 *sanctions* cross-package
symlinks — that is how two packages share a handoff kind — so the dangerous case
is a link in a neutral package pointing **into** the producer's zone, which is
lexically innocent and executes the producer's bytes. Measured over five checks ×
six layouts, wrong answers were 3 / 2 / 2 / 1 / 0; only `realpath` plus a trailing
separator gets all six right.

**The fail-closed direction is inverted relative to the sibling modules.** In
`handoff` §7 and `env_mgr` §4.3, *contained* means allow, so an unresolvable path
is denied. Here *contained* means **reject the validator**, so an unresolvable
path must be treated as **inside**. Importing `handoff.check_contained` and
negating at the call site would negate the fail-closed behaviour too, and a
dangling validator symlink would be accepted. Two uses of one idea, two failure
directions, and they are not one function.

Go's `internal` check is the sharpest prior art and it inverts for us: it
resolves symlinks **only to widen** access, so a link can never turn an allowed
import into a denied one. Ours is a rejection, so the corresponding asymmetry is
that resolution may only ever move a verdict toward *accept*. A deliberate
inversion of Go's risk posture, not a copy of it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from validator.protocols import SeparationViolation
from validator.spec import ValidatorSpec

__all__ = ["check_separation", "producer_zones", "reaches"]


def reaches(zone: Path, candidate: Path) -> bool:
    """Does `zone` reach `candidate`? **Unresolvable means yes.**

    The candidate is resolved strictly, because a validator's body is required to
    exist by the load-time check that runs before this one, so a candidate that
    will not resolve is a dangling symlink and must be rejected rather than
    admitted. The zone is resolved leniently, because a declared permission path
    is a declaration and need not exist on disk yet.

    The trailing separator is what separates `.../zone` from `.../zone-EVIL`, and
    it is the single difference between the check that got all six measured
    layouts right and the one that got five.
    """
    try:
        target = Path(os.path.realpath(candidate, strict=True))
    except OSError:
        return True  # unresolvable -> treated as inside -> the validator is rejected
    root = Path(os.path.realpath(zone))
    if target == root:
        return True
    return str(target).startswith(str(root) + os.sep)


def producer_zones(producer: Mapping[str, Any]) -> tuple[Path, ...]:
    """The paths a producing task's declared permissions reach.

    Reads the task spec's `permissions`, which is a *declaration* and therefore a
    plain document here rather than a `task_graph` object — this module never
    imports one. Two written shapes are accepted, because `task_graph` rev. 12's
    `Permissions` / `Grant` pair is not shipped and the spec key is a document
    either way: a list of path strings, or a list of mappings carrying a `path`.
    Anything else is not a path and is ignored rather than guessed at.
    """
    declared: Any = producer.get("permissions") or ()
    if isinstance(declared, Mapping):
        declared = declared.get("grants") or ()
    elif not isinstance(declared, (str, bytes, Sequence)):
        declared = getattr(declared, "grants", ())  # a task_graph.Permissions
    if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
        return ()
    out: list[Path] = []
    for entry in declared:
        if isinstance(entry, str):
            out.append(Path(entry))
        elif isinstance(entry, Mapping):
            if isinstance(entry.get("path"), str) and entry["path"]:
                out.append(Path(entry["path"]))
        elif isinstance(getattr(entry, "path", None), str) and entry.path:
            out.append(Path(entry.path))  # a task_graph.Grant
    return tuple(out)


def check_separation(
    validator: ValidatorSpec, producer: Mapping[str, Any], *, package_root: Path
) -> None:
    """Raise `SeparationViolation` if the producing task's declared permissions
    reach the validator's body.

    "Structural" means the layout, not anyone's assertion: the check consults no
    field in which a validator or a task claims its own independence, which is
    what makes it unfoolable by an author who is simply wrong about their own
    package.

    The honest ceiling, because it decides what the system may claim: this
    refuses **one shape** of a broader problem. Logic that lives elsewhere but was
    authored by the producing agent, influence through a shared upstream, a
    standard weakened before the producer ran — none of these have a path to
    compare. Spec §5.2's `logic_source` taxonomy is what carries that weight, and
    nothing verifies it.
    """
    if not validator.body:
        return  # a composite's implementation is its members, each checked itself
    zones = producer_zones(producer)
    if not zones:
        return
    body = validator.body
    parts: list[tuple[str, str]] = [("readme", body["readme"])]
    if body.get("entry"):
        parts.append(("entry", body["entry"]))
    parts += [("material", m) for m in body.get("materials", ())]

    for label, rel in parts:
        candidate = package_root / rel
        for zone in zones:
            if reaches(zone, candidate):
                raise SeparationViolation(
                    f"validator {validator.name!r}: {label} {candidate} lies inside "
                    f"the producing task's permission zone {zone} — the producer "
                    f"may execute in its own zone, but may not write the rule it "
                    f"will be judged by"
                )
