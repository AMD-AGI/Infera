# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Runner: select items, detect conflicts, dispatch stages, roll up status."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .outcome import Outcome, status_from
from .recipe import IMPORTANCE, Item, Target
from .registry import get_installer
from .versions import constraints_conflict

STAGES = ("check", "dry-run", "install", "bootstrap")


@dataclass
class Filters:
    tags: list[str] = field(default_factory=list)
    installer: str | None = None
    importance: str | None = None
    item: str | None = None


def _importance_rank(name: str) -> int:
    # required strictest (0) ... suggested loosest (2)
    return IMPORTANCE.index(name)


def select(items: list[Item], filters: Filters) -> list[Item]:
    out = []
    for it in items:
        if filters.tags and not (set(filters.tags) & set(it.tags)):
            continue
        if filters.installer and it.installer != filters.installer:
            continue
        if filters.importance and _importance_rank(it.importance) > _importance_rank(
            filters.importance
        ):
            continue
        if filters.item and it.name != filters.item:
            continue
        out.append(it)
    return out


def _has_explicit_identity(it: Item) -> bool:
    """True if the item declares a real identity (name/provides), not just
    the installer-fallback used by `Item.name`."""
    provides = it.spec.get("provides")
    if not isinstance(provides, str):
        provides = None
    return bool(it.spec.get("name") or provides)


def detect_conflicts(items: list[Item]) -> list[Outcome]:
    by_name: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        if not _has_explicit_identity(it):
            continue
        by_name[it.name].append(it)
    outs: list[Outcome] = []
    for name, group in by_name.items():
        versions = [g.version for g in group]
        conflict = any(
            constraints_conflict(versions[i], versions[j])
            for i in range(len(versions))
            for j in range(i + 1, len(versions))
        )
        if conflict:
            # A **list**, not a mapping. The detail used to be keyed on each
            # item's `layer`, and the layer model is gone: there is no field
            # left that distinguishes two items sharing a name. Keying on
            # anything still available — `installer`, `importance` — would
            # collide the moment the two conflicting items agree on it, and a
            # mapping that silently drops one of the two versions is worse than
            # no mapping. The conflicting constraints in declaration order are
            # the whole of what the reader needs to see.
            outs.append(
                Outcome(
                    "fail",
                    f"version conflict for {name}",
                    {"versions": [g.version for g in group]},
                )
            )
    return outs


def run(
    target: Target,
    items: list[Item],
    stage: str,
    filters: Filters,
    on_conflict: str = "fail",
) -> tuple[list[Outcome], str]:
    selected = select(items, filters)
    outs: list[Outcome] = []

    conflicts = detect_conflicts(selected)
    if conflicts and on_conflict == "fail":
        # fail = halt: record the conflict and return before any installer runs,
        # for every stage. install/bootstrap side effects (embed/oneline/claude)
        # are not idempotent, so we must not mutate the host on a fatal conflict.
        outs.extend(conflicts)
        return outs, status_from(outs)

    for it in selected:
        inst = get_installer(it.installer)
        if stage == "check":
            outs.extend(inst.check(it, target))
        elif stage == "dry-run":
            outs.extend(inst.plan(it, target))
        elif stage == "install":
            outs.extend(inst.install(it, target))
        elif stage == "bootstrap":
            outs.extend(inst.install(it, target))
            outs.extend(inst.bootstrap(it, target))
        else:
            raise ValueError(f"unknown stage {stage!r} (expected {STAGES})")

    return outs, status_from(outs)
