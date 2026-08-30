# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Domains: a registered region with a name, a root, and a kind. Design §3.3."""

from __future__ import annotations

import os
from collections.abc import Iterator

from env_mgr.fs.path import resolve_strict

# `Domain` and `DomainKind` are inert data and are declared once, in
# `protocols.py`. Defining them again here would give one fact two writers.
from env_mgr.protocols import Domain, DomainKind

__all__ = ["Domain", "DomainKind", "DomainRegistry", "subdir_for"]


#: The kind decides the layout, and only that — design §8.2. ``logs/`` is not a
#: domain: it is created with the zone and granted read-write like anything else
#: in it, which is what spec §6.1's last row asks for.
_SUBDIR = {
    DomainKind.HANDOFF_STORAGE: "handoffs",
    DomainKind.WORKSPACE: "workspace",
    DomainKind.PLAYGROUND: "playground",
}


def subdir_for(kind: DomainKind) -> str:
    """The per-task subdirectory this kind occupies inside a zone."""
    return _SUBDIR[kind]


class DomainRegistry:
    """Idempotent registration, and the one question the layout asks it.

    A getter over an internal dict would make every caller re-derive "which
    domain roots the task tree"; `storage_root` is that question answered here.
    """

    def __init__(self) -> None:
        self._domains: dict[str, Domain] = {}

    def register(self, name: str, root: str, kind: DomainKind) -> Domain:
        """Idempotent.

        Re-registering an existing name with the same root and kind returns the
        existing `Domain` and touches nothing on disk — which is what lets a
        playground survive a restart (spec §6.2). A different root or kind for
        a live name is an error, not an update.
        """
        os.makedirs(root, exist_ok=True)
        resolved = resolve_strict(root)
        if resolved is None:
            raise ValueError(f"domain {name!r}: root {root!r} does not resolve")
        resolved = resolved.rstrip(os.sep) or os.sep
        existing = self._domains.get(name)
        if existing is not None:
            if existing.root != resolved or existing.kind is not kind:
                raise ValueError(
                    f"domain {name!r} is already registered as "
                    f"{existing.kind.value} at {existing.root}; "
                    f"refusing to change it to {kind.value} at {resolved}"
                )
            return existing
        domain = Domain(name, resolved, kind)
        self._domains[name] = domain
        return domain

    def get(self, name: str) -> Domain:
        """Names the candidates on a miss, following `env_mgr/registry.py`."""
        try:
            return self._domains[name]
        except KeyError as e:
            raise KeyError(f"unknown domain {name!r} (have {sorted(self._domains)})") from e

    def storage_root(self) -> str:
        """The root the nested task tree hangs from.

        Exactly one `HANDOFF_STORAGE` domain, because two would be two answers
        to *where does a zone go* and the layout may only have one.
        """
        roots = [d for d in self._domains.values() if d.kind is DomainKind.HANDOFF_STORAGE]
        if len(roots) != 1:
            raise ValueError(
                f"the nested task tree needs exactly one {DomainKind.HANDOFF_STORAGE.value} "
                f"domain; {len(roots)} are registered ({sorted(d.name for d in roots)})"
            )
        return roots[0].root

    def kinds(self) -> tuple[DomainKind, ...]:
        """Which kinds are registered, in a stable order. The layout builds one
        subdirectory per kind and asks nothing else about the collection."""
        present = {d.kind for d in self._domains.values()}
        return tuple(k for k in DomainKind if k in present)

    def __iter__(self) -> Iterator[Domain]:
        return iter(tuple(self._domains.values()))

    def __contains__(self, name: object) -> bool:
        return name in self._domains

    def __len__(self) -> int:
        return len(self._domains)
