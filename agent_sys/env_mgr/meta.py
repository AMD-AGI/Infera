# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`env_mgr`'s own configuration. Spec §3.1.

Domain registrations, local↔remote mappings, and sync strength. Populated three
ways in decreasing order of preference: **auto-detected** where discoverable,
**produced by a designated task** and arriving as a knowledge handoff, and
**declared** for the rest.

The default granted system set lives here too, because spec §4.5.1 requires it
to be *a default in configuration, not a constant in code* — it is a policy
decision with a security cost and a site may need to narrow or widen it.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import NamedTuple

from env_mgr.fs.domain import DomainKind, DomainRegistry
from env_mgr.isolation.policy import DEFAULT_SYSTEM_SET, Granted, Mode

__all__ = [
    "CONVENTIONS",
    "Meta",
    "RemoteMapping",
    "Strength",
    "configured_path",
    "from_knowledge",
    "load",
    "save",
]

#: The file a cluster-conventions **knowledge handoff** carries. Spec §7: how to
#: detect and reach a remote is knowledge produced by a designated task and
#: delivered as a handoff, not configuration hard-coded here — so `env_mgr` is
#: itself a consumer of knowledge handoffs, reading the same conventions an
#: agent would.
CONVENTIONS = "conventions.json"


class Strength(str, Enum):
    """Spec §5.2. Recorded per mapping, not inferred at use time."""

    STRONG = "strong"  # the same bytes: one NFS or one mount
    WEAK = "weak"  # two copies, synchronised explicitly by rsync


class RemoteMapping(NamedTuple):
    local_root: str
    remote_root: str
    strength: Strength = Strength.WEAK
    transport: str = "ssh"
    target: str = ""  # host for ssh, container for docker exec


class Meta(NamedTuple):
    domains: tuple[tuple[str, str, str], ...] = ()  # name, root, kind
    mappings: tuple[RemoteMapping, ...] = ()
    system_set: tuple[Granted, ...] = DEFAULT_SYSTEM_SET
    #: Far-side roots this site accepts as **destroyable**. `sync` runs
    #: ``rsync --delete``, so a mapping decides what gets deleted — on a remote
    #: machine, from a file somebody edits. `sync.check_delete_scope` refuses any
    #: weak mapping whose `remote_root` is not under one of these.
    #:
    #: **Empty by default, and empty refuses.** An allow-list rather than a
    #: deny-list of `/`, `/usr`, `/home`: the next dangerous root is always the
    #: one nobody thought to add. Stating it here makes the acceptance explicit,
    #: reviewable, and attached to the mapping it authorises.
    deletable_roots: tuple[str, ...] = ()

    def registry(self) -> DomainRegistry:
        """Register every declared domain. Idempotent, so this is also reload."""
        reg = DomainRegistry()
        for name, root, kind in self.domains:
            reg.register(name, root, DomainKind(kind))
        return reg

    def weak(self) -> tuple[RemoteMapping, ...]:
        """The mappings with something to copy. A **strong** one is the same
        bytes on both sides — one mount — so there is nothing to synchronise.

        The filter lives here and not in each caller: `mapping_roots` and the
        transports built beside it must select the *same* mappings, and two
        comprehensions each carrying `if m.strength is WEAK` is how they would
        one day stop doing so.
        """
        return tuple(m for m in self.mappings if m.strength is Strength.WEAK)

    def mapping_roots(self) -> dict[str, str]:
        """The local→remote root map `sync` takes."""
        return {m.local_root: m.remote_root for m in self.weak()}


def configured_path(explicit: str | None = None) -> str:
    """Where this site's meta file is: an explicit path, then ``$ENV_MGR_META``,
    then ``~/.config/env_mgr/meta.json``.

    The order is not new — `inspection` has resolved it this way since the
    `domain` and `zone` sub-commands shipped. It moves here because the **run**
    path now reads the same file (`cli/main.py`), and an order resolved in two
    modules is one fact with two writers (§1). `inspection` calls this rather
    than keeping its own copy.

    Resolved per call, not at import: a caller that sets ``$ENV_MGR_META`` or
    ``$XDG_CONFIG_HOME`` after this module is imported — every test, and the
    R0 run — would otherwise get the value the interpreter started with.
    """
    return (
        explicit
        or os.environ.get("ENV_MGR_META")
        or os.path.join(
            os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
            "env_mgr",
            "meta.json",
        )
    )


def load(path: str) -> Meta:
    if not os.path.exists(path):
        return Meta()
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return Meta(
        domains=tuple(tuple(d) for d in raw.get("domains", ())),  # type: ignore[misc]
        mappings=tuple(
            RemoteMapping(
                local_root=m["local_root"],
                remote_root=m["remote_root"],
                strength=Strength(m.get("strength", "weak")),
                transport=m.get("transport", "ssh"),
                target=m.get("target", ""),
            )
            for m in raw.get("mappings", ())
        ),
        system_set=tuple(
            Granted(g["path"], Mode[g["mode"]], g.get("optional", False))
            for g in raw.get("system_set", ())
        )
        or DEFAULT_SYSTEM_SET,
        deletable_roots=tuple(raw.get("deletable_roots", ())),
    )


def from_knowledge(store_root: str, handoff_id: object, version: int) -> Meta:
    """Read cluster conventions out of a knowledge handoff. Criterion 21.

    Deliberately the *same* reader as `load`: a knowledge handoff is an ordinary
    versioned artefact at ``<store_root>/<hid>/v<N>/``, and pointing the reader
    at one is the whole mechanism. Changing the handoff changes this module's
    behaviour with no code change, which is what the criterion asks for.

    **Half the criterion, and the half that is this module's.** The designated
    system-level task that *produces* such a handoff is unspecified — spec §11
    concedes it — so nothing here can be tested end to end against a real one.
    What is tested is that the consumption route exists and is version-selected.
    """
    from env_mgr.fs.layout import handoff_version_dir

    return load(os.path.join(handoff_version_dir(store_root, handoff_id, version), CONVENTIONS))


def save(meta: Meta, path: str) -> None:
    payload = {
        "domains": [list(d) for d in meta.domains],
        "mappings": [
            {
                "local_root": m.local_root,
                "remote_root": m.remote_root,
                "strength": m.strength.value,
                "transport": m.transport,
                "target": m.target,
            }
            for m in meta.mappings
        ],
        "system_set": [
            {"path": g.path, "mode": g.mode.name, "optional": g.optional} for g in meta.system_set
        ],
        "deletable_roots": list(meta.deletable_roots),
    }
    os.makedirs(os.path.dirname(path) or os.curdir, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
