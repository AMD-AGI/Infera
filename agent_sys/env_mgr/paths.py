# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The declared names for a zone's own directories, local and remote.

**What this is a fact about.** Every name here answers *"where, inside this
attempt's zone, does X live"* — and that is a fact about the `Zone`, not about
the task, the agent, or the grant that covers it. `engineer_principle.md` §2
asks which part of the existing design a new thing belongs to before it is
written; the honest answer was *none of the three that already export
environment*, and this file is that answer rather than the nearest place:

| where it did not go | why not |
|---|---|
| `fs/layout.py` | owns filesystem **acts** — create, stage, copy. A variable name is not an act, and `layout` sits below `grants` where nothing should learn about environment spelling |
| `grants.py` | owns what the task **declared**. `output_env` and `input_env` are keyed by handoff kind because an author wrote that kind down. Nobody declares a zone's `logs/`; it exists because the zone does |
| `prepare.py` | is the composition. A sixth thing composed there is right; the sixth thing's *definition* living there is how `prepare` became 31 kB |

**Is this a sixth source?** `env_mgr/README.md` counts five contributors to
`Prepared.environment`, and `docs/ui-stage.md` asks the question directly. The
answer is that it is **a sixth call site and not a sixth kind of source**:
`AGENT_SYS_TASK_PACKAGE` was already a zone-path variable, exported by hand at
`prepare.py`'s step 6a, and it folds in here as one member of the family it was
the first of. So the count of *kinds* stays five — derived `PATH`, zone paths,
outputs, inputs, agent material — and the zone-path kind grows from one name to
six. `PACKAGE_ENV_VAR` is defined here and re-exported from `prepare`, because
one fact may not have two writers (§1) and `tests/cli/test_isolation_shown.py`
imports it from there.

## What is exported, and what the user asked for that is not

The requirement (`refine.task_package.define.md` item 3) lists eleven names in
two halves. The ``my_*`` half is here. **The four ``*_root`` names are not, and
that is a measurement rather than an omission.**

``agent_workspace_root``, ``agent_handoff_root`` and ``agent_playground_root``
resolve to registered *domain* roots, which sit outside the zone. Measured
against a real Landlock ruleset built by `isolation.apply` from exactly the
policy `prepare.py` composes, with an in-zone positive control reading
successfully in the same confined child
(`scratch/ui-yaml-2026-08/w2/p13_are_the_root_paths_reachable.py`):

    CONTROL my_agent_workspace       errno=0  OK
    agent_handoff_root (zone tree)   errno=13 EACCES
    agent_workspace_root (domain)    errno=13 EACCES
    agent_playground_root (domain)   errno=13 EACCES
    ctx.store_root                   errno=13 EACCES

All four read cleanly unconfined, so this is a denial and not an absence.
Exporting them would break the rule this module's nearest neighbour states for
`AGENT_SYS_OUTPUT_<KIND>` — *exported and granted agree by construction*,
because "an exported path we did not grant would be the evaporating allow-list
one level up: the body failing on our own instruction". `AGENT_SYS_MY_ZONE` is
the one root in the user's sense that **is** granted, and it is exported in
their place. The conflict is reported, not routed around.

## The remote half

`AGENT_SYS_*_REMOTE` mirrors each local name through `sync.remote_root`, and is
**absent entirely when no mapping covers the zone** — which is every production
configuration today (`cli/environment.py` passes an empty mapping and says
so). Absent rather than empty is `grants.output_paths`' rule: a path we could
not resolve must not arrive looking like one we resolved to nothing.

**The far side is less confined than the near side** (`remote/__init__.py`), so
the granted-and-exported rule above does not transfer: these are values to hand
to `remote.tools`' `env_remote_push` / `env_remote_run`, not paths this process
opens. That surface has no production caller yet either.
"""

from __future__ import annotations

import os

from env_mgr.fs.domain import DomainKind, subdir_for
from env_mgr.fs.layout import LOGS
from env_mgr.fs.zone import Zone

__all__ = [
    "AGENT_ASSETS_ENV_VAR",
    "ADDONS_ROOT_ENV_VAR",
    "HANDOFFS_ENV_VAR",
    "INSTALL_REPORT_ENV_VAR",
    "LOGS_ENV_VAR",
    "PACKAGE_ENV_VAR",
    "PLAYGROUND_ENV_VAR",
    "REMOTE_SUFFIX",
    "WORKSPACE_ENV_VAR",
    "ZONE_ENV_VAR",
    "remote_name",
    "zone_env",
]

#: **The granted root, and the honest answer to the user's four ``*_root``
#: names.** `prepare` grants ``Granted(zone.root, Mode.READ_WRITE)`` — the whole
#: zone, recursively — so this is the one path in the family that is a *root* in
#: the user's sense and is also reachable. A body handed it can compute anything
#: this family forgot; a body handed `agent_handoff_root` gets `EACCES`.
ZONE_ENV_VAR = "AGENT_SYS_MY_ZONE"

#: Where the staged task package went. **The user's ``task_package_root``**, and
#: the name is unchanged from `prepare.py`'s, where it lived alone.
#:
#: Our own namespace deliberately, and **not** a value any package invents for
#: itself: `examples/demo` reads `AGENT_SYS_DEMO_PACKAGE`, which `cli/main.py`
#: sets to the *original* checkout. Under `interfaces.md` §4.16 that path is no
#: longer granted, so a package with its own variable must derive it from this.
PACKAGE_ENV_VAR = "AGENT_SYS_TASK_PACKAGE"

#: **Where this agent's own asset directory went**, inside the staged package —
#: ``<staged package>/<AgentSpec.assets>`` — or absent when the spec has none.
#:
#: **A name defined here and bound elsewhere**, which is a shape no other member
#: of this family has and is worth the line. Every other name is a zone
#: subdirectory, so `zone_env` can compute the value from the `Zone` alone; this
#: one's value is `AgentSpec.assets` resolved against the staged package, and
#: neither of those is a fact about a zone. So `agent_assets.install` binds it and
#: this module owns the spelling — one writer for the name, one writer for the
#: value, and they are different modules because they know different things.
#:
#: **Exported and granted agree**, which is this module's rule for the whole
#: family: the staged package is inside the zone and `prepare` grants
#: ``Granted(zone.root, Mode.READ_WRITE)`` recursively, so unlike the four
#: ``*_root`` names above this one is reachable. It is not derived from
#: ``AGENT_SYS_TASK_PACKAGE`` by the body, because the relative part is the agent
#: spec's and a body has no route to an agent spec.
AGENT_ASSETS_ENV_VAR = "AGENT_SYS_AGENT_ASSETS"

#: ``agent_sys/env_mgr/addons/`` — this repository's own plugins, **outside the zone**.
#:
#: **The one exported path in this module that is not inside the zone, and the
#: only reason it is allowed is that it is granted.** The four ``*_root`` names
#: above are refused precisely because exporting an ungranted path is *"the
#: evaporating allow-list one level up: the body failing on our own
#: instruction"*. So this name is not a counter-example to that rule; it is the
#: rule applied in the other direction — `isolation/policy.py::addon_grants`
#: composes a **read** grant on this directory, `prepare` adds it beside
#: `agent_cli_grants`, and exported-and-granted agree by construction again.
#:
#: **Emitted only when the agent spec declares ``agent_plugins:``**, by the same
#: condition that emits the grant. A run that declares none gets neither, so the
#: two cannot fall out of step by one of them being unconditional.
#:
#: Read-only, and that is a decision: a component is *read* from here and every
#: member of its ``.claude/`` tree is copied into the zone before anything names
#: it (`agent_assets._place_tree` — *place by default*, three named exceptions).
#: If something ever has to run out of this directory, the answer is to copy that
#: component into the zone, not to widen the grant.
ADDONS_ROOT_ENV_VAR = "AGENT_SYS_ADDONS_ROOT"

#: ``<zone>/logs/agent_assets.install.json`` — what the three component levels
#: installed, per outcome, as JSON.
#:
#: **Promised rather than discoverable, and that is load-bearing rather than
#: convenient.** An agent asked to state what capabilities it has would otherwise
#: search for a file nobody named; worse,
#: `examples/env_checker`'s `check_capabilities_genuine` decides whether an
#: ``unavailable`` verdict is honest *by reading this file* — an ``unavailable``
#: beside a clean install report is a FAIL. A validator that cannot find it fails
#: the report for a reason that has nothing to do with the agent.
#:
#: Inside the zone, so it needs no grant of its own: `paths.py`'s rule is
#: satisfied by where it is written rather than by an addition to the policy.
INSTALL_REPORT_ENV_VAR = "AGENT_SYS_INSTALL_REPORT"

#: The user's ``my_agent_workspace`` — ``<zone>/workspace``, what `workspace.cut`
#: clones into.
#:
#: **``MY_`` and not ``MY_AGENT_``.** Every name in this namespace is already the
#: agent's, so the second word would say it twice; what ``MY_`` distinguishes is
#: *this attempt's* from the shared root, which is the distinction the user's
#: list draws between items 2 and 8. The user's list is lowercase and
#: unprefixed — `task_package_root` is already spelled `AGENT_SYS_TASK_PACKAGE`
#: in this tree and they did not object — so it reads as concepts to be mapped
#: onto the existing convention, not as literal spellings. Reported for a ruling.
WORKSPACE_ENV_VAR = "AGENT_SYS_MY_WORKSPACE"

#: The user's ``my_agent_playground`` — ``<zone>/playground``, spec §6.2's
#: unsynced scratch.
PLAYGROUND_ENV_VAR = "AGENT_SYS_MY_PLAYGROUND"

#: ``<zone>/handoffs`` — where `layout.stage_handoffs` puts this attempt's
#: **staged inputs**.
#:
#: **Not the handoff store, and the vocabulary here is genuinely confusing.**
#: `DomainKind.HANDOFF_STORAGE` names the domain that roots the *zone tree*
#: (`fs/domain.py:23`, `layout.create:88`), while the store the artefacts live in
#: is `Context.store_root`, a separate field. `cli/environment.py:465,472` sets
#: them to two different directories. Neither is exportable — both measured
#: `EACCES` — so what this name carries is the third thing, the one inside the
#: zone. `AGENT_SYS_INPUT_<KIND>` remains the way to address a *particular*
#: staged input; this is the directory they share.
HANDOFFS_ENV_VAR = "AGENT_SYS_MY_HANDOFFS"

#: ``<zone>/logs`` — spec §6.1's last row. Not on the user's list and included
#: because it is one of the four directories a zone has and the only one that
#: would otherwise have no name; ``等等`` invited the completion.
LOGS_ENV_VAR = "AGENT_SYS_MY_LOGS"

#: The user's ``_romote``, read as ``_remote``. A suffix rather than a second
#: family, so that a name and its counterpart cannot drift apart.
REMOTE_SUFFIX = "_REMOTE"


def remote_name(local: str) -> str:
    """The far-side spelling of a local variable name. One rule, one place."""
    return f"{local}{REMOTE_SUFFIX}"


#: variable → the zone-relative subdirectory it names. Ordered as spec §5.1's
#: layout diagram lists them, so the exported set reads like the documented one.
#:
#: The subdirectories come from `subdir_for` and `layout.LOGS` rather than from
#: literals: the kind→directory map is `fs/domain.py`'s, and a second copy here
#: would be a second answer to *where does a playground go*.
_SUBDIRS: tuple[tuple[str, str], ...] = (
    (HANDOFFS_ENV_VAR, subdir_for(DomainKind.HANDOFF_STORAGE)),
    (WORKSPACE_ENV_VAR, subdir_for(DomainKind.WORKSPACE)),
    (PLAYGROUND_ENV_VAR, subdir_for(DomainKind.PLAYGROUND)),
    (LOGS_ENV_VAR, LOGS),
)


def zone_env(
    zone: Zone,
    *,
    staged_package: str | None = None,
    remote_zone_root: str | None = None,
) -> dict[str, str]:
    """The declared name for each of this zone's own directories.

    Called at `prepare`'s step 6a, **after** the zone, the workspace and the
    package staging have happened and **before** `material.deploy`, so that an
    agent spec's declared ``env`` still outranks every name here — which is the
    precedence `test_a_declared_env_overrides_every_contributor` pins for the
    five that came before.

    **A directory that does not exist is not exported.** The zone's
    subdirectories are one per *registered domain kind* (`layout._subdirs`), so a
    run with no `PLAYGROUND` domain has no ``<zone>/playground`` — and a variable
    naming it would be an instruction to a body to use a path that is not there.
    `grants.output_paths` settled this shape already: a slot we could not resolve
    is absent rather than present-and-empty, because absence has to be
    unambiguous for the reader that renders the difference. The five stats this
    costs are per prepared task, once.

    `staged_package` is **passed, not computed**, although this module knows
    `layout.PACKAGE`. `stage_package` returns `None` when no package was
    configured, and recomputing the path here would export a package location
    for a run that staged nothing — one fact, two writers, and the second one
    wrong exactly when the first declined to answer.

    `remote_zone_root` comes from `sync.remote_root`, which owns the mapping.
    `None` — the configuration everything ships with today — exports no
    ``_REMOTE`` name at all rather than a half-formed one.
    """
    local: dict[str, str] = {ZONE_ENV_VAR: zone.root}
    for name, subdir in _SUBDIRS:
        path = os.path.join(zone.root, subdir)
        if os.path.isdir(path):
            local[name] = path
    if staged_package is not None:
        local[PACKAGE_ENV_VAR] = staged_package

    if remote_zone_root is None:
        return local

    # The far side mirrors the near side because `sync` rsyncs the zone whole,
    # and it creates ``playground/`` explicitly even though it excludes the
    # contents (`sync.py`, and spec §6.4's *"remote workspace, playground,
    # handoff storage"*). So the remote names are derived from the local ones
    # by relative path — never by a second join against a second subdirectory
    # table, which is how the two halves would come to disagree.
    far = {remote_name(ZONE_ENV_VAR): remote_zone_root}
    for name, path in local.items():
        if name == ZONE_ENV_VAR:
            continue
        rel = os.path.relpath(path, zone.root)
        far[remote_name(name)] = os.path.join(remote_zone_root, rel)
    return {**local, **far}
