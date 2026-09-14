"""Turning a closure into the root `Task`. **In the wrong package, knowingly.**

`docs/interfaces.md` §5.3 and `closure` D5: nobody owns this. `closure` declined
it and gave the right reason — a helper returning a `Task` would make `closure`
import `task_graph`, and would make the module that reads four other modules'
objects the thing that decides a task's initial permissions. The only two named
callers are this module's `show` and `--dry-run`, so the choice was to build it
or to have no verbs.

It moves to the whole-system CLI (`docs/TODO.md` item 5) **unchanged**: whoever
builds that imports these functions and deletes this file. Recorded so the move
is a relocation rather than a rediscovery.

Three public functions, and one body, because the ordering is the design.

A fourth lived here briefly and does not any more. `register_agent_specs`
bridged `agent_specs` into `task_graph`'s `AgentMgr`, because nothing did and
`scheduler.submit` raised `unknown agent spec` on every task (F-D2). Reported,
and `task_graph` moved it into `build_registry` with the argument that settles
the general case: **is this fact about *this graph*, or about the catalogue?**
A registered spec is registry state, so it belongs to the root; a `Task` is
graph state, so it belongs here. That question is the one to ask of anything
else this file accumulates.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from spec_loader import task_of
from task_graph import Access, Grant, HandoffId, Permissions, Task
from task_graph.models import agent_spec_for

__all__ = ["handoff_ids", "root_task", "wire"]

log = logging.getLogger(__name__)


def root_task(closure_name: str, registry: Any) -> Task:
    """The root `Task` for a closure. **Builds only the root.**

    `Task.unfold()` instantiates the subgraph from `self.closure`, with
    `parent = self.id` and the `is_start` / `is_end` marks the closure declares
    (`task_graph` design §8.5). So this sets `closure` on one `Task` and stops.

    That is worth stating because it bounds the unowned job: it is not *assemble
    the graph*, it is *make the first node*. Everything after it is
    `task_graph`'s, already designed, and already the subject of its criteria
    36–54.

    `parent` is left `None`, which is criterion 2's first half and is the
    default rather than an assignment — a root is a task nothing expanded from.
    """
    closures = registry.get("closures")
    if closure_name not in closures:
        raise KeyError(
            f"{closure_name!r} is not a declared closure; known: {sorted(closures.names())}"
        )
    doc = closures.get(closure_name)
    spec = task_of(doc)
    ids = handoff_ids(closure_name, registry)

    inputs = [ids[kind] for kind in spec.get("inputs") or ()]
    outputs = [ids[kind] for kind in spec.get("outputs") or ()]
    named = tuple(spec.get("inputs") or ()) + tuple(spec.get("outputs") or ())
    kinds = {ids[kind]: kind for kind in named}

    return Task(
        # **`task_graph`'s own function, not a copy of its rule.** This read
        # `doc["agent"]` and broke the moment a non-leaf stopped declaring one;
        # the fix was written here first, as a duplicate, and reported as a
        # defect — one invariant with two writers. `task_graph` promoted the
        # original to public in `b6249e6` and the duplicate is gone. The rule
        # itself stays theirs: a declared agent wins even on a non-leaf, a
        # non-leaf with none gets `SUBGRAPH_AGENT_SPEC`, and **a leaf with none
        # keeps its `KeyError`** — papering that over would dispatch a broken
        # catalogue under a name describing something it is not.
        agent_spec=agent_spec_for(doc, spec),
        closure=closure_name,
        inputs=inputs,
        outputs=outputs,
        kinds=kinds,
        resources=dict(spec.get("resources") or {}),
        permissions=_permissions(spec),
        monitor_spec=spec.get("monitor"),
    )


def handoff_ids(closure_name: str, registry: Any) -> dict[str, HandoffId]:
    """Kind name -> a fresh `HandoffId`, one per kind the closure names.

    The map the graph is wired with, and the map `env_mgr` resolves grants
    against (`closure` design D2): a `Grant` carries a **kind name**, because a
    grant is written at declaration time where no instance exists, and
    `env_mgr.grants.resolve` needs this to turn one into `<root>/<hid>/v<N>/`.

    Every kind the closure declares gets an id, not only the ones the root task
    names — a closure may declare a kind its subgraph uses internally, and
    `Task._instantiate` mints its own ids for those. This map is what a caller
    that needs to *name* one before the unfold uses.
    """
    closures = registry.get("closures")
    doc = closures.get(closure_name)
    spec = task_of(doc)
    declared = (
        tuple(doc.get("handoffs") or ())
        + tuple(spec.get("inputs") or ())
        + tuple(spec.get("outputs") or ())
    )
    return {kind: HandoffId.new() for kind in dict.fromkeys(declared)}


def wire(tasks: Sequence[Task]) -> None:
    """Fill `depends_on` from the handoff wiring, in place.

    **It exists because `depends_on` is `list[TaskId]` — runtime ids — so no
    spec document can carry one.** `scheduler._warn_depends_on` logs on every
    dispatch whose `depends_on` omits the producer of one of its inputs, and it
    is a warning by design there: *rejecting would make declaration order
    matter*. So if the builder does not derive it, the reference example of this
    system prints a warning on every run, which is precisely the kind of
    accepted noise a demo exists to prevent (`materials/08-demo.md` §5).

    Derived from the only source that has the information: the input and output
    sets of the tasks that exist at this moment. **Idempotent** — `Task.unfold`
    already derives it for a subgraph it instantiates in one pass, and a root
    plus a later `replace_with` does not, so this runs over whatever is in hand
    and adds nothing twice.
    """
    producer: dict[HandoffId, Any] = {}
    for task in tasks:
        for hid in task.outputs:
            producer.setdefault(hid, task.id)
    for task in tasks:
        missing = [
            producer[hid]
            for hid in task.inputs
            if hid in producer and producer[hid] != task.id and producer[hid] not in task.depends_on
        ]
        if missing:
            # Assignment rather than `.append`: `Model` sets
            # `validate_assignment=True`, so this is the checked path, and
            # mutating the list in place is the one that is not.
            task.depends_on = [*task.depends_on, *dict.fromkeys(missing)]


def _permissions(spec: Mapping[str, Any]) -> Permissions:
    """The declared grants as the type `Task` carries.

    `closure.check.covers` has already checked at load that these cover every
    kind the task names, over the same relation `Permissions.covers` implements
    — one relation with two bodies, guarded by
    `tests/interfaces/test_covers_agreement.py`. Nothing is re-checked here.
    """
    declared = (spec.get("permissions") or {}).get("grants") or ()
    return Permissions(
        grants=tuple(
            Grant(
                path=str(grant.get("path") or ""),
                access=Access(grant.get("access") or "read"),
                kind=grant.get("kind"),
            )
            for grant in declared
        )
    )
