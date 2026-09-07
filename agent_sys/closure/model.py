"""The two document types, and the accessors that keep key names in one file.

A spec is a plain `dict` throughout (main design §4.1), so `TaskSpec` and
`ClosureDoc` are aliases rather than typed models: a second declaration of the
schema's shape is a second source of truth, and a typed object built from a
schema accepts instances the schema rejects.

What the alias does not buy is autocompletion, which is why every key name this
package reads is read here and nowhere else. `check.py` and `query.py` call an
accessor; neither indexes a raw key.

**`Body`, `body_of`, `subgraph_of` and `task_of` are not declared here any
more.** They are `spec_loader`'s: one `$defs.body` in `_common.schema.json` had
grown three Python declarations — this module's, `agent`'s and `validator`'s —
and `subgraph` and `task` each had two readers. They are imported and
re-exported so this file stays the one place a `closure` reader looks for a key
name without being a second writer of one.

`task_of` went last and over this module's objection, recorded because the
ruling is the better of the two. The objection: `body` and `subgraph` are keys
of the *task spec*, sharing one `$defs.body`, while `task` and `agent` are keys
of the *closure document* — this package's subject — so the argument that moved
`Body` did not obviously reach `task_of`, and `task_graph` could have deleted its
copy by resolving `task_specs` by name instead. The deciding point was one the
objection had not weighed: leaving it means the next reader finds `Body` unified
and `task_of` not, **with no visible reason for the difference**. A rule that
holds in three places and not the fourth is not a rule anyone can apply.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from spec_loader import Body, body_of, subgraph_of, task_of
from spec_loader.protocols import ClosureDoc, TaskSpec

__all__ = [
    "Body",  # re-exported from spec_loader
    "ClosureDoc",
    "TaskSpec",
    "agent_of",
    "body_of",
    "declared_handoffs",
    "grants_of",
    "has_subgraph",
    "monitor_of",
    "named_inputs",
    "named_kinds",
    "named_outputs",
    "permissions_of",
    "phase_validators",
    "repos_of",
    "task_of",
]


def _strings(value: Any) -> tuple[str, ...]:
    """Every string in `value`, treating a non-sequence as empty.

    The schema is the enforcement point, so this is not validation — it is what
    lets a checker run over a document that never reached the schema, which is
    the case a unit test exercises and the case a malformed package produces
    when two problems are collected rather than raised.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def declared_handoffs(doc: ClosureDoc) -> tuple[str, ...]:
    """The kinds the closure lists explicitly, in declaration order.

    Derivable from the task's `inputs` and `outputs`, and listed anyway: the
    redundant field is checked *against* the derivable one, which catches the
    mistake that actually happens — adding an input and forgetting to bring its
    kind into the closure.
    """
    return _strings(doc.get("handoffs"))


def named_inputs(task: TaskSpec) -> tuple[str, ...]:
    return _strings(task.get("inputs"))


def named_outputs(task: TaskSpec) -> tuple[str, ...]:
    return _strings(task.get("outputs"))


def named_kinds(task: TaskSpec) -> tuple[str, ...]:
    """Every handoff kind the task names, inputs and outputs together, in
    declaration order with duplicates removed."""
    return tuple(dict.fromkeys(named_inputs(task) + named_outputs(task)))


def phase_validators(doc: ClosureDoc) -> tuple[str, ...]:
    """The validators for this task's two validation phases.

    A property of the task rather than of any one handoff kind, so the handoff
    specs cannot carry them. That is why `check_closures` hands them to
    `validator_specs.bind_phase` — the third of `docs/interfaces.md` §5.4's edge
    kinds, and the one `users_of` had no other way to learn.
    """
    return _strings(doc.get("validators"))


def agent_of(doc: ClosureDoc) -> str:
    """The agent spec name.

    Required, so a well-formed closure always has one and `""` is only reachable
    for a document that never passed the schema. What varies is the agent's
    `kind` — `ai`, `human`, or `program`; a `kind: program` spec is written by
    the package author and admitted by the ordinary registry, because this module
    synthesises nothing.
    """
    agent = doc.get("agent")
    return agent if isinstance(agent, str) else ""


def repos_of(task: TaskSpec) -> tuple[str, ...]:
    """The dependency repositories this task's work needs. Per task, unlike the
    main repository, which is one per run."""
    return _strings(task.get("repos"))


def monitor_of(task: TaskSpec) -> str | None:
    """Which monitor loop watches this task, by name. `None` takes the default."""
    monitor = task.get("monitor")
    return monitor if isinstance(monitor, str) else None


def permissions_of(task: TaskSpec) -> Mapping[str, Any]:
    """What this task's executor may reach. Read for one check and stored
    nowhere — permissions are a versioned *task* attribute."""
    perms = task.get("permissions")
    return perms if isinstance(perms, Mapping) else {}


def grants_of(task: TaskSpec) -> tuple[Mapping[str, Any], ...]:
    """The declared grants, in declaration order."""
    grants = permissions_of(task).get("grants")
    if isinstance(grants, str) or not isinstance(grants, Sequence):
        return ()
    return tuple(g for g in grants if isinstance(g, Mapping))


def has_subgraph(task: TaskSpec) -> bool:
    """Whether this task expands into a subgraph rather than doing the work.

    One line over `spec_loader.subgraph_of`, and it stays a *question* rather
    than becoming a second reader of the key: check 7 wants to know whether the
    task is a non-leaf, not what is in the expansion. **Nothing here reads inside
    an entry** — the entry shape is `task_graph`'s, and `Task.unfold` is what
    instantiates one.
    """
    return bool(subgraph_of(task))
