"""What leaves `closure/`.

A closure is a spec-level, code-management artefact: a composition of four parts,
a load checker, and read-only query helpers. **Nothing at runtime.** It is
consulted when a graph is assembled and never again.

The temptation this module exists to resist: it is the one place that sees all
four parts, so every cross-object question wants to live here. The rule is that
if a rule can be expressed against one object, it belongs to that object.

Declarations only. See `docs/interfaces.md` §4.5.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Any, Protocol

from spec_loader.protocols import (
    ClosureDoc,
    Problem,
    Registries,
    SpecRegistry,
    TaskSpec,
)

__all__ = [
    "ClosureDoc",
    "ClosureRegistry",
    "TaskSpec",
    "TaskSpecRegistry",
    "agent_of",
    "check_closures",
    "declared_handoffs",
    "named_kinds",
    "permissions_of",
    "phase_validators",
]


# --------------------------------------------------------------------------- #
# Accessors, so key names live in one file and no check reads a raw key.
#
# **`task_of` is no longer one of them.** It reads the closure document's `task`
# key, which three packages each wrote a reader for because the direct edges are
# forbidden, and it is `spec_loader.task_of` now — the same move that unified
# `Body`, `body_of` and `subgraph_of`. `closure` still re-exports the name so an
# in-flight caller keeps working; the re-export goes when nothing imports it.


def declared_handoffs(doc: ClosureDoc) -> tuple[str, ...]: ...


def named_kinds(task: TaskSpec) -> tuple[str, ...]:
    """Every handoff kind the task names, inputs and outputs together, in
    declaration order with duplicates removed."""
    ...


def phase_validators(doc: ClosureDoc) -> tuple[str, ...]: ...


def agent_of(doc: ClosureDoc) -> str:
    """The agent spec name. **Never `None`** — spec §2.2 rev. 8 makes `agent` a
    required key, and check 4 runs unconditionally.

    What varies is the agent's `kind`. A `kind: program` spec is not a degenerate
    case: a large part of the reference workflow is running a command someone
    already wrote, and wrapping *that* in an AI would add cost and
    non-determinism for nothing. Wrapping it in an agent **spec** costs a name
    and a command line, and buys one dispatch path and one answer to "what ran
    this task".
    """
    ...


def permissions_of(task: TaskSpec) -> Mapping[str, Any]: ...


# --------------------------------------------------------------------------- #
# The two registries


class TaskSpecRegistry(SpecRegistry, Protocol):
    """Adds nothing to the base.

    It exists as a separate object because the four spec registries are
    deliberately separate, and because `task_graph.check_graph` takes it alone.
    It has no package of its own — a task spec is not independently loadable, so
    a `task/` package would hold one registry and no other reason to exist.
    """


class ClosureRegistry(SpecRegistry, Protocol):
    """The closure table, plus the reverse index and its six queries.

    **Not a component `Registry`.** That one resolves collaborators late and
    permits replacement; this one is a name table that refuses it.
    """

    def handoff_kinds(self, closure: str) -> tuple[str, ...]:
        """Every kind this closure touches, inputs and outputs."""
        ...

    def validators_for(self, closure: str) -> tuple[str, ...]:
        """Every validator that will run: the phase validators, plus the
        per-handoff ones joined through the handoff registry."""
        ...

    def closures_using_kind(self, kind: str) -> tuple[str, ...]:
        """Reverse. Raises `SpecNotFound` if `kind` is not a known handoff kind;
        returns `()` for a known kind no closure uses.

        **"Not found" and "found, used by nothing" are different answers**, and
        the argument is validated against the catalogue *before* the index is
        touched, so the two never share a code path. dbt has this right in the
        data and loses it at all six call sites, and its user-facing message has
        to hedge across three causes.

        The universe is the loaded catalogue. A task package that is not loaded
        is not in the answer, and unlike Bazel's `rdeps` we cannot get that
        universe *wrong*, only narrow.
        """
        ...

    def closures_using_agent(self, agent: str) -> tuple[str, ...]: ...

    def closures_using_validator(self, name: str) -> tuple[str, ...]:
        """Reverse, for a **phase** validator: which closures name it as one.

        A sixth query, not in the spec's five. It was withdrawn and the
        withdrawal reversed (`docs/interfaces.md` §4.5). The premise that
        justified it — that `users_of` *structurally cannot see* this edge — was
        true when written and stopped being true when `check_closures` wired
        `bind_phase`, which this package did itself.

        What keeps it is a different argument: `users_of` is fed from both sides
        and answers *who names this, and how*, across edge kinds; this answers
        *which closures name it as a phase validator*, typed, within one kind.
        Two questions, not two indexes over one fact.
        """
        ...

    def agent_of(self, closure: str) -> str:
        """The agent spec name. Always present."""
        ...

    def freeze(self) -> None:
        """Build the reverse index and refuse further registration.

        Called once by the composition root, after the closure pass, over the
        closures that passed. **Frozen structurally, not by convention** — Sphinx
        is the argument: an index that can outlive its build is one somebody
        eventually has to purge, and its `clear_doc` obligation falls on every
        owner including third-party extensions.
        """
        ...


# --------------------------------------------------------------------------- #
# The pass


def check_closures(
    regs: Registries,
    handoff_report: Any,
    *,
    skip: AbstractSet[str] = frozenset(),
) -> list[Problem]:
    """The six checks, over every closure, in sorted name order.

    Returns problems; **raises nothing**, because a closure with a typo'd kind
    and a missing agent should report both.

    Runs at the composition root, once, after every package is loaded — not
    inside `load_package`, which runs per package and would fire the pass before
    a second package's specs existed.

    `skip` is the layering gate: a closure whose own spec failed is not checked
    again here, because "your task's handoff kind does not resolve" on top of
    "your schema is broken" is noise. Kubernetes CRD validation does exactly this
    and gives the reason — error messages that are not actionable.

    `handoff_report` is a `handoff.HandoffLoadReport`, typed `Any` here so this
    module imports no other module package.
    """
    ...
