"""`TaskSpecRegistry` — the fourth spec registry, homed with the document that
declares its contents."""

from __future__ import annotations

from typing import ClassVar

from spec_loader import BaseSpecRegistry

__all__ = ["TaskSpecRegistry"]


class TaskSpecRegistry(BaseSpecRegistry):
    """Adds nothing to the base.

    It exists as a separate object because the four spec registries are
    deliberately separate, and because `task_graph.check_graph` takes it alone.
    It has no package of its own — a task spec is not independently loadable, so
    a `task/` package would hold one registry and no other reason to exist.

    **It shares a key space with `ClosureRegistry`, and that is a decision.** A
    task spec is nested inside the closure and carries no `name`; the closure's
    name is the workflow step, and its task *is* that step. A separate name would
    be a second thing to keep unique, to typo, and for `Task.closure` and
    `check_graph` to disagree about.

    The consequence is that one duplicate closure would otherwise report twice,
    under two kinds. So `add` is called only from the closure admission path,
    never from a discovery pass of its own, and the duplicate is reported once —
    by `ClosureRegistry`, because that is the file the author wrote.
    """

    kind: ClassVar[str] = "task"
