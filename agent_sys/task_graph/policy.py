"""Ordering the eligible set — the one scheduling decision in the system.

No graph algorithm is required: the only graph operation anywhere is asking
whether a task's inputs are valid, and that is a query on the handoff.
"""

from typing import Protocol

from task_graph.ids import TaskId
from task_graph.models import Task

__all__ = ["SchedulePolicy", "FifoPolicy", "DepthFirstPolicy"]


class SchedulePolicy(Protocol):
    def select(self, eligible: list[Task], snapshot: dict[str, float]) -> list[TaskId]: ...


class FifoPolicy:
    """Expedited first, then submission order."""

    def select(self, eligible: list[Task], snapshot: dict[str, float]) -> list[TaskId]:
        # `not t.expedited` sorts False (expedited) before True. `snapshot` is
        # unused here; it is in the signature because a cost- or fit-aware
        # policy needs it and changing it later would break every implementation.
        return [t.id for t in sorted(eligible, key=lambda t: (not t.expedited, t.created_at))]


class DepthFirstPolicy:
    """Stack-like: run the subgraph on top of the stack as far down as it goes.

    **The order comes from the pool, not from a key computed over the task.**
    `eligible` arrives in promotion order, so depth-first is that order
    reversed — most recently promoted first. That works because of *where*
    promotion happens: a chain `P1 -> P2 -> P3` enters `WAITING_HANDOFF` at
    unfold, and each link is promoted only when its predecessor's output becomes
    valid, so the frontier that most recently advanced is the deepest one.

    It reads no field on `Task` except `expedited` — not `parent`, not
    `is_start`, not `is_end` — which is why criterion 42's blanking check passes
    trivially and criterion 43 is satisfied at the same time. Two keys computed
    over `Task` were designed first and both are rejected: sorting by subgraph
    membership satisfies 43 and fails 42, and LIFO on `created_at` abandons a
    subgraph mid-way as soon as an unrelated task is submitted while it runs.
    """

    def select(self, eligible: list[Task], snapshot: dict[str, float]) -> list[TaskId]:
        ordered = list(reversed(eligible))
        return [t.id for t in sorted(ordered, key=lambda t: not t.expedited)]
