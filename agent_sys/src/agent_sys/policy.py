"""Ordering the eligible set — the one scheduling decision in the system.

No graph algorithm is required: the only graph operation anywhere is asking
whether a task's inputs are valid, and that is a query on the handoff.
"""

from typing import Protocol

from agent_sys.ids import TaskId
from agent_sys.models import Task

__all__ = ["SchedulePolicy", "FifoPolicy"]


class SchedulePolicy(Protocol):
    def select(self, eligible: list[Task], snapshot: dict[str, float]) -> list[TaskId]: ...


class FifoPolicy:
    """Expedited first, then submission order."""

    def select(self, eligible: list[Task], snapshot: dict[str, float]) -> list[TaskId]:
        # `not t.expedited` sorts False (expedited) before True. `snapshot` is
        # unused here; it is in the signature because a cost- or fit-aware
        # policy needs it and changing it later would break every implementation.
        return [t.id for t in sorted(eligible, key=lambda t: (not t.expedited, t.created_at))]
