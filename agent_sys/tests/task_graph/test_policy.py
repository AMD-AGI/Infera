"""Ordering — criterion 10.

Expedite's whole effect on ordering lives here rather than in the scheduler, so
a replacement policy is free to ignore it.
"""

from datetime import datetime, timedelta, timezone

from task_graph.ids import TaskId
from task_graph.models import Task
from task_graph.policy import FifoPolicy, SchedulePolicy

BASE = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)


def task(seconds: int, *, expedited: bool = False) -> Task:
    return Task(
        agent_spec="profiler",
        created_at=BASE + timedelta(seconds=seconds),
        expedited=expedited,
    )


def test_fifo_orders_by_submission_time():
    third, first, second = task(30), task(10), task(20)
    assert FifoPolicy().select([third, first, second], {}) == [first.id, second.id, third.id]


def test_expedited_goes_ahead_of_earlier_tasks():
    early, late_but_expedited = task(10), task(90, expedited=True)
    order = FifoPolicy().select([early, late_but_expedited], {})
    assert order == [late_but_expedited.id, early.id]


def test_expedited_tasks_are_fifo_among_themselves():
    second, first = task(90, expedited=True), task(50, expedited=True)
    plain = task(10)
    order = FifoPolicy().select([plain, second, first], {})
    assert order == [first.id, second.id, plain.id]


def test_an_empty_set_is_empty():
    assert FifoPolicy().select([], {}) == []


def test_the_snapshot_is_accepted_and_ignored():
    """In the signature for a future cost-aware policy; FIFO does not read it."""
    a, b = task(10), task(20)
    assert FifoPolicy().select([a, b], {"gpu": 0.0}) == FifoPolicy().select([a, b], {"gpu": 8.0})


def test_the_input_list_is_not_reordered_in_place():
    tasks = [task(30), task(10)]
    original = list(tasks)
    FifoPolicy().select(tasks, {})
    assert tasks == original


def test_a_replacement_policy_changes_only_the_order():
    """The seam the mission asks for: the algorithm is decoupled."""

    class LifoPolicy:
        def select(self, eligible, snapshot):
            return [t.id for t in sorted(eligible, key=lambda t: t.created_at, reverse=True)]

    tasks = [task(10), task(20), task(30)]
    assert LifoPolicy().select(tasks, {}) == list(reversed(FifoPolicy().select(tasks, {})))


def test_the_protocol_is_structural_and_not_runtime_checkable():
    """Unlike `Resumable`, nothing does an isinstance against this one, so it
    stays a plain Protocol — a policy is supplied, never discovered."""
    import pytest

    with pytest.raises(TypeError):
        isinstance(FifoPolicy(), SchedulePolicy)
    assert callable(SchedulePolicy.select)


def test_select_returns_ids_not_tasks():
    """The scheduler re-reads each task and re-checks its status; handing back
    objects would invite dispatching from a stale one."""
    result = FifoPolicy().select([task(10)], {})
    assert all(isinstance(x, TaskId) for x in result)
