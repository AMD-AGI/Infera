"""Scope and authority — criteria 6, 8 and 10."""

from __future__ import annotations

import pytest

from monitor import EventKind, PusherMonitor, ScopeViolation, event
from task_graph.ids import TaskId
from task_graph.registry import Registry

from .conftest import Status, StubAttempt, StubTask, StubTaskMgr


def test_no_discovery_without_set_task(monitor: PusherMonitor, task_mgr: StubTaskMgr) -> None:
    """Criterion 10: **`set_task` is the only way a monitor learns what it
    watches.** It is told, and it does not go looking.

    Three tasks exist in the manager and the monitor was given one of them; it
    knows about that one. Nothing else in the class writes the watch set — in
    particular `report` does not, so an event for an unwatched task cannot
    smuggle one in.
    """
    a, b, c = (task_mgr.add(StubTask()) for _ in range(3))
    monitor.set_task(b.id)

    assert monitor.watches(b.id)
    assert not monitor.watches(a.id)
    assert not monitor.watches(c.id)

    monitor.report(event(EventKind.VALIDATION_FAILED, c.id))
    assert not monitor.watches(c.id), "report() must not widen the watch set"


def test_global_monitor_refuses_another_task(monitor: PusherMonitor, task_mgr: StubTaskMgr) -> None:
    """Criterion 8: **a global monitor may transition only the task `set_task`
    gave it**, and an attempt to act on another is refused rather than silently
    widened.

    The scope is the **verbs, not the filesystem** (spec §6.1): a monitor lives
    in the unconfined supervisor and no OS mechanism bounds it, so this
    in-process check is the whole of the boundary rather than a defence in depth.
    """
    mine, theirs = task_mgr.add(StubTask()), task_mgr.add(StubTask())
    monitor.set_task(mine.id)

    monitor._current = mine.id
    with pytest.raises(ScopeViolation) as unwatched:
        monitor._transition(theirs.id, "cancel")
    assert str(theirs.id) in str(unwatched.value)
    assert theirs.calls == []

    # And stricter than the criterion asks: one task's scope at a time, so even
    # a *watched* task cannot be transitioned while another is being handled.
    monitor.set_task(theirs.id)
    with pytest.raises(ScopeViolation) as busy:
        monitor._transition(theirs.id, "cancel")
    assert "is handling" in str(busy.value)
    assert theirs.calls == []


def test_the_scope_guard_spans_both_queues(monitor: PusherMonitor, task_mgr: StubTaskMgr) -> None:
    """`_current` is set and cleared by the loop's one guarded path, which is
    **how spec §5.2 rule 5 spans both queues**: whichever queue the work came
    from, one task is in `_current` at a time."""
    task = task_mgr.add(StubTask())
    monitor.set_task(task.id)

    assert monitor._current is None
    with pytest.raises(ScopeViolation):
        monitor._transition(task.id, "cancel")  # nothing is being handled


def test_every_action_is_a_transition_call(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner, registry: Registry
) -> None:
    """Criterion 6: **every monitor action is observable as a transition call**,
    never a status the monitor assigns.

    `task_graph`'s `test_authority.py` proves the same shape for the scheduler
    with a spy that logs every call; the same spy is applied here to the new
    caller. Every write of `Task.status` must have happened *inside* a verb —
    a monitor that assigned one directly would leave a write with no verb on the
    stack, and would be visible even though the write itself succeeded.
    """
    task = task_mgr.add(StubTask(status=Status.INPUT_VALIDATING))
    monitor.set_task(task.id)
    runner.attempts[task.id] = StubAttempt()

    monitor._run_guarded(task.id, monitor._advance, event(EventKind.PHASE_DONE, task.id))

    assert task.calls == [("enter_phase", Status.RUNNING)]
    assert task.status_writes == [True], "a status was written outside a transition verb"
    assert monitor._current is None, "the monitor holds nothing between calls"


def test_a_transition_on_an_unknown_task_does_not_reach_the_manager(
    monitor: PusherMonitor,
) -> None:
    """The guard runs before the lookup, so an id the monitor never held cannot
    even become a `KeyError` from someone else's collection."""
    monitor._current = None
    with pytest.raises(ScopeViolation):
        monitor._transition(TaskId.new(), "cancel")
