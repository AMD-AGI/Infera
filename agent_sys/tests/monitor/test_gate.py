"""The completeness gate's seam — criteria 4 and 18, this module's half.

**The gate itself is `agent`'s.** Spec §4.1.0 puts it in the runner, which
already holds the task, the handoff store and the `AgentResult`; design §8 draws
the line. What is testable here is everything on this side of it:

| | Owner |
|---|---|
| Running the four checks between the main phase and `OUTPUT_VALIDATING` | `agent` |
| The four `EventKind` values the checks produce | **here** |
| `report()`, and the record reaching disk before the buffer | **here** |
| Deciding what happens after a failure | **here** |
| Reporting a phase that finished *normally* — the same call, a different `kind` | `agent` |

So `StubGate` below is the contract the runner has to satisfy, written out: it
reports and it takes no corrective action of its own. It cannot prove the real
runner behaves that way; it pins the shape the real runner is written against,
and criterion 4's "a task does not enter `OUTPUT_VALIDATING` until the gate
passes" is `agent`'s to demonstrate.
"""

from __future__ import annotations

import dataclasses

import pytest

from monitor import EventKind, PusherMonitor, event
from monitor.pusher import GATE_KINDS
from task_graph.registry import Registry

from .conftest import Status, StubAttempt, StubTask, StubTaskMgr

#: The four independent failures of spec §4.1.0. Three are delivery failures;
#: the fourth is not a property of the delivery at all, but it is a gate failure
#: on the same path and it is what bounds the loop.
FOUR_FAILURES = (
    EventKind.OUTPUT_ABSENT,
    EventKind.OUTPUT_NOT_EXECUTABLE,
    EventKind.SELF_CHECK_UNSET,
    EventKind.BUDGET_EXCEEDED,
)


class StubGate:
    """The runner's completeness gate, reduced to its obligations to this module.

    **The runner does not push.** It reports, and the monitor decides — a runner
    that retried on its own would be a second failure policy, invisible to the
    record and unreachable by the analysing dispatcher that replaces the pusher
    later. `report()` is not an aside on the retry path; it *is* the retry path.
    """

    def __init__(self, monitor: PusherMonitor, task: StubTask) -> None:
        self.monitor = monitor
        self.task = task
        self.corrective_actions: list[str] = []  # must stay empty

    def run(self, failure: EventKind | None) -> bool:
        """One trip through the gate. Returns whether output validation may
        begin. **The runner does not branch on which monitor call to make** — it
        makes one, with a different `kind`."""
        kind = failure or EventKind.PHASE_DONE
        self.monitor.report(event(kind, self.task.id, attempt=0, reported_by="runner"))
        return failure is None


def test_four_failures_each_report(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, registry: Registry
) -> None:
    """Criterion 4: **four things fail the gate independently**, and each is a
    distinct `kind` the monitor receives.

    `OUTPUT_MALFORMED` is absent and its absence is measured, not overlooked:
    `put` raises before anything is created, inside the producing agent's zone,
    so a malformed handoff never reaches storage and the gate sees only an
    absence.
    """
    assert set(FOUR_FAILURES) - {EventKind.BUDGET_EXCEEDED} == GATE_KINDS

    for failure in FOUR_FAILURES:
        task = task_mgr.add(StubTask(status=Status.RUNNING))
        monitor.set_task(task.id)
        gate = StubGate(monitor, task)

        assert gate.run(failure) is False

        kinds = [
            r["kind"]
            for r in registry.get("store_mgr").read_all("event")
            if r["task_id"] == str(task.id)
        ]
        assert kinds == [failure.value], "the gate failure was not recorded as its own kind"
        assert len(monitor._planned) == 0, "a gate failure reached the planned queue"


def test_a_passing_gate_is_a_planned_event(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """The same call, at the same place: a gate that passes is a `PHASE_DONE`.

    The reporter never classifies what it is reporting — `report` routes on
    `kind in PLANNED` — which is exactly what lets the gate call one method
    whether it passed or failed.
    """
    task = task_mgr.add(StubTask(status=Status.RUNNING))
    monitor.set_task(task.id)
    runner.attempts[task.id] = StubAttempt()

    assert StubGate(monitor, task).run(None) is True
    assert len(monitor._planned) == 1
    assert len(monitor._buffer) == 0


def test_no_status_move_during_cycle(monitor: PusherMonitor, task_mgr: StubTaskMgr, runner) -> None:
    """Criterion 4: **the cycle stays below the scheduler.** Repeating it does not
    move task status, does not reach the scheduler, and does not call `on_done`.

    A task cycling through the gate several times is behaving normally, and the
    graph sees one task `RUNNING` throughout. `ForbiddenScheduler` is registered,
    so "does not reach the scheduler" is an assertion.
    """
    task = task_mgr.add(StubTask(status=Status.RUNNING))
    monitor.set_task(task.id)
    runner.attempts[task.id] = StubAttempt(None)
    gate = StubGate(monitor, task)

    for _ in range(5):
        gate.run(EventKind.OUTPUT_ABSENT)
        unit = monitor._buffer.get(0.05)
        assert unit is not None
        monitor._run_guarded(task.id, monitor._handle, unit, release=True)

    assert task.status is Status.RUNNING
    assert task.calls == [], "the gate cycle moved the task"
    assert task.status_writes == []


def test_runner_takes_no_corrective_action(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """Criterion 18: **no recovery action originates outside a monitor
    decision.**

    A module that retried internally would satisfy every other criterion here and
    still be wrong. The test for each reporter is the same: the failing path
    calls `report()` and takes no corrective action of its own — the push, when
    there is one, comes back out of the monitor.
    """
    task = task_mgr.add(StubTask(status=Status.RUNNING))
    monitor.set_task(task.id)
    backend = StubAttempt()
    runner.attempts[task.id] = backend
    gate = StubGate(monitor, task)

    gate.run(EventKind.SELF_CHECK_UNSET)
    assert gate.corrective_actions == []
    assert runner.pushed == []
    assert backend.executor.instructions == [], "the runner pushed"

    unit = monitor._buffer.get(0.05)
    assert unit is not None
    monitor._run_guarded(task.id, monitor._handle, unit, release=True)

    assert backend.executor.instructions == ["continue, do it until finished"]
    assert gate.corrective_actions == [], "the correction came from the monitor, not the gate"


def test_the_monitor_exposes_no_second_inbound_call() -> None:
    """Rev. 14 widened what arrives and **did not widen this** — which is the
    whole reason the routing is internal. A reporter has `report` and `set_task`
    and no third door to choose between."""
    from monitor import Monitor

    inbound = {"report", "set_task"}
    lifecycle = {"mainloop", "stop"}
    declared = {n for n in vars(Monitor) if not n.startswith("_")}
    assert declared - lifecycle == inbound


def test_budget_is_declared_here_and_read_by_the_runner() -> None:
    """`Budget` lives in this package because `BUDGET_EXCEEDED` does, and a
    threshold with no matching kind is unreportable. **It is one global value**:
    nobody yet knows what a normal task costs, so a per-task limit would be
    authored out of numbers no one has."""
    from monitor import Budget

    default = Budget()
    assert (default.max_tokens, default.max_seconds, default.max_turns) == (None, None, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        Budget(max_tokens=1).max_tokens = 2  # frozen: one value, not a mutable setting
