"""Threads — criteria 21 and 22, and the boundary is sharp.

**The thread belongs to `agent`.** Design §12 steps 9–11 put `TaskAttempt`,
`resume` and `attempt_of` in that package, and says plainly that criteria 21 and
22 are not testable without them. What this module owns, and what these tests
pin, is the half that decides *which* of the two shapes a task is and *what the
monitor asks for*:

- a leaf has a live attempt, so the monitor **wakes** it and takes no new thread;
- a non-leaf has none, so the monitor asks the runner to **resume** it;
- and the monitor never starts anything, which is what keeps thread count at the
  executing leaves.

`StubRunner` stands in for the real one. It cannot prove that a leaf holds
exactly one OS thread from dispatch to `on_done` — that is `agent`'s test — but
it does prove the monitor never asks for a second one.
"""

from __future__ import annotations

from monitor import EventKind, PusherMonitor, event

from .conftest import Status, StubAttempt, StubTask, StubTaskMgr


def test_leaf_holds_one_thread(monitor: PusherMonitor, task_mgr: StubTaskMgr, runner) -> None:
    """Criterion 21, this module's half: **a leaf's three phases borrow one
    thread in turn**, so the monitor wakes the attempt it already has and never
    asks for another.

    The attempt object is the same across both advances — one `Execution`, one
    thread, three phases.
    """
    leaf = task_mgr.add(StubTask(status=Status.INPUT_VALIDATING))
    monitor.set_task(leaf.id)
    attempt = StubAttempt()
    runner.attempts[leaf.id] = attempt

    for _ in range(2):  # INPUT_VALIDATING -> RUNNING -> OUTPUT_VALIDATING
        monitor._run_guarded(leaf.id, monitor._advance, event(EventKind.PHASE_DONE, leaf.id))

    assert leaf.calls == [
        ("enter_phase", Status.RUNNING),
        ("enter_phase", Status.OUTPUT_VALIDATING),
    ]
    assert attempt.woken == 2, "the same parked thread, woken twice"
    assert runner.resumed == [], "no second thread was ever asked for"
    assert leaf.executions == 0


def test_non_leaf_holds_no_thread(monitor: PusherMonitor, task_mgr: StubTaskMgr, runner) -> None:
    """Criterion 22, this module's half: **a non-leaf holds no thread while its
    subgraph runs.**

    Its thread ended at `unfold`; a new one is taken for output validation. An
    attempt that waited on a condition through its subgraph would hold one thread
    per ancestor, which is the cost this shape exists to avoid.
    """
    parent = task_mgr.add(StubTask(status=Status.RUNNING))
    monitor.set_task(parent.id)
    released = runner.make_released_attempt(parent.id)
    assert runner.attempt_of(parent.id) is released and not released.is_running

    monitor._run_guarded(
        parent.id,
        monitor._advance,
        event(EventKind.SUBGRAPH_DONE, parent.id, attributes={"from_task": "a-child"}),
    )

    assert runner.resumed == [parent.id]
    assert released.woken == 0
    assert parent.calls == [("enter_phase", Status.OUTPUT_VALIDATING)]


def test_the_monitor_never_starts_a_task(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """`resume` is not `start`: no new `Execution`, no new agent, the same
    attempt. The monitor has no verb that begins one."""
    assert not hasattr(monitor, "start")
    parent = task_mgr.add(StubTask(status=Status.RUNNING))
    monitor.set_task(parent.id)
    runner.make_released_attempt(parent.id)

    monitor._run_guarded(
        parent.id,
        monitor._advance,
        event(EventKind.SUBGRAPH_DONE, parent.id, attributes={"from_task": "x"}),
    )

    assert parent.executions == 0
    assert runner.resumed == [parent.id]


def test_a_released_attempt_is_resumed_not_woken(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """The regression test for the defect `p7_nonleaf_wake_is_silent.py` found.

    `_advance` used to branch on `attempt_of(tid) is None`, which **never fires
    for a non-leaf**: the attempt object survives its thread — `monitor` spec
    §5.3 says so, `agent`'s `release()` docstring says so, and
    `Runner._attempts` is emptied only by `stop`. So the monitor called `wake()`,
    which is `Event.set()` on an Event no thread is waiting on, and the parent
    sat in `OUTPUT_VALIDATING` for ever **with nothing reported**.

    The branch is now `attempt is None or not attempt.is_running`, and the
    predicate is the attempt's own because only it knows that all three of its
    terms matter — `halt()` sets `_halted` before the thread notices.

    **This test could not have caught the defect before the stub was fixed**,
    and that is the lesson worth keeping: the old `StubAttempt` had no
    `is_running` and modelled a non-leaf as an absent entry, so it agreed with
    the design rather than with the neighbour.
    """
    parent = task_mgr.add(StubTask(status=Status.RUNNING))
    monitor.set_task(parent.id)
    released = runner.make_released_attempt(parent.id)

    monitor._run_guarded(
        parent.id,
        monitor._advance,
        event(EventKind.SUBGRAPH_DONE, parent.id, attributes={"from_task": "x"}),
    )

    assert released.woken == 0, "wake() on a dead thread is the silent stall"
    assert runner.resumed == [parent.id], "the parent must be given a thread"
    assert released.is_running, "and resume() is what makes it live again"


def test_a_parked_leaf_is_woken_not_resumed(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """The other half, and why one call cannot serve both: `resume` is
    `begin()`, so using it on a parked leaf would start a **second** thread on
    one attempt while the first sits in `_await_wake`."""
    leaf = task_mgr.add(StubTask(status=Status.INPUT_VALIDATING))
    monitor.set_task(leaf.id)
    parked = StubAttempt()
    runner.attempts[leaf.id] = parked
    assert parked.is_running

    monitor._run_guarded(leaf.id, monitor._advance, event(EventKind.PHASE_DONE, leaf.id))

    assert parked.woken == 1
    assert runner.resumed == [], "a parked leaf must not be given a second thread"


def test_an_attempt_that_stop_removed_is_resumed(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """`attempt is None` is kept in the branch, for the case `Runner.stop` ran
    first — `stop` is the only thing that empties the map. The real `resume`
    then raises `KeyError`, which is correct and stays: a missing attempt at
    re-entry means the parent is stuck, and a no-op would make that silent."""
    parent = task_mgr.add(StubTask(status=Status.RUNNING))
    monitor.set_task(parent.id)
    assert runner.attempt_of(parent.id) is None

    monitor._run_guarded(
        parent.id,
        monitor._advance,
        event(EventKind.SUBGRAPH_DONE, parent.id, attributes={"from_task": "x"}),
    )

    kinds = [
        row["kind"]
        for row in monitor._r.get("store_mgr").read_all("event")
        if row["task_id"] == str(parent.id)
    ]
    assert kinds == ["handling_failed"], "the raise is recorded, not swallowed"
