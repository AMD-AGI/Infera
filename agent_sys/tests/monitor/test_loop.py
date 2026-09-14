"""The mainloop — criteria 3 and 20, and the two regressions `p4` produced."""

from __future__ import annotations

import threading
import time

import pytest

from monitor import BufferClosed, Escalate, EventKind, GiveUp, PusherMonitor, event
from monitor.record import EVENT_KIND
from task_graph.registry import Registry

from .conftest import Status, StubAttempt, StubTask, StubTaskMgr


def run(monitor: PusherMonitor) -> threading.Thread:
    thread = threading.Thread(target=monitor.mainloop, name=f"monitor-{monitor.name}", daemon=True)
    thread.start()
    return thread


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_mainloop_is_not_the_agents(monitor: PusherMonitor, watched: StubTask) -> None:
    """Criterion 3: a monitor runs its own loop, distinct from any agent's.

    **They are not one mechanism with two users** (spec §1.1). Conflating them
    gives the watched and the watcher one heartbeat, and an agent whose loop is
    wedged cannot be the thing that notices its own wedge — so the test is that
    the monitor makes progress *while* a stand-in agent loop is wedged.
    """
    wedged = threading.Event()
    agent_thread_id: dict[str, int] = {}

    def agent_mainloop() -> None:
        agent_thread_id["id"] = threading.get_ident()
        wedged.wait(5.0)  # this loop is never coming back

    agent = threading.Thread(target=agent_mainloop, daemon=True)
    agent.start()
    assert wait_until(lambda: "id" in agent_thread_id)

    loop = run(monitor)
    monitor.report(event(EventKind.VALIDATION_FAILED, watched.id))

    assert wait_until(lambda: monitor.sweeps > 0)
    assert agent.is_alive(), "the agent's loop is wedged, which is the point"
    assert monitor.last_beat > 0

    wedged.set()
    monitor.stop()
    loop.join(2.0)
    assert not loop.is_alive()


def test_handler_exception_does_not_kill_the_loop(
    monitor: PusherMonitor, watched: StubTask, registry: Registry
) -> None:
    """The `p4` failure, as a regression test.

    Measured: after an unguarded handler raises, the thread is dead, further
    reports are accepted, depth grows, and **no producer sees an error**. The
    broad catch is what prevents that, and criterion 9 is why the failure is
    recorded rather than swallowed.
    """
    boom = iter([True])

    def exploding_decide(unit):
        if next(boom, False):
            raise RuntimeError("the handler itself raised")
        return GiveUp("second time round")

    monitor.decide = exploding_decide  # type: ignore[method-assign]
    loop = run(monitor)

    monitor.report(event(EventKind.VALIDATION_FAILED, watched.id))
    assert wait_until(lambda: _kinds(registry, watched).count(EventKind.HANDLING_FAILED) == 1)

    # The loop is still turning: a second report is still handled.
    monitor.report(event(EventKind.VALIDATION_UNREACHED, watched.id))
    assert wait_until(lambda: EventKind.MONITOR_GAVE_UP in _kinds(registry, watched))

    monitor.stop()
    loop.join(2.0)
    assert not loop.is_alive()


def test_sweep_runs_once_per_idle_period(monitor: PusherMonitor) -> None:
    """The §4.3 seam exists and is called.

    The alpha builds no poller — a wedged agent is a wedged thread and Python
    cannot kill a thread, so detection would arrive without remedy — but the
    *period* and the *hook* are what make adding one later an edit rather than a
    refactor.
    """
    loop = run(monitor)
    assert wait_until(lambda: monitor.sweeps >= 3, timeout=3.0)
    monitor.stop()
    loop.join(2.0)


def test_one_task_one_handling_across_queues(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """Criterion 20's second half, and spec §5.2 rule 5 since rev. 14.

    A planned advance for task T while T's exception is being decided would have
    the monitor moving a task forward and repairing it at the same instant. The
    exclusion cannot live in either queue — neither can see the other — so it is
    `_current`, one level up.
    """
    task = task_mgr.add(StubTask(status=Status.INPUT_VALIDATING))
    monitor.set_task(task.id)
    runner.attempts[task.id] = StubAttempt()

    inside = threading.Event()
    release = threading.Event()
    seen: list[object] = []

    def slow_decide(unit):
        seen.append(monitor._current)
        inside.set()
        release.wait(3.0)
        return GiveUp("done deliberating")

    monitor.decide = slow_decide  # type: ignore[method-assign]
    loop = run(monitor)

    monitor.report(event(EventKind.VALIDATION_FAILED, task.id))
    assert inside.wait(2.0), "the decision never started"

    # A planned advance arrives for the same task, mid-decision.
    monitor.report(event(EventKind.PHASE_DONE, task.id))
    time.sleep(0.05)
    assert task.calls == [], "the phase advanced while the task was being decided"

    release.set()
    assert wait_until(lambda: ("enter_phase", Status.RUNNING) in task.calls)
    assert seen == [task.id]

    monitor.stop()
    loop.join(2.0)


def test_report_after_stop_is_refused_loudly(monitor: PusherMonitor, watched: StubTask) -> None:
    """Refuse new reports loudly, drain what is queued, then stop. A producer
    with an exception to report into a closing system learns about it."""
    monitor.stop()
    with pytest.raises(BufferClosed):
        monitor.report(event(EventKind.VALIDATION_FAILED, watched.id))
    with pytest.raises(BufferClosed):
        monitor.report(event(EventKind.PHASE_DONE, watched.id))


def test_escalate_decision_is_carried_out(monitor: PusherMonitor, watched: StubTask, registry):
    """`decide` and `_apply` are separate so that deciding and doing are
    separable; this is the seam between them."""
    monitor.decide = lambda unit: Escalate("because the test says so")  # type: ignore[method-assign]
    loop = run(monitor)
    monitor.report(event(EventKind.VALIDATION_FAILED, watched.id))
    assert wait_until(lambda: EventKind.ESCALATED in _kinds(registry, watched))
    monitor.stop()
    loop.join(2.0)


def _kinds(registry: Registry, task: StubTask) -> list[EventKind]:
    store = registry.get("store_mgr")
    return [
        EventKind(r["kind"]) for r in store.read_all(EVENT_KIND) if r["task_id"] == str(task.id)
    ]
