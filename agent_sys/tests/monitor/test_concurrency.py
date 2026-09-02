"""Blocking on the scheduler's lock — criterion 7."""

from __future__ import annotations

import threading
import time

from monitor import EventKind, PusherMonitor, event

from .conftest import Status, StubAttempt, StubTask, StubTaskMgr


def test_transition_blocks_then_proceeds(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """Criterion 7: a monitor calling a transition while the scheduler is
    mid-pass **blocks and then proceeds**, and it **holds nothing across the
    call**.

    That is not a new concurrency model. `task_graph` design §9's table already
    contemplates the case — *"an async runner calls `on_done` from its own thread
    → blocks on the lock until the current pass finishes"* — and a monitor is the
    same shape with a different caller. The stub's `lock` stands in for the
    scheduler's `RLock`, which every transition routes through.
    """
    lock = threading.RLock()
    task = task_mgr.add(StubTask(status=Status.INPUT_VALIDATING))
    task.lock = lock
    monitor.set_task(task.id)
    runner.attempts[task.id] = StubAttempt()

    held = threading.Event()
    finished = threading.Event()

    def scheduler_pass() -> None:
        with lock:
            held.set()
            time.sleep(0.2)

    passer = threading.Thread(target=scheduler_pass, daemon=True)
    passer.start()
    assert held.wait(2.0)

    started = time.monotonic()
    monitor._run_guarded(task.id, monitor._advance, event(EventKind.PHASE_DONE, task.id))
    elapsed = time.monotonic() - started
    finished.set()
    passer.join(2.0)

    assert elapsed >= 0.1, "the transition did not wait for the pass to finish"
    assert task.calls == [("enter_phase", Status.RUNNING)], "and then it proceeded"
    assert monitor._current is None, "it holds nothing between calls"


def test_the_monitor_holds_no_lock_of_its_own_across_report(
    monitor: PusherMonitor, task_mgr: StubTaskMgr
) -> None:
    """`report()` blocks on its own synchronous write and on nothing else.

    Rev. 3 of the spec justified the unbounded buffer by saying the reporter
    holds the scheduler's lock. **It does not** — §4.1.0 puts the call inside the
    runner's own gate loop, which holds no lock and touches no scheduler path.
    Two reporters on two threads therefore never wait on each other's handling.
    """
    a, b = task_mgr.add(StubTask()), task_mgr.add(StubTask())
    done = threading.Barrier(3, timeout=2.0)

    def reporter(task: StubTask) -> None:
        monitor.report(event(EventKind.VALIDATION_FAILED, task.id))
        done.wait()

    for task in (a, b):
        threading.Thread(target=reporter, args=(task,), daemon=True).start()

    done.wait()  # raises BrokenBarrierError if either reporter blocked
    assert len(monitor._buffer) == 2
