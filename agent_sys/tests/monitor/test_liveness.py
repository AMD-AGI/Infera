"""The monitor's own liveness — criteria 25 and 26.

Rev. 14 put the planned path through this module, so "nothing monitors the
monitor" stopped being a recorded risk and became a requirement: before, a dead
monitor meant exceptions went unhandled; now it means **every task stops
advancing, silently**.
"""

from __future__ import annotations

import threading
import time

from monitor import (
    NO_TASK,
    EventKind,
    NullUserSink,
    PusherMonitor,
    Recorder,
    check_liveness,
    install_excepthook,
)
from monitor.record import EVENT_KIND
from task_graph.store import MemoryStoreMgr


class FakeClock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        return self.t


class Monitorish:
    def __init__(self, name: str, last_beat: float) -> None:
        self.name = name
        self.last_beat = last_beat


# --------------------------------------------------------------------------- #
# Criterion 25 — the excepthook


def test_excepthook_records_and_surfaces() -> None:
    """Criterion 25: **an exception that escapes any thread produces a record and
    reaches the user.**

    The default does not, and this is measured rather than assumed
    (`probes-monitor/p4_thread_death.py`): a thread's uncaught exception prints
    to stderr, the thread dies, the process continues with its exit code
    unchanged, and producers see nothing.
    """
    store = MemoryStoreMgr()
    recorder, sink = Recorder(store), NullUserSink()
    saved = threading.excepthook
    install_excepthook(recorder, sink, chain=False)
    try:

        def raises() -> None:
            raise ValueError("nobody caught this")

        thread = threading.Thread(target=raises, name="a-task-thread")
        thread.start()
        thread.join(2.0)
    finally:
        threading.excepthook = saved

    (row,) = store.read_all(EVENT_KIND)
    assert row["kind"] == EventKind.THREAD_DIED.value
    assert row["exception_type"] == "ValueError"
    assert "nobody caught this" in row["exception_message"]
    assert "raises" in row["exception_stacktrace"]
    assert row["severity"] == 17
    assert row["reported_by"] == "a-task-thread"

    assert sink.delivered, "the record was written and nobody was told"
    assert sink.delivered[0][0].kind is EventKind.THREAD_DIED


def test_a_thread_may_attribute_itself_to_a_task() -> None:
    """Whoever spawned the thread may set `task_id` on it; nothing is inferred
    when they did not, and the nil id says so in one readable value."""
    from task_graph.ids import TaskId

    store = MemoryStoreMgr()
    saved = threading.excepthook
    install_excepthook(Recorder(store), NullUserSink(), chain=False)
    tid = TaskId.new()
    try:
        thread = threading.Thread(target=lambda: 1 / 0)
        thread.task_id = tid  # type: ignore[attr-defined]
        thread.start()
        thread.join(2.0)
    finally:
        threading.excepthook = saved

    (row,) = store.read_all(EVENT_KIND)
    assert row["task_id"] == str(tid)
    assert str(NO_TASK) != row["task_id"]


def test_install_excepthook_returns_what_it_replaced() -> None:
    """It is process-global, so the caller has to be able to put it back."""
    saved = threading.excepthook
    previous = install_excepthook(Recorder(MemoryStoreMgr()), NullUserSink())
    try:
        assert previous is saved
        assert threading.excepthook is not saved
    finally:
        threading.excepthook = saved


# --------------------------------------------------------------------------- #
# Criterion 26 — the heartbeat


def test_stall_needs_n_consecutive() -> None:
    """Criterion 26: a monitor whose loop has stopped turning is detected **after
    N consecutive stale periods rather than one**, in the `failureThreshold`
    shape §4.3 already chose over a heartbeat for the agent case."""
    clock = FakeClock()
    alive = Monitorish("alive", clock.t)
    stalled = Monitorish("stalled", clock.t)

    clock.t += 3.5  # threshold=3, period=1.0
    alive.last_beat = clock.t

    records = check_liveness([alive, stalled], period=1.0, threshold=3, now=clock)

    assert [r.attributes["monitor"] for r in records] == ["stalled"]
    assert records[0].kind is EventKind.LOOP_STALLED
    assert records[0].task_id == NO_TASK
    assert records[0].severity == 17


def test_one_slow_round_is_not_a_stall() -> None:
    clock = FakeClock()
    slow = Monitorish("slow", clock.t)
    clock.t += 2.5  # under threshold * period

    assert check_liveness([slow], period=1.0, threshold=3, now=clock) == []


def test_clock_is_injected() -> None:
    """The stall test uses a fake clock and does not sleep, and a monotonic jump
    *backwards* is not a stall — `time.monotonic`, never wall-clock, because a
    clock adjustment must not read as a dead monitor."""
    clock = FakeClock()
    monitor = Monitorish("m", clock.t)
    clock.t -= 60.0

    assert check_liveness([monitor], period=1.0, threshold=3, now=clock) == []


def test_check_liveness_writes_nothing() -> None:
    """Pure: it takes a clock and returns records. **The checker is a comparison
    of one float**, which is what makes it an answer rather than an infinite
    regress — small enough that nothing needs to watch *it*."""
    store = MemoryStoreMgr()
    clock = FakeClock()
    stalled = Monitorish("stalled", clock.t)
    clock.t += 100.0

    records = check_liveness([stalled], period=1.0, now=clock)

    assert records
    assert store.read_all(EVENT_KIND) == []


def test_a_running_loop_keeps_beating(monitor: PusherMonitor) -> None:
    """And the mainloop stamps it once per round, before any work, so a round
    spent entirely on planned work still counts as alive."""
    first = monitor.last_beat
    thread = threading.Thread(target=monitor.mainloop, daemon=True)
    thread.start()
    try:
        deadline = threading.Event()
        deadline.wait(0.2)
        assert monitor.last_beat > first
        assert check_liveness([monitor], period=0.01, threshold=3) == []
    finally:
        monitor.stop()
        thread.join(2.0)


# --------------------------------------------------------------------------- #
# Starting the loops — the gap `demo`'s first end-to-end run found (F-D7)


def test_start_monitors_gives_every_registered_monitor_a_thread(registry) -> None:
    """`demo` got through confinement, preflight, load and dispatch, and then sat
    in `INPUT_VALIDATING` for 300 s: **nothing ever started a monitor's loop.**

    `build_registry` registers `monitor:<name>` and gives it no thread, and a
    monitor that is never started still *accepts* reports — `report()` persists
    and enqueues exactly as it should — so the queue fills and the task never
    advances. That is `interfaces.md` §2.1 rev. 4's failure reached from the
    other direction: the name resolves and the loop is not running.
    """
    from monitor import DEFAULT_MONITOR_NAME, PusherMonitor, start_monitors

    for name in (DEFAULT_MONITOR_NAME, "careful"):
        registry.register(f"monitor:{name}", PusherMonitor(name, registry, period=0.01))

    running = start_monitors(registry)
    try:
        assert len(running) == 2
        beats = {m.name: m.last_beat for m in running.monitors}
        assert wait_for(lambda: all(m.last_beat > beats[m.name] for m in running.monitors))
    finally:
        stragglers = running.stop(timeout=2.0)
    assert stragglers == [], f"these loops did not return: {stragglers}"


def test_a_monitor_that_was_never_started_reads_as_stalled(registry) -> None:
    """The detection already existed and nothing called it.

    `last_beat` is stamped at construction and moved only by the loop, so a
    monitor nobody started is stale from the first period — `check_liveness`
    reports it without needing a new mechanism. An entry point that calls
    neither has a system that stops silently; one that calls both has one that
    says when it stopped.
    """
    from monitor import PusherMonitor, check_liveness

    never_started = PusherMonitor("forgotten", registry, period=0.01)
    clock = FakeClock()
    never_started.last_beat = clock.t
    clock.t += 10.0

    (record,) = check_liveness([never_started], period=1.0, threshold=3, now=clock)
    assert record.kind is EventKind.LOOP_STALLED
    assert record.attributes["monitor"] == "forgotten"


def test_stop_names_a_loop_that_would_not_return(registry) -> None:
    """`stop()` returns the names that did not stop rather than hanging or
    passing silently — a wedged loop at shutdown is a fact, not a delay."""
    from monitor import PusherMonitor, start_monitors

    wedged = PusherMonitor("wedged", registry, period=0.01)
    registry.register("monitor:wedged", wedged)

    stuck = threading.Event()
    original = wedged.mainloop
    wedged.mainloop = lambda: stuck.wait(10.0)  # type: ignore[method-assign]

    running = start_monitors(registry)
    try:
        assert running.stop(timeout=0.2) == ["wedged"]
    finally:
        stuck.set()
        wedged.mainloop = original  # type: ignore[method-assign]


def wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False
