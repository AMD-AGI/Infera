"""The two queues — criteria 15 and 20, and the invariant that carries the rules.

Spec §5.2's five rules are requirements on the *structure*, so these are tests of
the queues rather than of the loop. Criterion 20 in particular is written here on
purpose: a collapse added to the planned queue later "for symmetry" would be
invisible at the loop level.
"""

from __future__ import annotations

import random
import threading
import time

import pytest

from monitor import BufferClosed, EventKind, EventRecord, ExceptionBuffer, PlannedQueue, event
from task_graph.ids import TaskId


def gate(task_id: TaskId) -> EventRecord:
    return event(EventKind.OUTPUT_ABSENT, task_id)


def phase(task_id: TaskId) -> EventRecord:
    return event(EventKind.PHASE_DONE, task_id)


# --------------------------------------------------------------------------- #
# Rule 1 — never blocks, never refuses


def test_add_never_blocks() -> None:
    """Criterion 15. The buffer is **unbounded**: a `maxsize` would make the
    reporter wait on *the monitor*, coupling a common path to how busy the
    handler happens to be.

    Measured against a deadline rather than asserted, because "does not block" is
    a timing claim.
    """
    buffer = ExceptionBuffer()
    started = time.monotonic()
    for _ in range(5_000):
        buffer.add(gate(TaskId.new()))
    assert time.monotonic() - started < 2.0
    assert len(buffer) == 5_000


def test_add_from_many_threads_loses_nothing() -> None:
    buffer = ExceptionBuffer()
    ids = [TaskId.new() for _ in range(50)]

    def producer(task_id: TaskId) -> None:
        for _ in range(20):
            buffer.add(gate(task_id))

    threads = [threading.Thread(target=producer, args=(i,)) for i in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(buffer) == 50  # rule 2: depth cannot exceed the number of tasks
    total = 0
    while (unit := buffer.get(0.01)) is not None:
        total += len(unit.records)
        buffer.done(unit.task_id)
    assert total == 1_000  # rule 4: and not one record was overwritten


# --------------------------------------------------------------------------- #
# Rules 2 and 4 — bounded by dedup, and the collapse merges


def test_collapse_merges_and_loses_nothing() -> None:
    """Rule 4. `client-go` does last-wins here and that is its gap, not a pattern
    to inherit — criterion 9 requires *every* exception to be recorded, and
    `probes-monitor/p3` showed the naive version dropping four of five."""
    buffer = ExceptionBuffer()
    tid = TaskId.new()
    written = [gate(tid) for _ in range(5)]
    for record in written:
        buffer.add(record)

    assert len(buffer) == 1
    unit = buffer.get(0.01)
    assert unit is not None
    assert [r.id for r in unit.records] == [r.id for r in written]
    assert unit.newest.id == written[-1].id


def test_depth_is_bounded_by_the_number_of_tasks() -> None:
    """Rule 2, which is what makes rule 1 safe without a `maxsize`."""
    buffer = ExceptionBuffer()
    ids = [TaskId.new() for _ in range(3)]
    for _ in range(100):
        for tid in ids:
            buffer.add(gate(tid))
    assert len(buffer) == 3


# --------------------------------------------------------------------------- #
# Rule 5 — a task being handled is not handled twice


def test_requeued_exactly_once_while_processing() -> None:
    """Criterion 15. An event arriving while the monitor is inside a transition
    is re-queued **exactly once**, after the current handling completes."""
    buffer = ExceptionBuffer()
    tid = TaskId.new()
    buffer.add(gate(tid))

    first = buffer.get(0.01)
    assert first is not None
    for _ in range(4):  # four arrive while it is out with the loop
        buffer.add(gate(tid))
    assert len(buffer) == 0, "a task being handled must not be queued again"

    buffer.done(tid)
    assert len(buffer) == 1

    second = buffer.get(0.01)
    assert second is not None
    assert len(second.records) == 4  # all four, merged into one unit
    buffer.done(tid)
    assert len(buffer) == 0, "re-queued once, not once per arrival"


def test_done_without_arrivals_does_not_requeue() -> None:
    buffer = ExceptionBuffer()
    tid = TaskId.new()
    buffer.add(gate(tid))
    buffer.get(0.01)
    buffer.done(tid)
    assert len(buffer) == 0


# --------------------------------------------------------------------------- #
# The invariant


def test_invariant_holds_under_concurrent_add() -> None:
    """> Every element of `_order` is in `_dirty` and not in `_processing`.

    Asserted after a fuzz of interleaved `add` / `get` / `done`, because the
    three collections are the whole of the adopted shape and the invariant is
    what makes the five rules hold at once.
    """
    buffer = ExceptionBuffer()
    ids = [TaskId.new() for _ in range(8)]
    stop = threading.Event()

    def churn() -> None:
        rng = random.Random(1234)
        while not stop.is_set():
            buffer.add(gate(rng.choice(ids)))

    def drain() -> None:
        while not stop.is_set():
            unit = buffer.get(0.001)
            if unit is not None:
                buffer.done(unit.task_id)

    threads = [threading.Thread(target=churn) for _ in range(3)]
    threads += [threading.Thread(target=drain) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    stop.set()
    for t in threads:
        t.join()

    with buffer._cond:
        order = list(buffer._order)
        assert len(order) == len(set(order)), "a task id is queued twice"
        for task_id in order:
            assert task_id in buffer._dirty
            assert task_id not in buffer._processing


# --------------------------------------------------------------------------- #
# Shutdown


def test_shutdown_refuses_loudly_and_drains() -> None:
    """`client-go` discards on shutdown and says so in a comment; 3.13's
    `Queue.shutdown(immediate=True)` does the same. Both leave the producer
    believing the event was accepted, which is the one thing a reporting call
    must never do."""
    buffer = ExceptionBuffer()
    tid = TaskId.new()
    buffer.add(gate(tid))

    buffer.shutdown()

    with pytest.raises(BufferClosed):
        buffer.add(gate(TaskId.new()))

    unit = buffer.get(0.01)  # queued work is still delivered
    assert unit is not None and unit.task_id == tid
    buffer.done(tid)
    assert buffer.get(0.01) is None  # and then it stops, without blocking


def test_get_returns_none_on_timeout_rather_than_blocking() -> None:
    buffer = ExceptionBuffer()
    started = time.monotonic()
    assert buffer.get(0.05) is None
    assert time.monotonic() - started < 1.0


# --------------------------------------------------------------------------- #
# The planned queue — criterion 20


def test_planned_queue_never_collapses() -> None:
    """Criterion 20. Spec §2.2: "input validation finished" and "the main phase
    finished" are two advances of one task, and merging them is **a task that
    never runs its middle phase**. The bound comes from the domain instead — a
    task is in one phase at a time, so it has at most one outstanding advance."""
    queue = PlannedQueue()
    tid = TaskId.new()
    first, second = phase(tid), phase(tid)
    queue.add(first)
    queue.add(second)

    assert len(queue) == 2
    assert queue.get_nowait().id == first.id
    assert queue.get_nowait().id == second.id
    assert queue.get_nowait() is None


def test_planned_queue_is_fifo_across_tasks() -> None:
    queue = PlannedQueue()
    ids = [TaskId.new() for _ in range(4)]
    for tid in ids:
        queue.add(phase(tid))
    assert [queue.get_nowait().task_id for _ in ids] == ids


def test_planned_get_never_waits() -> None:
    """A planned advance is never the thing the loop should sleep for, because
    whatever produced it has already gone back to work."""
    queue = PlannedQueue()
    started = time.monotonic()
    assert queue.get_nowait() is None
    assert time.monotonic() - started < 0.05


def test_planned_shutdown_refuses_loudly() -> None:
    queue = PlannedQueue()
    queue.shutdown()
    with pytest.raises(BufferClosed):
        queue.add(phase(TaskId.new()))
