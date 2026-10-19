"""The two queues — design §4.

Spec §5.2's five rules, applied to each queue as that section's table assigns
them. Kubernetes `client-go/util/workqueue` is the shape of the **unplanned**
one; the planned one is a plain FIFO and is smaller for a reason §4.4 states.

The whole of the adoption is three collections and an invariant. It is copied
rather than imported because `client-go` is Go — a dependency on a Kubernetes
client to get sixty lines would be absurd — and `queue.Queue` supplies neither
dedup nor merge, while its `shutdown` is 3.13-only against a 3.10 target and
discards pending items silently.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from task_graph.ids import TaskId

from .protocols import BufferClosed
from .record import EventRecord

__all__ = ["ExceptionBuffer", "PlannedQueue", "Unit"]


@dataclass(frozen=True)
class Unit:
    """One task's outstanding unplanned work, and every record merged into it.

    **The buffer keys on task id and carries the records anyway**, which is where
    it departs from the prior art: `client-go` holds bare keys and re-reads state
    from an informer cache, there is no cache here, and a collapse that kept only
    the last payload would silently lose an observation. Rule 4 says merge, and
    `records` is the merge.
    """

    task_id: TaskId
    records: tuple[EventRecord, ...]

    @property
    def newest(self) -> EventRecord:
        return self.records[-1]


class ExceptionBuffer:
    """The unplanned channel. Unbounded, deduplicating, merging.

    The invariant, and it is what makes the five rules hold:

    > **Every element of `_order` is in `_dirty` and not in `_processing`.**
    """

    def __init__(self) -> None:
        self._order: deque[TaskId] = deque()  # FIFO across tasks
        self._dirty: set[TaskId] = set()  # queued, or pending re-queue
        self._processing: set[TaskId] = set()  # a unit is out with the loop
        self._records: dict[TaskId, list[EventRecord]] = {}
        self._cond = threading.Condition()
        self._closed = False

    def add(self, record: EventRecord) -> None:
        """Never blocks, never refuses. Raises only after `shutdown`.

        There is no `maxsize` anywhere: a full queue would make the reporter wait
        on *the monitor*, coupling a common path to how busy the handler happens
        to be.
        """
        with self._cond:
            if self._closed:
                raise BufferClosed(
                    f"buffer is closed; {record.kind.value} for {record.task_id} was refused"
                )
            self._records.setdefault(record.task_id, []).append(record)  # rule 4: merge
            if record.task_id in self._dirty:
                return  # already known; do not re-queue
            self._dirty.add(record.task_id)
            if record.task_id in self._processing:
                return  # rule 5: `done` re-queues it
            self._order.append(record.task_id)
            self._cond.notify()

    def get(self, timeout: float) -> Unit | None:
        """The next unit, or `None` when `timeout` elapses with nothing to do.

        Marks the task as processing, which is half of rule 5; `done` is the
        other half and **forgetting it wedges that task forever** — the one sharp
        edge in the prior art, which `client-go` flags in a comment.
        """
        with self._cond:
            if not self._order and not self._closed:
                self._cond.wait(timeout)
            if not self._order:
                return None
            task_id = self._order.popleft()
            self._dirty.discard(task_id)
            self._processing.add(task_id)
            return Unit(task_id, tuple(self._records.pop(task_id, [])))

    def done(self, task_id: TaskId) -> None:
        """Finish a unit, re-queuing the task if anything arrived meanwhile."""
        with self._cond:
            self._processing.discard(task_id)
            if task_id in self._dirty:  # arrived while it was being handled
                self._order.append(task_id)
                self._cond.notify()

    def shutdown(self) -> None:
        """Refuse new records **loudly**, let the queue drain, stop `get` waiting.

        `client-go` discards on shutdown and says so in a comment; 3.13's
        `Queue.shutdown(immediate=True)` does the same. Both leave the producer
        believing the event was accepted. A producer with an exception to report
        into a closing system learns about it instead.

        Dropping work is permitted; dropping a fact is not — by rule 3 the record
        is on disk before the buffer ever sees it, so a unit lost here is a
        handling that did not happen, visible in the record set as an occurrence
        with no outcome after it.
        """
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def __len__(self) -> int:
        with self._cond:
            return len(self._order)


class PlannedQueue:
    """The planned channel: a FIFO, and deliberately not the other one.

    **Everything the buffer does to bound itself is absent here, and that is the
    design.** Spec §2.2: two advances of one task are two phases, and merging
    them is a task that never runs its middle one.

    It is bounded anyway, and by a stronger fact than dedup — a task is in one
    phase at a time, so it has at most one outstanding advance, and depth cannot
    exceed the number of tasks. **The bound the buffer buys with a `_dirty` set,
    this queue gets from the domain.**

    Rule 5 it does share, and that check lives in neither queue: neither can see
    the other, so `BaseMonitor._current` is where one-task-one-handling spans
    both.
    """

    def __init__(self) -> None:
        self._items: deque[EventRecord] = deque()
        self._lock = threading.Lock()
        self._closed = False

    def add(self, record: EventRecord) -> None:
        """Append. Never blocks, never collapses. Raises after `shutdown`."""
        with self._lock:
            if self._closed:
                raise BufferClosed(
                    f"planned queue is closed; {record.kind.value} for {record.task_id} was refused"
                )
            self._items.append(record)

    def get_nowait(self) -> EventRecord | None:
        """The next advance, or `None` if empty. Never waits.

        A planned advance is never the thing the loop should sleep for, because
        whatever produced it has already gone back to work.
        """
        with self._lock:
            return self._items.popleft() if self._items else None

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
