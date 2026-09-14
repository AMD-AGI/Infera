"""Stubs for the collaborators `monitor` resolves by name, and nothing else.

`docs/implementation-stage.md` §4.1: *import the Protocol, never a sibling's
implementation module; if you need a neighbour's behaviour to run a test, satisfy
the Protocol with a stub in your own `tests/`.* Every stub here exists for that
reason, and each names the seam it stands in for.

**`StubTask` carries the phase members `task_graph.TaskStatus` does not have
yet.** `monitor.next_phase` resolves the successor by *name*, against whichever
enum the status came from, so the planned channel is testable today and binds to
the real `TaskStatus` unchanged when `task_graph` rev. 12 lands. `test_planned.py`
carries the guard that arms itself when it does.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Any

import pytest

from monitor import DEFAULT_MONITOR_NAME, PusherMonitor
from task_graph.ids import TaskId
from task_graph.models import TaskStateError
from task_graph.registry import Registry
from task_graph.store import MemoryStoreMgr


class Status(str, Enum):
    """The phase sequence of `task_graph` spec §3.2, by the names it uses."""

    WAITING_RESOURCE = "waiting_resource"
    INPUT_VALIDATING = "input_validating"
    RUNNING = "running"
    OUTPUT_VALIDATING = "output_validating"
    SUCCEEDED = "succeeded"


#: The same three names, in the same order, as `task_graph.models.PHASES`.
#: `test_planned.py::test_phase_order_names_are_task_graphs_when_they_exist`
#: is the guard that these have not diverged.
PHASE_NAMES = ["INPUT_VALIDATING", "RUNNING", "OUTPUT_VALIDATING"]


class StubTask:
    """What the monitor reads of a `Task`: `id`, `parent`, `monitor_spec`,
    `status`, and the transition verbs.

    It is also the **status-write spy** for criterion 6. `task_graph`'s
    `test_authority.py` proves the scheduler never writes handoff state by
    logging every call; the same shape is applied here to the new caller: every
    write of `status` is recorded with whether a verb was on the stack, so a
    monitor that assigned a status instead of calling a transition would be
    visible even though the write itself succeeded.
    """

    def __init__(
        self,
        *,
        parent: TaskId | None = None,
        monitor_spec: str | None = None,
        status: Status = Status.INPUT_VALIDATING,
        is_end: bool = True,
    ) -> None:
        object.__setattr__(self, "_in_verb", 0)
        self.id = TaskId.new()
        self.parent = parent
        self.monitor_spec = monitor_spec
        self.status = status
        # `Task.is_end` defaults to True — the subgraph's exit point — and the
        # default matters: it is what makes a completing subtask announce to its
        # parent. Defaulted the same way here so a test that does not mention it
        # gets the real class's behaviour rather than the stub's convenience.
        self.is_end = is_end
        self.calls: list[tuple[str, Any]] = []
        self.status_writes: list[bool] = []  # True when a verb was on the stack
        self.executions = 0
        self.lock: threading.RLock | None = None  # stands in for the scheduler's

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "status":
            self.__dict__.setdefault("status_writes", []).append(bool(self._in_verb))
        object.__setattr__(self, name, value)

    def _verb(self, name: str, payload: Any) -> None:
        self.calls.append((name, payload))

    def enter_phase(self, phase: Any) -> None:
        """`task_graph` design §3.4's transition, which rev. 14 makes a monitor
        caller. The `lock` models the scheduler's `RLock`: a real transition
        routes through `_move` under it.

        **It rejects what the real one rejects, and that was not true at first.**
        `Task.enter_phase` raises `TaskStateError` twice — once if the task is
        not in a phase state, once if `phase` is not the *successor* of the
        current one, so a runner cannot skip output validation by advancing
        twice. This stub used to accept any phase from any status, so a
        `next_phase` that returned the wrong member would have passed every test
        here and raised in production.

        A **permissive stub** is the shape behind four of this week's defects
        (`agent`'s `set_task` no-ops, my own three `_advance` tests): the double
        that accepts more than the real thing turns its suite into a record of
        what the author expected rather than of what the collaborator promises.
        """
        if self.lock is not None:
            with self.lock:
                self._enter(phase)
        else:
            self._enter(phase)

    def _enter(self, phase: Any) -> None:
        order = [s.name for s in Status]
        if self.status.name not in PHASE_NAMES:
            raise TaskStateError(
                f"cannot advance {self.id}: it is {self.status.value}, expected a phase state"
            )
        successor = PHASE_NAMES[PHASE_NAMES.index(self.status.name) + 1 :][:1]
        if not successor or phase.name != successor[0]:
            raise TaskStateError(
                f"cannot advance {self.id} from {self.status.value} to {phase.value}; "
                f"the phase sequence is {PHASE_NAMES} (of {order})"
            )
        object.__setattr__(self, "_in_verb", self._in_verb + 1)
        try:
            self._verb("enter_phase", phase)
            self.status = phase
        finally:
            object.__setattr__(self, "_in_verb", self._in_verb - 1)

    def cancel(self, **kw: Any) -> None:
        self._verb("cancel", kw)

    def fail(self, **kw: Any) -> None:
        self._verb("fail", kw)

    def push_execution(self, *a: Any, **kw: Any) -> None:
        """Present so that "the re-entry pushes no second execution" is a fact a
        test can check rather than an absence it has to trust."""
        self.executions += 1


class StubTaskMgr:
    """`task_graph.TaskMgr`, reduced to the one method the monitor calls."""

    def __init__(self) -> None:
        self.tasks: dict[TaskId, StubTask] = {}

    def add(self, task: StubTask) -> StubTask:
        self.tasks[task.id] = task
        return task

    def get(self, task_id: TaskId) -> StubTask:
        return self.tasks[task_id]


class StubBackend:
    """Satisfies `monitor.Pushable` — the three members and no more.

    Standing in for `agent.AgentBackend`, which this package may not import.
    `tests/interfaces/test_pushable.py` is what keeps the two declarations in
    step; this only has to satisfy the local one.
    """

    def __init__(self) -> None:
        self.status = "running"
        self.instructions: list[str] = []

    def instruct(self, message: str) -> None:
        self.instructions.append(message)

    def query(self) -> Any:
        return []


class StubAttempt:
    """`agent.TaskAttempt` (design §7.5): the thread and the executor handle.

    **`is_running` is the whole reason this stub was wrong once**, and the
    correction is worth keeping visible. The first version modelled a non-leaf
    as *no entry in the runner's map*, which is what `monitor` design §6.1
    assumed — and the real `TaskAttempt` **survives its thread**: `release()`
    ends the thread and says "the object survives", and `Runner._attempts` is
    emptied only by `stop`. The stub encoded this module's assumption instead of
    the neighbour's behaviour, so every test agreed with it and the defect was
    invisible until `agent`'s code was read.
    (`scratch/impl-2026-08/monitor/p7_nonleaf_wake_is_silent.py`.)
    """

    def __init__(self, executor: StubBackend | None = None, *, is_running: bool = True) -> None:
        self.executor = executor or StubBackend()
        self.is_running = is_running
        self.woken = 0

    def wake(self) -> None:
        """Real `wake()` is `Event.set()` — on a released attempt it is a silent
        no-op, not an error. The stub counts it for the same reason: a test has
        to be able to see the *wrong* call having been made."""
        self.woken += 1

    def release(self) -> None:
        """What a non-leaf's attempt does at `unfold`: the thread ends, the
        object stays."""
        self.is_running = False


class StubRunner:
    """`agent.Runner`, with the members the monitor actually calls.

    The two shapes, as the **real** runner presents them:

    | | `attempt_of` | `carry_on` returns |
    |---|---|---|
    | a leaf, thread parked between phases | the attempt | `"woken"` |
    | a non-leaf, thread ended at `unfold` | **the attempt** | `"resumed"` |
    | after `Runner.stop` popped it | `None` | raises |
    """

    def __init__(self) -> None:
        self.attempts: dict[TaskId, StubAttempt] = {}
        self.resumed: list[TaskId] = []
        self.woken: list[TaskId] = []
        self.pushed: list[TaskId] = []  # a runner must never fill this

    def attempt_of(self, task_id: TaskId) -> StubAttempt | None:
        return self.attempts.get(task_id)

    def carry_on(self, task_id: TaskId) -> str:
        """The whole operation the monitor now calls instead of branching.

        The real one does this under the attempt's own lock and returns
        `"woken"` or `"resumed"`; the stub records which so a test can assert
        the two shapes **without reaching into the runner**, which is the reason
        the return value was asked for.

        Raises when no attempt is live, as the real one does: a missing attempt
        at a phase boundary means the task is stuck, and a no-op hides it.
        """
        if task_id not in self.attempts:
            raise KeyError(f"no live attempt for task {task_id}")
        attempt = self.attempts[task_id]
        if attempt.is_running:
            attempt.woken += 1
            self.woken.append(task_id)
            return "woken"
        attempt.is_running = True
        self.resumed.append(task_id)
        return "resumed"

    def make_released_attempt(self, task_id: TaskId) -> StubAttempt:
        """A non-leaf awaiting re-entry: the attempt is present, its thread is
        not. The shape the first version of this stub could not express.

        **`make_` because the real `Runner` has no such method.** It was
        `released()`, which reads like a query a caller could make — and *a name
        a reader can take for a real method* is the same family as a `getattr`
        that looks like a field access and a double that looks like its subject.
        Nothing was wrong with it; the prefix is what makes it unmistakably
        fixture-side.
        """
        attempt = self.attempts.setdefault(task_id, StubAttempt())
        attempt.release()
        return attempt


class BareRunner:
    """A runner with neither `attempt_of` nor `resume` — the tree as shipped.

    Design §9.2: until `agent` builds them the pusher degrades to `Escalate`,
    and that degradation is a tested behaviour rather than an accident.
    """


class ForbiddenScheduler:
    """Any attribute access is a failure.

    Criterion 23: the scheduler is not in a non-leaf's re-entry. Registering this
    under `scheduler` turns "we do not think it is called" into an assertion.
    """

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the monitor touched scheduler.{name}; criterion 23 forbids it")


@pytest.fixture
def registry() -> Registry:
    r = Registry()
    r.register("store_mgr", MemoryStoreMgr())
    r.register("task_mgr", StubTaskMgr())
    r.register("runner", StubRunner())
    r.register("scheduler", ForbiddenScheduler())
    return r


@pytest.fixture
def task_mgr(registry: Registry) -> StubTaskMgr:
    return registry.get("task_mgr")


@pytest.fixture
def runner(registry: Registry) -> StubRunner:
    return registry.get("runner")


@pytest.fixture
def monitor(registry: Registry) -> PusherMonitor:
    """The default monitor, registered under the name a task with no
    `monitor_spec` resolves to."""
    m = PusherMonitor(DEFAULT_MONITOR_NAME, registry, period=0.01)
    registry.register(f"monitor:{m.name}", m)
    return m


@pytest.fixture
def watched(monitor: PusherMonitor, task_mgr: StubTaskMgr) -> StubTask:
    """One task, added to the manager and given to the monitor by `set_task`."""
    task = task_mgr.add(StubTask())
    monitor.set_task(task.id)
    return task
