"""The runner seam.

What actually executes an agent is harness-specific and out of scope. What this
system owes is the interface, and a fake that lets the whole scheduler be tested
without one.
"""

from collections.abc import Callable
from typing import Any, Protocol

from task_graph.ids import TaskId
from task_graph.models import PHASES, Agent, HandoffRef, HandoffStatus, Task, TaskStatus
from task_graph.registry import Registry

__all__ = ["OnDone", "OnStopped", "TaskRunner", "FakeRunner"]


class OnDone(Protocol):
    """What a runner calls when an attempt finishes.

    **A Protocol rather than a `Callable` alias, because of `detail`.**
    `Scheduler.on_task_done` has accepted a keyword-only `detail` since the
    phase states landed, and `Execution.detail` is documented "from the runner;
    for a human" — but the alias was
    `Callable[[TaskId, TaskStatus, dict[str, float]], None]`, which cannot
    express a keyword argument. So a runner holding an exception had the field,
    the scheduler's parameter and the plumbing all in place and **no declared
    way to pass the value**: every failed task recorded `detail=''`.
    `Callable[..., None]` would have widened it by giving up the first three as
    well.

    The first three are positional-only, so an implementation may name them
    what it likes — `Scheduler.on_task_done` calls its first parameter `tid`.

    `detail` has a default, so a runner that has nothing to say passes nothing;
    what changes for implementers of the *callback* is that they must now
    tolerate the keyword. A `lambda *a: ...` does not — `lambda *a, **k: ...`
    does.
    """

    def __call__(
        self,
        task_id: TaskId,
        status: TaskStatus,
        usage: dict[str, float],
        /,
        *,
        detail: str = "",
    ) -> None: ...


OnStopped = Callable[[TaskId], None]


class TaskRunner(Protocol):
    def start(self, task: Task, agent: Agent, on_done: OnDone) -> None: ...

    def stop(self, task_id: TaskId, on_stopped: OnStopped) -> None: ...


class FakeRunner:
    """Records what was started. The test drives completion explicitly.

    Never calls `on_done` from inside `start`, which keeps tests deterministic.
    Re-entrancy is still handled by the scheduler, because a real synchronous
    runner is a reasonable implementation.
    """

    def __init__(self) -> None:
        self.running: dict[TaskId, tuple[Task, Agent, OnDone]] = {}
        self.started: list[TaskId] = []
        self.stop_requested: list[TaskId] = []
        self.skipped: list[tuple[TaskId, TaskStatus, str]] = []
        self._acks: dict[TaskId, OnStopped] = {}

    def start(self, task: Task, agent: Agent, on_done: OnDone) -> None:
        self.running[task.id] = (task, agent, on_done)
        self.started.append(task.id)

    def stop(self, task_id: TaskId, on_stopped: OnStopped) -> None:
        self.stop_requested.append(task_id)
        self._acks[task_id] = on_stopped

    # ---- test-driven, standing in for the agent ----

    def produce(
        self, registry: Registry, task_id: TaskId, *, valid: bool = True, content: Any = None
    ) -> None:
        """What a real agent does to its outputs: take a version and seal it.

        The only thing in the test suite that writes handoff state. That is what
        makes the authority test meaningful.
        """
        handoff_mgr = registry.get("handoff_mgr")
        task = registry.get("task_mgr").get(task_id)
        agent = registry.get("agent_mgr").get(task.current.agent_id)
        verdict = HandoffStatus.VALID if valid else HandoffStatus.INVALID

        for hid in task.outputs:
            version = handoff_mgr.get(hid).open_next(task_id, agent.id)
            version.seal(verdict, content)
            handoff_mgr.persist(hid)
            agent.handoffs.append(HandoffRef(handoff_id=hid, version=version.version))
        registry.get("agent_mgr").persist(agent.id)

    # ---- test-driven, standing in for the phase machinery ----

    def advance(self, registry: Registry, task_id: TaskId) -> None:
        """Move the task to its next phase.

        Calls a transition; it never assigns a status. That is the whole of the
        authority rule as it applies to a runner, and making the fake do it
        correctly is what makes the status-write spy meaningful — a fake writing
        `task.status` would train the suite to accept the thing the rule forbids.
        """
        task = registry.get("task_mgr").get(task_id)
        index = PHASES.index(task.status)
        task.enter_phase(PHASES[index + 1])

    def skip_phase(self, registry: Registry, task_id: TaskId, reason: str) -> None:
        """Advance without running anything, and record why.

        The *structured* report belongs one layer down — `validator`'s
        `PhaseOutcome` folds `ran`, `reused` and `skipped` and a skipped
        validator produces a `SkipRecord`. What this design owes is the state
        transition, plus a note on the `Execution` for a human.
        """
        task = registry.get("task_mgr").get(task_id)
        skipped = task.status
        self.skipped.append((task_id, skipped, reason))
        self.advance(registry, task_id)
        if task.current is not None:
            note = f"skipped {skipped.value}: {reason}"
            task.current.detail = f"{task.current.detail}; {note}".lstrip("; ")
            registry.get("task_mgr").persist(task_id)

    def finish(
        self,
        task_id: TaskId,
        status: TaskStatus = TaskStatus.SUCCEEDED,
        usage: dict[str, float] | None = None,
        detail: str = "",
    ) -> None:
        """`detail` is what a real runner has and the fake did not.

        It reaches `Execution.detail` — the first field a human reads on a
        failed task, and empty for every one of them until `OnDone` could
        express it.
        """
        _, _, on_done = self.running.pop(task_id)
        on_done(task_id, status, usage or {}, detail=detail)

    def ack_stop(self, task_id: TaskId) -> None:
        self.running.pop(task_id, None)
        self._acks.pop(task_id)(task_id)
