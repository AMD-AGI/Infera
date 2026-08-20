"""The runner seam.

What actually executes an agent is harness-specific and out of scope. What this
system owes is the interface, and a fake that lets the whole scheduler be tested
without one.
"""

from collections.abc import Callable
from typing import Any, Protocol

from agent_sys.ids import TaskId
from agent_sys.models import Agent, HandoffRef, HandoffStatus, Task, TaskStatus
from agent_sys.registry import Registry

__all__ = ["OnDone", "OnStopped", "TaskRunner", "FakeRunner"]

OnDone = Callable[[TaskId, TaskStatus, dict[str, float]], None]
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

    def finish(
        self,
        task_id: TaskId,
        status: TaskStatus = TaskStatus.SUCCEEDED,
        usage: dict[str, float] | None = None,
    ) -> None:
        _, _, on_done = self.running.pop(task_id)
        on_done(task_id, status, usage or {})

    def ack_stop(self, task_id: TaskId) -> None:
        self.running.pop(task_id, None)
        self._acks.pop(task_id)(task_id)
