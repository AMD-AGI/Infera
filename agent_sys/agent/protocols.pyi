# Stub for agent. Signature-only view of the same contract.
#
# Generated from protocols.py and kept in step by
# tests/interfaces/test_stub_agreement.py, which fails if the two
# public surfaces diverge. Reasons live in the .py; this file is the
# shape a type checker reads.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Literal, Protocol

from task_graph.ids import TaskId

__all__ = [
    "AgentBackend",
    "AgentHistory",
    "AgentResult",
    "AgentStatus",
    "BackendUnavailable",
    "BackendUnsupported",
    "Executor",
    "Kind",
    "Rejection",
    "Runner",
    "Selection",
    "select_backend",
]

class BackendUnsupported(NotImplementedError): ...
class BackendUnavailable(RuntimeError): ...

class Kind(str, Enum):
    AI = "ai"
    PROGRAM = "program"
    HUMAN = "human"

class AgentStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

class AgentResult(Protocol):
    status: AgentStatus
    usage: Mapping[str, float]
    detail: str

class AgentHistory(Protocol):
    entries: Sequence[Mapping[str, Any]]
    session_ref: str | None

class Executor(Protocol):
    status: AgentStatus

    def start_async(self, on_started: Callable[[], None]) -> None: ...
    def wait(self) -> AgentResult: ...
    def start(self) -> AgentResult: ...
    def stop(self) -> None: ...
    def mainloop(self) -> None: ...

class AgentBackend(Executor, Protocol):
    def interrupt(self) -> None: ...
    def instruct(self, message: str) -> None: ...
    def query(self) -> AgentHistory: ...

class Rejection(Protocol):
    key: str
    reason: str

class Selection(Protocol):
    backend: Executor
    key: str
    source: Literal["cli", "spec", "config"]
    rejected: Sequence[Rejection]

def select_backend(
    spec: Any, *, override: str | None, config_order: Sequence[str], assignment: Any
) -> Selection: ...

class TaskAttempt(Protocol):
    task: Any
    agent: Any
    executor: Executor | None
    @property
    def environment(self) -> Mapping[str, str]: ...
    @property
    def is_running(self) -> bool: ...
    def wake(self) -> None: ...
    def release(self) -> None: ...

class Runner(Protocol):
    def start(self, task: Any, agent: Any, on_done: Any) -> None: ...
    def stop(self, task_id: TaskId, on_stopped: Any) -> None: ...
    def attempt_of(self, task_id: TaskId) -> TaskAttempt | None: ...
    def carry_on(self, task_id: TaskId) -> str: ...
