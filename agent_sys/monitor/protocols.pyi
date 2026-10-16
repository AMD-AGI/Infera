# Stub for monitor. Signature-only view of the same contract.
#
# Generated from protocols.py and kept in step by
# tests/interfaces/test_stub_agreement.py, which fails if the two
# public surfaces diverge. Reasons live in the .py; this file is the
# shape a type checker reads.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from task_graph.ids import AgentId, HandoffId, TaskId

__all__ = [
    "PLANNED",
    "Attempt",
    "AttemptRunner",
    "Budget",
    "BufferClosed",
    "EventKind",
    "EventRecord",
    "Monitor",
    "Pushable",
    "Recorder",
    "ScopeViolation",
    "UserSink",
]

class BufferClosed(RuntimeError): ...
class ScopeViolation(RuntimeError): ...

class EventKind(str, Enum):
    PHASE_DONE = "phase_done"
    SUBGRAPH_DONE = "subgraph_done"
    OUTPUT_ABSENT = "output_absent"
    OUTPUT_NOT_EXECUTABLE = "output_not_executable"
    SELF_CHECK_UNSET = "self_check_unset"
    BUDGET_EXCEEDED = "budget_exceeded"
    VALIDATION_FAILED = "validation_failed"
    VALIDATION_UNREACHED = "validation_unreached"
    PUSH_ATTEMPTED = "push_attempted"
    PUSH_INEFFECTIVE = "push_ineffective"
    ESCALATED = "escalated"
    MONITOR_GAVE_UP = "monitor_gave_up"
    HANDLING_FAILED = "handling_failed"
    LOOP_STALLED = "loop_stalled"
    THREAD_DIED = "thread_died"

PLANNED: frozenset[EventKind]

@dataclass(frozen=True)
class Budget:
    max_tokens: float | None = None
    max_seconds: float | None = None
    max_turns: int | None = None

class EventRecord(Protocol):
    id: Any
    fingerprint: tuple[str, ...]
    task_id: TaskId
    attempt: int
    agent_id: AgentId | None
    handoff_id: HandoffId | None
    kind: EventKind
    reported_by: str
    exception_type: str | None
    exception_message: str | None
    exception_stacktrace: str | None
    severity: int
    at: datetime
    attributes: Mapping[str, Any]

@runtime_checkable
class Pushable(Protocol):
    status: Any

    def instruct(self, message: str) -> None: ...
    def query(self) -> Any: ...

@runtime_checkable
class Attempt(Protocol):
    executor: Any

@runtime_checkable
class AttemptRunner(Protocol):
    def attempt_of(self, task_id: TaskId) -> Attempt | None: ...
    def carry_on(self, task_id: TaskId) -> str: ...

class Recorder(Protocol):
    def open(self, task_id: TaskId, attempt: int) -> None: ...
    def write(self, record: EventRecord) -> None: ...
    def read(self, task_id: TaskId, attempt: int) -> Sequence[EventRecord]: ...
    def is_open(self, task_id: TaskId, attempt: int) -> bool: ...

class UserSink(Protocol):
    def deliver(self, record: EventRecord, why: str) -> None: ...

class Monitor(Protocol):
    name: str
    last_beat: float

    def set_task(self, task_id: TaskId) -> None: ...
    def report(self, record: EventRecord) -> None: ...
    def mainloop(self) -> None: ...
    def stop(self) -> None: ...
