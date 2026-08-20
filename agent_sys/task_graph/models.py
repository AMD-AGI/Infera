"""Domain models.

Every model owns its own state machine and touches no other component. What it
does *not* own is its collection — that is the manager's.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from task_graph.ids import AgentId, HandoffId, TaskId

__all__ = [
    "Model",
    "TaskStatus",
    "HandoffStatus",
    "WAITING",
    "RESUMABLE",
    "HandoffStateError",
    "TaskStateError",
    "HandoffVersion",
    "Handoff",
    "Execution",
    "Task",
    "HandoffRef",
    "Agent",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # a typo'd field is an error, not a silent drop
        validate_assignment=True,  # task.status = X is validated
        use_enum_values=False,  # keep enum members; compare with `is`
    )


class TaskStatus(str, Enum):
    WAITING_HANDOFF = "waiting_handoff"
    WAITING_RESOURCE = "waiting_resource"
    RUNNING = "running"
    STOPPING = "stopping"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"


WAITING = frozenset({TaskStatus.WAITING_HANDOFF, TaskStatus.WAITING_RESOURCE})
RESUMABLE = frozenset({TaskStatus.FAILED, TaskStatus.SUSPENDED})


class HandoffStatus(str, Enum):
    CREATED = "created"  # declared; nothing written yet
    GENERATING = "generating"  # an agent has it open
    VALID = "valid"  # sealed, usable
    INVALID = "invalid"  # sealed, not usable


_VERDICTS = frozenset({HandoffStatus.VALID, HandoffStatus.INVALID})


class HandoffStateError(RuntimeError):
    """An illegal handoff transition."""


class TaskStateError(RuntimeError):
    """An illegal task transition."""


# ------------------------------------------------------------------ handoff


class HandoffVersion(Model):
    """One attempt at filling a slot. Immutable once sealed."""

    version: int
    status: HandoffStatus = HandoffStatus.CREATED
    producer_task_id: TaskId | None = None
    producer_agent_id: AgentId | None = None  # None until opened
    timestamp: datetime = Field(default_factory=_now)
    content: Any = None

    @property
    def is_valid(self) -> bool:
        return self.status is HandoffStatus.VALID

    def seal(self, status: HandoffStatus, content: Any = None) -> None:
        """GENERATING -> VALID | INVALID."""
        if self.status is not HandoffStatus.GENERATING:
            raise HandoffStateError(
                f"v{self.version} is {self.status.value}; only a GENERATING version can be sealed"
            )
        if status not in _VERDICTS:
            raise HandoffStateError(f"{status.value} is not a verdict; use VALID or INVALID")
        self.status = status
        self.content = content
        self.timestamp = _now()


class Handoff(Model):
    """The slot. `versions` is append-only and the list index is the version."""

    id: HandoffId
    type: str = ""
    versions: list[HandoffVersion] = Field(min_length=1)

    @property
    def latest(self) -> HandoffVersion:
        return self.versions[-1]

    @property
    def is_latest_valid(self) -> bool:
        return self.latest.is_valid

    def get(self, version: int) -> HandoffVersion:
        if not 0 <= version < len(self.versions):
            raise IndexError(f"{self.id} has no version {version}")
        return self.versions[version]

    def open_next(self, task_id: TaskId, agent_id: AgentId) -> HandoffVersion:
        """Hand an agent a version to write, GENERATING, and return it.

        Adopts `latest` in place if it is still CREATED; otherwise appends v+1.
        Raises if `latest` is GENERATING — someone else has it open.
        """
        latest = self.latest
        if latest.status is HandoffStatus.GENERATING:
            raise HandoffStateError(
                f"{self.id} v{latest.version} is already open by task {latest.producer_task_id}"
            )
        if latest.status is HandoffStatus.CREATED:
            version = latest
        else:
            version = HandoffVersion(version=len(self.versions))
            self.versions.append(version)
        version.producer_task_id = task_id
        version.producer_agent_id = agent_id
        version.timestamp = _now()
        version.status = HandoffStatus.GENERATING
        return version


# --------------------------------------------------------------------- task


class Execution(Model):
    """One attempt at running a task. The stack top is the live binding."""

    attempt: int
    agent_id: AgentId
    input_versions: dict[HandoffId, int] = Field(default_factory=dict)
    output_versions: dict[HandoffId, int] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    outcome: TaskStatus | None = None
    detail: str = ""  # from the runner; for a human

    @property
    def is_open(self) -> bool:
        return self.ended_at is None


class Task(Model):
    id: TaskId = Field(default_factory=TaskId.new)
    agent_spec: str  # a SPEC NAME, not an agent
    inputs: list[HandoffId] = Field(default_factory=list)
    outputs: list[HandoffId] = Field(default_factory=list)
    depends_on: list[TaskId] = Field(default_factory=list)  # the graph edge
    resources: dict[str, float] = Field(default_factory=dict)  # pool NAME -> amount
    status: TaskStatus = TaskStatus.WAITING_HANDOFF
    created_at: datetime = Field(default_factory=_now)
    expedited: bool = False
    history: list[Execution] = Field(default_factory=list)

    @property
    def current(self) -> Execution | None:
        return self.history[-1] if self.history else None

    @property
    def is_running(self) -> bool:
        current = self.current
        return current is not None and current.is_open

    def push_execution(
        self, agent_id: AgentId, input_versions: dict[HandoffId, int] | None = None
    ) -> Execution:
        """Bind an agent by appending an open record, attempt = len(history)."""
        if self.is_running:
            raise TaskStateError(f"{self.id} already has an attempt open")
        execution = Execution(
            attempt=len(self.history),
            agent_id=agent_id,
            input_versions=dict(input_versions or {}),
        )
        self.history.append(execution)
        return execution

    def close_execution(
        self,
        output_versions: dict[HandoffId, int] | None,
        outcome: TaskStatus,
        detail: str = "",
    ) -> None:
        """Seal the stack top."""
        if not self.is_running:
            raise TaskStateError(f"{self.id} has no open attempt to close")
        top = self.history[-1]
        top.output_versions = dict(output_versions or {})
        top.outcome = outcome
        top.detail = detail
        top.ended_at = _now()


# -------------------------------------------------------------------- agent


class HandoffRef(Model):
    handoff_id: HandoffId
    version: int


class Agent(Model):
    id: AgentId = Field(default_factory=AgentId.new)
    spec: str  # which kind
    task_id: TaskId | None = None  # what it is bound to
    handoffs: list[HandoffRef] = Field(default_factory=list)  # what it touched
    knowledge: Any = None  # left empty by the task definition
    config: dict[str, Any] = Field(default_factory=dict)
