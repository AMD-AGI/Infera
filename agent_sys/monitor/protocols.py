"""What leaves `monitor/`.

The task's event loop. Everything that happens to a task and is not the task's
own work arrives at a `Monitor` through one call, `report`, and lands on one of
two channels: **planned** advances, handled by code and never by a model, and
**unplanned** outcomes, which are a decision. `PLANNED` is the whole of that
routing rule and is why a reporter never has to classify what it is reporting.

Nothing here imports `agent`, and that is deliberate. The monitor needs
`instruct` on a live agent while the runner needs `report` from here; written the
obvious way that is a package cycle. `Pushable` breaks it structurally — see its
docstring.

Declarations only. See `docs/design.md` and `../docs/interfaces.md` §4.9.
"""

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


# --------------------------------------------------------------------------- #
# Errors


class BufferClosed(RuntimeError):
    """A report arrived after shutdown.

    Raised rather than dropped. `client-go`'s workqueue discards on shutdown and
    says so in a comment; Python 3.13's `Queue.shutdown(immediate=True)` does the
    same. Both leave the producer believing the event was accepted, which is the
    one thing a reporting call must never do.
    """


class ScopeViolation(RuntimeError):
    """A monitor tried to transition a task other than the one it is handling.

    Carries both ids. Spec criterion 8 is about the *verbs*, not the filesystem:
    a monitor is unconfined by construction (spec §6.1), so this in-process check
    is the whole of the boundary and it is not a defence in depth.
    """


# --------------------------------------------------------------------------- #
# Vocabulary


class EventKind(str, Enum):
    """What happened. A closed enum, and **no value is a benign default** —
    distinguishing "attempted and ineffective" from "never attempted" is a phase
    distinction (Erlang's `Context`), and free text answers it only to a human.
    """

    # -- planned. The whole of the planned channel; see PLANNED below --
    PHASE_DONE = "phase_done"
    SUBGRAPH_DONE = "subgraph_done"

    # -- the completeness gate, four independent failures (spec §4.1.0) --
    OUTPUT_ABSENT = "output_absent"
    OUTPUT_NOT_EXECUTABLE = "output_not_executable"
    SELF_CHECK_UNSET = "self_check_unset"
    BUDGET_EXCEEDED = "budget_exceeded"

    # -- the validator, both bad outcomes (spec §2.1) --
    VALIDATION_FAILED = "validation_failed"
    VALIDATION_UNREACHED = "validation_unreached"

    # -- the monitor's own actions (criterion 9) --
    PUSH_ATTEMPTED = "push_attempted"
    PUSH_INEFFECTIVE = "push_ineffective"
    ESCALATED = "escalated"
    MONITOR_GAVE_UP = "monitor_gave_up"
    HANDLING_FAILED = "handling_failed"

    # -- the monitor observing itself (spec §5.4) --
    LOOP_STALLED = "loop_stalled"
    THREAD_DIED = "thread_died"


#: The routing rule, in one place. `report()` consults this and nothing else.
#:
#: A kind absent from here reaches the unplanned queue, which is the safe
#: direction: the worst outcome is an event that gets decided instead of
#: switched. The reverse default would silently skip a decision.
PLANNED: frozenset[EventKind] = frozenset({EventKind.PHASE_DONE, EventKind.SUBGRAPH_DONE})


@dataclass(frozen=True)
class Budget:
    """The alpha's thresholds, one global value.

    Not per task and not per agent spec: **nobody yet knows what a normal task
    costs**, so a per-task limit would be authored out of numbers no one has and
    would mostly be a copy of a default — a global setting with extra steps and a
    worse failure mode, since a wrong local value is invisible where a wrong
    global one is felt immediately.

    Read by the runner at the completeness gate, not by the monitor. It lives
    here because `BUDGET_EXCEEDED` does, and a threshold with no matching kind is
    unreportable.
    """

    max_tokens: float | None = None
    max_seconds: float | None = None
    max_turns: int | None = None


class EventRecord(Protocol):
    """One occurrence, as the seam sees it.

    A **persisted value, never a log line** — a test that satisfies criterion 9
    with `caplog` is testing the logging configuration. Logging is a projection
    of this, rendered at the severity it already carries.

    The field vocabulary is a hybrid, at no dependency cost: OpenTelemetry's
    stable `exception.*` names and `SeverityNumber` supply the naming, Erlang's
    SASL supervisor record the structure, and Sentry's event/fingerprint split
    the identity. The OTel *SDK* is not adopted: it is emit-only, and §6 of the
    design must read back what it wrote.
    """

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


# --------------------------------------------------------------------------- #
# The seams


@runtime_checkable
class Pushable(Protocol):
    """The part of `agent.AgentBackend` a monitor uses.

    Declared here so `monitor` imports nothing from `agent`, which is what keeps
    the dependency one-way: consumers import the monitor, never the reverse.
    `AgentBackend` satisfies this structurally and neither package names the
    other.

    The cost is two declarations of one shape with nothing in either package
    noticing if they diverge, so `tests/interfaces/test_pushable.py` asserts the
    agreement — a test may import both, because tests are not under the import
    rule.
    """

    status: Any

    def instruct(self, message: str) -> None: ...

    def query(self) -> Any:
        """The agent's history. What makes `answer` reachable with nothing new
        built: the agent does not push a question, the monitor reads one."""
        ...


@runtime_checkable
class Attempt(Protocol):
    """The part of `agent.TaskAttempt` a monitor uses.

    Declared here for the same reason as `Pushable` and with the same cost: the
    monitor may not import `agent`, so the shape is written twice and
    `tests/interfaces/test_runner_seam.py` is what keeps the copies in step.

    **This one was undeclared until a defect made the case for it.** `_advance`
    branched on `attempt_of(tid) is None` and called that "the non-leaf case";
    the attempt object *survives its thread*, so the branch never fired, `wake()`
    set an Event nobody was waiting on, and the parent stalled in
    `OUTPUT_VALIDATING` silently. Nothing anywhere said what the runner
    guarantees about an attempt's lifetime, which is exactly what a declared
    seam is for.
    """

    #: A `Pushable` while the main phase runs, and `None` otherwise — including
    #: for the whole life of a non-leaf, which never has one. **`None` is a real
    #: value, not an error**: it is `None` for a parked leaf between phases too,
    #: so a caller must say *which* of those it found rather than flatten them.
    #:
    #: This is the only member left. `is_running` and `wake` were here and are
    #: not, because `_advance` stopped combining them into a branch — see
    #: `AttemptRunner.carry_on`. A Protocol declaring what its owner no longer
    #: calls is `interfaces.md` §4.12's other family, a capability reachable by
    #: nobody, and this package would rather not hold an instance of it.
    executor: Any


@runtime_checkable
class AttemptRunner(Protocol):
    """The part of `agent.Runner` a monitor uses — resolved as `runner`.

    Two verbs and no more. The scheduler holds a narrower protocol and sees
    neither of these; widening the one it holds would hand it verbs it has no
    use for.
    """

    def attempt_of(self, task_id: TaskId) -> Attempt | None:
        """The live attempt, or `None` once `stop` has removed it.

        **`None` does not mean "no thread"** — an attempt outlives its thread.
        That is `is_running`'s question.
        """
        ...

    def carry_on(self, task_id: TaskId) -> str:
        """The phase moved; take this attempt onward. Returns what it did.

        **The whole operation, in place of a predicate a caller had to combine
        with an action.** `_advance` used to read `is_running` for exactly one
        purpose — choosing between `wake()` and `resume()` — which is
        `engineer_principle.md` §3's stated symptom: *a caller that reads
        `a.b.c`, branches on it, and acts.* §4.4 says offer the computation
        instead, and this is that computation.

        It also closes the half of the check-then-act race that was dangerous,
        because the check and the wake happen under the attempt's own lock. That
        is a bonus rather than the reason: the shape argument stands with the
        race removed entirely.

        **A plain `str`, and the value is the contract** — `"woken"` or
        `"resumed"`. Not an enum, because this package may not import `agent`
        and a bare string compared against a member with `is` takes the wrong
        branch silently.

        Raises when no attempt is live. That raise is wanted: a missing attempt
        at a phase boundary means the task is stuck, and a no-op would make it
        silent.
        """
        ...


class Recorder(Protocol):
    """Persistence for events, over `task_graph`'s `StoreMgr`. Owns no policy.

    Two store kinds, because **absence is a signal**: a marker per attempt and
    one record per occurrence. A marker present with no occurrences reads as
    "nothing was recorded here"; a marker absent reads as "something is wrong".
    `handoff` applies the identical rule to `validation.yaml`.

    Append-only, one record per occurrence, never read-modify-write — a single
    container holding a list would put the runner's thread and the monitor's loop
    in a read/rewrite race that `JsonFileStoreMgr`'s per-record atomicity does
    not cover.
    """

    def open(self, task_id: TaskId, attempt: int) -> None:
        """Create the (empty) record set for one attempt. Idempotent."""
        ...

    def write(self, record: EventRecord) -> None: ...

    def read(self, task_id: TaskId, attempt: int) -> Sequence[EventRecord]:
        """Every record for that attempt, in written order. Empty if none."""
        ...

    def is_open(self, task_id: TaskId, attempt: int) -> bool: ...


class UserSink(Protocol):
    """Where an escalation goes when the task tree runs out.

    **How a monitor reaches a human is unspecified anywhere in this system.** The
    alpha registers a null implementation that records the arrival and does
    nothing else, so a branch that outruns every monitor is visible rather than
    silent. Inventing a channel here would be adding a requirement.
    """

    def deliver(self, record: EventRecord, why: str) -> None: ...


class Monitor(Protocol):
    """A task's event loop.

    **The inbound surface is `report` and `set_task`, and widening what arrives
    did not widen it** — which is the whole reason the two-channel routing is
    internal. `name`, `mainloop` and `stop` are lifecycle and identity: the
    composition root keys on `name`, and a monitor that could only be called
    could not notice a stall, which is the failure it exists for.
    """

    name: str

    #: Monotonic, stamped once per round. Read by a liveness check outside the
    #: loop, which is a comparison of one float and therefore small enough that
    #: nothing needs to watch *it* — the "make the top trivial" answer, against
    #: systemd's hardware watchdog and Ray's delegation to KubeRay.
    last_beat: float

    def set_task(self, task_id: TaskId) -> None:
        """Take this task under watch. The only way a monitor learns what it
        watches; it is told, and does not go looking."""
        ...

    def report(self, record: EventRecord) -> None:
        """Persist synchronously, then enqueue on the channel `record.kind`
        selects. Never blocks on a queue and never refuses a record.

        The two steps are one call so that "durable before queued" holds
        structurally: a caller that forgot the first of two would lose the event
        silently, which is the failure the rule exists to prevent.
        """
        ...

    def mainloop(self) -> None:
        """Drain both queues, planned first. Its own thread, never an agent's —
        giving the watched and the watcher one heartbeat is the failure a
        watchdog exists to avoid."""
        ...

    def stop(self) -> None:
        """Refuse new reports loudly, drain what is queued, then return."""
        ...
