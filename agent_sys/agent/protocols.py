"""What leaves `agent/`.

Two interface levels, kept apart as two protocols rather than one with holes in
it: level 1 is what a task runner talks to and every executor satisfies it; level
2 is the AI-harness abstraction and only an AI executor has one.

Declarations only. See `docs/interfaces.md` §4.4.
"""

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
    "TaskAttempt",
    "select_backend",
]


# --------------------------------------------------------------------------- #
# Errors


class BackendUnsupported(NotImplementedError):
    """An adapter does not implement a method it declared.

    Carries the backend key and the method name, so the error names the adapter
    that is incomplete rather than the interface.

    **This is about an incomplete adapter, not about an executor that has no
    level 2.** A program executor implements `Executor` and declares no
    `AgentBackend`, so there is no method to raise from — which is better than a
    raising stub, because a raise should mean *this adapter is incomplete* and a
    program is not an incomplete AI harness.
    """


class BackendUnavailable(RuntimeError):
    """Nothing in the chain could run here. Carries every rejection and its
    reason — you get the reasons or you get a broad catch, never both, unless the
    probe returns a structured result."""


# --------------------------------------------------------------------------- #
# Vocabulary


class Kind(str, Enum):
    """What an agent *is*. Main spec §4.8: every task has an agent, and `kind` is
    what varies. `human` is declarable and unimplemented — a `kind: human` spec
    loads and fails at *selection*, which is honest: the spec is well-formed and
    this alpha cannot run it."""

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
    """A named projection of the backend's result — never the backend's own
    message object.

    Criterion 16 is about the *system's* record, and a `ResultMessage` carries
    the final response text, structured output and permission denials. Exactly
    one of its fields is annotated "safe to log". Persisting the whole message
    would put prompt-derived text in the record; the adapter projects a subset.
    """

    status: AgentStatus
    usage: Mapping[str, float]
    detail: str


class AgentHistory(Protocol):
    """The backend's own data, fetched on demand and never stored.

    `entries` is deliberately untyped: giving it a schema here would make the
    backend's history the system's record, which spec §7 says it is not.
    """

    entries: Sequence[Mapping[str, Any]]
    session_ref: str | None


# --------------------------------------------------------------------------- #
# The two levels


class Executor(Protocol):
    """Level 1. What the task runner talks to, and every executor satisfies it —
    AI, human, or shell script."""

    status: AgentStatus

    def start_async(self, on_started: Callable[[], None]) -> None:
        """Return immediately. Invoke `on_started` when the agent *really* is
        running, which is later than "asked to start": deploying an environment
        and launching a harness takes long enough that the two are different
        events."""
        ...

    def wait(self) -> AgentResult: ...

    def start(self) -> AgentResult:
        """Sugar: `start_async` then `wait`."""
        ...

    def stop(self) -> None: ...

    def mainloop(self) -> None:
        """Drive this agent. Owns `status`, services the message queue, and is
        what `start_async` hands work to.

        **An agent is a live, stateful thing**, and the question that puts this
        here is the concrete one: `start()` returns immediately — then who is
        executing? An interface of five verbs with nothing behind them is not an
        interface.

        An adapter implements this. Nothing above level 1 calls it: the runner
        calls `start_async` and gets a callback. One thread per agent is the
        alpha's shape; attaching to a shared round-robin thread is roadmap.

        **The monitor's loop is a different loop with a different job** — it
        watches for the *task's* exceptions. Conflating them would give the thing
        being watched and the thing watching one heartbeat.
        """
        ...


class AgentBackend(Executor, Protocol):
    """Level 2. The AI-harness abstraction, and only AI executors have one.

    **The runner is typed against `Executor` and never against this.** Nothing on
    the `TaskRunner` path needs `interrupt`, `instruct` or `query`, so asking for
    level 2 would be asking for authority it does not use — and criterion 6
    becomes unwriteable-wrong rather than tested-right.
    """

    def interrupt(self) -> None:
        """End the current submission, keep the agent. **Not reachable from
        `TaskRunner`** — its caller is a monitor or an interactive surface."""
        ...

    def instruct(self, message: str) -> None: ...

    def query(self) -> AgentHistory: ...


# --------------------------------------------------------------------------- #
# Selection


class Rejection(Protocol):
    key: str
    reason: str


class Selection(Protocol):
    """Everything the selection learned, not just the answer.

    **Observing the selection must be separable from making it, from day one.**
    matplotlib learned it three times: `get_backend()` resolved *and committed*
    the choice, so a later `use()` became a silent no-op, and one of the three
    bugs destroyed the caller's open figures. All three are one shape —
    selection as a side-effecting operation disguised as a read.
    """

    backend: Executor
    """**Level 1, not `AgentBackend`.** A `kind: program` spec selects a
    `ProgramExecutor`, which has no level 2 by construction, and the runner
    holds level 1 only — so narrowing this to `AgentBackend` would make
    criterion 15 (*swapping the backend changes no other component*)
    unsatisfiable by type. Rev. 1 declared `AgentBackend` here; that was F5."""

    key: str
    source: Literal["cli", "spec", "config"]
    rejected: Sequence[Rejection]


def select_backend(
    spec: Any, *, override: str | None, config_order: Sequence[str], assignment: Any
) -> Selection:
    """Choose a backend, and report what was tried and why it was not chosen.

    Raises `BackendUnavailable` if nothing is usable.

    **`assignment` is required.** The probe is the constructor, so this is how
    an executor receives its `readme`, `entry`, `zone` and `environment`; a
    caller that omitted it would build an agent that starts and does nothing.
    Declared without a default per `interfaces.md` §4.11.

    **A CLI override does not fall through, and probes nothing else.** If it
    names something unusable that is an error, not a hint — "a run can be pinned
    when reproducing something" is worthless if the pin is advisory. matplotlib
    says it in a comment on the last line of `use()`: *do not helpfully
    fallback*.

    **Nothing is cached.** `env_mgr` deploys the environment, so a probe taken
    before deployment is taken at the one moment it is guaranteed to be wrong.
    """
    ...


# --------------------------------------------------------------------------- #
# The real TaskRunner, and the object one dispatch is carried on


class TaskAttempt(Protocol):
    """One attempt at one task, carried through its phases. Design §7.5.

    Maps 1:1 to the `Execution` the scheduler pushed, and holds the thread, the
    executor and which phase is next. **Declared here because `monitor` reaches
    it**: `monitor` design §6.1 duck-typed `attempt.executor` against nothing,
    which is a seam with one checkable side.

    **`executor` is `None` until the main phase begins**, and for a non-leaf it
    stays `None` — the scheduler runs a non-leaf's main phase by unfolding and
    the runner is never asked. A pusher must handle the `None`.

    An AI executor satisfies `monitor.protocols.Pushable` structurally, which is
    how the two packages talk without importing each other.

    **A present executor is not therefore pushable, and this said only the
    first half.** The type is `Executor` — level 1 — and a `ProgramExecutor` has
    no `instruct`, `interrupt` or `query` at all, deliberately
    (`backends/program.py`, spec §1.1 and §3.3.1: a raise should mean *this
    adapter is incomplete*, and a program is not an incomplete AI harness). So a
    pusher has **three** cases and this paragraph named two; the third is a live
    level-1 executor, and reading it as a `Pushable` is an `AttributeError` at
    the moment of the push rather than a capability answer. Measured, the
    honest question costs no import: `isinstance(executor, Pushable)` against
    the consumer's own runtime-checkable Protocol is `False` for a program
    executor and `True` for an AI one.
    """

    task: Any
    agent: Any
    executor: Executor | None

    @property
    def environment(self) -> Mapping[str, str]:
        """The **resolved configuration** this task was deployed with.

        Declared here because `validator` reaches it, for spec §8.2's producer
        row: at `OUTPUT_VALIDATING` a validation's default configuration is the
        validated task's. `_deploy` used to compute a `Prepared`, read four
        things off it and discard it, so nothing could reach it afterwards.

        A `Mapping`, not `env_mgr.Prepared`: `agent` may not import `env_mgr`,
        and the mapping is the whole of what was asked for.

        **Empty before the main phase**, like `executor` and for the same
        reason — it is computed inside `_deploy`. Read-only, because `env_mgr`
        hands out a `MappingProxyType` and downgrading that here would put one
        task's configuration one mutation away from another's.

        > **A configuration is not an environment.** `validator` spec §8.2:
        > *reusing a configuration is fine; inheriting an environment or a
        > conversation is not*, and criterion 21 makes a validation environment
        > a rebuild. Nothing here grants a zone, a handle, or a conversation.
        """
        ...

    @property
    def is_running(self) -> bool:
        """Whether a thread is currently carrying this attempt.

        **`executor` does not answer this**, and assuming it did was a live
        defect: it is `None` both for a non-leaf that released its thread and
        for a *parked leaf* between phases, because it is set inside the main
        phase. A caller that treated the two alike called `wake()` on an `Event`
        no thread was waiting on, and the task stalled with nothing reported
        (`monitor` design §6.1, measured).

        Kept because `carry_on` is built on it and either side's tests assert
        the two shapes through it. **A caller should reach for `carry_on`
        instead**: reading this to choose between two verbs is the branch
        `carry_on` owns, and doing it non-atomically.
        """
        ...

    def wake(self) -> None:
        """Resume a parked phase thread. The monitor's, after `enter_phase`."""
        ...

    def release(self) -> None:
        """End the thread. **The object survives**, and that is what it is for:
        a non-leaf's attempt outlives its subgraph and is re-entered by
        `Runner.carry_on` as the same `Execution`."""
        ...


class Runner(Protocol):
    """Satisfies `task_graph.TaskRunner`. Registered as `runner`.

    Resolves `agent_specs`, `env_mgr` and `phase_runner` from the registry by
    name at use time, and imports no backend. Runs three phases for the one task
    the scheduler dispatched, advancing by `task.enter_phase(next)` and **never**
    by assigning a status — a runner that assigned one would prove `FakeRunner`
    had been teaching the test suite a lie.

    **Two of these four are not on `task_graph.TaskRunner`, deliberately.** The
    scheduler holds the narrow protocol and sees neither `carry_on` nor
    `attempt_of`; widening it would hand the scheduler two verbs it has no use
    for, and one of them re-enters an attempt, which is precisely what it must
    not do (`task_graph` design §8.9). Their caller is the monitor.
    """

    def start(self, task: Any, agent: Any, on_done: Any) -> None: ...

    def stop(self, task_id: TaskId, on_stopped: Any) -> None: ...

    def attempt_of(self, task_id: TaskId) -> TaskAttempt | None:
        """The live attempt, or `None`. `monitor` design O1's accessor.

        It was not a missing accessor but a missing *object*; this returns the
        object, and `TaskAttempt.executor` is the live handle a pusher wants.
        """
        ...

    def carry_on(self, task_id: TaskId) -> str:
        """The phase moved; take this attempt onward. `"woken"` or `"resumed"`.

        **Prefer this to `is_running` plus `wake`.** A caller reading
        the predicate to choose between two verbs is doing a branch this owns
        (`engineer_principle.md` §3), and doing it non-atomically.

        Returns what it did — `"woken"` or `"resumed"`. **An observation, not a
        value to branch on**: `"resumed"` is a stand-in for *this was a
        non-leaf*, which is the same stand-in whose failure produced this verb.
        Safe to record or assert on; unsafe to decide with.

        **A plain string, not an enum**: the only caller may not import `agent`,
        and a bare string compared against an enum member with `is` silently
        takes the wrong branch — which is F3, one seam over.
        """
        ...
