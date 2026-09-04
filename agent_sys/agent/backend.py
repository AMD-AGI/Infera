"""The two interface levels, and the sugar layer every adapter shares.

**This module imports nothing from this repository** (`docs/design.md` §2), which
is what lets `selection`, `backends/` and `runner` all depend on it without a
cycle. It is two protocols, four value types and one base class.

Level 1 is `Executor`: what a task runner talks to, and every executor satisfies
it. Level 2 is `AgentBackend`: the AI-harness abstraction, and only an AI
executor has one. They are two protocols rather than one with holes in it, so
"a program executor never touches level 2" is a type rather than a raising stub
(design §5.1, D5).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

__all__ = [
    "TERMINAL",
    "ConfinementNotApplied",
    "AgentBackend",
    "AgentHistory",
    "AgentResult",
    "AgentStatus",
    "Assignment",
    "BackendUnsupported",
    "Executor",
    "ExecutorBase",
]


# --------------------------------------------------------------------------- #
# Errors


class ConfinementNotApplied(RuntimeError):
    """This executor cannot apply a rung-1 confinement, so the task must not run.

    A fourth reason a task does not start, beside `env_mgr`'s three, and it is
    here rather than there because the obligation is the **caller's**: on rung 1
    bubblewrap *is* the exec, so `apply()` confines nothing and whoever spawns
    the process builds the argv.

    **This is `spec.md` §3.3.1's shape and not the capability matrix it
    forbids** — a sentence that belongs next to the code, because the two look
    alike and a later reader will otherwise delete this citing the spec. §3.3.1
    is about a method a backend *declares* and cannot perform. This is
    `accept_confinement`, which every executor declares and which one kind of
    executor genuinely cannot honour: an AI harness spawns its own CLI, so
    there is no argv for anyone to wrap. An exception is one code path; a
    `supports_confinement` flag would be a branch at every call site, untested
    in exactly the configuration a site runs.
    """


class AgentNotListening(RuntimeError):
    """`instruct` was called on an executor whose loop has already returned.

    A `RuntimeError` rather than a new hierarchy, because the caller that needs
    it — `monitor.base._push` — already runs under `_run_guarded`'s broad catch
    and records the failure as an event. This exists to make the failure *have a
    type* at the seam, not to be caught specially anywhere.

    Raised where the alternative is a `put` into a queue nobody will read; see
    `ExecutorBase._enqueue_instruction` for the run this was measured on.
    """


class BackendUnsupported(NotImplementedError):
    """An adapter does not implement a method it declared, or cannot run here.

    Carries the backend key and the method name, so the error names the adapter
    that is incomplete rather than the interface (design §5.3).

    **This is about an incomplete adapter, not about an executor that has no
    level 2.** A program executor implements `Executor` and declares no
    `AgentBackend`, so there is no method to raise from.
    """

    def __init__(self, key: str, what: str, detail: str = "") -> None:
        self.key = key
        self.what = what
        self.detail = detail
        message = f"backend {key!r} cannot {what}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


# --------------------------------------------------------------------------- #
# Vocabulary


class AgentStatus(str, Enum):
    """Spec §4.3. `Task.status` is a superset of the stack-top agent's."""

    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


#: The three an agent does not leave. `mainloop` returns once one is reached.
TERMINAL: frozenset[AgentStatus] = frozenset(
    {AgentStatus.FINISHED, AgentStatus.FAILED, AgentStatus.INTERRUPTED}
)


class AgentResult(BaseModel):
    """A named projection of the backend's result — never the backend's own
    message object.

    Criterion 16 is about the *system's* record, and a `ResultMessage` carries
    the final response text, structured output and permission denials. The
    adapter projects a subset; §8.5 of the design says which.
    """

    status: AgentStatus
    usage: dict[str, float] = Field(default_factory=dict)
    detail: str = ""


class AgentHistory(BaseModel):
    """The backend's own data, fetched on demand and never stored.

    `entries` is deliberately untyped: giving it a schema here would make the
    backend's history the system's record, which spec §7 says it is not.
    """

    entries: list[dict[str, Any]] = Field(default_factory=list)
    session_ref: str | None = None


class Assignment(BaseModel):
    """What the runner hands an executor for the main phase — design §7.2.1.

    Rev. 5 of the design described the three phases in full and never said what
    the backend is given; §7.2.1 named the four things and left the call they
    arrive through unstated. They arrive here, at construction, because
    selection builds the executor and the probe is the constructor (§6.4).

    `readme` is the instruction for an AI backend and documentation a program
    does not consume; `entry` is what a `kind: program` executor runs.
    """

    readme: str = ""
    entry: str | None = None
    goal: str = ""
    zone: str = ""
    materials: tuple[str, ...] = ()

    #: What `env_mgr.material.deploy` computed — `CLAUDE_CONFIG_DIR`,
    #: `CLAUDE_CODE_TMPDIR`, `TMPDIR`, plus the agent spec's own `env`.
    #:
    #: **`CLAUDE_CONFIG_DIR` is the load-bearing one.** Measured: with `~/.claude`
    #: granted, a confined demo agent read the *operator's personal* `CLAUDE.md`
    #: and obeyed its language rule — a transcript that changes with whoever's
    #: dotfiles are on the machine. Pointing it into the zone is what removes the
    #: `$HOME` grant entirely.
    environment: dict[str, str] = Field(default_factory=dict)

    #: `env_mgr.remote.tools.ToolDef`s for this attempt's far side, or empty.
    #:
    #: **Spec §5.5 is why this is a field and not prose in the readme**: the
    #: remote surface reaches an agent as *tool calls*, because "an agent given a
    #: natural-language description of how to sync a directory will improvise,
    #: and the improvisation will be wrong in a way nobody notices". Until this
    #: field there was no route from `remote/tools.py` to any backend, so
    #: criterion 18 was built and reachable by nobody.
    #:
    #: Typed loosely for the same reason `confinement` is: `agent` may not import
    #: `env_mgr`. Each element has `.name`, `.description`, `.schema` and
    #: `.call`; a backend that cannot express tools ignores the field entirely,
    #: which is what every backend but `claude_sdk` does today.
    tools: tuple[Any, ...] = ()

    #: **External MCP servers this agent's components declared**, keyed by the
    #: name the model addresses them under. `env_mgr.Prepared.mcp_servers`,
    #: straight through.
    #:
    #: Typed loosely for `tools`' reason and one more of its own: the values are
    #: the *SDK's* server vocabulary — `{"type": "stdio", "command": …}` and the
    #: rest — so a type here would be `agent` declaring a shape it does not own
    #: and cannot check, on behalf of one of its backends.
    #:
    #: **A field and not prose in the readme**, for the reason spec §5.5 gives
    #: `tools`: an agent told in English to start an MCP server will improvise.
    #: A backend that cannot express external servers ignores this entirely,
    #: which is every backend but `claude_sdk` today.
    mcp_servers: dict[str, Any] = Field(default_factory=dict)

    #: `env_mgr.Confinement`, carried for an executor that wants to report what
    #: it will run under. Typed loosely because `agent` may not import `env_mgr`.
    #:
    #: **A prediction, not a report**, since `interfaces.md` split step 7:
    #: nothing has been applied when this arrives, and `spawn` realises it in
    #: the child. `mechanism`, `network`, `pid` and `abi` are all knowable at
    #: prepare time and are accurate — and the run does get them, because
    #: `spawn` cannot silently skip the confinement. So it is safe to record;
    #: what would be wrong is reading it as *already in force*.
    #:
    #: **The wrapper is deliberately not here.** It used to be, and that was the
    #: defect `closure`'s review found: a field an executor *may* read is a
    #: field an executor may silently *not* read, and the one that does not is
    #: the AI backend — the executor whose confinement matters most. It arrives
    #: through `accept_confinement` instead, which an executor that cannot
    #: honour it refuses.
    confinement: Any = None

    #: **What this task must deliver, as text for the agent** — each declared
    #: output, its kind, and where it goes. Empty for a task with no outputs and
    #: for a `kind: program` body, which is told by its environment instead.
    #:
    #: **Facts, never guidance**, which is the line `main` drew when ruling this
    #: onto the runner: the path is per-attempt and computed at dispatch, so no
    #: readme can name it and no package could. What the work *is* — the
    #: contract, what counts as done — stays in the readme, which is the
    #: package's to write and this field must never grow into.
    #:
    #: **Separate from `readme` on purpose.** `readme` means *the package's
    #: brief, as written*, and a program executor receives it too; machine text
    #: merged into it would make one field two things and leave no way to tell
    #: which half a reader is looking at.
    outputs_brief: str = ""

    #: `env_mgr.Prepared.agent_cli` — the CLI this environment was provisioned
    #: for, absolute, or `None` when the `Context` declared none. An AI backend
    #: pins it and **refuses** rather than letting the SDK pick a different
    #: binary (O2, `interfaces.md` §4.11).
    #:
    #: **A field rather than an entry in `environment`.** It was the latter for
    #: one revision, under a name `env_mgr` never published — so the lookup
    #: found nothing on every prepared run and the refusal fired every time. A
    #: declared field cannot drift that way: it is `Prepared`'s, `env_mgr` owns
    #: it, and the runner carries it across.
    agent_cli: str | None = None

    #: `env_mgr.Prepared.permissions_enforced` — **False unless the operator
    #: spelled the `AGENT_SYS_NO_PERMISSIONS` switch off** (`=0`). Off is the
    #: default since 2026-08-30; before that, False required setting it.
    #:
    #: An AI harness has a permission layer of its own, and it is not ours. With
    #: our enforcement switched off and the harness's left at its default, a run
    #: is **more** restricted than with permissions on: measured live, the agent
    #: could not `printenv`, could not `grep`, and could not `Write` inside its
    #: own zone — every tool call died at the SDK's ask-for-approval step with
    #: no approval channel to answer it.
    #:
    #: **Default `False` since 2026-08-30 — and here that is not "fails open".**
    #: The paragraph above is why: with *our* enforcement off and the harness's
    #: own layer left at its ask-for-approval default, an `Assignment` built
    #: without this field cannot run a single tool call. `True` was the safe
    #: direction while `True` was also the run's default; now it is the
    #: direction that silently breaks the agent, and `False` is the one that
    #: matches what `runner.py` actually passes on every real attempt.
    #:
    #: The safety property has not gone anywhere — it moved to where the fact
    #: is decided. `env_mgr.prepare.permissions_enforced()` is still the single
    #: reader, and an operator who spells the switch off gets enforcement in
    #: both layers, because `runner.py` passes the value rather than defaulting.
    permissions_enforced: bool = False


# --------------------------------------------------------------------------- #
# The two levels


@runtime_checkable
class Executor(Protocol):
    """Level 1. What the task runner talks to, and every executor satisfies it —
    AI, human, or shell script."""

    status: AgentStatus

    def start_async(self, on_started: Callable[[], None]) -> None:
        """Return immediately. Invoke `on_started` when the agent *really* is
        running, which is later than "asked to start"."""
        ...

    def wait(self) -> AgentResult: ...

    def start(self) -> AgentResult:
        """Sugar: `start_async` then `wait`."""
        ...

    def stop(self) -> None: ...

    def mainloop(self) -> None:
        """Drive this agent. Owns `status`, services the message queue, and is
        what `start_async` hands work to.

        The loop is the agent's; the thread is not — `TaskAttempt` owns it and
        the agent borrows it for the main phase (design §5.1.1, §7.5).
        """
        ...


@runtime_checkable
class AgentBackend(Executor, Protocol):
    """Level 2. The AI-harness abstraction, and only AI executors have one.

    **The runner is typed against `Executor` and never against this.** Nothing
    on the `TaskRunner` path needs `interrupt`, `instruct` or `query`, so asking
    for level 2 would be asking for authority it does not use — and criterion 6
    becomes unwriteable-wrong rather than tested-right.
    """

    def interrupt(self) -> None:
        """End the current submission, keep the agent. **Not reachable from
        `TaskRunner`** — its caller is a monitor or an interactive surface."""
        ...

    def instruct(self, message: str) -> None: ...

    def query(self) -> AgentHistory: ...


# --------------------------------------------------------------------------- #
# The sugar layer


class _Op(str, Enum):
    START = "start"
    INSTRUCT = "instruct"
    STOP = "stop"


#: How long `mainloop` sleeps between inbox polls. Small enough that a `stop`
#: arriving from another thread is observed promptly, large enough that an idle
#: loop is not a spin.
TICK = 0.02


class ExecutorBase:
    """What every adapter inherits: the loop, the queue, and the status field.

    **Every synchronous verb is sugar this level wraps** (spec §4.3, design
    §5.1.1). An adapter implements the asynchronous form — `_deploy`, `_run`,
    `_deliver`, `_terminate` — and never `start`, `wait` or `stop`. That is one
    rule rather than a per-method convention, and it is what keeps a backend
    from shipping a `stop()` that blocks differently from every other backend's.

    Not a Protocol and not exported across the seam: `Executor` is the type that
    crosses, and this is the shared implementation behind it.
    """

    def __init__(self, key: str, assignment: Assignment | None = None) -> None:
        self.key = key
        self.assignment = assignment or Assignment()
        self.status: AgentStatus = AgentStatus.PENDING
        self._inbox: queue.Queue[tuple[_Op, Any]] = queue.Queue()
        self._settled = threading.Event()
        self._result: AgentResult | None = None
        self._on_started: Callable[[], None] = lambda: None
        self._looping = False
        self._stopping = False
        self._spawn: Any = None

    # ---- what an adapter implements ------------------------------------- #

    def _deploy(self) -> None:
        """Bring the harness up. Returns when the agent really is running."""

    def _run(self) -> AgentResult:
        """Do the work of one submission and return its terminal result."""
        raise BackendUnsupported(self.key, "run a submission")

    def _deliver(self, message: str) -> None:
        """Hand a queued instruction to the live agent."""
        raise BackendUnsupported(self.key, "deliver an instruction")

    def _terminate(self) -> None:
        """Ask the harness to end. May be called from another thread."""

    # ---- rung 1, which the caller applies ---------------------------------- #

    def accept_confinement(self, spawn: Any) -> None:
        """Take responsibility for starting confined, or refuse.

        **The base refuses**, and every executor that does not spawn a command
        line of its own inherits the refusal. An AI harness spawns its own CLI,
        so a spawn handed to it would be ignored and the task would run with
        `prepare` reporting a sandbox that does not exist.

        `spawn(argv, **popen_kwargs)` is `env_mgr.Prepared.spawn` — one verb
        over three mechanisms, and the executor branches on none of them.

        **Every mechanism, not just bubblewrap**, since `interfaces.md` split
        step 7 (`b846c3c`): the runner's own process is deliberately left
        unconfined, so a child not started through `spawn` is not confined by
        inheritance either.

        **That is a wider statement and not a lost capability.** Pre-split,
        Landlock did confine the runner's thread — but only from a
        single-threaded caller, and `apply()` refuses above one thread, which
        this runner always is. So an AI task was never confined in any
        configuration that ran.

        **And it is in-process only.** `env_mgr` measured that a *grandchild*
        inherits, so a harness running inside a `spawn`-ed child would have its
        self-spawned CLI confined with nothing wrapping the CLI. What that
        costs is level 2: `interrupt`, `instruct` and `query` are built on an
        in-process client. Roadmap.
        """
        raise ConfinementNotApplied(
            f"backend {self.key!r} cannot start confined: it does not spawn a "
            f"command line of its own, so there is nothing for `Prepared.spawn` "
            f"to start and the sandbox would not exist. Refusing to start "
            f"(env_mgr criterion 14)"
        )

    # ---- level 1 --------------------------------------------------------- #

    def start_async(self, on_started: Callable[[], None]) -> None:
        """Queue one submission and return.

        A second call is a **second submission on the same agent** — Cursor's
        Agent/Run split, which spec §5.1 calls the most important structural
        lesson of the survey. It is how the completeness gate's cycle runs the
        agent again after a monitor pushed, without a new executor and without
        a new `Execution`.
        """
        self._on_started = on_started
        self._settled.clear()
        self._result = None
        self._inbox.put((_Op.START, None))

    def mainloop(self) -> None:
        """Drive this agent until it settles. Runs on the attempt's thread."""
        self._looping = True
        try:
            while True:
                try:
                    op, payload = self._inbox.get(timeout=TICK)
                except queue.Empty:
                    if self.status in TERMINAL or self._stopping:
                        break
                    continue
                if op is _Op.START:
                    self._service_start()
                elif op is _Op.INSTRUCT:
                    self._deliver(payload)
                elif op is _Op.STOP:
                    self._service_stop()
                if self.status in TERMINAL and self._inbox.empty():
                    break
        finally:
            self._looping = False
            self._settle(self.status)

    def wait(self) -> AgentResult:
        """Block until the loop settles. Somebody must be driving `mainloop`.

        **Unbounded, where `stop()` bounds itself, and the asymmetry is
        deliberate.** A bound here would be a timeout invented for an agent that
        may legitimately run for an hour; `stop()`'s bound is for a session that
        has already died. The shipped path cannot hang because the runner drives
        `mainloop()` itself before calling this — the hazard is only for a
        future third-party caller holding an executor, which is why it is said
        here rather than fixed with a number nobody can justify.
        """
        self._settled.wait()
        return self._result or AgentResult(status=self.status)

    def start(self) -> AgentResult:
        """Sugar: `start_async` then `wait`, lending this thread to the loop.

        A caller with no thread to spare for `mainloop` is the synchronous case,
        and it is the one this exists for. Criterion 7.
        """
        self.start_async(lambda: None)
        self.mainloop()
        return self.wait()

    def stop(self) -> None:
        """Sugar: request the stop, then wait for the loop to settle."""
        self._stopping = True
        self._terminate()
        self._inbox.put((_Op.STOP, None))
        if self._looping:
            self._settled.wait(timeout=_STOP_GRACE)
        else:
            self._service_stop()
            self._settle(self.status)

    # ---- internals ------------------------------------------------------- #

    def _service_start(self) -> None:
        """One submission, and **two failure paths signalled two ways** — which
        is deliberate rather than an oversight.

        A `_deploy` that raises is a harness that will not come up: that is not
        a *result*, so it is recorded and re-raised, and the attempt's outermost
        handler turns it into a reported failure. A `_run` that returns a
        `FAILED` result is a submission that completed and whose answer is "it
        failed" — an ordinary outcome, which goes on through the completeness
        gate like any other.
        """
        if self._stopping:
            return
        self.status = AgentStatus.DEPLOYING
        try:
            self._deploy()
        except Exception as exc:  # a harness that will not come up is a failure
            self._finish(AgentResult(status=AgentStatus.FAILED, detail=str(exc)))
            raise
        self.status = AgentStatus.RUNNING
        self._on_started()
        self._finish(self._run())

    def _service_stop(self) -> None:
        if self.status not in TERMINAL:
            self._finish(AgentResult(status=AgentStatus.INTERRUPTED, detail="stopped"))

    def _finish(self, result: AgentResult) -> None:
        self._result = result
        self.status = result.status

    def _settle(self, status: AgentStatus) -> None:
        if status not in TERMINAL:
            self.status = AgentStatus.FAILED
            self._result = self._result or AgentResult(
                status=AgentStatus.FAILED, detail="the loop ended before the agent settled"
            )
        self._settled.set()

    # ---- what the queue is for ------------------------------------------- #

    def _enqueue_instruction(self, message: str) -> None:
        """Level 2's `instruct`, as the queue operation it is. The loop
        delivers it; the caller does not touch the harness.

        **Refuses when no loop is running, and that refusal is the whole point.**
        `queue.Queue` is unbounded, so a `put` after `mainloop` has returned
        succeeds and the message is never read by anyone. Measured on
        2026-08-31: the agent finished, the seal was refused, the monitor pushed
        *continue, do it until finished*, `PUSH_ATTEMPTED` was written — and the
        message sat in this queue at `qsize=1` while the run hung for 65 minutes
        and was killed by the deadline.
        (`scratch/single-real-task-2026-08/probe_push_after_settle.py`.)

        **The cost of the silence, not of the failed push.** A push that cannot
        be delivered is an ordinary outcome — the agent settled first, which is
        a race the monitor cannot win from outside. What is not ordinary is
        reporting it as attempted and then waiting for ever for an answer that
        nothing will produce. One `put` into a dead queue converted a
        recoverable disagreement into a silent deadlock.

        **Raising is the sanctioned channel, not a new one.** `monitor/base.py`'s
        `_push` writes `PUSH_ATTEMPTED` *before* this call precisely so that "an
        `instruct` that raises still leaves the attempt visible" (criterion 9),
        and `_run_guarded` catches it and records `handling_failed`. So the
        caller was already built for this and the signature does not change; the
        path simply had no way to be taken.

        **The predicate is `_settled`, and `_looping` is the wrong one.** A first
        version refused whenever no loop was turning, and
        `test_instruct_does_not_end_run` caught it: `start_async()` then
        `instruct()` then `mainloop()` is a **shipped, legitimate order** — queue
        the work, then lend the loop a thread — and there `_looping` is false
        while the message is about to be consumed. "Has not started yet" and
        "has already finished" are different states and only the second is a
        fault. `start_async` clears `_settled` (`:389`) and `_settle` sets it
        (`:490`), so it is the flag that separates them.

        **One race is left and is named rather than papered over.** `mainloop`'s
        `finally` clears `_looping` before it calls `_settle`, so an `instruct`
        landing between those two lines is still queued and still lost. The
        window is a few instructions wide and its outcome is the old behaviour,
        not a worse one. Closing it needs the loop to drain and report on exit,
        which is a larger change than this defect justifies; `TODO.md` is where
        it belongs if it is ever seen in the wild.
        """
        if self._settled.is_set():
            raise AgentNotListening(
                f"{self.key!r}: instruct({message!r}) has no loop to deliver it. "
                f"The agent is {self.status.value} and `mainloop` has returned, "
                f"so this message would be queued and never read. Restart the "
                f"agent with `start_async` before instructing it."
            )
        self._inbox.put((_Op.INSTRUCT, message))


#: How long `stop()` waits for a running loop to settle before returning. A
#: bound rather than an indefinite wait, for the reason design §8.4 gives about
#: the interrupt drain: the case that most needs the call to return is the one
#: where the session has already died.
_STOP_GRACE = 30.0


def history_of(entries: Sequence[Mapping[str, Any]], session_ref: str | None) -> AgentHistory:
    """Build an `AgentHistory` without an adapter naming the model."""
    return AgentHistory(entries=[dict(e) for e in entries], session_ref=session_ref)
