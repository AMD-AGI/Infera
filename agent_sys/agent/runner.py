"""The real `TaskRunner` — design §7.

Spec §4.1: *"a middle-man between the task, the scheduler, and the agent … It
also runs the three phases"*. The runner orchestrates; the backend executes.

**Rev. 7: it is a factory and a registry, and it runs nothing itself.** `start`
creates one `TaskAttempt` per dispatch, keeps it, starts its thread, and
returns. The phases run on the attempt.

**`start` is called while the scheduler holds its `RLock`**, so what it does has
to be cheap. Creating and starting a thread is 71 μs measured, against a task
that will run an agent for seconds or minutes.

**The runner is typed against `Executor`, level 1, and never against
`AgentBackend`.** Criterion 6 becomes unwriteable-wrong rather than
tested-right: the runner cannot call an AI-only method because it does not hold
one.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agent.backend import AgentResult, Assignment, ConfinementNotApplied, Executor
from agent.gate import GateFailure, run_gate
from agent.selection import select_backend
from monitor import event, monitor_for
from monitor.protocols import EventKind
from spec_loader import body_of
from task_graph.ids import TaskId
from task_graph.models import HandoffStatus, Task, TaskStatus

__all__ = ["MonitorUnresolved", "Runner", "TaskAttempt", "ThreadAlreadyHeld"]

#: `validator.PhaseKind`, **as its value rather than as the member**. The seam
#: is `run_phase(kind: PhaseKind, task, registry)`, and `agent` may not import
#: `validator` (`docs/interfaces.md` §4.4, `tests/interfaces/test_import_rules.py`),
#: so there is no way to name the member here. `PhaseKind` is a `(str, Enum)`,
#: so `PhaseKind.INPUT == INPUT_PHASE` holds and a `run_phase` that compares by
#: value works; one that compares with `is` does not. **Reported in `README.md`
#: rather than worked around** — the fix is one side or the other, not here.
INPUT_PHASE = "input_validation"
OUTPUT_PHASE = "output_validation"

#: How long `begin` waits for a released thread to finish leaving. Bounded: the
#: thread is already returning, and a wait that could not end would turn a
#: handover into a hang.
HANDOVER_GRACE = 5.0

#: What `carry_on` did, as **plain strings rather than an enum**, and that is
#: F3's lesson applied before it could bite: `monitor` is the only caller and
#: may not import `agent`, so an enum member would reach them as a value
#: anyway — and a caller comparing a bare string against a member with `is`
#: silently takes the wrong branch. They only *record* it, so the value is the
#: whole contract.
WOKEN = "woken"
RESUMED = "resumed"

#: What `_report` moves out of its keyword arguments and into `attributes`.
#:
#: **Structured keys, never prose folded into the message.** `monitor` ruled the
#: shape (option 2): kinds name the *phase* a body terminated in — Erlang's
#: `Context` — and the cause is payload, so `exit_status` and `detail` ride here
#: rather than earning an `EventKind` of their own. `f"output absent ({detail})"`
#: would answer a human reading the file and nothing else.
#:
#: **Never put an enum member in a lifted value — pass `.value`.** This tuple
#: will grow, and `attributes` is `dict[str, Any]`, so neither pydantic nor the
#: store objects; the store narrows the type in passing. `monitor` measured the
#: round trip through `Recorder.write`'s `model_dump(mode="json")`: the member
#: goes in, the string comes back. A reader comparing against the member would
#: pass on the record it just built and fail on the one it loaded — and loading
#: is the path a reviewer is on.
#:
#: `seal_refused` carries why a pinned output would not publish, and it is
#: **provisional**: whether a seal refusal is an event kind, a verdict or a task
#: failure is `monitor`'s to settle with `agent`. It rides as an attribute so
#: that criterion 5's two cases — *never attempted* versus *wrote badly*, which
#: the store already distinguishes in its message — survive until then, without
#: an `EventKind` a later ruling would have to withdraw.
#:
#: `phase` is the phase that just finished, as `TaskStatus.name` — the key
#: `monitor.base.PHASE_ORDER` is built from. Lifted so `_advance` stops
#: re-deriving it from `task.status` after the fact, which raced `on_done` and
#: persisted a `HANDLING_FAILED` for every successful task.
#:
#: `nothing_to_attempt` is the gate's counterpart to `seal_refused`: the
#: producer was **never asked**, as against asked and refused. `monitor`
#: criterion 5 turns on keeping those apart, and `seal_refused`'s
#: *presence* is what says the producer was asked at all — so one key
#: cannot carry both.
_LIFTED = ("evidence", "exit_status", "detail", "seal_refused", "phase", "nothing_to_attempt")


class ThreadAlreadyHeld(RuntimeError):
    """`begin()` was called on an attempt that is already carrying a thread.

    One attempt, one thread — two would run its phases twice, report twice and
    call `on_done` twice.
    """


class MonitorUnresolved(RuntimeError):
    """`Task.monitor_spec` names a monitor the registry does not hold.

    `docs/interfaces.md` §2.1 rev. 4: such a task "never advances a phase". It
    fails loudly here instead, because a task that hangs for ever and a task
    that reports why are the same outcome told two ways, and only one of them
    is debuggable.
    """


class Runner:
    """Satisfies `task_graph.TaskRunner`. Registered as `runner`.

    Resolves, by name and at use time: `agent_specs`, `task_specs`, `env_mgr`,
    `phase_runner`, `handoff_store`, `handoff_mgr`, `budget`, `recorder`, and
    `monitor:<name>`. It imports no backend.

    **`handoff_mgr` is new in §2.1 rev. 6**, and this docstring said *"Not
    `handoff_mgr`"* until then — correctly, because nothing here used it. What
    changed is that the **agent-facing write path had no production caller at
    all**: `open_next`, `HandoffVersion.seal` and `HandoffMgr.persist` were
    reached only by `FakeRunner.produce`, a test double standing in for this
    runner. Criterion 14 says `persist` originates only from the agent, so the
    work could not go anywhere else.
    """

    def __init__(
        self,
        registry: Any,
        *,
        override: str | None = None,
        config_order: Sequence[str] = (),
        package_root: Path | None = None,
    ) -> None:
        """`package_root` resolves a task body's **package-relative** paths.

        `_common.schema.json` types `entry` as *"package-relative path to the
        entry.sh"*, and nothing else carries the package root into this package.
        `validator.PhaseRunner` takes it the same way for the same key, and
        `demo` F-D3 found that the two consumers of one schema key disagreed:
        theirs joined, mine did not, so a package that wrote the relative path
        the schema documents failed only under this executor and only at run
        time.

        An absolute path is unaffected — `Path("/a") / "/abs"` is `/abs` — so a
        package that renders its body paths absolute keeps working.
        """
        self._r = registry
        self.override = override
        self.config_order = tuple(config_order)
        self.package_root = package_root
        self._attempts: dict[TaskId, TaskAttempt] = {}
        self._lock = threading.RLock()

    # ---- the `TaskRunner` protocol ---------------------------------------- #

    def start(self, task: Task, agent: Any, on_done: Callable[..., None]) -> None:
        """Create the attempt and start its thread. Cheap, and it returns."""
        attempt = TaskAttempt(self, task, agent, on_done)
        with self._lock:
            self._attempts[task.id] = attempt
        attempt.begin()

    def stop(self, task_id: TaskId, on_stopped: Callable[[TaskId], None]) -> None:
        """The scheduler wants this task ended.

        **Not `interrupt`.** That is level 2, ends only the current submission,
        and has no `TaskRunner` route — its caller reaches it through
        `attempt_of` (design §7.4).
        """
        attempt = self.attempt_of(task_id)
        if attempt is not None:
            attempt.halt()
        with self._lock:
            self._attempts.pop(task_id, None)
        on_stopped(task_id)

    def shutdown(self) -> None:
        """Release every attempt when the process-level run is over.

        Attempts outlive their worker thread on purpose: a monitor can wake a
        parked task or submit to the same executor again while the run exists.
        Once all monitors have stopped and the CLI has produced its report,
        there is no remaining owner for those executors. Keeping them in
        `_attempts` leaked each Claude SDK reader task and subprocess transport
        until interpreter teardown.

        The map is emptied before any potentially blocking backend shutdown, so
        no concurrent observer can acquire an executor whose disposal started.
        Every attempt is given a chance to stop even when one backend raises;
        the first error is re-raised after all worker threads have been joined.
        """
        with self._lock:
            attempts = tuple(self._attempts.values())
            self._attempts.clear()

        first_error: Exception | None = None
        for attempt in attempts:
            try:
                attempt.halt()
            except Exception as exc:  # cleanup all owners before reporting one failure
                first_error = first_error or exc
        for attempt in attempts:
            attempt.join(HANDOVER_GRACE)
        if first_error is not None:
            raise first_error

    # ---- the monitor's two entrances, not on the protocol ----------------- #

    # `Runner.resume` is gone. `carry_on` subsumed it, `monitor`'s Protocol
    # shrink then stopped requiring it, and they enumerated all twelve of spec
    # §7.1's actions plus both open questions: **no path wants a bare re-entry
    # without the wake-or-resume decision.** A replacement monitor adopting a
    # dead one's tasks wants `carry_on` most of all, being the caller least able
    # to know whether a thread is parked.
    #
    # Removed on the plain ground that it was a published promise nobody made
    # use of. The stronger-sounding argument — that publishing it invites a
    # caller to decide which shape a task is — is real but bounded, because
    # `ThreadAlreadyHeld` makes a wrong re-entry loud; `monitor` said so against
    # their own case, and it is the more honest reason to record.
    #
    # **Not to be confused with the other `resume` in this system**: the
    # claude-agent-sdk session resume, the middle rung of push / resume /
    # restart, which costs ~5.5 s warm and loses `permission_mode`,
    # `--mcp-config`, `--settings` and `--add-dir`. That one is untouched and
    # lives in the backend.

    def attempt_of(self, task_id: TaskId) -> TaskAttempt | None:
        """Reach the live executor, for `instruct`. `monitor` design O1."""
        with self._lock:
            return self._attempts.get(task_id)

    def carry_on(self, task_id: TaskId) -> str:
        """The phase moved; take this attempt onward. Returns `WOKEN`/`RESUMED`.

        **The whole operation, instead of the predicate a caller had to combine
        with an action.** `monitor`'s `_advance` read `is_running` for exactly
        one purpose — choosing between `wake()` and `resume()` — which is
        `engineer_principle.md` §3's stated symptom: *a caller that reads
        `a.b.c`, branches on it, and acts*. §4.4 says offer the computation.

        **The shape argument stands on its own and the atomicity is a bonus**,
        which is `monitor`'s own correction to the position they first took. It
        also closes the dangerous half of the check-then-act race nobody could
        reproduce: the check and the wake happen under the attempt's lock, so a
        wake can no longer be lost to a thread that died between them.

        **The return is an observation, not a value to branch on.** It exists
        because a self-describing verb is what lets either side's tests assert
        which of the two shapes ran without reaching into the runner's threads —
        the only thing left to assert on otherwise, and internals.

        It was asked for so the outcome could go into the `PHASE_DONE` record's
        attributes, and **that turned out to be unimplementable**: `report()`
        persists before it enqueues (`monitor` spec §5.2 rule 3), so the record
        is on disk before this is called, and an append-only store has nothing
        left to amend. Kept anyway, for the reason above, with the original
        purpose recorded rather than quietly replaced.

        **Branching on it would be the proxy trap one step later.** `"resumed"`
        means a thread was taken, which is a stand-in for *this was a non-leaf*
        — the same stand-in whose failure produced this verb. It is safe to
        record and unsafe to decide with, and `monitor` design §6.1 is withdrawn
        rather than satisfied: the visibility it wanted was never in the branch.
        """
        attempt = self.attempt_of(task_id)
        if attempt is None:
            raise KeyError(f"no live attempt for task {task_id}")
        return attempt.carry_on()

    # ---- resolution, by name, at use time --------------------------------- #

    @property
    def registry(self) -> Any:
        """What `run_phase` is handed, so the phase reaches the managers."""
        return self._r

    def component(self, name: str, default: Any = None) -> Any:
        """A collaborator the runner can do without. `env_mgr` and
        `phase_runner` are absent in a unit test and their absence is a skip."""
        return self._r.get(name) if name in self._r else default

    def require(self, name: str) -> Any:
        """A collaborator whose absence is a wiring bug rather than a mode.

        `docs/interfaces.md` §2 registers it, so not finding it means the root
        was not run — and the failures that follow are silent ones.
        """
        return self._r.get(name)

    def monitor_for(self, task: Task) -> Any:
        """`Task.monitor_spec`, by name. Absent takes the default.

        **The rule is `monitor`'s and this delegates to it.** Rev. 1 had a
        second implementation, and the two disagreed about what an absent
        `monitor_spec` means: `monitor:default` there, `resolve("monitor:*")[0]`
        — whichever was registered *first* — here. Latent under
        `build_registry`'s own wiring, where only `default` exists, and live the
        moment anyone passes `monitors=[...]` with another name first, which it
        supports.

        It matters because the two resolutions are used at the two ends of one
        conversation: this one picks the monitor a phase is **reported to**, and
        `monitor`'s picks the one an escalation is **sent to**. Disagreeing
        gives one task two watchers, which is the thing `set_task` and the scope
        guard exist to prevent.

        Only the exception type stays ours, so a caller of the runner sees a
        runner error rather than a bare `KeyError` (`engineer_principle.md` §1:
        one writer for one fact).
        """
        try:
            return monitor_for(task, self._r)
        except KeyError as exc:
            raise MonitorUnresolved(str(exc)) from None

    def task_spec_of(self, task: Task) -> Any:
        """The task spec this task was instantiated from — design §7.2.1.

        **`task_specs`, not `closures`, and it is one lookup rather than two.**
        `closure/check.py` keys each closure's nested task spec into
        `task_specs` under the closure's own name, so `Task.closure` indexes it
        directly. `interfaces.md` §5.1b names this route — *"resolve them
        through `task.closure` → the task spec"* — and taking it means this
        package never touches a closure *document* and needs none of
        `closure`'s document accessors.

        **Reading a spec here is permitted.** Closure criterion 8's prohibition
        is on the *scheduler*; the narrower rule that holds is that the
        scheduler never reads a spec, and a task's runner may read the catalogue
        the task came from.
        """
        name = getattr(task, "closure", None)
        specs = self.component("task_specs")
        if not name or specs is None or name not in specs:
            return {}
        return specs.get(name)

    def resolve_path(self, declared: Any) -> str | None:
        """A package-relative body path against `package_root`. See `__init__`."""
        if not declared:
            return None
        if self.package_root is None:
            return str(declared)
        return str(self.package_root / str(declared))

    def agent_spec_of(self, task: Task) -> Any:
        return self._r.get("agent_specs").spec(task.agent_spec)


class TaskAttempt:
    """One attempt at one task, carried through its phases — design §7.5.

    Maps 1:1 to the `Execution` the scheduler pushed. Holds the thread, the
    executor, and which phase is next.

    **One attempt, possibly two threads.** A leaf's starts one and keeps it to
    `on_done`; a non-leaf's ends at `unfold` and takes another at `resume`.
    Neither is a second `Execution` — the parent was dispatched once, and
    `Execution.attempt` is what this object is named for.

    **No thread pool, and the reason is not the 50 μs.** The scheduler's
    resource leases are already this system's admission control; a pool smaller
    than the resource-permitted concurrency would leave a task with its lease
    taken, its status `RUNNING`, and its work sitting in a queue — a second
    admission policy the scheduler cannot see.
    """

    def __init__(
        self,
        runner: Runner,
        task: Task,
        agent: Any,
        on_done: Callable[..., None],
    ) -> None:
        self.runner = runner
        self.task = task
        self.agent = agent
        self.on_done = on_done
        self.executor: Executor | None = None
        self.usage: dict[str, float] = {}
        #: The resolved configuration, once `_deploy` has computed one. Empty
        #: before that, and empty is the honest answer — see `environment`.
        self._environment: Mapping[str, str] = _EMPTY
        #: Which outputs the *store* refused to publish, from the last pass.
        #: Read at close, so a refused artefact is never sealed `VALID` on the
        #: model — see `_seal_model_versions`.
        self._store_refusals: dict[Any, str] = {}
        #: Which outputs **this attempt** actually published to the store. The
        #: model seal requires membership here rather than absence from
        #: `_store_refusals`, because a skipped output is in neither.
        self._store_sealed: set[Any] = set()
        #: Whether this attempt holds an open model version. One writer, and it
        #: is what lets `run`'s `finally` close a slot `_close` never reached
        #: without double-sealing the one it did.
        self._model_open = False
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()
        self._halted = False
        #: Guards the thread's own bookkeeping — `_thread` and `_halted`, which
        #: `is_running` reads together. `Runner._lock` guards the *map*; this
        #: guards the *attempt*, and they are different facts with different
        #: writers: `release` runs on this attempt's thread, `halt` on the
        #: scheduler's, and `begin` on the monitor's.
        self._own = threading.RLock()

    # ---- lifecycle -------------------------------------------------------- #

    @property
    def environment(self) -> Mapping[str, str]:
        """The **resolved configuration** this task was deployed with.

        Requested by `validator` for spec §8.2's producer row: at
        `OUTPUT_VALIDATING` a validation's default configuration is the
        *validated task's*, and until now `_deploy` computed a `Prepared`, read
        four things off it and let it go — so after `_deploy` returned, nothing
        could reach it.

        **A mapping, not the `Prepared`.** `agent` may not import `env_mgr`, so
        carrying the tuple would put an `Any`-typed foreign object on this class
        and widen the seam to every field it has. The mapping is the whole of
        what was asked for.

        > **This is a configuration. It is not an environment, and the two words
        > collide here.** It holds what `material.deploy` computed —
        > `CLAUDE_CONFIG_DIR`, `CLAUDE_CODE_TMPDIR`, `TMPDIR`, and the agent
        > spec's own `env`. `validator` spec §8.2 is *"reusing a configuration is
        > fine; inheriting an environment or a conversation is not"*, and
        > criterion 21 makes a validation environment a **rebuild**. The hazard
        > is not reading this; it is a later reader treating *"I have the
        > producer's configuration"* as *"I may run in the producer's
        > environment"*. Nothing here grants a zone, a handle or a conversation.

        **Read-only, and empty before `_deploy`.** `env_mgr` hands out a
        `MappingProxyType` and this keeps that property rather than quietly
        downgrading it — a live `dict` here would be one task's configuration
        one mutation away from another's (`engineer_principle.md` §1).
        """
        return self._environment

    @property
    def is_running(self) -> bool:
        """Whether a thread is currently carrying this attempt.

        **The question `monitor` needs and could not ask.** Its `_advance`
        branched on `attempt_of(tid) is None` for "the non-leaf case: no live
        thread", and against this runner that is never true — `release()` ends
        the thread and keeps the object, and only `Runner.stop` drops it from
        the map. So the branch fell to `wake()`, which set an `Event` no thread
        was waiting on, and the parent entered `OUTPUT_VALIDATING` and never ran
        it. Measured by `monitor`'s `p7_nonleaf_wake_is_silent.py`: threads
        before 1, threads after 1.

        `executor` could not answer it — that is `None` for a *parked leaf*
        between phases too, because it is set inside `_main`.

        It is a property here rather than `_thread`/`_halted` published for an
        outsider to combine, which is `engineer_principle.md` §3: the attempt
        owns the thread, so it answers "am I holding one?" instead of handing
        over the parts. `halt()` sets `_halted` before the thread notices, so
        both terms are needed and only this object knows that.
        """
        with self._own:
            thread = self._thread
            return thread is not None and thread.is_alive() and not self._halted

    def begin(self) -> None:
        """Take a thread, **and refuse to take a second one.**

        Found by `closure`'s review, through `monitor`: this had no guard, so
        `Runner.resume` on a running attempt started a *second* thread on one
        attempt — measured at two live `attempt-*` threads, and both would run
        phases, both report, and both call `on_done`.

        **That is not a race and it is not unmeasured**: it is an unconditional
        missing precondition on a public verb, and it fires every time.
        `monitor` checks `is_running` first, but the check does not protect
        `resume`'s other callers and never protected `resume` itself.

        **Raising rather than no-op'ing**, for the reason `resume`'s own
        `KeyError` gives: a second `begin` means somebody believes this attempt
        is idle and it is not, and a silent no-op is how a wrong belief
        survives. The guard also converts one direction of the check-then-act
        race the same review found — a caller whose `is_running` went stale
        toward "not running" now gets a raise instead of a second thread.

        The other direction is not closed here; see `README.md` F11.
        """
        with self._own:
            if self.is_running:
                raise ThreadAlreadyHeld(
                    f"attempt for task {self.task.id} is already carrying a thread; "
                    f"a second would run its phases twice and call on_done twice"
                )
            previous = self._thread
            self._wake.clear()
            self._halted = False
            self._thread = threading.Thread(
                target=self.run, name=f"attempt-{self.task.id}", daemon=True
            )
            # `monitor/base.py`'s excepthook reads `thread.task_id` to attribute
            # a thread death to a task, and **nothing in production set it** —
            # every real thread death recorded `NO_TASK` and criterion 25's
            # attribution half was dead, with `monitor`'s own test the only
            # caller. The id is already in the thread *name*; passing it as the
            # declared attribute is the point, because recovering it from
            # `f"attempt-{...}"` would make a string format the contract.
            # **Set before `start`**: a thread that dies immediately must still
            # be attributable.
            self._thread.task_id = self.task.id  # type: ignore[attr-defined]
            self._thread.start()
        if previous is not None:
            # A released thread is on its way out and will not loop again —
            # `run` returns rather than re-reading `_halted`, which `begin` has
            # just cleared. Joining it anyway keeps "one thread per attempt"
            # true of the *process* and not only of the bookkeeping.
            previous.join(HANDOVER_GRACE)

    def run(self) -> None:
        """The thread's target: one phase, then report, then wait for the wake.

        **The `finally` closes the model slot, and it is not tidiness.** A
        version opened in `_main` is sealed in `_close`, and **a gate failure
        never reaches `_close`** — the same ordering that forced the store seal
        to precede the gate. Without this the slot stayed `GENERATING` for
        ever and the *next dispatch* raised `HandoffStateError: … is already
        open`, turning the ordinary failure — an agent that produced nothing
        usable — into a hard error on retry.

        Found by `task_graph` reading the plan rather than the code, and
        measured before it was fixed: the probe they prescribed is
        `test_a_gate_failure_does_not_deadlock_the_next_dispatch`.

        **A non-leaf is unaffected**: `_main` returns before `_open_outputs`,
        so nothing is open and `_close_model_slot` is a no-op — which is what
        keeps a released thread from sealing a version it never took.
        """
        try:
            self._open_recorder()
            while not self._halted:
                if not self._one_phase():
                    return
                if not self._await_wake():
                    return
        except BaseException as exc:  # noqa: BLE001 — a dead attempt must be reported
            self._crash(exc)
        finally:
            self._close_model_slot()

    def carry_on(self) -> str:
        """Wake this attempt if a thread is carrying it, otherwise take one.

        See `Runner.carry_on`, which is the caller-facing verb. **The decision
        and the wake are one critical section**; taking a thread is not, because
        `begin` joins a released predecessor and holding the lock across a join
        is how the dying thread's own `release` would deadlock against us.

        Leaving the lock before `begin` is safe rather than a smaller window:
        `begin` re-checks under it, so the worst case is a loud
        `ThreadAlreadyHeld` and never a second thread.
        """
        with self._own:
            if self.is_running:
                self.wake()
                return WOKEN
        self.begin()
        return RESUMED

    def wake(self) -> None:
        """The monitor, after `enter_phase`. **`carry_on` is the verb to prefer**
        — this one cannot tell a parked thread from a released one, so a caller
        choosing between it and `resume` is doing the branch `carry_on` owns."""
        self._wake.set()

    def release(self) -> None:
        """End the thread; **the object survives**, and that is what it is for."""
        with self._own:
            self._halted = True
        self._wake.set()

    def halt(self) -> None:
        """`Runner.stop`: end the executor and the thread."""
        with self._own:
            self._halted = True
        try:
            if self.executor is not None:
                self.executor.stop()
        finally:
            # A backend cleanup error must not leave the attempt parked forever.
            self._wake.set()

    def join(self, timeout: float | None = None) -> None:
        """For a caller that wants the thread settled. Tests, and `demo`."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # ---- the three phases ------------------------------------------------- #

    def _one_phase(self) -> bool:
        """Run whichever phase the task is in. `False` ends this thread."""
        phase = self.task.status
        if phase == TaskStatus.INPUT_VALIDATING:
            return self._validation(INPUT_PHASE)
        if phase == TaskStatus.RUNNING:
            return self._main()
        if phase == TaskStatus.OUTPUT_VALIDATING:
            return self._close()
        return False

    def _close(self) -> bool:
        """The last phase: run it, then call `on_done` once. The thread ends
        here either way, because a failed output phase is the monitor's to
        decide and not this runner's to retry.

        **The model version is sealed here, before `on_done`, and the order is
        load-bearing.** `on_done` reaches `close_execution` and `try_dispatch`,
        so a consumer can be dispatched inside that call — and its eligibility
        reads the model (`check_if_latest_valid` → `is_latest_valid` →
        `is_valid`), not the store. Sealing afterwards would race the consumer
        against its own input.
        """
        passed = self._validation(OUTPUT_PHASE)
        self._seal_model_versions(passed)
        if passed:
            self.on_done(self.task.id, TaskStatus.SUCCEEDED, dict(self.usage))
        return False

    def _validation(self, kind: str) -> bool:
        """A validation phase produces no output handoff — it calls
        `handoff.update_validation_status`, which is the handoff module's, and
        this runner does not read the verdict back. It owes the *transition*,
        not the report."""
        phase_runner = self.runner.component("phase_runner")
        if phase_runner is None:
            return self._report_planned()
        try:
            outcome = phase_runner.run_phase(kind, self.task, self.runner.registry)
        except Exception as exc:  # noqa: BLE001 — see `_unreached`
            return self._unreached(kind, exc)
        if outcome.blocks_the_task:
            self._report(
                EventKind.VALIDATION_FAILED,
                f"{kind} did not pass",
                evidence=_evidence(outcome),
            )
            return False
        return self._report_planned(evidence=_evidence(outcome))

    def _unreached(self, kind: str, exc: BaseException) -> bool:
        """A phase that **raised** reached no verdict. `monitor` spec §2.1.

        > *No verdict reachable — its `entry.sh` crashed, its agent died, its own
        > inputs were missing. Nothing was decided.*

        That is what `VALIDATION_UNREACHED` was always for, and `monitor` had to
        tell me so: I had mapped it onto `PhaseOutcome.empty`, which is *"there
        was nothing to check"* — a different sentence. Removing the wrong
        producer in F12 left the kind with no right one, and the cost was
        measured rather than argued: `validator/phase.py` raises in seven
        places, nothing here caught it, so it reached `_crash` and was reported
        `HANDLING_FAILED`. Their pusher routes `validation_unreached -> Escalate`
        and `handling_failed -> GiveUp`, so **a crashed validator died at its own
        monitor instead of walking `Task.parent` to the user** — the quietest
        possible dead branch, which is the defect §2.1 exists to close. And
        `HANDLING_FAILED` means *the monitor's handler raised*, so a wrong value
        was flowing on besides.

        **Caught by behaviour, not by type, and that is not a shortcut.**
        `monitor` suggested `except ValidatorInvalid`, and `agent` may not import
        `validator` — `docs/interfaces.md` §4.4 and
        `tests/interfaces/test_import_rules.py` — so naming it is not available
        and a string match on a class name would be worse than either. It is
        also the *better* rule: **any** exception out of `run_phase` means no
        verdict was reached, which is §2.1's sentence exactly, so this needs no
        answer to "is `ValidatorInvalid` the complete set". An unexpected error
        escalating rather than giving up is the safer direction too, and nothing
        diagnostic is lost — the type, the message and the stack all go into the
        record.

        `_crash` keeps `HANDLING_FAILED` for everything else, which is what that
        kind is for.
        """
        self._report(
            EventKind.VALIDATION_UNREACHED,
            f"{kind} reached no verdict: {exc}",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            exception_stacktrace="".join(traceback.format_exception(exc)),
        )
        return False

    # An empty phase advances, and the derivation for that now lives in
    # `validator.PhaseOutcome.blocks_the_task` rather than here — beside the
    # fold it depends on, where it can be maintained. `demo` argued for that
    # home and was right: `passed` answers *what the phase found*, this runner's
    # question is *may the task proceed?*, and the two coincide only when
    # something ran (`engineer_principle.md` §4.4).

    def _main(self) -> bool:
        """The one phase that touches a backend.

        **A non-leaf reaches none of this.** The scheduler runs its main phase
        by unfolding, so the attempt hands its thread back and the monitor calls
        `resume` when the subgraph finishes (design §7.2.1, §7.5).

        **It no longer places the container zone either.** That call was here and
        has moved to `Task.enter_phase`, whose docstring holds the derivation:
        `enter_phase(RUNNING)` unfolds *and submits*, and `submit` dispatches, so
        by the time the monitor woke this thread every child was already running
        and creating its own zone inside a parent zone that did not exist yet.
        `agent` cannot fix that from here — the ordering is decided in the
        monitor's thread, before this one is woken.
        """
        if self.task.has_subgraph():
            self.release()
            return False

        self.executor = self._deploy(self.runner.task_spec_of(self.task))
        # **Outside the loop, and two derivations agree on that.** `open_next`
        # sets GENERATING and refuses a slot already open, so a second pass
        # would raise; and one `Execution` means one version, which is what
        # settled the store side. The state machine and the ordering arrive at
        # the same place.
        self._open_outputs()
        while not self._halted:
            self.executor.start_async(lambda: None)
            self.executor.mainloop()
            result = self.executor.wait()
            self.usage = dict(result.usage)
            # **Before the gate, and that ordering is the whole ruling.** The
            # gate asks whether an output exists; the seal is what makes it so.
            refusals = self._store_refusals = self._seal_outputs()
            failures = self._gate(result)
            if not failures:
                return self._report_planned()
            for failure in failures:
                self._report(
                    failure.kind,
                    failure.message,
                    handoff_id=failure.handoff_id,
                    seal_refused=refusals.get(failure.handoff_id),
                    nothing_to_attempt=failure.nothing_to_attempt or None,
                    **_body_outcome(result),
                )
            # The runner never pushes. It reported; the monitor decides, and a
            # decision that says "try again" arrives here as a wake.
            if not self._await_wake():
                return False
        return False

    # ---- the main phase's four steps -------------------------------------- #

    def _deploy(self, spec: Any) -> Executor:
        """`env_mgr.prepare`, then selection, then the assignment.

        The order is the design: the environment is deployed *before* a backend
        is probed, because a probe taken before deployment is taken at the one
        moment it is guaranteed to be wrong (design §6.4).

        **Nothing `prepare` raises is caught here.** `NoConfinement`,
        `PrepareRefused` and `UnresolvedGrant` all mean the task does not start,
        and `env_mgr` criterion 14 is *no isolation, no start*. There is no
        `try` in this method: the attempt's own handler turns any of them into
        `on_done(FAILED)`, which is the task not starting, said out loud. What
        would delete the guarantee is a `try` that logged and *continued*, and
        `test_runner.py::test_a_refused_environment_never_reaches_an_executor`
        is what says there is not one.
        """
        agent_spec = self.runner.agent_spec_of(self.task)
        prepared = self._prepare(agent_spec)
        # `body_of` returns the mapping **as written**, so `{}` for a task that
        # declares none — falsy, and deliberately not a `Body(readme="")`, which
        # is truthy and reports a body that is present and empty.
        body = body_of(spec)
        staged = prepared.staged_package
        assignment = Assignment(
            readme=self._brief(body.get("readme"), staged),
            outputs_brief=self._outputs_brief(prepared),
            entry=self._body_path(body.get("entry"), staged),
            goal=str(spec.get("goal") or ""),
            zone=_zone_root(prepared),
            materials=tuple(str(m) for m in body.get("materials") or ()),
            environment=_environment(prepared),
            # Spec §5.5's tool surface, straight through. `getattr` because a
            # `Prepared` from before this field existed is still a valid one,
            # which is the same allowance every other optional field here gets.
            tools=tuple(getattr(prepared, "tools", ()) or ()),
            confinement=getattr(prepared, "confinement", None),
            agent_cli=prepared.agent_cli,
            # **Read, not inferred**, and that is the field's whole reason.
            # `confinement is None` means unconfined for two different reasons
            # — no mechanism on this machine, or the user's kill switch — and
            # an AI backend must tell them apart: with the switch on it also
            # stands down the *harness's* permission layer, and with it off it
            # must not.
            permissions_enforced=prepared.permissions_enforced,
        )
        # One writer, and it is here: the resolved configuration becomes
        # reachable at the same moment the executor is given it, so the attempt
        # and the assignment can never disagree about what this task ran with.
        self._environment = MappingProxyType(dict(assignment.environment))
        selection = select_backend(
            agent_spec,
            override=self.runner.override,
            config_order=self.runner.config_order,
            assignment=assignment,
        )
        _apply_confinement(prepared, selection.backend)
        return selection.backend

    def _outputs_brief(self, prepared: Any) -> str:
        """What this task must deliver, named in the one channel a model has.

        **Ruled onto the runner by `main`, with the line it drew:** *the runner
        states the facts only it possesses and does not author guidance.* The
        contract — what to write, what counts as grounded — stays in the readme.

        **Why not the readme.** `demo`'s first real model call produced nothing
        because the agent was never told where its output goes. The path is
        `<store>/<hid>/v<N>/content`, computed at dispatch and different every
        attempt, so no static text in a package can name it. The nearest a
        readme could get is naming `AGENT_SYS_OUTPUT_<KIND>` — a variable whose
        spelling is `env_mgr`'s, in prose nothing validates, that only helps an
        agent which chooses to run a shell and look. `demo-2`'s sentence is the
        argument: **an env var cannot instruct an agent.** A conversation is not
        a process reading `os.environ`.

        **An unresolved path is stated, never omitted** — `main`'s constraint,
        and it is `interfaces.md` §4.13's family: an agent told about two of
        three outputs writes two and finishes successfully, which is today's
        failure repeated one level up. The two cases that reach here without a
        path are a version `task_graph` did not pin and a kind `env_mgr` could
        not export, and neither is the agent's fault or the agent's to fix.
        """
        outputs = tuple(self.task.outputs)
        if not outputs:
            return ""
        paths = _output_paths(prepared)
        kinds = getattr(self.task, "kinds", {}) or {}
        lines = ["## What this task must deliver", ""]
        for hid in outputs:
            kind = kinds.get(hid) or "unknown kind"
            where = paths.get(hid)
            lines.append(
                f"- **{kind}** (`{hid}`): write it into `{where}`"
                if where
                else f"- **{kind}** (`{hid}`): **no resolved path** — this output "
                f"has nowhere to be written and cannot be delivered on this attempt"
            )
        return "\n".join(lines)

    def _body_path(self, declared: Any, staged: str | None) -> str | None:
        """A body path, resolved against **this attempt's** staged package.

        §4.16 copies the package into the zone and leaves the original outside
        every grant, so the path `Runner.resolve_path` produces now names a file
        the kernel refuses — `demo` measured `/bin/sh: cannot open …: Permission
        denied`, and only saw it because a failed body's output travels now.

        **`resolve_path` could not have been fixed in place.** `package_root` is
        a constructor argument and the staged copy is per *attempt*, so the
        value simply is not in scope there; it is in scope here, two lines after
        `prepare` returned. The fallback is still `resolve_path`: a runner built
        directly, and every test that prepares nothing, has no staged copy and
        the original tree is not confined away from it.

        An absolute declared path wins, which is `Path`'s rule and not a policy
        of mine — worth knowing because `demo`'s bodies were absolute until
        `086c12e`, and under that fill this resolution is a no-op rather than a
        wrong answer.
        """
        if not declared:
            return None
        if staged is None:
            return self.runner.resolve_path(declared)
        return str(Path(staged) / str(declared))

    def _brief(self, declared: Any, staged: str | None) -> str:
        """The agent's instructions — the **contents** of the readme, not its path.

        **`Assignment.readme` was the path for as long as it has existed**, and
        `backends/claude_sdk.py` hands it to the SDK as `system_prompt`. The
        schema is unambiguous that the declared value is a path
        (`_common.schema.json`: *"Package-relative path to the readme.md. For an
        agent task this IS the body"*), so a `kind: ai` task's brief was the
        path to its brief. It never crashed and no AI task has run, which is the
        whole reason it survived — `demo`'s `readme` note is what surfaced it.

        **A missing file raises rather than passing the string through.** The
        alternative is `interfaces.md` §4.11's named failure: the agent would
        receive a plausible prompt — its own path — and produce work nobody
        could tell from work done to a brief.
        """
        if not declared:
            return ""
        resolved = self._body_path(declared, staged)
        if resolved is None:  # pragma: no cover — `declared` is truthy here
            return ""
        path = Path(resolved)
        if not path.is_file():
            raise FileNotFoundError(
                f"the task body declares readme {declared!r}, which resolves to "
                f"{resolved} and is not a file. The readme is the agent's whole "
                f"brief, so this cannot fall back to the declared string"
                + (f" (staged package: {staged})" if staged else "")
            )
        return path.read_text(encoding="utf-8")

    def _prepare(self, agent_spec: Any) -> Any:
        env = self.runner.component("env_mgr")
        if env is None:
            return None
        return env.prepare(self.task, self.task.current, agent_spec)

    def _open_outputs(self) -> None:
        """Open this attempt's model slot per declared output — `GENERATING`.

        **The whole agent-facing write path had no production caller.**
        `open_next`, `HandoffVersion.seal` and `HandoffMgr.persist` were reached
        only by `FakeRunner.produce`, whose own docstring says it is *"the only
        thing in the test suite that writes handoff state"* and that it stands
        in for a real agent. So the model slot was never opened and never
        sealed, and a consumer waited for ever: `demo` measured the store
        holding `facts v0: SEALED, published, verdict PASS` while the model
        held `status=created, verdicts=0`.

        **Two version numbers for one artefact, and this is the other one**
        (`interfaces.md` §5.12). `_pin_outputs` allocates the *store* directory
        version at dispatch; this opens `HandoffMgr`'s *slot* version. They
        advance on different events and neither substitutes for the other.

        **Criterion 14 is why it is here**: `persist` must originate only from
        the agent, so `task_graph` may not do it and neither may `monitor` —
        `test_authority.py:235` asserts the scheduler calls neither verb.
        """
        mgr = self.runner.component("handoff_mgr")
        if mgr is None:
            return
        for hid in self.task.outputs:
            mgr.get(hid).open_next(self.task.id, self.agent.id)
            mgr.persist(hid)
        self._model_open = True

    def _seal_model_versions(self, passed: bool) -> None:
        """Seal the model slot with the verdict output validation reached.

        > **This is not the store seal.** `task_graph.HandoffVersion.seal` on the
        > *slot* version, after output validation. The other one is
        > `handoff.FilesystemStore.seal` on the *store directory* version, in
        > `_seal_outputs`, before the gate. **Two verbs, one name, two objects,
        > two moments, and this method is one of the two places in the system
        > where both appear.** Each is unambiguous inside its own package; the
        > collision exists here, so the disambiguation lives here.
        >
        > `interfaces.md` §5.12 is the same seam — two version numbers for one
        > artefact — and the names over them collide the same way. Raised by
        > `handoff` after checking the ordering **because the name made them**,
        > and finding nothing wrong: a thing that is correct but re-derived by
        > everyone who arrives costs more than a defect, which gets found once
        > and closed.

        **`seal` takes a verdict, so the event that decides it is the
        validation and not the write.** `VALID` is *"sealed, usable"* and
        `check_if_latest_valid` is what makes a consumer eligible — so sealing
        beside the store seal, in the main phase, would make a consumer
        eligible for an output that nothing has checked yet. That is the fault
        arm arriving through the back door.

        **A store refusal never yields a `VALID` model version, and this
        asserts it rather than relying on the ordering.** Today the gate
        reports `OUTPUT_ABSENT` for a refused seal and `_close` is not reached,
        so the case is already impossible — which is exactly the kind of
        guarantee that stops holding when someone reorders. If the store did
        not publish it, the model does not call it usable.

        `INVALID` rather than leaving the slot `GENERATING`: a hole is the
        honest record, and `open_next` refuses a slot someone else has open, so
        a re-dispatch would raise instead of appending `v+1`.
        """
        mgr = self.runner.component("handoff_mgr")
        # `_model_open` is what makes this callable twice. `run`'s `finally`
        # closes any slot `_close` did not, and on the success path `_close`
        # has already sealed — a second `seal` on a sealed version raises.
        if mgr is None or not self._model_open:
            return
        self._model_open = False
        for hid in self.task.outputs:
            # **Positive evidence, not the absence of a refusal.** `published`
            # used to mean `hid not in self._store_refusals`, which is true for
            # an output this attempt never even tried — `_seal_outputs` skips a
            # hid with no pinned version, and a skip is not a refusal. Measured:
            # with nothing pinned and a *previous* attempt's version in the
            # store, the gate passed on that older version, `_close` ran, and
            # the slot was sealed `VALID` for an attempt that published
            # nothing. `exists()` is not attempt-scoped; this set is.
            status = (
                HandoffStatus.VALID
                if (passed and hid in self._store_sealed)
                else HandoffStatus.INVALID
            )
            mgr.get(hid).latest.seal(status)
            mgr.persist(hid)

    def _close_model_slot(self) -> None:
        """Seal `INVALID` if the attempt ended with a version still open.

        The attempt is over and nothing validated the output, so the honest
        record is a hole. `_seal_model_versions` is idempotent through
        `_model_open`, so the success path — where `_close` already sealed —
        reaches here and does nothing.
        """
        self._seal_model_versions(passed=False)

    def _seal_outputs(self) -> dict[Any, str]:
        """Publish what the body wrote. **Ruled: here, and before the gate.**

        Returns `{handoff_id: refusal}` for the ones that would not publish.

        > **This is not the model seal.** `handoff.FilesystemStore.seal` on the
        > *store directory* version, before the gate. The other one is
        > `task_graph.HandoffVersion.seal` on the *slot* version, in
        > `_seal_model_versions`, after output validation. **A second `seal`
        > running later in this flow is not the store seal having moved** — the
        > two are different objects at different moments, and only their names
        > agree.

        **Why the runner and not `task_graph`.** The symmetric answer — *the
        allocator seals* — cannot be adopted: `task_graph` pins the version at
        dispatch (`scheduler.py:280`) and would seal at close, and **close is
        after the gate**. The gate asks whether an output exists, and
        `FilesystemStore.exists` means *published* — its own docstring says an
        allocated-but-unsealed directory is not a version that exists. So with
        nothing sealing before the gate, **every successful task reported
        `OUTPUT_ABSENT`**, which is what `demo` measured.

        **The caller needs no evidence, and that was the question worth
        asking.** `seal` is not a rubber stamp: it re-runs `put`'s admission
        checks, and it tests `content/` for **emptiness before contents** — so
        a body that exited 0 having written nothing gets back the reason
        *this attempt produced no content at all*, not a manifest. Sealing on
        the wrong evidence would erase the signal `monitor` criterion 5 depends
        on; the store is what prevents that, which is the answer arriving from
        the party that owns it.

        **A refusal is not an error here.** It leaves the version unsealed,
        which is exactly a hole, and the gate then reports the absence — the
        truth about the attempt. Raising instead would fail the attempt through
        the outermost handler and the gate would never speak.

        **There is no `try` here, and the absence is the contract.** This
        wrapped `store.seal` in `except Exception` for one commit, because
        `agent` may not import `handoff` (`docs/interfaces.md` §4) and there was
        no type to name. `handoff` answered the constraint by changing the
        boundary instead (`fd31a6c`): `seal` **returns** the reason and raises
        `NotSealable` — a wiring bug — and nothing else.

        **Against the new contract the catch was exactly backwards**: a refusal
        no longer raises, so `refused` would have stayed empty and the reason
        would have reached no record, while `NotSealable` — the one thing that
        must escape — would have been swallowed into it. Both halves silent, on
        a suite that stayed green because no test sealed twice.

        **`Malformed` is not a refusal either, and catching it would be the
        same mistake one type over.** Inside `seal` it is raised in exactly one
        place — a store built without a `KindSource` (`handoff/store.py:383`) —
        which is a composition error, not a fact about the producer. It escapes
        for the same reason `NotSealable` does. An earlier version of this
        docstring called the empty-content refusal a `Malformed`; `monitor`
        caught that, and it was left over from before `fd31a6c` moved the
        boundary.

        **Ruled by `monitor`: the attribute is right, and there is no new
        `EventKind`.** Kinds name the *phase* a body terminated in and causes
        ride in the payload, so a seal refusal is not a different phase — it is
        *why* the output is missing, the same shape as `exit_status`. The
        implication is one-way, which is what makes payload the correct home:
        a refusal always yields absence, but absence does not imply a refusal
        (a never-pinned output is absent with nothing refused). So **the
        attribute's presence carries criterion 5's *attempted versus never
        attempted*, and its content carries *wrote nothing versus wrote
        badly*.** A kind would be a synonym for `OUTPUT_ABSENT` plus a cause,
        and would then need a second kind to preserve the store's own
        distinction. Not a task failure either — an unpublishable attempt is
        not exceptional — and not a verdict, which is a roadmap risk nobody has
        opened.

        **A seal refusal is not a gate failure**, even though the two arrive
        adjacent. `refusals.get(failure.handoff_id)` keys the reason to the
        *same output* the finding is about, so it qualifies the finding it
        caused rather than being welded onto an unrelated record.

        **`seal_refused` has no reader outside these tests yet**, and that is
        deliberate: declaring a predicate for it now would be `interfaces.md`
        §4.12, a capability reachable by nobody. Criterion 5 makes the
        distinction normative, so when a consumer appears, `monitor` declares
        the key and the predicate — the way `reached_the_user` was done —
        rather than anyone string-matching an optional attribute.
        """
        store = self.runner.component("handoff_store")
        if store is None:
            return {}
        versions = dict(self.task.current.output_versions)
        refused: dict[Any, str] = {}
        for hid in self.task.outputs:
            version = versions.get(hid)
            if version is None:
                # Nothing was pinned for this output, so there is no directory
                # to seal. The gate reports the absence.
                continue
            reason = store.seal(hid, version, producer=self.task.id)
            if reason:
                refused[hid] = str(reason)
            else:
                # What this attempt actually published. `_seal_model_versions`
                # requires it — absence of a refusal is not evidence of a seal.
                self._store_sealed.add(hid)
        return refused

    def _gate(self, result: AgentResult) -> list[GateFailure]:
        """**No early return on a missing store**, and it cost two checks.

        This used to be `if store is None: return []`, which skipped not only
        the output questions but `_budget` as well — so a storeless run checked
        **nothing**, and a task that declared outputs and published none was
        reported succeeded. The gate decides what a missing store means; the
        runner's job is to ask it either way.
        """
        store = self.runner.component("handoff_store")
        return run_gate(
            list(self.task.outputs),
            result.usage,
            store=store,
            budget=self.runner.component("budget"),
        )

    # ---- reporting, which is the only thing that follows a phase ---------- #

    def _report_planned(self, **attributes: Any) -> bool:
        """A phase that finished normally. **The runner does not branch on which
        monitor call to make** — it makes one, with a different kind.

        **`phase` is lifted out of the prose because prose is the part that
        rots** — the same reasoning as `exit_status`. `monitor`'s `_advance`
        re-derived the finished phase by reading `task.status` *afterwards*,
        and `_close` reports a third `PHASE_DONE` for a three-phase order, so
        which error it raised depended only on whether it drained before
        `on_done` landed. A `HANDLING_FAILED` was persisted for every
        successful task. The value here is read **before** `on_done` runs, so
        it is the phase that actually finished rather than a race.

        **`.name`, not `.value`, and that difference is load-bearing.**
        `monitor` asked for `.value` and compares against
        `PHASE_ORDER = ("INPUT_VALIDATING", "RUNNING", "OUTPUT_VALIDATING")` —
        which is keyed by `.name`, as their own `next_phase` shows
        (`monitor/base.py:80`, `PHASE_ORDER.index(status.name)`). Measured:
        `"output_validating" == "OUTPUT_VALIDATING"` is `False`, so the fix
        they proposed would never have matched and the terminal phase would
        still have been treated as an advance. Two sides, one name, and only
        one of them checked — today's recurring shape.

        A plain string either way, so the round-trip rule `_LIFTED` states is
        satisfied: nothing here is an enum member.

        `setdefault` rather than a keyword, because `_validation` already
        passes `evidence=` and a second caller may pass `phase=` deliberately.
        """
        attributes.setdefault("phase", self.task.status.name)
        self._report(EventKind.PHASE_DONE, f"{self.task.status} finished", **attributes)
        return True

    def _report(self, kind: EventKind, message: str, **extra: Any) -> None:
        """One call site, and `kind` is the only thing that differs.

        `evidence` rides in `attributes` rather than being folded into the
        message: `validator` criterion 19 makes a phase whose passes are all
        `weak` `LOW_CONFIDENCE` rather than `ESTABLISHED`, and carrying only
        `passed` loses exactly what pytest's XPASS lost — distinguishable
        per-item rendering, exit code 0, green bar.

        **The lift below is required, not tidiness.** `EventRecord` inherits
        `extra="forbid"`, so **any `**extra` key that is not one of its fields
        raises** — deliberately, because a typo'd field silently dropped is what
        `forbid` exists to prevent. A second attribute needs the same treatment
        and the tuple already has the shape for it. (`monitor` measured the
        raise while checking whether this was a defect; it is not.)

        **The pop is unconditional and the `None` check follows it**, which the
        first version had the other way round: a lifted key passed as `None`
        stayed in `extra`, reached `event()` as a keyword `EventRecord` does not
        declare, and `forbid` raised on the very case the caller meant as "there
        is nothing to add".
        """
        attributes: dict[str, Any] = {"message": message}
        for key in _LIFTED:
            lifted = extra.pop(key, None)
            if lifted is not None:
                attributes[key] = lifted
        monitor = self.runner.monitor_for(self.task)
        monitor.report(
            event(
                kind,
                self.task.id,
                attempt=self._attempt_number(),
                agent_id=getattr(self.agent, "id", None),
                reported_by="agent.Runner",
                attributes=attributes,
                **extra,
            )
        )

    def _crash(self, exc: BaseException) -> None:
        """A dead attempt is reported, and closed **with a reason**.

        `Execution.detail` is *"from the runner; for a human"* and was empty for
        every failed task in the system — `demo` measured `detail=''` on a real
        run while the same exception sat complete in the monitor's record.
        **A failure that is recorded somewhere is not the same as a failure that
        is reported**, and the gap is paid by whoever holds the artefact rather
        than the source.

        The exception was in hand two lines above and the scheduler had always
        taken the argument; what was missing was a *type* that could express it.
        `task_graph` widened `OnDone` from a `Callable` alias to a Protocol
        (`f1faf74`) rather than my passing an undeclared keyword, which would
        have worked in production and broken every conforming callback.

        **Reporting is itself allowed to fail** — an unresolvable monitor is
        exactly the case that gets here — and the task must still be closed, or
        the scheduler waits on a completion nobody will send.
        """
        try:
            self._report(
                EventKind.HANDLING_FAILED,
                str(exc),
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                exception_stacktrace="".join(traceback.format_exception(exc)),
            )
        except Exception:  # noqa: BLE001, S110 — see the docstring
            pass
        finally:
            self.on_done(self.task.id, TaskStatus.FAILED, dict(self.usage), detail=_one_line(exc))

    # ---- internals -------------------------------------------------------- #

    def _await_wake(self) -> bool:
        self._wake.wait()
        self._wake.clear()
        return not self._halted

    def _attempt_number(self) -> int:
        current = self.task.current
        return current.attempt if current is not None else 0

    # `Monitor.set_task` is **not** called here. `main` ruled it into
    # `Scheduler._dispatch_pass` (`interfaces.md` §2.1 rev. 5) after three
    # rulings and a period where `task_graph` and `agent` had both built it —
    # safe only because `_Watch.add` dedupes, which neither of us designed for.
    #
    # The argument that settled it is about the *interface* rather than either
    # site: `TaskRunner` declares `start` and `stop` and says nothing about
    # monitoring, so a `set_task` inside a runner implementation is a
    # per-implementation obligation of a contract that does not state it — and
    # the registered default is `FakeRunner`, which would not discharge it.
    #
    # Two sentences worth keeping, which three of us reached independently:
    # `set_task` is idempotent, so a re-dispatch needs no guard; and an
    # unresolvable `monitor_spec` raises, because that is `monitor_for`'s to say.

    def _open_recorder(self) -> None:
        """`Recorder.open(task_id, attempt)` — `monitor` design §9.2's obligation.

        **On the attempt's thread, not in `__init__`, and that is a measurement
        rather than a preference.** `__init__` runs inside `Runner.start`, which
        the scheduler calls while holding its `RLock`; `monitor` measured the
        call at **102 µs against `JsonFileStoreMgr`** and 5.1 ms for a fifty-task
        unfold — a third of the whole lock hold, and the only term in it that is
        a filesystem write, which is the one that degrades badly on a slow or
        networked store. Here it holds nothing and the *n* of them stop being
        serialised behind each other.

        The obligation survives the move: `open` is idempotent, so `resume()`
        taking a second thread is free, and criterion 14 still holds — an
        attempt whose thread ran and reported nothing has its marker. The window
        the move opens, where a report could arrive before the marker, is closed
        on the other side: `Recorder.write` calls `open()` itself.

        **Absent is a wiring bug, and it is loud.** `docs/interfaces.md` §2
        registers `recorder`, so a missing one silently voids criterion 14's
        empty-versus-missing distinction — the same argument as
        `MonitorUnresolved`.
        """
        self.runner.require("recorder").open(self.task.id, self._attempt_number())


def _one_line(exc: BaseException) -> str:
    """The exception as `Execution.detail` should carry it.

    **`type(exc).__name__: exc`, not `str(exc)`**, and that is measured rather
    than preferred: `str(KeyError('agent'))` is `"'agent'"` — a bare quoted word
    with nothing saying it is a lookup failure, in the field a human reads
    first because it is on the task.

    It is also exactly the join of `exception_type` and `exception_message`,
    which `_report` already puts in the monitor's record — so the one-line form
    on the `Execution` and the structured form in the recorder are **one fact
    in two renderings rather than two facts**.
    """
    return f"{type(exc).__name__}: {exc}"


def _evidence(outcome: Any) -> str | None:
    """`PhaseOutcome.evidence` as its value — `nothing_ran` / `failed` /
    `low_confidence` / `established`. `None` from a phase runner that predates
    the field, which is not an error here."""
    found = getattr(outcome, "evidence", None)
    return getattr(found, "value", None) if found is not None else None


def _apply_confinement(prepared: Any, executor: Any) -> None:
    """Hand the confinement to the executor, **after selection**, and let it refuse.

    **`interfaces.md` split step 7** (`b846c3c`): `prepare` now *checks* that a
    mechanism exists and refuses early; `prepared.spawn(argv, **kw)` *applies*
    it in the child. So the executor is started confined rather than started
    into a confinement, and the caller branches on no mechanism.

    **That widens the refusal, and the widening is the point.** Rev. 3 asked
    only about bubblewrap, because on Landlock `prepare` had already confined
    the runner's thread and a child inherited the domain. It no longer does —
    deliberately, since confining the supervisor is what the split exists to
    avoid — so **a child not started through `spawn` is unconfined under every
    mechanism**, not just rung 1. An AI harness spawns its own CLI, so a
    `kind: ai` task is now unconfinable under Landlock too. `README.md` F8.

    **Three earlier versions were wrong in the same direction**, and the
    progression is the record: *"does a wrapper exist?"* (refused the honest
    case, admitted the dishonest one, and `wrap_argv` landing silently disarmed
    it because a bound method is truthy); *"is `AgentSpec.kind` ai?"* (the kind
    is a proxy for the executor and a CLI override breaks the proxy —
    measured, `probe_r1_override.py`); and now this, which asks the executor.
    `ExecutorBase.accept_confinement` refuses by default — spec §3.3.1's own
    shape, an unimplementable method raising, and not the capability matrix
    §3.3.1 forbids.

    **`bwrap` is absent on this machine and Landlock leaves the process
    unconfined**, so neither suite exercises the live path either way. Every
    version of this has been a refusal rather than a warning for that reason.
    """
    # **Read, not asked for.** `Prepared` declares `confinement`, so a
    # `getattr` default here would answer "no confinement" to a *missing field*
    # — the one answer that must never be guessed, because its consequence is a
    # task starting unconfined. A probe over the suite showed the default arm
    # taken 34 times, all of them by a thin stub and none by production.
    if prepared.confinement is None:
        return
    spawn = getattr(prepared, "spawn", None)
    if spawn is None:
        raise ConfinementNotApplied(
            "the environment reports a confinement and exposes no `spawn` to "
            "start the executor through — refusing to start rather than run "
            "unconfined (env_mgr criterion 14)"
        )
    executor.accept_confinement(spawn)


#: What `TaskAttempt.environment` answers before `_deploy` has computed one.
#: A shared read-only empty, so an attempt that has not deployed cannot be told
#: apart from one that deployed with nothing — which is true, and is the answer.
_EMPTY: Mapping[str, str] = MappingProxyType({})


def _output_paths(prepared: Any) -> Mapping[Any, str]:
    """`Prepared.output_paths` — slot → `<store>/<hid>/v<N>/content`.

    **Read, not asked for.** This was a `getattr` default for one commit, while
    `env_mgr` had not landed the field, and it was defensible only because an
    absent mapping renders every output as *no resolved path* — louder than the
    truth rather than quieter. The field exists now (`1e82d13`), so the default
    would answer *"nothing resolved"* to a **missing field**, which is the shape
    this package has deleted four times this week.

    **Absence within the mapping still means something**, and it is `env_mgr`'s
    statement rather than a gap: a slot with no pinned version is absent rather
    than present-and-empty, so the difference against `task.outputs` is
    unambiguous and each side reads only what it owns.
    """
    return prepared.output_paths


def _body_outcome(result: AgentResult) -> dict[str, Any]:
    """How the executor's own submission ended, for a gate report's payload.

    **The runner read `result.usage` and nothing else**, so a `kind: program`
    body that exited 3 with a traceback produced a perfectly good `FAILED`
    result that no reader ever saw: all that travelled was the gate's
    `output_absent`, which is the same observation for a body that crashed on
    line 1, a body that exited 0 having written to the wrong path, and a body
    never launched. `demo` measured the cost of that at about an hour.

    Two of those three separate here — `exit_status` tells them apart, and
    `detail` carries what the body said before it stopped (`2d33282`). The third
    never reaches the gate at all and lands as `HANDLING_FAILED`; it is a
    different route and not addressed by this.

    `status.value` rather than the member, because an attribute is data a reader
    renders and this record is persisted, not logged.
    """
    return {"exit_status": result.status.value, "detail": result.detail or None}


def _environment(prepared: Any) -> dict[str, str]:
    """What `material.deploy` computed.

    **F7 is closed and this docstring said otherwise for weeks**: it claimed
    `Prepared` was a five-field `NamedTuple` awaiting a sixth. Ruling 2 landed
    `environment`, and the real `Prepared` has all six — so the defensive read
    was a fallback whose reason had expired, and the empty dict it produced was
    a legal value that would have hidden the field going away again.
    """
    found = prepared.environment
    return {str(k): str(v) for k, v in found.items()} if found else {}


def _zone_root(prepared: Any) -> str:
    return str(prepared.zone.root or "")
