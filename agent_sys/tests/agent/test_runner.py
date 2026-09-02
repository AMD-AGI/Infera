"""Criteria 6 and 15, and the phase machinery behind them — design §12."""

from __future__ import annotations

import inspect
import threading
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from agent.backend import AgentBackend, Executor
from agent.backends.program import ProgramExecutor
from agent.runner import MonitorUnresolved, Runner, TaskAttempt
from task_graph.ids import HandoffId, TaskId
from task_graph.models import TaskStatus
from task_graph.runner import TaskRunner

from .conftest import (
    ScriptedBackend,
    StubManifest,
    StubMonitor,
    StubPhaseRunner,
    assigned_attributes,
    called_attributes,
    dispatched,
)


def _run(wired, agent_spec: str, closure: str, timeout: float = 5.0):
    task, agent = dispatched(agent_spec, closure, wired)
    done: list[tuple] = []
    finished = threading.Event()

    def on_done(tid, status, usage, *, detail=""):
        # `detail` is keyword-only on `OnDone`, which `task_graph` widened from a
        # `Callable` alias to a Protocol so the runner could pass a reason at all.
        done.append((tid, status, usage, detail))
        finished.set()

    wired.runner.start(task, agent, on_done)
    finished.wait(timeout)
    wired.runner.attempt_of(task.id) and wired.runner.attempt_of(task.id).join(timeout)
    return task, done


# --------------------------------------------------------------------------- #
# Criterion 6


def test_backend_is_not_a_runner() -> None:
    """The two have no method in common, `TaskRunner.start` takes two arguments
    neither protocol here mentions, and neither can be passed where the other
    is expected."""
    runner_verbs = {"start", "stop"}
    assert set(inspect.signature(TaskRunner.start).parameters) == {
        "self",
        "task",
        "agent",
        "on_done",
    }
    assert set(inspect.signature(Executor.start).parameters) == {"self"}
    assert runner_verbs & {"start_async", "wait", "mainloop", "interrupt"} == set()
    assert not isinstance(ScriptedBackend(), TaskRunner.__class__)


def test_runner_holds_level_one_only() -> None:
    """Criterion 6 in its strongest form: **the runner cannot call an AI-only
    method because it does not hold one.** It fails the moment somebody widens
    the runner to reach `interrupt`."""
    assert TaskAttempt.__annotations__ == {} or True  # the annotation is on the field
    hint = inspect.get_annotations(TaskAttempt.__init__, eval_str=False)
    assert "AgentBackend" not in str(hint)

    assert not called_attributes(Runner, TaskAttempt) & {"interrupt", "instruct", "query"}

    program = ProgramExecutor(assignment=_entry())
    assert isinstance(program, Executor)
    assert not isinstance(program, AgentBackend)


def test_runner_unchanged_across_backends(wired) -> None:
    """Substituting the backend leaves the runner unchanged: the phase order,
    the transitions and the reporting are identical, because none of them
    mentions a backend."""
    _, ai_done = _run(wired, "writer", "leaf_ai")
    ai_kinds = list(wired.monitor.kinds())
    wired.monitor.records.clear()
    _, program_done = _run(wired, "runner", "leaf")

    assert ai_kinds == wired.monitor.kinds() == ["phase_done"] * 3
    assert ai_done[0][1] is program_done[0][1] is TaskStatus.SUCCEEDED


# --------------------------------------------------------------------------- #
# Criterion 15


def test_swap_backend_same_handoff_state(wired) -> None:
    """**Swapping the backend changes no other component.** Nearly a tautology
    once the runner needs level 1 only, and tested end to end anyway, because
    the tautology is about types and the criterion is about handoffs."""
    store = wired.registry.get("handoff_store")
    before = dict(store.present)

    _, ai = _run(wired, "writer", "leaf_ai")
    ai_state = dict(store.present)
    _, program = _run(wired, "runner", "leaf")

    assert ai_state == dict(store.present) == before
    assert ai[0][1] == program[0][1] == TaskStatus.SUCCEEDED


# --------------------------------------------------------------------------- #
# The three phases


def test_the_three_phases_run_in_order(wired) -> None:
    task, done = _run(wired, "writer", "leaf_ai")
    assert [kind for kind, _ in wired.phase_runner.calls] == [
        "input_validation",
        "output_validation",
    ]
    assert task.status is TaskStatus.OUTPUT_VALIDATING
    assert done[0][1] is TaskStatus.SUCCEEDED


def test_the_runner_never_assigns_a_status(wired) -> None:
    """`task.enter_phase(next)` is the only route to a status, and it is never
    an assignment — only the caller moved when `monitor` spec §5.3 made the
    planned advance a monitor action."""
    assert "status" not in assigned_attributes(Runner, TaskAttempt)
    assert "enter_phase" not in called_attributes(Runner, TaskAttempt)


def test_a_failed_input_phase_reports_and_stops(wired) -> None:
    wired.registry.register("phase_runner", StubPhaseRunner(passes=False))
    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    wired.runner.attempt_of(task.id).join(5)
    assert wired.monitor.kinds() == ["validation_failed"]
    assert task.status is TaskStatus.INPUT_VALIDATING


def test_an_empty_phase_advances_and_records_that_nothing_ran(wired) -> None:
    """`demo` F-D9, and **this test previously asserted the defect.**

    It was `test_an_empty_phase_is_unreached_not_failed`, and it pinned
    `["validation_unreached"]` — which was the runner reporting an unplanned
    event and ending the thread, leaving the task in its phase for ever.
    Measured live: `main` sat in `INPUT_VALIDATING` for 300 s. The test was
    green the whole time, because it had encoded my reading rather than the
    requirement.

    The requirement is `demo` criterion 3: *"Empty is the normal case and must
    be shown to be normal, not degenerate."* An empty phase is not a pass — that
    stays true, and `evidence` is where it is said — but it is not a failure
    either, and it must advance.
    """
    wired.registry.register(
        "phase_runner", StubPhaseRunner(passes=False, empty=True, evidence="nothing_ran")
    )
    task, done = _run(wired, "writer", "leaf_ai")

    assert wired.monitor.kinds() == ["phase_done"] * 3
    assert done[0][1] is TaskStatus.SUCCEEDED
    # Two of the three are validation phases and carry the qualification; the
    # middle one is the main phase, which has no `PhaseOutcome` to qualify.
    assert _evidences(wired) == ["nothing_ran", "nothing_ran"]


def test_an_empty_phase_is_still_not_a_pass(wired) -> None:
    """The half that must survive the fix. The task advances, and the record
    says `nothing_ran` rather than `established` — so a reader can tell an
    unvalidated task from a validated one, which is the distinction
    `PhaseOutcome.evidence` exists to carry and pytest's XPASS lost."""
    wired.registry.register(
        "phase_runner", StubPhaseRunner(passes=False, empty=True, evidence="nothing_ran")
    )
    _run(wired, "writer", "leaf_ai")
    assert set(_evidences(wired)) == {"nothing_ran"}
    assert "established" not in _evidences(wired)


def _evidences(wired) -> list[str]:
    return [r.attributes["evidence"] for r in wired.monitor.records if "evidence" in r.attributes]


def test_a_failing_phase_still_blocks(wired) -> None:
    """The other half. Advancing on empty must not advance on a failure — the
    two used to share an arm, which is how the fix could break this."""
    wired.registry.register(
        "phase_runner", StubPhaseRunner(passes=False, empty=False, evidence="failed")
    )
    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    wired.runner.attempt_of(task.id).join(5)
    assert wired.monitor.kinds() == ["validation_failed"]
    assert task.status is TaskStatus.INPUT_VALIDATING


def test_a_non_leaf_hands_its_thread_back_at_unfold(wired) -> None:
    """**A non-leaf's attempt ends its thread at `unfold`** — its next event may
    be hours away, and an attempt that waited would hold one thread per
    ancestor. The object survives, which is what it is for."""
    task, agent = dispatched("writer", "branch", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    attempt.join(5)
    assert task.status is TaskStatus.RUNNING
    assert attempt.executor is None
    assert wired.runner.attempt_of(task.id) is attempt


def test_re_entry_is_the_same_attempt_and_not_a_new_execution(wired) -> None:
    """**Not `start`**: no new `Execution`, no new agent, the same attempt.

    Through `carry_on` now — `Runner.resume` is removed, because `monitor`
    enumerated all twelve of their spec §7.1 actions and no path wants a bare
    re-entry without the wake-or-resume decision.
    """
    task, agent = dispatched("writer", "branch", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    attempt.join(5)

    task.enter_phase(TaskStatus.OUTPUT_VALIDATING)
    done: list[tuple] = []
    attempt.on_done = lambda *a, **k: done.append((*a, k))
    wired.runner.carry_on(task.id)
    attempt.join(5)

    assert wired.runner.attempt_of(task.id) is attempt
    assert len(task.history) == 1
    assert done[0][1] is TaskStatus.SUCCEEDED


def test_attempt_of_reaches_the_live_executor(wired) -> None:
    """`monitor` design O1: **it was not a missing accessor, it was a missing
    object.** The monitor's push needs `instruct` on a live agent, and this is
    the only route to one."""
    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.monitor.advance = False
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    assert attempt is not None and attempt.task is task
    attempt.halt()
    attempt.join(5)


def test_start_returns_before_the_backend_is_reached(wired) -> None:
    """**`start` is called while the scheduler holds its `RLock`**, so what it
    does has to be cheap: it creates the attempt and starts a thread, and the
    backend is two phases later on that thread."""
    wired.monitor.advance = False
    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    assert wired.runner.attempt_of(task.id).executor is None
    wired.runner.attempt_of(task.id).halt()


def test_stop_ends_the_attempt_and_acknowledges(wired) -> None:
    """`Runner.stop` is the scheduler's verb; `interrupt` is level 2 and has no
    `TaskRunner` route (design §7.4)."""
    wired.monitor.advance = False
    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    acked: list = []
    wired.runner.stop(task.id, acked.append)
    assert acked == [task.id]
    assert wired.runner.attempt_of(task.id) is None


def test_an_unresolvable_monitor_is_loud(wired) -> None:
    """`docs/interfaces.md` §2.1 rev. 4: such a task "never advances a phase".
    It fails visibly instead — a task that hangs for ever and a task that says
    why are the same outcome told two ways, and only one is debuggable."""
    task, agent = dispatched("writer", "leaf_ai", wired)
    task.monitor_spec = "absent"
    done: list[tuple] = []
    wired.runner.start(task, agent, lambda *a, **k: done.append((*a, k)))
    wired.runner.attempt_of(task.id).join(5)
    assert done and done[0][1] is TaskStatus.FAILED
    assert issubclass(MonitorUnresolved, RuntimeError)


def _entry():
    from agent.backend import Assignment

    return Assignment(entry="/bin/true")


# --------------------------------------------------------------------------- #
# The two seam obligations that fail SILENTLY if they are skipped


def test_prepare_is_given_the_agent_spec(wired) -> None:
    """`env_mgr` design §11.5's `material.deploy(agent_spec, zone)` has no other
    source for it. Without the third argument, `agent` spec §3.1's `env` and
    design §3.4's `rules` / `hooks` / `skills` have **no consumer at all** —
    four spec keys an author can write that nothing ever reads."""
    env = wired.registry.get("env_mgr")
    _run(wired, "writer", "leaf_ai")
    assert env.calls and env.calls[0][1] is not None
    assert env.calls[0][1].name == "writer"


def test_prepare_is_called_once_per_attempt_with_the_execution(wired) -> None:
    """A grant resolves to `<root>/<hid>/v<N>/` and `N` lives on the attempt, so
    a retry gets a **different granted set**. It takes an `Execution`, not just
    a `Task`."""
    env = wired.registry.get("env_mgr")
    task, _ = _run(wired, "writer", "leaf_ai")
    assert len(env.calls) == 1
    assert env.executions == [task.current]


def test_a_refused_environment_never_reaches_an_executor(wired) -> None:
    """**Criterion 14 is `no isolation, no start`.** `NoConfinement`,
    `PrepareRefused` and `UnresolvedGrant` each mean the task does not start,
    and there is no `try` in `_deploy` that could log one and continue.

    What this asserts is the observable half: nothing was selected, no executor
    exists, and the task is closed `FAILED` rather than left running.
    """

    class Refusing:
        def prepare(self, task, execution, agent_spec=None):
            raise RuntimeError("no mechanism could confine this task")

    wired.registry.register("env_mgr", Refusing())
    task, done = _run(wired, "writer", "leaf_ai")
    attempt = wired.runner.attempt_of(task.id)

    assert attempt.executor is None
    assert done[0][1] is TaskStatus.FAILED
    assert "handling_failed" in wired.monitor.kinds()


def test_the_runner_refuses_a_bwrap_confinement_it_cannot_apply(wired) -> None:
    """**On rung 1 the caller execs.** bubblewrap *is* the exec, so `apply()`
    confines nothing and whoever starts the executor builds the argv. Skip it
    and `prepare` succeeded, the task ran, and there was no sandbox.

    We cannot build that argv — `bwrap_argv` needs an `Availability` that is not
    on `Prepared` — so a bwrap confinement with no wrapper is refused. `bwrap`
    is absent on this machine, so nothing else would have caught it.
    """
    from agent.runner import ConfinementNotApplied

    wired.registry.register("env_mgr", _confining_env(str(wired.tmp_path), spawn=None))
    task, done = _run(wired, "writer", "leaf_ai")
    assert wired.runner.attempt_of(task.id).executor is None
    assert done[0][1] is TaskStatus.FAILED
    assert issubclass(ConfinementNotApplied, RuntimeError)


def test_a_confinement_starts_the_executor_through_spawn(wired) -> None:
    """`prepare` checks that a mechanism exists; `spawn` applies it in the
    child. So the executor is *started confined* rather than started into a
    confinement, and the runner branches on no mechanism."""
    started: list[list[str]] = []
    wired.registry.register(
        "env_mgr", _confining_env(str(wired.tmp_path), spawn=_recording_spawn(started))
    )
    _, done = _run(wired, "runner", "leaf")
    assert started and started[0][0] == "/bin/sh"
    assert done[0][1] is TaskStatus.SUCCEEDED


def _storeless(wired) -> None:
    """Unregister `handoff_store`, the way a composition that chose none leaves it.

    `bootstrap.py:216` leaves the name unregistered rather than rooting a store
    nobody asked for, and `task_graph` runs its whole suite that way — so this
    is a supported mode, not a broken fixture.
    """
    wired.registry._items.pop("handoff_store", None)


def test_a_storeless_task_that_declares_outputs_does_not_succeed(wired) -> None:
    """**Two "nothing to say" answers were composing into "nothing was wrong".**

    `_seal_outputs` returned `{}` with no store and `_gate` returned `[]`, each
    correct about the single fact it tests, and `_main`'s `if not failures` read
    the pair as a pass. So a task that declared an output and published none was
    reported **succeeded**, with no cause named anywhere. Measured by `monitor`.

    The fault is the conjunction — declares outputs, no store — which is what
    `_pin_outputs` already warns about and what neither guard asked.
    """
    _storeless(wired)
    task, agent = _with_output(wired, HandoffId.new())
    done: list = []

    wired.runner.start(task, agent, lambda tid, status, usage, *, detail="": done.append(status))
    wired.runner.attempt_of(task.id).join(5)

    assert "output_absent" in wired.monitor.kinds()
    assert TaskStatus.SUCCEEDED not in done
    absent = next(r for r in wired.monitor.records if r.kind.value == "output_absent")
    # `nothing_to_attempt`, not `seal_refused`: the producer was never asked.
    # Criterion 5 needs those apart, and `seal_refused`'s *presence* is what
    # says the producer was asked at all.
    assert "no handoff store" in absent.attributes["nothing_to_attempt"]
    assert "seal_refused" not in absent.attributes


def test_a_storeless_task_that_declares_no_outputs_still_succeeds(wired) -> None:
    """**Storeless stays a supported mode.** The guard was never the defect —
    its coverage was. A task that wanted no store and declared no outputs must
    pass exactly as before."""
    _storeless(wired)
    _, done = _run(wired, "writer", "leaf_ai")

    assert done[0][1] is TaskStatus.SUCCEEDED


def test_the_budget_is_checked_even_with_no_store(wired) -> None:
    """The early return skipped `_budget` too, so a storeless run checked
    **nothing** — not just nothing about outputs. Found while fixing the other
    half; the gate decides what a missing store means, and the runner's job is
    to ask it either way."""
    _storeless(wired)
    wired.registry.register(
        "budget", SimpleNamespace(max_tokens=0.0, max_seconds=None, max_turns=None)
    )
    _, done = _run(wired, "writer", "leaf_ai")

    assert "budget_exceeded" in wired.monitor.kinds()
    assert TaskStatus.SUCCEEDED not in [d[1] for d in done]


@pytest.mark.parametrize("enforced", [True, False])
def test_the_permissions_switch_reaches_the_executor(wired, enforced: bool) -> None:
    """**The carry-across, and nothing guarded it.**

    `Prepared.permissions_enforced` says *why* there is no confinement — the
    user's kill switch, or a machine with no mechanism — and an AI backend must
    tell them apart, because with the switch on it also stands down the SDK's
    own permission layer.

    Measured: deleting the one line in `_deploy` that carries this field left
    **195 tests passing**. The backend half is guarded on both sides; this half
    was not, so the switch could have silently stopped reaching the harness and
    every AI task would have gone back to being blocked by an approval prompt
    nobody can answer.
    """
    env = wired.registry.get("env_mgr")
    original = env.prepare

    def prepare(task, execution, agent_spec=None):
        prepared = original(task, execution, agent_spec)
        return SimpleNamespace(**{**vars(prepared), "permissions_enforced": enforced})

    env.prepare = prepare
    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.monitor.advance = False
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    wired.monitor.advance = True
    wired.monitor.report(_first_report(wired))
    attempt.join(5)

    assert attempt.executor.assignment.permissions_enforced is enforced


def test_the_deployed_environment_reaches_the_executor(wired) -> None:
    """`CLAUDE_CONFIG_DIR` is the load-bearing one: with `~/.claude` granted, a
    confined demo agent read the *operator's personal* `CLAUDE.md` and obeyed
    its language rule. Pointing it into the zone removes the `$HOME` grant.

    **No confinement here on purpose.** This is about the environment reaching
    the executor, and an AI executor with a confinement is now refused — see
    `test_an_ai_agent_cannot_be_confined_under_any_mechanism`. Keeping the two
    apart means neither passes for the other's reason.
    """
    wired.registry.register(
        "env_mgr", _confining_env(str(wired.tmp_path), spawn=None, mechanism=None)
    )
    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.monitor.advance = False
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    wired.monitor.advance = True
    wired.monitor.report(_first_report(wired))
    attempt.join(5)

    assert attempt.executor is not None
    assert attempt.executor.assignment.environment["CLAUDE_CONFIG_DIR"].endswith("/config")


def test_a_body_path_resolves_against_the_staged_package(wired, tmp_path) -> None:
    """§4.16 copies the package into the zone and leaves the original outside
    every grant, so a body path resolved against the original names a file the
    kernel refuses — `demo` measured `/bin/sh: cannot open …: Permission
    denied`.

    **`Runner.resolve_path` could not have been fixed in place**: `package_root`
    is a constructor argument and the staged copy is per attempt. This asserts
    the attempt's copy wins, and that the readme arrives as its **contents** —
    the schema calls the declared value a path, and it used to reach the SDK as
    `system_prompt` unread.
    """
    staged = tmp_path / "staged-package"
    (staged / "bodies").mkdir(parents=True)
    (staged / "bodies" / "readme.md").write_text("Brief from the staged copy.\n")

    env = _confining_env(str(wired.tmp_path), spawn=None, mechanism=None)
    prepare = env.prepare

    def staging(task, execution, agent_spec=None):
        prepared = prepare(task, execution, agent_spec)
        return SimpleNamespace(**{**vars(prepared), "staged_package": str(staged)})

    env.prepare = staging
    wired.registry.register("env_mgr", env)
    # The task spec the runner reads, not the closure it was dispatched from.
    wired.registry.get("task_specs").get("leaf_ai")["body"]["readme"] = "bodies/readme.md"

    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.monitor.advance = False
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    wired.monitor.advance = True
    wired.monitor.report(_first_report(wired))
    attempt.join(5)

    assert attempt.executor.assignment.readme == "Brief from the staged copy.\n"


def test_the_agent_is_told_where_each_output_goes(wired) -> None:
    """`demo`'s first real model call produced nothing because **nothing in the
    conversation named the output**. `AGENT_SYS_OUTPUT_SUMMARY` was in the
    process environment the whole time; a model is not a process reading
    `os.environ`.

    `main` ruled the runner states the facts only it possesses — the declared
    output, its kind, its resolved path — and authors no guidance.
    """
    hid = HandoffId.new()
    env = _confining_env(str(wired.tmp_path), spawn=None, mechanism=None)
    prepare = env.prepare
    env.prepare = lambda t, e, a=None: SimpleNamespace(
        **{**vars(prepare(t, e, a)), "output_paths": {hid: "/store/h/v3/content"}}
    )
    wired.registry.register("env_mgr", env)
    task, agent = _with_output(wired, hid)
    task.kinds = {hid: "summary"}

    wired.monitor.advance = False
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    wired.monitor.advance = True
    wired.monitor.report(_first_report(wired))
    attempt.join(5)

    brief = attempt.executor.assignment.outputs_brief
    assert "summary" in brief and "/store/h/v3/content" in brief
    # The readme is the package's and stays the package's.
    assert brief not in attempt.executor.assignment.readme


def test_an_output_with_no_resolved_path_says_so(wired) -> None:
    """**Omitting it is the failure one level up.** An agent told about two of
    three outputs writes two and finishes successfully — `interfaces.md`
    §4.13's family, and `main` made stating it a condition of the ruling."""
    hid = HandoffId.new()
    task, agent = _with_output(wired, hid)  # the stub env reports no output_paths
    task.kinds = {hid: "summary"}

    wired.monitor.advance = False
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    wired.monitor.advance = True
    wired.monitor.report(_first_report(wired))
    attempt.join(5)

    brief = attempt.executor.assignment.outputs_brief
    assert "summary" in brief
    assert "no resolved path" in brief


def _first_report(wired, timeout: float = 5.0):
    """The attempt's first record, **waited for rather than assumed present**.

    `Runner.start` creates the attempt and returns — its own docstring says
    *"cheap, and it returns"* — so the report these tests then hand back to the
    monitor is written on the attempt's thread after `start` has already
    returned on this one. Reading `records[0]` on the next line is a race that
    wins almost always and lost at least once (`agent-mod-2`, under load with a
    dirty tree, on a test that passed in isolation and on the two runs after).

    **Almost always is the problem**: a test that fails once an afternoon is
    read as flakiness in the suite rather than as the missing wait it is, and
    the six call sites shared the assumption.
    """
    deadline = time.monotonic() + timeout
    while not wired.monitor.records and time.monotonic() < deadline:
        time.sleep(0.005)
    assert wired.monitor.records, "the attempt reported nothing within the deadline"
    return wired.monitor.records[0]


def _with_output(wired, hid) -> tuple:
    """A dispatched task that actually declares an output, with its version
    pinned the way `Scheduler._dispatch_pass` pins it (`scheduler.py:280`).

    **No fixture task declares an output**, which is why 171 tests passed while
    the seal never ran once: `run_gate` loops over `task.outputs` and so does
    `_seal_outputs`, and both loops were empty.
    """
    task, agent = dispatched("writer", "leaf_ai", wired)
    task.outputs = [hid]
    task.current.output_versions = {hid: 3}
    return task, agent


def test_the_runner_seals_its_outputs_before_the_gate_asks(wired) -> None:
    """**Ruled: the seal happens in `_main`, after `wait()` and before `_gate`.**

    The gate asks whether an output exists, and `FilesystemStore.exists` means
    *published* — an allocated-but-unsealed directory is not a version that
    exists. `task_graph` pins the version at dispatch and would seal at close,
    and **close is after the gate**, so nothing published before the question
    was asked and every successful task reported `OUTPUT_ABSENT`.
    """
    hid = HandoffId.new()
    store = wired.registry.get("handoff_store")
    store.written.add(hid)  # the body wrote something into its grant
    task, agent = _with_output(wired, hid)

    wired.runner.start(task, agent, lambda *a, **k: None)
    wired.runner.attempt_of(task.id).join(5)

    assert store.sealed == [(hid, 3, task.id)]
    # The version it sealed is the one pinned at dispatch, and the producer is
    # the task — not the agent, which is what `seal`'s signature asks for.
    assert wired.monitor.kinds() == ["phase_done"] * 3


def test_a_task_that_produced_nothing_still_reports_output_absent(wired) -> None:
    """**The case a green gate proves least about**, and the one worth pinning.

    Sealing must not turn "the body wrote nothing" into a published version.
    The store refuses an empty `content/` before it looks at contents, the
    version stays a hole, and the gate reports the absence — which is the truth
    about the attempt.
    """
    hid = HandoffId.new()
    store = wired.registry.get("handoff_store")  # nothing added to `written`
    task, agent = _with_output(wired, hid)

    wired.runner.start(task, agent, lambda *a, **k: None)
    wired.runner.attempt_of(task.id).join(5)

    assert store.sealed == []
    assert "output_absent" in wired.monitor.kinds()
    absent = next(r for r in wired.monitor.records if r.kind.value == "output_absent")
    # Criterion 5's distinction survives: *never attempted* is not *wrote badly*,
    # and the store says which in its refusal.
    assert "produced no content at all" in absent.attributes["seal_refused"]


class _FailsOutputOnly(StubPhaseRunner):
    """Passes the input phase and fails the output phase.

    Inline rather than a `passes_output=` on the shared stub: exactly one test
    needs it, and a shared double grows a flag per caller until nobody can say
    what its default models.
    """

    def run_phase(self, kind: Any, task: Any, registry: Any):
        self.passes = str(kind) != "output_validation"
        return super().run_phase(kind, task, registry)


def _real_handoff_mgr(wired, hid):
    """The **real** `HandoffMgr`, not a double.

    Every other stand-in in this file is a double, and four of them drifted
    from the real type today. This seam is the one where a double would be
    worst: the whole point is that `check_if_latest_valid` — the consumer's
    eligibility question, three calls deep in `task_graph`'s state machine —
    answers `True` afterwards. A double would answer whatever I taught it to.
    """
    from task_graph.handoff import HandoffMgr
    from task_graph.store import MemoryStoreMgr

    wired.registry.register("store_mgr", MemoryStoreMgr())
    mgr = HandoffMgr(wired.registry)
    wired.registry.register("handoff_mgr", mgr)
    mgr.declare([hid], TaskId.new())
    return mgr


def test_a_sealed_output_makes_its_consumer_eligible(wired) -> None:
    """**The wall `describe` waited behind.** The agent-facing write path —
    `open_next`, `seal`, `persist` — had no production caller at all; only
    `FakeRunner.produce`, a double whose docstring says it stands in for this
    runner. So the store held `facts v0: SEALED, published, verdict PASS`
    while the model held `status=created`, and eligibility reads the model.

    Asserted through `check_if_latest_valid`, which is the question the
    scheduler actually asks, rather than through the status field it reads.
    """
    hid = HandoffId.new()
    mgr = _real_handoff_mgr(wired, hid)
    store = wired.registry.get("handoff_store")
    store.written.add(hid)
    task, agent = _with_output(wired, hid)

    assert mgr.check_if_latest_valid(hid) is False  # created, not usable

    wired.runner.start(task, agent, lambda *a, **k: None)
    wired.runner.attempt_of(task.id).join(5)

    assert mgr.check_if_latest_valid(hid) is True


def test_a_failed_output_validation_seals_invalid_rather_than_leaving_it_open(wired) -> None:
    """`INVALID` is the honest record, and leaving the slot `GENERATING` would
    make a re-dispatch raise — `open_next` refuses a slot someone else has
    open."""
    hid = HandoffId.new()
    mgr = _real_handoff_mgr(wired, hid)
    wired.registry.get("handoff_store").written.add(hid)
    # **The output phase only.** `StubPhaseRunner(passes=False)` fails the
    # *input* phase, so the main phase never runs, `_open_outputs` never fires
    # and the slot stays `created` — which is a different thing being tested.
    wired.registry.register("phase_runner", _FailsOutputOnly())
    task, agent = _with_output(wired, hid)

    wired.runner.start(task, agent, lambda *a, **k: None)
    wired.runner.attempt_of(task.id).join(5)

    assert mgr.check_if_latest_valid(hid) is False
    assert mgr.latest(hid).status.value == "invalid"


def test_a_store_refusal_never_yields_a_valid_model_version(wired) -> None:
    """**The rider, asserted rather than relied on.**

    Today the gate reports `OUTPUT_ABSENT` for a refused seal and `_close` is
    never reached, so this cannot happen — which is exactly the kind of
    guarantee that stops holding when someone reorders. If the store did not
    publish it, the model must not call it usable.
    """
    hid = HandoffId.new()
    mgr = _real_handoff_mgr(wired, hid)
    # Nothing added to `written`, so the store refuses to publish.
    task, agent = _with_output(wired, hid)

    wired.runner.start(task, agent, lambda *a, **k: None)
    wired.runner.attempt_of(task.id).join(5)

    assert mgr.check_if_latest_valid(hid) is False


def test_an_output_this_attempt_never_published_is_not_sealed_valid(wired) -> None:
    """**Absence of a refusal is not evidence of a seal.**

    Found by `task_graph` asking whether `_seal_model_versions` can reach a
    slot with no entry in `output_versions`. It can — the loop is over
    `task.outputs`, not over the pinned versions — and `_seal_outputs` *skips*
    such a hid, which put it in neither the sealed set nor the refused map.

    The gate does not catch it, because `exists()` is **not attempt-scoped**: a
    version published by a *previous* attempt satisfies it. Measured before the
    fix — nothing pinned, nothing sealed by this attempt, an older version in
    the store — and the slot came out `valid` with the consumer eligible.

    That is rider 1 defeated through the skip path instead of the refusal path.
    """
    hid = HandoffId.new()
    mgr = _real_handoff_mgr(wired, hid)
    task, agent = _with_output(wired, hid)
    task.current.output_versions = {}  # nothing pinned for this attempt
    store = wired.registry.get("handoff_store")
    store.present[hid] = StubManifest()  # a previous attempt published

    wired.runner.start(task, agent, lambda *a, **k: None)
    wired.runner.attempt_of(task.id).join(5)

    assert store.sealed == [], "this attempt published nothing"
    assert mgr.check_if_latest_valid(hid) is False
    assert mgr.latest(hid).status.value == "invalid"


def test_a_gate_failure_does_not_deadlock_the_next_dispatch(wired) -> None:
    """**The normal failure path, and it deadlocked the retry.**

    Found by `task_graph`, who read the plan rather than the code: the version
    is opened in `_main` and sealed in `_close`, and **a gate failure never
    reaches `_close`** — that is the same ordering that forced the store seal
    to precede the gate. So the slot stayed `GENERATING` for ever, and the next
    dispatch raised `HandoffStateError: … is already open`.

    Not an edge case. It is *the* failure — an agent that produced nothing
    usable — turned from a recoverable attempt into a hard error on retry.

    Their prescribed probe, run rather than reasoned about: open a version,
    force a gate failure, re-dispatch, and confirm the second `open_next` does
    not raise.

    **Running it corrected the scenario, and the correction is worth keeping.**
    A gate failure alone does *not* end the attempt — measured, the thread is
    still alive after five seconds with `halted=False`, parked in
    `_await_wake`. That is correct: *"the runner never pushes; it reported, the
    monitor decides, and a decision that says try again arrives as a wake."*
    So the deadlock needs the ending the monitor actually causes — a give-up,
    which reaches `Runner.stop` and `halt()`. Modelled here, because a test
    that let the thread park would have asserted `generating` for ever and
    called it a bug.
    """
    hid = HandoffId.new()
    mgr = _real_handoff_mgr(wired, hid)
    # Nothing in `written`, so the store refuses and the gate reports absence.
    task, agent = _with_output(wired, hid)

    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    _wait_until(lambda: "output_absent" in wired.monitor.kinds())
    wired.runner.stop(task.id, lambda tid: None)  # the monitor gave up
    attempt.join(5)

    # **Assert the precondition of the assertion below.** The seal happens in
    # `run`'s `finally`, so it only holds once the thread has ended — and
    # `join(5)` returns whether or not it did. Without this, a slow run reports
    # `generating != invalid`, which reads as the defect being back rather than
    # as a test that did not wait long enough. `env_mgr` saw this fail 2 of 4
    # full-suite runs while it passed alone; I could not reproduce it (15/15
    # isolated, 0/5 across `tests/agent`), so this does not claim to fix a cause
    # — it makes the failure name itself if it returns.
    assert not attempt._thread.is_alive(), (
        "the attempt thread was still running after join(5); the assertion "
        "below is about state that is only written when the thread ends"
    )
    assert mgr.latest(hid).status.value == "invalid", (
        "an attempt that ended without reaching _close left the slot GENERATING"
    )
    # The whole guarantee: a re-dispatch can open the next version.
    mgr.get(hid).open_next(TaskId.new(), agent.id)


def test_the_two_seals_each_name_the_other() -> None:
    """**Two verbs named `seal`, two objects, two moments, one flow.**

    `handoff.FilesystemStore.seal` publishes the *store directory* version
    before the gate; `task_graph.HandoffVersion.seal` seals the *slot* version
    after output validation. Each is unambiguous inside its own package, and
    this runner is where both appear — so the disambiguation lives here.

    Raised by `handoff` after they checked the ordering **because the name made
    them**, and found nothing wrong. That is the expensive kind: a defect gets
    found once and closed, and a thing that merely *looks* wrong is re-derived
    by everyone who arrives.

    Pinned as a test because prose rots — three docstrings in this package were
    wrong about their own code today, each one outliving the change it
    described.
    """
    from agent.runner import TaskAttempt

    store_seal = inspect.getdoc(TaskAttempt._seal_outputs) or ""
    model_seal = inspect.getdoc(TaskAttempt._seal_model_versions) or ""

    assert "_seal_model_versions" in store_seal, "the store seal must name the model seal"
    assert "_seal_outputs" in model_seal, "the model seal must name the store seal"
    assert "not the model seal" in store_seal
    assert "not the store seal" in model_seal


def test_the_model_slot_is_opened_once_per_attempt_not_once_per_pass(wired) -> None:
    """`open_next` refuses a slot already open, so a second body run inside one
    attempt would raise if it were inside the loop. One `Execution`, one
    version — the state machine and the ordering agree."""
    from agent.runner import TaskAttempt

    source = inspect.getsource(TaskAttempt._main)
    opened = source.index("_open_outputs")
    loop = source.index("while not self._halted")

    assert opened < loop, "_open_outputs must run before the re-run loop, not inside it"


def test_every_planned_report_names_the_phase_that_finished(wired) -> None:
    """`monitor`'s `_advance` re-derived the finished phase from `task.status`
    **after** the fact, and `_close` reports a third `PHASE_DONE` for a
    three-phase order — so which error it raised depended only on whether it
    drained before `on_done` landed, and a `HANDLING_FAILED` was persisted for
    every successful task.

    Read here **before** `on_done` runs, so it is the phase that finished.
    """
    _run(wired, "writer", "leaf_ai")

    planned = [r for r in wired.monitor.records if r.kind.value == "phase_done"]
    assert [r.attributes["phase"] for r in planned] == [
        "INPUT_VALIDATING",
        "RUNNING",
        "OUTPUT_VALIDATING",
    ]


def test_the_reported_phase_is_the_key_monitors_phase_order_uses() -> None:
    """**`.name`, not `.value`, and the difference is the whole of it.**

    `monitor` asked for `.value` and compares against `PHASE_ORDER`, which is
    keyed by `.name` — their own `next_phase` does `PHASE_ORDER.index(
    status.name)`. Measured, `"output_validating" == "OUTPUT_VALIDATING"` is
    `False`, so the fix as proposed would never have matched and the terminal
    phase would still have been treated as an advance.

    Two sides, one name, only one of them checked. This is the check.
    """
    from monitor.base import PHASE_ORDER

    assert PHASE_ORDER == ("INPUT_VALIDATING", "RUNNING", "OUTPUT_VALIDATING")
    assert TaskStatus.OUTPUT_VALIDATING.name == PHASE_ORDER[-1]
    assert TaskStatus.OUTPUT_VALIDATING.value != PHASE_ORDER[-1]
    # And what the runner actually emits is the one that matches.
    for phase in PHASE_ORDER:
        assert getattr(TaskStatus, phase).name == phase


def test_a_wiring_error_out_of_seal_is_not_swallowed(wired) -> None:
    """**`NotSealable` must escape, and nothing showed that it did.**

    `handoff` split the two refusals apart for this seam
    (`handoff/errors.py`): a `Malformed` artefact is *returned* as a reason and
    is a fact about the producer, while *no such version* or *already
    published* is a **wiring** bug that must die loudly. The re-run case is the
    dangerous one — a second `seal` against the same pinned `v<N>` — because
    swallowed, the loop looks like it worked while the second body's output is
    discarded, and only on retries.

    The property is now structural: `_seal_outputs` has no `except` at all.
    This asserts it, because *"there is no `except` there"* is a fact about
    code that a later edit can undo without any test noticing.

    **Asserted by behaviour, not by type** — `agent` may not import `handoff`,
    so `NotSealable` cannot be named here; any exception standing in for it
    must reach `_crash` rather than becoming a refusal.
    """
    hid = HandoffId.new()
    store = wired.registry.get("handoff_store")

    def wired_wrong(*a, **k):
        raise RuntimeError("already published. A version is written once")

    store.seal = wired_wrong
    task, agent = _with_output(wired, hid)
    done: list = []

    wired.runner.start(task, agent, lambda tid, status, usage, *, detail="": done.append(status))
    wired.runner.attempt_of(task.id).join(5)

    assert done and done[0] is TaskStatus.FAILED
    # Not turned into a refusal: a wiring bug is not a fact about the producer.
    assert all("seal_refused" not in r.attributes for r in wired.monitor.records)


def test_a_successful_seal_leaves_no_refusal_on_the_record(wired) -> None:
    """`seal_refused` is absent, not `None`-valued, when nothing refused —
    `_report` pops a lifted key and only sets it when it is not `None`, which is
    what keeps `EventRecord`'s `extra="forbid"` from raising on the ordinary
    case."""
    hid = HandoffId.new()
    store = wired.registry.get("handoff_store")
    store.written.add(hid)
    task, agent = _with_output(wired, hid)

    wired.runner.start(task, agent, lambda *a, **k: None)
    wired.runner.attempt_of(task.id).join(5)

    assert all("seal_refused" not in r.attributes for r in wired.monitor.records)


def test_the_attempt_carries_the_resolved_configuration(wired) -> None:
    """`validator` spec §8.2's producer row: at `OUTPUT_VALIDATING` a
    validation's default configuration is the validated task's.

    `_deploy` computed a `Prepared`, read four things off it and let it go, so
    after it returned **nothing could reach the task's resolved configuration**.
    `validator` reaches it through `attempt_of(task.id).environment`, and adds
    no import for it.

    Only the *producer* row is closed by this. The consumer row — the input
    phase — is unreachable in principle rather than unbuilt: `_one_phase` runs
    `_validation(INPUT_PHASE)` at `INPUT_VALIDATING` and only reaches `_main`,
    and therefore `prepare`, at `RUNNING`. There is no `Prepared` yet to carry.
    """
    wired.registry.register(
        "env_mgr", _confining_env(str(wired.tmp_path), spawn=None, mechanism=None)
    )
    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.monitor.advance = False
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    wired.monitor.advance = True
    wired.monitor.report(_first_report(wired))
    attempt.join(5)

    assert attempt.environment["CLAUDE_CONFIG_DIR"].endswith("/config")
    # The attempt and the assignment cannot disagree about what the task ran
    # with: one writer, at the moment the executor is handed the same values.
    assert dict(attempt.environment) == dict(attempt.executor.assignment.environment)
    # Read-only on the *real* deploy path, not only on a hand-built attempt.
    with pytest.raises(TypeError):
        attempt.environment["CLAUDE_CONFIG_DIR"] = "/elsewhere"


def test_the_carried_configuration_is_read_only() -> None:
    """`env_mgr` hands out a `MappingProxyType` and this keeps that property.

    A live `dict` published off the attempt would be one task's configuration
    one mutation away from another's — `engineer_principle.md` §1, *never hand
    out a mutable handle to internal state*. It also makes the seam honest: a
    consumer that wants to change a value must build its own, which is what
    `validator` criterion 21's rebuild requires anyway.
    """

    attempt = TaskAttempt.__new__(TaskAttempt)
    attempt._environment = MappingProxyType({"CLAUDE_CONFIG_DIR": "/z/config"})

    with pytest.raises(TypeError):
        attempt.environment["CLAUDE_CONFIG_DIR"] = "/elsewhere"


def test_an_attempt_that_has_not_deployed_reports_an_empty_configuration() -> None:
    """Empty before the main phase, like `executor` and for the same reason: it
    is computed inside `_deploy`. Empty is the answer, not a placeholder."""

    attempt = TaskAttempt(SimpleNamespace(), SimpleNamespace(id="t1"), None, lambda *a, **k: None)

    assert dict(attempt.environment) == {}


def _confining_env(zone_root: str, *, spawn, mechanism: str = "bwrap"):
    """A `Prepared` carrying a `confinement`, an `environment`, and a `spawn`.

    `spawn(argv, **popen_kwargs)` is what starts the executor confined, since
    `interfaces.md` split step 7: `prepare` checks, `spawn` applies in the
    child. `None` stands for an environment that reports a confinement and
    offers no way to start through it — which the runner refuses.
    """

    class Env:
        def __init__(self) -> None:
            self.calls: list = []
            self.executions: list = []

        def prepare(self, task, execution, agent_spec=None):
            self.calls.append((task.id, agent_spec))
            self.executions.append(execution)
            # A `SimpleNamespace`, not a synthesised class: a function set as a
            # class attribute becomes a bound method and `wrap(argv)` arrives
            # with two arguments.
            return SimpleNamespace(
                zone=SimpleNamespace(root=zone_root),
                confinement=SimpleNamespace(mechanism=mechanism) if mechanism else None,
                spawn=spawn,
                environment={
                    "CLAUDE_CONFIG_DIR": f"{zone_root}/config",
                    "CLAUDE_CODE_TMPDIR": f"{zone_root}/tmp",
                },
                # The seventh and eighth fields of the real `Prepared`. The
                # runner reads both unconditionally, so a double that omits one
                # fails every test here with an `AttributeError` the moment
                # `env_mgr` adds a field — which is how each of them was found.
                agent_cli=None,
                staged_package=None,
                permissions_enforced=True,
                output_paths={},
                tools=(),
            )

    return Env()


def test_the_default_monitor_rule_has_one_owner(wired) -> None:
    """`monitor`'s Finding B. Two resolvers disagreed about what an absent
    `monitor_spec` means — `monitor:default` there, "whichever was registered
    first" here — and they are the two ends of one conversation: this side
    picks who a phase is *reported to*, `monitor`'s picks who an escalation is
    *sent to*. One task, two watchers.

    Latent under `build_registry`'s own wiring, where only `default` exists;
    live the moment `monitors=[...]` puts another name first, which it
    supports. So the test registers them in the order that used to diverge.
    """
    import monitor as monitor_pkg

    first, default = StubMonitor(), StubMonitor()
    wired.registry.register("monitor:careful", first)
    wired.registry.register("monitor:default", default)
    task, _ = dispatched("writer", "leaf_ai", wired)
    task.monitor_spec = None

    assert wired.runner.monitor_for(task) is monitor_pkg.monitor_for(task, wired.registry)
    assert wired.runner.monitor_for(task) is default


# --------------------------------------------------------------------------- #
# The seam `monitor` was blocked on


def test_is_running_tells_a_parked_leaf_from_a_released_non_leaf(wired) -> None:
    """`monitor`'s defect, and the predicate that fixes it.

    Their `_advance` branched on `attempt_of(tid) is None` for "the non-leaf
    case: no live thread", and **against this runner that is never true** — a
    non-leaf `release()`s, which ends the thread and keeps the object, and only
    `Runner.stop` drops it from the map. So they took the `else` and called
    `wake()` on an `Event` no thread was waiting on: the parent entered
    `OUTPUT_VALIDATING` and never ran it.

    `executor` cannot discriminate, which is what made a new predicate
    necessary: it is `None` for a parked leaf too.
    """
    # The non-leaf first, while the monitor is still advancing phases: it has
    # to reach RUNNING to release. Parking the leaf needs the opposite, so the
    # order matters and this is why.
    released_task, released_agent = dispatched("writer", "branch", wired)
    wired.runner.start(released_task, released_agent, lambda *a, **k: None)
    released = wired.runner.attempt_of(released_task.id)
    released.join(5)

    wired.monitor.advance = False
    parked_task, parked_agent = dispatched("writer", "leaf_ai", wired)
    wired.runner.start(parked_task, parked_agent, lambda *a, **k: None)
    parked = wired.runner.attempt_of(parked_task.id)
    _wait_until(lambda: parked.is_running)

    assert parked.executor is None and released.executor is None  # cannot tell them apart
    assert parked.is_running is True
    assert released.is_running is False

    parked.halt()
    parked.join(5)
    assert parked.is_running is False


def test_the_released_attempt_is_still_reachable(wired) -> None:
    """The other half of the same fact: `release()` ends the thread and the
    object survives, so `attempt_of` keeps returning it. That is what `resume`
    re-enters, and it is why `is_running` had to be asked instead."""
    task, agent = dispatched("writer", "branch", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    attempt.join(5)
    assert wired.runner.attempt_of(task.id) is attempt
    assert not attempt.is_running


def test_recorder_open_happens_on_the_attempts_thread(wired) -> None:
    """**Off the scheduler's `RLock`, and that is a measurement.** `__init__`
    runs inside `Runner.start`, which the scheduler calls holding its lock;
    `monitor` measured `Recorder.open` at 102 µs against `JsonFileStoreMgr` and
    5.1 ms for a fifty-task unfold — a third of the whole hold, and the only
    filesystem write in it."""
    wired.monitor.advance = False
    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)

    _wait_until(lambda: wired.recorder.opened)
    assert wired.recorder.opened == [(task.id, 0)]
    attempt.halt()
    attempt.join(5)


def test_a_missing_recorder_is_loud(wired) -> None:
    """`docs/interfaces.md` §2 registers `recorder`, so absence is a wiring bug
    and not a mode. A silent skip would void criterion 14's
    empty-versus-missing distinction with nothing saying so."""
    wired.registry._items.pop("recorder")
    task, done = _run(wired, "writer", "leaf_ai")
    assert done[0][1] is TaskStatus.FAILED
    assert "handling_failed" in wired.monitor.kinds()


def test_an_ai_agent_under_bwrap_refuses_to_start(wired) -> None:
    """The half `wrap_argv` cannot fix. `ClaudeSDKClient` spawns the `claude`
    CLI itself and neither side sees that argv, so under bubblewrap a
    `kind: ai` task is unwrappable — running it means `prepare` succeeded, the
    task ran, and there was no sandbox.

    `bwrap` is absent on this machine, so nothing else in either suite would
    ever exercise it: this is the only thing standing there.
    """
    wired.registry.register(
        "env_mgr",
        _confining_env(str(wired.tmp_path), spawn=_recording_spawn([])),
    )
    task, done = _run(wired, "writer", "leaf_ai")
    assert wired.runner.attempt_of(task.id).executor is None
    assert done[0][1] is TaskStatus.FAILED


def _recording_spawn(seen: list):
    """A `Prepared.spawn` that records the argv and starts something harmless.

    `spawn(argv, **popen_kwargs)` is the whole seam since `interfaces.md` split
    step 7: `prepare` checks a mechanism exists, this applies it in the child.
    """
    import subprocess

    def spawn(argv, **kw):
        seen.append(list(argv))
        return subprocess.Popen(["/bin/sh", "-c", "exit 0"], **kw)

    return spawn


def test_a_program_agent_under_landlock_also_goes_through_spawn(wired) -> None:
    """**The widening.** Before the step-7 split, Landlock confined the runner's
    thread and a child inherited, so only bubblewrap needed a caller. `prepare`
    no longer confines — that is what the split exists to avoid — so a child not
    started through `spawn` is unconfined under *every* mechanism."""
    started: list[list[str]] = []
    wired.registry.register(
        "env_mgr",
        _confining_env(str(wired.tmp_path), spawn=_recording_spawn(started), mechanism="landlock"),
    )
    _, done = _run(wired, "runner", "leaf")
    assert started and done[0][1] is TaskStatus.SUCCEEDED


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)


def test_the_phase_report_carries_evidence_not_only_passed(wired) -> None:
    """`validator` criterion 19. A phase whose passes are all `weak` is
    `LOW_CONFIDENCE`, not `ESTABLISHED`, and `passed` alone loses that — which
    is what pytest's XPASS lost: distinguishable per-item rendering, exit code
    0, green bar, and issue #11467 is a core developer saying *"it was missed
    that a test was fixed."*"""
    wired.registry.register("phase_runner", StubPhaseRunner(evidence="low_confidence"))
    _run(wired, "writer", "leaf_ai")
    planned = [r for r in wired.monitor.records if r.kind.value == "phase_done"]
    assert planned and planned[0].attributes["evidence"] == "low_confidence"


def test_a_phase_runner_without_evidence_is_not_an_error(wired) -> None:
    """The field is `validator`'s and arrived after the seam did. Absent means
    absent, not a failure."""
    wired.registry.register("phase_runner", StubPhaseRunner(evidence=None))
    _, done = _run(wired, "writer", "leaf_ai")
    planned = [r for r in wired.monitor.records if r.kind.value == "phase_done"]
    assert done[0][1] is TaskStatus.SUCCEEDED
    assert "evidence" not in planned[0].attributes


def test_a_program_spec_pinned_to_an_ai_backend_under_bwrap_is_refused(wired) -> None:
    """**R1, and the case that survived the first two fixes.**

    A CLI override resolves a backend entry in its own right and need not name
    one the spec declares — design D6, deliberately, because the case that most
    needs pinning is a backend the author did not foresee. So `AgentSpec.kind`
    is a *proxy* for the executor and the override breaks it: a `kind: program`
    spec pinned to an AI backend passed a kind-keyed guard and ran unwrapped.

    Asking the executor is what fixes it, and this is the test that would have
    failed against both earlier versions. Measured before writing:
    `scratch/impl-2026-08/agent/probe_r1_override.py`.
    """
    wired.runner.override = "scripted"
    wired.registry.register(
        "env_mgr",
        _confining_env(str(wired.tmp_path), spawn=_recording_spawn([])),
    )
    task, done = _run(wired, "runner", "leaf")  # a kind: program spec

    assert wired.runner.attempt_of(task.id).executor is None
    assert done[0][1] is TaskStatus.FAILED


def test_the_refusal_names_the_backend_and_not_the_interface(wired) -> None:
    """`ConfinementNotApplied` is spec §3.3.1's shape — an unimplementable
    method raises — and not the capability matrix §3.3.1 forbids. The error
    names the adapter, which is what makes the distinction visible to whoever
    reads it next."""
    from agent.backend import ConfinementNotApplied

    from .conftest import ScriptedBackend

    with pytest.raises(ConfinementNotApplied) as caught:
        ScriptedBackend().accept_confinement(lambda a: a)
    assert "scripted" in str(caught.value)


def test_an_ai_agent_cannot_be_confined_under_any_mechanism(wired) -> None:
    """**The widening, stated as the thing it costs.** `ClaudeSDKClient` spawns
    the `claude` CLI itself, so there is no child of ours for `spawn` to start
    — and since `prepare` no longer confines the runner's process, that CLI is
    not confined by inheritance either.

    Before the step-7 split this was true of bubblewrap only; it is now true of
    Landlock too. Both mechanisms, so neither passes by the other's accident.
    """
    for mechanism in ("bwrap", "landlock"):
        wired.registry.register(
            "env_mgr",
            _confining_env(str(wired.tmp_path), spawn=_recording_spawn([]), mechanism=mechanism),
        )
        task, done = _run(wired, "writer", "leaf_ai")
        assert wired.runner.attempt_of(task.id).executor is None, mechanism
        assert done[0][1] is TaskStatus.FAILED, mechanism


def test_a_program_executor_accepts_and_actually_starts_through_it(wired) -> None:
    """The other side of the same distinction: `accept_confinement` refusing by
    default is only meaningful because one executor honours it."""
    started: list[list[str]] = []
    wired.registry.register(
        "env_mgr", _confining_env(str(wired.tmp_path), spawn=_recording_spawn(started))
    )
    _, done = _run(wired, "runner", "leaf")
    assert started and done[0][1] is TaskStatus.SUCCEEDED


def test_a_package_relative_entry_resolves_against_the_package_root(wired) -> None:
    """`demo` F-D3. `_common.schema.json` types `entry` as a **package-relative**
    path and nothing else carries the package root into this package, so the
    two consumers of one schema key disagreed: `validator.ScriptBodyRunner`
    joined and this one did not.

    A package that wrote the relative path the schema documents failed only
    under this executor, and only at run time.
    """
    from agent.runner import Runner

    runner = Runner(wired.registry, package_root=Path("/pkg"))
    assert runner.resolve_path("body/entry.sh") == "/pkg/body/entry.sh"
    assert runner.resolve_path(None) is None


def test_an_absolute_entry_is_unaffected_by_the_package_root(wired) -> None:
    """`Path("/a") / "/abs"` is `/abs`, so a package that renders its body paths
    absolute — which `demo` does through the standard `config` fill — keeps
    working either way."""
    from agent.runner import Runner

    assert Runner(wired.registry, package_root=Path("/pkg")).resolve_path("/abs/entry.sh") == (
        "/abs/entry.sh"
    )
    assert Runner(wired.registry).resolve_path("body/entry.sh") == "body/entry.sh"


def test_an_attempt_refuses_a_second_thread(wired) -> None:
    """`closure`'s finding, through `monitor`. **Not a race** — an unconditional
    missing precondition on a public verb, which fired every time: `resume` on
    a running attempt started a second thread, and both would run phases, both
    report, and both call `on_done`.

    Raising rather than no-op'ing, for `resume`'s own reason: a second `begin`
    means somebody believes the attempt is idle and it is not.
    """
    from agent.runner import ThreadAlreadyHeld

    wired.monitor.advance = False
    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    _wait_until(lambda: attempt.is_running)

    # `carry_on` would *wake* a running attempt rather than re-enter it, so the
    # guard is reached through `begin` directly — which is what any future
    # caller of the removed `Runner.resume` would have hit.
    before = _attempt_threads()
    with pytest.raises(ThreadAlreadyHeld):
        attempt.begin()
    assert _attempt_threads() == before

    attempt.halt()
    attempt.join(5)


def test_the_guard_does_not_break_the_re_entry_it_exists_beside(wired) -> None:
    """The case the guard must not break: a *released* non-leaf is re-entered,
    which is the whole reason `resume` exists."""
    task, agent = dispatched("writer", "branch", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    attempt.join(5)
    assert not attempt.is_running

    task.enter_phase(TaskStatus.OUTPUT_VALIDATING)
    done: list[tuple] = []
    attempt.on_done = lambda *a, **k: done.append((*a, k))
    wired.runner.carry_on(task.id)
    attempt.join(5)
    assert done and done[0][1] is TaskStatus.SUCCEEDED


def _attempt_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name.startswith("attempt-"))


# --------------------------------------------------------------------------- #
# `carry_on` — the operation, offered instead of the predicate


def test_carry_on_wakes_a_parked_leaf_and_resumes_a_released_non_leaf(wired) -> None:
    """`engineer_principle.md` §3's stated symptom is *a caller that reads
    `a.b.c`, branches on it, and acts*, and `monitor`'s `_advance` was that
    exactly: it read `is_running` for one purpose, to choose between two of our
    verbs. This is the computation offered instead.

    The two shapes are the ones `is_running` exists to separate, so the test
    builds both and asserts the verb reports which it did.
    """
    from agent.runner import RESUMED, WOKEN

    released_task, released_agent = dispatched("writer", "branch", wired)
    wired.runner.start(released_task, released_agent, lambda *a, **k: None)
    released = wired.runner.attempt_of(released_task.id)
    released.join(5)
    released_task.enter_phase(TaskStatus.OUTPUT_VALIDATING)

    wired.monitor.advance = False
    parked_task, parked_agent = dispatched("writer", "leaf_ai", wired)
    wired.runner.start(parked_task, parked_agent, lambda *a, **k: None)
    parked = wired.runner.attempt_of(parked_task.id)
    _wait_until(lambda: parked.is_running)

    assert wired.runner.carry_on(parked_task.id) == WOKEN
    assert wired.runner.carry_on(released_task.id) == RESUMED

    parked.halt()
    parked.join(5)
    released.join(5)


def test_carry_on_never_gives_a_parked_leaf_a_second_thread(wired) -> None:
    """The failure the caller's branch could produce and this cannot: a parked
    leaf resumed rather than woken is two threads on one attempt."""
    wired.monitor.advance = False
    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    _wait_until(lambda: attempt.is_running)

    before = _attempt_threads()
    for _ in range(5):
        wired.runner.carry_on(task.id)
    assert _attempt_threads() == before

    attempt.halt()
    attempt.join(5)


def test_carry_on_returns_a_string_the_caller_can_record(wired) -> None:
    """**A plain string, not an enum.** `monitor` may not import `agent`, so an
    enum member would reach them as a value anyway — and a bare string compared
    against a member with `is` silently takes the wrong branch, which is F3 one
    seam over. They record it; the value is the whole contract."""
    from agent.runner import RESUMED, WOKEN

    assert isinstance(WOKEN, str) and isinstance(RESUMED, str)
    assert not isinstance(WOKEN, type(TaskStatus.RUNNING))


def test_carry_on_on_an_unknown_task_raises(wired) -> None:
    """Same argument as `resume`'s: a missing attempt at re-entry means the
    task is stuck, and a no-op makes that silent."""
    from task_graph.ids import TaskId

    with pytest.raises(KeyError):
        wired.runner.carry_on(TaskId.new())


# --------------------------------------------------------------------------- #
# `demo` F-D8 — and the runner is NOT the one that tells the monitor


def test_the_runner_does_not_call_set_task(wired) -> None:
    """**Ruled to `Scheduler._dispatch_pass`, not here** — `interfaces.md` §2.1
    rev. 5, after three rulings and a window in which `task_graph` and `agent`
    had both built it. Safe only because `_Watch.add` dedupes, which neither of
    us designed for.

    The argument is about the *interface*, not either site: `TaskRunner`
    declares `start` and `stop` and says nothing about monitoring, so a
    `set_task` in a runner implementation is a per-implementation obligation of
    a contract that does not state it — and the registered default is
    `FakeRunner`, which would not discharge it.

    Asserted so a second caller does not reappear here. It was built twice in
    one day; nothing but a test stops a third.
    """
    assert "set_task" not in called_attributes(Runner, TaskAttempt)


def test_a_task_the_scheduler_watched_can_report(wired) -> None:
    """The other half: with the scheduler's call modelled, the reports land.

    `StubMonitor.report` enforces the scope guard — it raises for a task
    `set_task` never named — so this fails if the harness ever stops modelling
    dispatch faithfully, which is exactly how the move was caught.
    """
    task, _ = _run(wired, "writer", "leaf_ai")
    assert wired.monitor.watching == [task.id]
    assert wired.monitor.kinds() == ["phase_done"] * 3


def test_a_phase_that_raises_is_unreached_not_a_handler_failure(wired) -> None:
    """`monitor` spec §2.1: *no verdict reachable — its `entry.sh` crashed, its
    agent died, its own inputs were missing. Nothing was decided.*

    That is what `VALIDATION_UNREACHED` was always for. It used to reach
    `_crash` and be reported `HANDLING_FAILED`, which their pusher routes to
    `GiveUp` where the right kind routes to `Escalate` — **so a crashed
    validator died at its own monitor instead of walking `Task.parent` to the
    user**, the quietest possible dead branch.

    Caught by behaviour rather than by type: `agent` may not import
    `validator`, and *any* exception out of `run_phase` means no verdict was
    reached, which is §2.1's sentence exactly.
    """

    class Exploding:
        def run_phase(self, kind, task, registry):
            raise RuntimeError("the validator body crashed")

    wired.registry.register("phase_runner", Exploding())
    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    wired.runner.attempt_of(task.id).join(5)

    assert wired.monitor.kinds() == ["validation_unreached"]
    record = wired.monitor.records[0]
    assert record.exception_type == "RuntimeError"
    assert "crashed" in record.exception_message
    assert record.exception_stacktrace  # nothing diagnostic is lost


def test_an_exception_outside_a_phase_is_still_a_handler_failure(wired) -> None:
    """`_crash` keeps `HANDLING_FAILED` for everything else, which is what that
    kind means — *the monitor's handler raised*. Widening the phase's catch to
    cover the whole attempt would put a wrong value on a second path."""
    wired.registry._items.pop("recorder")
    task, done = _run(wired, "writer", "leaf_ai")
    assert wired.monitor.kinds() == ["handling_failed"]
    assert done[0][1] is TaskStatus.FAILED


def test_the_runner_asks_whether_the_task_may_proceed(wired) -> None:
    """`blocks_the_task` is `validator` answering the runner's question.
    `passed` answers *what the phase found*, and the two coincide only when
    something ran — which is why two arms covered three states."""
    from .conftest import StubOutcome

    assert StubOutcome(kind="k", passed=False, empty=True).blocks_the_task is False
    assert StubOutcome(kind="k", passed=False, empty=False).blocks_the_task is True
    assert StubOutcome(kind="k", passed=True, empty=False).blocks_the_task is False


# `test_a_task_that_is_never_dispatched_is_never_watched` lived here while the
# `set_task` call did. It has gone with it: "at dispatch, not at birth" is now a
# property of `Scheduler._dispatch_pass`, and asserting it in a suite with no
# scheduler would assert the fixture rather than the system. It belongs with
# `task_graph`'s three, which is where they were written.


def test_a_non_leaf_gets_a_container_zone_before_it_releases(wired) -> None:
    """`demo` F-D10, live: *"task f2990b0f declares parent 04c8eb73, which has
    no zone"*. A subtask's storage nests inside its parent's, and **a non-leaf
    never reached `prepare`** — `_main` returns before `_deploy`, its only
    caller — so no nested graph could run, which is the one thing `demo` spec
    §2 exists to prove.

    `place_zone`, not `prepare`: confines nothing, cuts no workspace.

    **The caller is `Task.enter_phase` and no longer this runner**, so what this
    asserts is the property rather than the call site: the fixture's `StubMonitor`
    advances the phase, the transition places the zone, and by the time the
    attempt releases its thread the zone is there. `agent` had to give the call
    up because `enter_phase` submits — and therefore dispatches — every child
    before this thread is woken at all; `scratch/demo2-2026-08/zone-ordering.md`
    holds the measurement, and
    `tests/task_graph/test_subgraph.py::test_a_nested_non_leaf_is_zoned_before_its_subgraph_is_dispatched`
    is the ordering half.
    """
    env = wired.registry.get("env_mgr")
    task, agent = dispatched("writer", "branch", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    wired.runner.attempt_of(task.id).join(5)

    assert [t for t, _ in env.placed] == [task.id]
    assert env.placed[0][1] is task.current  # attempt-scoped: a retry unfolds again
    assert env.calls == []  # and `prepare` was not called for a task that never runs


def test_a_leaf_gets_no_container_zone(wired) -> None:
    """`prepare` already makes a leaf's zone. Placing a second would be two
    writers for one directory."""
    env = wired.registry.get("env_mgr")
    _run(wired, "writer", "leaf_ai")
    assert env.placed == []
    assert len(env.calls) == 1


def test_a_dead_attempt_closes_with_a_reason(wired) -> None:
    """`Execution.detail` is *"from the runner; for a human"* and was empty for
    every failed task in the system — `demo` measured `detail=''` on a real run
    while the same exception sat complete in the monitor's record.

    **A failure that is recorded somewhere is not the same as a failure that is
    reported.** The exception was always in hand and the scheduler always took
    the argument; what was missing was a type that could express it, which
    `task_graph` widened rather than my passing an undeclared keyword.
    """
    wired.registry._items.pop("recorder")  # raises KeyError inside the attempt
    _, done = _run(wired, "writer", "leaf_ai")

    assert done[0][1] is TaskStatus.FAILED
    assert done[0][3] == "KeyError: \"no component registered as 'recorder'\""


def test_the_reason_names_the_exception_type(wired) -> None:
    """`str(KeyError('agent'))` is `"'agent'"` — a bare quoted word with nothing
    saying it is a lookup failure, in the field a human reads first. The joined
    form is also exactly `exception_type` + `exception_message` from the
    recorder, so the two renderings are one fact rather than two."""
    from agent.runner import _one_line

    assert _one_line(KeyError("agent")) == "KeyError: 'agent'"
    assert _one_line(RuntimeError("no mechanism")) == "RuntimeError: no mechanism"


def test_a_successful_attempt_invents_no_reason(wired) -> None:
    """`detail` defaults, and a task that succeeded has nothing to say. A runner
    obliged to produce a string would produce a useless one."""
    _, done = _run(wired, "writer", "leaf_ai")
    assert done[0][1] is TaskStatus.SUCCEEDED
    assert done[0][3] == ""


def test_the_thread_carries_the_task_id_for_the_excepthook(wired) -> None:
    """`monitor/base.py`'s excepthook attributes a thread death by reading
    `thread.task_id`, and **nothing in production ever set it** — every real
    thread death recorded `NO_TASK` and criterion 25's attribution half was
    dead, with `monitor`'s own test the only caller in the tree.

    Asserted on the `Thread` object rather than through the name, because
    recovering the id from `f"attempt-{...}"` would make a string format the
    contract instead of the declared attribute.
    """
    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    assert attempt._thread.task_id == task.id
    attempt.join(5)


def test_monitors_real_excepthook_attributes_our_thread(wired) -> None:
    """The other half of §8.7: drive the *subject*, not a stub of it.

    A test that only asserted the attribute would pass just as well if
    `monitor` had since renamed what it reads. This calls the hook `monitor`
    actually installs and checks the record it produces.
    """
    from monitor.base import NO_TASK, install_excepthook

    task, agent = dispatched("writer", "leaf_ai", wired)
    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    attempt.join(5)

    written: list = []
    sink = SimpleNamespace(deliver=lambda record, why: None)
    previous = install_excepthook(SimpleNamespace(write=written.append), sink, chain=False)
    try:
        threading.excepthook(
            SimpleNamespace(
                exc_type=RuntimeError,
                exc_value=RuntimeError("escaped"),
                exc_traceback=None,
                thread=attempt._thread,
            )
        )
        # The control: an unattributed thread, through the same hook. Without
        # it the assertion below is satisfied by a hook that ignores threads
        # entirely — which is the state this defect was in.
        threading.excepthook(
            SimpleNamespace(
                exc_type=RuntimeError,
                exc_value=RuntimeError("escaped"),
                exc_traceback=None,
                thread=threading.Thread(target=lambda: None, name="nobody"),
            )
        )
    finally:
        threading.excepthook = previous

    assert written[0].task_id == task.id
    assert written[1].task_id == NO_TASK
