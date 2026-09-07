"""The alpha's decision function — criterion 12."""

from __future__ import annotations

import pytest

from monitor import (
    PUSH_MESSAGE,
    Escalate,
    EventKind,
    GiveUp,
    Push,
    PusherMonitor,
    Unit,
    event,
)
from monitor.pusher import GATE_KINDS
from task_graph.registry import Registry

from .conftest import BareRunner, StubAttempt, StubBackend, StubTask, StubTaskMgr


def unit_for(task: StubTask, kind: EventKind, attempt: int = 0) -> Unit:
    return Unit(task.id, (event(kind, task.id, attempt=attempt),))


@pytest.mark.parametrize("kind", sorted(GATE_KINDS, key=lambda k: k.value))
def test_decision_table_pushes_a_live_agent(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner, kind: EventKind
) -> None:
    """A gate kind, a live handle, and no push yet for this attempt → `Push`.

    **"Keep going" is expressible on a returned agent. Measured, not assumed**: a
    `ResultMessage` ends a *turn*, not the session and not the process, and a
    live probe pushed a returned agent on the same session id in ~2 s.
    """
    task = task_mgr.add(StubTask())
    monitor.set_task(task.id)
    backend = StubBackend()
    runner.attempts[task.id] = StubAttempt(backend)

    decision = monitor.decide(unit_for(task, kind))

    assert isinstance(decision, Push)
    assert decision.handle is backend
    assert decision.message == PUSH_MESSAGE == "continue, do it until finished"


def test_the_pusher_never_reaches_for_restart(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """Spec §7's cost table is the reason and it is measured: push ~2 s and
    lossless; resume ~5.5 s warm and drops `permission_mode`, `--mcp-config`,
    `--settings` and `--add-dir` — the per-attempt wiring `env_mgr` prepared;
    restart loses all context plus the zone.

    **An agent that returned nearly-finished work and merely failed to publish is
    the cheapest possible fault, and restart is the most expensive possible
    reaction to it.**
    """
    task = task_mgr.add(StubTask())
    monitor.set_task(task.id)
    runner.attempts[task.id] = StubAttempt()

    for kind in EventKind:
        decision = monitor.decide(unit_for(task, kind))
        assert not isinstance(decision, str)
        assert "restart" not in repr(decision).lower()


def test_a_second_gate_failure_is_ineffective_and_escalates(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner, registry: Registry
) -> None:
    """Criterion 9's third state, and the concrete reason spec §8.2 rejected the
    OpenTelemetry SDK: **OTel is emit-only by construction, and this decision has
    to read back what it wrote.**"""
    task = task_mgr.add(StubTask())
    monitor.set_task(task.id)
    backend = StubBackend()
    runner.attempts[task.id] = StubAttempt(backend)

    first = unit_for(task, EventKind.OUTPUT_ABSENT)
    monitor._run_guarded(task.id, monitor._handle, first)
    assert backend.instructions == [PUSH_MESSAGE]

    second = unit_for(task, EventKind.OUTPUT_ABSENT)
    decision = monitor.decide(second)
    assert isinstance(decision, Escalate)

    kinds = [
        r["kind"]
        for r in registry.get("store_mgr").read_all("event")
        if r["task_id"] == str(task.id)
    ]
    assert kinds.count(EventKind.PUSH_ATTEMPTED.value) == 1
    assert kinds.count(EventKind.PUSH_INEFFECTIVE.value) == 1


def test_budget_exceeded_escalates_rather_than_pushing(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """The budget is what **bounds** the gate loop (spec §4.1.3). Pushing past it
    would remove the bound, which is the whole reason the exit is a monitor
    decision rather than a retry count buried in the runner."""
    task = task_mgr.add(StubTask())
    monitor.set_task(task.id)
    runner.attempts[task.id] = StubAttempt()

    decision = monitor.decide(unit_for(task, EventKind.BUDGET_EXCEEDED))
    assert isinstance(decision, Escalate)
    assert "bound" in decision.why


def test_both_validator_outcomes_escalate(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    task = task_mgr.add(StubTask())
    monitor.set_task(task.id)
    runner.attempts[task.id] = StubAttempt()

    for kind in (EventKind.VALIDATION_FAILED, EventKind.VALIDATION_UNREACHED):
        assert isinstance(monitor.decide(unit_for(task, kind)), Escalate)


def test_an_unhandled_kind_gives_up_loudly(monitor: PusherMonitor, task_mgr: StubTaskMgr) -> None:
    """`GiveUp` is recorded as `MONITOR_GAVE_UP`. Nothing here may default to a
    benign outcome, and "the pusher had no action" is not silence."""
    task = task_mgr.add(StubTask())
    monitor.set_task(task.id)
    decision = monitor.decide(unit_for(task, EventKind.LOOP_STALLED))
    assert isinstance(decision, GiveUp)
    assert "loop_stalled" in decision.why


def test_no_live_agent_escalates_rather_than_pushing(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """`Push` needs a live agent, and a non-leaf never has one.

    **This test used to assert the wrong thing.** It registered a runner with no
    `attempt_of` at all and asserted the pusher degraded — which was right while
    `agent` was declaration-only and became meaningless the day `attempt_of`
    landed. Worse, `live_handle`'s `getattr` fallback made a *renamed* accessor
    produce the same `Escalate`, so a regression would have looked exactly like
    the documented degraded state — `interfaces.md` §4.11's first row.

    The case that actually occurs is this one: the attempt exists and has no
    executor, because it has not reached its main phase or is a non-leaf and
    never will. `closure`'s review, M3.
    """
    task = task_mgr.add(StubTask())
    monitor.set_task(task.id)
    runner.attempts[task.id] = StubAttempt()
    runner.attempts[task.id].executor = None  # before the main phase

    decision = monitor.decide(unit_for(task, EventKind.OUTPUT_ABSENT))

    assert isinstance(decision, Escalate)
    assert "not in its main phase" in decision.why


def test_the_two_reasons_for_no_handle_are_distinguishable(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """`None` is two real situations and the record must not flatten them.

    `agent-mod` flagged it: `executor` is `None` for an attempt before its main
    phase *and* for a non-leaf that never has one, while **no attempt at all**
    means `Runner.stop` removed it and the agent is gone. At a gate failure the
    first is surprising — the gate runs at the *end* of the main phase — so
    putting both behind one sentence would hide the surprising one behind the
    ordinary one. That is this module's own thesis applied to its own escalation
    reason.
    """
    gone, early = task_mgr.add(StubTask()), task_mgr.add(StubTask())
    for task in (gone, early):
        monitor.set_task(task.id)
    runner.attempts[early.id] = StubAttempt()
    runner.attempts[early.id].executor = None
    # `gone` has no entry at all: stop() removed it.

    why_gone = monitor.decide(unit_for(gone, EventKind.OUTPUT_ABSENT)).why
    why_early = monitor.decide(unit_for(early, EventKind.OUTPUT_ABSENT)).why

    assert why_gone != why_early
    assert "agent is gone" in why_gone
    assert "main phase" in why_early


def test_a_missing_accessor_is_loud_not_a_degraded_push(
    registry: Registry, task_mgr: StubTaskMgr
) -> None:
    """A runner without `attempt_of` now **raises**, and the raise is recorded.

    It must not read as "no live agent": that is the failure mode
    `interfaces.md` §4.11 names, where a check that reports nothing is
    indistinguishable from a check that found nothing. `_run_guarded` turns it
    into a `HANDLING_FAILED` record rather than swallowing it.
    """
    registry.register("runner", BareRunner())
    monitor = PusherMonitor("default", registry, period=0.01)
    registry.register("monitor:default", monitor)
    task = task_mgr.add(StubTask())
    monitor.set_task(task.id)

    with pytest.raises(AttributeError):
        monitor.decide(unit_for(task, EventKind.OUTPUT_ABSENT))

    monitor._run_guarded(task.id, monitor._handle, unit_for(task, EventKind.OUTPUT_ABSENT))
    kinds = [
        r["kind"]
        for r in registry.get("store_mgr").read_all("event")
        if r["task_id"] == str(task.id)
    ]
    assert kinds == [EventKind.HANDLING_FAILED.value]


def test_push_is_recorded_before_the_call(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner, registry: Registry
) -> None:
    """So an `instruct` that raises still leaves the attempt visible."""

    class Exploding(StubBackend):
        def instruct(self, message: str) -> None:
            raise RuntimeError("the transport was not ready for writing")

    task = task_mgr.add(StubTask())
    monitor.set_task(task.id)
    runner.attempts[task.id] = StubAttempt(Exploding())

    monitor._run_guarded(task.id, monitor._handle, unit_for(task, EventKind.OUTPUT_ABSENT))

    kinds = [
        r["kind"]
        for r in registry.get("store_mgr").read_all("event")
        if r["task_id"] == str(task.id)
    ]
    assert EventKind.PUSH_ATTEMPTED.value in kinds
    assert EventKind.HANDLING_FAILED.value in kinds


class ProgramLikeExecutor:
    """`agent.backends.program.ProgramExecutor`'s shape: level 1 only.

    Verified against the real class — `status`, `instruct` and `query` are all
    absent, which is `interfaces.md` §4.4 by construction: *"a program executor
    implements `Executor` and has no level 2 to raise from."*
    """

    def start_async(self, on_done): ...
    def wait(self): ...
    def stop(self): ...


def test_a_program_body_is_escalated_not_pushed(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """A gate failure on a `kind: program` body has no agent to instruct.

    **This was a live defect, reachable today through `OUTPUT_ABSENT` and
    nothing to do with `done_by_self_check`.** `live_handle` returned
    `attempt.executor` whenever it was not `None`; for a program body that is a
    `ProgramExecutor`, so `decide` returned `Push` and `_push` called `instruct`
    on something that has none — measured as `PUSH_ATTEMPTED` then
    `HANDLING_FAILED`, an `AttributeError` where a decision belongs
    (`scratch/impl-2026-08/monitor/p11_program_body_push.py`).

    The guard is `isinstance(executor, Pushable)`, which is **what
    `runtime_checkable` was declared for** — until now the decorator was kept
    for a check nobody could run.
    """
    task = task_mgr.add(StubTask(parent=None))
    monitor.set_task(task.id)
    runner.attempts[task.id] = StubAttempt(ProgramLikeExecutor())

    decision = monitor.decide(unit_for(task, EventKind.OUTPUT_ABSENT))

    assert isinstance(decision, Escalate)
    assert "program body" in decision.why

    monitor._run_guarded(task.id, monitor._handle, unit_for(task, EventKind.OUTPUT_ABSENT))
    kinds = [
        r["kind"]
        for r in monitor._r.get("store_mgr").read_all("event")
        if r["task_id"] == str(task.id)
    ]
    assert EventKind.PUSH_ATTEMPTED.value not in kinds, "a program was pushed"
    assert EventKind.HANDLING_FAILED.value not in kinds, "instruct raised"
    assert EventKind.ESCALATED.value in kinds


def test_the_three_reasons_for_no_handle_are_all_distinct(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """Gone, not-yet-in-its-main-phase, and no-agent-at-all are three facts.

    The third arrived with the program-body fix; the first two were separated
    earlier for the same reason. A `None` handle now always says which.
    """
    gone, early, program = (task_mgr.add(StubTask()) for _ in range(3))
    for t in (gone, early, program):
        monitor.set_task(t.id)
    runner.attempts[early.id] = StubAttempt()
    runner.attempts[early.id].executor = None
    runner.attempts[program.id] = StubAttempt(ProgramLikeExecutor())

    whys = [
        monitor.decide(unit_for(t, EventKind.OUTPUT_ABSENT)).why for t in (gone, early, program)
    ]
    assert len(set(whys)) == 3, whys
    assert "agent is gone" in whys[0]
    assert "main phase" in whys[1]
    assert "program body" in whys[2]
