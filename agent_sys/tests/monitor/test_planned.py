"""The planned channel — criteria 19, 22, 23 and 24.

**Program, always.** Every test here is a test that nothing decides anything.
"""

from __future__ import annotations

import threading

import pytest

from monitor import BaseMonitor, EventKind, GiveUp, PusherMonitor, event, next_phase
from monitor.base import LAST_PHASE, NoNextPhase
from monitor.record import EVENT_KIND
from task_graph.models import TaskStatus
from task_graph.registry import Registry

from .conftest import Status, StubAttempt, StubTask, StubTaskMgr


class SpyAgentMonitor(BaseMonitor):
    """What an `AnalysingMonitor` will be: a different body behind `decide`.

    Its "agent" is a counter. Criterion 19's test is that the counter stays at
    zero through a planned advance.
    """

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.agent_consulted = 0

    def decide(self, unit):
        self.agent_consulted += 1
        return GiveUp("the analysing dispatcher is roadmap scope")


# --------------------------------------------------------------------------- #
# next_phase — the mapping, and the guard that arms itself


def test_next_phase_is_a_mapping_not_a_computation() -> None:
    assert next_phase(Status.INPUT_VALIDATING) is Status.RUNNING
    assert next_phase(Status.RUNNING) is Status.OUTPUT_VALIDATING


def test_next_phase_refuses_what_is_not_a_phase() -> None:
    """Nothing here may default to a benign outcome: a planned advance for a task
    that is not in a phase is an error, not a no-op."""
    with pytest.raises(NoNextPhase):
        next_phase(Status.WAITING_RESOURCE)
    with pytest.raises(NoNextPhase):
        next_phase(Status.OUTPUT_VALIDATING)  # completion is not an advance


def test_phase_order_names_are_task_graphs_when_they_exist() -> None:
    """A guard that arms itself.

    `task_graph` rev. 12 adds `INPUT_VALIDATING` and `OUTPUT_VALIDATING` to
    `TaskStatus`; the shipped enum is at rev. 10 and has neither, so today this
    asserts that none is present. The moment they land it asserts that
    `PHASE_ORDER` names **all three** exactly — which is the only way a
    name-resolved mapping can be checked against a type that does not exist yet.
    """
    from monitor import base

    present = [name for name in base.PHASE_ORDER if hasattr(TaskStatus, name)]
    assert present in ([], list(base.PHASE_ORDER)), (
        f"TaskStatus has {present} of {list(base.PHASE_ORDER)}; monitor.PHASE_ORDER "
        f"and task_graph's phase sequence have diverged"
    )


# --------------------------------------------------------------------------- #
# Criterion 19


def test_advance_is_one_transition(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner, registry: Registry
) -> None:
    """Criterion 19: **a planned event advances the phase and does nothing
    else.** No action set, no analysis, no threshold.

    A leaf's attempt has a thread parked on a condition, so it is woken; nothing
    is resumed and no execution is pushed.
    """
    task = task_mgr.add(StubTask(status=Status.INPUT_VALIDATING))
    monitor.set_task(task.id)
    attempt = StubAttempt()
    runner.attempts[task.id] = attempt

    monitor._run_guarded(task.id, monitor._advance, event(EventKind.PHASE_DONE, task.id))

    assert task.calls == [("enter_phase", Status.RUNNING)]
    assert attempt.woken == 1
    assert runner.resumed == []
    assert task.executions == 0


def test_agent_monitor_uses_the_same_advance(
    registry: Registry, task_mgr: StubTaskMgr, runner
) -> None:
    """Criterion 19's mechanical half: **a monitor built with an agent handles a
    planned event through the same code as one built without, and the agent is
    never consulted.**

    `_advance` is on `BaseMonitor` and no subclass replaces it, so this is
    testable by construction rather than by inspecting a prompt.
    """
    analysing = SpyAgentMonitor("analysing", registry, period=0.01)
    registry.register("monitor:analysing", analysing)
    task = task_mgr.add(StubTask(status=Status.RUNNING, monitor_spec="analysing"))
    analysing.set_task(task.id)
    runner.attempts[task.id] = StubAttempt()

    assert type(analysing)._advance is BaseMonitor._advance

    analysing._run_guarded(task.id, analysing._advance, event(EventKind.PHASE_DONE, task.id))

    assert task.calls == [("enter_phase", Status.OUTPUT_VALIDATING)]
    assert analysing.agent_consulted == 0, "a model reached the ordinary path"


def test_advance_cannot_be_overridden_by_accident() -> None:
    """Stronger than the spec asks, and the only form in which criterion 19 is
    testable by construction: `decide` is the subclass's, `_advance` is not."""
    assert PusherMonitor._advance is BaseMonitor._advance
    assert PusherMonitor.decide is not BaseMonitor.decide


# --------------------------------------------------------------------------- #
# Criterion 22 and 23 — the non-leaf re-entry


def test_non_leaf_takes_a_thread_from_the_runner(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """Criterion 22: **a non-leaf holds no thread while its subgraph runs.** Its
    thread ended at `unfold`, so there is nothing to wake and a thread must be
    made.

    That is why `wake()` and `resume()` are two calls and not one: collapsing
    them would hide, at the one place it matters, which of the two shapes a task
    is.
    """
    parent = task_mgr.add(StubTask(status=Status.RUNNING))
    monitor.set_task(parent.id)
    released = runner.make_released_attempt(parent.id)
    # The real shape: the attempt SURVIVES its thread. `attempt_of` returns it.
    assert runner.attempt_of(parent.id) is released
    assert not released.is_running

    monitor._run_guarded(
        parent.id,
        monitor._advance,
        event(EventKind.SUBGRAPH_DONE, parent.id, attributes={"from_task": "a-child"}),
    )

    assert parent.calls == [("enter_phase", Status.OUTPUT_VALIDATING)]
    assert runner.resumed == [parent.id]
    assert released.woken == 0, "wake() on a released attempt is a silent no-op"


def test_reentry_pushes_no_second_execution(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """Criterion 22: **the re-entry is the same `Execution`.** The parent was
    dispatched once, and its output-validation phase is that same attempt
    resuming — no second execution record, no second agent bound."""
    parent = task_mgr.add(StubTask(status=Status.RUNNING))
    monitor.set_task(parent.id)
    released = runner.make_released_attempt(parent.id)

    monitor._run_guarded(
        parent.id,
        monitor._advance,
        event(EventKind.SUBGRAPH_DONE, parent.id, attempt=0, attributes={"from_task": "a-child"}),
    )

    assert parent.executions == 0
    assert runner.resumed == [parent.id], "resume, which is not start"
    assert runner.attempt_of(parent.id) is released, "the same attempt object"


def test_scheduler_untouched_by_reentry(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, registry: Registry, runner
) -> None:
    """Criterion 23: **the scheduler is not involved in a non-leaf's re-entry.**

    Routing it there would make one task's progress depend on the scheduler
    *observing another task's status*, against `task_graph` §2 principles 2 and 4
    and against §3.2.1's own rule that `is_end` gets no special treatment at
    completion. `ForbiddenScheduler` raises on any attribute access, so this is
    an assertion rather than a belief.
    """
    parent = task_mgr.add(StubTask(status=Status.RUNNING))
    monitor.set_task(parent.id)
    runner.make_released_attempt(parent.id)

    # Called outside `_run_guarded`, deliberately: the guard's broad catch would
    # turn the spy's `AssertionError` into a `HANDLING_FAILED` record instead of
    # a failing test.
    monitor._current = parent.id
    monitor._advance(event(EventKind.SUBGRAPH_DONE, parent.id, attributes={"from_task": "a-child"}))
    monitor._current = None

    assert parent.calls == [("enter_phase", Status.OUTPUT_VALIDATING)]
    with pytest.raises(AssertionError):
        _ = registry.get("scheduler").anything  # the guard is real


# --------------------------------------------------------------------------- #
# Criterion 24 — the tree walk


def test_subtask_monitor_does_not_transition_parent(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, registry: Registry, runner
) -> None:
    """Criterion 24: a subtask's monitor **reports** the subgraph's completion to
    the parent's monitor; it does not transition the parent.

    The record arrives re-keyed to the parent, because the parent's monitor will
    run it through `_transition`, which refuses a task that is not the one it is
    handling. A record keeping the child's `task_id` would be a scope violation
    on arrival.
    """
    parent = task_mgr.add(StubTask(status=Status.RUNNING))
    child = task_mgr.add(StubTask(parent=parent.id, monitor_spec="child-watch"))

    child_monitor = PusherMonitor("child-watch", registry, period=0.01)
    registry.register("monitor:child-watch", child_monitor)
    child_monitor.set_task(child.id)
    monitor.set_task(parent.id)  # the default monitor watches the parent

    child_monitor._run_guarded(
        child.id, child_monitor._advance, event(EventKind.SUBGRAPH_DONE, child.id)
    )

    assert child.calls == [], "the subtask's monitor transitioned its own task"
    assert parent.calls == [], "the subtask's monitor transitioned the PARENT"

    forwarded = monitor._planned.get_nowait()
    assert forwarded is not None, "the parent's monitor was never told"
    assert forwarded.task_id == parent.id
    assert forwarded.kind is EventKind.SUBGRAPH_DONE
    assert forwarded.attributes["from_task"] == str(child.id)

    # And the parent's own monitor is what advances it.
    monitor._run_guarded(parent.id, monitor._advance, forwarded)
    assert parent.calls == [("enter_phase", Status.OUTPUT_VALIDATING)]


def test_the_root_of_the_tree_is_a_completion_not_an_escalation(
    monitor: PusherMonitor, task_mgr: StubTaskMgr
) -> None:
    """The two walks share a mechanism and differ at the top.

    The root of the task tree is the *system whole task*, and its `is_end`
    completing means the system finished — a completion, not something to
    surface as an escalation.
    """
    root = task_mgr.add(StubTask(parent=None, status=Status.RUNNING))
    monitor.set_task(root.id)

    monitor._notify_parent_done(event(EventKind.SUBGRAPH_DONE, root.id))

    assert len(monitor._planned) == 0
    assert root.calls == []


# --------------------------------------------------------------------------- #
# Prioritisation without starvation


def test_starvation(monitor: PusherMonitor, task_mgr: StubTaskMgr, runner) -> None:
    """The ordering in the mainloop **prioritises; it does not starve.**

    A planned advance is a fixed non-blocking transition, so the planned queue
    drains to empty in bounded time and the loop reaches `buffer.get` on the next
    round. A saturated planned queue is therefore still not a stuck decision.
    """
    unplanned_task = task_mgr.add(StubTask())
    monitor.set_task(unplanned_task.id)
    handled = threading.Event()

    def decide(unit):
        handled.set()
        return GiveUp("seen")

    monitor.decide = decide  # type: ignore[method-assign]

    for _ in range(200):
        task = task_mgr.add(StubTask(status=Status.INPUT_VALIDATING))
        monitor.set_task(task.id)
        runner.attempts[task.id] = StubAttempt()
        monitor.report(event(EventKind.PHASE_DONE, task.id))
    monitor.report(event(EventKind.VALIDATION_FAILED, unplanned_task.id))

    thread = threading.Thread(target=monitor.mainloop, daemon=True)
    thread.start()
    assert handled.wait(5.0), "200 planned advances starved one decision"
    monitor.stop()
    thread.join(2.0)


def test_planned_work_is_taken_before_a_decision(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner
) -> None:
    """A task waiting to advance is a task doing nothing, and a decision can
    block on the scheduler's lock while a transition cannot meaningfully be
    deferred. So the advance is taken first even though it was reported second.
    """
    advancing = task_mgr.add(StubTask(status=Status.INPUT_VALIDATING))
    deciding = task_mgr.add(StubTask())
    for task in (advancing, deciding):
        monitor.set_task(task.id)
    runner.attempts[advancing.id] = StubAttempt()

    order: list[str] = []
    done = threading.Event()

    def decide(unit):
        order.append("advanced first" if advancing.calls else "decided first")
        done.set()
        return GiveUp("x")

    monitor.decide = decide  # type: ignore[method-assign]

    monitor.report(event(EventKind.VALIDATION_FAILED, deciding.id))  # queued first
    monitor.report(event(EventKind.PHASE_DONE, advancing.id))

    thread = threading.Thread(target=monitor.mainloop, daemon=True)
    thread.start()
    assert done.wait(3.0)
    monitor.stop()
    thread.join(2.0)

    assert order == ["advanced first"]


# --------------------------------------------------------------------------- #
# the end of the channel — the last phase's PHASE_DONE is a completion


def _kinds_for(registry: Registry, task: StubTask) -> list[str]:
    store = registry.get("store_mgr")
    return [r["kind"] for r in store.read_all(EVENT_KIND) if r["task_id"] == str(task.id)]


def test_the_last_phases_report_is_a_completion_not_an_advance(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner, registry: Registry
) -> None:
    """`PHASE_ORDER` has three phases and two advances, but `agent`'s `_close`
    reports a **third** `PHASE_DONE` and then completes the task itself.

    Before this, `next_phase` refused — correctly — and `_run_guarded` turned the
    refusal into a persisted `HANDLING_FAILED` **for every successful task**.
    `demo-2` saw it on the first task ever to succeed.
    """
    task = task_mgr.add(StubTask(status=Status.OUTPUT_VALIDATING))
    monitor.set_task(task.id)
    runner.attempts[task.id] = StubAttempt()

    monitor._run_guarded(
        task.id,
        monitor._advance,
        event(EventKind.PHASE_DONE, task.id, attributes={"phase": LAST_PHASE}),
    )

    assert task.calls == [], "the last phase was advanced out of"
    assert EventKind.HANDLING_FAILED.value not in _kinds_for(registry, task), (
        "a successful task recorded a handling failure"
    )


def test_the_completion_is_read_from_the_record_not_the_status(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner, registry: Registry
) -> None:
    """**The status has usually already moved.** `_close` reports and *then*
    calls `on_done`, so by the time the monitor drains, the task is `SUCCEEDED`
    — which is why the old failure alternated between two different refusals
    depending on who won.

    Keying on the record makes it deterministic. `SUCCEEDED` is not in
    `PHASE_ORDER` at all, so this passes only because the record is consulted
    first.
    """
    task = task_mgr.add(StubTask(status=TaskStatus.SUCCEEDED))
    monitor.set_task(task.id)

    monitor._run_guarded(
        task.id,
        monitor._advance,
        event(EventKind.PHASE_DONE, task.id, attributes={"phase": LAST_PHASE}),
    )

    assert task.calls == []
    assert EventKind.HANDLING_FAILED.value not in _kinds_for(registry, task)


def test_a_phase_done_without_the_attribute_still_refuses_loudly(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner, registry: Registry
) -> None:
    """**Absence is not granted a benign default.** A reporter that omits
    `phase` falls through to the old behaviour and the refusal is recorded, so
    the guard cannot quietly absorb a producer that stops naming the phase."""
    task = task_mgr.add(StubTask(status=Status.OUTPUT_VALIDATING))
    monitor.set_task(task.id)

    monitor._run_guarded(task.id, monitor._advance, event(EventKind.PHASE_DONE, task.id))

    assert task.calls == []
    assert EventKind.HANDLING_FAILED.value in _kinds_for(registry, task)


def test_a_subgraph_re_entry_carrying_a_stale_phase_still_advances(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, runner, registry: Registry
) -> None:
    """**Why the check is scoped to `PHASE_DONE`.**

    `_notify_parent_done` rekeys to `SUBGRAPH_DONE` and copies the child's
    attributes, so a non-leaf re-entry can arrive carrying a `phase` that
    belongs to another task. Matching on the kind keeps that record on the
    advancing path; matching on the attribute alone would strand the parent.
    """
    parent = task_mgr.add(StubTask(status=Status.RUNNING))
    monitor.set_task(parent.id)
    runner.attempts[parent.id] = StubAttempt()

    stale = event(
        EventKind.SUBGRAPH_DONE,
        parent.id,
        attributes={"phase": LAST_PHASE, "from_task": "some-child"},
    )
    monitor._run_guarded(parent.id, monitor._advance, stale)

    assert parent.calls == [("enter_phase", Status.OUTPUT_VALIDATING)]


# --------------------------------------------------------------------------- #
# criterion 24's producing half — what creates the first SUBGRAPH_DONE


def test_an_is_end_subtask_announces_its_subgraphs_completion(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, registry: Registry, runner
) -> None:
    """**The producer that never existed.**

    `SUBGRAPH_DONE` was declared, consumed by `_advance`, and re-emitted by
    `_notify_parent_done` — and **nothing anywhere created the first one.** A
    non-leaf's `_main` ends its thread at `unfold` with the task in `RUNNING`,
    and this re-entry is the only thing that moves it, so every non-leaf — the
    root included — sat in `main: running` for ever and no run of the demo
    terminated cleanly.

    `test_subtask_monitor_does_not_transition_parent` passed throughout, because
    **it constructs the `SUBGRAPH_DONE` by hand.** The test was the missing
    producer: it proved the relay forwards, which was true, and could not see
    that nobody upstream ever spoke.

    A subgraph is finished exactly when its `is_end` subtask runs out of phases,
    so that is the trigger.
    """
    parent = task_mgr.add(StubTask(status=Status.RUNNING))
    child = task_mgr.add(StubTask(parent=parent.id, monitor_spec="child-watch", is_end=True))

    child_monitor = PusherMonitor("child-watch", registry, period=0.01)
    registry.register("monitor:child-watch", child_monitor)
    child_monitor.set_task(child.id)
    monitor.set_task(parent.id)

    child_monitor._run_guarded(
        child.id,
        child_monitor._advance,
        event(EventKind.PHASE_DONE, child.id, attributes={"phase": LAST_PHASE}),
    )

    assert child.calls == [], "the completing subtask was transitioned"
    assert parent.calls == [], "the subtask's monitor transitioned the parent"

    forwarded = monitor._planned.get_nowait()
    assert forwarded is not None, "the parent was never told its subgraph finished"
    assert forwarded.kind is EventKind.SUBGRAPH_DONE
    assert forwarded.task_id == parent.id
    assert forwarded.attributes["from_task"] == str(child.id)

    # End to end: the parent leaves RUNNING, which is the wall this removes.
    monitor._run_guarded(parent.id, monitor._advance, forwarded)
    assert parent.calls == [("enter_phase", Status.OUTPUT_VALIDATING)]


def test_a_subtask_that_is_not_the_end_announces_nothing(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, registry: Registry, runner
) -> None:
    """Only the exit point ends the subgraph. A middle subtask finishing its
    phases is its own completion and says nothing about its siblings."""
    parent = task_mgr.add(StubTask(status=Status.RUNNING))
    child = task_mgr.add(StubTask(parent=parent.id, monitor_spec="mid-watch", is_end=False))

    child_monitor = PusherMonitor("mid-watch", registry, period=0.01)
    registry.register("monitor:mid-watch", child_monitor)
    child_monitor.set_task(child.id)
    monitor.set_task(parent.id)

    child_monitor._run_guarded(
        child.id,
        child_monitor._advance,
        event(EventKind.PHASE_DONE, child.id, attributes={"phase": LAST_PHASE}),
    )

    assert len(monitor._planned) == 0, "a non-terminal subtask ended the subgraph"
    assert parent.calls == []


def test_the_root_completing_announces_to_nobody(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, registry: Registry, runner
) -> None:
    """`is_end` defaults to True, so the root reaches the announcement and must
    survive having no parent. `_notify_parent_done` returns on a `None` parent;
    this reaches it through the completion branch rather than by calling it."""
    root = task_mgr.add(StubTask(parent=None, status=Status.OUTPUT_VALIDATING))
    monitor.set_task(root.id)

    monitor._run_guarded(
        root.id,
        monitor._advance,
        event(EventKind.PHASE_DONE, root.id, attributes={"phase": LAST_PHASE}),
    )

    assert root.calls == []
    assert len(monitor._planned) == 0
    assert EventKind.HANDLING_FAILED.value not in _kinds_for(registry, root)
