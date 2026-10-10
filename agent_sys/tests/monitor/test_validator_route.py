"""The validator's two outcomes — criteria 16 and 18.

**The route itself is `validator`'s to build**; spec §9 records it as missing on
both sides. What is this module's, and what is tested here, is that both outcomes
have a kind, that the two are distinguishable in the record, and that a reporter
which takes no corrective action of its own is a shape the monitor supports.
"""

from __future__ import annotations

from monitor import EventKind, PusherMonitor, event
from monitor.record import EVENT_KIND
from task_graph.registry import Registry

from .conftest import StubTask, StubTaskMgr


class StubValidatorPhase:
    """A validation phase, reduced to its obligation to this module.

    Both of §2.1's outcomes are reported and they differ in *kind*, never in
    whether the monitor hears about them:

    | | | Reported |
    |---|---|---|
    | a verdict of "fail" | the validator worked; the answer is *no*, **and the task is now terminal, so the plan is broken** | yes |
    | no verdict reachable | its `entry.sh` crashed, its agent died, its own inputs were missing. Nothing was decided | yes |
    """

    def __init__(self, monitor: PusherMonitor) -> None:
        self.monitor = monitor
        self.reruns = 0  # must stay zero

    def finish(self, task: StubTask, *, verdict_reached: bool, passed: bool) -> None:
        if verdict_reached and passed:
            kind = EventKind.PHASE_DONE
        elif verdict_reached:
            kind = EventKind.VALIDATION_FAILED
        else:
            kind = EventKind.VALIDATION_UNREACHED
        self.monitor.report(event(kind, task.id, reported_by="phase_runner"))


def kinds_for(registry: Registry, task: StubTask) -> list[str]:
    return [
        r["kind"]
        for r in registry.get("store_mgr").read_all(EVENT_KIND)
        if r["task_id"] == str(task.id)
    ]


def test_fail_and_unreached_are_different_kinds(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, registry: Registry
) -> None:
    """Criterion 16: **both of the validator's bad outcomes reach the monitor**,
    and they are distinguishable in the record by `kind`.

    The first is the one worth being explicit about, because it is where the
    wrong test leads somewhere bad. "The validator worked correctly" is true and
    irrelevant: a task that failed its output validation is terminal, its
    dependents will never run, and **the graph will not finish.**
    """
    failed, unreached = task_mgr.add(StubTask()), task_mgr.add(StubTask())
    for task in (failed, unreached):
        monitor.set_task(task.id)
    phase = StubValidatorPhase(monitor)

    phase.finish(failed, verdict_reached=True, passed=False)
    phase.finish(unreached, verdict_reached=False, passed=False)

    assert kinds_for(registry, failed) == [EventKind.VALIDATION_FAILED.value]
    assert kinds_for(registry, unreached) == [EventKind.VALIDATION_UNREACHED.value]
    assert len(monitor._buffer) == 2
    assert len(monitor._planned) == 0


def test_a_failed_validation_leaves_a_record_and_does_not_go_quiescent(
    monitor: PusherMonitor, task_mgr: StubTaskMgr, registry: Registry
) -> None:
    """Criterion 16's second half — **the defect main spec §10 recorded.**

    Main spec §10 and ROADMAP §2 both say that in the alpha such a branch goes
    quiescent and *nothing surfaces an error*. Under principle 1 it does not go
    quiescent: it is reported, it is recorded, and it is escalated, whether or
    not the alpha can do anything clever about it. **The point is that the branch
    is no longer silent**, which was the whole of the recorded defect.
    """
    task = task_mgr.add(StubTask(parent=None))
    monitor.set_task(task.id)
    StubValidatorPhase(monitor).finish(task, verdict_reached=True, passed=False)

    unit = monitor._buffer.get(0.05)
    assert unit is not None
    monitor._run_guarded(task.id, monitor._handle, unit, release=True)

    recorded = kinds_for(registry, task)
    assert recorded[0] == EventKind.VALIDATION_FAILED.value
    assert EventKind.ESCALATED.value in recorded


def test_a_passing_validation_is_a_planned_event(
    monitor: PusherMonitor, task_mgr: StubTaskMgr
) -> None:
    """The same call and the same reporter — only the `kind` differs. That is
    what makes the two channels one inbound surface."""
    task = task_mgr.add(StubTask())
    monitor.set_task(task.id)
    StubValidatorPhase(monitor).finish(task, verdict_reached=True, passed=True)

    assert len(monitor._planned) == 1
    assert len(monitor._buffer) == 0


def test_validator_does_not_rerun_itself(monitor: PusherMonitor, task_mgr: StubTaskMgr) -> None:
    """Criterion 18, for the second reporter the alpha knows about.

    A validator that quietly re-ran itself would be a private recovery path:
    invisible to the record, unreachable by the analysing dispatcher, and
    untunable from outside. **One decision-maker is worth more than the local
    cleverness it displaces.**
    """
    task = task_mgr.add(StubTask())
    monitor.set_task(task.id)
    phase = StubValidatorPhase(monitor)

    phase.finish(task, verdict_reached=False, passed=False)

    assert phase.reruns == 0
    assert len(monitor._buffer) == 1, "the outcome went to the monitor and nowhere else"
