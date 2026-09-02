"""The completeness gate — `monitor` spec §4.1.0, owned by `agent`.

`monitor` criteria 4 and 18 are the monitor's; what is tested here is the half
this package owes: the four checks produce four kinds, and **the runner takes no
corrective action of its own.**
"""

from __future__ import annotations

import threading

from agent.gate import run_gate
from agent.runner import Runner, TaskAttempt
from monitor.protocols import Budget, EventKind
from task_graph.ids import HandoffId
from task_graph.models import TaskStatus

from .conftest import StubManifest, StubStore, called_attributes, dispatched


def _hid() -> HandoffId:
    return HandoffId.new()


def test_four_failures_each_report() -> None:
    """Four independent mechanical failures, and all four run: reporting one
    absence while a budget was also blown would hide half the situation."""
    missing, present, wrong = _hid(), _hid(), _hid()
    content = type("C", (), {"items": {"script": type("I", (), {"path": "/no/such/file"})()}})()
    store = StubStore(
        {
            present: StubManifest(done_by_self_check=False),
            # The declaration lives in the copied-out content, never on the
            # manifest — see `test_gate_against_the_real_store.py`.
            wrong: StubManifest(done_by_self_check=True),
        }
    )
    store.contents[wrong] = content

    failures = run_gate(
        [missing, present, wrong],
        {"tokens": 10.0},
        store=store,
        budget=Budget(max_tokens=5),
    )
    assert {f.kind for f in failures} == {
        EventKind.OUTPUT_ABSENT,
        EventKind.SELF_CHECK_UNSET,
        EventKind.OUTPUT_NOT_EXECUTABLE,
        EventKind.BUDGET_EXCEEDED,
    }
    assert str(missing) in next(f.message for f in failures if f.kind is EventKind.OUTPUT_ABSENT)


def test_a_complete_delivery_passes_the_gate() -> None:
    hid = _hid()
    store = StubStore({hid: StubManifest(done_by_self_check=True)})
    assert run_gate([hid], {"tokens": 1.0}, store=store, budget=Budget(max_tokens=5)) == []


def test_self_check_absent_is_not_a_failure() -> None:
    """`done_by_self_check` **does not exist on `handoff` yet** (`monitor` spec
    §4.1.2, §9). Absent is not the producing agent's fault, so only a
    present-and-false reports — the check activates the day the field lands."""
    hid = _hid()
    store = StubStore({hid: StubManifest(done_by_self_check=None)})
    assert run_gate([hid], {}, store=store, budget=None) == []


def test_no_budget_is_no_budget_failure() -> None:
    hid = _hid()
    store = StubStore({hid: StubManifest()})
    assert run_gate([hid], {"tokens": 1e9}, store=store, budget=None) == []


def test_the_gate_is_not_the_validator() -> None:
    """It asks whether there is something to check, never whether the work is
    right. Nothing in it reads a verdict."""
    import agent.gate

    called = called_attributes(agent.gate)
    assert not called & {"read_verdicts", "run_phase"}


def test_runner_takes_no_corrective_action(wired) -> None:
    """**The runner never pushes.** The failing path calls `report()` and
    returns; a runner that retried internally would be a second failure policy
    the record cannot see and the analysing dispatcher cannot reach."""
    called = called_attributes(Runner, TaskAttempt)
    assert not called & {"instruct", "interrupt", "restart", "cancel", "resume_task"}


def test_a_failing_gate_reports_and_the_task_stays_running(wired) -> None:
    """**The cycle stays below the scheduler**: nothing in it moves task
    status, and the graph sees one task `RUNNING` throughout."""
    hid = _hid()
    wired.registry.register("handoff_store", StubStore())
    wired.registry.register("budget", Budget())
    task, agent = dispatched("writer", "leaf_ai", wired)
    task.outputs.append(hid)

    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    _settle(attempt)
    attempt.halt()
    attempt.join(5)

    assert "output_absent" in wired.monitor.kinds()
    assert task.status is TaskStatus.RUNNING


def test_a_gate_report_carries_how_the_body_ended(wired) -> None:
    """`output_absent` is the same observation for a body that crashed and one
    that exited 0 having written to the wrong path. **`monitor` ruled the
    discriminator is payload, not a kind** — kinds name the phase a body
    terminated in, and both of those terminated in the same phase.

    Separate keys rather than prose folded into the message: a value a reader
    branches on. The `None` case is pinned too, because the first version of
    the lift left a `None` in `**extra`, where `EventRecord`'s `extra="forbid"`
    raised on the very case meaning "there is nothing to add".
    """
    wired.registry.register("handoff_store", StubStore())
    wired.registry.register("budget", Budget())
    task, agent = dispatched("writer", "leaf_ai", wired)
    task.outputs.append(_hid())

    wired.runner.start(task, agent, lambda *a, **k: None)
    attempt = wired.runner.attempt_of(task.id)
    _settle(attempt)
    attempt.halt()
    attempt.join(5)

    absent = next(r for r in wired.monitor.records if r.kind is EventKind.OUTPUT_ABSENT)
    assert isinstance(absent.attributes["exit_status"], str)
    assert "never delivered" in absent.attributes["message"]
    assert "exit_status" not in absent.attributes["message"]


def _settle(attempt, timeout: float = 5.0) -> None:
    deadline = threading.Event()
    deadline.wait(0.2)
