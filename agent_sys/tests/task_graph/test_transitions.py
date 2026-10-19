"""A task owns its transitions — criteria 45, 46, 47, 48.

`Scheduler._move` was already the single writer of `task.status`. What these
criteria add is that a *transition* becomes the only caller of the paths that
reach it: a monitor, an agent or a runner acts by calling a verb on the task,
never by assigning a field.

The instrument is the marker span `test_authority.py` established, not stack
inspection — a marker records who claimed responsibility, while a stack frame
only records who happened to be on it.
"""

import ast
import inspect

import pytest

from task_graph.models import PHASES, Task, TaskStateError, TaskStatus
from task_graph.scheduler import Scheduler

from .conftest import make_task, new_handoffs

TRANSITIONS = ("cancel", "restart", "fail", "replace_with", "enter_phase", "unfold")


class Trace:
    """A log of status writes, dispatch requests, and who claimed them."""

    def __init__(self, monkeypatch) -> None:
        self.log: list[str] = []
        self._install(monkeypatch)

    def _install(self, monkeypatch) -> None:
        log = self.log
        original_setattr = Task.__setattr__

        def setattr_(self, name, value):
            if name == "status":
                log.append(f"status={value.value}")
            return original_setattr(self, name, value)

        monkeypatch.setattr(Task, "__setattr__", setattr_, raising=False)

        for name in TRANSITIONS:
            monkeypatch.setattr(Task, name, self._wrap(getattr(Task, name), f"transition:{name}"))
        monkeypatch.setattr(
            Scheduler, "try_dispatch", self._wrap(Scheduler.try_dispatch, "try_dispatch")
        )

    def _wrap(self, func, label):
        log = self.log

        def wrapper(*args, **kwargs):
            log.append(f"<{label}>")
            try:
                return func(*args, **kwargs)
            finally:
                log.append(f"</{label}>")

        return wrapper

    def actor(self, name: str):
        """A context manager standing in for a monitor, an agent, or a runner."""
        return _Span(self.log, name)

    def spans(self, prefix: str) -> list[range]:
        out, stack = [], []
        for i, entry in enumerate(self.log):
            if entry.startswith(f"<{prefix}"):
                stack.append(i)
            elif entry.startswith(f"</{prefix}"):
                out.append(range(stack.pop(), i))
        return out

    def inside(self, prefix: str) -> set[int]:
        return {i for span in self.spans(prefix) for i in span}


class _Span:
    def __init__(self, log, name) -> None:
        self.log, self.name = log, name

    def __enter__(self):
        self.log.append(f"<{self.name}>")
        return self

    def __exit__(self, *exc):
        self.log.append(f"</{self.name}>")
        return False


# ------------------------------------------------------------- criterion 45


def test_no_frame_outside_a_transition_assigns_a_status(scheduler, runner, registry, monkeypatch):
    """A monitor, an agent and a runner each act; every status write they cause
    happens inside a task transition."""
    trace = Trace(monkeypatch)
    task = make_task()
    scheduler.submit(task)

    with trace.actor("runner"):
        runner.advance(registry, task.id)
    with trace.actor("monitor"):
        registry.get("task_mgr").get(task.id).fail("judged dead")
    with trace.actor("monitor"):
        registry.get("task_mgr").get(task.id).restart()

    outsiders = trace.inside("runner") | trace.inside("monitor")
    in_transition = trace.inside("transition")
    offenders = [
        i
        for i, e in enumerate(trace.log)
        if e.startswith("status=") and i in outsiders - in_transition
    ]
    assert not offenders, [trace.log[i] for i in offenders]
    assert any(e.startswith("status=") for e in trace.log), "the spy saw nothing at all"


def test_the_spy_would_catch_a_status_written_from_outside(scheduler, registry, monkeypatch):
    """The test is only meaningful if it can fail. Simulate the erosion."""
    trace = Trace(monkeypatch)
    task = make_task(inputs=new_handoffs(1))
    scheduler.submit(task)

    with trace.actor("monitor"):
        registry.get("task_mgr").get(task.id).status = TaskStatus.CANCELLED

    outsiders = trace.inside("monitor") - trace.inside("transition")
    assert [i for i, e in enumerate(trace.log) if e.startswith("status=") and i in outsiders]


# ------------------------------------------------------------- criterion 46


def test_every_dispatch_originates_in_a_transition_or_the_public_api(
    scheduler, runner, registry, monkeypatch
):
    trace = Trace(monkeypatch)
    mid, never = new_handoffs(2)
    producer = make_task(outputs=[mid])
    consumer = make_task(inputs=[mid, never])  # `never` keeps it queued to cancel
    scheduler.submit(producer)
    scheduler.submit(consumer)

    with trace.actor("agent"):
        runner.produce(registry, producer.id)
    with trace.actor("runner"):
        runner.finish(producer.id)
    with trace.actor("monitor"):
        registry.get("task_mgr").get(consumer.id).cancel("no longer wanted")

    outsiders = trace.inside("agent") | trace.inside("monitor")
    in_transition = trace.inside("transition")
    stray = [
        i
        for i, e in enumerate(trace.log)
        if e == "<try_dispatch>" and i in outsiders - in_transition
    ]
    assert not stray, "a dispatch was requested from outside a transition"


def test_an_agent_writing_a_handoff_triggers_no_dispatch_of_its_own(
    scheduler, runner, registry, monkeypatch
):
    """Downstream promotion is not pushed; it falls out of the next pass."""
    trace = Trace(monkeypatch)
    (mid,) = new_handoffs(1)
    producer = make_task(outputs=[mid])
    scheduler.submit(producer)
    scheduler.submit(make_task(inputs=[mid]))

    before = len(trace.spans("try_dispatch"))
    with trace.actor("agent"):
        runner.produce(registry, producer.id)
    assert len(trace.spans("try_dispatch")) == before


# ------------------------------------------------------------- criterion 47


def test_the_monitors_whole_action_set_is_transitions(scheduler, runner, registry, task_mgr):
    """Restart a failed task, submit a copy, reconcile a related one — each is a
    call on a task. Without `fail()` the monitor would have no verb for the
    third and would reach for the field."""
    (mid,) = new_handoffs(1)
    stuck = make_task(outputs=[mid])
    downstream = make_task(inputs=[mid])
    scheduler.submit(stuck)
    scheduler.submit(downstream)

    task_mgr.get(stuck.id).fail("stalled")
    assert task_mgr.get(stuck.id).status is TaskStatus.FAILED

    task_mgr.get(stuck.id).restart()
    assert task_mgr.get(stuck.id).status is not TaskStatus.FAILED

    copy = make_task(outputs=new_handoffs(1))
    scheduler.submit(copy)  # "submit a copy" is the ordinary public entrance

    report = task_mgr.get(downstream.id).cancel("reconciled")
    assert task_mgr.get(downstream.id).status is TaskStatus.CANCELLED
    assert report.reached[0][1] == "reconciled"


@pytest.mark.parametrize("verb", ["cancel", "restart", "fail"])
def test_a_transition_refuses_a_status_it_does_not_apply_to(scheduler, task_mgr, verb):
    task = make_task()
    scheduler.submit(task)  # dispatched: a phase state
    if verb == "fail":
        return  # `fail` is the one that *is* legal here
    with pytest.raises(TaskStateError):
        getattr(task_mgr.get(task.id), verb)()


def test_fail_releases_the_lease_and_closes_the_attempt(scheduler, registry, task_mgr):
    """It is `on_task_done`'s effect with an explicit caller, not a second path
    to the same state that could drift from it."""
    task = make_task(resources={"gpu": 4})
    scheduler.submit(task)
    task_mgr.get(task.id).fail("judged dead")

    assert registry.get("resource:gpu").available == 8
    top = task_mgr.get(task.id).history[-1]
    assert top.outcome is TaskStatus.FAILED and top.detail == "judged dead"


def test_restart_is_resume_task_expressed_as_a_transition(scheduler, runner, task_mgr):
    task = make_task()
    scheduler.submit(task)
    runner.finish(task.id, TaskStatus.FAILED)
    task_mgr.get(task.id).restart()
    assert len(task_mgr.get(task.id).history) == 2


# ------------------------------------------------------------- criterion 48


def test_models_names_no_scheduler_symbol_anywhere_in_its_ast():
    """Not a grep. `scheduler` and `try_dispatch` appear throughout prose and
    docstrings — measured on the sibling case, `"scheduler" in runner.py` is
    True today from two docstring mentions alone, while an AST walk returns 0.
    Copying the grep would produce a test that fails for the wrong reason.
    """
    from task_graph import models

    tree = ast.parse(inspect.getsource(models))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if "scheduler" in a.name]
        elif isinstance(node, ast.ImportFrom):
            offenders += [node.module] if node.module and "scheduler" in node.module else []
            offenders += [a.name for a in node.names if "scheduler" in a.name.lower()]
        elif isinstance(node, ast.Attribute) and node.attr in ("Scheduler",):
            offenders.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in ("Scheduler", "try_dispatch"):
            offenders.append(node.id)
    assert offenders == [], offenders


def test_the_module_graph_stays_acyclic():
    """`models` sits below every manager, and `scheduler` imports it."""
    from task_graph import models
    from task_graph import scheduler as scheduler_module

    def imported(module):
        tree = ast.parse(inspect.getsource(module))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
            elif isinstance(node, ast.Import):
                names |= {a.name for a in node.names}
        return {n for n in names if n.startswith("task_graph")}

    assert "task_graph.models" in imported(scheduler_module)
    assert imported(models) <= {"task_graph.ids", "task_graph.permissions"}


def test_a_transition_reaches_the_scheduler_by_name(scheduler, task_mgr, registry):
    """Resolution at use time creates no import edge — which is the whole reason
    a transition may reach the scheduler at all."""
    task = make_task(inputs=new_handoffs(1))
    scheduler.submit(task)
    calls = []

    class Recording:
        def __getattr__(self, name):
            calls.append(name)
            return getattr(scheduler, name)

    registry.register("scheduler", Recording())
    task_mgr.get(task.id).cancel()
    assert "cascade_cancel" in calls


def test_phases_is_an_ordered_tuple_and_not_a_set():
    """`PHASES[i + 1]` is "the next phase", which `enter_phase` uses to reject a
    skip. A frozenset would make the membership tests read identically and
    silently lose the sequence."""
    assert isinstance(PHASES, tuple)
    assert PHASES == (
        TaskStatus.INPUT_VALIDATING,
        TaskStatus.RUNNING,
        TaskStatus.OUTPUT_VALIDATING,
    )
