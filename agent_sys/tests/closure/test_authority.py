"""Criterion 8 — the scheduler never reads a closure.

Verified the way `task_graph`'s criterion 14 is: a spy over the closure registry
records every read across a full submit → dispatch → complete cycle, and a
meta-test proves the spy would have caught the thing it reports absent. A test
that passes because nothing read anything is a test that proves nothing, so the
meta-test is not an extra — it is what makes the negative result mean something.

**The assertion cannot be "no read at all", and that is the whole difficulty.**
The prohibition is on the *scheduler*. A `Task` transition may resolve `closures`
— `unfold` does it on every non-leaf's `enter_phase(RUNNING)`, and `replace_with`
does it too — and that adds no `Scheduler → spec registry` edge. So the spy
attributes each read to its **calling frame** and asserts that no entry names
`task_graph.scheduler`.

Frame inspection is fragile: a decorator, a `functools.wraps`, or a C-level call
would change what it sees. It is chosen anyway, because the alternative is a
`caller=` argument threaded through `SpecRegistry.get`, which changes a base
class shared by five registries to satisfy one test.

A test may import `task_graph`; `docs/interfaces.md` §4's import rule is about
what leaves a package, and `tests/` is not under it.
"""

from __future__ import annotations

import sys

import pytest

from closure.registry import ClosureRegistry
from task_graph import scheduler as scheduler_module
from task_graph.bootstrap import build_registry
from task_graph.models import Task, TaskStatus
from task_graph.runner import FakeRunner
from task_graph.scheduler import Scheduler
from task_graph.store import MemoryStoreMgr

from .conftest import make_closure


def _caller_module() -> str:
    """The module that called the method one frame up.

    `sys._getframe` rather than `inspect.stack()`: the latter builds a full
    `FrameInfo` — source lines and all — for every frame on the stack, and one
    frame's `__name__` is the whole of what this needs.
    """
    return sys._getframe(2).f_globals.get("__name__", "?")


class SpyClosureRegistry(ClosureRegistry):
    """The real registry, with attribution."""

    def __init__(self) -> None:
        super().__init__()
        self.log: list[str] = []

    def get(self, name: str):
        self.log.append(_caller_module())
        return super().get(name)


def _world() -> tuple:
    registry = build_registry(store=MemoryStoreMgr(), runner=FakeRunner())
    spy = SpyClosureRegistry()
    spy.add("collect_trace", make_closure(), origin="collect_trace.jsonnet")
    registry.register("closures", spy)
    registry.get("agent_mgr").register("profiler")
    return registry, spy


def _full_cycle(registry) -> None:
    """Submit → dispatch → complete, with a re-run in the middle.

    The scheduler touches every one of its own paths here: eligibility, resource
    admission, dispatch, failure, resume, and completion.
    """
    scheduler, runner = registry.get("scheduler"), registry.get("runner")
    handoff_mgr = registry.get("handoff_mgr")

    from task_graph.ids import HandoffId, TaskId

    mid = HandoffId.new()
    producer = Task(id=TaskId.new(), agent_spec="profiler", inputs=[], outputs=[mid])
    consumer = Task(id=TaskId.new(), agent_spec="profiler", inputs=[mid], outputs=[])
    scheduler.submit(producer)
    scheduler.submit(consumer)

    runner.produce(registry, producer.id)
    runner.finish(producer.id, TaskStatus.FAILED)
    scheduler.resume_task(producer.id)
    runner.produce(registry, producer.id)
    runner.finish(producer.id)
    runner.produce(registry, consumer.id)
    runner.finish(consumer.id)

    assert handoff_mgr.check_if_latest_valid(mid)


def test_scheduler_never_reads_closure() -> None:
    """No read of the closure registry is attributed to a scheduler frame."""
    registry, spy = _world()
    _full_cycle(registry)

    assert "task_graph.scheduler" not in spy.log, (
        f"the scheduler read a closure; the log is {spy.log}"
    )


def test_the_spy_would_catch_a_scheduler_read() -> None:
    """The meta-test. Plant a read inside a scheduler method and assert the spy
    names the scheduler as its caller.

    Without this, `test_scheduler_never_reads_closure` would pass equally well
    against a spy that logs nothing.
    """
    registry, spy = _world()

    # Defined in a copy of the scheduler module's own globals, so the frame the
    # spy inspects says `task_graph.scheduler` for the same reason a real
    # scheduler method's would. Setting `__module__` on a function defined here
    # would not do it: the spy reads `f_globals`, not the attribute, and that is
    # itself worth knowing about the mechanism.
    leaked: dict = {**vars(scheduler_module), "_original": Scheduler.submit}
    exec(  # noqa: S102
        "def leaking_submit(self, task):\n"
        "    self._r.get('closures').get('collect_trace')\n"
        "    return _original(self, task)\n",
        leaked,
    )

    monkey = pytest.MonkeyPatch()
    monkey.setattr(Scheduler, "submit", leaked["leaking_submit"])
    try:
        _full_cycle(registry)
    finally:
        monkey.undo()

    assert "task_graph.scheduler" in spy.log, (
        "the spy did not attribute a read made from a scheduler frame; the "
        "negative result in the sibling test would therefore mean nothing"
    )


def test_the_spy_distinguishes_a_transition_from_the_scheduler() -> None:
    """A read from `task_graph.models` is expected and permitted.

    `Task.unfold` reads a closure on every non-leaf's `enter_phase(RUNNING)`, and
    `replace_with` reads one too. The prohibition is on the scheduler, so the spy
    has to tell the two frames apart rather than count reads. This plants a read
    in a models frame and asserts it is attributed there and not to the
    scheduler — the same mechanism the real `unfold` will exercise once
    `task_graph` rev. 12 lands it.
    """
    registry, spy = _world()

    # A frame whose `__name__` is `task_graph.models`, which is what the spy
    # reads. Built with `exec` because that is the shortest honest way to get
    # one without waiting on `Task.unfold`, which is rev. 12 material and does
    # not exist yet.
    transition = {"__name__": "task_graph.models", "closures": registry.get("closures")}
    exec("def read():\n    closures.get('collect_trace')\n", transition)  # noqa: S102
    transition["read"]()

    assert spy.log == ["task_graph.models"]
    assert "task_graph.scheduler" not in spy.log
