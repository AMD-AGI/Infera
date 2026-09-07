"""A monitor is not a task — criterion 11."""

from __future__ import annotations

import inspect

from monitor import Monitor, PusherMonitor
from task_graph.models import Task

from .conftest import StubTask, StubTaskMgr


def test_monitor_holds_no_lease_no_zone_and_is_not_in_the_graph(
    monitor: PusherMonitor, task_mgr: StubTaskMgr
) -> None:
    """Criterion 11. Decided 2026-08-27: **the gap between a monitor and the task
    model is too wide to be worth closing.** A task is a function
    `<handoffs, agent>` with inputs, outputs and validators, and a monitor has
    none of those.

    The consequence is stated rather than hidden: nothing monitors the monitor,
    which is what `test_liveness.py` is the alpha's answer to.
    """
    for absent in ("zone", "lease", "agent_spec", "inputs", "outputs", "depends_on", "resources"):
        assert not hasattr(monitor, absent), f"a monitor acquired {absent}, which is a task's"

    # It is identified by a name, not by a `TaskId` — which is also why the
    # composition root can key on `monitor:<name>`.
    assert monitor.name
    assert not hasattr(monitor, "id")

    task_mgr.add(StubTask())
    monitor.set_task(next(iter(task_mgr.tasks)))

    assert monitor not in task_mgr.tasks.values(), "a monitor appeared in the graph"
    assert not isinstance(monitor, Task)


def test_the_protocol_declares_nothing_a_task_would_need() -> None:
    """The interface is the check that outlives the implementation: a later
    monitor that grew a zone would have to widen `Monitor` to say so."""
    members = set(getattr(Monitor, "__annotations__", {})) | {
        name for name, _ in inspect.getmembers(Monitor, inspect.isfunction)
    }
    assert not members & {"zone", "lease", "agent_spec", "prepare", "acquire", "release"}
    assert {"set_task", "report", "mainloop", "stop"} <= members
