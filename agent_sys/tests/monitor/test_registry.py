"""Resolving a monitor by name — criteria 1 and 2."""

from __future__ import annotations

import pytest

from monitor import DEFAULT_MONITOR_NAME, PusherMonitor, monitor_for
from task_graph.registry import Registry

from .conftest import StubTask, StubTaskMgr


def test_monitor_spec_resolves_by_name(
    registry: Registry, monitor: PusherMonitor, task_mgr: StubTaskMgr
) -> None:
    """Criterion 1: a task with no `monitor_spec` is watched by the default
    monitor; a task naming one is watched by that one, resolved by name from the
    component registry.

    **A name and not an object**, for the reason every other collaborator here is
    a name: an object on the model is a handle the task holds across restarts,
    and `model_validate` returns `None` for it.
    """
    special = PusherMonitor("careful", registry)
    registry.register("monitor:careful", special)

    plain = task_mgr.add(StubTask(monitor_spec=None))
    named = task_mgr.add(StubTask(monitor_spec="careful"))

    assert monitor_for(plain, registry) is monitor
    assert monitor_for(named, registry) is special
    assert monitor.name == DEFAULT_MONITOR_NAME


def test_unknown_monitor_spec_names_the_value(
    registry: Registry, monitor: PusherMonitor, task_mgr: StubTaskMgr
) -> None:
    """Criterion 2: an unregistered name is **rejected with the offending value
    named**, at the same point every other by-name collaborator is resolved.

    Rev. 4 of `interfaces.md` raised the stakes: a `monitor:<name>` that will not
    resolve is no longer merely an unwatched task — since the planned channel
    runs through this module, **it is a task that never advances a phase**. So
    the failure has to be loud and has to say which name.
    """
    task = task_mgr.add(StubTask(monitor_spec="typo-in-the-jsonnet"))

    with pytest.raises(KeyError) as raised:
        monitor_for(task, registry)

    message = str(raised.value)
    assert "typo-in-the-jsonnet" in message
    assert "monitor:typo-in-the-jsonnet" in message
    assert str(task.id) in message


def test_the_default_is_absent_rather_than_wrong(registry: Registry) -> None:
    """A task with no `monitor_spec` and no default registered fails the same
    way, naming the default — not silently unwatched."""
    task = StubTask(monitor_spec=None)
    with pytest.raises(KeyError) as raised:
        monitor_for(task, registry)
    assert DEFAULT_MONITOR_NAME in str(raised.value)
