"""Shared fixtures. Every test builds its own registry; nothing is global."""

import pytest

from task_graph.bootstrap import build_registry
from task_graph.ids import HandoffId
from task_graph.models import Task
from task_graph.resource import GpuMgr, TokenMgr
from task_graph.store import MemoryStoreMgr


@pytest.fixture
def store():
    return MemoryStoreMgr()


@pytest.fixture
def registry(store):
    r = build_registry(store=store)
    r.get("agent_mgr").register("profiler")
    r.get("agent_mgr").register("tuner")
    return r


@pytest.fixture
def scheduler(registry):
    return registry.get("scheduler")


@pytest.fixture
def runner(registry):
    return registry.get("runner")


@pytest.fixture
def handoff_mgr(registry):
    return registry.get("handoff_mgr")


@pytest.fixture
def task_mgr(registry):
    return registry.get("task_mgr")


@pytest.fixture
def agent_mgr(registry):
    return registry.get("agent_mgr")


def rebuild(store, **kw):
    """A restart: fresh components over the same store."""
    r = build_registry(store=store, **kw)
    r.get("agent_mgr").register("profiler")
    r.get("agent_mgr").register("tuner")
    return r


def make_task(*, spec: str = "profiler", **kw) -> Task:
    return Task(agent_spec=spec, **kw)


def new_handoffs(n: int = 1) -> list[HandoffId]:
    return [HandoffId.new() for _ in range(n)]


def gpu(registry) -> GpuMgr:
    return registry.get("resource:gpu")


def token(registry) -> TokenMgr:
    return registry.get("resource:token")


def run_to_completion(registry, task, *, valid=True, status=None, usage=None):
    """Submit, let the fake agent write its outputs, and finish."""
    from task_graph.models import TaskStatus

    registry.get("scheduler").submit(task)
    registry.get("runner").produce(registry, task.id, valid=valid)
    registry.get("runner").finish(task.id, status or TaskStatus.SUCCEEDED, usage or {})
