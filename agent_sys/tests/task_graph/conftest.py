"""Shared fixtures. Every test builds its own registry; nothing is global."""

import pytest

from task_graph.bootstrap import build_registry
from task_graph.ids import HandoffId
from task_graph.models import PHASES, Task, TaskStatus
from task_graph.resource import GpuMgr, TokenMgr
from task_graph.store import MemoryStoreMgr

# Where the scheduler lands a task it dispatched. A leaf then advances
# INPUT_VALIDATING -> RUNNING -> OUTPUT_VALIDATING under its runner, holding one
# lease across all three. Before spec rev. 9 there were no phase states and
# dispatch landed in RUNNING; these tests say "dispatched", which is what they
# always meant.
DISPATCHED = TaskStatus.INPUT_VALIDATING
LIVE = frozenset({*PHASES, TaskStatus.STOPPING})


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


class FakeClosures:
    """A `SpecRegistry` over a dict — which is what the Protocol is for.

    The real one is `closure.ClosureRegistry`. Importing a sibling's
    implementation into this package's tests would be the edge the whole
    interface contract exists to prevent, so the tests satisfy the shape.
    """

    kind = "closure"

    def __init__(self, docs: dict) -> None:
        self._docs = dict(docs)

    def add(self, name: str, spec, *, origin: str = "") -> None:
        self._docs[name] = spec

    def get(self, name: str):
        return self._docs[name]

    def names(self) -> list[str]:
        return sorted(self._docs)

    def __contains__(self, name: str) -> bool:
        return name in self._docs


def closure_doc(
    name: str,
    *,
    agent: str = "profiler",
    inputs=(),
    outputs=(),
    resources=None,
    subgraph=None,
    monitor=None,
    permissions=None,
) -> dict:
    """One closure document, in the shape `task_graph.models.subgraph_entries` reads.

    The subgraph key's name and shape are this design's convention: no spec in
    the set gives it one. See `SubgraphEntry`.
    """
    task: dict = {"inputs": list(inputs), "outputs": list(outputs)}
    if resources:
        task["resources"] = dict(resources)
    if subgraph:
        task["subgraph"] = list(subgraph)
    if monitor:
        task["monitor"] = monitor
    if permissions:
        task["permissions"] = permissions
    return {"name": name, "agent": agent, "task": task}


def task_specs(docs: dict) -> dict:
    """The catalogue as `check_graph` sees it — task specs, keyed by closure name.

    `TaskSpecRegistry` shares a key space with `ClosureRegistry` (a task spec is
    nested inside its closure and carries no name of its own), and its values are
    the *inner* object, not the wrapping document.
    """
    return {name: doc["task"] for name, doc in docs.items()}


def with_closures(registry, docs: dict):
    """Register a closure catalogue and every agent spec it names."""
    registry.register("closures", FakeClosures(docs))
    for doc in docs.values():
        # A non-leaf may carry no `agent` at all (main spec §4.8 at rev. 10);
        # `build_registry` has already registered the name the system supplies
        # for one, so there is nothing to register here.
        if doc.get("agent"):
            registry.get("agent_mgr").register(doc["agent"])
    return registry


def advance_to_end(registry, task_id) -> None:
    """Drive a leaf from INPUT_VALIDATING to OUTPUT_VALIDATING, as a runner does."""
    runner = registry.get("runner")
    runner.advance(registry, task_id)
    runner.advance(registry, task_id)


def run_to_completion(registry, task, *, valid=True, status=None, usage=None):
    """Submit, let the fake agent write its outputs, and finish."""
    from task_graph.models import TaskStatus

    registry.get("scheduler").submit(task)
    registry.get("runner").produce(registry, task.id, valid=valid)
    registry.get("runner").finish(task.id, status or TaskStatus.SUCCEEDED, usage or {})
