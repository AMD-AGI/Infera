"""The composition root.

The only module that imports every manager. Registration order is free —
components resolve at use time, not construction time — so this reads top-down
for a human rather than being constrained by dependencies.
"""

from collections.abc import Sequence

from task_graph.agent import AgentMgr
from task_graph.handoff import HandoffMgr
from task_graph.policy import FifoPolicy, SchedulePolicy
from task_graph.registry import Registry
from task_graph.resource import GpuMgr, ResourceMgr, TokenMgr
from task_graph.runner import FakeRunner, TaskRunner
from task_graph.scheduler import Scheduler
from task_graph.store import MemoryStoreMgr, StoreMgr
from task_graph.task import TaskMgr

__all__ = ["build_registry"]


def build_registry(
    store: StoreMgr | None = None,
    runner: TaskRunner | None = None,
    policy: SchedulePolicy | None = None,
    resources: Sequence[ResourceMgr] | None = None,
) -> Registry:
    """Wire a system. Every default is overridable, which is how a test
    substitutes a fake runner, a spy manager, or a different policy."""
    r = Registry()
    r.register("store_mgr", store or MemoryStoreMgr())
    r.register("handoff_mgr", HandoffMgr(r))
    r.register("task_mgr", TaskMgr(r))
    r.register("agent_mgr", AgentMgr(r))
    r.register("runner", runner or FakeRunner())
    r.register("policy", policy or FifoPolicy())
    for pool in resources if resources is not None else _default_resources(r):
        r.register(f"resource:{pool.name}", pool)
    r.register("scheduler", Scheduler(r))
    return r


def _default_resources(r: Registry) -> list[ResourceMgr]:
    return [GpuMgr(r, capacity=8), TokenMgr(r, capacity=1_000_000)]
