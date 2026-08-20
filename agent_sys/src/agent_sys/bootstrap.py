"""The composition root.

The only module that imports every manager. Registration order is free —
components resolve at use time, not construction time — so this reads top-down
for a human rather than being constrained by dependencies.
"""

from collections.abc import Sequence

from agent_sys.agent import AgentMgr
from agent_sys.handoff import HandoffMgr
from agent_sys.policy import FifoPolicy, SchedulePolicy
from agent_sys.registry import Registry
from agent_sys.resource import GpuMgr, ResourceMgr, TokenMgr
from agent_sys.runner import FakeRunner, TaskRunner
from agent_sys.scheduler import Scheduler
from agent_sys.store import MemoryStoreMgr, StoreMgr
from agent_sys.task import TaskMgr

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
