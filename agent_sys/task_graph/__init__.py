"""Task-management substrate for Infera's agent-driven optimization loop.

What leaves this package is `docs/interfaces.md` §4.7's list, and nothing else.
Importing this module reaches pydantic and this package only; the composition
root defers every sibling import to call time.
"""

from task_graph.agent import AgentMgr
from task_graph.bootstrap import build_registry
from task_graph.graph import check_graph
from task_graph.handoff import HandoffMgr
from task_graph.ids import AgentId, HandoffId, Id, TaskId
from task_graph.models import (
    PHASES,
    RESUMABLE,
    WAITING,
    Agent,
    CascadeReport,
    Execution,
    Handoff,
    HandoffRef,
    HandoffStatus,
    Task,
    TaskStatus,
)
from task_graph.ordered import OrderedIdSet
from task_graph.permissions import Access, Grant, Permissions
from task_graph.policy import DepthFirstPolicy, FifoPolicy, SchedulePolicy
from task_graph.registry import RESUME_ORDER, Registry, Resumable, resume_all
from task_graph.resource import ConsumableMgr, GpuMgr, RenewableMgr, ResourceMgr, TokenMgr
from task_graph.runner import FakeRunner, TaskRunner
from task_graph.scheduler import Scheduler
from task_graph.store import JsonFileStoreMgr, MemoryStoreMgr, StoreMgr
from task_graph.task import TaskMgr

__all__ = [
    # ids
    "Id",
    "TaskId",
    "AgentId",
    "HandoffId",
    # models
    "Task",
    "Execution",
    "Handoff",
    "HandoffRef",
    "Agent",
    "TaskStatus",
    "HandoffStatus",
    "CascadeReport",
    "PHASES",
    "WAITING",
    "RESUMABLE",
    # permissions
    "Permissions",
    "Grant",
    "Access",
    # wiring
    "Registry",
    "Resumable",
    "RESUME_ORDER",
    "resume_all",
    "StoreMgr",
    "MemoryStoreMgr",
    "JsonFileStoreMgr",
    # managers
    "HandoffMgr",
    "TaskMgr",
    "AgentMgr",
    "ResourceMgr",
    "RenewableMgr",
    "ConsumableMgr",
    "GpuMgr",
    "TokenMgr",
    # runner, policy, scheduler
    "TaskRunner",
    "FakeRunner",
    "SchedulePolicy",
    "FifoPolicy",
    "DepthFirstPolicy",
    "OrderedIdSet",
    "Scheduler",
    # composition and load-time checks
    "build_registry",
    "check_graph",
]
