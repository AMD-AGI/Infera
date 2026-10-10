# `task_graph` — which task runs when, and nothing else

The task-management substrate. An AI agent is treated as a function that is not
very procedural; a handoff is that function's input or output. **This package
decides which task runs when and never inspects what a task does.**

| | |
|---|---|
| Specification | [`docs/spec.md`](docs/spec.md) — 54 acceptance criteria |
| Design | [`docs/design.md`](docs/design.md) |
| Seam | [`../docs/interfaces.md`](../docs/interfaces.md) §4.7 |
| Tests | `../tests/task_graph/` |

## Who uses it

Everything above it. `agent` runs a task and reports back; `validator` runs a
phase inside one; `monitor` acts on a task through its own transitions;
`env_mgr` reads a task's grants; `cli` builds the root task and starts the
scheduler. **This package imports `spec_loader` and nothing else of ours.**

Content-agnosticism is the load-bearing property: the scheduler never reads a
spec and never learns what a handoff means, which is why a package can declare
anything without this code changing.

## The interface

```python
from task_graph import (
    Task, Execution, Handoff, HandoffRef, Agent,   # the objects
    TaskId, AgentId, HandoffId, Id,
    TaskStatus, HandoffStatus, PHASES, WAITING, RESUMABLE,
    Scheduler, SchedulePolicy, FifoPolicy, DepthFirstPolicy,
    TaskRunner, FakeRunner,                        # what the scheduler dispatches to
    TaskMgr, HandoffMgr, AgentMgr, ResourceMgr,    # the managers
    GpuMgr, TokenMgr, RenewableMgr, ConsumableMgr,
    StoreMgr, MemoryStoreMgr, JsonFileStoreMgr,    # persistence
    Permissions, Grant, Access,                    # what a task may reach
    Registry, build_registry, check_graph,         # assembly, and the load check
    Resumable, resume_all, RESUME_ORDER, CascadeReport, OrderedIdSet,
)
```

**Ask the object, do not read its fields.** `Handoff.open_next()` hands back a
version to write and never tells a caller which case it was in;
`check_if_latest_valid` answers a question rather than publishing a status to
compare against. That is `engineer_principle.md` §3, and this package is where
the examples come from.

## What is inside

| | |
|---|---|
| `models.py` | `Task`, `Execution`, `Agent` and the status enums |
| `task.py`, `agent.py`, `handoff.py` | the three managers over them |
| `graph.py` | the graph itself, and `check_graph` |
| `scheduler.py` | eligibility, dispatch, leases, and the cascade |
| `policy.py` | ordering: FIFO, and depth-first |
| `resource.py` | renewable and consumable pools — GPUs, tokens |
| `permissions.py` | grants, typed by handoff kind name |
| `store.py` | the store managers, in-memory and on disk |
| `runner.py` | the `TaskRunner` seam, and a fake for tests |
| `bootstrap.py` | `build_registry` — loading packages into one registry set |
| `ids.py`, `ordered.py` | identity, and an insertion-ordered id set |

## Two rules that protect it

**The scheduler never names a spec registry.** It schedules; interpreting a spec
is somebody else's job, and `tests/closure/test_authority.py` enforces it.

**A task's status has one writer.** Every transition goes through the task's own
`_move`; a monitor acts by *calling* a transition, never by assigning a status.
Both rules exist so that the graph stays reasonable when several things are
happening at once.
