# `agent` — what wraps a task spec for execution

Takes a dispatched task and runs it: deploys the environment, executes the body,
runs the validation phases, seals the outputs, and reports every boundary to a
monitor.

| | |
|---|---|
| Specification | [`docs/spec.md`](docs/spec.md) — 16 acceptance criteria |
| Design | [`docs/design.md`](docs/design.md) |
| Seam | [`../docs/interfaces.md`](../docs/interfaces.md) §4.4 |
| Contract | [`protocols.py`](protocols.py) + `protocols.pyi` |
| Tests | `../tests/agent/` |

## Who uses it

`task_graph`'s scheduler dispatches into it through the `TaskRunner` seam.
`cli` constructs it. It reaches `env_mgr` for a zone, `validator` for a phase,
and `monitor` for every boundary — and **nothing imports it back**; the monitor
declares `Pushable` locally rather than depend on this package.

## Two interface levels, kept apart

| | |
|---|---|
| `Executor` | **level 1** — what a task runner talks to. Every executor satisfies it, AI or not |
| `AgentBackend` | **level 2** — the AI-harness abstraction. Only an AI executor has one |

They are two protocols rather than one with holes in it, because a program node
is a first-class executor and not a degenerate AI.

## The interface

```python
from agent import (
    Runner, TaskAttempt, Executor, ExecutorBase,   # running a task
    AgentSpec, AgentSpecRegistry, AgentStatus, Kind,
    AgentBackend, BackendDecl, select_backend, Selection,
    Assignment, AgentResult, AgentHistory,
    ValidatorExecutor, run_gate,                   # the completeness gate
    KnowledgeRef, KnowledgeReport, KNOWLEDGE_TYPES,
    RESUMED, WOKEN, TERMINAL,
    BackendUnavailable, BackendUnsupported, ConfinementNotApplied,
    GateFailure, KnowledgeWarning, MonitorUnresolved, Rejection,
    ThreadAlreadyHeld, ValidatorExecutorUnconfigured,
)
```

**`backends/claude_sdk.py` is not imported here, and must never be.** The SDK is
a 376 MB extra costing about 1.3 s to import; a missing extra must be a
`BackendUnsupported` naming it, not an `ImportError` at start-up.

## What is inside

| | |
|---|---|
| `protocols.py` / `.pyi` | the two levels above, declarations only |
| `runner.py` | the runner and `TaskAttempt` — one attempt object per dispatch, owning the thread |
| `backend.py` | the level-2 abstraction, and what a push means |
| `backends/claude_sdk.py` | the Claude Code SDK backend. Imported lazily, never at module load |
| `backends/program.py` | a task whose body is a program, not a model |
| `gate.py` | the completeness gate: an agent that delivered nothing is pushed, not failed |
| `selection.py` | which backend runs this task |
| `spec.py`, `registry.py` | the agent spec, and where agent specs live |
| `validator_executor.py` | the executor a validation phase borrows |

## Three properties worth knowing before changing anything

**The loop is the agent's; the thread is not.** An attempt owns the thread and
lends it; an agent has a `mainloop` and borrows one. That answers in one place
who spawns a phase thread and what survives a non-leaf's subgraph.

**Each phase gets a fresh executor.** That is how a phase becomes separately
attributable — one client with several session ids does not work, because
`interrupt()` takes no session id and acts on the whole connection.

**A `claude` child does not exit when its task completes.** Measured: they
accumulate one per task for the life of a run. Harmless on a small package, not
on a large one — [`../docs/TODO.md`](../docs/TODO.md) item 9, and whose it is
remains open.
