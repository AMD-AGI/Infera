# Agent Task Graph — Specification

| | |
|---|---|
| Status | Draft, pending review |
| Date | 2026-08-19 |
| Scope | Task management substrate for the Infera AI-optimization agent loop |
| Source | `mission.md`; prior-art report `agent-task-graph-prior-art.html` (rev. 2, 2026-08-18) |

---

## 1. Purpose

Provide the task-management substrate for Infera's agent-driven performance
optimization loop.

An AI agent is treated as a *function that is not very procedural*. A handoff is
that function's input or output. This system decides **which task runs when**,
and nothing else. It never inspects what a task does.

### 1.1 In scope

- Task lifecycle: declaration, state, persistence.
- Dependency resolution through handoffs.
- Resource admission control for two resource classes: GPU and token.
- Scheduling decisions, with the algorithm itself replaceable.
- Recovery of the whole system from persisted state.

### 1.2 Out of scope

A task is assumed to succeed, and its input handoffs are assumed to arrive.
The following are explicitly delegated to other mechanisms; this system only
guarantees it does not obstruct them:

- Retry, backoff, and failure recovery.
- Cascading invalidation of downstream tasks.
- Re-running a finished task. `SUCCEEDED` is final. A caller dissatisfied with a
  result submits a *new* task and wires up its dependencies.
- Agent internals: knowledge, prompts, agent configuration.
- The runtime that actually executes an agent. That is `TaskRunner`, a registered
  interface; this system defines the interface and ships only a fake.
- Judging whether produced content is usable. Agents decide that; the system only
  records the verdict (§3.1).

---

## 2. Design principles

| # | Principle | Consequence |
|---|---|---|
| 1 | Composition over inheritance | Inheritance appears only in the `ResourceMgr` hierarchy. Everything else is a `Protocol` resolved from the registry. |
| 2 | Content-agnostic scheduler | A resource is a `(name, amount)` pair the task declares. The scheduler does arithmetic on counters and nothing more. |
| 3 | Single source of truth | Task state lives in `TaskMgr`. The scheduler's pools are a derived index, never a second copy. |
| 4 | The engine decides when | An agent may submit tasks. It may not redirect the graph. |
| 5 | Algorithm decoupled from mechanism | Ordering lives behind `SchedulePolicy`. The first implementation is naive FIFO. |
| 6 | Simplicity is a requirement | Where a mature solution exists, use it (§9). Where none fits, the implementation stays small enough to read in one sitting. |

---

## 3. Domain model

### 3.1 Handoff

A handoff is the unit of transfer between tasks. It is **instantiated when its
producing task is created**, not when its content is produced. A downstream task
therefore holds the handoff — and its `uuid` — from birth. The upstream task only
fills in the content later.

This is what makes dependency resolution a counter rather than a matching engine.

| Field | Type | Meaning |
|---|---|---|
| `uuid` | `str` | Identity, assigned at instantiation |
| `type` | `str` | Opaque to the scheduler; meaningful to agents |
| `produced_by` | `str \| None` | Task id of the producer; `None` for externally supplied handoffs |
| `status` | `HandoffStatus` | See below |
| `timestamp` | `float` | Creation time |
| `content` | `Any` | `None` until produced |

```
CREATED  ──→  GENERATING  ──→  VALID     (terminal)
   │                       └─→  INVALID   (terminal)
   └──────────────────────────→ INVALID   (producer failed before starting)
```

- `CREATED` — instantiated, empty.
- `GENERATING` — a producer has begun writing. Content is dirty and must not be read.
- `VALID` — content is complete and usable. **Terminal.**
- `INVALID` — produced but unusable, or the producer failed. **Terminal.** Distinct
  from "not ready yet": a consumer of an `INVALID` handoff will never become
  eligible, and that is correct.

Because `SUCCEEDED` is final (§3.2), `VALID` never reverts. A downstream task's
input precondition, once satisfied, stays satisfied. The system therefore contains
no invalidation-propagation logic at all.

#### Who decides `VALID`

**No manager judges handoff validity.** Whether produced content is usable is a
question about the content, and only the agents on either side of the handoff can
answer it. The producing agent declares its output `VALID` or `INVALID`; a
consuming agent is free to check its inputs again on its own terms.

`HandoffMgr` therefore records and serves — it never adjudicates. It writes down
the verdict an agent reports and answers queries about it. It has no validation
logic, no content schema, and no opinion.

The scheduler *reads* `VALID` to determine eligibility, but it reads it as an
opaque fact. Reading a verdict is not making one. This keeps the scheduler
content-agnostic (§2, principle 2) while still letting it resolve dependencies.

The practical consequence is in the completion signature: the runner reports a
per-handoff verdict supplied by the agent, rather than the scheduler inferring
validity from whether the task exited cleanly (§6.3).

### 3.2 Task

| Field | Type | Meaning |
|---|---|---|
| `id` | `str` | Identity |
| `agent` | `str` | Agent name, resolved through `AgentMgr` |
| `inputs` | `list[str]` | Handoff uuids required to run |
| `outputs` | `dict[str, HandoffStatus]` | Handoff uuids this task will fill, with the verdict recorded for each (§7) |
| `resources` | `dict[str, float]` | Declared, never inferred, e.g. `{"gpu": 2, "token": 100_000}` |
| `status` | `TaskStatus` | See below |
| `created_at` | `float` | FIFO ordering key |
| `expedited` | `bool` | Set by `expedite()`; a policy hint |

```
      submit
        │
        ├──────────────────┐
        ↓                  ↓
WAITING_HANDOFF ──→ WAITING_RESOURCE ──→ RUNNING ──→ SUCCEEDED  [final]
  (Ineligible)         (Eligible)          │  │
       ↑                    ↑              │  └─→ FAILED     ┐
       │                    │              ↓                 │ resumable
       │                    │          STOPPING ─→ SUSPENDED ┘
       └────────────────────┴───── resume ──────────────────┘

remove_queued:  WAITING_HANDOFF | WAITING_RESOURCE  ──→  CANCELLED  [final]
```

| State | Meaning | Exit |
|---|---|---|
| `WAITING_HANDOFF` | Not all input handoffs are `VALID` | Last input turns `VALID` → `WAITING_RESOURCE` |
| `WAITING_RESOURCE` | Inputs ready, competing for resources | Full resource set granted atomically → `RUNNING` |
| `RUNNING` | Holds a lease; the runner is executing | Runner reports done, or `stop()` → `STOPPING` |
| `STOPPING` | Stop requested; the runner has not confirmed | `on_stopped()` → `SUSPENDED` |
| `SUCCEEDED` | The task ran to completion. Its outputs carry whatever verdict the agent gave — not necessarily `VALID` (§6.3). **Final** | — |
| `FAILED` | Finished unsuccessfully. Resumable | `resume()` |
| `SUSPENDED` | Stopped on request. Resumable | `resume()` |
| `CANCELLED` | Removed while queued. **Final** | — |

`STOPPING` is a transient of `RUNNING`, present because a runner cannot terminate
a process instantaneously. It holds its resources until `on_stopped()` confirms.
Recovery treats it as `SUSPENDED` (§6.4).

`resume()` accepts `FAILED` and `SUSPENDED`. It rejects `SUCCEEDED` and
`CANCELLED`. The resumed task re-enters `WAITING_RESOURCE` if all its inputs are
`VALID`, otherwise `WAITING_HANDOFF` — the landing pool is recomputed, not
remembered.

**Why the two waits are separate collections.** `WAITING_HANDOFF` is an unordered
set: there is no decision to make, only a counter to watch. `WAITING_RESOURCE` is
ordered: it is the single place in the system where a scheduling decision occurs.
Merging them into one queue with a filter would hide that distinction.

### 3.3 Resource

Two release semantics, distinguished at the abstract-base level:

| Class | On release | Example |
|---|---|---|
| Renewable | Returned in full | GPU |
| Consumable | Only the unused reservation is returned | Token budget |

Consumables follow **reserve-then-settle**: reserve an estimate before the task
starts, settle the actual amount on completion. An estimate alone overspends;
post-hoc accounting alone bounds nothing.

---

## 4. Architecture

```
                    ┌───────────────────────────────┐
                    │           Registry            │
                    │   register(name, component)   │
                    │   get(name) -> component      │
                    └───────────────┬───────────────┘
                                    │
      everything registers here; everything resolves from here
                                    │
   ┌────────────┬─────────────┬─────┴──────┬──────────────┬────────────┐
   │            │             │            │              │            │
┌──▼──────┐ ┌───▼──────┐ ┌────▼─────┐ ┌────▼─────┐ ┌──────▼───┐ ┌──────▼─────┐
│Scheduler│ │HandoffMgr│ │ TaskMgr  │ │ResourceMgr│ │ AgentMgr │ │TaskRunner  │
│         │ │          │ │          │ │ ├ GpuMgr  │ │          │ │            │
│ pools:  │ │ records, │ │  owns    │ │ └ TokenMgr│ │ get(name)│ │ start/stop │
│ WAIT_HO │ │  never   │ │  state   │ │          │ │          │ │            │
│ WAIT_RES│ │adjudicates│ │          │ │          │ │          │ │            │
│ RUNNING │ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────────┘
│ STOPPED │       ┌────────────┐  ┌──────────────┐
└─────────┘       │  StoreMgr  │  │SchedulePolicy│
                  │ (json file)│  │ (FIFO first) │
                  └────────────┘  └──────────────┘
```

Module dependency remains strictly one-way and acyclic. The registry decouples
*instance wiring*, not *import direction*:

```
core     ← everyone      (pure data: dataclasses and enums, no behaviour)
registry ← everyone      (holds instances; imports nothing but core)
```

No module imports another module's implementation. A component that needs a
collaborator resolves it by name at use time.

### 4.1 Registry

**Every component is registered — managers, the store, the runner, the policy,
the scheduler itself.** There is no separate dependency-injection path: a
component that needs a collaborator calls `registry.get(name)`.

| Registered name | Component |
|---|---|
| `handoff_mgr` | `HandoffMgr` |
| `task_mgr` | `TaskMgr` |
| `store_mgr` | `StoreMgr` — the persistence backend |
| `agent_mgr` | `AgentMgr` |
| `runner` | a `TaskRunner` implementation |
| `policy` | a `SchedulePolicy` implementation |
| `scheduler` | `Scheduler` |
| `resource:<name>` | one `ResourceMgr` per pool, e.g. `resource:gpu`, `resource:token` |

Requirements:

- **`Registry()` is constructible.** Each instance is isolated. A test builds its
  own registry, populates it, and discards it; nothing leaks between tests. A
  process-wide default instance exists for convenience but is never mandatory.
- **Resolution happens at use time, not construction time.** A component holds
  the registry, not its collaborators. This is what allows any registration order
  and lets a test swap an implementation after wiring.
- **Missing registration fails loudly.** `get()` on an unregistered name raises
  immediately with the name in the message. A silent `None` would surface far
  from its cause.

The trade-off is accepted deliberately: the registry makes dependencies implicit
at the call site rather than visible in a constructor signature. The mitigations
are the three requirements above — isolated instances, loud failure, and the
fixed name table.

### 4.2 StoreMgr

Persistence is a registered component like any other, not a sub-object of
`TaskMgr`. `TaskMgr` resolves `store_mgr` when it writes.

```python
class StoreMgr(Protocol):
    def save(self, task: Task) -> None: ...
    def load_all(self) -> list[Task]: ...
    def delete(self, task_id: str) -> None: ...
```

The first implementation is `JsonFileStoreMgr`: one JSON file per task in a
directory. No database. Chosen for inspectability while the shape of the data is
still settling.

### 4.3 Managers

| Manager | Responsibility | Does *not* |
|---|---|---|
| `HandoffMgr` | Record the status an agent reports; answer queries | **Judge validity** (§3.1). Store large payloads inline (§8.2) |
| `TaskMgr` | Own task state; write through `store_mgr` | Make scheduling decisions |
| `ResourceMgr` | Account for one named pool | Know about tasks |
| `AgentMgr` | `get(name) -> Agent` | Anything else. Deliberately trivial |

### 4.4 Pools as an index

The scheduler's four pools are `dict[TaskStatus, set[task_id]]`, maintained in
lockstep with `TaskMgr` through a single internal transition method. They are an
index over task state, not a second copy of it. It is therefore structurally
impossible for a pool to disagree with `TaskMgr`.

---

## 5. Interfaces

All of the following are registered in the `Registry` (§4.1) and resolved by name.

```python
# ---- registry ----
class Registry:
    def register(self, name: str, component: Any) -> None: ...
    def get(self, name: str) -> Any: ...          # raises KeyError if absent

# ---- store ----  (registered as "store_mgr", see §4.2)
class StoreMgr(Protocol):
    def save(self, task: Task) -> None: ...
    def load_all(self) -> list[Task]: ...
    def delete(self, task_id: str) -> None: ...

# ---- handoff ----  (records verdicts; never forms one)
class HandoffMgr:
    def register(self, handoffs: list[Handoff]) -> None: ...
    def get(self, uuid: str) -> Handoff: ...
    def mark_generating(self, uuid: str) -> None: ...
    def record(self, uuid: str,
               status: HandoffStatus,       # VALID | INVALID, decided by the agent
               content: Any = None) -> None: ...
    def is_valid(self, uuid: str) -> bool: ...   # a read, not a judgement

# ---- resource ----
class ResourceMgr(ABC):
    name: str
    @abstractmethod
    def can_afford(self, amount: float) -> bool: ...
    @abstractmethod
    def take(self, amount: float) -> None: ...
    @abstractmethod
    def give_back(self, amount: float, actual: float | None = None) -> None: ...

class RenewableMgr(ResourceMgr):   # give_back returns the full amount
class ConsumableMgr(ResourceMgr):  # give_back returns amount - actual

# ---- runner ----
class TaskRunner(Protocol):
    def start(self, task: Task, agent: Any, on_done: Callable) -> None: ...
    def stop(self, task_id: str) -> None: ...

# ---- policy ----
class SchedulePolicy(Protocol):
    def select(self, eligible: list[Task],
               snapshot: dict[str, float]) -> list[str]: ...
```

`can_afford` and `take` are separate so the scheduler can check every declared
resource before mutating any of them (§6.2).

### 5.1 Scheduler API

| Method | Effect | Rejects |
|---|---|---|
| `submit(task)` | Register task and its output handoffs; place in the pool its inputs dictate; dispatch | Duplicate id; undeclared resource pool |
| `expedite(task)` | `submit`, but requires all inputs already `VALID` and marks the task for front-of-queue ordering | Any input not `VALID` |
| `remove_queued(tid)` | `→ CANCELLED` | Task not in a waiting pool |
| `stop(tid)` | `RUNNING → STOPPING`; calls `runner.stop(tid)` | Task not `RUNNING` |
| `resume(tid)` | `FAILED \| SUSPENDED →` recomputed waiting pool; dispatch | `SUCCEEDED`, `CANCELLED`, or any live state |
| `update_task(tid, ...)` | Sugar: remove and re-submit under the same id, with new inputs/outputs/resources | Task not queued |
| `on_task_done(tid, result)` | Release resources; record the agent's handoff verdicts; promote dependents; dispatch | Task not `RUNNING` |
| `on_stopped(tid)` | `STOPPING → SUSPENDED`; release resources | Task not `STOPPING` |
| `resume_system()` | Rebuild all state from the store | — |
| `try_dispatch()` | Grant resources and start tasks, as capacity allows | — |

`stop()` records intent and delegates; `on_stopped()` completes the transition
(§3.2). A task in `STOPPING` still holds its resources.

---

## 6. Data flow

### 6.1 Submit

```
submit(task)
  ├─ TaskMgr.add(task)                      → persisted
  ├─ HandoffMgr.register(task.outputs)      → each CREATED
  ├─ all inputs VALID ?  WAITING_RESOURCE : WAITING_HANDOFF
  └─ try_dispatch()
```

### 6.2 Dispatch — atomic, all-or-nothing

Called at every decision point: after `submit`, `on_task_done`, `resume`,
`expedite`.

```
eligible = pools[WAITING_RESOURCE]
for tid in policy.select(eligible, snapshot):
      task = TaskMgr.get(tid)
      if not all(mgr(r).can_afford(n) for r, n in task.resources.items()):
            continue                        # grant nothing; leave in the pool
      for r, n in task.resources.items():
            mgr(r).take(n)
      TaskMgr.set_status(tid, RUNNING)
      runner.start(task, agent_mgr.get(task.agent), on_done=self.on_task_done)
```

**The whole declared set is verified before anything is mutated.** A task that
does not fit takes nothing and stays queued. Never acquire incrementally; never
let a queued task hold anything. This makes hold-and-wait deadlock structurally
impossible, and costs one extra loop.

### 6.3 Completion — the principal state-change entry point

The runner reports a `TaskResult`. Its handoff verdicts come from the producing
agent (§3.1); the scheduler does not derive them.

```python
@dataclass
class TaskResult:
    ok: bool                                  # did the task itself complete
    verdicts: dict[str, HandoffStatus]        # uuid -> VALID | INVALID, from the agent
    contents: dict[str, Any]                  # uuid -> content, or a reference
    actual_usage: dict[str, float]            # for consumable settlement
```

```
on_task_done(tid, result)
  ├─ release lease   (renewable: full; consumable: settle result.actual_usage)
  ├─ for each output handoff uuid:
  │       HandoffMgr.record(uuid, result.verdicts[uuid], result.contents.get(uuid))
  │       # the manager writes down the agent's verdict; it does not form one.
  │       # an output with no verdict is recorded INVALID — silence is not success
  ├─ for each handoff now VALID:
  │       for each waiting consumer:  pending -= 1
  │            pending == 0  →  WAITING_HANDOFF → WAITING_RESOURCE
  ├─ TaskMgr.set_status(tid, SUCCEEDED if result.ok else FAILED)   → persisted
  └─ try_dispatch()
```

Note that `result.ok` and the handoff verdicts are independent. A task can
complete cleanly (`ok=True`) and still declare its output `INVALID` — the agent
ran fine and concluded the result is unusable. The task is `SUCCEEDED`; its
consumers never become eligible. The scheduler treats both facts as reported,
not inferred.

A failed task releases its resources. Its outputs carry whatever verdict the
agent supplied, defaulting to `INVALID`. Downstream tasks stay in
`WAITING_HANDOFF` indefinitely. That is deliberate: the system does not cancel
them, and does not pretend to.

### 6.4 Recovery

```
resume_system()
  ├─ tasks = store_mgr.load_all()
  ├─ rebuild handoff records from each task's stored output verdicts (§7)
  ├─ recompute every task's pending-input counter
  ├─ RUNNING → WAITING_RESOURCE       (leases do not survive a restart)
  ├─ STOPPING → SUSPENDED
  ├─ reset every resource pool to full capacity
  └─ try_dispatch()
```

---

## 7. Persistence

Only **task state** is persisted. It contains everything needed to rebuild the
scheduler's pools and the handoff dependency counters, so persisting the
scheduler or the handoff manager separately would store the same facts twice.

- `StoreMgr` is a `Protocol`, registered under `store_mgr` (§4.2). `TaskMgr`
  resolves it from the registry when it writes; it is not a sub-object.
- The first implementation is `JsonFileStoreMgr`: one JSON file per task, in a
  directory. No database. Chosen for inspectability while the shape of the data
  is still settling.
- Every state-mutating operation on `TaskMgr` writes through. Completion is the
  principal trigger but not the only one — submission, cancellation, and stop all
  mutate state and must survive a crash.

The persisted task record must therefore include the **verdict recorded for each
of its output handoffs**, not just the uuid list. Without it, recovery cannot tell
a `VALID` output from an `INVALID` one and would recompute eligibility wrongly.
Handoff status is then rebuilt from task records rather than stored separately.

Handoff *content* is a different question, deliberately left open (§8.2). The
persisted record holds a reference, not a payload.

---

## 8. Deliberate omissions

### 8.1 Not built

| Omitted | Why |
|---|---|
| Resource kinds beyond a name | The scheduler cannot tell a GPU from an API quota. That is the point. |
| Duration model | Agent durations are unknown, heavy-tailed, and not reproducible. |
| Graph library | Two operations are needed: decrement a counter, and detect a cycle on insert. |
| Cascading failure | Out of scope by §1.2. |
| Event bus | Deferred. A hook callback is left at each transition point, and the registry (§4.1) gives a later bus a natural place to live. |
| Lease TTL sweep | Deferred; noted in §10. |

### 8.2 Handoff content storage

`HandoffMgr` keeps handoff *metadata*. Where large payloads live is a larger
question, deliberately left open — the interface accommodates a content reference
rather than mandating inline storage. Storing large agent payloads in the state
record is a known way to make the state store the bottleneck.

---

## 9. Build-versus-adopt

Per `mission.md` rule 3, mature solutions are preferred. The prior-art survey
(rev. 2) evaluated the field and concluded: build it.

| Candidate | Verdict |
|---|---|
| `graphlib.TopologicalSorter` | **Rejected.** `add()` raises after `prepare()`. This graph grows at runtime, so the sorter would have to be rebuilt on every submission, losing completion state. CPython issue 91301 tracks the limitation. |
| `networkx` | **Rejected.** No graph algorithms are required. Importing it invites modelling the system as a graph object that must then be kept in sync with the state machine doing the real work. |
| Prefect global concurrency limits | **Rejected.** The closest existing match to the resource stage, including atomic multi-pool acquisition. But limits live server-side: adopting a server to obtain one primitive. |
| Hatchet concurrency keys | **Rejected.** A platform, not a library. DAGs must be declared; this graph is dynamic. |
| Ray, Temporal/Restate/Inngest/DBOS, Airflow/Dagster, Slurm/K8s | **Rejected.** Wrong layer or wrong problem; see prior-art §08. |

Adopted from prior art, as design rather than dependency:

- **RCPSP** (Hartmann & Briskorn, EJOR 2021) — the formal name for this model.
  The two waiting pools are its *ineligible* and *eligible* activity sets.
- **Parallel schedule generation scheme** — at each decision point, consider the
  eligible set in priority order and start everything that fits.
- **A2A task-state vocabulary** — state names adopted rather than invented.
- **Reserve-then-settle** (HiveMind, arXiv 2604.17111) — for consumable pools.
- **The engine owns routing** (GraphBit, arXiv 2605.13848) — agents submit; the
  scheduler decides.

The rationale is recorded in `agent_sys/README.md` as required by mission rule 3.

---

## 10. Open questions

| Item | Status |
|---|---|
| Lease TTL and sweep | A runner that dies while `RUNNING` never reports done, and its resources leak until `resume_system()`. Deferred, not solved. A TTL plus a periodic sweep is the known fix. |
| Handoff payload storage | §8.2. The interface is left open. |
| Fairness across submitters | Naive FIFO lets one submitter monopolise a pool. A future `SchedulePolicy` keyed by submitter is the reference solution. |
| Better ordering than FIFO | The composite rule from the prior art (priority tier → estimated cost → most-total-successors → FIFO) is a drop-in `SchedulePolicy`. Not built first. |
| Cycle detection | A task whose inputs transitively depend on its own outputs will never run. A ten-line DFS at submit would reject it at the boundary. Not in the first version. |

---

## 11. Acceptance criteria

The implementation is complete when each of the following is demonstrated by a
test:

1. A task whose inputs are all `VALID` at submit lands in `WAITING_RESOURCE`.
2. A task with an unfilled input lands in `WAITING_HANDOFF` and is promoted
   exactly when its last input becomes `VALID`.
3. A task requiring more of any pool than is available starts nothing and
   **consumes nothing** — verified by asserting every pool is unchanged.
4. A renewable pool returns to its prior level after completion; a consumable
   pool is debited by the settled actual, not the reservation.
5. `stop` → `on_stopped` → `resume` returns a task to the correct pool, recomputed
   from its inputs.
6. `resume` on a `SUCCEEDED` or `CANCELLED` task is rejected.
7. A failed task releases its resources, records `INVALID` for outputs the agent
   gave no verdict for, and leaves its dependents in `WAITING_HANDOFF`.
8. `resume_system()` reconstructs pools, handoff verdicts, and counters from the
   store alone, with `RUNNING` and `STOPPING` tasks demoted to
   `WAITING_RESOURCE` and `SUSPENDED` respectively.
9. `expedite` places a task ahead of earlier-submitted eligible tasks, and is
   rejected when any input is not `VALID`.
10. Swapping `SchedulePolicy` changes dispatch order and nothing else.
11. `update_task` on a queued task replaces its inputs/outputs/resources under the
    same id and recomputes its pool; on a `RUNNING` task it is rejected.
12. A pool never disagrees with `TaskMgr`: after any sequence of operations, the
    union of the pools equals the set of non-final tasks, and each task appears in
    exactly the pool matching its stored status.
13. A task reporting `ok=True` with an `INVALID` output becomes `SUCCEEDED` while
    its consumers stay in `WAITING_HANDOFF` — completion and validity are
    independent (§6.3).
14. No manager derives a handoff verdict: `HandoffMgr` records exactly what it is
    given, and given the same verdict for identical content it stores the same
    result regardless of `ok`.
15. Two isolated `Registry()` instances do not share components, and `get()` on an
    unregistered name raises with that name in the message.
