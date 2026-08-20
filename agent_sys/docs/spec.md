# Agent Task Graph — Specification

| | |
|---|---|
| Status | Draft, pending review |
| Revision | 3 — 2026-08-20. Handoffs are versioned; agents own their state |
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
- Re-running a `SUCCEEDED` task. That state is final; a caller dissatisfied with
  a result submits a *new* task and wires up its dependencies. (`FAILED` and
  `SUSPENDED` tasks *are* resumable — see §3.2.)
- Agent internals: knowledge, prompts, agent configuration.
- **An agent's own process, history, and recovery.** An agent is responsible for
  its own durability, including re-establishing what it was doing after a restart.
  This system records that a task is bound to an agent uuid (§3.2); it does not
  manage the agent's lifecycle.
- The runtime that actually executes an agent. That is `TaskRunner`, a registered
  interface; this system defines the interface and ships only a fake.
- Judging whether produced content is usable, and **advancing handoff state at
  all**. Agents own both; the scheduler only asks whether a handoff's latest
  version is valid (§3.1).

---

## 2. Design principles

| # | Principle | Consequence |
|---|---|---|
| 1 | Composition over inheritance | Inheritance appears only in the `ResourceMgr` hierarchy. Everything else is a `Protocol` resolved from the registry. |
| 2 | Content-agnostic scheduler | A resource is a `(name, amount)` pair the task declares. The scheduler does arithmetic on counters and asks one yes/no question about each input. |
| 3 | Single source of truth | Task state lives in `TaskMgr`; handoff state lives in `HandoffMgr`. The scheduler's pools are a derived index, never a second copy. |
| 4 | The engine decides when, the agent decides what | The scheduler owns task state and never writes handoff state. An agent owns its handoffs' content and validity, and may submit tasks, but may not redirect the graph. |
| 5 | Algorithm decoupled from mechanism | Ordering lives behind `SchedulePolicy`. The first implementation is naive FIFO. |
| 6 | Simplicity is a requirement | Where a mature solution exists, use it (§9). Where none fits, the implementation stays small enough to read in one sitting. |
| 7 | One fact, one place — reuse the implementation | Where two operations mean the same thing, one is expressed in terms of the other rather than reimplemented. `update_task` is `remove_queued` + `submit` (§5.1); "which agent is running this" is `history[-1].agent_uuid`, not a field (§3.2); eligibility is a query, not a cached counter (§3.2). Structural clarity comes first — this is not licence to collapse two genuinely different concerns into one. |

---

## 3. Domain model

### 3.1 Handoff

A handoff is the unit of transfer between tasks. It is **instantiated when its
producing task is created**, not when its content is produced. A downstream task
therefore holds the handoff — and its `uuid` — from birth. The upstream task only
fills in the content later.

This is what makes dependency resolution a lookup by uuid rather than a matching
engine.

#### Versions

A handoff has a stable identity and an **append-only list of versions**. Each
version is one attempt to produce the content; a re-run of the producing task
appends a new one rather than overwriting.

| Handoff field | Type | Meaning |
|---|---|---|
| `uuid` | `str` | Stable identity, assigned at instantiation |
| `type` | `str` | Opaque to the scheduler; meaningful to agents |
| `produced_by` | `str \| None` | **Task uuid** of the producer; `None` if externally supplied |
| `versions` | `list[HandoffVersion]` | Append-only. The last entry is *latest* |

| Version field | Type | Meaning |
|---|---|---|
| `version` | `int` | Monotonic, starting at 0 |
| `status` | `HandoffStatus` | `GENERATING` while open; `VALID \| INVALID` once sealed |
| `produced_by_agent` | `str \| None` | **Agent uuid** of the run that produced it |
| `timestamp` | `float` | When this version was opened |
| `content` | `Any` | Content, or a reference to it (§8.2) |

```
handoff declared          versions: []            check_if_latest_valid -> False
  agent starts    ──→     v0 [GENERATING]         -> False
  agent seals     ──→     v0 [VALID]              -> True      ← latest
  producer re-run ──→     v0 [VALID], v1 [GENERATING]  -> False
  agent seals     ──→     v0 [VALID], v1 [VALID]  -> True      ← latest
```

A version is `GENERATING` from the moment it is opened and becomes `VALID` or
`INVALID` when the agent seals it. **A sealed version is never rewritten.**
Re-running a producer appends `v+1`; `v` stays exactly as it was, which is what
makes the execution history (§3.2) auditable.

There is no `CREATED` status. A handoff between `declare` and its agent's first
write simply has an empty version list, and `check_if_latest_valid` returns
`False` for it — the same answer as for a handoff mid-write or one sealed
`INVALID`. The scheduler never needs to tell those cases apart; only agents do,
and they can read `versions` directly.

`HandoffStatus` is therefore `GENERATING | VALID | INVALID`.

#### Everything links back by uuid

Three identities exist — task, agent, handoff — and each artefact records which
task it belongs to. The links are uuid references, resolved through the owning
manager; no object holds a pointer to another.

| From | To | Where | Purpose |
|---|---|---|---|
| handoff | task | `Handoff.produced_by` | which task is responsible for filling it |
| handoff version | agent | `HandoffVersion.produced_by_agent` | which run wrote this version |
| agent | task | agent-side record | which task the agent was bound to |
| task | agent | `history[-1].agent_uuid` | which agent is running it now (§3.2) |
| task | handoff | `Task.inputs` / `Task.outputs` | declared dependencies |

Handoff and agent maintain the pairing between them from both sides: a version
names the agent that wrote it, and the agent's own record names the handoffs it
touched. Neither side is authoritative for the other — the pair is what makes a
run reconstructible from either end.

#### Agents own handoff state entirely

**No manager and no scheduler advances a handoff.** Whether content is usable is a
question about the content, and only the agents on either side can answer it. The
producing agent opens a new version when it starts writing and records the verdict
when it finishes. A consuming agent may re-check its inputs on its own terms.

`HandoffMgr` records and serves. It has no validation logic, no content schema,
and no opinion. It does not open versions on its own initiative and it never
derives a status.

The scheduler touches handoffs in exactly three ways, none of which set a status:

| Call | When | Effect on state |
|---|---|---|
| `declare(uuids, produced_by)` | at `submit` | creates identity; **no version** |
| `check_if_latest_valid(uuid)` | at each decision point | none — a read |
| `latest_version(uuid)` | at dispatch, to pin the history | none — a read |

Asking is not deciding: the scheduler reads a fact an agent established. This
keeps it content-agnostic (§2, principle 2) while still resolving dependencies.

#### Why versioning removes the need for propagation

Re-running an upstream task does not disturb anything downstream:

- Older versions are immutable, so a consumer that already ran against `v0`
  retains a truthful record of what it consumed (§3.2).
- A consumer that has not yet run re-asks `check_if_latest_valid` at dispatch
  time, so it sees whatever is current then — no cached counter to invalidate.
- A consumer currently running is unaffected; it holds the version it was given.

The system therefore contains **no invalidation-propagation logic at all**, and
did not have to forbid re-running a producer to achieve that.

One consequence is deliberate and stated plainly: because a new version is opened
by the producing agent when it starts writing, there is a window between resuming
a task and its agent actually beginning. During that window `latest` is still the
previous version, so a downstream task may be dispatched against it. See §10.

### 3.2 Task

| Field | Type | Meaning |
|---|---|---|
| `id` | `str` | Identity |
| `agent` | `str` | Agent *name* — what to run, resolved through `AgentMgr` |
| `inputs` | `list[str]` | Handoff uuids required to run |
| `outputs` | `list[str]` | Handoff uuids this task will fill |
| `resources` | `dict[str, float]` | Declared, never inferred, e.g. `{"gpu": 2, "token": 100_000}` |
| `status` | `TaskStatus` | See below |
| `created_at` | `float` | FIFO ordering key |
| `expedited` | `bool` | Set by `expedite()`; a policy hint |
| `history` | `list[Execution]` | Append-only; one entry per run |

Note `outputs` is a plain uuid list again. Verdicts live on handoff versions
(§3.1), not on the task.

#### Execution history — a stack, not an archive

A task may run more than once — `resume` after `FAILED` or `SUSPENDED`. Each run
appends a record.

**The top of the stack is the current run.** It is not merely the newest history
entry; it *is* where the live facts about an executing task are kept. "Which
agent is running this, against which handoff versions" is read from
`history[-1]`, never from a field duplicating it on the task. A run is in
progress exactly when `history[-1].ended_at is None`.

Everything below the top is immutable history.

| Field | Type | Meaning |
|---|---|---|
| `attempt` | `int` | 0-based |
| `agent_uuid` | `str` | Which agent instance ran it |
| `input_versions` | `dict[str, int]` | uuid → version actually consumed |
| `output_versions` | `dict[str, int]` | uuid → version produced |
| `started_at` / `ended_at` | `float \| None` | `ended_at` is `None` while running |
| `outcome` | `TaskStatus \| None` | Terminal status of this attempt |

This is what makes a re-run auditable rather than destructive. `input_versions`
pins exactly which upstream content a run saw, so a later re-run of an upstream
producer cannot retroactively change the record of what happened.

`Task` carries no `agent_uuid` of its own. Binding an agent means pushing a record
whose `agent_uuid` is set; asking which agent is bound means reading the top. One
fact, one place (§2, principle 7).

`Task.status` is *not* derived from the stack, and the two are not redundant.
Status is a scheduling fact the scheduler owns; the stack is an execution fact the
agent's run produces. Most of the statuses — `WAITING_HANDOFF`,
`WAITING_RESOURCE`, `SUSPENDED`, `CANCELLED` — describe a task that is not running
and have nothing to derive from.

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
| `WAITING_HANDOFF` | Not every input's latest version is `VALID` | Re-checked at each decision point; all valid → `WAITING_RESOURCE` |
| `WAITING_RESOURCE` | Inputs ready, competing for resources | Full resource set granted atomically → `RUNNING` |
| `RUNNING` | Holds a lease; the runner is executing | Runner reports done, or `stop()` → `STOPPING` |
| `STOPPING` | Stop requested; the runner has not confirmed | `on_stopped()` → `SUSPENDED` |
| `SUCCEEDED` | The task ran to completion. Its outputs carry whatever verdict the agent recorded — not necessarily `VALID` (§6.3). **Final** | — |
| `FAILED` | Finished unsuccessfully. Resumable | `resume()` |
| `SUSPENDED` | Stopped on request. Resumable | `resume()` |
| `CANCELLED` | Removed while queued. **Final** | — |

`STOPPING` is a transient of `RUNNING`, present because a runner cannot terminate
a process instantaneously. It holds its resources until `on_stopped()` confirms.
Recovery treats it as `SUSPENDED` (§6.4).

`resume()` accepts `FAILED` and `SUSPENDED`. It rejects `SUCCEEDED` and
`CANCELLED`. The resumed task re-enters `WAITING_RESOURCE` if
`check_if_latest_valid` holds for every input, otherwise `WAITING_HANDOFF` — the
landing pool is recomputed, not remembered. `resume()` does **not** touch the
task's output handoffs; the agent opens new versions when it starts (§3.1).

**Eligibility is a query, not a cached counter.** A task's readiness is
recomputed by asking `check_if_latest_valid` for each input at every decision
point. There is no dependency counter to keep in sync, and therefore nothing that
can go stale when an upstream producer re-runs.

**Why the two waits are separate collections.** `WAITING_HANDOFF` is an unordered
set: there is no decision to make there, only a question to re-ask.
`WAITING_RESOURCE` is ordered: it is the single place in the system where a
scheduling decision occurs. Merging them into one queue with a filter would hide
that distinction.

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
   ┌────────────┬─────────────┬─────┴──────┬──────────────┬─────────────┐
   │            │             │            │              │             │
┌──▼───────┐ ┌──▼────────┐ ┌──▼───────┐ ┌──▼─────────┐ ┌──▼───────┐ ┌───▼────────┐
│Scheduler │ │HandoffMgr │ │ TaskMgr  │ │ResourceMgr │ │ AgentMgr │ │ TaskRunner │
│          │ │           │ │          │ │ ├ GpuMgr   │ │          │ │            │
│ pools =  │ │ versions; │ │  owns    │ │ └ TokenMgr │ │ get(name)│ │ start/stop │
│  status  │ │ records   │ │  state   │ │            │ │          │ │            │
│  index   │ │ only      │ │          │ │            │ │          │ │            │
└────┬─────┘ └─────▲─────┘ └──────────┘ └────────────┘ └──────────┘ └─────┬──────┘
     │             │                                                       │
     │  check_if_latest_valid (read only)                                  │
     └─────────────┘                       new_version / record ───────────┘
                                           (agent, via the runner)

              ┌────────────┐  ┌──────────────┐
              │  StoreMgr  │  │SchedulePolicy│
              │ (json file)│  │ (FIFO first) │
              └─────▲──────┘  └──────────────┘
                    │  write-through from TaskMgr and HandoffMgr (§7)
```

The two arrows into `HandoffMgr` are the whole story of §3.1: the scheduler only
reads, and only agents write.

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
| `store_mgr` | `StoreMgr` — the persistence backend, shared by `TaskMgr` and `HandoffMgr` |
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
    def save(self, kind: str, key: str, record: dict) -> None: ...
    def load_all(self, kind: str) -> list[dict]: ...
    def delete(self, kind: str, key: str) -> None: ...
```

`kind` separates the key spaces: `TaskMgr` writes `"task"`, `HandoffMgr` writes
`"handoff"` (§7). Keeping the store ignorant of both types is what lets one
implementation serve both managers.

The first implementation is `JsonFileStoreMgr`: one JSON file per record, in a
directory per kind. No database. Chosen for inspectability while the shape of the
data is still settling.

### 4.3 Managers

| Manager | Responsibility | Does *not* |
|---|---|---|
| `HandoffMgr` | Hold versions; record what an agent reports; answer `check_if_latest_valid`; persist and restore them | **Judge validity, or advance state on its own** (§3.1). Store large payloads inline (§8.2) |
| `TaskMgr` | Own task state and execution history; write through `store_mgr` | Make scheduling decisions |
| `ResourceMgr` | Account for one named pool | Know about tasks |
| `AgentMgr` | `get(name) -> Agent` | Anything else. Deliberately trivial |

### 4.4 Pools as an index

The scheduler keeps `pools: dict[TaskStatus, set[task_id]]` — one bucket per
`TaskStatus`, so a task is in exactly one. Only three are load-bearing for
scheduling (`WAITING_HANDOFF`, `WAITING_RESOURCE`, `RUNNING`); the rest exist so
that "which tasks are suspended" is a lookup rather than a scan.

The pools are an index over task state, not a second copy of it. Every transition
goes through one internal method that updates `TaskMgr` and the index together,
so the two cannot drift **as long as that method is the only writer**. It is an
invariant maintained by construction, not one enforced by the type system;
acceptance criterion 12 asserts it directly.

Because eligibility is a query (§3.2), `WAITING_HANDOFF` membership is a hint
about where a task was last seen, not a claim about its current readiness.
Dispatch re-checks.

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
    def save(self, kind: str, key: str, record: dict) -> None: ...
    def load_all(self, kind: str) -> list[dict]: ...
    def delete(self, kind: str, key: str) -> None: ...

# ---- handoff ----
# Called by the SCHEDULER (read-only):
class HandoffMgr:
    def declare(self, uuids: list[str], produced_by: str) -> None:
        """Instantiate handoffs with an empty version list. No status is set."""
    def check_if_latest_valid(self, uuid: str) -> bool:
        """True iff the newest version exists and is VALID. A read, not a judgement."""
    def latest_version(self, uuid: str) -> int | None:
        """For pinning into an Execution record (§3.2)."""
    def get(self, uuid: str) -> Handoff: ...
    def restore(self) -> None:
        """Reload handoffs and versions from the store, at startup (§6.4)."""

# Called by the AGENT, through its runner (write). Each of these persists:
    def new_version(self, uuid: str, agent_uuid: str) -> int:
        """Open version v+1 as GENERATING. Returns the new version number."""
    def record(self, uuid: str, version: int,
               status: HandoffStatus,       # VALID | INVALID, decided by the agent
               content: Any = None) -> None:
        """Seal that version. Raises if it is already terminal."""

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

The split is deliberate. `can_afford` / `take` are separate so the scheduler can
check every declared resource before mutating any of them (§6.2). The two halves
of `HandoffMgr` are separate so the read/write asymmetry of §3.1 is visible in the
interface: the scheduler calls only the first group, agents only the second.

### 5.1 Scheduler API

| Method | Effect | Rejects |
|---|---|---|
| `submit(task)` | `declare` its output handoffs; place in the pool its inputs dictate; dispatch | Duplicate id; undeclared resource pool |
| `expedite(task)` | `submit`, but requires every input already valid and marks the task for front-of-queue ordering | Any input failing `check_if_latest_valid` |
| `remove_queued(tid)` | `→ CANCELLED` | Task not in a waiting pool |
| `stop(tid)` | `RUNNING → STOPPING`; calls `runner.stop(tid)` | Task not `RUNNING` |
| `resume(tid)` | `FAILED \| SUSPENDED →` recomputed waiting pool; dispatch. Does not touch handoffs | `SUCCEEDED`, `CANCELLED`, or any live state |
| `update_task(tid, ...)` | Sugar: remove and re-submit under the same id, with new inputs/outputs/resources | Task not queued |
| `on_task_done(tid, result)` | Release resources; close the execution record; dispatch | Task not `RUNNING` |
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
  ├─ TaskMgr.add(task)                          → persisted
  ├─ HandoffMgr.declare(task.outputs, task.id)  → identity only, no version yet
  ├─ pool = all(check_if_latest_valid(h) for h in task.inputs)
  │            ? WAITING_RESOURCE : WAITING_HANDOFF
  └─ try_dispatch()
```

`declare` creates the handoff so downstream tasks can reference its uuid. It does
not open a version — the producing agent does that when it starts (§3.1).

### 6.2 Dispatch — atomic, all-or-nothing

Called at every decision point: after `submit`, `expedite`, `resume`,
`on_task_done`, `on_stopped`.

```python
handoff_mgr = registry.get("handoff_mgr")
task_mgr    = registry.get("task_mgr")

# 1. re-check eligibility; a hint in WAITING_HANDOFF may have become ready,
#    and one in WAITING_RESOURCE may have gone stale. Snapshot first: move()
#    mutates the very sets being scanned.
for tid in list(pools[WAITING_HANDOFF] | pools[WAITING_RESOURCE]):
      task  = task_mgr.get(tid)
      ready = all(handoff_mgr.check_if_latest_valid(h) for h in task.inputs)
      move(tid, WAITING_RESOURCE if ready else WAITING_HANDOFF)

# 2. order the eligible set — the one scheduling decision in the system
eligible = [task_mgr.get(t) for t in pools[WAITING_RESOURCE]]
for tid in registry.get("policy").select(eligible, resource_snapshot()):
      task = task_mgr.get(tid)
      pool = lambda r: registry.get(f"resource:{r}")

      # 3. all-or-nothing: verify the FULL set before mutating anything
      if not all(pool(r).can_afford(n) for r, n in task.resources.items()):
            continue                          # grant nothing; leave it queued
      for r, n in task.resources.items():
            pool(r).take(n)

      # 4. bind an agent by PUSHING a record; the stack top is the binding (§3.2)
      agent = registry.get("agent_mgr").get(task.agent)
      task_mgr.push_execution(
            tid, agent_uuid=agent.uuid,
            input_versions={h: handoff_mgr.latest_version(h) for h in task.inputs},
      )
      task_mgr.set_status(tid, RUNNING)
      registry.get("runner").start(task, agent, on_done=self.on_task_done)
```

**The whole declared set is verified before anything is mutated.** A task that
does not fit takes nothing and stays queued. Never acquire incrementally; never
let a queued task hold anything. This makes hold-and-wait deadlock structurally
impossible, and costs one extra loop.

Step 1 is what replaces a dependency counter. Re-checking every waiting task at
each decision point is O(waiting × inputs) — acceptable at this scale, and it
means no cached readiness can ever be wrong. If the waiting set grows large
enough to matter, a reverse index from handoff to consumers is the optimisation,
and it does not change any of the semantics above.

Step 4 pins the input versions into the record *before* the run starts, so the
history says what the run actually saw. Pushing the record *is* the act of binding
an agent — there is no separate field to assign (§3.2).

### 6.3 Completion

The runner reports a `TaskResult`. Handoff state has **already been written by
the agent** through `new_version` / `record`; the scheduler neither sets nor
infers it.

```python
@dataclass
class TaskResult:
    ok: bool                            # did the run itself complete
    output_versions: dict[str, int]     # uuid -> version this run produced
    actual_usage: dict[str, float]      # for consumable settlement
```

```
on_task_done(tid, result)
  ├─ release lease   (renewable: full; consumable: settle result.actual_usage)
  ├─ TaskMgr.close_execution(tid, result.output_versions,      # seals the stack top
  │                        outcome = SUCCEEDED if result.ok else FAILED)
  ├─ TaskMgr.set_status(tid, SUCCEEDED if result.ok else FAILED)   → persisted
  └─ try_dispatch()          # step 1 re-checks every waiter against latest
```

There is no handoff manipulation in this flow at all. Downstream promotion is not
pushed here; it falls out of the re-check in `try_dispatch`.

`result.ok` and handoff validity are independent. An agent can complete cleanly
and still record its output `INVALID` — it ran fine and concluded the result is
unusable. The task is `SUCCEEDED`; its consumers stay in `WAITING_HANDOFF`.

A failed task releases its resources and is recorded `FAILED`. Whatever its agent
last wrote to the handoff stands; if the agent opened a version and never sealed
it, that version remains `GENERATING`, and `check_if_latest_valid` is false — so
consumers correctly stay blocked. Downstream tasks are not cancelled. That is
deliberate: the system does not cancel them, and does not pretend to.

### 6.4 Recovery

```
resume_system()
  ├─ tasks = store_mgr.load_all("task")
  ├─ handoff_mgr.restore()            ← see below
  ├─ RUNNING → WAITING_RESOURCE       (leases do not survive a restart)
  ├─ STOPPING → SUSPENDED
  ├─ close a dangling stack top (ended_at unset) as interrupted
  ├─ reset every resource pool to full capacity
  └─ try_dispatch()                   # eligibility is recomputed, never restored
```

Eligibility needs no reconstruction: it is a query, so recovery only has to
ensure the handoff versions the query reads are present.

#### Handoffs persist themselves

`HandoffMgr` owns handoff persistence. The reason is not a crash window — it is
that **the information is not present anywhere else.**

`ok` and a version's verdict are independent facts (§6.3): an agent may finish
cleanly and still seal its output `INVALID`. A task record carries only `ok`.
Rebuilding handoff state from task records would therefore mean guessing
`ok=True → VALID`, which is wrong in exactly the case §6.3 describes — not
because of restart timing, but because the verdict was never written down there
at all. Guessing it would also be the inference §3.1 exists to prevent.

Two further gaps: a handoff supplied externally has no producing task to replay,
and a version abandoned mid-write by a crashed agent appears in no completed
execution record.

So `HandoffMgr` persists what it owns, symmetrically with `TaskMgr`:

```python
class HandoffMgr:
    def restore(self) -> None:
        """Reload handoffs and their versions from the store."""
```

It writes through on `declare`, `new_version`, and `record` — the same
write-through discipline as §7 — using the same `store_mgr` under a separate key
space. This costs one more persisted record type and removes an impossible
reconstruction.

A version left `GENERATING` across a restart stays `GENERATING`. Its agent is gone
and will never seal it, so `check_if_latest_valid` stays false and consumers stay
blocked — correct, and the same outcome as the crash case in §6.3. Resuming the
producing task appends a fresh version rather than adopting the abandoned one.

---

## 7. Persistence

Two things are persisted: **task state** and **handoff versions**. The scheduler
persists nothing of its own — its pools are an index over task status (§4.4), so
reading tasks back rebuilds them.

Handoff state cannot be derived from task records: a version's verdict is
independent of the producing run's `ok` flag, and only `HandoffMgr` is ever told
it (§6.4). Each manager persists what it owns.

Handoff persistence is a larger subject than this document treats it as — a
handoff is the durable artefact the whole system exists to pass around, and
archival, retention, and content addressing all belong to it. `HandoffMgr` is
where that lives. Only the part the scheduler depends on is specified here.

- `StoreMgr` is a `Protocol`, registered under `store_mgr` (§4.2). `TaskMgr`
  resolves it from the registry when it writes; it is not a sub-object.
- The first implementation is `JsonFileStoreMgr`: one JSON file per task, in a
  directory. No database. Chosen for inspectability while the shape of the data
  is still settling.
- Every state-mutating operation on `TaskMgr` writes through. Completion is the
  principal trigger but not the only one — submission, cancellation, and stop all
  mutate state and must survive a crash.

The persisted task record includes its **execution history** (§3.2), not merely
its current status. The history is the audit trail — which agent ran, against
which input versions — and it is recoverable from nowhere else.

`HandoffMgr` follows the same write-through discipline: `declare`, `new_version`,
and `record` each persist. Because those calls come from agents, a handoff is
persisted at the moment the agent acts rather than at task completion.

Handoff *content* is a different question, deliberately left open (§8.2). The
persisted record holds a reference, not a payload.

**Not atomic.** A crash between `TaskMgr`'s write and `HandoffMgr`'s can leave the
two disagreeing. Recovery tolerates this in the safe direction: anything short of
a sealed `VALID` version reads as not-valid, so a consumer stays blocked rather
than running against unverified content. Cross-manager atomicity is not attempted
(§10).

---

## 8. Deliberate omissions

### 8.1 Not built

| Omitted | Why |
|---|---|
| Resource kinds beyond a name | The scheduler cannot tell a GPU from an API quota. That is the point. |
| Duration model | Agent durations are unknown, heavy-tailed, and not reproducible. |
| Graph library | The only graph operation is asking whether a task's inputs are valid. No traversal, no topological order. |
| Cascading failure | Out of scope by §1.2. Versioning (§3.1) removes the need for it in the re-run case. |
| Event bus | Deferred. A hook callback is left at each transition point, and the registry (§4.1) gives a later bus a natural place to live. |
| Lease TTL sweep | Deferred; noted in §10. |

### 8.2 Handoff content storage

`HandoffMgr` keeps handoff *metadata* and versions. Where large payloads live is a
larger question, deliberately left open — `content` accommodates a reference
rather than mandating inline storage. Storing large agent payloads in the state
record is a known way to make the state store the bottleneck.

Versioning sharpens the question rather than answering it: N runs of a producer
mean N payloads, and nothing here says when an old one may be discarded.

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

Two departures from that survey, both consequences of decisions taken here:

- It recommends a **dependency counter** as the graph representation. This design
  queries `check_if_latest_valid` at each decision point instead, because
  versioned handoffs make a cached counter the thing most likely to go stale
  (§3.1). The counter remains the optimisation if the waiting set grows (§10).
- It treats a produced artefact as **immutable and singular**. Versioning is an
  addition, and it is what lets a producer be re-run without the cascading
  invalidation the survey identifies as an unresolved design gap.

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
| **Re-run window** | A new version is opened by the producing agent when it starts writing (§3.1), so between `resume(t)` and that moment, `latest` is still the previous version. A downstream task dispatched in that window runs against stale-but-valid content. Accepted as the price of the scheduler never touching handoff state. Mitigations if it bites: have the runner open versions at `start`, or have `resume` mark outputs pending. Neither is built. |
| Cross-manager atomicity | `TaskMgr` and `HandoffMgr` persist independently, so a crash between the two writes leaves them briefly inconsistent (§7). Recovery fails safe — a consumer stays blocked — but the window exists. A shared transaction, or a single append-only log both managers write to, is the known fix. Not built. |
| Version retention | Nothing says when an old version's content may be discarded (§8.2). Unbounded re-runs mean unbounded payloads. |
| Re-check cost | Eligibility is recomputed for every waiting task at every decision point (§6.2, step 1). Fine at this scale; a reverse handoff→consumer index is the known optimisation. |

---

## 11. Acceptance criteria

The implementation is complete when each of the following is demonstrated by a
test:

1. A task whose inputs all report `check_if_latest_valid` lands in
   `WAITING_RESOURCE` at submit.
2. A task with an unfilled input lands in `WAITING_HANDOFF` and is dispatched
   exactly when a later decision point finds its last input valid.
3. A task requiring more of any pool than is available starts nothing and
   **consumes nothing** — verified by asserting every pool is unchanged.
4. A renewable pool returns to its prior level after completion; a consumable
   pool is debited by the settled actual, not the reservation.
5. `stop` → `on_stopped` → `resume` returns a task to the correct pool, recomputed
   from its inputs.
6. `resume` on a `SUCCEEDED` or `CANCELLED` task is rejected.
7. A failed task releases its resources and leaves its dependents in
   `WAITING_HANDOFF`; an output whose version the agent left `GENERATING` does not
   satisfy `check_if_latest_valid`.
8. `resume_system()` reconstructs pools from persisted tasks and versions from
   persisted handoffs, with `RUNNING` and `STOPPING` tasks demoted to
   `WAITING_RESOURCE` and `SUSPENDED` respectively.
9. `expedite` places a task ahead of earlier-submitted eligible tasks, and is
   rejected when any input is not valid.
10. Swapping `SchedulePolicy` changes dispatch order and nothing else.
11. `update_task` on a queued task replaces its inputs/outputs/resources under the
    same id and recomputes its pool; on a `RUNNING` task it is rejected.
12. A pool never disagrees with `TaskMgr`: after any sequence of operations, the
    union of the pools equals the set of all tasks, and each task appears in
    exactly the pool matching its stored status.
13. A run reporting `ok=True` whose agent recorded its output `INVALID` becomes
    `SUCCEEDED` while its consumers stay in `WAITING_HANDOFF` — completion and
    validity are independent (§6.3).
14. **The scheduler never writes handoff state.** Across a full submit → dispatch →
    complete → resume → re-dispatch cycle, a `HandoffMgr` spy records calls to
    `new_version` and `record` originating only from the agent, and calls from the
    scheduler only to `declare`, `check_if_latest_valid`, and `latest_version`.
15. Two isolated `Registry()` instances do not share components, and `get()` on an
    unregistered name raises with that name in the message.
16. Re-running a producer appends a version and leaves earlier ones byte-identical;
    a consumer that already ran retains `input_versions` pointing at the version it
    actually consumed.
17. A consumer dispatched after a producer's re-run reads the new version, with no
    invalidation call made anywhere in the system.
18. Execution history grows by exactly one entry per run, each carrying the bound
    `agent_uuid` and the pinned input/output versions.
19. A handoff sealed `INVALID` before a restart is still `INVALID` after
    `restore()`, and one left `GENERATING` is still `GENERATING` — neither is
    re-derived from the producing run's `ok` flag (§6.4).
20. Resuming a task whose previous run left a `GENERATING` version appends a new
    version rather than reusing the abandoned one.
21. The bound agent is readable only from `history[-1]`: `Task` exposes no
    `agent_uuid`, and after a `resume` the top reports the new run's agent while
    the entry beneath still reports the previous one.
22. Every handoff resolves to its owning task by uuid, and every version to the
    agent that wrote it; a run is reconstructible starting from either the task or
    the handoff (§3.1).
23. `update_task` produces the same observable state as `remove_queued` followed
    by `submit` with the new arguments — verified by comparing against that
    sequence, not by re-asserting the outcome (§2, principle 7).
