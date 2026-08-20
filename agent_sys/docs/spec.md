# Agent Task Graph — Specification

| | |
|---|---|
| Status | Draft, pending review |
| Revision | 5 — 2026-08-20. Handoff owns its versions; per-version provenance; `depends_on`; no result object |
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
  This system records the binding between a task, an agent, and the handoff
  versions it wrote (§3.3); it does not manage the agent's lifecycle.
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
| 7 | One fact, one place — reuse the implementation | Where two operations mean the same thing, one is expressed in terms of the other rather than reimplemented. `update_task` is `remove_queued` + `submit` (§5.1); "which agent is running this" is `history[-1].agent_id`, not a field (§3.2); eligibility is a query, not a cached counter (§3.2). Structural clarity comes first — this is not licence to collapse two genuinely different concerns into one. |

---

## 3. Domain model

### 3.1 Handoff

A handoff is the unit of transfer between tasks. It is **instantiated when its
producing task is created**, not when its content is produced. A downstream task
therefore holds the handoff — and its `uuid` — from birth. The upstream task only
fills in the content later.

This is what makes dependency resolution a lookup by uuid rather than a matching
engine.

#### A handoff owns its versions

A `Handoff` is the **slot** in the graph: one stable identity, one place a
downstream task points at. What fills it is a `HandoffVersion`, and a handoff
holds an append-only list of them.

| `Handoff` field | Type | Meaning |
|---|---|---|
| `id` | `HandoffId` | Stable identity of the slot |
| `type` | `str` | Opaque to the scheduler; meaningful to agents |
| `versions` | `list[HandoffVersion]` | Append-only. The last is *latest* |

| `HandoffVersion` field | Type | Meaning |
|---|---|---|
| `version` | `int` | Monotonic from 0. `(handoff id, version)` names this artefact |
| `status` | `HandoffStatus` | `CREATED → GENERATING → VALID \| INVALID` |
| `producer_task_id` | `TaskId \| None` | Task that produced **this version**; `None` if externally supplied |
| `producer_agent_id` | `AgentId \| None` | Agent run that wrote it. `None` until opened |
| `timestamp` | `datetime` | When this version was opened |
| `content` | `Any` | Content, or a reference to it (§8.2) |

**Provenance is per version, not per slot.** A re-run writes a new version with a
different `producer_agent_id`; `update_task` can even change which task produces
a slot, so `producer_task_id` belongs on the version too. A slot-level producer
field could not say who wrote v0 after either of those.

The names say what they hold. A field called `produced_by` on a type with two
different kinds of producer is the ambiguity typed ids (below) exist to remove,
and leaving it in the name would undo half of that.

```
declare           v0 [CREATED]                        check_if_latest_valid -> False
  agent opens ──→ v0 [GENERATING]                                           -> False
  agent seals ──→ v0 [VALID]                                                -> True
  re-run      ──→ v0 [VALID], v1 [GENERATING]                               -> False
  agent seals ──→ v0 [VALID], v1 [VALID]         ← latest                   -> True
```

`CREATED` is the declared-but-untouched state: the slot exists so downstream
tasks can name it, and nothing has been written. It is a real state, not the
absence of one, and it is what a `declare` produces.

**A sealed version is never rewritten.** Re-running a producer appends `v+1`;
`v` stays exactly as it was, which is what makes the execution history (§3.2)
auditable.

`HandoffStatus` is `CREATED | GENERATING | VALID | INVALID`. The scheduler
distinguishes none of them — it asks one question, `check_if_latest_valid`, and
the first three answer it identically. Agents read the status directly.

#### Behaviour belongs to the objects

Version bookkeeping is the handoff's own business, and each version owns its
transitions. Neither is the manager's:

| On `Handoff` | Effect |
|---|---|
| `latest` | The newest version, or `None` |
| `is_latest_valid` | `latest is not None and latest.is_valid` |
| `open_next(task_id, agent_id)` | Give an agent a version to write. Adopts an untouched `CREATED` v0 in place; otherwise appends `v+1` as `GENERATING` |
| `get(version)` | One version by number |

| On `HandoffVersion` | Effect |
|---|---|
| `is_valid` | `status is VALID` |
| `seal(status, content)` | `GENERATING → VALID \| INVALID`. Raises if not open |

`open_next` is one verb, not two, and that is the point: the caller says "I am
about to write this handoff" and does not have to know whether it is the first
run or the fourth. Contiguity of version numbers is then structural — the list
index is the version — rather than something a manager has to check.

`HandoffMgr` reimplements none of this. It manages the *collection*: which
handoffs exist, and how they persist (§4.3).

#### Everything links back by uuid

Three identities exist — task, agent, handoff — and each artefact records which
task it belongs to. The links are uuid references, resolved through the owning
manager; no object holds a pointer to another.

**Each has its own type.** `TaskId`, `AgentId`, and `HandoffId` are distinct
types, not interchangeable `str`s. A signature reading `list[str]` says nothing
about which of the three it wants; `list[HandoffId]` says it exactly. They wrap
`uuid.UUID`, so identity generation, comparison, and formatting come from the
standard library rather than from string conventions.

| From | To | Where | Purpose |
|---|---|---|---|
| version | task | `HandoffVersion.producer_task_id` | which task produced this version |
| version | agent | `HandoffVersion.producer_agent_id` | which run wrote it |
| agent | task | `Agent.task_id` | which task the agent is bound to |
| agent | version | `Agent.handoffs: list[HandoffRef]` | which artefacts it touched |
| task | agent | `history[-1].agent_id` | which agent is running it now (§3.2) |
| task | handoff | `Task.inputs` / `Task.outputs`: `list[HandoffId]` | declared slots |
| task | task | `Task.depends_on: list[TaskId]` | the dependency edge, for traversal (§3.2) |

`HandoffRef` is `(HandoffId, version)` — the pair that names a concrete artefact.
An agent records the slot alone only when it has not written yet.

**Both directions are stored.** A version names the agent that wrote it, and the
agent names the versions it touched. Neither side is authoritative for the other;
the pair is what makes a run reconstructible from either end, which is what
acceptance criterion 22 asserts.

References are ids, never objects. Two reasons, and the first is the load-bearing
one:

- **Serialisation would copy.** A `Task` holding `list[Handoff]` dumps the whole
  handoff record inside the task record, so handoff state exists in the handoff
  store *and* in every consumer's record. Validating back produces distinct
  objects — the aliasing that made it one fact in memory is gone.
- **The graph has cycles.** `HandoffVersion → Task` and `Task → Handoff` point at
  each other; object references would recurse on dump.

In memory a Python reference is genuinely shared, so nothing is duplicated
there — the problem appears the moment state crosses the store, which is every
mutation (§7).

#### Agents own handoff state entirely

**No manager and no scheduler advances a handoff.** Whether content is usable is a
question about the content, and only the agents on either side can answer it. The
producing agent opens a new version when it starts writing and records the verdict
when it finishes. A consuming agent may re-check its inputs on its own terms.

`HandoffMgr` holds the handoffs and serves queries over them. It has no
validation logic, no content schema, and no opinion. It does not open versions on
its own initiative and it never derives a status. Version bookkeeping is
`Handoff.open_next`; sealing is `HandoffVersion.seal` (§3.1). The mgr persists
what those produced.

The scheduler touches handoffs in exactly three ways, none of which set a status:

| Call | When | Effect on state |
|---|---|---|
| `declare(ids, producer_task_id)` | at `submit` | creates v0 in state `CREATED` |
| `check_if_latest_valid(hid)` | at each decision point | none — a read |
| `latest(hid)` | at dispatch, to pin the history | none — a read |

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
| `id` | `TaskId` | Identity |
| `agent_spec` | `str` | *Which kind* of agent to run. Names an entry in `AgentMgr`'s spec table |
| `inputs` | `list[HandoffId]` | Handoff slots required to run |
| `outputs` | `list[HandoffId]` | Handoff slots this task will fill |
| `depends_on` | `list[TaskId]` | Upstream tasks. The graph edge (below) |
| `resources` | `dict[str, float]` | Pool **name** → amount. Declared, never inferred, e.g. `{"gpu": 2, "token": 100_000}` |
| `status` | `TaskStatus` | See below |
| `created_at` | `datetime` | FIFO ordering key |
| `expedited` | `bool` | Set by `expedite()`; a policy hint |
| `history` | `list[Execution]` | Append-only; one entry per run |

`outputs` holds ids, not handoffs. Verdicts live on the handoff version (§3.1),
not on the task.

`agent_spec` is a *spec* name, not an agent. An `Agent` is a concrete object with
an `AgentId`, created per run; the task says which kind to create. `AgentMgr`
holds both — the spec table and the instances (§4.3) — and the field name has to
distinguish them or every reader has to guess.

`resources` is keyed by pool *name*, and that `str` is genuinely a name, not an
identity. Resource pools are singletons registered under `resource:<name>`; there
is no fourth id type.

#### `depends_on` is the graph; the scheduler does not use it

`depends_on` records the dependency edges directly. Nothing in scheduling reads
it: eligibility is `check_if_latest_valid` over `inputs` (§3.2 below), and that
stays true. The field exists because **everything else** needs the graph —
topological order for display, impact analysis, cycle detection (§10), progress
reporting — and reconstructing it by joining `outputs` against `inputs` across
every task is an O(n²) scan of something the submitter knew for free.

Three fields, three jobs, and they are not redundant:

| Field | Answers | Read by |
|---|---|---|
| `depends_on` | which *tasks* must come first | traversal, display, analysis |
| `inputs` | which *slots* must be valid | the scheduler, at every decision point |
| `history[-1].input_versions` | which *artefacts* this run actually consumed | audit |

The first is structure, the second is the readiness question, the third is what
happened. `inputs` cannot be replaced by `input_versions`: a queued task has no
history yet, and that is precisely when the scheduler needs to know what it
depends on.

Keeping `depends_on` consistent with `inputs` is the submitter's job. The system
does not derive one from the other, and §10 records that nothing yet checks they
agree.

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
| `agent_id` | `AgentId` | Which agent instance ran it |
| `input_versions` | `dict[HandoffId, int]` | Handoff → version actually consumed |
| `output_versions` | `dict[HandoffId, int]` | Handoff → version produced |
| `started_at` / `ended_at` | `datetime \| None` | `ended_at` is `None` while running |
| `outcome` | `TaskStatus \| None` | Terminal status of this attempt |

This is what makes a re-run auditable rather than destructive. `input_versions`
pins exactly which upstream content a run saw, so a later re-run of an upstream
producer cannot retroactively change the record of what happened.

`Task` carries no `agent_id` of its own. Binding an agent means pushing a record
whose `agent_id` is set; asking which agent is bound means reading the top. One
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
| `FAILED` | Finished unsuccessfully. Resumable | `resume_task()` |
| `SUSPENDED` | Stopped on request. Resumable | `resume_task()` |
| `CANCELLED` | Removed while queued. **Final** | — |

`STOPPING` is a transient of `RUNNING`, present because a runner cannot terminate
a process instantaneously. It holds its resources until `on_stopped()` confirms.
Recovery treats it as `SUSPENDED` (§6.4).

`resume_task()` accepts `FAILED` and `SUSPENDED`. It rejects `SUCCEEDED` and
`CANCELLED`. The resumed task re-enters `WAITING_RESOURCE` if
`check_if_latest_valid` holds for every input, otherwise `WAITING_HANDOFF` — the
landing pool is recomputed, not remembered. `resume_task()` does **not** touch the
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

### 3.3 Agent

A spec is a *kind* of agent; an `Agent` is one created to run one task.

| Field | Type | Meaning |
|---|---|---|
| `id` | `AgentId` | Identity of this instance |
| `spec` | `str` | Which kind — the name `Task.agent_spec` gave |
| `task_id` | `TaskId \| None` | The task it is bound to |
| `handoffs` | `list[HandoffRef]` | Versions it touched, as `(HandoffId, version)` |
| `knowledge` | `Any` | Left empty (`mission.md`) |
| `config` | `dict` | Left empty (`mission.md`) |

`task_id` and `handoffs` are the agent's half of the two-way links (§3.1). Without
them a run is reconstructible only from the task end, and the pairing the spec
claims does not exist.

One agent, one run. A re-run creates a new agent with a new id, which is what
makes `Execution.agent_id` distinguish attempts.

What an agent *does* — its process, its prompts, its own durability — is out of
scope (§1.2). This record is the system's index of it, not its state.

### 3.4 Resource

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
│ pools =  │ │ handoffs  │ │  tasks   │ │ └ TokenMgr │ │  agents  │ │ start/stop │
│  status  │ │ keyed by  │ │ keyed by │ │            │ │ keyed by │ │            │
│  index   │ │ HandoffId │ │  TaskId  │ │ a counter  │ │ AgentId  │ │            │
└────┬─────┘ └─────▲─────┘ └──────────┘ └────────────┘ └──────────┘ └─────┬──────┘
     │             │                                                       │
     │  check_if_latest_valid (read only)                                  │
     └─────────────┘                    open/seal, then append ────────────┘
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
- **`resolve(pattern)` handles the one-to-many case.** `resource:*` expands to
  every registered pool, in registration order. This exists because recovery
  (§6.4) and dispatch both need "all resource managers" and neither should hold a
  hard-coded list of pool names.

The trade-off is accepted deliberately: the registry makes dependencies implicit
at the call site rather than visible in a constructor signature. The mitigations
are the three requirements above — isolated instances, loud failure, and the
fixed name table.

### 4.2 StoreMgr

Persistence is a registered component like any other, not a sub-object of
`TaskMgr`. `TaskMgr` resolves `store_mgr` when it writes.

A store is a CRUD store; it offers all four operations.

```python
class StoreMgr(Protocol):
    def create(self, kind: str, key: str, record: dict) -> None: ...  # raises if present
    def read(self, kind: str, key: str) -> dict | None: ...
    def read_all(self, kind: str) -> list[dict]: ...
    def update(self, kind: str, key: str, record: dict) -> None: ...  # raises if absent
    def delete(self, kind: str, key: str) -> None: ...
    def exists(self, kind: str, key: str) -> bool: ...
```

`create` and `update` are separate rather than one upsert because their
preconditions differ and are worth enforcing: `TaskMgr.add` means *new*, and a
collision is a bug the store should surface; `set_status` means *existing*, and
writing a record that vanished is equally a bug. A single `save` would silently
accept both mistakes.

`read` exists as the single-key counterpart to `read_all`. Recovery uses
`read_all`; nothing in the scheduler path reads a single record today, because
the managers hold their collections in memory. It is in the interface because a
store missing point reads is not a store.

`kind` separates the key spaces: `TaskMgr` writes `"task"`, `HandoffMgr` writes
`"handoff"` (§7). Keeping the store ignorant of both types is what lets one
implementation serve both managers.

The first implementation is `JsonFileStoreMgr`: one JSON file per record, in a
directory per kind. No database. Chosen for inspectability while the shape of the
data is still settling.

### 4.3 Managers

**A manager owns a collection.** That is the word's basic meaning and every one
of these honours it: a manager holds a set of things of one kind, and offers
add / get / query / remove over that set, plus persistence of it. Behaviour
belonging to a single member is a method on that member, not on its manager.

| Manager | Owns the collection of | Offers | Does *not* |
|---|---|---|---|
| `HandoffMgr` | handoffs, keyed by `HandoffId` | declare, get, query, persist | **Judge validity, or transition anything** (§3.1) — `Handoff.open_next` and `HandoffVersion.seal` do that. Store large payloads inline (§8.2) |
| `TaskMgr` | tasks, keyed by `TaskId` | add, get, all, by-status, remove, persist | Make scheduling decisions. Own the execution *transitions* — `Task.push_execution` / `.close_execution` do that |
| `AgentMgr` | agents, keyed by `AgentId`, plus the spec table | register a spec, instantiate, get by id, list by spec, retire | Run an agent, or know what it does |
| `ResourceMgr` | one named pool's accounting | can_afford, take, give_back | Know about tasks. This is the one that manages a *quantity*, not a set — §3.4 |

`ResourceMgr` is deliberately the exception, and the name is kept because
`mission.md` uses it. What it manages is a counter with two release disciplines,
not a collection.

`AgentMgr` is no longer a bare factory. `mission.md` asks for `get(name) ->
agent`, which stays, but an agent that is instantiated and then forgotten leaves
`Execution.agent_id` pointing at nothing — the audit trail (§3.2) would name
agents the system cannot resolve. So the mgr keeps what it created.

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
    def get(self, name: str) -> Any: ...          # exact; raises KeyError if absent
    def resolve(self, pattern: str) -> list[Any]:  # "resource:*" -> every pool (§6.4)
        ...

# ---- recovery ----  (§6.4)
@runtime_checkable
class Resumable(Protocol):
    def resume_system(self) -> None: ...
    # Implemented by HandoffMgr, TaskMgr, ResourceMgr, Scheduler.
    # Not implemented by AgentMgr, SchedulePolicy, StoreMgr — they hold no state.
    # Named resume_system, not resume: "rebuild yourself from persistence" is a
    # different act from Scheduler.resume_task(tid), and one class needs both.

RESUME_ORDER = ["handoff_mgr", "task_mgr", "resource:*", "scheduler"]

def resume_all(registry: Registry) -> None: ...

# ---- store ----  (registered as "store_mgr", see §4.2)
class StoreMgr(Protocol):
    def create(self, kind: str, key: str, record: dict) -> None: ...
    def read(self, kind: str, key: str) -> dict | None: ...
    def read_all(self, kind: str) -> list[dict]: ...
    def update(self, kind: str, key: str, record: dict) -> None: ...
    def delete(self, kind: str, key: str) -> None: ...
    def exists(self, kind: str, key: str) -> bool: ...

# ---- handoff: the objects own their transitions ----
class HandoffVersion(Model):
    version: int
    status: HandoffStatus
    producer_task_id: TaskId | None
    producer_agent_id: AgentId | None
    ...
    @property
    def is_valid(self) -> bool: ...
    def seal(self, status: HandoffStatus, content: Any = None) -> None:
        """GENERATING -> VALID | INVALID. Raises if not open."""

class Handoff(Model):
    id: HandoffId
    type: str
    versions: list[HandoffVersion]
    @property
    def latest(self) -> HandoffVersion | None: ...
    @property
    def is_latest_valid(self) -> bool: ...
    def get(self, version: int) -> HandoffVersion: ...
    def open_next(self, task_id: TaskId, agent_id: AgentId) -> HandoffVersion:
        """Adopt an untouched CREATED v0, else append v+1. Returns it GENERATING."""

# ---- handoff: the mgr owns the collection ----
# Called by the SCHEDULER (read-only):
class HandoffMgr:
    def declare(self, ids: list[HandoffId], producer_task_id: TaskId,
                types: dict[HandoffId, str] | None = None) -> None:
        """Create each with v0 in state CREATED. Idempotent."""
    def check_if_latest_valid(self, hid: HandoffId) -> bool:
        """`get(hid).is_latest_valid`, False if unknown. A read, not a judgement."""
    def latest(self, hid: HandoffId) -> HandoffVersion | None:
        """For pinning into an Execution record (§3.2)."""
    def get(self, hid: HandoffId) -> Handoff: ...
    def get_many(self, ids: list[HandoffId]) -> list[Handoff]: ...
    def all_ids(self) -> list[HandoffId]: ...
    def produced_by_task(self, tid: TaskId) -> list[HandoffId]: ...
    def resume_system(self) -> None:
        """Reload every handoff from the store, at startup (§6.4)."""

# Called by the AGENT, through its runner (write):
    def persist(self, hid: HandoffId) -> None:
        """Write a handoff back after open_next() or seal() mutated it."""

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

# ---- agent ----
class AgentMgr:
    def register(self, spec: str, **config) -> None:
        """Declare a kind of agent. Specs are the vocabulary Task.agent_spec draws on."""
    def instantiate(self, spec: str, task_id: TaskId) -> Agent:
        """Mint a new Agent with a fresh AgentId, bound to that task, and keep it."""
    def get(self, ref: AgentId | str) -> Agent:
        """By id: that agent. By spec name: the mission.md `get(name) -> agent`."""
    def by_spec(self, spec: str) -> list[Agent]: ...
    def all(self) -> list[Agent]: ...
    def retire(self, aid: AgentId) -> None: ...

# ---- task ----
class TaskMgr:
    def add(self, task: Task) -> None: ...
    def get(self, tid: TaskId) -> Task: ...
    def all(self) -> list[Task]: ...
    def by_status(self, status: TaskStatus) -> list[Task]: ...
    def remove(self, tid: TaskId) -> None: ...
    def persist(self, tid: TaskId) -> None:
        """Write a task back after a caller mutated it through Task's own methods."""
    def resume_system(self) -> None: ...

# ---- runner ----
class TaskRunner(Protocol):
    def start(self, task: Task, agent: Agent, on_done: OnDone) -> None: ...
    def stop(self, task_id: TaskId, on_stopped: Callable[[TaskId], None]) -> None: ...

# on_done(task_id, status, usage) — status is SUCCEEDED or FAILED; usage is
# pool name -> amount spent, for consumable settlement (§6.3). There is no
# result object: everything else the scheduler needs it can read itself.
OnDone = Callable[[TaskId, TaskStatus, dict[str, float]], None]

# ---- policy ----
class SchedulePolicy(Protocol):
    def select(self, eligible: list[Task],
               snapshot: dict[str, float]) -> list[TaskId]: ...
```

The splits are deliberate:

- `can_afford` / `take` are separate so the scheduler can check every declared
  resource before mutating any of them (§6.2).
- The two halves of `HandoffMgr` are separate so the read/write asymmetry of
  §3.1 is visible in the interface: the scheduler calls only the first group,
  agents only the second.
- Transitions are on `Handoff` and `Task`; collection operations are on their
  managers. `HandoffMgr` has no `seal`, and `TaskMgr` no `set_status` — a caller
  mutates the object and asks the mgr to `persist` it. The mgr is not a proxy
  for its members' behaviour.

### 5.1 Scheduler API

| Method | Effect | Rejects |
|---|---|---|
| `submit(task)` | `declare` its output handoffs; place in the pool its inputs dictate; dispatch | Duplicate id; undeclared resource pool |
| `expedite(task)` | `submit`, but requires every input already valid and marks the task for front-of-queue ordering | Any input failing `check_if_latest_valid` |
| `remove_queued(tid)` | `→ CANCELLED` | Task not in a waiting pool |
| `stop(tid)` | `RUNNING → STOPPING`; calls `runner.stop(tid, self.on_stopped)` | Task not `RUNNING` |
| `resume_task(tid)` | `FAILED \| SUSPENDED →` recomputed waiting pool; dispatch. Does not touch handoffs | `SUCCEEDED`, `CANCELLED`, or any live state |
| `update_task(tid, ...)` | Sugar: remove and re-submit under the same id, with new inputs/outputs/resources | Task not queued |
| `on_task_done(tid, outcome)` | Release resources; close the execution record; dispatch | Task not `RUNNING` |
| `on_stopped(tid)` | `STOPPING → SUSPENDED`; release resources; close the execution record | Task not `STOPPING` |
| `resume_system()` | Rebuild the pool index from task status; demote interrupted runs. Called by `resume_all`, last (§6.4) | — |
| `try_dispatch()` | Grant resources and start tasks, as capacity allows | — |

`stop()` records intent and delegates; `on_stopped()` completes the transition
(§3.2). A task in `STOPPING` still holds its resources.

The two resumes are named apart: `resume_task(tid)` puts one task back in a
queue; `resume_system()` rebuilds this component from persistence. They are
different acts, one class needs both, and `@runtime_checkable` matches on method
*name* alone — so a single `resume` would have `resume_all` calling the
task-level one with no argument. `on_stopped` takes the
callback the runner was handed, which is why `TaskRunner.stop` carries one: a
runner that had to resolve `scheduler` from the registry would be the only
component depending on it by name.

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

`declare` creates v0 in state `CREATED` so downstream tasks can reference the id.
It does not open it — the producing agent does that when it starts (§3.1).

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
      agent = registry.get("agent_mgr").instantiate(task.agent_spec, tid)
      task.push_execution(                          # the Task's own transition
            agent_id=agent.id,
            input_versions={h: handoff_mgr.latest(h).version for h in task.inputs},
      )
      task.status = RUNNING
      task_mgr.persist(tid)                         # the mgr's job: durability
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

The runner reports the terminal status and what it spent. Handoff state has
**already been written by the agent** through `open_next` / `seal`; the
scheduler neither sets nor infers it.

```python
on_done(task_id, status: TaskStatus, usage: dict[str, float]) -> None
```

**There is no result object.** Taking the previous `TaskResult` field by field:

| Field | Verdict |
|---|---|
| `ok` / `outcome` | It *was* the task status, one indirection away. `SUCCEEDED` and `FAILED` are already `TaskStatus` members; a boolean or a parallel enum said the same thing in a second vocabulary the scheduler then translated. The runner says the status. |
| `output_versions` | Redundant and less accurate. At completion `handoff_mgr.latest(h)` is the authoritative answer, and the scheduler reads it directly — the same way it obtained `input_versions` at dispatch. |
| `actual_usage` | Kept, as the `usage` argument. Only the runtime knows what was spent, and consumable settlement needs it. |
| `detail` | Belongs on the `Execution` record, which is where a human looks. |

One dict does not need a class. The earlier argument for the struct — that the
scheduler must apply everything as one step — survives, and a single call with
three arguments satisfies it exactly as well.

The cost is that a runner must know `TaskStatus`. It already knows `Task`.

Completion and validity remain independent (§3.1): an agent may finish cleanly
and still seal its output `INVALID`. That is why the runner reports a *task*
status and never touches the handoff. It is also why a boolean was the wrong
shape — "true" reads as "it worked", and whether it worked is not the runner's
to say.

```
on_task_done(tid, status, usage)
  ├─ release lease   (renewable: full; consumable: settle `usage`)
  ├─ output_versions = {h: handoff_mgr.latest(h).version for h in task.outputs}
  ├─ task.close_execution(output_versions, status)     # seals the stack top
  ├─ task.status = status
  ├─ TaskMgr.persist(tid)
  └─ try_dispatch()          # step 1 re-checks every waiter against latest
```

The scheduler **reads** the output versions rather than being told them, exactly
as it read the input versions at dispatch. Symmetry aside, the read is the more
truthful of the two: it reports what the handoff actually holds.

There is still no handoff *manipulation* here. Downstream promotion is not pushed;
it falls out of the re-check in `try_dispatch`.

A task is `SUCCEEDED` when its run completed, whatever verdict its agent sealed.
An agent that ran fine and concluded the output is unusable leaves its consumers
in `WAITING_HANDOFF` — a `SUCCEEDED` task with an `INVALID` output is a legal and
expected state.

A failed task releases its resources and is recorded `FAILED`. Whatever its agent
last wrote to the handoff stands; if the agent opened a version and never sealed
it, that version remains `GENERATING`, and `check_if_latest_valid` is false — so
consumers correctly stay blocked. Downstream tasks are not cancelled. That is
deliberate: the system does not cancel them, and does not pretend to.

### 6.4 Recovery

Recovery is not one component's job. Rebuilding handoff versions, task state,
resource pools, and the scheduler's index are four different reconstructions, and
`Scheduler.resume_system()` doing all four would put a component in charge of
state it does not own (§2, principle 3).

**Each component restores itself.** A separate `resume_all` sequences them.

```python
class Resumable(Protocol):
    def resume(self) -> None:
        """Rebuild this component's own state from whatever it persisted."""
```

Components with nothing to restore — `AgentMgr`, `SchedulePolicy`, `StoreMgr`
itself — simply do not implement it. Not implementing `Resumable` is the
declaration that a component is stateless; an empty `resume_system()` would be
indistinguishable from one somebody forgot to write.

#### Order matters

The sequence is a real dependency, not a convention:

| # | Component | Restores | Depends on |
|---|---|---|---|
| 1 | `HandoffMgr` | handoffs and their versions | nothing |
| 2 | `TaskMgr` | tasks and their execution stacks | nothing |
| 3 | `ResourceMgr` (each) | pools to full capacity — no lease survives a restart | nothing |
| 4 | `Scheduler` | its pool index; demotes interrupted runs | 1, 2, 3 |

The scheduler is last because everything it rebuilds is derived: its index is a
projection of task status (§4.4), and the eligibility it recomputes reads handoff
versions (§3.2). Restoring it against an empty `HandoffMgr` would classify every
waiting task as blocked and then never re-check, because nothing would have
changed to trigger a decision point.

1 through 3 are mutually independent, so the order among them is free; only
"scheduler last" is load-bearing.

```python
RESUME_ORDER = ["handoff_mgr", "task_mgr", "resource:*", "scheduler"]

def resume_all(registry) -> None:
    for pattern in RESUME_ORDER:                    # declared, not discovered
        for component in registry.resolve(pattern): # "resource:*" expands to every pool
            if isinstance(component, Resumable):
                component.resume_system()
```

`Registry.resolve(pattern)` returns one component for a plain name and all
matching ones for a `prefix:*` pattern, in registration order. It is the only
place a wildcard is honoured; `get()` stays exact.

`Resumable` must be `@runtime_checkable` for that `isinstance` to work — it is a
Protocol, and an unmarked one raises at runtime rather than returning `False`.

The order is declared rather than inferred. Inferring it would mean a dependency
graph among managers, which is exactly the machinery the registry exists to avoid;
a four-entry list that fails loudly when a name is missing is the smaller thing.

#### What each restore does

```
HandoffMgr.resume_system()
  └─ load handoffs and versions. A version left GENERATING stays GENERATING:
     its agent is gone and will never seal it, so check_if_latest_valid remains
     false and consumers stay blocked — the same outcome as the crash case in §6.3

TaskMgr.resume_system()
  └─ load tasks with their execution stacks. A dangling stack top (ended_at
     unset) is closed as interrupted

ResourceMgr.resume_system()
  └─ reset to full capacity. Leases do not survive a restart, so nothing is held

Scheduler.resume_system()
  ├─ rebuild the pool index from each task's stored status
  ├─ RUNNING  → WAITING_RESOURCE      (its lease is gone)
  ├─ STOPPING → SUSPENDED             (the runner it was waiting on is gone)
  └─ try_dispatch()                   # eligibility is recomputed, never restored
```

Eligibility needs no reconstruction: it is a query, so recovery only has to
ensure the handoff versions the query reads are present.

`resume_all` is the whole-system entry point. `Scheduler.resume_system()` no
longer exists; calling one component's resume without the others is a partial
recovery and the API should not offer it.

#### Why handoffs persist themselves

`HandoffMgr` restores from its own records, not from task records. The reason is
not a crash window — it is that **the information is nowhere else.**

A run's terminal status and a version's verdict are independent facts (§6.3): an agent may
finish cleanly and still seal its output `INVALID`. A task record carries only
the run outcome. Rebuilding handoff state from task records would therefore mean
guessing `COMPLETED → VALID`, which is wrong in exactly the case §6.3 describes —
not because of restart timing, but because the verdict was never written there at
all. Guessing it would also be the inference §3.1 exists to prevent.

Two further gaps: a handoff supplied externally has no producing task to replay,
and a version abandoned mid-write by a crashed agent appears in no completed
execution record.

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

`HandoffMgr` follows the same write-through discipline: `declare` and `persist`
each write. Because `persist` is called by agents, a handoff is stored at the
moment the agent acts rather than at task completion.

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

Adopted as dependencies:

| Adopted | For | Why |
|---|---|---|
| **pydantic v2** | every domain model in §3 | Already installed — `fastapi` is a repository dependency and pulls it. It supplies validation, `model_dump` / `model_validate` for §7, and enum and `datetime` coercion, all of which would otherwise be hand-written `*_from_dict` constructors with no inverse. The state machines of §3.1 and §3.2 are exactly what a validator should enforce rather than a comment. |
| **`uuid.UUID`** | `TaskId`, `AgentId`, `HandoffId` | Identity generation, comparison, and formatting are a solved problem in the standard library. The three types wrap it so they stay mutually incompatible; a bare `str` makes `list[str]` unreadable and lets a `TaskId` be passed where a `HandoffId` belongs. |

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
| Lease TTL and sweep | A runner that dies while `RUNNING` never reports done, and its resources leak until the next `resume_all`. Deferred, not solved. A TTL plus a periodic sweep is the known fix. |
| Handoff payload storage | §8.2. The interface is left open. |
| Fairness across submitters | Naive FIFO lets one submitter monopolise a pool. A future `SchedulePolicy` keyed by submitter is the reference solution. |
| Better ordering than FIFO | The composite rule from the prior art (priority tier → estimated cost → most-total-successors → FIFO) is a drop-in `SchedulePolicy`. Not built first. |
| Cycle detection | A task whose inputs transitively depend on its own outputs will never run. A ten-line DFS at submit would reject it at the boundary. Not in the first version. |
| **Re-run window** | A new version is opened by the producing agent when it starts writing (§3.1), so between `resume_task(t)` and that moment, `latest` is still the previous version. A downstream task dispatched in that window runs against stale-but-valid content. Accepted as the price of the scheduler never touching handoff state. Mitigations if it bites: have the runner open versions at `start`, or have `resume` mark outputs pending. Neither is built. |
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
5. `stop` → `on_stopped` → `resume_task` returns a task to the correct pool,
   recomputed from its inputs.
6. `resume_task` on a `SUCCEEDED` or `CANCELLED` task is rejected.
7. A failed task releases its resources and leaves its dependents in
   `WAITING_HANDOFF`; an output whose version the agent left `GENERATING` does not
   satisfy `check_if_latest_valid`.
8. `resume_all` reconstructs pools from persisted tasks and versions from
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
13. A run reporting `SUCCEEDED` whose agent sealed its output `INVALID` leaves its
    consumers in `WAITING_HANDOFF` — completion and validity are independent
    (§6.3). The scheduler records `output_versions` by reading `HandoffMgr`, not
    from anything the runner passed.
14. **The scheduler never writes handoff state.** Across a full submit → dispatch →
    complete → resume → re-dispatch cycle, a `HandoffMgr` spy records `persist`
    originating only from the agent, and calls from the scheduler only to
    `declare`, `check_if_latest_valid`, and `latest`. No `open_next` or `seal` is
    called from a scheduler frame.
15. Two isolated `Registry()` instances do not share components, and `get()` on an
    unregistered name raises with that name in the message.
16. Re-running a producer appends a version and leaves earlier ones byte-identical;
    a consumer that already ran retains `input_versions` pointing at the version it
    actually consumed.
17. A consumer dispatched after a producer's re-run reads the new version, with no
    invalidation call made anywhere in the system.
18. Execution history grows by exactly one entry per run, each carrying the bound
    `agent_id` and the pinned input/output versions.
19. A handoff sealed `INVALID` before a restart is still `INVALID` after
    `resume_all`, and one left `GENERATING` is still `GENERATING` — neither is
    re-derived from the producing run's terminal status (§6.4).
20. Resuming a task whose previous run left a `GENERATING` version appends a new
    version rather than reusing the abandoned one.
21. The bound agent is readable only from `history[-1]`: `Task` exposes no
    `agent_id`, and after a `resume_task` the top reports the new run's agent
    while the entry beneath still reports the previous one.
22. **A run is reconstructible from either end.** From the task:
    `history[-1].agent_id` resolves through `AgentMgr`, and its `input_versions`
    keys resolve through `HandoffMgr`. From the handoff: a version's
    `producer_task_id` and `producer_agent_id` both resolve, and that agent's own
    `task_id` and `handoffs` name the same pair back (§3.1).
23. `update_task` produces the same observable state as `remove_queued` followed
    by `submit` with the new arguments — verified by comparing against that
    sequence, not by re-asserting the outcome (§2, principle 7).
24. `resume_all` calls `resume_system()` on every `Resumable` in the declared order and
    skips components that do not implement it, with the scheduler last.
25. Resuming the scheduler against a `HandoffMgr` that has not resumed yet leaves
    every waiting task blocked — the failure the order exists to prevent, asserted
    directly so the ordering constraint is tested rather than assumed.
26. A `HandoffVersion` refuses an illegal transition: `seal` on a `CREATED` one
    and a second `seal` each raise. The state machine is enforced by the object,
    not by whoever calls it.
27. `TaskId`, `AgentId`, and `HandoffId` are mutually incompatible: passing one
    where another is expected is a type error the checker reports, and equality
    across two types with the same underlying UUID is `False`.
28. `AgentMgr` retains what it instantiates: after a run, `get(execution.agent_id)`
    returns that agent with its `task_id` and `handoffs` populated, and `by_spec`
    lists every instance made under a spec name.
29. A store rejects `create` on an existing key and `update` on a missing one, and
    a round-trip through `read` returns a record equal to the one written.
30. `Handoff.open_next` adopts an untouched `CREATED` v0 in place — the list stays
    length 1 — and appends `v+1` on every later call, so version numbers are
    contiguous without anyone checking.
31. `Task.depends_on` supports a topological sort of the submitted graph, and no
    scheduling path reads it: blanking it on every task changes no dispatch order
    and no pool membership.
