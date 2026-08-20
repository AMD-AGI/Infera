# Agent Task Graph — Design

| | |
|---|---|
| Status | Draft, pending review |
| Revision | 1 — 2026-08-20 |
| Implements | `docs/spec.md` rev. 3 |
| Language | Python ≥ 3.10, standard library only |

---

## 1. Scope

This document turns `spec.md` into files, classes, and method bodies. It adds no
requirements. Where it makes a choice the spec left open, the choice is stated
here; where implementing the spec exposed a contradiction in it, §13 says so
rather than papering over it.

The spec's 25 acceptance criteria are the definition of done. §11 maps every one
of them to a named test.

Everything is standard library. There is no third-party dependency other than
`pytest` for the tests, which the repository already carries. §10 records why,
per module, as `mission.md` rule 3 requires.

---

## 2. Layout and import graph

`agent_sys/` *is* the package. `docs/` and `tests/` sit inside it and are not
imported.

```
agent_sys/
├── README.md              build-versus-adopt record (mission rule 3)
├── __init__.py            re-exports the public names
├── core.py                enums + dataclasses. No behaviour, no imports
├── registry.py            Registry, Resumable, RESUME_ORDER, resume_all
├── store.py               StoreMgr Protocol, JsonFileStoreMgr, MemoryStoreMgr
├── handoff.py             HandoffMgr
├── task.py                TaskMgr
├── resource.py            ResourceMgr, RenewableMgr, ConsumableMgr, GpuMgr, TokenMgr
├── agent.py               AgentMgr
├── runner.py              TaskRunner Protocol, FakeRunner
├── policy.py              SchedulePolicy Protocol, FifoPolicy
├── scheduler.py           Scheduler
├── bootstrap.py           build_registry() — the composition root
├── docs/
│   ├── spec.md
│   └── design.md
└── tests/
```

### Import graph

```
                        core.py          (stdlib only)
                           ▲
   ┌─────────┬─────────┬───┴─────┬──────────┬─────────┐
handoff    task     runner    policy   scheduler   agent
   ▲         ▲         ▲         ▲         ▲         ▲
   └─────────┴─────────┴────┬────┴─────────┴─────────┘
                            │
                       bootstrap.py      (the only module that imports managers)

registry.py — imported by the managers for the `Registry` type annotation only
store.py, resource.py — stdlib only, imported by nobody but bootstrap and tests
```

**No manager imports another manager.** A component holds the `Registry` and
resolves collaborators by name at call time. `bootstrap.py` is the single
composition root; it is the only place with a wide import fan-in, which is what a
composition root is for.

`core.py` importing nothing is load-bearing: it is what keeps the graph acyclic
without anyone thinking about it.

---

## 3. Data model — `core.py`

Pure data. Every type is a dataclass or an enum; none has a method that touches
another component.

```python
class TaskStatus(str, Enum):
    WAITING_HANDOFF  = "waiting_handoff"
    WAITING_RESOURCE = "waiting_resource"
    RUNNING          = "running"
    STOPPING         = "stopping"
    SUCCEEDED        = "succeeded"
    FAILED           = "failed"
    SUSPENDED        = "suspended"
    CANCELLED        = "cancelled"

# module level, alongside the enum
WAITING   = frozenset({TaskStatus.WAITING_HANDOFF, TaskStatus.WAITING_RESOURCE})
RESUMABLE = frozenset({TaskStatus.FAILED, TaskStatus.SUSPENDED})

class HandoffStatus(str, Enum):
    GENERATING = "generating"
    VALID      = "valid"
    INVALID    = "invalid"
```

Both enums subclass `str`. This is not cosmetic: `json.dumps` serialises a
`str`-Enum as its value with no encoder hook, which is what lets `store.py` stay
type-ignorant (§5). Python 3.11's `StrEnum` would be the same thing; the repo
targets 3.10, so `(str, Enum)` it is.

These two exist because the guards in `remove_queued`, `update_task`, and
`resume_task` would otherwise repeat the same tuple literal. There is no
`LIVE` or `FINAL` set: every other guard tests a single status, and a constant
with one call site is worse than the literal.

```python
@dataclass
class HandoffVersion:
    version: int
    status: HandoffStatus
    produced_by_agent: str | None
    timestamp: float
    content: Any = None

@dataclass
class Handoff:
    uuid: str
    type: str = ""
    produced_by: str | None = None
    versions: list[HandoffVersion] = field(default_factory=list)

    @property
    def latest(self) -> HandoffVersion | None:
        return self.versions[-1] if self.versions else None

@dataclass
class Execution:
    attempt: int
    agent_uuid: str
    input_versions: dict[str, int] = field(default_factory=dict)
    output_versions: dict[str, int] = field(default_factory=dict)
    started_at: float = 0.0
    ended_at: float | None = None
    outcome: TaskStatus | None = None

@dataclass
class Task:
    id: str
    agent: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    resources: dict[str, float] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.WAITING_HANDOFF
    created_at: float = field(default_factory=time.time)
    expedited: bool = False
    history: list[Execution] = field(default_factory=list)

    @property
    def current(self) -> Execution | None:
        return self.history[-1] if self.history else None

@dataclass
class Agent:
    name: str
    uuid: str
    knowledge: Any = None        # left empty per mission.md
    config: dict = field(default_factory=dict)

@dataclass
class TaskResult:
    ok: bool
    output_versions: dict[str, int] = field(default_factory=dict)
    actual_usage: dict[str, float] = field(default_factory=dict)
```

`Handoff.latest` and `Task.current` are properties, not stored fields — spec §2
principle 7. `Task.current` is the execution stack top: the live binding, not
merely the newest entry.

### Serialisation

`dataclasses.asdict` handles the write side for all of them, including nesting.
The read side needs two hand-written constructors, because `asdict` has no
inverse:

```python
def task_from_dict(d: dict) -> Task
def handoff_from_dict(d: dict) -> Handoff
```

Each is a dozen lines: rebuild the nested list, coerce enum fields via
`TaskStatus(d["status"])`. They live in `core.py` because they are pure data
operations and putting them in `store.py` would give the store knowledge of the
types it is designed not to have.

---

## 4. Registry and recovery — `registry.py`

```python
class Registry:
    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def register(self, name: str, component: Any) -> None:
        self._items[name] = component            # replacement is the swap mechanism

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"no component registered as {name!r}") from None

    def resolve(self, pattern: str) -> list[Any]:
        if pattern.endswith(":*"):
            prefix = pattern[:-1]                # "resource:*" -> "resource:"
            return [c for n, c in self._items.items() if n.startswith(prefix)]
        return [self.get(pattern)]
```

`register` **overwrites deliberately.** Spec §4.1 requires that a test can swap an
implementation after wiring; rejecting duplicates would forbid exactly that. The
protection against typos is `get`'s loud failure, not a registration guard.

`resolve` honours `prefix:*` and nothing else — no globbing, no regex. It exists
for `resource:*` and returns registration order, which `dict` preserves.

```python
@runtime_checkable
class Resumable(Protocol):
    def resume(self) -> None: ...

RESUME_ORDER = ["handoff_mgr", "task_mgr", "resource:*", "scheduler"]

def resume_all(registry: Registry) -> None:
    for pattern in RESUME_ORDER:
        for component in registry.resolve(pattern):
            if isinstance(component, Resumable):
                component.resume()
```

`@runtime_checkable` is mandatory: an unmarked Protocol raises on `isinstance`
rather than returning `False`.

One caveat this creates and the implementation must respect: `isinstance` against
a runtime-checkable Protocol checks *method presence only*. Any component that
happens to have a zero-argument `resume` will be called. That is why
`Scheduler`'s task-level resume is named `resume_task` (§13, D1) — otherwise the
scheduler would satisfy `Resumable` by accident with the wrong method.

---

## 5. Persistence — `store.py`

```python
class StoreMgr(Protocol):
    def save(self, kind: str, key: str, record: dict) -> None: ...
    def load_all(self, kind: str) -> list[dict]: ...
    def delete(self, kind: str, key: str) -> None: ...
```

Two implementations:

```python
class MemoryStoreMgr:
    """A dict of dicts. Survives a manager restart because the store object does."""

class JsonFileStoreMgr:
    """<root>/<kind>/<quoted-key>.json, one file per record."""
```

`JsonFileStoreMgr.save` writes to `<name>.json.tmp` and calls `Path.replace`,
which is atomic on POSIX. That gives per-record atomicity for free. It does *not*
give cross-record or cross-manager atomicity — spec §7 says so explicitly and
§10 leaves it open.

Keys are task ids and handoff uuids, used directly as filenames, so they go
through `urllib.parse.quote(key, safe="")`. Callers pass opaque strings and
should not have to know they become paths.

`MemoryStoreMgr` is the default in tests. Recovery is still testable with it:
"restart" means constructing fresh managers over the *same* store object, which
is precisely what recovery does. `test_store.py` and one end-to-end case in
`test_recovery.py` exercise the JSON implementation against `tmp_path`.

### On-disk shape

```
<root>/
├── task/
│   └── t1.json      {"id": "t1", "agent": "profiler", "status": "running",
│                     "history": [{"attempt": 0, "agent_uuid": "...", ...}], ...}
└── handoff/
    └── h1.json      {"uuid": "h1", "produced_by": "t1",
                      "versions": [{"version": 0, "status": "valid", ...}]}
```

Directly readable, which is the whole reason for choosing files over sqlite while
the shape of the data is still settling.

---

## 6. Managers

### 6.1 `HandoffMgr` — `handoff.py`

Holds `dict[str, Handoff]`. Every write persists under kind `"handoff"`.

```python
class HandoffMgr:
    def __init__(self, registry: Registry) -> None:
        self._r = registry
        self._handoffs: dict[str, Handoff] = {}

    # ---- scheduler-facing: read only ----
    def declare(self, uuids, produced_by, types=None) -> None:
        for u in uuids:
            if u in self._handoffs:              # idempotent — see D6
                continue
            self._handoffs[u] = Handoff(uuid=u, produced_by=produced_by,
                                        type=(types or {}).get(u, ""))
            self._persist(u)

    def check_if_latest_valid(self, uuid) -> bool:
        h = self._handoffs.get(uuid)
        return h is not None and h.latest is not None \
               and h.latest.status is HandoffStatus.VALID

    def latest_version(self, uuid) -> int | None:
        h = self._handoffs.get(uuid)
        return h.latest.version if h and h.latest else None

    def get(self, uuid) -> Handoff: ...          # raises KeyError

    # ---- agent-facing: write ----
    def new_version(self, uuid, agent_uuid) -> int:
        h = self.get(uuid)
        v = HandoffVersion(version=len(h.versions), status=HandoffStatus.GENERATING,
                           produced_by_agent=agent_uuid, timestamp=time.time())
        h.versions.append(v)
        self._persist(uuid)
        return v.version

    def record(self, uuid, version, status, content=None) -> None:
        v = self.get(uuid).versions[version]
        if v.status is not HandoffStatus.GENERATING:
            raise ValueError(f"{uuid} v{version} is already sealed as {v.status}")
        v.status, v.content = status, content
        self._persist(uuid)

    def resume(self) -> None:
        self._handoffs = {d["uuid"]: handoff_from_dict(d)
                          for d in self._r.get("store_mgr").load_all("handoff")}
```

Three decisions worth stating:

**`check_if_latest_valid` on an unknown uuid returns `False`, not an exception.**
A consumer submitted before its producer references a uuid nobody has declared
yet. `False` — "not ready" — is the correct answer and keeps `submit` ordering
free (D5).

**`declare` is idempotent.** `update_task` is `remove_queued` + `submit`, so
`declare` runs twice for the same outputs. Overwriting would destroy versions an
agent had already written (D6).

**`new_version` numbers by `len(versions)`,** never by a stored counter. The list
is the counter. One fact, one place.

`HandoffMgr` has no validation logic and never sets `VALID`/`INVALID` itself.
That is the whole content of spec §3.1, and criterion 14 tests it with a spy.

### 6.2 `TaskMgr` — `task.py`

Holds `dict[str, Task]`. Every mutator persists under kind `"task"`.

| Method | Effect |
|---|---|
| `add(task)` | Store and persist. Raises if the id exists and is not `CANCELLED` (D3) |
| `get(tid)` / `all()` | Read |
| `set_status(tid, status)` | Write status, persist |
| `push_execution(tid, agent_uuid, input_versions)` | Append an `Execution` with `attempt=len(history)`, `started_at=now`, `ended_at=None` |
| `close_execution(tid, output_versions, outcome)` | Seal `history[-1]`: set `output_versions`, `ended_at`, `outcome` |
| `resume()` | Reload from the store; close any dangling stack top |

```python
def resume(self) -> None:
    self._tasks = {d["id"]: task_from_dict(d)
                   for d in self._r.get("store_mgr").load_all("task")}
    for t in self._tasks.values():
        e = t.current
        if e is not None and e.ended_at is None:      # interrupted by the restart
            e.ended_at, e.outcome = time.time(), TaskStatus.SUSPENDED
            self._persist(t.id)
```

The interrupted attempt is closed as `SUSPENDED` — the attempt was cut short, not
failed on its merits (D8). The *task's* landing state is a separate decision made
by `Scheduler.resume()` a moment later, and it is `WAITING_RESOURCE`: a new
attempt will be pushed on top.

`push_execution` is the act of binding an agent. There is no `Task.agent_uuid` to
assign — spec §3.2.

### 6.3 `ResourceMgr` — `resource.py`

The one place inheritance appears (spec §2 principle 1).

```python
class ResourceMgr(ABC):
    def __init__(self, name: str, capacity: float) -> None:
        self.name, self.capacity, self.available = name, capacity, capacity

    @abstractmethod
    def can_afford(self, amount: float) -> bool: ...
    @abstractmethod
    def take(self, amount: float) -> None: ...
    @abstractmethod
    def give_back(self, amount: float, actual: float | None = None) -> None: ...

    def resume(self) -> None:
        self.available = self.capacity           # no lease survives a restart

class RenewableMgr(ResourceMgr):
    def can_afford(self, amount): return amount <= self.available
    def take(self, amount):
        if amount > self.available:
            raise ValueError(f"{self.name}: cannot take {amount} of {self.available}")
        self.available -= amount
    def give_back(self, amount, actual=None):
        self.available += amount                 # `actual` is meaningless here

class ConsumableMgr(ResourceMgr):
    # can_afford / take identical to renewable; only release differs
    def give_back(self, amount, actual=None):
        spent = amount if actual is None else min(actual, amount)
        self.available += amount - spent         # the reservation, less what was used

class GpuMgr(RenewableMgr):    # name defaults to "gpu"
class TokenMgr(ConsumableMgr): # name defaults to "token"
```

`GpuMgr` and `TokenMgr` add nothing today beyond a default name. They exist
because `mission.md` names them and because GPU-specific accounting (topology,
per-node pools) has an obvious home when it arrives. That they are currently
empty is the honest state, not an oversight.

`actual=None` on a consumable means "consumed everything reserved" — the
conservative reading for a budget. `on_stopped` relies on it (D7).

**`ConsumableMgr.resume()` refilling to capacity is wrong and is inherited from
the spec.** See §14, O1. It is implemented as specified and flagged, not silently
fixed.

### 6.4 `AgentMgr` — `agent.py`

```python
class AgentMgr:
    def __init__(self) -> None:
        self._specs: dict[str, dict] = {}

    def register(self, name: str, **config) -> None:
        self._specs[name] = config

    def get(self, name: str) -> Agent:
        if name not in self._specs:
            raise KeyError(f"no agent named {name!r}")
        return Agent(name=name, uuid=str(uuid4()), config=dict(self._specs[name]))
```

**`get` mints a fresh uuid on every call.** This is forced by criterion 21: after
a resume, `history[-1].agent_uuid` must differ from the entry beneath it. A
cached singleton would make every attempt report the same agent and the execution
history would stop being an audit trail (D4).

`knowledge` and `config` stay empty per `mission.md`. `AgentMgr` is deliberately
trivial and holds no state, so it does not implement `Resumable`.

---

## 7. Runner and policy

### 7.1 `TaskRunner` — `runner.py`

```python
class TaskRunner(Protocol):
    def start(self, task: Task, agent: Agent,
              on_done: Callable[[str, TaskResult], None]) -> None: ...
    def stop(self, task_id: str,
             on_stopped: Callable[[str], None]) -> None: ...
```

`stop` takes a callback, symmetric with `start` (D2). The alternative — the
runner resolving `scheduler` from the registry — would make the runner the one
component that knows the scheduler by name, and would invert the dependency the
whole design keeps one-way.

```python
class FakeRunner:
    """Records what was started. The test drives completion explicitly."""
    def start(self, task, agent, on_done):
        self.running[task.id] = (task, agent, on_done)
    def finish(self, task_id, result: TaskResult) -> None:
        _, _, on_done = self.running.pop(task_id)
        on_done(task_id, result)
    def stop(self, task_id, on_stopped):
        self.stop_requested.append(task_id)
        self._acks[task_id] = on_stopped
    def ack_stop(self, task_id) -> None:
        self.running.pop(task_id, None)
        self._acks.pop(task_id)(task_id)
```

`FakeRunner` never calls `on_done` from inside `start`. Tests stay deterministic,
and dispatch is not re-entered on the common path. Re-entrancy is still handled
(§9) because a real synchronous runner is a perfectly reasonable implementation
and must not deadlock or recurse.

The fake also exposes an `agent_writes(...)` helper that performs the
`new_version` → `record` pair on the agent's behalf, so tests can express "the
agent produced a valid output" without a real agent. This is the only place a
test writes handoff state, which is what makes the criterion-14 spy meaningful.

### 7.2 `SchedulePolicy` — `policy.py`

```python
class SchedulePolicy(Protocol):
    def select(self, eligible: list[Task],
               snapshot: dict[str, float]) -> list[str]: ...

class FifoPolicy:
    def select(self, eligible, snapshot):
        return [t.id for t in sorted(eligible,
                                     key=lambda t: (not t.expedited, t.created_at))]
```

`not t.expedited` sorts `False` (expedited) before `True`. Expedited tasks go
first, then submission order — that is the whole of `expedite`'s effect on
ordering, and it lives in the policy rather than in the scheduler, where a
replacement policy is free to ignore it.

`snapshot` is unused by FIFO. It is in the signature because a cost- or
fit-aware policy needs it and changing the signature later would break every
implementation.

---

## 8. Scheduler — `scheduler.py`

```python
class Scheduler:
    def __init__(self, registry: Registry) -> None:
        self._r = registry
        self.pools: dict[TaskStatus, set[str]] = {s: set() for s in TaskStatus}
        self._lock = threading.RLock()
        self._in_dispatch = False
        self._dispatch_again = False
```

One bucket per status, so a task is in exactly one pool. Only three are
load-bearing for scheduling; the rest make "which tasks are suspended" a lookup.

### 8.1 The single writer

```python
def _move(self, tid: str, status: TaskStatus) -> None:
    task_mgr = self._r.get("task_mgr")
    for pool in self.pools.values():
        pool.discard(tid)
    self.pools[status].add(tid)
    if task_mgr.get(tid).status is not status:
        task_mgr.set_status(tid, status)         # persists
```

Every transition in the system goes through this method, and nothing else writes
`pools` or calls `set_status`. Criterion 12 — the index never disagrees with
`TaskMgr` — holds by construction because there is exactly one writer.

Discarding from all eight pools rather than from the task's recorded status makes
`_move` idempotent and self-healing: it cannot leave a stale entry behind even if
called on a task whose stored status was already wrong.

### 8.2 The API

| Method | Body |
|---|---|
| `submit(task)` | validate id and resource names → `task_mgr.add` → `handoff_mgr.declare(task.outputs, task.id)` → `_move` to the pool `_ready` dictates → `try_dispatch` |
| `expedite(task)` | reject unless every input passes `check_if_latest_valid` → `task.expedited = True` → `submit(task)` |
| `remove_queued(tid)` | reject unless status in `WAITING` → `_move(CANCELLED)` |
| `stop(tid)` | reject unless `RUNNING` → `_move(STOPPING)` → `runner.stop(tid, self.on_stopped)` |
| `resume_task(tid)` | reject unless status in `RESUMABLE` → `_move` to the recomputed waiting pool → `try_dispatch` |
| `update_task(tid, **fields)` | `remove_queued(tid)` → `submit(replace(old, **fields, status=WAITING_HANDOFF, history=[]))` |
| `on_task_done(tid, result)` | reject unless `RUNNING` → release → `close_execution` → `_move(SUCCEEDED\|FAILED)` → `try_dispatch` |
| `on_stopped(tid)` | reject unless `STOPPING` → release (consumables at full reservation) → `close_execution(..., SUSPENDED)` → `_move(SUSPENDED)` → `try_dispatch` |
| `resume()` | `Resumable`: rebuild the index, demote interrupted runs, `try_dispatch` |
| `try_dispatch()` | §8.3 |

`update_task` is written as literally those two calls, not as an equivalent
reimplementation. Criterion 23 then holds by construction rather than by
assertion — spec §2 principle 7.

`dataclasses.replace` preserves `created_at`, so an update does not cost a task
its place in FIFO order. Criterion 23 compares the two arms field by field with
`created_at` excluded, since the manually re-submitted task is constructed later
and will differ there and only there.

`resume_task` is *not* named `resume`; §13 D1 explains why the spec's naming
cannot stand.

### 8.3 Dispatch

```python
def try_dispatch(self) -> None:
    with self._lock:
        if self._in_dispatch:                 # re-entered from a synchronous runner
            self._dispatch_again = True
            return
        self._in_dispatch = True
        try:
            while True:
                self._dispatch_pass()
                if not self._dispatch_again:
                    return
                self._dispatch_again = False
        finally:
            self._in_dispatch = False

def _dispatch_pass(self) -> None:
    handoff_mgr, task_mgr = self._r.get("handoff_mgr"), self._r.get("task_mgr")

    # 1. re-check eligibility. Snapshot: _move mutates the sets being scanned.
    for tid in list(self.pools[WAITING_HANDOFF] | self.pools[WAITING_RESOURCE]):
        task = task_mgr.get(tid)
        self._move(tid, WAITING_RESOURCE if self._ready(task) else WAITING_HANDOFF)

    # 2. order the eligible set — the one scheduling decision in the system
    eligible = [task_mgr.get(t) for t in self.pools[WAITING_RESOURCE]]
    for tid in self._r.get("policy").select(eligible, self._snapshot()):
        task = task_mgr.get(tid)
        if task.status is not TaskStatus.WAITING_RESOURCE:
            continue            # moved since selection — see §9, case 1

        # 3. all-or-nothing: verify the FULL set before mutating anything
        pools = {r: self._r.get(f"resource:{r}") for r in task.resources}
        if not all(pools[r].can_afford(n) for r, n in task.resources.items()):
            continue                                     # take nothing; stay queued
        for r, n in task.resources.items():
            pools[r].take(n)

        # 4. bind an agent by PUSHING a record; the stack top is the binding
        agent = self._r.get("agent_mgr").get(task.agent)
        task_mgr.push_execution(
            tid, agent_uuid=agent.uuid,
            input_versions={h: handoff_mgr.latest_version(h) for h in task.inputs},
        )
        self._move(tid, RUNNING)
        self._r.get("runner").start(task, agent, on_done=self.on_task_done)
```

The `list(...)` in step 1 is not defensive style: `_move` mutates the two sets
being iterated, and without the snapshot this raises `RuntimeError`.

Step 3 verifies before mutating. A task that does not fit takes nothing, so no
queued task ever holds a resource, so hold-and-wait deadlock is structurally
impossible. The cost is one extra loop over a dict that has at most a handful of
entries.

Step 4 pins input versions *before* the run starts. The history then records what
the run actually saw, and a later re-run of an upstream producer cannot
retroactively rewrite it.

The status re-check in the loop guards the one way the ordered list can go stale
between selection and use: a synchronous runner that completes inside `start`
re-enters the scheduler and may move a *later* entry of the same list (§9).
Selecting once and dispatching from a snapshot is what keeps the policy call
cheap; the re-check is what makes that safe.

`_ready(task)` is `all(check_if_latest_valid(h) for h in task.inputs)` — the query
that replaces a dependency counter. `_snapshot()` is
`{m.name: m.available for m in self._r.resolve("resource:*")}`, passed to the
policy so a future cost-aware implementation has what it needs; FIFO ignores it.

### 8.4 Recovery

```python
def resume(self) -> None:
    self.pools = {s: set() for s in TaskStatus}
    for task in self._r.get("task_mgr").all():
        status = {TaskStatus.RUNNING:  TaskStatus.WAITING_RESOURCE,  # lease is gone
                  TaskStatus.STOPPING: TaskStatus.SUSPENDED,         # runner is gone
                  }.get(task.status, task.status)
        self._move(task.id, status)
    self.try_dispatch()
```

Eligibility is not restored — it is recomputed by `try_dispatch`, which is why
`HandoffMgr` must have resumed first. Criterion 25 asserts that failure directly:
resume the scheduler against an unresumed `HandoffMgr` and every waiting task
stays blocked, with no later event to unblock it.

### 8.5 `bootstrap.py` — the composition root

```python
def build_registry(store=None, runner=None, policy=None, resources=None) -> Registry:
    r = Registry()
    r.register("store_mgr",  store  or MemoryStoreMgr())
    r.register("handoff_mgr", HandoffMgr(r))
    r.register("task_mgr",    TaskMgr(r))
    r.register("agent_mgr",   AgentMgr())
    r.register("runner",      runner or FakeRunner())
    r.register("policy",      policy or FifoPolicy())
    for m in resources or [GpuMgr(capacity=8), TokenMgr(capacity=1_000_000)]:
        r.register(f"resource:{m.name}", m)
    r.register("scheduler",   Scheduler(r))
    return r
```

Registration order is free — components resolve at use time, not construction
time (spec §4.1) — so this reads top-down for a human rather than being
constrained by dependencies. Every default is overridable, which is how a test
substitutes `FakeRunner`, a spy `HandoffMgr`, or a different policy.

This is the only module that imports every manager. Nothing imports it except
tests and an eventual entry point.

---

## 9. Concurrency

`mission.md` asks for the most optimistic simple implementation and no extra code
for atomicity. The design is therefore:

**Single-threaded by default.** All mutation happens on the caller's thread. The
`FakeRunner` never calls back spontaneously.

**One `threading.RLock` in `Scheduler`, and nothing else.** It guards
`try_dispatch` and covers the two ways a real runner can violate the
single-thread assumption:

| Case | Behaviour |
|---|---|
| A synchronous runner calls `on_done` from inside `start` | Same thread, so the `RLock` is re-entrant and does not deadlock. `_in_dispatch` is already set, so the nested call sets `_dispatch_again` and returns; the outer loop makes another pass. No recursion. |
| An async runner calls `on_done` from its own thread | Blocks on the lock until the current pass finishes. |

The `_dispatch_again` flag is what makes case 1 a loop instead of a stack. Without
it, a synchronous runner would recurse once per dispatched task and a long queue
would exhaust the stack.

The managers are not individually locked. Their mutations happen under the
scheduler's lock in every path the scheduler drives; an agent writing handoffs
from its own thread races only with other agents on the same handoff, which is a
contradiction in the model (one producer per handoff) rather than a case to
defend against.

---

## 10. Build versus adopt

`mission.md` rule 3 requires recording, per module, which library was chosen and
why. `README.md` carries the same table for readers who never open `docs/`.

| Module | Considered | Chosen | Why |
|---|---|---|---|
| `core` | `pydantic`, `msgspec`, `attrs` | stdlib `dataclasses` + `enum` | No validation or coercion is required — these are internal records, not a wire format. `msgspec` is already a repository dependency and would be the drop-in if serialisation ever shows up in a profile; adopting it now would couple `agent_sys` to `infera`'s dependency set for no measured gain. |
| `registry` | `dependency-injector`, `pluggy`, `punq` | stdlib `dict` | Every candidate is built around constructor injection, which spec §4.1 explicitly rejects in favour of resolve-at-use-time. Their remaining feature — a name→instance map — is nine lines. |
| `store` | `sqlite3` (stdlib), `shelve` (stdlib), `tinydb`, `diskcache` | `json` + `pathlib` | The user asked for filesystem + JSON. Records are inspectable with `cat`, which matters while the schema is still moving. `Path.replace` gives per-record atomicity. **`sqlite3` is the named upgrade path** — it is stdlib, and it would supply the cross-manager transaction §14 O2 wants — but it hides the data behind a client and buys nothing else today. The `StoreMgr` Protocol exists so that swap is a one-file change. |
| `handoff` | content-addressed stores (git, DVC, S3) | own implementation | Versioning here is metadata bookkeeping, not content storage. Where payloads live is deliberately open (spec §8.2); when it is decided, a content store plugs in behind `HandoffVersion.content` without touching this module. |
| `task` | — | own implementation | A dict with write-through. Nothing to adopt. |
| `resource` | `threading.Semaphore`, Prefect global concurrency limits | own implementation | A semaphore cannot express the consumable (reserve-then-settle) half, and cannot do the all-or-nothing multi-pool acquisition of §8.3 without a second layer on top. Prefect's limits do exactly what is wanted but live server-side; adopting a server for one primitive is the trade spec §9 rejected. |
| `agent` | any agent framework | own implementation | Two methods, both trivial. `mission.md` leaves agent internals empty on purpose. |
| `runner` | Claude Code / Codex / Cursor CLIs, `subprocess` | Protocol + a fake | The real implementations are harness-specific and out of scope (spec §1.2). What this system owes is the seam. |
| `policy` | `graphlib.TopologicalSorter`, `networkx`, OR-Tools | `sorted()` | Spec §9 records the rejections. FIFO is one `sorted` call; the composite priority rule from the prior art is a drop-in replacement behind the same Protocol. |
| `scheduler` | Prefect, Hatchet, Temporal, Ray, Airflow, Slurm | own implementation | Spec §9, from the prior-art survey. Every candidate is a platform whose scheduling core is not separable. |
| tests | — | `pytest` | Already a dev dependency of the repository. |

The short version: **nothing here has a mature off-the-shelf answer at the size
this system needs it.** Every candidate is either a platform (adopt the server to
get the primitive) or a library for a problem this system does not have (graph
traversal, dependency injection, schema validation). The two genuine upgrade
paths — `sqlite3` for the store, `msgspec` for serialisation — are both recorded
above and both sit behind an interface that already exists.

---

## 11. Test plan

`pytest`. Run with `pytest agent_sys/tests` from the repository root. The root
`testpaths = ["tests"]` only supplies a default when no path is given, so a bare
`pytest` still collects exactly the existing suite and nothing changes for it.

`agent_sys/tests/` gets an `__init__.py`, which makes it a package and keeps
pytest's `prepend` import mode from putting the test directory on `sys.path`
ahead of the repository root — the root is what `import agent_sys` needs.

Two other repository facts, checked rather than assumed:
`[tool.setuptools.packages.find] include = ["infera*"]`, so `agent_sys` is not
packaged and needs no change there; `norecursedirs` does not exclude it.

Every test builds its own `Registry` via `bootstrap.build_registry(...)` with a
`MemoryStoreMgr` and a `FakeRunner`. Nothing is process-global.

| File | Covers | Criteria |
|---|---|---|
| `test_registry.py` | isolation, loud failure, `resolve` wildcard, replacement | 15 |
| `test_store.py` | JSON round-trip, key quoting, atomic replace, `delete` | — |
| `test_resource.py` | renewable vs consumable release, settle-at-actual | 4 |
| `test_handoff.py` | declare/idempotence, version append, seal-once, `check_if_latest_valid` on unknown/empty/generating/valid | 16 (part) |
| `test_task.py` | execution push/close, attempt numbering, write-through | 18 |
| `test_policy.py` | FIFO order, expedited first, swap changes order only | 10 |
| `test_submit.py` | landing pool from input state, undeclared-resource rejection | 1, 2 |
| `test_dispatch.py` | all-or-nothing, agent binding, pinned input versions | 3, 18 |
| `test_lifecycle.py` | stop → on_stopped → resume_task, rejections, `update_task` | 5, 6, 11, 23 |
| `test_completion.py` | resource release, `ok` vs validity independence, failure path | 4, 7, 13 |
| `test_expedite.py` | ordering ahead of earlier tasks, rejection on invalid input | 9 |
| `test_versioning.py` | append not overwrite, earlier versions untouched, consumer sees new version, no invalidation call | 16, 17, 20 |
| `test_linkage.py` | uuid links resolve from either end; agent readable only from `history[-1]` | 21, 22 |
| `test_authority.py` | the `HandoffMgr` spy: scheduler reads only | 14 |
| `test_invariants.py` | pools vs `TaskMgr`, after arbitrary operation sequences | 12 |
| `test_recovery.py` | `resume_all` order, skipping non-`Resumable`, demotion, verdicts not re-derived, the scheduler-first failure | 8, 19, 24, 25 |

Two of these carry more weight than their size suggests:

**`test_authority.py`** registers a `HandoffMgr` subclass that appends every call
to a log, then drives a full submit → dispatch → complete → resume → re-dispatch
cycle in which the test itself makes the agent's writes. Each write is bracketed
by a marker the test pushes into the same log, so "who called this" is recorded
rather than inferred from the call stack. The assertion: every `new_version` and
`record` entry falls between a pair of markers, and every entry outside them is
one of `declare`, `check_if_latest_valid`, `latest_version`. This is the one
mechanical check that spec §3.1's authority boundary has not eroded.

**`test_invariants.py`** runs a fixed sequence of a few dozen operations and, after
each, asserts that the union of the pools equals the set of all task ids and that
each task sits in the pool matching its stored status. It is the check that `_move`
really is the only writer.

Criterion 25 deserves its own note: it asserts a *failure*. Resume the scheduler
before `HandoffMgr` and every waiting task stays in `WAITING_HANDOFF` with no
subsequent event to release it. Without this test the ordering in `RESUME_ORDER`
is a comment.

---

## 12. Implementation order

Test first, in dependency order. Each step is independently runnable and green
before the next begins.

| # | Module | Depends on |
|---|---|---|
| 1 | `core` | — |
| 2 | `registry` | — |
| 3 | `store` | `core` (for the round-trip test only) |
| 4 | `resource` | — |
| 5 | `handoff` | 1–3 |
| 6 | `task` | 1–3 |
| 7 | `agent`, `runner`, `policy` | 1 |
| 8 | `bootstrap` | 1–7 |
| 9 | `scheduler` — submit and dispatch | 1–8 |
| 10 | `scheduler` — lifecycle, completion, expedite | 9 |
| 11 | `resume_all` and recovery | 9, 10 |

Steps 1–8 are small enough that the interesting work is entirely in 9–11. That is
the intent: the components are dull so the scheduler can be read in one sitting.

Before step 1, per `mission.md` rule 5 and the global working rules: back up the
project `CLAUDE.md`, write a fresh one, and create the scratch workspace. No
temporary experiment leaves it.

---

## 13. Deviations from the spec

Each of these is a place where implementing the spec literally does not work.
None changes an acceptance criterion; several are forced *by* one.

| # | Spec says | Design does | Why |
|---|---|---|---|
| D1 | §5.1 lists both `resume(tid)` and `resume()` on `Scheduler` | task-level resume is `resume_task(tid)`; `resume()` is the `Resumable` one | Python cannot have two methods of one name. Worse, `resume(tid)` would satisfy `@runtime_checkable Resumable` by name alone, so `resume_all` would call it with no argument and raise. This is a genuine spec defect, not a style choice. |
| D2 | `TaskRunner.stop(task_id)` | `stop(task_id, on_stopped)` | Otherwise the runner must resolve `scheduler` from the registry — the only component that would need to. The callback is symmetric with `start(..., on_done=)`. |
| D3 | `submit` rejects a duplicate id | rejects an id that exists and is **not** `CANCELLED` | Forced by criterion 23. `update_task` must be `remove_queued` + `submit` under the same id, and `remove_queued` leaves the task `CANCELLED`. Reviving a cancelled id replaces the record with fresh history. |
| D4 | `AgentMgr.get(name) -> Agent` | returns a **new** `Agent` with a fresh uuid on every call | Forced by criterion 21: after a resume the stack top must report a different `agent_uuid` than the entry beneath it. |
| D5 | `check_if_latest_valid(uuid)` | returns `False` for an unknown uuid rather than raising | A consumer may be submitted before its producer declares the handoff. "Not ready" is the right answer and it keeps submission order unconstrained. |
| D6 | `declare(uuids, produced_by)` | idempotent; skips uuids already known | `update_task` re-declares. Overwriting would delete versions an agent had already written. |
| D7 | `on_stopped` releases resources | consumables settle at the **full reservation** | `on_stopped` carries no `TaskResult`, so actual usage is unknown. Assuming the agent spent what it reserved is the safe direction for a budget. |
| D8 | "a dangling stack top is closed as interrupted" | `outcome = SUSPENDED`, `ended_at = now` | The spec does not name the outcome. `SUSPENDED` says the attempt was cut short rather than judged. |
| E1 | `declare(uuids, produced_by)` | optional third argument `types: dict[str, str]` | Additive, and the only write path `Handoff.type` has. The scheduler does not use it — it passes nothing, because `Task` carries no type map (O4). It exists for a caller that declares an externally-supplied handoff directly. |
| E2 | §5.1 `on_stopped` only releases and transitions | also calls `close_execution(..., outcome=SUSPENDED)` | A run is in progress exactly when `history[-1].ended_at is None` (spec §3.2). Leaving the stack top open on a stopped task would make that predicate lie, and the next `resume_task` would push a second open record on top of the first. |

---

## 14. New open questions

These are found by this design and are **not** in spec §10.

| # | Question |
|---|---|
| **O1** | **Consumable budgets do not survive a restart.** Spec §6.4 has `ResourceMgr.resume()` reset to full capacity, and §7 persists only tasks and handoffs. For a renewable pool that is exactly right — no lease survives. For a consumable one it resurrects every token ever spent, which is a correctness hole in one of the two mandated resource types. The fix is small: `ConsumableMgr` persists `available` under a third kind, `"resource"`, and `resume()` reads it back. It needs a spec change to §7 ("two things are persisted") and it keeps recovery step 3 independent of steps 1 and 2. **Recommended, pending review.** Implemented as specified until then. |
| **O2** | A completion that arrives after `stop()` **raises into the runner**. `on_task_done` rejects a non-`RUNNING` task per spec §5.1, so a runner that finishes in the window between `stop()` and its own acknowledgement gets an exception where it expected a callback — and the finished work is discarded, since the task then goes `SUSPENDED`. Nothing leaks, because `on_stopped` still releases the resources. Two candidate fixes: accept `STOPPING` in `on_task_done` and treat the run as complete, or make the rejection a no-op return rather than a raise. The first is better and changes the state machine. Neither is done. |
| **O3** | `Task.resources` names pools that must already be registered, and `submit` rejects unknown ones. Nothing yet rejects a task whose declared amount exceeds a pool's total capacity: it is accepted and waits in `WAITING_RESOURCE` forever. A one-line check at submit would surface it immediately. Not built. |
| **O4** | **`Handoff.type` has no route from `submit`.** Spec §3.1 gives a handoff a `type`, but `Task` (§3.2) has no field naming the types of its outputs, so the scheduler declares them with `type=""` and nothing later fills it in. The field is dead on the scheduler path. Either `Task` gains an `output_types: dict[str, str]` — a §3.2 change — or `type` is acknowledged as being for directly-declared external handoffs only. Nothing depends on it today: the scheduler is content-agnostic and never reads it. |
