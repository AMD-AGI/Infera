# Agent Task Graph — Design

| | |
|---|---|
| Status | Draft, pending review |
| Revision | 2 — 2026-08-20. Code under `src/`; pydantic models; typed ids; behaviour on the objects |
| Implements | `docs/spec.md` rev. 4 |
| Language | Python ≥ 3.10. Standard library plus pydantic v2 |

---

## 1. Scope

This document turns `spec.md` into files, classes, and method bodies. It adds no
requirements. Where it makes a choice the spec left open, the choice is stated
here; where implementing the spec exposed a contradiction in it, §13 says so
rather than papering over it.

The spec's 29 acceptance criteria are the definition of done. §11 maps every one
of them to a named test.

The only runtime dependency is **pydantic v2**, which the repository already
installs — `fastapi` pulls it. §10 records why, per module, as `mission.md`
rule 3 requires.

---

## 2. Layout and import graph

All code lives under `src/`. `docs/` and `tests/` are siblings of it, not
children, and nothing importable sits at the `agent_sys/` top level.

```
agent_sys/
├── README.md              build-versus-adopt record (mission rule 3)
├── src/
│   └── agent_sys/         the importable package
│       ├── __init__.py    re-exports the public names
│       ├── ids.py         TaskId, AgentId, HandoffId
│       ├── models.py      Handoff, Task, Execution, Agent, TaskOutcome + enums
│       ├── registry.py    Registry, Resumable, RESUME_ORDER, resume_all
│       ├── store.py       StoreMgr Protocol, JsonFileStoreMgr, MemoryStoreMgr
│       ├── handoff.py     HandoffMgr
│       ├── task.py        TaskMgr
│       ├── resource.py    ResourceMgr, RenewableMgr, ConsumableMgr, GpuMgr, TokenMgr
│       ├── agent.py       AgentMgr
│       ├── runner.py      TaskRunner Protocol, FakeRunner
│       ├── policy.py      SchedulePolicy Protocol, FifoPolicy
│       ├── scheduler.py   Scheduler
│       └── bootstrap.py   build_registry() — the composition root
├── docs/
│   ├── spec.md
│   └── design.md
└── tests/
```

The `src/` layout is the packaging default for good reason: a test can only
import `agent_sys` if it is genuinely installed or on the path, never by
accident of the working directory. `conftest.py` at `agent_sys/` adds
`src/` to `sys.path`, which is the two-line version of an editable install and
avoids touching the repository's `pyproject.toml` while this is unreleased.

### Import graph

```
                     ids.py             (uuid only)
                        ▲
                    models.py           (pydantic + ids)
                        ▲
   ┌─────────┬─────────┬┴────────┬──────────┬─────────┐
handoff    task     runner    policy   scheduler   agent
   ▲         ▲         ▲         ▲         ▲         ▲
   └─────────┴─────────┴────┬────┴─────────┴─────────┘
                            │
                      bootstrap.py      (the only module that imports managers)

registry.py — imported by the managers for the `Registry` type annotation only
store.py, resource.py — imported by nobody but bootstrap and tests
```

`ids.py` is separate from `models.py` because everything imports the ids and
almost nothing needs to import the models — the managers deal in ids. Splitting
them keeps that visible in the import lines.

**No manager imports another manager.** A component holds the `Registry` and
resolves collaborators by name at call time. `bootstrap.py` is the single
composition root; it is the only place with a wide import fan-in, which is what a
composition root is for.

That `ids.py` and `models.py` import nothing from this package is load-bearing:
it is what keeps the graph acyclic without anyone thinking about it.

---

## 3. Data model — `ids.py`, `models.py`

Every model owns its own state machine and touches no other component. What it
does *not* own is its collection — that is the manager's (§6).

### 3.1 Typed identities — `ids.py`

```python
class _Id(uuid.UUID):
    """A UUID that is not interchangeable with a UUID of another kind."""
    __slots__ = ()

    @classmethod
    def new(cls) -> "Self":
        return cls(uuid.uuid4().hex)

    def __eq__(self, other) -> bool:
        return type(other) is type(self) and self.int == other.int

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.int))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"

    @classmethod
    def _coerce(cls, v) -> "Self":
        if isinstance(v, cls):
            return v
        if isinstance(v, uuid.UUID):
            return cls(v.hex)
        return cls(str(v))                      # UUID.__init__ rejects a malformed one

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return core_schema.no_info_plain_validator_function(
            cls._coerce,
            serialization=core_schema.plain_serializer_function_ser_schema(
                str, return_schema=core_schema.str_schema(), when_used="json"),
        )

class TaskId(_Id): ...
class AgentId(_Id): ...
class HandoffId(_Id): ...
```

Subclassing `uuid.UUID` rather than wrapping it: generation, parsing, ordering,
and `str()` come from the standard library, and `UUID.__init__` already rejects a
malformed value.

The equality overrides are what make criterion 27 pass. `UUID.__eq__` compares
`self.int` against any `UUID`, so without them a `TaskId` and a `HandoffId` built
from the same bytes would be equal and would collide in one dict. Overriding
`__eq__` alone silently sets `__hash__` to `None` and makes the type unhashable,
so both are defined together.

**`__get_pydantic_core_schema__` is required, not optional.** pydantic does *not*
handle a `UUID` subclass natively — it raises `PydanticSchemaGenerationError` on
the unknown type. The plain validator is the smallest thing that works, and
`when_used="json"` keeps `model_dump()` returning real id objects while
`model_dump(mode="json")` returns strings. Verified against pydantic 2.13:
round-trip preserves the type for both plain fields and `dict[HandoffId, int]`
keys.

One thing it does not buy: `_coerce` accepts any string, so a `HandoffId`'s
digits parsed into a `TaskId`-annotated field become a `TaskId`. Deserialisation
cannot detect a mix-up that the serialised form does not record. The protection
is static — `TaskId` and `HandoffId` are distinct classes, so passing one where
the other is annotated is an error the checker reports — plus runtime
distinctness for values that were never flattened to a string.

### 3.2 Models — `models.py`

```python
class Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",              # a typo'd field is an error, not a silent drop
        validate_assignment=True,    # task.status = X is validated
        use_enum_values=False,       # keep enum members; compare with `is`
    )
```

`validate_assignment` is the point of using pydantic here. State is mutated in
place — `task.status = RUNNING` — and this makes every such assignment go
through validation rather than trusting the caller.

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
    CREATED    = "created"       # declared; nothing written yet
    GENERATING = "generating"    # an agent has it open
    VALID      = "valid"         # sealed, usable
    INVALID    = "invalid"       # sealed, not usable
```

Both enums subclass `str` so a dumped record is plain JSON. Python 3.11's
`StrEnum` is the same thing; the repo targets 3.10, so `(str, Enum)` it is.

These two exist because the guards in `remove_queued`, `update_task`, and
`resume_task` would otherwise repeat the same tuple literal. There is no
`LIVE` or `FINAL` set: every other guard tests a single status, and a constant
with one call site is worse than the literal.

#### `Handoff` — one class, owning its transitions

```python
class Handoff(Model):
    uuid: HandoffId                        # the slot; shared by every version
    version: int = 0                       # (uuid, version) is this artefact
    type: str = ""
    status: HandoffStatus = HandoffStatus.CREATED
    produced_by: TaskId | None = None
    produced_by_agent: AgentId | None = None
    timestamp: datetime = Field(default_factory=_now)
    content: Any = None

    @property
    def is_valid(self) -> bool:
        return self.status is HandoffStatus.VALID

    def open(self, agent_id: AgentId) -> None:
        if self.status is not HandoffStatus.CREATED:
            raise HandoffStateError(f"{self!r}: cannot open a {self.status.value} handoff")
        self.status, self.produced_by_agent = HandoffStatus.GENERATING, agent_id
        self.timestamp = _now()

    def seal(self, status: HandoffStatus, content: Any = None) -> None:
        if self.status is not HandoffStatus.GENERATING:
            raise HandoffStateError(f"{self!r}: cannot seal a {self.status.value} handoff")
        if status not in (HandoffStatus.VALID, HandoffStatus.INVALID):
            raise HandoffStateError(f"a verdict must be VALID or INVALID, got {status}")
        self.status, self.content = status, content

    def successor(self, agent_id: AgentId) -> "Handoff":
        return Handoff(uuid=self.uuid, version=self.version + 1, type=self.type,
                       status=HandoffStatus.GENERATING, produced_by=self.produced_by,
                       produced_by_agent=agent_id)
```

The guards are the state machine. Criterion 26 tests them directly: seal a
`CREATED` one, open a sealed one, seal twice — each raises. No caller can move a
handoff illegally, because no caller is the one moving it.

`successor` returns `GENERATING`, not `CREATED`: a re-run version exists only
because an agent started writing it. `CREATED` is reachable only through
`declare`.

#### `Task` — same treatment

```python
class Execution(Model):
    attempt: int
    agent_id: AgentId
    input_versions: dict[HandoffId, int] = Field(default_factory=dict)
    output_versions: dict[HandoffId, int] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    outcome: TaskStatus | None = None

    @property
    def is_open(self) -> bool:
        return self.ended_at is None

class Task(Model):
    id: TaskId = Field(default_factory=TaskId.new)
    agent: str                                      # a NAME, resolved via AgentMgr
    inputs: list[HandoffId] = Field(default_factory=list)
    outputs: list[HandoffId] = Field(default_factory=list)
    resources: dict[str, float] = Field(default_factory=dict)   # pool NAME -> amount
    status: TaskStatus = TaskStatus.WAITING_HANDOFF
    created_at: datetime = Field(default_factory=_now)
    expedited: bool = False
    history: list[Execution] = Field(default_factory=list)

    @property
    def current(self) -> Execution | None:
        return self.history[-1] if self.history else None

    @property
    def is_running(self) -> bool:
        return self.current is not None and self.current.is_open

    def push_execution(self, agent_id: AgentId,
                       input_versions: dict[HandoffId, int]) -> Execution:
        if self.is_running:
            raise TaskStateError(f"{self.id!r}: attempt {self.current.attempt} is still open")
        self.history.append(Execution(attempt=len(self.history), agent_id=agent_id,
                                      input_versions=input_versions))
        return self.current

    def close_execution(self, output_versions: dict[HandoffId, int],
                        outcome: TaskStatus) -> None:
        if not self.is_running:
            raise TaskStateError(f"{self.id!r}: no open attempt to close")
        self.current.output_versions = output_versions
        self.current.ended_at, self.current.outcome = _now(), outcome
```

`push_execution` refusing to stack a second open attempt is what makes
`is_running` trustworthy, and it is why `on_stopped` must close the record
(spec §5.1) — otherwise the next `resume_task` would trip this guard.

`Task.agent` and the keys of `Task.resources` are the two `str`s that are
genuinely names, not identities. Both resolve to singletons registered by name.

#### The rest

```python
class Agent(Model):
    id: AgentId = Field(default_factory=AgentId.new)
    name: str
    knowledge: Any = None                            # left empty per mission.md
    config: dict[str, Any] = Field(default_factory=dict)

class RunOutcome(str, Enum):
    COMPLETED = "completed"      # the run finished; says nothing about content
    CRASHED   = "crashed"        # exception, kill, timeout

class TaskOutcome(Model):
    outcome: RunOutcome
    output_versions: dict[HandoffId, int] = Field(default_factory=dict)
    actual_usage: dict[str, float] = Field(default_factory=dict)
    detail: str = ""                                 # for a human; never parsed
```

`Agent.id` is `AgentId`, not `uuid` — the field is an identity of a known kind
and the name should say so.

### Serialisation

`model_dump(mode="json")` out, `model_validate` in. Nothing hand-written per
model: no `*_from_dict` constructors, no enum coercion, no `datetime` parsing.
The `HandoffId` keys of `input_versions` survive the round trip, validated back
into the declared key type by the schema in §3.1 — checked against pydantic 2.13,
not assumed.

This is the concrete payoff of adopting pydantic over `dataclasses`: the read
side of persistence was going to be two hand-maintained constructors that drift
from the models every time a field is added, and `asdict` has no inverse.

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

A full CRUD interface — spec §4.2.

```python
class StoreMgr(Protocol):
    def create(self, kind: str, key: str, record: dict) -> None: ...   # raises if present
    def read(self, kind: str, key: str) -> dict | None: ...
    def read_all(self, kind: str) -> list[dict]: ...
    def update(self, kind: str, key: str, record: dict) -> None: ...   # raises if absent
    def delete(self, kind: str, key: str) -> None: ...                 # raises if absent
    def exists(self, kind: str, key: str) -> bool: ...
```

`create` and `update` are separate because their preconditions differ and both
are worth enforcing: `TaskMgr.add` means *new* and a collision is a bug;
`persist` means *existing* and writing a vanished record is equally a bug. An
upsert would accept both mistakes silently. Criterion 29 tests the two
rejections.

Two implementations:

```python
class MemoryStoreMgr:
    """dict[kind][key] -> deepcopy(record). Survives a manager restart because
       the store object does; that is exactly what recovery needs to be testable."""

class JsonFileStoreMgr:
    """<root>/<kind>/<quoted-key>.json, one file per record."""
```

`MemoryStoreMgr` deep-copies on the way in and out. Without it a caller would
hold a live reference into the store, and "reload from persistence" in a test
would return the same objects it never actually wrote — recovery tests would
pass vacuously.

`JsonFileStoreMgr` writes to `<name>.json.tmp` and calls `Path.replace`, which is
atomic on POSIX. That gives per-record atomicity for free. It does *not* give
cross-record or cross-manager atomicity — spec §7 says so and §10 leaves it open.

Keys arrive as `str(some_id)` and become filenames, so they go through
`urllib.parse.quote(key, safe="")`. Callers pass opaque strings and should not
have to know they become paths.

The store never sees a model — managers pass `model_dump(mode="json")` and
validate on the way back. That is what lets one implementation serve both kinds.

`MemoryStoreMgr` is the default in tests. Recovery is still testable with it:
"restart" means constructing fresh managers over the *same* store object, which
is precisely what recovery does. `test_store.py` and one end-to-end case in
`test_recovery.py` exercise the JSON implementation against `tmp_path`.

### On-disk shape

```
<root>/
├── task/
│   └── 3f2b...c1.json        {"id": "3f2b...c1", "agent": "profiler",
│                              "status": "running", "created_at": "2026-08-20T...",
│                              "history": [{"attempt": 0, "agent_id": "9a4e...",
│                                           "input_versions": {"7d1c...": 0}}]}
└── handoff/
    ├── 7d1c...%3A0.json      {"uuid": "7d1c...", "version": 0, "status": "valid",
    │                          "produced_by": "3f2b...c1", "produced_by_agent": "..."}
    └── 7d1c...%3A1.json      {"uuid": "7d1c...", "version": 1, "status": "generating", ...}
```

**One file per version, not per handoff.** The key is `f"{uuid}:{version}"`
(the colon percent-encoded by the filename quoting), so appending v1 writes a new
file and never rewrites v0's. Spec §3.1 says a sealed version is never rewritten;
this makes that true on disk and not merely in memory.

The cost is that `HandoffMgr.resume` must regroup by `uuid` and sort by version,
since directory order is arbitrary. That is the two lines at the end of `resume`
(§6.1).

Directly readable, which is the whole reason for choosing files over sqlite while
the shape of the data is still settling.

---

## 6. Managers

Each manages a collection: add / get / query / remove over a set of one kind of
thing, plus persistence of it. Transitions belong to the members (§3.2), so no
manager has a `seal` or a `set_status`.

`ResourceMgr` is the acknowledged exception — it manages a quantity, not a set.
The name is kept because `mission.md` uses it.

### 6.1 `HandoffMgr` — `handoff.py`

Owns the **lineages**: `dict[HandoffId, list[Handoff]]`, each list ordered by
version. The lineage is a property of the collection, which is why it lives here
and not as a field on any `Handoff`. Every write persists under kind
`"handoff"`, keyed `f"{uuid}:{version}"` — one record per version, so a sealed
version is never rewritten on disk either.

```python
class HandoffMgr:
    def __init__(self, registry: Registry) -> None:
        self._r = registry
        self._lineages: dict[HandoffId, list[Handoff]] = {}

    # ---- scheduler-facing: read only ----
    def declare(self, ids, produced_by, types=None) -> None:
        for hid in ids:
            if hid in self._lineages:                 # idempotent — D6
                continue
            h = Handoff(uuid=hid, version=0, produced_by=produced_by,
                        status=HandoffStatus.CREATED, type=(types or {}).get(hid, ""))
            self._lineages[hid] = [h]
            self._persist(h)

    def check_if_latest_valid(self, hid) -> bool:
        latest = self.latest(hid)
        return latest is not None and latest.is_valid      # the Handoff answers

    def latest(self, hid) -> Handoff | None:
        lineage = self._lineages.get(hid)
        return lineage[-1] if lineage else None

    def lineage(self, hid) -> list[Handoff]:
        return list(self._lineages.get(hid, []))           # a copy; not the live list
    def get(self, hid, version) -> Handoff: ...
    def all_ids(self) -> list[HandoffId]: ...
    def produced_by(self, tid) -> list[HandoffId]: ...

    # ---- agent-facing: write ----
    def append(self, handoff: Handoff) -> None:
        """Place a version into its lineage. The transition already happened."""
        lineage = self._lineages.setdefault(handoff.uuid, [])
        if lineage and handoff.version != lineage[-1].version + 1:
            raise ValueError(f"{handoff.uuid}: v{handoff.version} does not follow "
                             f"v{lineage[-1].version}")
        lineage.append(handoff)
        self._persist(handoff)

    def persist(self, handoff: Handoff) -> None:
        """After an in-place open()/seal() on a member. Same record, updated."""
        self._persist(handoff)

    def resume(self) -> None:
        self._lineages = {}
        for d in self._r.get("store_mgr").read_all("handoff"):
            h = Handoff.model_validate(d)
            self._lineages.setdefault(h.uuid, []).append(h)
        for lineage in self._lineages.values():
            lineage.sort(key=lambda h: h.version)          # file order is not version order
```

The mgr does not decide anything about a handoff. An agent calls `h.open(...)`
or `h.successor(...)` — those are the transitions — and then hands the result to
`append` or `persist`. The only rule the mgr enforces is a collection rule:
versions arrive contiguously.

Three decisions worth stating:

**`check_if_latest_valid` delegates to `Handoff.is_valid`** rather than
comparing a status itself. There is one definition of "usable", and it is on the
object.

**An unknown id returns `False`, not an exception** (D5). A consumer may be
submitted before its producer declares the handoff; "not ready" is the right
answer and keeps submission order unconstrained.

**`declare` is idempotent** (D6). `update_task` is `remove_queued` + `submit`, so
it runs twice for the same outputs; overwriting would discard versions an agent
had already written.

`HandoffMgr` has no validation logic and never sets `VALID`/`INVALID`. That is
the whole content of spec §3.1, and criterion 14 tests it.

### 6.2 `TaskMgr` — `task.py`

Owns `dict[TaskId, Task]`, persisted under kind `"task"`.

| Method | Effect |
|---|---|
| `add(task)` | `store.create`; raises if the id exists and is not `CANCELLED` (D3) |
| `get(tid)` / `all()` | Read |
| `by_status(status)` | The collection query the pools index would otherwise be the only way to get |
| `remove(tid)` | Drop and `store.delete` |
| `persist(tid)` | `store.update` after a caller mutated the task through its own methods |
| `resume()` | Reload; close any dangling stack top |

There is no `set_status` and no `push_execution` here. A caller does
`task.status = RUNNING` or `task.push_execution(...)` — the transitions are the
`Task`'s (§3.2), with its own guards — and then `mgr.persist(tid)`. The mgr's
job is durability and lookup, not proxying its members' behaviour.

```python
def resume(self) -> None:
    self._tasks = {}
    for d in self._r.get("store_mgr").read_all("task"):
        t = Task.model_validate(d)
        self._tasks[t.id] = t
        if t.is_running:                     # the restart cut this attempt short
            t.close_execution({}, TaskStatus.SUSPENDED)
            self.persist(t.id)
```

The interrupted attempt closes as `SUSPENDED` — cut short, not judged (D8). The
*task's* landing state is a separate decision `Scheduler.resume()` makes a moment
later, and it is `WAITING_RESOURCE`; a new attempt gets pushed on top.

Closing it here is not tidiness. `Task.push_execution` refuses to stack on an
open attempt, so leaving it open would make the first `resume_task` after a
restart raise.

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

Owns two collections: the **specs**, a name → config table of what kinds of
agent exist, and the **instances**, `dict[AgentId, Agent]`, of what has actually
been created.

```python
class AgentMgr:
    def __init__(self) -> None:
        self._specs: dict[str, dict] = {}
        self._agents: dict[AgentId, Agent] = {}

    # ---- the spec table ----
    def register(self, name: str, **config) -> None:
        self._specs[name] = config
    def names(self) -> list[str]: ...
    def is_registered(self, name: str) -> bool: ...

    # ---- the instance collection ----
    def instantiate(self, name: str) -> Agent:
        if name not in self._specs:
            raise KeyError(f"no agent named {name!r}; registered: {sorted(self._specs)}")
        agent = Agent(name=name, config=dict(self._specs[name]))   # id auto-generated
        self._agents[agent.id] = agent
        return agent

    def get(self, ref: AgentId | str) -> Agent:
        """By id: that agent. By name: instantiate — the mission.md signature."""
        return self._agents[ref] if isinstance(ref, AgentId) else self.instantiate(ref)

    def by_name(self, name: str) -> list[Agent]: ...
    def all(self) -> list[Agent]: ...
    def retire(self, aid: AgentId) -> None: ...
```

**The mgr keeps what it creates.** Previously it was a factory that instantiated
and forgot, which left `Execution.agent_id` pointing at nothing — the audit
trail (§3.2) would name agents nobody could resolve, and criterion 22 requires
that a run be reconstructible from either end. Criterion 28 tests retention
directly.

**`instantiate` mints a fresh id every call.** Forced by criterion 21: after a
resume the stack top must report a different `agent_id` than the entry beneath.
One run, one agent instance.

`get` accepts both because `mission.md` asks for "提交agent名，返回agent对象" and
the audit path needs lookup by id. The dispatch path calls `instantiate`
explicitly — relying on the overload there would make "a new agent is created
here" invisible at the call site.

Instances are **not persisted**, so `AgentMgr` does not implement `Resumable`.
An agent owns its own durability (spec §1.2), and after a restart the ids in the
history refer to agents this process never made. That is a real gap and §14 O5
records it.

`knowledge` and `config` stay empty per `mission.md`.

---

## 7. Runner and policy

### 7.1 `TaskRunner` — `runner.py`

```python
class TaskRunner(Protocol):
    def start(self, task: Task, agent: Agent,
              on_done: Callable[[TaskId, TaskOutcome], None]) -> None: ...
    def stop(self, task_id: TaskId,
             on_stopped: Callable[[TaskId], None]) -> None: ...
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
    def stop(self, task_id, on_stopped):
        self.stop_requested.append(task_id)
        self._acks[task_id] = on_stopped

    # ---- test-driven, standing in for the agent ----
    def produce(self, registry, task_id, *, valid=True, content=None) -> None:
        """What a real agent does to its outputs: open (or fork) a version and seal it."""
        hm, task = registry.get("handoff_mgr"), registry.get("task_mgr").get(task_id)
        agent_id = task.current.agent_id
        for hid in task.outputs:
            latest = hm.latest(hid)
            if latest.status is HandoffStatus.CREATED:
                latest.open(agent_id); hm.persist(latest); h = latest
            else:                                     # a re-run forks a new version
                h = latest.successor(agent_id); hm.append(h)
            h.seal(VALID if valid else INVALID, content)
            hm.persist(h)

    def finish(self, task_id, outcome=RunOutcome.COMPLETED) -> None:
        task, _, on_done = self.running.pop(task_id)
        on_done(task_id, TaskOutcome(outcome=outcome, output_versions={
            h: ... for h in task.outputs}))

    def ack_stop(self, task_id) -> None:
        self.running.pop(task_id, None)
        self._acks.pop(task_id)(task_id)
```

`FakeRunner` never calls `on_done` from inside `start`. Tests stay deterministic,
and dispatch is not re-entered on the common path. Re-entrancy is still handled
(§9) because a real synchronous runner is a reasonable implementation and must
not deadlock or recurse.

`produce` is the agent's half of the contract in one place: it is the *only*
thing in the test suite that calls `open`, `successor`, or `seal`. That is what
makes criterion 14 meaningful — if the scheduler ever started writing handoff
state, this would no longer be the only writer and the test would catch it.

The `CREATED` / re-run branch is the whole reason `CREATED` is a real state: the
first run adopts the declared v0 in place, a re-run forks v+1.

### 7.2 `SchedulePolicy` — `policy.py`

```python
class SchedulePolicy(Protocol):
    def select(self, eligible: list[Task],
               snapshot: dict[str, float]) -> list[TaskId]: ...

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
        self.pools: dict[TaskStatus, set[TaskId]] = {s: set() for s in TaskStatus}
        self._lock = threading.RLock()
        self._in_dispatch = False
        self._dispatch_again = False
```

One bucket per status, so a task is in exactly one pool. Only three are
load-bearing for scheduling; the rest make "which tasks are suspended" a lookup.

### 8.1 The single writer

```python
def _move(self, tid: TaskId, status: TaskStatus) -> None:
    task_mgr = self._r.get("task_mgr")
    for pool in self.pools.values():
        pool.discard(tid)
    self.pools[status].add(tid)
    task = task_mgr.get(tid)
    if task.status is not status:
        task.status = status          # the Task's own field, validated on assignment
        task_mgr.persist(tid)         # the mgr's job: durability
```

Every transition in the system goes through this method, and nothing else assigns
`task.status` or writes `pools`. Criterion 12 — the index never disagrees with
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
| `on_task_done(tid, outcome)` | reject unless `RUNNING` → release → `task.close_execution` → `_move(SUCCEEDED\|FAILED)` → `try_dispatch` |
| `on_stopped(tid)` | reject unless `STOPPING` → release (consumables at full reservation) → `task.close_execution(..., SUSPENDED)` → `_move(SUSPENDED)` → `try_dispatch` |
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
        agent = self._r.get("agent_mgr").instantiate(task.agent)
        task.push_execution(                          # the Task's own transition
            agent_id=agent.id,
            input_versions={h: handoff_mgr.latest(h).version for h in task.inputs},
        )
        self._move(tid, RUNNING)                      # _move persists both
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
| `ids` | bare `str`, `NewType`, `typing.Annotated` | `uuid.UUID` subclasses | Generation, parsing, and formatting are solved in the stdlib. `NewType` gives static distinctness but erases at runtime, so two ids of different kinds would still be equal and collide in one dict. Subclassing gives both. It costs a ten-line `__get_pydantic_core_schema__` — pydantic does not accept a `UUID` subclass without one (§3.1). |
| `models` | `dataclasses`, `msgspec`, `attrs` | **pydantic v2** | Already installed — `fastapi` pulls it, so this adds nothing to the dependency set. It supplies `model_dump` / `model_validate`, which removes the two hand-written `*_from_dict` constructors `dataclasses.asdict` would have required (it has no inverse) and which would drift from the models on every field added. `validate_assignment` also makes in-place mutation checked, which matters because status is assigned directly. `msgspec` is faster and also present, but has no validation-on-assignment and a thinner enum/`datetime` story. |
| `registry` | `dependency-injector`, `pluggy`, `punq` | stdlib `dict` | Every candidate is built around constructor injection, which spec §4.1 explicitly rejects in favour of resolve-at-use-time. Their remaining feature — a name→instance map — is nine lines. |
| `store` | `sqlite3` (stdlib), `shelve` (stdlib), `tinydb`, `diskcache` | `json` + `pathlib` | The user asked for filesystem + JSON. Records are inspectable with `cat`, which matters while the schema is still moving. `Path.replace` gives per-record atomicity. **`sqlite3` is the named upgrade path** — it is stdlib, and it would supply the cross-manager transaction §14 O2 wants — but it hides the data behind a client and buys nothing else today. The `StoreMgr` Protocol exists so that swap is a one-file change. |
| `handoff` | content-addressed stores (git, DVC, S3) | own implementation | Versioning here is metadata bookkeeping, not content storage. Where payloads live is deliberately open (spec §8.2); when it is decided, a content store plugs in behind `Handoff.content` without touching this module. |
| `task` | — | own implementation | A dict with write-through. Nothing to adopt. |
| `resource` | `threading.Semaphore`, Prefect global concurrency limits | own implementation | A semaphore cannot express the consumable (reserve-then-settle) half, and cannot do the all-or-nothing multi-pool acquisition of §8.3 without a second layer on top. Prefect's limits do exactly what is wanted but live server-side; adopting a server for one primitive is the trade spec §9 rejected. |
| `agent` | any agent framework | own implementation | Two methods, both trivial. `mission.md` leaves agent internals empty on purpose. |
| `runner` | Claude Code / Codex / Cursor CLIs, `subprocess` | Protocol + a fake | The real implementations are harness-specific and out of scope (spec §1.2). What this system owes is the seam. |
| `policy` | `graphlib.TopologicalSorter`, `networkx`, OR-Tools | `sorted()` | Spec §9 records the rejections. FIFO is one `sorted` call; the composite priority rule from the prior art is a drop-in replacement behind the same Protocol. |
| `scheduler` | Prefect, Hatchet, Temporal, Ray, Airflow, Slurm | own implementation | Spec §9, from the prior-art survey. Every candidate is a platform whose scheduling core is not separable. |
| tests | — | `pytest` | Already a dev dependency of the repository. |

**pydantic v2 is adopted; everything else is standard library.** For the rest,
every candidate is either a platform (adopt the server to get the primitive) or a
library for a problem this system does not have — graph traversal, dependency
injection. The named upgrade path is `sqlite3` for the store, and it sits behind
an interface that already exists.

---

## 11. Test plan

`pytest`. Run with `pytest agent_sys/tests` from the repository root. The root
`testpaths = ["tests"]` only supplies a default when no path is given, so a bare
`pytest` still collects exactly the existing suite and nothing changes for it.

`agent_sys/conftest.py` puts `src/` on `sys.path`. That is the two-line editable
install, and it keeps the repository's `pyproject.toml` untouched while this is
unreleased — `[tool.setuptools.packages.find] include = ["infera*"]` means
`agent_sys` is not packaged today and needs no entry there.

`agent_sys/tests/` gets an `__init__.py` so pytest's `prepend` import mode does
not put the test directory itself on `sys.path`.

Every test builds its own `Registry` via `bootstrap.build_registry(...)` with a
`MemoryStoreMgr` and a `FakeRunner`. Nothing is process-global.

| File | Covers | Criteria |
|---|---|---|
| `test_ids.py` | cross-type inequality, hashing, `new()`, round-trip through `str` | 27 |
| `test_models.py` | `Handoff` and `Task` state-machine guards; `model_dump`/`model_validate` round-trip incl. `HandoffId` dict keys | 26 |
| `test_registry.py` | isolation, loud failure, `resolve` wildcard, replacement | 15 |
| `test_store.py` | CRUD, `create` on existing, `update` on missing, key quoting, atomic replace | 29 |
| `test_resource.py` | renewable vs consumable release, settle-at-actual | 4 |
| `test_handoff.py` | declare/idempotence, lineage ordering, contiguity check, `check_if_latest_valid` on unknown/created/generating/valid | 16 (part) |
| `test_task.py` | `add`/`by_status`/`remove`/`persist`, attempt numbering | 18 |
| `test_agent.py` | spec table, `instantiate` retains, fresh id per call, `get` by id vs name, `by_name` | 28 |
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
cycle in which `FakeRunner.produce` makes the agent's writes. `produce` brackets
itself with a marker in the same log, so "who called this" is recorded rather
than inferred from the call stack. The assertion: every `append` and `persist`
entry falls between a pair of markers, and every entry outside them is one of
`declare`, `check_if_latest_valid`, `latest`. This is the one mechanical check
that spec §3.1's authority boundary has not eroded.

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
| 1 | `ids` | — |
| 2 | `models` | 1 |
| 3 | `registry` | — |
| 4 | `store` | 2 (for the round-trip test only) |
| 5 | `resource` | — |
| 6 | `handoff` | 1–4 |
| 7 | `task` | 1–4 |
| 8 | `agent`, `runner`, `policy` | 1, 2 |
| 9 | `bootstrap` | 1–8 |
| 10 | `scheduler` — submit and dispatch | 1–9 |
| 11 | `scheduler` — lifecycle, completion, expedite | 10 |
| 12 | `resume_all` and recovery | 10, 11 |

Steps 1–9 are small enough that the interesting work is entirely in 10–12. That
is the intent: the components are dull so the scheduler can be read in one
sitting. Step 2 carries more than its size suggests — the state-machine guards
live there, and everything downstream relies on them holding.

Before step 1, per `mission.md` rule 5 and the global working rules: back up the
project `CLAUDE.md`, write a fresh one, and create the scratch workspace. No
temporary experiment leaves it.

---

## 13. Deviations from the spec

Each of these is a place where implementing the spec literally does not work.
None changes an acceptance criterion; several are forced *by* one.

Four deviations from revision 1 of this document — the `resume` name collision,
`TaskRunner.stop`'s callback, `on_stopped` closing the execution record, and the
`declare(types=...)` argument — are gone from this table because spec rev. 4
adopted them. They are now specification, not deviation.

| # | Spec says | Design does | Why |
|---|---|---|---|
| D3 | `submit` rejects a duplicate id | rejects an id that exists and is **not** `CANCELLED` | Forced by criterion 23. `update_task` must be `remove_queued` + `submit` under the same id, and `remove_queued` leaves the task `CANCELLED`. Reviving a cancelled id replaces the record with fresh history. |
| D5 | `check_if_latest_valid(hid)` | returns `False` for an unknown id rather than raising | A consumer may be submitted before its producer declares the handoff. "Not ready" is the right answer and it keeps submission order unconstrained. |
| D6 | `declare(ids, produced_by)` | idempotent; skips ids already known | `update_task` re-declares. Overwriting would delete versions an agent had already written. |
| D7 | `on_stopped` releases resources | consumables settle at the **full reservation** | `on_stopped` carries no `TaskOutcome`, so actual usage is unknown. Assuming the agent spent what it reserved is the safe direction for a budget. |
| D8 | "a dangling stack top is closed as interrupted" | `outcome = SUSPENDED`, `ended_at = now` | The spec does not name the outcome. `SUSPENDED` says the attempt was cut short rather than judged. |
| D9 | `HandoffMgr.append(handoff)` | plus a `persist(handoff)` | `Handoff.open()` mutates the declared v0 **in place** — it does not create a version — so there is nothing to append, only an updated record to write. Two verbs because there are two situations: a first run adopts v0, a re-run forks v+1. |

---

## 14. New open questions

These are found by this design and are **not** in spec §10.

| # | Question |
|---|---|
| **O1** | **Consumable budgets do not survive a restart.** Spec §6.4 has `ResourceMgr.resume()` reset to full capacity, and §7 persists only tasks and handoffs. For a renewable pool that is exactly right — no lease survives. For a consumable one it resurrects every token ever spent, which is a correctness hole in one of the two mandated resource types. The fix is small: `ConsumableMgr` persists `available` under a third kind, `"resource"`, and `resume()` reads it back. It needs a spec change to §7 ("two things are persisted") and it keeps recovery step 3 independent of steps 1 and 2. **Recommended, pending review.** Implemented as specified until then. |
| **O2** | A completion that arrives after `stop()` **raises into the runner**. `on_task_done` rejects a non-`RUNNING` task per spec §5.1, so a runner that finishes in the window between `stop()` and its own acknowledgement gets an exception where it expected a callback — and the finished work is discarded, since the task then goes `SUSPENDED`. Nothing leaks, because `on_stopped` still releases the resources. Two candidate fixes: accept `STOPPING` in `on_task_done` and treat the run as complete, or make the rejection a no-op return rather than a raise. The first is better and changes the state machine. Neither is done. |
| **O3** | `Task.resources` names pools that must already be registered, and `submit` rejects unknown ones. Nothing yet rejects a task whose declared amount exceeds a pool's total capacity: it is accepted and waits in `WAITING_RESOURCE` forever. A one-line check at submit would surface it immediately. Not built. |
| **O4** | **`Handoff.type` has no route from `submit`.** Spec §3.1 gives a handoff a `type`, but `Task` (§3.2) has no field naming the types of its outputs, so the scheduler declares them with `type=""` and nothing later fills it in. The field is dead on the scheduler path. Either `Task` gains an `output_types: dict[HandoffId, str]` — a §3.2 change — or `type` is acknowledged as being for directly-declared external handoffs only. Nothing depends on it today: the scheduler is content-agnostic and never reads it. |
| **O5** | **`AgentMgr` instances do not survive a restart.** The mgr now retains what it creates (§6.4), so `Execution.agent_id` resolves — until the process restarts. Agents are not persisted, because spec §1.2 makes an agent responsible for its own durability, so after `resume_all` every id in the restored history dangles and criterion 22 holds only within one process lifetime. Two candidate answers: persist the `Agent` record (id, name, config — not the agent's own state, which stays its business), or accept that the audit trail names agents the system cannot resolve after a restart and say so in §1.2. **The first is cheap and I recommend it**; it is one more `kind` in the store and one more `Resumable`. Not built, because it extends what §7 persists and that is a spec decision. |
