# Agent Task Graph — Design

| | |
|---|---|
| Status | Implemented — `src/` and `tests/` follow this document |
| Revision | 8 — 2026-08-20. Second review: D14–D18 and O13. (rev. 7: D10–D13, O10–O12, D3's scope) |
| Implements | `docs/spec.md` rev. 6 |
| Language | Python ≥ 3.10. Standard library plus pydantic v2 |

---

## 1. Scope

This document turns `spec.md` into files, classes, and interfaces. It adds no
requirements. Where it makes a choice the spec left open, the choice is stated
here; where implementing the spec exposed a contradiction in it, §13 says so
rather than papering over it.

The spec's 35 acceptance criteria are the definition of done. §11 maps every one
of them to a named test.

**This document specifies interfaces, not bodies.** A method appears as a
signature and a sentence of semantics; a body appears only where the ordering of
steps *is* the design decision — which is `try_dispatch` (§8.3) and nothing
else.

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
│       ├── models.py      Handoff, HandoffVersion, Task, Execution, Agent + enums
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

#### `Handoff` and `HandoffVersion`

The handoff is the slot; the version is what fills it. The slot owns version
bookkeeping, the version owns its own transition.

```python
class HandoffVersion(Model):
    version: int
    status: HandoffStatus = HandoffStatus.CREATED
    producer_task_id: TaskId | None = None
    producer_agent_id: AgentId | None = None       # None until opened
    timestamp: datetime = Field(default_factory=_now)
    content: Any = None

    @property
    def is_valid(self) -> bool: ...                # status is VALID

    def seal(self, status: HandoffStatus, content: Any = None) -> None:
        """GENERATING -> VALID | INVALID.
        Raises HandoffStateError unless currently GENERATING, and unless
        `status` is one of the two verdicts."""

class Handoff(Model):
    id: HandoffId
    type: str = ""
    versions: list[HandoffVersion]                 # index == version; never empty

    @property
    def latest(self) -> HandoffVersion: ...        # versions[-1]
    @property
    def is_latest_valid(self) -> bool: ...         # latest.is_valid
    def get(self, version: int) -> HandoffVersion: ...

    def open_next(self, task_id: TaskId, agent_id: AgentId) -> HandoffVersion:
        """Hand an agent a version to write, GENERATING, and return it.
        Adopts `latest` in place if it is still CREATED; otherwise appends
        v+1. Raises if `latest` is GENERATING — someone else has it open."""
```

`open_next` is one verb where the previous revision had two (`open` for the first
run, `successor` for a re-run). The caller says "I am about to write this" and
does not have to know which case it is in. Two consequences fall out:

- **Version numbers are contiguous structurally.** The list index *is* the
  version, so nothing checks it. The previous design had `HandoffMgr.append`
  validating `v == last.v + 1` — a rule that only existed because versions were
  loose objects a caller assembled.
- **`versions` is never empty.** `declare` creates v0, so `latest` needs no
  `None` branch and neither does anything downstream of it.

The guards are the state machine, and criterion 26 tests them directly. Refusing
`open_next` on a `GENERATING` latest is new: two agents writing one slot
concurrently is a contradiction in the model, and it should raise rather than
silently fork.

#### `Task` and `Execution`

```python
class Execution(Model):
    attempt: int
    agent_id: AgentId
    input_versions: dict[HandoffId, int] = Field(default_factory=dict)
    output_versions: dict[HandoffId, int] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    outcome: TaskStatus | None = None
    detail: str = ""                               # from the runner; for a human

    @property
    def is_open(self) -> bool: ...                 # ended_at is None

class Task(Model):
    id: TaskId = Field(default_factory=TaskId.new)
    agent_spec: str                                # a SPEC NAME, not an agent
    inputs: list[HandoffId] = Field(default_factory=list)
    outputs: list[HandoffId] = Field(default_factory=list)
    depends_on: list[TaskId] = Field(default_factory=list)      # the graph edge
    resources: dict[str, float] = Field(default_factory=dict)   # pool NAME -> amount
    status: TaskStatus = TaskStatus.WAITING_HANDOFF
    created_at: datetime = Field(default_factory=_now)
    expedited: bool = False
    history: list[Execution] = Field(default_factory=list)

    @property
    def current(self) -> Execution | None: ...     # history[-1], the live binding
    @property
    def is_running(self) -> bool: ...              # current is not None and current.is_open

    def push_execution(self, agent_id, input_versions) -> Execution:
        """Bind an agent by appending an open record, attempt = len(history).
        Raises TaskStateError if an attempt is already open."""

    def close_execution(self, output_versions, outcome: TaskStatus,
                        detail: str = "") -> None:
        """Seal the stack top. Raises TaskStateError if none is open."""
```

`push_execution` refusing to stack a second open attempt is what makes
`is_running` trustworthy, and it is why `on_stopped` must close the record —
otherwise the next `resume_task` trips this guard.

`agent_spec` and the keys of `resources` are the two `str`s that are genuinely
names. `agent_spec` says *spec* because `AgentMgr` holds two collections and this
one names an entry in the spec table, not an `Agent`.

`depends_on` is read by nothing in this design. It is the graph, recorded for
traversal, display, and analysis; the scheduler's dependency question stays
`inputs` + `check_if_latest_valid`. Criterion 31 asserts both halves — that a
topological sort works, and that blanking the field changes no scheduling
behaviour.

#### `Agent`

```python
class HandoffRef(Model):
    handoff_id: HandoffId
    version: int

class Agent(Model):
    id: AgentId = Field(default_factory=AgentId.new)
    spec: str                                      # which kind
    task_id: TaskId | None = None                  # what it is bound to
    handoffs: list[HandoffRef] = Field(default_factory=list)   # what it touched
    knowledge: Any = None                          # left empty per mission.md
    config: dict[str, Any] = Field(default_factory=dict)
```

`task_id` and `handoffs` are the agent's half of the two-way links (spec §3.1).
Without them a run is reconstructible only from the task end, and criterion 22
fails. They are written by whoever binds and whoever writes — `instantiate` sets
`task_id`, and the agent appends a `HandoffRef` each time it calls `open_next`.

There is **no result or outcome model.** The runner calls
`on_done(task_id, status, usage)` and everything else the scheduler reads for
itself (spec §6.3).

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
    def resume_system(self) -> None: ...

RESUME_ORDER = ["handoff_mgr", "agent_mgr", "task_mgr", "resource:*", "scheduler"]

def resume_all(registry: Registry) -> None:
    for pattern in RESUME_ORDER:
        for component in registry.resolve(pattern):
            if isinstance(component, Resumable):
                component.resume_system()
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
├── handoff/
│   └── 7d1c...9f.json        {"id": "7d1c...9f", "type": "profile",
│                              "versions": [{"version": 0, "status": "valid",
│                                            "producer_task_id": "3f2b...c1",
│                                            "producer_agent_id": "9a4e..."}]}
├── agent/
│   └── 9a4e...20.json        {"id": "9a4e...20", "spec": "profiler",
│                              "task_id": "3f2b...c1",
│                              "handoffs": [{"handoff_id": "7d1c...9f", "version": 0}]}
└── resource/
    └── token.json            {"name": "token", "available": 700000.0}
```

**Four kinds, one record each** — spec §7. A handoff's versions nest inside it
rather than being separate files; the previous revision keyed on
`f"{uuid}:{version}"`, which forced `resume_system` to regroup and sort because
directory order is not version order.

The `resource` directory holds one file per *consumable* pool and nothing else.
A renewable pool writes no record, so its absence is not a missing file — it is
the correct representation of a thing with no durable state.

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

Owns `dict[HandoffId, Handoff]`, persisted under kind `"handoff"` — one record
per handoff, its versions nested inside it.

```python
class HandoffMgr:
    def __init__(self, registry: Registry) -> None: ...

    # ---- scheduler-facing: read only ----
    def declare(self, ids, producer_task_id, types=None) -> None:
        """Create each handoff with a single CREATED v0. Idempotent: an id
        already present is skipped, never overwritten (D6)."""
    def check_if_latest_valid(self, hid) -> bool:
        """`self._handoffs[hid].is_latest_valid`; False if unknown (D5)."""
    def latest(self, hid) -> HandoffVersion | None: ...
    def get(self, hid) -> Handoff: ...                     # raises KeyError
    def get_many(self, ids) -> list[Handoff]: ...
    def all_ids(self) -> list[HandoffId]: ...
    def produced_by_task(self, tid) -> list[HandoffId]: ...

    # ---- agent-facing: write ----
    def persist(self, hid) -> None:
        """Write the handoff back after open_next() or seal() mutated it."""

    def resume_system(self) -> None:
        """Reload from the store. No regrouping: one record per handoff."""
```

**One write verb.** The previous revision needed `append` *and* `persist`,
because a first run mutated v0 in place while a re-run created a loose object
someone had to insert. With `open_next` on the handoff, every write is "this
handoff changed, store it" — and the mgr no longer validates version contiguity,
because the list index guarantees it.

**One record per handoff, not per version.** The previous revision keyed on
`f"{uuid}:{version}"`, which required `resume_system` to regroup and sort because
directory order is not version order. Nesting removes that. The spec's "a sealed
version is never rewritten" still holds — it is a rule about the version's fields,
which nothing touches after sealing, not about file granularity.

**`check_if_latest_valid` delegates** to `Handoff.is_latest_valid`, which
delegates to `HandoffVersion.is_valid`. One definition of "usable", on the object
that has it.

**An unknown id returns `False`, not an exception** (D5). A consumer may be
submitted before its producer declares the handoff; "not ready" is the right
answer and keeps submission order unconstrained.

The mgr decides nothing. Criterion 14 tests that.

### 6.2 `TaskMgr` — `task.py`

Owns `dict[TaskId, Task]`, persisted under kind `"task"`.

| Method | Effect |
|---|---|
| `add(task)` | `store.create`; raises if the id exists and is not `CANCELLED` (D3) |
| `get(tid)` / `all()` | Read |
| `by_status(status)` | The collection query the pools index would otherwise be the only way to get |
| `remove(tid)` | Drop and `store.delete` |
| `persist(tid)` | `store.update` after a caller mutated the task through its own methods |
| `resume_system()` | Reload; close any dangling stack top as `SUSPENDED` |

There is no `set_status` and no `push_execution` here. A caller does
`task.status = RUNNING` or `task.push_execution(...)` — the transitions are the
`Task`'s (§3.2), with its own guards — and then `mgr.persist(tid)`. The mgr's
job is durability and lookup, not proxying its members' behaviour.

`resume_system` reloads every record and, for any task whose stack top is still
open, closes it with `outcome = SUSPENDED`: the restart cut the attempt short, it
was not judged (D8). The *task's* landing state is a separate decision
`Scheduler.resume_system()` makes a moment later, and it is `WAITING_RESOURCE`;
a new attempt gets pushed on top.

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

    @abstractmethod
    def resume_system(self) -> None: ...     # the classes differ here — spec §3.4

class RenewableMgr(ResourceMgr):
    def can_afford(self, amount) -> bool: ...        # amount <= available
    def take(self, amount) -> None: ...              # raises if it does not fit
    def give_back(self, amount, actual=None) -> None:
        """Return the full amount. `actual` is meaningless for a renewable."""
    def resume_system(self) -> None:
        """available = capacity. No lease survives a restart, so nothing is held.
        Persists nothing: there is nothing to remember."""

class ConsumableMgr(ResourceMgr):
    # can_afford / take identical to renewable; release and recovery differ
    def give_back(self, amount, actual=None) -> None:
        """Return `amount - spent`, where spent is `actual` (clamped to amount),
        or the whole reservation when `actual` is None. THEN persist the
        balance: this is the moment spend becomes final."""
    def resume_system(self) -> None:
        """Read the balance back. Missing record -> capacity (first run)."""

class GpuMgr(RenewableMgr):    # name defaults to "gpu"
class TokenMgr(ConsumableMgr): # name defaults to "token"
```

**Only `give_back` persists, never `take`.** A reservation is a lease and dies
with its process; a settlement is spend and must not be un-spent. The store write
therefore sits in exactly one place, and a crash mid-run costs the reservation —
which is correct, because the task holding it is not running any more (criterion
33).

The record is one row under kind `"resource"`, keyed by pool name:
`{"name": "token", "available": 700000.0}`. Renewable pools write nothing, so
the directory holds one file per consumable and no more.

`GpuMgr` and `TokenMgr` add nothing today beyond a default name. They exist
because `mission.md` names them and because GPU-specific accounting (topology,
per-node pools) has an obvious home when it arrives. That they are currently
empty is the honest state, not an oversight.

`actual=None` on a consumable means "consumed everything reserved" — the
conservative reading for a budget. `on_stopped` relies on it (D7).

The `resume_system` split is why the renewable/consumable distinction lives at
the abstract-base level rather than being a boolean flag: the two classes differ
in *three* behaviours — release, persistence, recovery — and a flag would mean
three conditionals kept in agreement by hand.

### 6.4 `AgentMgr` — `agent.py`

Owns two collections: the **spec table**, `dict[str, dict]`, of what kinds of
agent exist, and the **instances**, `dict[AgentId, Agent]`, of what has been
created.

```python
class AgentMgr:
    # ---- the spec table ----
    def register(self, spec: str, **config) -> None: ...
    def specs(self) -> list[str]: ...
    def is_registered(self, spec: str) -> bool: ...

    # ---- the instance collection ----
    def instantiate(self, spec: str, task_id: TaskId) -> Agent:
        """Mint an Agent with a fresh AgentId, bound to that task, and keep it.
        Raises KeyError naming the registered specs if the spec is unknown."""
    def get(self, ref: AgentId | str) -> Agent:
        """By id: that instance. By spec name: instantiate one, unbound — the
        `get(name) -> agent` mission.md asks for."""
    def by_spec(self, spec: str) -> list[Agent]: ...
    def by_task(self, tid: TaskId) -> list[Agent]: ...
    def all(self) -> list[Agent]: ...
    def retire(self, aid: AgentId) -> None: ...

    # ---- persistence ----
    def persist(self, aid: AgentId) -> None:
        """Write an agent back after it appended to its `handoffs`."""
    def resume_system(self) -> None:
        """Reload instances under kind "agent". The spec table is NOT restored."""
```

**The mgr keeps what it creates.** Previously it was a factory that instantiated
and forgot, which left `Execution.agent_id` pointing at nothing — the audit trail
would name agents nobody could resolve. Criterion 28 tests retention.

**`instantiate` mints a fresh id every call and binds `task_id`.** The fresh id
is forced by criterion 21: after a resume the stack top must report a different
`agent_id` than the entry beneath. The binding is the agent's half of the
two-way link (§3.2).

`get` accepts both forms because `mission.md` asks for "提交agent名，返回agent对象"
and the audit path needs lookup by id. Dispatch calls `instantiate` explicitly —
relying on the overload there would make "a new agent is created here" invisible
at the call site, and `instantiate` is where the task binding is supplied.

**Instances persist; the spec table does not.** The asymmetry is the point:

| | Restored by | Because |
|---|---|---|
| instances | `resume_system` | They are state — `task_id` and `handoffs` are links nothing else records, and every `agent_id` in a restored execution history points at one |
| spec table | whoever builds the registry | It is configuration. Reading it back from a store would mean the system remembers a spec the operator has since removed |

A restored instance is a *record*, not a live agent. Nothing tries to resume the
agent's own process — that is out of scope (spec §1.2) — and nothing dispatches
against a restored instance either: `instantiate` always creates a new one.
Restoration exists so the audit trail resolves (criterion 34).

`AgentMgr` therefore implements `Resumable` and sits at position 2 in
`RESUME_ORDER` (§8.4). Nothing depends on it being there — no other component
reads an agent during recovery — so the position is free; it is early because
grouping the three independent reloads together reads better than scattering
them.

`knowledge` and `config` stay empty per `mission.md`.

---

## 7. Runner and policy

### 7.1 `TaskRunner` — `runner.py`

```python
OnDone = Callable[[TaskId, TaskStatus, dict[str, float]], None]

class TaskRunner(Protocol):
    def start(self, task: Task, agent: Agent, on_done: OnDone) -> None: ...
    def stop(self, task_id: TaskId,
             on_stopped: Callable[[TaskId], None]) -> None: ...
```

`on_done` carries the terminal status (`SUCCEEDED` or `FAILED`) and what was
spent. Nothing else: the scheduler reads output versions from `HandoffMgr`
itself, exactly as it read the input versions at dispatch (spec §6.3).

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
        """What a real agent does to its outputs: take a version and seal it."""
        hm, task = registry.get("handoff_mgr"), registry.get("task_mgr").get(task_id)
        agent = registry.get("agent_mgr").get(task.current.agent_id)
        for hid in task.outputs:
            version = hm.get(hid).open_next(task_id, agent.id)
            version.seal(VALID if valid else INVALID, content)
            hm.persist(hid)
            agent.handoffs.append(HandoffRef(handoff_id=hid, version=version.version))

    def finish(self, task_id, status=TaskStatus.SUCCEEDED, usage=None) -> None:
        _, _, on_done = self.running.pop(task_id)
        on_done(task_id, status, usage or {})

    def ack_stop(self, task_id) -> None:
        self.running.pop(task_id, None)
        self._acks.pop(task_id)(task_id)
```

`FakeRunner` never calls `on_done` from inside `start`. Tests stay deterministic,
and dispatch is not re-entered on the common path. Re-entrancy is still handled
(§9) because a real synchronous runner is a reasonable implementation and must
not deadlock or recurse.

`produce` is the agent's half of the contract in one place: it is the *only*
thing in the test suite that calls `open_next` or `seal`. That is what makes
criterion 14 meaningful — if the scheduler ever started writing handoff state,
this would no longer be the only writer and the test would catch it.

`open_next` collapses what used to be a two-branch conditional here (adopt v0 on
a first run, fork v+1 on a re-run). The fake no longer knows which case it is in,
which is the same simplification a real runner gets.

Appending the `HandoffRef` is the agent-side half of the two-way link. It is
`produce`'s job because it is the agent's job (spec §3.3) — the scheduler never
writes it, and criterion 22 checks both directions resolve.

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
| `submit(task)` | validate the agent spec, the resource names, and the resource **amounts** (D9) → `task_mgr.add` → `handoff_mgr.declare(task.outputs, task.id)` → `_warn_depends_on(task)` → `_move` to the pool `_ready` dictates → `try_dispatch` |
| `expedite(task)` | reject unless every input passes `check_if_latest_valid` → `task.expedited = True` → `submit(task)` |
| `remove_queued(tid)` | reject unless status in `WAITING` → `_move(CANCELLED)` |
| `stop(tid)` | reject unless `RUNNING` → `_move(STOPPING)` → `runner.stop(tid, self.on_stopped)` |
| `resume_task(tid)` | reject unless status in `RESUMABLE` → `_move` to the recomputed waiting pool → `try_dispatch` |
| `update_task(tid, **fields)` | `remove_queued(tid)` → `submit(model_copy(old, update={**fields, "status": WAITING_HANDOFF, "history": []}))` |
| `on_task_done(tid, status, usage)` | reject unless `RUNNING` → release, settling consumables at `usage` → read `output_versions` from `handoff_mgr` → `task.close_execution` → `_move(status)` → `try_dispatch` |
| `on_stopped(tid)` | reject unless `STOPPING` → release (consumables at full reservation) → `task.close_execution(..., SUSPENDED)` → `_move(SUSPENDED)` → `try_dispatch` |
| `resume_system()` | `Resumable`: rebuild the index, demote interrupted runs, `try_dispatch` |
| `try_dispatch()` | §8.3. A task that cannot be launched is released and moved to `FAILED` (D10); the pass continues |

`update_task` is written as literally those two calls, not as an equivalent
reimplementation. Criterion 23 then holds by construction rather than by
assertion — spec §2 principle 7.

`model_copy(update=...)` preserves `created_at`, so an update does not cost a
task its place in FIFO order. Criterion 23 compares the two arms field by field with
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
        agent = self._r.get("agent_mgr").instantiate(task.agent_spec, tid)
        task.push_execution(                          # the Task's own transition
            agent_id=agent.id,
            input_versions={h: handoff_mgr.latest(h).version for h in task.inputs},
        )   # instantiate() bound agent.task_id; the agent fills agent.handoffs
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

Step 4 is also the only part of the loop that can fail for reasons outside the
scheduler's control — an unknown spec, an agent factory that is down, an
unreachable harness — and by the time it runs the lease is already held. It is
therefore wrapped: `_abort_launch` gives the whole reservation back at
`actual=0`, closes the half-open attempt, and parks the task in `FAILED` without
re-raising (D10). Both halves matter. Without the release, one bad task shrinks
a pool permanently; without swallowing, one bad task aborts the pass and every
other queued task stays queued with no later event to release it.

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

`_warn_depends_on` implements spec §3.2's check: for each input, look up its
latest version's `producer_task_id` and warn through `logging` if it is absent
from `task.depends_on`. It warns and continues — rejecting would make declaration
order matter, and repairing would make `depends_on` derived and unable to express
a dependency that shares no handoff. Criterion 35 asserts both the warning and
the submission.

```python
def resume_system(self) -> None:
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
| `store` | `sqlite3` (stdlib), `shelve` (stdlib), `tinydb`, `diskcache` | `json` + `pathlib` | The user asked for filesystem + JSON. Records are inspectable with `cat`, which matters while the schema is still moving. `Path.replace` gives per-record atomicity. **`sqlite3` is the named upgrade path** — it is stdlib, and it would supply the cross-manager transaction spec §10 wants — but it hides the data behind a client and buys nothing else today. The `StoreMgr` Protocol exists so that swap is a one-file change. |
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
| `test_models.py` | `HandoffVersion.seal` and `Task.push/close_execution` guards; `open_next` adopt-then-append; round-trip incl. `HandoffId` dict keys | 26, 30 |
| `test_registry.py` | isolation, loud failure, `resolve` wildcard, replacement | 15 |
| `test_store.py` | CRUD, `create` on existing, `update` on missing, key quoting, atomic replace | 29 |
| `test_resource.py` | renewable vs consumable release, settle-at-actual, balance persists across a rebuild, unsettled reservation is not charged | 4, 32, 33 |
| `test_handoff.py` | declare/idempotence, `check_if_latest_valid` on unknown/created/generating/valid, `get_many`, `produced_by_task` | 16 (part) |
| `test_task.py` | `add`/`by_status`/`remove`/`persist`, attempt numbering | 18 |
| `test_agent.py` | spec table, `instantiate` retains and binds `task_id`, fresh id per call, `get` by id vs spec, `by_spec`, `by_task` | 28 |
| `test_policy.py` | FIFO order, expedited first, swap changes order only | 10 |
| `test_submit.py` | landing pool from input state, undeclared-resource rejection, the `depends_on` warning and that it does not reject | 1, 2, 35 |
| `test_dispatch.py` | all-or-nothing, agent binding, pinned input versions | 3, 18 |
| `test_lifecycle.py` | stop → on_stopped → resume_task, rejections, `update_task` | 5, 6, 11, 23 |
| `test_completion.py` | resource release, `ok` vs validity independence, failure path | 4, 7, 13 |
| `test_expedite.py` | ordering ahead of earlier tasks, rejection on invalid input | 9 |
| `test_versioning.py` | append not overwrite, earlier versions untouched, consumer sees new version, no invalidation call | 16, 17, 20 |
| `test_linkage.py` | both link directions resolve; agent readable only from `history[-1]`; `depends_on` sorts topologically and drives no scheduling | 21, 22, 31 |
| `test_authority.py` | the `HandoffMgr` spy: scheduler reads only | 14 |
| `test_invariants.py` | pools vs `TaskMgr`, after arbitrary operation sequences | 12 |
| `test_recovery.py` | `resume_all` order, skipping non-`Resumable`, demotion, verdicts not re-derived, agents resolve after a restart, the scheduler-first failure | 8, 19, 24, 25, 34 |

Two of these carry more weight than their size suggests:

**`test_authority.py`** registers a `HandoffMgr` subclass that appends every call
to a log, then drives a full submit → dispatch → complete → resume → re-dispatch
cycle in which `FakeRunner.produce` makes the agent's writes. `produce` brackets
itself with a marker in the same log, so "who called this" is recorded rather
than inferred from the call stack. The assertion: every `persist` entry falls
between a pair of markers, and every entry outside them is one of `declare`,
`check_if_latest_valid`, `latest`. This is the one mechanical check
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
| D3 | `submit` rejects a duplicate id | rejects an id that exists and is **not** `CANCELLED` | Forced by criterion 23. `update_task` must be `remove_queued` + `submit` under the same id, and `remove_queued` leaves the task `CANCELLED`. Reviving a cancelled id replaces the record with fresh history. **The exemption is wider than the reason for it** (noted by review): it lives on `TaskMgr.add`, not on the update path, so *any* caller can resubmit a cancelled id and spec §3.2's "`CANCELLED` is final" is weaker than it reads. Scoping it — a private `revive=True` only `update_task` passes — is a one-line change deliberately not made, because nothing in the system depends on finality and the narrower API would exist only to enforce a rule no caller wants to break. |
| D5 | `check_if_latest_valid(hid)` | returns `False` for an unknown id rather than raising | A consumer may be submitted before its producer declares the handoff. "Not ready" is the right answer and it keeps submission order unconstrained. |
| D6 | `declare(ids, producer_task_id)` | idempotent; skips ids already known | `update_task` re-declares. Overwriting would delete versions an agent had already written. |
| D7 | `on_stopped` releases resources | consumables settle at the **full reservation** | `on_stopped` carries no usage figures, so actual spend is unknown. Assuming the agent spent what it reserved is the safe direction for a budget. |
| D8 | "a dangling stack top is closed as interrupted" | `outcome = SUSPENDED`, `ended_at = now` | The spec does not name the outcome. `SUSPENDED` says the attempt was cut short rather than judged. |
| D9 | `submit` validates resource **names** | also rejects an amount that is negative or non-finite, and `can_afford` returns `False` for a negative one | Found by review during implementation. A negative amount passes `can_afford`, so step 3's all-or-nothing check succeeds — and then `take` raises partway through step 3's loop, after earlier pools were already debited. That is precisely the partial reservation criterion 3 exists to make impossible. Rejecting at submit keeps a malformed task out of the dispatch loop entirely; the `can_afford` guard makes a pool safe regardless of who calls it. |
| D10 | dispatch is "take, then bind, then start" | steps 3–4 are wrapped: a failure anywhere after `take` releases the whole reservation at `actual=0`, closes the half-open attempt, moves the task to `FAILED`, logs, and **does not re-raise** | Found by review. The lease is acquired before the agent is minted and before the runner is called, so an unknown spec, a downed agent factory, or an unreachable harness leaked it permanently — the same shape as D9, one step later. Two separate consequences needed fixing: the leak, and that the exception aborted the whole dispatch pass, so one bad task stopped every other queued task from ever starting. `FAILED` rather than back in the queue, because the next pass would retry it and fail identically forever; `resume_task` is the operator's move once the cause is fixed. This also keeps `resume_all` alive when the operator has removed a spec that persisted tasks still name. |
| D11 | `update_task` is `remove_queued` + `submit` | if the `submit` rejects, the original is re-submitted before the error propagates | Found by review. The cancel has already happened by then, so a rejected update — an unknown pool name, an unregistered spec — left the task `CANCELLED` and gone: the call silently destroyed the thing it was asked to change. The rollback keeps criterion 23 intact, since the successful path is still literally those two calls. |
| D12 | a consumable persists its balance | persists **`spent`**, the running total of settlements, and derives `available = capacity - spent` on resume | Found by review, and the most serious bug in the implementation. `available` is live and nets out every *outstanding reservation*, so persisting it baked each in-flight lease into the durable record — and a lease is supposed to die with its process (spec §3.4, criterion 33). Every interrupted run permanently shrank the budget by its reservation, and it compounded across restarts: measured a 400-token overcharge from one interrupted task. `spent` is a function of settlements alone and cannot carry a lease. `available` is still written to the record for a human with `cat`, and is not read back. |
| D13 | `_release` gives every lease back | one pool raising is logged and the rest still release | The run is over and cannot be un-finished. An escaping exception left the task `RUNNING` with an open execution record and its other leases held, escapable only by a restart. Same shape as D10, on the completion path instead of the dispatch path. |
| D14 | D10 wraps the launch | `_abort_launch` guards **each of its own steps** too, and pool resolution moved *inside* the guard | Found by the second review. D10 fixed the launch but left the handler itself bare, and a handler that raises has nowhere to go: the exception propagated out of `submit`, leaving the task `RUNNING` with a half-open attempt that only a restart clears. Two live triggers — a wedged pool's `give_back`, and `ConsumableMgr._persist` hitting a full disk. Separately, `self._r.get(f"resource:{name}")` sat *outside* the wrapped block, so a pool an operator deleted between restarts raised a `KeyError` that escaped the whole dispatch pass: `resume_all` died and every healthy task stayed queued. That is the exact failure D10 fixed for a removed *agent spec*, in the sibling case it did not reach. Reaching `FAILED` matters more than any single cleanup step succeeding. |
| D15 | eligibility is re-checked once per dispatch pass | `_ready(task)` is re-asked **per task**, immediately before its lease is taken | Found by the second review. Step 1 clears every queued task, then the loop starts them one at a time — and each `runner.start` can run agent code. A producer that opens its output slot on start invalidates a handoff a *later* task in the same pass was already cleared for; that task then pinned a `GENERATING` version, recording an input whose content does not exist yet and making criterion 18's audit record false. Reproduced single-threaded with a synchronous runner; the threaded case is the same bug via an agent thread. The re-check sits before `take`, so a task found stale returns to `WAITING_HANDOFF` without ever holding a lease. |
| D16 | `update_task(tid, **fields)` | validates `fields` against `Task.model_fields` and refuses `id` / `status` / `history` / `created_at` | Found by the second review. `model_copy(update=...)` writes straight to `__dict__`, honouring neither `extra="forbid"` nor `validate_assignment`. A misspelled field name was therefore accepted, reported success, and changed nothing; `id=` was worse — it left the original `CANCELLED` and created a *second* task under the new id, violating criterion 11's "under the same id" while keeping the pool invariant intact, so `test_invariants.py` could not see it. |
| D17 | `TaskMgr.remove(tid)` | refuses a task the scheduler still indexes | Found by the second review. `remove` deleted the record but nothing told the scheduler, leaving an id in a pool with no task behind it — and every subsequent dispatch pass then raised `KeyError` at the eligibility re-check, permanently. Nothing in the system calls it today, but it is a public method on a public manager that breaks "one writer per invariant" from the outside. The guard resolves the scheduler at use time and tolerates its absence, so `TaskMgr` still depends on no manager. |
| D18 | `ConsumableMgr.resume_system` sets `available = capacity - spent` | clamps at zero and warns | Found by the second review; the sibling of O7. An operator lowering a budget below what is already spent produced a negative `available`, and `can_afford` then returned `False` for *every* request including zero — each task naming the pool queued forever with no diagnostic. Clamping makes an exhausted budget behave like an exhausted budget. |

---

## 14. New open questions

These are found by this design and are **not** in spec §10.

Three entries from revision 3 are gone: O1 (consumable balances), O5 (agent
persistence), and O6 (`depends_on` unchecked) were accepted and are now
specified — spec §3.4, §7, and §3.2 respectively. They were spec decisions, which
is why they were raised here rather than fixed here.

| # | Question |
|---|---|
| **O2** | A completion that arrives after `stop()` **raises into the runner**. `on_task_done` rejects a non-`RUNNING` task per spec §5.1, so a runner that finishes in the window between `stop()` and its own acknowledgement gets an exception where it expected a callback — and the finished work is discarded, since the task then goes `SUSPENDED`. Nothing leaks, because `on_stopped` still releases the resources. Two candidate fixes: accept `STOPPING` in `on_task_done` and treat the run as complete, or make the rejection a no-op return rather than a raise. The first is better and changes the state machine. Neither is done. |
| **O3** | `Task.resources` names pools that must already be registered, and `submit` rejects unknown ones. Nothing yet rejects a task whose declared amount exceeds a pool's total capacity: it is accepted and waits in `WAITING_RESOURCE` forever. A one-line check at submit would surface it immediately. Not built. |
| **O4** | **`Handoff.type` has no route from `submit`.** Spec §3.1 gives a handoff a `type`, but `Task` has no field naming the types of its outputs, so the scheduler declares them with `type=""` and nothing later fills it in. Either `Task` gains an `output_types: dict[HandoffId, str]` — a §3.2 change — or `type` is acknowledged as being for directly-declared external handoffs only. Nothing depends on it today: the scheduler is content-agnostic and never reads it. |
| **O7** | **A consumable's capacity and its balance can disagree after a config change.** `resume_system` reads the stored `available` and ignores the `capacity` passed to the constructor, so raising a budget from 1M to 2M has no effect until the record is deleted by hand. That is the right default — the operator did not intend to hand back spend — but it means the constructor argument silently stops mattering. Either log when the two disagree at startup, or add the explicit `refill` that spec §10 already wants. Not built. |
| **O8** | **`_warn_depends_on` reads a version that may not exist yet.** The check looks up each input's `producer_task_id`, which is `None` until the producing task is submitted and `declare`s the slot. Submitting a consumer before its producer therefore warns about nothing — the check silently passes on exactly the graph most likely to be miswired. Re-running it at dispatch would catch that, at the cost of warning repeatedly. Accepted for now: the warning is a convenience, and the scheduling behaviour it guards is unaffected either way. |
| **O9** | **A handoff payload must be JSON-serialisable.** The scheduler is content-agnostic and `test_authority.py` proves it never inspects a payload — but `HandoffMgr.persist` dumps the whole handoff, so an arbitrary Python object raises `PydanticSerializationError` at the agent's `seal`, not at the scheduler. Found while writing the authority test and asserted there rather than left to be discovered by the first agent that returns a live object. The fix is the same one spec §8.2 already leaves open: decide where payloads live, and put a content store behind `Handoff.content`. |
| **O10** | **A version left `GENERATING` by a crash deadlocks its own retry.** Spec §6.4 requires that recovery not re-derive a verdict, and it does not — but `Handoff.open_next` refuses a slot that is already open, and nothing seals the abandoned one: the agent that opened it is dead, and the scheduler is forbidden from writing handoff state. The task is demoted, re-dispatched, and its new agent cannot write its own output. Found by review; asserted as it behaves in `test_recovery.py`. Criterion 20 is therefore satisfied only for the case where *something* closes the abandoned version off, not for a crash. Three candidate fixes, all of them spec decisions: let `open_next` adopt a version whose producing execution is no longer open; have `TaskMgr.resume_system` seal it `INVALID` alongside the execution it already closes (but that is a manager writing handoff state); or state that an operator seals abandoned versions by hand. |
| **O11** | **Criterion 27's static half is not enforced by anything.** The criterion says passing a `TaskId` where a `HandoffId` is expected is "a type error the checker reports", but no `mypy` or `pyright` configuration exists in the repository, so no checker runs in CI. `test_ids.py` asserts the structural fact the claim rests on — three unrelated leaf classes — and the runtime half is thoroughly covered, but the static half is an assertion about a tool nobody invokes. Either add a `mypy --strict` gate over `agent_sys/` or reword the criterion. Raised by review. |
| **O12** | **`test_authority.py` cannot see `open_next` or `seal`.** Criterion 14 is worded as "no `open_next` or `seal` is called from a scheduler frame", but those are methods on `Handoff` / `HandoffVersion`, and the spy wraps `HandoffMgr`. What is actually enforced is that `persist` only happens inside an agent span. Since the scheduler does hold live `Handoff` objects (through `get` and `latest`), a scheduler that mutated one and did not persist would pass. The inference is strong — an unpersisted mutation is a bug in its own right — but it is an inference. Closing it means wrapping the returned `Handoff` in the spy, or asserting on version-status transitions rather than on `persist`. Raised by review. |
| **O13** | **`output_versions` is misattributed when two tasks produce the same handoff.** `on_task_done` records `handoff_mgr.latest(h).version`, which is whatever the handoff holds *at completion time* — not what this run's agent wrote. With A and B both outputting `h`: A writes v0, B writes v1, A completes, and A's execution records `{h: 1}` while A's own agent's `handoffs` correctly says `[0]`. Criterion 22's "reconstructible from either end" then disagrees with itself depending on which end you read from. **This is spec-mandated, not an oversight**: criterion 13 requires the scheduler to record `output_versions` "by reading `HandoffMgr`, not from anything the runner passed", and spec §8.2 calls `latest(h)` the authoritative answer. `agent.handoffs` is the accurate source and reading it would not violate the authority rule — it is an agent record, not handoff state — but switching would contradict a criterion, so it is a spec decision. Raised by review; not fixed. |
