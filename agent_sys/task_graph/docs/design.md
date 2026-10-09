# Agent Task Graph — Design

| | |
|---|---|
| Status | Partly implemented — §1.1 says which half. `task_graph/` and `tests/task_graph/` follow this document up to spec rev. 7 |
| Revision | 14 — 2026-08-28. **§8.5's completion sentence is withdrawn.** It said a parent "reaches `SUCCEEDED` when its `is_end` subtask completes", which broke spec §3.2.1 twice — it skipped `OUTPUT_VALIDATING`, and it had the scheduler act on `is_end` where the spec says it "does not treat `is_end` specially at completion". The corrected route runs monitor-to-monitor and leaves this module out of it; §8.9 says so, and records that `enter_phase` and `Task.parent` are now monitor-facing. (rev. 13: 2026-08-27. **The user-interface brief**, spec rev. 13. `Task` gains `kinds` and `monitor_spec`, and `closure` is regularised from D23 into the link back to the task spec (§3.6, §3.7, §3.8). `submit` finally passes `types=` to `declare`, which rev. 11 omitted and which left every `Handoff.type` empty. §8.9 says what this module owes the monitor now that it is alpha scope. (rev. 12: 2026-08-27. **The stage-three consistency pass**, which found four defects in rev. 11 and one of them fatal on its first use. `Grant.handoff` becomes a kind **name**, adopting `closure` D2 (§3.5). The cascade no longer assigns `Task.cancel_reason`, a field that does not exist and whose assignment raises — measured (§8.6). `resume_system` rebuilds `OrderedIdSet` pools, not `set` (§8.4). `check_graph` returns `Problem`, not a second problem type (§8.7). (rev. 11: 2026-08-26. Subgraph nesting, the two validation phases, task-owned transitions, cascading cancel, and leaf-only acquisition — spec rev. 8–12, criteria 36–54. (rev. 10: references to the task-definition file made self-contained. rev. 9: renamed `agent_sys` → `task_graph`, flattened alongside `env_mgr`)))) |
| Implements | `docs/spec.md` rev. 13 |
| Language | Python ≥ 3.10. Standard library plus pydantic v2 |

---

## 1. Scope

This document turns `spec.md` into files, classes, and interfaces. It adds no
requirements. Where it makes a choice the spec left open, the choice is stated
here; where implementing the spec exposed a contradiction in it, §13 says so
rather than papering over it.

The spec's 54 acceptance criteria are the definition of done. §11 maps every one
of them to a named test.

**This document specifies interfaces, not bodies.** A method appears as a
signature and a sentence of semantics; a body appears only where the ordering of
steps *is* the design decision — which is `try_dispatch` (§8.3), the cascade
walk (§8.6), and nothing else.

The only runtime dependency is **pydantic v2**, which the repository already
installs — `fastapi` pulls it. §10 records why, per module, as the task
definition requires.

### 1.1 What is built, and what this revision adds

Revision 10 of this document was implemented in full and is green: **358 tests**
in `tests/task_graph`, 423 counting `env_mgr`. That is spec rev. 7 — criteria
1–35 — and none of it is redesigned here.

Revision 11 covers spec rev. 8–12 and **criteria 36–54**:

| Spec rev. | Subject | Criteria | Where |
|---|---|---|---|
| 8 | Subgraph nesting — `parent`, `is_start`, `is_end`, the system whole task | 36–38, 42 | §3.3, §8.5 |
| 9 | The two validation phases and their statuses; permissions as a task attribute | 39–41, 44 | §3.2, §3.5, §7.1, §8.2 |
| 9 | The default policy is depth-first | 43 | §7.2, §8.1.1 |
| 10 | A task owns its transitions | 45–48 | §3.4, §9 |
| 10 | Cascading cancel, and the graph-level load checks | 49–52 | §8.6, §8.7 |
| 11 | Only a leaf acquires resources | 53–54 | §8.3, §8.7 |

**Two of the nineteen cannot be fully designed here**, and §14 says so rather
than papering over it: criterion 49 depends on two rows of spec §10 at once, and
criterion 37's cancelled-subgraph case on a third. Everything else is settled
below.

**An increment to a shipped component is a different kind of document from a
greenfield one.** Where a rev. 10 decision is reversed, this document says which
one and why the earlier argument expired — §13 D19 and D20 are both of that
shape. Where a rev. 10 mechanism is untouched, it is not restated.

---

## 2. Layout and import graph

`agent_sys/` is a container holding two independent components, `env_mgr` and
this one. Each is a top-level package declared by `agent_sys/pyproject.toml`,
each owns its own `docs/` and `README.md`, and their tests are siblings under a
shared `tests/`. Nothing importable sits at the `agent_sys/` top level.

```
agent_sys/
├── pyproject.toml         declares both packages; ruff and pytest settings
├── env_mgr/               the sibling component — not this document's subject
├── task_graph/            this package
│   ├── README.md          build-versus-adopt record
│   ├── docs/
│   │   ├── spec.md
│   │   └── design.md
│   ├── __init__.py        re-exports the public names
│   ├── ids.py             TaskId, AgentId, HandoffId
│   ├── models.py          Handoff, HandoffVersion, Task, Execution, Agent + enums
│   ├── permissions.py     NEW. Permissions — carried, never interpreted. §3.5
│   ├── ordered.py         NEW. OrderedIdSet — the pool's collection type. §8.1.1
│   ├── registry.py        Registry, Resumable, RESUME_ORDER, resume_all
│   ├── store.py           StoreMgr Protocol, JsonFileStoreMgr, MemoryStoreMgr
│   ├── handoff.py         HandoffMgr
│   ├── task.py            TaskMgr — and the two derived indexes. §6.2.1
│   ├── resource.py        ResourceMgr, RenewableMgr, ConsumableMgr, GpuMgr, TokenMgr
│   ├── agent.py           AgentMgr
│   ├── runner.py          TaskRunner Protocol, FakeRunner
│   ├── policy.py          SchedulePolicy Protocol, FifoPolicy, DepthFirstPolicy
│   ├── scheduler.py       Scheduler
│   ├── graph.py           NEW. The graph-level load checks. §8.7
│   └── bootstrap.py       build_registry() — the composition root
└── tests/
    ├── env_mgr/
    └── task_graph/
```

**Four new modules, and each exists because putting it anywhere else would
create an edge the design forbids.**

| Module | Why not somewhere else |
|---|---|
| `permissions.py` | A `Permissions` type defined in `env_mgr` and imported here is a cross-package edge; defined in `models.py` and imported *there* is the same edge reversed. Its own module below `models.py` keeps both packages free of each other — `task_graph` carries the value, `env_mgr` interprets it. §3.5 |
| `ordered.py` | Nine lines. It is separate because `scheduler.py` and its tests both need it, and putting a collection type in `scheduler.py` invites the scheduler to grow collection behaviour |
| `graph.py` | The load checks run **before** the scheduler exists (§8.7), over task *specs* rather than over `Task` objects. Nothing in `scheduler.py` may reference them, or `test_authority.py`'s boundary erodes |
| `DepthFirstPolicy` in `policy.py` | Not a new module. It joins `FifoPolicy` behind the same Protocol — that is the whole point of criterion 10 |

**Why not `src/`.** Revisions 1–8 of this document specified `src/task_graph/`
with a `conftest.py` injecting it into `sys.path` — the two-line editable
install — chosen so the repository's `pyproject.toml` stayed untouched. That
reasoning expired when `env_mgr` arrived with its own `agent_sys/pyproject.toml`:
there is now a manifest to declare this package in, so declaring it is strictly
better than a path hack, and matching `env_mgr`'s flat layout matters more than
the `src/` convention's one guarantee. `pip install -e agent_sys` installs both.

### Import graph

```
                     ids.py             (uuid only)
                        ▲
                 permissions.py         (pydantic + ids)
                        ▲
                    models.py           (pydantic + ids + permissions)
                        ▲
   ┌─────────┬─────────┬┴────────┬──────────┬─────────┬─────────┐
handoff    task     runner    policy   scheduler   agent     graph
   ▲         ▲         ▲         ▲         ▲         ▲         ▲
   └─────────┴─────────┴────┬────┴─────────┴─────────┴─────────┘
                            │
                      bootstrap.py      (the only module that imports managers)

registry.py — imported by the managers for the `Registry` type annotation only
ordered.py  — imported by scheduler.py and its tests; imports nothing
store.py, resource.py — imported by nobody but bootstrap and tests
```

The graph stays acyclic and one-way, and rev. 11 does not weaken that. The one
place it could have is spec §3.2.3's requirement that a transition reach the
scheduler: written the obvious way, `models.py` would import `scheduler.py`,
which already imports `models.py`, and the package would fail at import time.
**The registry is what prevents it** — §3.4 — and criterion 48 asserts the
absence mechanically.

`graph.py` sits at the same level as the managers rather than below them: it
imports `models` for the types it checks and nothing else in the package.

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
    WAITING_HANDOFF   = "waiting_handoff"
    WAITING_RESOURCE  = "waiting_resource"
    INPUT_VALIDATING  = "input_validating"      # rev. 11
    RUNNING           = "running"
    OUTPUT_VALIDATING = "output_validating"     # rev. 11
    STOPPING          = "stopping"
    SUCCEEDED         = "succeeded"
    FAILED            = "failed"
    SUSPENDED         = "suspended"
    CANCELLED         = "cancelled"

# module level, alongside the enum
WAITING   = frozenset({TaskStatus.WAITING_HANDOFF, TaskStatus.WAITING_RESOURCE})
RESUMABLE = frozenset({TaskStatus.FAILED, TaskStatus.SUSPENDED})
PHASES    = (TaskStatus.INPUT_VALIDATING,        # rev. 11 — ORDERED, not a set
             TaskStatus.RUNNING,
             TaskStatus.OUTPUT_VALIDATING)

class HandoffStatus(str, Enum):
    CREATED    = "created"       # declared; nothing written yet
    GENERATING = "generating"    # an agent has it open
    VALID      = "valid"         # sealed, usable
    INVALID    = "invalid"       # sealed, not usable
```

Both enums subclass `str` so a dumped record is plain JSON. Python 3.11's
`StrEnum` is the same thing; the repo targets 3.10, so `(str, Enum)` it is.

`WAITING` and `RESUMABLE` exist because the guards in `remove_queued`,
`update_task`, and `resume_task` would otherwise repeat the same tuple literal.

**`PHASES` is new, and rev. 10 argued against exactly this** — "There is no
`LIVE` or `FINAL` set: every other guard tests a single status, and a constant
with one call site is worse than the literal." **That argument expired with the
two new members.** Three guards now test the same three statuses (`stop`,
`on_task_done`, and recovery's demotion — §8.2), and a literal repeated three
times is what the constant exists to prevent. Recorded as D19 rather than
changed silently, because the earlier reasoning was right when it was written.

**`PHASES` is a tuple, not a `frozenset`, and the order is load-bearing.** It is
the phase *sequence* — `INPUT_VALIDATING → RUNNING → OUTPUT_VALIDATING` — so
`PHASES[i + 1]` is "the next phase", which §3.4's phase transitions use. A
`frozenset` would make the membership tests read identically and silently lose
that. Membership is `status in PHASES`, which is O(3).

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

    # ---- rev. 11: structure. Nothing in scheduling reads these (criterion 42) ----
    parent: TaskId | None = None                   # which task this expanded from
    is_start: bool = True                          # the subgraph's entry point
    is_end: bool = True                            # the subgraph's exit point
    closure: str | None = None                     # THE LINK to its task spec. §3.6
    permissions: Permissions = Field(default_factory=Permissions)

    # ---- rev. 12 ----
    kinds: dict[HandoffId, str] = Field(default_factory=dict)   # slot -> kind. §3.7
    monitor_spec: str | None = None                # which monitor loop. §3.8

    # ---- rev. 11: the collaborator, kept out of persistence (§3.4) ----
    _registry: "Registry | None" = PrivateAttr(default=None)

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

**`is_start` and `is_end` default to `True`, not `False`.** Spec §3.2.1: "A task
that is its own start and end entry subtask is a leaf: it expands into nothing,
and `is_start` and `is_end` are both true of itself. That is the common case."
Defaulting to the common case is also what keeps every rev. 10 record loading
unchanged — measured, §5's on-disk shape.

**`closure` is this design's addition, it is not in spec §3.2's field table, and
D23 declares it.** The spec requires a subgraph to be "declared in the task's
spec" (§3.2.1) and `replace_with` to instantiate only declared closures
(criterion 51) — but gives the runtime `Task` no link back to its own
declaration: `agent_spec` names the agent spec and nothing names the task's. A
name is the smallest thing that closes that, resolved against the `closures`
registry at use time (§3.4), never an object — the same discipline `agent_spec`
already uses. It is `None` for a task submitted directly rather than
instantiated from a closure.

### 3.6 `closure` is the link, and it is why this model stays small

Spec §3.2.5 rev. 13 promotes what rev. 11 added as D23. The runtime `Task` is an
instance of a *task spec*, and that spec says far more than this model does — the
goal, the body, the materials, the dependency repositories (`closure` spec §2.5).

**None of it is copied here.** The task carries the *name* of its declaration and
whoever needs the rest resolves `closures` at use time, which is the discipline
`agent_spec` already uses and which `unfold` and `replace_with` already rely on.

A field per spec key was the alternative and it is rejected for the reason
§3.3 gives about `children`: the runtime model would become a second copy of the
spec, and two copies of one fact can disagree about what a task is.

**Who reads it.** `Task.unfold` and `replace_with` (§3.4); `agent`'s `Runner`, for
the body it hands the executor; `env_mgr`, for the dependency repositories. **Not
the scheduler** — closure criterion 8's prohibition is on the scheduler, and
nothing here changes that.

### 3.7 `kinds` — a uuid does not say what it is

`inputs` and `outputs` are ids. The **spec** names kinds (`inputs: ['facts']`);
the **runtime** names instances; `kinds` is the mapping, and it is derivable from
neither side alone. A task may legitimately hold two inputs of one kind — which is
precisely why lookup is by uuid (handoff spec §5.1) — so positional
correspondence with the spec's ordered list is not a substitute.

**`submit` passes it straight through:**

```python
handoff_mgr.declare(task.outputs, producer_task_id=task.id, types=task.kinds)
```

`declare` has taken `types` since rev. 4 of the spec and **nothing has ever passed
it**, because `Task` had no field to pass. So every `Handoff.type` is `""`.

That was inert while nothing read it, and rev. 12's `Grant.kind` (§3.5) is what
made it live: `env_mgr` design §6.1 resolves a permission grant by matching a
declared kind name against `Handoff.type`, so an unset type matches **no**
handoff. The executor is then confined to a zone containing none of its own
inputs, and finds out by failing to read one. `env_mgr` now raises rather than
returning an empty granted set, which makes the omission loud; this field is what
stops it happening.

**Only the ids are the graph.** `kinds` is a label map and nothing in scheduling
reads it — the same category as `parent` / `is_start` / `is_end`, and criterion
42's blanking test covers it on the same terms.

### 3.8 `monitor_spec` — a name, like every other collaborator

Spec §3.5 rev. 13 moves the monitor into the alpha. `monitor_spec` names which
loop watches this task, resolved from the component `Registry` at use time;
`None` takes the global default.

**A name and not an object**, for the reason every other collaborator here is a
name: an object on the model is a handle the task holds across restarts, and
§3.4 already measured what that costs — `model_validate` returns `None` for the
registry reference under both candidate mechanisms, so anything of that kind has
to be rehydrated by `TaskMgr` or it is dead.

**Nothing in this module runs a monitor.** §8.9 says what `task_graph` owes and
what it does not.

### 3.3 The four structural fields are declared, and nothing here reads them

Spec §3.2.1 is emphatic — "**nothing in scheduling reads them**" — and criterion
42 is the mechanical check: blank all three on every task and neither dispatch
order nor pool membership changes.

They earn their place the same way `depends_on` does (§3.2, rev. 10): the graph
exists for traversal, display, progress reporting, and the parent's own
completion accounting, and reconstructing it by scanning is work the submitter
already did for free.

**Where the design has to be careful is the two places that *want* to read
them**, and neither does:

| Wants to | Reads instead |
|---|---|
| Order the subgraph before a sibling (criterion 43) | The pool's own order. §8.1.1 — the ordering is a property of *where a task was placed*, not of a field it carries |
| Decide whether a task is a leaf (criterion 53) | `TaskMgr.children(tid)`. §6.2.1 — leaf-ness is the absence of children, not a marker pair |

#### Three things `Task` deliberately does not gain

Spec §3.2.1 rules each of these out, and none carries a criterion, so they are
recorded here rather than left to be re-proposed:

- **No `phase` field.** "Only the `main` phase produces scheduler-visible tasks,
  so every task in a pool is by definition a `main`-phase task and the field
  would be a constant." Where a task is within its run is `status` — that is what
  the two new members are for.
- **No `children` list.** The forward edge is derived (§6.2.1). Storing both
  directions on the model would make them two facts that can disagree, which is
  the shape spec §3.2 already warns about for `depends_on` versus `inputs`.
- **No agent status.** Spec §3.2.1: "**`Task.status` is a superset of the status
  of the agent at the top of its execution stack**" — it adds the states that
  exist when no agent is bound (`WAITING_HANDOFF`, `WAITING_RESOURCE`,
  `CANCELLED`) and the two phase states. The agent's own status lives on the
  agent (`agent` spec), and mirroring it here would be the second copy spec §2
  principle 3 forbids.

#### The system whole task

One task has `parent = None`. Spec §3.2.1: "the **system whole task**, whose
expansion is the entire graph. It exists so that 'has the system finished' is the
same question as 'has this task's `is_end` completed'."

It is an ordinary `Task` — no subclass, no flag beyond the `None`. Criterion 36
requires *exactly* one per graph, and that is a **graph-level** property, so it
is checked where the other graph-level properties are (§8.7) rather than at
`submit`, which sees one task at a time and could only ever say "a second root
appeared" after the fact.

**Deriving leaf-ness from `is_start and is_end` is rejected, and the reason is
in the spec's own wording.** Spec §3.2.1 states both-markers as a *consequence* of
leafness, not a test for it; using it as the test inverts the implication, which
is sound only if no non-leaf can carry both — and the spec never says that. It is
also unprecedented: in every surveyed system leaf-ness is a *type* distinction
(Dagster's `OpDefinition` vs `GraphDefinition`), never a flag pair.

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

### 3.4 A task owns its transitions — `models.py`

Spec §3.2.3: "**A task's state changes through its own transition functions, and
a transition is the only thing that triggers the scheduler.** Nothing outside a
task writes its status."

`Scheduler._move` stays the single writer of `task.status` and of the pools (§8.1,
unchanged). What rev. 11 adds is that **a transition becomes the only caller of
the paths that reach it.**

#### The reference, and why it is a `PrivateAttr`

A transition must reach the scheduler without `models.py` importing it. The
spec's answer is `self._registry.get("scheduler")` — resolution by name at use
time, which creates no import edge. That makes `Task` stop being pure data, and
spec §3.2.3 hands the design two questions: how the reference is supplied, and
how it is kept out of `model_dump`.

Both candidate mechanisms were measured against this package's own `Model`
config (`extra="forbid"`, `validate_assignment=True`):

| | `PrivateAttr` | field with `exclude=True` |
|---|---|---|
| `model_dump()` / `model_dump(mode="json")` | absent | absent |
| `model_copy()`, `model_copy(update={...})` | preserved | preserved |
| `model_validate(dump)` | **`None`** | **`None`** |
| assigning a wrong type | unchecked | `ValidationError` |
| needs a `model_config` change | no | **`arbitrary_types_allowed=True`** |

**`PrivateAttr` is chosen.** The type check `exclude=True` buys is worth little —
there is exactly one supplier and it is internal — while `arbitrary_types_allowed`
would have to go on the shared `Model` base and relax validation for *every*
model in the package to type one field on one of them. `PrivateAttr` is also the
honest declaration: the registry is not part of the task's data.

Two consequences the measurement forces:

- **Rehydration on load is not a choice.** `model_validate` returns a `Task` with
  `_registry = None` under either mechanism, so whoever reconstructs a task from
  the store must re-supply it. `TaskMgr` is the only thing that does — §6.2.1 —
  and a task obtained any other way has a dead transition path. `_require_registry`
  raises with the task id rather than letting `None.get` surface as an
  `AttributeError` three frames away.
- **`model_copy(deep=True)` and `copy.deepcopy` silently *clone* the registry**
  rather than sharing or dropping it, which is the worst of the three: the copy
  then drives a scheduler nobody else can see. Nothing in this package deep-copies
  a model — `MemoryStoreMgr` deep-copies plain dicts — so this is stated as a
  hazard rather than defended against. O16.

`update_task` is unaffected: it is `model_copy(update=...)`, which preserves the
reference. That matters, because criterion 23 holds by construction only if
`update_task` stays literally those two calls (§8.2).

#### The transition set

Each is a method on `Task`. Each checks its own precondition, mutates through
`_move`, and calls the scheduler exactly once at the end.

```python
def cancel(self, reason: str = "") -> CascadeReport:
    """A waiting state -> CANCELLED, then cascade downstream within this
    graph (§8.6). Rejects any other status. Returns what the cascade reached."""

def restart(self) -> None:
    """FAILED | SUSPENDED -> the recomputed waiting pool. `resume_task`
    expressed as a transition; cascades nothing."""

def fail(self, detail: str = "") -> None:
    """A phase state -> FAILED. Cascades nothing. Already reachable through
    on_task_done; this is the same effect with an explicit caller."""

def replace_with(self, closure_name: str) -> CascadeReport:
    """A waiting or terminal state. Cancels this graph's downstream, then
    instantiates `closure_name` — which must be in the `closures` registry
    (criterion 51). §8.6."""

# ---- the phase transitions, called by the runner ----
def enter_phase(self, phase: TaskStatus) -> None:
    """Advance to the next member of PHASES. Rejects a phase that is not the
    successor of the current one, so a runner cannot skip output validation
    by advancing twice."""

def unfold(self) -> list["Task"]:
    """Instantiate this task's declared expansion. Called on entry to the main
    phase, for a non-leaf. §8.5. Rejects a task with no `closure`."""
```

**`fail()` exists although `on_task_done` already reaches `FAILED`.** Spec §3.2.3
lists it, and the reason is criterion 47: the monitor's action set includes
failing a task it has judged dead, and the rule is that every monitor action is a
transition it *calls*, never a status it *assigns*. Without `fail()` the monitor
would have no verb and would reach for the field.

**The phase transitions are how the runner moves a task without writing its
status.** Criterion 45's spy records writes originating only inside a task
transition; a runner calling `task.enter_phase(RUNNING)` satisfies that, and a
runner assigning `task.status = RUNNING` does not. The guard — successor only —
is what makes the phase sequence enforceable rather than advisory, and it is why
`PHASES` is an ordered tuple (§3.2).

#### Reaching the scheduler

```python
def _sched(self):
    if self._registry is None:
        raise TaskStateError(f"task {self.id} has no registry; it was not "
                             f"loaded through TaskMgr")
    return self._registry.get("scheduler")
```

`registry.get` resolves by name at use time and creates no import edge — the
scheduler already does exactly this for `agent_mgr` and `runner`. Criterion 48
asserts the absence of the edge directly, over the module's AST rather than its
text (§11).

**`replace_with` also resolves `closures`, which is a *spec* registry**, and
main design §7 says the scheduler "has no name for a spec registry; it never
acquires one". Read closely, that prohibition is scoped to the **scheduler**, and
`replace_with` is a transition on `Task`. Resolving `closures` from a task adds
no `Scheduler → spec registry` edge, so `test_authority.py` and main criterion 10
are untouched. Said out loud here because the sentence reads as a whole-package
prohibition on a first pass; the rule that actually holds is sharper — *the
scheduler never reads a spec; the task may read the catalogue it came from.*

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
    knowledge: Any = None                          # left empty by the task definition
    config: dict[str, Any] = Field(default_factory=dict)
```

`task_id` and `handoffs` are the agent's half of the two-way links (spec §3.1).
Without them a run is reconstructible only from the task end, and criterion 22
fails. They are written by whoever binds and whoever writes — `instantiate` sets
`task_id`, and the agent appends a `HandoffRef` each time it calls `open_next`.

There is **no result or outcome model.** The runner calls
`on_done(task_id, status, usage)` and everything else the scheduler reads for
itself (spec §6.3).

### 3.5 `Permissions` — `permissions.py`

`task_graph` spec §3.2 types the field as `Permissions`. **Nothing in the spec
set defines that type.** `agent` spec §3.2 and `closure` spec §1.1 both point
back here; spec §3.2.2 gives the *default content* — own handoffs, workspace,
playground, log location, and everything belonging to subtasks recursively —
which is a default, not a shape. So this design is the first to give it one, and
it does so as narrowly as the consumers allow.

**What the consumers actually require is much less than spec §3.2.2's list suggests:**

| Consumer | What it needs from the field |
|---|---|
| `env_mgr` §4.5, the granted **read** set | Row 3 of four — "whatever else its permissions name". Rows 1, 2 and 4 are the system default, the task's own zone, and nothing-else |
| `env_mgr` §4.5, the **write** rule | **Nothing.** "A task's executor may not write outside its zones. Local or remote, no exception" — zones alone decide writes |
| `env_mgr` §5.1, subtree coverage | **Nothing.** "'may this task reach that path' is containment" — the recursive half is derived from the nested layout, not enumerated in the field |
| `closure` §4 load check 6 | The field must be **queryable against a specific handoff**, and must **distinguish read from write** per entry |

So criterion 44's "covers its subtasks recursively" is a property of the storage
layout that `env_mgr` builds, not of anything stored here. That is the single
most useful thing the consumption sites settle.

```python
class Access(str, Enum):
    READ  = "read"
    WRITE = "write"

class Grant(Model):
    path: str                                  # opaque here; env_mgr resolves it
    access: Access = Access.READ
    kind: str | None = None                    # a handoff KIND NAME, never an id

class Permissions(Model):
    grants: tuple[Grant, ...] = ()

    def covers(self, kind: str, access: Access) -> bool: ...
```

#### Rev. 11 typed that field `HandoffId`, and it could not do the job it was added for

**Adopted from `closure` design D2, which measured it.** Rev. 11 wrote
`handoff: HandoffId | None` and `covers(hid: HandoffId, ...)`, and named
`closure`'s load check 6 as the reason the type exists. But check 6 runs **at
load**, where a closure names handoff *kinds by name* — there is no instance yet,
so there is no id yet. Measured against this module's own shipped `ids.py`:

```
HandoffId('trace')  ->  ValueError: badly formed hexadecimal UUID string
_coerce('trace')    ->  ValueError: badly formed hexadecimal UUID string
```

The pydantic coercion path raises identically, so a declared grant naming a kind
could not be loaded into the declared type at all. **The type was correct for a
runtime question and unusable for the only question that asks it.**

The survey behind `closure` D2 is a three-system negative and it is worth
repeating here, because the instinct this corrects is a good one: Kubernetes RBAC
has **no UID field anywhere in its permission model** and `RoleRef` is immutable
across updates; Dagster carries `asset_key: str`, the declaration-time key
stringified; Android lint compares bare permission-name strings. **No surveyed
permission model references a runtime instance id.** The consistent shape is a
declared name plus a resolution step.

The resolution step is `env_mgr`'s, at zone build, and it already has what it
needs: `Handoff.type` carries the kind name on the runtime object
(`env_mgr` design §6.1). **What nothing yet does is fill that field** — O4 below,
and it stops being cosmetic here, because the granted set is now computed from
it.

**`task_graph` carries this and never interprets it.** No method here resolves a
path, compares a prefix, or decides containment — every one of those is
`env_mgr`'s, specified against measured behaviour in its §4.3. `covers` is a
lookup over declared entries, which is what `closure`'s load check 6 needs and
nothing more.

**The precedent for carrying-without-interpreting is inside this module.**
`Task.resources` is `dict[str, float]` keyed by pool *name*; the scheduler does
arithmetic on counters and never learns what a GPU is (spec §2 principle 2).
Permissions is the same shape one level up, and it is what lets the type live
here without either package importing the other.

`permissions.py` imports pydantic and `ids`, and nothing else. `ids.py` imports
only `uuid` and sits at the bottom of §2's graph, so this adds no cycle and lets
`Grant.handoff` carry a real `HandoffId` rather than a bare `str` — which is the
whole argument §3.1 makes for typed identities in the first place.

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
`Scheduler`'s task-level resume is named `resume_task` (spec §5.1) — otherwise the
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
│   └── 3f2b...c1.json        {"id": "3f2b...c1", "agent_spec": "profiler",
│                              "status": "input_validating",
│                              "created_at": "2026-08-20T...",
│                              "parent": "0b17...4e", "is_start": true,
│                              "is_end": false, "closure": "collect_trace",
│                              "permissions": {"grants": []},
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
The name is kept because the task definition uses it.

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

#### 6.2.1 Two derived indexes, and one supplier of the registry — rev. 11

Three things rev. 11 needs from `TaskMgr`, and all three are the same kind of
thing: a **forward edge derived from a stored back edge**.

```python
    def children(self, tid: TaskId) -> list[Task]:
        """Tasks whose `parent` is `tid`. Empty means leaf (criterion 53)."""
    def consumers(self, hid: HandoffId) -> list[Task]:
        """Tasks naming `hid` in `inputs`. The downstream direction (criterion 49)."""
```

`Task` stores `parent`, not `children`; it stores `inputs`, not consumers. Both
questions are asked often enough to matter — leaf-ness at every acquisition
point, consumers at every cascade level.

**They are maintained as indexes, not computed as scans, and that is a
departure.** `by_status` is a scan today, and the in-module precedent is "a
collection query is a scan at this scale". The reason to break it is not
performance:

- Spec §3.2.4 promotes the reverse index **from an optimisation to a
  requirement** — "a cascade needs to know a task's consumers, and nothing
  provides that."
- Every surveyed system with a reverse edge maintains it **eagerly and
  symmetrically, in the same statement** as the forward one — Airflow's
  `_set_relatives`, Dask's `add_dependency`, Kubernetes' `insertNode`. **Nobody
  computes it on demand.**

**These two methods are not in spec §5's `TaskMgr` listing, and they are not an
overreach: spec §3.2.4 asks for exactly them.** "What it is keyed by, who
maintains it, and how `submit` and `update_task` keep it current must be
specified before a cascade can be." Three questions, three answers:

| §3.2.4 asks | Answer |
|---|---|
| **Keyed by what** | `children` by `TaskId` — the value of `parent`. `consumers` by `HandoffId` — a member of `inputs`. Both are the stored back edge inverted, so neither introduces a key that is not already a field |
| **Maintained by whom** | `TaskMgr`, and nothing else. It is the only object that sees a task enter or leave the collection |
| **How `submit` and `update_task` keep it current** | Neither touches the index. Both reach it through `add` / `remove`, which is where the maintenance sits |

So the design question is not eager-versus-lazy. It is **which write path is the
single edge-creation site**, and the answer has to cover `submit`, `update_task`,
`remove`, and `unfold`:

```python
    def add(self, task: Task) -> None:
        """...unchanged, plus: task._registry = self._r, then _index(task)."""
    def remove(self, tid: TaskId) -> None:
        """...unchanged, plus: _unindex(task) before the delete."""
```

`add` and `remove` are the only two places a task enters or leaves the
collection — `update_task` is `remove_queued` + `submit`, and `unfold` submits —
so indexing there covers every path by construction, the same way criterion 23
holds by construction.

**A dangling edge is not an error.** `children(tid)` for an unknown `tid` returns
empty and `consumers(hid)` for an undeclared handoff returns empty, matching
`check_if_latest_valid`'s D5: a consumer may be submitted before its producer.
Kubernetes reaches the same rule from the other direction — a dangling
`ownerReference` is *treated as absent* — and it is what keeps submission order
unconstrained.

**The index finds candidates; it never authorises the destructive act.** §8.6's
cascade re-reads each task's status from `TaskMgr` before cancelling it rather
than trusting what the index said a moment ago. That is Kubernetes' shape, and
its own source is blunt about why: `getDependents` "does not provide any
synchronization guarantees; items could be added to or removed from
ownerNode.dependents the moment this function returns."

**`add` is also where the registry reference is supplied.** §3.4 measured that
`model_validate` returns a task with `_registry = None`, so rehydration is
forced; `TaskMgr` is the only component that constructs tasks from the store, and
`resume_system` re-supplies it on every reloaded record for the same reason.

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
because the task definition names them and because GPU-specific accounting (topology,
per-node pools) has an obvious home when it arrives. That they are currently
empty is the honest state, not an oversight.

`actual=None` on a consumable means "consumed everything reserved" — the
conservative reading for a budget. `on_stopped` relies on it (D7).

#### 6.3.1 A non-leaf spends without reserving — rev. 11

Spec §6.2 permits it explicitly: "**A non-leaf's validation phases may still run
an AI agent and spend tokens.** That does not reintroduce the problem: the rule
is about *acquisition*, and a non-leaf acquires nothing, so **its phases' spend
is recorded rather than reserved.**"

**The rev. 10 mechanism cannot record it, and fails silently.** Two independent
reasons, both in shipped code:

- `on_task_done` settles by iterating `task.resources`, which is empty for a
  non-leaf (criterion 53), so `give_back` is never called at all.
- Even if it were, the clamp would erase the figure:
  `spent = amount if actual is None else min(max(actual, 0.0), amount)`. With
  `amount = 0` and `actual = 100`, `spent` is **0**.

That clamp is right for what it was written for — a leaf must not settle for more
than it reserved — so it is kept, and a second verb is added rather than the
first being loosened:

```python
class ConsumableMgr(ResourceMgr):
    def charge(self, amount: float) -> None:
        """Record spend that was never reserved. `available` may go negative;
        `spent` grows. Persists, because a settlement is what makes spend final."""
```

**`charge` may drive `available` below zero, and that is the honest outcome.**
The alternative is refusing to record spend that has already happened, which is
the one thing a budget must never do. D18 already clamps `available` at zero on
*resume* and warns — an exhausted budget then behaves like an exhausted budget —
and `charge` reaches the same state by the same route.

`on_task_done` calls `charge` for any pool named in `usage` that the task did not
declare, and `give_back` for the ones it did. For a leaf the second branch is
empty in practice and for a non-leaf the first is; neither needs a leaf test.

**Renewables have no equivalent and must not get one.** A GPU that was never
taken cannot be given back, and "spend" is not a concept a renewable has —
spec §3.4's two release semantics are exactly this distinction. `usage` naming a
renewable pool a task did not declare is a caller error and is logged, not
silently absorbed. O20.

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
        `get(name) -> agent` the task definition asks for."""
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

`get` accepts both forms because the task definition asks for "submit an agent
name, get back an agent object" and the audit path needs lookup by id. Dispatch
calls `instantiate` explicitly —
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

`knowledge` and `config` stay empty, as the task definition requires.

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

`stop` takes a callback, symmetric with `start` (spec §5.1). The alternative — the
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

#### The three phases, and why `FakeRunner` must grow them

Spec §3.2.1: a task runs in three phases, and **only the main one is a graph**.
The two validation phases "are not tasks it dispatches, they take no pool slot,
and no policy orders them" — `TaskRunner` runs all three for the one task the
scheduler dispatched.

**The `TaskRunner` Protocol does not change.** `start` and `stop` are the whole
interface, and the real runtime is out of scope — spec §1.2: "That is
`TaskRunner`, a registered interface; this system defines the interface and ships
only a fake." What runs inside `start` is the runner's business, which is exactly
the boundary spec §2 principle 4 already draws.

**But criteria 39, 40 and 41 are assertions about a *task's* observable states**,
so if the fake does not move through them, none of the three is testable. So:

```python
class FakeRunner:
    def start(self, task, agent, on_done):
        self.running[task.id] = (task, agent, on_done)
        # the task is already INPUT_VALIDATING; the scheduler put it there

    # ---- test-driven, standing in for the phase machinery ----
    def advance(self, task_id) -> None:
        """task.enter_phase(next) — the runner's only route to a status (§3.4)."""
    def skip_phase(self, task_id, reason: str) -> None:
        """Advance without running anything, and record why. Criterion 41."""
```

**`advance` calls a transition; it never assigns a status.** That is the whole
of criterion 45 as it applies to a runner, and making the fake do it correctly is
what makes criterion 45's spy meaningful — a fake that wrote `task.status`
directly would train the test suite to accept the thing the criterion forbids.

**A validation phase produces no output handoff, and that is why `on_task_done`
needs no new argument.** Spec §3.2.1: a phase "calls
`handoff.update_validation_status(versioned_handoff, ...)` instead. That update
is persisted — a YAML record maintained alongside — and is **excluded from the
handoff's checksum**, so recording a verdict does not change the artefact's
identity." Both halves are the handoff module's — `handoff` design §4.1 places
`validation.yaml` beside the content subtree and outside the digest — and
nothing in `task_graph` writes or reads either. It is stated here because a
reader looking for "where does the verdict go" would otherwise expect it on the
`Execution` record, and putting it there would be the scheduler holding handoff
state, which criterion 14 forbids.

**Criterion 41's "the skip is reported" is already owned one layer down.**
`validator` design §5.2's `run_phase` returns a `PhaseOutcome` folding `ran`,
`reused` and `skipped`, and a skipped validator produces a `SkipRecord` naming
the validator and why. `TaskRunner` is the seam between the two modules. So this
design owes the *state transition* — the task advances rather than stalling — and
`FakeRunner.skip_phase` records the reason on the `Execution` for a human;
the structured report is the validator's, and duplicating it here would give the
system two answers to one question.

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

#### `DepthFirstPolicy` — the default from rev. 11

```python
class DepthFirstPolicy:
    def select(self, eligible: list[Task],
               snapshot: dict[str, float]) -> list[TaskId]:
        """Stack-like. `eligible` arrives in promotion order (§8.1), so this is
        that order reversed — most recently promoted first — with expedited
        tasks lifted to the front as `FifoPolicy` also does."""
```

**It reads no field on `Task` except `expedited`.** Not `parent`, not `is_start`,
not `is_end` — so criterion 42's mechanical check (blank all three, nothing
changes) passes trivially, and criterion 43 is satisfied at the same time.

**The order comes from the pool, not from a key computed over the task.**
`eligible` arrives in promotion order (§8.1), and depth-first is "take the most
recently promoted". That works because of where promotion happens: a subgraph's
`P1 → P2 → P3` chain enters `WAITING_HANDOFF` at unfold, and each link is
promoted to `WAITING_RESOURCE` only when its predecessor's output becomes valid.
**The frontier that most recently advanced is the deepest one**, which is what
depth-first means operationally.

**Two keys computed over `Task` were designed first and both are rejected.**
Recorded because a reader will otherwise propose them again:

| Rejected key | Why |
|---|---|
| Sort by `parent` / subgraph membership | Satisfies criterion 43, fails criterion 42. The conflict is not resolvable inside the policy |
| LIFO on `created_at` | Reads no structural field and passes criterion 43's *worked example* — but measured on a chain `P1 → P2` with an unrelated `Q` submitted later, it picks `Q` over `P2` and abandons the subgraph mid-way. That is the opposite of spec §5.2's "as far down as it will go, before starting a sibling". It satisfies the criterion's example and not the sentence the criterion is drawn from |
| A stamped `ready_since` timestamp on `Task` | Correct — it is promotion order by another name — but it adds a mutable derived field, and a field that can go stale, to express something the collection already knows. Rejected as redundant once the pool became ordered. D20 |

**Prior art is split, and the split is instructive.** No workflow engine surveyed
— Argo, Dagster, Airflow, Prefect, Nextflow, Snakemake — separates ordering from
graph structure; all order by something flat. **kube-scheduler does**:
`PrioritySort.Less` sorts on priority then `QueuedPodInfo`'s timestamp, which is
*"The time entity added to the scheduling queue"* — assigned by the scheduler on
entry, not the object's creation time. The sort plugin never reads the object
graph. That is this design's shape, with a position where kube-scheduler uses a
timestamp.

**And the mechanism that gave Airflow depth-first is the one that deadlocked and
was deleted.** Airflow #35689 asks for exactly criterion 43 — "does not maintain
the sequential integrity of each group" — and names `SubDagOperator` as the only
workaround; `SubDagOperator` is what §8.3 exists to prevent. Worth knowing that
the two capabilities arrived together elsewhere and are being separated here.

`FifoPolicy` is unchanged and stays available; criterion 43 requires swapping
back to it to change the order and nothing else. It sorts by `created_at` and
ignores promotion order entirely, which is what makes the two policies genuinely
different rather than two spellings of one.

---

## 8. Scheduler — `scheduler.py`

```python
class Scheduler:
    def __init__(self, registry: Registry) -> None:
        self._r = registry
        self.pools: dict[TaskStatus, OrderedIdSet] = {s: OrderedIdSet() for s in TaskStatus}
        self._lock = threading.RLock()
        self._in_dispatch = False
        self._dispatch_again = False
        self._cascade: deque[tuple[TaskId, str]] = deque()   # rev. 11, §8.6
```

One bucket per status, so a task is in exactly one pool. **Ten buckets from
rev. 11** — the two phase statuses create two more by construction, since the
dict comprehends over the whole enum.

**Four are load-bearing for scheduling, not three.** Spec §4.4 names
`WAITING_HANDOFF`, `WAITING_RESOURCE` and `RUNNING`, which was right before the
phase states existed. Dispatch now lands a task in `INPUT_VALIDATING`, so the
third entry has become a three-member group — a task holding a lease is in
whichever of `PHASES` it has reached, and every guard that used to name
`RUNNING` now names the group (§8.2). The rest still exist so that "which tasks
are suspended" is a lookup rather than a scan. D24(b).

### 8.1 The single writer, and the pool's order

```python
def _move(self, tid: TaskId, status: TaskStatus) -> None:
    task_mgr = self._r.get("task_mgr")
    if tid in self.pools[status]:         # rev. 11: already here — keep its place
        return self._sync(tid, status)
    for pool in self.pools.values():
        pool.discard(tid)
    self.pools[status].add(tid)           # appended: promotion order (§8.1.1)
    task = task_mgr.get(tid)
    if task.status is not status:
        task.status = status          # the Task's own field, validated on assignment
        task_mgr.persist(tid)         # the mgr's job: durability
```

Every transition in the system goes through this method, and nothing else assigns
`task.status` or writes `pools`. Criterion 12 — the index never disagrees with
`TaskMgr` — holds by construction because there is exactly one writer.

Discarding from all ten pools rather than from the task's recorded status makes
`_move` idempotent and self-healing: it cannot leave a stale entry behind even if
called on a task whose stored status was already wrong.

#### The early return is not an optimisation — it is what makes depth-first work

**This is the one line rev. 11 could most easily have got wrong.** With a `set`,
`discard` then `add` is genuinely idempotent and position is meaningless. With an
ordered collection it is not: it moves the task to the end.

And `_dispatch_pass` step 1 calls `_move` on **every** waiting task, **every**
pass:

```python
for tid in list(self.pools[WAITING_HANDOFF] | self.pools[WAITING_RESOURCE]):
    self._move(tid, WAITING_RESOURCE if self._ready(task) else WAITING_HANDOFF)
```

Without the early return, each pass would re-append every eligible task in
iteration order and **the promotion order would be destroyed on the pass after it
was established** — silently, with dispatch degrading to something arbitrary and
no test failing unless one specifically checks order across two passes. §11 gives
that check its own test.

So the invariant `_move` now carries is: **a task's position changes only when
its pool changes.** `_sync` is the remainder of the old body — reconcile
`task.status` and persist — with the pool write skipped.

### 8.1.1 `OrderedIdSet` — `ordered.py`

```python
class OrderedIdSet:
    """Insertion-ordered set of TaskId. A dict with None values."""
    def add(self, tid: TaskId) -> None: ...        # append; no-op if present
    def discard(self, tid: TaskId) -> None: ...
    def __contains__(self, tid) -> bool: ...
    def __iter__(self): ...                        # insertion order
    def __len__(self) -> int: ...
    def __or__(self, other) -> "OrderedIdSet": ...  # step 1 unions two pools
```

Nine lines over a `dict`, which has preserved insertion order since 3.7 and gives
O(1) membership and deletion — the two operations `_move` does most.

**Every pool gets one, not just `WAITING_RESOURCE`.** Spec §3.2 calls
`WAITING_HANDOFF` "an unordered set", and it stays one in the sense that matters:
nothing reads its order. Making the collections uniform costs nothing and avoids
a `dict[TaskStatus, set | OrderedIdSet]` that a reader has to keep straight.

**Spec §3.2 already required this**, and the rev. 10 implementation was the thing
that diverged: "`WAITING_HANDOFF` is an unordered set… **`WAITING_RESOURCE` is
ordered**: it is the single place in the system where a scheduling decision
occurs." Rev. 10 used a `set` and recovered an order by sorting inside the
policy. That worked while the only order wanted was `created_at`; it is what made
depth-first look like a problem about *keys* rather than about *placement*. D20.

### 8.2 The API

| Method | Body |
|---|---|
| `submit(task)` | validate the agent spec, the resource names, and the resource **amounts** (D9) → `task_mgr.add` → `handoff_mgr.declare(task.outputs, task.id, types=task.kinds)` (§3.7 — rev. 11 omitted `types` and every `Handoff.type` was `""`) → `_warn_depends_on(task)` → `_move` to the pool `_ready` dictates → `try_dispatch` |
| `expedite(task)` | reject unless every input passes `check_if_latest_valid` → `task.expedited = True` → `submit(task)` |
| `remove_queued(tid)` | reject unless status in `WAITING` → `_move(CANCELLED)` |
| `stop(tid)` | reject unless status in **`PHASES`** (D19) → `_move(STOPPING)` → `runner.stop(tid, self.on_stopped)` |
| `resume_task(tid)` | reject unless status in `RESUMABLE` → `_move` to the recomputed waiting pool → `try_dispatch` |
| `update_task(tid, **fields)` | `remove_queued(tid)` → `submit(model_copy(old, update={**fields, "status": WAITING_HANDOFF, "history": []}))` |
| `on_task_done(tid, status, usage)` | reject unless status in **`PHASES`** (D19) → release **if this task holds a lease** (§8.3) → read `output_versions` from `handoff_mgr` → `task.close_execution` → `_move(status)` → `try_dispatch` |
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

`resume_task` is *not* named `resume`; spec §5.1 explains why the naming
cannot stand.

#### The blast radius of the two phase states is five sites

Enumerated by `grep -n "RUNNING\|STOPPING" task_graph/*.py` against the shipped
implementation, so this is the complete list rather than an estimate. Each has a
determinate answer already in the spec:

| Site | Rev. 10 | Rev. 11 | Spec |
|---|---|---|---|
| `stop` guard | `{RUNNING}` | `PHASES` | §3.2: "`stop()` is accepted in all three phase states" |
| `on_task_done` guard | `{RUNNING}` | `PHASES` | §5.1 rejects "Task not in a phase state" |
| dispatch's `_move` | `RUNNING` | `INPUT_VALIDATING` | §3.2: `WAITING_RESOURCE → INPUT_VALIDATING` |
| recovery demotion | `RUNNING → WAITING_RESOURCE` | all three → `WAITING_RESOURCE` | criterion 8: "All three lease-holding phase states demote identically, because the lease is gone in each" |
| recovery demotion | `STOPPING → SUSPENDED` | unchanged | §6.4 |

**Dispatch now lands a task in `INPUT_VALIDATING`, not `RUNNING`.** The task
reaches `RUNNING` when the runner calls `enter_phase` — which is the point of the
two statuses: spec §3.2.1, "without them, 'where is this task' has no answer
during the phase that most often blocks."

`STOPPING` is unchanged and still transient across all three, since spec §3.2
says all three are "a running task from the outside".

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

        # 3. all-or-nothing: verify the FULL set before mutating anything.
        #    A non-leaf declares nothing (criterion 53), so this loop is empty
        #    for it and it acquires nothing — the hold-and-wait invariant.
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
        self._move(tid, INPUT_VALIDATING)             # _move persists both
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

#### Only a leaf acquires, and the rule is enforced at load rather than here

Step 3 needs no `if task.is_leaf` branch, and that is the design. **A non-leaf's
`resources` is empty because a non-leaf declaring `resources` was rejected at
load** (criterion 53, §8.7), so the all-or-nothing loop runs over an empty dict
and takes nothing. Spec §6.2's invariant then holds by construction:

> the only thing that holds is a leaf; a leaf has no subtasks; so nothing that
> holds is ever waiting on something that must first hold.

A branch here would express the same thing twice and put the load check's
correctness at the mercy of a runtime test agreeing with it.

**This is the failure Airflow shipped, diagnosed, and eventually deleted.**
`SubDagOperator` was a real task instance holding a worker *and* a pool slot
while its inner tasks queued for the same pool; its own docstring conceded it
"can occupy a pool/concurrency slot… to avoid potential deadlock", #14338 is the
user report — "the scheduler thinks there is a task running in the dag… so it
won't schedule any more tasks" — and PR #41390 removed it in 2024, 3918 lines.
Prefect has the same deadlock, reproduced, and declined to fix it: PR #21800,
"It does not change the deadlock semantics — it just makes them debuggable."
Dagster is the shape adopted here — a `@graph` produces **no execution step at
all**, measured as 4 nodes → 2 steps.

**Airflow also shipped a load-time check for this and it is too narrow, which is
the argument for the blanket rule.** `_validate_pool` filters `Pool.slots == 1` —
so a two-slot pool with a parent holding one and a child needing two deadlocks
and parses clean. Airflow detects *a* deadlock; criterion 53 forbids *the class*.

Temporal is the one genuinely different answer and it is unavailable here: a
worker slot is held for a Workflow Task — milliseconds — not for the workflow's
life, so a parent awaiting a child occupies nothing. Our leaf holds a GPU lease
across all three phases precisely because the work is long. Named as the road not
taken so nobody re-proposes it.

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
    self.pools = {s: OrderedIdSet() for s in TaskStatus}   # NOT set() — §8.1.1
    for task in self._r.get("task_mgr").all():
        status = {TaskStatus.RUNNING:  TaskStatus.WAITING_RESOURCE,  # lease is gone
                  TaskStatus.STOPPING: TaskStatus.SUSPENDED,         # runner is gone
                  }.get(task.status, task.status)
        self._move(task.id, status)
    self.try_dispatch()
```

**The pool type is the correction rev. 12 makes here.** Rev. 11 rebuilt the pools
as plain `set()`, which is what `__init__` did before §8.1.1 introduced
`OrderedIdSet` — a line that was right in rev. 10 and was not updated. A `set`
after every restart destroys promotion order, so `DepthFirstPolicy` silently
degrades to whatever iteration order the set happens to give, on exactly the path
where nobody is watching. That is the same failure §8.1's early return exists to
prevent, arriving through the recovery path instead of the dispatch path, and
§11's "two passes preserve promotion order" test would not catch it because it
never restarts.

Eligibility is not restored — it is recomputed by `try_dispatch`, which is why
`HandoffMgr` must have resumed first. Criterion 25 asserts that failure directly:
resume the scheduler against an unresumed `HandoffMgr` and every waiting task
stays blocked, with no later event to unblock it.

### 8.5 Unfolding a subgraph

Spec §3.2.1's phase table says who runs the main phase: "**The scheduler**, if
there is a subgraph; the runner, if this is a leaf."

So for a non-leaf, `enter_phase(RUNNING)` is where the expansion is instantiated:

```python
def unfold(self) -> list[Task]:          # on Task, §3.4
    """Instantiate the declared expansion of `self.closure`, parent = self.id.
    Returns the subtasks; the caller submits them. Rejects a task whose
    `closure` is None, and a closure absent from the registry."""
```

Each subtask is an ordinary `Task` — same states, same dispatch, same audit
record — carrying `parent = self.id` and the `is_start` / `is_end` marks its
closure declares. They are submitted through `Scheduler.submit`, which is what
puts them through `TaskMgr.add` and therefore through the indexes (§6.2.1) and
the registry supply (§3.4).

**The parent stays `RUNNING` while its subgraph runs, holding nothing** — no
resources and, since rev. 14 of the monitor spec, **no thread either**: its
attempt's thread ends at `unfold` and a new one is taken later for phase 3.

**What it does *not* do is go straight to `SUCCEEDED`.** Rev. 11 of this document
said it "reaches `SUCCEEDED` when its `is_end` subtask completes", and that
sentence broke spec §3.2.1 twice:

| | |
|---|---|
| It skipped `OUTPUT_VALIDATING` | The phase chain is `RUNNING → OUTPUT_VALIDATING → SUCCEEDED`, and a non-leaf has both validation phases — §3.2.1 discusses them for the non-leaf case explicitly ("For a non-leaf they hold nothing") |
| It had the scheduler act on `is_end` | §3.2.1: the markers "are not gates", and the scheduler "**does not treat `is_end` specially at completion**" |

**The corrected route puts the scheduler outside it entirely.** `is_end`'s
completion is reported by that subtask's monitor to the *parent's* monitor, which
calls `parent.enter_phase(OUTPUT_VALIDATING)` and asks the runner for a thread —
spec §3.2.1's "how a non-leaf gets back to its output validation", mechanism in
[`../../monitor/docs/spec.md`](../../monitor/docs/spec.md) §5.3. Nothing here
observes another task's status, which is what principle 4 forbids and what the
withdrawn sentence required.

`is_end` still earns its place, and for what §3.2.1 actually says: *"'has this
subgraph finished' … which display, progress reporting, and the parent's own
completion accounting all need and which is otherwise a search over the subtask
set."* **Answering the question is not the same as the scheduler acting on the
answer**, and the withdrawn sentence conflated the two.

**Unfolding does not violate "a subgraph is declared, never generated."** Spec
§3.2.2 forbids a task *inventing* a subtask nobody declared; `unfold` instantiates
what the closure already names. That is main spec §6.1's framing exactly — the
catalogue is static, the instance count need not be — and it is the same
mechanism criterion 51 governs from the other entrance, `replace_with`.

**So `unfold` raises rather than improvising, and the improvisation has a
different exit.** Spec §3.2.2: "What a task can do when it finds it needs an
undeclared step is report it, **through the risk exit**, and let a human amend
the recording." That exit is not this module's — nothing in `task_graph` opens
it — and the design's obligation is only to leave the failure loud enough to be
reported: `unfold` on a task whose `closure` is `None`, or whose closure is not
in the catalogue, raises with the task id and the known names, exactly as
`replace_with` does.

**Where the ordering decision actually happens is not here.** A chained subgraph
`P1 → P2 → P3` unfolds all at once, and only `P1` is eligible; `P2` and `P3` land
in `WAITING_HANDOFF`. `P2` reaches `WAITING_RESOURCE` later, when `P1`'s output
becomes valid and step 1 promotes it — and *that* promotion is what puts it at
the top of the pool. Depth-first is a property of promotion order, not of unfold
order (§8.1). Getting this backwards would give a design that unfolds onto a
stack and then loses the order at the first dependency.

**A cycle through `parent` must be impossible.** `unfold` sets `parent` on tasks
it just created, so it cannot close a loop; `replace_with` and any future linking
API can. Argo #16376 is the cautionary case — its node ids are an FNV-32a hash of
the node name, a collision made a node its own ancestor, and the controller stack
overflowed. Our ids are `uuid4` and never name-derived, which removes that
mechanism; the residual risk is an explicit `parent` assignment, and §8.7's load
check covers the declared graph.

### 8.6 Cascading cancel

Spec §3.2.4: "A task maintains its own subgraph's consistency. Cancelling
cascades to its downstream, **level by level**, reporting upward."

#### The discipline is not a free choice — spec §3.2.4 already made it

Spec §3.2.3 leaves the design "either the work is queued and drained at the top
of the call, or the recursion is explicitly bounded — the design stage picks one,
and the choice is not optional." **spec §3.2.4's "level by level" picks it.** Measured
on a diamond `A → {B, C, F}`, `B → D`, `C → D`, `D → E`:

```
recursion (depth-first) : ['A', 'B', 'D', 'E', 'C', 'F']
drained queue (level)   : ['A', 'B', 'C', 'F', 'D', 'E']
```

Level-by-level *is* the drained queue; recursion produces depth-first order, and
the two differ observably on any graph that is not a chain. So the design does not
weigh two equal options — it records that one of them contradicts a sentence in
the spec.

The depth measurement is the secondary argument and it is still worth stating: a
recursive cascade costs four stack frames per level in an optimistic modelled
shape and dies at a chain of 250 against CPython's default limit of 1000. Real
frames per level are higher — `validate_assignment` runs on each status write,
`_move` persists, the registry lookup adds one. **Airflow left the same reasoning
in a comment**: its relative-walk is "intentionally implemented as a loop, instead
of calling `get_direct_relative_ids()` recursively, since Python has significant
limitation on stack level, and a recursive implementation can blow up if a DAG
contains very long routes."

**And Dask is this design written the obvious way, broken.** `stimulus_cancel`
recurses over its reverse index with no visited set and no bound; driven on a
live cluster it raised `RecursionError` inside the scheduler's event loop **while
the client printed `cancel OK`** — the caller was told the cascade succeeded over
a silently half-cancelled graph.

#### The walk

```python
def _drain_cascade(self) -> CascadeReport:
    seen: set[TaskId] = set()
    reached, refused = [], []
    while self._cascade:
        tid, reason = self._cascade.popleft()      # level order (FIFO)
        if tid in seen:
            continue                               # a diamond, not an error
        seen.add(tid)
        task = task_mgr.get(tid)                   # re-read; never trust the index
        if task.status not in WAITING:
            refused.append((tid, task.status))     # §14 O14 — not this design's call
            continue
        self._move(tid, CANCELLED)
        reached.append((tid, reason))          # the reason travels in the report
        for consumer in task_mgr.consumers_of_outputs(tid):
            if self._same_graph(consumer, tid):    # criterion 49
                self._cascade.append((consumer.id, f"upstream {tid} cancelled"))
    return CascadeReport(reached=reached, refused=refused)
```

Four things in that body are the design, and each is there for a measured reason.

**The visited set is not hygiene.** On the same diamond with no seen-set, `D` and
`E` are reached twice — ordinary for a DAG walk, except that spec §3.2.3 gives
`cancel()` the precondition "a waiting state", and after the first visit `D` is
`CANCELLED`, which is not one. **A diamond cascade with no dedupe raises**, on a
graph shape that is not exotic. Airflow's loop carries the same `if task_id in
relatives: continue`.

**The re-read is Kubernetes' shape.** The index finds candidates; the
authoritative status comes from `TaskMgr` immediately before acting. K8s says why
in its own source — `getDependents` "does not provide any synchronization
guarantees; items could be added to or removed from ownerNode.dependents the
moment this function returns" — and compensates by always reaching the API server
before the destructive act.

**The queue is separate from `_dispatch_again`.** Akka drains system messages to
empty before *every* user message — "don't ever execute normal message when system
message present!" — and a cancel cascade is structurally a system message. Mixing
it into the dispatch trampoline would also hit that trampoline's real limitation:
**`_dispatch_again` is a boolean, not a queue**, so it coalesces N requests into
one extra pass. Coalescing is right for "re-check eligibility", which is
idempotent; it is wrong for "cancel these seven specific tasks".

**`reason` travels with each entry, and it goes in the report — not onto the
task.** Rev. 11's body wrote `task.cancel_reason = reason`, and there is no such
field: §3.2's rev. 11 listing does not have one, and `Model` sets `extra="forbid"`
with `validate_assignment=True`. Measured against the shipped model:

```
$ t = Task(agent_spec='x'); t.cancel_reason = 'boom'
ValidationError: 1 validation error for Task
fields: ['agent_spec','created_at','depends_on','expedited','history','id',
         'inputs','outputs','resources','status']
```

**The cascade raised on its first entry.** Found in the stage-three consistency
pass. Adding the field was the other repair and it is rejected: a reason on the
task would be a second record of something the report already carries, and
`CascadeReport` is what the caller gets. The survey below is what makes that the
right direction rather than merely the cheaper one — a *state plus a reason per
unit* is the shape that survives an incomplete cascade, and `reached` is that
list.

Of the three
upward-reporting shapes in the survey — enumerate-before-acting, aggregate-after,
or a state plus a reason on each unit — the third is the only one that survives an
incomplete cascade, and incompleteness is guaranteed: **no surveyed system makes a
cascade atomic.** Argo writes a reason string per node; Airflow marks
`RESTARTING`; Kubernetes sets `deletionTimestamp`. Aggregation was measured lossy
even in the standard library — two children raising inside an
`asyncio.TaskGroup` or a trio nursery produce an `ExceptionGroup` carrying one,
with the cascade's own `CancelledError` absent entirely.

`CascadeReport` goes to the **caller**, not to an ancestor. Airflow's `dry_run` is
structural about this: the same code computes the affected set and acts, and the
report is returned to whoever asked. **A parent veto over a running cascade is
unprecedented** — veto exists everywhere but always as child-over-parent
(Kubernetes' `BlockOwnerDeletion`), self (trio's `shield`, asyncio's `uncancel`),
or parent-declared-in-advance (Temporal's `ParentClosePolicy = ABANDON`). No
surveyed system consults an ancestor at cascade time. Spec §3.2.4's "reporting
upward" is read here as *returning a report*, and §14 O15 says why the stronger
reading is left alone.

**Cancel is not invalidation, and criterion 52 asserts it directly**: nothing in
this walk touches a handoff version or a validation record. Spec §3.2.4 draws
that line, and it is worth knowing the line is ours — **Airflow actively
conflates them.** One `clear` both terminates a `RUNNING` task and resets a
`SUCCESS` task so it re-runs, separated only by caller booleans `only_failed` /
`only_running`, **neither set by default**. The nearest support is Temporal's
cancel/terminate split: two operations, different authority, both about execution
and neither about content.

#### `replace_with`

`replace_with(closure_name)` is `cancel()`'s cascade followed by `unfold` of a
named closure. Criterion 51 requires the closure be declared:

```python
closures = self._registry.get("closures")        # a Task may; the scheduler may not (§3.4)
if closure_name not in closures:
    raise TaskStateError(f"{closure_name!r} is not a declared closure; "
                         f"known: {sorted(closures.names())}")
```

Enumerating the candidates follows module 1's error-message rule. **Without this
check `replace_with` is the hole through which the whole record-and-replay
constraint is bypassed** — spec §3.2.4 says so in those words.

Its containment claim — cancel this graph's downstream "without propagating
outside" — is sound only if a subgraph's handoffs do not escape it, which is
exactly what §8.7 checks at load.

### 8.7 The graph-level load checks — `graph.py`

Criteria 50 and 53 both say "at load", and `task_graph` has no load step:
`submit` is the entry point and it sees one task at a time.

**Neither neighbour will take it.** `closure` spec §4.1: "**Whether the closures
compose into a valid graph.** A closure is one step; a graph is many… A partial
version here would put the check in two places and satisfy neither." Main design
§6.3: "This design does not smuggle a partial version in… **it is still not this
pass's.**"

And `task_graph` spec §3.2.4 misattributes where it went: it says the check is one
"`closure` spec §4.1 defers to 'the system whole task'", but read whole, closure
§4.1 defers it to *nobody* — the phrase appears at `closure/docs/spec.md:242`
inside an **open question**, "it has to live somewhere. The likely home is the
system whole task." **The two documents that touch this check each think the other
is holding it.** D21.

This design claims it, on the argument that the graph is this module's subject.

```python
def check_graph(specs: Mapping[str, TaskSpec], *,
                skip: Set[str] = frozenset()) -> list[Problem]:
    """Every graph-level check, over the declared catalogue. Returns problems;
    raises nothing. Called by the composition root before the scheduler exists."""
```

**It returns `spec_loader`'s `Problem`, not a type of its own.** Rev. 11 wrote
`list[GraphProblem]`, and the composition root would then have collected two
differently-shaped problem lists from two passes running back to back and
reported them in one place. One report format across every load-time check is
main design §6.2's rule and `closure` design §10 already adopts it; a second shape
here would have made the whole-run error output depend on which pass found the
fault.

`skip` arrives for the same reason `check_closures` takes one: a task spec whose
closure already failed should not be walked for containment, because its handoff
names may not resolve (`closure` design §7.2).

**`Problem` and `TaskSpec` both come from `spec_loader`**, which `graph.py` may
import — it is the leaf every module depends on (main design §2.3), and importing
it creates none of the edges §2 forbids.

**It runs over task *specs*, not over `Task` objects, and that is forced.** The
catalogue is static (main spec §6.1) while `submit` accepts tasks at any time, so
a runtime pass could never see a complete graph. Main design §7's composition root
loads packages and *then* registers the scheduler — "a graph cannot be assembled
from specs that have not been admitted" — which is the only moment when every spec
is present and nothing has run.

Two checks:

| # | Check | Criterion |
|---|---|---|
| 1 | A task declaring a subgraph declares no `resources` | 53 |
| 2 | No handoff produced inside a subgraph is consumed outside it, except through the end entry subtask's outputs | 50 |

**Check 1's message copies Airflow's shape** — name both sides, end with the
consequence:

```
task 'optimize_kernel' declares resources {'gpu': 2} and expands into a subgraph
of 4 tasks. Only a leaf may acquire; a parent holding a lease while its subtasks
queue for the same pool is the deadlock this rule prevents. Move the declaration
onto the subtasks that do the work.
```

**Check 2 has no precedent as a check, and the reason is worth recording.**
Containment is enforced statically everywhere it exists — Dagster at definition
time, Argo at submit/lint — but always in the *other* direction: the inner name
simply does not exist outside, so the violation is unrepresentable. Dagster is
the sharpest illustration and its asymmetry is instructive: a declared `GraphOut`
left unmapped **errors**, while an inner output nobody mapped is **legal and
silently discarded**, and consuming one from outside fails as a type error about
`None` rather than a containment message.

**Ours is representable because handoff ids are global** — `HandoffMgr` is one
flat namespace (spec §3.1), so an outside task can name an inner handoff and the
lookup succeeds. That is a consequence of a §3.1 design choice, not an accident,
and check 2 exists to close it. The alternative *with* precedent — scope handoff
ids to the subgraph and require re-export — is a spec question, O18.

### 8.8 `bootstrap.py` — the composition root

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

### 8.9 What this module owes the monitor, and what it does not

Spec §3.5 rev. 13 brings the monitor into the alpha. **Almost none of it is this
module's**, and saying which part is matters, because the monitor is the second
thing in the system with a loop of its own.

| | Owner |
|---|---|
| The monitor's loop, its `set_task`, its two queues, its two kinds | the monitor. Not designed here |
| **The verbs it calls** | **this module.** `cancel`, `restart`, `fail`, `replace_with` (§3.4) are already the whole action set |
| **`enter_phase`, which is now also a monitor caller** | **this module**, §3.4. Rev. 14 of the monitor spec makes the *planned* advance a monitor action too, and it is the same shape: a transition it calls, never a status it assigns |
| **That those verbs are safe from another thread** | **this module.** §9 |
| `Task.monitor_spec`, and resolving it by name | §3.8 |
| **`Task.parent`, readable** | **this module**, §3.3. Read by the monitor for escalation and for the non-leaf re-entry (§8.5). It stays in the "structure only — nothing in scheduling reads these" category: **the monitor is not scheduling** |

**The authority rule is what makes a second loop cheap rather than dangerous.**
Spec §2 principle 4 and §3.2.3: every monitor action is a transition it *calls*,
never a status it *assigns*. Every transition goes through `_move`, `_move` is the
single writer, and `try_dispatch` holds the scheduler's `RLock`. So a monitor
thread mutates nothing directly — it observes, and it calls the same verbs an
operator would, blocking on the lock for the duration and holding nothing between
calls.

That is not a new concurrency model. §9's table already contemplates the case:
*"An async runner calls `on_done` from its own thread → Blocks on the lock until
the current pass finishes."* A monitor is the same shape with a different caller.

**One thing rev. 12 does not build and does not pretend to**: nothing here polls
anything. `Task.status` is readable, `TaskMgr.by_status` is a scan at this scale
(§6.2.1), and that is the whole of what a monitor needs from this side.

**And one thing rev. 14 of the monitor spec deliberately does not ask for.** The
non-leaf re-entry (§8.5) does **not** route through the scheduler: a task in
`OUTPUT_VALIDATING` is not in the `WAITING_RESOURCE` pool that `_dispatch_pass`
scans, and making it dispatchable would mean the scheduler deciding one task's
progress by observing another's. **The monitor calls `enter_phase` and asks the
runner for a thread; this module is not in that path** — which is why principles 2
and 4 need no amendment for any of rev. 14.


---

## 9. Concurrency

The task definition asks for the most optimistic simple implementation and no extra code
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

### 9.1 Transition re-entrancy — rev. 11

Spec §3.2.3 states the problem and hands it to this stage: "A transition calls
the scheduler, which dispatches, which completes a task, which fires a
transition… Either the work is queued and drained at the top of the call, or the
recursion is explicitly bounded — **the design stage picks one, and the choice is
not optional.**"

**Queued and drained.** §8.6 gives the measurement that decides it and §8.6's
walk is the implementation. Two things this section adds.

#### The two queues are separate, and coalescing is right for one and wrong for the other

| | `_dispatch_again` | `_cascade` |
|---|---|---|
| Type | `bool` | `deque[(TaskId, reason)]` |
| Semantics | "make another pass" | "cancel these, in this order" |
| Coalescing N requests into one | **correct** | **loses work** |

`_dispatch_pass` recomputes eligibility for every waiting task from scratch, so
two requests to dispatch are indistinguishable from one — a boolean is exactly
right, and rev. 10's trampoline stays as it is. A cascade entry names a specific
task and a specific reason; collapsing two into one drops a task from the walk.

This is also why the cascade is drained **before** the dispatch trampoline
resumes. Akka's mailbox does the same and says why in a comment — "don't ever
execute normal message when system message present!" — and a cancel is
structurally a system message: it changes what work exists, and dispatching
against a set that is about to shrink is wasted at best.

#### What is deliberately *not* built, and why it is recorded

- **No depth bound.** With a drained queue there is no stack to overflow, so a
  bound would be a policy limit rather than a safety one. The only clean numeric
  precedent found is SQL Server's 32 nested trigger levels, and **exceeding it
  fails silently** — the trigger just terminates, with no error number located.
  No surveyed source gives a *derivation* for its number. A bound that fails
  silently hides the bug it was added to expose, so none is invented here.
- **No backoff.** Kubernetes' answer to a non-terminating drain is exponential
  backoff plus a rate limiter, not detection. There is no analogue here and no
  measurement saying one is needed. O17.
- **No cycle detection.** systemd is the third option the spec's binary framing
  omits — it recurses but uses a `generation` marker and, on a cycle, logs "Found
  ordering cycle" and deletes a job to break it. §8.6's visited set makes a cycle
  terminate rather than diverge, so the cascade is safe without detection; what it
  does *not* do is tell anyone the cycle exists. O17.

#### The transitions are the event bus's future attachment points

Spec §8.1 defers an event bus and says what is left in its place: "A hook
callback is left at each transition point, and the registry (§4.1) gives a later
bus a natural place to live."

Rev. 11 multiplies the transition points — six on `Task` plus the phase
advances — and it would be easy to read that as six new places a bus has to be
wired. It is the opposite: **every one of them funnels through `_move`**, which
was already the single writer and is still the only place a status changes. A bus
attaches there and sees everything, including the cascade, without any transition
knowing it exists. Nothing is built, and nothing needs to be.

#### One existing correctness property is now load-bearing enough to test

`_in_dispatch` is set inside a `try/finally`. SQLAlchemy #13485 is the failure
mode if it is not: `_bulk_save_mappings` sets `_flushing = True` outside its
`try/finally`, and one failure leaves the Session permanently unusable — neither
`rollback()` nor `close()` clears it. The rev. 10 implementation is already
correct here; §11 adds a test so it stays that way, because a guard flag that
sticks is indistinguishable from a hung scheduler.

**And a repeat request that produces no change is the danger, not recursion.**
Kubernetes #77081: a second foreground delete on an object already holding the
finalizer left the garbage collector doing nothing at all — "no logs, nothing" —
until the controller manager restarted. The cascade queue is drained
unconditionally at the top of `try_dispatch` rather than being conditioned on
anything having changed, which is what avoids that shape.

---

## 10. Build versus adopt

The task definition requires recording, per module, which library was chosen and
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
| `agent` | any agent framework | own implementation | Two methods, both trivial. The task definition leaves agent internals empty on purpose. |
| `runner` | Claude Code / Codex / Cursor CLIs, `subprocess` | Protocol + a fake | The real implementations are harness-specific and out of scope (spec §1.2). What this system owes is the seam. |
| `policy` | `graphlib.TopologicalSorter`, `networkx`, OR-Tools | `sorted()` | Spec §9 records the rejections. FIFO is one `sorted` call; the composite priority rule from the prior art is a drop-in replacement behind the same Protocol. |
| `scheduler` | Prefect, Hatchet, Temporal, Ray, Airflow, Slurm | own implementation | Spec §9, from the prior-art survey. Every candidate is a platform whose scheduling core is not separable. |
| tests | — | `pytest` | Already a dev dependency of the repository. |

Revision 11 adds four modules and adopts nothing new:

| Module | Considered | Chosen | Why |
|---|---|---|---|
| `permissions` | a shared type in `env_mgr`, a `TypedDict`, a bare `dict` | two small pydantic models | The type must live where neither package imports the other (§3.5), which rules out the first. A `dict` would work — `task_graph` never interprets it — but `closure`'s load check 6 has to ask "does this cover that handoff, for reading or writing", and a method on a model is a better home for that question than a convention about keys |
| `ordered` | `collections.OrderedDict`, `sortedcontainers`, a plain `list` | a nine-line wrapper over `dict` | `dict` has preserved insertion order since 3.7 and gives O(1) membership and deletion, which are the two operations `_move` does most. `OrderedDict` adds `move_to_end` and a doubly-linked list this design never needs; a `list` makes `discard` O(n); `sortedcontainers` sorts by a key, which is the thing §8.1.1 is deliberately not doing |
| `graph` | `networkx`, `graphlib` | own implementation | Spec §9 rejected both for scheduling and the rejection holds here. Check 2 is one pass over declared inputs and outputs grouped by `parent` — a dict of sets, not a graph algorithm. Importing a graph library for it would invite modelling the catalogue as a graph object that then has to be kept in sync |
| cascade | `graphlib.TopologicalSorter`, a recursive walk | a `deque` | The order required is level-by-level, which is `popleft` (§8.6). A topological sorter answers a different question and, per spec §9, "`add()` raises after `prepare()`" — this graph grows. Recursion is the option the spec's own wording excludes and that Dask demonstrates breaking |

**pydantic v2 is adopted; everything else is standard library.** For the rest,
every candidate is either a platform (adopt the server to get the primitive) or a
library for a problem this system does not have — graph traversal, dependency
injection. The named upgrade path is `sqlite3` for the store, and it sits behind
an interface that already exists.

---

## 11. Test plan

`pytest`. Run with `pytest agent_sys/tests/task_graph` from the repository root,
or `pytest agent_sys` for both components. The repository's root
`testpaths = ["tests"]` only supplies a default when no path is given, so a bare
`pytest` at the root still collects exactly the existing suite and nothing
changes for it.

The package is declared in `agent_sys/pyproject.toml` and reached by
`pip install -e agent_sys`. The repository's own `pyproject.toml` is untouched:
`[tool.setuptools.packages.find] include = ["infera*"]` does not cover
`agent_sys`, and does not need to.

`agent_sys/tests/task_graph/` gets an `__init__.py` so pytest's `prepend` import
mode does not put the test directory itself on `sys.path` — and so its module
basenames cannot collide with `tests/env_mgr/`'s.

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

### 11.1 Criteria 36–54 — rev. 11

Five new files. Criteria 1–35 keep the files above; every one of them must still
pass unchanged, which is what makes this an increment rather than a rewrite.

| File | Covers | Criteria |
|---|---|---|
| `test_subgraph.py` | `parent` agreement and the single `parent=None`; `is_start` / `is_end` observability; a leaf is its own start and end; unfold instantiates a declared closure and submits its subtasks | 36, 37, 38 |
| `test_phases.py` | the three phase states in sequence; a single lease across them; a skipped phase advances and is reported; `stop` accepted in all three; recovery demotes all three identically | 39, 40, 41, 8 (extended) |
| `test_structure_blind.py` | blanking `parent`/`is_start`/`is_end` changes no order and no pool; depth-first vs FIFO; permissions versioned and inherited | 42, 43, 44 |
| `test_transitions.py` | the status-write spy; `try_dispatch` origin; the monitor's actions as transitions; the AST check for the scheduler import | 45, 46, 47, 48 |
| `test_cascade.py` | cascade within a graph and no further; the load checks; `replace_with`'s closure check; no validation record changes; leaf-only acquisition; parent/child on one pool | 49, 50, 51, 52, 53, 54 |

Six of these carry more weight than their line count, and three of them assert
something a naive test would get wrong.

**Criterion 42 — `test_blanking_structure_changes_nothing`.** Run a fixed
scenario, record the dispatch order and the pool membership after each step; then
rebuild the same graph with `parent=None`, `is_start=True`, `is_end=True` on
every task and assert both records are identical. This is criterion 31's
instrument, reused. It passes trivially under §7.2's policy because nothing reads
the fields — which is the point, and a policy that started reading `parent` would
fail here rather than in review.

**Criterion 43 — and the trap in it.** The scenario is a parent whose subgraph is
dispatchable and an unrelated sibling of equal age. **A structure-blind key can
appear to pass by accident**: in an early probe two wrong keys passed only
because their tiebreaker was `t.id` and the unrelated task's name happened to
sort late. **The test must name the unrelated task so that it sorts *first* under
any tiebreak**, and assert the subgraph runs anyway. Without that, the test
certifies the bug.

A second case is needed and is the one the criterion's wording does not reach:
`P1 → P2` inside a subgraph, with an unrelated `Q` submitted *while `P1` runs*.
When `P1` completes, depth-first must pick `P2`. That is the case where LIFO on
`created_at` fails (§7.2), and without it the suite accepts a policy that
abandons a subgraph mid-way.

**A third, and it guards §8.1's early return.** Run at least two dispatch passes
and assert the promotion order survives the second. Without the early return in
`_move`, step 1 re-appends every waiting task each pass and the order is
destroyed silently — dispatch still works, nothing else fails, and depth-first
quietly stops happening.

**Criterion 45 — the status-write spy.** `Task.status` is a pydantic field with
`validate_assignment=True`, so the assignment is observable through a field
validator installed for the test. The assertion is on *origin*: every write must
occur inside a `Task` transition or inside `_move` called from one. The existing
`test_authority.py` marker-span pattern is the instrument — brackets written into
the same log — rather than stack inspection, because a marker records who claimed
responsibility while a stack frame only records who was on it.

**Criterion 48 — and why it is not a `grep`.** `test_authority.py`'s existing
static half is a substring check over `inspect.getsource`, and it is honest about
why ("no test can observe it"). It does not generalise here: criterion 48's
tokens are `scheduler` and `try_dispatch`, which appear throughout prose and
comments. Measured on the sibling case — `"scheduler" in runner.py` is **`True`
today**, from two docstring mentions alone, while an AST walk over names,
attributes and imports returns **0**. So criterion 48 walks `ast.parse(models.py)`
for `Import` / `ImportFrom` / `Attribute` / `Name` nodes naming the scheduler
module. Copying the grep would produce a test that fails for the wrong reason,
or passes for one.

**Criterion 53 — two halves, and the second is the interesting one.** That a
non-leaf declaring `resources` is rejected at load is a `graph.check_graph` unit
test. That a parent holds nothing while its subgraph runs is asserted by
snapshotting every pool at each of the parent's phase transitions and requiring
them unchanged — the same instrument criterion 40 uses for a leaf's single lease,
pointed at the opposite expectation.

**Criterion 54 — the deadlock that is supposed not to happen.** A pool sized to
satisfy the parent *or* the child but not both; assert the graph completes. It
passes because the parent declares nothing, so there is no "both". Worth writing
as a test that would hang under the rev. 7 model rather than as an assertion about
a counter, since hanging is what the rule prevents.

**One test with no criterion**, and it is the re-entrancy guard: assert that a
`try_dispatch` whose inner pass raises still leaves `_in_dispatch` false. §9.1
says why — SQLAlchemy #13485 is one failure permanently bricking the object, and
a stuck flag here is indistinguishable from a hung scheduler.

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

Scratch experiments stay in `agent_sys/scratch/`, which is gitignored. No
temporary experiment leaves it.

### 12.1 Rev. 11's order — an increment over a green suite

Steps 1–12 are done. **The rule for every step below is that the 358 existing
tests stay green after it**, not merely at the end; a step that needs them
changed is a step that has misunderstood something.

| # | Step | Depends on | Why here |
|---|---|---|---|
| 13 | `permissions.py`, and `Permissions` on `Task` | 2 | Leaf-most new module, carried and never read. Nothing else needs it, so it cannot break anything |
| 14 | `ordered.py` + `_move`'s early return | 3, 10 | **Before** anything reads order. Its own test is "two passes preserve promotion order", which fails today and passes after |
| 15 | `TaskStatus` +2, `PHASES`, the five guard sites | 2, 10 | Mechanical, and the five sites are enumerated in §8.2. The suite catches any missed one |
| 16 | The structural fields; `TaskMgr.children`; `unfold` | 13, 15 | Structure before behaviour. Criterion 42's blanking test is written here and must pass immediately |
| 17 | `enter_phase`, `FakeRunner`'s three phases | 15 | Criteria 39–41 become testable only now |
| 18 | `DepthFirstPolicy`, and the default swapped | 14, 16 | Needs the ordered pool. One existing test is rebuilt — see below |
| 19 | The registry reference, `_sched`, `cancel` / `restart` / `fail` | 16 | The transitions, before anything cascades |
| 20 | `TaskMgr.consumers`, the cascade queue, `_drain_cascade` | 19 | Criterion 49, as far as §14 permits |
| 21 | `graph.py`, wired into the composition root | 16 | Criteria 50 and 53. Last because it checks what the earlier steps made expressible |
| 22 | `replace_with` | 20, 21 | Needs both the cascade and the closure catalogue |

**Step 18 changes one existing test, and only one.** Measured by overriding the
default policy and running the suite: **1 of 358 fails**, and it is D15's
regression test, whose *setup* depends on which task a single dispatch pass
reaches first — not its assertion. It is rebuilt to pin the order it needs
explicitly instead of relying on the default. Recorded because "the default
policy changed and one test moved" is the kind of thing that looks like a
regression a year later.

Steps 19–22 are where the interesting work is, exactly as 10–12 were in rev. 10.
13–18 are deliberately dull.

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
| D19 | rev. 10 of *this document* argued "There is no `LIVE` or `FINAL` set: every other guard tests a single status" | `PHASES` is added, as an **ordered tuple** | The premise expired with spec rev. 9. Three guards now test the same three statuses — `stop`, `on_task_done`, and recovery's demotion — and a literal repeated three times is what the constant prevents. It is a tuple rather than a `frozenset` because `PHASES[i+1]` is "the next phase", which `enter_phase` uses to reject a skip; a set would make the membership tests read identically and silently lose the sequence. Recorded rather than changed silently, because rev. 10's reasoning was right when written |
| D20 | spec §4.4 types the index `pools: dict[TaskStatus, set[task_id]]`, and rev. 10 implemented that, recovering order by sorting inside the policy | every pool is an `OrderedIdSet`, and `_move` preserves a task's position when its pool does not change | **The spec contradicts itself here and this resolves it toward §3.2**, which says "`WAITING_RESOURCE` is ordered: it is the single place in the system where a scheduling decision occurs." §4.4's `set` is the weaker statement — it is describing the index's *shape*, one bucket per status, not arguing for unordered — while §3.2's sentence is load-bearing for criterion 43. Recorded as a spec defect in D24 as well as a deviation here. Sorting a `set` inside the policy worked while the only order wanted was `created_at`, and it is what made depth-first look like a problem about ordering *keys* rather than about *placement*. Two keys were designed and discarded before the collection was questioned — a policy reading `parent` (satisfies criterion 43, fails 42) and LIFO on `created_at` (passes criterion 43's example, abandons a subgraph mid-way on a measured counter-case). A third, a stamped `ready_since` field, is correct but adds a mutable derived field to express what the collection already knows. **`_move`'s early return is the load-bearing line**: without it, step 1 re-appends every waiting task on every pass and the order dies silently on the pass after it is set |
| D21 | spec §3.2.4 says the subgraph boundary check is one "`closure` spec §4.1 defers to 'the system whole task'" | this design claims it, in `graph.py`, run by the composition root | The citation does not hold. Read whole, `closure` §4.1 defers graph-level checks to **nobody** — it names only `task_graph` §10's cycle detection — and the phrase "the system whole task" lives at `closure/docs/spec.md:242` inside an **open question**: "it has to live somewhere. The likely home is the system whole task." Main design §6.3 declines it explicitly too: "it is still not this pass's." **The two documents that touch this check each think the other is holding it**, and criterion 50 requires it to exist. Claimed here on the argument that the graph is this module's subject; the spec sentence needs correcting either way |
| D22 | spec §3.2.1 says "**Four** fields carry the structure" | three are listed and three are designed — `parent`, `is_start`, `is_end` | The table immediately below that sentence has three rows, and no fourth field is named anywhere in §3.2.1. The sentence continues "They join `depends_on` in the category §3.2 already establishes", which suggests the count absorbed `depends_on` and then the table did not. Nothing depends on the resolution; recorded so a reader does not go looking for a missing field |
| D23 | ~~spec §3.2's `Task` field table has fourteen fields and no `closure`~~ **No longer a deviation** — spec §3.2 rev. 13 declares it, and §3.2.5 gives it the wider job of being the link back to the whole task spec | `Task` gains `closure: str \| None` | Criterion 51 requires `replace_with` to instantiate **only declared closures**, and §3.2.1 requires a subgraph to be "declared in the task's spec" — but **the runtime `Task` has no link back to its own declaration.** `agent_spec` names the agent spec and nothing names the task's. So either the catalogue is searched for whichever closure happens to contain this task — a scan whose answer is not unique — or the task carries the name. It carries the name: a `str` resolved against the `closures` registry at use time, never an object, the same discipline `agent_spec` already uses. It is `None` for a task submitted directly rather than instantiated from a closure. **This is the smallest addition that makes criteria 51 and the `unfold` path expressible**, and it is an addition to a spec'd model, so it is declared here rather than left in §3.2 |
| D24 | three places in the spec were not updated when rev. 9 added the phase states and rev. 10 added the cascade | the design follows the section that is current in each case, and names the stale one | **Reported, not fixed — a design does not amend the spec.** (a) **§6.2's dispatch pseudocode still ends `task.status = RUNNING`**, while §3.2's diagram and status table both say `WAITING_RESOURCE → INPUT_VALIDATING`; §8.3 follows the diagram. (b) **§4.4 names the load-bearing pools as `WAITING_HANDOFF`, `WAITING_RESOURCE`, `RUNNING`** — dispatch now lands in `INPUT_VALIDATING`, so the third entry is one member of a three-member phase group rather than the state a task is dispatched into. (c) **§8.1 still lists "The downstream index" under *Not built*** with the justification "Today's cascade-free paths do not need it" — rev. 10 added the cascade and criterion 49 asserts it, so §6.2.1 builds it. Each of the three is a sentence that was true of an earlier revision |

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
| **O4** | **`Handoff.type` has no route from `submit`, and as of rev. 12 something depends on it.** Spec §3.1 gives a handoff a `type`; `HandoffMgr.declare(ids, producer_task_id, types=None)` can carry it; and `Task` has no field naming the types of its outputs, so `submit` calls `declare` without types and the field stays `""`. Rev. 11 closed with *"nothing depends on it today"* — **that is no longer true.** §3.5's `Grant.kind` is a kind name, and `env_mgr` design §6.1 resolves a grant by matching it against `Handoff.type`. An unset `type` therefore matches no grant, and the agent receives an **empty granted set rather than an error** (`env_mgr` O4, from the other side). Three candidate routes: `Task` gains `output_types: dict[HandoffId, str]` and `submit` forwards it (a spec §3.2 change); the graph builder calls `declare(types=...)` itself before `submit` (no spec change, but the correctness of `type` becomes an unchecked convention); or `env_mgr` is handed the builder's kind→id map and stops reading `type` at all. **Reported to the user in the stage-three pass and not answered, so nothing is chosen here.** What *is* done, because it is a design decision and not a spec one: `env_mgr` design §6.1 now **raises** when a grant's kind matches no handoff, instead of returning an empty set — so whichever route is taken, forgetting it is loud. |
| **O7** | **A consumable's capacity and its balance can disagree after a config change.** `resume_system` reads the stored `available` and ignores the `capacity` passed to the constructor, so raising a budget from 1M to 2M has no effect until the record is deleted by hand. That is the right default — the operator did not intend to hand back spend — but it means the constructor argument silently stops mattering. Either log when the two disagree at startup, or add the explicit `refill` that spec §10 already wants. Not built. |
| **O8** | **`_warn_depends_on` reads a version that may not exist yet.** The check looks up each input's `producer_task_id`, which is `None` until the producing task is submitted and `declare`s the slot. Submitting a consumer before its producer therefore warns about nothing — the check silently passes on exactly the graph most likely to be miswired. Re-running it at dispatch would catch that, at the cost of warning repeatedly. Accepted for now: the warning is a convenience, and the scheduling behaviour it guards is unaffected either way. |
| **O9** | **A handoff payload must be JSON-serialisable.** The scheduler is content-agnostic and `test_authority.py` proves it never inspects a payload — but `HandoffMgr.persist` dumps the whole handoff, so an arbitrary Python object raises `PydanticSerializationError` at the agent's `seal`, not at the scheduler. Found while writing the authority test and asserted there rather than left to be discovered by the first agent that returns a live object. The fix is the same one spec §8.2 already leaves open: decide where payloads live, and put a content store behind `Handoff.content`. |
| **O10** | **A version left `GENERATING` by a crash deadlocks its own retry.** Spec §6.4 requires that recovery not re-derive a verdict, and it does not — but `Handoff.open_next` refuses a slot that is already open, and nothing seals the abandoned one: the agent that opened it is dead, and the scheduler is forbidden from writing handoff state. The task is demoted, re-dispatched, and its new agent cannot write its own output. Found by review; asserted as it behaves in `test_recovery.py`. Criterion 20 is therefore satisfied only for the case where *something* closes the abandoned version off, not for a crash. Three candidate fixes, all of them spec decisions: let `open_next` adopt a version whose producing execution is no longer open; have `TaskMgr.resume_system` seal it `INVALID` alongside the execution it already closes (but that is a manager writing handoff state); or state that an operator seals abandoned versions by hand. |
| **O11** | **Criterion 27's static half is not enforced by anything.** The criterion says passing a `TaskId` where a `HandoffId` is expected is "a type error the checker reports", but no `mypy` or `pyright` configuration exists in the repository, so no checker runs in CI. `test_ids.py` asserts the structural fact the claim rests on — three unrelated leaf classes — and the runtime half is thoroughly covered, but the static half is an assertion about a tool nobody invokes. Either add a `mypy --strict` gate over `agent_sys/task_graph/` or reword the criterion. Raised by review. |
| **O12** | **`test_authority.py` cannot see `open_next` or `seal`.** Criterion 14 is worded as "no `open_next` or `seal` is called from a scheduler frame", but those are methods on `Handoff` / `HandoffVersion`, and the spy wraps `HandoffMgr`. What is actually enforced is that `persist` only happens inside an agent span. Since the scheduler does hold live `Handoff` objects (through `get` and `latest`), a scheduler that mutated one and did not persist would pass. The inference is strong — an unpersisted mutation is a bug in its own right — but it is an inference. Closing it means wrapping the returned `Handoff` in the spy, or asserting on version-status transitions rather than on `persist`. Raised by review. |
| **O13** | **`output_versions` is misattributed when two tasks produce the same handoff.** `on_task_done` records `handoff_mgr.latest(h).version`, which is whatever the handoff holds *at completion time* — not what this run's agent wrote. With A and B both outputting `h`: A writes v0, B writes v1, A completes, and A's execution records `{h: 1}` while A's own agent's `handoffs` correctly says `[0]`. Criterion 22's "reconstructible from either end" then disagrees with itself depending on which end you read from. **This is spec-mandated, not an oversight**: criterion 13 requires the scheduler to record `output_versions` "by reading `HandoffMgr`, not from anything the runner passed", and spec §8.2 calls `latest(h)` the authoritative answer. `agent.handoffs` is the accurate source and reading it would not violate the authority rule — it is an agent record, not handoff state — but switching would contradict a criterion, so it is a spec decision. Raised by review; not fixed. |

### 14.1 Raised by revision 11

**Two criteria this revision cannot close, and it says so rather than inventing
the spec material.** Both are named in spec §10 already; what is new is that a
criterion now depends on each.

| # | Question |
|---|---|
| **O14** | **What a cascade does on reaching a `RUNNING` task, and whether a cascade is atomic.** §8.6's walk records such a task in `refused` and moves on, which is the *narrowest* behaviour and therefore the one that presumes least — but it is not an answer, it is a placeholder for one. Spec §10 flags the stakes: "The first changes `cancel()`'s signature if the answer is 'stop it', so it is not a detail." **Airflow is the counter-example to that claim** and is worth putting in front of whoever decides: its `clear` offers exactly two of the three answers behind one caller-supplied parameter — `prevent_running_task` either raises and refuses the whole operation, or sets `RESTARTING`, which is *a request rather than a wait*, so `clear` stays synchronous. "Skip it" is offered by nobody surveyed. On atomicity there is no ambiguity left: **no surveyed system makes a cascade atomic**, and every one of them names the intermediate state — Kubernetes `deletionTimestamp`, Kotlin `Cancelling`, Airflow `RESTARTING`, Dagster `CANCELING`, Temporal `CancelRequested`. Spec §10's "half-cancelled is a state nothing describes" is true **only of ours**, and naming it is a spec change |
| **O15** | **What "reporting upward" means, and `is_end` under a cancelled subgraph.** §8.6 returns a `CascadeReport` to the caller, which is the reading with precedent — Airflow's `dry_run` is structural about the report going to whoever asked. The stronger reading, an ancestor consulted mid-cascade, **is unprecedented in everything surveyed**: veto exists widely but always as child-over-parent, self, or parent-declared-in-advance. Kotlin supplies the cleanest available answer to the direction question — `cancel` takes only a `CancellationException`, "which does not lead to cancellation of its parent", while a child's *failure* does. **Two causes, two upward paths.** Underneath sits spec §10's own row: a cancelled subgraph never completes its `is_end` subtask, so "has this subgraph finished" has no answer for it, and §3.2.1's markers assume completion. Criterion 37 depends on that, so it is only partly assertable |

Three more, smaller:

| # | Question |
|---|---|
| **O16** | **`model_copy(deep=True)` and `copy.deepcopy` silently clone a `Task`'s registry reference** rather than sharing or dropping it — measured. A cloned task then drives a scheduler nobody else can see, which is worse than the reference being absent, because the absent case raises (§3.4). Nothing in this package deep-copies a model, so it is stated rather than defended against. A `__deepcopy__` that drops the reference is the one-line fix if a caller ever appears |
| **O17** | **Nothing bounds or backs off a non-terminating cascade.** §8.6's visited set makes a cycle *terminate*; it does not make anyone aware the cycle existed. systemd is the third option the spec's binary framing omits — it detects the cycle, logs "Found ordering cycle", and breaks it by deleting a job. Kubernetes' answer is exponential backoff plus a rate limiter. Neither is built, and there is no measurement saying either is needed |
| **O18** | **Handoff ids are global, which is what makes criterion 50's violation expressible at all.** `HandoffMgr` is one flat namespace (spec §3.1), so a task outside a subgraph can name a handoff produced inside it and the lookup succeeds. Everywhere else containment is enforced by *scope* — the inner name does not exist outside, so the violation cannot be written. Scoping handoff ids to a subgraph and requiring re-export is the alternative with precedent, and it would make §8.7's check 2 unnecessary rather than merely automated. A spec question, and a large one |
| **O20** | **`charge` can drive a consumable's `available` negative, and nothing acts on that.** §6.3.1 argues the negative balance is the honest record — refusing to book spend that already happened is worse — and D18 already clamps at zero on resume. But nothing warns while running, and a pool sitting at −40 000 tokens rejects every request including zero, so every task naming it queues for ever with no diagnostic. That is D18's failure mode reached from the other direction. A warning at the crossing, or the explicit `refill` spec §10 already wants, would close it. Not built |
