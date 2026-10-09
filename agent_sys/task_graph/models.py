"""Domain models.

Every model owns its own state machine and touches no other component. What it
does *not* own is its collection — that is the manager's.

A `Task` also owns its own *transitions*: nothing outside a task writes its
status. A transition reaches the scheduler through the registry, by name, so
this module imports no scheduler symbol and the package graph stays acyclic.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, NamedTuple, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from spec_loader import subgraph_of as declared_subgraph
from spec_loader import task_of
from task_graph.ids import AgentId, HandoffId, TaskId
from task_graph.permissions import Permissions

__all__ = [
    "Model",
    "TaskStatus",
    "HandoffStatus",
    "WAITING",
    "RESUMABLE",
    "PHASES",
    "TERMINAL",
    "HandoffStateError",
    "TaskStateError",
    "HandoffVersion",
    "Handoff",
    "Execution",
    "Task",
    "HandoffRef",
    "Agent",
    "CascadeReport",
    "SubgraphEntry",
    "subgraph_entries",
    "DerivedEdge",
    "derived_edges",
    "SUBGRAPH_AGENT_SPEC",
    "agent_spec_for",
]


#: The agent spec a **non-leaf** runs under, supplied by the system because the
#: author no longer writes one (main spec §4.8, narrowed at rev. 10).
#:
#: **Not `None`, and the reason is that the field is read unconditionally on a
#: path a non-leaf takes.** `scheduler.py:53` gates `submit` on
#: `agent_mgr.is_registered(task.agent_spec)` for every task, and
#: `scheduler.py:274` calls `instantiate(task.agent_spec, tid)` and feeds
#: `agent.id` into a required `Execution.agent_id`. So a hole here does not stop
#: at the model — it reaches the execution record, which is the shape
#: `docs/interfaces.md` §5.13 already has open for `Verdict.agent_id`, and a
#: second instance of it would make every downstream reader re-derive
#: "non-leaf ⇒ no attribution".
#:
#: **A name, not a document.** `AgentMgr.register` takes a bare name with empty
#: config (`agent.py:28`), which satisfies both the submit gate and
#: `instantiate`. Nothing resolves it in `agent_specs`: the only reader that
#: would is `runner.agent_spec_of`, reached from `_deploy`, and `_main` returns
#: before `_deploy` for a non-leaf (`agent/runner.py:679-685`). So the system
#: invents no spec — it registers one more name, the way the composition root
#: already supplies `DEFAULT_MONITOR_NAME` and `FakeRunner`.
#:
#: It is also **more** truthful than what it replaces, not less. Today a non-leaf
#: is minted a real `Agent` under the author's name and the demo reports
#: `main dispatched to agent 'compose'` while `compose`'s backend is never
#: deployed. This name cannot be mistaken for a leaf's executor.
SUBGRAPH_AGENT_SPEC = "subgraph"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # a typo'd field is an error, not a silent drop
        validate_assignment=True,  # task.status = X is validated
        use_enum_values=False,  # keep enum members; compare with `is`
    )


class TaskStatus(str, Enum):
    WAITING_HANDOFF = "waiting_handoff"
    WAITING_RESOURCE = "waiting_resource"
    INPUT_VALIDATING = "input_validating"
    RUNNING = "running"
    OUTPUT_VALIDATING = "output_validating"
    STOPPING = "stopping"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"


WAITING = frozenset({TaskStatus.WAITING_HANDOFF, TaskStatus.WAITING_RESOURCE})
RESUMABLE = frozenset({TaskStatus.FAILED, TaskStatus.SUSPENDED})
TERMINAL = frozenset({TaskStatus.SUCCEEDED, TaskStatus.CANCELLED})

# ORDERED, not a set: `PHASES[i + 1]` is "the next phase", which `enter_phase`
# uses to reject a runner trying to skip output validation by advancing twice.
# A frozenset would make the membership tests read identically and silently lose
# the sequence.
PHASES = (
    TaskStatus.INPUT_VALIDATING,
    TaskStatus.RUNNING,
    TaskStatus.OUTPUT_VALIDATING,
)


class HandoffStatus(str, Enum):
    CREATED = "created"  # declared; nothing written yet
    GENERATING = "generating"  # an agent has it open
    VALID = "valid"  # sealed, usable
    INVALID = "invalid"  # sealed, not usable


_VERDICTS = frozenset({HandoffStatus.VALID, HandoffStatus.INVALID})


class HandoffStateError(RuntimeError):
    """An illegal handoff transition."""


class TaskStateError(RuntimeError):
    """An illegal task transition."""


# ------------------------------------------------------------------ handoff


class HandoffVersion(Model):
    """One attempt at filling a slot. Immutable once sealed."""

    version: int
    status: HandoffStatus = HandoffStatus.CREATED
    producer_task_id: TaskId | None = None
    producer_agent_id: AgentId | None = None  # None until opened
    timestamp: datetime = Field(default_factory=_now)
    content: Any = None

    @property
    def is_valid(self) -> bool:
        return self.status is HandoffStatus.VALID

    def seal(self, status: HandoffStatus, content: Any = None) -> None:
        """GENERATING -> VALID | INVALID."""
        if self.status is not HandoffStatus.GENERATING:
            raise HandoffStateError(
                f"v{self.version} is {self.status.value}; only a GENERATING version can be sealed"
            )
        if status not in _VERDICTS:
            raise HandoffStateError(f"{status.value} is not a verdict; use VALID or INVALID")
        self.status = status
        self.content = content
        self.timestamp = _now()


class Handoff(Model):
    """The slot. `versions` is append-only and the list index is the version."""

    id: HandoffId
    type: str = ""
    versions: list[HandoffVersion] = Field(min_length=1)

    @property
    def latest(self) -> HandoffVersion:
        return self.versions[-1]

    @property
    def is_latest_valid(self) -> bool:
        return self.latest.is_valid

    def get(self, version: int) -> HandoffVersion:
        if not 0 <= version < len(self.versions):
            raise IndexError(f"{self.id} has no version {version}")
        return self.versions[version]

    def open_next(self, task_id: TaskId, agent_id: AgentId) -> HandoffVersion:
        """Hand an agent a version to write, GENERATING, and return it.

        Adopts `latest` in place if it is still CREATED; otherwise appends v+1.
        Raises if `latest` is GENERATING — someone else has it open.
        """
        latest = self.latest
        if latest.status is HandoffStatus.GENERATING:
            raise HandoffStateError(
                f"{self.id} v{latest.version} is already open by task {latest.producer_task_id}"
            )
        if latest.status is HandoffStatus.CREATED:
            version = latest
        else:
            version = HandoffVersion(version=len(self.versions))
            self.versions.append(version)
        version.producer_task_id = task_id
        version.producer_agent_id = agent_id
        version.timestamp = _now()
        version.status = HandoffStatus.GENERATING
        return version


# --------------------------------------------------------------------- task


@dataclass(frozen=True)
class CascadeReport:
    """What a cascade reached, and what it declined to touch.

    A state plus a reason *per unit* is the only one of the three surveyed
    upward-reporting shapes that survives an incomplete cascade — and
    incompleteness is guaranteed, because no surveyed system makes a cascade
    atomic. The reason travels here rather than onto the task: a field on the
    task would be a second record of what this already carries.
    """

    reached: tuple[tuple[TaskId, str], ...] = ()
    refused: tuple[tuple[TaskId, TaskStatus], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.reached or self.refused)


@dataclass(frozen=True)
class SubgraphEntry:
    """One declared member of a task's expansion.

    **The entry's shape is this package's to define**, and `spec_loader`'s
    `task.schema.json` says so in as many words: `subgraph` is typed
    `array of object` on purpose, because "the shape of an entry belongs to
    `task_graph.Task.unfold`, which instantiates it, and nothing in `closure`
    reads inside one". This is that shape — a declared closure name, optionally
    with its entry marks.

    The key and the shape are a **documented convention and not a specification**:
    no spec in the set fixes either. Hosting the accessor in `spec_loader` gave
    the key one reader; it did not promote the convention to a rule.

    **`froms` is the exception, and it arrived the other way round.** `closure`
    spec §2.7 fixes it and `task.schema.json` makes it required, so it is
    *declared* where the other three are declared-or-defaulted. It names entries
    of **this** subgraph by their `closure`, and the names must refer to earlier
    entries — see `check_graph`, which is where that is enforced.
    """

    closure: str
    is_start: bool
    is_end: bool
    froms: tuple[str, ...] = ()


def subgraph_entries(task_spec: Mapping[str, Any]) -> tuple[SubgraphEntry, ...]:
    """The declared expansion of a **task spec**, normalised, or `()` for a leaf.

    Named apart from `spec_loader.subgraph_of` on purpose: that one hands back
    the entries **as written**, this one hands back what they *mean*. Two
    functions with one name and two return types is a reader's trap, and the
    split itself is right — `spec_loader` owns that the key exists and is called
    `subgraph`; the marks mean nothing until an entry is a `SubgraphEntry`, and
    that type is not its to name.

    A task spec is the inner object — `goal`, `body`, `version`, `inputs`,
    `outputs`, `resources`, `subgraph` — not the closure document that wraps it
    under a `task` key. `check_graph` is handed task specs by name;
    `Task.unfold` reads a closure document and passes `doc["task"]` here.

    Absent marks default to "first is the start, last is the end", which is the
    only reading that makes a one-entry subgraph well formed.

    **`froms` is not defaulted, and absent is read as `()`.** The schema makes
    it required, and re-implementing `required` here would give one rule two
    writers — main spec §4.4's "the schema is the only enforcement point". A
    document that reaches this function without it never passed the schema, and
    `bootstrap` puts exactly those names in `check_graph`'s `skip`.
    """
    entries = list(declared_subgraph(task_spec))
    out = []
    for i, entry in enumerate(entries):
        name = entry["closure"] if isinstance(entry, Mapping) else str(entry)
        marks: Mapping[str, Any] = entry if isinstance(entry, Mapping) else {}
        raw_froms = marks.get("froms")
        froms = tuple(f for f in raw_froms if isinstance(f, str)) if _is_seq(raw_froms) else ()
        out.append(
            SubgraphEntry(
                closure=name,
                is_start=bool(marks.get("is_start", i == 0)),
                is_end=bool(marks.get("is_end", i == len(entries) - 1)),
                froms=froms,
            )
        )
    return tuple(out)


def _is_seq(value: Any) -> TypeGuard[Sequence[Any]]:
    """A `TypeGuard`, not a `bool`: the caller iterates `raw_froms` inside the
    guard, and a plain `bool` narrows nothing, so a checker reads that as
    iterating a possible `None`. `typing.TypeGuard` is 3.10, which is the floor."""
    return isinstance(value, (list, tuple))


class DerivedEdge(NamedTuple):
    """One edge the handoff wiring implies, and the kind that implies it.

    `producer_index` is a position in the subgraph's entry list rather than a
    closure name, because a subgraph may list one closure twice and only
    positions stay unambiguous when it does. It is **not** spelled `index`: a
    `NamedTuple` is a `tuple`, and that field name would shadow `tuple.index()`
    and turn `.index(x)` into a `TypeError`. `kind` is the first input kind that
    produced this edge, carried so a diagnostic can name *why* the edge exists;
    a caller that only wants the graph reads `producer_index`.
    """

    producer_index: int
    kind: str


def derived_edges(
    entries: Sequence[SubgraphEntry],
    task_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[DerivedEdge, ...], ...]:
    """The subgraph's edges, derived from handoff wiring plus list order.

    **The single writer of "which entry depends on which".** `Task._instantiate`
    used to derive this inline while it allocated handoff ids, and `check_graph`
    now has to ask the same question to cross-check `froms`. Two walks would be
    two writers of one invariant (`engineer_principle.md` §1), and the copy in
    the checker is the one that would silently stop matching — so the walk lives
    here and `_instantiate` calls it.

    The rule, unchanged from `_instantiate`: an entry consuming kind K depends on
    whichever *earlier* entry last produced K. A kind nobody inside produces
    resolves to the parent's own input and yields no edge, which is why the walk
    needs no knowledge of the parent.

    **Every edge therefore points strictly backwards, by construction.** The
    derived graph is acyclic and its listing order is always a valid topological
    order; only a *declared* `froms` can violate either. That is the whole reason
    `check_graph` compares indices instead of running a topological sort.

    A member absent from `task_specs` contributes no kinds rather than raising —
    `check_graph` already walks a catalogue that may be missing one, and
    `_instantiate` raises on the same fault itself, with the catalogue in the
    message.
    """
    producer_of: dict[str, int] = {}
    out: list[tuple[DerivedEdge, ...]] = []
    for index, entry in enumerate(entries):
        member = task_specs.get(entry.closure) or {}
        edges: list[DerivedEdge] = []
        seen: set[int] = set()
        for kind in _kinds_of(member, "inputs"):
            producer = producer_of.get(kind)
            if producer is None or producer in seen:
                continue
            seen.add(producer)
            edges.append(DerivedEdge(producer, kind))
        for kind in _kinds_of(member, "outputs"):
            producer_of[kind] = index
        out.append(tuple(edges))
    return tuple(out)


def _kinds_of(task_spec: Mapping[str, Any], key: str) -> list[str]:
    return list(task_spec.get(key) or ())


def agent_spec_for(doc: Mapping[str, Any], task_spec: Mapping[str, Any]) -> str:
    """Which agent spec a task runs under, given its closure document.

    **Public, and it became public because it had a second writer.** It was
    module-private and `cli/build.py::root_task` — the one place a `Task` is
    built outside `unfold`, and a job `docs/interfaces.md` §5.3 records as
    nobody's — copied it, named the copy a defect in its own docstring, and
    reported it here rather than living with it. That was the right call and
    this is the other half of it: the rule is one invariant, so it gets one
    writer, and the caller deletes its copy.

    The invariant, stated once: **a declared agent wins even on a non-leaf; a
    non-leaf with none gets `SUBGRAPH_AGENT_SPEC`; a leaf with none is a load
    failure and keeps its `KeyError`.**

    A declared one wins even on a non-leaf: an author may still name one, and
    silently replacing it would be this module deciding something the document
    already said.

    **The fallback is conditional on being a non-leaf, not on the key being
    absent.** `doc.get("agent") or SUBGRAPH_AGENT_SPEC` would read the same on
    every document the loader admits, and differ on the one that matters: a
    *leaf* with no agent is a load failure (`closure/check.py` check 4, and the
    schema's `else`), and papering it with a built-in name here would let a
    broken catalogue dispatch under a name that describes something it is not.
    So a leaf keeps the `KeyError`, which is the loud failure `unfold`'s
    docstring asks for at this level.
    """
    declared = doc.get("agent")
    if declared:
        return str(declared)
    if subgraph_entries(task_spec):
        return SUBGRAPH_AGENT_SPEC
    return doc["agent"]  # a leaf with no agent: load should have caught it, loudly


class Execution(Model):
    """One attempt at running a task. The stack top is the live binding."""

    attempt: int
    agent_id: AgentId
    input_versions: dict[HandoffId, int] = Field(default_factory=dict)
    output_versions: dict[HandoffId, int] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    outcome: TaskStatus | None = None
    detail: str = ""  # from the runner; for a human

    @property
    def is_open(self) -> bool:
        return self.ended_at is None


class Task(Model):
    id: TaskId = Field(default_factory=TaskId.new)
    agent_spec: str  # a SPEC NAME, not an agent
    inputs: list[HandoffId] = Field(default_factory=list)
    outputs: list[HandoffId] = Field(default_factory=list)
    depends_on: list[TaskId] = Field(default_factory=list)  # the graph edge
    resources: dict[str, float] = Field(default_factory=dict)  # pool NAME -> amount
    status: TaskStatus = TaskStatus.WAITING_HANDOFF
    created_at: datetime = Field(default_factory=_now)
    expedited: bool = False
    history: list[Execution] = Field(default_factory=list)

    # ---- structure. Nothing in scheduling reads these (criterion 42) ----
    parent: TaskId | None = None  # which task this expanded from
    is_start: bool = True  # the subgraph's entry point
    is_end: bool = True  # the subgraph's exit point
    closure: str | None = None  # THE LINK back to its task spec
    permissions: Permissions = Field(default_factory=Permissions)
    kinds: dict[HandoffId, str] = Field(default_factory=dict)  # slot -> handoff kind
    monitor_spec: str | None = None  # which monitor loop watches this task

    # The collaborator, kept out of persistence. `model_validate` returns None
    # for it under every candidate mechanism, so `TaskMgr` re-supplies it on
    # every task it constructs — there is no other supplier.
    _registry: Any = PrivateAttr(default=None)

    def __deepcopy__(self, memo: dict | None = None) -> "Task":
        """Share the registry; never clone it.

        `model_copy(deep=True)` — which `update_task` is — otherwise deep-copies
        the private attribute, and a cloned registry means the copy drives a
        scheduler nobody else can see. It also simply fails: the scheduler holds
        an `RLock`, which cannot be copied. Seeding the memo is the whole fix.
        """
        memo = {} if memo is None else memo
        registry = (self.__pydantic_private__ or {}).get("_registry")
        if registry is not None:
            memo[id(registry)] = registry
        return BaseModel.__deepcopy__(self, memo)

    @property
    def current(self) -> Execution | None:
        return self.history[-1] if self.history else None

    @property
    def is_running(self) -> bool:
        current = self.current
        return current is not None and current.is_open

    def push_execution(
        self,
        agent_id: AgentId,
        input_versions: dict[HandoffId, int] | None = None,
        output_versions: dict[HandoffId, int] | None = None,
    ) -> Execution:
        """Bind an agent by appending an open record, attempt = len(history).

        **Both version maps are pinned here, and that is `interfaces.md`
        §4.14.** An output used to be recorded when the attempt closed, which
        made `Execution.output_versions` empty for the whole of the attempt
        that was supposed to fill it — so `env_mgr`'s kind-named write grant had
        no ``N`` to build a store path from and raised `UnresolvedGrant` before
        the body ran. Pinning at dispatch is what lets the agent write into its
        own grant. Which paths under `v<N>/` that grant covers is `env_mgr`'s
        (`grants.py::_version_paths`) and not stated here — it has already
        narrowed once, and a copy of it would go stale where the pointer does
        not. **This field carries the number, nothing else.**
        """
        if self.is_running:
            raise TaskStateError(f"{self.id} already has an attempt open")
        execution = Execution(
            attempt=len(self.history),
            agent_id=agent_id,
            input_versions=dict(input_versions or {}),
            output_versions=dict(output_versions or {}),
        )
        self.history.append(execution)
        return execution

    def close_execution(self, outcome: TaskStatus, detail: str = "") -> None:
        """Seal the stack top.

        **It does not touch `output_versions`, and it used to.** Since §4.14
        that map is pinned by `push_execution`, so writing it again here would
        give one field two writers — and the two would not even agree: what
        close derived was `HandoffMgr`'s *slot* version, while `env_mgr` reads
        this field as the *store* version the grant resolved to. They coincide
        at v0 and diverge at the first retry, which is the shape of defect that
        surfaces months later.
        """
        if not self.is_running:
            raise TaskStateError(f"{self.id} has no open attempt to close")
        top = self.history[-1]
        top.outcome = outcome
        top.detail = detail
        top.ended_at = _now()

    # ------------------------------------------------------------ transitions
    #
    # A task's state changes through these and nothing else. Each checks its own
    # precondition and then goes through `Scheduler._move`, which stays the
    # single writer of `status` and of the pools.

    def cancel(self, reason: str = "") -> CascadeReport:
        """A waiting state -> CANCELLED, then cascade downstream in this graph.

        Returns what the cascade reached and what it refused. A task already
        running is refused rather than stopped — the narrowest behaviour, and
        the one that presumes least (design O14).
        """
        if self.status not in WAITING:
            raise TaskStateError(
                f"cannot cancel {self.id}: it is {self.status.value}, expected a waiting state"
            )
        return self._sched().cascade_cancel(self.id, reason)

    def restart(self) -> None:
        """FAILED | SUSPENDED -> the recomputed waiting pool. Cascades nothing."""
        if self.status not in RESUMABLE:
            raise TaskStateError(
                f"cannot restart {self.id}: it is {self.status.value}, expected failed or suspended"
            )
        self._sched().resume_task(self.id)

    def fail(self, detail: str = "") -> None:
        """A phase state -> FAILED. Cascades nothing.

        Reachable through `on_task_done` already; this is the same effect with an
        explicit caller, and it exists so a monitor that has judged a task dead
        has a verb instead of reaching for the field.
        """
        if self.status not in PHASES:
            raise TaskStateError(
                f"cannot fail {self.id}: it is {self.status.value}, expected a phase state"
            )
        self._sched().on_task_done(self.id, TaskStatus.FAILED, {}, detail=detail)

    def replace_with(self, closure_name: str) -> CascadeReport:
        """Cancel this graph's downstream, then instantiate a declared closure.

        Only a declared closure: regenerating tasks is graph construction, and
        without the check this is the hole through which the whole
        record-and-replay constraint is bypassed.
        """
        if self.status not in WAITING | TERMINAL:
            raise TaskStateError(
                f"cannot replace {self.id}: it is {self.status.value}, "
                f"expected a waiting or terminal state"
            )
        closures = self._closures()
        if closure_name not in closures:
            raise TaskStateError(
                f"{closure_name!r} is not a declared closure; known: {sorted(closures.names())}"
            )
        scheduler = self._sched()
        report = scheduler.cascade_cancel(
            self.id, f"replaced by {closure_name}", include_self=False
        )
        for subtask in self._instantiate(closure_name, closures):
            scheduler.submit(subtask)
        return report

    # ---- the phase transitions, called by the runner and by the monitor ----

    def enter_phase(self, phase: TaskStatus) -> None:
        """Advance to the next member of PHASES.

        Rejects anything that is not the successor of the current phase, so a
        runner cannot skip output validation by advancing twice. A non-leaf
        entering RUNNING unfolds its declared expansion here — the phase table's
        "the scheduler, if there is a subgraph".
        """
        if self.status not in PHASES:
            raise TaskStateError(
                f"cannot advance {self.id}: it is {self.status.value}, expected a phase state"
            )
        expected = PHASES[PHASES.index(self.status) + 1 :][:1]
        if not expected or phase is not expected[0]:
            raise TaskStateError(
                f"cannot advance {self.id} from {self.status.value} to {phase.value}; "
                f"the phase sequence is {[s.value for s in PHASES]}"
            )
        scheduler = self._sched()
        scheduler._move(self.id, phase)
        if phase is TaskStatus.RUNNING and self.has_subgraph():
            self._place_container_zone()
            for subtask in self.unfold():
                scheduler.submit(subtask)

    def _place_container_zone(self) -> None:
        """This non-leaf's zone, **before its children exist**.

        `env_mgr.place_zone` — `prepare`'s first step and none of the rest. The
        verb is `env_mgr`'s and is unchanged; what is settled here is *when* it
        is called, which `agent/runner.py::_place_container_zone` explicitly left
        unruled: *"The scheduler at `unfold` and the attempt before it releases
        its thread are both plausible ... the choice is not this module's."* It
        is this one's, and the answer is `unfold`.

        **Because `submit` dispatches, and a dispatched child creates its own
        zone inside this one.** `submit` ends in `try_dispatch`, so the loop
        below starts an attempt thread per child before this method returns to
        the monitor; the monitor then wakes *this* task's attempt, which is where
        the call used to live. So the parent's zone was created **after** every
        child was already running — the ordering was never in the parent's
        favour, only in its favour by a margin.

        The margin is one monitor round, and it is not enough. Measured, in
        `scratch/demo2-2026-08/zone-ordering.md`: `layout.create` for a non-leaf
        begins with `find_zone_dir`, an `os.walk` of the **whole** zones tree to
        locate its own parent's zone, and that walk grows with the run — 11 ms
        over an empty tree, 540 ms over the 1669 directories a demo2 run had
        accumulated by the time `grade` unfolded. A child's input-validation
        phase is shorter than that, so the deeper into a run a non-leaf unfolds,
        the more reliably it loses. `full2.log`: *"task caf4fb37 declares parent
        bd890c07, which has no zone"*, twice, for `grade`'s two children.

        **Attempt-scoped, and that is required rather than incidental** — a retry
        re-enters this phase and unfolds again, and those children must nest
        under *that* attempt's directory. `layout.create` is idempotent
        (`exist_ok`), so a resume that re-enters RUNNING reloads rather than
        rebuilds.

        Resolved by name, never imported: `task_graph` §4.7 *"resolves
        everything, by name"*, and a system assembled without `env_mgr` (which
        `interfaces.md` §2.4 permits, and every `tests/task_graph` fixture is)
        simply has no zones to place.
        """
        registry = self._require_registry()
        if "env_mgr" not in registry:
            return
        registry.get("env_mgr").place_zone(self, self.current)

    def _already_unfolded(self) -> bool:
        """Has this task's expansion been built before?

        **`has_subgraph` asks the declaration; this asks the run.** The two were
        the same question only while nothing ever entered RUNNING twice, and a
        resume does exactly that.

        The test is `TaskMgr.children` — which `has_subgraph`'s own docstring
        already names as what leaf-ness means *after* unfolding. That sentence
        was in this file, correct, and not consulted at the one site where the
        distinction decides anything.
        """
        return bool(self._require_registry().get("task_mgr").children(self.id))

    def has_subgraph(self) -> bool:
        """Does this task's declaration expand into one? Asked before unfolding.

        Leaf-ness *after* unfolding is the absence of children
        (`TaskMgr.children`), never the `is_start` / `is_end` pair: the spec
        states both-marks as a consequence of leafness, not as a test for it.
        """
        if self.closure is None or self._registry is None:
            return False
        closures = self._closures()
        if self.closure not in closures:
            return False
        return bool(subgraph_entries(task_of(closures.get(self.closure))))

    def unfold(self) -> list["Task"]:
        """Instantiate this task's declared expansion, `parent = self.id`.

        Returns the subtasks; the caller submits them. Raises on a task with no
        `closure`, and on a closure absent from the catalogue — improvising an
        undeclared step is what the risk exit is for, and that exit is not this
        module's, so the obligation here is only to fail loudly.
        """
        if self.closure is None:
            raise TaskStateError(f"cannot unfold {self.id}: it has no closure")
        if self._already_unfolded():
            # **Idempotent, for the same reason `HandoffMgr.declare` is.** A
            # resumed non-leaf re-enters its main phase, and a non-leaf's main
            # phase *is* the unfold — so without this a resume builds a second
            # subgraph beside the first. `demo` measured it end to end: one run
            # then one `--resume` left 2x every subtask and 2x every handoff
            # slot, all parented to the one correctly-resumed root.
            #
            # Nothing to submit is the honest answer, not an error: the
            # expansion this task declares already exists, with the attempt
            # history and sealed handoff versions its children have accumulated.
            # Rebuilding would be `declare`'s "overwriting would delete versions
            # an agent had already written", one level up.
            return []
        closures = self._closures()
        if self.closure not in closures:
            raise TaskStateError(
                f"{self.closure!r} is not a declared closure; known: {sorted(closures.names())}"
            )
        return self._instantiate(self.closure, closures, parent=self.id)

    # -------------------------------------------------------------- internals

    def _instantiate(
        self, closure_name: str, closures: Any, *, parent: TaskId | None = None
    ) -> list["Task"]:
        """Build one `Task` per declared subgraph entry, wired by handoff kind.

        A kind an entry consumes resolves to the slot an earlier entry produced;
        a kind nobody inside produces resolves to this task's own input of that
        kind. Symmetrically the end entry's outputs *are* this task's outputs —
        which is the declared boundary criterion 50 checks nothing escapes
        through.

        **`depends_on` is the derived edges union the declared `froms`**, in that
        order. The union is what makes `froms` operative rather than decorative:
        `closure` spec §2.7 says it is *the place* to write down a dependency
        that shares no handoff, and derivation cannot see such an edge, so
        dropping it here would leave the field checked and unused —
        `scheduler._warn_depends_on` would still have no way to be satisfied for
        one. Deriving stays first because `check_graph` requires `froms` to
        contain every derived edge, so on an admitted catalogue the union only
        ever adds the handoff-free ones.
        """
        entries = subgraph_entries(task_of(closures.get(closure_name)))
        if not entries:
            raise TaskStateError(f"closure {closure_name!r} declares no subgraph")

        specs_by_name = {
            entry.closure: task_of(closures.get(entry.closure))
            for entry in entries
            if entry.closure in closures
        }
        edges = derived_edges(entries, specs_by_name)

        available: dict[str, HandoffId] = {}
        for hid in self.inputs:
            kind = self.kinds.get(hid)
            if kind is not None:
                available.setdefault(kind, hid)
        mine: dict[str, HandoffId] = {}
        for hid in self.outputs:
            kind = self.kinds.get(hid)
            if kind is not None:
                mine.setdefault(kind, hid)

        ids: list[TaskId] = []
        latest_named: dict[str, int] = {}
        subtasks: list[Task] = []
        for index, entry in enumerate(entries):
            if entry.closure not in closures:
                raise TaskStateError(
                    f"{closure_name!r} names subtask closure {entry.closure!r}, which is "
                    f"not declared; known: {sorted(closures.names())}"
                )
            doc = closures.get(entry.closure)
            task_spec = task_of(doc)
            tid = TaskId.new()
            kinds: dict[HandoffId, str] = {}
            inputs: list[HandoffId] = []
            depends_on: list[TaskId] = [ids[edge.producer_index] for edge in edges[index]]
            for name in entry.froms:
                if name not in latest_named:
                    raise TaskStateError(
                        f"{closure_name!r} entry {index} ({entry.closure!r}) declares "
                        f"froms {name!r}, which names no earlier entry of this subgraph; "
                        f"earlier entries: {sorted(latest_named)}"
                    )
                declared = ids[latest_named[name]]
                if declared not in depends_on:
                    depends_on.append(declared)
            for kind in _kinds_of(task_spec, "inputs"):
                hid = available.get(kind)
                if hid is None:
                    hid = HandoffId.new()
                    available[kind] = hid
                inputs.append(hid)
                kinds[hid] = kind
            outputs: list[HandoffId] = []
            for kind in _kinds_of(task_spec, "outputs"):
                hid = mine.get(kind) if entry.is_end else None
                if hid is None:
                    hid = HandoffId.new()
                outputs.append(hid)
                kinds[hid] = kind
                available[kind] = hid
            ids.append(tid)
            latest_named[entry.closure] = index
            subtasks.append(
                Task(
                    id=tid,
                    agent_spec=agent_spec_for(doc, task_spec),
                    inputs=inputs,
                    outputs=outputs,
                    depends_on=depends_on,
                    resources=dict(task_spec.get("resources") or {}),
                    parent=parent if parent is not None else self.parent,
                    is_start=entry.is_start,
                    is_end=entry.is_end,
                    closure=entry.closure,
                    # **The subtask's own declared permissions**, not the
                    # parent's. Passing `self.permissions` down discarded what
                    # the sub-closure declared and gave every subtask the root's
                    # full set — so a subtask held kind-named grants for kinds it
                    # has no slot for, and `env_mgr.resolve` treats "no slot has
                    # that kind" as an error rather than a no-op. `demo`
                    # measured it: `produce` carrying a `summary` grant.
                    #
                    # Inheriting was never required. `closure` check 6 already
                    # validates at load that a task's declared permissions cover
                    # its own handoffs, and design §3.5 records that criterion
                    # 44's "covers its subtasks recursively" is a property of
                    # the storage layout `env_mgr` builds — containment — rather
                    # than of anything stored in this field.
                    permissions=Permissions.model_validate(task_spec.get("permissions") or {}),
                    kinds=kinds,
                    monitor_spec=task_spec.get("monitor", self.monitor_spec),
                )
            )
        return subtasks

    def _sched(self) -> Any:
        """`registry.get` resolves by name at use time and creates no import
        edge — which is the whole reason a transition may reach the scheduler."""
        return self._require_registry().get("scheduler")

    def _closures(self) -> Any:
        """A `Task` may read the catalogue it came from; the *scheduler* may not
        name a spec registry. Resolving it here adds no scheduler edge."""
        return self._require_registry().get("closures")

    def _require_registry(self) -> Any:
        if self._registry is None:
            raise TaskStateError(
                f"task {self.id} has no registry; it was not loaded through TaskMgr"
            )
        return self._registry


# -------------------------------------------------------------------- agent


class HandoffRef(Model):
    handoff_id: HandoffId
    version: int


class Agent(Model):
    id: AgentId = Field(default_factory=AgentId.new)
    spec: str  # which kind
    task_id: TaskId | None = None  # what it is bound to
    handoffs: list[HandoffRef] = Field(default_factory=list)  # what it touched
    knowledge: Any = None  # left empty by the task definition
    config: dict[str, Any] = Field(default_factory=dict)
