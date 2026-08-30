# Monitor — Design

| | |
|---|---|
| Status | Not implemented. No file of this module exists yet |
| Revision | 2 — 2026-08-28. **Spec rev. 14: two channels.** A second queue for planned advances, an `EventRecord` where rev. 1 had an `ExceptionRecord`, the phase-advance handler (§6.1), and the liveness pair §5.6 owes to the fact that ordinary progress now runs through here. **O1 is closed** — `Runner.attempt_of` is the accessor, and `TaskAttempt` the owner rev. 1 could not find. (rev. 1: 2026-08-27, against spec rev. 13) |
| Implements | [`spec.md`](spec.md) rev. 14 |
| Language | Python ≥ 3.10. Standard library plus pydantic v2 |

---

## 1. Scope

This document turns [`spec.md`](spec.md) into files, classes, and interfaces. It
adds no requirements. Where it makes a choice the spec left open, the choice is
stated here; where turning the spec into interfaces exposed something the spec
assumed and the system does not have, §13 and §14 say so rather than papering
over it.

The spec's 26 acceptance criteria are the definition of done. §11 maps every one
of them to a named test.

**This document specifies interfaces, not bodies.** A method appears as a
signature and a sentence of semantics; a body appears only where the ordering of
steps *is* the design decision — which is `ExceptionBuffer.add` / `done` (§4.2),
`BaseMonitor.report` (§5.3), the mainloop (§5.4), `_advance` (§6.1), and nothing
else.

### 1.1 What this module owns

| | Where |
|---|---|
| The `Monitor` interface, and the two kinds behind it | §5 |
| `EventRecord`, `EventKind`, and the store kind they live in | §3 |
| **The two queues**, and the five rules of spec §5.2 against each | §4 |
| **The planned advance handler** — one transition, sometimes one `resume` | §6.1 |
| The pusher's decision function, and escalation | §6, §7 |
| **The monitor's own liveness** — the excepthook and the heartbeat | §5.6 |
| The vocabulary the *reporters* use — the kinds, the record, the call | §3, §5.3 |

### 1.2 What it does not

| | Owner |
|---|---|
| **The completeness gate itself** | `agent`. Spec §4.1.0 puts it in the runner; §8 here says exactly what this module owes it and what it does not |
| The four transition verbs, and their thread safety | `task_graph` spec §3.4, design §9 |
| Resolving `Task.monitor_spec` by name | `task_graph` design §3.8 |
| The analysing dispatcher — ten of §7.1's twelve actions | [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §2.3 |
| Reaching a human at the top of an escalation chain | Nothing owns it. Spec §11 records it; §7.3 here gives it a seam and no implementation |

---

## 2. Layout and import graph

```
monitor/
├── __init__.py          re-exports the frozen surface
├── protocols.py         Monitor, Pushable, UserSink, Budget — the seam
├── protocols.pyi        the stub, agreeing with protocols.py
├── record.py            EventId, EventKind, EventRecord, Recorder
├── buffer.py            PlannedQueue, ExceptionBuffer, Unit
├── base.py              BaseMonitor — loop, set_task, report, escalate
└── pusher.py            PusherMonitor — the alpha's decision function
```

Six modules, and the split is the spec's: **the record survives the buffer**
(spec §5.2 rule 3), so `record.py` may not import `buffer.py`, and it does not.

### 2.1 The import graph

```
protocols.py  ──▶ task_graph.ids
record.py     ──▶ task_graph.ids, task_graph.models (Model), task_graph.store
buffer.py     ──▶ record.py
base.py       ──▶ protocols.py, record.py, buffer.py, task_graph.registry
pusher.py     ──▶ base.py
```

**`monitor` imports `task_graph` and nothing else of ours.** The row to add to
`tests/interfaces/test_import_rules.py::ALLOWED` is `"monitor": {"task_graph"}`,
and the two consumer rows widen: `"agent"` and `"validator"` each gain
`"monitor"`.

### 2.2 The direction is one-way, and the awkward half is deliberate

The monitor's push needs `instruct()` on a live agent — level 2, `AgentBackend`,
which lives in `agent`. The runner's gate needs `report()` and `EventRecord`,
which live here. Written naively that is a package cycle.

**It is broken here, not there**, by declaring the handle structurally:

```python
class Pushable(Protocol):                       # protocols.py
    """The part of `agent.AgentBackend` a monitor uses. Declared here so
    `monitor` imports nothing from `agent`, which is what keeps the dependency
    one-way: consumers import the monitor, never the reverse."""
    status: Any
    def instruct(self, message: str) -> None: ...
    def query(self) -> Any: ...
```

`AgentBackend` satisfies it structurally and neither package knows about the
other. **The cost is real and is a drift risk** — two declarations of one shape,
and nothing in either package notices if they diverge. The countermeasure is one
test that may legally import both, because tests are not under the import rule:

```python
def test_agent_backend_satisfies_pushable() -> None:      # tests/interfaces/
    assert isinstance(SomeBackend(), Pushable)            # runtime_checkable
```

The alternative — putting `Monitor` in `task_graph`, the one package everyone may
import — was rejected because it makes `task_graph` the owner of an interface
spec §9 says this module defines, and `task_graph` design §8.9 has already
written down that almost none of the monitor is its.

---

## 3. The record — `record.py`

Spec §8: a persisted value written through `task_graph`'s `StoreMgr`, never a log
line. The vocabulary is the three-source hybrid §8.2 fixed; this section turns it
into fields.

### 3.1 `EventId` — a fourth typed identity

```python
class EventId(_Id): ...          # task_graph.ids._Id, alongside TaskId
```

`task_graph/ids.py` already has the base and the pydantic schema hook. **The new
subclass is declared here, not there** — a `task_graph` that had to learn a
monitor id would be the dependency §2.2 just avoided — and `_Id` is imported for
the purpose. That makes `_Id` a private name crossing a package boundary, which
§13 records as a deviation with its alternative.

### 3.2 `EventKind` — the closed enum, and nothing defaults to benign

Spec §8.2.1: Erlang's `Context` widened. **Every value names a phase, and none of
them is a safe default** — the rule `validator` learned the hard way and spec
§8.2.1 restates.

```python
class EventKind(str, Enum):
    # --- PLANNED (spec §2.2). The whole of the planned channel ---
    PHASE_DONE           = "phase_done"            # a phase finished as planned
    SUBGRAPH_DONE        = "subgraph_done"         # an is_end subtask completed

    # --- the gate (spec §4.1.0), four independent failures ---
    OUTPUT_ABSENT        = "output_absent"         # a declared output is missing
    OUTPUT_NOT_EXECUTABLE = "output_not_executable" # claims executable, is not
    SELF_CHECK_UNSET     = "self_check_unset"      # done_by_self_check false
    BUDGET_EXCEEDED      = "budget_exceeded"       # §4.1.3

    # --- the validator (spec §2.1), both outcomes ---
    VALIDATION_FAILED    = "validation_failed"     # worked; the answer is no
    VALIDATION_UNREACHED = "validation_unreached"  # no verdict was produced

    # --- the monitor's own actions (criterion 9) ---
    PUSH_ATTEMPTED       = "push_attempted"
    PUSH_INEFFECTIVE     = "push_ineffective"
    ESCALATED            = "escalated"
    MONITOR_GAVE_UP      = "monitor_gave_up"
    HANDLING_FAILED      = "handling_failed"       # the handler itself raised
    LOOP_STALLED         = "loop_stalled"          # §5.6, the heartbeat check
    THREAD_DIED          = "thread_died"           # §5.6, the excepthook


PLANNED: frozenset[EventKind] = frozenset({EventKind.PHASE_DONE,
                                           EventKind.SUBGRAPH_DONE})
```

**The routing is `kind in PLANNED`, in one place and nowhere else.** Spec §5's
requirement is that a reporter never chooses a queue; a frozenset consulted at the
head of `report()` is what makes that structural. **A new kind that nobody adds to
`PLANNED` lands on the unplanned queue** — which is the safe default, because the
worst outcome is an event that gets decided instead of switched.

**Fifteen values in four groups.** The first two are the planned channel; the next
six are the reporters'; the next five the monitor's own actions; the last two are
the monitor observing *itself* (§5.6). That grouping is what makes criterion 9
mechanical: a push that was
never attempted has no `PUSH_ATTEMPTED` row, one that did nothing has both
`PUSH_ATTEMPTED` and `PUSH_INEFFECTIVE`, and one that worked has the first
without the second. The sequence is the history, exactly as `Task.history` is.

**`OUTPUT_MALFORMED` is not here, and its absence is measured rather than
overlooked.** Spec §4.1.1: `put` raises `Malformed` *before* anything is created,
inside the producing agent's own zone, so a malformed handoff never reaches
storage and the gate sees only an absence. There is no phase in which this module
could observe it. The producer-side record spec §9 asks for would carry it, and
it would be written by `agent`, not here.

### 3.3 `EventRecord`

```python
class EventRecord(Model):                    # task_graph.models.Model
    # --- identity (Sentry's event / fingerprint split, spec §8.2) ---
    id: EventId = Field(default_factory=EventId.new)
    fingerprint: tuple[str, ...]

    # --- correlation. No standard supplies this; it is ours ---
    task_id: TaskId
    attempt: int                                 # (task_id, attempt) IS the Execution
    agent_id: AgentId | None = None
    handoff_id: HandoffId | None = None          # which declared output, when one

    # --- classification (Erlang SASL, spec §8.2.1) ---
    kind: EventKind                          # Erlang's `Context`
    reported_by: str                             # Erlang's `Supervisor`

    # --- payload (OTel stable `exception.*` names, as naming only) ---
    exception_type: str | None = None
    exception_message: str | None = None
    exception_stacktrace: str | None = None      # only when one was raised

    # --- severity (OTel SeverityNumber; the number, not the text) ---
    severity: int = 13                           # 13 WARN handled, 17 ERROR gave up

    at: datetime = Field(default_factory=_now)
    attributes: dict[str, Any] = Field(default_factory=dict)
```

**`Model` is `task_graph`'s**, which gives `extra="forbid"` and the id
serialisation for free and keeps one pydantic configuration in the system rather
than two.

**`fingerprint` is a tuple of strings, not a hash**, because `JsonFileStoreMgr`
writes one readable JSON file per record and the whole reason files were chosen
over sqlite is that a human can `cat` them while the schema is still moving.

The default is a module-level function so it can be replaced without touching the
model:

```python
def default_fingerprint(r: EventRecord) -> tuple[str, ...]:
    return (r.kind.value, str(r.task_id), str(r.handoff_id or ""), r.exception_type or "")
```

**`attempt` is excluded on purpose, and it is a choice rather than a detail.**
Excluding it groups the same failure across attempts into one issue; including it
makes every fingerprint unique and grouping a no-op. Spec §11 keeps it open, and
this is the alpha's answer, put in one function so reversing it is one edit.

### 3.4 `Recorder` — two store kinds, and why the second one exists

```python
class Recorder:
    """Writes records through `StoreMgr`. Owns no policy."""

    def __init__(self, store: StoreMgr) -> None: ...

    def open(self, task_id: TaskId, attempt: int) -> None:
        """Create the (empty) record set for one attempt. Idempotent."""

    def write(self, record: EventRecord) -> None:
        """Persist one occurrence. Append-only; never updates an existing key."""

    def read(self, task_id: TaskId, attempt: int) -> list[EventRecord]:
        """Every record for that attempt, in written order. `[]` if none."""

    def is_open(self, task_id: TaskId, attempt: int) -> bool: ...
```

| Store kind | Key | Holds |
|---|---|---|
| `exception_set` | `f"{task_id}#{attempt}"` | the marker. `{"task_id":…, "attempt":…, "opened_at":…}` |
| `exception` | `f"{task_id}#{attempt}#{exception_id}"` | one `EventRecord`, `model_dump(mode="json")` |

**Two kinds, because absence is a signal (spec §8.3, criterion 14).** The marker
is what makes an empty set distinguishable from a lost one: marker present with
no occurrences reads as *nothing was recorded here*, marker absent reads as
*something is wrong*. This is the rule `handoff` already applies to
`validation.yaml`, which is created empty at publication for the identical
reason.

**Append-only, one store record per occurrence — never read-modify-write.** The
alternative, a single container record holding a list, would have the runner's
gate thread and the monitor's loop thread reading and rewriting one JSON file.
`JsonFileStoreMgr` is atomic *per record* (`tmp.replace(path)`) and not across a
read, so that shape needs a lock the append-only shape does not. Choosing the
key to be unique per occurrence removes the race instead of guarding it.

**What it costs**: `read` is `store.read_all("exception")` filtered by prefix — a
scan of the kind. At alpha scale that is the same trade `TaskMgr.by_status`
already takes (`task_graph` design §6.2.1), and it is noted rather than optimised.

**`Execution.detail` is not touched.** Spec §8.4 asks which it is: the runner
writes it, this module does not read it and does not write it, and there is no
second writer. The record and `detail` overlap in content and not in ownership;
duplicating a sentence is cheaper than sharing a field.

---

## 4. The two queues — `buffer.py`

Spec §5.2's five rules, applied to each queue as spec §5.2's table assigns them.
Kubernetes `client-go/util/workqueue` is the shape of the **unplanned** one; the
planned one is a plain FIFO and is smaller for a reason §4.4 states.

The whole of the adoption is three collections and an invariant; §10 says why it
is copied rather than imported.

### 4.1 `Unit` — what the loop gets

```python
@dataclass(frozen=True)
class Unit:
    task_id: TaskId
    records: tuple[EventRecord, ...]     # every record merged into this unit
```

**The buffer keys on task id and carries the records anyway**, which is where it
departs from the prior art. `client-go` holds bare keys and re-reads state from
an informer cache; there is no cache here, and a collapse that kept only the last
payload would silently lose an observation — measured, `probes-monitor/p3` case 1.
Rule 4 says merge, and `records` is the merge.

### 4.2 `ExceptionBuffer`

```python
class ExceptionBuffer:
    def add(self, record: EventRecord) -> None:
        """Never blocks, never refuses. Raises only after `shutdown`."""

    def get(self, timeout: float) -> Unit | None:
        """The next unit, or None when `timeout` elapses with nothing to do.
        Marks the task as processing."""

    def done(self, task_id: TaskId) -> None:
        """Finish a unit. Re-queues the task if anything arrived while it was
        being handled."""

    def shutdown(self) -> None:
        """Refuse new records loudly; let the queue drain; stop `get` blocking."""

    def __len__(self) -> int: ...            # depth, for the tests
```

State, and the invariant that makes the rules hold:

```python
_order: deque[TaskId]                 # FIFO across tasks
_dirty: set[TaskId]                   # queued or pending re-queue
_processing: set[TaskId]              # a unit is out with the loop
_records: dict[TaskId, list[EventRecord]]
_cond: threading.Condition
```

> **Every element of `_order` is in `_dirty` and not in `_processing`.**

`add`, where the ordering of the steps *is* the design:

```python
with self._cond:
    if self._closed:
        raise BufferClosed(...)                    # loudly. Not a silent drop
    self._records.setdefault(record.task_id, []).append(record)   # rule 4: merge
    if record.task_id in self._dirty:
        return                                     # already known; do not re-queue
    self._dirty.add(record.task_id)
    if record.task_id in self._processing:
        return                                     # rule 5: re-queue in `done`
    self._order.append(record.task_id)
    self._cond.notify()
```

`done`, which is the other half of rule 5:

```python
with self._cond:
    self._processing.discard(task_id)
    if task_id in self._dirty:                     # arrived while handling
        self._order.append(task_id)
        self._cond.notify()
```

`get` pops from `_order`, moves the id from `_dirty` into `_processing`, and
takes `_records.pop(task_id)` as the unit's records.

**How each of the spec's five rules lands:**

| Rule | Where |
|---|---|
| 1 — never blocks, never refuses | `add` takes the lock, appends, returns. No `maxsize` anywhere |
| 2 — bounded by dedup on task id | the `_dirty` short-circuit. Depth ≤ number of tasks |
| 3 — record first, enqueue second | **not here.** It is `BaseMonitor.report`'s ordering, §5.3 |
| 4 — merge, never overwrite | `_records[...].append`, and `Unit.records` is the whole list |
| 5 — a task being handled is not handled twice | `_processing`, plus the re-queue in `done` |

**`done` must be called, and forgetting it wedges the task forever** — the one
sharp edge in the prior art, and `client-go` says so in a comment. The loop calls
it in a `finally` (§5.4) and nothing else calls it.

### 4.3 Shutdown is hand-rolled, and it is the well-behaved variant

`queue.Queue.shutdown` is **3.13-only** and the target is 3.10, so there is
nothing to inherit even if the semantics were right. They are not:
`immediate=True` discards pending items silently, and `client-go` does the same
on its own shutdown path — *"new items will silently be ignored"*.

**Refuse loudly, drain what is queued, then stop.** `add` after `shutdown` raises
`BufferClosed`; `get` returns queued units until `_order` is empty and then
returns `None` rather than blocking. A producer that has an exception to report
into a closing system learns about it, which is the whole difference.

**Dropping work is permitted; dropping a fact is not.** By rule 3 the record is
already on disk before the buffer ever sees it, so a unit lost at shutdown is a
handling that did not happen — visible in the record set as an occurrence with no
outcome after it, which is precisely what criterion 9 asks to be distinguishable.

### 4.4 `PlannedQueue` — a FIFO, and deliberately not the other one

```python
class PlannedQueue:
    def add(self, record: EventRecord) -> None:
        """Append. Never blocks, never collapses. Raises after `shutdown`."""

    def get_nowait(self) -> EventRecord | None:
        """The next advance, or None if empty. Never waits."""

    def shutdown(self) -> None: ...
    def __len__(self) -> int: ...
```

**Everything the buffer does to bound itself is absent here, and that is the
design.** No `_dirty`, no collapse, no `_records` merge — spec §2.2: two advances
of one task are two phases, and merging them is a task that never runs its middle
one. **Criterion 20 is the test, and it is written as a test of this class rather
than of the loop**, because a collapse added later "for symmetry" would be
invisible at the loop level.

**It is bounded anyway, and by a stronger fact than dedup.** A task is in one
phase at a time, so it has at most one outstanding advance; depth cannot exceed
the number of tasks. **The bound the buffer buys with a `_dirty` set, this queue
gets from the domain.**

**What it does share is rule 5 — one task, one handling, across both queues.**
That check cannot live in either queue, because neither can see the other; it
lives in `BaseMonitor` as a single `_current` and the `_inflight` set (§5.4). The
queues stay independent; the exclusion is one level up.

**No blocking `get`.** The loop waits on the buffer's condition and drains this
one first without waiting (§5.4) — a planned advance is never the thing the loop
should sleep for, because whatever produced it has already gone back to work.

---

## 5. The monitor — `protocols.py`, `base.py`

### 5.1 `Monitor` — the frozen surface

```python
class Monitor(Protocol):
    name: str                                    # the composition root keys on it

    def set_task(self, task_id: TaskId) -> None:
        """Take this task under watch. The only way a monitor learns what it
        watches (criterion 10)."""

    def report(self, record: EventRecord) -> None:
        """Persist synchronously, then enqueue. Does not block on the buffer."""

    def mainloop(self) -> None:
        """Drain the buffer. Runs on its own thread; returns after `stop`."""

    def stop(self) -> None:
        """Close the buffer and let `mainloop` return."""
```

**`report` and `set_task` are the inbound surface** — spec §5.1. `mainloop`,
`stop` and `name` are lifecycle and identity, not inbound; §13 records the
distinction rather than letting the count quietly differ.

Two more protocols live beside it:

```python
class UserSink(Protocol):
    def deliver(self, record: EventRecord, why: str) -> None: ...

@dataclass(frozen=True)
class Budget:
    max_tokens: float | None = None
    max_seconds: float | None = None
    max_turns: int | None = None
```

`Budget` is one global value (spec §4.1.3), registered as `budget` and resolved
by the *runner*, which is where it is read. It is declared here because
`BUDGET_EXCEEDED` is one of this module's kinds and a threshold with no matching
kind is unreportable.

### 5.2 `BaseMonitor` — everything both kinds share

```python
class BaseMonitor:
    def __init__(self, name: str, registry: Registry, *,
                 period: float = 1.0, sink: UserSink | None = None) -> None: ...

    # ---- the interface ----
    def set_task(self, task_id: TaskId) -> None: ...
    def report(self, record: EventRecord) -> None: ...
    def mainloop(self) -> None: ...
    def stop(self) -> None: ...

    # ---- what a subclass supplies ----
    def decide(self, unit: Unit) -> Decision:
        """What to do about this unit. The one method the two kinds differ in."""
        raise NotImplementedError

    # ---- what NO subclass may replace: the planned channel ----
    def _advance(self, record: EventRecord) -> None:
        """One transition, then wake or resume. Program, always — §6.1."""
    def _notify_parent_done(self, unit: Unit) -> None: ...   # §7.4

    # ---- what a subclass calls, and may not bypass ----
    def _transition(self, task_id: TaskId, verb: str, **kw) -> None: ...
    def _escalate(self, unit: Unit, why: str) -> None: ...
    def _beat(self) -> None:
        """Stamp `last_beat`. Once per round, before any work — §5.6."""
    def _sweep(self) -> None:
        """Called once per idle period. Does nothing. The §4.3 seam."""
```

`Decision` is a small closed union — `Push`, `Escalate`, `ReportToUser`,
`GiveUp` — and it exists so that *deciding* and *doing* are separable: the
pusher's decision is testable without a live agent, and the analysing
dispatcher replaces `decide` and nothing else.

**Two kinds, one class.** Spec §3: the second is not a different module, it is a
different body behind the same interface. `PusherMonitor` overrides `decide`;
an `AnalysingMonitor` would too, and would touch nothing else.

**Per-task and global are one class as well.** A global monitor is a
`BaseMonitor` whose `set_task` has been called more than once. Nothing in the
loop cares how many ids are in the watch set — the round-robin the roadmap
describes is what FIFO across tasks already is (§4.1).

### 5.3 `report` — the ordering is the design

```python
def report(self, record: EventRecord) -> None:
    self._recorder.write(record)                        # 1. durable
    if record.kind in PLANNED:                          # 2. route — §3.2
        self._planned.add(record)
    else:
        self._buffer.add(record)
```

**The routing is here and nowhere else**, which is spec §5's requirement that a
reporter never picks a queue. One `if`, over one frozenset, at the single inbound
call.

**Three statements, in that order, inside one call.** Spec §5.1 decided it as one
call rather than two so rule 3 holds structurally: a caller that forgot the first
of two steps would lose the event silently, and that is the failure the rule
exists to prevent. Making it one call removes the opportunity rather than
documenting it.

`write` is a single small JSON file on no lock and no scheduler path — spec
§4.1.0 puts the call inside the runner's own gate loop — so the synchronous half
delays one task's cycle and nothing else.

**A `write` that raises propagates to the reporter.** It does not fall through to
`add`: a record that is not durable must not become queued work, because then the
buffer holds the only copy of a fact and rule 3 is a comment rather than an
invariant.

### 5.4 The mainloop

```python
def mainloop(self) -> None:
    while True:
        self._beat()                         # §5.6. One store, every round

        record = self._planned.get_nowait()  # planned first — spec §5
        if record is not None:
            self._run_guarded(record.task_id, lambda: self._advance(record))
            continue

        unit = self._buffer.get(timeout=self._period)
        if unit is None:
            if self._stopping:
                return
            self._sweep()                    # §4.3's seam. Does nothing yet
            continue
        self._run_guarded(unit.task_id, lambda: self._handle(unit),
                          done=lambda: self._buffer.done(unit.task_id))
```

`_run_guarded` is the one place the broad catch and the `_current` bookkeeping
live, so neither path can forget either:

```python
def _run_guarded(self, tid, work, done=None):
    self._current = tid                      # §5.5's scope guard reads this
    try:
        work()
    except Exception:                        # noqa: BLE001 — see below
        self._record_handling_failed(tid)
    finally:
        self._current = None
        if done: done()
```

**Planned work is taken first and without waiting.** A task waiting to advance is
a task doing nothing, and a decision can block on the scheduler's `RLock` while a
transition cannot meaningfully be deferred. **It cannot starve the buffer**: a
planned advance is a fixed non-blocking transition, so the planned queue drains to
empty in bounded time and the loop reaches `buffer.get` on the next round.

**`self._beat()` is called before either**, so a round spent entirely on planned
work still counts as alive.

**The broad catch is deliberate and is criterion 9.** A handler that raises must
not take the loop with it — measured: after an unguarded handler raises, the
thread is dead, further pushes are accepted, depth grows, and **no producer sees
an error** (`probes-monitor/p4`). The failure is recorded as `HANDLING_FAILED`
and the loop continues.

**`_sweep` is a few lines that do nothing, and that is the point** (spec §4.3).
The alpha builds no poller — a wedged agent is a wedged thread and Python cannot
kill a thread, so detection would arrive without remedy — but the *period* and
the *hook* are what make adding one later an edit instead of a refactor.

**One consumer per monitor.** Rule 5 would make several safe; nothing needs the
throughput, and one consumer keeps §6's "one task's scope at a time" true by
construction rather than by argument.

### 5.5 The scope guard — criterion 8

Every transition goes through `_transition`, and `_transition` refuses a task
that is not the one currently being handled:

```python
def _transition(self, task_id: TaskId, verb: str, **kw) -> None:
    if task_id != self._current:
        raise ScopeViolation(f"{self.name} is handling {self._current}, not {task_id}")
    getattr(self._task_mgr.get(task_id), verb)(**kw)     # the task owns its transitions
```

`self._current` is set and cleared by `_run_guarded` (§5.4), which is also **how
spec §5.2 rule 5 spans both queues**: one task is in `_current` at a time
regardless of which queue its work came from, so a planned advance and a decision
for the same task cannot overlap. **This is stricter than criterion 8 asks** — the criterion says "the task
`set_task` gave it", and a global monitor was given several — and the stricter
form is what spec §6 means by holding one task's scope at a time.

**It is also the whole of the authority rule at this end.** `verb` names a method
on `Task`, every one of which routes through the scheduler's `_move`; nothing
here assigns a status, and criterion 6's existing status-write spy sees the new
caller without being amended. **`enter_phase` is one of those verbs**, so the
planned advance (§6.1) needs no exception to any of this.

### 5.6 Liveness — `base.py` and one hook in the composition root

Spec §5.4. Two mechanisms, ~40 lines together, and neither is new machinery.

**1. The excepthook, and why it is not optional.** Measured, not assumed
(`scratch/design/probes-monitor/p4_thread_death.py`, re-run 2026-08-28 on
3.13.13): a `Thread` whose target raises prints a traceback to stderr and dies;
the process keeps running, **the exit code is unchanged, and producers see no
error** — subsequent `add`s are accepted and pile up behind a consumer that no
longer exists.

```python
def install_excepthook(recorder: Recorder, sink: UserSink) -> None:
    """Turn an escaped thread exception into a record and surface it.
    Installed once, by the composition root, for every thread — not only ours."""
```

**It belongs in the composition root, not in `BaseMonitor`.** `threading.excepthook`
is process-global; a module that installed it on construction would be a library
reaching into the interpreter's global state, and two monitors would fight over
it. §12 puts it in the bootstrap step.

**2. The heartbeat, and the checker that needs no checker.**

```python
class BaseMonitor:
    last_beat: float          # monotonic, stamped by _beat() each round (§5.4)

def check_liveness(monitors, *, period, threshold=3, now=time.monotonic) -> list[EventRecord]:
    """Report a LOOP_STALLED for each monitor whose last_beat is older than
    `threshold * period`. Pure: takes a clock, returns records, writes nothing."""
```

**`threshold` consecutive periods, not one** — the `failureThreshold` shape spec
§4.3 already chose over a heartbeat for the agent case, for the same reason: one
slow round is not a death.

**The checker is a pure function over a timestamp**, called from the main thread,
which is already sitting there. That is what makes this an answer rather than an
infinite regress — and it is deliberately the *cheapest* of the three answers the
prior art gives (`scratch/design/findings-arch-super.md`): s6 makes its top-level
supervisor unable to fail (no heap, under 500 lines, a full DFA), systemd hands
the problem to a hardware watchdog, Ray hands it outward to KubeRay. **Comparing
one float is at the trivial end of that scale.**

**`time.monotonic`, not wall-clock**, because a clock adjustment must not read as
a stalled monitor. The clock is injected so the test does not sleep.

**Neither mechanism recovers anything**, and the design says so rather than
implying otherwise: a stalled monitor is *reported*. What happens to the tasks it
was watching is spec §11's open question, and §14 keeps it.

---

## 6. The two handlers

One per channel. **They live in different classes on purpose**: the planned
handler is on `BaseMonitor`, where no subclass can replace it, and the decision is
on the subclass, where replacing it is the entire point.

### 6.1 `_advance` — the planned handler, on `BaseMonitor`

```python
def _advance(self, record: EventRecord) -> None:
    task = self._task_mgr.get(record.task_id)
    self._transition(task.id, "enter_phase", phase=NEXT_PHASE[task.status])
    attempt = self._runner.attempt_of(task.id)
    if attempt is None:                     # the non-leaf case: no live thread
        self._runner.resume(task.id)
    else:
        attempt.wake()
```

**Three lines of behaviour, and no branch on anything but liveness.** There is no
policy here, no threshold, no model — spec §2.2's "program, always". `NEXT_PHASE`
is a mapping over `TaskStatus`, not a computation.

**It is on `BaseMonitor` and it is not overridable.** An `AnalysingMonitor`
inherits it unchanged; that is the mechanical form of spec criterion 19's "a
monitor built with an agent handles a planned event through the same code as one
built without", and it is why criterion 19 can be tested by construction rather
than by inspecting a prompt.

**`SUBGRAPH_DONE` arrives already re-keyed to the parent**, by the subtask's
monitor (§7.4), so this function does not know which of the two planned kinds it
is handling. That is the point: one handler, and the tree walk happens where the
tree walk already lives.

**Why `wake()` and `resume()` are two calls, not one.** A leaf's attempt has a
thread parked on a condition — waking it is free. A non-leaf's ended at `unfold`,
so there is nothing to wake and a thread must be made. Collapsing them into one
runner call would hide, at the one place it matters, which of the two shapes a
task is.

### 6.2 `PusherMonitor.decide` — the unplanned handler

```python
class PusherMonitor(BaseMonitor):
    def decide(self, unit: Unit) -> Decision: ...
```

The decision function, in full, because it is short and because what it *declines*
to do is the load-bearing part:

| The unit's newest record is | Decision |
|---|---|
| a gate kind (`OUTPUT_*`, `SELF_CHECK_UNSET`) **and** a live handle exists **and** no push has been attempted for this attempt | `Push` |
| a gate kind, and a push was already attempted for this attempt | record `PUSH_INEFFECTIVE`, then `Escalate` |
| `BUDGET_EXCEEDED` | `Escalate`. The budget is what bounds the loop (spec §4.1.3); pushing past it would remove the bound |
| `VALIDATION_FAILED` or `VALIDATION_UNREACHED` | `Escalate`. The task is terminal and no agent is running — there is nothing to push |
| anything else | `GiveUp`, recorded as `MONITOR_GAVE_UP` |

**"A push was already attempted" is read back from the record set**, which is the
concrete reason spec §8.2 rejected the OpenTelemetry SDK: OTel is emit-only by
construction, and this decision needs to read what it wrote.

**`Push` is `handle.instruct("continue, do it until finished")`**, and the
handle is a `Pushable` (§2.2). It is recorded as `PUSH_ATTEMPTED` *before* the
call, so an `instruct` that raises still leaves the attempt visible.

**The pusher never reaches for `restart`.** Spec §7's cost table is the reason and
it is measured: push ~2 s and lossless; resume ~5.5 s warm and drops
`permission_mode`, `--mcp-config`, `--settings` and `--add-dir` — the per-attempt
wiring `env_mgr` prepared; restart loses all context plus the zone. An agent that
returned nearly-finished work and merely failed to publish is the cheapest
possible fault, and `restart` is the most expensive possible reaction to it.

**Where the live handle comes from is the module's one unsolved input**, and §9
states it rather than inventing a route.

---

## 7. Escalation — `base.py`

### 7.1 The chain is `Task.parent`

```python
def _escalate(self, unit: Unit, why: str) -> None:
    task = self._task_mgr.get(unit.task_id)
    self._recorder.write(record(kind=ESCALATED, task_id=task.id, attributes={"why": why}))
    if task.parent is None:
        return self._to_user(unit, why)                       # §7.3
    parent = self._task_mgr.get(task.parent)
    self._monitor_for(parent).report(rekeyed(unit, parent.id))
```

**Up the task tree, never the monitor topology** (spec §3.1). Global monitors are
a flat pool; the tree that matters is `task_graph`'s, and `Task.parent` is the
edge `unfold` sets. So the target is always *the monitor of my task's parent*,
whichever kind either one is.

`_monitor_for(task)` is `registry.get(f"monitor:{task.monitor_spec or DEFAULT}")` —
the same by-name resolution `task_graph` design §3.8 already specifies, and the
same one that rejects an unregistered name.

**The escalated record is re-keyed to the parent**, with the child's id in
`attributes["from_task"]`. It must be: the parent's monitor will run it through
`_transition`, and `_transition` refuses a task that is not the one it is
handling. A record that kept the child's `task_id` would be a scope violation on
arrival.

### 7.2 It terminates, and the reason is structural

`unfold` sets `parent` on tasks it has just created and therefore cannot close a
loop (`task_graph` design §8.5). A cycle would need an explicit `parent`
assignment from some future linking API, and §8.7's load check covers the
declared graph. **The walk needs no visited set**, and this document says so
rather than adding one defensively — a second guard for one invariant is the
two-writers failure.

### 7.3 The top of the chain is a seam with nothing behind it

`UserSink.deliver`. The alpha registers `NullUserSink`, which records
`kind=ESCALATED` with `attributes={"target": "user"}` and does nothing else.

**How a monitor reaches a human is unspecified anywhere in the system** — spec
§11 carries it as an open question, and inventing a channel here would be adding
a requirement. What the alpha owes is that the chain is defined and the arrival
is recorded, so a branch that outruns every monitor is visible rather than silent.

### 7.4 The same walk carries `SUBGRAPH_DONE` upward

The non-leaf re-entry (spec §5.3) is one hop of the escalation walk with a
different payload:

```python
def _notify_parent_done(self, unit: Unit) -> None:
    task = self._task_mgr.get(unit.task_id)          # the is_end subtask
    if task.parent is None:                          # the system whole task
        return
    parent = self._task_mgr.get(task.parent)
    self._monitor_for(parent).report(rekeyed(unit, parent.id, SUBGRAPH_DONE))
```

**It reuses `_monitor_for` and the `rekeyed` re-addressing from §7.1**, and for
the same reason: the parent's monitor will run this through `_transition`, which
refuses a task that is not the one it is handling. A record keeping the child's
`task_id` would be a scope violation on arrival.

**The subtask's monitor does not transition the parent.** It reports; the parent's
monitor decides nothing and advances (§6.1). That is criterion 24, and it is why
the planned channel needs no exception to the scope guard.

**`parent is None` returns rather than escalating to the user.** The root of the
task tree is the *system whole task* (`task_graph` spec §3.2.1), and its `is_end`
completing means the system finished — which is a completion, not something to
surface as an escalation. **The two walks share a mechanism and differ at the
top**, and this is the one place that difference shows.

---

## 8. What this module owes the gate, and what it does not

Spec §4.1.0 puts the completeness gate **in the runner**, which is `agent`'s.
Criterion 4 is nevertheless a monitor criterion, so the boundary has to be exact.

| | Owner |
|---|---|
| Running the four checks between the main phase and `OUTPUT_VALIDATING` | **`agent`.** Its runner already holds `task`, the handoff store, and the `AgentResult` |
| The four `EventKind` values the checks produce | **this module**, §3.2 |
| `report()`, and the record reaching disk before the buffer | **this module**, §5.3 |
| Deciding what happens after a failure | **this module.** The runner reports and takes no corrective action of its own |
| `Recorder.open(task, attempt)` when the attempt starts | **`agent`.** §9.2 records that this is a new obligation |
| **Reporting a phase that finished *normally*** | **`agent`.** The same `report()` call, at the same place. A gate that passes is a `PHASE_DONE`; a gate that fails is one of §3.2's four. **The runner does not branch on which monitor call to make** — it makes one, with a different `kind` |
| The thresholds the budget check compares against | `Budget`, §5.1, registered as `budget` and read by the runner |

**The runner never pushes**, and criterion 18 is the test for it: the failing
path calls `report()` and returns. A runner that retried internally would satisfy
every other criterion and still be wrong — it would be a second failure policy
the record cannot see and the analysing dispatcher cannot reach.

**The cycle stays below the scheduler.** Nothing in the gate loop moves task
status, reaches the scheduler, or calls `on_done`; the graph sees one task
`RUNNING` throughout, however many times it cycles.

---

## 9. What this module needs from `agent`, and how O1 closed

### 9.1 O1 is closed, and it closed by changing the other side

Rev. 1 recorded, as this module's largest gap: *"Nothing in the system maps a task
id to the live executor running it."* What existed and did not answer it:

| | Why not |
|---|---|
| `AgentMgr.by_task(tid)` | Returns `Agent` **records**. `task_graph` design §6.4 is explicit: *"A restored instance is a record, not a live agent"* |
| `Execution.agent_id` | An id, with no resolver to a running process |
| The `report()` call | Spec §5.1 freezes the inbound surface at `report` and `set_task`, and a live handle cannot be a field of a persisted record |

Rev. 1 proposed `executor_of(task_id) -> Executor | None` on `agent.Runner`,
reasoning that the runner *must already* hold the mapping because
`stop(task_id, on_stopped)` takes an id. **Two things were wrong with that.**
"Already" was false — `agent/` is declaration-only and the sole `TaskRunner` is
`FakeRunner`. And the map a runner is forced to hold is
`dict[TaskId, tuple[Task, Agent, OnDone]]` (`FakeRunner.running`), which contains
**no executor at all**; `executor_of` assumed more than existed
(`scratch/design/findings-arch-ours.md`).

**`agent` design rev. 7 supplies the owner rather than the accessor.**
`TaskAttempt` — one object per dispatch, holding the thread, the executor and the
next phase — and `Runner.attempt_of(task_id)` reaches it. This module gets what it
needed and, because the same object owns the thread, two other gaps closed with
it: who spawns and joins the phase thread, and what survives a non-leaf's
subgraph.

**It was not a missing accessor. It was a missing object**, and the survey said
so before this document did: four of the six comparable systems make their
runner-equivalent per-unit-of-work, and each does it because that object holds the
per-task state a supervisory loop reads (`scratch/design/findings-arch-workflow.md`).
Ours was shared and kept that state nowhere.

### 9.2 What is still owed, and by whom

| Required | From | Status |
|---|---|---|
| `Runner.attempt_of`, `Runner.resume` | `agent` design §7.1, §7.5 | **Designed rev. 7**, not built |
| `TaskAttempt.wake()` / `release()` | `agent` design §7.5 | **Designed rev. 7**, not built |
| A `report()` route from the attempt, for **both** channels | `agent` | Does not exist. One handle, resolved by name, used at every phase boundary |
| `Recorder.open` at attempt start | `agent` | An obligation spec §9's table does not list, because §8.3's container rule predates this document placing the call. Reported, not edited |
| `install_excepthook` called once | the composition root | §5.6. Process-global, so not this module's to install |

**One consequence worth stating plainly.** Until `attempt_of` exists,
`PusherMonitor.decide` returns `Escalate` where it would have returned `Push` —
a working monitor with a disabled pusher, costing criterion 12 while 4, 9, 15, 17
and 18 hold. **The planned channel is unaffected either way**, because §6.1 needs
`resume`, not the executor.

One item remains in the sequencing class rev. 1 opened:

- **`Task.parent` and the four verbs are designed and not yet implemented.**
  `task_graph` design rev. 11 puts `cancel` / `restart` / `fail` / `replace_with`
  on `Task` (§3.4) and adds `parent` (§3.3); the shipped `models.py` is at design
  rev. 10 and has neither — the verbs are still `Scheduler.stop` /
  `resume_task`. Spec §9's table says these "exist", and against the *design* they
  do. §5.5 and §7 are written to the design, so they are correct and untestable
  until rev. 11 lands. Not a defect; a sequencing fact, and it is what puts steps
  5 and 6 last in §12. **Rev. 2 adds `enter_phase` to that list** — §6.1's one
  line of behaviour is a `task_graph` design rev. 11 method too.

---

## 10. Build versus adopt

| Piece | Decision | Why |
|---|---|---|
| The workqueue triple | **Copy the shape, ~60 lines** | `client-go` is Go. The shape is three collections and an invariant; a dependency on a Kubernetes client to get it would be absurd, and the invariant is one assertion in a test |
| `queue.Queue` | **No** | `shutdown` is 3.13-only against a 3.10 target, and its `immediate=True` semantics discard pending items silently. Neither dedup nor merge is expressible on it |
| OpenTelemetry SDK | **No — names only** | Three packages, and it answers the wrong question: OTel is emit-only, and §6's decision must read back what it wrote. The stable `exception.*` names cost nothing to adopt as naming |
| Sentry SDK | **No — the split only** | The event/fingerprint/issue idea is the value; the client is a network reporter |
| A logging library | **No** | Spec §8.1: the carrier was never open. Logging is a projection of the record — one handler, rendering at the severity the record already carries |
| pydantic v2 | **Yes** | Already installed, and `task_graph.models.Model` is the configuration this repository has settled on |
| `threading.Condition` | **Yes** | Stdlib. The buffer needs exactly one wait/notify |

**Net new runtime dependencies: none.**

---

## 11. Test plan

`tests/monitor/`, and two files that extend existing suites rather than
duplicating them.

| # | Criterion | Test |
|---|---|---|
| 1 | default vs named monitor | `test_registry.py::test_monitor_spec_resolves_by_name` |
| 2 | unregistered name rejected, value named | `test_registry.py::test_unknown_monitor_spec_names_the_value` |
| 3 | its own loop | `test_loop.py::test_mainloop_is_not_the_agents` |
| 4 | the gate blocks `OUTPUT_VALIDATING`; four failures; cycle below the scheduler; runner never pushes | `test_gate.py::test_four_failures_each_report`, `::test_no_status_move_during_cycle`, `::test_runner_takes_no_corrective_action` |
| 5 | refused `put` ≠ never called | `test_record.py::test_absent_output_kind_does_not_claim_malformed` — and the positive half is `agent`'s, §9 |
| 6 | every action is a transition call | `tests/task_graph/test_authority.py`, **unamended**, with a monitor added as a caller |
| 7 | blocks on the lock, holds nothing | `test_concurrency.py::test_transition_blocks_then_proceeds` |
| 8 | may transition only its current task | `test_scope.py::test_global_monitor_refuses_another_task` |
| 9 | every exception recorded; push attempted vs ineffective vs never | `test_record.py::test_push_attempted_ineffective_never`, `::test_handling_failure_is_recorded` |
| 10 | `set_task` is the only way | `test_scope.py::test_no_discovery_without_set_task` |
| 11 | not a task | `test_identity.py::test_monitor_holds_no_lease_no_zone_and_is_not_in_the_graph` |
| 12 | the alpha's reaction is the pusher; recording and escalation still work | `test_pusher.py::test_decision_table`, `test_escalation.py::test_unpushable_still_records` |
| 13 | records, not log lines | `test_record.py::test_suite_passes_with_logging_disabled` |
| 14 | empty record set ≠ missing one | `test_record.py::test_open_creates_an_empty_set` |
| 15 | `report()` does not block; handled exactly once | `test_buffer.py::test_add_never_blocks`, `::test_requeued_exactly_once_while_processing` |
| 16 | both validator outcomes arrive, distinguishable by `kind` | `test_validator_route.py::test_fail_and_unreached_are_different_kinds` |
| 17 | escalates up the tree to the root, each step recorded | `test_escalation.py::test_walks_parent_chain_to_root` |
| 18 | no recovery outside a monitor decision | `test_gate.py::test_runner_takes_no_corrective_action`, `test_validator_route.py::test_validator_does_not_rerun_itself` |
| 19 | a planned event advances and nothing else; no model on the path | `test_planned.py::test_advance_is_one_transition`, `::test_agent_monitor_uses_the_same_advance` — the second builds an `AnalysingMonitor` with a spy agent and asserts the agent is never called |
| 20 | the planned queue does not collapse; one task not handled twice across both queues | `test_buffer.py::test_planned_queue_never_collapses`, `test_loop.py::test_one_task_one_handling_across_queues` |
| 21 | a leaf holds one thread for its three phases | `test_threads.py::test_leaf_holds_one_thread` |
| 22 | a non-leaf holds none during its subgraph; the re-entry is the same `Execution` | `test_threads.py::test_non_leaf_holds_no_thread`, `test_planned.py::test_reentry_pushes_no_second_execution` |
| 23 | the scheduler is not in the re-entry | `test_planned.py::test_scheduler_untouched_by_reentry` — a spy on `Scheduler` sees no call, and `pools[WAITING_RESOURCE]` never contains the parent |
| 24 | a subtask's monitor reports to the parent's; it does not transition it | `test_planned.py::test_subtask_monitor_does_not_transition_parent` |
| 25 | an escaped thread exception produces a record and reaches the user | `test_liveness.py::test_excepthook_records_and_surfaces` |
| 26 | a stalled loop is detected after N stale periods, not one | `test_liveness.py::test_stall_needs_n_consecutive`, `::test_one_slow_round_is_not_a_stall` |

Beyond the criteria:

| | |
|---|---|
| `test_buffer.py::test_invariant_holds_under_concurrent_add` | every element of `_order` is in `_dirty` and not in `_processing`, asserted after a fuzz of interleaved `add`/`get`/`done` |
| `test_buffer.py::test_collapse_merges_and_loses_nothing` | five reports for one task collapse to one unit carrying five records — the case `probes-monitor/p3` showed the naive version last-wins |
| `test_buffer.py::test_shutdown_refuses_loudly_and_drains` | `add` raises `BufferClosed`; queued units are still delivered |
| `test_loop.py::test_handler_exception_does_not_kill_the_loop` | the `p4` failure, as a regression test |
| `test_loop.py::test_sweep_runs_once_per_idle_period` | the §4.3 seam exists and is called |
| `tests/interfaces/test_import_rules.py` | `monitor` imports only `task_graph` |
| `tests/interfaces/test_pushable.py` | `AgentBackend` satisfies `Pushable` — §2.2's drift guard |
| `test_record.py::test_every_kind_is_routed` | every `EventKind` member is either in `PLANNED` or reaches the buffer. A kind added later cannot fall through both |
| `test_planned.py::test_starvation` | a saturated planned queue still reaches `buffer.get` — the ordering in §5.4 prioritises, it does not starve |
| `test_liveness.py::test_clock_is_injected` | the stall test uses a fake clock and does not sleep; a monotonic jump backwards is not a stall |

---

## 12. Implementation order

Each step leaves the suite green.

| | | Unblocks |
|---|---|---|
| 1 | `record.py` — id, kind, `PLANNED`, record, `Recorder` | criteria 5, 9, 13, 14 |
| 2 | `buffer.py` — both queues | criteria 15, 20, and the invariant test |
| 3 | `protocols.py` + `.pyi`, and the `interfaces.md` rows | criteria 1, 2 |
| 4 | `base.py` — `report` + routing, `mainloop`, `_run_guarded`, `_transition`, `_sweep` | criteria 3, 6, 7, 8, 10, 11 |
| 5 | **`_beat` + `check_liveness` + `install_excepthook`** | criteria 25, 26. **Independent of everything below**, and first because the rest of this module is now on the happy path |
| 6 | `_advance` (§6.1) | criterion 19 — **blocked on `enter_phase`**, §9.2 |
| 7 | `_escalate` + `_notify_parent_done` + `NullUserSink` | criteria 17, 22, 23, 24 — **blocked on `Task.parent`**, §9.2 |
| 8 | `pusher.py` | criterion 12, partially — **the push half is blocked on `attempt_of`**, §9.2 |
| 9 | `TaskAttempt` + `resume` + `attempt_of`, in `agent` | criteria 21, 22. Not this package |
| 10 | the gate, in `agent` | criteria 4, 18. Not this package |
| 11 | the validator's route | criteria 16, 18 |

**Step 5 moved to the front in rev. 2, and the reason is the whole of rev. 2.**
While the monitor only handled exceptions, its own liveness could follow the
features it protected. Now that every phase advance runs through this loop, a
monitor that can die silently is a system that can stop silently — so the two
mechanisms that make that visible are built before the thing they watch.

Steps 9–11 are in other packages and are listed because criteria 4, 16, 18, 21 and
22 are not testable without them. They are the propagation set, not this module's
implementation.

---

## 13. Deviations from the spec

| | |
|---|---|
| **The interface has five members, not two** | Spec §5.1: *"The inbound surface is `report()` and `set_task`, and nothing else."* `Monitor` also carries `name`, `mainloop` and `stop`. Those are lifecycle and identity — the composition root keys on `name`, and spec §1.1 requires the loop to exist. The *inbound* surface is still two |
| **`EventId` subclasses `task_graph.ids._Id`, a private name** | The alternative is a fourth id in `task_graph/ids.py`, which would make `task_graph` carry a monitor concept — the dependency §2.2 exists to avoid. Recorded as the smaller of two costs, not as a clean choice |
| **`Pushable` duplicates part of `AgentBackend`** | §2.2. Two declarations of one shape, and the guard is a test rather than the type system |
| **`OUTPUT_MALFORMED` is not an `EventKind`** | §3.2. Spec §4.1.1 measured that a malformed handoff never reaches storage, so no phase of this module can observe it. Its record is producer-side and belongs to `agent` |
| **The default fingerprint is fixed here** | Spec §11 leaves it open. The alpha excludes `attempt`; it is one function so reversing it is one edit |
| **`install_excepthook` is not on `Monitor`** | Spec §5.4 states the requirement without saying whose it is. `threading.excepthook` is process-global, so a module installing it on construction would be a library mutating interpreter state and two monitors would fight over it. §12 step 5 puts it in the composition root |
| **`_advance` cannot be overridden** | Spec §2.2 says the planned channel is program-always. This design makes that structural by putting `_advance` on `BaseMonitor` and leaving only `decide` to subclasses — stronger than the spec asks, and the only form in which criterion 19 is testable by construction |

---

## 14. New open questions

| | |
|---|---|
| ~~**O1 — who gives the monitor a live handle**~~ | **Closed in rev. 2** by `agent` design rev. 7's `TaskAttempt` and `Runner.attempt_of` — §9.1. It was not a missing accessor but a missing object, and rev. 1's proposed `executor_of` assumed a `TaskId → Executor` map that no runner held |
| **O6 — the planned queue's latency is now everyone's latency** | §5.4 takes planned work first and without waiting, so the cost of an advance is a thread handoff rather than a poll period. **That is derived from the design, not measured**, and it is now multiplied by every phase of every task. The number that would settle it is the same one O4 names |
| **O7 — what a stalled monitor's tasks should do** | §5.6 reports; it does not recover. Adopting them to another monitor, failing them, or waiting are three different answers and the spec (§11) does not choose. **Rev. 2 raised the cost of getting this wrong**: those tasks are no longer merely unwatched, they are stopped |
| **O8 — `_beat` measures the loop, not the handler** | A monitor blocked inside one long `_handle` stops beating and reads as stalled; one spinning fast on nothing beats happily. Both are the wrong answer, and neither is distinguishable from a timestamp. A work counter beside the timestamp would separate them, and is not built |
| **O2 — what `PUSH_INEFFECTIVE` actually measures** | §6 declares a push ineffective when the *same attempt* fails the gate again. An agent that improved the delivery without completing it is recorded identically to one that did nothing, and the two are different situations |
| **O3 — `read` is a scan of a store kind** | §3.4. Fine at alpha scale, and it is on the decision path of every push, so it is the first thing that will hurt. A per-attempt index is the obvious answer and is not built |
| **O4 — the loop's period is unmeasured** | §5.4 defaults to 1.0 s. Nothing measured what latency an exception can tolerate, and the number that would settle it — how long a monitor blocks on the scheduler's `RLock` — is spec §11's third question and is not measurable while `FakeRunner.start` returns immediately |
| **O5 — one monitor thread per monitor, and no bound on monitors** | `build_registry` registers whatever `monitors=` holds. A per-task monitor for every task would be a thread per task, which is the cost the global form exists to avoid — and nothing enforces choosing it |
