# Agent — Design

| | |
|---|---|
| Status | Draft — stage two of spec → design → test & code |
| Revision | 7 — 2026-08-28. **The runner becomes a factory; `TaskAttempt` becomes the thread's owner** (§7.1, §7.5). One attempt object per dispatch holds the thread, the executor and the next phase — answering in one place who spawns the phase thread (which no design document said, for either loop), what maps a task id to a live executor (`monitor` design O1), and what survives a non-leaf's subgraph so its re-entry is the same `Execution`. The attempt no longer advances itself: each phase ends in `report()` and the monitor calls `enter_phase` (§7.2). An agent keeps its `mainloop` and stops owning a thread — it borrows the task's (§5.1.1). **No thread pool**, and not for the 50 μs: the scheduler's leases are already the admission control, and a pool would be a second one it cannot see (§7.5). (rev. 6: 2026-08-27. **§7.2.1 says what the runner hands the executor**, which rev. 5 described the phases without ever stating. Found by the spec-key-to-runtime trace: `body` was declared and reachable and nobody walked the route. `prepare` now takes the agent spec, so the runner passes it. (rev. 5: 2026-08-27. **An agent has its own `mainloop()`**, following spec §4.3 rev. 5 (§5.1, §5.1.1). Rev. 4 had five verbs and no answer to "who is executing after `start()` returns". Every synchronous verb becomes sugar this level wraps, as one rule rather than per method. The program executor gets a loop too. (rev. 4: 2026-08-27. **The stage-three consistency pass.** §7.1 named three registry entries and two of them were registered by nobody; the names and their owners are now stated, and [`../../docs/interfaces.md`](../../docs/interfaces.md) §2 is the normative listing. O6 splits into a requirement `validator` owns and a mechanism this module owns, which is what the two documents were actually disagreeing about. (rev. 3: 2026-08-27. D1 is resolved upstream: main spec §4.8 rev. 9 makes every task have an agent and `kind` the thing that varies, so §9's program executor **is** what a task without an AI has (§9, D1). No interface changes. (rev. 2: 2026-08-26. Spec-consistency pass: the two levels are **two protocols**, `Executor` and `AgentBackend`, because rev. 1 read spec §1.1's level 1 as `TaskRunner` and merged the levels (D5). The runner holds level 1 only; the program executor raises nothing. (rev. 1: initial)))))) |
| Implements | [`spec.md`](spec.md) rev. 6, acceptance criteria 1–16 |
| Language | Python ≥ 3.10. pydantic v2; `claude-agent-sdk` an **extra**, imported lazily (§8.1) |

---

## 1. Scope

This document turns [`spec.md`](spec.md) into files, classes, and interfaces. It
adds no requirements. Where it makes a choice the spec left open, the choice is
stated here; where implementing the spec exposed a contradiction, §14 says so
rather than papering over it.

The spec's 16 acceptance criteria are the definition of done. §12 maps every one
to a named test.

**This document specifies interfaces, not bodies.** A method is a signature and
a sentence. A body appears only where the ordering of steps *is* the design
decision — which is §6.2's selection chain, §7.2's phase loop, §8.4's interrupt
drain, and nothing else.

### 1.1 What this module owns

- The `AgentSpec` model and what admits one (§3).
- `AgentSpecRegistry` — one of the four the main design reserves — with its
  load-time checks (§4).
- The two interface levels — `Executor` and `AgentBackend` — and the status
  model (§5).
- **Backend selection**, the three-source precedence chain and its structured
  result (§6).
- **The real `TaskRunner`** — the one `task_graph` declares and does not
  implement (§7).
- Two backends: the `claude-agent-sdk` adapter (§8) and the program executor
  (§9).

### 1.2 What it does not

| Deferred to | What |
|---|---|
| `task_graph` spec §3.3, design §3.5 | **The runtime `Agent` record** — id, spec name, `task_id`, `HandoffRef`s. §10 is about what the *system records*; there is no second agent object |
| `task_graph` spec §3.2.2 | **Permissions.** A versioned *task* attribute. This module carries nothing and interprets nothing (§3.3) |
| `env_mgr` | **Preparing the environment**, and interpreting `Permissions` into a confinement zone (its spec §4) |
| o11y — [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §1 | **Logging**, and per-run metrics |
| A module that does not exist yet | **The format transform helper.** §4.4 of the spec calls it "an independent module"; §3.5 here places it and declines to build it |
| The backend | **What happens inside it** — spec §4.2 |
| The agent | **Its own durability.** Spec §1.3: its business, not the system's. Nothing here checkpoints, resumes or replays a backend |

---

## 2. Layout and import graph

```
agent_sys/agent/
├── __init__.py
├── spec.py            AgentSpec, Kind, BackendDecl, KnowledgeRef
├── registry.py        AgentSpecRegistry(SpecRegistry)
├── backend.py         Executor + AgentBackend protocols, AgentStatus,
│                      AgentResult, AgentHistory, BackendUnsupported
├── selection.py       Selection, select_backend, BackendUnavailable
├── runner.py          Runner — the real TaskRunner
├── backends/
│   ├── __init__.py    the entry-point / dotted-path resolver
│   ├── claude_sdk.py  the claude-agent-sdk adapter — level 2   (extra)
│   └── program.py     ProgramExecutor — level 1 only
└── docs/
```

Import direction, one way, left to right:

```
spec   backend   →   selection   →   runner
  ↓       ↑                            ↓
registry  └──── backends/ ─────────────┘
```

- **`backend.py` imports nothing of ours.** It is two protocols and four value
  types, so both `selection` and `backends/` can depend on it without a cycle.
- **`runner.py` does not import `backends/`.** It resolves a backend through
  `selection`, which resolves it through the registry at use time. The one
  concrete backend a runner ever names is none.
- **`backends/claude_sdk.py` is never imported at module scope by anything
  here.** §8.1.

`agent` imports `task_graph` for `TaskRunner`, `Task`, `Agent`, `TaskId` and
`TaskStatus`. It does not import `handoff`, `validator` or `closure`; where it
needs one it takes it from the registry by name — the discipline
[`../../task_graph/docs/design.md`](../../task_graph/docs/design.md) §3.4 sets
out, and the reason `engineer_principle.md` §1 gives for it.

---

## 3. The agent spec — `spec.py`

### 3.1 The model

Spec §3.1's nine keys, as a pydantic model. It is the *validated* form: the JSON
Schema pass in the main design §4 has already run, so this model restates the
shape rather than defending it.

```python
class Kind(str, Enum):
    AI = "ai"
    PROGRAM = "program"
    HUMAN = "human"


class BackendDecl(BaseModel):
    key: str                       # unique within this spec
    backend_entry: str             # §6.3 — a dotted path or an entry-point name
    config: dict[str, Any] = {}


class AgentSpec(BaseModel):
    name: str
    version: str                   # maintenance metadata; nothing reads it
    description: str
    kind: Kind
    backends: list[BackendDecl] = []      # ordered: the order IS the preference
    env: dict[str, Any] = {}
    knowledge: list[KnowledgeRef] = []
    rules: list[str] = []                 # handoff-free material, §3.5
    hooks: list[str] = []
    skills: list[str] = []
```

**`backends` is a list, not a dict, even though the spec permits either.** The
order is load-bearing (spec §3.3, criterion 3), and a dict that carries an
ordering is a dict whose ordering nobody can see at the call site. A mapping
form in the YAML is normalised to this list at load, preserving declaration
order.

**`version` gets a field and no reader.** Closure spec §1.2 says it is
maintenance metadata. Stating that here is cheaper than a future reader
discovering the field and assuming it means something.

**`kind: human` is declarable and unimplemented, deliberately.** Spec §9 leaves
it open — "what `query()` and `interrupt()` mean for a person is undecided" — so
the enum carries the value and nothing satisfies it. A `kind: human` spec
**loads** and fails at selection with `BackendUnavailable` naming the kind, which
is the honest outcome: the spec is well-formed and this alpha cannot run it.
Rejecting it at load would make the spec ill-formed, which it is not.

### 3.2 Knowledge is a reference, not content

Spec §3.4: knowledge arrives as **knowledge handoffs**, not inline prose.

```python
class KnowledgeRef(BaseModel):
    kind: str                      # a handoff kind name — resolved in the
                                   # handoff registry at load (§4)
    knowledge_type: str            # one of the six, and the list is open
    required: bool = False
```

**`knowledge_type` is a `str`, not an enum.** Spec §3.4 says "six is where it
stands, not a closed vocabulary". An enum would make the seventh type a code
change in this module, which is the opposite of what the spec asked for. The six
are a documented constant used for the coverage report, not a validator.

**Type 6 — runtime-generated — is declarable but produced later.** It is a
knowledge handoff like the rest, so the registry check at load is the same: does
the *kind* resolve. Whether an instance exists is a runtime question and no load
check asks it. **Whether such a handoff outlives its run, and whether a
later task may consume it, is spec §9's third open question and this design does
not settle it** — it declares the reference and stops there.

### 3.3 Permissions are not here, and nothing carries them

Spec §3.2 and `task_graph` spec §3.2.2 both put permissions on the task.
`task_graph` design §3.5 carries the type; `env_mgr` interprets it.

**This module has no permissions field, no permissions parameter, and no
permissions argument on any backend method.** That is criterion 5, and it is
satisfied structurally rather than by a check: there is nowhere to put one.

The runner receives a `Task` and hands the backend an already-prepared
environment. What the environment permits was decided before the backend existed.

### 3.4 Material is stored in Claude Code's canonical form

Spec §4.5. `rules`, `hooks` and `skills` hold **paths into the task package**,
in Claude Code's format, and this module does not parse them — it hands them to
`env_mgr` to deploy and to the backend to consume.

`knowledge` is format-free because it is a handoff (§3.2), which is why it has a
different type from the other three.

### 3.5 The transform helper is placed here and built elsewhere

Spec §4.4 calls the transform helper "an independent module … Not part of the
agent spec". This design agrees and goes no further than placing it:

> **It is not in `agent/`, and this revision does not create a package for it.**

Two reasons, and the second is the one that decides it:

- `engineer_principle.md` §2: a new module needs a reason no existing one can
  host it. The helper converts *between two third parties' formats*. It needs
  neither `AgentSpec` nor `AgentBackend`, and putting it under `agent/` would
  make every consumer of a Cursor rule file import the Claude SDK adapter's
  neighbourhood.
- **Its release cadence is the union of the harnesses' cadences.** Measured:
  `rulesync`, which does exactly this job for ~30 harnesses, has **299 releases
  and seven major versions inside two weeks**, and its recent commits are almost
  entirely per-target fixes. A module on that cadence cannot share a version with
  a spec model that changes twice a year.

**And the acceptance criterion it exists to satisfy is not testable as written.**
That is O1; §14 does not swallow it.

---

## 4. The registry — `registry.py`

`AgentSpecRegistry(SpecRegistry)`, the fourth of the four the main design §5
reserves. The base supplies the dict, the duplicate policy — error by default,
byte-identical re-registration a no-op — and the error shape that enumerates
candidates.

Spec §3.6's four load-time checks, and where each lands:

| # | Check | Where |
|---|---|---|
| 1 | The YAML validates; the name is unique | the base; the schema pass ran earlier |
| 2 | **Every declared backend resolves** | `add()`, via `backends.resolve(entry)` — §6.3 |
| 3 | **Every knowledge handoff kind resolves** in the handoff registry | `check_knowledge(handoff_specs)`, a second pass |
| 4 | **Knowledge coverage is reported** — warn by default, fatal under the flag | same pass |

**Check 2 resolves the entry; it does not probe availability.** Resolution
answers "does this name denote something"; availability answers "can it run
here", and the second is a runtime question whose answer changes after `env_mgr`
deploys. Conflating them would take the availability reading at the one moment
it is guaranteed to be wrong. SQLAlchemy draws the same line deliberately —
`get_dialect()` resolves the name with the driver absent, and `create_engine`
fails later.

**Check 3 is a separate pass, for the reason the main design §6 already gives**:
a registry cannot see another registry's contents during its own load. It runs
where the other cross-registry checks run, and it takes the handoff registry as
an argument rather than reaching for it.

### 4.1 Knowledge coverage: a warning that names what is absent

Spec §3.5. Default is a warning naming each absent piece; a run-config flag makes
it fatal.

```python
class KnowledgeReport(BaseModel):
    spec: str
    missing: list[KnowledgeRef]       # declared required, kind unresolvable
    types_absent: list[str]           # of the six, which are not represented
```

**The report is a value, not a log line.** The warning is rendered from it, and
the fatal mode raises from the same value, so the two modes cannot drift into
disagreeing about what is missing. Criterion 2 asserts exactly that: the same
spec loads in one mode and is rejected in the other.

**`types_absent` is advisory and never fatal.** A spec that declares no
`runnable` knowledge is not malformed; §3.4's list is a checklist for a human,
and treating a checklist as a schema is how "strongly suggested" becomes
"hardcoded mandatory" — the thing spec §3.5 exists to prevent.

---

## 5. The two levels

### 5.1 Two protocols, because the spec has two levels

Spec §1.1 draws the levels and its principle 2 says to keep them apart, so they
are **two protocols, not one with holes in it** — `backend.py`:

```python
class AgentStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class Executor(Protocol):
    """Level 1. What the task runner talks to. Thin, and every executor
       satisfies it — AI, human, or shell script."""

    status: AgentStatus

    # ---- what a caller invokes ----
    def start_async(self, on_started: Callable[[], None]) -> None:
        """Return immediately. Invoke `on_started` when the agent really is
           running, which is later than 'asked to start' (§8.3)."""

    def wait(self) -> AgentResult: ...
    def start(self) -> AgentResult:
        """Sugar: start_async then wait."""

    def stop(self) -> None: ...

    # ---- what actually runs it. §5.5 ----
    def mainloop(self) -> None:
        """Drive this agent. Owns its `status`, services its message queue,
           and is what `start_async` hands work to. An adapter implements this;
           nothing above level 1 calls it."""


class AgentBackend(Executor, Protocol):
    """Level 2. The AI-harness abstraction, and only AI executors have one."""

    def interrupt(self) -> None: ...
    def instruct(self, message: str) -> None: ...
    def query(self) -> AgentHistory: ...
```

**The split is exactly where spec §1.1 puts it.** Level 1 is "start, stop,
status, result"; level 2 is "history, interrupt, message queue, hooks,
permissions, sessions". The first four names are `Executor`; `interrupt`,
`instruct` and `query` are the three that only an AI harness has.

**A program executor implements `Executor` and never touches level 2**, which is
§1.1's sentence turned into a type rather than into a runtime error. It is also
what lets §7 state that the runner needs level 1 only — and that is criterion 6
in its strongest form: the runner cannot call an AI-only method because it does
not have one to call.

**No task parameter, anywhere.** Spec §4.3: the task uuid is already in the
runtime agent's schema. A backend that needed it would be a second source of
truth for a binding `task_graph` owns.

`AgentResult` and `AgentHistory` are the two value types crossing the seam:

```python
class AgentResult(BaseModel):
    status: AgentStatus              # a terminal one
    usage: dict[str, float] = {}     # what on_done carries; §8.5
    detail: str = ""                 # for a human, never parsed


class AgentHistory(BaseModel):
    entries: list[dict[str, Any]]    # the backend's own shape, opaque here
    session_ref: str | None = None   # §8.6
```

**`AgentHistory.entries` is deliberately untyped.** Spec §7: the history "is the
backend's data, fetched on demand, not the system's record". Giving it a schema
here would make it the system's.

### 5.1.1 `mainloop()`, and why the interface was incomplete without it

Spec §4.3 rev. 5. The question that settles it is the concrete one: **`start()`
returns immediately — then who is executing?** Rev. 3 of this document had five
verbs and no answer; an interface of five verbs with nothing behind them is not
an interface.

**An agent is a live, stateful thing.** It has a status that changes while nobody
is calling it, a message queue that has to be serviced, and a backend session
that outlives any one call. Without a loop of its own there is nothing for a
caller to interact *with* between calls.

```
caller                      the agent
──────                      ─────────
start_async(cb) ──────────► queue the work, return
                            mainloop() picks it up, status = DEPLOYING
                            connect() handshake returns  ──► status = RUNNING, cb()
instruct("...") ──────────► queue it
                            mainloop() delivers it
                            ...
                            terminal ──► status = FINISHED
```

**The loop is the agent's; the thread is not.** Rev. 3 of this document said "one
thread per agent" and left unsaid — as `scratch/design/findings-arch-ours.md`
confirmed across every design document, with zero occurrences of `Thread(` or
`.join()` anywhere — **who creates that thread and who ends it.**

**The answer is that the task owns the thread and the agent borrows it** (§7.5).
`TaskAttempt` starts one thread per dispatch; during the main phase it is
`mainloop()` that runs on it, and when the main phase ends the agent hands it
back. So an agent still has a loop of its own — the question "`start()` returns
immediately, then who is executing?" still gets an answer — while the *lifetime*
question has an owner it did not have.

**Two things fall out.** A graph running K leaf tasks has K threads, not K agent
threads plus K task threads; and `ROADMAP.md` §7.1's refinement — attaching an
agent's loop to a shared round-robin thread — is now about sharing *task* threads,
which is the same trade the global monitor already makes.

**The monitor's loop is still a separate thread, and must be.** Sharing a thread
between the watched and the watcher is the one merge this design does not make —
`monitor` spec §1.1, and the failure mode a watchdog exists to avoid.

**Two loops, two jobs, and they are not one mechanism.** The monitor's loop
(`task_graph` spec §3.5) watches for the *task's* exceptions — an agent that
stopped behaving, a broken node. This one *runs the agent*. Conflating them would
give the thing being watched and the thing watching one heartbeat, which is the
failure mode a watchdog exists to avoid.

**Every synchronous verb is sugar this level wraps.** Spec §4.3: an adapter
implements the asynchronous form and does not implement the synchronous one. Rev.
3 applied that only to `start()`; it is now the rule, so a backend cannot ship a
`stop()` that blocks differently from every other backend's. One helper does it
for all of them.

### 5.2 `TaskRunner` is the caller of level 1, not level 1 itself

Easy to get wrong, so it is stated: **the levels are on the executor side of the
seam.** `TaskRunner` is what *talks to* level 1 (spec §1.1), and spec §4.1's
whole point is that it is neither level.

```python
class TaskRunner(Protocol):                     # task_graph/runner.py — frozen
    def start(self, task: Task, agent: Agent, on_done: OnDone) -> None: ...
    def stop(self, task_id: TaskId, on_stopped: OnStopped) -> None: ...
```

`task_graph/runner.py` declares it and ships only a fake; its design §7.1 says so
explicitly. **This module supplies the real one** (§7), and the protocol does not
change.

Criterion 6 — "a backend is not a `TaskRunner`" — then holds three times over:
the two have no method in common, `TaskRunner.start` takes two arguments that
neither protocol here mentions, and neither can be passed where the other is
expected.

### 5.3 An unimplementable method raises

Spec §3.3.1: no capability matrix, no per-capability degradation.

```python
class BackendUnsupported(NotImplementedError):
    """Raised by a backend whose adapter does not implement a method.

    Carries the backend key and the method name so the error names the adapter
    that is incomplete, not the interface."""
```

**It subclasses `NotImplementedError`.** Prior art is weaker than we recorded:
PEP 249's `NotSupportedError` is a normal, sanctioned channel, but it sits
*alongside* capability introspection rather than having replaced it. So it
supports "raising is normal" and not "matrices are wrong". The argument against
the matrix comes from elsewhere and is stronger:

- **LiteLLM's capability table is 3212 × 37 and is 90.4% empty** — 11 356 filled
  cells of 118 844 — and **every absent cell resolves to `False`**. Issue
  #20885's DeepSeek bug is that shape: two entries for one model, the resolver
  reaching only the incomplete one, and `_supports_factory` returning a confident
  `False` from it. The remedy was *two more lookup paths* plus an
  `except Exception:` that also returns `False`.
- **SQLAlchemy did not remove its matrix** — ~40 `supports_*` flags remain on
  `DefaultDialect`. What it did, for `supports_statement_cache` alone, is make
  the flag **non-inheritable on purpose**, reading
  `self.__class__.__dict__.get(...)` to bypass the MRO, and **warn when a
  dialect is silent** rather than defaulting quietly.

The transferable rule is narrower than "declare locally" and is exactly spec
§3.3.1's stance:

> **A capability a third party must actively verify may not be inherited, and the
> absence of a declaration must be loud, not defaulted.**

An exception is the loudest possible absence, and it is one code path rather
than a branch at every call site.

**This is about an incomplete *adapter*, not about an executor that legitimately
has no level 2.** Spec §3.3.1's example is Cursor, whose hooks are file-based:
"its adapter's job is to make file-based hooks satisfy §4.3 — not to report that
hooks are unavailable." A program executor is the other thing entirely — it
implements `Executor` and declares no `AgentBackend`, so there is no method to
raise from. **Conflating the two is precisely what §3.3.1 warns against**, and
§5.1's split is what keeps them apart.

**Where the exception goes**, spec §3.3.1's table, and the only row this module
owns:

| Caller | Route |
|---|---|
| o11y | Its own; it asked, it deals with the answer |
| The monitor | Its own, like any task failure (validator spec §3.4) |
| **`Runner`, mid-execution** | **The task fails** — the runner lets it propagate to `on_done(FAILED, …)`, and the monitor takes it from there |

The runner does not catch `BackendUnsupported` specially. A backend that cannot
do what the phase needs is a failed task, and giving it its own path would be the
per-capability branch §3.3.1 rules out.

### 5.4 No runtime configuration interface

Spec §4.4. There is no `configure`, `reload`, `set_rules` or `set_hooks` on
`AgentBackend`. Criterion 14 is satisfied structurally, like criterion 5.

Three capabilities are **deliberately unused** and named here so a later reader
does not take their absence for an oversight:

- `set_permission_mode()` and `set_model()` both exist on `ClaudeSDKClient` and
  both mutate a live agent. They are not runtime *negotiation of rules*, so §4.4
  does not forbid them outright — but nothing in the alpha needs them, and adding
  a passthrough would create the interface §4.4 says must not exist.
- **Cursor's `reload()`**, which spec §5.1 flags as "worth adopting later,
  neither in the alpha" with the reason spelled out: §4.4 forbids runtime
  *negotiation*, and re-reading already-prepared config is a different thing. It
  is not on `AgentBackend`, and the day it is, it belongs on level 2.

---

## 6. Backend selection — `selection.py`

### 6.1 The result is a triple, not a backend

Spec §3.3 names three sources. The single most consequential thing the prior art
says about implementing them:

> **Observing the selection must be separable from making it, from day one.**

matplotlib learned it three times. `get_backend()` *resolved and committed* the
choice, so a later `use()` became a silent no-op (#12362); resolution also
**destroyed the caller's open figures** (#23298); the fix — a non-resolving read
plus `get_backend(*, auto_select=True)` — landed between 3.0 (2018) and 3.10,
with a provisional flag and a deprecation runway (PR #29039). All three bugs are
one shape: **selection was a side-effecting operation disguised as a read.**

So selection returns everything it learned, and answering "which backend?" is
never what performs the selection:

```python
class Rejection(BaseModel):
    key: str
    reason: str                     # the adapter's own words


class Selection(BaseModel):
    backend: AgentBackend
    key: str
    source: Literal["cli", "spec", "config"]
    rejected: list[Rejection]       # in the order tried


def select_backend(spec: AgentSpec, *, override: str | None,
                   config_order: Sequence[str]) -> Selection:
    """Choose a backend, and report what was tried and why it was not chosen.
       Raises BackendUnavailable if nothing is usable."""
```

`source` is what makes criterion 3 assertable without reading logs, and it is
what keyring lacks: its issue #632 asks for exactly this — *"show which backend
is being used and why … (envvar, config, CLI argument or by priority)"* — and
the `diagnose` command added in reply prints the config path and the data root,
naming neither.

**`rejected` carries reasons, not names.** keyring #316 is the cautionary case: an
uncaught exception from inside a probe made `import keyring` unusable, the fix
broadened the catch to `ExceptionTrap`, and every rejection reason vanished with
it. `--list-backends` now shows only the viable ones. **You get one or the other
unless the probe returns a structured result** — which is what virtualenv does,
with a `defaultdict(list)` keyed by rejection reason, and it is the only surveyed
project whose failure names what it tried.

### 6.2 The chain, and the one place a body is the design

```python
def select_backend(spec, *, override, config_order):
    rejected = []

    if override is not None:                       # 1. CLI — pins the run
        return Selection(backend=_load(spec, override), key=override,
                         source="cli", rejected=[])

    for decl in spec.backends:                     # 2. the spec's own order
        try:
            return Selection(backend=_probe(decl), key=decl.key,
                             source="spec", rejected=rejected)
        except BackendUnsupported as exc:
            rejected.append(Rejection(key=decl.key, reason=str(exc)))

    for key in config_order:                       # 3. the global fallback
        ...
    raise BackendUnavailable(spec.name, rejected)
```

Three properties, each with a reason:

**A CLI override does not fall through.** If it names something unusable, that is
an error, not a hint. keyring does the same — both its explicit sources go
through `load_keyring()`, which evaluates the priority "to ensure it is viable,
or raise". And matplotlib says why in a comment on the last line of `use()`:
`# if the user has asked for a given backend, do not helpfully fallback`. Spec
§3.3's "forces one backend for the whole run, so a run can be pinned when
reproducing something" is worthless if the pin is advisory.

**An override probes nothing else.** keyring shipped this as a fix — NEWS #404,
*"Improve import times when a backend is specifically configured by lazily
calling `get_all_keyring`"* — after #162, *"Import of keyring takes 25s"*, whose
traceback ran through a backend's viability probe into D-Bus name activation. Our
probes are the same kind: they ask whether a CLI is on `PATH` and whether an
import succeeds.

**Sources 2 and 3 accumulate rejections; source 1 has none.** An override that
fails raises on the spot, so there is nothing to accumulate.

`virtualenv`'s `default = next(iter(choices))` — the first available candidate in
declared order — **is criterion 3's second mechanism, literally**, and it is the
only place in the survey where our shape already exists.

**Where `config_order` comes from.** Spec §3.5: configuration in the alpha is
"one global YAML file with well-classified partitions that everyone reads", with
a real dispatch system on the roadmap. So `config_order` is a parameter here and
the composition root reads it — this module neither locates the file nor parses
it, and when the roadmap's config system arrives only the caller changes.

### 6.3 `backend_entry` accepts both forms

Spec leaves the form unspecified. It takes both, because the surveyed projects
that use both do so for different jobs and neither substitutes for the other:

| Form | Example | What it is for |
|---|---|---|
| **Dotted path** | `agent.backends.claude_sdk:ClaudeSdkBackend` | Naming *this exact one*. What a CLI override and the config order carry |
| **Entry point** | `claude_sdk` in group `agent_sys.backends` | Discovering *what exists*, so a third party can add one without this module knowing it |

The dotted string buys one thing an entry point cannot, and it is worth the
whole cost: **a per-entry failure message.** fsspec re-raises `ImportError` as
`ImportError(bit.get("err"))`, so a user gets *"Install adlfs to access Azure
Datalake Gen2"* rather than a traceback about a missing submodule. So
`BackendDecl` may carry an `err` alongside `backend_entry`.

Its cost is on record too: fsspec's `_import_class` grew a literal
`is_s3 = mod == "s3fs"` special case and a version check, so the registry ends up
holding knowledge about specific third-party packages, and its own docstring
warns that it "can import arbitrary modules".

**What we accept as a `key`** follows matplotlib's entry-point validation, which
is a ready-made checklist: a key may not shadow a declared one with a different
entry, may not be duplicated within a spec, and an *identical* duplicate is
tolerated rather than an error — that last because duplicates arise from
packaging outside the declarer's control (their #28367).

### 6.4 "Available" means the probe did not raise

keyring's shape, adopted:

```python
def _probe(decl: BackendDecl) -> AgentBackend:
    """Construct the backend and let it decide whether it can run here.
       Raises BackendUnsupported with a message naming the cause."""
```

**One expression carries both facts.** keyring's `viable` is literally "reading
`priority` did not raise", and its docstring instructs backends to raise
`RuntimeError` naming the underlying cause. There is no separate
`is_available()` to drift out of sync with the ordering — and drift is the
failure mode a boolean probe plus a rank invites.

**Nothing is cached.** keyring caches with `@once` and matplotlib resolves once
and freezes; both are wrong for us for a reason neither of them has: **`env_mgr`
deploys the environment**, so a probe taken before deployment is taken at the one
moment it is guaranteed to be wrong. Selection happens per dispatch, and the
probes are an import and a `PATH` lookup.

---

## 7. The runner — `runner.py`

### 7.1 What it is

Spec §4.1: *"a middle-man between the task, the scheduler, and the agent … It
also runs the three phases"*. The backend is *"a real thing that runs a defined
agent … **after** the environment, workspace, handoffs, and playground have been
deployed"*.

```python
class Runner:                      # satisfies task_graph's TaskRunner
    def __init__(self, registry: Registry, *, override: str | None = None,
                 config_order: Sequence[str] = ()) -> None: ...

    def start(self, task: Task, agent: Agent, on_done: OnDone) -> None: ...
    def stop(self, task_id: TaskId, on_stopped: OnStopped) -> None: ...

    # ---- rev. 7: not on the TaskRunner protocol; the monitor's entrances ----
    def resume(self, task_id: TaskId) -> None: ...
    def attempt_of(self, task_id: TaskId) -> "TaskAttempt | None": ...
```

It resolves `agent_specs`, `closures`, `env_mgr` and `phase_runner` **from the registry, by
name, at use time**. It imports no backend.

**Rev. 7: it is a factory and a registry, and it runs nothing itself.** `start`
creates one `TaskAttempt` per dispatch (§7.5), keeps it, starts its thread, and
returns. The phases run on the attempt, not here.

**This is not new state; it is state that was already forced and never
declared.** `stop(task_id, on_stopped)` takes an *id*, so any implementation must
already hold a per-task map — `FakeRunner` holds four such containers
(`running`, `started`, `stop_requested`, `_acks`) while both `TaskRunner` and this
`Runner` declare zero attributes. Rev. 7 names the thing that map is a map *of*.

**Two entrances beyond the `TaskRunner` protocol, both the monitor's:**

| | For |
|---|---|
| `resume(task_id)` | Give an existing attempt a thread again — the non-leaf re-entry (`monitor` spec §5.3). **Not `start`**: no new `Execution`, no new agent, the same attempt |
| `attempt_of(task_id)` | Reach the live executor, for `instruct` (`monitor` spec §7). This is the gap `monitor` design §9 recorded as O1 |

**The scheduler holds the narrow protocol and sees neither.** `TaskRunner` stays
`start` + `stop`; widening it would hand the scheduler two verbs it has no use
for, and one of them (`resume`) is precisely the one the scheduler must not have
(`task_graph` design §8.9).

**Rev. 3 named those three and no document registered two of them.** The
stage-three consistency pass found it while assembling one normative composition
root: `env_mgr` design rev. 1 never mentions the component registry, and
`validator` design rev. 1 registers nothing either. Both now do —
`env_mgr` design §11.4 and `validator` design §5.5 — and
[`../../docs/interfaces.md`](../../docs/interfaces.md) §2 is the single listing
that has all of them. The names this runner resolves are exactly:

| Name | Type | Owner |
|---|---|---|
| `agent_specs` | `AgentSpecRegistry` | §4 here |
| `env_mgr` | `EnvManager`, one method | `env_mgr` design §11.4 |
| `phase_runner` | `PhaseRunner`, `run_phase(kind, task, registry)` | `validator` design §5.5 |

**The runner is typed against `Executor`, level 1, and never against
`AgentBackend`.** Nothing on the `TaskRunner` path needs `interrupt`, `instruct`
or `query` — §7.4 — so asking for level 2 would be asking for authority it does
not use. Two things fall out that are worth more than the type annotation:

- **Criterion 6 becomes unwriteable-wrong rather than tested-right.** The runner
  cannot call an AI-only method because it does not hold one.
- **The program executor needs no stubs** (§9), because level 1 is all it is ever
  asked for.

### 7.2 The three phases

`task_graph` spec §3.2.1 and its design §7.1: three phases, only the middle one
is a graph, and the two validation phases "take no pool slot and no policy orders
them".

```
INPUT_VALIDATING  →  RUNNING  →  OUTPUT_VALIDATING  →  on_done
  validator's         the           validator's
  run_phase           backend       run_phase
```

**Rev. 7: the attempt does not advance itself — it reports, and the monitor
advances.** `monitor` spec §5.3 made the planned advance a monitor action, so each
phase ends with `report()` and a wait, and `task.enter_phase(next)` is called by
the monitor:

```
     attempt thread                         monitor loop
     ──────────────                         ────────────
     run INPUT_VALIDATING
     report(planned) ─────────────────────▶ enter_phase(RUNNING)
     wait ◀───────────────────────────────── wake
     run RUNNING (mainloop borrows this thread; the gate at its end)
       gate passes → report(planned) ─────▶ enter_phase(OUTPUT_VALIDATING)
       gate fails  → report(unplanned) ───▶ decide (§7 of the monitor spec)
     wait ◀───────────────────────────────── wake
     run OUTPUT_VALIDATING → on_done
```

**`task.enter_phase(next)` is still the only route to a status, and it is still
never an assignment** — only the caller moved. Criterion 45 over in `task_graph`
forbids the *scheduler* assigning a status; the same argument applies unchanged to
this runner and now to the monitor, which is why `monitor` spec §2 principle 2
required every monitor action to be a transition it calls.

**A non-leaf's thread ends after `unfold` instead of waiting.** Its next event may
be hours away, and an attempt that waited on a condition through its subgraph
would hold one thread per ancestor. The monitor calls `resume` when the subgraph
finishes (§7.5).

**Only the middle phase touches a backend.** That is criterion 6 stated as
control flow: swap the backend and the phase order, the transitions, and the skip
handling are untouched, because none of them mentions it.

**A validation phase produces no output handoff** — it calls
`handoff.update_validation_status`, which is the handoff module's, and the runner
does not read the verdict back. `validator` design §5.2's `PhaseOutcome` carries
`ran` / `reused` / `skipped`, and a skip is recorded there. This runner owes the
*transition*, not the report.

### 7.2.1 What the runner actually hands the executor

Rev. 5 of this document described the three phases in full and **never said what
the backend is given**. The spec-key-to-runtime trace found it: `closure` spec
§2.6 declares a `body`, `task_graph` spec §3.2.5 makes it reachable through
`Task.closure`, and no document said who walks that route. Reachable is not read.

```python
spec = registry.get("closures").get(task.closure)     # a Task may; the scheduler may not
body = closure.body_of(closure.task_of(spec))         # readme, entry, materials
```

| The executor gets | From |
|---|---|
| **`readme.md`** | `body.readme`. For an AI backend it is the instruction; for a program it is documentation the run does not consume |
| **`entry.sh`**, when present | `body.entry`. What a `kind: program` executor runs, and what makes it programmatic |
| `goal` | The task spec's one sentence. Context, not instruction — at most 100 characters, it cannot carry a procedure |
| Its prepared zone | `env_mgr.prepare` (§7.1), which has already placed the agent's own rules, hooks and skills (`env_mgr` design §11.5) |

**The `kind` decides which of the first two is load-bearing, and nothing else
branches on it.** `agent` spec §3.1's `kind: program` selects `ProgramExecutor`
(§9), which runs `entry.sh`; `kind: ai` selects a backend, which is given
`readme.md`. Both are `Executor` at the type the runner holds (§7.1), so this is
one line of dispatch and not a second control flow.

**A task with a subgraph reaches none of this**, because the scheduler runs its
main phase by unfolding (`task_graph` design §8.5) and the runner is never asked.
Its `readme.md` still exists and is still required — it is what a reviewer reads —
which is the whole content of `closure` spec §2.6's rule that the exclusion is
`entry.sh`-versus-subgraph rather than body-versus-subgraph.

**This module reads a closure and that is permitted.** Closure criterion 8's
prohibition is on the *scheduler*; `task_graph` spec §3.2.5 states the narrower
rule that actually holds — the scheduler never reads a spec, and a task's runner
may read the catalogue the task came from.

### 7.3 `start` is asynchronous and `on_done` is called once

**Rev. 7: `start` returns after starting the attempt's thread**, not after handing
the backend `start_async` — the backend is not reached until the main phase, which
is two steps later on that thread. The terminal callback still runs from the
attempt's completion path, carrying the terminal `TaskStatus` and the usage dict —
`OnDone`'s existing signature, unchanged.

**`start` is called while the scheduler holds its `RLock`**
(`task_graph/scheduler.py`, `_dispatch_pass` runs inside `try_dispatch`'s `with
self._lock`), so what it does has to be cheap. Creating and starting a thread is:
**71 μs measured** (`scratch/design/probes-arch/p5_thread_cost.py`, 3.13.13,
start+join over 2000 iterations), against a task that will run an agent for
seconds or minutes.

**Re-entrancy is `task_graph`'s problem and is already solved** (its design §9).
A synchronous backend that completed inside `start` would re-enter dispatch; the
scheduler's trampoline handles it. This runner adds no guard of its own, because
a second guard for one invariant is the two-writers failure
(`engineer_principle.md` §1).

### 7.4 `stop` versus `interrupt`

Two different verbs and the design keeps them apart:

| | Means | Backend call |
|---|---|---|
| `Runner.stop` | The scheduler wants this task ended. `task_graph` moved it to `STOPPING` and waits for `on_stopped` | `backend.stop()` |
| `backend.interrupt` | End the *current submission*, keep the agent | **not reachable from `TaskRunner`** — it is level 2, and the runner holds level 1 |

`interrupt` has no `TaskRunner` route because nothing in the scheduler asks for
one, and §7.1 makes that structural rather than a convention. It exists on
`AgentBackend` because spec §4.3 requires it and criterion 9 tests it; its caller
is a monitor or a future interactive surface — **reaching it through
`Runner.attempt_of` (§7.1), which is the route rev. 7 supplies.**

### 7.5 `TaskAttempt` — one object per dispatch, and the thread's owner

```python
class TaskAttempt:
    """One attempt at one task, carried through its phases.

    Maps 1:1 to the `Execution` the scheduler pushed. Holds the thread, the
    executor, and which phase is next.
    """
    task: Task
    agent: Agent
    executor: Executor | None          # None until the main phase begins
    phase: TaskStatus

    def run(self) -> None: ...         # the thread's target; one phase, then report
    def wake(self) -> None: ...        # the monitor, after enter_phase
    def release(self) -> None: ...     # end the thread; the object survives
```

**Why an object rather than three functions.** It is the answer to three separate
questions that all wanted the same holder:

| Question | Answer |
|---|---|
| Who spawns and joins the phase thread (`findings-arch-ours.md`, observation a) | this object |
| What maps a task id to the live executor (`monitor` design §9, O1) | this object, via `Runner.attempt_of` |
| What survives a non-leaf's subgraph so the re-entry is the same attempt | this object; only its thread ends |

**One attempt, possibly two threads.** A leaf's attempt starts one thread and
keeps it to `on_done`. A non-leaf's ends at `unfold` and takes another at
`resume`. **Neither is a second `Execution`** — the parent was dispatched once,
and `Execution.attempt` is what `TaskAttempt` is named for.

**No thread pool, and the reason is not the 50 μs.** A pool is measurably cheaper
per attempt (21 μs against 71, `p5_thread_cost.py`) and is still the wrong
mechanism: **the scheduler's resource leases are already this system's admission
control.** A pool smaller than the resource-permitted concurrency would leave a
task with its lease taken, its status `RUNNING`, and its work sitting in a queue —
a second admission policy the scheduler cannot see, which is the failure
`monitor` spec §2 principle 1 names in its own words. Fifty microseconds is not
worth a mechanism that lets the system lie about what is running.

---

## 8. The `claude-agent-sdk` backend — `backends/claude_sdk.py`

Verified against `claude-agent-sdk` **0.2.144**. Spec §5.1's 13-row capability
table was checked row by row and **is accurate**; what follows is what it does
not say.

### 8.1 An extra, imported lazily

| Measured | |
|---|---|
| Wheel | **99 MB** |
| Installed | **376 MB**, of which **328 MB** is `_bundled/claude`, a single executable |
| Extra packages | **26**, including `cryptography`, `uvicorn`, `starlette`, `opentelemetry-api`, `jsonschema` |
| **Import time** | **~1.3 s** (1.437 / 1.272 / 1.333 s) |

The import time decides it. A hard dependency makes every `agent_sys` entry
point — including `env-mgr check`, which has nothing to do with agents — pay
1.3 s and 376 MB.

```toml
[project.optional-dependencies]
claude = ["claude-agent-sdk>=0.2.144"]
```

**`backends/claude_sdk.py` imports the SDK inside `_probe`, not at module
scope.** A missing extra is then a `BackendUnsupported` naming the extra, which
is the per-entry error message §6.3 exists for — not an `ImportError` at
start-up in a process that was never going to use it.

*(The recorded figures — "103 MB, pulls `mcp` + `sniffio`" — were both wrong. The
wheel figure was close; the installed cost is 3.6× and the dependency count 13×.)*

### 8.2 Our `PreToolUse` hook must never return `allow`

**The single easiest thing to get wrong here, and the spec does not mention it.**

The SDK's permission evaluation is six steps, not one gate: hooks → deny rules →
ask rules → permission mode → allow rules → `can_use_tool`. Hooks run **first**,
and a hook *deny* applies even under `bypassPermissions`.

But `can_use_tool`'s own docstring records the coupling:

> To observe or gate *every* tool call regardless of permission rules, use a
> `PreToolUse` hook via `hooks` instead — **but note that a `PreToolUse` hook
> returning an *allow* decision also skips this callback.**

And `permissionDecision` is `NotRequired`; omitting it lets the call "flow
through the normal permission evaluation".

> **MUST: the adapter's `PreToolUse` hook returns `deny`, or omits
> `permissionDecision`. It never returns `allow`.**

The natural phrasing — *return allow when the check passes* — silently disables
every downstream check for that call. Two further traps from the same source,
both recorded so nobody re-derives them:

- **`allowed_tools` does not constrain `bypassPermissions`.**
  `allowed_tools=["Read"]` with `permission_mode="bypassPermissions"` still
  approves `Bash`, `Write` and `Edit`.
- **A bare-name deny rule removes the tool from context** before evaluation
  begins; only scoped rules like `Bash(rm *)` are checked at the deny step.

**The hook is still not the boundary**, and this design does not upgrade it. Spec
§5.3 is right: a hook sees `{"tool_name": "Bash", "command": "python3 x.py"}`
with no file path in it. `env_mgr` spec §4 owns confinement. What is new is that
the SDK now ships its own bash sandbox (`SandboxSettings`), whose documentation
says filesystem and network restrictions are configured *by permission rules, not
by it*, and whose `enableWeakerNestedSandbox` is documented as reducing security.
It is a third layer, not a replacement for either.

### 8.3 `on_started` fires when `connect()` returns

Spec §4.3 asks for a callback invoked when the agent *really* starts, because
"deploying an environment and launching a harness takes long enough that
'started' and 'asked to start' are different events".

There is a real signal. `connect()` performs an `initialize` **control-protocol
handshake** with a timeout and stores `_initialization_result`, which
`get_server_info()` returns — documented as "the initialization result that was
already obtained during connect". So:

```
start_async()  →  status = DEPLOYING
   env_mgr prepares, ClaudeSDKClient(...).connect() runs the handshake
connect() returns  →  status = RUNNING, on_started()
```

The handshake payload — available commands, output styles, server capabilities —
is what a monitor wants, and `get_server_info()` is where a monitor gets it.

**A docstring here would have added a constraint that does not exist.**
`interrupt()` says "(only works with streaming mode)" and the reference tabulates
interrupts as unsupported in single mode — but **both** entry points hard-code
`is_streaming_mode=True` (`client.py:191`; `_internal/client.py:137`,
`# Always streaming internally`). The adapter is *not* forced to pass an
`AsyncIterable` prompt to get `interrupt()`.

### 8.4 The interrupt drain, and why it needs no counting

Spec §5.2's second caveat is confirmed by the reference verbatim, and the
mechanism is visible in the source: `interrupt()` sends a **control request** on
a channel separate from the message stream, so nothing touches the buffer.

The drain is not "consume N messages". `ResultMessage.terminal_reason` is
`"aborted_streaming"` or `"aborted_tools"` for an interrupted turn, so **the
interrupted submission's own result is self-identifying**:

```python
def interrupt(self) -> None:
    """Interrupt, then drain to the aborted ResultMessage.

    Drains until a ResultMessage arrives; the turn was ours to abandon if its
    terminal_reason is an aborted_* value. Bounded by the drain timeout, since
    terminal_reason is None on older CLIs and on a synthesized result after a
    fatal session failure."""
```

**The bound is load-bearing.** `terminal_reason` is documented as `None` on CLI
versions predating the field, on results that bypassed the query loop, and on
synthesized error results after a fatal failure. A drain that waits for an
`aborted_*` value and nothing else hangs in exactly the case where the session
has already died.

`_send_control_request` waits for an acknowledgement, so a synchronous `stop()`
over the same channel is implementable.

**One spec-internal tension, resolved toward §4.3.** Spec §5.1 draws the
structural lesson from Cursor's Agent/Run split and concludes that "`interrupt`
belongs to the submission rather than to the agent" — but §4.3's protocol, which
is what criterion 9 tests, puts `interrupt()` on the backend. This design follows
§4.3, and the SDK agrees with it: `ClaudeSDKClient.interrupt()` takes no
submission identifier and aborts the turn in flight. There is no `Run` object to
hang it on, and inventing one to honour a sentence in §5.1 would add a type the
criterion does not mention. **O6** is where the residue of that shows up.

### 8.5 What crosses into `AgentResult`, and what must not

`ResultMessage` carries the metrics spec §5.1 lists — `duration_ms`,
`num_turns`, `total_cost_usd`, `usage`, `model_usage` — and also `result` (the
final response text), `structured_output`, and `permission_denials`. Exactly one
field is annotated *"Safe to log (no message content)"*: `api_error_status`.

> **The adapter projects a named subset into `AgentResult`. It never stores the
> `ResultMessage`.** Criterion 16 is about the *system's* record, and persisting
> the whole message would put prompt-derived text in it.

**Cost accounting, and the shape of spec §9's open question.** Measured from the
SDK's own reference:

- A "run" is **one `query()` call**. Each reports its own `total_cost_usd`.
- **The SDK provides no session-level total** — "accumulate the totals yourself".
- Per-step usage must be **deduplicated by message id**; all messages in a turn
  share one.
- `/clear` starts a **new `session_id`** and its result covers only since the
  reset.
- **A session crash may zero every cost field** — `error_during_execution`
  "may carry it zeroed".

The last one matters to `task_graph`'s consumable pool, which expects a settled
figure at completion: the failure mode is **a settled figure of zero**, not an
absent one. `task_graph` design §6.3.1's `charge()` — "record spend that was
never reserved; `available` may go negative" — is the mechanism that survives it,
because a later true-up is expressible. Nothing here changes; it is recorded so
the zero is not mistaken for a free run.

### 8.6 Sessions, transcripts, and subagents

**`AgentId` and the SDK's session id are different things** (spec §5.4). The
adapter records the correspondence in `AgentHistory.session_ref` and nowhere
else; `task_graph`'s `Agent` gains no field.

**The SDK writes the full transcript outside any workspace we grant** —
`~/.claude/projects/<encoded-cwd>/*.jsonl`, the directory name being the absolute
cwd with every non-alphanumeric character replaced by `-`. Subagent transcripts
sit under `<session>/subagents/` and `list_subagents()` finds them by **scanning
that directory**, not by a protocol call.

Criterion 16 stays true — *the system's* record holds no prompt text — but the
backend's record holds all of it, in `$HOME`, outside the confinement zone. Three
levers exist and this design picks none of them, because the choice is
`env_mgr`'s and the storage is its to manage (spec §6): `CLAUDE_CONFIG_DIR`
relocates it, `CLAUDE_CODE_SKIP_PROMPT_HISTORY` suppresses it, and a
`SessionStore` adapter mirrors it. **O3.**

**Only the main agent is interactable** (spec §4.2, criterion 12). No method on
`AgentBackend` takes a subagent id, and the adapter calls neither
`list_subagents` nor `get_subagent_messages`. Two facts make this non-trivial
rather than automatic:

- **Subagent output reaches our stream by default.** `forward_subagent_text` is
  `False`, but subagent `tool_use` / `tool_result` blocks are emitted regardless,
  as messages whose `parent_tool_use_id` is the spawning Agent tool-use id. The
  adapter must decide what it does with them rather than inheriting a default.
- **Subagents inherit the parent's permission mode**, and `bypassPermissions`,
  `acceptEdits` and `auto` **cannot be overridden per subagent**. A subagent is
  not a weaker principal than its parent.

### 8.7 Which `claude` CLI runs, and why it matters

`_find_cli()` prefers the **bundled** executable and falls back to
`shutil.which("claude")` only if there is none. `cli_path` overrides it.

`env_mgr/installers/claude.py` assumes a `claude` **already on `PATH`** — it runs
`claude plugin list` and reports "claude CLI not available" if that fails — and
installs only *plugins*.

> **So there are two CLIs, and by default the backend runs the one `env_mgr`
> never touched.** An agent would not see the plugins its environment recipe
> installed.

The adapter sets `cli_path` from the prepared environment when `env_mgr` reports
one, and falls back to the bundled CLI otherwise. Neither spec mentions this
interaction; **O2** records that the decision is really `env_mgr`'s.

---

## 9. The program backend — `backends/program.py`

Spec §1.1: *"A program executor implements [level 1] directly and never touches
level 2."* Taken literally, which is what §5.1's split makes possible:

```python
class ProgramExecutor:          # satisfies Executor. Not an AgentBackend.
    status: AgentStatus
```

| Level 1 method | Program executor |
|---|---|
| `start_async` | Spawns the declared command; `on_started` when the process exists |
| `wait` / `start` | Waits for exit; a non-zero status is `FAILED` |
| `stop` | Terminates it |
| **`mainloop`** | Waits on the process and maintains `status`. Short, and it is not optional — §5.1.1's argument holds for a program too: something has to notice the process exited |

**It raises nothing, because it declares nothing it cannot do.** `interrupt`,
`instruct` and `query` are level 2, a program has no level 2, and there is
therefore no method to raise from — which is better than a raising stub for the
reason spec §3.3.1 gives about Cursor: a raise should mean *this adapter is
incomplete*, and a program is not an incomplete AI harness.

Criterion 15 — swapping the backend changes no other component — is then a
statement about the runner, and it is nearly a tautology once the runner needs
level 1 only (§7.1): the SDK backend and this one are substitutable *at the type
the runner uses*. §12 still tests it end to end by running the demo both ways and
comparing handoff state, because the tautology is about types and the criterion
is about handoffs.

**And this *is* what a task without an AI has.** Main spec §4.8 rev. 9 settles
the question three revisions of this section deferred: every task has an agent,
and `kind` is what varies. A `kind: program` spec is not a workaround for the
absence of an agent — it is the agent. §14 D1 records the resolution.

---

## 10. What the system records

Spec §7, and there is nothing to build: every recorded field already belongs to
`task_graph`'s `Agent` (its spec §3.3). This module writes none of them.

| Recorded, by `task_graph` | Not recorded, by construction |
|---|---|
| `AgentId`, spec name, `task_id`, `HandoffRef`s | The prompt — nothing carries it |
| The execution record | The reasoning — `AgentHistory` is fetched, never stored |
| | Tool calls, the backend's internal structure |

The one place this could go wrong is §8.5, and it is why the projection is a MUST
rather than a convention.

### 10.1 The log tool is the one piece of logging that is ours

Spec §6 sends logging to o11y and then keeps exactly one thing back: "an agent
has a log tool, its levels are `debug` / `info` / `warning` / `error`, and three
things can require an entry — system policy, the agent's own rules, or a
skill/rule/hook."

**So the log tool is agent material, deployed like a rule or a skill** (§3.4),
and this module's whole involvement is that `rules` / `hooks` / `skills` can name
something that requires an entry. It declares no logger, defines no level enum,
and opens no sink: **where the entries go and what is done with them is o11y's**,
and a level enum here would be the second definition of a vocabulary that
subsystem owns.

The distinction matters because the two are easy to merge: *requiring* a log
entry is a property of the agent's configured material; *routing* one is not.

---

## 11. Build versus adopt

| Piece | Decision | Why |
|---|---|---|
| The spec model | **Build** — pydantic | Nine keys; the schema pass already validated the shape |
| The registry | **Adopt** `SpecRegistry` (main design §5.1) | The collision policy and the enumerating error are shared |
| The selection chain | **Build**, shaped by keyring | Six lines. No library expresses a three-source chain with a structured result |
| Backend resolution | **Adopt** `importlib.metadata` + a dotted-path loader | Both mechanisms, §6.3 |
| The status enum | **Build** | Spec §4.3 fixes the six names |
| The SDK adapter | **Build** — thin | `ClaudeSDKClient` is the durable handle; the adapter is a projection and a drain |
| The three-phase loop | **Build** | It is `TaskRunner`'s whole job, and no prior art has our phase model |
| The transform helper | **Neither, this revision** — §3.5 | Independent module; its cadence is the union of the harnesses' |

---

## 12. Test plan

`agent_sys/tests/agent/`. Every criterion maps to a named test.

| # | Criterion | Test | File |
|---|---|---|---|
| 1 | Unregistered backend or unresolvable knowledge handoff rejected at load, offending value named | `test_load_rejects_unknown_backend`, `test_load_rejects_unresolvable_knowledge` | `test_registry.py` |
| 2 | **Missing knowledge warns by default, fatal under the flag** — the same spec loads in one mode and is rejected in the other | `test_knowledge_missing_warns_then_fatal` | `test_registry.py` |
| 3 | **Declared order; first available wins; config order when none; CLI beats both** | `test_selection_precedence` (parametrised over the three sources), `test_cli_override_does_not_fall_through` | `test_selection.py` |
| 4 | **An unimplemented backend method raises**, and surfaces to its caller | `test_unsupported_method_raises`, `test_unsupported_fails_the_task` | `test_backend.py` |
| 5 | **The agent spec carries no permissions**; the same spec on two tasks reaches two different sets | `test_agent_spec_has_no_permissions_field`, `test_same_spec_two_tasks_two_reaches` | `test_spec.py` |
| 6 | **A backend is not a `TaskRunner`**; substituting the backend leaves the runner unchanged | `test_backend_is_not_a_runner`, `test_runner_holds_level_one_only`, `test_runner_unchanged_across_backends` | `test_runner.py` |
| 7 | `start_async` returns immediately, callback on real start; `wait` blocks; `start` ≡ both | `test_start_async_returns_before_started`, `test_start_equals_async_plus_wait` | `test_backend.py` |
| 8 | Status transitions pending → deploying → running → finished; `Task.status` is a superset | `test_status_sequence`, `test_task_status_is_superset` | `test_backend.py` |
| 9 | **`interrupt()` stops the agent and the backend drains** — the next query's response is the new one's | `test_interrupt_drains_before_next_query` | `test_claude_sdk.py` |
| 10 | `instruct()` reaches a running agent without ending the run | `test_instruct_does_not_end_run` | `test_claude_sdk.py` |
| 11 | `query()` returns history; the session corresponds to the recorded `AgentId` | `test_query_history_session_matches_agent_id` | `test_claude_sdk.py` |
| 12 | **Only the main agent is interactable** | `test_no_interface_reaches_a_subagent` | `test_backend.py` |
| 13 | Canonical storage; the helper converts losslessly for what both support | `test_material_stored_canonically`; **`test_transform_lossless` — xfail, see O1** | `test_spec.py`, `test_transform.py` |
| 14 | **No runtime interface for changing rules or hooks** | `test_backend_has_no_configuration_method` | `test_backend.py` |
| 15 | **Swapping the backend changes nothing else** — SDK and program produce identical handoff state | `test_swap_backend_same_handoff_state` | `test_runner.py` |
| 16 | No prompt text and no reasoning in the persisted records | `test_records_hold_no_prompt_text` | `test_records.py` |

Three notes on how the harder ones are tested:

**Criterion 9 asserts on `terminal_reason`, not on a message count.** The fake
transport emits the interrupted turn's `ResultMessage` with
`terminal_reason="aborted_streaming"` followed by the new query's, and the test
asserts the drain consumed the first and returned the second. A count-based
assertion would pass against a backend that drained the wrong number of messages
for the wrong reason.

**Criteria 4, 5, 6, 12 and 14 are structural**, and the tests say so: they
assert over the protocol's signature — no `permissions` parameter, no subagent
parameter, no configuration method, and no level-2 method on the type the runner
holds — rather than over behaviour. A structural criterion tested behaviourally
passes for the wrong reason as soon as someone adds the field.

`test_runner_holds_level_one_only` is the one worth naming: it asserts that
`ProgramExecutor` satisfies `Executor` and **does not** satisfy `AgentBackend`,
and that the runner's annotation is the former. That is criterion 6's content,
and it fails the moment someone widens the runner to reach `interrupt`.

**Criterion 13 splits, and only half of it is an `xfail`.** "Stored in Claude
Code's canonical format" is testable now and passes — the material is paths into
the task package, unparsed, and `test_material_stored_canonically` asserts it.
"Converts … losslessly for what both support" is the half with no testable
formulation, and O1 explains why; a green test asserting something the criterion
does not mean would be worse than an honest red one. Marking the whole row
`xfail` would have been the opposite error — hiding a satisfied requirement
behind an unsatisfiable one.

Backends are exercised against a **fake transport**, not a live harness. The two
tests that need the real SDK — 9 and 11 — are marked `claude` and skipped when
the extra is absent, which is also how CI runs without a key.

---

## 13. Implementation order

1. `spec.py` and `backend.py`. No dependencies; they unblock everything.
2. `registry.py` plus check 2. Criterion 1's first half.
3. `backends/program.py`. A real backend with no extra, so `selection` and
   `runner` can be built and tested against something.
4. `selection.py`. Criterion 3.
5. `runner.py` with the three phases. Criteria 6 and 15's first half.
6. The knowledge pass and its report. Criterion 2.
7. `backends/claude_sdk.py`. Criteria 9, 10, 11 and the drain.
8. `test_records.py`, the projection, and criterion 16.

Step 3 before step 4 is deliberate: building `selection` against two fakes would
test the chain against nothing that can actually fail to be available.

---

## 14. Deviations from the spec

| | | | |
|---|---|---|---|
| **D1** | ~~**A task with no agent at all**~~ — was `closure` spec §2.2 and criterion 3, `demo` criterion 7 | **Resolved upstream, and this design's assumption was the right one.** Main spec §4.8 rev. 9: every task has an agent; `kind` — `ai`, `human`, `program` — is what varies | This entry stood for three revisions as "reported, not resolved", and it was correct that the alternative was unaffordable: `Task.agent_spec` is `str`, `scheduler.py` calls `instantiate` unconditionally, and `TaskRunner.start` requires an `Agent` — all in code that is frozen and 423-tests green. What was wrong was only *who* supplies the `kind: program` spec: this design said the loader, `closure` D1 declined, and the answer is that the **package author writes it**, because it is an ordinary agent spec and not a synthesised one. `agent` criterion 15's "a program executor" now means a spec, and the specs say so |
| **D2** | Spec §3.1 permits `backends` to be "a list or dict" | **A list.** A mapping form is normalised at load, preserving order | The order is load-bearing (criterion 3). A dict that carries an ordering hides it from every call site |
| **D3** | Spec §3.4 lists six knowledge types | `knowledge_type` is a `str`, not an enum | §3.4 says the list is "extensible … not a closed vocabulary". An enum makes the seventh type a code change here |
| **D4** | Spec §5.1 lists `can_use_tool` and `PreToolUse` together under "permission gate" | The adapter uses a `PreToolUse` hook that **never returns `allow`** | They are not alternatives: a hook returning `allow` skips `can_use_tool`. §8.2 |
| **D5** | Spec §4.3 shows one `AgentBackend` protocol carrying all seven methods | **Two protocols** — `Executor` (level 1) and `AgentBackend(Executor)` (level 2) | §4.3's listing is a merge of both levels, but §1.1 draws them apart and principle 2 says to keep them apart. Splitting where §1.1 splits — "start, stop, status, result" against "history, interrupt, … sessions" — is what makes §1.1's own sentence, "a program executor implements it directly and never touches level 2", true by type rather than by a raising stub. It also lets the runner hold level 1 only (§7.1) |
| **D6** | Spec §3.3 does not say whether a CLI override must name a backend the spec declares | **It need not.** The override is resolved as a `backend_entry` in its own right; a key matching a declared one uses that declaration's `config` and `err` | §3.3's purpose for the override is "a run can be pinned when reproducing something", and the case that most needs pinning is a backend the spec's author did not foresee. Requiring prior declaration would mean editing the spec to reproduce a bug |
| **D7** | ~~Spec §4.3: *"One thread per agent is the alpha's shape."*~~ | **Resolved upstream the same day.** Spec rev. 6: the loop is the agent's, the thread is the task's, and the agent borrows it (§5.1.1, §7.5) | Raised here as reported-not-edited, and the user amended the spec rather than carrying the divergence. **The count was never in question** — K leaf tasks give K threads either way, because a leaf has one agent — so what the old sentence got wrong was ownership, at a time when *no* document named an owner for either loop's thread. Kept as an entry rather than deleted, because the shape recurs: a spec sentence that answers a question nobody had asked yet will name whatever the writer had in mind |

---

## 15. New open questions

| | |
|---|---|
| **O1** | **Criterion 13 is not testable as written.** "Losslessly for what both support" requires knowing the intersection of two harnesses' feature sets, and no converter computes it — everyone hand-maintains a table. Both reference implementations' tables fail invisibly: **pandoc** classifies a dropped block as `INFO`, so `--fail-if-warnings` exits 0 with the content gone, and attribute-level loss (a link title) is logged at *no* level; **kompose**'s 25-entry unsupported-key table has **no production caller** — its only caller is its own unit test, and the exported sibling is invoked with an empty map, so eight declared-unsupported keys converted clean with exit 0. The only executable formulation found anywhere is `rulesync`'s per-(target, feature) fixtures asserting the *canonical* value. **The criterion needs to name the artefact that defines "what both support" and the test that keeps it honest** — and to separate *unsupported* (the target cannot express it) from *unknown* (the converter did not handle it), which GitHub Actions Importer does and which criterion 13 conflates |
| **O2** | **Which `claude` CLI the backend runs is undecided, and it is `env_mgr`'s call.** The SDK prefers its bundled 328 MB executable; `env_mgr` installs plugins into whatever is on `PATH`. Unless `cli_path` is set from the prepared environment, an agent does not see the plugins its own recipe installed. §8.7 |
| **O3** | **The backend's transcript lands outside the confinement zone.** `~/.claude/projects/<encoded-cwd>/*.jsonl`, containing prompts and reasoning, written by default. Criterion 16 is about the system's record and stays true, but "an agent reaches only its own zone" does not. Three levers exist (`CLAUDE_CONFIG_DIR`, `CLAUDE_CODE_SKIP_PROMPT_HISTORY`, a `SessionStore`); choosing one is `env_mgr`'s, and its spec does not mention the directory |
| **O4** | **§4.5's "Claude Code's format" is ambiguous.** The declarative `.claude/settings.json` surface and the SDK's `ClaudeAgentOptions(hooks={...})` callbacks are different execution models. Every surveyed converter targets the first; **nobody converts programmatic callbacks at all.** §4.5 must name which surface is canonical, and if it is the callbacks, criterion 13 has no prior art of any kind behind it |
| **O5** | **Two objects hold "the agent spec table".** `AgentSpecRegistry` here, and `AgentMgr.register(spec, **config)` in `task_graph`, which copies its dict onto every minted `Agent.config`. This design assumes the loader feeds the second from the first and that nothing else writes either — but the direction is not stated in any spec, and `engineer_principle.md` §1 forbids two writers for one fact |
| **O6** | **How each phase becomes separately attributable** — narrowed twice, and the question is now smaller than rev. 3 stated it. `validator` design §8.2 rev. 2 owns the **requirement**: a phase must carry an `agent_id`, because criterion 10 there is untestable otherwise, and the SDK's `agent_id` is *"absent on the main thread"*. This module owns the **mechanism**, and one candidate is ruled out: not one client with several `session_id`s, because `interrupt()` takes no `session_id` and acts on the whole connection (§8.4). `fork_session`, `resume`, a subagent per phase, and a second client remain, and none was tested. The stage-three consistency pass found this document and `validator`'s giving different answers to what turned out to be two different questions; splitting them is what made the residue this small |
| **O7** | **Mid-run backend failure.** §3.3's "pins the whole run" implies no fallback after the chosen backend dies, and every surveyed project except LiteLLM agrees. LiteLLM's cost is on record — a depth bound, an attempted-targets set against looping graphs, a pin predicate, cooldown feedback and per-failure-class chains, threaded through a loosely-typed `kwargs` at four call sites. Worth knowing before anyone proposes it, and worth stating in the spec either way |
