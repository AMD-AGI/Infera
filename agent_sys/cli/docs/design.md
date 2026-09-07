# Demo — Design

| | |
|---|---|
| Status | Draft, rev. 2 |
| Revision | 4 — 2026-08-29. **The package format is YAML and the layout changed with it** (`docs/ui-stage.md`, `spec.md` rev. 7). §3's tree and §8.4's paths are updated to what is on disk; the rest of this document's prose about jsonnet is **kept as the record of a decision that was reversed**, not as a description of the tree — the reasoning is still why things are shaped as they are. Three substantive changes: the per-kind directories are gone (the directory no longer claims a kind, `module:` does), `bodies/` and `logic/` and `bin/` are gone into `assets/` where every filename is found by convention, and `examples/demo/broken/` is now the sibling package `examples/demo-broken/` because the YAML scan reaches every `*.yaml` under a root. (rev. 3: 2026-08-27. **Every task in the demo package carries a body** (§3, §4.2.1), following `closure` spec §2.6 rev. 9: `produce` and `consume` are `readme.md` + `entry.sh`, `describe` is `readme.md` alone, and the non-leaf `main` has a readme and no entry — which is what makes the entry-versus-subgraph rule visible. (rev. 2: 2026-08-27. Every task has an agent (main spec §4.8 rev. 9, `demo` spec rev. 6): §5 rewritten and D1 retired. No other section changes. (rev. 1: initial, written after the composition research recorded in [`../../docs/design-stage.md`](../../docs/design-stage.md) "Module 8 — demo")) |
| Implements | [`spec.md`](spec.md) rev. 7, §6 — 17 acceptance criteria |
| Language | Python ≥ 3.10, YAML. `ruff`, line length 100 |
| Part of | [`../../docs/design.md`](../../docs/design.md) — the whole-system design |

---

## 1. Scope

The demo is the last module and the only one that composes the other seven. It
adds no mechanism. Everything below is either a *choice* the other designs left
open, or a *collision* between two of them that only shows up when they run
together.

### 1.1 What this document owns

- **The split into two artefacts** — a task package and a runner (§2), and why
  keeping them apart is what makes spec §1.1's rule checkable.
- **The graph the package declares** (§4), including which validator fails and
  why it fails *structurally*.
- **The one step nobody owns**: turning a closure into the root `Task` (§6).
- **The event stream** (§7) — one stream, two renderings, a versioned schema.
- **The three verbs and their exit codes** (§8).
- **The environment the demo asks for** (§9), measured rather than assumed.
- **What CI does with the demo** (§11), which spec §5 leaves as a tension.

### 1.2 What it does not own

| | |
|---|---|
| The whole-system CLI | Spec §1.3, [`../../docs/TODO.md`](../../docs/TODO.md) item 5. The demo's entry point is a stand-in for it, and §8.5 says which parts should migrate |
| The e2e test | Spec §5. A separate artefact at the system-level implementation stage, modelled on this shape, choosing its own tasks |
| The isolation properties | `env_mgr` spec §10 — criteria 2–14 there are CI-enforced in `tests/env_mgr` on every commit. The demo *shows* one of them to a person (§9.4) |
| Any component's behaviour | Criterion 15. If the demo needs something, that is a missing feature, not a demo detail (spec §1.1) |

---

## 2. Two artefacts, not one

Spec §1.1 makes one rule load-bearing:

> **the demo may use nothing a task package outside the repository could not
> use.** It has no privileged import, no private loader path, no schema of its
> own.

Spec §4.1 makes a second: *"One entry point, runnable after `pip install -e
agent_sys`."* Those two pull in opposite directions, because an entry point is a
Python console script and a task package outside this repository will not have
one — it will be run by the whole-system CLI that does not exist yet.

**So the demo is two artefacts, and the split is the enforcement of §1.1:**

| | `examples/demo/` | `cli/` |
|---|---|---|
| What | The task package: YAML specs, and the one program the program node runs | The runner: `run`, `show`, `--dry-run` |
| Contains | No Python that any component imports; no `__init__.py`; not on `sys.path` | An ordinary top-level Python package like `handoff/` or `closure/` |
| Installed | **No.** Found by path (§12) | Yes — it is where `[project.scripts]` points |
| May import `agent_sys` components | Never. It is data | Freely. It is the composition root's caller |
| Imported by a component | Never — criterion 15 | Never — criterion 15 |

Main design §2 projects this differently: *"`demo/` docs only — the package
itself is `examples/demo/`"*. That projection cannot carry a console script
(§12 measured what happens when it tries), and it also weakens §1.1 — the moment
`examples/demo/` holds an installed Python module, the example stops looking like
what an out-of-repository package looks like. **§16 D2.**

The rule that keeps the split honest is one sentence, and §14 tests it:

> `examples/demo/` contains no file that `cli/` imports as a module.

`cli/` locates the package by path and hands it to `load_package` exactly as
the whole-system CLI would hand it any other. That is criterion 16 in one line:
there is no privileged path because there is no path at all — only a directory
argument.

---

## 3. Layout and the import graph

```
agent_sys/
├── demo/                       NEW. The runner
│   ├── __init__.py
│   ├── cli.py                  argparse: run | show, --dry-run. §8
│   ├── package.py              locating examples/demo/. §12
│   ├── build.py                closure -> the root Task. §6
│   ├── events.py               Event, EventKind, SCHEMA_VERSION. §7.1
│   ├── stream.py               the emitter; two renderers subscribe. §7.2
│   ├── render/
│   │   ├── __init__.py
│   │   ├── human.py            the narration a person reads. §7.3
│   │   └── machine.py          JSON Lines. §7.4
│   ├── environment.py          the demo's Context for env_mgr.prepare. §9
│   └── docs/
│       ├── spec.md
│       └── design.md           this document
│
├── examples/demo/              The task package. Not installed, not imported
│   ├── README.md               what a reviewer reads before running it
│   ├── main.yaml               MANDATORY. the outermost graph. §4.1
│   ├── shared.yaml             what more than one step uses — the `collect` agent
│   ├── steps/                  one file per step, holding what that step introduces
│   │   ├── produce.yaml          check_facts, facts, produce
│   │   ├── describe.yaml         describe (agent), check_grounded, summary, describe
│   │   └── consume.yaml          consume
│   └── assets/                 MANDATORY. every body found by filename convention
│       ├── main.task/readme.md              a non-leaf: readme, no entry.sh
│       ├── produce.task/                    readme.md, entry.sh, collect.py. §4.2
│       ├── describe.task/readme.md          agent-bodied: no entry.sh
│       ├── consume.task/                    readme.md, entry.sh, render.py
│       ├── check_facts.validator/           readme.md, entry.sh, check.py
│       ├── check_grounded.validator/        readme.md, entry.sh, check.py
│       └── lib/store.py                     shared by the two validator bodies
│
├── examples/demo-broken/       loaded only by --dry-run --with-broken. §8.4
│   ├── main.yaml               the dangling closure
│   └── assets/dangling.task/readme.md
│
└── tests/cli/
    ├── __init__.py
    ├── test_package_loads.py   the CI half. §11
    ├── test_build.py           §6
    ├── test_events.py          §7
    └── test_isolation_shown.py §9.4
```

### 3.1 The import graph, and the wall

```
                    cli/main.py
                        │
        ┌───────────────┼────────────────┬──────────────┐
        ▼               ▼                ▼              ▼
    cli/package      cli/build       cli/stream     cli/environment
        │               │                │              │
        ▼               ▼                ▼              ▼
   spec_loader      closure +       (nothing)        env_mgr
                    task_graph
        │               │                               │
        └───────────────┴───────────────────────────────┘
                        ▼
              handoff, validator, agent

  ═══════════════════ the wall ═══════════════════
      NOTHING above this line is imported by any
      component.  Nothing below imports `cli`.
      examples/demo/ is imported by nobody at all.
```

The wall is criterion 15, and it is checked twice in §14: once by grepping every
component package for the token `cli`, and once by asserting that
`examples/demo/` holds no `__init__.py`.

`cli/stream.py` importing nothing is deliberate. The event stream is the
demo's own vocabulary (§7), and a stream that imported `task_graph` would
tempt someone to put a `Task` in an event instead of the two fields the renderer
needs.

---

## 4. The graph the package declares

Spec §2.1 gives an illustrative shape and §7 leaves *what the task actually
does* open, noting it is package content rather than a system decision, and
asking for "something small, verifiable, and not contrived, so a validator has
something to be honestly `strong` about". This design picks one. Changing it
later changes no code in `cli/`.

### 4.1 The shape

```
system whole task
└── main                    parent = None, closure = "main", no executor of its own
    │
    ├── produce             kind: program — runs assets/produce.task/collect.py
    │   │                   input validation:  EMPTY (it has no inputs)
    │   │                   output: facts        [handoff kind: facts]
    │   └── output validation
    │       └── check_facts          completeness / strong        PASSES
    │
    ├── describe            kind: ai, backend claude-agent-sdk
    │   │                   input validation:  POPULATED — check_facts on `facts`
    │   │                   output: summary      [handoff kind: summary]
    │   └── output validation
    │       └── check_grounded       trustworthiness / strong     FAILS
    │
    └── consume             kind: program — would render the report
                            input: summary. Never runs: its input never becomes valid
```

`produce` is the `is_start` subtask, `consume` the `is_end` one. The run ends
quiescent with `consume` in `WAITING_HANDOFF`, and the demo says so as the
expected outcome (§7.5).

Criteria 2, 3 and 4 fall out of that shape and are not arranged for:

| Criterion | Satisfied by |
|---|---|
| 2 — a root with `parent = None`, a subtask with a non-`None` parent | `main` and its three subtasks. `Task.unfold()` sets `parent`; `task_graph` design §8.5 |
| 3 — one input validation phase empty, another populated | `produce` has no inputs, so its input phase runs nothing. `describe` consumes `facts`, so its input phase runs `check_facts` — *the same validator*, in the other phase, which is the cheapest possible demonstration that a phase is a position and not a kind of validator |
| 4 — one dispatch per task, no validator in a pool | `describe` has both phases populated and is dispatched once. `validator` design §4: `TaskRunner` runs all three phases for the one task the scheduler dispatched |

### 4.2 What `produce` actually does

`examples/demo/assets/produce.task/collect.py` walks a directory the closure names — by default
the demo package itself — and writes a `facts` handoff: a JSON document of one
row per file, each row `{path, lines, sha256_prefix}`, plus a `totals` object.

Three properties, each chosen for a reason:

- **It is verifiable by a machine.** Every number in it can be recomputed, which
  is what lets `check_grounded` be honestly `strong` rather than a rubric.
- **It needs no GPU, no network and no cluster** (spec §3), and it is
  deterministic given a directory.
- **It is not contrived.** A manifest of what is in a tree is a thing people
  actually produce, and it is the smallest artefact that has enough internal
  structure for a summary to get wrong.

### 4.2.1 The three bodies, and what the demo shows with them

`closure` spec §2.6 rev. 9: every task has a `readme.md`; a programmatic one adds
an `entry.sh`. The demo's three leaves cover both shapes and the contrast is worth
showing rather than only satisfying:

| Task | Body | What a reviewer sees |
|---|---|---|
| `produce` | `readme.md` + `entry.sh` | The exact command that ran. Reproducible by hand, outside the system |
| `describe` | `readme.md` only | The instruction the agent worked from — **the same file kind**, read by a different executor |
| `consume` | `readme.md` + `entry.sh` | Never runs; its readme still says what it would have done |

**That `describe` and `produce` differ by one file is the point.** A reader
comparing the two folders sees the whole of what "an agent task" versus "a program
task" means in this system, without reading any design document — which is what
`demo` spec §4.2 asks the demo to teach.

`main`, the non-leaf, carries a `readme.md` and no `entry.sh`: its work is its
subgraph, and the two are mutually exclusive (`closure` spec §2.6). It is the
demo's evidence that the rule is *`entry.sh`-versus-subgraph* rather than
*body-versus-subgraph* — the non-leaf still has to say what it is for.

### 4.3 Why `check_grounded` fails, and why that is not rigging

Criterion 5 needs a failing verdict, and a demo whose failure is arranged proves
nothing. So the failure is **structural and reproducible**, and it comes from the
handoff contract rather than from the validator's taste:

- The `summary` handoff kind's contract says: *every number appearing in the
  summary must also appear in the `facts` artefact it summarises.* That is a
  groundedness rule, it is `trustworthiness / strong`, and it is checkable by
  extracting the numerals from both and testing set inclusion.
- `describe`'s goal template asks the agent for a short summary **including the
  wall-clock time the collection took**.
- `facts` does not carry a duration. `collect.py` does not measure one, and its
  handoff kind does not declare one.

So the agent is asked, in good faith, for a figure its input cannot ground. It
will produce *something* — that is what a language model does — and
`check_grounded` will find a numeral in the summary that is not in the facts, and
record `False`.

The failure therefore does not depend on the model behaving badly, on a prompt
trick, or on a validator that is secretly `return False`. It depends on the
task having been specified with a gap in it, which is the failure this whole
system exists to catch, and it is exactly the shape of the kickoff report's
appendix A teardown — *the candidate writes the exam, the answer key, and grades
it* — inverted: here the candidate writes the answer and **something else**
grades it.

Two consequences the design must accept:

1. **If the model ever answers "the facts do not record a duration"**, the
   summary contains no ungrounded numeral and the verdict passes. That is an
   XPASS in pytest's sense (§7.5), and the demo must **report it loudly**, not
   silently succeed. §14 makes this a strict expectation.
2. `check_grounded`'s numeral extraction is deliberately crude — digits, not a
   parser. A crude check that is honestly described is a `strong` validator; a
   sophisticated one that is silently approximate is not.

### 4.4 The four spec kinds, and what each carries

Nothing here is novel; it is the ordinary shape, listed so §14 can check that the
package uses no key a package outside this repository could not use.

| Kind | Files | Notable |
|---|---|---|
| handoff | `facts`, `summary` | `summary` names `check_grounded` as its validator (`handoff` spec §5.3), so the kind is never valid without one |
| validator | `check_facts`, `check_grounded` | Both carry `brief`, `dimension`, `strength` with no defaults (`validator` design §3.2) |
| agent | `collect` (`kind: program`), `describe` (`kind: ai`) | §5 |
| closure | `main`, `produce`, `describe`, `consume` | `main` declares the subgraph; the three leaves declare `is_start` / `is_end` |

---

## 5. The program node — an agent that is not an AI

Criterion 7 rev. 6: *"One node runs on `claude-agent-sdk` and one runs as a
program — an agent of `kind: program`, with no AI in it — and the handoff state
each produces is indistinguishable in kind."*

Both halves are satisfiable, and the second is the interesting one. The first
half read differently until 2026-08-27, and the change is worth recording
because a chain of four modules ended with it.

**What the criterion used to say** was *"a program with no agent at all"*, and
that is not expressible. Measured in shipped and frozen code (findings M12):

| | |
|---|---|
| `Task.agent_spec: str` | Required. `Task(agent_spec=None)` raises `string_type` |
| `Execution.agent_id: AgentId` | Required. Every *attempt* binds an agent, not just every task |
| `scheduler._dispatch_pass` | Calls `agent_mgr.instantiate(task.agent_spec, tid)` unconditionally, before `push_execution` |

The chain: `agent` design D1 assumed the loader would supply a `kind: program`
agent spec; `closure` design D1 **is** that loader and declined to synthesise
one, on the correct ground that it would make `closure` spec §2.2's "names no
agent spec" true of the file and false of the loaded object; `env_mgr` carried
the question forward; and this module is where it had to be answered.

**It was answered by removing the gap rather than filling it.** Main spec §4.8
rev. 9: *every task has an agent, and an agent need not be an AI* — `ai`,
`human`, or `program`. `closure` spec §2.2 rev. 8 makes `agent` a required key.
The three designs follow: `agent` §9's `ProgramExecutor` **is** what a task
without an AI has, `closure` §3.3 demands an agent spec and still synthesises
nothing, and the shipped `Task.agent_spec: str` turns out to have been right all
along.

### 5.1 What the demo declares

The `collect` agent, in `examples/demo/shared.yaml`, is an ordinary agent spec (shown here in the jsonnet this design was written against; it is YAML now, unchanged in content):

```jsonnet
{ name: 'collect', kind: 'program', version: '1',
  description: 'Walks a directory and writes a facts manifest',
  command: ['python3', 'bin/collect.py'] }
```

Nothing is synthesised, nothing is inferred, and `closure`'s `agent_of()`
returns it because it is there — written by the package author, which is what
`closure` D1 said the loader must not do on its behalf.

### 5.2 What the demo says about it, in its own output

The distinction is the point of the node, so the narration draws it rather than
leaving a reader to infer it from an absence:

> `produce` runs a program. Its agent is `collect`, `kind: program` — an agent,
> and not an AI: no model call, no level-2 backend, and nothing non-deterministic
> in it.

### 5.3 The half the criterion is really about

`test_program_and_sdk_handoff_state_identical` (§14) compares the `Handoff`
records the two nodes produce and asserts they differ only in content — same
states, same version numbering, same verdict shape, same `Execution` binding
record. `agent` design §9 makes that nearly a tautology at the type level,
because the runner needs level 1 only; the demo tests it at the handoff level,
which is where the criterion is written and where a regression would actually
show.

---

## 6. Turning a closure into the root `Task`

`closure` design D5 and §9.2: **nobody owns this.** The only attribution in the
whole design set is main design §7's *"the scheduler is what assembles it"*,
which `closure` criterion 8 forbids. `closure` declined to claim it; the two
named callers are the demo's `show` and `--dry-run`.

So `cli/build.py` builds it. Three sentences of interface, and one body,
because the ordering is the design.

```python
def root_task(closure_name: str, registry: Registry) -> Task:
    """The root `Task` for a closure. Builds only the root: `Task.unfold()`
    instantiates the subgraph from `self.closure` (task_graph design §8.5)."""

def handoff_ids(closure_name: str, registry: Registry) -> dict[str, HandoffId]:
    """Kind name -> a fresh HandoffId, one per kind the closure names. The map
    the graph is wired with, and the map `env_mgr` resolves grants against
    (closure design D2)."""

def wire(tasks: Sequence[Task]) -> None:
    """Fill `depends_on` from the handoff wiring: a task depends on every task
    that declares one of its inputs as an output. §6.2."""
```

### 6.1 Why the root is the only `Task` built

`Task.unfold()` (`task_graph` design §8.5) instantiates the declared expansion of
`self.closure` with `parent = self.id` and the `is_start` / `is_end` marks the
closure declares. So the builder sets `closure = "main"` on one `Task` and stops.

That is worth stating because it bounds the unowned job: it is not "assemble the
graph", it is "make the first node". Everything else is `task_graph`'s, already
designed, and already the subject of criteria 36–54 there.

### 6.2 `depends_on` has no spec that can carry it, and omitting it warns

Measured (M11). `scheduler._warn_depends_on` logs on every dispatch whose
`depends_on` omits the producer of one of its inputs:

```
<consumer>: depends_on omits <producer>, which produces <handoff>
```

It is a warning by design there — *"rejecting would make declaration order
matter"* — but `depends_on` is `list[TaskId]`, **runtime** ids, so no jsonnet
spec can hold one. If the builder does not derive it, the reference example of
this system prints a warning on every run, which is precisely the kind of
accepted noise a demo exists to prevent.

`wire()` derives it from the only source that has the information: the
input/output handoff sets of the tasks that exist at that moment. It runs after
`unfold`, in `cli/main.py`, and §14 asserts the log is silent.

### 6.3 This is the whole-system CLI's job, borrowed

`cli/build.py` is ~60 lines and it is in the wrong package. When
[`../../docs/TODO.md`](../../docs/TODO.md) item 5 is built, these three functions
move into it unchanged and the demo imports them. Recorded here so the move is a
relocation rather than a rediscovery — and so the next reader knows that a
component boundary was crossed knowingly. **§16 D3.**

---

## 7. The event stream

Spec §4.2: the output *is* the deliverable, it must show six things legibly, and
*"machine-readable output alongside the human-readable form, because the
acceptance criteria are assertions over it"*. Criterion 14 sharpens that: the
machine-readable output must be sufficient to assert criteria 2–10 **without
parsing prose**.

That makes it an interface. Terraform's answer to owning one (findings S4, S5) is
adopted whole: a closed enumeration of message types, a version constant with a
comment obliging a bump, and **one stream rendered twice** rather than two
writers that can drift.

### 7.1 `events.py`

```python
SCHEMA_VERSION = "1.0"
"""The schema of the machine-readable stream. Criterion 14 makes this an
interface: bump it on any change to EventKind, to Event's fields, or to what
render/machine.py emits."""


class EventKind(str, Enum):
    # the run itself
    RUN_START = "run_start"
    RUN_COMPLETE = "run_complete"
    # loading — everything --dry-run can reach
    PACKAGE_LOADED = "package_loaded"
    CLOSURE_RESOLVED = "closure_resolved"
    SPEC_REJECTED = "spec_rejected"
    GRAPH_BUILT = "graph_built"
    # the environment
    CONFINEMENT_APPLIED = "confinement_applied"
    ZONE_PREPARED = "zone_prepared"
    ACCESS_DENIED = "access_denied"
    # the task lifecycle
    TASK_DISPATCHED = "task_dispatched"
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    VERDICT_RECORDED = "verdict_recorded"
    HANDOFF_TRANSITION = "handoff_transition"
    TASK_COMPLETE = "task_complete"
    # the end
    TASK_FINAL_STATE = "task_final_state"
    EXPECTED_FAILURE = "expected_failure"
    UNEXPECTED_SUCCESS = "unexpected_success"
```

```python
class Event(NamedTuple):
    kind: EventKind
    message: str            # the human sentence. Complete on its own
    fields: Mapping[str, JSONValue]   # the typed payload. Never a model object
    at: datetime
```

**`message` and `fields` are produced by the same call**, which is the whole
point of S5: a demo whose narration and whose JSON disagree fails at the one job
it has. `fields` holds JSON scalars, never a `Task` or a `Handoff` — the renderer
must not be able to reach into a model and start rendering it.

### 7.2 `stream.py`

```python
class Stream:
    def emit(self, kind, message, **fields) -> None:
        """Stamp and fan out to every attached renderer, in attach order."""
    def attach(self, renderer: Renderer) -> None: ...
```

```python
class Renderer(Protocol):
    def on_event(self, event: Event) -> None: ...
```

Two renderers ship. A third — a progress bar, a web view — is an `attach` call
and changes nothing else, which is the same substitutability argument
`agent` design §7.1 makes about the runner.

### 7.3 `render/human.py`

One line per event, the taxonomy spelled out rather than abbreviated, because
spec §4.2 asks the demo to *teach* the taxonomy by using it:

```
  dispatch   describe                        agent=describe (ai, claude-agent-sdk)
    phase    input validation                1 validator
    verdict  check_facts                     completeness / strong    PASS
    phase    execution
    denied   /tmp/agentsys-demo/outside      landlock  EACCES         expected
    phase    output validation               1 validator
    verdict  check_grounded                  trustworthiness / strong FAIL
  handoff    summary  v1                     INVALID
```

Criterion 10 — *every verdict carries its dimension and strength* — is a
property of this renderer and of `render/machine.py`, and §14 asserts it against
both.

### 7.4 `render/machine.py`

JSON Lines on a separate stream, one object per event:

```json
{"schema":"1.0","kind":"verdict_recorded","at":"2026-08-27T09:14:02.881Z",
 "message":"check_grounded: FAIL",
 "validator":"check_grounded","dimension":"trustworthiness","strength":"strong",
 "handoff":"7d1c…","version":1,"result":false,"phase":"output_validation",
 "task":"a91b…","expected":true}
```

JSON Lines rather than one document, for the reason Terraform gives implicitly by
using a streaming logger: a run that is interrupted (criterion 12) must still
have produced valid, complete output for everything that happened before the
interrupt. A single top-level JSON object would be truncated and unparseable —
which would make the criterion-12 demonstration unassertable.

**Where it goes.** `--json <path>` writes it to a file; with no flag it is
suppressed. Not stdout: criterion 6's message arrives on stdout from the backend
(M6), and interleaving a byte stream we do not control with a machine-readable
one is how a parser learns to be lenient.

### 7.5 Expected failures are a category, not a tone

Spec §2: items 4 and 7 *"are the ones a demo is tempted to skip, because both are
failures and a demo that fails looks broken"*. pytest solved this naming problem
(findings S6, verified first-hand), and the vocabulary is adopted directly:

| | |
|---|---|
| `expected: true` on an event | This failure is the demonstration. `check_grounded`'s FAIL, `consume` stuck in `WAITING_HANDOFF`, the denied write |
| `EXPECTED_FAILURE` | Emitted at the end for each one that happened. The demo is **green** |
| `UNEXPECTED_SUCCESS` | Emitted when an expected failure did **not** happen. The demo **fails**, loudly, and §8.3 gives it a distinct exit code |

The third row is the one that matters. A demo that prints "all good" because the
sandbox stopped blocking, or because the validator stopped failing, is the single
worst outcome available to this artefact — it would assert, in the most visible
place in the repository, that a safety property holds when it does not. pytest
calls this `xfail(strict=True)`; here it is not optional.

---

## 8. The three verbs

### 8.1 `show`

Loads the package, resolves every closure, builds the root `Task`, unfolds it,
and prints the graph. Dispatches nothing, prepares no environment, needs no
credentials and no sandbox.

This is spec §4.1's *"prints the graph without running: closures, tasks, phases,
validators"*, and it is one of `closure` D5's two named callers — which is why
§6's builder must exist before this verb does.

### 8.2 `run [--dry-run] [--json PATH] [--package DIR]`

Without `--dry-run`: everything. With it: **resolve and validate everything,
dispatch nothing**, which is criterion 11.

`--dry-run` is `show` plus the load-time checks that `show` does not need to
reach: `check_closures`, `check_graph`, and every schema validation, with
failures collected rather than raised (main design §3.6). The difference between
the two verbs is not the amount of work; it is the audience. `show` answers
"what is this graph"; `--dry-run` answers "would this run".

### 8.3 Exit codes

| | |
|---|---|
| `0` | The run completed, and every expected failure was observed |
| `1` | A load error — a rejected spec, an unresolvable closure, a broken graph. The offending file path is in the message (criterion 11) |
| `2` | Missing credentials (criterion 6), or no sandbox available (criterion 9). The precondition, not the run, is what failed |
| `3` | **An expected failure did not happen.** §7.5 |
| `4` | An unexpected failure — anything else |

`2` is separated from `1` because spec §3.2 is explicit that refusing to start
without a sandbox *"is itself correct behaviour, not a demo failure"*, and a
reviewer on a machine with neither `bwrap` nor Landlock needs to see that
distinction without reading prose.

### 8.4 The deliberately broken closure

Criterion 11's second half: *"a deliberately broken closure makes it fail with
the offending file path"*. That collides with criterion 13 — running twice
without hand-editing — if the broken file is in the package.

`examples/demo-broken/` is a **sibling package** that the ordinary discovery pass
does not reach, and `--dry-run --with-broken` adds it. One flag, two runs, no
editing:

```
$ demo run --dry-run                 → exit 0
$ demo run --dry-run --with-broken   → exit 1
    examples/demo-broken/main.yaml::$.task: closure 'dangling' names handoff
    kind 'nonexistent', which does not resolve. known kinds: facts, summary
```

**It was a subdirectory of the package and could not stay one.** `YamlPackage`
scans every `*.yaml` under a root except `assets/`, so a broken document anywhere
inside `examples/demo/` would load on every ordinary run and criterion 13 would
be dead. Two mandatory names are what make a directory a package, so a sibling
with both is the smallest thing that is unambiguously outside the demo.

The alternative — Airflow's answer, a skip-list of known-bad files (findings S2)
— is what a suite grows *after* it has more than one example. With one example,
a separate directory and an explicit flag is both smaller and honest about what
it is for. §16 records the moment that stops being true.

### 8.5 What migrates to the whole-system CLI

`run` and `--dry-run` are the whole-system CLI's verbs wearing a demo's name.
When TODO item 5 arrives, `cli/main.py` should shrink to: locate the package,
call the real CLI, keep `show`. Recorded so the demo is not later mistaken for a
second CLI to maintain.

---

## 9. The environment the demo asks for

`env_mgr` owns every mechanism here. What the demo owns is the `Context` it
hands to `prepare()`, and that context turned out to be the module's most
surprising research result: **`env_mgr` design §5.2's default granted set does
not start the backend.** Three grants were discovered by being refused, and each
broke differently.

### 9.1 The measured granted set

| | |
|---|---|
| read-exec | `/usr` `/bin` `/sbin` `/lib` `/lib64` `/etc` `/proc`; the backend's own install directory; the CA bundle if one is configured; **`/run/systemd/resolve/stub-resolv.conf`** |
| read-write | the zone (containing `config/` and `tmp/`); **`/dev/urandom`**; `/dev/null` |

Eleven entries. `$HOME` appears only as the parent of the backend's install
directory and the CA file, and §9.3 is why that matters.

**`/dev/urandom`** (M1). The `claude` executable on the development machine is a
standalone Bun binary. Without `/dev/urandom` it aborts inside 3 ms with
`panic(main thread): abort() called`, the words *"This indicates a bug in Bun,
not your code"*, and a crash-report URL for the wrong project. Granting the whole
of `$HOME` read-write does not help; that one character device does.

**`/run/systemd/resolve/stub-resolv.conf`** (M2). `/etc/resolv.conf` is a symlink
to `../run/systemd/resolve/stub-resolv.conf`, and Landlock rules apply to the
resolved path — so granting `/etc` grants the symlink and not its target. Under
the same domain:

| | `getent hosts api.anthropic.com` | `claude -p` |
|---|---|---|
| `/etc` only | rc=2 in **0.0 s** | **hung ~184 s**, then `Request timed out` |
| + the stub file | rc=0, answer | **rc=0 in 5.5 s** |

One missing file, two tools, and symptoms with nothing in common. This is
`env_mgr`'s M7 — *an ungranted file looks broken, not absent* — in its third and
fourth shapes, and the demo is where an operator meets it first.

### 9.2 The zone carries the backend's temp directory

The backend refuses to start if its temp directory is unreadable, and says so
well (M3): it names the path, both plausible causes, the remedy, and
`CLAUDE_CODE_TMPDIR`. The demo takes the suggestion, and the result is better
than a workaround: `<zone>/tmp/`, mode `0700`, created by the demo's context and
destroyed with the zone. Per attempt, like everything else in `env_mgr` D6.

### 9.3 The zone carries the backend's *config* directory, and this is not optional

Measured (M5): with `~/.claude` granted read-write, the confined demo agent read
the operator's personal `CLAUDE.md` and obeyed it — it answered in Chinese. A
demo whose transcript changes with the reviewer's dotfiles is not a demo, and the
run is not reproducible in the sense criterion 13 needs.

So `cli/environment.py` sets `CLAUDE_CONFIG_DIR=<zone>/config`, places the
credentials there from config (spec §3.1), and grants no part of `$HOME` beyond
the two files in §9.1. Measured cost: the confined agent then answers in **2.8 s**
with no `$HOME` grant at all. **§16 D5.**

### 9.4 The block, as a scene

Criterion 8 wants a scripted out-of-zone write blocked, the agent told why, and
the block in the output as an expected event naming the mechanism. Measured end
to end (M4), in one Landlock domain with cwd set to the zone:

| | |
|---|---|
| a real model call | rc=0, **8.4 s** |
| a write **inside** the zone | rc=0, the file is there |
| `echo leaked > <outside>/leak.txt` **via bash** | rc=0 for the agent, **nothing written** |

and the agent's own report of the third:

> the command ran; verbatim output `permission denied: /tmp/…/leak.txt`; exit
> code 1; nothing was written. That path is outside the working directory, and
> `ls` returns Permission denied as well.

Two of criterion 8's three clauses are therefore already true without the demo
doing anything: the OS blocks it and the agent is told why, by the OS, and
understands it. What the demo adds is the third — an `ACCESS_DENIED` event with
`expected: true`, naming `landlock` and the errno, so the block reads as a
demonstration rather than as a fault.

The scripted form is required, not decorative: `agent` spec §5.3 and the module-3
measurement both record that the SDK's `Bash{'command': 'python3 reader.py'}`
hook returns ALLOW. The hook is the attributable layer; the OS is the boundary.

### 9.5 Credentials, and the message the demo must not swallow

Criterion 6 is already satisfied by the backend (M6). With an empty config
directory and `ANTHROPIC_*` scrubbed:

```
rc=1   0.6 s   stdout: 'Not logged in · Please run /login'
```

identical confined and unconfined. Two consequences. The demo invents no
credentials check — it runs the preflight, catches the non-zero exit, and
surfaces the backend's own words, which name what is missing better than a
rewrite would. And it must **not print only stderr**: the message that satisfies
this criterion arrives on stdout.

The check runs **before** any zone is built, so a reviewer with no credentials
waits 0.6 s and not 4 seconds of `git clone`.

### 9.6 `extensions.preciousObjects` is a precondition on the reviewer's checkout

`env_mgr` design §7.2 makes the workspace a `git clone --shared` and requires
`extensions.preciousObjects` on the main repository, so a `gc` cannot delete a
pack an agent's clone is reading through `objects/info/alternates`. `prepare()`
enforces it.

For the demo that means: **the first thing the demo does to a reviewer's machine
is set a git config on the repository they just cloned.** It is one config key,
it is reversible, and it is genuinely required — but it happens before anything
has been demonstrated, and the demo must say what it is doing and why in one
line rather than doing it silently. §16 O1.

---

## 10. Idempotence, and the resume

### 10.1 Criterion 13 is a statement about zone naming

*"Running twice in succession succeeds without hand-editing anything."* Two runs
produce two sets of `TaskId`s and `HandoffId`s, and `env_mgr` design §8.1 names
a zone `task.<uuid>.<version>.<hash>` — so two runs never collide. What can
collide is the store root and the accumulated zones.

```
<demo-root>/                       default: $XDG_STATE_HOME/agent-sys-demo
├── runs/<run-id>/                 one directory per run; run-id is a timestamp
│   ├── store/                     JsonFileStoreMgr root
│   └── zones/                     env_mgr's <root>
└── latest -> runs/<run-id>        what `--resume` follows
```

`--clean` removes `runs/` and exits. The demo does not clean automatically:
criterion 12 needs the previous run's state to still be there.

### 10.2 Criterion 12, and where the interrupt lands

Measured (M8, M10) in two real processes against the shipped
`JsonFileStoreMgr`, killing the writer with `os._exit(9)` at each of the twelve
record writes an equivalent graph performs:

- **All twelve resume with no exception and no unreadable record.** The store's
  per-record `tmp.replace(path)` holds. This is a stronger claim than
  `tests/task_graph/test_recovery.py` makes, which restarts with fresh managers
  over the *same live store object* in one process.
- The interrupted attempt is re-run: `attempts=[(0, 'SUSPENDED'), (1, None)]`.

So `demo run --resume` is: point `JsonFileStoreMgr` at `latest/store`, rebuild
the registry, `resume_all(registry)`, and continue. The demo prints the attempt
numbers, because that is what makes "continued from persisted state" observable
rather than asserted — a task at attempt 1 with attempt 0 recorded `SUSPENDED` is
the evidence.

**Where the reviewer should interrupt matters**, and the demo says so: after
`produce` and before `describe` completes, the resume costs a second model call
(M10). The README suggests interrupting during `produce`, which is a program and
free to re-run, and the demonstration is identical.

---

## 11. What CI does with the demo

Spec §5 says plainly: *"The demo is not a test, and CI does not run it."*
Spec §1 says the demo is *"the first thing to break when one of them drifts"*.
Both cannot be true of an artefact nothing checks.

Airflow has the answer (findings S1), and it is not a compromise. Every shipped
example DAG is covered by three tests, and **not one of them executes a DAG**:

```python
def test_should_be_importable(example, ...):
    dagbag = DagBag(dag_folder=example)
    assert len(dagbag.import_errors) == 0
    assert len(dagbag.dag_ids) >= 1

def test_should_not_do_database_queries(example, ...): ...
def test_should_not_run_hook_connections(example, ...): ...
```

**CI loads the example on every commit; a human runs it.** Our load-time half
already exists and already has a name: `--dry-run`, which criterion 11 defines as
resolving everything and dispatching nothing. It needs no credentials, no
sandbox, no model, and no network, so putting it in CI contradicts nothing in
§5 — the thing §5 excludes from CI is the *run*.

`tests/cli/test_package_loads.py` therefore carries three tests shaped like
Airflow's, the last two being the "what loading must not do" kind:

| Test | Asserts |
|---|---|
| `test_package_loads_clean` | Every spec renders, validates and admits; `check_closures` and `check_graph` are silent; the graph has ≥ 4 tasks |
| `test_dry_run_dispatches_nothing` | After `--dry-run`, no task left `WAITING_HANDOFF`, no `Agent` was instantiated, no zone was created |
| `test_loading_needs_no_credentials_and_no_sandbox` | The load path touches neither. Run with `ANTHROPIC_*` scrubbed and a `Confinement` factory that raises if called |

Airflow also shows the price (S2): two hand-maintained exemption tuples and a
per-file timeout table, accumulated one postponement at a time. With one example
those lists are empty. **The moment this repository holds a second task package,
the exemption list is the thing to watch**, and spec §1.1 says there will not be
one — so the guard on that cost is the same rule that keeps the demo unique.

The counter-example is worth the sentence: dbt's `jaffle_shop`, the most-copied
example in its ecosystem, 544 stars, **archived, and `.github/workflows` is a
404** (S3).

---

## 12. Packaging

Measured against a throwaway project shaped like `agent_sys`, in both install
modes (M13):

| `packages.find` `include` | `pip install -e` | `pip install` (wheel) |
|---|---|---|
| `examples*` absent | script installs, **dies with `ModuleNotFoundError`** | same |
| `examples*` present | works; the `.jsonnet` beside the module is reachable | imports, **but the `.jsonnet` is gone** |

Two facts follow, and together they decide §2's split.

**A console script pointing into an unpackaged directory is not an install-time
error.** `pip` writes the script, the install reports success, and the failure
arrives when a reviewer runs it. That is the worst available failure mode for the
first command someone types.

**setuptools ships `.py` only.** A task package's specs are data, so from a wheel
they simply are not there.

So `cli/` is the installed package and `examples/demo/` is data found by path:

```toml
[project.scripts]
agent-sys = "cli.main:main"

[tool.setuptools.packages.find]
include = ["env_mgr*", "task_graph*", "agent_sys_helper*",
           "spec_loader*", "handoff*", "validator*", "agent*", "closure*", "cli*"]
```

`cli/package.py` resolves the package directory in three steps, and the third is
what keeps it honest:

1. `--package DIR`, if given. Always wins.
2. `<the repository root>/agent_sys/examples/demo`, derived from
   `demo.__file__`. This is the editable-install case, and criterion 1 says
   `pip install -e`.
3. Otherwise **fail with the two paths it tried**, and the sentence *"the demo
   task package is not installed with the wheel; run from a checkout"*.

Step 3 is a deviation worth its own line (§16 D7): a wheel install of
`agent_sys` gives a working `agent-sys` command that refuses to run. That is
the correct behaviour — the alternative is packaging the specs as package data
and shipping an example that behaves differently depending on how it was
installed — but it is a refusal, and it must name why.

**`agent_sys/pyproject.toml` is this design's only edit outside its own two
directories**, and it is additive: the `include` list, the script, and the three
runtime dependencies main design O1 records as missing (`_jsonnet`,
`jsonschema`, `jsonpath-ng`). The repository's own `pyproject.toml` is not
touched.

---

## 13. Build versus adopt

| Piece | Considered | Chosen | Why |
|---|---|---|---|
| CLI parsing | `click`, `typer`, `argparse` | **`argparse`** | Two verbs and five flags. `click` and `typer` are undeclared dependencies bought for nothing, and the demo is the one artefact whose install must be boring |
| event stream | `logging`, `structlog`, own | **own, ~40 lines** | §7. `logging` fans out to handlers, which is the right shape, but its record carries a formatted string and arbitrary `extra` — criterion 14 needs a *closed* set of kinds and a versioned payload, and enforcing that on top of `logging` is more code than `Event` plus `Stream` |
| machine format | JSON document, JSON Lines, `--junitxml` | **JSON Lines** | §7.4. An interrupted run must still have emitted valid output for what happened before the interrupt, which criterion 12 requires and a single document cannot give |
| stream shape | invented | **Terraform's**, adopted | S4, S5. A closed `MessageType` enum, start/complete/errored per operation, `JSON_UI_VERSION` with a comment obliging a bump. Three conventions that are free now and unaffordable later |
| expected-failure vocabulary | invented | **pytest's**, adopted | S6. `xfail` / `xpass` / `strict` already name exactly the three cases §7.5 needs, and every Python reviewer already knows them |
| CI guard | run the demo, run nothing, load the demo | **load it**, Airflow's shape | S1. It is the only option that satisfies both spec §1 and spec §5 |
| the task's content | a toy, a benchmark, a file manifest | **a file manifest** | §4.2. Verifiable by recomputation, deterministic, needs nothing, and has enough structure for a summary to get wrong |
| store | `MemoryStoreMgr`, `JsonFileStoreMgr` | **`JsonFileStoreMgr`** | Criterion 12 needs a durable store across processes, and M8 measured it surviving all twelve interruption points. Its records also stay readable with `cat`, which is a demo virtue |

Nothing new is built that any component already provides. The three files with
real content — `build.py`, `events.py`, `environment.py` — exist because §6, §7
and §9 are decisions nobody else made, not because something was missing.

---

## 14. Test plan

`pytest`. `agent_sys/tests/cli/`, with an `__init__.py`, for main design §9's
reason. **No test in this directory makes a model call, requires credentials, or
requires a sandbox** — §11 is why that is a property and not a limitation.

### 14.1 The criteria, mapped

| # | Criterion | Test | File |
|---|---|---|---|
| 1 | Install, then run, under a minute excluding model latency, no GPU/cluster/setup | `test_overhead_budget` — asserts the non-model overhead of a `--dry-run` plus four zone builds is under 30 s, half the budget. Measured at 2.97 s (M14) | `test_package_loads.py` |
| 2 | A root with `parent = None`, a subtask with a non-`None` parent | `test_root_and_subtasks` | `test_build.py` |
| 3 | One input phase empty and runs nothing, another populated and runs | `test_produce_input_phase_empty`, `test_describe_input_phase_runs` | `test_build.py` |
| 4 | One dispatch per task; no validator in any pool | `test_one_dispatch_per_task`, `test_no_validator_in_any_pool` | `test_events.py` |
| 5 | A failing verdict, the consumer left `WAITING_HANDOFF`, written by the phase | `test_grounded_fails_and_consume_waits`, `test_verdict_author_is_the_phase` | `test_events.py` |
| 6 | No credentials → a clear message naming what is missing, no fake fallback | `test_missing_credentials_message`, `test_no_fake_backend_exists` | `test_package_loads.py` |
| 7 | One SDK node, one program node, handoff state indistinguishable in kind | `test_program_and_sdk_handoff_state_identical` | `test_build.py` |
| 8 | A scripted out-of-zone write blocked, the agent told, an expected event naming the mechanism | `test_access_denied_event_shape` — the *event*; the property itself is `env_mgr`'s and CI-enforced there (§1.2) | `test_isolation_shown.py` |
| 9 | No sandbox → refuses to start and says so | `test_no_confinement_refuses_with_exit_2` — with an injected probe reporting nothing available (`env_mgr` O-note on M16) | `test_isolation_shown.py` |
| 10 | Every verdict carries dimension and strength | `test_human_verdict_line_carries_taxonomy`, `test_machine_verdict_carries_taxonomy` | `test_events.py` |
| 11 | `--dry-run` resolves and validates everything, dispatches nothing; a broken closure fails with the file path | `test_dry_run_dispatches_nothing`, `test_broken_closure_names_its_file` | `test_package_loads.py` |
| 12 | Interrupt and restart continues from persisted state | `test_resume_continues_from_disk` — two processes, `os._exit` mid-run, assert attempt 0 is `SUSPENDED` and attempt 1 exists | `test_build.py` |
| 13 | Running twice succeeds without hand-editing | `test_two_runs_do_not_collide` — two `--dry-run`s over one demo root | `test_package_loads.py` |
| 14 | The machine-readable output suffices for criteria 2–10 without parsing prose | `test_machine_output_answers_criteria_2_to_10` — one test that asserts each of the nine, reading only `fields` | `test_events.py` |
| 15 | The components import nothing from the demo | `test_no_component_imports_demo`, `test_examples_has_no_init` | `test_package_loads.py` |
| 16 | Loads through the ordinary task-package path: YAML, discriminated and schema-validated, no privileged import, no schema of its own | `test_loads_as_an_ordinary_package`, `test_package_declares_no_schema` | `test_package_loads.py` |
| 17 | Every filename found by convention, nothing bound by hand | `test_no_body_is_bound_by_hand` — no `body` key in any source **and** no `explicit-binding` warning from the loader; `test_body_paths_are_package_relative`'s `checked >= 8` is what stops an empty package satisfying it | `test_package_loads.py` |

### 14.2 Tests beyond the criteria

| Test | Guards |
|---|---|
| `test_expected_failure_that_passes_fails_the_run` | §7.5. Injects a passing `check_grounded` and asserts exit code 3 and an `UNEXPECTED_SUCCESS` event. **The most important test in this directory** — it is the one that stops the demo from reporting success when a safety property has stopped holding |
| `test_depends_on_is_derived_and_the_log_is_silent` | §6.2, M11. Captures `task_graph.scheduler`'s logger and asserts no `depends_on omits` record |
| `test_schema_version_matches_the_emitted_field` | §7.1. A constant and the thing it describes drift the moment they are two facts |
| `test_every_event_kind_has_a_human_rendering` | §7.2. Parametrised over `EventKind`; a new kind that renders as `<Event object>` fails here rather than in front of a reviewer |
| `test_no_event_field_holds_a_model` | §7.1. Asserts every value in `fields` is a JSON scalar, list or dict |

### 14.3 What is deliberately not tested here

The confined agent's behaviour. §9's measurements are real and reproducible
(`scratch/design/probes-demo/p3_confined_agent.py`), but a test that starts a
model call is non-deterministic, costs money, and fails on a fork — the reasons
spec §5 gives. The *properties* it demonstrates are CI-enforced in
`tests/env_mgr`; what `tests/cli` checks is that the demo **reports** them
correctly, which is a test about the event stream and needs no agent.

---

## 15. Implementation order

The demo is last for a reason: it needs every other module built. It also needs
something not yet built inside a module that is: `Task.parent`, `is_start`,
`is_end`, `closure` and `unfold()` are `task_graph` **design rev. 11**, criteria
36–54, and the shipped code is rev. 7. Criterion 2 is unreachable until those
land.

| # | Step | Depends on |
|---|---|---|
| 0 | *(precondition)* `task_graph` criteria 36–54 — nesting, `unfold`, `parent` | — |
| 1 | `events.py` + `stream.py` + the two renderers | nothing. Testable alone, and everything else emits into it |
| 2 | `examples/demo/` — the specs, `assets/produce.task/collect.py`, the two validator logics | `spec_loader`, `handoff`, `validator`, `agent`, `closure` |
| 3 | `package.py` + `test_package_loads.py` — **the CI half of §11** | 2 |
| 4 | `build.py` + `show` | 3, step 0 |
| 5 | `--dry-run` and the broken-closure case | 4 |
| 6 | `environment.py` — the context, the granted set, the preflight | `env_mgr` |
| 7 | `run` end to end | 5, 6, `agent` |
| 8 | `--resume` | 7 |

Steps 1–5 need no credentials, no sandbox and no model, and they carry ten of the
sixteen criteria. That is the ordering's payoff: most of the demo is testable
before the parts that need a machine with an API key exist.

---

## 16. Deviations, and new open questions

### 16.1 Deviations from the spec

Places where implementing the specification literally does not work. Each names
what was measured.

| # | Spec says | Design does | Why |
|---|---|---|---|
| **D1** | ~~Criterion 7 — one node runs as a program **"with no agent at all"**~~ | **No longer a deviation.** Criterion 7 rev. 6 asks for an agent of `kind: program`, which is what the package declares | §5. Rev. 1 of this document reported the old wording as unsatisfiable and measured why: `Task.agent_spec: str` and `Execution.agent_id: AgentId` are both required and `_dispatch_pass` calls `instantiate` unconditionally, all in code that is frozen and 423-tests green. Four modules had deferred it — `agent` D1, `closure` D1, `env_mgr`'s carried notes, this one. **It was resolved by removing the gap rather than filling it**: main spec §4.8 rev. 9 makes every task have an agent and `kind` the thing that varies. The demo's program node was going to declare a `kind: program` spec either way; what changed is that it is now the specified answer rather than a workaround |
| **D2** | Main design §2 — *"`demo/` docs only — the package itself is `examples/demo/`"* | `cli/` is an installed Python package (the runner); `examples/demo/` is specs and data, not installed | §2, §12, M13. A console script pointing into an unpackaged directory installs successfully and dies with `ModuleNotFoundError` when run. Making `examples*` an installed Python package would fix that and break something better: spec §1.1's rule that the demo uses nothing an out-of-repository package could use. The split is what makes that rule checkable — §14's `test_examples_has_no_init` |
| **D3** | `closure` D5 — nobody owns turning a closure into the root `Task` | `cli/build.py` owns it, in ~60 lines, and says it is in the wrong package | §6. The two named callers are this module's `show` and `--dry-run`, so the choice was to build it or to have no verbs. It moves to the whole-system CLI (TODO item 5) unchanged. **This is a component boundary crossed knowingly**, recorded so the move is a relocation and not a rediscovery |
| **D4** | `env_mgr` §4.5.1 / design §5.2's default granted set | Three more grants: `/dev/urandom`, `/run/systemd/resolve/stub-resolv.conf`, and a zone-local temp directory | §9.1, M1–M3. Each was found by being refused, and each broke differently: a 3 ms abort blaming Bun, a 184-second hang, and a clean actionable refusal. `env_mgr` D2 already extended §4.5.1 once for the same reason; this is the second extension and it will not be the last, which is O2 |
| **D5** | Nothing in the spec set mentions the backend's config directory | `CLAUDE_CONFIG_DIR=<zone>/config`, and no part of `$HOME` is granted beyond the backend's install directory and the CA file | §9.3, M5. Measured: with `~/.claude` granted, the confined demo agent read the operator's personal `CLAUDE.md` and changed language. A demo whose transcript depends on the reviewer's dotfiles is not reproducible, and criterion 13 is a reproducibility claim |
| **D6** | Spec §5 — *"The demo is not a test, and CI does not run it"* | CI runs `--dry-run` over the demo package on every commit | §11, S1. Not a contradiction, and the distinction is Airflow's: CI **loads** the example, a human **runs** it. `--dry-run` dispatches nothing, needs no credentials, no sandbox and no model. Without it, spec §1's claim that the demo is "the first thing to break when one of them drifts" is guarded by nobody — which is what happened to `jaffle_shop` (S3) |
| **D7** | Criterion 1 — *"`pip install -e agent_sys` then the run verb"* | Implemented as written, and a **wheel** install produces a working command that refuses to run, naming why | §12, M13. setuptools ships `.py` only, so a wheel carries the runner and not the specs. Packaging the specs as package data would make the example behave differently depending on how it was installed, which is worse than a clear refusal — but the refusal is a deviation from what a reader would expect of an installed command, so it is named |

### 16.2 New open questions

Found by this design, and not in spec §7.

| # | Question |
|---|---|
| **O1** | **The demo's first act is to modify the reviewer's repository.** `env_mgr` §7.2 requires `extensions.preciousObjects` on the main repository and `prepare()` enforces it, so the first thing that happens after `pip install -e` is a `git config` write on the checkout — before anything has been demonstrated. It is one reversible key and it is genuinely required. Whether a demonstration is allowed to do that silently, prompt for it, or refuse without `--allow-repo-config`, is not this design's to settle alone |
| **O2** | **Nothing enumerates what a second backend needs granted.** D4's three grants were found by running one binary and watching it break in three different ways. A different agent harness — a node-based CLI, a different vendor — has a different set, discoverable by the same method and no other. `env_mgr` M7 says an allow-list makes an ungranted file look *broken*; this module says the list is also **undiscoverable in advance**, which is the harder half |
| **O3** | **Criterion 12 costs a model call.** Resume re-runs the interrupted attempt (M10), correctly. So *where* a reviewer interrupts determines whether the demonstration costs one model call or two, and the demo can only suggest. If a future task caches an attempt's output, that changes; nothing plans to |
| **O4** | **`SCHEMA_VERSION` has no owner once the whole-system CLI exists.** §7 makes the demo's event stream a versioned interface because criterion 14 asserts over it. The whole-system CLI will want the same stream, and at that point two artefacts share one version constant with no policy for who bumps it. Terraform's answer is one constant in one package with a comment; ours would need the same, in `cli/` or somewhere better |
| **O5** | **Which model, and which tools** — spec §7's third open question, still open. This design fixes only what it must: the SDK's default is Claude Code's full tool set and system prompt, and §9.3 already narrows the config directory. What `describe` should actually be *given* — a model name, an allowed-tools list, a system prompt — is a package-content choice, and pinning a model name in a checked-in example has a cost the moment that name is retired |
| **O6** | **`JsonFileStoreMgr` has no cross-record transaction**, as its own docstring says, and M9 measured a task record being written *before* the handoff it names. Twelve interruption points produced no dangling reference only because the consumer happens to be written after the handoff — an accident of this graph's write order. The demo is the first artefact that kills a real process on purpose, so it is where this would first be seen |
| **O7** | **The exemption list starts the moment there are two examples.** Airflow's example suite carries two hand-maintained ignore tuples and a per-file timeout table (S2), accumulated one postponement at a time. Spec §1.1 says this repository holds exactly one task package, so the list is empty and stays empty — but that guarantee is a *spec* rule, and §11's CI guard silently depends on it |
