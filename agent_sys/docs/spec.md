# Agent Work System — Specification

| | |
|---|---|
| Status | Draft, revised after review |
| Revision | 12 — 2026-08-29. **Three §7 rows corrected against the landed code, not against the plan.** `spec_loader` now parses with `ruamel.yaml` round-trip and `validate` takes a parsed document, so: the **PyYAML** row no longer claims it reads specs (it does not — `env_mgr/recipe.py`, `handoff/verdict.py` and `handoff/store.py` still use it); the **`ruamel.yaml`** row records that round-trip mode is YAML 1.2 and rejects duplicate keys, which **closes the two traps rev. 10 flagged as live** — closed by parser choice, not by convention, and exactly one parser may touch a package document because the two disagree on real values; and §4.4's **shared-constants** row says the values are supplied to a package rather than declared in one, with the measurement (across the 21 replaced sources, every reference was to a run-level fact and no package declared a constant of its own), so no `vars:` block is built. No criterion changed. (rev. 11: 2026-08-29. **`main.yaml` is a role, not a requirement on every directory.** Rev. 10's criterion 16 fused two rules of different arity and one of them over-reached. `assets/` is about *being a package* — every document may write an unqualified path — so it is required of every package and that half is unchanged. `main.yaml` is about *being a run's entry*, which is one per **run**: `task_graph/bootstrap.py:47` takes `packages: Sequence[Any]` and `:255` loads each into one shared set of registries, so demanding the file of every package answers "where does a run start" N times and therefore not at all — and makes a kinds-only library package inexpressible, three paragraphs after §4.3 permits exactly that. **Presence is now the statement**: a package carrying `main.yaml` is runnable, one without is a library, and a `main.yaml` present but declaring no `module: task` is rejected because it is an entry to nothing. The per-run half — exactly one entry package — has no owner (nothing reads the file's contents; `cli/main.py:668,687` choose the root by closure name) and is a new §10 row naming both sides of the seam. Criterion 16 split, 18 added, nothing deleted. (rev. 10: 2026-08-29. **The user-interface stage: YAML replaces jsonnet, and the agent rule becomes leaf-only.** Four amendments, and the first three are one change. **§4.4** is now `YAML → schema`: the render step is gone, because measurement over all 21 `.jsonnet` / `.libsonnet` sources found the whole computation surface to be constants, string concatenation, and default-if-absent. `config` / `extVar` is replaced by naming where each of those three now lives — a package-level variable set for the first two, a schema `default` for the third — and "the schema is the only enforcement point" survives intact and slightly stronger. **§2.3**'s three tiers become two: tier ② was jsonnet and stopped existing rather than moving. **§4.3** admits **two mandatory names**, `main.yaml` and `assets/`, and replaces the unqualified "the loader does not interpret a package's layout" with what the exception costs. **§7** moves jsonnet to rejected with the measurement, replaces Kustomize's reason with the real one (its API is the Go package `krusty`; there is no Python binding), and adopts `ruamel.yaml` for a parse that carries source positions. **§4.8** narrows to leaf-only: a non-leaf's `agent` was read by nothing (`agent/runner.py:682`, `env_mgr/prepare.py:447`, `validator/environment.py:140`). Criteria 3, 4, 5, 6 and 9 amended in place, 16 and 17 added; this is the first revision to amend a criterion rather than only add one. (rev. 9: 2026-08-27. **Every task has an agent, and an agent need not be an AI** — `ai`, `human` or `program` (§4.8). The previous wording, "a task may have no agent at all", was inexpressible in `task_graph` and had been deferred by four design modules; `kind: program` is what a task without an AI has. (rev. 8: 2026-08-26. Four diagrams added to §2, no specification changed: the system as a reader meets it — four core subjects framed by `task_graph`, substrate below, registries above, O11y and the monitor vertical; the same structure as a strict dependency order; and a spec's three tiers of definition — contract, template, declaration — which together are what lets the core run a package end to end (§2.3). Plus the life of one task as a state machine and the same life zoomed out to a task package (§2.4). (rev. 7: 2026-08-26. Restructured most-important-first: a whole-system architecture diagram, a component map, and the life of one task in a new §2; the design principles follow as §3; a reading guide in §1.4. No specification changed — the survey count and the leaf/lease wording were reconciled with the component specs. (rev. 6: 2026-08-26. The monitor acts through a task's own transitions, never on its status; cascading cancel is in scope, cascading invalidation is not (§5.1). (rev. 5: 2026-08-26. Runtime fan-out is a framing, not a prohibition: the catalogue is static, the instance count is not (§6.1). Producer means the agent, not the task (§5.2). Isolation criteria are CI-enforced. (rev. 4: A workflow's specs live in a task package outside this repository; this repository holds the schemas, the loader, and the general specs. jsonnet adopted for templating, with the JSON Schema as the only enforcement point (§4.3–§4.5). rev. 3: Failure reaction belongs to the graph's monitor, not the scheduler (§5.1). rev. 2: Review of PR #132: `decoupled` promoted to the first principle; the use-the-existing-wheel rule; a task may have no agent; validations invisible to the scheduler; two research-reversed decisions recorded. rev. 1: initial)))))))) |
| Date | 2026-08-24 |
| Scope | The whole system: what it is, which components exist, and what each owes the others |
| Source | The task definition; a survey of evaluation frameworks and of agent-harness isolation (§7.1) |

---

## 1. Purpose

Provide the substrate for an agent-driven workflow whose every step is
**reproducible, checkable, and auditable**.

The system's one claim is this:

> **A task is a function.** Its signature is `<handoffs, agent>`. Quality is
> guaranteed by standardising the inputs and outputs, not by trusting the
> executor.

Everything else follows. A handoff is the function's typed input or output; a
validator is what makes that type real rather than aspirational; a task graph is
the composition of those functions; an agent is one executor of one task, and may
equally be a human or a program.

```
system  =  task graph  =  graph of <handoffs, agent>
```

### 1.1 Why this, and not a prompt

The first-phase target is a fixed workflow — LLM inference performance
optimisation — where the same six steps run over and over against different
models and hardware. A system for that does not need agents that improvise. It
needs the opposite: **the same procedure, executed identically, with each step's
output verified before the next one starts.**

Two properties of this domain decide the design. **The expensive gate is very
expensive** — an end-to-end benchmark costs GPU-hours, so anything catchable
earlier must be caught earlier. And **the interesting failures are not crashes**:
a trace that silently omits half the kernels, an operator that is fast because it
returns NaN, a configuration that is valid and unrepresentative. Those pass every
check that only asks "did it run".

So the system puts cheap, early, independent checks in front of the expensive
one — which is what §5.2 makes structural.

### 1.2 In scope

- The **handoff**: what a unit of transfer must carry and how it is versioned.
- The **validator**: what makes a handoff checkable, and how much a given check
  can be trusted.
- The **task graph**: which task runs when, including subgraph nesting.
- The **agent**: what an executor must declare, and the backend abstraction that
  keeps the system independent of any one harness.
- The **closure**: the predefined binding of the four.
- The **environment**: all interaction with the operating system — storage,
  workspaces, isolation, local↔remote mapping.

### 1.3 Out of scope

- **What an agent does internally.** If a backend organises its own multi-agent
  structure, the system does not see it and does not manage it
  (see `../agent/docs/spec.md` §4.2).
- **The runtime environment of the program under test.** sglang, vllm, and infera
  environments and their before/after consistency belong to the handoffs and
  validators that name them, not to `env_mgr` (see `../env_mgr/docs/spec.md` §7).
- **Frameworking the test code a validator runs.** The validator system specifies
  where a check plugs in, not how the check's own code is organised; that belongs
  to the owning external test system (see `../validator/docs/spec.md` §6).
- **Dynamic task graphs.** §6 states the record-and-replay scope and its cost.

### 1.4 How to read this document, and the eight others

This document is written **most important first**. A reader who stops after §5
has the whole architecture and both constraints the design turns on; §6 onward is
scope, justification, and bookkeeping.

| Section | What it settles | Stop here if you want |
|---|---|---|
| §2 | The architecture: the flow, the seven components, the structure — what surrounds what, what cuts across, and the two tiers that make a spec real — and the life of one task at three zoom levels | the shape of the system |
| §3 | The principles that decide daily arguments | to know *why* it is shaped that way |
| §4 | The four objects, where their specs live, and how one is loaded | to write or load a spec |
| §5 | **The two authority boundaries.** The load-bearing constraints | to know what must never be weakened |
| §6–§7 | The v1 scope, and what was adopted versus built | to review a technical decision |
| §8–§10 | System-level criteria, the index into each component's, open questions | to check or extend the work |

**The component specs are read on demand, and none is a prerequisite for
another.** Where two need the same fact, one states it and the other links —
so a section that seems thin is usually deferring, not omitting.

| Read | When you are |
|---|---|
| [`handoff`](../handoff/docs/spec.md) | defining what a step produces or consumes |
| [`validator`](../validator/docs/spec.md) | deciding how a handoff is checked, and how far to trust the check |
| [`task_graph`](../task_graph/docs/spec.md) | working on scheduling, subgraphs, task state, or recovery |
| [`agent`](../agent/docs/spec.md) | adding an executor or a backend |
| [`closure`](../closure/docs/spec.md) | binding the four into one workflow step |
| [`env_mgr`](../env_mgr/docs/spec.md) | touching storage, workspaces, or isolation |
| [`demo`](../cli/docs/spec.md) | checking that all of it actually composes |

Two documents beside these carry work rather than specification:
[`ROADMAP.md`](ROADMAP.md) for deferred subsystems, [`TODO.md`](TODO.md) for
near-term decisions. Neither specifies anything; a rule that lives only there is
not yet a rule.

---

## 2. Architecture

### 2.1 The whole system on one page

Read top to bottom: a workflow is declared, loaded, scheduled, and executed. Each
band is a different kind of thing, and the boundaries between them are what the
rest of this document specifies.

```
 DECLARED   ┌ a task package ─────────────────┐   ┌ this repository ───────────┐
  §4.3      │  handoff · validator · task ·   │   │  5 JSON Schemas  §4.3      │
            │  agent · closure — YAML         │   │  general specs   §4.5      │
            │  documents, one workflow's own  │   │  the loader                │
            └────────────────┬────────────────┘   └─────────────┬──────────────┘
                             │                                  │
 LOADED     ┌────────────────▼──────────────────────────────────▼──────────────┐
  §4.4      │  scan & discriminate ──► validate (JSON Schema) ──► admit         │
            │  the source is never seen; the package delivers parsed documents  │
            └────────────────────────────────┬──────────────────────────────────┘
                                             ▼
            ┌ four independent registries ────────────────────┐  ┌ closure ─────┐
  §4.1      │  handoff · validator · task · agent             │◄─┤ the binding  │
            │  name → spec. The vocabulary every other        │  │ of the four. │
            │  component's references are drawn from          │  │ Load-time    │
            └────────────────────────────────┬────────────────┘  │ only  §1.1   │
                                             │                   └──────────────┘
 SCHEDULED  ┌────────────────────────────────▼──────────────────────────────────┐
            │                          task_graph                               │
            │   decides WHEN a task runs. Never WHAT it does.      ◄── BOUNDARY 1
            │   reads: are this task's inputs valid? do its resources fit?       │
            └────────────────┬──────────────────────────────┬───────────────────┘
                             │ dispatches ONE task          │ read-only:
                             ▼                              │ is this input's
 EXECUTED   ┌ TaskRunner ─────────────────┐                 │ latest version
            │  1. input validation  ──────┼─────────────────┘ VALID?
            │  2. main ──► agent, or a    │
            │     subgraph of tasks       │   the two validation phases are
            │  3. output validation ──────┼──► invisible to the scheduler  §5.1
            └──────┬───────────────┬──────┘    and write the verdict       §5.2
                   │               │                                  ◄── BOUNDARY 2
                   ▼               ▼
            ┌ handoff storage ┐  ┌ env_mgr ────────────────────────────────────┐
            │ versioned slots │  │ workspace · playground · storage · isolation │
            └─────────────────┘  └──────────────────────────────────────────────┘
```

Three readings of that picture are worth stating, because each is a rule
elsewhere:

- **Declaration and system are separate boxes.** A workflow's specs are not in
  this repository; nothing here changes to add, change, or retire one (§4.3).
- **The scheduler's two arrows are both thin.** One dispatch out, one yes/no
  question in. It never reaches the bottom band at all (§5.1).
- **The verdict is written in the third phase, not the second.** A task is one
  producer from outside and three isolated phases from inside (§5.2).

### 2.2 Components

Seven. Each has its own specification; this table is the map.

| Component | Owns | Specification |
|---|---|---|
| `handoff` | What a unit of transfer carries: content shape, digest, scope tags, validator binding | [`../handoff/docs/spec.md`](../handoff/docs/spec.md) |
| `validator` | What makes a handoff checkable, and how far a check can be trusted | [`../validator/docs/spec.md`](../validator/docs/spec.md) |
| `task_graph` | Which task runs when. Nothing else — it never inspects what a task does | [`../task_graph/docs/spec.md`](../task_graph/docs/spec.md) |
| `agent` | What an executor declares, and the backend abstraction | [`../agent/docs/spec.md`](../agent/docs/spec.md) |
| `closure` | The predefined binding of the four objects | [`../closure/docs/spec.md`](../closure/docs/spec.md) |
| `env_mgr` | All interaction with the operating system, including isolation | [`../env_mgr/docs/spec.md`](../env_mgr/docs/spec.md) |
| `demo` | The runnable proof that the above compose | [`../cli/docs/spec.md`](../cli/docs/spec.md) |

`task_graph` and `env_mgr` are implemented; the rest are specified here and built
in later stages.

### 2.3 The structure: what surrounds what, what cuts across, what makes a spec real

§2.1 is the flow; this is the *structure*. Three cuts through it: what the pieces
are and what surrounds what; the same thing as a strict dependency order; and
then a cut the other way, through a single spec.

**The system as a reader meets it.** Four core subjects in the middle, the graph
that composes them around those, the substrate underneath, the registries above,
and the mechanisms that stand vertically through all of it.

```
   the user's TASK PACKAGE — YAML specs, outside this repository  §4.3
                             │  reaches the system ONLY through the registries
 ════════════════════════════▼══════════════════════════════════════════  ┌───┐
 ┌ REGISTRIES — the vocabulary  §4.1 ────────────────────────────────────┐│ O │
 │  handoff · validator · task · agent   +  closure: the binding of the  ││ 1 │
 │  four, checked at load and read no further  (closure spec §1.1)       ││ 1 │
 └───────────────────────────────────────────────────────────────────────┘│ Y │
 ┌ task_graph — composes the subjects into a workflow  §5.1 ─────────────┐│   │
 │   decides WHEN a task runs, never WHAT it does                        ││ + │
 │  ┌ THE FOUR CORE SUBJECTS ─────────────────────────────────────────┐  ││   │
 │  │                                                                 │  ││ M │
 │  │      TASK ─── is ───►  < HANDOFFS , AGENT >                     │  ││ O │
 │  │       │                     ▲          ▲                        │  ││ N │
 │  │       │ may hold a          │          └── rules · hooks ·      │  ││ I │
 │  │       │ SUBGRAPH of         │              skills · knowledge   │  ││ T │
 │  │       ▼ tasks               │              (6 types) · backends │  ││ O │
 │  │      ( … )              VALIDATOR                               │  ││ R │
 │  │                         makes that type real, not merely        │  ││   │
 │  │                         claimed — and cannot be the producer    │  ││ e │
 │  └─────────────────────────────────────────────────────────────────┘  ││ v │
 │   Scheduler + status pools · TaskRunner · SchedulePolicy ·            ││ e │
 │   TaskMgr · HandoffMgr · AgentMgr · ResourceMgr · StoreMgr            ││ r │
 └───────────────────────────────────────────────────────────────────────┘│ y │
 ┌ SUBSTRATE — everything above rests on it, and it knows none of it ────┐│   │
 │  env_mgr:  filesystem · isolation · remote · metadata ·               ││ b │
 │            agent environment (workspace, playground) · storage        ││ a │
 └───────────────────────────────────────────────────────────────────────┘│ n │
       wiring:  the component Registry — name → component, resolved at    │ d │
       use time. No module imports another   (task_graph spec §4.1)       └───┘
                                                                         not yet
```

Read off it:

- **The four subjects are the whole model.** `task = <handoffs, agent>` is §1's
  claim drawn; the validator is what makes the handoff a *type* rather than a
  hopeful label; and a graph is the composition of those functions.
- **`task_graph` surrounds the subjects rather than sitting above them.** It is
  the frame: it composes and sequences, and never looks inside (§5.1). The
  validator sits inside the frame with the other three and is still unreachable
  from the producer — that is §5.2, and it is a property of the *phases*, not of
  the box.
- **A task package touches the registries and nothing else.** No workflow reaches
  the substrate, the scheduler, or another package directly (§4.3).
- **The small pieces belong to a subject, not to the system.** Rules, hooks,
  skills, knowledge, and backends hang off the agent — which is why they are
  drawn in its corner and specified in
  [`../agent/docs/spec.md`](../agent/docs/spec.md) §3, not here. A piece with no
  owner ends up owned by everyone.
- **O11y and the monitor are vertical by nature** — they must see every band or
  they see nothing. **Neither exists yet** (§3.1 principles 4–5,
  [`ROADMAP.md`](ROADMAP.md)); the column marks where they attach.

**The same structure as a dependency order.** The picture above says what
surrounds what; this one says what may know about what, which is the stricter and
more useful statement. Five layers, each depending only on the one above it, plus
the two things that genuinely span them.

```
                                                           ┌ SPANS ────────────┐
┌ 1  DECLARATION — outside this repository  §4.3 ─────────┐ │                   │
│  task package(s): YAML documents for the five kinds     │ │  closure   §4.1   │
│  ├ handoff kinds   ├ validators   ├ tasks               │ │  the group of the │
│  ├ agents          └ closures                           │ │  four. A load     │
│  general specs — workflow-independent, in this repo §4.5│ │  checker plus     │
│  the demo package — the only one here  (demo spec §1.1) │ │  read-only        │
└─────────────────────────────────────────────────────────┘ │  queries.         │
┌ 2  ADMISSION — this repository  §4.4 ───────────────────┐ │  LAYERS 1–2 ONLY: │
│  the five JSON Schemas — the spec of each spec          │ │  nothing at       │
│  the loader:  scan & discriminate ─► validate           │ │  runtime points   │
│  four independent spec registries  §4.1                 │ │  at one.          │
│  ├ handoff   ├ validator   ├ task   ├ agent             │ │                   │
└─────────────────────────────────────────────────────────┘ ├───────────────────┤
┌ 3  SCHEDULING — task_graph.  Decides WHEN  §5.1 ────────┐ │                   │
│  Scheduler + the status pools  │  SchedulePolicy        │ │  handoff          │
│  TaskMgr · HandoffMgr · AgentMgr · ResourceMgr          │ │  the unit of      │
│  StoreMgr — write-through, four record kinds            │ │  transfer, and    │
└─────────────────────────────────────────────────────────┘ │  the system's     │
┌ 4  EXECUTION — one dispatch  §5.2 ──────────────────────┐ │  only interface.  │
│  TaskRunner: input validation ─► main ─► output valid.  │ │  ALL FIVE LAYERS: │
│  validator: the four elements, dimension, strength      │ │  declared at 1,   │
│  agent: spec, backend list, knowledge, lifecycle        │ │  versioned at 3,  │
└─────────────────────────────────────────────────────────┘ │  written at 4,    │
┌ 5  PLATFORM — env_mgr.  All contact with the OS ────────┐ │  stored at 5.     │
│  filesystem · isolation · remote · metadata             │ │                   │
│  agent environment (workspace, playground) · storage    │ │                   │
└─────────────────────────────────────────────────────────┘ └───────────────────┘

    wiring, not a layer:  the component Registry  (task_graph spec §4.1)
    name → component, resolved at use time. Layers 3–5 hold the registry and
    never import each other. It is how the boxes above connect at all.
```

Four things that diagram is claiming:

- **No layer knows about the layers above it.** `env_mgr` has never heard of a
  task; `task_graph` has never heard of a spec registry or a closure. That is
  §3.0 made structural rather than aspirational
  ([`../env_mgr/docs/spec.md`](../env_mgr/docs/spec.md) §7 is that module's own
  statement of it).
- **The layers are not a call stack.** Layer 3 dispatches into layer 4; layer 4
  uses layer 5. Layer 3 does not reach layer 5 at all (§5.1), and layer 2 is
  consumed *before* a graph exists rather than called during one.
- **`closure` spans only the top two.** It is declared in a package and checked at
  load, and no runtime object points at one — which is what keeps it from
  becoming a fifth object
  ([`../closure/docs/spec.md`](../closure/docs/spec.md) §1.1).
- **The handoff is the one thing present in every layer**, which is why its spec
  is the one every other component reads.

**Observability and intervention (§3.1, principles 4–5) would be a third span,
and do not exist yet.** They are on [`ROADMAP.md`](ROADMAP.md); the shape above is
where they attach.

**A third cut, in the other direction.** The two pictures above slice the
*system*. This one slices a **spec** — a handoff kind, a validator, a task, an
agent — and answers "what does it take before one of these means anything".

Each is defined in **two tiers, by two different authors**. Neither alone is a
usable spec; the two together are one, and only then can the core run the package
end to end. §4.4 specifies the mechanism; this is the shape.

**It was three until rev. 10**, and the middle one was jsonnet — "the shape of one
recurring kind, with `config` as its declared fill interface". §4.4 records the
measurement that removed it: nothing was using the language it needed. The tier
did not move somewhere else, it stopped existing, and the numbering is kept so
that ① and ③ still mean what every other section says they mean.

```
   TIER          WHO WRITES IT, AND WHAT IT FIXES
 ┌───────────────────────────────────────────────────────────────────────┐
 │ ① CONTRACT      this repository                        §4.3, §4.6     │
 │                 the data model · the JSON Schema · the interface the  │
 │                 core calls.  Identical for every workflow, fillable   │
 │                 by nobody.  A field sealed here is sealed for good —  │
 │                 the schema is checked over the DELIVERED document,    │
 │                 so no amount of cleverness upstream evades it  §4.4   │
 │                 It also carries every `default`, which is where a     │
 │                 reader learns a field's fallback  §4.4                │
 ├───────────────────────────────────────────────────────────────────────┤
 │ ② ——— gone at rev. 10.  Was jsonnet templating; measured unused §4.4  │
 ├───────────────────────────────────────────────────────────────────────┤
 │ ③ DECLARATION   the task package — or a general spec  §4.3, §4.5      │
 │                 THIS trace kind, THIS check, THIS step, THIS          │
 │                 executor.  Where the semantics enter — and now also   │
 │                 where customisation happens, through the package's    │
 │                 own variable set, substituted before it emits  §4.4   │
 │                 A general spec is one that ships here rather than in  │
 │                 a package; there is nothing else special about it §4.5│
 └───────────────────────────────────────────────────────────────────────┘
        ①+③, for each of the four objects, and only then:
                                 │
                                 ▼   scan & discriminate ─► validate  §4.4
 ┌───────────────────────────────────────────────────────────────────────┐
 │  A REAL SPEC — one with meaning.  "collect the trace", not "a task"   │
 └───────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  agent_sys core  +  the user's task definitions  =  a running system  │
 │                                                                       │
 │  the core calls each object through ①'s interface and nothing else,   │
 │  and the package's own INPUT becomes its OUTPUT.  This is the thing   │
 │  the user actually wanted; everything above exists to make it         │
 │  reproducible, checkable, and auditable    §1                         │
 └───────────────────────────────────────────────────────────────────────┘
```

Three things follow, and each is a rule elsewhere:

- **Semantics live only in ③.** ① knows a task has inputs and outputs; only ③
  knows this one collects a trace. That is why the system can be specified without
  any workflow existing, and why §4.3 can put the workflow outside this
  repository.
- **Both guarantees now come from the two surviving tiers, and one of them
  changed hands.** Safety is still ①'s — the schema, checked over the delivered
  document. Customisation was ②'s and is now ③'s: a package parameterises itself
  with its own variable set rather than filling a template someone else wrote.
  **That is a real loss and it is worth naming**: nothing is reusable across
  packages any more, because there is no artefact to reuse. §4.4's measurement is
  the reason it is an acceptable loss — the reusable thing was carrying constants
  and string concatenation, not a shape.
- **Nothing runs until both are present for all four objects.** A missing ③ is a
  load error, not a run-time surprise; the closure check is where a group of four
  is confirmed complete
  ([`../closure/docs/spec.md`](../closure/docs/spec.md) §4).

### 2.4 The life of one task

Three views, zooming out: one task's states, then the package those tasks come
from, then the same life as a sequence.

**First, the state machine**, because a reader's first question is what a task
can be and what moves it:

```
                    submit
                      │
                      ▼
        ┌ WAITING_HANDOFF ─────────────┐   is every input's latest
        │                              │   version VALID?
        └──────────────┬───────────────┘
                       │ yes                            ▲
                       ▼                                │ resume
        ┌ WAITING_RESOURCE ────────────┐                │
        │ a LEAF competes for its full │   remove_      │
        │ set; a non-leaf acquires     ├── queued ──► CANCELLED [final]
        │ nothing   task_graph §6.2    │
        └──────────────┬───────────────┘
                       │ granted, all-or-nothing — the scheduler's
                       │ ONE dispatch. It sees phase 2 and no other  §5.1
  ═════════════════════▼═══════════════════════════════════════════════
   ONE DISPATCH, ONE LEASE
   ┌ 1 ─────────────┐   ┌ 2 ─────────────┐   ┌ 3 ──────────────────┐
   │ INPUT          │──►│ MAIN           │──►│ OUTPUT VALIDATION   │
   │ VALIDATION     │   │ the agent, or  │   │ fresh environment,  │
   │ cheap checks   │   │ a subgraph of  │   │ unreachable by the  │
   │ over inputs    │   │ tasks          │   │ agent. The verdict  │
   │                │   │                │   │ is written HERE §5.2│
   └───────┬────────┘   └───┬────────┬───┘   └─────┬──────────┬────┘
           │                │        │ stop()      │          │
  ═════════╪════════════════╪════════╪═════════════╪══════════╪═════
           │                │        ▼             │          ▼
           │                │    STOPPING          │      SUCCEEDED [final]
           │                │        │             │      Its outputs carry
           ▼                ▼        ▼             ▼      whatever verdict was
         FAILED [resumable]     SUSPENDED       FAILED    recorded — not always
                                [resumable]               VALID  task_graph §6.3
```

**The authoritative state table is `task_graph` spec §3.2**, including every exit
condition and the two states this drawing folds away. What is worth reading off
*this* version is the horizontal rules: they are the two authority boundaries of
§5. Above the first, only the scheduler acts. Between them, only the runner does,
and the scheduler is told nothing except that phase 2 happened. Below the second,
the task is over.

**Second, one zoom level out: the life of a task package**, from the sources on
disk to the artefacts a run leaves behind.

```
 AUTHORED   a task package: YAML documents for its handoff kinds, validators,
   §4.3     tasks, agents, and closures. Outside this repository; versioned and
            shipped on its own cadence. `main.yaml` and `assets/` are the only
            two names it does not choose for itself
                             │
                             ▼   scan ─► discriminate ─► validate against the schema
 ADMITTED   four spec registries populated + the closure check, which is the
   §4.4     only pass that sees all four at once. Any failure here and nothing
  closure   runs — a load error, not a run error   (closure spec §4)
   §4         │
              ▼   a graph is assembled from admitted closures. AFTER this
                  point no closure is read again  (closure spec §1.1)
 SUBMITTED  ┌──────────────────────────────────────────────────────────────┐
            │  task_graph: the DAG. Every task's output slots declared,     │
            │  empty. The graph's EDGES ARE handoff slots — one task's      │
            │  output slot is the next task's input                         │
            └──────────────────────────┬───────────────────────────────────┘
                                       ▼
 RUNNING    ┌ run ─┐  out v1     ┌ run ─┐  out v1     ┌ run ─┐
            │ task ├── VALID ───►│ task ├── INVALID ─╫│ task │  never runs: its
            │  A   │             │  B   │            ╫│  C   │  input's latest
            └──────┘             └──────┘             └──────┘  version is not
              each box is one copy of the state machine above   VALID, so it
                                                                stays in
                                                                WAITING_HANDOFF
 LEFT       versioned handoffs with their verdicts · execution records, one per
  BEHIND    run · agent bindings · the store's four record kinds. This is the
            audit trail, and it is what "reproducible" is claimed against  §1
```

Three rules that picture is the clearest statement of:

- **A load failure and a run failure are different events.** Everything above
  `SUBMITTED` fails before any work starts, and names a file and a value.
- **The closure exists between those two bands and nowhere else.** It is read to
  assemble the graph and never again.
- **An `INVALID` output is not a task failure.** Task B above reached `SUCCEEDED`;
  its output simply never became eligible input for C, which waits. The two facts
  are recorded separately, and `task_graph` spec §6.3 is why.

**Third, the same life as a sequence**, because the ordering is what a reader
most often needs and neither diagram gives it:

| # | What happens | Who does it | Where specified |
|---|---|---|---|
| 1 | Its closure is loaded: the group of four resolves, or the load fails | the loader | `closure` spec §4 |
| 2 | It is submitted; its output handoff slots are declared, empty | `task_graph` | `task_graph` spec §6.1 |
| 3 | It waits until every input's latest version is `VALID` | `task_graph` | `task_graph` spec §3.2 |
| 4 | It waits for resources. **A leaf acquires; a non-leaf never does** | `task_graph` | `task_graph` spec §6.2 |
| 5 | Its environment is built: storage, handoffs staged, workspace, sandbox. **No sandbox, no start** | `env_mgr` | `env_mgr` spec §8 |
| 6 | **Input validation.** Cheap checks over its inputs | `TaskRunner` | `validator` spec §3 |
| 7 | **Main.** Its agent works, or its subgraph expands into scheduler-visible tasks | the agent, or `task_graph` | `task_graph` spec §3.2.1 |
| 8 | **Output validation**, in an environment the main phase's agent cannot reach. The verdict is written here | `TaskRunner` | `validator` spec §3, §8 |
| 9 | It reaches `SUCCEEDED` or `FAILED` and releases what it held. A failure is a task failure and nothing more | `task_graph` | `task_graph` spec §6.3 |
| 10 | Reacting to a failure — escalate, restart, cancel — is the monitor's, through the task's own transitions | the monitor | `task_graph` spec §3.2.3 |

Steps 6–8 are one dispatch. The scheduler saw step 7 and nothing else.

---

## 3. Design principles

### 3.0 The first principle: **decoupled**

> **Everything is decoupled, or stays easy to decouple.**

It is listed first and separately because it decides more arguments than the rest
combined, and because it is the one that erodes silently. Three consequences that
come up constantly:

- **The agent system and its usage are separate things.** `agent_sys` is not "the
  LLM optimisation pipeline"; the pipeline is one thing built on it. A rule that
  only makes sense for the optimisation workflow does not belong in the system.
- **The system does not care which backend runs an agent** — not because backends
  are unimportant, but because caring would couple every component to one
  harness's lifecycle.
- **Long-term evolution and maintenance is the reason.** This is a repository
  worked on by agents; a coupled repository is one where a change in any file can
  require reading every file. Decoupling is what keeps a change local.

### 3.1 The rest

| # | Principle | Consequence |
|---|---|---|
| 1 | **Reproducible or it did not happen** | Any deliverable, conclusion, or performance number rests on a reproducible basis |
| 2 | **Develop against the validator** | The goal is not "the output looks right" but "the output passes its validator". The validator is specified first, the implementation second |
| 3 | **`<context, worker>`** | A worker is `<executor, knowledge, rules>`; a context is `<content, protocol, validation programs>`. Each has a clear, complete, checkable input and output, and does only its own job |
| 4 | **Observability** | Levelled logging, an outside view of whether an agent has drifted or is looping, and a final process summary. A system that can be observed can be repaired. The subsystem is on the roadmap |
| 5 | **Interventionability** | An agent can be interrupted and instructed; a control surface can abort |
| 6 | **Risk has an exit** | What an agent cannot decide, it reports. Some reports may not be self-issued, which is why §5.2 exists |
| 7 | **Measurable** | A scoring mechanism exists. Where two results are equal, per-token and per-time efficiency break the tie |
| 8 | **Composable and pluggable** | The optimisation flow does not change, so each new external tool integrates through a thin wrapper against a standard interface |
| 9 | **Composition over inheritance** | Inheritance appears only where two things genuinely differ in behaviour |
| 10 | **One fact, one place** | Where two operations mean the same thing, one is expressed in terms of the other — not licence to collapse two genuinely different concerns |
| 11 | **Use the existing wheel** | §3.2 |

### 3.2 Use the existing wheel

**Research before building. Do not invent a wheel that already exists.**

The rule, in order:

1. Find out whether a mature, widely adopted library, interface, or CLI tool
   already does this.
2. If a de-facto standard exists, **use it directly** — do not reimplement it.
3. If a thin wrapper over a standard tool would do, **write the wrapper**.
4. Only build it yourself when nothing fits, and **record why** in the module's
   README.

This is not only about saving effort. Two of the decisions in this document were
reversed by research after they had been written down and looked reasonable: the
validator's template system (no surveyed system does it; two explicitly prohibit
the nesting variant) and the isolation mechanism (path-prefix matching is a
CVSS 9.1 CVE in the harness we build on). Both would have shipped. §7 records
what was adopted and what was rejected.

---

## 4. The four objects

Brief §9 asks for a uniform treatment of four objects. It is honoured literally:

**Each of `handoff`, `validator`, `task`, and `agent` has five things:**

| | |
|---|---|
| A **static spec** | A YAML document, validated against a JSON Schema |
| A **runtime object** | Identified by a uuid, distinct per run |
| A **runtime manager** | Owns the collection of live objects |
| A **spec registry** | Name → spec. Knows what kinds exist |
| A **schema** | The spec of the spec. Lives in this repository (§4.3) |

Where the spec *files* live is a separate question, and the answer is **not "a
folder in this repository"** — see §4.3.

### 4.1 The four registries are deliberately separate

The four registries do structurally similar work — load YAML, validate against a
schema, map a name to a spec, scan a folder — and one generic mechanism serving
all four is the obvious consolidation. It is not taken, for three reasons:

- **They diverge where it matters.** A validator registry must maintain the
  two-way binding to handoff kinds and the used-by index (validator spec §7); a
  handoff registry must not. Generic machinery would grow per-kind branches until
  it was four implementations sharing a directory.
- **The registry is the vocabulary.** `Task.agent_spec` names an entry in the
  agent registry; a handoff spec names entries in the validator registry. Keeping
  them separate keeps those references unambiguous in a signature.
- **The shared part is small.** Loading a schema-constrained YAML file is a few
  lines against a mature library (§7). Sharing a few lines is not worth coupling
  four vocabularies.

The cost is accepted and named: four places to change when the loading mechanism
changes. If that becomes painful, the fix is a shared *loader* the four
registries call — not a shared registry.

### 4.2 Spec versus instance, in one line each

| Object | The spec says | The instance is |
|---|---|---|
| handoff | what this *kind* of artefact must carry | one slot in one graph, with its versions |
| validator | how this *kind* of check is performed | one run of that check against concrete handoffs |
| task | what this *step* is: its handoffs and its agent kind | one node in one graph, with its execution history |
| agent | what this *kind* of executor knows and may touch | one executor bound to one task run |

The distinction is load-bearing and is why `Task.agent_spec` is a *spec name*
rather than an agent (`task_graph` spec §3.2).

### 4.3 This repository holds schemas; task packages hold specs

**A concrete workflow's specs do not live here.** One end-to-end workflow — its
handoff kinds, its validators, its tasks, its agents, its closures — is a **task
package**: a self-contained directory, outside this repository, holding every
spec that workflow needs.

| This repository | A task package |
|---|---|
| The five **JSON Schemas** — one per object, plus the closure's | The concrete specs: `collect_trace`, `check_trace_shape`, … |
| The **loader, the registries, and the loading pipeline** (§4.4) | The YAML sources, organised however the package likes — **except for two names** |
| **General specs** that are workflow-independent (§4.5) | Everything workflow-specific |

This is principle 2.0 applied to the one place it is easiest to lose. Maintaining
a concrete workflow is not this system's job; a `collect_trace` handoff kind
living in this repository would make every change to that workflow a change to
the system.

**Two packages may reference each other**, and they do it themselves — a relative
symlink from one package into another. The loader resolves a path, and it does not
know that packages can be nested, adjacent, or shared. Anything more would be the
coupling this rule exists to avoid.

#### Two mandatory names, and what they cost

Earlier revisions said flatly that the loader "does not interpret a package's
layout". **The system now fixes exactly two names at a package root** — it fixes
the *names*, which is not the same as demanding both files of every package, and
the subsection after next is about that difference. Leaving the old sentence
unqualified beside them would make this document disagree with itself:

| Name | What it is | Why it cannot be left to convention |
|---|---|---|
| `main.yaml` | The entry declaring the **outermost** graph | A package holds many documents and more than one may declare a graph. Something has to say where a run starts, and *inferring* it — take the one nothing references, say — makes a package's meaning depend on a global property of the whole tree, so adding an unreferenced draft would silently move the entry point |
| `assets/` | The root that unqualified paths resolve against, and the tree in which files are found by filename convention | A path written in a document has to be relative to *something*. The document's own directory would mean moving a file changes what it points at; the package root would put the scanned documents and the assets in one namespace, so a `*.yaml` under `assets/` would be both an object and an asset |

#### The two names are not the same kind of rule, and rev. 10 fused them

Rev. 10 wrote both into one criterion — *"a package missing `main.yaml` fails
naming the root, and so does one missing `assets/`"*. **That is right for one of
them and an over-reach for the other, and the difference is arity:**

| | `assets/` | `main.yaml` |
|---|---|---|
| What the rule is about | **being a package.** Every document may write an unqualified path, and every such path resolves against this directory. There is no package for which the question does not arise | **being a run's entry.** A run starts in one place |
| Its arity | one per **package** | one per **run** |
| Consequently | required of every package, and a missing one is fatal | required of the package a run starts from — which is a role, not a property of the directory |

**Requiring `main.yaml` of every package defeats the reason given for it.**
`task_graph/bootstrap.py:47` takes `packages: Sequence[Any]` — several — and
`:255` loads each into one shared set of registries. A two-package run holds two
files each claiming to declare *the outermost graph*, and the rationale above —
"something has to say where a run starts" — is answered twice and therefore not
at all.

**And it makes a shape this very section permits inexpressible.** "Two packages
may reference each other" is three paragraphs up; the natural form of that is a
package shipping only shared handoff kinds, so two workflows can agree on what a
"trace" is. Such a package has no graph, so it has no outermost graph, so its
`main.yaml` could only be a file written to satisfy a check.

**So the rule is presence, not existence:**

> **A package that carries `main.yaml` is runnable; one that does not is a
> library.** Absence is a statement, not a fault. A run has exactly one entry
> package and that package must carry one — and *that* is where the failure
> belongs, named against the run rather than against every directory.

Three things follow, and the third is a gap rather than a design.

**The predicate is the file itself, and deliberately not something computed.**
"A package that declares any `module: task` needs one" is the obvious narrowing
and it is wrong: a package shipping reusable closures for another package's graph
to instantiate is exactly as much a library as a kinds-only one, and that rule
would force it to ship the same meaningless file. A new key for the author to
state it would be a second declaration of a fact the filesystem already carries.
Presence costs nothing to compute and cannot drift.

**What is checkable per package is the file's contents, not its existence.** The
rule is that `main.yaml` *is* the outermost graph's entry, so a `main.yaml` that
declares no `module: task` is an entry to nothing and is rejected naming the
file. That is this sentence read literally, and it recovers most of what the
unconditional rule was buying: an author who meant to ship a runnable package and
misspelt the name still gets a library rather than an error, but one who wrote
the name and forgot the graph is told.

**The other half has no home today, and is not invented here.** Nothing can
currently ask "is this the run's entry": `spec_loader` is handed one package at a
time (`bootstrap.py:255`, `load_package(pkg, views)` per package), and the root
is chosen by *closure name* by whoever calls in — `cli/main.py:668,687` pass the
literal `"main"` to `build.root_task`. **`main.yaml`'s contents are read by
nothing today**: `spec_loader/package.py` references the name at `:184` for this
check and at `:237` to sort the file last in the scan, and nowhere else. §10
carries the question.

**Which sentence changes, and which does not.** The two names bind **what a task
package is**. They do not reach the loader: the scan and the assets lookup happen
inside the package, upstream of the seam, and what crosses is documents (§4.4).
So "the loader does not interpret a package's layout" is *still* true and is now
true structurally rather than by convention — while the weaker claim a reader
would take from it, that *nothing* in the system fixes a filename, is false and
is why this subsection exists.

**Everything else is still the package's own.** Nothing fixes how the rest of the
tree is arranged, how many objects a file holds, which directory a kind lives in,
or whether there is a directory per kind at all. A package with one file and a
package with two hundred are equally well formed.

The cost is one sentence long: a package cannot rename these two. That is a line
in whatever tool creates a package, and the alternative is inference over the
tree. Kubernetes takes the same trade with `kustomization.yaml`, and so does every
build tool with a well-known root file.

**Consequences worth stating, because they are what the rule buys:**

- Nothing in this repository needs updating to add, change, or retire a workflow.
- A package can be versioned, reviewed, and shipped on its own cadence.
- Two workflows that disagree about what a "trace" is do not have to be
  reconciled — they are different packages.
- The main repository's tests never depend on a workflow's correctness.

### 4.4 The loading pipeline: YAML → schema

A spec is a **YAML document**. Loading one is two steps, in this order:

```
   *.yaml  ──scan & discriminate──►  a document  ──validate──►  the spec, admitted
                                          │
                                          └── against the object's JSON Schema (§4.3)
```

**There is no third step, and there used to be.** Revisions 4–9 put a jsonnet
render in front of this, adopted for the reason
[Kustomize](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomize/)'s
overlays are: a workflow runs the same shape against different models and
hardware, and the alternative to templating is copy-paste.

**Measured, that templating was never used.** Counted over every non-comment line
of all 21 `.jsonnet` and `.libsonnet` sources in `examples/demo/**` and
`validator/general_specs/**`, the entire computation surface is three things: a
shared library of constants, string concatenation to build paths, and
default-if-absent. No arithmetic, no loops, no comprehensions, no overlays. Every
general spec uses exactly one construct, the same one —
`if std.objectHas(config, 'inputs') then config.inputs else ['any']`. A language
was being carried for three needs that do not require a language. **jsonnet is
deleted** (§7), and the three needs are rehomed below.

**The document a package supplies is what is validated.** That was the whole
point of the old ordering and it survives the deletion unchanged: the loader does
not read, audit, or constrain how a package produced its documents. Inline
definitions, a shared variable set, several objects in one file, whatever
directory tree the author likes — all of it is the package's business. Only the
result is checked.

The deletion makes that promise **structural rather than conventional.** A
package now delivers *parsed documents* across the seam, so the loader has no
parameter through which a path could arrive and cannot open a source file even by
accident. It was previously an ordering convention inside one function. Criterion
4 pins the new form.

#### What replaced `config`: three needs, two homes

The removed mechanism was a single variable, `std.extVar("config")`, through
which a template declared what a package could fill. With no computation there is
no `extVar` — and no template to fill — so the question is not *what replaces the
variable* but *where each of the three measured needs now lives*.

| The need | Where it lives now |
|---|---|
| **Shared constants** — one value written once, referenced from many documents | A **package-level variable set**, substituted by the package before it emits a document. **The values are supplied to the package, not declared inside one**, which is narrower than it sounds: measured across the 21 sources this replaced, every reference was to `config.package_root`, `config.outside` or `config.inputs` — run-level facts — and not one package declared a constant of its own. A `vars:` block would be a construct with no measured user (`engineer_principle.md` §2), so it is not built; if a package ever needs one, this is the row that changes |
| **Path concatenation** — a path built from a root and a leaf | The same variable set, plus `${TASK_PACKAGE_ASSERT_DIR}` for the assets root (§4.3) |
| **Default-if-absent** | `default` on the field, in the object's JSON Schema |

Two things about that table are load-bearing, and both are easy to get wrong.

**The substitution is the package's work, not the loader's.** It happens inside
the package, upstream of everything this repository validates, and what crosses
the seam has no variables left in it. So the enforcement point below still sees
the final document, exactly as it saw the rendered one.

**`default` *declares* a fallback; it does not apply one.** Verified: `jsonschema`
validates `{}` against `{"access": {"type": "string", "default": "read"}}` with
zero errors and leaves the document untouched, and `spec_loader` installs no
default-filling extension. A schema `default` is therefore where a reader learns
the fallback and where a reviewer sees it change — the value itself is supplied by
whichever model consumes the document. That is one fact with two writers, and it
is named here rather than left to be discovered by whoever first wonders why the
key is still absent after validation.

**The tier count drops from three to two**, which is what §2.3's third diagram
now shows. Tier ② was the jsonnet template — "the shape of one recurring kind,
with `config` as its declared fill interface". Nothing plays that part any more:
the *shape* of a kind is the JSON Schema at tier ①, with its `description`s and
its `default`s, and the *declaration* is the YAML at tier ③. A general spec
(§4.5) stops being "a template whose `config` is empty" and is simply a document
this repository ships. **No guarantee was lost in the collapse** — customisation
came from ② and safety from ①, and the measurement above is that ② was carrying
constants and string concatenation, which tier ③ can do for itself. **Something
else was lost and it is not a guarantee**: with no template artefact there is
nothing to *share* between packages, so two packages that want the same shape now
each write it. §2.3 states that cost where the tiers are drawn.

#### The schema is the only enforcement point

**Immutability is a schema property**, expressed the way JSON Schema already
expresses it:

| | |
|---|---|
| A field that must not change | `const`, or a tight `enum` |
| A field nobody may add | `additionalProperties: false` |
| What a field is for | `description` — which is also the field's documentation |

Verified against `jsonschema`, and re-verified over a hand-written YAML document
now that there is no render to go through: giving a `const` field another value
fails with `'reproducible' was expected`, and smuggling an undeclared field fails
with `Additional properties are not allowed ('secretly_added' was unexpected)`.
The probe is `scratch/ui-yaml-2026-08/w1/probe_enforcement_point.py`.

**The deletion of jsonnet strengthened this rather than weakening it**, and it is
worth saying why, because the obvious reading is the opposite — a layer was
removed, so surely something is less guarded:

- **The source format never enforced anything.** jsonnet has no `final`: there is
  no immutable-field marker in the language, `+` means "the right side wins", and
  hidden fields (`::`) are overridable too. Every way of sealing a field *inside*
  jsonnet was a mechanism the reader had to learn and none survived being called
  from a package that did not want to cooperate. YAML makes that plain rather than
  changing it — a data format has no sealing construct to be tempted by.
- **The schema is checked over the document the package delivers**, so it cannot
  be evaded by any amount of cleverness upstream of it. That was true of a
  rendered document and is true of an emitted one; §4.4's opening is the only
  thing that changed shape.
- **A reader gets a kind's shape from the schema**, with `description` on every
  field — and now that tier ② is gone, the schema is the *only* place that shape
  is written down, so the artefact matters more than it did.

### 4.5 General specs are documents this repository happens to ship

Some specs are workflow-independent: a JSON Schema validator, a
digest-recomputation check. These live in this repository, and **they are
ordinary documents** — same file format, same pipeline, same schemas as a task
package's. Until rev. 10 they were "the degenerate case of a template, one whose
`config` is empty"; with tier ② gone (§4.4) there is no template for them to be a
case of, and the description simplifies to nothing at all. Being uninteresting is
the point.

The uniformity is the point: the main repository does not get a private path for
its own specs.

They live in **their own directory**, separate from any task package, so the
distinction is visible in a directory listing rather than only in a registry.

### 4.6 Identity

Every runtime object is identified by a uuid, and the id types are mutually
incompatible so that a signature reading `list[HandoffId]` says what it wants.
`task_graph` spec §3.1 specifies this for `TaskId`, `AgentId`, and `HandoffId`;
a `ValidatorId` joins them on the same terms.

### 4.7 The agent node is coarse

**The system's agent is the large node, not the detail.** Whatever multi-agent
structure a backend organises internally — subagents, planners, critics — is
invisible to this system, which always considers the work handed to a backend to
have been done by one agent.

This is what makes the backend swappable. It is also what makes the *system's*
observability claims honest: the system reports that an agent ran and what it
touched, never what it thought.

### 4.8 Every *leaf* task has an agent, and an agent need not be an AI

**Every leaf task has exactly one agent. What an agent *is* is open**: an AI
harness, a human, or a program — an executable, some code, or a thread. `kind` on
the agent spec says which.

**A non-leaf has none, and rev. 9 was wrong to demand one.** A task contains a
task graph or it is a leaf that does the work itself (`closure` spec §2.1), so a
non-leaf's work *is* its subgraph and there is no executor to name. The previous
wording forced every author to invent a name for a thing that does not run, and
**that name was read by nothing** — three code reads, not an argument:

| Claim | Evidence |
|---|---|
| A non-leaf never deploys its agent spec | `agent/runner.py:682` returns before `_deploy`, so `env_mgr/prepare.py:447`'s `material.deploy(agent_spec, zone)` is never reached for one |
| Its zone stages nothing | `_place_container_zone` "confines nothing, cuts no workspace, stages nothing" (`agent/runner.py:678,687`) |
| Output validation is already safe without one | `producer=None` falls through to the GLOBAL row (`validator/environment.py:140`) — the path exists and is exercised |

So dropping the requirement removes a field, deletes no behaviour, and needs no
new fallback. The rule is now enforced in the two places it was enforced before,
both narrowed the same way: `closure.schema.json` makes `agent` required unless
`task.subgraph` is non-empty, and `closure/check.py` check 4 reports a leaf that
names none, with the known agent specs listed
([`../closure/docs/spec.md`](../closure/docs/spec.md) §2.2, criterion 3).

A program agent is not a degenerate case to be tolerated; a large part of the
reference workflow is running a command someone already wrote, and wrapping that
in an *AI* would add cost and non-determinism for nothing. What it is wrapped in
instead is an agent spec of `kind: program`, which costs a name and a command
line and buys uniformity: one dispatch path, one binding record, one place the
system reports what ran.

So where there *is* an executor, the axis the system is decoupled along is **not
"agent or no agent"** — it is what an agent may be. Rev. 9 read that as "therefore
every task names one", which conflated two questions: whether a task has an
executor, and what an executor may be. Only the second was ever the interesting
one.

The system's interface is the handoff, so an executor that produces the right
handoffs is a valid executor whatever it is. That is §3.0 applied to the executor
itself: the system is decoupled from *what runs the work*, not merely from which
AI harness runs it.

---

## 5. Two authority boundaries

Everything else in this document is structure. These two are the load-bearing
constraints, and both exist because the alternative has been observed to fail.

### 5.1 The scheduler decides *when*, never *what*

The scheduler asks one question about each input — is the latest version valid? —
and does arithmetic on resource counters. It does not inspect content, does not
judge quality, cannot advance a handoff, and does not know validators exist.

**And it owns almost nothing.** Reading the implementation: `Scheduler` holds a
`pools` dict, a lock, and two re-entrancy flags. That is all. The pools are a
*derived index* over task status — rebuilt from `TaskMgr` on recovery — and the
scheduler **persists nothing of its own**. Task state lives in `TaskMgr`, handoff
state in `HandoffMgr`, balances in the consumable pools.

So it is a decision procedure with a cache, not an owner of state. That is why
swapping the policy changes dispatch order and nothing else, and why recovery can
rebuild it from records it did not write.

**Reacting to a failure is nobody's job here either.** Any phase of a task can
fail, and every failure is the same thing: the task ends `FAILED` and stops being
scheduled. What to *do* about it — escalate to the user, restart an upstream
producer, cancel a branch, assign a helper — is a policy decision belonging to the
monitor of the graph the task is in (`task_graph` spec §3.5), not to the
scheduler (validator spec §3.4, `task_graph` spec §6.3).

**And the monitor acts through the task, not on it.** A task owns its own
transitions, and a transition is the only thing that triggers the scheduler
(`task_graph` spec §3.2.3). So the monitor calls `cancel()` or `restart()` and
the task decides what follows; it never assigns a status. That is what lets the
system gain cascading cancel — which it now has, within a graph
(`task_graph` spec §3.2.4) — without the scheduler acquiring an opinion about
*what*. Cascading **invalidation** remains out of scope on the same reasoning:
deciding that derived content is bad is a judgement about content, and only a
validator makes those.

Specified in `task_graph` spec §2 principle 4 and §3.1. **Enforced mechanically**
by `agent_sys/tests/task_graph/test_authority.py`, which drives a full
submit → dispatch → complete → resume cycle against a spy and asserts that every
handoff write originates inside an agent span. That test is the reason this
boundary can be relied on rather than merely intended.

### 5.2 The producer cannot grade its own output

**A validator runs in a context the producing agent cannot reach**, and the
separation is enforced by hook rather than by convention.

Three things follow, and each of them is a rule elsewhere in this spec set:

| Rule | Where |
|---|---|
| The checking logic is externally supplied, never written by the agent under test | validator spec §5 |
| The checking standard is externally supplied, and the agent may not see it | validator spec §5, agent spec §3 |
| The verdict is recorded by the validator, not reported by the producer | handoff spec §3, validator spec §4 |

**"Producer" here means the agent, not the task.** `task_graph` spec §3.1 says
agents own handoff state and the producing task records the verdict when it
finishes — which reads like a contradiction and is not. A task is one producer
from outside and three isolated phases from inside: the main phase writes the
content, the output validation phase records the verdict, and the second cannot
be reached by the first. The task answers for its output; no agent grades what it
wrote.

The kickoff appendix records what happens without them, at three levels of
severity: a correctness gate satisfied by a regex over the agent's own prose; a
tolerance table permitting 10% element mismatch and still passing; and an agent
that, told not to repeat a proposal with identical parameters, added a `tag`
field so the fingerprint differed and queued fourteen identical baselines in
thirteen minutes.

The last one is the general lesson and is worth stating as a rule of its own:

> **An agent will comply literally.** A rule that exists only in a prompt is a
> rule the agent can satisfy without honouring. Anything that can be enforced in
> code is enforced in code; the agent's job is to fill blanks in a fixed
> procedure, not to invent the procedure.

---

## 6. Record and replay is the v1 scope

**Task graphs are statically defined.** A task's internal subgraph is declared in
its spec, not produced at runtime; there are no dynamic task specs and therefore
no dynamic handoff specs.

The reason is the target: phase one automates a workflow that is already fixed.
A system that can express a fixed workflow exactly is more useful there than one
that can express any workflow approximately.

What this forecloses, stated plainly so nobody discovers it later:

- A task cannot decide at runtime that it needs a step nobody declared. It can
  report that it does — that is what the risk exit (§3.1 principle 6) is for — but
  it cannot add one.
- A graph cannot be generated from a natural-language goal.
- A task's **own declared expansion** is fixed. What it contains is in its spec.

The system is nonetheless not frozen: a closure can be added, a validator can be
added to a handoff kind, and an agent spec can be swapped, all without touching
the engine. The static constraint is on *what may exist*, not on how many of each
there are.

### 6.1 The catalogue is static; the instance count need not be

An earlier revision said fan-out over a runtime-discovered set "is not
expressible as N tasks". **That was too strong**, and it contradicted
`task_graph` spec §3.2.2, which states that the graph may grow while a task's own
expansion stays fixed — an agent is permitted to *submit* tasks (§2 principle 4
forbids only redirecting the graph).

The accurate constraint:

> **The catalogue is static. The instance count is not.** Runtime fan-out
> instantiates a *declared* closure N times with different inputs. It never
> invents a task.

That is consistent with record-and-replay — a replay reads the same catalogue —
and it is the same question `closure/docs/spec.md` §6 calls "parameterised
closures". The two were one question and are now one, homed in §10 of this
document.

It matters because the reference workflow's fourth step is exactly this shape:
take the top-k operators from a trace analysis and optimise each, where k is
discovered. Declaring it foreclosed would foreclose the system's most valuable
step.

**It is not an alpha capability yet**, and four things must be decided first —
§10 carries them.

---

## 7. Build versus adopt

The task definition requires researching whether a mature solution exists before
building, and recording the outcome. The system-level summary is here; per-module
reasoning goes in each module's `README.md` at code stage, which is where the
task definition asks for it.

| Need | Adopted | Why |
|---|---|---|
| Domain models, validation, serialisation | **pydantic v2** | Already installed via `fastapi`, so it costs nothing. `model_dump` / `model_validate` remove hand-written deserialisers that would drift on every field added |
| YAML files elsewhere in the system | **PyYAML** | Already an `agent_sys` dependency. **It no longer parses a package document, and the row is kept because the reason is instructive.** `spec_loader/validate.py` used to `safe_load` and argue that "neither the YAML 1.1 `norway: NO` trap nor the duplicate-key trap can reach us — jsonnet quotes every string and rejects a duplicate field statically". Deleting jsonnet removed that upstream and made both traps live; the row below closed them by parser choice, and `validate` now takes a parsed document and does not parse at all. PyYAML remains in use where nothing is hand-authored against a schema — `env_mgr/recipe.py`, `handoff/verdict.py`, `handoff/store.py` |
| **The parse of a package document** | **`ruamel.yaml`, round-trip mode** | A diagnostic that cannot say *which line* sends an author to grep a package. Round-trip mode keeps `lc.line`, `lc.key(k)`, `lc.value(k)` and `lc.item(i)` on the parsed node and reports a syntax error as `MarkedYAMLError.problem_mark`, both 0-based; PyYAML exposes no equivalent. **It also settles the two traps the row above raises, and that was not why it was chosen** — round-trip is **YAML 1.2**, so `NO` is the string `NO`, and duplicate keys raise `DuplicateKeyError` with a position where `safe_load` silently keeps the last. `CommentedMap` subclasses `dict`, so `jsonschema` validates the position-carrying tree directly and `json_path` is correct. **Exactly one parser may touch a package document**: the two disagree on real values (`12:30` is `'12:30'` or `750`), so a second reader is how one document comes to mean two things |
| Spec schema constraint | **jsonschema** | Already installed. The task definition requires the YAML be schema-constrained, and JSON Schema is the standard for that — and §4.4 makes it the system's only enforcement point, so `const` and `additionalProperties` carry real weight |
| Identity | **`uuid.UUID` subclasses** | Generation, comparison, and formatting are solved in the standard library. Subclassing keeps the id types mutually incompatible |
| Agent backend | **claude-agent-sdk** | Satisfies every level-2 capability — history, interrupt, instruct, hooks, permission callback, sessions. Mapping in agent spec §5, verified against the SDK reference |
| **Isolation** | **bubblewrap, else Landlock** | §7.1. Both are what the surveyed harnesses use, and the measured alternative does not work |
| Test runner | **pytest** | Already a dev dependency |

| Need | Rejected | Why |
|---|---|---|
| Scheduling | Prefect, Hatchet, Temporal, Ray, Airflow, Slurm | Each is a platform whose scheduling core is not separable; adopting a server to obtain one primitive. Recorded in full in `task_graph` spec §9 |
| Graph algorithms | networkx, `graphlib` | The only graph question asked at dispatch time is whether one task's inputs are valid. `graphlib.TopologicalSorter` additionally refuses nodes after `prepare()`, and this graph grows at runtime |
| Plugin framework | pluggy | Solves 1-to-N hook broadcast with ordering and wrappers. A validator is a 1-to-1 call and needs none of it. A decorator writing to a dict is the whole requirement; setuptools entry points come later, if validators ever ship out of tree |
| **Spec templating** | **jsonnet** — adopted at rev. 4, **removed at rev. 10** | §4.4. Not rejected on principle: it was adopted, shipped, and then measured. Across every non-comment line of all 21 `.jsonnet` / `.libsonnet` sources in the tree, the whole computation surface is constants, string concatenation, and default-if-absent — and every general spec uses one construct, `if std.objectHas(config, 'inputs') then config.inputs else ['any']`. No arithmetic, no loops, no comprehensions, no overlays. A runtime dependency with a compiled extension and a fallback binding (`jsonnet` had no aarch64 wheel, hence `rjsonnet`) was buying three things a variable set and a schema `default` do for nothing. **Recorded here rather than quietly dropped, because §3.2's rule cuts both ways**: research before building, and re-measure before continuing to carry |
| Templating | Jinja2 | **Text** templating: it renders strings and can emit a document that is not valid YAML at all. That was the reason at rev. 4 and it still holds — with the template layer itself now gone (§4.4), nothing is looking for a replacement |
| Config overlays | **Kustomize** | Rev. 4 rejected the *patch model* as the wrong shape. That reasoning is superseded: the real blocker is that Kustomize is not adoptable from Python at all. Its embeddable API is the Go package `krusty`, and there is no Python binding — the three routes are cgo, a subprocess, or a Go sidecar, all to obtain a patch engine this system does not need. **One idea from it is still wanted and is not adopted here**: `LoadRestrictions`, which refuses to load anything outside the package root. That is `spec_loader` design **O3** and is open |

### 7.1 Two decisions research reversed

Both had been written down and both looked reasonable. Recording them because
§3.2 is otherwise an untested rule.

**The validator template system.** The spec had a template validator with
declared blanks, composing other validators recursively to a configured depth. A
survey of sixteen evaluation, data-quality, and policy-as-code systems — Inspect
AI, DeepEval, Ragas, LangSmith, OpenAI Graders, promptfoo, HELM, Great
Expectations, Pandera, dbt, Soda, Deequ, OPA/Rego, Conftest, Gatekeeper, and
Dagster asset checks — found **none** that templates source with blanks. All use one small callable,
parameterised instances from a `{name, args}` table, and a name→factory registry.
Two prohibit the nesting variant outright. **Removed** — validator spec §6.

> At rev. 4 this had to be reconciled with §4.4 adopting jsonnet, because the two
> read alike: the removed system templated a validator's **checking logic** — its
> source, with blanks an agent filled in — while §4.4 templated a spec's
> **configuration**, and filling in a parameter is not writing code. **The
> reconciliation is no longer needed**, since rev. 10 removed the spec templating
> too (§7) — for a different and weaker reason, that nothing was using it. The
> distinction is kept on the record anyway: it is the one that decides whether a
> future proposal to template something is this rule or that one.

**Path-prefix isolation.** The spec confined an agent with a `PreToolUse` hook
matching an unguessable path prefix. Measured: an agent that writes a Python
script and runs it defeats the hook entirely — the hook sees `python3 x.py` and
no path at all — and prefix matching is CVE-2025-54794 in Claude Code itself,
CVSS 9.1, three defeats reproduced. An unguessable prefix is
security-by-obscurity, since the confined agent holds the value. **Replaced** by
canonical containment plus an OS sandbox — `env_mgr` spec §4.

Adopted as *design* rather than as a dependency: RCPSP terminology and its two
waiting sets; the parallel schedule generation scheme; the A2A task-state
vocabulary; reserve-then-settle for consumable resources; "the engine owns
routing"; Cursor's Agent/Run split (agent spec §4.3); Inspect AI's
`multi_scorer` + pluggable reducer as the composition primitive (validator
spec §6.2); and the three-stage anti-hallucination pattern — code computes every
number into JSON, the agent only renders, a checker verifies the report copied
those fields verbatim.

---

## 8. Acceptance criteria

System-level. Each component spec carries its own; §9 indexes them.

1. **The four objects are uniform.** Each of handoff, validator, task, and agent
   has a static spec kind, a uuid-identified runtime object, a runtime manager, a
   spec registry, and a JSON Schema — demonstrated by instantiating one of each
   from a spec on disk.
2. **A spec that violates its schema is rejected at load time**, with the file
   path and the offending field in the message. It does not fail later, at use.
3. **The schema is the only enforcement point.** A `const` field given another
   value is rejected, and an undeclared field smuggled in is rejected — whatever
   the package did to produce the document (§4.4). *Amended at rev. 10: it said
   "a template's `const` field overridden after rendering … whatever the jsonnet
   did". There is no template and no render, and the property being pinned never
   depended on either — it is that the check runs over the delivered document.*
4. **The loader never sees a package's source.** It is handed parsed documents
   and has no parameter through which a path could arrive, so two packages whose
   files are organised wildly differently — one document per file in a directory
   per kind, or two hundred objects inline across a handful of files — are
   indistinguishable to it and both load. *Amended at rev. 10: it said "renders
   before it validates", which no longer happens. The rest is strengthened rather
   than relaxed — this was an ordering convention inside `load_package` and is now
   a type boundary (§4.4).*
5. **This repository contains no workflow-specific spec.** Every handoff kind,
   validator, task, agent, and closure here is either a schema, or a general spec
   (§4.5), or the demo package's (demo spec §1.1). *Amended at rev. 10: "a general
   spec whose `config` is empty" — there is no `config`.*
6. **A task package resolves without the loader knowing its layout**: a package
   whose validator symlinks point into a second package loads, and a dangling
   symlink fails naming the path. *Amended at rev. 10 by having its scope made
   explicit rather than by changing: the two names §4.3 now fixes are the
   package's, not the loader's, so this criterion survives — but a test that
   demonstrated it by reaching into the loader would no longer be demonstrating
   it, since the loader is handed documents (criterion 4). What a package insists
   on is criterion 16.*
7. **The producing agent cannot reach the validation's context.** A spy records
   every context read, and no read originating in a producer frame reaches a
   validation's checking standard. The denial is by OS confinement, not by
   convention — `env_mgr` spec §4.
8. **Every handoff kind a loaded package declares resolves to at least one
   validator**, unless the escape-hatch flag is set — and setting it is reported,
   not silent.
9. **A closure is well-formed or it is rejected**: every handoff kind its task
   names resolves, every handoff has a validator, **a leaf names an agent spec
   and it exists**, a non-leaf's is optional and still must exist if named, and
   the task's permissions cover its handoffs. *Amended at rev. 10: "its agent spec
   exists if one is named" was silent on whether one had to be named, so §4.8's
   narrowing would have left it meaning the same thing before and after — which
   is the failure mode a criterion is supposed to prevent.*
10. **The scheduler still never writes handoff state, and never sees a
   validation** — `test_authority.py` passes unchanged with validation phases,
   subgraphs, and a real backend present.
11. **The six-step reference workflow is expressible.** The kickoff report's main
   loop — prepare e2e, collect, analyse, optimise, integrate, verify — is
   declared as closures and runs as a graph, with each step's handoffs and
   validators named.
12. **An agent's work is reconstructible after a restart** from its history and
   its workspace, without the agent's cooperation. The playground may help and is
   not depended on (`env_mgr` spec §6.2).
13. **Swapping the agent backend changes no other component.** Demonstrated by
   running the demo graph with the `claude-agent-sdk` backend and with a program
   executor, and observing identical handoff state.
14. **No isolation, no start.** With neither `bwrap` nor Landlock available, a
    task refuses to start rather than running unconfined, and says so.
15. **A scripted bypass is blocked.** A subprocess opens a file outside its zone
    and fails at both the read and the write, while the same script succeeds
    inside the zone.
16. **`assets/` is required of every package, and it is the whole of the layout
    a package must have** (§4.3). A package missing `assets/` fails naming the
    root; a package that declares every object in a single file and one that
    gives each its own both load, with no directory-per-kind anywhere. Separate
    from criterion 6 because the two test opposite things: 6 is what may vary,
    this is what may not. *Amended at rev. 11: as written at rev. 10 this also
    demanded `main.yaml` of every package, which defeated the reason §4.3 gives
    for the name — a run over several packages would hold several files each
    claiming to declare the outermost graph — and made a kinds-only library
    package inexpressible, three paragraphs after §4.3 permits one. The
    `main.yaml` half is criterion 18. **Nothing was deleted**: the rule was
    split, and the half that is genuinely per-package is stated here unchanged.*
17. **No source format survives the deletion** (new at rev. 10). No `.jsonnet` or
    `.libsonnet` file remains in the tree, nothing imports `_jsonnet` or
    `rjsonnet`, and neither is a declared dependency. Stated as a criterion rather
    than left to a grep because a deletion that is 95% done is the state in which
    a second format quietly comes back.
18. **`main.yaml` states that a package is runnable, and says what it is**
    (§4.3, new at rev. 11). A package with no `main.yaml` loads, and its
    documents are admitted — a package shipping only shared handoff kinds is a
    library, and its absence of a graph is a statement rather than a fault. A
    `main.yaml` that is present but declares no `module: task` is rejected naming
    the file, because a file whose whole definition is "the outermost graph's
    entry" cannot be an entry to nothing. **What is not yet demonstrable is the
    other half** — that a run has exactly one entry package and that package
    carries one. That check has no owner today and §10 carries it; a criterion
    asserting it now would be untestable, which is worse than one openly
    deferred.
**Criteria 14 and 15 are CI-enforced**, in `tests/env_mgr` on every commit —
neither needs a model, and both are the properties the system's safety claim
rests on. `env_mgr` spec §10 says which criteria that covers and what happens on
a runner with no sandbox mechanism. The demo *additionally* shows criterion 15
happening to a real agent (demo spec §5).

---

## 9. Index of the spec set

Nine documents, each with its own numbered acceptance criteria. The 18 above are
system-level and are demonstrated *across* components; these are demonstrated
within one.

| Document | Rev. | Criteria | Its own open questions |
|---|---|---|---|
| **this document** | 12 | §8 — 18 | §10 |
| [`handoff`](../handoff/docs/spec.md) | 4 | §9 — 17 | §10 |
| [`validator`](../validator/docs/spec.md) | 6 | §11 — 21 | §12 |
| [`task_graph`](../task_graph/docs/spec.md) | 12 | §11 — 54 | §10 |
| [`agent`](../agent/docs/spec.md) | 4 | §8 — 16 | §9 |
| [`closure`](../closure/docs/spec.md) | 10 | §5 — 12 | §6 |
| [`env_mgr`](../env_mgr/docs/spec.md) | 3 | §10 — 22 | §11 |
| [`demo`](../cli/docs/spec.md) | 5 | §6 — 16 | §7 |

**176 criteria in total, and three facts about the set are worth stating:**

- **Rev. 10 was the first revision to amend criteria rather than only add them,
  and rev. 11 amended one of its own.** Rev. 10: this document's 3, 4, 5, 6 and 9
  changed wording and `closure`'s 3 changed meaning, with 16, 17 and `closure` 12
  added. Rev. 11: 16 was split, its `main.yaml` half becoming 18. Each says so
  inline and **none was deleted**. Flagged because until rev. 10 the set had the
  property in the next bullet, and a reader who assumes it still holds everywhere
  will trust a stale quotation.
  Flagged because until now the set had the property in the next bullet, and a
  reader who assumes it still holds everywhere will trust a stale quotation.

- **`task_graph`'s 54 are stratified by revision**: 1–35 at rev. 7 are
  **implemented and green** (423 tests); 36–44 (subgraphs and validation phases),
  45–52 (task-owned transitions, cascading cancel), and 53–54 (leaf-only
  acquisition) are specified and unbuilt. No earlier criterion was amended by a
  later revision.
- **`env_mgr`'s 2–14 and this document's 14–15 are CI-enforced** and are the only
  ones with that status today. They are the isolation properties, which is
  deliberate: they are what the system's safety claim rests on (§8).

---

## 10. Open questions

System-level only. Component-level questions live in each component's spec.

| Item | Status |
|---|---|
| **Scoring** | §3.1 principle 7 wants a scoring mechanism; the validator spec reserves the field. Nothing specifies how a score is produced, compared across runs, or aggregated to a task. v1 is boolean throughout, because a threshold set before run-to-run variance is measured is indistinguishable from noise |
| **The observer, and the monitor** | §3.1 principle 4 wants an outside view of whether an agent has drifted or is looping. It is not the scheduler's job (§5.1) and an agent cannot be trusted to report it of itself (§5.2). **The monitor moved into the alpha on 2026-08-27** — `task_graph` spec §3.5 specifies it, with its own mainloop. **On 2026-08-28 its job widened from the task's exceptions to the task's events**, on two channels: planned phase advances handled by code, and the unplanned outcomes this row is about (`monitor` spec §2.2). The alpha still ships a simple pusher, and the *analysing* dispatcher stays in [`ROADMAP.md`](ROADMAP.md) §2 — bound to the unplanned channel, so no model is ever on the ordinary path |
| **The control surface** | §3.1 principle 5 wants abort and instruct. The backend exposes both (agent spec §4.3); what drives them — a CLI, a panel, a queue — is not specified. The whole-system CLI ([`TODO.md`](TODO.md)) is where it will land |
| **Cross-closure knowledge accumulation** | Knowledge handoffs are specified, but nothing says how a good result *becomes* one, or who decides. "Excellent work feeds back into the few-shot examples" is a goal, not a mechanism |
| **Runtime fan-out** | §6.1 settles the framing — the catalogue is static, the instance count is not — and merges `closure` spec's "parameterised closures" into this row. Four things are undecided: **who submits** (the executing agent, permitted today, or the monitor, which is roadmap); **`is_end` accounting**, which breaks if N siblings appear after a statically declared end entry subtask, so "has this subgraph finished" can report finished while children run; **parentage**, since a runtime-submitted task needs a parent for its storage to nest; and **what bounds N**, because 200 tasks against an 8-GPU pool is fine for the scheduler and probably not for the operator |
| **A failed branch is reported, but the alpha cannot repair it** | ~~The branch stops and nothing surfaces an error.~~ **Withdrawn 2026-08-27.** `monitor` spec §2 principle 1 makes every departure from the plan the monitor's, and a task that fails a validation is terminal — its dependents will never run and the graph will not finish, whether or not any component malfunctioned. So the failure **is reported and recorded** (`monitor` spec §2.1); it does not go quiescent. What remains a limitation is only the *response*: the alpha's pusher has no push for a terminal task with no agent running, so deciding what to do about a dead branch — retry with more knowledge, reassign, escalate to a human — waits for the analysing dispatcher ([`ROADMAP.md`](ROADMAP.md) §2.3). **The ceiling is on the reaction, never on the reporting.** `demo` criterion 5 still requires demonstrating a validation failure |
| **Which package a run starts from** | §4.3 settles the per-package half — `main.yaml` present means runnable, absent means library — and leaves the per-run half unowned. A run takes `packages: Sequence[Any]` (`task_graph/bootstrap.py:47`) and loads each into one shared set of registries (`:255`); nothing designates one as the entry, and nothing reads `main.yaml`'s contents at all. Today the root is chosen by *closure name* by the caller — `cli/main.py:668,687` pass the literal `"main"` to `build.root_task`. So three things are undecided: **who selects the entry package**, **what happens when two of the packages carry `main.yaml`** (legal, since each may be runnable alone) and when none does, and **whether the closure-name route survives** or is replaced by reading the entry package's `main.yaml`. The seam is `task_graph/bootstrap.py`'s `packages=` parameter on one side and `spec_loader`'s `TaskPackage` on the other; neither can answer it alone, which is why it is here and not in either module |
| **Multi-graph concurrency** | One system whole task is specified. Two running at once against the same handoff storage is not. Nested per-task storage (`env_mgr` spec §5.1) handles most of it; two runs of the *same* task is the case it does not |
| **Where agent work quality is scored** | Distinct from scoring a handoff: the task definition wants agent work quality quantified over time. o11y records the metrics ([`ROADMAP.md`](ROADMAP.md) §1); nothing turns them into a judgement |
| **The isolation ceiling** | `env_mgr` spec §4.6 states it plainly: a process sandbox is necessary and not inviolable, and for genuinely untrusted input the answer is a VM per task. The alpha runs trusted-but-fallible agents, so this is a threat-model judgement — and it should be revisited if that changes |
| **Standards for admitting parts to `agent_sys`** | There should be a rule, a checker, and a review process for adding to the system, with each module carrying hard and soft quality standards. It is the meta-item that would make several other open questions enforceable. [`ROADMAP.md`](ROADMAP.md) §8 |
