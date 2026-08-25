# Agent Work System — Specification

| | |
|---|---|
| Status | Draft, pending review |
| Revision | 1 — 2026-08-24 |
| Date | 2026-08-24 |
| Scope | The whole system: what it is, which components exist, and what each owes the others |
| Source | The task definition; the Infera × Hyperloom kickoff report (v1.0, 2026-08) and its research appendix |

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

The research appendix behind this system (kickoff report §5A) is a teardown of an
existing implementation that got this backwards. Its finding, in one line:

> The absence of cheap gates pushed the entire verification burden onto the most
> expensive gate, so dead ends and live ones cost the same.

Its most-cited concrete failure is the validator problem this system exists to
solve: correctness was decided by grepping the agent's own report for the string
`"correctness passed"`, against a harness the agent wrote, using a reference
implementation the agent also wrote. **The candidate wrote the exam, the answer
key, and graded it.** §5.2 of this document is the structural answer.

### 1.2 In scope

- The **handoff**: what a unit of transfer must carry and how it is versioned.
- The **validator**: what makes a handoff checkable, and how much a given check
  can be trusted.
- The **task graph**: which task runs when, including subgraph nesting.
- The **agent**: what an executor must declare, and the backend abstraction that
  keeps the system independent of any one harness.
- The **closure**: the predefined binding of the four.
- The **environment**: all interaction with the operating system — storage,
  workspaces, permission zones, local↔remote mapping.

### 1.3 Out of scope

- **What an agent does internally.** If a backend organises its own multi-agent
  structure, the system does not see it and does not manage it (§4.4).
- **The runtime environment of the program under test.** sglang, vllm, and infera
  environments and their before/after consistency belong to the handoffs and
  validators that name them, not to `env_mgr` (see `../env_mgr/docs/spec.md` §7).
- **Frameworking the test code a validator runs.** The validator system specifies
  where a check plugs in, not how the check's own code is organised; that belongs
  to the owning external test system (see `../validator/docs/spec.md` §6).
- **Dynamic task graphs.** §6 states the record-and-replay scope and its cost.

---

## 2. Design principles

Eight principles, adopted from the kickoff report §2 and restated here as
system-level constraints. They are listed in the order in which they decide
arguments.

| # | Principle | Consequence |
|---|---|---|
| 1 | **Reproducible or it did not happen** | Any deliverable, conclusion, or performance number rests on a reproducible basis. A result that cannot be re-obtained is not a result |
| 2 | **Develop against the validator** | The goal is not "the output looks right" but "the output passes its validator". The validator is specified first and the implementation second |
| 3 | **`<context, worker>`** | A worker is `<executor, knowledge, rules>`; a context is `<content, protocol, validation programs>`. Every part has a clear, complete, checkable input and output, and does only its own job |
| 4 | **Observability** | Levelled logging, an outside observer that can see whether an agent has drifted from its goal or entered a loop, and a final working-process summary. A system that can be observed can be repaired |
| 5 | **Interventionability** | An agent can be interrupted and appended to; a control surface can abort |
| 6 | **Risk has an exit** | What an agent cannot decide, it reports. Some reports may not be self-issued, which is why §5.2 exists |
| 7 | **Measurable** | A scoring mechanism exists. Where two results are equal, per-token and per-time efficiency break the tie |
| 8 | **Composable and pluggable** | The optimisation flow does not change, so each new external tool is integrated through a thin wrapper against a standard interface |

Three further rules are inherited from `task_graph` and apply system-wide because
they turned out not to be local:

| # | Principle | Consequence |
|---|---|---|
| 9 | **Composition over inheritance** | Inheritance appears only where two things genuinely differ in behaviour. Everything else is a `Protocol` resolved by name |
| 10 | **One fact, one place** | Where two operations mean the same thing, one is expressed in terms of the other. This is not licence to collapse two genuinely different concerns |
| 11 | **Simplicity is a requirement** | Where a mature solution exists, use it (§7). Where none fits, the implementation stays small enough to read in one sitting |

---

## 3. Components

Seven components. Each has its own specification; this table is the map.

| Component | Owns | Specification |
|---|---|---|
| `handoff` | What a unit of transfer carries: schema, digest, scope tags, validator list | [`../handoff/docs/spec.md`](../handoff/docs/spec.md) |
| `validator` | What makes a handoff checkable, and how far a check can be trusted | [`../validator/docs/spec.md`](../validator/docs/spec.md) |
| `task_graph` | Which task runs when. Nothing else — it never inspects what a task does | [`../task_graph/docs/spec.md`](../task_graph/docs/spec.md) |
| `agent` | What an executor declares, and the backend abstraction | [`../agent/docs/spec.md`](../agent/docs/spec.md) |
| `closure` | The predefined binding of the four objects | [`../closure/docs/spec.md`](../closure/docs/spec.md) |
| `env_mgr` | All interaction with the operating system | [`../env_mgr/docs/spec.md`](../env_mgr/docs/spec.md) |
| `demo` | The runnable proof that the above compose | [`../demo/docs/spec.md`](../demo/docs/spec.md) |

`task_graph` and `env_mgr` are implemented; the rest are specified here and built
in later stages.

### 3.1 How they fit

```
                        closure
         the predefined <handoff set, task, agent, validators>
                            │  looked up, never inferred
                            ▼
   ┌────────────────────────────────────────────────────────┐
   │                      task_graph                        │
   │      decides WHEN a task runs. Never WHAT it does.     │
   └───────┬─────────────────────────────────┬──────────────┘
           │ dispatches                      │ asks one question:
           ▼                                 │ is this input's latest
        agent  ──── produces ────►  handoff ─┘ version VALID?
           │                          ▲
           │                          │ writes the verdict
           │                     validator
           │            a single-node task, run like any other,
           │            in a context the producer cannot reach
           ▼
        env_mgr
   workspace · playground · handoff storage · permission zones
```

Two arrows carry most of the design:

- **`task_graph → handoff` is read-only.** The scheduler asks whether a handoff's
  latest version is valid and never writes one. This is mechanically enforced
  today (`task_graph` spec §3.1, criterion 14).
- **`validator → handoff` is the only write path for a verdict**, and it runs in
  a context the producing agent cannot reach (§5.2).

---

## 4. The four objects

Brief §9 asks for a uniform treatment of four objects. It is honoured literally:

**Each of `handoff`, `validator`, `task`, and `agent` has five things:**

| | |
|---|---|
| A **static spec** | A YAML file with a schema constraint |
| A **runtime object** | Identified by a uuid, distinct per run |
| A **runtime manager** | Owns the collection of live objects |
| A **spec registry** | Name → spec. Knows what kinds exist |
| A **predefined-spec folder** | Where the YAML files live, on disk |

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

### 4.3 Identity

Every runtime object is identified by a uuid, and the id types are mutually
incompatible so that a signature reading `list[HandoffId]` says what it wants.
`task_graph` spec §3.1 specifies this for `TaskId`, `AgentId`, and `HandoffId`;
a `ValidatorId` joins them on the same terms.

### 4.4 The agent node is coarse

**The system's agent is the large node, not the detail.** Whatever multi-agent
structure a backend organises internally — subagents, planners, critics — is
invisible to this system, which always considers the work handed to a backend to
have been done by one agent.

This is what makes the backend swappable. It is also what makes the *system's*
observability claims honest: the system reports that an agent ran and what it
touched, never what it thought.

---

## 5. Two authority boundaries

Everything else in this document is structure. These two are the load-bearing
constraints, and both exist because the alternative has been observed to fail.

### 5.1 The scheduler decides *when*, never *what*

The scheduler owns task state and never writes handoff state. It asks one
question about each input — is the latest version valid? — and does arithmetic on
resource counters. It does not inspect content, does not judge quality, and
cannot advance a handoff.

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
  report that it does — that is what the risk exit (§2 principle 6) is for — but
  it cannot add one.
- A graph cannot be generated from a natural-language goal.
- Fan-out over a set discovered at runtime is not expressible as N tasks. It is
  expressible as one task that does N things internally, which is a different
  thing with different observability.

The system is nonetheless not frozen: a closure can be added, a validator can be
added to a handoff kind, and an agent spec can be swapped, all without touching
the engine. The static constraint is on *graph shape at runtime*, not on the
catalogue.

---

## 7. Build versus adopt

The task definition requires researching whether a mature solution exists before
building, and recording the outcome. The system-level summary is here; per-module
reasoning goes in each module's `README.md` at code stage, which is where the
task definition asks for it.

| Need | Adopted | Why |
|---|---|---|
| Domain models, validation, serialisation | **pydantic v2** | Already installed via `fastapi`, so it costs nothing. `model_dump` / `model_validate` remove hand-written deserialisers that would drift on every field added |
| YAML spec files | **PyYAML** | Already an `agent_sys` dependency; the format the task definition asks for |
| Spec schema constraint | **jsonschema** | Already installed. The task definition requires the YAML be schema-constrained; JSON Schema is the standard for that and its `$ref` support is what makes template composition expressible |
| Identity | **`uuid.UUID` subclasses** | Generation, comparison, and formatting are solved in the standard library. Subclassing keeps the four id types mutually incompatible |
| Agent backend | **claude-agent-sdk** | Satisfies every capability the backend abstraction requires — history, interrupt, message-queue append, hooks, permission callback. See agent spec §5 for the mapping, verified against the SDK reference |
| Test runner | **pytest** | Already a dev dependency |

| Need | Rejected | Why |
|---|---|---|
| Scheduling | Prefect, Hatchet, Temporal, Ray, Airflow, Slurm | Each is a platform whose scheduling core is not separable; adopting a server to obtain one primitive. Recorded in full in `task_graph` spec §9 |
| Graph algorithms | networkx, `graphlib` | The only graph question asked at dispatch time is whether one task's inputs are valid. `graphlib.TopologicalSorter` additionally refuses nodes after `prepare()`, and this graph grows at runtime |
| Templating for validator blanks | Jinja2 | Deferred, not rejected: whether a blank is filled by text substitution or by a typed config value is a design-stage question. The spec constrains *what* a blank is, not how it is filled |

Adopted as *design* rather than as a dependency: RCPSP terminology and its two
waiting sets; the parallel schedule generation scheme; the A2A task-state
vocabulary; reserve-then-settle for consumable resources; "the engine owns
routing"; and — from the kickoff appendix — the three-stage anti-hallucination
pattern (code computes every number into JSON; the agent only renders; a checker
verifies the agent's report copied those fields verbatim), which is the most
directly reusable pattern the survey found.

---

## 8. Acceptance criteria

System-level. Each component spec carries its own; §9 indexes them.

1. **The four objects are uniform.** Each of handoff, validator, task, and agent
   has a static spec kind, a uuid-identified runtime object, a runtime manager, a
   spec registry, and a predefined-spec folder — demonstrated by instantiating
   one of each from a spec on disk.
2. **A spec that violates its schema is rejected at load time**, with the file
   path and the offending field in the message. It does not fail later, at use.
3. **The producing agent cannot reach the validator's context.** Demonstrated the
   way criterion 14 demonstrates the scheduler boundary: a spy records every
   context read, and no read originating in a producer frame reaches a
   validator's checking standard.
4. **Every handoff kind in the predefined folder resolves to at least one
   validator**, unless the escape-hatch flag is set — and setting it is reported,
   not silent.
5. **A closure is well-formed or it is rejected**: every handoff kind its task
   names resolves, every handoff has a validator, and its agent spec exists.
6. **The scheduler still never writes handoff state** after the whole system is
   assembled — `test_authority.py` passes unchanged with validators, subgraphs,
   and a real backend present.
7. **The six-step reference workflow is expressible.** The kickoff report's main
   loop — prepare e2e, collect, analyse, optimise, integrate, verify — is
   declared as closures and runs as a graph, with each step's handoffs and
   validators named.
8. **An agent's work is reconstructible after a restart** from its history, its
   playground, and its workspace, without the agent's cooperation.
9. **Swapping the agent backend changes no other component.** Demonstrated by
   running the demo graph with the `claude-agent-sdk` backend and with a program
   executor, and observing identical handoff state.

---

## 9. Index of component criteria

| Component | Criteria |
|---|---|
| `handoff` | [`../handoff/docs/spec.md`](../handoff/docs/spec.md) §10 |
| `validator` | [`../validator/docs/spec.md`](../validator/docs/spec.md) §10 |
| `task_graph` | [`../task_graph/docs/spec.md`](../task_graph/docs/spec.md) §11 — 42 criteria: 1–35 at rev. 7, 36–42 added at rev. 8 |
| `agent` | [`../agent/docs/spec.md`](../agent/docs/spec.md) §8 |
| `closure` | [`../closure/docs/spec.md`](../closure/docs/spec.md) §5 |
| `env_mgr` | [`../env_mgr/docs/spec.md`](../env_mgr/docs/spec.md) §10 |
| `demo` | [`../demo/docs/spec.md`](../demo/docs/spec.md) §5 |

---

## 10. Open questions

System-level only. Component-level questions live in each component's spec.

| Item | Status |
|---|---|
| **Scoring** | §2 principle 7 requires a scoring mechanism and the validator spec reserves the field, but nothing specifies how a score is produced, compared across runs, or aggregated to a task. v1 is boolean throughout. The kickoff report's own guidance — measure run-to-run variance first, then derive the threshold from the variance — is the starting point, not a specification |
| **The observer** | §2 principle 4 requires an outside view that can tell whether an agent has drifted from its goal or entered a loop. Where it lives is undecided: it is not the scheduler's job (§5.1), and an agent cannot be trusted to report it of itself (§5.2). A system-level task is the likely answer |
| **The control surface** | §2 principle 5 requires abort and injection. The agent backend exposes both (agent spec §5); what drives them — a CLI, a panel, a queue — is not specified |
| **Cross-closure knowledge accumulation** | Knowledge handoffs are specified (handoff spec §4) but nothing says how a good result becomes one, or who decides. "Excellent work products feed back into the few-shot examples" is a goal, not a mechanism |
| **Multi-graph concurrency** | One system whole task is specified. Two running at once against the same handoff storage and the same permission zones is not, and the isolation requirement is unclear |
| **Where the scoring of agent work products lives** | Related to but distinct from scoring a handoff: the task definition wants agent work quality quantified over time. No component currently owns it |
