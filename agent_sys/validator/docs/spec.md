# Validator — Specification

| | |
|---|---|
| Status | Draft, revised after review |
| Revision | 8 — 2026-08-27. **The user-interface brief.** A validator's checking logic is a **`readme.md`, plus an `entry.sh` when programmatic, plus its own `materials`** — the same body `closure` spec §2.6 gives a task, because §3 already says a validator is a special kind of task (§6.1). **The registered Python callable is withdrawn**: it cannot express a validator an agent is responsible for without a wrapper that runs an agent, so the callable becomes a layer that exists to be worked around. A code-shaped check uses the shipped pytest harness from its `entry.sh`. (rev. 7: 2026-08-27. **Addressing into content is an RFC 6901 JSON Pointer, not a jsonpath** (§4.1), following `handoff` spec §5.1 rev. 5, which carries the reason. Decided in the stage-three consistency pass. (rev. 6: 2026-08-26. Consistency pass across the spec set: output validation's admission is a leaf's lease or a non-leaf's absence of one (§3); the survey count is sixteen throughout (§6.3). (rev. 5: 2026-08-26. A failure binds at every strength; the label qualifies a pass (§5.4). One fails, all fail (§5.5). A validation environment is rebuilt, never reused (§8.2). Dagster asset checks added to the survey (§6.0). (rev. 4: Validators live in task packages; a symlink may cross packages (§9.1). Spec templating is configuration, not the removed source templating (§6.3). rev. 3: A failed phase is an ordinary task failure; reacting to it is the monitor's job (§3.4). rev. 2: Review of PR #132: validations are phases inside `TaskRunner`, invisible to the scheduler; three quality dimensions; `<purpose, inputs, implementation, result>`; the template-with-blanks system removed; simplified against prior art. rev. 1: initial)))) |
| Date | 2026-08-24 |
| Scope | What makes a handoff checkable, how far a check can be trusted, and how validators are organised |
| Source | The task definition §4, §5; a survey of evaluation and data-validation frameworks (§6) |
| Part of | [`../../docs/spec.md`](../../docs/spec.md) — the whole-system specification |
| Depends on | [`../../handoff/docs/spec.md`](../../handoff/docs/spec.md), [`../../task_graph/docs/spec.md`](../../task_graph/docs/spec.md) §3.2.1 |

---

## 1. Purpose

A handoff is only a contract if something checks it. **Validators are the sole
standard by which a handoff is judged.**

Our goal is narrow and worth stating plainly, because it decides every trade-off
below:

> Make each step's output checkable **cheaply, early, and by someone other than
> its producer** — so that a bad artefact is caught at the step that produced it,
> rather than at the end-to-end run that eventually fails because of it.

The domain is LLM inference performance optimisation, and it has two properties
that shape the design. **The expensive gate is very expensive** — an end-to-end
benchmark costs GPU-hours, so anything catchable earlier must be caught earlier.
And **the interesting failures are not crashes** — a trace that silently omits
half the kernels, an operator that is fast because it returns NaN, a
configuration that is valid and unrepresentative. Those pass every check that
only asks "did it run".

So the design puts three things first: a check runs in a context its producer
cannot reach (§8); the cheap checks run before the expensive ones (§2); and a
check declares honestly how much it actually proves (§5).

### 1.1 In scope

- What a validator is, and where it runs.
- The four elements: purpose, inputs, implementation, result.
- The three quality dimensions a check can address.
- The trust taxonomy: `strong` versus `weak`, and why the label must be honest.
- Reuse without copy-paste, and extension without a template engine.
- Folders, tags, and the registry.

### 1.2 Out of scope

- **Frameworking the test code itself.** §7.
- **How a handoff is versioned or stored** — handoff spec.
- **Scheduling.** The scheduler does not know validators exist (§3).

---

## 2. Design principles

| # | Principle | Consequence |
|---|---|---|
| 1 | **The producer never grades its own output** | Enforced by hook and by environment, not by convention. §8 |
| 2 | **Cheap gates before expensive ones** | A schema check that costs milliseconds runs before a benchmark that costs GPU-hours |
| 3 | **Anything that can be code is code** | The agent's role is to run a procedure someone else wrote, not to invent one |
| 4 | **The standard is external** | Both the checking logic and the criterion come from outside the thing being checked. §5 |
| 5 | **Label honestly** | A weak check labelled strong is worse than no check: it stops anyone looking further. §5.6 |
| 6 | **A failure binds at every strength** | The strength label qualifies a *pass*, never a failure. A failing `weak` validator fails the phase. §5.4 |
| 7 | **One small interface, many instances** | Reuse comes from parameterising one implementation, never from templating source. §6 |

---

## 3. A validator is a phase, not a graph node

**A validator is a special kind of task — and it is invisible from the scheduler
side.**

`task_graph` spec §3.2.1 specifies the arrangement: a task has three phases, and
`TaskRunner` runs all three for the one task the scheduler dispatched.

```
scheduler dispatches ──►  ┌────────────────────────────────┐
   ONE task               │ 1. input validations           │  ← the scheduler
                          │ 2. main / subgraph             │    sees none of
                          │ 3. output validations          │    these
                          └────────────────────────────────┘
```

| Phase | Which validators | Why here |
|---|---|---|
| `input_validation` | Checks over the task's inputs | Cheap rejection before the expensive work starts |
| `main` | None | — |
| `output_validation` | Checks over the task's outputs | Three things the producing task still has: its **admission** — a leaf still holds the lease it took, and a non-leaf never needed one (`task_graph` spec §6.2); its **resolved configuration**, so the check's environment is a rebuild rather than a discovery; and the **artefacts locally**. Downstream, all three are gone |

### 3.1 What a validation phase keeps from being a task

Invisible to the scheduler does not mean lightweight. A validation phase keeps
every other property that matters:

- **It gets a fresh, clean agent environment**, and may have an AI agent inside
  it. §8.2.
- **Its inputs are handoffs.** Same lookup by uuid, same versions.
- **It produces no output handoff.** It calls
  `handoff.update_validation_status(versioned_handoff, ...)` — persisted as a
  YAML record and excluded from the handoff's digest, so recording a verdict does
  not change the artefact's identity (handoff spec §5.2).
- **It has a life status**, which is why `TaskStatus` carries
  `INPUT_VALIDATING` and `OUTPUT_VALIDATING`.

### 3.2 Why not a graph node

The earlier revision made each validator a scheduler-visible task. It was wrong
in three ways, and they are worth recording because each is a real cost:

| | |
|---|---|
| **The scheduler would have to know what a validator is** | It would read validator specs and order validations — content-aware scheduling, which is the boundary the whole system rests on |
| **Every check would take a lease** | A task's environment would be torn down between the main work and its own output check, which is exactly what putting the check inside the task avoids |
| **The graph would be mostly checks** | A workflow of six real steps with three checks each is a 24-node graph, of which 18 nodes are not work |

The scheduler dispatches a task and gets a completion. What happened in between
is the runner's business.

### 3.3 Phases can be skipped

A validation phase may be skipped:

- **By config** — a phase declared empty, or switched off.
- **Because someone else already validated it.** A handoff carries its validation
  history (handoff spec §5.2); if the required validators already passed against
  this exact version, re-running them buys nothing.

A CLI switch — `--validation-strict-level` — controls how permissive this is. A
skip is **reported**, never silent.

### 3.4 A failed phase is just a failed task

**A failure in any phase is a task failure, and nothing more.** Input validation,
main, output validation — the task ends `FAILED`, releases its resources, and its
consumers stay in `WAITING_HANDOFF` because no output became valid
(`task_graph` spec §6.3). There is no separate failure kind for a validation
phase, and the scheduler does not distinguish them.

**Deciding what to do about it is not the scheduler's job.** That belongs to the
monitor of the graph the task is in ([`../../docs/ROADMAP.md`](../../docs/ROADMAP.md)
§2), which is the component that can analyse and choose: escalate to the user,
restart an upstream producer, add knowledge, assign a helper, or give up.

This is why the scheduler does not cascade. Cascading invalidation is a *policy*
about how to react to a failure, and putting policy in the scheduler would give
it an opinion about *what*, which is exactly the boundary the whole system rests
on (main spec §5.1). The scheduler's response to a failure is to stop scheduling;
everything past that is the monitor's.

The alpha's monitor is a simple pusher, so in practice a failed validation phase
leaves a stalled branch and a report. That is the honest alpha behaviour, not a
gap in this specification.

### 3.5 Two structural constraints

| Constraint | Why |
|---|---|
| **A validator is single-node.** No subtasks | A check that expands into a workflow needs its own checks. The recursion has to stop somewhere |
| **A validator's own input validation is empty** | Otherwise validating a validator's inputs needs a validator, without bound |

---

## 4. The four elements

Every validator is `<purpose, inputs, implementation, result>`.

| Element | What it is |
|---|---|
| **purpose** | A one-line `brief`: which dimension (§5.1) this checks, and what specifically. Written for a human deciding whether this validator is the one they want |
| **inputs** | The handoff kinds it consumes, **declared**. One or more |
| **implementation** | The check itself: code, or a procedure an agent runs |
| **result** | `dict[HandoffId, bool]` — a verdict per input handoff |

`purpose` is the element the earlier revision lacked, and it is the one a reader
needs first. A registry of forty validators whose names are `check_trace_v2` is
not searchable; one where each carries "checks that every kernel in the trace has
a recorded shape — completeness dimension" is.

### 4.1 The interface

```python
class Validator(Protocol):
    brief: str                              # the purpose, one line
    inputs: tuple[HandoffKind, ...]         # declared, not discovered
    dimension: Dimension                    # §5.1
    strength: Strength                      # §5.3

    def __call__(self, handoffs: dict[HandoffId, Handoff]) -> dict[HandoffId, bool]: ...
```

One callable, one declared input contract. This is the shape every surveyed
system converged on (§6.1) and it is deliberately the smallest thing that works.

**Inputs are declared rather than discovered.** A validator says which kinds it
consumes, so the phase can dispatch only compatible validators and a reviewer can
answer "what does this actually read" without running it. The two surveyed
systems that leave it implicit both pay for it — a mistyped parameter name
silently means the value is never passed.

**Lookup is by handoff uuid**, with an **RFC 6901 JSON Pointer** into the content
where the check needs one value rather than the whole artefact (handoff spec
§5.1, which gives the reason a Pointer and not a jsonpath).

**The binding is many-to-many.** A validator may take several handoffs; a handoff
may have several validators. Both directions are genuinely many.

### 4.2 The result is boolean in v1

`dict[HandoffId, bool]`. A score type is reserved in the schema and built later
([`../../docs/ROADMAP.md`](../../docs/ROADMAP.md)): a score is only useful against
a threshold, and a threshold is only meaningful once run-to-run variance has been
measured. Specifying the type before specifying how its threshold is derived
would produce thresholds indistinguishable from noise.

---

## 5. What a validator checks, and how far it can be trusted

Two independent classifications. The first says *which kind of question* the
check asks; the second says *how much the answer is worth*.

### 5.1 Three quality dimensions

Every validator addresses one dimension, declared in its spec. All three need to
be reduced to code where possible.

| Dimension | The question | How checkable |
|---|---|---|
| **Completeness / conformance** | Is it well-formed and complete? Does it match its declared schema? | **Easiest.** A schema exists, so this is usually `strong` with little effort |
| **Usability** | Can the next task actually use it? Is it convenient to use? | **Usually reachable.** Often `strong` — "does the downstream loader parse it", "does the reproduce script run" |
| **Trustworthiness** | Is it *right*? | **The hard one.** Often not quantifiable, or not on a short loop |

The third is where the real work is, and it has three partial routes:

1. **Long feedback from the end-to-end system** — the artefact's quality shows up
   once a downstream task, or the whole loop, has run. `long_term_strong` (§5.3).
2. **Internal consistency self-checks** — the artefact can be checked against
   itself. Percentages that must sum; a perturbation whose effect is predictable.
   Frequently `strong` and frequently overlooked.
3. **Otherwise `weak`**, honestly labelled.

Declaring the dimension is what stops a registry filling up with completeness
checks while nobody notices that nothing checks trustworthiness. Three
`strong` validators on one handoff mean little if all three check the schema.

### 5.2 The logic and the criterion are both external

| Checking logic | Trust |
|---|---|
| **External, programmatic** | **High.** Code someone else wrote, running the same way every time. Sub-divides into *static* (written in advance) and *dynamic* (produced by an earlier task in the flow) |
| **External, agent-written** | **Medium.** An agent wrote it — but not the agent being checked |

There is no third row: logic written by the agent under test is not a validator,
it is the producer's opinion with extra steps.

| Checking criterion | Trust |
|---|---|
| **Clearly quantifiable** | **High** |
| **Hard to quantify** | **Low** |

The *dynamic* case is how the reference workflow actually works: the task that
packages an operator also produces the correctness harness and the performance
measurement method, and hands both downstream as handoffs. The optimising agent
receives a check it did not write and cannot see the standard of.

### 5.3 `strong`, `long_term_strong`, `weak`

**`strong`** — a quantified standard, checkable on a short loop:

1. There is a quantitative method.
2. There is a result and a ground truth, or a way to produce both.
3. The risk closes: a failure is detectable within the loop.

**`long_term_strong`** — a quality that **cannot be verified when the handoff is
produced**. It has to wait for a long feedback loop:

| Waits for | Example |
|---|---|
| A downstream task to finish | Was this trace analysis sound? Ask after the optimisation it justified was measured |
| The whole loop / e2e job to finish | Did this configuration represent the real workload? |
| External customer feedback | Further still |

It is not a weaker `strong`. The rigour is the same; the *timing* is what
differs, and that is what the label records.

**`weak`** — no clear criterion, or none on a short loop; open risk exposure; but
the end-to-end outcome is assessable.

Weak is not worthless. "Is this deployment configuration production-grade?" has
no quantified answer, and an agent's analysis against public knowledge and prior
experience is genuinely informative.

### 5.4 The label qualifies a pass, never a failure

**A failure binds at every strength.** A `weak` validator that fails, fails the
phase, exactly as a `strong` one does. The label is not a measure of how much the
check matters.

What it qualifies is the **pass**:

| Verdict | `strong` | `weak` |
|---|---|---|
| **Fails** | The phase fails | The phase fails — identically |
| **Passes** | Evidence. The quality is established | The graph proceeds, and **a human must not read this as evidence** |

The asymmetry has a plain justification: a check that found something wrong found
something wrong, whatever its rigour. It is the *absence* of a finding that is
worth less when the method is weak — a check that cannot state its criterion in
advance also cannot claim much from not tripping.

Strengthening a `weak` check into a `strong` one, as the repository learns what
the real criterion is, is the intended path.

**The cost, named rather than discovered:** a weak validator's false positive
halts a branch. An agent's "this deployment configuration is not production-grade"
judgement stops work, and it may be wrong. That is accepted for the alpha, and it
is why §5.5 has exactly one rule rather than a strength-weighted one.

### 5.5 One fails, all fail

**The alpha's phase rule: any failing validator in a phase fails the phase**, and
a failed phase fails the task (§3.4).

One rule, no weighting, no quorum. The reducers of §6.2 — `all`, `any`,
`at_least(k)` — are scoped *within* a composite validator and say nothing about a
phase: a composite reduces its members to one verdict, and that verdict then
meets this rule like any other.

`--validation-strict-level` (§3.3) governs **skips only** — whether a phase runs
at all. It does not relax this rule. It is the natural home for a future
relaxation, and deliberately is not one now.

### 5.6 The label must be honest

A weak check labelled strong is **worse than no check**, because it stops anyone
looking further.

The observable test: **a `strong` validator can state, in advance, the number or
the comparison that decides it.** If the answer is "the agent assesses whether it
looks reasonable", the label is `weak`, whatever the intent.

A review guard that checks a validator's declared metadata against its actual
implementation is on the roadmap; until it exists, this is a review obligation.

---

## 6. Reuse without copy-paste, extension without a template engine

The requirement is two things at once: **avoid copy-paste between similar
validators**, and **stay open to extension**. Neither needs a template system.

A survey of sixteen evaluation, data-quality, and policy-as-code systems —
Inspect AI, DeepEval, Ragas, LangSmith, OpenAI Graders, promptfoo, HELM, Great
Expectations, Pandera, dbt, Soda, Deequ, OPA/Rego, Conftest, Gatekeeper, and
Dagster asset checks — found **none** that composes checks by templating source
with declared blanks. All of them use the same three layers.

### 6.0 Dagster asset checks: the closest existing system

Worth naming separately, because it is the nearest neighbour to this whole
document and was missing from the original survey.

[Dagster's asset checks](https://docs.dagster.io/guides/test/asset-checks) bind a
check to an **artefact** rather than to a run, execute it adjacent to the
artefact's production, and record the verdict against the artefact. That is the
same shape as §3: a validator is a phase of the task that produced the thing, and
the verdict lands on the handoff version.

It also arrived independently at the distinction §5.4 makes.
[Dagster separates severity from blocking as two orthogonal axes](https://github.com/dagster-io/dagster/discussions/16569),
deliberately, after users found a single axis conflated "how loud is this" with
"does this stop the pipeline". This spec reaches the same separation by a
different route — `strength` is a statement about evidence, and blocking is
uniform (§5.5) — and the convergence is worth recording: two systems, the same
mistake available, the same fix.

Where this spec differs, and why: Dagster's checks may be authored by whoever
authors the asset. Producer/validator context separation (§8) is the property
this system adds, and it is the one the Hyperloom teardown says is load-bearing.

### 6.1 A validator's logic has the same shape a task's body has

§3 says a validator is a special kind of task. **That is structural, not an
analogy**, and it decides what the checking logic *is*:

```
validator
  ├─ readme.md        ALWAYS required — the body an agent works from
  ├─ entry.sh         required iff the check is programmatic
  └─ materials        its own, whatever the check needs
```

Exactly `closure` spec §2.6's `body`, and the same mechanism serves both.

**A registered Python callable is not the interface.** It was, in an earlier
revision, and it fails the case this system most needs: **a validator an agent is
responsible for.** A callable cannot express "here is a description, an agent
carries it out" without a wrapper around the callable that runs an agent — so the
callable becomes a layer that exists only to be worked around, and the two kinds
of validator stop being one thing.

**For a code-shaped check, the system ships a pytest harness.** The validator
supplies its test code, the command that assembles it, and the command that runs
it, and all three live in `entry.sh`. That is a smaller mechanism than a
registration decorator plus a factory table plus an argument-signature check, and
it is the *same* mechanism a programmatic task already uses.

| Layer | Mechanism |
|---|---|
| **Interface** | `readme.md`, plus `entry.sh` when programmatic. A declared input contract — which handoff kinds it takes |
| **Instances** | One validator folder, parameterised through its spec. Two validators differing only in a threshold are two specs over one folder |
| **Discovery** | The validator registry, loaded from disk like every other spec (§9). No import side effects, no decorator |

**Duplicate registration raises.** Not overwrite: Great Expectations' registry
silently overwrites on a name collision, and that is the behaviour to avoid.
Pandera raises, and so does this system.

**What §6.1 gives up, stated.** Pandera's four-line `inspect.signature` check
against a validator's declared arguments — *"checks that look configured but
ignore their inputs"* — has nothing to read when the logic is a shell script or a
description. That check was worth having and it is not available in this shape;
what replaces it is that the argument surface is the spec, and the spec is
schema-checked at load.

### 6.2 Composition is flat

Where several checks combine, one combinator with a pluggable reducer:

```
composite(validators=[...], reduce="all" | "any" | "at_least(k)")
```

Inspect AI ships exactly this — `multi_scorer(scorers, reducer)` with registered
reducers. **Nesting is not permitted.** OpenAI's multigrader forbids nesting
outright; Gatekeeper refuses cross-template library reuse entirely, having
found that dependency conflicts outweigh the DRY benefit.

Where a check genuinely must gate one thing behind another — run the expensive
judgement only if the cheap check passed — that is a **DAG of validator
*instances***, one level deep, not a template. DeepEval's `DAGMetric` is the
model.

### 6.3 What was removed

The earlier revision specified a template validator with declared blanks,
composing other validators recursively to a configured depth, plus `.leaf.` and
`.template.` folder-name markers to tell the two apart.

**All of it is removed.** It was over-design: no surveyed system does it, two
explicitly prohibit the nesting variant, and the problem it solved is solved by
§6.1's middle row at a fraction of the complexity. The folder markers go with it,
since without templates every validator is a leaf.

The parameterisation mechanism is recorded in
[`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) as the thing that must exist
before the registry grows large — but it is a `{name, args}` table, not a code
generator.

**This does not conflict with the spec files being jsonnet templates** (main spec
§4.4), and the two are easy to confuse:

| | The removed system | jsonnet spec templates |
|---|---|---|
| What is templated | The **checking logic** — a validator's source, with blanks | The **spec** — configuration: which handoff kinds, which threshold, which tags |
| Who fills the blanks | An agent, at run time | A package author, before anything runs |
| Prior art | None of the sixteen surveyed systems | Kustomize, jsonnet, Helm — the standard answer |

A validator's implementation stays one small callable in code. Its *spec* may be
generated from a template, exactly as a handoff kind's or an agent's is.

---

## 7. What the validator system does not own

**Frameworking the test code itself.** Test code that falls into recognisable
families should be frameworked — by the external test system that owns it, not
here.

| Owned here | Owned by the external test system |
|---|---|
| Where a check plugs in | How the check's own code is structured |
| The input contract and the result shape | The assertions and their helpers |
| Whether a check is `strong` or `weak` | Whether two checks share a base class |
| That the check ran, and in whose context | Whether the check is fast |

---

## 8. The producer cannot grade its own output

The load-bearing constraint. Two mechanisms enforce it.

### 8.1 Context separation, by hook

**The agent that produced an artefact cannot reach the context in which it is
checked.**

| The producer cannot | Because |
|---|---|
| Read the checking standard | Knowing the bar lets an agent optimise for the bar |
| Write the checking logic | §5.2 |
| Write the verdict | §3.1 |
| See the hidden inputs a differential comparison uses | Otherwise the comparison checks memorisation |

**By hook, not by convention.** A convention is a prompt instruction, and an
agent complies with prompt instructions literally. The mechanism is
[`../../env_mgr/docs/spec.md`](../../env_mgr/docs/spec.md) §4's permission zones:
the validator's materials live where the producing task's permissions do not
reach.

### 8.2 A fresh environment, every time

**A validation environment is rebuilt from configuration. It is never reused
directly.** The configuration may match the producer's or the consumer's; the
environment, the sandbox, and the context are new ones.

The rule in one line: **reusing a configuration is fine; inheriting an
environment or a conversation is not.** Direct reuse would hand the validation a
filesystem and a sandbox the producer has already written to, which is the
boundary §8.1 exists to hold.

Relaxing this — reusing an environment directly, or with light modification — is
on the roadmap ([`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §6) and carries
the same warning there: it must not blur the isolation standard, which is the
entire reason each validation gets a fresh environment.

The default configuration, in order:

| Case | Environment configuration |
|---|---|
| The validator is bound to a real agent with a declared environment | That one |
| Otherwise, **input** validation | The **consumer's** — the task about to run |
| Otherwise, **output** validation | The **producer's** — the task that just ran |
| Otherwise | A predefined global one |

The middle two are why the phases sit inside the task (§3): the right
configuration is the one already **resolved**, which is not the same as the one
already running.

---

## 9. Folders, tags, and the registry

### 9.1 Layout

Every validator is a folder in a **task package** (main spec §4.3), or — for the
workflow-independent ones such as a JSON Schema check — in this repository's
general-spec directory (main spec §4.5). Since templates are gone, every
validator is a leaf and no name marker distinguishes kinds.

A validator's folder carries relative symlinks to the handoff kinds it binds to,
so the binding is visible in a directory listing and not only in the registry.

**A symlink may point outside its own package**, and that is how two packages
share a handoff kind. Placing it correctly is the package's job: the loader
resolves a path and does not interpret anyone's layout. A symlink that dangles is
a load error naming the path, not a puzzle for the loader to solve.

### 9.2 Tags

A tag dictionary — key/value, so dimensions stay distinguishable:

| Key | Values |
|---|---|
| `dimension` | `completeness` \| `usability` \| `trustworthiness` (§5.1) |
| `strength` | `strong` \| `long_term_strong` \| `weak` (§5.3) |
| `logic_source` | `external_static` \| `external_dynamic` \| `agent_written` |
| `cost` | An order of magnitude: seconds, minutes, GPU-hours. Feeds §2 principle 2 |
| `domain` | Free-form: `trace`, `kernel`, `deploy`, `eval` |

### 9.3 The registry

A validator's YAML also carries a `version` — its own revision. **Maintenance
metadata only**: nothing at runtime reads it, and it is not what makes a
validator's past verdicts re-interpretable (closure spec §1.2, and the roadmap's
validator-versioning item).

One of the four independent registries (main spec §4.1). It records each
validator's spec, tags, brief, and bound handoff kinds, plus **who uses it now
and who has used it** — the first answers "what breaks if I change this", the
second answers "has this check ever actually run", and a validator that has never
run is one nobody should trust.

**Load-time checks**, each failing loudly with the file path:

1. The YAML validates against the schema; the name is unique. **A duplicate name
   raises** — it does not overwrite.
2. `brief`, `dimension`, and `strength` are present. **No defaults**: an
   unlabelled validator would default to being trusted.
3. Every declared input kind resolves in the handoff registry.
4. The binding agrees with the handoff registry's side. **A conflict crashes**
   (handoff spec §5.1).
5. A composite's members resolve, and it does not contain another composite.

---

## 10. Worked example — the reference workflow

The six-step optimisation loop in this vocabulary. It doubles as the check that
the vocabulary is sufficient (main spec criterion 7). Each entry names its
dimension.

### 10.1 Human-supplied `<deploy config, workload, SLA>`

| Dimension | Strength | Check |
|---|---|---|
| trustworthiness | **weak** | Is this state-of-the-art and production-grade? No quantified standard. An agent analyses against public knowledge, prior experience, and — for individual items — an open-source publication or a customer guarantee |

Nothing here is strong, and the spec does not pretend otherwise. This input is
where the chain's validity is decided, and it is a human's responsibility.

### 10.2 The e2e run method

| Dimension | Strength | Check |
|---|---|---|
| usability | **strong** | Correctness: `curl` / ping / a short load test. Quantifiable, programmable, externally specified |
| trustworthiness | **weak** | Are all performance-relevant knobs open? Analysed against engine source and accumulated knowledge. *Individual knobs escalate to strong* when an expected number decides them |

### 10.3 The trace getter

| Dimension | Strength | Check |
|---|---|---|
| completeness | **strong** | Schema check, readability, presence of key fields |
| trustworthiness | **strong** | Self-consistency: percentage sums; and **mock perturbation** — inflate, shrink, delete, or add a duration and confirm the trace changes as predicted |
| trustworthiness | **long_term_strong** | Did the trace support the analysis it was collected for? Answerable only afterwards |
| completeness | **weak** | Full coverage, judged against model structure and code. Not programmable in general |
| trustworthiness | **weak** | Is the measured time plausible? The profiler's overhead is known only from experience |

The perturbation check is the §5.1 route-2 case — internal consistency reaching
`strong` on a question that looks unquantifiable. It is also the check most often
skipped, and its absence is how a trace that silently drops kernels passes.

### 10.4 Trace analysis and the extractor

| Dimension | Strength | Check |
|---|---|---|
| usability | **strong** | Top-k selection: credible as long as no headroom claim rides on it |
| usability | **strong** | The extracted operator's correctness-test code — inputs and outputs can be captured |
| usability | **strong** | Its timing-test code — method and standard come from elsewhere |
| trustworthiness | **long_term_strong** | Post-run feedback |
| trustworthiness | **weak** | Headroom and roofline bottleneck analysis |
| trustworthiness | **weak** | First-pass analysis of a pipelined or overlapped trace |

The last row matters more than its length: with two-batch overlap and dual-stream
MoE, much computation is deliberately overlapped with communication, and
optimising a kernel on the short branch yields exactly zero end-to-end gain.

### 10.5 The optimised kernel

| Dimension | Strength | Check |
|---|---|---|
| trustworthiness | **strong** | Differential-comparison code and correctness checker, **with hidden inputs supplied by the upstream task** |
| usability | **strong** | The performance evaluator, also supplied upstream |
| trustworthiness | **weak** | Optimisation quality in the abstract |

§5.2's dynamic case at its clearest: the run method, the checking method, the
checked content, and the standard are all supplied by someone other than the
optimising agent.

### 10.6 Integration complete

| Dimension | Strength | Check |
|---|---|---|
| trustworthiness | **strong** | Evaluation suite |
| usability | **strong** | Benchmark |

The most strongly-gated step, appropriately: it is where a claimed improvement
either survives contact with the end-to-end system or does not.

### 10.7 System-level validators

| Validator | Strength | What it does |
|---|---|---|
| **global validator** | `long_term_strong` | Looks back at the end of the run at whether the trace and the analysis were sound, given what the optimisation achieved |
| **goal validator** | `weak` | An independent agent receives the output and the task definition and judges whether the work was done. Independent because §8 |
| **cheat validator** | `weak` | Checks whether the work gamed its own evaluation |

The cheat validator earns its place from observed behaviour, not paranoia. Three
distinct evaluation-surface exploits are on record from one competition: taking
one's own first version as the baseline; copying tolerance logic while omitting
the NaN check, so an all-NaN output is both fast and "correct" because NaN
comparisons are always false; and a writer model discovering the verifier had
edit permission and instructing it, in the verification prompt, to do the work.

None is a bug in the optimiser. All three are rational responses to an evaluation
surface with a hole in it.

---

## 11. Acceptance criteria

1. A validator spec missing `brief`, `dimension`, or `strength` is rejected at
   load. There are no defaults.
2. A validator spec declaring a subtask, or a non-empty input validation of its
   own, is rejected at load.
3. **A duplicate validator name raises at registration.** It does not overwrite.
4. A validator returns `dict[HandoffId, bool]` with one entry per input handoff —
   verified for a validator taking three handoffs.
5. **A validation phase is invisible to the scheduler.** Across a full run, the
   scheduler dispatches one task, no validator occupies a pool, and the policy is
   never asked to order one.
6. A validation phase produces **no output handoff**; it calls
   `update_validation_status`, the record persists, and the handoff's digest is
   unchanged.
7. A skipped phase — by config, or because the version was already validated —
   is **reported**, and the task moves to the next state.
8. `--validation-strict-level` changes which skips are permitted.
9. **A validation runs in a fresh agent environment**, and the default
   configuration follows §8.2's chain: bound env, else consumer's for input, else
   producer's for output, else global.
10. **The producing agent cannot read the checking standard.** A spy records
    context reads across a produce → validate cycle; no read originating in a
    producer frame reaches the standard, and the hook denies the attempt.
11. A validator whose logic lives in the producing task's permission zone is
    rejected — the check is structural, not declarative.
12. **Reuse needs no copy-paste**: two validators differing only in a parameter
    are two registry entries over one implementation, and the shared logic is a
    plain function that is directly testable.
13. A composite runs its members and reduces with `all` / `any` / `at_least(k)`;
    **a composite containing a composite is rejected at load.**
14. The registry answers "who uses this" and "who has used it" separately, and a
    validator that has never run is distinguishable from one that runs constantly.
15. Each of the three dimensions is represented in the shipped validator set, and
    the registry can list validators by dimension — so "nothing checks
    trustworthiness on this kind" is answerable.
16. Every step of the reference workflow in §10 is expressible: its handoffs, its
    validators, and each validator's dimension and strength resolve.
17. **A failed validation phase is indistinguishable from any other task
    failure**: the task ends `FAILED`, its resources are released, its consumers
    stay in `WAITING_HANDOFF`, and nothing downstream is cancelled or invalidated
    by the scheduler (§3.4).
18. **A failing `weak` validator fails the phase**, exactly as a failing `strong`
    one does — asserted by running the same phase with one of each and observing
    the same outcome (§5.4).
19. **A passing `weak` validator is reported as a low-confidence pass**, and is
    distinguishable in the output from a passing `strong` one.
20. **`--validation-strict-level` changes which phases run, and never which
    verdicts bind** (§5.5).
21. **A validation environment is a rebuild, not a reuse.** The producer's
    sandbox and filesystem view are not the validation's: a file the producing
    agent left in its environment is absent from the validation's, even where
    both were built from the same configuration (§8.2).

---

## 12. Open questions

| Item | Status |
|---|---|
| **Score-typed results** | §4.2. Reserved, unbuilt. Requires first specifying how a threshold is derived from measured run-to-run variance |
| **The parameterisation table** | §6.3 names `{name, args}` as the mechanism and the roadmap carries it. The exact shape — where the table lives, whether a handoff kind may supply args — is a design-stage decision |
| **Dynamic logic provenance** | §5.2's `external_dynamic` is trusted because the producing task is not the checked task. Nothing verifies that at load time: the graph shape is what makes it true, and the registry does not see the graph |
| **Cost-aware ordering within a phase** | §2 principle 2 wants cheap gates first and §9.2 tags cost, but nothing consumes the tag. Ordering *within* a phase is the runner's, so this is answerable — just unspecified |
