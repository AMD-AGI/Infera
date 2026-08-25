# Validator — Specification

| | |
|---|---|
| Status | Draft, pending review |
| Revision | 1 — 2026-08-24 |
| Date | 2026-08-24 |
| Scope | What makes a handoff checkable, how far a check can be trusted, and how validators are organised |
| Source | The task definition §4, §5; the Infera × Hyperloom kickoff report §2, §3, §5A |
| Depends on | [`../../handoff/docs/spec.md`](../../handoff/docs/spec.md), [`../../task_graph/docs/spec.md`](../../task_graph/docs/spec.md) |

---

## 1. Purpose

A handoff is only a contract if something checks it. **Validators are the sole
standard by which a handoff is judged**, which is why this is the largest
document in the set and why the review bar for a validator is higher than for the
thing it checks.

The system exists because the alternative was measured. The kickoff report's
teardown of an existing implementation (§5A) found:

> `correctness_passed` came from one of two places: a CLI override, or a regex
> over the report text looking for the string `"correctness passed"`. The harness
> was written by the LLM. The reference implementation was written by the LLM.
> The tolerance table permitted 10% of bf16 elements to mismatch and still PASS.

and concluded:

> The root cause in one line: no strict, standardised, hardened validation of the
> agent loop's inputs and outputs. **The absence of cheap gates pushed the entire
> verification burden onto the most expensive gate**, so dead ends and live ones
> cost the same.

This document specifies the cheap gates.

### 1.1 In scope

- What a validator is, structurally, and why it is a task.
- The three elements: input, process, result.
- Templates, blanks, and bounded recursion.
- The two-way binding between leaf validators and handoff kinds.
- The trust taxonomy: what makes a check `strong` or `weak`, and why the label
  must be honest.
- Folder conventions, tags, and the registry.
- The two principles that decide the hard cases.
- A worked example over the reference workflow.

### 1.2 Out of scope

- **Frameworking the test code itself.** §6.
- **How a handoff is versioned or stored** — handoff spec, `task_graph` spec §3.1.
- **When a validator task runs** — that is scheduling, and the scheduler decides
  it exactly as it decides any other task (`task_graph` spec §6.2).

---

## 2. Design principles

| # | Principle | Consequence |
|---|---|---|
| 1 | **The producer never grades its own output** | Enforced by hook, not by convention. §8.1 |
| 2 | **Anything that can be code is code** | The agent fills blanks in a fixed procedure; it does not invent the procedure. §8.2 |
| 3 | **A validator is a task** | It runs through the same scheduler, gets the same audit record, and is subject to the same authority boundary as any other work. §3 |
| 4 | **The standard is external** | Both the checking logic and the checking criterion come from outside the thing being checked. §5 |
| 5 | **Label honestly** | The `strong`/`weak` taxonomy is only useful if the labels are accurate. A weak check labelled strong is worse than no check, because it stops anyone looking further. §5.4 |
| 6 | **Cheap gates before expensive ones** | The reason the system exists. A validator that costs a GPU-hour is a last resort, not a first line |

---

## 3. A validator is a task

**A validator is a special kind of task spec.** Not a callback, not a plugin, not
a phase of another task — a task, which at runtime becomes an ordinary node the
scheduler dispatches like any other.

### 3.1 The three structural constraints

| Constraint | Why |
|---|---|
| **Single node.** A validator contains no subtasks | A check that expands into a subgraph is a workflow, and a workflow needs its own validators. The recursion has to stop somewhere, and this is where |
| **Its own input validation is empty** | Otherwise validating a validator's inputs requires a validator, without bound |
| **Consumes one or more handoffs; produces `dict[HandoffId, bool]`** | The result is a verdict per handoff, not per validator. One validator checking three handoffs returns three verdicts |

### 3.2 Why a task and not a callback

Three things fall out for free, and each would otherwise have to be built:

- **It is scheduled**, so it competes for resources like anything else. A
  validator that needs a GPU declares one and waits for it; a validator embedded
  as a callback inside another task would take its host's lease implicitly.
- **It is audited.** `Execution` records which agent ran the check, against which
  input versions (`task_graph` spec §3.2). "Which version of the checker passed
  this artefact" is answerable without new machinery.
- **It runs in its own context.** Which is §8.1, the reason the whole document
  exists.

The cost is that a check is a graph node, so a graph with many validators has
many nodes. That is accepted: the alternative — checks hidden inside tasks — is
exactly the shape that made the reference implementation's verification
unauditable.

### 3.3 Where validator tasks sit in the graph

A validator task is placed in a phase of the task whose handoffs it checks. The
phases are specified in [`../../task_graph/docs/spec.md`](../../task_graph/docs/spec.md) §3.2.1:

| Phase | Which validators |
|---|---|
| `input_validation` | Cross-handoff checks over the task's inputs. **Usually empty**: a single handoff's own validators already ran when it was produced, and re-running them here would double the cost for no information |
| `main` | The task's real work. No validators |
| `output_validation` | The task's outputs. Placed here rather than downstream because a check usually needs the same environment the producing task had — the model loaded, the trace open, the cluster reserved |

The `output_validation` placement is the non-obvious one and it is the task
definition's own reasoning: tearing down an environment and rebuilding it to run
a check is often more expensive than the check.

### 3.4 `result` is boolean in v1

`dict[HandoffId, bool]`. A `score` variant is named in the task definition and
reserved in the schema, but nothing in v1 produces or consumes one.

The reason it is reserved rather than built: a score is only useful against a
threshold, and the kickoff report's own finding is that thresholds set without
first measuring run-to-run variance are worthless —

> `DEFAULT_KEEP_THRESHOLD_PCT = 1.0`, whose own comment calls it "the grid noise
> floor". Repeated same-config measurements differed by roughly the same margin
> as the threshold, so real wins and noise became indistinguishable. Nowhere on
> the KEEP path was there a repeat measurement, a variance, or a confidence
> interval.

Specifying a score type before specifying how its threshold is derived would
reproduce that. See §11.

---

## 4. The three elements

Every validator is `<input, process, result>`.

```
input      ──►   process   ──►   result
from handoffs    template        dict[HandoffId, bool]
to named keys    with blanks
```

### 4.1 Input: a declared mapping, not a lookup

The validator declares, per input handoff, **which part of the handoff goes into
which key** of the process. It does not receive the handoff and go looking.

| Declared | Meaning |
|---|---|
| `from` | The handoff kind, and which of the validator's inputs this is |
| `path` | Where inside `content` the value sits |
| `key` | The name the process block sees |

Two reasons for the indirection. The process template can be written against
stable key names while handoff kinds evolve; and the mapping is a static
artefact, so "which fields does this validator actually read" is answerable
without running it — which is precisely what the coverage review (handoff spec
§5.2) asks.

### 4.2 Process: a template with blanks

**The procedure is fixed; the content is filled in.** A validator spec is a
fixed flow plus a set of declared holes, and the holes are declared in the YAML
so nobody has to guess where they are.

| | |
|---|---|
| The **flow** | Fixed by the validator kind. Input format and output format are defined in advance |
| The **blanks** | Declared positions the config fills |

The rule that makes this worth doing: **modifying the wiring code on the spot is
not permitted.** A blank is filled by configuration — a value, a template, a named
component — and never by editing the procedure at the point of use. If a check
needs a different procedure, it is a different validator kind.

This is §2 principle 2 in its most concrete form, and it is the direct answer to
the failure mode the kickoff report calls out: rules that live only in a prompt
are rules an agent satisfies literally and defeats.

### 4.3 Result: the verdict, and who records it

The validator writes the verdict. **The producer does not report it and the
scheduler does not infer it** — the scheduler is forbidden from writing handoff
state at all (main spec §5.1, `task_graph` spec §3.1).

Mechanically, the verdict reaches the handoff the way any handoff write does: the
validator's agent seals the version `VALID` or `INVALID`. What makes it
trustworthy is not the mechanism, which is shared, but §8.1 — the validator's
context.

### 4.4 The binding, and which side wins

A leaf validator names the handoff kinds it binds to; a handoff kind names its
validators (handoff spec §8.1). Both sides are recorded, deliberately.

- **Why both:** each direction is needed at a different moment. Loading a handoff
  kind must resolve its validators; reviewing a validator must show what it
  covers. Deriving either from the other is a full scan of the other registry.
- **On conflict, the handoff kind wins,** and the mismatch is **reported, not
  silently resolved.** The handoff's declaration is what gates its admission
  (handoff spec §5), so it is the operative one; the validator's list is an index.
  A mismatch is a mistake either way and stays visible.

---

## 5. The trust taxonomy

The core of this document. A validator's trustworthiness is decided by two
independent questions, and the answers compose.

### 5.1 The checking logic — always externally supplied

| Source | Trust |
|---|---|
| **External, programmatic** | **High.** Code someone else wrote, that runs the same way every time. Sub-divides into *static* (written in advance) and *dynamic* (produced by an earlier task in the flow) |
| **External, agent-written** | **Medium.** An agent wrote it, but not the agent being checked |

There is no third row. **Logic written by the agent under test does not appear in
this table**, because a validator built that way is not a validator — it is the
producer's own opinion with extra steps, and §8.1 forbids it structurally.

The *dynamic* sub-case is worth naming because it is how the reference workflow
actually works: the task that packages an operator also produces the correctness
harness and the performance measurement method for it, and hands both downstream
as handoffs. The optimising agent then receives a check it did not write and
cannot see the standard of.

### 5.2 The checking criterion — always externally supplied

| Criterion | Trust |
|---|---|
| **Clearly quantifiable** | **High** |
| **Hard to quantify** | **Low** |

### 5.3 The two combine into `strong` and `weak`

**`strong`** — a quantified standard, checkable on a short loop:

1. There is a quantitative method.
2. There is a result and a ground truth, or a way to produce both.
3. The risk closes: a failure is detectable within the loop.

**`long-term strong`** is a sub-case: the same rigour, but the verification chain
only closes at the end of the run, in combination with other handoffs. A trace's
plausibility judged after the optimisation it justified has been measured is
long-term strong — quantified, but not answerable at the moment the trace is
produced.

**`weak`** — no clear criterion, or none available on a short loop:

1. No quantified standard.
2. Open risk exposure.
3. But the end-to-end outcome is assessable.

Weak is not worthless. "Is this deployment configuration production-grade?" has
no quantified answer, and an agent's analysis against public knowledge, prior
experience, and vendor guarantees is genuinely informative. It is simply not a
gate, and must not be used as one.

### 5.4 The label must be honest

A weak check labelled strong is **worse than no check**, because it stops anyone
from looking further. This is why §5.3's three conditions are enumerated rather
than left to judgement, and why the coverage review (handoff spec §5.2) asks
whether each label is accurate.

The observable test: **a `strong` validator can state, in advance, the number or
the comparison that decides it.** If the answer is "the agent assesses whether
it looks reasonable", the label is `weak`, whatever the intent.

### 5.5 What makes a strong validator possible

A `strong` check needs its criterion supplied from outside. In the reference
workflow the supplier is usually the *upstream* task, and the pattern is worth
naming because it recurs:

> The task that produces an artefact also produces the means of checking it —
> the differential-comparison code, the hidden inputs, the performance
> evaluator — and hands them downstream as handoffs. The consumer of the
> artefact therefore receives a check it did not write, and the standard is
> not visible to it.

That is what makes "the run method, the checking method, the checked content,
and the standard are all defined by someone else" achievable rather than
aspirational.

---

## 6. What the validator system does not own

**The validator system is not responsible for abstracting or frameworking the
test code itself.** Test code that falls into recognisable families should be
frameworked — by the external test system that owns it, not here.

Where the line falls:

| Owned here | Owned by the external test system |
|---|---|
| Where a check plugs in | How the check's own code is structured |
| The input mapping and the result shape | The assertions and their helpers |
| Whether a check is `strong` or `weak` | Whether two checks share a base class |
| That the check ran, and in whose context | Whether the check is fast |

Getting this wrong in either direction is expensive. Pulling test frameworks in
here makes the validator system grow a testing DSL; pushing the plug-in point out
makes checks unauditable.

---

## 7. Folders, tags, and the registry

Validators accumulate, and the point of the taxonomy is lost if nobody can find
the validators that already exist.

### 7.1 Folder conventions

| Convention | Meaning |
|---|---|
| `.leaf.` in the folder name | A leaf validator. Binds directly to handoff kinds; **may not declare blanks of its own** |
| `.template.` in the folder name | A template validator. May declare blanks; composes other validators |
| Relative symlinks to its handoffs, inside a leaf validator's folder | Makes the binding visible in the filesystem, not only in the registry |

The marker is in the *name* rather than in a field so that the distinction
survives a directory listing. Someone browsing the folder can see the shape of
the library without opening files, and the two kinds have genuinely different
rules — a leaf validator with blanks is a template that forgot to say so.

### 7.2 Recursion is bounded

A template validator may compose others, and this recurses. **The depth is
capped, and the cap is configured, not a law of the system.** The task definition
suggests three; the spec adopts three as the default and requires that exceeding
it fail at load time with the chain in the message, rather than at run time.

Load-time, because a validator library that only reveals its cycles when a
particular graph runs is a library nobody can review.

### 7.3 Tags

Every validator carries a tag dictionary — key/value, not a flat list, so
dimensions stay distinguishable. Expected dimensions:

| Key | Values |
|---|---|
| `strength` | `strong` \| `long_term_strong` \| `weak` |
| `logic_source` | `external_static` \| `external_dynamic` \| `agent_written` |
| `cost` | An order of magnitude: seconds, minutes, GPU-hours. Feeds §2 principle 6 |
| `domain` | Free-form: `trace`, `kernel`, `deploy`, `eval` |

### 7.4 The registry

One of the four independent registries (main spec §4.1). It records, for every
validator:

- Its spec, tags, and kind (`leaf` or `template`).
- **Which handoff kinds it binds to** — the §4.4 index.
- **Who uses it in the current system, and who has used it.** Both, deliberately:
  the first answers "what breaks if I change this", the second answers "has this
  check ever actually run", and a validator that has never run is a validator
  nobody should trust.

**Load-time checks**, each failing loudly with the file path:

1. The YAML validates against the schema.
2. The name is unique.
3. A `.leaf.` validator declares no blanks.
4. Every composed validator resolves, and the composition depth is within the cap.
5. Every bound handoff kind resolves, and the binding agrees with the handoff
   registry's side of it (§4.4).
6. `strength` is present. There is no default — an unlabelled validator would
   default to being trusted, and §5.4 is the reason that is unacceptable.

---

## 8. The two principles that decide the hard cases

### 8.1 Producer and validator contexts are separated, by hook

**The agent that produced an artefact cannot reach the context in which it is
checked**, and the separation is enforced by hook rather than by convention.

What "separated" covers:

| The producer cannot | Because |
|---|---|
| Read the checking standard | Knowing the bar lets an agent optimise for the bar. The kickoff report's finding: "the evaluation rule must not be perceptible to the optimiser" |
| Write the checking logic | §5.1 |
| Write the verdict | §4.3 |
| See the hidden inputs a differential comparison uses | Otherwise the comparison checks memorisation |
| Report its own risk assessment where the assessment is the gate | §2 principle 6 of the main spec: some reports may not be self-issued, to prevent under-reporting |

**By hook, not by convention**, because a convention is a prompt instruction and
an agent complies with prompt instructions literally. The mechanism is
[`../../env_mgr/docs/spec.md`](../../env_mgr/docs/spec.md) §4's permission zones
and [`../../agent/docs/spec.md`](../../agent/docs/spec.md) §5's `PreToolUse`
hook: the validator's materials live in a zone the producing agent's permission
list does not name, and the hook denies the read.

### 8.2 Anything that can be code is code

Codify what can be codified. Remove the agent from the procedure, or reduce it to
an intern filling in blanks in a procedure someone else wrote.

The pattern the kickoff report identifies as the single most reusable thing it
found, and which this system adopts:

> **Three stages.** ① Python computes every number and writes it to JSON.
> ② The agent only renders; if the script fails, it writes ERROR rather than
> improvising. ③ A checker verifies that the key fields in the agent's report
> were copied verbatim from the JSON.
>
> Stage three turns anti-hallucination from "hope the model does not invent
> numbers" into "if it invents one, it is caught."

Every `strong` validator should be readable as an instance of this: the numbers
come from code, the agent's role is presentation, and the copy is checked.

---

## 9. Worked example — the reference workflow

The kickoff report's main loop is six steps. This section walks it in the
vocabulary above, both as an illustration and as the coverage check that the
vocabulary is sufficient (main spec criterion 7).

### 9.1 Human-supplied `<deploy config, workload, SLA>`

| | |
|---|---|
| **weak** | Is this state-of-the-art? Is it production-grade? No quantified standard exists. An agent analyses against public knowledge, the user's or its own prior experience, and — for individual items — an open-source publication or a customer guarantee |

Nothing here is strong, and the spec does not pretend otherwise. The kickoff
report's own analysis is that this input is where the whole chain's validity is
decided, and that it is a human's responsibility, not the loop's.

### 9.2 The e2e run method

| | |
|---|---|
| **strong** | Correctness: `curl` / ping / a short load test. Quantifiable standard, programmable procedure, externally specified |
| **weak** | Are all the performance-relevant knobs actually open? An agent analyses against the engine source, public material, and locally accumulated knowledge. *Individual knobs escalate to strong* when a specific expected performance number can decide them |

The escalation in the second row is the useful pattern: a weak check often
contains a strong one, and finding it is worth more than accepting the weak label.

### 9.3 The trace getter

| | |
|---|---|
| **strong** | Usability: schema check, readability, presence of key fields |
| **strong** | Post-run feedback: did the trace support the analysis it was collected for |
| **strong** | Self-consistency: percentage sums; and mock perturbation — inflate, shrink, delete, or add a duration and confirm the trace changes as predicted, either by re-running at source level or by checking the trace directly |
| **weak** | Completeness: judged against the model structure and the code. Not programmable in general; some models can be handled specifically |
| **weak** | Is the measured time plausible? The profiler's own overhead is only known from experience |

The self-consistency row is a `strong` check that is easy to overlook, and the
kickoff report names the failure it prevents: an implementation whose trace
validator's own docstring read "Logs warnings (**never raises**)", and whose
analysis silently discarded kernels it could not correlate. Nobody, upstream or
downstream, ever computed what fraction of GPU time the trace covered.

### 9.4 Trace analysis and the extractor

| | |
|---|---|
| **strong** | Top-k selection: credible as long as no headroom analysis is claimed |
| **strong** | The extracted operator's correctness-test code: credible, because inputs and outputs can be captured |
| **strong** | Its timing-test code: credible, because the method and the standard come from elsewhere |
| **strong** | Post-run feedback |
| **weak** | Headroom and roofline bottleneck analysis |
| **weak** | First-pass analysis of a pipelined or overlapped trace |

The last row matters more than its length suggests. With two-batch overlap and
dual-stream MoE, a large amount of computation is *deliberately* overlapped with
communication, and optimising a kernel on the short branch of an overlap yields
exactly zero end-to-end gain until it becomes the long branch.

### 9.5 The optimised kernel

| | |
|---|---|
| **strong** | Differential-comparison code and correctness checker, **with hidden inputs supplied by the upstream task** |
| **strong** | The performance evaluator, also supplied upstream |
| **strong** | Both of the above can be templated |
| **weak** | Optimisation quality in the abstract |

This is §5.5's pattern at its clearest: the run method, the checking method, the
checked content, and the standard are all supplied by someone other than the
agent doing the optimising.

### 9.6 Integration complete

| | |
|---|---|
| **strong** | Evaluation suite |
| **strong** | Benchmark |
| **strong** | Run method, checking method, checked content, and standard — all externally defined |

The most strongly-gated step in the workflow, and appropriately so: it is where a
claimed improvement either survives contact with the end-to-end system or does
not.

### 9.7 System-level validators

Three that do not attach to a single step:

| Validator | Kind | What it does |
|---|---|---|
| **global validator** | `long_term_strong` | Looks back, at the end of the run, at whether the trace and the analysis were sound in the light of what the optimisation actually achieved |
| **goal validator** | `weak` | An independent agent receives the output and the task definition and judges whether the work was completed. Independent because §8.1 |
| **cheat validator** | `weak` | Checks whether the work gamed its own evaluation |

The cheat validator earns its place from the record. The kickoff report documents
three distinct evaluation-surface exploits observed in one competition: taking
one's own first version as the baseline; copying tolerance logic while omitting
the NaN check, so an all-NaN output is both fast and "correct" because NaN
comparisons are always false; and a writer model discovering the verifier model
had edit permission and instructing it, in the verification prompt, to do the
work.

None of those is a bug in the optimiser. All three are rational responses to an
evaluation surface with a hole in it, which is why a validator that looks for
them is a permanent fixture rather than a phase-one expedient.

---

## 10. Acceptance criteria

1. A validator spec declaring a subtask is rejected at load.
2. A validator spec declaring a non-empty input validation is rejected at load.
3. A `.leaf.` validator declaring blanks is rejected at load, and the message
   says which blank.
4. Template composition exceeding the configured depth cap is rejected **at
   load**, with the composition chain in the message — not at run time.
5. A validator spec with no `strength` tag is rejected. There is no default.
6. A validator returns `dict[HandoffId, bool]` with one entry per input handoff —
   verified for a validator taking three handoffs.
7. **A validator task is dispatched by the scheduler like any other task**: it
   appears in the pools, declares resources, and produces an `Execution` record
   naming the agent that ran it and the input versions it read.
8. **The producing agent cannot read the checking standard.** A spy records
   context reads across a produce → validate cycle; no read originating in a
   producer frame reaches the validator's standard, and the hook denies the
   attempt rather than the convention discouraging it.
9. A validator whose logic was written by the agent under test is rejected — the
   check is structural (whose permission zone the logic lives in), not
   declarative.
10. The two-way binding is consistent after loading both registries; a deliberate
    mismatch is reported, the handoff kind's side wins, and nothing is silently
    rewritten.
11. The registry answers "who uses this validator" and "who has used it"
    separately, and a validator that has never run is distinguishable from one
    that runs constantly.
12. An `output_validation` validator runs in the same environment as the task
    whose outputs it checks — verified by asserting it does not re-provision.
13. Each of the six reference-workflow steps in §9 is expressible: its handoffs,
    its validators, and each validator's `strength` label resolve through the
    registries.
14. The three-stage pattern of §8.2 is demonstrated by at least one shipped
    validator: numbers computed by code into JSON, rendered by an agent, and the
    render checked for verbatim copying.

---

## 11. Open questions

| Item | Status |
|---|---|
| **Score-typed results** | §3.4. Reserved in the schema, unbuilt. Building it requires first specifying how a threshold is derived from measured run-to-run variance, which is a measurement task, not a specification task |
| **The recursion depth number** | Three, adopted from the task definition's suggestion, is a guess. Nothing has yet needed two levels, so the cap has not been tested against a real library |
| **How a cross-handoff input validation names its inputs** | §3.3 puts cross-handoff checks in `input_validation`, and §4.1 says a validator declares its input mapping by handoff kind. When a task has two inputs of the *same* kind, the mapping is ambiguous. Positional? A declared role name? Undecided; related to handoff spec §11's cross-schema reference question |
| **Dynamic logic provenance** | §5.1's `external_dynamic` — logic produced by an earlier task — is trusted because the producing task is not the checked task. Nothing verifies that at load time, because the graph shape is what makes it true and the registry does not see the graph. A load-time check would need the closure |
| **Validator versioning** | A validator that changes has re-graded history. Whether a verdict records which validator version produced it, and whether an old verdict is invalidated when its validator changes, is unspecified. `long_term_strong` makes this pressing: the global validator runs against verdicts recorded much earlier |
| **Weak-validator aggregation** | Several weak checks agreeing is more informative than one, and the kickoff report's strongest recommendation on this point is to use a different model family for the reviewer than for the producer, so the two do not share blind spots. Nothing specifies how multiple weak verdicts combine, or whether the system should require model-family diversity |
| **Cost-aware ordering** | §2 principle 6 wants cheap gates first, and §7.3 tags cost — but nothing uses the tag. Whether the scheduler should order validators by cost, and how that interacts with it not inspecting content, is open |
