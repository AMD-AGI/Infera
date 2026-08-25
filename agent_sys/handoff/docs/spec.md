# Handoff — Specification

| | |
|---|---|
| Status | Draft, pending review |
| Revision | 1 — 2026-08-24 |
| Date | 2026-08-24 |
| Scope | What a unit of transfer carries: schema, digest, scope tags, validator binding |
| Source | The task definition §3; the Infera × Hyperloom kickoff report §2, §4 |
| Depends on | [`../../task_graph/docs/spec.md`](../../task_graph/docs/spec.md) §3.1 — the versioned slot |

---

## 1. Purpose

A handoff is a module's input or output. It is the only thing that crosses
between tasks, which makes it the only place where quality can be enforced
system-wide.

**This document specifies the schema and lifecycle layer.** The runtime slot —
identity, versions, the `CREATED → GENERATING → VALID | INVALID` machine, who may
write it — is already specified in `task_graph` spec §3.1 and is **not** restated
here. The two layers meet at exactly one point: a `Handoff.type` names a handoff
*kind*, and a kind is what this document specifies.

```
task_graph spec §3.1        this document
─────────────────────       ──────────────────────────────
Handoff       the slot      HandoffSpec    the kind
  .id                         name
  .versions[]                 required fields
  .type ─────────────────────►schema
  .open_next() / .seal()      validators
                              scope tag
```

### 1.1 In scope

- The required schema of a handoff's content, and the digest that makes it
  tamper-evident.
- The scope and lifecycle vocabulary: `fixed` versus `addons`.
- The rule that every handoff kind carries at least one validator, and the
  review obligation that comes with admitting a new kind.
- Locality independence — what makes a handoff replayable on another machine.
- The YAML spec file and its registry.

### 1.2 Out of scope

- **The versioned slot** — `task_graph` spec §3.1.
- **How a validator works** — [`../../validator/docs/spec.md`](../../validator/docs/spec.md).
  This document specifies only that a binding exists and what a handoff must
  declare about it.
- **Where large payloads physically live.** `task_graph` spec §8.2 leaves this
  open deliberately; §6 below narrows it slightly by requiring the digest, which
  a content-addressed store would supply for free.

---

## 2. Design principles

| # | Principle | Consequence |
|---|---|---|
| 1 | **A handoff is standardised, or it is not a handoff** | What it contains and where each thing sits is fixed by the kind. This is what lets a validator be a filled-in template rather than bespoke code |
| 2 | **Checkable by construction** | A kind with no validator cannot be admitted. §5 |
| 3 | **Local-independent** | A handoff names its dependencies and contains no local paths, so any suitable machine can consume it. §7 |
| 4 | **Tamper-evident** | Runtime output carries a digest. §3.2 |
| 5 | **The kind is the contract; the slot is the instance** | Two layers, one join point, no duplicated specification. §1 |

---

## 3. What a handoff carries

### 3.1 Required fields

Every handoff version carries the following. Fields already specified on
`HandoffVersion` (`task_graph` spec §3.1) are marked and not respecified.

| Field | Type | Meaning |
|---|---|---|
| `kind` | `str` | The handoff kind. Resolves in the registry (§8). This is `Handoff.type` |
| `date` | `datetime` | When this version was produced. Already carried as `HandoffVersion.timestamp` |
| `digest` | `str` | `sha256` over the content. §3.2 |
| `version` | `int` | Already carried by `HandoffVersion` |
| `content` | `Content` | §3.3 |
| `validators` | `list[str]` | Validator spec names bound to this kind. §5 |
| `dependencies` | `Dependencies` | §7 |

`producer_task_id`, `producer_agent_id`, and `status` are also on every version
and are specified in `task_graph` spec §3.1.

### 3.2 The digest is `sha256`, and the choice is deliberate

The task definition asks for "md5/hash/any one" tamper-evidence mechanism. This
spec picks one, because a field whose algorithm varies per producer is not a
mechanism — a consumer would have to be told which one to use, and that
negotiation is a bigger cost than the choice.

`sha256`, because it is in the standard library, is not collision-broken, and is
what a content-addressed store would use if §1.2's open question is ever closed
in that direction.

**The digest covers the content, not the metadata.** Recomputing it must be
possible from what a consumer holds, and a consumer holding a copy in its
playground does not hold the producer's timestamp. Precisely: the digest is over
the canonical serialisation of `content`, and the canonicalisation rule is a
design-stage decision recorded in `§11` as an open item — but the *scope* of the
digest is specified here, because getting it wrong makes the field useless.

**What it is and is not for.** It detects accidental corruption and casual
tampering, and it makes "is this the same artefact I validated?" answerable.
It is not a security boundary: an agent that can write a handoff can write its
digest too. The boundary that matters is §5.2 of the main spec — the producer
cannot grade its own output — and no hash substitutes for it.

### 3.3 Content is executable, result, or both

```
content
  ├─ executable :  null | command | recipe-and-scripts
  └─ result     :  result_schema + result_content
```

| Part | Shape | Meaning |
|---|---|---|
| `executable` | `null` | This handoff carries no way to run anything |
| | `command` | A single command line, with its declared environment |
| | `recipe` + `scripts` | A named procedure plus the files it needs |
| `result` | `result_schema` | The schema the result content conforms to |
| | `result_content` | The result itself, or a reference to it |

The split is what makes a handoff replayable. A `result` alone says what
happened; an `executable` alone says how to make it happen again; the pair is
what lets a downstream consumer *reproduce* rather than trust — which is §2
principle 1 of the main spec, made concrete.

**A handoff may carry either part alone.** A knowledge handoff (§4) typically has
no `executable`; a run-method handoff may be produced before it has ever been
run, and carries no `result` until it has.

**`result_schema` is required whenever `result_content` is present.** A result
whose shape is undeclared cannot be checked by a filled-in template, which
defeats §2 principle 1.

---

## 4. Scope and lifecycle tags

A closed vocabulary. A handoff carries exactly one tag.

```
fixed                          declared by the recorded task graph
  ├─ required                  the task cannot run without it
  └─ optional                  the task runs; behaviour may differ

addons                         injected, outside the declared interface
  ├─ temp                      run-local: scratch, intermediate, this-run-only
  └─ knowledge                 long-lived accumulated experience
```

### 4.1 What each tag changes

| Tag | Storage | Permission | Retention |
|---|---|---|---|
| `fixed.required` | Handoff storage | Named in the consuming agent's permission list | Kept for the life of the graph run, and archived with it |
| `fixed.optional` | Handoff storage | Same | Same |
| `addons.temp` | Playground | Reachable by the injecting and receiving agent only | Discardable when the run ends |
| `addons.knowledge` | Handoff storage, in a long-lived domain | Broadly readable; narrowly writable | Outlives every run. Versioned like any other handoff |

The interesting row is the last one. A knowledge handoff is **the mechanism by
which expert experience enters an agent's context** — deployment guides, cluster
conventions, hard-won pitfalls, few-shot examples. It is a handoff and not
configuration precisely so that it is versioned, digested, and checkable like
everything else.

`env_mgr` is itself a consumer of knowledge handoffs: its cluster-usage and
deployment conventions arrive that way rather than being hard-coded, which is why
[`../../env_mgr/docs/spec.md`](../../env_mgr/docs/spec.md) §7 names a small set
of system-level tasks that produce them.

### 4.2 Why `addons` exists at all

A recorded graph declares its interface, and a fixed interface is the point of
the system. `addons` is the escape valve: a user or another agent can inject
something that helps without amending the recording.

The constraint that keeps it honest: **an `addons` handoff cannot satisfy a
`fixed.required` input.** If it could, the declared interface would be advisory.
A task blocked on a missing required input stays blocked; injecting an addon does
not unblock it.

---

## 5. Every handoff kind carries at least one validator

**A kind with no validator cannot be admitted to the registry.** This is the
system's second principle (main spec §2) reduced to a load-time check.

### 5.1 The escape hatch is a flag, and it is loud

A command-line flag permits running with unvalidated handoff kinds. Three
constraints on it:

- It is **off by default**.
- Every kind it lets through is **reported**, by name, at startup and in the run
  record. A silent bypass would make the guarantee unfalsifiable.
- It does not disable existing validators; it only permits absent ones.

It exists for bring-up — writing a new handoff kind before its validator — and
for debugging. It is not a mode anything ships in.

### 5.2 Admitting a kind is a review of coverage, not of presence

When a new handoff kind is submitted, the reviewable artefact is not "does it
have a validator" — that is checked mechanically. It is **how much of the kind
the validators actually cover.**

The review asks, and the answer is recorded with the kind:

| Question | Why |
|---|---|
| Which fields of `content` does no validator read? | An unread field is unchecked, whatever the count of validators says |
| Which failure modes would pass? | The useful question. A schema check catches shape, not substance |
| Is each validator `strong` or `weak`, and is that honest? | The taxonomy is only useful if the labels are accurate. See validator spec §5 |
| Does any validator depend on the producer? | If it does, §5.2 of the main spec is violated and the kind is rejected |

The kickoff appendix supplies the failure this guards against directly: a
correctness gate whose tolerance table permitted 10% of elements to mismatch and
still reported PASS. It had a validator. Its coverage was the problem.

### 5.3 The binding is recorded on both sides

A handoff spec names its validators; a leaf validator names the handoff kinds it
binds to. The redundancy is deliberate and is specified — including which side
wins on conflict — in [`../../validator/docs/spec.md`](../../validator/docs/spec.md) §4.

---

## 6. Where the payload lives

`task_graph` spec §8.2 leaves payload storage deliberately open, and this
document does not close it. It adds one constraint:

**Whatever holds the payload must be able to return it byte-identical, and the
digest is how that is checked.** A store that normalises, re-encodes, or
re-orders content on the way through breaks the digest and is therefore not a
valid store for this system.

The consequence for the design stage: an inline JSON payload works, a filesystem
path works, a content-addressed store works and would supply the digest for
free — and a store that rewrites content does not.

---

## 7. Locality independence

**A handoff contains no local information.** It declares its hardware and
software dependencies, its container image if it needs one, and nothing about the
machine that produced it. Any machine satisfying the declaration can consume it.

| `dependencies` field | Example |
|---|---|
| `hardware` | GPU model and count, interconnect, memory floor |
| `software` | Framework and version, driver floor |
| `image` | A container image reference |
| `resources` | The pool names and amounts a consuming task should declare |

This is what makes replay meaningful. A handoff carrying `/home/someone/run3/`
is a record of one machine's afternoon, not a transferable artefact — and the
system's first principle is that a result that cannot be re-obtained is not a
result.

**The consumption protocol follows from this** and is specified in
[`../../env_mgr/docs/spec.md`](../../env_mgr/docs/spec.md) §6: an agent copies a
handoff from storage into its playground and works on the copy. It never edits
the stored artefact, and it never depends on the storage path.

---

## 8. The spec file and the registry

### 8.1 The YAML file

A handoff kind is declared in a YAML file, in the predefined-spec folder,
constrained by a JSON Schema. It declares at minimum:

| Key | Meaning |
|---|---|
| `name` | The kind name. Unique in the registry; this is what `Handoff.type` holds |
| `description` | What this kind of artefact is, for a human |
| `scope` | One of the four §4 tags |
| `content_schema` | The schema `content` conforms to, including `result_schema` when a result is expected |
| `validators` | Validator spec names. At least one, unless the §5.1 flag is set |
| `dependencies` | §7 |

Schema-constrained rather than free-form, because §2 principle 1 requires the
kind to be a contract, and a contract nobody checks is a comment.

### 8.2 The registry

Name → spec, over the predefined-spec folder. It is one of the four independent
registries (main spec §4.1). It answers:

- Which kinds exist.
- Which validators a kind is bound to.
- Which kinds a given validator covers — the reverse index, maintained because
  §5.2's review needs it and scanning every kind to answer it does not scale.

**Load-time checks**, all of which fail loudly with the file path in the message:

1. The YAML validates against the schema.
2. The name is unique.
3. Every named validator resolves in the validator registry.
4. At least one validator is named, unless the §5.1 flag is set — in which case
   the kind is admitted and reported.
5. The two-way binding agrees with the validator registry's side of it
   (validator spec §4).

---

## 9. Data flow

Nothing here is new; it is the two layers seen together once.

```
declare        task_graph creates the slot, v0 CREATED
               (scheduler; task_graph spec §6.1)

produce        agent opens v_n GENERATING, writes content,
               computes the digest, seals
               (agent, via its runner; task_graph spec §3.1)

validate       a validator task runs, reads the handoff, and records
               the verdict against the kind's declared validators
               (validator spec §4; a separate context — main spec §5.2)

consume        a downstream agent copies the handoff into its playground
               and works on the copy
               (env_mgr spec §6)
```

The scheduler appears once, at `declare`, and thereafter only reads. That is main
spec §5.1, and it is why this diagram has no arrow from the scheduler to
`produce` or `validate`.

---

## 10. Acceptance criteria

1. A handoff spec that omits a required key, or whose `content_schema` is not a
   valid schema, is rejected at load with the file path and the offending key in
   the message.
2. A handoff kind naming no validator is rejected, unless the escape-hatch flag
   is set — in which case it loads **and** its name appears in the startup report
   and the run record.
3. A handoff kind naming a validator that does not resolve is rejected, and the
   message names both.
4. The registry's reverse index answers "which kinds does validator V cover"
   without scanning every kind.
5. Recomputing `sha256` over a consumed handoff's content reproduces its
   `digest`, after a round trip through storage and a copy into a playground.
6. A handoff whose content declares an absolute local path fails its
   locality-independence check.
7. Each of the four scope tags produces the storage, permission, and retention
   behaviour §4.1 tabulates — verified by asserting where the artefact lands and
   who can read it, not by reading the tag back.
8. An `addons` handoff does not satisfy a `fixed.required` input: the consuming
   task stays in `WAITING_HANDOFF`.
9. A handoff carrying `executable` but no `result` is legal and loads; so is the
   reverse. A `result_content` with no `result_schema` is rejected.
10. The two-way validator binding is consistent after loading both registries,
    and a deliberate mismatch is reported rather than silently resolved.

---

## 11. Open questions

| Item | Status |
|---|---|
| **Digest canonicalisation** | §3.2 fixes the algorithm and the scope but not the serialisation. Key order, float formatting, and line endings all move the digest. JSON with sorted keys is the obvious answer; it is a design-stage decision and is called out here so it does not get made by accident |
| **Payload storage** | Inherited from `task_graph` spec §8.2 and narrowed by §6 only. Still open |
| **`addons.temp` retention** | "Discardable when the run ends" is not "discarded". Nothing sweeps a playground, and an agent's playground is part of what makes it resumable (env_mgr spec §5), so the sweep cannot be unconditional |
| **Knowledge handoff provenance** | A knowledge handoff outlives every run, so `producer_task_id` may point at a task that no longer exists in any live graph. Whether that reference must remain resolvable — and what archives it if so — is unspecified |
| **Cross-handoff schema references** | A validator that checks two handoffs against each other (validator spec §3) needs to name fields in both. Whether `content_schema` may `$ref` another kind's schema is undecided; permitting it couples kinds, forbidding it duplicates definitions |
| **Version retention** | Inherited from `task_graph` spec §10: nothing says when an old version's payload may be discarded, and re-running a producer is unbounded |
