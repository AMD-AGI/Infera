# Closure — Specification

| | |
|---|---|
| Status | Draft, revised after review |
| Revision | 10 — 2026-08-29. **`agent` becomes leaf-only, and a closure is YAML.** Rev. 8 made `agent` required of every task; that was wrong and the review did not catch it. A non-leaf has no executor — its work *is* its subgraph (§2.1) — and the name it was forced to invent was read by nothing: `agent/runner.py:682` returns before `_deploy`, so `env_mgr/prepare.py:447`'s `material.deploy(agent_spec, zone)` is never reached for one, and output validation already falls through to the GLOBAL row on `producer=None` (`validator/environment.py:140`). §2's key table, §2.2, §2.3's `agent_of`, §4 check 4 and criterion 3 all narrow together. Separately, a closure is a **YAML document in a task package** rather than a jsonnet source: main spec §4.4 deleted the render step after measuring that its templating was unused. Two smaller consequences: §2.5's subgraph entries now carry a required **`froms`** (main spec §4.4, `task.schema.json`), and §4 gains check 9 for it. (rev. 9: 2026-08-27. **The user-interface brief.** A task spec gains **`body`** — always a `readme.md`, plus an `entry.sh` when programmatic — which is how a package author states what the task *is*, what to run, and how (§2.5, §2.6). `goal` shrinks to one sentence of at most 100 characters. `materials`, `repos` and `monitor` join the key list. `entry.sh` and a subgraph are mutually exclusive; `readme.md` is required of every task, leaf or not. (rev. 8: 2026-08-27. **Every task has an agent; what varies is its `kind` — `ai`, `human` or `program`.** `agent` becomes a required key and criterion 3 rejects a closure that names none (§2.2). Closes a question four design modules had deferred. (rev. 7: 2026-08-26. Consistency pass: a leaf binds to an *executor* runtime, which may be a program (§2.1, reconciling with main spec §4.8). (rev. 6: 2026-08-26. The task spec's key list names `goal`, reconciling it with `agent` spec §1 (§2). (rev. 5: 2026-08-26. Parameterised closures merged into main spec §10's runtime-fan-out question. (rev. 4: A closure is a jsonnet source in a task package (§2). rev. 3: A closure carries no version; each of the four member specs carries its own (§1.2). rev. 2: Review of PR #132: a closure is a spec-level / code-management artefact — a load checker plus read-only query helpers, and nothing at runtime. rev. 1: initial)))))) |
| Date | 2026-08-24 |
| Scope | The predefined binding of a task's handoffs, its agent, and its validators |
| Source | The task definition §3.3 |
| Part of | [`../../docs/spec.md`](../../docs/spec.md) — the whole-system specification |
| Depends on | [`../../handoff/docs/spec.md`](../../handoff/docs/spec.md), [`../../validator/docs/spec.md`](../../validator/docs/spec.md), [`../../agent/docs/spec.md`](../../agent/docs/spec.md), [`../../task_graph/docs/spec.md`](../../task_graph/docs/spec.md) |

---

## 1. What a closure is

A task spec and its handoff specs are strongly bound in practice: a task's
handoffs are not reusable across arbitrary tasks, and a handoff kind exists
because some task produces or consumes it. The agent spec that executes the task
and the validators that check the handoffs complete the group.

**A closure is that group of four, named and predefined:**

```
closure = < handoff spec set, task spec, agent spec, validator set >
```

**It is a spec-level, code-management artefact.** Concretely, it is three things
and nothing more:

| | |
|---|---|
| **A composition of four parts** | The table above |
| **A load checker** | §4. It refuses an incoherent group at load |
| **Read-only query helpers** | §3.2. Convenience over the four registries |

It is predefined because the system does not support dynamic task specs (main
spec §6), so the handoff specs cannot be dynamic either. Everything the system
will ever run is declared before it runs.

### 1.1 Nothing at runtime

**A closure is consulted when a graph is assembled, and never again.** The
scheduler does not read one, no manager holds one, and no runtime object points
at one.

This is what keeps it from becoming a fifth object. The closure is the one place
that sees all four parts, so every cross-object question wants to live here;
resisting that is the design. The rule:

> **If a rule can be expressed against one object, it belongs to that object.**
> The closure records which objects go together, and checks that the group is
> coherent.

| Question | Owner | Why not the closure |
|---|---|---|
| Does this handoff kind have a validator? | handoff spec §5.3 | A property of the kind, true whatever closure uses it |
| Is this validator `strong` or `weak`? | validator spec §5 | Same |
| May this task's executor read that path? | `task_graph` spec §3.2.2 | Permissions are a versioned task attribute |
| When does this task run? | `task_graph` spec §6 | The scheduler decides when, and never reads a closure |
| Which revision of this step is this? | each of the four specs | Each spec carries its own `version` (§1.2) |
| **Does this task's agent spec exist, and do its handoff kinds resolve?** | **closure** | A statement about the group, and nothing else sees the whole group |

### 1.2 A closure has no version of its own

**The four member specs each carry a `version` string; the closure does not.**

The temptation is the reverse: a closure is the thing that names a whole workflow
step, so versioning *it* looks like versioning the step. It is not taken, for the
same reason §1.1 gives — a closure that carried a version would be a fifth object
with state, and a run would then have two answers to "what was this made of": the
closure's version, and the four members' versions, which can disagree.

The member versions are the single answer. `Execution` already pins the agent and
the handoff versions it actually used (`task_graph` spec §3.3), and those are
runtime facts; the spec `version` is a different thing and is explicitly **not**
runtime state:

> A spec `version` exists to make maintenance friendly — so a reviewer can see
> that a handoff kind changed between two branches, and so a changelog has
> something to key on. Nothing at runtime reads it, nothing pins to it, and
> loading two versions of one spec at once is not supported.

---

## 2. What a closure records

A **YAML document in a task package**, validated against the closure JSON Schema
(main spec §4.3, §4.4). Through rev. 9 it was a jsonnet source rendered to YAML;
main spec §4.4 removed the render after measuring that nothing used the language
it required.

**Users write `task`, never `closure`.** A package author declares a
`module: task` document; the closure and the task spec are produced from it
internally, which is what `closure/check.py:709` already does by splitting one out
of the other. `closure` is the name of the group, not a thing anyone types.

| Key | Meaning |
|---|---|
| `name` | Unique. By convention it names the workflow step: `prepare_e2e`, `collect_trace` |
| `description` | What this step is, for a human |
| `task` | The task spec — §2.5. `goal`, `body`, `materials`, `repos`, `monitor`, inputs, outputs, resources, permissions, its own `version`, and — if it has one — its subgraph |
| `agent` | The agent spec name that executes it. **Required of a leaf, and of nothing else**; where there is one, its `kind` may be `ai`, `human` or `program` (§2.2) |
| `handoffs` | The handoff kinds the task names, listed explicitly |
| `validators` | The validators for this task's input and output validation phases |

### 2.1 A task contains a task graph, or nothing

Restating `task_graph` spec §3.2.1 from the closure's side, because it is what
decides whether a closure has a subgraph at all:

1. **A task contains a task graph, or nothing** — in which case it is itself, a
   leaf task.
2. **Every leaf task is bound to an executor runtime.** Where that executor is an
   AI agent, the binding may choose among several agent specs or backends at run
   time; where the task's agent is `kind: program` (§2.2), the executor is that
   program.
3. **Every task has a monitor.** A thread, a thread with an AI in it, or a slot
   in a global round-robin monitor. Specified in
   [`../../task_graph/docs/spec.md`](../../task_graph/docs/spec.md) §3.5, and
   named per task by `monitor` (§2.5); the alpha ships a simple pusher and
   [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §2 keeps the analysing
   dispatcher.

### 2.2 `agent` is required of a leaf, and its `kind` is what varies

**A leaf closure always names an agent spec, and the load checker demands one.**
What varies is the agent's `kind` (main spec §4.8, `agent` spec §3.1): `ai`,
`human`, or `program` — an executable, a script, or a thread.

A `kind: program` agent is not a degenerate case. A large part of the reference
workflow is running a command someone already wrote, and wrapping that in an
*AI* would add cost and non-determinism for nothing. Wrapping it in an agent
**spec** costs a name and a command line, and it is what keeps one dispatch
path, one binding record, and one answer to "what ran this task".

#### A non-leaf names none, and rev. 8 was wrong to demand one

Rev. 8 made `agent` unconditionally required and closed a question four design
modules had deferred. It closed it the wrong way. A task contains a task graph or
it is a leaf that does the work itself (§2.1), so **a non-leaf has no executor to
name** — and the name it was nonetheless forced to invent was read by nothing.
Three code reads rather than an argument:

| Claim | Evidence |
|---|---|
| A non-leaf never deploys its agent spec | `agent/runner.py:682` returns before `_deploy`, so `env_mgr/prepare.py:447`'s `material.deploy(agent_spec, zone)` is never reached for one |
| Its zone stages nothing to deploy into | `_place_container_zone` "confines nothing, cuts no workspace, stages nothing" (`agent/runner.py:678,687`) |
| Output validation is already correct without one | `producer=None` falls through to the GLOBAL row (`validator/environment.py:140`) — an existing, exercised path, not a new fallback |

So the requirement was costing every package author a decision with no
consequence, and the usual defence of a redundant field — that it is *checked
against* something (§2.3) — does not apply, because there was nothing to check it
against.

**Dropping it removes a field and deletes no behaviour.** What replaces it is the
same rule with a leaf-only qualifier, in the same two places it lived before:

| Writer | What it now says |
|---|---|
| `closure.schema.json` | `agent` leaves the top-level `required`; an `if`/`else` pair reinstates it unless `task.subgraph` is present and non-empty. An **empty** subgraph counts as a leaf, which is the safe direction — it demands the name rather than waiving it |
| `closure/check.py` check 4 | The "names no agent spec" branch becomes conditional on the closure being a leaf. The message, which lists the known agent specs, is unchanged |

Two writers for one fact is deliberate here and predates this revision: the schema
catches it for any document that reached the schema, and the check catches it for
any document that did not (§4). **Both must carry the qualifier or they
disagree**, and a closure that one accepts and the other rejects is worse than
either rule alone.

### 2.3 Why `handoffs` is listed rather than derived

It is derivable — the task's `inputs` and `outputs` name the kinds. Listing it
anyway is the trade `task_graph` spec §3.2 already makes for `depends_on`: the
redundant field is *checked against* the derivable one rather than replacing it,
and the check catches the mistake that actually happens — adding an input and
forgetting to bring its kind into the closure.

The check is one-directional (§4): every kind the task names must appear in
`handoffs`. A kind in `handoffs` that the task does not name is legal, because a
closure may declare a kind its subgraph uses internally.

### 2.4 Why `validators` is here and not only on the handoffs

A handoff kind names its own validators, and those run when the handoff is
produced. The closure's `validators` are the **phase** validators — the checks
that run in this task's `input_validation` and `output_validation` phases
(validator spec §3). They are a property of the task, not of any one handoff
kind, which is why the handoff specs cannot carry them.

### 2.5 What a task spec carries, and why `body` is the important one

| Key | Required | Meaning |
|---|---|---|
| `goal` | **yes** | **One sentence, at most 100 characters.** What this task is *for*, for a human deciding whether they want it. The limit is a global configuration value, not a per-package one |
| `body` | **yes** | **What this task actually is** — §2.6 |
| `materials` | no | Things this task may need for itself. The system wraps them into a handoff; the author is free about what goes in |
| `repos` | no | The dependency repositories this task's work needs — `sglang`, `mooncake`, `aiter`. **Per task**, unlike the main repository, which is one per run (`env_mgr` spec §6.1) |
| `monitor` | no | Which monitor loop watches it, by name. Absent takes the global default |
| `inputs` / `outputs` | yes | Handoff **kind names**. What the task consumes and produces |
| `resources`, `permissions`, `version` | — | As before |
| subgraph | no | As before, except that **every entry now carries `froms`** — §2.7 |

**`goal` and `body` are not two names for one thing, and the split is the point.**
A goal is a sentence; a body is an artefact. A hundred characters of prose cannot
tell a system what to run or how, and a shell script cannot tell a reviewer why
the step exists.

### 2.6 `body` — how a user says what the task *is*

Without a body there is no channel through which a package author states a task's
semantics: what the work is, what to execute, and how to execute it. A `goal`
does not carry it, and nothing else in this document did.

```
body
  ├─ readme.md        ALWAYS required
  └─ entry.sh         required iff the body is programmatic
```

**`readme.md` is required of every task, leaf or not.** For an agent task it *is*
the body — the description the agent works from. For a programmatic task it is
still required, because a task nobody can read is a step nobody can review, and
that holds for a non-leaf too.

**`entry.sh` is the exact execution entry point**, and it is what makes a
programmatic task programmatic. It may run something on the system or run material
a handoff supplied.

**`entry.sh` and a subgraph are mutually exclusive.** A task contains a task
graph, or it is a leaf that does the work itself (§2.1); a non-leaf's work *is*
its subgraph. `readme.md` is required either way, so the exclusion is between
`entry.sh` and the subgraph — not between `body` and the subgraph.

**This is the same shape a validator has** (validator spec §6), and deliberately:
a validator is a special kind of task, so "what does this thing do and how do I
run it" should have one answer in the system, not two.

### 2.7 `froms` — a subgraph entry states its own predecessors

Every entry in a task's subgraph carries **`froms`**: the entries it depends on,
by their `closure` name, **required even when empty**. `[]` is how an entry says
it has no predecessor; omitting the key is how an author forgets to think about
it, which is why the schema demands it rather than defaulting it.

**The edge already exists and is derived.** `task_graph/models.py:560-569` walks
each entry's input kinds and takes the producer recorded in `available[kind]`, so
today the graph is a consequence of handoff wiring plus list order, and nothing in
the spec says what it is. That derivation is **kept**, and `froms` is checked
against it; a mismatch is an error naming both.

**This is the trade §2.3 already makes for `handoffs`**, and for the same reason.
The redundant field is not a second source of truth — it is a second statement of
one, positioned to catch the mistake that actually happens. For `handoffs` that
mistake is adding an input and forgetting to declare its kind; here it is wiring a
handoff and not noticing an edge appeared, or removing one and not noticing an
edge vanished.

Unlike `handoffs`, the check is **two-directional**. A declared kind the task does
not name is legal because a subgraph may use it internally (§2.3); there is no
corresponding reading of a declared edge that no handoff supports — except the one
case that is the point of the field:

> **`froms` can express a dependency that shares no handoff, and derivation
> cannot.** `task_graph/scheduler.py:639`'s `_warn_depends_on` warns rather than
> rejects precisely so `depends_on` may hold an edge no input accounts for. Until
> now there was nowhere to *declare* such an edge; `froms` is that place.

**Listing order must be a valid topological order**, so a name in `froms` must
refer to an entry earlier in the list. That constraint is this repository's and is
not adopted: Argo puts every edge in the task's own `dependencies` and resolves by
name with no ordering requirement, and Tekton derives from data flow and keeps
`runAfter` for the rest. It is taken anyway because a subgraph is read top to
bottom by a human far more often than it is executed, and a list whose order means
nothing is a list that has to be traced to be understood.

---

## 3. Naming, lookup, and queries

### 3.1 Lookup

By name, through the closure registry. Whoever assembles a graph looks one up;
nothing else does.

The closure registry is **not** one of the four (main spec §4) — a closure is not
one of the four objects. It is a table over them.

### 3.2 Read-only query helpers

The registry offers convenience queries over the four registries, so a caller
does not hand-join them:

| Query | Answers |
|---|---|
| `handoff_kinds(closure)` | Every kind this closure touches, inputs and outputs |
| `validators_for(closure)` | Every validator that will run, phase validators and per-handoff ones together |
| `closures_using(handoff_kind)` | Which closures a kind appears in — the reverse index |
| `closures_using(agent_spec)` | Same for an agent spec |
| `agent_of(closure)` | The agent spec, or `None` for a non-leaf. **`None` is a real answer, not a missing one** — §2.2 requires a name only of a leaf, so a caller that treats `None` as an error would reject every valid subgraph parent. Rev. 9 promised "never `None`" and that promise is withdrawn |

**All read-only.** No helper mutates a registry, and none is called at run time.
The last two exist because "what breaks if I change this" is otherwise a scan of
every closure — which is exactly the question a code-management artefact should
answer.

---

## 4. Well-formedness

A closure is well-formed or it is rejected at load. Each check fails with the
closure file path and the offending value:

1. **The YAML validates against the schema**, and the name is unique.
2. **Every handoff kind the task names resolves** in the handoff registry, and
   appears in the declared `handoffs` (§2.3).
3. **Every handoff kind has at least one validator** — enforced by the handoff
   registry itself, and re-asserted here so a closure assembled from kinds
   admitted under the escape-hatch flag reports that it was.
4. **A leaf names an agent spec, and any name given resolves** (§2.2). Two
   distinct failures: a leaf with no name at all, and a name — leaf or not — that
   does not resolve. A non-leaf with no name is not a failure.
5. **Every phase validator resolves**, and each is a validator rather than a
   general task.
6. **The task's permissions cover its handoffs.** A task that may not read an
   input it is given, or may not write an output it must produce, is a graph that
   deadlocks at run time for a reason visible at load time.
7. **`entry.sh` and a subgraph are not both present** (§2.6).
8. **Every subgraph entry names a declared closure.**
9. **Every subgraph entry's `froms` matches the derived edges, and refers
   backwards** (§2.7). Three failures, and they are different: a declared
   predecessor the derivation did not produce, a derived predecessor `froms` does
   not declare, and a name referring to an entry that appears later in the list or
   not at all. Each names the entry and the edge.

Check 6 is the one only a closure can perform: it needs the task's handoffs and
its permissions together, and neither registry sees both.

Checks 7 and 8 were already implemented and unlisted here; they are written down
at rev. 10 rather than added.

### 4.1 What is deliberately not checked

**Whether the closures compose into a valid graph.** A closure is one step; a
graph is many. Cycle detection, reachability, and whether every input has a
producer are graph-level questions — `task_graph` spec §10 already carries cycle
detection as an open item. A partial version here would put the check in two
places and satisfy neither.

---

## 5. Acceptance criteria

1. A closure naming a handoff kind that does not resolve is rejected at load,
   with the kind name in the message.
2. A closure whose task names an input absent from its declared `handoffs` is
   rejected; the reverse — a declared kind the task does not name — loads.
3. A closure naming an agent spec that does not exist is rejected, and **so is a
   LEAF naming no agent spec at all**; a `kind: program` spec is how a task runs a
   plain executable. **A non-leaf naming none loads**, and one that names an agent
   anyway is still checked for resolution. *Amended at rev. 10: it said "so is a
   closure naming no agent spec at all", which rev. 8 had made unconditional. §2.2
   records the three code reads that show a non-leaf's `agent` is read by nothing.
   The criterion needs four cases tested, not two — leaf/non-leaf crossed with
   named/absent — because the change turns one of the four from reject to accept
   and a test suite that only covers the leaf column would pass either way.*
4. A closure naming a phase validator that resolves to a general task is
   rejected.
5. **A closure whose task permissions do not cover its handoffs is rejected at
   load**, naming both the handoff and the missing permission — rather than
   deadlocking at dispatch.
6. A closure assembled from a handoff kind admitted under the escape-hatch flag
   loads, and reports that it did.
7. Two closures may share a handoff kind, an agent spec, and a validator; none is
   exclusive to one closure.
8. **The scheduler never reads a closure.** Verified the way criterion 14 is: a
   spy over the closure registry records no read from a scheduler frame across a
   full submit → dispatch → complete cycle.
9. Every read-only query in §3.2 answers without a caller-written join, and none
   mutates anything.
10. The six reference-workflow steps are each expressible as one closure, and the
    set loads without error (main spec criterion 7).
11. **A closure declares no version, and each of its four members declares its
    own.** A `version` key on a closure is rejected at load; nothing at runtime
    reads a member's `version` (§1.2).
12. **A subgraph entry's `froms` is required, is cross-checked, and must refer
    backwards** (§2.7, new at rev. 10). An entry with no `froms` key is rejected;
    `froms: []` is accepted for an entry with no predecessor; an entry declaring a
    predecessor the derivation did not produce is rejected naming both, and so is
    one omitting a predecessor the derivation *did* produce; and a name referring
    to a later entry is rejected naming the entry and the edge. **The derivation
    stays the reference** — `task_graph/models.py:560-569` is not replaced, and a
    test that only checks `froms` against itself would pass with the derivation
    deleted.

---

## 6. Open questions

| Item | Status |
|---|---|
| **Graph-level composition** | §4.1 leaves it out on purpose, and it has to live somewhere. The likely home is the system whole task, which is the only thing that sees every closure in a graph |
| **Parameterised closures** | **Merged into main spec §10's "runtime fan-out" row.** They were one question: instantiating a declared closure N times with different inputs is what both describe. The framing is settled in main spec §6.1 — the catalogue is static, the instance count is not — and what remains undecided is carried there |
| **Change propagation** | Criterion 7 permits sharing, and §3.2's reverse indexes answer *who is affected* when a shared kind changes. What nothing answers is whether the change is safe — that needs a diff of the kind's schema, not a list of users |
