# Closure — Specification

| | |
|---|---|
| Status | Draft, pending review |
| Revision | 1 — 2026-08-24 |
| Date | 2026-08-24 |
| Scope | The predefined binding of a task's handoffs, its agent, and its validators |
| Source | The task definition §3.3 |
| Depends on | [`../../handoff/docs/spec.md`](../../handoff/docs/spec.md), [`../../validator/docs/spec.md`](../../validator/docs/spec.md), [`../../agent/docs/spec.md`](../../agent/docs/spec.md), [`../../task_graph/docs/spec.md`](../../task_graph/docs/spec.md) |

---

## 1. Purpose

A task spec and its handoff specs are strongly bound in practice: a task's
handoffs are not reusable across arbitrary tasks, and a handoff kind exists
because some task produces or consumes it. The agent spec that executes the task
and the validators that check the handoffs complete the group.

**A closure is that group, named and predefined:**

```
closure = < handoff spec set, task spec, agent spec, validator set >
```

It is predefined because the system does not support dynamic task specs (main
spec §6), so the handoff specs cannot be dynamic either. Everything the system
will ever run is declared before it runs.

### 1.1 It supplies no semantics of its own

**A closure is an abstract table — a wrapper — and nothing more.** This is the
task definition's own framing and the spec keeps it that way deliberately.

The temptation is obvious: the closure is the one place that sees all four
objects, so every cross-object question wants to live here. Resisting it is the
point. A closure that grew behaviour would become a fifth object with a fifth
registry, and the four-object symmetry (main spec §4) would be a lie.

The rule, stated so it can be applied without argument:

> **If a rule can be expressed against one object, it belongs to that object.**
> The closure only records which objects go together, and checks that the group
> is coherent.

Worked examples of the boundary:

| Question | Owner | Not the closure because |
|---|---|---|
| Does this handoff kind have a validator? | handoff spec §5 | It is a property of the kind, true whatever closure uses it |
| Is this validator `strong` or `weak`? | validator spec §5 | Same |
| May this agent read that handoff? | agent spec §3.2 | The permission list is the agent's |
| When does this task run? | `task_graph` spec §6 | The scheduler decides when, and never reads a closure |
| **Do this task's declared inputs all resolve to kinds that exist?** | **closure** | It is a statement about the group, and nothing else sees the whole group |

### 1.2 Out of scope

- **Everything each object already specifies.** §1.1.
- **Runtime.** A closure is consulted when a graph is assembled, not while it
  runs. The scheduler never reads one.

---

## 2. What a closure records

A YAML file in the predefined-spec folder, constrained by a JSON Schema.

| Key | Meaning |
|---|---|
| `name` | The closure name. Unique |
| `description` | What step of the workflow this is, for a human |
| `task` | The task spec: its inputs, outputs, resources, and — if it has one — its subgraph. See [`../../task_graph/docs/spec.md`](../../task_graph/docs/spec.md) §3.2.1 |
| `agent` | The agent spec name that executes it |
| `handoffs` | The handoff kinds the task names, listed explicitly |
| `validators` | The validators that run in this closure's `input_validation` and `output_validation` phases |

### 2.1 Why `handoffs` is listed rather than derived

It is derivable — the task's `inputs` and `outputs` name the kinds. Listing it
anyway is the same trade `task_graph` spec §3.2 makes for `depends_on`: the
redundant field is checked against the derivable one rather than replacing it,
and the check catches the mistake that actually happens — adding an input and
forgetting to bring its kind into the closure's declared set.

The check is one-directional and stated in §4: every kind the task names must
appear in `handoffs`. A kind in `handoffs` that the task does not name is legal
and unchecked, because a closure may legitimately declare a kind its subgraph
uses internally.

### 2.2 Why `validators` is here and not only on the handoffs

A handoff kind names its validators (handoff spec §8.1), and those run when the
handoff is produced. The closure's `validators` are the *phase* validators — the
cross-handoff input checks and the output checks that run in this task's
environment (validator spec §3.3). They are a property of the task, not of any
one handoff kind, which is why the handoff specs cannot carry them.

---

## 3. Naming and lookup

| | |
|---|---|
| **Name** | Unique across the registry. By convention it names the workflow step: `prepare_e2e`, `collect_trace`, `analyse_trace` |
| **Lookup** | By name, through the closure registry |
| **Who looks one up** | Whoever assembles a graph. Not the scheduler |

The closure registry is **not** one of the four (main spec §4) — a closure is not
one of the four objects. It is a table over them, and it is registered like any
other component so it can be resolved by name at use time.

---

## 4. Well-formedness

A closure is well-formed or it is rejected at load. Six checks, each failing with
the closure file path and the offending value in the message:

1. **The YAML validates against the schema**, and the name is unique.
2. **Every handoff kind the task names resolves** in the handoff registry, and
   appears in the closure's declared `handoffs` (§2.1).
3. **Every handoff kind has at least one validator** — enforced by the handoff
   registry itself (handoff spec §5), and re-asserted here so a closure cannot be
   assembled from kinds admitted under the escape-hatch flag without the closure
   also reporting it.
4. **The agent spec exists** in the agent registry.
5. **Every phase validator resolves** in the validator registry, and each is a
   validator, not a general task.
6. **The agent's permission list covers the task's handoffs.** An agent that may
   not read an input it is given, or may not write an output it must produce, is
   a graph that deadlocks at run time for a reason visible at load time.

Check 6 is the one only a closure can perform: it needs the task's handoffs and
the agent's permissions together, and neither registry sees both.

### 4.1 What is deliberately not checked

**Whether the closures compose into a valid graph.** A closure is one step; a
graph is many. Cycle detection, reachability, and whether every input has a
producer are graph-level questions, and `task_graph` spec §10 already carries
cycle detection as an open item. Adding a partial version here would put the
check in two places and satisfy neither.

---

## 5. Acceptance criteria

1. A closure naming a handoff kind that does not resolve is rejected at load,
   with the kind name in the message.
2. A closure whose task names an input absent from its declared `handoffs` is
   rejected; the reverse — a declared kind the task does not name — loads.
3. A closure naming an agent spec that does not exist is rejected.
4. A closure naming a phase validator that resolves to a general task, rather
   than a validator, is rejected.
5. **A closure whose agent's permission list does not cover its task's handoffs
   is rejected at load**, naming both the handoff and the missing permission —
   rather than deadlocking at dispatch.
6. A closure assembled from a handoff kind admitted under the escape-hatch flag
   loads, and reports that it did.
7. Two closures may share a handoff kind, an agent spec, and a validator; none of
   the three is exclusive to one closure.
8. **The scheduler never reads a closure.** Verified the way criterion 14 is
   verified: a spy over the closure registry records no read from a scheduler
   frame across a full submit → dispatch → complete cycle.
9. The six reference-workflow steps are each expressible as one closure, and the
   set loads without error (main spec criterion 7).

---

## 6. Open questions

| Item | Status |
|---|---|
| **Closure versioning** | A closure is the recording. Changing one changes what "the same workflow" means, and nothing records which version of a closure a past run used. The `Execution` record names the agent and the handoff versions (`task_graph` spec §3.2) but not the closure |
| **Graph-level composition** | §4.1 leaves it out on purpose, and it has to live somewhere. The likely home is the system whole task (`task_graph` spec §3.2.1), which is the only thing that sees every closure in a graph |
| **Parameterised closures** | The reference workflow runs the same six steps against different models and hardware. Whether that is six closures with runtime parameters, or a closure per configuration, is undecided — and the answer interacts with main spec §6's static-graph constraint, since a parameter that changes the graph's *shape* is a dynamic spec by another name |
| **Sharing between closures** | §5 criterion 7 permits sharing. Nothing says what happens when a shared handoff kind's spec changes: every closure using it is affected, and nothing enumerates them. The handoff registry's reverse index (handoff spec §8.2) answers the validator half of this; the closure half has no index |
