# Engineering principles

| | |
|---|---|
| Status | **Binding.** Read before writing any spec, design, or code in `agent_sys/` |
| Scope | Every module. No exceptions, and no "just this once" |

These are **MUST**s, not preferences. A design that violates one is wrong even if
it works, because the cost lands on whoever touches the code next — and that is
usually a different agent, in a different session, with none of the context that
made the shortcut look reasonable.

If a principle seems to block a requirement, **say so and stop**. Do not route
around it quietly. A stated conflict is a useful deliverable; a silent violation
is a defect that takes months to surface.

---

## 0. Spec first, and one home for every fact

**The order is `spec → design → code`, and it is an order of *decisions*, not a
schedule.** Each stage answers a question the next one is not allowed to reopen:

| Stage | Answers | Does not answer |
|---|---|---|
| **spec** | What must be true. The properties, guarantees, invariants, failure modes, and the acceptance criteria that would falsify them | How anything is built, or what anything is called |
| **design** | How it is built. Modules, types, the public surface, the call sequences, the libraries adopted and why | What the code looks like line by line |
| **code** | The rest | Anything the two above already settled |

Work backwards and the cost is not a bad document — it is a property nobody can
tell you was ever intended. **If implementation forces a change, change the
spec.** A design that quietly diverges is worse than one that says it deviates:
`interfaces.md` is normative for what crosses a module boundary precisely
because that is where divergence used to hide.

### 0.1 What each document is for

**One fact, one home.** A reader must be able to answer "where would this be
written down?" without opening anything. Duplication is not redundancy here — it
is a second thing to keep true, and it is always the copy that goes stale.

| Document | Holds | Never holds |
|---|---|---|
| `<module>/README.md` | **≤150 lines.** What the module is for, who uses it, how to call it, and a table of what is inside | Rationale, prior art, measurements, findings, history |
| `<module>/docs/spec.md` | The properties, and numbered acceptance criteria | An API, a file layout, a library choice |
| `<module>/docs/design.md` | The build: types, public surface, call sequences, adopted libraries and the reason for each. Its deviations from the spec, stated | Properties the spec should own |
| `docs/spec.md` | The whole system: architecture, the four objects, the authority boundaries, system-level criteria, the index of every component's criteria | Anything one module could own alone |
| `docs/design.md` | How the system is assembled, and the loader | — |
| `docs/interfaces.md` | **Normative for what crosses a module boundary**, and for nothing else. The composition root, the shared vocabulary, the permitted import edges, and the seams left deliberately open | Anything internal to one module |
| `docs/spec.<topic>.md` | A **cross-cutting spec**: a property that spans several modules and therefore belongs to none of their `docs/`. Its header names which modules it spans. `spec.provisioning.md` is the one that exists | A property one module could own alone |
| `docs/ROADMAP.md` | Long-term subsystems and deferred design questions | Near-term work |
| `docs/TODO.md` | Near-term decisions and pieces, each with what would close it | Long-term subsystems |
| `spec_loader/schemas/*.json` | The spec of a spec. The only enforcement point | Prose a reader needs |

A module with no `docs/spec.md` is not exempt; it means some other document is
specifying it, and its README must say which.

**Every spec and every design opens with the same four rows**, in this order,
and nothing else goes above them:

| Row | |
|---|---|
| `Status` | Normative, and for what |
| `Version` | A single integer. **It lives here and nowhere else** — a second copy in another document's `Implements` row is a second thing to keep true, and it drifted in three of nine before anyone read them against each other |
| `Updated` | The date this version was written |
| `Summary` | One sentence: what this document settles |

**No revision history.** What changed at version N−1 is what the document said
at version N−1, and the document no longer says it. A changelog inside a
normative document is read as though it were still normative, and it is the part
that cannot be checked against anything.

### 0.1.1 The two canonical structures

A reader who knows one document knows where to look in all of them.

| | **spec** | **design** |
|---|---|---|
| 1 | Purpose | Scope |
| 2 | Design principles | Layout and import graph |
| 3…N | the module's own substance | the module's own substance |
| N+1 | *(optional)* What this module does not own | Build versus adopt |
| N+2 | **Acceptance criteria** | **Test plan** — including the criterion→test map |
| N+3 | **Open questions** — always last | Deviations from the spec |
| N+4 | — | **New open questions** — always last |

**There is no "implementation order" section.** A build plan — *write this
module first, then that one* — answers a question that stops existing the moment
the code exists, which is §0.3's test. What is durable about a dependency order
is the import graph, and that is §2 of every design.

**Section numbers are load-bearing and are not renumbered casually**: roughly 700
places in this repository cite a document by `§N`. Reordering a *body* section is
a repo-wide edit; reordering or removing a tail section is usually free, and was
both times, because almost nothing cites the tail. **Check rather than assume** —
a citation that no longer resolves is silent, so the check is worth writing once
and re-running.

### 0.2 A document names a file, and stops there

**Never cite a line number.** Not in a document, not in a docstring, not in a
comment. A position is not a fact about the code — it is a fact about a
particular revision of a file, it goes stale on the next edit, and it goes stale
*silently*: the citation still looks valid and now points at something else.

| Instead of | Write |
|---|---|
| `runner.py` + a line number | `runner.py`, or `runner.py::_seal_outputs` |
| "the check at line 40" | the name of the check |
| a path into working space | the conclusion the probe reached |

The same rule is why **evidence is stated rather than pointed at**. A measurement
belongs in the document as its result and its conditions; a reader who cannot
check the claim without opening something that is not in the repository cannot
check it at all.

### 0.3 Keep the reason, drop the incident

A comment that explains **why the code is the way it is, stated as a property of
the code**, stays and is worth more than the code. A comment that narrates the
day it was discovered, who found it, which run, or how many attempts it took is
process information and belongs in neither the code nor the documents.

> *"`expect_ranks` must equal `tp`, because the probe counts ranks and the
> engine counts shards"* — a reason. Keep it.
>
> *"m3 found this on the 4th of September after two failed runs"* — an incident.
> It answers a question nobody will ask again.

The same test applies to a dated measurement presented as a constant, to a test
count quoted in a README, and to a revision history: each is a fact that was true
once, written where a reader will take it as true now.

---

## 1. A module is independent, and owns its own consistency

**Outside code may not read or write a module's internal state.** It calls an
interface, and the module keeps itself coherent.

| MUST | |
|---|---|
| **Expose as few interfaces as possible, and each as narrow as possible** | Every public name is a promise. A method added "because a caller needs it just now" is a promise nobody meant to make |
| **Maintain internal consistency inside the call** | When an interface returns, every invariant the module owns holds. A caller must never have to remember to also call something else afterwards, or to call two things in the right order |
| **Never let an invariant have two writers** | If one fact is maintained in two places they will disagree. Pick one writer and route everything through it |
| **Never hand out a mutable handle to internal state** | Returning a live object is the same as making the field public, one indirection later |
| **Depend on names, not on imports, where the graph would otherwise cycle** | Resolve a collaborator at use time. An import edge is permanent; a name lookup is not |

**Why it is a MUST and not a style note.** Once outside code touches inside
state, the module can no longer be reasoned about, tested, or replaced on its
own — and every later change has to consider every caller. That is the cost that
compounds.

## 2. Put it where it belongs, or do not put it anywhere yet

**Before adding any interface, attribute, or module, work out which part of the
existing design it belongs to semantically.** Write the answer down. If there is
no good home, that is a finding to report, not a reason to pick the nearest file.

| MUST | |
|---|---|
| **Never add a responsibility an existing mechanism can already cover** | Before introducing a concept, find the mechanism that already owns that job and **change it**. Modifying, clarifying or narrowing an existing component's responsibility is the normal move; adding a parallel one is not. A new module or concept is admissible only after a full analysis shows no existing component can host the job, **and** that its responsibility conflicts with nobody else's — and the analysis is written down, not asserted |
| **Name the owner before writing the code** | "Which object is this a fact *about*?" usually answers it in one sentence |
| **Never put something in a semantically wrong module because it is convenient** | Convenience is a one-time saving. A misplaced concept is paid for at every later read |
| **Never fuse two modules to avoid deciding** | A merge is easy and a split is expensive. When in doubt, keep them apart |
| **A new module needs a reason no existing one can host it** | State that reason in the design |
| **If the right home does not exist, say so** | An unowned concept reported is worth more than an owned concept in the wrong place |

**The most common failure is not putting a thing in the wrong module — it is
adding a thing at all so that some *other* module can compute with it.** That is
§3, and §4.4 is what it looks like in practice.

**A second concept covering a job the system already has a mechanism for is the
same failure wearing a friendlier face, and it is harder to see because nothing
is obviously in the wrong place.** Two mechanisms for one job do not merely cost
twice; they drift, and every later reader has to learn which one is authoritative
in which case — a question the code cannot answer for them. **More concepts make
more to maintain**, and the maintenance is paid by whoever is not in the room
when the second one is added.

So the order is: **find the owner, change the owner, and only then consider a new
component.** "The existing mechanism does not quite fit" is the beginning of the
analysis, not its conclusion — most of the time the right change is to the
existing mechanism, and the reason it looked unfit was that its responsibility had
never been stated precisely enough to argue with.

---


## 3. Whoever owns it does the work

**Do not take another module's property and compute with it.** If a piece of
logic is *about* module X's state, X performs it and returns the answer.

The symptom to watch for is a caller that reads `a.b.c`, branches on it, and
acts. Every one of those branches is logic that belongs to whoever owns `c`, and
that has been copied out to a place that will not be updated when `c` changes.

| MUST | |
|---|---|
| **Expose whole operations, not the parts to assemble one** | An interface should let a caller say *what it wants done*, not *how to do it*. "Give me the next version to write" is an operation; "tell me the status so I can decide whether to append" is a leaked internal |
| **Digest internally** | Branching on internal state belongs inside. If two callers each need the same branch, that branch was always the module's job |
| **Answer questions; do not publish fields to be interrogated** | `is_this_usable()` beats a `status` getter every caller compares against the same constant |
| **A getter/setter pair is a smell, not an interface** | It means the module has no opinion about its own state and has delegated that opinion to everyone else. Add one only when the value is genuinely inert data with no rules attached |
| **A manager is not a proxy for its members** | Collection operations belong to the collection; behaviour belongs to the member. Do not forward every member method through the owner |

**Why this is a MUST.** Logic copied into a caller cannot be found later. When
the rule changes, the owning module is updated and the four callers that
reimplemented it are not — and each of them is now silently wrong in a way no
test names, because no test knows the rule was ever there.

---

## 4. Worked examples

All four are from this codebase.

### 4.1 One verb instead of a branch — `Handoff.open_next`

An agent about to write a handoff needs a version to write into. The internal
situation differs: the slot may hold an untouched `CREATED` v0 to adopt, or it
may need `v+1` appended.

**Before**, two methods — `open` for the first run, `successor` for a re-run — so
a caller had to read the slot's state, decide which case it was in, and call the
matching one. The branch depends on a state machine only the handoff owns, and
every caller that ever appeared would have carried its own copy of it.

**After**, one verb:

```python
def open_next(self, task_id, agent_id) -> HandoffVersion:
    """Hand an agent a version to write, GENERATING, and return it.
    Adopts `latest` in place if it is still CREATED; otherwise appends v+1."""
```

The caller says *what it wants* — "I am about to write this" — and never learns
which case it was in. Two things fell out that nobody designed:

- **Version numbers became contiguous structurally.** The list index *is* the
  version, so nothing checks it. The old design needed a manager method
  validating `v == last.v + 1`, and that rule existed only because versions were
  loose objects a caller assembled.
- **The mgr's two write verbs collapsed into one.** `append` and `persist`
  became just `persist` — "this handoff changed, store it".

**A rule that lives inside cannot be got wrong from outside.**

### 4.2 Ask a question, do not read a field — `check_if_latest_valid`

The scheduler needs to know whether a task's input is usable. It could read
`handoff.versions[-1].status` and compare against `VALID`. It does not:

```python
def check_if_latest_valid(self, hid: HandoffId) -> bool:
    """A read, not a judgement."""
```

Three consequences, and the third is the point:

- The definition of "usable" lives in exactly one place, and it delegates further
  down — `HandoffMgr` → `Handoff.is_latest_valid` → `HandoffVersion.is_valid`.
- The scheduler stays content-agnostic, which is a stated principle it would
  otherwise violate by knowing what `VALID` means.
- **An unknown id returns `False` rather than raising.** That decision belongs to
  the handoff module, and it can only make it because the question comes to it. A
  caller reading `versions[-1]` would have got a `KeyError` and each caller would
  have invented its own answer.

### 4.3 Not a proxy — `HandoffMgr` has no `seal`

Transitions live on the objects: `Handoff.open_next`, `HandoffVersion.seal`. The
manager owns the *collection* — which handoffs exist, and how they persist — and
forwards none of it.

A caller mutates the object through its own guarded method and then asks the mgr
to `persist` it. The alternative — `mgr.seal(hid, status)` — reads more
conveniently and would put the version state machine's enforcement one level away
from the state it guards, in a class whose job is storage.

**The test.** If a method on the owner would do nothing but locate a member and
call one of its methods, it should not exist.

### 4.4 Hand over the answer, not the raw material

`SchedulePolicy` had to order eligible tasks depth-first — a parent's subgraph
before an unrelated sibling. The relation lives in `Task.parent`.

**Two wrong attempts, and both are this principle's failure mode.**

1. **Let the policy read `Task.parent`** and walk the structure. That is an
   outsider taking another object's property and doing graph logic with it — and
   graph structure is not the ordering policy's business.
2. **Stamp a `ready_since` timestamp onto `Task`** for the policy to sort on.
   This *looks* better, because the policy stops walking anything. It is the same
   mistake one step further out: a fact the scheduler observed, published as a
   property on someone else's object, so an outsider can compute with it. That is
   adding a getter to avoid doing the work inside.

**What was correct.** The scheduler decides when a task becomes eligible and owns
the collection eligible tasks sit in. So it should hand the policy a list already
in the right order, and the policy's job shrinks to "given ordered candidates,
choose". No field was added; no outsider reads structure; the mechanical check
that ordering ignores the structural fields passes without anyone defending it.

**The generalisation:**

> When a caller seems to need one of your properties, ask what it intends to
> *compute* with it — then offer that computation instead. If the answer is a
> whole operation, expose the operation. If it is an ordering, hand over the
> ordering.

Two smells that you are about to publish raw material:

- You are adding a field so that **someone else** can derive something from it.
- You are computing a **key** to recover an order that the module handing you the
  data already knew.

### A closing note on how 4.4 was found

Not by analysis. Days of prior-art research went into finding a better *sort
key* — and answered that question well, while the question was wrong. It broke
when someone asked *"why does the policy need a key at all?"*

**When a problem resists, re-examine the shape of the question before researching
the answer harder.** Re-read the specification first; in that case it had said
"that collection is ordered" from the beginning.
