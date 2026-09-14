# agent_sys — the implementation stage

| | |
|---|---|
| Status | **The plan for stage four.** Nine specs and nine designs are frozen; this turns them into code |
| Revision | 1 — 2026-08-28 |
| Companion | [`interfaces.md`](interfaces.md) is normative for seams; [`design-stage.md`](design-stage.md) records what the research settled. This file adds **schedule, gates, and the definition of done**, and nothing else |
| Binding | [`../engineer_principle.md`](../engineer_principle.md) |

---

## 1. What this file is for

`interfaces.md` §6 answers *who can start when*. It does not answer *what
"finished" means*, *what stops a broken package reaching the others*, or *when
the `task_graph` debt gets paid*. Those three are here.

**Nothing here amends a spec or a design.** Where this file and a design differ
on a signature, the design wins and this file is wrong.

---

## 2. The measured starting state

Run on the tree at 2026-08-28. `pytest agent_sys` — **454 passed in 2.31 s**.

| Package | Shipped | Reality |
|---|---|---|
| `spec_loader` | 220 lines | `protocols.py` + `__init__.py`. **Declaration only** |
| `handoff` | 283 | Declaration only |
| `validator` | 249 | Declaration only |
| `agent` | 231 | Declaration only |
| `closure` | 183 | Declaration only |
| `monitor` | 272 | Declaration only |
| `env_mgr` | 1115 | **Below the wall only** — `recipe`, `layer`, `runner`, `outcome`, `report`, `registry`, `versions`, `installers/`. The whole design §2 subtree (`meta.py`, `fs/`, `isolation/`, `grants.py`, `workspace.py`, `material.py`, `sync.py`, `remote/`, `prepare.py`) does not exist |
| `task_graph` | 1459 | Implemented, and **behind its own design** — §3 |
| `demo` | 0 | No `.py` at all |

Six packages are declaration-only, one is half, one is in debt, one is empty.
**The 454 green tests cover `env_mgr` below the wall, `task_graph` as shipped,
and the three interface tests.** They are the regression guard for everything
below and must stay green at every step.

---

## 3. The `task_graph` debt, and where it goes

### 3.1 The two numbers were counting different things

`interfaces.md` §6 puts `task_graph` in **wave 2**, as *"three small changes to
unimplemented rev. 11 material"*. `materials/00-architecture.md` §4 calls it
**the first thing to fix** and tabulates eight differences. Both are right, and
they are not measuring the same distance:

| | |
|---|---|
| shipped (≈ rev. 7) → design **rev. 11** | Large. Two `TaskStatus` members, seven `Task` fields, six verbs, `OrderedIdSet`, three absent files, an eleven-parameter `build_registry` |
| design rev. 11 → **rev. 12** | Genuinely three small changes: `Grant.handoff` → `Grant.kind: str`, the cascade reason moves into the report, `resume_system` rebuilds `OrderedIdSet` pools |

§6 counted only the second. **The total is the sum, and the sum is not a wave-2
task.**

### 3.2 Therefore it is wave 0

Three wave-1 packages import the missing pieces, not merely the package:

| Package | Needs, and does not have |
|---|---|
| `agent` | `TaskStatus.INPUT_VALIDATING` / `OUTPUT_VALIDATING`, `Task.closure`, `Task.enter_phase` |
| `env_mgr` | `Task.permissions`, `Grant.kind` (rev. 12), `Execution` at its rev. 11 shape |
| `monitor` | `Task.parent` to escalate along, `Task.enter_phase` to advance a phase, `Task.monitor_spec` to resolve by name |

Coding those three against types that do not exist is coding against a document
twice — once for the seam, once for the type. **`task_graph` catch-up runs in
wave 0, beside `spec_loader`.** The two do not touch: `task_graph` imports
pydantic, and `spec_loader` imports nothing of ours.

### 3.3 Two properties of the package that make the debt sharp

- `Model` sets `extra="forbid"` and `validate_assignment=True`. **Adding a field
  is a real change** — assigning an undeclared one raises.
- A record written by new code is **unreadable by old code**, for the same
  reason. There is no rollback across a `Task` field addition, so the field set
  lands once, deliberately, and not field by field.

---

## 4. The waves

Derived from `interfaces.md` §6, with §3 applied.

| Wave | Package | May import | Starts when |
|---|---|---|---|
| **0** | `spec_loader` | nothing of ours, ever | now |
| **0** | `task_graph` rev. 12 | pydantic; `spec_loader` in `graph.py` only | now |
| **1** | `handoff` | `spec_loader` | now, against `protocols.py` |
| **1** | `validator` | `spec_loader`, `handoff` | now, against `protocols.py` |
| **1** | `closure` | `spec_loader` | now, against `protocols.py` |
| **1** | `monitor` | `task_graph` ids only | now, against `protocols.py` |
| **1** | `env_mgr` | `task_graph` | now, against `protocols.py` |
| **1** | `agent` | `spec_loader`, `task_graph` | now, against `protocols.py` |
| **2** | `demo` | all eight | when wave 0 and 1 are green |

### 4.1 Why wave 1 starts now and not after wave 0

**Because the seams are already on disk.** Seven `protocols.py` files exist, are
declaration-only, and are *frozen by construction* (`interfaces.md` §1.2). A
wave-1 package is written against the Protocol, not against a neighbour's
in-flight implementation.

The rule that makes this safe, and it is the one rule wave 1 may not bend:

> **Import the Protocol, never a sibling's implementation module.** If you need
> a neighbour's behaviour to run a test, satisfy the Protocol with a stub in
> your own `tests/`.

`interfaces.md` §6 says the same thing for `validator`: *"Stub both against
`protocols.py` and start immediately."*

### 4.2 What each wave-1 package can build with no dependency at all

Named because it is the work to do first, while wave 0 is still moving:

| | |
|---|---|
| `handoff` | `digest`, `readme`, `pointer`, `locality` — pure functions over bytes |
| `closure` | the six checks; `Registries` is a Protocol a test satisfies with five dicts |
| `validator` | the composite fold, the verdict shapes, `PhaseOutcome` |
| `env_mgr` | `fs/path.py` — imports only `os` and `pathlib`, and three modules sit on it |
| `monitor` | the buffer, `EventRecord`, `EventKind` / `PLANNED` |
| `agent` | `backend.py` and `select_backend` — they import nothing of ours |

---

## 5. Done, per package

A package is done when **all five** hold. Four are mechanical; the first is the
one that takes judgement.

1. **Every acceptance criterion in `<module>/docs/spec.md` maps to a named test
   that exists and passes.** The design's §11 test plan already names them —
   that mapping is the deliverable, not a formality. 172 criteria across the
   eight stage-one specs; `docs/spec.md` §9 is the index.
2. `pytest agent_sys` green — **the whole suite**, not your directory.
3. `ruff check` and `ruff format --check` clean, at the repository's settings.
4. **The import rule holds**: nothing leaves your package that `interfaces.md`
   §4 does not list, and nothing enters that it does not permit.
5. `<module>/README.md` names every library adopted and why — mission rule 5.

### 5.1 The criteria count, per package

| Package | Spec rev. | Criteria |
|---|---|---|
| main / architecture | 9 | 15 |
| `handoff` | 5 | 17 |
| `validator` | 7 | 21 |
| `task_graph` | 12 | 54 — **1–35 are built and green** |
| `agent` | 4 | 16 |
| `closure` | 8 | 11 |
| `env_mgr` | 3 | 22 |
| `demo` | 6 | 16 |
| `monitor` | 14 | see its §11 |

---

## 6. The gates

Three, and they run in this order. A package that fails one does not advance.

**A number measured mid-write is not stale — it is unattributable.** Two people
nearly reported another package as broken on a run that was green minutes later:
a `build_registry` parameter that did not exist yet, and a `NameError`, both from
in-flight writes by somebody else. Re-run before quoting. **Quote a suite number
with the commit it was measured at.** Eight agents wrote
to one tree; a run that starts mid-write imports a half-written module, and the
*collected* count changed between consecutive runs of the same command. Several
hours were spent on failures that were tree-state artefacts rather than defects
— and one genuine deterministic bug was nearly dismissed as flake because of
them. `git log --oneline -1` beside the result, always.

| Gate | What it runs | Blocks |
|---|---|---|
| **G1 — package** | `pytest agent_sys/tests/<module>` + `ruff` on the package | that package's own claim of done |
| **G2 — suite** | `pytest agent_sys` entire, plus `tests/interfaces/` | merging into the shared tree |
| **G3 — composition** | `build_registry` builds; the demo's `--dry-run` loads with no credentials, no sandbox, no model | wave 2, and the release |

### 6.0a Run the whole thing early — it is worth more than per-package rigour

**The single most actionable finding of this stage.** `demo` — the first artefact
that assembles all eight packages — found **three defects in one live run**, and
in the week before it existed, seven packages found **none of them**.

They were not missed through carelessness. They were **unreachable from inside any
package**, and the two mechanisms differ in whether anything cheap fixes them:

| | fixable by |
|---|---|
| **the double accepts more than the real thing** — a `set_task` that no-ops, a `StubTask.enter_phase` that permits what the real one refuses | **stricter stubs**, compared against the declaration. `tests/interfaces/test_declaration_conformance.py` closes this |
| **the test is the missing caller** — `Monitor.mainloop` had a docstring saying *"its own thread"*, 90 green tests exercising it **on a thread its own tests started**, and no caller anywhere in the system | **nothing inside the package.** Only assembly finds it |

`HandoffStore.put` with no caller, the escape-hatch flag with no route from the
root, `AgentSpecRegistry.check_knowledge` never called, `Monitor.set_task` never
called, `bind_phase` never fed — **all the second kind.**

> **A package's own suite cannot find this class, because the test is the missing
> caller.**

**So the process change: assemble early, even against stubs, and run it.** The
wave-2 integration artefact was scheduled last because it depends on everything;
its *value* is inverted — it is the only role that can find a whole family of
defect, and every week it does not exist is a week those defects accumulate
silently behind green suites.

### 6.0b What a guard is worth, measured honestly

The conformance test over all seven `protocols.py` found **one** non-conforming
package on its first run, and that one had already been reported by a person.
Its author said so rather than claiming a better result:

> Six of seven conformed on the first run. The three defects this test was
> written for were all *already found*, by three different people reading seams.
> **What it buys is that the fourth one does not need somebody reading.**

That is the right way to value a guard. **Its worth is not what it catches on the
day it lands; it is that it removes the dependency on someone noticing.** Every
defect this stage found was caught by a person reading a seam, and people do not
scale to the next eight packages.

### 6.0 G3 gains the assertion that would have caught all five — added 2026-08-28

Five defects reached the assembled system this stage and **not one was catchable
by any package's own suite**, because each of us asserted only our own half
returned the right value. Every one degraded to a **plausible empty value**
rather than an error: a `getattr` default over a renamed accessor, a check
reading the wrong nested key, a registry whose only producer was a test fixture,
a `skip=` filter comparing origins against names, and a store built without its
kind resolver.

So **G3 asserts that the assembled system reports *something* on a package built
to fail** — one deliberately-broken fixture package, loaded through the real
`load_package`, the real `check_closures`, the real `check_graph`, asserting a
non-empty problem list naming the fault.

That is the only gate shaped like the defect. "Returns `[]` on a good catalogue"
is what passed while three criteria were inert.

**G3 is load-time on purpose.** Airflow tests every shipped example and every one
of those tests is parse-only — `test_should_be_importable`,
`test_should_not_do_database_queries`. That dissolves the tension between *"the
example is the first thing to break when a module drifts"* and *"CI cannot make
model calls"*: **CI loads it on every commit; a human runs it.**

### 6.1 Two standing rules for the tests

**An expected failure is strict.** The demo's failing validator and its blocked
write are `xfail(strict=True)` — an expected failure that *passes* is a FAILURE.
A demo that reports "all good" when its sandbox stopped working is the worst
outcome available to it.

**`PhaseOutcome` never defaults to success.** `PhaseOutcome.empty()` is its own
outcome, is not a pass, and an unrecognised outcome is an error. Four systems
reached that independently.

---

## 7. The dependency declaration, which is a gate blocker

`pyproject.toml` declares **three** runtime dependencies; the design set needs
**eight**. Nine packages are installed and declared nowhere, so *the suite is
green by accident*.

One of them is not a formality:

> **`python-jsonpath` was the only one NOT installed** — and it is the library
> `handoff` §8.4 chose after measuring six. Both libraries it **rejected**
> (`jsonpath-ng`, `jsonpointer`) are present. **A Pointer test written today
> would have passed using a rejected library and failed on a clean install.**
> Installed 2026-08-28, before any Pointer code was written.

`handoff` owns landing this block, since five of the eight are its:

```toml
dependencies = [
  "pyyaml>=6", "packaging>=23", "pydantic>=2",
  "ruamel.yaml>=0.18",      # main design §8 — a parse that carries positions
  "jsonschema>=4.18",       # main design §8 — the only enforcement point
  "python-jsonpath>=1.1",   # handoff §8.4 — Pointer, three-way failure
  "markdown-it-py>=3",      # handoff §9.2 — CommonMark, not a regex
  "rfc8785>=0.1",           # handoff §4.6 — JCS that raises instead of rounding
]
[project.optional-dependencies]
claude = ["claude-agent-sdk>=0.2.144"]     # agent §8.1 — 376 MB
dev = ["pytest>=8"]
```

**Amended 2026-08-29.** This block listed `jsonnet>=0.20`, and the
user-interface stage deleted the render step (main spec §4.4 rev. 10). The line
above is the replacement, not an addition: `ruamel.yaml` is imported by
`spec_loader/yaml_source.py` and jsonnet is imported by nothing —
`tests/interfaces/test_import_rules.py::test_no_source_format_survives_the_deletion`
forbids it and passes. **Whether the file on disk has been updated is not checked
here**, and `agent_sys/pyproject.toml` has no owner in the user-interface stage's
wave table, which is a gap worth someone claiming.

The paragraph this replaces recorded that `rjsonnet` 0.5.6 was installed and that
main design O2's aarch64 concern therefore already had a fallback present, needing
"one function in `render.py`". Both the concern and the function are gone;
`docs/design.md` O2 closes it as closed-by-deletion and says why the cost is still
worth recording.

**No type checker is installed** — neither `mypy` nor `pyright`. Seven
`protocols.py` files are checkable and nothing checks them. Adding `mypy` to
`dev` plus one CI step is the cheap fix; it is not in scope here, and it is the
first thing to add if the seams start drifting.

---

## 8. Rules for the team

One package per person. Eight in parallel, then `demo`.

| | |
|---|---|
| **Read the whole spec set before writing your module** | You need the global picture to know which of your choices is somebody else's problem |
| **Never change a cross-module signature quietly** | `interfaces.md` §1.1. A seam has two sides and only one is in front of you. Raise it, name both sides |
| **Report a blocking principle; do not route around it** | A stated conflict is a useful deliverable |
| **Scratch lives in `scratch/impl-2026-08/<module>/`** | Probe scripts are kept, not deleted — they are the evidence. **"Kept" means kept in this worktree: `scratch/` is gitignored, so nothing in it survives a fresh clone.** A document citing a probe must therefore carry **the number it measured**, not only the path — the result is the durable claim, the script is reproduction. Found by `handoff`, who noticed "kept" reads as "published" |
| **Do not edit the repository's `pyproject.toml`** | `agent_sys/pyproject.toml` is yours; the repository's is not |
| **`git commit -s`, signed off as yourself** | DCO. An unsigned commit is a broken PR |

### 8.2 Handing a change across a live seam — verified three times

Three signature changes landed under packages that were mid-write, without
breaking any of them. Same shape every time:

1. **Land additive.** The new parameter, the new name, the new bridge — with the
   old path constructing exactly what it constructed before.
2. **Measure that the old and new paths coexist.** Not reason about it. *What
   happens if both run at once?*
3. **Tell the other side it can move at its own pace**, and say so explicitly.
4. **Remove the default only when both halves are in**, in one commit.

`build_registry(registries=...)`, the `agent_specs` → `agent_mgr` bridge, and
`_Id` → `Id` all went this way.

> **Step 2 is the load-bearing one and the easy one to skip.** Each time the
> answer was *"nothing happens"* — a duplicate registration, a second registry
> construction, an alias beside its new name — **but only because it was
> checked.** Had `AgentMgr.register` raised on a duplicate instead of
> overwriting, the same plan would have broken the other package's suite the
> moment it was committed.

Two refinements worth keeping:

- **Better than warning the other side is making the collision cost nothing.**
  The instruction given was *"do not let them discover a double registration"*;
  what was built instead was `test_bridging_twice_is_harmless`, so both bridges
  running is a no-op and the other package deletes theirs at a quiet moment.
- **A public name spelled with a leading underscore is self-contradictory.**
  `_Id` was ruled public and exported as `Id`, with `_Id = Id` retained as an
  alias so the subclassing package pays nothing until it moves. §1.2 defines the
  underscore as *named in one package*; exporting it under that spelling would
  have made every reader wrong.

### 8.9 Verify a whole section once, instead of a row at a time when reported

§4's `Imports` rows were corrected **five times**, each after somebody hit one.
That is a row-at-a-time repair of a section-wide property, and §1.0b's rule
covers the fold-back but not the audit.

**One AST sweep over all eight packages settles it in a minute:**

```
spec_loader  []                              handoff     [spec_loader, task_graph]
validator    [handoff, spec_loader,          agent       [monitor, spec_loader,
              task_graph]                                 task_graph]
closure      [spec_loader]                   env_mgr     [task_graph]
monitor      [task_graph]                    task_graph  [spec_loader]
```

**One discrepancy in eight, and it points the other way**: §4.3 listed `monitor`
for `validator`, and `validator` does not import it. **The document was wider
than the code, not narrower.**

That is worth distinguishing. A row that is *narrower* than the code hides an
edge somebody is relying on. A row that is *wider* records a **permission nobody
uses** — harmless today, and tomorrow the route by which something arrives that
nobody argued for. Left permitted and **recorded as unused**, rather than
narrowed, because `test_import_rules.py`'s `ALLOWED` is a permission table and
the row should not be read as a description of behaviour.

**The general form:** when a section has been corrected more than twice
reactively, stop correcting rows and verify the section.

### 8.8 Prose that *became* stale is harder to find than prose written stale

**Three stale-prose instances this stage, and they are not one kind.** Two were
**written stale** — the sentence was wrong, or half-implemented, on the day it was
typed. The third **became stale while the file sat untouched**, because a change
was made on that module's behalf:

> `monitor/record.py` argued that *"importing the private `_Id` is the smaller of
> the two costs"*. Somebody else promoted `_Id` to `Id`, keeping an alias. **The
> import kept working. The justification stopped being true.** A reader would
> then find a reasoned case for accepting a cost that had been removed.

**It is the hardest of the three to find on purpose: no diff to review, no test
to fail, and the code beneath it correct.** The tell is the same one that catches
the other two — *the prose is more complete than the code beneath it* — but here
**the prose did not change and the world did**, so reading the file carefully
does not surface it.

**The countermeasure differs, which is why the separation is worth keeping.** The
first two are caught by one person reading one file. This one is caught **only by
somebody who knows the change happened** — which makes it the responsibility of
whoever made the change, not whoever owns the file.

And the fix is deletion rather than adjustment: the argument no longer applies at
all, so amending it would leave a weaker version of a case that should not be
made.

### 8.7 Every authored artefact is a snapshot, and authority suppresses the check

**The unification of §8.3 and §8.6, and it came from the last round.** A lead's
*message* and that same lead's *commit* disagreed. The implementer followed the
commit and stayed where they were; had they obeyed the message they would have
reverted a third time and left the tree contradicting the contract.

> **It is not a rule about teammates being unreliable. It is a rule about any
> authored artefact being a snapshot.** A ruling, a contract row, a docstring, a
> bug report and a direct instruction are all descriptions of a moving thing —
> and **every stale item this stage produced was true when it was written.**

The corollary is the operational half:

> **The more authoritative the source, the more expensive it is to skip the
> check — because authority is exactly what suppresses checking.**

A stale `interfaces.md` row cost forty minutes of building **precisely because it
was trusted enough to stop reading.** A stale instruction from the lead would
have cost a third reversal for the same reason.

**Three people made the same error on the same day, and all three made it on the
last step rather than in the analysis** — *measured the fix instead of the trap*,
*reproduced a snippet instead of the behaviour*, *confirmed one authority and
called it verification*. The reasoning was sound in each case; the final check
was skipped in each case.

### 8.6 A ruling can make somebody a worse reader of their own package

**The most expensive thing that happened in this stage was not a wrong ruling.**
It was what a ruling did to the person nearest the code:

> *"After the first ruling I stopped looking for reasons it might be wrong, and
> my own design's line was sitting three paragraphs from something I had already
> read. **A ruling made me a worse reader of my own package than I had been an
> hour earlier.**"*

One question went through **four positions, three of them the lead's and wrong**,
and every correction came from an implementer doing exactly the checking the
ruling had discouraged. The churn cost a build; the discouragement cost the
checks that would have prevented it.

**Two practices, and the second is the one that generalises:**

- **When you reverse a ruling, the person building against the first one is the
  first person to tell** — and fold it into *every* place the first one was
  written, not only the one you argued in. A stale row sent somebody at a
  reversed decision for forty minutes.
- **Verification is not dissent.** The shape to want is *comply and check*: do
  what was ruled, and say what you found while doing it. That is how a wrong
  ruling stays a message instead of becoming a defect.

**And check more than one authority.** *"Two sources agreeing is evidence; one
source consulted is a habit that feels like evidence."* The contract was
confirmed and the package's own design was never opened — with the sentence that
decided the question three lines above a table that had been read.

### 8.5 A probe carries its own bugs across the wire, with the authority of a measurement

**This is the failure mode of the practice that made the stage work.** People here
sent each other *runnable probes* rather than claims, and almost every finding
came from one. The hazard is specific to that:

> **A measurement relayed between people loses its provenance in one hop.**

One implementer ran a naive `split(':')` in a throwaway probe, saw it break, and
reported it as the behaviour of the correct implementation. The receiving package
then **verified the snippet rather than the behaviour** — which is what
verification looks like when the thing handed over is code — and committed a
docstring and a test asserting two false claims, both quoted from the sender.

**And a retraction has to go everywhere the claim went.** The sender corrected
*upward* and assumed it would flow back down. It did not: by then the claim was in
another package's committed code. **They had sent it to two people and retracted to
one.**

Two practices follow, and they cost nothing:

- **Run the behaviour, not the snippet.** A probe from a teammate is a hypothesis
  with runnable steps attached, not a result.
- **Retract along the same edges you asserted along.** List who received the
  claim before correcting it; correcting the record upward does not reach the
  person who already acted.

What survived on a true premise is worth keeping: the guard is still warranted,
because **a tagged string that *looks* parseable invites the naive parse.** It is
a format that punishes carelessness, not a lossy one — which is a different and
smaller claim than the one that propagated.

### 8.4 When your own guard blocks a correct change, convert it

`env_mgr` had `test_env_manager_has_exactly_one_method`, written deliberately to
hold *"one method, and it stays one"*. A ruling then required a second method.

**Three options, and only one keeps what the guard was for:**

| | |
|---|---|
| delete it | removes the pressure — a third method would then arrive unremarked |
| leave it at one | blocks a change that is correct |
| **pin the set** | `test_env_manager_exposes_exactly_these`. A third method still fails a test and still needs a decision |

**And amend the rule in writing rather than reading around it.** The tempting
move was a second registered *component* — which keeps the guard literally true
and makes the object graph worse. `EnvManager` **is a bound `Context`**; a
validation zone needs the same `ctx.domains` and `ctx.store_root`, so a second
component would bind one configuration twice. **One fact, two writers —
`engineer_principle.md` §1, and precisely what the one-method rule was protecting
against.** The letter preserved, the purpose broken.

The rule's owner made that call, not the person ruling. A rule reinterpreted from
outside to fit a decision is worth less afterwards than one its author amended.

### 8.3 Write down the doubt where the code is

`demo` needed the `agent_specs` bridge to make progress, built it in their own
package, and **recorded the ownership doubt in the code**: *"if the intended
owner is the composition root, this moves there and nothing else changes."*

Relocating it then cost nothing. The alternative — building it silently and
correctly — would have left the next reader unable to tell a considered choice
from an expedient one.

### 8.1 What integration is allowed to change

The user's rule, and it is the right one: **at integration, code may be debugged,
changed, and adapted.** Nothing here promises the first assembly works.

What integration should *not* have to do is discover that two modules meant
different things by one name. That is what `interfaces.md` §3 and §4 are for,
and it is what the fifteen consistency findings were.
