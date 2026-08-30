# agent_sys — the module interface contract

| | |
|---|---|
| Status | Stage four. **Normative for what crosses a module boundary**, and for nothing else |
| Revision | 5 — 2026-08-28. **Written from implementation, which is the first time any of this was executed.** Eight packages were built in parallel against rev. 4 and reported back; every change below is a finding with both sides named, in the shape §1.1 asks for. Three `Imports` rows were provably wrong — `tests/interfaces/test_import_rules.py` enforces an `ALLOWED` table that had already diverged from §4, so **the test was the truth and the document was stale**. §3 gains `BaseSpecRegistry`, which two packages independently found missing from opposite ends. §4.7 gains the composition-root clause that §2 always needed. §5 gains three seams that only appeared once there was code. (rev. 4: 2026-08-28. **`monitor` gets a seam (§4.9) and a `protocols.py`.** It was registered in §2 by a name nothing defined — the same shape as finding C1, one day old. `budget` and `install_excepthook` join it, both named by `monitor` spec rev. 14 and neither having anywhere to live. `monitor` enters wave 1 rather than waiting on `agent`, which is what §4.9's locally-declared `Pushable` buys. (rev. 3: 2026-08-27. §5.1b, from the spec-key-to-runtime trace. (rev. 2: 2026-08-27. The user-interface brief: the composition root gains `monitor:<name>`, `agent` gains `mainloop()`, `validator` swaps its callable for a `Body`, and §5.1's `Handoff.type` blank closes — `Task.kinds` now carries it)) |
| Companion | `spec_loader/`, `handoff/`, `validator/`, `agent/`, `closure/`, `env_mgr/`, `monitor/` — each a `protocols.py` plus its `.pyi`, the same contract importable and type-checkable |
| Produced by | The stage-three consistency pass. Findings in `scratch/design/findings-consistency.md` |

---

## 1. What this file is for, and how binding it is

Eight design documents exist and every one of them specifies interfaces. What
none of them could specify is **the other side of its own seams**, and the
consistency pass found the cost: five of the eight modules cannot be wired as
written, because four documents each wrote a different part of one composition
root and two components were resolved by name and registered by nobody.

So this file exists to hold exactly what no single module can hold:

- **§2 — the composition root.** One listing. Every other document defers to it.
- **§3 — the shared vocabulary.** The types more than one module names.
- **§4 — the seams**, module by module: what each exports, what it may import,
  and what it resolves at run time.
- **§5 — the seams left deliberately blank**, with a direction and the reason it
  is not decided.
- **§6 — who can start, and when.** Nine people, nine packages.
- **§7 — the dependency declaration**, which nine undeclared packages need.

### 1.0b §5 decides; §4 is where implementers look

**Three times in one day a settled §5 entry was not folded back into §4**, and
each time the §4 row told an implementer to do the thing that had just been
ruled against — §5.11 settled `wrap_argv` while §4.6 still described the
uncallable `bwrap_argv`; §5.15 and §4.3 both described `prepare_validation` while
§4.6 still said *one method*.

**A settled §5 entry that has not reached §4 does not read as history. It reads
as an unresolved contradiction, to exactly the person about to implement against
it.**

The mechanical check is cheap and is now the rule: **every §5 entry marked
*settled* names the §4 row it changed.** If it names none, either the decision
changed no seam or the fold-back has not happened.

### 1.1 This contract is reportable, not sacred

**If implementing something shows an interface here to be wrong, say so and
propose the change.** That is not a failure of the contract; it is the contract
working. A design document is a prediction about code that does not exist yet,
and this one is a prediction about eight of them at once.

What is asked instead is narrow: **do not change a cross-module signature
quietly.** A seam has two sides and only one of them is in front of you. Raise
it, name both sides, and let it be changed in one place — which is how the
fifteen contradictions in `findings-consistency.md` would each have been caught
at the cost of one message instead of a whole design revision.

Inside a module, nothing here binds you. Choose your own file split, your own
private helpers, your own names. The rule is only that what leaves the package
matches.

### 1.2 Two words used precisely

| | |
|---|---|
| **frozen** | Named by another module. Changing it breaks somebody who is not you |
| **internal** | Named only inside one package. Yours |

Every symbol in the seven `protocols.py` files is frozen by construction — that
is what those files are for. Everything else is internal until it appears here.

---

## 2. The composition root

`task_graph/bootstrap.py::build_registry`. **This listing is normative**; main
design §7, `task_graph` design §8.8, `closure` design §7.1 and `handoff` design
§6.5 each show their own rows and defer the whole to here.

```python
def build_registry(
    *,
    store: StoreMgr | None = None,
    runner: TaskRunner | None = None,
    policy: SchedulePolicy | None = None,
    resources: Sequence[ResourceMgr] | None = None,
    packages: Sequence[TaskPackage] = (),
    env: EnvManager | None = None,
    strict_level: StrictLevel = StrictLevel.DEFAULT,
    monitors: Sequence[Monitor] | None = None,
    config_order: Sequence[str] = (),
    handoff_root: str | None = None,
    knowledge_root: str | None = None,
) -> Registry:

    r = Registry()

    # 1 ── task_graph: the runtime managers. None of these reads a spec.
    r.register("store_mgr",   store or MemoryStoreMgr())
    r.register("handoff_mgr", HandoffMgr(r))
    r.register("task_mgr",    TaskMgr(r))
    r.register("agent_mgr",   AgentMgr())
    for m in resources or [GpuMgr(capacity=8), TokenMgr(capacity=1_000_000)]:
        r.register(f"resource:{m.name}", m)
    r.register("policy",      policy or DepthFirstPolicy())

    # 2 ── the five spec registries. Before the stores, since a store needs one.
    r.register("handoff_specs",   HandoffSpecRegistry())
    r.register("validator_specs", ValidatorSpecRegistry())
    r.register("task_specs",      TaskSpecRegistry())
    r.register("agent_specs",     AgentSpecRegistry())
    r.register("closures",        ClosureRegistry())

    # 3 ── handoff: two stores, one implementation, different roots.
    #      The KindSource closes over both halves of a mapping neither
    #      component holds alone — see 2.6. Without it `put` is unchecked.
    kinds = _KindSource(r)     # kind_for(hid): handoff_mgr.type_of
                               #                -> handoff_specs.kind_of
    r.register("handoff_store",   FilesystemStore(handoff_root,   kinds=kinds))
    r.register("knowledge_store", FilesystemStore(knowledge_root, kinds=kinds))

    # 4 ── load, per package. No cross-registry check happens inside.
    views   = RegistryViews(r)                      # satisfies Registries
    reports = [load_package(pkg, views) for pkg in packages]

    # 5 ── the two whole-catalogue passes, in this order. Both return problems.
    failed    = failed_names(reports)
    problems  = list(chain.from_iterable(rep.problems for rep in reports))
    problems += check_closures(views, views.handoff_specs.load_report(), skip=failed)
    problems += check_graph(views.task_specs, skip=failed | rejected(problems))
    if any(p.fatal for p in problems):
        raise SpecInvalid(format_problems(problems))
    views.closures.freeze()

    # 6 ── the executors. They resolve specs, so they are built after loading.
    if env is not None:                       # rev. 5: no default — see 2.4
        r.register("env_mgr", env)
    r.register("phase_runner", PhaseRunner(strict_level))
    r.register("runner",       runner or FakeRunner())
    r.register("budget",       Budget())          # monitor spec §4.1.3
    r.register("recorder",     Recorder(r.get("store_mgr")))   # rev. 5 — see 2.5
    for m in monitors or [PusherMonitor(DEFAULT_MONITOR_NAME, r)]:
        r.register(f"monitor:{m.name}", m)

    # 6b ── process-global, and therefore nobody's constructor.
    # An uncaught exception in a thread prints a traceback and vanishes:
    # the process lives, the exit code is unchanged, producers see nothing
    # (measured — monitor spec §5.4). Installed once, for every thread.
    install_excepthook(recorder=r.get("recorder"), sink=NullUserSink())

    # 7 ── the scheduler, last.
    r.register("scheduler", Scheduler(r))
    return r
```

### 2.1 The names, and who owns each

| Name | Type | Owner | Resolved by |
|---|---|---|---|
| `store_mgr` | `StoreMgr` | `task_graph` | every manager, for durability |
| `handoff_mgr` | `HandoffMgr` | `task_graph` | scheduler, `validator.phase`, `FakeRunner`, **and `agent.Runner` — added rev. 6.** The whole agent-facing write path (`open_next` → `seal` → `persist`) had **no production caller**: `FakeRunner.produce` was all three, in a docstring saying it stands in for `agent`. **The model slot was never opened, so `describe` waited on `status=created` while the store said sealed-and-passed.** A resolve, not an import — nothing in §4.6 moves |
| `task_mgr` | `TaskMgr` | `task_graph` | scheduler, the cascade |
| `agent_mgr` | `AgentMgr` | `task_graph` | scheduler at dispatch, `validator.phase` |
| `resource:<name>` | `ResourceMgr` | `task_graph` | scheduler, by pattern |
| `policy` | `SchedulePolicy` | `task_graph` | scheduler, once per pass |
| `handoff_store` | `HandoffStore` | `handoff` | `agent.Runner`, `validator.phase` |
| `knowledge_store` | `HandoffStore` | `handoff` | `agent.Runner`, for knowledge refs |
| `handoff_specs` · `validator_specs` · `task_specs` · `agent_specs` · `closures` | `SpecRegistry` | `handoff` · `validator` · `closure` · `agent` · `closure` | the two passes; `Task.unfold` and `replace_with` read `closures` |
| `env_mgr` | `EnvManager` | `env_mgr` | `agent.Runner`, before the main phase |
| `phase_runner` | `PhaseRunner` | `validator` | `agent.Runner`, twice per dispatch |
| `runner` | `TaskRunner` | `agent` supplies the real one; `task_graph` ships the fake | scheduler at dispatch |
| `monitor:<name>` | `Monitor` | `monitor` | `Task.monitor_spec`, by name. Absent takes the default. **Rev. 4: a name that will not resolve is no longer merely an unwatched task — it is a task that never advances a phase** (`monitor` spec §5.3). **Who calls `set_task` is held by §2.7 and by the code. This row does not restate it** — it has carried four different answers, and the last two were wrong while reading as settled, one of them costing `task_graph` a build. A row asserting a position is a state; states go stale where a pointer does not (§8.4). This row sanctioned the edge from the beginning and nothing had ever traversed it |
| `budget` | `Budget` | `monitor` | `agent.Runner`, at the completeness gate. One global value: nobody yet knows what a normal task costs, so a per-task limit would be authored out of numbers no one has |
| `validator_executor` | `Executor` | `agent` | `validator.phase` resolves it for an agent-bodied validator. **Registered by nobody — added to this table rev. 5 so integration reads it rather than discovers it.** It is C1's third instance and the least bad one: `validator` raises naming the component and citing `agent` O6, so it fails loudly. Whoever closes O6 registers the name |
| `scheduler` | `Scheduler` | `task_graph` | every `Task` transition, via `_registry` |
| `agent_spec:subgraph` | a **name**, not a spec | `task_graph` | `AgentMgr.is_registered` at `Scheduler.submit`, and `instantiate` at dispatch. **Added by the UI stage, and it is a name nobody authored.** A non-leaf no longer carries an `agent` (main spec §4.8, leaf-only), but `submit` gates every task on the name resolving and `_dispatch_pass` writes `agent.id` into a required `Execution.agent_id` — so removing the field would have reached persistence, and `interfaces.md` §5.13 is already open on exactly that shape. `SUBGRAPH_AGENT_SPEC = "subgraph"` is registered unconditionally by `_bridge_agent_specs`, **before** the `agent_specs` loop, because a graph can be built with no agent specs admitted at all and a non-leaf in one still has to pass the gate. **No document is invented**: `AgentMgr`'s table is `dict[str, dict]` and the one reader that would need a real spec, `runner.agent_spec_of`, is unreachable for a non-leaf — `_main` releases and returns at `agent/runner.py:679-685`, before `_deploy`. An authored spec of the same name warns rather than colliding |

**`env_mgr`, `phase_runner` and `monitor:<name>` are the three that were missing.**
The first two were resolved by name and registered by nobody; the monitor arrived
with `task_graph` spec §3.5 rev. 13, which moved it out of the roadmap. The first
two are the whole of finding C1, and they are why this file exists.

**Rev. 4 closes the monitor's half of that, which stood open for a day.** It was
registered here by a name nothing defined; `monitor/protocols.py` now defines it
— §4.9. The same revision adds `budget` and `install_excepthook`, and both arrived
the way C1's originals did: named by a design that had nowhere to put them.

### 2.7 Watching begins at dispatch, not at birth — rev. 5

**Nobody called `Monitor.set_task`**, so no task was watched and every planned
advance raised `ScopeViolation`. §4.12's family again, and the one with teeth,
since §2.1 already says an unwatched task is *a task that never advances a phase*.
Neither spec named a caller: `task_graph` spec §3.5 says only *"a monitor is told
what to watch"*, `monitor` spec §6 says only *"`set_task` is the interface"*.

**Only `task_graph` sees every task at birth** — the root is submitted from
outside, but `Task.unfold` instantiates subtasks *inside* `enter_phase(RUNNING)`,
so no external caller ever holds one.

**The two birth sites were measured and rejected, on a fact neither a call graph
nor a reading shows:**

```
add       watched 3 of 3 tasks in the graph
dispatch  watched 3 of 3 tasks in the graph

after a restart:  `add` fires on none of them
                  dispatched again after resume : 3
```

**`TaskMgr.resume_system` reconstructs the collection from the store without
going through `add`, and cannot** — `add` is the new-task path and raises on a
duplicate id. So birth needs **two** call sites, and the second is easy to
forget. **That is how this gap arose in the first place.**

**The site is `Scheduler._dispatch_pass`, immediately before `runner.start`.
Settled after four positions, three of them mine and wrong.**

**The argument that survives is about the interface, not about today's runner:**

```python
class TaskRunner(Protocol):
    def start(self, task, agent, on_done) -> None: ...
    def stop(self, task_id, on_stopped) -> None: ...
```

**Nothing about monitoring.** A runner that dispatches a task and does not
register it with the monitor produces an unwatched task, so calling `set_task` is
an obligation of *any* real runner — **declared nowhere and carried by each
implementation separately.** §4.12's shape.

**`FakeRunner` is not a counterexample to that**, and the measurement that looked
like one was mine: `advance()` calls `task.enter_phase` directly and never routes
through a monitor. **A double that does not participate is not evidence that a
real implementation should not be obliged** — it escapes the obligation by not
having the collaborator at all.

**One consequence stated in an earlier revision here was wrong and is corrected
by measurement:** *"`tests/task_graph`, `demo`'s dry-run and the G3 gate would
hit `ScopeViolation` on the first advance."* They do not — nothing in
`task_graph` raises, because `FakeRunner.advance` bypasses the monitor. The
`ScopeViolation` appears only where `monitor._advance` is the route, which today
is `demo` with a loop running.

**What the site buys, precisely:** *the watch set is populated wherever a task is
dispatched, whatever runner is installed* — **not** that the planned channel is
exercised end to end, which no current configuration does.

`task_graph/docs/design.md:480`'s *"Nothing in this module runs a monitor"* holds:
**calling one method is not running one**, and the implementation resolves
`monitor_for` by name, so the scheduler learns neither the default-name rule nor
the resolution.

Every task that produces an event is dispatched first — a non-leaf runs
`INPUT_VALIDATING` before it unfolds — so **nobody needs to observe a task being
born**, and recovery is free because resume re-dispatches.

Dispatch covers the unfold-born subtasks, **covers recovery for free** because
resume re-dispatches, and is one site — but the reason to keep if only one
survives is:

> **It is where a task first has a phase to advance.** A `WAITING_HANDOFF` task
> has no planned advance to make and no agent to poll for a stall; watching it is
> watching nothing. Spec §3.5's *"every task has a monitor"* is satisfied by
> every task that **runs**, and a task that never dispatches never needed one.

**And `monitor` supplied the structural version of that, measured in their own
code:** `_transition` is called from **exactly one place**, `_advance`. The
unplanned channel never transitions — `Escalate` reports, `Push` instructs,
`GiveUp` records. **So the watch set gates phase advances and nothing else**, and
a task with no phase to advance cannot need watching.

**The genuinely uncovered case is recorded as uncovered rather than
half-covered.** A task never dispatched at all — waiting forever on a handoff
that never arrives — is a departure from the plan that nobody notices.
**Watching it at birth would not fix it**: it would put an id in a set with no
mechanism behind it, since nothing reports on such a task, the monitor *receives
and does not hunt*, and its loop has no poll over the watch set.

> **The illusion of coverage plus the same silence is worse than the honest gap,
> because a closed-looking gap stops being looked at.**

`monitor._sweep()` is the seam if a poller is ever built. **§4.12's shape with the
sign reversed** — a capability that exists and reaches nobody — except that here
somebody declined to create one.

**The stated cost: a queued task that is cancelled is never watched.** That is a
claim about `monitor`'s semantics rather than `task_graph`'s, and it was put to
them rather than assumed. If some unplanned outcome does need a queued task
watched — resource starvation is the candidate nobody has ruled out — that is a
finding, and it makes it the two birth sites with `TaskMgr` growing a monitor
dependency.

### 2.8 Everything after `load_package` is a phase, and it has two failure signatures

**A correction to how these were grouped.** `set_task` was counted with the
*"passes the root must run"* — it is not one. It is **per-task, at run time, in
the scheduler**. The others are **whole-catalogue, once, after load**, and two
more of that shape were already here:

> **Everything after `load_package` is work no single package can do for itself,
> because each needs a registry some *other* package filled.**

That is §2.2's ordering argument generalised from *these two are ordered* to
*this is a phase*. Three current members and a fourth expected:

| | |
|---|---|
| **passes that report** | `check_closures`, `check_graph`, and `check_knowledge` when it lands. They return `Problem`s and raise nothing |
| **passes that effect** | `_bridge_agent_specs`. It registers, and without it the root returns something that **cannot dispatch** |

**The two failure signatures are why the distinction earns its place:**

- **A skipped *report* pass reports nothing — which reads exactly like a clean
  catalogue.** That was the `skip=` origin/name mismatch and `check_graph`
  reading the wrong shape: two of this week's defects, both in this phase, both
  invisible.
- **A skipped *effect* pass leaves a system that fails later and elsewhere.**
  That was `agent_mgr` empty after a successful load, which `demo` met as
  `unknown agent spec 'collect'` at submit — **three layers from its cause, and
  the signature that gets reported as somebody else's bug.**

#### `check_knowledge`'s `mandatory` flag — ruled: a run configuration object

`AgentSpecRegistry.check_knowledge(handoff_specs, *, mandatory=False)` drops into
the existing accumulator in one line, because it returns `Problem`s. **`mandatory`
is the problem** — spec §3.5 promises an operator that flag and **nothing in the
assembled system can set it.** Third instance of the escape-hatch gap.

**And `registries=` does not close this one**, which is the distinction worth
keeping: the escape-hatch flag was a **constructor** argument, so passing the
registry carried it. `mandatory` is a **call** argument to a pass the root makes,
so a caller has to reach *the call*, not the object.

**Ruled: route 3, a run configuration object.** `strict_level`, `config_order`,
`handoff_root`, `knowledge_root` and this are **five parameters that are one
thing** — the same argument that produced `registries=`, one level up. A
thirteenth parameter is the accumulation already argued against; folding it into
`strict_level` gives `validator`'s type a second meaning.

**Not called until the route exists.** Wiring it with `mandatory=False`
hardcoded would be a capability with no route, added knowingly, in the very
function whose comment explains why that keeps happening.

### 2.4 `env` has no default, because a `Context` cannot be guessed — rev. 5

Rev. 4 wrote `env or EnvManager(Context(...))`, and that `...` is unwritable.
`Context` is composition-time configuration — domains, store root, main
repository, sync mapping, tier — **none of it derivable at run time**. An
`EnvManager` over a fabricated `Context` is worse than an unregistered name: a
task would be prepared against an environment nobody configured, and `env_mgr`
criterion 14 is *no isolation, no start*.

An unpassed `env` therefore leaves the name unregistered and whoever resolves it
fails loudly. The demo and the whole-system CLI pass `env=EnvManager(ctx)`, which
is what the thin object exists for. `env_mgr` and `task_graph` reached this
independently.

### 2.5 `recorder` is a registered name — rev. 5

Rev. 4 registered `budget` and wrote `install_excepthook(recorder=..., sink=...)`
with a literal placeholder, so **the root already had to build a `Recorder` and
nothing in the listing built one.** The row finishes an unfinished line rather
than adding a component: monitor-owned, root-built, runner-read — `budget`'s
shape exactly.

**Open-on-first-use is not the alternative.** `Monitor.report` already opens the
container, because `Recorder.write` calls `open()` first. What stays uncovered is
therefore **an attempt that reports nothing at all** — the only case the marker
exists for. `monitor` criterion 14 distinguishes *never attempted* from *the store
lost it*, and a lazily-opened container cannot.

`agent/runner.py::_open_recorder` resolves the name and **skips silently when it
is absent**. With the row registered that skip becomes loud: a wiring bug that
quietly voids criterion 14 is worse than a raise.

### 2.6 The store needs a `KindSource`, or `put` checks nothing — rev. 5

Rev. 4 built the stores as `FilesystemStore(handoff_root)`, with no kind
resolver, and `put` **treated that as normal and degraded silently**. Required
README sections come from the content type, so with no kind it checked **none**;
`items` is checked against the kind's `items_schema`, so with no kind it checked
**nothing**; the manifest recorded `kind: ""`. Measured through exactly this
call:

```
content/README.md   -> one section, where a `reproducible` kind requires five
content/items/junk  -> an item no content type defines
FilesystemStore(root).put(...)   ->  published version: 0
                                     manifest kind: ''
```

**`handoff` criteria 2 and 3, unenforced in the assembled system, while all 135
tests in that package passed** — because every one of them injects a resolver.

A `KindSource` answers `hid -> HandoffKind` and needs two halves that **no single
component holds**: `hid -> Handoff.type` is `task_graph`'s `HandoffMgr`, and
`type -> HandoffKind` is `handoff`'s `HandoffSpecRegistry`. So the root supplies
it — five lines, no frozen signature changed, and §2.2's freedom to reorder is
what pays for it.

The two rejected routes, and why: **adding `kind` to `put`** changes a frozen
signature §4.2 calls out by name, on two sides, for something the root can just
supply. **`handoff` resolving `handoff_mgr` itself** is forbidden — §4.2 is *it
is called, it does not call*, and that is the whole of its position in the graph.

`put` now raises and names the wiring rather than publishing something
half-checked. **Loud and unwired beats quiet and wrong**, and `demo` will meet
this on day one, which is the point.

**The pseudo-code this section first carried had three wrong names**, and the
implementer checked all three against the shipped code rather than against the
listing — §8.2, *check before you edit, including when the instruction came from
the lead.* `KindSource` is a Protocol with `kind_for`; the registry accessor is
`kind_of`; `type_of` did not exist and `handoff` declined to guess it.

**All three fail loudly, and this section previously claimed one of them would
not.** That claim was wrong and was corrected by measurement: a `KindSource`
returning the raw mapping makes `put` die at
`AttributeError: 'dict' object has no attribute 'content_type'`, before the
staging directory exists.

**Keeping the correction is the point.** *A wrong spelling is an ordinary typo
the runtime catches.* Folding typos into §4.11 would make that category mean
"mistakes at seams" — which is everything, and therefore nothing. **Four
instances have the property; three typos do not**, and the four are the ones no
package suite can catch.

**Two shape notes from building it, both non-obvious:**

- **`type_of` is a question, not a getter.** The alternative was the root calling
  `get(hid)` and reading `.type` off a live `Handoff` — the mutable handle
  `test_authority.py` keeps out of the scheduler. That rule names the scheduler
  by letter, but the hazard is the handle, so the root takes the narrow read and
  a test pins that it does not reach for `get(`.
- **`None` for an unresolvable id is not a fallback.** It is the value the
  Protocol asks for, and `handoff.put` turns it into a raise naming the wiring.
  The removed `getattr(..., lambda: None)` produced **the same value as a guess**
  — same value, opposite meanings, and nothing downstream could tell them apart.

  Which gives §4.11 its most actionable sentence, and it tells you *where* to
  look rather than *when*:

  > **`None` and `""` are fine as answers and dangerous as defaults.**

### 2.2 Three things about the ordering, because two of them are not constraints

**Registration order is free.** Components resolve by name at use time
(`task_graph` design §8.8), so steps 1, 2, 3 and 6 may be reordered freely. They
read top-down for a human.

**`merged(reports)` is withdrawn — rev. 5, and it could never have been written.**
`reports` is a list of `spec_loader.LoadReport`, which is `(admitted, problems)`;
`check_closures` wants a `handoff.HandoffLoadReport`, which is
`(admitted, without_validator)`. **A `LoadReport` has no `without_validator` to
fold**, so the two types do not connect through any function — the §3.1 name
split is exactly what made this visible. One `HandoffSpecRegistry` receives every
package, so `load_report()` is already the whole-catalogue answer and a fold would be
a second writer of a fact the registry holds. `handoff` and `spec_loader` reached
this independently, from opposite ends.

**Steps 4 and 5 are genuinely ordered, and it is the one real constraint here.**
`load_package` needs the five registries to exist; `check_closures` needs *every*
package loaded, which is `closure` design §7.1's whole argument; `check_graph`
runs after it and takes the closures it rejected in `skip`; `freeze()` comes
after both, because the reverse index is built over the closures that passed
(`closure` design §8.2).

**The scheduler is last, and that is a statement rather than a constraint.** A
graph cannot be assembled from specs that have not been admitted. Main design
rev. 1 continued *"and the scheduler is what assembles it"*; that clause is
withdrawn — `closure` criterion 8 forbids the scheduler reading a closure, and
who builds the root `Task` is §5.3 below.

### 2.3 The default runner stays `FakeRunner`, deliberately

`agent` supplies the real `Runner`, and `build_registry`'s **default does not
change**. `tests/task_graph` is 358 tests written against a fake whose completion
the test drives, and swapping the default would rewrite the suite that is the
regression guard for everything else.

So the real runner is passed in, by the demo and by the whole-system CLI:

```python
r = build_registry(packages=[pkg], runner=Runner(r, config_order=cfg.backends))
```

The circularity there is apparent, not real — `Runner.__init__` stores the
registry and resolves nothing until `start`. If it becomes awkward, the fix with
precedent is a two-phase root: build, then `r.register("runner", Runner(r))`
before the scheduler.

---

## 3. The shared vocabulary

Types more than one module names. All live in `spec_loader/`, which imports
nothing from this repository and must stay that way (main design §2.3).

| Type | What | Frozen because |
|---|---|---|
| `Problem` | `(origin, path, keyword, message, fatal)` — one load-time fault | Five modules produce them; one function formats them |
| `SpecRegistry` | `add` / `get` / `names` / `__contains__`, duplicate-is-an-error | Four registries **are typed by it** |
| `BaseSpecRegistry` | the concrete base: the dict, the collision policy, and `_validate` as the per-kind override point | **Added rev. 5.** Four registries *subclass* it. A `Protocol` with `...` bodies has no dict and no policy, so "four registries subclass it" could not be true of the Protocol — four packages would each rewrite the thing main design §5.1 says the base exists to prevent. Found independently by `spec_loader` and by `closure`, which had grown a provisional `_BaseSpecRegistry` commented *"the real base belongs in `spec_loader`"* |
| `Registries` | a read-only view over the five spec registries | `load_package`, `check_closures` and `check_graph` all take one |
| `LoadReport` | `(admitted, problems)` — one package's load | `build_registry` merges them |
| `TaskPackage` | `discover()` / `config_for()` | the composition root's input |
| `ImportResolver` | `(base, rel) -> (path, bytes)` | main design §3.3's substitution point |
| `TaskSpec`, `ClosureDoc` | `Mapping[str, Any]` aliases | `task_graph.check_graph` and `closure.check` both name them |
| `SpecNotFound` · `SpecInvalid` · `SpecInconsistent` | the three error classes | main design §6.2 |

### 3.1 Two names that were one, and are now two

The pass found two collisions where one name meant two things. Both are resolved
by giving the second thing a different name, not by merging them.

| Was | Now | Why they are different |
|---|---|---|
| `handoff.LoadReport` and `spec_loader.LoadReport` | `HandoffLoadReport` and `LoadReport` | `(admitted, without_validator)` is the escape-hatch report; `(admitted, problems)` is a package load. `closure/check.py` holds both |
| `env_mgr.Access` and `task_graph.Access` | `env_mgr.Mode` and `task_graph.Access` | `Access` is what an author **declared** — read or write. `Mode` is what the **kernel** gets — combinable, and `READ_EXEC` has no declaration-side meaning. `prepare()` mixed both in one `Policy` |

And one that was two and is now one: `Verdict` was declared by both `handoff` and
`validator`. It is **`handoff`'s** — the module that persists it owns the shape —
and `validator` re-exports the name and keeps `VerdictRecord` as its own *view*
of one.

### 3.2 `check_closures` takes its report as an argument

```python
def check_closures(regs: Registries, handoff_report: HandoffLoadReport, *,
                   skip: Set[str] = frozenset()) -> list[Problem]: ...
```

`closure` design rev. 2 wanted `handoff_report` as a **field on `Registries`**,
and that is withdrawn: the field's type lives in `handoff`, so a Protocol in
`spec_loader` declaring it would make the leaf name a type in a module package.
A parameter also says something true — the escape-hatch report is a fact about
*this load*, while the five registries outlive it.

---

## 4. The seams, module by module

Each row: what leaves the package, what it may import, what it resolves by name.
The signatures are in the matching `protocols.py`.

### 4.1 `spec_loader` — the leaf

| | |
|---|---|
| **Exports** | §3's whole table, plus `validate`, `load_package`, `report`. **Rev. 5 added eight**: `RenderError`, `FileImportResolver`, `DirectoryPackage`, `schema_for` / `KINDS`, and `format_problems` / `failed_names` / `merged` / `rejected`. **The UI stage deletes four of those and `render` with them** — see the row below |
| **Imports** | `ruamel.yaml`, jsonschema. **Nothing from this repository, ever.** jsonnet and PyYAML are gone — UI stage |
| **Resolves** | nothing |

**The eight added exports each close a case where rev. 4 already named something with nowhere to come from** — finding C1's shape, one layer down:

| | Why it had to exist |
|---|---|
| `RenderError` | `protocols.render`'s own docstring says it raises this, and §3's table had no row |
| `FileImportResolver` | `render`'s `resolver` parameter has no default |
| `DirectoryPackage` | `build_registry` takes `Sequence[TaskPackage]` and nothing constructed one |
| `schema_for` / `KINDS` | the five schemas live here and four modules own their contents; without one accessor each hand-rolls the `importlib.resources` read main design D1 exists to prevent |
| `format_problems`, `failed_names`, `merged`, `rejected` | **§2's composition root calls all four and §4 assigned them to nobody.** They are derivations over `Problem` and `LoadReport`, both owned here — `engineer_principle.md` §3, whoever owns it does the work. `merged` / `rejected` touch `HandoffLoadReport`, which is `handoff`'s, so they **take it as a parameter** exactly as `check_closures` does per §3.2 |

**The UI stage removes five of these, and the removal is the point rather than a
tidy-up.** `render`, `RenderError`, `FileImportResolver`, the `ImportResolver`
Protocol and `SpecSource` are gone, and `DirectoryPackage` is now `YamlPackage`.
There is no render step: a package hands over parsed documents
(`TaskPackage.documents() -> PackageContents`), so main spec §4.4's promise that
the loader never inspects a source stops being an ordering convention inside
`load_package` and becomes a type boundary.

`validate` changed with it — `(data: bytes, ...) -> tuple[Any, list[Problem]]`
became `(doc: Any, ...) -> list[Problem]`. **Still path-free**, which is the
property main spec criterion 4 rests on and which `test_validate_takes_no_path`
still guards. Keeping `bytes` would have meant re-serialising a `ruamel`-parsed
document so `validate` could parse it again with PyYAML — and the two disagree on
ordinary scalars, because ruamel round-trip is YAML 1.2 and `safe_load` is 1.1
(`12:30` -> `'12:30'` vs `750`; `NO` -> `'NO'` vs `False`). One document, two
readings, built into the seam.

The one rule with teeth: `validate(data: bytes, schema, *, origin: str)` has **no
parameter through which a path could reach it**, which is how main spec §4.4's
"the loader does not read a package's jsonnet" is enforced rather than asserted.
`test_validate_takes_no_path` guards the signature.

**Where the leaf rule actually falls — endorsed rev. 5, and it is load-bearing
for everything after it.** `spec_loader` now exports accessors (`body_of`,
`subgraph_of`) that read keys out of a document, and the same person had earlier
argued that `load_package` reaching into a closure for `doc["task"]` would break
the rule. Both are right, and only if the line is stated:

> **`spec_loader` may declare and expose the vocabulary; it may not act on it
> during a load.** Exporting `body_of` is declaration-side. Having `load_package`
> change what it does based on a document's contents is action-side, and that is
> what main spec §4.4 makes structural.

The `$ref` in `closure.schema.json` is on the declaration side by the same test:
it declares a shape, rather than reading a document to find a nested object.

### 4.2 `handoff`

| | |
|---|---|
| **Exports** | `HandoffStore` (Protocol) and `FilesystemStore`; `Content`, `Item`, `Manifest`, `Verdict`, `Scope`, `ContentType`, `HandoffKind`; `HandoffSpecRegistry`; `tree_digest`, `resolve` (RFC 6901), `check_contained`; `HandoffLoadReport`; `version_dir` |
| **Imports** | `spec_loader`, **`task_graph.ids`** — corrected rev. 5 against `test_import_rules.py` |
| **Resolves** | nothing. It is called, it does not call |

Two frozen shapes carry more weight than the rest. **`copy_out(hid, version,
dst)` has no default for `dst`** — MLflow's equivalent returns the store's own
path and an agent handed it edits the store in place; the guarantee is the
signature, so the signature is tested. And **`put` is the commit token, not
`rename`** — if rename were the interface, an object-store backend would have
nothing to implement.

`Verdict` is the type `record_verdict` writes and `validator` reads. Its fields
are handoff spec criterion 8's: validator, result, strength, dimension, task,
agent, environment, timestamp.

### 4.3 `validator`

| | |
|---|---|
| **Exports** | `Validator`, `Reducer` (Protocols); ~~`Body`~~ — **withdrawn rev. 5, it is `spec_loader`'s** (below); `Dimension`, `Strength`, `PhaseKind`, `StrictLevel`; `PhaseOutcome`, `SkipRecord`, `VerdictRecord`; `ValidatorSpec`, `ValidatorSpecRegistry`; **`PhaseRunner`** |
| **Imports** | `spec_loader`, `handoff`, **`task_graph`**. `monitor` is **permitted and unused** — `test_import_rules.py`'s `ALLOWED` is a permission table, and an AST sweep of all eight packages finds no `monitor` import here. Left permitted rather than narrowed, and recorded so it is not read as a description. The `handoff` edge is for **`Verdict` and `HandoffStore`**; *"and the Pointer resolver"* is withdrawn — see §5.8 |
| **Resolves** | `handoff_mgr`, `agent_mgr`, `handoff_store`, **`closures`**, **`env_mgr`**, **`runner`** — at call time, never by import |

**`runner` is added for §8.2's `producer` row, and `agent` built the field it
reads.** `validator` spec §8.2 gives output validation *"the producer's — the
task that just ran"* configuration, and until `agent` `3155ca2` that value was a
discarded local of `_deploy`. It is now `TaskAttempt.environment`, a read-only
`Mapping[str, str]`, reached as `attempt_of(task.id).environment`. **A resolve,
not an import** — `test_import_rules.py`'s `ALLOWED` is unchanged and `validator`
still may not import `agent`.

**Two facts a reader of that row needs.** The component registered as `runner` is
**not one protocol**: `task_graph/bootstrap.py:101` registers the shipped
`FakeRunner`, which has `start` and `stop` and no attempts, so the capability is
checked rather than assumed. And **`TaskAttempt.environment` is `{}` until
`_deploy`** — which for a **non-leaf** is forever, since the scheduler runs its
main phase by unfolding — so `validator` reads empty as *absent* and falls
through to the global row. §8.2's row is *the configuration already resolved*,
and a task that resolved none has not got one.

**`consumer` stays unreachable, and in principle rather than for want of a
field.** `env.prepare` has one call site, `agent/runner.py:668`, inside
`_deploy`, and `_one_phase` reaches `_main` only in `RUNNING` — so at
`INPUT_VALIDATING` no `Prepared` exists. §8.2 calls that row *"the task about to
run"*, and about-to-run is exactly before `prepare`.

**`env_mgr` is added rev. 5, for `prepare_validation(task, execution, phase)`.**
A validation zone is placed as a **sibling of the producing task's zone, never a
descendant** — `env_mgr` design D5, and **criterion 13 is untrue without it**:
anything under the producing task's directory is inside its subtree and reachable.

Before this, zones were `tempfile.mkdtemp` in `/tmp`, so the separation held
**by accident of location rather than by placement**. An accident is not a
property; it is waiting.

The fit needs no signature change on either side — `build_environment(root, …)`
already takes a root and allocates inside it, so **a fresh directory inside a
correctly-placed sibling is still a sibling**. That the seam was already in the
signature, with the doubt documented beside it, is why the ruling cost one line.

**`closures` is added rev. 5, and without it a closure's declared phase
validators never run.** Found by `demo`'s first assembly. `PhaseRunner._select`
built a phase's set from the **handoff kind's** list; the closure's `validators`
list was read by nothing in the tree — `grep phase_validators` hits `closure`
and nowhere else. `closure.schema.json` says why the kind cannot carry them:

> *"They are a property of the task rather than of any one handoff kind, **which
> is why the handoff specs cannot carry them**."*

So a closure declaring `validators: ['check_grounded']` ran **nothing**, and a
handoff kind declaring one ran it in both phases of every task touching that
kind. Two behaviours; the specs describe the first.

**The fix is not "read the closure's list as well" — it is "ask `closures` for
the set and stop deriving it here."** `closure/query.py:99` already computes the
union and `closure` spec §217 states it: *"Every validator that will run, phase
validators and per-handoff ones together."* Reading both lists here would make
`validator` a second computer of something `closure` already computes —
`engineer_principle.md` §3, whoever owns it does the work.

**`PhaseRunner` is the seam `agent` calls**, and it is new in this pass:

```python
class PhaseRunner:
    def __init__(self, strict_level: StrictLevel) -> None: ...
    def run_phase(self, kind: PhaseKind, task: Task, registry: Registry) -> PhaseOutcome: ...
```

The strict level is bound once because it is a run-wide policy; the registry is
per call because that is how the phase reaches the managers.

**`PhaseOutcome` never defaults to success.** `PhaseOutcome.empty()` is its own
outcome and is not a pass — four systems reached that independently and none of
them spells the third state "pass". An unrecognised outcome is an error.

**`PhaseOutcome` gains `verdicts_expected: bool` — the §4.15 fold-back, per
§1.0b.** It is the field that makes §4.15's two `empty`s distinguishable:
`False` means nothing was asked of this phase (the level is `NONE`, or the task
has no handoff in this position), `True` means verdicts were expected — so an
empty **output** phase is the fault and `blocks_the_task`. `Evidence` gains
`UNCHECKED`, which is what `agent._evidence` puts on the `VALIDATION_FAILED`
record instead of the misleading `nothing_ran`.

**`fold`'s parameters are unchanged**, which is how criterion 20 stays
structural: the level reaches the *choice of constructor* and nothing else. The
two "nothing was asked" sites call `PhaseOutcome.nothing_expected(kind, skipped=…)`.

**Read narrowly, and measured.** §4.15's sentence is *nothing checked what this
task **produced***, so a task with **no output handoff** has nothing unchecked.
The wide reading blocks `examples/demo/closures/main.jsonnet` — `outputs: []`,
`validators: []` — which is the demo's root. `agent`'s side is unchanged: it
reads `blocks_the_task` and `evidence` through `getattr`.

**`Body` is `spec_loader`'s, not this module's — rev. 5.** One shape had two
declarations: a frozen dataclass here and a `TypedDict` behind
`_common.schema.json#/$defs/body`, which `task.schema.json` and
`validator.schema.json` both `$ref`. **The shape is shared with `task`** —
validator spec §6.1 says *a validator is a special kind of task* and `closure`
§2.6 gives the task the identical body — so one writer beats two, exactly as for
the schema.

The construction argument decides it, and it is the same one that settled `Body`
for `agent` and `closure`: **a dataclass has to construct, and constructing means
inventing a value for a field the document does not have.** That is what made
`Body(readme='')` truthy where `{}` is falsy. Against it, *"a `TypedDict` is a
weaker type at the seam"* — which loses, because the schema is the enforcement
point (main design §8), and **a stronger Python type that can construct an
invalid state is worse than a weaker one that cannot.**

The instance that proves it is in this package: `spec.body.entry` with
`entry: ""` is falsy, so a **programmatic validator with an empty entry path was
silently run as an agent-bodied one** — an executable check quietly becoming an
agent's opinion. `_common.schema.json` gives `readme`, `entry` and every
`material` `minLength: 1`, so **`spec_loader`'s gate rejected it and this
module's did not.** The two-gates rule catching its own author.

**A validator's implementation is a `Body`** — `readme.md` always, `entry.sh` when programmatic, plus its own `materials`. The registered Python callable is withdrawn: it cannot express a validator an agent is responsible for without a wrapper that runs an agent. What goes with it is pandera's `inspect.signature` argument check, which has nothing to read on a script.

The binding field is **`inputs`**. Both this design and `handoff` called it
`binds_to` in rev. 1, and no model has that key.

### 4.4 `agent`

| | |
|---|---|
| **Exports** | `Executor`, `AgentBackend` (Protocols); `AgentStatus`, `AgentResult`, `AgentHistory`, `BackendUnsupported`; `AgentSpec`, `AgentSpecRegistry`; `Selection`, `select_backend`; **`Runner`** — the real `TaskRunner`; `ProgramExecutor` |
| **Imports** | `spec_loader`, `task_graph` (`TaskRunner`, `Task`, `Agent`, `TaskId`, `TaskStatus`), **`monitor`** — added rev. 5. The runner reports every phase boundary, planned or not, so `runner.py` and `gate.py` name `EventKind` and `Budget`. **The edge is one-way and that is the whole of §4.9**: `agent` imports `monitor` concretely; `monitor` declares `Pushable` structurally and imports nothing back |
| **Resolves** | measured rev. 5: `agent_specs`, `task_specs`, `env_mgr`, `phase_runner`, `handoff_store`, `budget`. **And `handoff_mgr` as of rev. 6** — rev. 4 declared it, rev. 5 measured that nothing resolved it, and rev. 6 is why: **the agent-facing write path had no production caller at all.** `agent` raised the contradiction rather than resolving a name this row denied them; **the conclusion they had drawn was right and the premise I gave for it was false**, which is the distinction §8.8 is about. `task_specs` is §5.1b's route — `task.closure` → the task spec |

**`TaskAttempt`, `Runner.attempt_of` and `Runner.carry_on` are declared — rev. 5,
and this paragraph is the fifth §4 row to have trailed a settled decision.**
Measured rather than restated:

```
agent/protocols.py:232   class TaskAttempt(Protocol)
agent/protocols.py:281   class Runner(Protocol)
agent/protocols.py:301       def attempt_of(task_id) -> TaskAttempt | None
agent/protocols.py:309       def carry_on(task_id) -> str
```

They were owed and undeclared, and `monitor` reached the live handle as
`attempt.executor` — a name from `agent` design §7.5, **checkable against
nothing.** That is closed from both sides: `monitor/protocols.py` now declares
`Attempt` and `AttemptRunner` beside `Pushable`, and `tests/interfaces/test_runner_seam.py`
is their price.

**`Runner.resume` is gone**, subsumed by `carry_on` — one runner verb instead of a
two-call branch on `attempt_of(...) is None`, which was a **proxy for
leaf-versus-non-leaf** and was already wrong once. `carry_on` returns what it did,
so the shape reaches the `PHASE_DONE` record: **reading it to record is not
branching on it to decide.**

**An agent has its own `mainloop()`**, on level 1. `start()` returns immediately, and something has to be executing after it does; five verbs with nothing behind them is not an interface. The monitor's loop is a *different* loop watching for the task's exceptions.

**`Selection.backend` is `Executor`, not `AgentBackend` — corrected rev. 5.**
`agent/protocols.py` declared the narrower type and the implementation is right:
a `kind: program` spec selects a `ProgramExecutor` that has no level 2 *by
construction*, so the declared type cannot describe criterion 15's case. This
file already said so in prose, one paragraph down — **the contract contradicted
its own `protocols.py`.**

**The two levels are two protocols, and the runner holds level 1 only.** A
program executor implements `Executor` and has no level 2 to raise from; the
runner cannot call an AI-only method because it does not hold one. That is
criterion 6 as a type rather than as a test.

`backends/claude_sdk.py` is **never imported at module scope** — the SDK is a
376 MB extra costing ~1.3 s to import, so a missing extra must be a
`BackendUnsupported` naming it and not an `ImportError` at start-up.

### 4.5 `closure`

| | |
|---|---|
| **Exports** | `ClosureRegistry` with **six** queries; `TaskSpecRegistry`; `check_closures`; the six accessors over a `ClosureDoc` |
| **Imports** | `spec_loader`, and nothing else in this repository |
| **Resolves** | nothing at run time — that is criterion 8, and a spy proves it |

The import rule matters more here than anywhere: this is the module whose whole
job is looking at four other modules' objects, so it is where an import would be
easiest to justify and hardest to remove. It reaches them through `Registries`.

**`closures_using_validator` stays — the withdrawal above was reversed, and the
reversal is worth reading because the arguments on both sides were wrong in
different ways.**

**What was ruled and why it was wrong.** `closure` design **D4** argued for a
sixth query because *"leaving it out ships a known-wrong answer from a different
module"* — `validator`'s `users_of` could not see the closure edge. Wiring
`bind_phase` removed that, so D4's premise did go. The query was withdrawn on the
reasoning *"two indexes, one fact — `closures_using_validator` is a filter over
`users_of`'s output."*

**Nobody constructed the filter, including the person ruling.** That is the rule
`closure` articulated on retracting their own proposal:

> **Before proposing that one of two things is unnecessary, construct the call
> that replaces it.**

**What the two actually answer.** `users_of` is fed from *both* sides — `bind`
from the handoff kind, `bind_phase` from the closure — so it is the only place
that can answer **who names this *and how***, across kinds. `closures_using_validator`
answers **which closures name this as a phase validator**, typed. Folding the
first into the second would make `closure` learn about handoff-kind edges, which
is the leaked knowledge §4.5 exists to prevent. **Two questions, not two indexes
over one fact.**

**The escaping argument went through three states and the settled one is
narrow.** Offered as *recovery breaks on a closure named `a:b`*, dismissed here as
false, then over-restored. All three measurements in one place:

```
users_of('shape')           -> ['closure:a:b', 'handoff_kind:trace']
split(':', 1)[1]  correct   -> ['a:b']      exact
split(':')[1]     naive     -> ['a']        silently wrong
```

**The claim as made was false** — the tagging is injective and correct recovery is
exact. Its author retracted it and named the cause: *they ran the naive split, saw
it break, and reported it as the correct implementation's behaviour.*

**A measurement presented beside an inference reads as one thing.** A probe
printed `threads alive at prepare: 2` — **measured** — and
`verdict: prepare would REFUSE` beside it, which was **inferred from a line
number in another package's file**, and it stayed on screen for an hour after it
had stopped being true.

> **The failure was not a wrong fact. It was a fact that was load-bearing for a
> decision already made, and nobody asked what it actually exercised.**

The same probe output carried both halves of that correction, two days apart:
*threads alive: 2* proved an AI task was never confined in any runnable
configuration, and the report nonetheless described the pre-split arrangement as
though it had worked.

> **A bug in your own throwaway probe is not a finding.** When a probe shows a
> failure, check that the failing version is the one anybody would write.

**A naive recovery is nonetheless a real trap**, and `validator` pinned it with a
test asserting the **wrong** answer — `a` is a plausible closure name and nothing
raises. Documenting the trap rather than the fix is the right way round: a fix
reads as *this is handled*; a trap does not.

What the separation actually rests on is neither: recovery means **one package
parsing another's display format**, and the two answer different questions.

**The recorded non-option**, so it is not rediscovered: if `users_of` returned
structured pairs rather than tagged strings, the sixth query would be cleanly
derivable and could go. That is `validator`'s API, the tagged form reads better at
a call site, and churning it to delete one query is not worth it.

**And `users_of` is the better answer, not merely the surviving one.** It records
the **edge kind** — `closure:collect_trace` beside `kind:trace` — where §8.5's
plan was to union two answers at the composition root. **The union never carried
*how* a validator is reached, and "how" is what *what breaks if I change this*
actually wants.** Two indexes over one fact was the shape this revision spent the
day removing.

**§5.4 closes with it**: the enumeration now has one owner and one
representation, rather than a union nobody hosts that *"does not scale to a
fourth"*. The known counter — `users_of` returns tagged strings where the removed
query returned bare names — is `validator`'s to address in its return type if it
ever matters, and is not a reason for a second index.

`ClosureRegistry.freeze()` is called by the composition root and makes the index
immutable. `add` raises afterwards — Sphinx is the argument for making it
impossible rather than discouraged.

### 4.6 `env_mgr`

| | |
|---|---|
| **Exports** | **`EnvManager`** — `prepare` (**checks; no longer confines** — §5.15), `prepare_validation`, `place_zone`; `Prepared` (six fields, plus **`wrap_argv`** and **`spawn`** — §5.15), **`ValidationZone`**, **`Zone`**, **`Availability`**, `Zone`, `Confinement`, `Policy`, `Granted`, `Mode`, `Context`; `contained`; `NoConfinement`, `UnresolvedGrant`, `PrepareRefused` |
| **Imports** | `task_graph` (`Task`, `Execution`, `Handoff`, `Permissions`, `Grant`, `Access`) |
| **Resolves** | nothing. `Context` is bound at composition |

```python
class EnvManager:
    def __init__(self, ctx: Context) -> None: ...
    def prepare(self, task: Task, execution: Execution, agent_spec: Any = None) -> Prepared: ...
```

**Two methods, and the set is pinned** — `test_env_manager_exposes_exactly_these`.
It was one, and `prepare_validation` was added under a ruling; the guard was
**converted rather than deleted**, so a third still fails a test and still needs a
decision. `EnvManager` **is a bound `Context`**, and a validation zone needs the
same `ctx.domains` and `ctx.store_root` — a second registered *component* would
have bound one configuration twice, which is the thing the one-method rule was
protecting against. Letter preserved, purpose broken. A second is how the runner would start making environment decisions.

**`Prepared.wrap_argv(argv) -> list[str]` is not a second method, and the
distinction is the whole of this row.** It is on the *returned value*, not on the
component: the runner is not making an environment decision, it is asking the
prepared environment a question about itself. The alternative was
`bwrap_argv(policy, availability, argv)` — and `Availability` is not a type
`agent` may import, so that instruction **handed the caller raw material to
assemble something the owner should compute.** `engineer_principle.md` §4.4's
exact smell, discovered by `env_mgr` in its own instruction to `agent`.

It returns `argv` unchanged under Landlock, the bwrap command under rung 1, and
raises `NoConfinement` when there is nothing to wrap with — **including the binary
having vanished since probe time, resolved at exec rather than remembered.**

**`agent_spec` is the third parameter — added rev. 5, and without it four spec
keys have no consumer.** `agent` spec §3.1's `env` and `agent` design §3.4's
`rules` / `hooks` / `skills` all name this module as the thing that reads them,
and `prepare` had no parameter they could arrive through. That is exactly the
mechanical check `materials/00-architecture.md` §7 describes — *a document says X
consumes Y; check whether X's signature can accept Y* — and it failed. The
default keeps every existing two-argument call working.

**`Prepared` gains a sixth field for the deployed environment — added rev. 5, and
this one is functional rather than tidy.** `material.deploy` computes
`CLAUDE_CONFIG_DIR`, `CLAUDE_CODE_TMPDIR` and the spec's own `env`, and a
five-field frozen `NamedTuple` had nowhere to put them, so `prepare` dropped them
and the runner could not see them. Measured (`demo` research, and it is the
reason this is not cosmetic): **with `~/.claude` granted, a confined agent read
the *operator's personal* `CLAUDE.md` and obeyed its language rule.** Pointing
`CLAUDE_CONFIG_DIR` into the zone is what removes the `$HOME` grant entirely. The
rejected alternative was `agent` calling `material.deploy` itself, which puts an
environment decision in the runner — the thing *one method, and it stays one*
exists to prevent.

**One caller obligation, because it fails silently.** On the bubblewrap rung
`apply()` confines nothing: **bwrap *is* the exec**, so whoever starts the
executor must wrap the command line — **`prepared.wrap_argv(argv)`**, above. Skip
it and `prepare` succeeded, the task ran, and there was no sandbox. `bwrap` is
absent on the development machine, so nothing local catches it, and §5.11 is the
half of this that is still undecided.

Two properties the caller must not work around. **`prepare` takes an
`Execution`**, because a grant resolves to `<root>/<hid>/v<N>/` and `N` lives on
the attempt — a retry gets a different granted set. And **nothing it raises is
caught by the runner**: `NoConfinement`, `PrepareRefused` and `UnresolvedGrant`
all mean the task does not start. Criterion 14 is *no isolation, no start*.

### 4.7 `task_graph` — shipped, and the two rules that protect it

| | |
|---|---|
| **Exports** | `Task`, `Execution`, `Handoff`, `HandoffVersion`, `Agent`, `HandoffRef`, the three ids, `TaskStatus`, `HandoffStatus`, `Permissions`, `Grant`, `Access`; `Registry`, `StoreMgr`, the four managers, `TaskRunner`, `SchedulePolicy`, `Scheduler`, `build_registry`, `check_graph` |
| **Imports** | **at module scope**: pydantic, and `spec_loader` in `graph.py` for `TaskSpec` and `Problem`. `bootstrap.build_registry` resolves six sibling packages **at call time** — see below |
| **Resolves** | everything, by name — no manager imports another |

**§2 and this row contradicted each other, and §2 could not be built as written —
resolved rev. 5.** The composition root constructs `FilesystemStore`,
`HandoffSpecRegistry`, `ValidatorSpecRegistry`, `TaskSpecRegistry`,
`ClosureRegistry`, `AgentSpecRegistry`, `PhaseRunner`, `EnvManager`,
`PusherMonitor`, `Budget` and calls `load_package` and `check_closures`. At module
scope that cannot coexist with "pydantic and `spec_loader`".

**The rule is about module scope, and the composition root defers.** A root that
may not name its components is not a root. `engineer_principle.md` §1 wrote the
mechanism years before it was needed here: *"Depend on names, not on imports,
where the graph would otherwise cycle. Resolve a collaborator at use time. **An
import edge is permanent; a name lookup is not.**"*

So: function-local guarded imports inside `build_registry`, and
`tests/task_graph/test_bootstrap.py::test_importing_task_graph_reaches_no_sibling_package`
runs a **fresh interpreter** and asserts `import task_graph` pulls in none of the
six. That is the same bargain §8 describes — the test is not a nicety attached to
the decision, it is the decision's price.

**`RegistryViews` is `task_graph.bootstrap`'s — settled rev. 5.** §2 named it and
§4 assigned it to nobody. `Registries` is a `spec_loader` Protocol; the concrete
class needs all five registries at once, and the component `Registry` is the only
object holding them.

**`_Id` is promoted to a declared export — rev. 5.** `monitor.record.EventId`
subclasses it, so a leading-underscore name is now named in two packages and §1.2
says internal means named in one. The alternative — a fourth id class in
`task_graph/ids.py` — would make this package carry a monitor concept, against
`engineer_principle.md` §2. The edge is declared rather than smuggled, and
`task_graph` has undertaken not to change `_Id`'s shape or name without messaging
`monitor` first.

**423 tests are green and stay green.** Rev. 12 changes three things inside the
package and each is small: `Grant.handoff: HandoffId` becomes `Grant.kind: str`,
the cascade puts its reason in the report rather than on the task, and
`resume_system` rebuilds `OrderedIdSet` pools. None is in shipped code — all
three are rev. 11 material that was never implemented.

**The scheduler never names a spec registry**, and `test_authority.py` enforces
it. The narrower rule that actually holds: *the scheduler never reads a spec; a
`Task` transition may read the catalogue it came from.* `unfold` and
`replace_with` resolve `closures`, and that adds no scheduler edge.

### 4.8 `demo`

| | |
|---|---|
| **Exports** | nothing. **Nothing imports `demo`**, and a test greps every component package for the token to prove it |
| **Imports** | all seven |
| **Resolves** | via `build_registry` |

`cli/build.py` holds `root_task`, `handoff_ids` and `wire` — §5.3.

### 4.9 `monitor` — added rev. 4

| | |
|---|---|
| **Exports** | `Monitor`, `Recorder`, `UserSink`, `Pushable`, `EventRecord` (Protocols); `EventKind`, `PLANNED`, `Budget`; `BufferClosed`, `ScopeViolation`. **Rev. 5 adds four §2 already called and this row omitted**: `PusherMonitor`, `install_excepthook`, `check_liveness`, `DEFAULT_MONITOR_NAME` |
| **Imports** | `task_graph` (`TaskId`, `AgentId`, `HandoffId`) — **and nothing else of ours** |
| **Resolves** | `task_mgr`, `store_mgr`, `runner`, **`recorder`**, and `monitor:<name>` for the parent task's monitor when escalating or reporting a finished subgraph |

**It is the task's event loop, on two channels.** Planned phase advances are
handled by code and never by a model; unplanned outcomes are a decision. One
inbound call, `report`, and the routing is `kind in PLANNED` — a reporter never
classifies what it is reporting, which is what lets the gate call the same method
whether it passed or failed.

**`monitor` imports `agent` in neither direction, and `Pushable` is why.** The
monitor needs `instruct` on a live agent; the runner needs `report` from here.
Written the obvious way that is a package cycle, so the monitor declares the three
members it uses as a local Protocol and `AgentBackend` satisfies it structurally.
The cost is two declarations of one shape, and
`tests/interfaces/test_pushable.py` is what keeps them in step — a test may import
both, because tests are not under §4's rule.

**Putting `Monitor` in `task_graph` was the alternative and was rejected.**
`task_graph` is the one package everyone may import, so it would have worked; it
would also have made `task_graph` the owner of an interface the monitor spec says
this module defines, against `task_graph` design §8.9's own statement that almost
none of the monitor is its.

### 4.10 An enum a caller cannot import — the trap this stage actually found

**Added rev. 5, from a live bug, and it generalises to every row above.**

`validator.PhaseRunner.run_phase(kind: PhaseKind, ...)` takes an enum. Its only
caller is `agent.Runner`, which may not import `validator` — so `agent` passed the
**value** `"input_validation"`. Inside, `validator` branched with `is`:

```python
return list(task.inputs if kind is PhaseKind.INPUT else task.outputs)
```

A bare string takes the `else`. **An input phase would have validated the task's
outputs** — no exception, a plausible `PhaseOutcome`, the wrong artefacts checked.
A second site would have fallen through to the global environment just as
quietly. Fixed by coercing at each boundary so that a value which is not a phase
raises naming itself, and pinned by a test asserting the string form reaches the
*same branch* as the enum rather than merely not crashing.

**The rule, for any seam:**

> If a Protocol takes an enum that its only caller cannot import, the caller will
> pass the value. Coerce at the boundary and raise on an unknown one. **Never
> compare a crossing parameter with `is`.**

**How it was found is the point.** Neither side had shipped against the other.
`agent` read `validator`'s declared signature against its own import rule and
asked one question. That is §1.1 working — and it is the strongest argument this
document has that a seam has two sides and only one of them is ever in front of
you.

A sweep of the tree found 27 identity comparisons against enum members in package
code; the rest are on parameters whose callers can import the enum, or on values
the package owns end to end.

### 4.11 A seam that fails by producing a plausible empty value

**The unifying diagnosis, and it is also the repair.** Every instance below has
one tell:

> **A field or a parameter whose *absent* case was filled with the nearest
> available value instead of with nothing.**

`getattr(..., lambda: None)`. `kind: ""`. The producer's `agent_id`, in a verdict
whose purpose is to name its checker. `merged(reports)`. Each reached for
whatever was to hand rather than saying *there is nothing here*.

**The repair is the same every time: make absence representable and let the type
say it.** `Verdict.agent_id` became `AgentId | None`; `kind_for` returns `None`
and `put` raises on it; the `getattr` default went and the accessor is called
plainly.

**And it is cheap in every case except the one where somebody has already
persisted the wrong value.** That is the argument for doing this during wave 1
rather than after — the `agent_id` case was still cheap **only because nothing had
read it yet.**


**Added rev. 5. Three live defects had this shape, all found the same day, none
catchable by any package's own suite.**

| | What it produced instead of an error |
|---|---|
| `bootstrap.py`'s `getattr(reg, "load_report", lambda: None)()` versus a registry spelling it `report()` | `None` → `check_closures` returns early → **an escape-hatch admission went unreported in the assembled system**, with `tests/handoff`, `tests/closure` and `tests/task_graph` all green |
| `check_graph` reading the closure-document shape when handed a task spec | no `subgraph` key → every task reads as a leaf → **criteria 50 and 53 return `[]` on a catalogue that violates both** |
| `TaskSpecRegistry` whose only `add` call site is a test fixture | an empty registry → `check_graph` walks nothing |
| `FilesystemStore(root)` with no `KindSource` | `put` checked **no** README sections and **no** items, and wrote `kind: ""` — `handoff` criteria 2 and 3 unenforced in the assembled system, with all 135 of that package's tests green because every one injects a resolver. §2.6 |
| `skip=` carrying **origins** where `check_closures` filters on **names** | nothing matched, so **the layering gate was inert** — a closure whose own spec had already failed was walked again and reported twice. Kubernetes CRD validation's stated reason for the gate is that errors on top of a broken schema are not actionable; we had the gate without the effect |

**The sharpest instance is not an empty value but an inverted one, and it was
persisted.** `validator` wrote `Verdict(..., agent_id=task.current.agent_id)` —
**the producing agent**. So every verdict that outlives the run recorded the
producer as the agent that validated the artefact: the exact claim `validator`
spec §8.1 forbids, in the one artefact `handoff` criterion 8 asserts over.
Criterion 10's attribution leg was not weak, it was **inverted**. Nothing was
empty, nothing was absent, and a reader would have taken it as an independent
check.

It was found only when `agent` registered `validator_executor` and the real
`AgentId` had somewhere to arrive — **the eighth package assembling made a wrong
value visible that seven could not.**

**One rule.** *A check that reports nothing is indistinguishable from a check
that found nothing.*

**And a second, about defaults rather than shapes: a fallback is a decision that
the absent case is normal — and the danger is not the fallback, it is the
fallback that outlives its reason.**

**Not one of these was wrong when it was written.** Each was written while the
other side was declaration-only, when the absent case genuinely *was* normal.
They became wrong the day the other side landed, and **nothing re-examines a
default at that moment.** That is the shape, and it is a property of building
eight packages in parallel rather than of anyone's carelessness. `bootstrap`'s `getattr(..., lambda: None)` was
written when `handoff` was declaration-only and a missing method genuinely *was*
normal — and it **outlived that condition silently**. `handoff_specs` is a
registry the root already refuses to run without, so by the time it mattered the
default was tolerance for a case that cannot occur, buying nothing and costing
exactly this. **Every package has guarded code from the same period**; wave 1
began before wave 0 finished, and that is when those defaults were written. Every seam above degraded to silence, and silence is the
worst output a load-time check has — it is the shape `closure` named as *"the
worst failure available to it"*.

**And the corollary that let one of these survive a test that already existed:**

> **A signature test that compares parameter names and not defaults is not a
> signature test. The default is the drift.**

`closure`'s `test_check_closures_matches_its_declaration` compared names only, so
it stayed **green** while the implementation carried a default the declaration did
not have. **A guard that passes over the defect it is named for is worse than no
guard, because it is counted.** `agent`'s `select_backend` has the identical
drift: an undeclared fourth parameter carrying `readme`, `entry`, `zone`,
`environment` and `wrap_argv`, defaulted — so a caller using the *declared*
signature builds an agent with no instruction, no entry point and no zone, and it
starts and does nothing.

`tests/interfaces/` therefore carries a conformance test comparing every
implementation against its own `protocols.py`, **full signature including
defaults**, across all seven packages.

**And the sharpest instance is a guard that had the defect it was written to
catch.** `monitor`'s new call-site walker — built to close the *undeclared seam*
finding — matched `runner.*` and `attempt.*` as **bare names**, so it never looked
at `self._runner.resume`. **It passed because it had not checked.** §4.11 arriving
inside a test written to enforce §4.11, one layer in.

The only reason it was caught: **the author printed what the matcher actually
collected instead of trusting the green run.** Then mutation-verified it —
removing `resume` from the declaration fails it now and did not before.

**And a guard for a silent-*incompleteness* defect must assert a positive answer
from the real collaborator, not from a stub.** `closure`'s first four tests for
`bind_phase` asserted on their own stub's `phase_edges` dict: they proved the
call happens, not the thing the call exists for. **That is one step removed from
the failure — and exactly the distance that let `check_graph` walk an empty
catalogue while every unit test passed.**

> **A reverse index that is merely incomplete does not raise.** Every
> intermediate representation of it can be correct while the answer is wrong.

Rewritten against `validator`'s real `ValidatorSpecRegistry`, asking the question
a user asks — *what breaks if I change `check_shape`?* — with two closures running
it and no handoff kind naming it, which is Airflow #58058's exact shape.

**Check whether an unreproducible thing is one defect or two before deferring
it** — *"nobody can reproduce it"* is what makes a thing safe to defer, so it is
the claim to get right. Two people relayed one finding as *a property, not a
reproduction*, over **both** halves of it. Only one half was:

| | |
|---|---|
| `begin()` unguarded | **not a race.** An unconditional missing precondition on a public verb: `resume()` on a running attempt starts a second thread **every time**, and both threads run phases, both report, both call `on_done` |
| `is_running` going stale toward *running* | genuinely unreproducible. Still open |

The reason the first was never seen is that **one caller happens to check
`is_running` first** — which says nothing about `resume`'s other callers, or about
`resume` itself. Two reviewers flattened it; the third separated them and was
right.

**The requirement, rather than the advice: a guard for a silent-failure class
must be watched to fail once.** That class is *defined* by tests passing while
guarding nothing, so a guard nobody has seen fail is a guard nobody knows works.
Three packages have now done it deliberately — `closure` neutralised
`_admit_task_specs` and watched three tests go red, `task_graph` dropped `kinds=`
and stubbed `type_of` to `""` and watched two, and both restored and re-ran.

**A proxy for a thing is not the thing, and the third version of one guard was
still wrong.** `agent`'s bwrap guard was fixed three times, each wrong in the same
direction. Version two keyed the refusal on `AgentSpec.kind` — **and the kind is a
proxy for the executor.** Design D6 makes a CLI override resolve a backend entry
in its own right, so a `kind: program` spec pinned to an AI backend passed the
guard and ran unwrapped:

```
kind=program  override=scripted  -> by kind: accept  executor ScriptedBackend
                                                     <-- accepted, nothing wraps
```

Ask the executor, not its proxy: `ExecutorBase.accept_confinement` refuses by
default and `ProgramExecutor` overrides it. **And `Assignment` lost `wrap_argv`
entirely, which is the part to generalise: a field an executor *may* read is one
it may silently *not* read.** The wrapper now arrives only through a call that can
be **refused**, so not-wrapping takes an explicit override rather than an
omission.

**A fix on one side of a seam can silently disarm a guard on the other.**
`agent`'s bwrap guard tested a falsy attribute. `env_mgr` landing
`Prepared.wrap_argv` made it a **bound method — truthy** — so the guard stopped
firing and **a `kind: ai` task under bubblewrap would have run unwrapped.**
Nothing failed: the guard was still there, still called, and always passing.
`bwrap` is absent on this machine, so no suite would ever have said so. **The
change that disarms a guard need not be in the guard's file, or in its package.**

**And the procedural recommendation, which all four instances share: verify a
seam from the *consuming* side, against the code and not its description.** Not
one of the four was found by the package that owned the defect —

| found by | in |
|---|---|
| `task_graph`, reading `spec_loader`'s handover note | their own `check_graph` |
| `handoff`, taking `task_graph`'s generalisation and going looking | their own `put` |
| `closure`, after two other people grepped for the call site | their own missing producer |
| `agent` and `spec_loader`, reading declared shapes | `validator`'s, three times |

That is a fact about the method rather than about anyone's diligence: **an owner
tests the side they can see.**

**A path can be repaired at every point and still not deliver.** The
escape-hatch admission — `closure` criterion 6's whole subject — reached nobody,
**after three repairs and a dedicated gate on that exact path**:

1. `handoff` renamed `load_report` so the root's accessor matched.
2. `task_graph` removed the `getattr` default that had been turning the mismatch
   into `None`.
3. The G3 composition test was written **specifically to catch that path**.

And at the end of it, `bootstrap.py`:

```python
fatal = [p for p in problems if p.fatal]
if fatal:
    raise SpecInvalid(format_problems(fatal))
closures.freeze()          # everything non-fatal: computed, then dropped
```

**`Problem.fatal=False` has exactly one producer** — that admission — so the
non-fatal channel had one message and threw it away.

**Why none of the three found it, and this is the part to keep:** `handoff` tested
their accessor, `task_graph` tested their caller, the G3 gate tested that *a
report exists*. **Nobody asserted that a non-fatal finding reaches a human.**
Every repair was correct, verified, and left the end-to-end property false.

> **Test the delivery, not the links.** A chain of correct hops is not a
> statement that anything arrives.

**And the sharpest form a test can take this shape: an assertion satisfied by
the failure it was written to exclude.** `test_a_non_leaf_hands_its_thread_back_at_unfold`
asserted `status is RUNNING` — and a broken fixture made `Task.unfold` raise
`TaskStateError`, **which left the status exactly there.**

> **A test named for unfolding had never once unfolded, and was green
> throughout.**

Found only when unrelated work moved a call and twenty-seven tests failed loudly
— *the fixture working*, because that harness enforces the scope guard. Fixing
those exposed the two fixture defects that had been swallowing the raise.

**The tell is an assertion that a failure can also satisfy.** `status is RUNNING`
is true of *did the work and stayed* and of *raised before starting*; the test
needed the positive artefact — here, that the scheduler received submissions.

**Three practices, each of which would have caught one of them:**

- **No `getattr` default over a required collaborator.** If `load_report` is not
  optional, `r.get("handoff_specs").load_report()` fails loudly next time
  somebody renames it. A default turns a rename into a silent behaviour change.
- **No `| None = None` on a parameter the composition root always supplies**, and
  no early return on it. That default is what hid the first row for as long as it
  was hidden.
- **Test that the check *reports* on real input**, not merely that it returns
  `[]` on good input. A hand-built fixture is exactly the thing that hides a
  shape mismatch, because it is built to the shape the code expects.

**Eight packages tested in isolation cannot catch any of these.** All three were
found by one teammate asking another about a shape rather than assuming it.

### 4.12 A capability the spec requires, built and reachable by nobody

**A second family, and it is not §4.11's.** That one is a wrong value flowing on;
this one is **a right thing never invoked**. Three instances, all surfaced the
same day and all by someone standing outside the owning package:

| found by | capability | the missing route |
|---|---|---|
| `demo`, as the first caller of the whole system | **`HandoffStore.put`** | **no call site anywhere** outside `handoff`'s own tests |
| `validator` / `handoff` | `handoff.resolve` | no in-process consumer — §5.8 |
| `handoff`, writing the composition test | spec §5.3's **escape-hatch flag** | `bootstrap` builds registries as `parts[name]()`, so nothing in the assembled system can turn it on. Criterion 12's second half is reachable from a direct construction and not from the root |

**Its mirror image: a seam that is called and declared nowhere.** `monitor` calls
`runner.attempt_of`, `runner.resume`, `attempt.is_running`, `attempt.wake` — and
`attempt.executor`, which **this paragraph originally omitted.** The enumeration
was written by hand and was one short, which is §5.4's own failure committed
while describing a different one.

**No Protocol anywhere named any of them** — `Pushable` declares the *backend*
half and §4.9 says `monitor` resolves `runner` without saying what a runner must
provide. `monitor` did the hard version correctly for the backend and got neither
declaration nor test for the runner.

**That seam's undeclaredness is what produced the silent non-leaf defect**:
`_advance` branched on `attempt_of(tid) is None` because nothing stated what the
runner guarantees about an attempt's lifetime, and the fix then added a *fourth*
undeclared member to the same seam.

**Closed as two Protocols, not one — `AttemptRunner` (`attempt_of`, `resume`)
and `Attempt` (`executor`, `is_running`, `wake`).** A single `Runnable` was
proposed and is wrong:

> **The seam spans two objects with two lifetimes, and one Protocol cannot say
> so.** `attempt_of` returns something that outlives its thread; `is_running` is a
> question about *that object*, not about the runner. One Protocol declaring all
> five **would say a runner has `is_running` — which is the sentence that was
> wrong in the first place**, and whose absence caused the defect.

A flattened declaration would have documented the confusion rather than fixed it.

**And it bounds the conformance test in `tests/interfaces/`, which is worth
knowing before relying on a green run:** that test compares implementations
against what a `protocols.py` **declares**. A seam nothing declares is invisible
to it.

> **The conformance test closes drift, not absence.** `monitor` is green, and
> that green means less than it looks.

**And a second limit, found the same way: a constructor never named in a Protocol
is outside its reach.** `protocols.PhaseRunner` declares `run_phase` and no
`__init__`, so nothing compared `PhaseRunner.__init__`'s
`package_root: Path | None = None` against anything — and that default resolved a
**package-relative** body path against `Path.cwd()`, i.e. wherever the process
happened to start. It finds nothing with a puzzling message, or a **different file
of the same name**.

**Everything the composition root wires has that shape** — `PhaseRunner(strict_level)`,
`EnvManager(ctx)`, `Runner(r, config_order=...)`. The rule in §4.11 is right; the
test implements it well; **the coverage the rule implies is wider than the
coverage the test has.**

**Why no package suite can catch this, and it is a different reason from
§4.11's.** §4.11's defects hide because each side asserts its own half. These
hide because **a package can always construct its own object and prove the
capability works.** In `handoff`'s words:

> **Every one of my 137 tests exercises the flag; none of them could tell me the
> root cannot set it.**

**The sharper name, and it tells you where the sixth one is.** The family is not
quite *works when you construct it yourself*. It is:

> **A producer or a consumer with no counterpart — and the missing half is
> invisible from the side that exists.**

`validator` could see their index was unfed **because they own the write**.
Nobody could see `put` had no caller from inside `handoff`, **because a store
with no caller looks exactly like a store.**

**So look for the half nobody owns.** A name resolved by one package and
registered by none is finding C1; a method exported by one and called by none is
this. The second is harder precisely because the exporting package's tests all
pass — **they construct the caller themselves.**

**And the cheap detector is inconsistency *within* one package rather than
absence *across* two.** `monitor` had declared `Pushable` for the backend half
and nothing for the runner half, and **the asymmetry is what made the gap
visible.** A package that had declared neither would have looked consistent.

**The check is not "does it work" but "who calls it".** A grep for call sites
outside the owning package's tests answers it in seconds and nothing else does.

One consolation, worth knowing because it bounds the class: **guessing at a
*signature* fails loudly; guessing at a *value* does not.** A wrong parameter name
raises on the spot. That is why §4.11's family is the dangerous one and this
family is merely invisible.

---

### 4.13 A real value, produced and never delivered

**Distinct from §4.11, and the distinction is the whole point.** §4.11 is *a
plausible value produced and consumed as if it were real*. This is the mirror:
**a real value produced, and the artefact a human reads is empty.** No wrong
answer anywhere. Every component correct in isolation.

`agent-mod`'s phrasing, kept over a tidier one:

> **A failure that is recorded somewhere is not the same as a failure that is
> reported, and the gap is always paid by whoever is holding the artefact rather
> than the source.**

Three instances in one afternoon, all *"the information exists"* mistaken for
*"the information arrives"*:

| | |
|---|---|
| non-fatal `Problem`s | computed by `check_closures`, filtered away by the composition root — `closure` criterion 6's reporting reached nobody after **three correct repairs on that path**. **A fourth landed and closed it** (`bootstrap.py:280`), and `closure` then did the thing the other three did not: measured *arrival* against the real `build_registry`, **with a negative control**, because a probe showing a pass deserves the same scepticism as one showing a failure. `scratch/impl-2026-08/closure/probe_criterion_6_arrives.py`, kept |
| `Execution.detail` | the runner held it, the field had a home, `on_task_done` took it — and the declared callback type could not carry it, so every failed task recorded `''` |
| `demo`'s `_why_failed` | falls back to `Recorder.read` on every failure — a second reader of a fact that already has a home |

**A path can be repaired at every point and still not deliver.** That is the
non-fatal `Problem` case exactly: three repairs, each correct, none of which
made the value arrive.

**No package's suite can see this, because each ends at its own boundary.** The
G3 composition gate is the only instrument that could, and as written it asserts
that problems are **produced**, not that they **arrive** — so it would have
passed all three. That is a gap in the gate, recorded here rather than quietly
widened, because changing what G3 asserts changes what every package owes it.

`interfaces.md` §4.7 does not name `OnDone`, by shape or by name, so the
Protocol change needs nothing here. **The break was real and located** —
`tests/agent/test_runner.py:226` and `:739`, `lambda *a:` → `lambda *a, **k:` —
and 1642 green says nothing else depended on the alias.

**Found by a caller refusing a signature that would have worked.** `agent-mod`
had the exception in hand and declined to pass it: it would have succeeded in
production and broken every callback conforming to the declared type, including
their own doubles. That is §1.1 exercised in the direction it is usually not —
the caller refusing to exploit a signature that happens to accept more than it
declares.

### 4.14 Ruled: output versions are pre-allocated at dispatch

**The user's ruling, and it dissolves §5.14 rather than deciding it.**

The conflict was read as *`monitor` §4.1.1 says the producer calls `put` from
inside its zone; `env_mgr` §4.5 says an executor may not write outside its
zones.* **Measured, the mechanism was already most of the way there:**
`env_mgr/grants.py:96` resolves a kind-named grant to
`<store_root>/<hid>/v<N>/` — **a store path, not a zone path.** An agent holding
a `WRITE` grant on its own output is already granted a location in the store.

**What blocked it was chicken-and-egg on `N`, not the write rule.**

| | pinned when |
|---|---|
| **input** version | at **dispatch** — `Task.push_execution(agent_id, input_versions)`, `models.py:322` |
| **output** version | at **close** — and *read*, not allocated: `scheduler.py:187` takes `handoff_mgr.latest(hid)` |

So `versions.get(hid)` is `None` during the attempt, `resolve` skips the slot,
and `UnresolvedGrant` fires. **`put` was the thing that allocated the version**,
which is why it had to be called from inside.

**The ruling: allocate the output version at dispatch.** The
`<store>/<hid>/v<N>/` directory is created and granted before the body runs; the
agent writes into it directly, **inside its own grant, violating nothing.** `put`
stops being *the write* and becomes *the seal* — which a runner may do outside
the confinement.

**Failure is handled by changing what `latest` means:** it returns **the highest
*successful* version, not the stack top.** A pre-allocated version whose attempt
failed is a hole, and a hole is skipped rather than compacted.

**This survives what the pull model did not:** it needs **no spec withdrawn.**
`monitor` §4.1.1's *"the producer calls `put`, from inside its own zone"* was
protecting criterion 5 — *refused* versus *never attempted*, which only the
producer can distinguish — and the producer still writes from inside. Only the
version allocation moved.

**Measured, and it does pollute** — against the real `agent/gate.py`:

```
nothing at all            -> OUTPUT_ABSENT "never delivered"      (correct today)
a pre-allocated empty v0  -> RAISED FileNotFoundError: …/v0/manifest.yaml
    store.exists -> True      store.list_versions -> [0]
```

**Naive pre-allocation kills criterion 5's refused-versus-never-attempted
distinction, and kills it as an uncaught `OSError` rather than a verdict.**

**The fix is an invariant, not a signature** (`handoff`): **`manifest.yaml`
becomes the seal marker and `list_versions` reports only sealed versions.** Every
existing reader then works unedited, §4.14's *"highest successful version"* falls
out store-side for free, and it **keeps** `handoff` design §6.1's rule — *a
half-written version must not be visible as a version* — with the commit token
moving from `rename` to the manifest's existence. `put` was always the commit
token; the Protocol never promised `rename`.

**Two consequences, both ruled.**

**Holes are normal, not exceptional.** The version must exist before
`env_mgr.prepare` resolves the grant, and `prepare` raising is *"no isolation, no
start"* — so **every refused dispatch leaves a hole**, not only every failed run.
Allocating later is impossible because the grant needs the number. **A sealed-only
`list_versions` makes a hole inert** — invisible, never reused — so **correctness
needs no reaper.** Disk hygiene does, and **has no owner**; named rather than
absorbed.

**The grant moves to `<store>/<hid>/v<N>/content/`.** Verified:
`handoff/store.py:114` is `v<N>/{content/, validation.yaml, manifest.yaml}` and
`layout.handoff_version_dir` returns the `v<N>` level — so **a WRITE grant today
reaches the manifest.** Harmless while `put` did the writing; **with the manifest
as the seal, an agent could forge a seal.** Spec §3.3 says the digest is not a
security boundary, so this is inside the stated threat model — but the *seal*
being agent-writable is new, and free to fix now.

### 4.15 Ruled: an unchecked output is a fault, and it blocks

**The user's ruling.** `StrictLevel.NONE` means *no validation is performed*.
**Under every other level, an output phase that nothing checked is a fault, and
it blocks the task.**

**`validator`'s objection does not survive it, and the reason is worth keeping**
— they had ruled the answer *cannot* be "it blocks", because `NONE` folds every
phase to `empty` and criterion 20 forbids the level deciding outcomes. **The
ruling makes two kinds of `empty` distinguishable:**

| | `empty` means |
|---|---|
| under `NONE` | **no validation was asked for.** Expected, not a fault |
| under any other level | **nothing checked what this task produced.** A fault |

**Checked against criterion 20 rather than assumed.** Its wording is
*"`--validation-strict-level` changes which phases run, and **never which
verdicts bind**"* (spec:738). `NONE` not running a phase is **which phases
run** — squarely the level's business. The fault rule is **identical at every
non-`NONE` level**, so the level does not change which verdicts bind. **No spec
change is needed.**

**And the asymmetry the whole question turns on**, `validator`'s, recorded so
the distinction is not flattened later:

> An empty **input** phase means *this task consumes nothing.*
> An empty **output** phase means ***nothing checked what this task produced.***
>
> **Different claims, and only the second is a candidate fault.**

### 4.16 F19 reverses a third time: **stage, not grant**

**Ruled by the user, and the reason is a scoping correction rather than a new
argument.** I had been treating the package grant as a security-hardening
question. It is not:

> **Permission management here is wide. Its only job is to stop several agents
> cross-contaminating each other.** System security caused by an agent's own
> behaviour is **out of scope** — that belongs to the user's harness
> configuration, and in practice a harness runs with `bypassPermissions` on.

**Under that scope the two criteria separate cleanly**, and only one of them is
ours:

| | |
|---|---|
| `env_mgr` criterion **14** — *a sibling zone created after this sandbox was built is unreachable* | **This is the cross-contamination property, and it already holds.** The spec notes it is what an **allow-list** buys and a deny-list could not: a zone nobody anticipated is absent from the list, so it is unreachable by construction |
| criterion **13** — *a producing task cannot read a validation's checking standard* | **Anti-gaming, not cross-contamination.** The spec says it is *"resolved entirely by §5.1's containment"* — sibling placement, outside the producer's grant. **F19's package grant opened a second route the spec never anticipated**, because `validators/` lives in the package |

**So the ruling: stage.** A task receives a copy of what it needs; the package
root is not granted wholesale.

**The launcher measurement that reversed F19 to *grant* still stands** — a body
runs `python3 <package>/bin/collect.py`, so staging `entry.sh` alone is not
enough. **What changes the premise is the user taking ownership of the package
layout**: if a task's executable set can be named without `validators/` in it,
staging that set is coherent. `TODO.md` item 4a carries it, **and until it holds
staging moves the leak rather than closing it.**

**The cost, stated and accepted rather than discovered later.** A staged copy
lands in the **zone**, which the agent can write, so *a task may not modify the
package it was loaded from* stops being kernel-enforced — an agent can edit its
own body mid-attempt. **Under the scope above that is explicitly not ours to
prevent**, which is what makes the reversal consistent rather than a retreat.

**Third position for F19, and each move came from a measurement rather than a
preference** — stage (argued by two packages), grant (reversed on *a body is a
launcher*), stage (reversed on *what permissions are for*). **The middle
position was right about the mechanism and wrong about the goal.**

## 5. Seams left blank on purpose

Each is a decision that is not the implementer's to make alone. Each has a
direction, and the direction is not a default: **build to the interface, leave
the decision open, and raise it when you reach it.**

### 5.0a Index — what is open, and whose it is

**Applying §5.0 retroactively.** Nineteen entries, and most are not decisions
anyone is waiting on. Ordered by who is blocked.

#### Blocking a live run — **nothing, since §4.14**

| | |
|---|---|
| ~~**§5.14**~~ | **CLOSED — see §4.14.** Ruled: **output versions are pre-allocated at dispatch**, so the agent writes into its own granted store path and `put` becomes the seal rather than the write. **No spec is withdrawn** — `monitor` §4.1.1's producer still writes from inside; only the version allocation moved. The supervisor-side pull is superseded |

**`done_by_self_check` is CLOSED, and the entry above was wrong about why.**
It claimed *"no manifest exists when the gate runs"*. **Measured,
`agent/gate.py:90` does `store.get_manifest(hid, version)`** — the gate runs
against the **store**, so a manifest is exactly what it has. The claim came from
relaying a README rather than reading the call path.

**It is a `handoff` schema field on the `Manifest`**, and `gate.py:101` is
already written for it — deliberately tolerant of absence (*"absent means
`handoff` has not landed the field; only present-and-false is a failure"*), so
**it activates the day `handoff` lands it, with no change to `agent`.** It is
the one guard the `getattr` sweep kept.

#### The user's, but blocking nothing

| | |
|---|---|
| ~~**§5.11**~~ | **MOVED to `ROADMAP.md` §6.1 as P0**, and this row was wrong about the mechanism. It said *out-of-process works with no shim, the price is level 2 entirely* — **that was a fabricated causal story.** The real block is that `ClaudeSDKClient` takes a `cli_path` and **launches the CLI itself**, so the fork is not ours and our ruleset never reaches it. Whether level 2 is affected was never measured. The roadmap carries the two-layer mitigation and the reason it cannot be tested here |
| ~~**§5.16**~~ | **RULED — see §4.15.** `StrictLevel.NONE` is *no validation is performed*; under **every other level, an output phase that nothing checked is a fault and blocks the task.** Checked against criterion 20 and it does not conflict |
| ~~**new, from §5.19**~~ | **RULED — see §4.16.** *Stage, not grant*, on a scoping correction: permission management is **wide** and exists only to stop agents cross-contaminating. Criterion 14 (sibling zones unreachable) is that property and already holds; criterion 13 is anti-gaming and the spec resolves it by containment. Precondition is `TODO.md` 4a — the package layout must separate a task's `bin` from the validators', and the user owns it |

#### Ruled, waiting only on a build

**§5.15** — `prepared.spawn` is **decided and built, and not reachable until
`agent` calls it.**

#### Dissolved rather than decided

**§5.19** — F19 reverses to *grant*, so no fourth `prepare` parameter and no
`Prepared.body`. **§5.12** — two version allocators, dissolved with §5.14's
settled half.

#### Open and owned — nobody blocked

§5.1b, §5.2, §5.3, §5.6, §5.7, §5.9, §5.10. **Closed:** §5.1, §5.4, §5.5, §5.8,
§5.13.

#### The property nobody has, recorded so it is not assumed

**Retry comparability.** Absent under *both* F19 candidates — the staging case
for it was false twice, once on a retracted argument and once on the launcher
hop. If wanted, the body must be **pinned at dispatch** the way `input_versions`
pins an input. Nobody has proposed it.

### 5.0 Every entry carries **who is waiting**, and settling it means telling them

**A mechanism, because the failure it fixes happened three times in one day and
care did not prevent it.** Three decisions were settled *in a commit* and the
person blocked on each was not told; two of the three had that person **named in
the entry's own text**.

> *"Remember to tell them"* is an instruction that is never executed — the same
> reason the stub-expiry rule needed a failing test rather than a note.

So: **an entry names who is waiting on it, and closing it means sending that
line.** The marker was never the missing part — *Settled* and *Ruled* were
already written. **The missing field is the addressee.**

Proposed by the package that was waiting on one of the three, and scoped
correctly: the failure is not that decisions are slow, it is that **they are
broadcast to the document and not to a person**, and one field fixes that.

### 5.1 ~~Who fills `Handoff.type`~~ — closed

**Closed 2026-08-27.** `Task` gains `kinds: dict[HandoffId, str]`
(`task_graph` spec §3.2.6) and `submit` passes it to
`declare(..., types=...)`, which has taken the argument since spec rev. 4 and
which **nothing had ever passed**, because `Task` had no field to pass.

The route chosen is the one where a load-time check is possible: the mapping
comes from the task spec's declared kinds, so a name that is not a registered
kind is catchable before anything runs.

`env_mgr.resolve` keeps the `UnresolvedGrant` raise that was added while this was
open. It is no longer the only thing standing between a forgotten mapping and a
silently empty granted set, and it stays because being loud costs one raise.

### 5.1b `materials` is wrapped into a handoff by nobody

**Status: found by the spec-key-to-runtime trace, and it needs one decision.**

`closure` spec §2.5 rev. 9 gives a task spec a `materials` key — *things this task
may need for itself*, which the user's brief says *"最后会被系统包装成 handoff，
但是这里可以给用户自由"*. `validator` design §3.8 gives a validator body the same
key. **Nothing reads either.** Every occurrence in the design set is a reference
to the key existing.

The wrapping is a real step and not a formality. It means, at least:

| | The question |
|---|---|
| **Which kind** | A handoff needs one, and a kind needs a validator (`handoff` spec §5.3). Is there a built-in `materials` kind, or does the author declare one per task? |
| **Which checks** | `put` runs the README check and the locality check first. A `materials` folder a user was told they could be free about will fail the README check |
| **Which scope** | `addons.temp` fits — run-local, discardable — but that puts it in the playground rather than the handoff store |
| **Who mints it** | The graph builder has the spec; `env_mgr.prepare` has the zone. Only one of them can also `put` |

**The direction, and why it is not taken here.** `addons.temp` plus a built-in
kind whose validators are the minimum a store will accept is the cheap answer, and
it makes materials a first-class artefact for free — versioned, digested,
reachable through the ordinary grant mechanism.

**What makes it a decision rather than a default** is the user's own clause:
*"这里可以给用户自由"*. The freedom and the checks pull against each other. A
handoff that skipped the README check would be the first artefact in the system
that did, and `handoff` design §6.3 runs those checks *before* publication
precisely because retraction is unsolved everywhere.

**Build against the interface, not the answer**: whoever needs materials should
resolve them through `task.closure` → the task spec, and treat "how they become a
handoff" as one call behind a name.

### 5.2 `ValidatorId`

**Status: reported, not answered.** Main spec §4.6 asks for a fourth typed id
joining `TaskId` / `AgentId` / `HandoffId`. Three designs key validators by name
instead, consistently, and `validator` design §3.2 gives the reason — a validator
is not instantiated per run the way an `Agent` is, so what it needs is a unique
vocabulary entry rather than a per-object identity.

Nothing is blocked. Build on names; if the id arrives it is an addition, because
a name is what the verdict record carries either way.

### 5.3 Turning a closure into the root `Task`

**Status: owned by the wrong package, knowingly.** `closure` D5 declined it —
a helper returning a `Task` would make `closure` import `task_graph`. `demo` D3
took it, in ~60 lines, and said so.

```python
def root_task(closure_name: str, registry: Registry) -> Task: ...
def handoff_ids(closure_name: str, registry: Registry) -> dict[str, HandoffId]: ...
def wire(tasks: Sequence[Task]) -> None: ...
```

It moves to the whole-system CLI (`TODO.md` item 5) **unchanged**. Whoever builds
that imports these three and deletes `cli/build.py`.

`wire()` exists because `depends_on` is `list[TaskId]` — runtime ids — so no
jsonnet spec can carry it, and omitting it makes `scheduler._warn_depends_on` log
on every run.

### 5.4 ~~Which reference kinds "who uses this" enumerates~~ — **closed**

**Closed 2026-08-28 — by giving the enumeration one owner, not by deleting a
query.** `users_of` now spans **every** edge kind, because `check_closures` feeds
`bind_phase`. So there is no union at the composition root for a fourth kind to
break, which is what this section was open about.

`ClosureRegistry.closures_using_validator` **survives beside it**, answering a
different and typed question — §4.5. A withdrawal of it was ruled and then
reversed; the reversal is the settled position. The direction was
*"whoever adds a fourth edge kind owns making the enumeration derived rather than
restated."* What happened instead: `validator`'s `users_of` was made to record the
**edge kind** on each entry, and `closure`'s `bind_phase` was wired so the third
kind is fed. One owner, one representation, and no union at the composition root
for a fourth kind to break.

The record of what it was, and why it fired anyway:


**This one has now fired, in the package that cited the precedent.** `validator`'s
`EDGE_KINDS` was `("handoff_kind", "composite")` — **two of the three** — with a
docstring directly above it citing Airflow #58058, *false-positive deadness from
an unenumerated reference kind*, as the reason to keep the list in one place. So
`users_of` reported a validator that two closures run as **used by nothing**.

> **Citing a failure is not the same as being protected from it.** The docstring
> named the paper, the bug and the mechanism, and the tuple beside it was still
> wrong.

Only assembling all eight packages found it. The direction below stands and is
now overdue rather than hypothetical.


**Status: three edge kinds in three modules, no owner.** A handoff kind naming a
validator; a composite naming a member; a closure naming a phase validator.
`closure` §8.5 unions two of them at the composition root and says that does not
scale to a fourth. Two independent systems have shipped the failure this
prevents — Airflow #58058 and dbt #14436, both silent false-negative deadness.

Direction: whoever adds a fourth edge kind owns making the enumeration derived
rather than restated.

### 5.5 How a validation phase becomes attributable

**Status: split, and the residue is small.** `validator` §8.2 owns the
*requirement* — a phase must carry an `agent_id`, or criterion 10 is untestable.
`agent` O6 owns the *mechanism*, and one candidate is ruled out: not several
`session_id`s on one client, because `interrupt()` is connection-wide.
`fork_session`, `resume`, a subagent per phase and a second client all remain,
and none was tested.

Build the requirement as an assertion; choose the mechanism when there is a
backend to choose it against.

### 5.6 Garbage collection between an artefact and its verdict

**Status: unsolved everywhere, and deliberately not solved here.**
`delete_version` is absent from `HandoffStore` on purpose. OCI
distribution-spec#378 has been open since 2023. Nix's direction — roots point
*at* content, so a content-orphan cannot arise — is the one design that makes the
harmful case unrepresentable. Decide before anything is deleted, not after.

### 5.7 `SCHEMA_VERSION`'s owner

**Status: fine until the whole-system CLI exists.** `demo`'s event stream is a
versioned interface because criterion 14 asserts over it. The CLI will want the
same stream, and then two artefacts share one constant with no bump policy.

### 5.8 Who materialises the value a JSON Pointer addresses

**Status: opened by implementation, and it is the same shape as §5.1b.**
Measured, not predicted: **`handoff.resolve` has zero callers anywhere in the
tree.** Three documents say `validator` consumes it, and two of them are
`validator`'s own — §4.3 and §6 here, `validator` spec §4.1, and `validator`
design §3.6's *"this module consumes it rather than restating it"*.

The reason is structural rather than an oversight. `validator` spec rev. 8 made a
validator's implementation a **`Body`** — an `entry.sh` or a `readme.md` — so
whatever addresses into content is a shell script or an agent, running in the
zone. **There is no Python in that package between the spec and the content for a
pointer to be resolved by.** Rev. 8 withdrew the Python callable in the same
document whose §4.1 still names the pointer.

| | The question |
|---|---|
| **A body addresses its own content** | RFC 6901 becomes a convention a body follows over its whole `content/` copy, and nothing in Python resolves anything |
| **Something outside the zone resolves it** | The body is handed one value, and the caller is **whoever prepares the zone** — not `validator` |

**Not decided here, and deliberately not by either implementer.** Both `handoff`
and `validator` reached it independently, measured it, and declined to invent a
call site to justify a line in a document — which is the correct output.
`validator` spec §4.1 is **spec**, and a design does not amend a spec, so this
cannot be closed by editing three lines.

**`handoff.resolve` ships exported with no in-process consumer in wave 1. That is
a reported fact, not a defect.**

### 5.11 An AI backend cannot be confined **in-process** — soluble, measured

**Who is waiting: `agent`.** The word *"cannot"* was wrong and the measurement
says so — `scratch/impl-2026-08/env_mgr/p5_grandchild_inherits.py`:

```
unconfined grandchild            rc=0     read succeeded
grandchild of a confined child   rc=13    EACCES
```

The child was confined the way `spawn` confines it, then spawned the grandchild
**itself, with no wrapper** — exactly the shape of an SDK spawning its own CLI.
**The kernel denied it, because a Landlock domain is inherited by every
descendant and not only the immediate child.**

> **So F8 is *"an AI task cannot be confined in-process"*, not *"cannot be
> confined"*.** If the harness runs inside a `spawn`-ed child, the `claude` CLI it
> spawns is confined — **no `cli_path` shim, no argv interception, and no
> cooperation from the SDK.**

The candidate this entry used to record — a bwrap-wrapped shim behind `cli_path`
— **is not needed, and would not have been the cheapest answer.**

**The price, spelled out so that *soluble* is not read as *small* — it costs
level 2 entirely, not a handle:**

| | |
|---|---|
| `interrupt`, `instruct`, `query` | **all three of `AgentBackend`'s methods** are calls on an in-process `ClaudeSDKClient` |
| `monitor`'s `Pushable` | **is** that handle, reached through `attempt_of(task_id).executor`. Out-of-process the pusher degrades to escalate-only — the state it was in for a day and which was deliberately fixed |
| `interrupt`'s drain | reads `terminal_reason` **off the message stream**, so it needs the stream and not merely a control channel |

> **That is not a plumbing change. It is the whole reason `AgentBackend` is a
> second protocol.**

**The trade may well be worth taking** — a confined agent that cannot be
interrupted could beat an unconfined one that can — **but it is a decision about
what the system is. Roadmap, not alpha.**

#### And the step-7 split did **not** reduce what the system can safely run

**That was recorded here and it is half wrong.** Pre-split, Landlock confining the
runner's thread did mean a self-spawned CLI inherited the domain — **true in the
design, and reachable only from a single-threaded caller.** Under `agent.Runner`,
`apply()` refused.

> **In any configuration that actually ran, an AI task was never confined.** The
> split removed a guarantee that existed only in an arrangement that could not
> run.

`agent`'s test asserting the refusal under **both** mechanisms is right and stays
— it describes the current architecture honestly. **It is the reason in this
finding that was wrong, not the assertion.**

### 5.12 Two version allocators, and nothing joins them

**Found by `handoff` while answering `demo`'s question about who calls `put`.**
There are two independent version numbers for one artefact:

| | allocates |
|---|---|
| `task_graph.Handoff.open_next` | the **slot** version — `v+1` on the handoff record |
| `handoff.FilesystemStore.put` | the **store** version — the directory the bytes land in |

`HandoffVersion.seal(status, content=...)` is the only field that could carry a
reference between them, **and nothing passes anything there today.**

So publishing is not *"call `put`"*. It is ***"`put`, then `seal` with a
reference to the store version"*** — and **what that reference is has no owner.**
Built without it, the slot reads `VALID` and nothing records which stored bytes
it means: a validated artefact whose content is unidentified.

`agent.Runner` already resolves `handoff_store`, and `agent/gate.py` already
calls `exists`, `list_versions`, `get_manifest` and `copy_out` on it — **four
reads and no write, on the exact path that then fails for the artefact's
absence.** So the wiring already anticipates the runner as the writer; it is the
join that is missing, not the caller.

### 5.20 A non-leaf may declare an output no entry can produce — **`task_graph`'s, and it has no check**

**Found by `validator` while fixing §5.12's runtime half, on
`scratch/demo2-2026-08/depth2`.** `models.py::_instantiate` wires a parent's
outputs to its subgraph through **the end entry alone**:

```python
hid = mine.get(kind) if entry.is_end else None
```

So a kind the parent declares and the end entry does not produce gets a **fresh
id nobody writes**, while the entry that does produce that kind gets a *second,
unrelated* id of the same kind. Measured on `depth2`, whose `mid` declares
`outputs: [facts, notes]` over entries `produce` (facts, not end) and `second`
(notes, end):

| handoff | declared by | written by | published |
|---|---|---|---|
| `ec41080d` `facts` | `main`, `mid` | **nobody** | none — `main` pinned v0, `mid` v1, both holes |
| `a098acd1` `facts` | `produce` | `produce` | v0 |
| `e82176e8` `notes` | `main`, `mid`, `second` | `second` | v2 |

**`check_graph` admits it.** Its four checks are leaf-only acquisition, subgraph
containment, `froms`, and the `is_start`/`is_end` marks; containment fires only
when an **outsider consumes** the kind, and in `depth2` nobody does. So the
package loads, four tasks run, two artefacts are published — and `mid` dies in
`output_validating` on `Malformed: cannot read verdicts of ec41080d v1: it is
not published (published: none)`, a message about verdicts for a fault about
graph shape.

**The missing check, and it is `task_graph`'s side of the seam.** Per nesting
level, as `_check_subgraph_containment` already argues for itself: *a non-leaf's
declared output kinds must be a subset of its end entry's output kinds.* Reported
against the parent, naming the kind and the end entry — the same wording
containment already uses, *"export the kind from the end entry subtask"*. The
per-level form is what makes a transitive case report once: `depth2`'s `main` is
unfillable only because `mid` is, and `main` becomes correct the moment `mid`
does.

**`validator`'s side is the other half and is already built.** `PhaseRunner._targets`
asks `handoff` which version **is** published rather than trusting the parent's
pin, which is what makes the ordinary case — a parent whose end entry *did*
publish — work at any depth. Measured on `probe-depth2-no-orphan`, the same
package with the orphan removed and nothing else changed: one `notes` slot with
`main` pinned v0, `mid` v1, `second` v2 and published at v2, three distinct
numbers, and both non-leaves' output phases pass. It cannot rescue an orphan,
because there is no version to resolve to.

**Only `depth2` has one.** Scanned across the tree: `examples/demo`'s `main`
declares `outputs: []`, and `demo2`'s `main`/`grade` and `bringup/n1`'s `main`
each match their end entry exactly. So the shape is rare enough to have gone
unnoticed and cheap enough to reject at load.

### 5.13 A script body has no agent, and `Verdict.agent_id` is required

**Reported by `validator`, and the field is `handoff`'s.** An agent-bodied
validator now lands a fresh unbound `AgentId` in its verdict, distinct from the
producer's. **A programmatic body has no agent at all**, and at the time
of the report its verdict fell back to the producer's id with `attributed: False`
beside it. **Neither is true now** — see the closure below; `f9142aa` and
`b2411bf` removed both, and `attributed` appears nowhere in the tree. Reported by
`validator`, whose own file it is not: **the entry was closed at the bottom and
left describing the old behaviour in the present tense at the top**, so a reader
skimming for the problem statement got a decided question back as an open one.

That is deliberately ugly, and inventing an id is the one thing that must not
happen: **a reader takes an invented id for an independent checker**, which is
the misreading criterion 10 exists to prevent. Three honest routes, none chosen:

| | |
|---|---|
| `agent_id` becomes optional | the record says "no agent" by having no agent |
| a sentinel for "no agent ran" | explicit, and every reader must know it |
| the system accepts that programmatic validators are unattributed | and the schema says so |

**Closed 2026-08-28: `Verdict.agent_id` is `AgentId | None`** — route (a), and
the two rejected routes are worth keeping.

**A sentinel is strictly worse than `None`.** An `AgentId` is a UUID, so a reader
who does not know the sentinel **takes it for a real agent**, and one who resolves
it in `agent_mgr` finds nothing — §4.11's shape, in the field whose entire purpose
is attribution. The objection to *inventing* an id does not stop applying because
the invention is deliberate. **Documenting it** leaves the record stating a
falsehood with the correction in a side channel.

**It does not weaken criterion 8, and that reading is the persisting layer's to
make.** Criterion 8 says the history names the versioned agent; it is satisfied by
recording **the truth** about the agent, and a field holding a false id defeats it
rather than satisfying it. A stricter reading, not a relaxed one — and a reading
rather than a spec change, which is why it was made rather than asked for.

The stated cost was measured before the route was taken: **nothing in the tree
reads `Verdict.agent_id`**, and the widening breaks no constructor and no stored
record.

### 5.19 A confined `kind: program` task cannot read its own `entry.sh` — ruled: stage the body

**Who was waiting: `agent`, and `demo` next.** Found by the first end-to-end run
of the confinement seam anywhere in the tree:

```
entry.sh inside the zone  -> finished    home-read=DENIED  zone-write=yes
entry.sh outside the zone -> failed
    /bin/sh: 0: cannot open /tmp/agent-e2e-…/entry.sh: Permission denied
```

`Assignment.entry` is **package-relative**, joined against `Runner(package_root=)`
— the *task package*, which is not the zone — and **nothing stages the task body
into the zone.** Handoffs are staged; `material.deploy` places `rules` / `hooks`
/ `skills` under `<zone>/config/`; **`closure` spec §2.6's `readme.md` and
`entry.sh` are placed by nobody.**

M3's shape one artefact over, with the same signature: **the sandbox refusing an
artefact, reported as the artefact being missing.**

#### Ruled: **grant the package root read-execute.** Do not stage.

**Reversed.** I ruled *stage* on a separation argument, and it is not a
discriminator — a measurement shows the grant is needed either way.

**A body is a launcher.** `bodies/produce/entry.sh` execs
`python3 <package>/bin/collect.py`, and `closure` spec §2.2 makes that the normal
case: *"a large part of the reference workflow is running a command someone
already wrote."*

> **So staging `entry.sh` and re-execing it five times still launches a file the
> package holds. Half-immutable is not immutable** — and the package has to be
> readable regardless, so staging does not avoid the grant. It adds to it.

**And then the safety comparison inverts.** A grant is **read-execute, never
write**. A staged copy lands in the zone, and **the zone is writable by the
agent** — so under staging an agent can edit its own body mid-attempt and nothing
notices, while under a grant *"a task may not modify the package it was loaded
from"* is **enforced by the kernel**.

§4.5 supports it directly rather than tolerating it: writes are *no exception*,
reads are an allow-list *"generously declared, not open"*. **A task package is the
paradigm case for a declared read.**

**Derived, not hand-written per `Context`** — the way `interpreter_grants()`
derives the interpreter's prefix. `demo` has built it (`demo_grants(..., package=)`
with `test_the_package_is_granted_read_only` asserting both halves including
absence when no package is named); it moves into `env_mgr`, because every package
needs it and hand-written-per-caller is the `stub-resolv` shape.

**`materials` cannot substitute**: it is for *artefacts a task needs*, §5.1b's
"how materials become a handoff" is open, and enumerating every transitively
reached file of a program would be got wrong constantly — with F19's own failure
mode, a shell error naming the file, once per import.

**So F19 dissolves rather than being fixed.** No fourth `prepare` parameter, no
`Prepared.body`, no `deploy_body`.

#### The separation concern is real and is **not** F19's

A granted package root lets a `kind: program` task read every other task's body
and the `validators/` it is about to be judged against — criterion 13's subject.
**That was my argument for staging, and it does not discriminate**, because a
launcher body needs the package granted either way. It is a property of the
system under both options, and therefore **a separate open item: what a package
grant should contain.** Narrowing it is a real question; F19 was not the place it
gets answered.

#### And the property nobody has

**Retry comparability is absent under both candidates**, and the staging case for
it was false **twice** — once on `env_mgr`'s retracted reason, once on the
launcher hop. If it is wanted, the body must be **pinned at dispatch** the way
`input_versions` pins an input. Nobody has proposed it; it is F-D12's shape a
third time.

#### The mechanism: a fourth parameter, and `Prepared.body`

```python
prepare(task, execution, agent_spec, body) -> Prepared    # body: Mapping[str, str]
```

**The fourth parameter exists for exactly the reason the third does.**
`agent_spec` is passed because `env_mgr` must not read the agent registry; the
body is passed because **it must not read the task-spec registry**. `agent` is the
caller holding both, already resolves `package_root`, and hands over absolute
paths — so `env_mgr` learns no key meaning and no path convention.

The precedent is in `env_mgr`'s own package: **`material.deploy(agent_spec, zone)
-> dict[str, str]`** takes a spec, places files, parses nothing, returns a
mapping. The body wants the same verb one spec over.

**This does not contradict §2.8's run-configuration ruling.** That one merged five
parameters that are **per-run** — `strict_level`, `config_order`, the two roots,
`mandatory`. `agent_spec` and `body` are **per-task**, arguments to a call about
one task. Different axis, and collapsing them would be the mistake, not the fix.

**`Prepared` gains `body`** rather than the two packages agreeing a convention
like `<zone>/body/<name>` — **a convention is two declarations of one fact in two
packages**, the shape that produced four separate findings this stage. And
`agent` needs it regardless: after staging, `Assignment.entry` must point at the
**staged** copy, or the executor runs the un-staged file and the comparability
property buys nothing.

#### The asymmetry is the whole content of the decision

Ruling **grant** costs `agent` nothing — `Assignment.entry` keeps pointing into
the package and no one writes anything. Ruling **stage** is one line for `agent`
and a small build for `env_mgr`.

> **So the cost falls on the system rather than on either package: one way keeps
> a reproducibility property, the other loses it silently, and neither shows up
> in any test.**

### 5.14 What publishes a handoff — **CLOSED, and not the way this entry says**

> **Read §4.14 first. This entry is kept for its reasoning, not its conclusion.**
> The supervisor-side pull below was correct about the problem and **superseded**
> by the user's ruling: allocate the output version at dispatch, and the agent
> writes into its own grant. **Everything below describes the state before that
> ruling.** It is left because the objections it survived still apply to any
> future proposal, and because deleting the losing argument is how the next
> person re-proposes it.

**Who was waiting: `demo`, `agent`, `handoff`, `task_graph`, `env_mgr` — all five.**
Three entries turned out to be one question, and three independent constraints
close it. **The output is written into the zone and published from there.**

| | |
|---|---|
| **F-D1** | nothing calls `put`; who publishes? |
| **F-D1a / §5.12** | `put` and `open_next` are two version allocators; what carries the reference? |
| **F-D12** | a write grant needs a version that does not exist yet |

#### The three constraints, each measured by a different package

**1. The scheduler cannot pre-fill the version.** `Execution.output_versions` is
set by `close_execution`, at the *end* of an attempt. For the scheduler to put a
number there at dispatch it would have to decide which version the run will
write — and allocating one is `Handoff.open_next`'s, called by the agent.
`tests/task_graph/test_authority.py:235` **fails if a scheduler frame reaches
`open_next` or `seal`**, and `scheduler.py` never mentions it. So any answer that
resolves an output grant from the `Execution` asks `task_graph` to break its own
authority boundary.

**2. The runner cannot publish.** `agent/gate.py:73,81` calls `store.exists` and
`store.list_versions` — **the gate reads the store, not the zone.** Publish after
it and the gate finds nothing (`OUTPUT_ABSENT` for every task, which is what
`demo` observes). Publish before it and **the gate is checking its own
publication** and passes by construction — `handoff` spec §5.3 and `monitor`
§4.1.1.

**3. A store-path write grant is unresolvable.** `env_mgr.grants.resolve` needs
`<store_root>/<hid>/v<N>/` and takes `N` off the `Execution`:

```
at prepare time (output_versions empty)  -> UnresolvedGrant
after the task closed  (v0 recorded)     -> ['/tmp/store/<hid>/v0']
```

And it is universal, not a demo shape: **`closure` check 6 requires a write grant
for every declared output.**

**4. And a fourth constraint I did not check before ruling.** `env_mgr` spec §4.5,
verbatim:

> **Write — A task's executor may not write outside its zones. Local or remote,
> no exception.**

The store root **is** outside the zone — that is why inputs are *staged in*
rather than granted in place. **So a confined producer cannot call `put` either.**

> **The half that is settled: an output is written into the zone, not into the
> store.** Four constraints agree on that.
>
> **The half I ruled and should not have: *the producer publishes*. §4.5 forbids
> it, and I called the shape "forced" on three constraints without checking the
> fourth.**

**And `resolve` producing a store path for a write grant is a defect in `env_mgr`
and has been since it was written** — the `UnresolvedGrant` `demo` hit is that
defect failing loudly instead of quietly granting something the spec forbids.
Candidate 2 (`<root>/<hid>/`, unversioned) is **worse** on this axis rather than a
compromise.

#### The precedent, already shipped and measured

`env_mgr.workspace.collect`, and it is the same problem solved once for git:

> *"of the three ways work can come back only this one is admissible: the agent
> pushing needs the main repository writable, which is the grant §7.1 forbids. So
> the write happens on the main side, performed by the process that already holds
> write access, **outside the agent's confinement**."*

**Publication as a supervisor-side *pull* from the zone, rather than an
agent-side *push* to the store.**

#### What is actually open, and it is one question

> **Does publication happen outside the confinement, or does the write rule have
> an exception nobody has written down?** If the second, it should be written
> deliberately rather than discovered.

**`agent`'s objection is the one thing the precedent does not answer**: if a
supervisor publishes, the completeness gate is checking its own publication.
`collect` never faces that because nothing gates a git branch.

#### Resolved by the pull model — and it survives the objection that produced it

`agent` proposed it **against their own position**, having argued the opposite:

```
agent writes outputs into the zone   inside the confinement — §4.5 holds
gate checks the ZONE                 the agent's work, not the runner's act
runner calls put                     outside the confinement — collect's shape
OUTPUT_VALIDATING                    validators read the store, unchanged
```

> **The gate asks *did the agent deliver*, and publication happens after it. That
> is a different question from *was something published*, which is the one that
> passes by construction.**

**And one of the two cited sources breaks, not both.** `handoff` §5.3 (*"hands the
store a directory"*) is compatible either way. **`monitor` spec §4.1.1:373 is the
contradiction**, verbatim: *"**The producer calls `put`, from inside its own
zone**"*. **Two frozen specs disagree** — `monitor` §4.1.1 against `env_mgr` §4.5
— which is a finding for the user, not an edit for anyone here.

**`monitor`'s own open requirement closes for free.** §4.1.1 asks that *"attempted
and refused"* be captured at the moment `put` refuses, and assigns it to `agent`.
Under the pull model the runner calls `put` and sees the `Malformed` directly.
**The sentence that contradicts §4.5 and the requirement it states are answered
by the same change.**

#### What it costs, stated now rather than discovered

Two of three are improvements: `OUTPUT_ABSENT` becomes *"the agent did not produce
this"* — what the check always meant — and `OUTPUT_NOT_EXECUTABLE` loses its
`copy_out`-to-a-temp-dir.

**The third has a deadline.** `done_by_self_check` is read off the `Manifest`, and
no manifest exists when the gate runs under this model. Measured: it appears
**only in `agent`'s test stubs**, and their own test docstring says *"does not
exist on `handoff` yet"* — `monitor` §4.1.2 and §9 carry it as an unbuilt
`handoff` item.

> **So it is cheap to land as a zone artefact now and expensive to move after
> `handoff` builds it on the manifest.** That is the one piece of this with a
> clock on it.

Nothing needs a version before the write. §5.12's two-allocator join dissolves,
and a write grant on an output kind **should not resolve to a store path at all**.

#### What remains, and it is the user's

**Does the confined agent hold a `HandoffStore`** — making the store root writable
from inside the zone — **or write to a conventional path that something else
commits?** The second reintroduces a publisher the gate then checks, so the first
is the shape; **whether that grant is acceptable is a security decision, not a
mechanical one.**

**Four correct refusals produced this.** `handoff` declined the zone→content
convention (*"the store has no opinion about the playground's shape"*); `agent`
declined to publish from the runner; `task_graph` cannot pre-fill the version;
`env_mgr` says the caller must hold the parent's `Execution`. **Each was right,
and what they were all refusing was the same unowned step.**



**§4.12's first instance, now with an argument about the owner.** Nothing calls
`HandoffStore.put`, so `agent`'s completeness gate reports `OUTPUT_ABSENT` for
every task in an assembled run and `demo`'s `run` verb is blocked.

**The runner is not the missing caller, and the design says so twice.** `handoff`
spec §5.3 and `monitor` §4.1.1 both put `put` **inside the producing agent's own
zone**, and §4.1.1 describes the gate from the other side — *"the runner… sees
only the later absence."*

> **If the runner published, the gate would be checking its own publication and
> would pass by construction.**

So the gap is the **tooling an agent uses to publish from inside its zone** —
agent material, like the log tool. Two shapes, and the choice is a security
decision rather than a mechanical one:

| | |
|---|---|
| the tool holds a `HandoffStore` **in the confined agent's process** | keeps `put` as the commit token and the gate honest. But the store root becomes writable from inside the zone, and a confined agent can reach versions it does not own |
| the agent writes to a **conventional path**, something else commits | a narrower grant — and it makes the runner a publisher again, which is what this section just argued against |

#### The gate reads the store, which rules out the second shape

Measured — `agent/gate.py:73,81`:

```python
if not store.exists(hid): ...          # OUTPUT_ABSENT
versions = store.list_versions(hid)    # "exists with no version"
```

**The gate checks the store, not the zone.** So:

- **The runner cannot publish *after* the gate** — the gate would find nothing,
  because publication has not happened. It reports `OUTPUT_ABSENT` for every task,
  which is what `demo` observes today.
- **The runner cannot publish *before* the gate** — then the gate is checking its
  own publication and passes by construction, which is `handoff` spec §5.3's and
  `monitor` §4.1.1's objection.

> **So publication must happen inside the producing agent's zone, before the
> gate runs. That is forced, not chosen.**

**What remains open is narrower than two shapes: it is what the agent holds.** A
`HandoffStore` in the confined process keeps `put` as the commit token and needs
the store root writable from inside the zone; a conventional path needs something
else to commit, and that something is back to being a publisher the gate then
checks.

**Nothing is being worked around while it is open**, deliberately: *a green demo
whose sandbox and completeness gate are both no-ops is worse than a red one.*

### 5.15 What applies a policy to a body — **decided, not yet reachable**

**The question is answered and the answer is `prepared.spawn`.** Three
independent routes arrived at it — a validation body, a task body, and the
mechanism itself — and one call confines a validator's `entry.sh` and a program
task's argv alike.

**It is not closed, and the distinction matters here more than anywhere:**

```
env_mgr/protocols.py:277   def spawn(argv, **popen_kwargs)     declared
env_mgr/prepare.py:79      def spawn(...) -> subprocess.Popen  built
agent/runner.py:789        "...prepared.spawn(argv, **kw) applies"   a comment
```

**`agent` has not called it.** Until it does, `spawn` is **§4.12's own shape — a
capability built, correct, tested, and reachable by nobody** — which is the trap
this contract has now catalogued five times. **A decision is not a delivery.**

This entry closes when there is a call site, not when there is an implementation.



**Retitled from *"what confines a validation body"*, and the misfiling was the
finding.** The validation case surfaced first only because that ruling landed
first. **It is about every body** — the same measurement says the identical thing
about a *task* body, so filing it as a validation question made it look smaller
than it is and gave it to the wrong owner.

**Measured, one process, ABI 3, with the guard removed on purpose — to see what
it refuses rather than to predict it:**

| | |
|---|---|
| worker thread, writing outside | **denied**, `errno=13` |
| worker thread, its own zone | writable |
| **MAIN thread, writing outside** | **WRITABLE** |
| **subprocess of the worker** | **denied** |

**Row 4 decides the shape and is the good news.** A subprocess inherits the
domain, so **the property is reachable in this architecture**: confinement applied
in the process that becomes the executor, between fork and exec.

**Row 1 is the killer, and it is functional rather than a policy problem.** The
thread that applies confinement **is itself confined, irreversibly** — and a
runner thread has to write the store afterwards to record the outcome. **It would
surface as a store bug, not a sandbox one.**

**Row 3 is the honesty problem**: the main thread stays unconfined while
`Confinement(filesystem=True)` reports success — true of the executor, false of
the process.

#### `env_mgr.prepare` can never succeed under `agent.Runner` on Landlock

`apply()` refuses when `threading.active_count() > 1`, correctly, for row 3's
reason. **`agent` runs one thread per attempt** (design §7.5, no pool), so when
`Runner` calls `prepare` there are **two threads and two is the floor** —
`MainThread` plus `attempt-…`. A monitor loop adds one; each concurrent task adds
another.

It does not fire today only because an earlier raise gets there first. **Fixing
that one does not unblock anything; it moves the wall one line.**

**This was escalated as a four-way architecture decision. It is not — three of
the four are refuted against the measurement, and one survives:**

| shape | |
|---|---|
| confine before any thread starts | the process is then confined to one task's zone before it can run anything, so it can only run that task. **That is "a process per task" with extra steps** |
| **a process per task** | **the only one that works.** Confinement applied in the process that becomes the executor, between fork and exec. Measured (row 4), and `tests/env_mgr/conftest.py::run_confined` already does it ~20× per suite run |
| require Landlock **ABI 8** and `all_threads()` | **wrong on its own terms, not merely unavailable here.** `all_threads` restricts *every* thread in the process — which is the supervisor. The runner would lose the ability to write the store, dispatch other tasks, or reach any zone but this one. A newer kernel does not make it correct |
| accept **serial execution** | **does not fix it.** Serial still leaves `MainThread` plus the attempt thread — and even at `active_count() == 1`, running the attempt *on* the main thread confines the runner process itself, and row 1 says it then cannot write the store. Irreversibly |

**So the architecture does not have to change.** What has to change is *where the
syscall happens*, and the answer is the one `wrap_argv` already implements for
bubblewrap.

#### Ruled: step 7 splits. `prepare` checks; `spawn` applies.

**`prepare` stops confining.** It **checks** that a mechanism exists and refuses
early; **`prepared.spawn(argv, **kw)` applies it in the child.** One verb, three
mechanisms, and the caller branches on none.

**Design §11.1's *"confinement last"* survives in the form that mattered.** That
rule existed so the supervisor and every prior process stay outside the domain —
and moving the syscall into the child achieves it **by construction rather than
by ordering**, which is strictly stronger than the sequence that expressed it.

**Criteria 8 and 14 are intact and one is improved:** *no isolation, no start*
now refuses **before the workspace is cut** rather than after.

**The alternative is not a trade, it is a dead end.** If `prepare` keeps
confining, `agent` cannot use `spawn` at all and there is no answer at all —
that is what the four-shapes-collapse-to-one result establishes: **no arrangement
exists in which a threaded runner confines itself and still works.**

##### And `wrap_argv`'s shape does not transfer, which is why this is a spawn

bubblewrap **is** the exec, so its confinement crosses fork/exec as **data in a
command line**. Landlock is a syscall against a live thread and must run in the
child. **There is nothing to put in an argv.**

##### The evidence, and its limits, stated by the person who first overstated it

The mechanism was first offered as *"proven — `run_confined` does it twenty times
a test run."* **pytest is single-threaded**, and the hazard is fork in a
*threaded* process, so that fixture said nothing about it. Measured properly, 150
rounds each under four threads contending the allocator and import lock:

| | hangs | |
|---|---|---|
| A build the ruleset in the child | 0 | 7.12 s |
| B build in parent, restrict in child | 0 | **111.14 s** |
| C `Popen` + `preexec_fn` on a pre-built ruleset | 0 | shipped |

**C ships because a ruleset fd survives fork**, so the child does `prctl` +
`restrict_self` and **nothing that allocates**.

**Evidence, not proof** — a fork deadlock is probabilistic and CPython warns on
principle. And keep B: **building a ruleset in a threaded parent is GIL-bound and
~15× slower** than in a single-threaded child.

#### The shape of the verb

**`prepared.spawn(argv, …)`, not a bare post-fork callable, and not
`ProgramExecutor` calling `apply` itself.** Three reasons and the first is the
one that decides it:

- **The rules for what is safe between fork and exec in a threaded process stay
  with the package that knows them.** `subprocess.Popen(preexec_fn=…)` is
  documented unsafe in exactly `agent`'s situation, and `env_mgr.apply()` already
  refuses above one thread for the ABI-3 reason. Handing `agent` a callable hands
  them that constraint without the knowledge that goes with it —
  `engineer_principle.md` §3.
- **`wrap_argv` is the precedent and it is on `Prepared`, not on `EnvManager`** —
  the returned value answering a question about itself, not the component making
  a decision. `spawn` is the same shape for the other mechanism.
- **It unifies both mechanisms into one verb**, so `ProgramExecutor` stops
  branching on which rung it got. That is a consequence rather than the argument.

**Unchanged: the AI backend.** The SDK spawns its own CLI, so there is no fork of
`agent`'s to confine in, and `accept_confinement`'s refusal is still the only
thing standing there.

**Half the answer already exists: `wrap_argv` is exactly this for bubblewrap.**
bwrap *is* the exec, so wrapping the argv puts confinement in the child by
construction. **Landlock has no twin and needs one of the same shape.**

The candidate *"the caller applies a returned policy"* fails for **both** bodies,
for one structural reason: **the caller is the supervisor**, so applying there
confines the supervisor.

**Nothing runs unconfined meanwhile.** Every raise fails in the correct direction
— `NoConfinement` → `HANDLING_FAILED` → task `FAILED`. The system refuses rather
than pretending, which is criterion 14 working and why this is a decision rather
than an emergency.

**And criterion 13's separation rests on placement, not the kernel.**
`prepare_validation` confines nothing: the standard is unreachable *by where it
is*, not *by what the kernel permits*.

### 5.9 Who calls `install_excepthook`

**Status: named by §2, called by nobody.** `task_graph` declined it and gave the
right reason: §2 writes both arguments as `...`, and **installing a process-global
hook over guessed objects is worse than leaving it out.** The need is real and
measured — a thread that raises prints a traceback and dies with the process exit
code unchanged and every producer seeing nothing.

**The question is sharper than "who supplies the arguments", and the arguments are
now buildable anyway** — `recorder` is a registered row and `NullUserSink`
exists. `threading.excepthook` is **one slot for the whole interpreter**. So the
real question is *may a composition root mutate interpreter-global state at all*,
and the answer is no: a library function that claims that slot takes a decision
belonging to whoever owns the process, and two registries built in one test
session would fight over it.

It belongs to the entry point — the whole-system CLI (`TODO.md` item 5) or
`demo` — which calls it with `r.get("recorder")`. Until one exists, an uncaught
exception in an attempt thread is silent.

### 5.10 Who owns `subgraph_of`

**Status: built in the wrong package, knowingly, and reported.** `unfold`,
`replace_with` and `check_graph` all need to read a task spec's subgraph.
`closure` spec §2 says a task spec carries *"— if it has one — its subgraph"* and
**never names the key**; `closure/protocols.py` has six accessors and no
`subgraph_of`. `task_graph` defined `models.subgraph_of` reading `task.subgraph`
as `[{closure, is_start?, is_end?}]`, marks defaulting to first and last.

`closure` owns the other six accessors over a `ClosureDoc`, so a seventh living in
`task_graph` is one accessor away from its family. Direction: it moves to
`closure`, and `task_graph` deletes its copy — but the *key and shape* are named
by no spec, so somebody has to fix those first.

---

## 6. Nine packages, and who can start when

Nothing here is a schedule. It is the dependency truth, so that nine people can
work without blocking on each other.

| Wave | Package | Depends on | Notes |
|---|---|---|---|
| **0** | `spec_loader` | — | Everything waits on `Problem` and `SpecRegistry`. Start here, and finish it first |
| **1** | `handoff` | `spec_loader` | `digest`, `readme`, `pointer`, `locality` are pure functions over bytes and can be written before anything else in the package exists |
| **1** | `validator` | `spec_loader`, `handoff` | Only for `Verdict` and the Pointer resolver. Stub both against `protocols.py` and start immediately |
| **1** | `closure` | `spec_loader` | The six checks need only `Registries`, which is a Protocol a test satisfies with five dicts |
| **1** | `agent` | `spec_loader`, `task_graph` | Both exist. `backend.py` imports nothing of ours |
| **1** | `env_mgr` | `task_graph` | Exists. The new subtree is below the wall and touches no shipped file except `cli.py` |
| **2** | `task_graph` rev. 12 | — | Three small changes to unimplemented rev. 11 material. Do it early; two other modules read the result |
| **1** | `monitor` | `task_graph` | Exists. Imports nothing else of ours — `Pushable` is declared locally so the dependency stays one-way (§4.9) |
| **3** | `demo` | all eight | The integration surface, by construction |

**Wave 1 is six packages in parallel and no two of them touch.** That is what
the import graph was for. `monitor` joins it rather than waiting on `agent`,
and that is the whole payoff of §4.9's structural declaration: the two packages
that must talk to each other at runtime can be written at the same time by
different people.

### 6.1 What integration is allowed to change

The user's rule, and it is the right one: **at integration, code may be debugged,
changed, and adapted.** Nothing here is a promise that the first assembly works.

What integration should *not* have to do is discover that two modules meant
different things by one name. That is what the fifteen findings were, and it is
what §3 and §4 are for.

---

## 7. The dependency declaration

**Landed 2026-08-28** — `handoff` owns the block and all eight entries are in
`agent_sys/pyproject.toml`, plus `[tool.setuptools.package-data]` from
`spec_loader` so the five schemas actually ship. What follows is the measurement
that produced it. `agent_sys/pyproject.toml` declared three
runtime dependencies and the design set needs twelve. **`monitor` adds none** —
its design §10 says why for each candidate, and the short version is that the
workqueue shape is sixty lines of Go worth copying rather than a dependency, and
the OpenTelemetry SDK answers the wrong question because it is emit-only. **Nine are installed on
this machine and declared nowhere** — most as transitive dependencies of
something else — so the suite is green by accident.

```toml
dependencies = [
  "pyyaml>=6",
  "packaging>=23",
  "pydantic>=2",
  "jsonnet>=0.20",          # main design §8 — render. No aarch64 wheel; see below
  "jsonschema>=4.18",       # main design §8 — the only enforcement point
  "python-jsonpath>=1.1",   # handoff §8.4 — Pointer with three-way failure
  "markdown-it-py>=3",      # handoff §9.2 — CommonMark, not a regex
  "rfc8785>=0.1",           # handoff §4.6 — JCS that raises instead of rounding
]

[project.optional-dependencies]
claude = ["claude-agent-sdk>=0.2.144"]     # agent §8.1 — 376 MB, ~1.3 s to import
dev = ["pytest>=8"]
```

Three corrections the measurement forced, each against a design that recorded the
opposite:

- **`python-jsonpath` is the one that is NOT installed**, and it is the library
  `handoff` §8.4 chose after measuring six. Both libraries it *rejected* —
  `jsonpath-ng` and `jsonpointer` — are present. A test written today would pass
  using a rejected library and fail on a clean install.
- **`rfc8785` 0.1.4 IS installed.** `handoff` O2 recorded it as absent.
- **`rjsonnet` 0.5.6 is installed too**, which main design O2 did not know when
  it flagged `_jsonnet`'s missing aarch64 wheel. The fallback is already present
  and the seam O2 asks for is one function in `render.py`.

**No type checker is installed** — neither `mypy` nor `pyright`. `task_graph` O11
already notes that criterion 27's static half asserts a tool nobody runs, and it
now applies to the six `protocols.py` files as well: they are checkable, and
nothing checks them. Adding `mypy` to `dev` and one CI step is the cheap fix and
is not made here.

---

## 8. The importable half

Seven files, one per package, carrying the same contract as §3 and §4:

```
spec_loader/protocols.py      Problem, SpecRegistry, Registries, LoadReport, …
handoff/protocols.py          HandoffStore, Content, Verdict, Manifest, …
validator/protocols.py        Validator, Reducer, PhaseRunner, PhaseOutcome, …
agent/protocols.py            Executor, AgentBackend, AgentResult, Runner, …
closure/protocols.py          ClosureRegistry queries, check_closures, …
env_mgr/protocols.py          EnvManager, Prepared, Context, Mode, …
monitor/protocols.py          Monitor, EventKind, PLANNED, Recorder, Pushable, …
```

Each has a `.pyi` beside it. **Declarations only** — Protocols, enums, aliases,
exception classes, and signatures whose bodies are `...`. No behaviour, so
importing one costs nothing and a circular import is impossible.

They exist so that a seam is checkable rather than remembered: an implementation
can be written against `handoff.protocols.HandoffStore` and a test can assert it
satisfies the Protocol, without `handoff` and `validator` being written by the
same person in the same week.

**And they are reportable, exactly as §1.1 says.** If one is wrong, it gets
changed — in one place, with both sides named.

**`monitor/protocols.py` is the case for all of this, written small.** The
monitor and the agent must call each other at run time and may not import each
other, so the shape they share is declared twice — once as `AgentBackend`, once as
`Pushable`. That is precisely the duplication `engineer_principle.md` §1 names,
and it is admissible here only because `tests/interfaces/test_pushable.py` fails
when the two drift. **The test is not a nicety attached to the decision; it is
the decision's price**, and without it the honest move would have been the
package cycle.

### 8.1 Forced duplication requires a drift test — the general rule, rev. 5

**`Pushable` was not a one-off, and implementation proved it five times over.**
Asked directly whether the rule generalises, the answer is yes:

> **When the import graph forbids the edge that would give one fact one writer,
> the fact may be declared twice — and only if a test in `tests/interfaces/`
> fails when the two drift.** No test, and the choice is between removing the
> duplication and taking the import edge. A test may import both packages;
> tests are not under §4's rule.

**And the test for whether a duplicate should be relocated at all — the same
question asked one step earlier:**

> **Is the duplication forced by the import graph?** Two packages need one
> accessor, across an edge one may not cross, and **no lookup either can make
> instead**.

The criterion tried first and wrong is **consistency** — *"the next reader finds
`Body` unified and `task_of` not, with no visible reason."* It reduces to *any key
with two readers moves*, which relocates §4.5's export list into the leaf one key
at a time without anyone deciding to. Sorted against the right criterion:

| | Forced? | |
|---|---|---|
| `Body` | **yes** | `$ref`ed by two kinds; three declarations in three packages, and they disagreed |
| `task_of` | **yes** | `closure` and `task_graph._instantiate` both hold closure documents; `task_graph` may not import `closure`, and the reverse is a cycle |
| `agent`'s `task_of` | **no** | a by-name lookup existed. **Deleting the *need* was the better fix**, and they did it |
| `closure`'s other five accessors | **no** | one reader each. `task_graph` reads `doc["agent"]` **raw** and never calls `agent_of` |

**The slope was looked for and measured empty**, which is why *no* is available
rather than merely asserted. `closure` remains the owner — `model.py` re-exports,
so it stays the one place a `closure` reader looks while no longer being a second
writer.

**A name in an `__all__` can be a re-export.** `validator` believed dropping its
`Body` dataclass needed a ruling because §4.3 lists `Body` in its exports. It does
not: `from spec_loader.protocols import Body` satisfies §4.3 unchanged and leaves
one declaration — probed against both interface tests rather than reasoned, since
an import is not a `ClassDef` and the stub comparison sees it in neither file.

Five instances, all found on 2026-08-28:

| One fact | Two writers | Forced by | Guard |
|---|---|---|---|
| the pushable shape | `monitor.Pushable` · `agent.AgentBackend` | the cycle §4.9 breaks | `test_pushable.py` |
| `<root>/<hid>/v<N>/` | `handoff.version_dir` · `env_mgr.fs.layout.handoff_version_dir` | §4.6 gives `env_mgr` only `task_graph` | `test_handoff_layout.py` |
| reading the `task` key | `closure.task_of` · `task_graph.models._task_of` | §4.7 gives `task_graph` only `spec_loader` | `test_task_of_agreement.py` |
| **the covering relation** | `task_graph.Permissions.covers` · `closure.check.covers` | at load time there is no `Task` — `closure` holds a `Mapping` and may not import `task_graph` | `test_covers_agreement.py` |
| the stubbed shape of a shipped type | a wave-1 package's `tests/stubs.py` · the real type | wave 1 began before wave 0 finished | each package's `test_the_stub_matches_the_shipped_shape` |

**The fourth is why the rule had to be stated rather than left as precedent: it
had already drifted into opposite answers and no test said so.**

**All of them are silent by construction, and the reason is worth stating
exactly** — it is why the agreement test is not optional here the way it might be
elsewhere:

> **A duplicated *type* drifts and something eventually fails to typecheck. A
> duplicated *decision* drifts and both sides keep returning a boolean.**
>
> So: **a duplicated fact needs an agreement test; a duplicated decision needs one
> that fires.** Every one of the predicates above had a passing test asserting an
> empty list. What each needed was a test asserting a *positive* result on real
> input.

The fifth names an expiry date nobody had written down — *the moment the real type
lands, a stub that agrees with the document and disagrees with the code is worse
than no stub, because every test using it is testing a fiction.*

**A cost argument where a correctness argument exists is a bug in the argument.**
Twice in one day a *"not worth it"* conclusion was upgraded by someone else to
*"wrong independent of the cost"* — `_validate`'s per-occurrence work (returning
early would make a kind's load-time checks run or not **depending on package
ordering**, which does not go on scales against a millisecond), and the by-name
route for `task_of` (`_admit_task_specs` runs inside `check_closures`, not inside
`ClosureRegistry.add`, so `closures.add(...)` alone leaves `task_specs` empty and
`unfold` raises on a closure that is demonstrably declared).

Both were the arguer's to notice and neither was. **The tell is pricing something
that should not have been on the scales at all** — and the author of a cost
argument is the least likely person to spot it, because they have already
accepted the framing.

**A statement is not enough; the rule needs a mechanism.** *"Remember to delete
your stub"* is the kind of instruction that is never executed.
`test_the_stub_matches_the_shipped_shape` works because it **fails on the day the
shapes diverge and names both sides** — nobody has to remember anything. Write
that, not a cleanup ticket.

**The sharpest instance, and it was a live silent defect.** `monitor` design §6.1
branched on `attempt_of(tid) is None` for *"the non-leaf case: no live thread"*.
Against the real runner that condition is **never true** — a non-leaf calls
`release()`, which ends the thread and keeps the object, and `_attempts.pop`
happens only in `Runner.stop`. So the branch took the `else` and called `wake()`,
`Event.set()` on an Event no thread waits on. Measured: threads before 1, threads
after 1. **The parent enters `OUTPUT_VALIDATING` and never runs it** — a task that
never advances a phase, silently, which is the exact failure the two-channel
design exists to prevent.

Ninety tests passed over it because **`StubRunner.attempt_of` returned `None`,
encoding the design's assumption rather than the neighbour's behaviour.** The
tests confirmed the thing they were derived from. `monitor` spec §5.3 had it
right all along — *"the thread ENDS. The attempt object stays"* — so this was one
package's design document contradicting its own spec, one section apart, and only
reading the neighbour's shipped code found it.

**A drift test goes when the *last* copy does, not the first.** While any two
declarations remain it is guarding a live duplication — so whoever removes the
final copy takes the test with them. Stated because the opposite instruction was
given and was wrong: it invites deleting the guard early *or* preserving it out
of respect for the rule.

### 8.3 On a seam, read the other side's code — not its description

**The deepest finding of the stage, and it is not about anyone's care.**

> **Confidence in a message tracks how well the sender has thought about their
> own package, not how right they are about the seam** — and a seam has two
> sides, only one of which the sender can see.

**Every stale item this week was true when it was written.** `handoff`'s
accessor name was one commit old when quoted; `task_graph`'s `models.py:435`
was three; three of §2.6's spellings were wrong by the time the implementer read
them, and that section was written by the person ruling on it.

**That is not carelessness and it is not fixable by writing more carefully. A
description of a moving thing is stale on arrival.** Eight packages changed under
each other all day; any prose about another package's shape has a shelf life
measured in commits.

The proof that it is not a wave-0 authority effect is `task_graph`'s, against
themselves: their `check_graph` read `doc["task"]` confidently and wrongly for
three commits, **about their own package**. Fluency and correctness are
independent.

**Every finding this week was caught by someone reading code instead of prose**,
and every wrong recommendation was caught the same way by its owner. It pairs
with §8.1's criterion: that one says when a duplication is admissible, this one
says how to check a seam before acting on it.

**And it applies to an index, which is the case that catches the person who
wrote it.** §5.0a lists §5.19 as *"F19 reverses to grant"* under **dissolved
rather than decided** — housekeeping, nothing owed. `env_mgr` read §5.19 itself
rather than the line summarising it and found the ruling **assigned them two
things**: `package_grants` to move into their package, and a second route into
`validators/` that criterion 13 had not accounted for. Both landed in `7a93fae`.

The line was not wrong. It described the outcome accurately and **omitted the
work**, which is the failure mode of a summary written by whoever ruled — the
ruling's *content* is vivid to them and its *consequences for someone else* are
not. An index entry saying "dissolved" reads as *nothing to do here*, and that
is precisely when nobody opens the section.

**So: a summary of a ruling is not the ruling, including when the summarizer is
the one who ruled.** The index says where to look. It is not a substitute for
looking, and its author is the last person able to tell the difference.

### 8.2 Confirm another module's key with its owner before treating it as settled

**The standing rule, from the package that had to guess five schemas.**

> **The spec fixes the vocabulary, the design fixes the shape, and only the owner
> has read both closely enough.**

Four of `spec_loader`'s five schemas needed the owner's *design* to be correct,
and the spec alone was never enough. The two worst guesses would each have been
fatal and silent:

| | |
|---|---|
| `agent`'s `knowledge` items | the guessed shape and `KnowledgeRef` **shared no key name at all**, so under `additionalProperties: false` every knowledge-bearing agent spec in the system would have been rejected |
| `agent`'s `backends` | spec §3.1 says *"a list or dict"*; an array-only schema would have rejected the dict **before** the owner's normaliser ran — the feature dead on arrival, with the owner's own test passing because it never reached the schema |

**And the hazard that makes this a rule rather than advice: the failure mode of a
well-argued message from wave 0 is that people believe it.** Six packages were
building against types one package shipped. Twice, a teammate caught a wrong
recommendation by **measuring instead of adopting** — `task_graph` checked
`agent in task.schema.json.properties → False` before dropping an accessor on
that advice, and `agent` did the same with the knowledge shape.

Neither would have been caught by the person giving the advice. **Check before
you edit, including when the instruction came from the lead.**

**Two mechanisms worth reusing**, both from landing these:

- **`xfail(strict=True)` as a handshake between two packages.** The `covers` test
  was written with the divergent row marked strict-xfail so it would land green
  and go red the instant the other side widened — no synchronisation needed. It
  XPASSed on the first run, because they had already widened. §6.1's rule used as
  a protocol rather than as a check.
- **Assert identity, not equality**, where both callers read further into what
  they get back — and use a decoy document with a *second* nested object. Two
  readers agreeing on a document that happens to have exactly one nested mapping
  proves less than it looks.

### 8.4 `task_of` — settled, with the objection kept

**`spec-loader`'s README points here and at `closure` for whether the move
stands. It stands.** Measured, not summarised: `task_of` is declared in
`spec_loader/access.py:81`, `closure/model.py:35` imports and re-exports it, and
`closure/check.py:337` calls it. Neither package declares a second copy.

It went **over `closure`'s objection**, and the objection is recorded in
`closure/model.py` rather than dropped, which is the right disposal of a losing
argument that was not a bad one:

> `body` and `subgraph` are keys of the *task spec*, sharing one `$defs.body`,
> while `task` and `agent` are keys of the *closure document* — this package's
> subject. So the argument that unified `Body` did not obviously reach `task_of`,
> and `task_graph` could have deleted its copy by resolving `task_specs` by name.

**The deciding point was one the objection had not weighed**, and it is the
transferable part:

> Leaving it means the next reader finds `Body` unified and `task_of` not, **with
> no visible reason for the difference**. A rule that holds in three places and
> not the fourth is not a rule anyone can apply.

That is a cost §8.1 does not price. §8.1 asks *is this duplication forced by the
import graph?* and answers per seam; this asks what the **set** of answers looks
like to someone who did not watch them being made. An inconsistency each of whose
instances is defensible is still unusable as a rule.

**`closure` corrects the paragraph above, and the correction is right.** They
wrote `spec-loader`'s **criterion** into their README rather than this deciding
point, on the grounds that *"the next reader finds `Body` unified and `task_of`
not"* is **a reason to be consistent, not a test anyone can apply to the next
case.** It settles this instance and gives the next one nothing. *Is the
duplication forced by the import graph?* is the durable half. Recorded in the
losing order here — the vivid argument first, the usable one second — which is
how a ruling's author tends to rank them.

Their statement of it, which is sharper than `spec-loader`'s and than mine:

> **Is the duplication forced by the import graph?** Two packages needing one
> accessor, across an edge one of them may not cross, **with no lookup either can
> make instead.**

That last clause is what makes it a test. It answers the `agent_of` slope in
advance and answers **no** — one reader, and `task_graph` reads `doc["agent"]`
raw.

**And the objection failed on its carrying claim, not on the tie-break.**
`closure` argued `task_graph` could delete its copy by resolving `task_specs` by
name; `_admit_task_specs` runs inside `check_closures` and **not** inside
`ClosureRegistry.add` — `closure`'s own design decision — so a closure can be
declared with its task spec absent, and `unfold` would raise on a closure that is
demonstrably declared. `task-graph` found it; `closure` re-ran it rather than
taking it. **The consistency argument above was never load-bearing**, which is
worth knowing before anyone reaches for it as precedent.

**The dispute survived hours after it was over, and the mechanism is worth
naming.** `closure` retracted long before the marker came down: *"the retraction
went upward to you and I never sent it sideways."* A hub sees a state change; the
counterpart holding the stale row does not, and neither party is at fault. **A
retraction is only spent when it reaches whoever is acting on the old answer** —
which in a hub-and-spoke team is nobody by default.

**The precise mechanism, in `closure`'s words rather than mine:** *a row
asserting a **state** goes stale in a file **its owner does not control**.*
`spec-loader` could not see the retraction, so their row stayed true-when-written
and wrong. That is narrower and more useful than "states go stale" — a row about
your own package you will notice; a row about someone else's you structurally
cannot.

**And `closure` then committed the same fault one level out, inside the hour,
while documenting this**: they wrote *"recorded because `spec_loader` listed it
as still disputed"* — a claim about the state of a file they do not own — and it
stopped being true when `spec-loader` changed it. Fixed in `773cccd`, and logged
rather than quietly corrected. **Knowing the rule buys no exemption from it**,
which is what §2.1's row demonstrated about whoever wrote §8.3.

### 8.6 Two weak arguments converging read as corroboration

**`env_mgr` named this and said they have no mechanism for it, which is why it
is here rather than in a README.**

They and `agent-mod` both argued *stage* for F19 — `agent-mod` from consistency,
`env_mgr` from a spec clause they had misread. **Two independent arguments, one
conclusion, and the agreement felt like evidence.** Neither had checked the
other's premise; each saw a peer reach their answer by a different route and read
that as the route being unnecessary to check.

Both were wrong, and the ruling reversed on a measurement **neither made** — that
a body is a launcher, so staging `entry.sh` still launches a file the package
holds.

Independent derivation is real evidence when the premises are independent. **Here
they were merely different**, and different is not independent when nobody has
read the other's. The tell is available and cheap: *did the other party's premise
get checked by anyone, or only their conclusion agreed with?*

This is why §5's entries carry both sides rather than the outcome. **Consensus
between two implementers is not a substitute for a measurement, and it is most
convincing exactly when neither has made one.**

**`env_mgr` refines the tell, and the refinement is what makes it usable.**
Neither of them could have answered *"was their premise checked?"* — because
**neither had stated a premise plainly enough to be checked.** `agent-mod`:
*"the zone is what an agent reaches."* `env_mgr`: *"§6.3 rule 2 means copying
makes a re-run comparable."* Both read as reasons. **Both were conclusions with
the load-bearing assumption tucked inside**, which is why the agreement had
nothing to bite on.

So the operational form, theirs:

> **When you agree with a peer, say which of their premises you checked — and if
> the answer is none, say that instead of agreeing.** *"Agreed"* and *"agreed,
> and I checked nothing"* are different messages, and today they were sent as the
> same one, twice.

That costs a clause and needs no mechanism, which matters: **§8.6 is the only
finding of the stage with no test-shaped enforcement available.** Everything else
here bought a guard; this one buys a habit or nothing.

**And the habit produced its first result one round later, between the same two
kinds of party.** `validator` asked `env_mgr` whether `ValidationZone.materials`
should carry the handoff-id association, and offered to drop it if it cost more
than a couple of lines.

`env_mgr`'s account of what made it decidable is the whole point:

> I would have accepted *"multi-input validators need the association"* as
> reasonable-sounding. What they actually brought: `phase.py` **sorts** the ids,
> `layout.stage` walks **declaration order** and skips slots with no version —
> different order, possibly different length, so the two lists cannot be zipped.

**A measurement in place of a plausible requirement.** The plausible version
would have been agreed to, and the zip workaround it licensed **would have been
green on today's single-input data and stopped being green silently.** §4.11's
family, prevented before it was written rather than found after.

The change landed as `tuple[str, ...]` → `Mapping[HandoffId, str]`, announced
rather than made quietly, and it turned out **not to be a judgement at all**:
`layout.stage` had the id in hand and discarded it, leaving the caller to recover
it by parsing `<materials>/<hid>/v<N>` — `engineer_principle.md` §4.4's second
named smell verbatim. Twelve lines.

**And then it broke `validator` silently anyway, which is the part that keeps
this from being a success story.** `phase.py` did `tuple(placed.materials)` —
**over a mapping that yields the keys** — so `materials.json` would have carried
handoff ids where a body expects paths. `tuple` and `Mapping` are both iterable,
so nothing raised.

**Every test passed.** The stub in `conftest` still returned a tuple, so the
suite asserted **the old contract with full confidence.**

Everything upstream of this was done right: the consumer asked, the owner
measured, the change was announced rather than made quietly, both sides recorded
it. **None of that moves a test double.** The announcement is a message to a
person; the stub is code, and it went on describing a shape that no longer
existed. §4.11's family again — *a plausible value produced and consumed as if
real* — reached by a route with no careless step in it.

The guard is the one `handoff`'s store stub already has:
`test_the_validation_zone_stub_matches_the_real_seam` compares the stub's field
types against the real `NamedTuple`, **and its author checked it fails against
the old shape** rather than passing for no reason — §4.11's own rule applied to
the guard for §4.11.

**Second time a neighbour's type moved under this package, and the first that
would have shipped.** The rule that falls out is narrow and worth having:
**announcing a type change protects the reader; only a conformance test protects
the doubles** — and the doubles are what the suite believes.

### 8.7 A stub guard must compare types, and mine was broadcast comparing names

**I broadcast `handoff`'s stub-conformance pattern to nine packages and five
wrote guards that compare field *names*.** A field set stays correct through
exactly the change that breaks a caller — which is what happened to `validator`
— so those five guards cannot catch the case that motivated them.

`env_mgr` upgraded theirs to compare **types** and it immediately found a
**shipped defect of their own**:

> **`stubs.Task.repos` does not exist on the real `Task` and never has.**

`prepare` read it as `getattr(task, "repos", ())`, which **looks like a field
access and is not**, so against every real task it yielded `()`. The workspace's
dependency-repo path **has been unreachable in production since it was written**
and looked live, because a stub in its own tests had invented the field it
needed. Design §7.1.1 says `repos` comes from the task spec via `task.closure`
and adds no runtime field — read, and papered over anyway.

**Two distinct failures, and the rule as I broadcast it covers only one.**

| | |
|---|---|
| `validator`'s | the stub was right, then a neighbour moved. **Drift** — what the rule anticipates |
| `env_mgr`'s | the stub was **wrong on the day it was written** and never matched anything. Drift-detection cannot see it; only a comparison against the real type can |

So: **compare types, not names, and the guard must be shown able to fail.**
`env_mgr`'s `test_the_stub_check_can_tell_two_shapes_apart` drives their guard
against a deliberately drifted stub in **both** directions — *a guard never shown
capable of failing is indistinguishable from one that cannot*, and they note they
wrote the name-only version **an hour after saying that to someone else.**

**`handoff` supplies what the two halves actually cost, having checked both.**
Their drift guard already compared types — `Signature.__eq__` includes
annotations, so `dst: Path` → `dst: str` is caught with the name unchanged — and
they confirmed it by drifting a parameter's **type** rather than asserting it
would work. **One limit written down rather than left to be discovered:** both
files carry `from __future__ import annotations`, so what is compared is
annotation **source text**. `Path` and `pathlib.Path` are one type and will not
compare equal. **A false positive, which is the safe direction** — but somebody
will hit it and should not have to work out why.

**The second half needed real work, and their statement of why is the best one
available:**

> Drift **has a trigger** — a neighbour changed — so *"re-check when they move"*
> is a rule someone can follow. **Wrong-on-the-day has no trigger at all**:
> nothing ever happens, and the stub is as wrong on day 300 as on day 1. The only
> thing that catches it is running the stub's **subject** rather than the stub,
> once, on purpose.

Theirs was `check_bindings` doing `validators.get(vname).get("inputs")` against a
fake registry returning a plain dict. **Every test in that file would pass if the
real registry returned a model** — `.get()` would not exist, `check_bindings`
would raise in the assembled system, and criterion 10 would be **dead while
green**. They drove the two real registries against each other rather than
reasoning about it; the stub turned out **narrower** than the real spec rather
than different, *"the good kind of wrong — but that was luck until it was
measured."*

**And the reason nobody writes this test by default:**

> The expensive part of the check is the part the code does not use.

Driving a real `ValidatorSpecRegistry` meant constructing a full validator spec —
`brief`, `dimension`, `strength`, `tags`, `body` — **none of which
`check_bindings` reads.** They also note what made it cheap enough to bother
with: the registry named the offending fields and enum values on each of two
failed attempts. **A loud admission path is a precondition for anyone driving the
real thing.**

**And the chain is the finding.** `validator` reported a near-miss **they had
already fixed**, asking nothing — that is the only reason `env_mgr` looked, and
it cost four sentences. **A convention of reporting fixed near-misses at a seam,
not only live breaks**, is what turned one package's caught bug into another
package's found bug. Nothing in a suite produces that.

`repos` joins the open list as a fourth no-route item and **pairs with
`Context.repo_locations`** — a route for *which* repos and a map of *where* they
are are two halves of one thing.

### 8.7a In a shared worktree, an edit is published the moment it is written

**A team-wide hazard, invisible from inside any one package, and it gets worse
the more of us work at once.**

`spec-loader` wrote a schema key, ran the suite, found it broke a `validator`
test, wrote a long message about it, and **then** reverted. Ten minutes,
uncommitted. **In that window `validator` read `validator.schema.json`, saw the
key with its description and `$comment`, took it as landed, and built their step
2 on it** — model field, accessor, resolve. Then it vanished. They found via
`git log` that it had never been committed, and reverted.

Nothing was lost. But **they had opened exactly the two-gates window they had
spent the day telling everyone to avoid**, in the direction where their model
accepts a document the gate rejects.

**Two halves, and the reader-side one is the weaker:**

| | |
|---|---|
| reader | *"I verified the file's content and not that the content was stable."* Defence: `git log -1 -- <path>` before acting on a file another agent owns |
| **writer** | **decide in `scratch/`, or commit promptly — but do not leave a half-decided edit in a shared file** |

**The reader-side defence puts the whole cost on the reader, who must suspect
every file in the tree. The writer-side one costs the writer nothing.**

**And the reader-side defence does not cover the worst case, measured
2026-08-29.** `main` read `cli/environment.py`, saw `uuid.uuid4()` with no
`import uuid`, and was composing the defect report when a re-read showed the
author had added the import a minute earlier. **`git log -1 -- <path>` would not
have helped**: the edit was uncommitted *and still in progress*, so the last
commit date says nothing. `spec-loader`'s case was *written, then reverted* —
a window with two stable endpoints. This one had no endpoint yet.

> **The shared worktree makes every file a moving target for everyone except its
> writer, and there is no signal for "mid-edit."** — `demo`

So the reader-side row is weaker than it reads: it defends against a file that
*was* something else, not against one that is *not yet* anything. The only
reliable form is **read it twice** before reporting, which costs a second and is
the sole defence that does not depend on the writer's discipline. §8.7a's write
side and this are the two halves of one fact.

`spec-loader` had already noticed the *test* consequence — their uncommitted edit
was reddening everyone's suite while they deliberated, which is why they reverted
rather than sat on it — **and did not carry it one step further to the obvious
one: somebody might read the file.** `scratch/` exists for an undecided edit.

**And a refinement to §8's pattern, which makes two axes rather than one.**
`spec-loader` had: *a docstring contract whose violation is local is a cost you
can pay; one whose violation is global is a design error.* `validator` added the
second axis with better evidence:

> **One whose violation is silent is worse than either.**

`spec-loader`'s exception-type defect was **global and loud**, so it was found the
first time anyone drove the real registries. `validator`'s was **global and
silent**, and survived a full criterion mapping, a README audit and six weeks of
tests.

**So the axes are not independent: a loud global violation is nearly
self-correcting; a silent one is only ever caught by someone running the subject
rather than the stub.** Which is the argument for §8.7's check being **a standing
habit rather than a sweep.**

**The mechanism §8.7a's announcement half was missing, from `env_mgr`, about
their own change.** I put it to them that they announced the `materials` type
change properly and it **still** broke `validator` silently. Their answer is not
*announce harder*:

> The announcement went to the right person with the right content and **their
> stub did not read it.** What caught it was a test — theirs, then mine. **An
> announcement is evidence that a human knew; it is not a mechanism.** Its value
> was that they knew where to look *after* it broke, which is real and is not
> prevention.

> **When a type changes incompatibly on a shared surface, rename the field.**

`tuple[str, ...]` → `Mapping[HandoffId, str]` **kept the name.** Both are
iterable, so `tuple(placed.materials)` stayed valid Python and silently yielded
**keys where paths were expected** — invisible at the only site that mattered.
Shipping `materials_by_id` would have made that line an `AttributeError` **at the
point of use, on the first run, in the consumer's own suite.**

**The tell is whether the old code would still run.** If it would, and would mean
something different, **the name must move with the type.** Not a general rule — a
rename churns callers for *compatible* changes — but for an incompatible one it
converts a silent misinterpretation into a loud failure **at the same one-line
cost as the announcement.**

**And the honest asymmetry, which is theirs and is the part that settles it:**

> My case has **no reader-side defence at all.** `validator` could have run `git
> log` on `protocols.py` and seen a committed, announced, intentional change, and
> their stub would still have been wrong. **The reader cannot defend against a
> change that was correctly made and correctly told to them.**

So for an incompatible type change **the cost is entirely the writer's, and the
writer has exactly one instrument: make the old code stop compiling.**

It generalises past types — it is the same instrument as renaming §5.15 from
*"what confines a validation body"* to *"what applies a policy to a body"*, where
the old title **would still have read as true and meant something narrower.**

**And the mirror of this section, from `handoff`, about a document I own.**
`implementation-stage.md` §8 said *"probe scripts are kept, not deleted — they
are the evidence"*, while `scratch/.gitignore` says *"nothing here is
committed."* Both hold. Together they mean **kept in this worktree, not kept in
the repository** — 0 design probes and 0 implementation probes are tracked — so a
citation like `handoff/docs/design.md:285` **dangles on a fresh clone.**

> §8.7a is *an edit is published the moment it is written*; a gitignored
> directory is ***an artefact is never published however long it sits*** — and
> **"kept" reads like it was.**

**`monitor` measured how far this goes, house-wide:**

```
spec_loader 4  handoff 1  validator 11  agent 5  closure 1
env_mgr 10     task_graph 1  monitor 11  demo 4        = 48 citations
distinct paths: 36        of those, in git: 0
```

**Every package cites `scratch/` from its shipped source, and not one of the 36
paths exists in a fresh clone.** *"Measured, not assumed"* followed by a path that
resolves to nothing. **No test, no lint, no error — and it only breaks for the
reader who was not here, which is the reader the citation is for.**

**Ruled: leave all 48, and make it a known limit rather than a broken link.**
Committing the probes contradicts the rule that keeps the tree clean;
re-pointing at `docs/design-stage.md` is a doc edit in nine packages **and that
file does not carry every probe.** `monitor` was right that a package diverging
alone would make the tree inconsistent in a **new** way, which is worse than
being consistently wrong in a **documented** one — so the statement goes in §8
above, once, where it governs, rather than into 48 sites.

**What makes it survivable is `demo`'s property, and it is the condition of the
ruling:** every measurement is **quoted inline** — the three `sh` PATH rows, the
three grant-resolution rows, the store's `detail=''`. **The path is provenance;
the numbers in the document are the artefact.** A citation whose result is *not*
inline is a broken link and must be fixed, not accepted.

**`monitor` corrected that wording and they are right — mine was too narrow.**
*"The number is the artefact"* **holds only where there is a number.** Where the
measurement is **categorical**, the artefact is the **stated finding**, and the
test is the same either way:

> **Can a reader check the claim without opening `scratch/`?**

**Their own tooling produced the false positive that shows why it matters.** They
grepped their commit messages for a numeric result beside each path, and it
flagged `4670f56` as bare. **Reading it, it is not** — the result is stated in
prose, because *that* measurement has no number: **a branch that never fires and
a `wake()` that is a no-op are categorical findings**, and the regex knew only
`ms` and `n failed`. **Had they trusted the grep, the "fix" would have been to
invent a number for a result that does not have one.**

That is *candidates, not verdicts* arriving **in their own tooling, within the
hour of their saying it about `closure`'s tell.**

**All eleven of theirs pass, and they named the weakest rather than counting it
equal:** spec §5.2's `findings-monitor-loop.md`, cited with **no number at all**,
passes on the stronger reading because **the artefact there is a structure and
the structure is the next sixty lines** — all five rules, in full, in the
document. **Nothing has to be fetched to check the claim**, which is the
condition's substance.

**And `demo`'s forward rule, adopted:**

> If a measurement is ever load-bearing enough that **a reader must re-run it**,
> it belongs in `tests/` where a clone can execute it — **not in a citation.**

**They proposed no change and were right not to.** The durable part of a
measurement is its **result**, which lives in docstrings and commit messages
where a reader meets it; the script is reproduction. **The fix, if it ever
matters, is not committing `scratch/` — it is that a document citing a probe
should carry the number it measured, not only the path.** §8's wording now says
so.

**And they verified my "you are clean" rather than taking it** — `git ls-files
--others --exclude-standard` over their paths — which is the rank-1 check applied
to a repository. Their generalisation of their own ranking is better than the
version I recorded: **it is not a property of tests, but of any claim you can
check for having ever been exercised.** They had also hit the pathspec hole once
today, on `tests/handoff/test_load.py`, and caught it **only because they run
`git status` out of habit — nothing told them.**

### 8.8 A test whose subject is reachable and whose premise is not

**`validator` applied `env_mgr`'s finding to their own package and found the
same shape, also shipped.** Three `getattr`s for an `environment` field that
neither `ValidatorSpec` nor `task_graph.Task` has — and `ValidatorSpec` sets
`extra="forbid"`, so no document can add one. **All three returned `None` on
every call ever made**, so every validation takes the GLOBAL row and **three of
spec §8.2's four rows are unreachable.** Criterion 9 is half implemented: the
fresh environment is real and tested, the chain is not.

**The generalisable part is why their own test did not catch it.**
`test_configuration_chain_order` exercises all four rows and passes. It is not a
bad test and it is not wrong — `choose_configuration` is pure, its logic is
correct, and the test calls it directly with the four arguments.

> **Its coverage was real and its implication was not.** Nothing asserted that
> the caller could supply those arguments, and the caller could supply exactly
> one.

**A unit test of a function whose real call site cannot reach three of its
branches stays green forever.** This is distinct from everything already in
§4.11: not a value that is wrong-but-not-type-wrong, not a rule half
implemented, not a guard asserting on a stub. **The subject is reachable; the
premise is not.**

The tell, theirs: ***does anything assert that the caller can produce these
inputs?***

**`demo` supplies the other half of this defect, from an integration test.**
`validator`'s unit test could not see it **because it called the function
directly**. The demo could not see it either, for the opposite reason:

> The demo registers `validation_env`, which **is** the GLOBAL row — it exercises
> the one row that works and **cannot distinguish the other three being dead from
> their being unused.**

**A green end-to-end run over the one live branch looks exactly like coverage of
the chain.** Both instruments were honest and neither could report the gap. That
is why *"the demo would have caught it"* is not a substitute for §8.7, and `demo`
measured rather than assumed: **of the three defects they were asked about,
`run` would have shown none** — one path never entered because no task spec
declares `repos`, one masked as above, and one (`check_bindings`) with **no
production caller at all**, so `--dry-run` never reaches it.

**Their statement of the structural limit is the one to keep:**

> A `getattr` default for a field the type does not have yields a value that is
> **legal**, so nothing downstream misbehaves — the symptom is *an absence that
> looks like a configuration nobody set*. The demo's output can show that a value
> **was** applied. **It has no way to show that a value should have been and was
> silently dropped.**

Not a gap to close with assertions. **The demo and §8.7's check are
complementary rather than overlapping** — worth knowing when someone proposes
that one covers the other.

**What they did instead of the tempting fix** is the other half. They did **not**
invent an `environment` field — that is a key in the shared
`validator.schema.json` and a decision about where a task's resolved
configuration lives, and neither is theirs. They replaced the three silent
`getattr`s with `CONFIGURATION_SOURCES`, which enumerates all four rows and
states for each whether it has a source **and why not**, plus
`test_only_the_global_row_is_reachable_today` asserting it **through the phase
runner** rather than the function. **The dead branch becomes a declared gap with
a test pinning its size.**

**And the criterion table keeps the caveat rather than staying uniform** —
`env_mgr`'s precedent, who reported 21½ of 22 rather than 22 over a
half-covered property. A mapping is a claim about the system, not about the
tests; uniformity that hides a half-implemented criterion is the thing this
stage has spent itself finding.

**One seam alone produced four, and none was found by the package that owned
it.** `spec-loader`↔`validator`, in a day:

| | found by |
|---|---|
| `ValidatorInvalid` escaping `load_package`, aborting the whole multi-package load | `spec-loader` driving `validator`'s real registry |
| the `Malformed` kinship question, where **the enumeration was the defect** | `handoff` measuring their own raise sites |
| a malformed config sidecar aborting the load — **`spec-loader`'s file, thirty lines from the fix they had just written** | `validator` asking why one file had two catch breadths |
| `origin_of` missing from the Protocol | `closure` deleting a guard **correct against an incorrect contract** |

**Every one sat in code with a passing suite.** That is the number that makes
§8.7 a standing habit rather than an agreement.

**And `validator`'s synthesis explains the mechanism rather than asserting the
practice:**

> **The writer cannot un-know the intent, and the reader cannot know it** — and
> those produce **different error sets**, which is why both passes are worth
> running and **neither replaces the other.**

**And an eighth from the same seam, hours later, found by the author checking
their own claim before relaying it.** `validator` told `closure` *"`agent_of` is
exported"* — it was exported **from `validator.spec`, not from `validator`**. So
**the accessor written specifically to spare another package from reading their
internals was reachable only by reading their internals.**

> An assertion about a seam **is not verified by the code on my side compiling.**

It existed, was in an `__all__`, carried a docstring saying why it was exported,
and passed every test they had. **Every one of those facts was about the module.
None was about the name the other package types.** `closure` had not tried the
import yet, so **nothing had failed** — this is the second finding today caught
*before* the consumer built on it, and both times by someone verifying their own
statement rather than by a test.

**The guard asserts the root name, not the submodule name** — moving the function
would leave the submodule name working — **and they verified it by removing the
export and watching it go red**, rather than reasoning that it would.

**The other half of the hazard is worse and has no guard:** nothing inside
`validator` imports `agent_of`. **An accessor with no in-package caller reads as
dead to whoever tidies up, and the breakage lands in a package whose tests that
person is not running.** It is `demo`'s *seams that must be called in production*
question wearing the opposite face — there, no production caller meant an
untested path; here it means an **inviting deletion.**

**And the criterion did not fire in advance, for a reason worth more than the
criterion.** `closure` demonstrated the block rather than arguing it — added
`from validator import agent_of`, ran the interface test, quoted the failure,
reverted. **The accessor is correct, needed, and unreachable from its only
cross-package caller.**

**Ruled: it goes to `spec_loader`, under a name that is not `agent_of`** —
verified, `validator/spec.py:208` returns `str | None` over a validator spec and
`closure/model.py:107` returns `str` over a `ClosureDoc`, **both take a mapping
and neither raises on the wrong document.** Two of that name in the leaf puts the
collision **where nobody can alias around it.** `subgraph_of`'s precedent: **the
leaf owns that the key exists, the owner owns what it means.**

**I ruled it on the forced-duplication criterion and that was wrong.**
`spec-loader` corrected it while landing `eff9a18`: **that criterion needs two
packages wanting one accessor**, and `validator` had measured that **nothing
inside their package calls theirs** — one reader, not two. **The right argument
is `body_of`'s and it is simpler:**

> **`spec_loader` declares the key in `validator.schema.json`, so hosting its
> reader adds no interpretation of a package's content.**

The outcome was the same and the reasoning was not, which matters for the reason
they give: ***a criterion cited loosely is how the next four keys get argued
wrongly.***

**`validator`'s own reading of why they cited it is sharper than the
correction.** They supplied the fact that falsifies the criterion **in the same
message that cited it** — it needs two packages wanting one accessor, and they
had just measured that theirs had zero readers:

> Their version discriminates where mine does not. Mine gave the same answer
> either way, **which means it was not deciding — a criterion that cannot come
> out the other way is a label.**

**And the second correction against them the same hour is the load-bearing kind.**
They described `agent-mod`'s absent-case default as *"the validator runs as the
task's agent."* It does not — `self.agent_spec` is a composition-root choice and
nothing about the task enters it.

> **My wrong version is the load-bearing kind**: if it were true, the obvious
> tidy-up is making the environment follow the agent, **which is criterion 10's
> separation deleted by someone who thinks they are fixing an inconsistency.**

**They wrote the sentence that motivates the forbidden change, inside a message
asking `agent-mod` to write it down.** `agent-mod` wrote the correct version
instead.

**A third the same afternoon, theirs against `closure`.** `closure` reported
*"your `body` and `tags` are both required."* `tags` is; **`body` is not** — it
defaults to `{}`. **The wrong half is the harmful one:** a fixture author adding a
stub body to satisfy a requirement that does not exist **has made an agent-bodied
validator by accident**, which is the `entry: ""` failure by another route.

**`validator`'s synthesis of all three is the day's most transferable finding
about how this team communicates, as distinct from how it codes:**

> The shape is identical each time: **a plausible generalisation stated in
> passing, where the wrong half points at a change somebody would make.** Not one
> of the three was a claim anyone was making carefully; **all three were asides
> inside messages whose main point was correct.**
>
> **The aside is the part nobody verifies, including the person writing it.**

**A fourth, and it names the mechanism the other three share.** `validator` told
me and `env_mgr` that §8.2's `consumer` and `producer` rows were **one question**
— same task, both phases inside `TaskRunner`. **I endorsed it.** Measured
first-hand against `agent/runner.py`, it is wrong:

```
env.prepare       exactly one call site, runner.py:668, from _deploy
_deploy           reached from _main, 604/587
_one_phase        reaches _main only in RUNNING, 466-471
_validation(INPUT_PHASE)  runs at INPUT_VALIDATING — strictly before that
```

**Same task, yes. Same moment, no.** The `consumer` row is **unreachable in
principle** — no `Prepared` exists yet, and §8.2's own phrase is *"the task about
to run"*, which **is** exactly before `prepare`. The `producer` row is reachable:
by `OUTPUT_VALIDATING` the configuration existed and was **discarded**, a local of
`_deploy`. So criterion 9 goes from half to **three-quarters and no further**, for
**two different reasons** rather than one.

> **A simplification is a claim.**

**And it carries less scrutiny than an assertion, precisely because it reduces
rather than adds.** I accepted it because it was tidy — a smaller ask arriving as
a smaller ask, not as a new fact. That is the mechanism the other three share and
none of them stated: **an aside is unverified because it is not doing the work;
a simplification is unverified because it appears to be doing less work.**

`env_mgr` produced the day's fourth in the same exchange — *"`Runner.attempt_of`
is declared and unbuilt"*. **It is built** (`runner.py:163`, declared at
`protocols.py:313`). Their conclusion survived it; the aside did not.

**I produced the fourth within the hour, in a ruling.** Ruling the `minLength: 1`
clause into `tests/interfaces/`, I wrote *"nothing asserts this one"* — **it was
asserted**, at `tests/spec_loader/test_access.py:209`. My grep was
`grep -rn minLength tests/ | head -3`, and **I concluded from a truncated
output.** An aside inside a ruling, which is the worst carrier for one, since a
ruling is the thing nobody re-derives.

**The ruling survived for a different reason than the one I gave**, and
`spec-loader` supplied it: the existing assertion sat inside a test **about
naming**, where the `minLength` line was incidental — so a reader would not learn
that relaxing it silently breaks two other packages, and **it would vanish
unnoticed if that test were restructured, with nothing downstream failing.** They
**moved** it rather than adding a second: one writer, in the place that names who
depends on it.

**The mechanism, worth keeping because it is what a relaxed clause costs:**

| on `agent: ""` | |
|---|---|
| `spec_loader.validator_agent_of` | returns `None` — **the same answer as an absent key**, so §8.2's global row is taken |
| `agent/validator_executor.py` | falls back with `or`; their own test says *"the `or` cannot reach this case: `minLength: 1` means a present name is never falsy"* |

**Relax the clause and an author writing `agent: ""` gets a silent default
environment instead of an error, with both suites green** — each package
behaving exactly as designed, on an input the schema was supposed to have made
impossible.

**And `spec-loader` framed the new file for its siblings rather than for this
clause:**

> **A schema clause authored by one package and depended on by others cannot be
> enforced by the dependants, only assumed, so it has to fail on the producer's
> side.**

That is `Pushable`'s bargain with the wall in a different place, **and there will
be more of them.**

**Two more from the same exchange.**

**A defect reported once existed twice, and the second was found by the person
*receiving* the report asking whether the first was alone.** `validator` reported
`validator_agent_of`'s annotation; `body_of` had the identical defect — its own
first line reads *"a task's **or a validator's** declared body"* while annotated
`TaskSpec` — and nobody had named it. **`validator`'s note on themselves: *I did
not ask that question about my own finding.***

**And the remedy attached to a measured finding was itself unmeasured.**
`validator` proposed `ValidatorSpec: TypeAlias` in the leaf, which would collide
with `validator.ValidatorSpec`, a pydantic model — **the leaf exporting a second
importable name for a different thing, which is the collision
`validator_agent_of`'s name exists to prevent, one level up.** They had argued
that collision for a *function* name an hour earlier.

> **I was careful about the finding and careless about the remedy attached to
> it.** The finding was measured; the fix was a plausible suggestion in the same
> message, **and it is the half that was wrong.**

**And a fifth of a different class, which `validator` separated rather than
folded in.** They told `closure` *"`body` is not required"* — **true of
`ValidatorSpec(...)`, false of `admit()` and `ValidatorSpecRegistry.add()`**,
which is what a fixture author calls. **The example in their own
fixture-guidance section did not load.**

> The three before it were unverified asides. **This one was measured** — I ran
> the constructor, the result was correct — **and the sentence was still wrong,
> because I did not say which of four gates I had measured.**

`closure`'s rule: **when a claim is about what is required, say required by
what.** And the general form, which is the sharpest thing produced today:

> **A measurement carries its setup and a sentence does not.** *"I checked it"*
> establishes that **some** claim is true, not that **the claim I wrote down** is
> the one I checked.

**That applies to this document and every ruling in it** — each measurement above
reached its reader as prose, stripped of what was run. **No villain in it:**
`closure` reported the gate they **hit**, `validator` the gate they **ran**, and
each meant the one they had most recently touched. Four gates in that package
disagree, and **the disagreement is now a table rather than something two people
rediscover.**

**And a sixth, where the error made the arguer's own case stronger.**
`validator` told me and `spec-loader` that on `agent: ""` their side quietly
takes the global row. `spec-loader` measured it:

```
agent ""  ->  RAISES ValidatorInvalid: names agent spec '', which does not resolve
```

**One silent dependant, not two, and `validator`'s is the loud one.** The
conclusion survives — `agent`'s `spec.agent or self.agent_spec` is genuinely
silent, and one silent dependant in a third package whose tests cannot reach the
clause is enough — **but the count and the mechanism were wrong in the direction
that strengthened the argument.**

**Where the belief came from is the record-worthy part.** For a few hours that
package had **two readers of one key that disagreed on exactly this input**: the
withdrawn `agent_of` normalised `""` to `None` deliberately, as the `entry: ""`
lesson, and `_bound_environment` never called it. **They described the
accessor's semantics as the package's, one hour after deleting the accessor** —
reporting the intent they had built for as the behaviour of a function that never
implemented it. `spec-loader` then relayed that sentence into a test docstring as
fact.

**`spec-loader`'s framing inverts the obvious intuition and is the one to keep:**

> **A description of a package's own internals is the weakest source, not the
> strongest** — the author cannot un-know the intent.

**Same premise, two errors, opposite directions**: they took the account as
authoritative *because* it was its owner's, and it was given without reading the
code *because* it was its owner's. **It also names why §8.3's *read the other
side's code, not its description* was insufficient — it only catches their
half.**

Three instances today of someone **wrong about their own package**: `validator`'s
`environment` `getattr`, `spec-loader`'s `$comment` owner, this.
**Ownership is what removes the reason to look.**

**Every defect recorded above was found in code. These were found in
prose**, in messages between packages — and each would have become code, because
each named the change a competent reader would then make. `closure`'s fixture
claim was inside a correct bug report; `validator`'s agent framing was inside a
correct request; their criterion citation was inside a correct conclusion.

Both are one shape: **reaching for a conclusion and then finding a name for it.**
One was a precedent that *felt* right; the other an inference from a fallback
expression **read but never driven.** Neither needed a hard measurement — *"one
grep and one `git grep ValidatorExecutor(`, both of which the other party ran."* I cited it loosely in a ruling, which is the worst place to.

**Their naming argument is also sharper than mine.** The other three accessors
are `<key>_of` **because their key is unambiguous across documents.** This one is
not — `closure.agent_of` reads a closure document and returns `str`, this reads a
validator spec and returns `str | None`. **The broken naming pattern is the
signal**, not an inconvenience: inside one package the ambiguity is aliasable,
**exported from the leaf it is not.** And `str | None` is **the two specs
disagreeing rather than an inconsistency** — `agent` is required on a closure
document and optional on a validator spec, where absence is ordinary and takes
§8.2's global row.

**Three accessors have now needed this move for the same reason — `body_of`,
`task_of`, this — and each was found by the consuming package hitting the import
wall, one at a time.** `spec-loader`'s criterion was written down after the first
and did not fire for either of the others:

> **The person who writes the accessor is the owner, and the owner never hits the
> wall.**

**A written criterion checkable only by someone with no reason to check it is not
a working criterion.** `closure`'s replacement is one grep by the party with the
motive:

> **When you build an accessor because another package asked for it, check that
> package's import row before you build it.**

**And `closure` generalised it past accessors, which is the structural finding:**

> Every rule we have written this week is **checkable by the consumer and
> authored by the producer** — the import row, the `getattr` guard, the default
> on a parameter the root supplies, the stub whose subject was never driven.
> **The producer never hits the wall, so a rule aimed at producers has to be
> enforced by something that runs on their side.**

**`validator` reached the same conclusion from the opposite direction**, having
declared `agent_of` in `protocols.py` **on `closure`'s suggestion rather than
their own**, which put it under `tests/interfaces/`'s signature comparison:

> **A guard in a test the package does not own is stronger than any guard it
> writes for itself**, because **the package cannot weaken it while making its
> own suite green** — which is the exact failure mode of a guard that gets in the
> way of a refactor.

Their own root-export guard lives in `tests/validator/` and **they could delete it
in the same commit as the thing it protects.** The conformance one they would have
to **disarm deliberately, in a file that belongs to the seam.**

**Two packages independently concluding that `tests/interfaces/` is the
enforcement point rather than a convenience is a fact about how this repository
is put together**, and it is probably that directory's real job description.

### 4.17 A green run can be the corruption

**`task_graph`, building §4.14.** They first derived the output version from
`HandoffMgr.latest` — **and it makes `demo run` pass.**

> On a retry `latest` is the **previous attempt's** version, so the grant hands
> the retry **the previous version's directory**, and it overwrites bytes
> criterion 16 promises are byte-identical forever.
>
> **Green demo, silent corruption.**

**The allocator is what avoids it**, and the difference between the two is
invisible in the artefact that was supposed to prove the change worked. This is
`demo`'s own limit — *the output can show a value was applied; it has no way to
show one should have been and was silently dropped* — met from the inside, **by
the person the green would have reassured.**

**The allocator is `handoff.HandoffStore.allocate`, not `HandoffMgr`**, and
`task_graph`'s README asserted the opposite: *"allocating a version is
`Handoff.open_next`'s and criterion 14 forbids the scheduler writing handoff
state."* **It named the wrong allocator** — §5.12's two version numbers, one of
which is the store's. **Criterion 14 holds literally**: no transition, no
`persist` from a scheduler frame.

### 8.14 Stand-ins versus instruments — the distinction the doubles rule was missing

**`monitor`'s concept, `agent`'s framing, and it survives both of their
overreaches.**

> **Production calls a stand-in**, so it must match the real type in surface
> **and** contract. **Only the test reads an instrument**, so the real type has no
> opinion about it.
>
> **Conform on what production calls; leave alone what the test reads.**

**Both parties overreached in opposite directions within one hour.** `agent`'s
first draft flagged **a spy's bookkeeping** (`queries`, `responses`, `connected`)
as surplus — it would have deleted legitimate code. `monitor` was about to turn a
spy's `**kw` into a `TypeError` and **break four tests.**

**All five recorded instances of the underlying bug are stand-ins**, which is why
the rule survives both mistakes rather than being weakened by them.

### 8.14a A fix that looks applied

**`monitor` asked `agent` for `phase=self.task.status.value`. It would never have
matched.**

```
PHASE_ORDER[-1]                      'OUTPUT_VALIDATING'
TaskStatus.OUTPUT_VALIDATING.value   'output_validating'   → False
TaskStatus.OUTPUT_VALIDATING.name    'OUTPUT_VALIDATING'   → True
```

`PHASE_ORDER` is **name-keyed**, and `monitor`'s own `next_phase` proves it —
`base.py:80` is `PHASE_ORDER.index(status.name)`. **With `.value` their
`finished == PHASE_ORDER[-1]` is dead: the terminal phase stays an advance, the
`HANDLING_FAILED` still lands on every success, and the fix looks applied.**

**`agent` emitted `.name` and pinned the agreement from their side**, so a re-key
on either end goes red rather than quiet.

**And the reason `monitor` reached for `.value` is worth more than the bug.** It
came from `agent`'s own round-trip rule, written that morning — **but that rule is
about enum members versus strings, and `.name` and `.value` are both plain
strings that survive `model_dump(mode="json")`.**

> **The rule does not choose between them; the consumer's key does.**

**It is now cited across three packages and it does not say what people are
reading into it.** A rule that has spread further than its own scope is a stale
document that nobody has to re-read, because it lives in citations rather than in
a file.

### 8.13 Four instances of one bug, and the rule was always one widening behind

**The day's tally, `agent`'s:**

| | the double | found by |
|---|---|---|
| 1 | **invented** members the real client lacked (`session_id`, `get_session_messages`) | **running the real thing** |
| 2 | **missing** a field the real `Prepared` gained (`agent_cli`) | the rule |
| 3 | **absent** method the runner called (`StubStore.seal`) | the rule |
| 4 | **wrong contract** — `-> None` and raising, against `-> str \| None` and returning | the rule |

> **Only the first was found by running the real thing. The other three were
> found by the rule — each time one widening later than the bug.**

**The rule chased the bug and never got in front of it.** Presence-of-member
would not have caught (4); return-annotation now would, and it is *"the cheapest
part of a contract to compare and exactly what moved."* **Each widening was
correct and each was retrospective.**

**And `agent`'s synthesis is the day's, so it goes here rather than in a
package:**

> `handoff` wrote *"a seam has two sides and I shipped having checked only
> one."* **Two and a half minutes' notice is not the failure — notification is a
> person remembering.** The failure is that **the other side's double still spoke
> the old contract and nothing compared them.**

**That is §8.10's standing habit with the missing half supplied.** The
enumeration is owed — **and a test is the only version of it that cannot be
forgotten.**

### 8.12a The ruling was made moot by a better answer, and the better answer was structural

**`seal_output` is withdrawn.** `handoff`'s `fd31a6c` makes `seal` **return** the
reason instead of raising it:

```
None    published
str     why the artefact was not publishable
raises  NotSealable only — no such version, already published
```

**That is the ruling's entire purpose** — the distinction crossing a wall `agent`
may not import — **with no new function, no root registration and no import
edge.** Their constraint, stated exactly: *"a return value crosses that boundary
and an exception type does not."*

**And it makes list item 3 structural rather than guarded.** With the refusal
returned, the runner has **no `except` at all**, so `NotSealable` **cannot** be
swallowed by construction. `agent` still pinned it by behaviour — *"there is no
`except` there" is a fact a later edit undoes silently* — and verified
non-vacuity by reinstating the `except` and watching it go red.

**A correction I recorded wrongly, withdrawn by its author against himself.**
I wrote here that `handoff` *"inferred the red rather than running it."* **They
ran it** — `1 failed, 173 passed` at `test_runner.py:488`, **inside the
two-minute window** between `fd31a6c` (06:58) and `0e6bf7e` (07:00).

**`agent` saw green afterwards, inferred backwards about how the other party had
reached red, and reported the inference to me as fact** — on the day whose whole
subject is that distinction. They withdrew it to me and to `handoff`, and I had
already committed it.

> **We were both reporting truthfully about different minutes.**

**A tree moving this fast needs a timestamp on a claim about it, not just a
result.** *"The tree is green"* with no clock on it is not a weaker claim than
one with a clock; **it is a different claim, and a false one.**

**And green was still the worse answer** at the moment it was true: the double
implemented the old raise-contract, so **the suite measured the old world while
the real store returned a reason.**

### 8.12 Member-level conformance is not enough, and the proof was inside the fix

**`monitor` found the third instance of *the double accepts a shape the real
thing never produces* — and it was sitting inside the guard built for the second
one.**

`handoff.seal` **returns** the refusal reason (`str | None`) and raises only for
wiring bugs. `agent`'s `StubStore.seal` was `-> None` and signalled refusal by
**raising**.

> **Member-existence conformance passes, because the member is present on both
> sides. Only the signature disagrees.**

**That is the sharpest available argument that the rule needs to be
signature-level**, and it is sharp because of where it was found: **the
doubles-conformance test written that morning, against the previous instance of
this exact class, did not catch this one.** A guard is not proven by the case it
was written for.

**Already closed by the time it was reported** — `conftest.py:282` is now
`-> str | None`, and its docstring records that the method *"was missing when the
runner started calling it"*, which is §4.20's first layer.

**And `monitor` ran the signature-level version on their own package** rather
than only recommending it: six differences, all correct by intent, and **the one
member reached by dynamic dispatch matches exactly.** *A rule is not adopted
until it is pointed somewhere its author did not have in mind* — here, at the
author's own doubles, immediately.

**Ruled, on `monitor`'s reasoning: a seal refusal is recorded as the existing
`seal_refused` attribute.** No new `EventKind`, no verdict, no task failure.
**Kinds name phases; causes go in the payload** — `OUTPUT_ABSENT` already names
the phase and the refusal is why. **The implication runs one way**: a refusal
always yields absence, absence does not imply a refusal — so the attribute's
**presence** carries *attempted versus never attempted* and its **content**
carries *wrote nothing versus wrote badly*. **Criterion 5 intact without a kind
per cause.**

**One deferral tracked rather than lost:** `seal_refused` has no reader outside
`agent`'s tests, so declaring a predicate in `monitor` today would be §4.12. But
**criterion 5 makes the distinction normative, and a normative fact read by
string-matching an optional attribute** is what `demo` objected to with
`attributes["target"] == "user"`. `monitor` declares the key and predicate **the
moment a consumer appears**. Same pattern maturing at **three sites**.

### 8.11d The index is shared too, and a control you do not read is decoration

**Two failures in one commit (`db6d485`), by `main`, hours after writing §8.11c.**

**The mechanism is a fourth form, not a repeat.** `git add <one file of mine>` was
run, and it did not scope the commit: **in a shared worktree the index is shared**,
and `git commit` with no pathspec commits *the index* — including whatever another
session staged into it. `git add` adds; **it does not restrict.** A teammate's
`StubTaskMgr` landed inside a documentation commit, under a message about something
else.

| | |
|---|---|
| §8.7a / `task-graph` | *their files are my pathspec* — **what** lands |
| `env-mgr` | *a pathspec bounds files, not authorship* — **whose changes** land |
| §8.11c | *`pytest` then `git add` is not atomic* — **when** it lands |
| **this** | **`git add` then `git commit` is not scoped** — **whose index** it is |

**And the second failure is the worse one.** The verification §8.11c demands *was
run*, and printed exactly the right answer:

```
agent_sys/docs/interfaces.md      45 +
agent_sys/tests/agent/conftest.py 26 +      ← not mine
```

**`git diff --cached --stat && git commit` was chained into one command, so the
commit executed before the output was read.** The control engaged, produced the
correct signal at the correct moment, and the sequencing made it decoration.

**That is `task-graph-2`'s `-p no:randomly` one step worse.** Theirs was a flag
against an uninstalled plugin — a control that never engaged. This one engaged and
was not consulted. **A verification step that cannot block is not a verification
step**, and chaining it to the action it is meant to gate is how a real check
becomes a ritual.

**Disposition:** stage, read, then commit as a **separate** invocation, with an
explicit pathspec on the commit itself. Not because the read is unreliable — because
a read whose result cannot change what happens next is not being used.

**`git commit -s -F - -- <paths>` is the mechanical form**, and it is stronger than
any `git add` discipline: it commits those paths from the **working tree** and
ignores the index, so it cannot pick up another session's staging *or* your own
stale staging. Hunk-splitting via `git apply --cached` survives §8.11d only by
accident — it would still commit a third party's staged file.

#### 8.11d-i A grep for what you expect cannot detect what you do not

**The discipline of §8.11c had a second defect, and its own author found it.** The
gate they had been chaining was

```bash
git diff --cached | grep -cE '<my load-bearing lines>' && git commit
```

**It asks *"are my lines present?"* and never *"is anything else present?"*** It
would have passed a commit carrying somebody else's staged file, cleanly, every
time. `main` ran the same shape — a `grep -c` for a section anchor — on four
consecutive commits, and on the one that swept a teammate's hunk the whole-index
inventory was printed **beside** the grep and was the line not read.

**A confirmatory check cannot answer an exclusionary question.** The two look
identical in a terminal and differ in what they are able to fail on.

The corrected form, all three parts load-bearing:

```bash
git add <paths>
git diff --cached --stat        # separate command; an inventory, not a grep for my own lines
git commit -s -F - -- <paths>   # pathspec here, so the index cannot decide what lands
```

`--stat` **because the question is *what is in the index*, and only an inventory
answers it.**

**And when four commits come back clean, that is a measurement of the window, not of
the discipline.** `env_mgr` checked `ad730a2`, `8eba17a`, `93f047d`, `9375517`: all
bare commits, none swept — *"because nobody else had staged in those windows, not
because anything stopped it."* Same sentence as §8.11c's, turned on its author.

#### 8.11d-ii A session transcript records neither the system prompt nor the environment

**Three people concluded from that file today and all three were wrong in the same
direction.** Settled with positive controls rather than by looking harder:

```
system_prompt = "ZZQQ-…-MARKER-7431. Whatever you are asked, reply with exactly BANANA."
query         = "What is 2+2?"
answer        = 'BANANA'                         ← the prompt demonstrably arrived
MARKER anywhere in the transcript = False        ← and left no trace, in any of six line types
```

Reproduced across three runs. The same construction for the environment: a marker
variable set, **absent from the transcript.**

**So a grep of `config/projects/<zone>/*.jsonl` cannot answer "did the brief
arrive?" or "was the agent given an output path?"** — it is the wrong file for both,
and it returns a real number about something else.

**The only construction that settles an absence claim is a positive control**: plant
something findable, show it works, and see whether the search finds *that*. A second
probe did it from the other side — a `system_prompt` demanding the reply begin
`PLATYPUS-7731`, against a goal that never mentions a token: **obeyed, and invisible
to a grep of the same run.**

**And two zeros from one instrument are one measurement, not two.** The output-path
zero was initially read as a second, independent finding; it is the same artefact.
It was only tested because someone was told *do not let resolving the first question
close the second*.

**What did settle it was a different file entirely** — the brief itself, 37 static
lines: `AGENT_SYS_DEMO_OUTSIDE` named twice, no output path anywhere. **The
transcript was never going to answer a question the source could.**

#### 8.11d-iii A running process is the tree at its start, not the tree you are reading

**Nine minutes and a live model conversation, spent watching pre-fix behaviour.**

```
run started    09:37:03
dd5d6fc        09:41:13   the fix the run was launched to exercise
```

The observer had HEAD in front of them, the fix in it, and a process that had
imported the module **four minutes before the line was written.** Every refusal they
recorded was correct, expected, and about a version of the system that no longer
existed on disk.

**The family's sixth form, and the axis is new:**

| | |
|---|---|
| §8.7a | *their files are my pathspec* — **what** |
| `env-mgr` | *a pathspec bounds files, not authorship* — **whose changes** |
| §8.11b | *a grep of the committed tree is not a grep of the tree* — **which copy** |
| §8.11c | *`pytest` then `git add` is not atomic* — **when it lands** |
| §8.11d | *`git add` then `git commit` is not scoped* — **whose index** |
| **this** | **a long-running process is the tree at its start** — **when it loaded** |

**It is the only one where the stale copy is in memory rather than on disk**, so
every disk-based check — `git log`, `git status`, reading the file — confirms the
*new* code while the thing under observation runs the old. **The instrument and the
subject disagree, and the instrument is right about the wrong object.**

**Cheap disposition, given nine agents committing continuously:** stamp the start
time of any long run, and compare it against `git log -1 --format=%cd <the fix>`
before interpreting a single line of output. **A measurement of a process is dated
by the process, not by the tree.**

#### 8.11e And the shell has to agree the control can block

**`task-graph` found the worse form in their own habit, and measured it:**

```bash
pytest ... | tail -2 && ruff check && git commit ...
```

```
python -c "print('FAILED'); sys.exit(1)" | tail -2 && echo COMMIT  →  COMMIT WOULD HAVE RUN
python -c "print('FAILED'); sys.exit(1)"          && echo commit   →  correctly blocked
```

**The pipe to `tail` discards the exit status** — a pipeline reports the *last*
command's, and `tail` always succeeds — **so every `&&` after it was
unconditional**, all day, on every commit.

**One notch worse than §8.11d's second half.** There a control's output was not
read; here the output **was** read, and the shell had already discarded the verdict
the `&&` appeared to depend on. The number was true and the gate was unrelated to
it. *"I was reading the number and believing the `&&`."*

**So the rule generalises past sequencing: a verification step must be able to
block, and the shell has to agree that it can.** `set -o pipefail`, or no pipe when
the exit code is the point.

#### 8.11g The day the instrument was the thing that never worked

**One package's summary, and it holds across all nine:**

> Three today where **the evidence, not the code, was the thing that never worked.**

§8.11f says the reasoning was aimed one step to the side. This is narrower and
worse: **the apparatus that produced the evidence was broken, and it reported
successfully.** Not a wrong conclusion from a good measurement — *no measurement,
formatted as one.*

Every instance from a single day, each found by someone other than its author:

| the instrument | what it actually did |
|---|---|
| `test_subtask_monitor_does_not_transition_parent` | **constructed the `SUBGRAPH_DONE` by hand** — the test was the missing producer |
| a drift guard | `assert CLI_ENV_VAR == "AGENT_SYS_CLAUDE_CLI"` — **compared a literal to a copy of itself** |
| a synthetic `ProgramBody` | stood in for the thing whose absence it should have detected |
| a stub returning `is_running=True` for every task | in a test written to catch exactly that class |
| a probe constructing the backend directly | **removed the condition it was there to observe** |
| `grep -c "$pat" $F`, glob unexpanded | printed empty; **empty was read as `0`** |
| `pytest … \| tail -2 && git commit` | the pipe discarded the exit status; every `&&` was unconditional |
| `-p no:randomly` | the plugin was not installed; **the flag was a no-op** |
| the criterion 8 leak probe | wrote to a root-owned `/`; a convincing `Permission denied` **with the sandbox off entirely** |
| `git diff --cached \| grep -c '<my lines>'` | a confirmatory check answering an exclusionary question |
| a grep of a session transcript | **the file records neither the system prompt nor the environment** |
| `test_two_runs_do_not_collide` | gave each run **its own root**, so it proved two roots do not collide — **a case nothing threatened.** The collision is two runs under *one* root, which is what typing the command twice produces |
| `tests/validator`'s output-phase suite | thirty tests over `PhaseRunner._targets`, **every fixture a single dispatch that wrote** — so the slot version and the store version were both `0` in all of them and *which of the two it read* was unobservable. `MemoryHandoffStore.read_verdicts` compounded it by returning `[]` for a version nobody wrote where the real store *raises*, so the divergence was **unrepresentable** as well as absent. Green for the whole of `_targets`' life; found by running `bringup/n1`, the first package whose non-leaf declares an output |
| `test_repo_holds_only_schemas_general_and_demo` | scanned `*.jsonnet`. **Criterion 5's guard, aimed at a format the repository no longer contains** — measured, a stray `.jsonnet` in a forbidden component still failed it, a stray `.yaml` passed. Not dead; **aimed at the one thing nobody can add any more and blind to the one they can** |
| `test_no_source_format_survives_the_deletion` | criterion 17's guard, **written this stage specifically to close a hole**, and its first draft walked the nine package directories and **missed `tests/` — where the only surviving `import _jsonnet` lived.** Caught by its own author, before shipping, by planting the import back |

**The last two are one author's, hours apart, and they share a cause worth stating
separately from the list.** Their own summary:

> Each scanned the set that was **convenient** rather than the set the criterion
> was **about**. Neither was found by reading — both took a planted file.

That is the generalisation §8.11g had been missing. Every row above is a broken
instrument; these two say **how** a guard becomes one without anybody being
careless. The scan is written against the tree in front of you, which is the tree
where the fault is absent — so the set that is easy to enumerate and the set the
claim is about coincide **today**, and diverge on the day the guard matters.

**The defence is cheap and already existed in this repository**, at
`tests/env_mgr/test_imports.py:225`:

```python
cited = set(re.findall(r"`(test_\w+)`", readme))
assert cited, "the README cites no tests at all; the mapping has been lost"
```

**A non-vacuity assertion on the scanned set, before the per-item claim** — and it
must itself be tested by disabling the scan, or the repair is the same defect with
a newer date. And the discriminator that says which guards need it:

| shape | what an empty scan means |
|---|---|
| `assert not scan()` | **still works.** Empty *is* the claim; it fails the day something appears |
| `assert not [p for p in scan() if pred]` | **vacuous.** The claim is about each item, and there are none |

Those two sit twenty lines apart in intent and behave oppositely, which is why a
sweep for this class has to read assertions and cannot count scan sizes.

**The last row is the variant to watch for**, because it is the one that looks most
like coverage: not a broken instrument but **a working instrument pointed at the safe
case.** It passed for a year while `layout_for`'s second-resolution run id let two
runs started inside one second share a directory — `create()` uses `exist_ok=True`, so
the second silently adopted the first's store, criterion 13 failing without a word and
criterion 12's resume then continuing the wrong graph. Found by its own author
(`0b9b55c`), while answering a question about something else.

**They are not carelessness and that is the point.** Every one was written by someone
competent, for the right reason, and each *passed* — which is the property that makes
this class distinct from a bug. A failing instrument fails loudly; **an instrument
that fabricates its own input succeeds, and everything downstream is true of a world
the system cannot reach.**

**The standing consequence, cheap and mechanical:** *show me the guard failing.*
Neutralise the thing under test and watch the instrument go red **before** trusting
it green — mutation, a positive control, a planted marker. Every instrument on this
list was disproved that way in under a minute once someone thought to try.

**And the corollary for reporting:** an instrument that returns nothing has told you
nothing. *Empty is not zero, silence is not absence, and a pipeline's exit status is
not its output's.*

#### 8.11f The one substitution under all of them

`task-graph`'s unification, and it covers every instance in §8.7a, §8.11b–e:

> **I reasoned about the thing I could see — my files, my changes, my test run — and
> the mechanism operated on something adjacent: their files, their changes, the
> shared index, the pipeline's exit status.**

Each entry is that substitution with a different adjacent thing. The visible object
is a real object and the reasoning about it is sound; **what fails is the assumption
that it is the object the mechanism consumes.** That is why none of these are caught
by being more careful about the thing in view — the care is already correct, and
aimed one step to the side.

### 8.11c `pytest` then `git add` is not atomic when another writer is live

**The third form of the tree/HEAD hazard, and the first two do not cover it.**
`env_mgr-2` reported *"`f4c55ac`. Whole suite 1802 passed"* — **true of their
working tree and false of the commit.**

```
pytest .              →  1802 passed        the line WAS in the working tree
git add …layout.py    →  captured whatever the file held at that instant
git commit
```

Another writer was live in the same file. **Between those two commands the file
changed underneath, and `git add` took the later state.** The green was real, of a
tree that no longer existed by the time it shipped.

| | |
|---|---|
| §8.7a / `task-graph` | *their files are my pathspec* — **what** lands |
| `env-mgr` | *a pathspec bounds files, not authorship* — **whose changes** land |
| **this** | **the tree you tested is not the tree you ship** — **when** it lands |

**The artefact is distinctive and that is the dangerous part.** The commit carried
the narrowing's docstring, three new tests and four fixture corrections — **and not
the one line that performs it.** A commit with every justification present and the
behaviour absent **reads as someone else's breakage rather than as an incomplete
change**, and the next person to run the suite spends the measurement finding out.
`env_mgr` spent one establishing it before touching anything, which is the only
reason it did not become a wild goose chase. (§4.17c is this seen from the suite's
side; this entry is the mechanism.)

**The fix is mechanical: verify what is *staged*.** Stage first and test the staged
state, or at minimum read `git diff --cached` for the load-bearing line before
committing. `git add <path>` had been treated as a snapshot of what was tested; **it
is a snapshot of now.**

**And it devalues a specific artefact**: a suite number attached to a commit hash.
From here, say what was verified and when, or verify the staged state.

### 8.11b A grep of the committed tree is not a grep of the tree

**§8.7a's writer-side rule has a reader-side twin, and nobody had written it
down.** `handoff` told `task_graph` *"nothing calls `allocate` yet"*, citing
`scheduler.py:187-192`.

> **That was a read of `HEAD`, not of the working tree** — the wiring was sitting
> uncommitted in front of them.

**In a shared worktree those are two different documents.** §8.7a says an edit is
published the moment it is written; **the consequence for a reader is that
`git grep` and `grep` answer about different worlds**, and the one that reads as
authoritative is the one that is behind.

**It is §8.11's shelf life with a second axis.** That one says a `grep` result
decays over time; this says **a `grep` can be stale at the instant it is run**,
because it was pointed at the wrong copy.

**And the same session produced the case where checking worked:** told to run
`git status` before building, `task_graph` found the §4.14 wiring already present
in their own package **and stopped instead of building a second copy.** *"That is
what saved the afternoon."*

**One more instance of the drift class, found in the right direction:** six
commits landed in `task_graph/` while they probed, **three of them correcting
docstrings of theirs that had gone stale** — `4c325ce`, where their comments said
the grant is `v<N>/` and it had not been since `d0681b5`. **Their own comments,
found by someone else.**

### 8.11 A grep result has a shelf life

**`handoff` changed `seal`'s contract two and a half minutes after `agent`
committed against it.** `dad0a46` at 06:55:56, `fd31a6c` at 06:58:32, and the
notice went out **after** the second commit.

**And this is not today's other failure.** The other three were *a document not
re-read after a ruling moved its premise*. This one is a **live seam**, and they
**did** name both sides — they messaged both consumers. Their own diagnosis:

> **Announcing a change is not the same as landing one safely**, and in a shared
> worktree with concurrent agents the gap between the two is measured in minutes.
>
> The rule that would have caught it is not *"tell the other side"* — I did that.
> It is **"before changing a signature, look at whether anyone has started
> calling it"**, which is one `grep` and which I had done **earlier the same
> hour** and concluded *zero callers*.
>
> **That conclusion had a shelf life I did not think about.**

**A grep result is a measurement, and it decays like any other.** Every stale-doc
finding this week was a claim written down; this is a claim that was **never
written down at all** — held in the author's head for forty minutes and acted on
as if still true. **The unwritten measurement is the one with no re-read
protocol.**

**Both failure modes it produced were the ones the design had guarded against:**
a refusal that no longer raises, so the reason is **lost**; and `except
Exception` now **swallowing `NotSealable`** — the wiring bug, the re-run case,
**and precisely what `ea832fa` split the type out to prevent.** *"My next commit
handed it straight back."*

**Green again by the time it was reported** — `tests/agent` 174 passed — because
the fix went with the report and the caller applied it. **The change was correct
and the landing was not**, which is a distinction worth keeping separate.

### 8.11a A caller depending on an error's prose is an undeclared seam

`agent` propagates `handoff`'s refusal **text** into a monitor record as
`seal_refused`, so **two error messages are load-bearing outside the package that
writes them** — and neither side declared it. `handoff` has committed to not
rewording without notice and offered a structured reason instead.

**Cheap now, expensive later**, and the general form: **a seam can exist without
either party having created one.** Nothing was designed, nothing was agreed, and
a string is now a contract.

### 4.24 A container whose element type changed still iterates

**Wall 8 read as an environment bug all day and was three layers from its cause.**
The symptom:

```
KeyError: 'AGENT_SYS_DEMO_STORE'   in store_root(), examples/demo/logic/store.py:53
```

The cause: `validator` writes `materials.json` as a **JSON object**, `hid -> staged
path`. `store.materials()` did

```python
[Path(entry) for entry in json.loads(...)]
```

and **iterating a dict yields its keys**. So it returned handoff *ids* as relative
paths, `staged_content` answered `None`, `check.py`'s `or store.content_dir(hid)`
fell through to the fallback, and the fallback died on a variable **that was never
the problem.**

**Nothing raised anywhere.** A dict iterates; a missing directory is falsy. Two
silent degradations in series, which is why three layers of distance looked like
one bug at the bottom.

**The transferable shape: the producer improved and the consumer kept parsing.** A
list became a mapping — strictly better, positional correspondence replaced by
explicit pairs — and the consumer's `for` loop went on succeeding against the new
type while meaning something else. **A type change that would have been caught by a
signature is invisible across a serialisation boundary**, because JSON has one
bracket for *sequence of things* and another for *mapping to things*, and `for`
accepts both.

**§8.7d, a third time.** `staged_content`'s own docstring called positional
correspondence *"the one guess left"* and named the mapping as the fix. The prose
was right, in the file, and not acted on until the defect surfaced somewhere else
entirely.

**Two dispositions worth copying.** The guess was **deleted, not defended** —
lookup is by handoff id and the positional path is gone rather than kept as a
fallback. And the fix was verified against a real validation zone from a real run
**with `AGENT_SYS_DEMO_STORE` absent from the environment entirely**: a control,
not a check. Patching at the symptom would have "worked" and left all of this in
place.

#### 4.22b The third row withdrawn — the ruling contradicted its own line

`9375517`. **`stage` no longer widens when permissions are off**, and the reversal
applies this section's own sentence against this section's own table:

> **Materialisation is not permission management.** If you find yourself disabling
> something that makes a file appear where a task needs it, you have crossed the
> line.

**Where a body finds its input is materialisation.** Widening moves every staged
input down one level — `<materials>/<hid>/v<N>/content/…` instead of
`<materials>/<hid>/v<N>/…` — so the switch would have **broken a body by moving its
input**, on the one day the demo runs with the switch on, and it would have
presented as *a body reading one level short* rather than as a switch.

**Two changes, each correct alone, composing into a break**, landing in the same
hour: `demo` removed that hop from `render.py` on `env_mgr`'s own measurement, and
the switch put it back. Caught by running **both staging modes side by side**, which
neither owner could do from one side.

**And the argument that settles it is the one the ruling could not have made:** with
nothing confined, a body **can read the store directly** — an unconfined control
succeeds on all four reads — so narrowing the *copy* denies it nothing it could not
already reach. The row's stated purpose, *"the narrowing decides what a consumer may
see"*, is void in this mode. What is left is a path convention, and **a switch that
changes a path convention is not a permission switch.**

`stage(narrow=)` stays and stays tested; only the wiring from the switch is gone.

**The general form, better than §4.22's:** *a switch must be checked at every
consumer of the thing it switches off, not only at the producer.* `spawn` was step
7's other consumer; `render.py` was `stage`'s.

#### 4.22c Three amendments to one prediction, and the amendments were the wrong part

`task-graph-2` predicted: **a hole presents as an empty input directory, with no
error.** It was twice "improved" by the module that owns the code, and both
improvements were wrong:

| amendment | why it was wrong |
|---|---|
| *"a narrowed `stage` skips an absent `content/`, so a hole stages **nothing** — one notch louder"* | skipping needs `content/` **absent**, and `handoff.allocate` always creates it |
| *"the switch gives a debugger a signature: an input holding exactly `content/` and `claim/`"* | withdrawn with §4.22b — the switch no longer widens |

**The base prediction has now survived both and stands exactly as written.**

**The diagnosis is the author's own and it generalises:** *a fact about another
package's function, asserted from my own control flow.* Reasoning forward from the
skip branch they had written that morning, without checking what the **allocator**
leaves behind. **An outside prediction beat two inside ones**, because the inside
view supplies a mechanism confidently and the mechanism was a guess about a
neighbour.

Each was corrected to the affected party before it cost them, and the standing
correction went into `env_mgr/README.md` rather than only into a message — *"since
the message is what was wrong the first time."*

### 4.26 A correct behaviour resting on a cross-package invariant neither package asserts

**The strongest form of the "interesting direction" family, and there was no weak
guard to notice — there was no guard, and nothing that looked like one.**

`agent/runner.py`'s `_apply_confinement` **infers** rather than reads:

```python
if prepared.confinement is None:
    return          # hands `spawn` to nobody
```

**The inference is correct, and it is correct because of `env_mgr`:**

```python
env_mgr/prepare.py:465-467
    conf = None
    if enforcing:
        conf = _apply.confinement_for(select(av), av.landlock_abi)   # select RAISES
```

`select` raises rather than returning `None`, so **`confinement is None` implies
`permissions_enforced is False`**, and the case the early return would mishandle
cannot be produced. **Today.**

**What it costs if the invariant moves is not small.** `_apply_confinement`'s early
return means the executor never receives `spawn`, and `backends/program.py:109` is
`start = self._spawn or subprocess.Popen` — **a `ProgramExecutor` that never receives
`spawn` runs with the operator's privileges.** A one-line change in `env_mgr` —
returning `None` where it now raises, or a fourth reason for an absent confinement —
**makes a program task silently unconfined, with a green suite on both sides.**

**The gap was never the record.** It was that **nothing checked the thing making the
behaviour safe.** `d779274` pins it, shown failing: replacing `if enforcing` with
`if False` in `env_mgr` turns it red.

**Two dispositions worth copying.**

**A test, not a runtime refusal**, and the reasoning is in the file. Refusing inside
`_apply_confinement` is the stronger guard and the wrong shape: **the invariant is
`env_mgr`'s to keep, asserting it at run time puts one rule in two places**, and the
fixtures that report `confinement=None` while enforcing are stand-ins for exactly the
composition the invariant says cannot exist.

**Source-level rather than behavioural, and stated as such.** `bwrap` is absent here
and Landlock present, so the branch that matters is unreachable on this machine and
**a green behavioural run would prove nothing** — §8.7 applied to one's own test
rather than to someone else's.

**And the general lesson is about where to look, not what to write.** The author went
in to change a *record*, found the record did not need changing, and found this
instead — *"had you not put it on the queue I would not have opened the file at
all."* **An invariant held by one package and depended on by another, asserted by
neither, is invisible to both suites and to any reviewer of either side.**

### 4.25 An `Any` the implementation could have named is a defect, not a convention

**Ruled as a class 2026-08-29**, after the same shape was ruled twice as instances
and a third arrived:

> **An `Any` on a cross-module surface that the implementation *could* name is a
> defect.** It covers fields, parameters and return annotations alike. **The forced
> ones stay a named exemption list**, because *"cannot name this type"* is a fact
> about a module's import surface, and a new entry should be a decision someone
> writes down.

| | |
|---|---|
| `Prepared.confinement` | `Confinement` was importable at `prepare.py:41` — **never an exemption case at all** |
| `Prepared.zone` | one intra-package import away |
| `EnvManager.place_zone -> Any` | same, in a **return** annotation, so outside the field ruling's words *and* outside the test it produced |
| `Prepared.output_paths` | **genuinely forced** — naming `HandoffId` means importing `task_graph`; the seam is one-way |

**Why the class and not the instances:** two-character changes queuing behind rulings
is a cost, and each instance ruled separately leaves the next one looking novel. The
module-local form of the same rule already existed — *this module names types from
its own package and leaves every other package's as `Any`* — and promoting it costs
nothing.

**And the refinement that says when a ruling is needed at all**, from someone who
queued a request they did not need:

> The question is not *does this touch a frozen surface* but **which side of it is
> wrong.**

`Prepared.confinement` needed a ruling because **the declaration had to move** — a
frozen surface changing meaning, two sides, §1.1. `EnvManager.place_zone -> Any`
needed none, because **the declaration was already right and only the implementation
disagreed**: bringing an implementation into line with a contract it already violates
is not a cross-module change, it is a fix. **Read the declaration before asking.**
That is the test for whether §1.1 applies, and it is cheaper than every ruling it
saves.

#### 4.25a Three surfaces, three axes, and each test priced one of them

**The recurring shape underneath**, found three times in one file in one morning:

| surface | can diverge in | the test checked |
|---|---|---|
| `Prepared` (fields) | names, defaults, **annotations** | names |
| `EnvManager` (methods) | parameters, defaults, **returns** | parameters |
| the exemption list itself | whether it is non-empty | *nothing, until a control was added* |

**A test whose entire justification is pricing two declarations of one shape, pricing
one of the things that can diverge.** And the second instance survived because the
first was fixed alone: *"I fixed it for `Prepared` this morning and did not look at
`EnvManager` beside it."*

> ~~When a defect is found in one declaration-pair, sweep the file for the others~~
> **Sweep `protocols.py` for what it declares, then the package for what redeclares
> it.** The narrow-fix habit, not a new failure mode — which is why it needs a rule
> rather than a diagnosis.

**Corrected by the person who applied it.** *"Sweep the file"* was shaped by where the
first two pairs happened to live; **the pairs are not in one file** — `Zone` is in
`fs/zone.py`, `Policy` in `isolation/policy.py`. **The declaration is the index, not
the file**, and the corrected sweep found **six pairs where the ruling had said two**,
four of them with no comparison at all — *not because anyone had judged them safe*
(`02a3f4c`).

It also found a live divergence the rule had not predicted: `Policy.granted` is
**required** in the declaration and **defaults to `()`** in the implementation. The
argument for removing the default is better than the rule that found it:

> **A convenience that makes the refused state the default state is not one.**

Eighty lines from `protocols.py:66`'s *"`UnresolvedGrant` is raised rather than
resolving to an empty granted set"*, the module whose central rule is that an empty
granted set must be **loud** had made *grant nothing* the easiest thing to write.

**Three dispositions from that sweep worth reusing anywhere:**

- **Name what is *not* a pair and why.** Eight types are imported from `protocols`
  rather than redeclared — one definition, cannot drift — and saying so means *"why
  is X not here"* has an answer instead of being re-derived.
- **Compare defaults by value *and* type.** An empty `dict` against a
  `MappingProxyType` is an equal value and **a different mutability contract**.
- **Derive the pair list by AST, and make dropping an entry fail.** *A hand-typed
  list of what to check is the same instrument as the wall test's hand-typed
  `ABOVE` — silent about whatever nobody added, and **the failure is always the
  unlisted one.*** That is §8.11g's general form: **an inventory maintained by hand
  has a blind spot in exactly the case it exists to catch.**

#### 4.25b Why the seams were where the findings were

**The empirical argument for reading across a boundary, from two people who could not
see their own defect and found each other's in an afternoon:**

> The pattern is not that we were careless; **it is that the thing each of us could
> not see was in the *other* package's five lines.**

One had written an early return that depended on a neighbour's invariant **without
knowing they were depending on anything**; the neighbour had never asserted it. The
other's race was in five lines of `Runner.start`, and the reader's first instinct was
a dirty file in another package — **plausible, adjacent, wrong.**

> **Neither was found by looking harder at our own code. Both were found because
> someone else's obligation forced a read across a seam.**

That is also why `place_zone` needed no ruling: the class rule worked on its first
instance, and **nobody had to ask.**

**And the third statement of §8.7d's mechanism arrived here**, from the person who
refused to write a confidently wrong note:

> **A stated rule is worse than an unstated one when it is wrong, because the next
> reader stops checking.**

The wrong rule was *"`HandoffId` is the only type `prepare.py` cannot name"* — false;
`task_graph`, `agent` and `validator` supply three more. It generalised from the
exemption list, which covers **fields**, to the module, which is mostly
**signatures**.

#### 4.22e The frozen declaration went wrong the moment the switch landed

**`protocols.Prepared.confinement: Confinement` → `Confinement | None`, `5e9e063`.**

`None` was not an edge case: it is **the declared way to say unconfined**, branched on
in four places across three packages — `prepare.spawn`, `agent/runner.py:749` and
`:1343`, and `demo`, which verified it as *"a supported value on a declared field"*
before building the §4.17a banner on it. `runner.py:749`'s comment exists **because**
a `getattr` default would answer *"no confinement"* to a missing field. **The
declaration forbade the value whose careful handling is the reason that comment was
written.**

**And the declaration documented the value its own type forbade** — nine lines above
the annotation, `protocols.py:329`: *"`confinement is None` would otherwise mean
unconfined for two different reasons…"* §8.7d again, in its worst placement: **two
contradictory statements nine lines apart in one file**, with no boundary to cross in
order to miss it, and **the prose is the more convincing of the two.**

**The timing is the finding.** At `ad730a2~1`, `prepare` ended
`conf = confinement_for(select(av), …)` and `select` **raises** — so no `Prepared`
this module produced could carry `None`. **The declaration was true until the kill
switch made it false, eight commits earlier the same day.**

> **A frozen declaration goes wrong at the moment its implementation gains a
> capability, not slowly.**

Not stale prose found late — a live divergence, created that afternoon, with another
package already building on the new value. It is the argument for a
declaration-versus-implementation test that a slow-drift story cannot make.

**The test that now holds it**, and the three axes matter:

| axis | rule |
|---|---|
| names | strict equality |
| defaults | strict equality — measured to agree today, so green for a reason |
| annotations | **an exemption list, not a subtyping rule** |

A list rather than a compatibility check because *"cannot name this type"* is a fact
about a module's **import surface**, and a subtyping check would be inferring intent
from types. **A third entry then has to be written down by someone.**

**And an exemption list decays in its own way:** `output_paths` is genuinely forced
(naming `HandoffId` means importing `task_graph` into `prepare.py`, a one-way seam);
`zone: Any` was **an intra-package import away**. Ruled: narrow it. **The moment a
fixable thing rests in the list because it once sat beside a forced one, the list
stops meaning *cannot* and starts meaning *did not*** — the same decay as a stale
docstring, inside a structure built to prevent decay.

**The control worth copying:** emptying the exemption list must fail. *Without it, a
loop that skipped everything would look identical to one that checked everything* —
§8.11g's purest form, and the author also caught their own broken injection, which
had hit the first of two identical `agent_cli: str | None = None` lines and passed.

#### 4.22d The switch made the run *more* restricted, and the banner said the opposite

**With `AGENT_SYS_NO_PERMISSIONS=1` set, the second real model call could not write
anything — and not because of anything this system did.** The agent knew the
destination, and tried three times:

```
Bash: echo "OUTPUT_SUMMARY=[$AGENT_SYS_OUTPUT_SUMMARY] …"   blocked
Bash: grep -rn "AGENT_SYS_OUTPUT_SUMMARY" …                  blocked
Bash: printenv AGENT_SYS_OUTPUT_SUMMARY                      blocked
```

Its own report: *"`AGENT_SYS_OUTPUT_SUMMARY` cannot be read — intercepted by the
permission layer — and `Write` into the zone's own `playground/summary/` returned
'requested permissions … but you haven't granted it yet'."*

**The `claude` CLI was still in its default ask-for-approval mode, with no approval
channel.** It could not `printenv`; it could not write **inside its own zone**.

> **We switched off our sandbox and left the SDK's on. The run is *more* restricted
> with permissions "off" than it would be with them on.**

**§4.16's own aside predicted it** — *"in practice a harness runs with
`bypassPermissions` on"* — and nobody set it. §4.22 enumerated three rows for
`env_mgr` and **the fourth consumer of "permissions off" was in `agent`**, which is
§4.22b's general form arriving a third time: *a switch must be checked at every
consumer of the thing it switches off.* `spawn` was step 7's other consumer,
`render.py` was `stage`'s, and `ClaudeAgentOptions` is the whole switch's.

**The disposition:** gate it on `permissions_enforced`, never unconditionally. **A
`bypassPermissions` that fires while our own enforcement is on is worse than this
bug** — silent, and in the direction that matters.

**And the display was the exact inverse of the truth.** The banner read *"nothing is
confined, no grant is enforced"* while the harness was refusing every tool call.
§4.17a is about a control a display can silence; **this is a display asserting an
absence of control that was not absent.** A run's own account of its permissions was
wrong in the reassuring direction *about being unreassuring*, which is a shape
nobody had allowed for.

### 4.23 A deliberate loudness, silenced twice by tolerant readers

**`task_graph/bootstrap.py:211` arranges for a loud failure, in a comment that
states the reasoning:**

> A root that was not supplied leaves the name unregistered: an artefact store
> rooted at a default nobody chose is **worse than a loud `KeyError` at the first
> resolution.**

**Two consumers then convert that `KeyError` into nothing.**

| | |
|---|---|
| `scheduler.py:355` | `if "handoff_store" not in self._r: return {}` — *"an absent `handoff_store` pins nothing, and is a supported mode rather than a guard"* |
| `agent/runner.py:1046` | `store = runner.component("handoff_store"); if store is None: return {}` |

Each is locally justified. `tests/task_graph` runs entirely without a store, so the
storeless mode is real and neither reader can raise. **But at the point of
consumption an intentional absence is indistinguishable from a real one**, and the
two skips are in series: nothing pins, then nothing seals, and the gate reports
`OUTPUT_ABSENT` for a task whose body exited 0 and wrote a complete artefact into
its granted path.

**§4.11 and §4.13 in one code path** — a plausible empty value produced, and a real
absence never reported. The symptom `demo` measured is what it looks like from the
far end: an artefact you can `ls`, and a run that says it was never delivered.

**Amended: there are four readers, not two, and the fourth changes the defect.**
`monitor` measured a storeless run against `agent`'s real methods:

```
_seal_outputs -> {}
_gate         -> []
```

**`_main` treats an empty failure list as a pass.** So the task **succeeds**, having
published nothing it declared, **and the record names no cause at all.** This entry
originally said the gate reports `OUTPUT_ABSENT` — a *wrong* report. It reports
nothing. **A misleading report and a silent success are different defects and the
milder one was written down.**

**And the fault is the conjunction, not any one reader.** Storeless is a supported
mode and none of the four is wrong about the single fact it tests; what nothing
tests is *declares outputs* **and** *no store*. That is why `task_graph`'s
`_pin_outputs` warning (`99b1c8e`) is the right shape and the two early returns are
not — one reader asks the conjunction, the others each ask half of it and answer
correctly.

**No new seam was added for it, and the reason is worth keeping.** A
`report_fault(task_id, kind, …)` into `monitor` founders on `kind`: the caller
cannot import `EventKind` either, so the parameter degrades to a string — **and the
closed enum is what makes `read()` enumerable.** A seam that costs the property the
record exists for is not worth having; the party that can report is the one that
already owns the channel. Nor per-task: `default_fingerprint` includes `task_id` and
`Recorder.write` is append-only, so one cause would show as N copies.

**The general shape, which is what this entry is for:** a module can arrange for a
loud failure and be overruled by its readers without anyone editing its code or its
comment. **The comment is still true and still describes an outcome that no longer
happens.** §8.7d is the single-file version of this; here the distance between the
intent and its defeat is a package boundary, so no reviewer of either side sees
both. The tolerance is not wrong — **it is unreported**, and that is the fixable
part: a dispatch that pins nothing for a task **that declares outputs** should say
so in the record, not in a log.

### 4.22 The permission kill switch — one reader, and it must be loud

**Ruled by the user 2026-08-29**, to get an end-to-end run today:

> 所有权限管理追加一键关闭开关，本次调通 demo 把权限管理模块关了。不进行任何权限管理。

**`AGENT_SYS_NO_PERMISSIONS=1`, read at exactly one place: `env_mgr.prepare`.**
Not in `agent`, not in `demo`, not separately in `grants` and `layout`. **A switch
with three readers is three switches**, and they drift the first time one of them
is refactored.

| step | on | **off** |
|---|---|---|
| 2 `grants.resolve_all` | resolves, **raises** on a missing granted path | resolves best-effort, never raises |
| 7 `select(av)` / `confinement_for` | picks a mechanism, `NoConfinement` if none | **not called at all** — `confinement=None` |
| ~~`layout.stage` narrowing to `content/`~~ | ~~narrows~~ | ~~does not narrow~~ **— row withdrawn, see §4.22b** |

**Step 7 must not be attempted and then discarded.** `select` raises
`NoConfinement` when no mechanism exists, and `bwrap` is absent on this machine —
computing it and throwing the result away leaves a live exception on the path of a
run that asked for no permission management at all.

**Materialisation is not permission management.** Staging inputs, deploying the
agent spec, cutting the workspace, `staged_package`, `environment` — unchanged.
The switch disables *confinement and grant enforcement* and nothing else. Anything
that makes a file appear where a task needs it is on the other side of the line.

**`confinement is None` now means two different things, and that is one too many.**
No mechanism on this machine, and the switch. `Prepared` must carry **which**, so
`demo` can display it and the runner can read it rather than infer it — §4.17a
applied before the fact instead of after: *a run that is unconfined because someone
exported a variable must not read the same as one that is unconfined because the
kernel offered nothing.*

**It does not repeal `ConfinementNotApplied`.** `agent`'s refusal (§3.3.1, ROADMAP
§6.1 P0) stands unchanged and is bypassed by an explicit, recorded, user-authorised
switch — not weakened. The distinction matters because the refusal is the only
thing standing between an AI task and an unconfined run, and a switch that
*softened* it would be indistinguishable from a switch that *disabled* it the next
time someone reads this code.

**And the switch is not the whole path.** Measured by `demo` the same day: with the
switch alone the run still dies at `produce`'s output validation, in the validator
body, on a `KeyError` for an environment name the body hard-reads and nobody
supplies. **Two walls, two owners** — a permissions wall and a plumbing wall, and
only one of them is what the switch is for.

#### 4.22a Built — and the two rows the ruling did not name

`ad730a2`. **The field is `Prepared.permissions_enforced`**, and the name is better
than what was asked for: **it states the fact, not the switch**, so neither reader
learns the variable's name and the field stays true if the switch is renamed,
replaced, or joined by a second one. It defaulted to `True`, so a hand-built
`Prepared` claimed the ordinary case — **the failing-open default would have been
`False`.**

> **Superseded 2026-08-30 — the field now defaults to `False`.** The paragraph
> above is kept because its *rule* survives and only its answer moved: the default
> claims **the ordinary case**, and the ordinary case is now unenforced. See
> §4.22f. The rest of this section is unchanged and still describes the built
> behaviour.

The one-reader rule is enforced **over the AST, not a substring**: three modules
name the variable in prose, so a text search answers a different question. §8.12's
*compare types, not names*, in a new place. Step 7 is asserted *not attempted* by
replacing `probe` with one that fails if called.

**Two rows the ruling missed, both wrong in the dangerous direction:**

| | |
|---|---|
| **`spawn` falls back to `select(probe())` when `confinement is None`** | `bwrap` is absent here, so **the kill switch fails closed in the one mode whose whole point is not to** |
| **`AGENT_SYS_NO_PERMISSIONS=0` read as ON** | every non-empty string treated as *set* disables enforcement for exactly the operator saying *no thanks* |

**The first is the instructive one.** It would not have presented as a switch bug —
it surfaces as `NoConfinement` from inside `spawn`, on a run explicitly told to
confine nothing, and sends its reader after confinement rather than after the
switch. **The ruling covered step 7 in `prepare` and stopped there, and `spawn` is
step 7's other half** — the half this very section names (*"the syscall happens in
the child"*) and then did not follow. Found by measurement, not reading: a real
child runs unconfined and returns `ran`.

**One more, ruled: the suite does not inherit the variable.** A `conftest` fixture
~~deletes it~~ **pins it to `0`** for `tests/env_mgr` — see §4.22f, a `delenv` now
selects the mode this directory does *not* assert. With the switch exported,
fourteen tests fail *correctly* — and **name the assertion rather than the
variable**, which is this module's own symptom-names-the-wrong-cause defect pointed
at its own suite. *A run whose result changes with the reviewer's dotfiles is not
reproducible, and a suite is a run.*

#### 4.22f The default is now OFF — a user ruling, 2026-08-30

**Ruled by the user, verbatim:**

> 关闭agent_sys的权限系统（干脆改成默认关闭吧）

Off, and **off as the default** rather than as something an operator opts into.
This reverses the direction §4.22a argued for and the reversal is a ruling, not a
finding — nothing measured says failing closed was wrong. It is recorded here so
that the code and this document do not disagree, which is worse than either
position.

**What changed: four defaults and no logic.**

| | before | after |
|---|---|---|
| `prepare.permissions_enforced({})` — the unset variable | `True` | **`False`** |
| `prepare.Prepared.permissions_enforced` | `True` | **`False`** |
| `protocols.Prepared.permissions_enforced` | `True` | **`False`** |
| `agent.Assignment.permissions_enforced` | `True` | **`False`** |

`permissions_enforced()` is still the **single reader**, still a function and not a
constant, and every spelling still means what it meant. The one edited expression is
its fallback: `env.get(NO_PERMISSIONS_ENV_VAR, "1")`.

**Enforcement is still reachable, and that is the half that makes this a default
change rather than a removal.** `AGENT_SYS_NO_PERMISSIONS=0` — or `false`, `no`,
`off`, or empty — enforces exactly as before. The `_FALSE` set was written in §4.22a
so a typo could not fail open; it is now load-bearing as the **opt-in path**.

**Why the default moved rather than the variable.** A companion
`AGENT_SYS_ENFORCE_PERMISSIONS` would be two variables that can contradict each
other, needing a precedence rule a reader has to look up — the same defect §4.22
names as *a switch with three readers is three switches*, in its two-variable form.
A rename would break every operator command line, the exported
`NO_PERMISSIONS_ENV_VAR`, three documents and `test_the_switch_is_read_in_exactly_one_place`,
and buy no behaviour. The cost of the chosen shape is one double negative — set *no
permissions* to *0* to get permissions — and it is written into the docstring rather
than left to be discovered.

**`Assignment.permissions_enforced` is the row that is not cosmetic.** §4.22d
measured it: with our enforcement off and the Claude SDK's own layer left at its
ask-for-approval default, the agent could not run **one** tool call. So the old
`True` default is no longer the safe direction — under the new run default it is the
direction that silently breaks the agent, while `claude_sdk.py`'s
`permission_mode: bypassPermissions` stays gated on the same fact and unchanged.

**What the flip does NOT touch, deliberately:**

| | |
|---|---|
| `closure/check.py`'s *every kind needs a covering grant* | a **spec-load** gate, never read by `prepare`. Out of the ruling's scope (权限系统 = runtime enforcement) |
| `isolation/policy.anchor_zone_root` | the CVE-2025-59532-class cwd guard, independent of the flag and always on |
| every `pytest.raises` denial assertion | kept. A test asserting a denial now **states its mode** — `tests/env_mgr/conftest.py` pins `=0` for that directory, and `test_the_harness_layer_stays_on_while_we_enforce` names it inline |
| `PERMISSIONS_DISABLED` (§4.17a, `cli/main.py:359`) | unchanged, and now fires on an ordinary run. That is the point: the loud banner is what keeps the new default from being silent |

**The pair that proves it is a default and not a removal**, because a green suite
proves neither on its own: `test_unset_leaves_it_off` (the default) against
`test_these_spellings_leave_it_on` (`=0` still enforces), and behaviourally
`test_by_default_a_run_that_would_be_refused_proceeds`, whose first half must still
raise `UnresolvedGrant` under an explicit `=0` before its second half deletes the
variable.

### 4.21 A function with tests, and a caller in nobody's package

**`demo`'s end-to-end drive found three, and two of them were live blockers.**
This is the class a unit suite structurally cannot find and an e2e driver finds
first.

| | consequence |
|---|---|
| `handoff.registry.check_bindings` | never called — criterion 10's load-path check does not run |
| `env_mgr.workspace.ensure_precious` | never called — **so `cut` could never succeed, and no workspace had ever been cut in this repository** |
| `handoff.store.seal` | never called — **so content written under §4.14 was never published** |

**The shape:** a function that exists, is tested, is correct, and whose only
callers are tests. **Each has a green suite proving the function works and
nothing proving anyone runs it.** `demo`'s framing is the one to keep:

> **Tests that these seams are *reached* — a production caller exists for `seal`,
> for `ensure_precious`, for `check_bindings` — belong to the packages that own
> the callers, once the callers exist.**

**And the artefacts that found them must not become tests.** Every probe asserts
*"X is currently broken in another package"* by patching around it; committed,
they would **encode other packages' bugs as expectations and go red the moment
each is fixed.** That is the inverse of a regression guard, and it is the correct
reading of the *load-bearing measurements belong in `tests/`* rule rather than an
exception to it.

**One more, from the same drive:** *a hand-run that succeeds proves the body
works; it proves nothing about whether the body was ever launched.* `demo`
reproduced by hand a failure that was **behind** the real one and reported the
wrong wall; `agent`'s stderr capture, landing between two runs, is what caught it.

### 4.21a A relay with no source, and the test was the missing producer

**§4.21 has a worse form and `monitor` found it in their own package, on their own
criterion.** `04a5b76`:

```
protocols.py:78   SUBGRAPH_DONE = "subgraph_done"        declared
base.py:602                                              consumed
base.py:682   rekeyed(record, parent.id, SUBGRAPH_DONE)  re-emitted
```

**`:682` only forwards one it already received. Nothing anywhere created the
first.** Criterion 24's forwarding half was built, worked, and was tested; **its
producing half was never written.** A non-leaf's `_main` ends its thread at
`unfold` with the task in `RUNNING`, so **every non-leaf, root included, sat in
`main: running` for ever** — and no run of the demo had ever terminated cleanly.

**Why no suite saw it, and this is the turn of the screw past §4.21:**

> `test_subtask_monitor_does_not_transition_parent` has passed all along — **because
> it constructs the `SUBGRAPH_DONE` by hand.**

In §4.21 the test is the only *caller*, so the suite is merely silent about
production. **Here the test performs the missing production**, so the suite does not
just fail to notice the gap — **it fills it**, and every assertion downstream is
true of a world the system cannot reach.

**The method that finds it, which generalises to every enum in the system:**

> **Ask of each kind *"who writes this?"* rather than *"is this handled?"***

**And the naive form of that method passes for `SUBGRAPH_DONE`, before the fix and
after.** Its author ran it to completion over all fifteen kinds rather than stopping
at the one that bit, and found the trap in their own method:

```
monitor/base.py:724   rekeyed(record, parent.id, EventKind.SUBGRAPH_DONE)
```

**A re-emission is a write.** Classify hits as read-or-write and ask *does a write
exist?* and the answer is yes, for a kind that nothing originates. The corrected
form, which **no name-based scan can perform** — all fifteen had to be read:

> **For each kind, find the write sites; then for each one ask whether its input is
> already a record of that kind. If it is, that is a relay, and the question is
> unanswered.**

`SUBGRAPH_DONE` was the only kind with the relay-only shape, which is exactly why it
was the only one that hid.

**Two notes for anyone running it across packages.** `demo` declares its *own*
`EventKind` — `RUN_START`, `PACKAGE_LOADED`, `UNEXPECTED_SUCCESS` — a different enum
with the same name, so a cross-package sweep returns ~30 hits that are not
`monitor`'s. And a producerless member is **not automatically a defect**:
`ReportToUser` has no production producer because `PusherMonitor.decide` legitimately
has no case for it, which is the same standing as `decide` being replaceable at all.
The discriminator applied was the right one — *does anything require it now?*
Criterion 24 required `SUBGRAPH_DONE`'s producer and its absence broke the running
system; and `_to_user` was checked to be reachable by another route
(`base.py:793`), so the capability `demo` depends on has a live path.

**The companion rule, and it is the general one:**

> **An absence claim needs a clock *and* an instrument that could have found a
> presence.**

§8.11 gave the clock — a grep result has a shelf life. This gives the other half,
and it is the same requirement as §8.11d-ii's positive control: the grep that
reported *no producer* could not have found one, because a producer of
`SUBGRAPH_DONE` need not contain the string. The fix's own new producer does not
either, which is why the rekey site now carries a comment naming both callers.

The author had run that check for `EventKind` producers weeks earlier and **stopped
at `PHASE_DONE`.** They had also written, earlier the same day, that this class
could not be found from inside a package; the retraction is the useful half — it
can, with the right question.

**And one thing about how it was nearly missed twice.** An hour before, the symptom
had been explained *completely and correctly* — the `is_end` subtask never runs, so
no last phase, so no notification — with the conclusion *"no defect"*. Every step
was right. **A complete explanation of a symptom is not a demonstration that the
mechanism works**, and the explainer had said so themselves: *"that route has never
executed in this repository… I am telling you there is no evidence of a defect, not
that it works."* That sentence is why anyone kept looking.

### 4.20 Two layers of nothing under a green suite

**The seal landed and `tests/agent` was 169 green — and worthless.** `agent`
found it because the ruling asked for one specific check: *confirm the gate
reports `OUTPUT_ABSENT` for a task that genuinely produced nothing.*

**Layer one: `StubStore` had no `seal`.** The runner catches broadly on purpose —
the refusal is `handoff.Malformed` and `agent` may not import `handoff`, so
**there is no type to name** — so **every `AttributeError` became "the seal
refused."** 169 green **with the seal never executing once.**

**Layer two: no fixture task declares an output.** `run_gate` loops over
`task.outputs` and so does `_seal_outputs` — **both loops were empty in every
test that has ever run in that package.** So even with a working `seal`, green
meant nothing.

**Neither layer is visible from the suite.** The first is a double missing a
member; the second is a fixture missing a field. **Together they made a feature
that had never executed look tested.**

**Their own rule caught the first, hours after they wrote it**, once pointed at
the store rather than at the SDK: `missing(agent's store reads, StubStore) ==
{'seal'}`. **A rule is not adopted until it is pointed somewhere its author did
not have in mind when writing it.**

**And they verified by removal rather than by passing:** with the seal disabled
both new tests fail, restored both pass, file byte-identical afterwards. **The
case the ruling named is pinned** — a task that produced nothing still reports
`OUTPUT_ABSENT`, carrying the store's own words, *"produced no content at all."*

**One thing stronger than the ruling assumed:** **no production code called the
store's `seal` at all.** The only non-test `.seal(` was `task_graph/runner.py:101`
— `HandoffVersion.seal`, **a different verb on a different object.**

**And one dependency flagged rather than assumed:** `Execution.output_versions`
is pinned at `push_execution` and never touched afterwards. **If a retry ever
re-pinned mid-attempt, the seal would target the wrong directory silently.**

#### 8.10a The live instance — §4.14 moved publication, and a class of predicate went false

**This entry described the shape weeks before we did it.**

§4.14 moved publication from close to `_seal_outputs` (`agent/runner.py:636`), which
is **before the gate and well before `OUTPUT_VALIDATING`**. So `demo`'s
`_summary_was_published` — *does a summary version exist in the store?* — **flips true
at a moment the validator provably has not run.** It was defensible when `put`
happened at close, and it silently became false on the day the ruling landed.

The consequence was the loudest line the demo prints: **an `UNEXPECTED` label on an
outcome that was `UNTESTED`** — the artefact whose whole purpose is telling *never
attempted* from *attempted and failed*, doing the opposite.

**And the generalisation is the finding, not the instance:**

> **Anything in the tree that infers *this was checked* from *the artefact is in the
> store* has the same defect now.**

**The correct predicate names the judgement, not the target**: *a verdict exists
against that version*. With it, every route to an absence — **crashed body, never
selected, phase never reached, level `NONE`** — lands in one honest classification
instead of being read as a promise that stopped being kept.

**A sweep was commissioned rather than assumed**, with a stated discriminator (*does
this read presence-in-store and conclude something about judgement?*), read-only,
findings named to their owners — **and required to report a clean result as a
result.** A sweep that cannot report something is §8.11g's instrument again.

**Result: one instance, the one already known — and the sweep carried its own
positive control.** The known defect was left in scope, so *the discriminator fires
on it, and a clean result elsewhere is a result rather than an instrument that cannot
fail.* **Build the control into the sweep rather than bolting one on**; it costs
nothing when a known instance exists.

The per-case reasons are what make it re-readable a month later — not *"these nine
are fine"* but **the gate concludes *delivery*, and existence is the right question
for delivery**; `logic/store.py` filters on the manifest and its docstring cites
§4.14; `monitor/record.py` is the object store, a different question entirely.
**Someone re-running it can check the reasoning instead of repeating the grep.**

**Two limits, stated by the sweeper rather than found later.** *Code read, not
executed* — a predicate assembled at runtime would not appear, and **absence in a
grep is weaker than absence in a run.** And the one that matters:

> **I swept one premise.** §4.14 moved at least a second — *the version number is
> stable across attempts* — and that is the slot-versus-store divergence pinned in
> `fb97797`.

**That is this very entry, one level up: the ruling moved several premises and the
enumeration covered one.** The commission was written in terms of the premise in
front of the commissioner. **§4.14's blast radius has never been enumerated** — two
premises are named and there is no reason to believe that is the list. Open, and not
closed by a clean sweep of one of them.

**Re-run when the join lands**, with the same discriminator. That is the return on
having written the discriminator down.

**And the general form of the correct predicate arrived twice from opposite ends of
one afternoon.** `agent/runner.py:945` seals a slot `VALID` only on
`passed and hid in self._store_sealed` — **positive evidence that this attempt
published** — after measuring a slot sealed `VALID` for an attempt that published
nothing. Same rule as the predicate given to `demo`: **require positive evidence, not
the absence of a negative.**

### 8.10 A ruling moves a premise, and nobody enumerates what rested on it

**`handoff`'s diagnosis, and it is about my own role rather than anyone's code.**
Three instances in one afternoon, one decision:

| | |
|---|---|
| `handoff.allocate`'s docstring | still said `v<N>/` was the agent's grant — **written before `0c2df28` moved the grant off it** |
| my two rulings | the claim location and the grant narrowing, **contradictory the moment the second landed** |
| `agent/gate.py`'s four questions | written when `monitor` §4.1.1 had the producer `put` from inside its zone, so **a published version genuinely existed at gate time.** §4.14 moved publication after the gate and nobody re-read the gate against the new ordering |

> **A ruling moves a premise, and the documents that rested on it are not
> enumerated.**

**It is not carelessness and more care will not fix it.** The author of a ruling
is the one person who cannot see its dependents, because to them the ruling is a
conclusion rather than a premise — §8.3's *"the owner never hits the wall"*,
pointed at a decision instead of a seam.

**What the ruler owes is the list.** Produced once, at the end of the day, and
**it caught an error of mine in one experiment.** I had written that *the
runner's `False` branch **is** the gate's absence signal — if it returns `False`
and the gate does not report absence, a body that wrote nothing passes.*

**`agent` patched the runner to discard every refusal and re-ran the
produced-nothing test:**

```
assert "output_absent" in wired.monitor.kinds()          PASSED
assert "produced no content..." in ...["seal_refused"]   KeyError
```

**`OUTPUT_ABSENT` still fires.** It is signalled by the version being
**unsealed** — `exists()` means *published* — not by anything the runner returns.
**The refusal carries only the reason.**

**So the coupling to guard is one layer down**, and the corrected entry is:
*`exists()` must keep meaning published.* If that changes, **the gate stops
reporting absence no matter what the runner does with the return value.** What
the `False` branch actually protects is **criterion 5's distinction** — *never
attempted* versus *wrote badly* — not the absence.

**That is the list working rather than failing.** Their words: *"you produced
that list this time, which is why I could find an error in it in one experiment
instead of at the next reader."* **A wrong entry in an enumerated list is
falsifiable; a wrong premise nobody enumerated is not.**

I produced it zero times earlier and withdrew
**three rulings on one decision** — fresh-allocation-per-pass, seal-after-the-loop,
remove-the-loop — **each killed by a measurement that cost less than the ruling
did.**

**And the fourth version was not mine.** `agent` walked the passes and found the
loop is safe in every case but one, which reduced *choose a shape* to **one
question**: may a gate failure on an **already-sealed** version be retried inside
the attempt? Ruled **no**, and **on the cost of the alternative rather than on
retry policy** — *yes* buys a fresh `prepare` mid-attempt, which is step 7's
hazard reopened to save a re-dispatch.

**The self-correcting part is worth as much as the ruling.** `agent` withdrew
their own option (2) after checking it against their loop; `handoff` withdrew the
*gate questions are stale* framing once the seal moved in front of the gate, and
with it **a cost table I had been about to price the decision against.** Each
correction was the other party's, neither was mine.

**And `ea832fa` came out of it**: `Malformed` was catching *already published*,
so a runner catching a refusal would have **silently swallowed the re-run case**.
Split into `Malformed` (the artefact is bad — catch it) and `NotSealable` (the
call could not have succeeded — let it escape), **and it was not a new
distinction** — `Malformed`'s own docstring is *"content that does not satisfy its
kind"*, so *already published* was outside its stated meaning **the day it was
written.** Found by **a caller reading an interface before writing against it**,
which is the same shape as `env_mgr` catching the allocator.

### 4.19 The store can tell *refused* from *never attempted*, so F-D1 shrank

**A premise §5.14 was argued hard from is false.** The argument ran: *`monitor`
§4.1.1's "the producer calls `put`, from inside its own zone" was protecting
criterion 5 — refused versus never attempted, which **only the producer can
distinguish**.* §4.14 preserved it by keeping the producer writing from inside.

**The store distinguishes them, and already does.** Verified at
`handoff/store.py:381-392` — `seal` runs the admission checks rather than
blessing whatever bytes are present, and **an empty grant directory is checked
before its contents**, so the two cases fail at *different* checks:

```
NEVER ATTEMPTED   nothing was written to …/v0/content. That directory is the
                  agent's grant and it is empty, so this attempt produced no
                  content at all
WROTE BADLY       …/v1/content/README.md: every handoff opens with a README.md
```

**Criterion 5's distinction is readable from the state of the grant, without
asking the producer anything.** `handoff` did not design it — it fell out of
checking for an empty `content/` first — and reported it as weakening an argument
they had helped make.

**Nothing is undone.** §4.14 stands on its own merits and the producer still
writes from inside. **What changes is that a future ruling here has one fewer
constraint than the record suggests.**

**And F-D1's residue is much smaller than F-D1 as stated.** Two consequences:

| | |
|---|---|
| **The caller needs no evidence to attempt a seal.** | It may call unconditionally on every closed attempt: **a half-written body cannot be blessed, because the checks refuse.** *On what evidence* was a question only if `seal` were a rubber stamp, and it is not |
| **What remains open is what the runner *records* when a seal refuses** | The store hands back a `Malformed` naming **which of the two cases** it was. Whether that becomes an event, a verdict or a task failure is `agent`'s and `monitor`'s — **not a spec conflict** |

**Also measured, closing §4.14's last unmeasured item** — the retry case `demo`
could not reach from the CLI:

```
attempt 1: allocated v0, content written, abandoned
attempt 2: allocated v1  -> hole kept, v0 on disk, list_versions [] latest None
```

**Holes are permanent and numbers are never reused** — previously a claim in a
docstring, now evidence. **And the bound is worse than "per failed run":** the
version must exist before `prepare` resolves the grant and `prepare` raising is
*no isolation, no start*, so **holes accrue per refused dispatch**, which in a
bring-up run is most of them. Reaping remains unowned.

**One thing `handoff` refused, and it belongs in the record.** `demo` thanked them
for fixing a false-`UNEXPECTED` hazard *"before they asked"*. They declined it:
the `MANIFEST_FILE` filter went in because they measured `agent/gate.py` raising,
and `cli/main.py` was **the second reader the same guard happened to cover** —
*"one shared cause and luck, not foresight."* **Credit correctly refused is as
useful as a finding**, because the alternative is a record that says someone
anticipated a case nobody was looking at.

### 4.17b A probe made robust against unexpected input stops measuring the strict thing

**`env_mgr`'s second measurement error of the session, and they named the shape
both of them share.**

Their monkeypatched narrowing carried **a fallback to the wide copy** when a
source had no `content/`. **The real `stage` skips.** So the probe measured a
strictly **more permissive** change and reported five green suites — while the
real narrowing took **four tests red**.

> **The fallback was there to stop the probe crashing on fixtures.**

**And what the four red tests turned out to be is the second half:** they had
fabricated a bare `v<N>/` — **a layout no write route in `handoff.store`
produces** — and passed only because the copy was wide enough to hide it. Fixed
**at the fixture, not the assertion**, because a test asserting against a shape
the store never makes is the green-that-would-be-green-if-the-property-were-false
case, and editing the assertion preserves it.

> **Both of my measurement errors this session came from making the probe more
> tolerant than the code.**

**Same family as §4.17a's `-q`-silenced control, one layer up.** There, the
control's output channel could be dropped; here, **the probe's own robustness
removes the condition it was built to observe.** A probe that cannot crash on the
input the code rejects is not measuring the code.

### 4.17c A commit whose tests are all present and whose behaviour is absent

**`f4c55ac` shipped the narrowing's docstring, its three tests and four fixture
corrections — and not the one line that performs it.** HEAD was **6 failed** until
`5976502`. Two agents were editing `layout.py` concurrently; **no mechanism is
offered.**

**It is a third shape beside the two pathspec findings.** Those were changes that
landed **unintentionally**. This is a change that **did not land while everything
justifying it did** — and the result is worse than either:

> **A commit whose tests are all present and whose behaviour is absent reads as
> someone else's breakage to whoever next runs the suite.**

The rationale, the tests and the fixtures all argue that the change is there. **The
only thing that says otherwise is the failure**, and it points away from its
cause.

### 4.17a A control a display flag can silence is not a control

**`env_mgr`, one round into the staging measurement.** Their positive control was
a `pytest_report_header` returning a string — **and `-q` suppresses the header.**

> *"The monkeypatch never applied"* and *"the narrowing broke nothing"* **printed
> identically**, and I read a green five-suite run as evidence.

**Now an `assert` in `pytest_configure`.** This is §4.17's shape with a new
mechanism: not a wrong value but **a control whose only output channel the test
runner is entitled to drop.** Anyone whose probe **prints** its control rather
than **asserting** it has the same hole.

**And their first item is the same species one level up.**
`tests/env_mgr/test_imports.py` enforces the decoupling wall by classifying
modules against **two hand-written lists** — and `harness.py`, added above the
wall, was in **neither**, so it was checked by **neither direction**. Measured
before changing anything: injecting a live violation left the file at **41
passed.**

**Two holes, not one.** The AST reader also missed a fourth import spelling —
`from env_mgr import harness` puts the module in `names`, not `module` — **and
that is the spelling `material.py` actually uses**, so it is the one a
below-the-wall module would copy.

> **An enumerated allow-list inside a structural test decays exactly like a
> §5.x-settled-but-§4.x-stale doc row: it keeps reading as true.**

Fixed by **deriving** the above-the-wall set rather than listing it, so an
unclassified new module gets the **safe default instead of invisibility**, with a
partition test keeping the two sides exhaustive. **If another package has a test
that classifies by a typed list, inject the violation once and see whether it is
caught.**

### 4.18 The allocator contradicted the narrowing, one level below the rulings

`handoff.allocate` created `v<N>/` **and nothing else**, its docstring stating
that `v<N>/` was the agent's grant — **written before `0c2df28` moved the grant
off `v<N>/`.** The same failure as the two contradicting rulings above, one level
down, and **`env_mgr` caught it by measuring both outcomes** for a granted path
that does not exist:

| | |
|---|---|
| non-optional | `FileNotFoundError` — **every output-producing dispatch dies in `prepare`** (`landlock.py:198` opens every granted path; `Granted.optional` defaults `False`) |
| optional | the rule is dropped **silently** |

**And the agent cannot create it either** — `mkdir` inside `v<N>/` needs write on
`v<N>/`, which the narrowing removes.

**Ruled: the allocator creates every directory it expects to be granted.**
`env_mgr` declining to `mkdir` at prepare time is right — their design §1.2 defers
the layout to `handoff`, and doing it themselves makes them a second writer of it.
`seal`'s content check moves from *absent* to *empty*, since `content/` now always
exists and its presence can no longer stand in for the agent having written.

**Two granted paths cost `env_mgr` nothing**, measured: `resolve` already returns
a tuple, `Policy.granted` is flat, and the **16 cap is on layers, not rules**.

**Sequencing ruled: the narrowing and the second granted path land in one
change**, so there is never a window where `content/` is granted and the claim
location is real-but-unwritable.

### 8.7d A docstring that states the correct distinction, beside code that does not make it

**Three instances in one package in one afternoon, all found by someone else
running the real path.** `has_subgraph`'s own docstring says:

> Leaf-ness **after** unfolding is the absence of children (`TaskMgr.children`),
> never the `is_start`/`is_end` pair

**That sentence was in the file, correct, and not consulted at the one site where
the distinction decides anything.** `--resume` on a non-leaf rebuilt the entire
subgraph — a non-leaf's main phase *is* the unfold, so a resume re-enters it, and
`has_subgraph()` asks the **declaration**, which is just as true the second time.

**It is worse than a stale comment, and that is the whole entry.** A stale comment
announces its own age. This one **reads as evidence the author considered the
case** — so the next reader stops looking, and the docstring actively protects the
defect it describes. §8.11's shelf life does not apply: nothing here decayed. The
prose was right when written and right now.

The sibling instances, same package, same day: `FakeRunner.produce`'s *"the only
thing in the test suite that writes handoff state"* — which `demo` read as the
defect it was naming — and a `getattr`/`hasattr` pair whose comment drew the
distinction the code then dropped.

**And the reason no suite could see it.** `tests/cli/test_resume_continues_from_disk`
is a genuine two-process test with `os._exit(9)` — and it resumes **a leaf**:

> *A guard that exists, passes, and covers a narrower case than the real path.*

Same shape as the five test-only functions (§4.21): every package's resume was
tested, **the composition's resume was not**, and it was broken in the one way that
only appears with a subgraph. **The strongest argument yet for G3 asserting
arrival.**

**One rejection worth keeping.** Making the new `task_mgr` lookup tolerant was
considered and refused — *a stub that wants tolerance answers the method*, and **a
harness with no task graph is not the same fact as a task with no children**. The
cross-package cost (one line in `tests/agent`'s harness) was raised to the other
side by name instead.

### 8.7c A quote attributed to a teammate is a claim like any other

**`handoff` was asked to disambiguate a position they had never taken.** A
message to them opened *"Your line to the team lead was 'the highest successful
version, not the stack top'"* — **they never said it.** It is §4.14's own text,
the user's ruling, quoted back at them as theirs.

**Nothing was lost, and the mechanism is why it is here rather than the
instance.** Had they answered inside the offered frame, they would have been
**defending a position they never took, about a file that is not theirs, with the
weight of having supposedly already ruled on it** — and that defence would then
be *citable*, which is how it becomes load-bearing.

> **Second-hand attribution is how a claim acquires authority it was never
> given.**

**It is §4.11 one level up:** a plausible value flowing on undetected, where the
value is ***who said it***. A wrong fact gets checked against the code; a wrong
attribution gets checked against nothing, because the named party is assumed to
have already done the checking.

**The cheap defence is the one that worked:** they did not recognise the quote,
so they went and read §4.14. **The document is the `grep` for a quotation, the
way the tree is the `grep` for a claim.**

**Misremembering who said a thing is the most ordinary error there is** — which
is precisely why the defence has to be structural rather than a matter of care.

### 8.7b An outward measurement is the one that most needs a control

**`validator` nearly told two teammates they were both wrong.** Their probe of
`closure`'s import-wall claim was a string replace **against text that was not in
the file** — it applied nothing, the test passed, and for a minute they held a
green result contradicting two correct measurements.

> **I caught it because green was the wrong answer, not because the probe told
> me.**

> **The tamper-and-restore probe needs `assert tampered != orig` as much as the
> test it is checking needs to fail.**

**They had written exactly that assertion into another probe an hour earlier.**
The discipline was known, held once, and **dropped at the moment the result was
about somebody else's claim rather than their own.**

> **The check on a claim about another package is the one where a wrong
> measurement does the most damage, because the correction travels outward.**

**And their refinement names what the dropped assertion actually was:**

> `assert tampered != orig` **does not check the property; it checks that the
> experiment happened.** It is the **control**, not the test. **Every negative
> result needs one — and a probe is all negative results.**

That is why the omission was invisible: **a probe reporting "nothing happened" is
indistinguishable from a probe that did not run**, and the only difference is an
assertion nobody misses when it is absent.

Theirs would have gone to two people **who had each measured correctly.**
`spec-loader` hit the mirror image the same day — reading `handoff` and seeing
their own bug where `handoff`'s was contained, which they named **a reader with a
hypothesis.** Same class: **the outward-facing measurement most needs a control,
and is where the incentive to double-check is weakest.**

**And `closure` supplied the near-miss that shows a related gap.** They named
their own live instance of `validator`'s scaffolding-rot argument — a `task_of`
re-export with a comment saying it goes when the last caller does — **then
checked, and the last caller had gone hours earlier.** The comment had outlived
its reason **in the file of the person who had just finished explaining that
comments do that.**

> The only reason I looked is that I had written it down as an example. **That is
> not a method — it is luck with a paper trail.**
>
> **Naming your own instance of a pattern is not the same as guarding it, and it
> feels like it is.**

**And a live collision, reported before it could fire:** `closure/check.py`
already has `from .model import agent_of`. Importing `validator`'s at module
scope **shadows theirs silently** — two near-identical accessors over **different
document types**, and **neither raises on the wrong input.** §4.11's family,
prevented rather than found.

**They also name a third state nobody had: a reader with a hypothesis.** Worse
than either, **because they carry the writer's blindness without the writer's
knowledge.** `spec-loader`'s instance: reading `handoff` and **seeing their own
bug where `handoff`'s was contained.** The defence is the one `validator` used on
them twice — **check before relaying.**

**Seven packages have run this and seven found something, none of it found by a
suite** — each found by one package taking another's finding and pointing it at
itself.

**`spec-loader`'s is the most severe, and it inverted the function's first
documented property.** `load_package` promises *"failures are collected, not
raised: one broken spec must not hide the other nine."* It caught `SpecInvalid`
and `SpecInconsistent`, which is what `BaseSpecRegistry._validate`'s docstring
tells subclasses to raise. Measured against all four real registries:

| | raises | caught |
|---|---|---|
| `handoff`, `agent` | `SpecInvalid` | yes |
| `validator` | **`ValidatorInvalid`** — a `ValueError`, neither of ours | **no** |

**One package's choice of exception type aborted the entire multi-package load**,
for every other package in the run, silently, **with all nine package suites
green.** The property had become *die on the first*.

**And nothing had ever driven it.** All 96 of their tests pass `FakeRegistries`
whose `_validate` does nothing; `tests/interfaces/test_composition.py` uses a
double too, and its own comment says only `handoff_specs` is exercised. **The
subject of the most-used double in the tree had never been run by anyone** —
`spec-loader`'s inverse case: not a stub of a collaborator, but standing in for a
**caller** never driven.

**Their lesson is the one to keep, and it is not the obvious one:**

> The lesson is not *"`validator` should have subclassed `SpecInvalid`."* They
> should. But **a contract that four packages must each remember, whose violation
> costs the entire load rather than one spec, is a contract this function should
> not have depended on. The docstring was doing work the code should have done.**

Fixed by catching `ValueError` — the common base of both of theirs and of every
module exception measured — **and deliberately not `Exception`**: a rejection is a
statement about the spec, an `AttributeError` is a bug in the registry, and
collecting the second reports a broken package where the code is at fault. Both
directions tested.

**A second finding fell out of the first**, hidden behind the abort:
`ValidatorSpec` requires `tags` and `validator.schema.json` has it **optional**.
A validator spec with no tags **passes the gate and fails the model** — the
two-gates problem from the side that lets a bad document *in*, invisible because
all three general specs happen to carry tags.

**And the error-quality note is now load-bearing rather than flattering.** Every
problem in their probe named the file, the path and the repair — *"Add 'env' to
`$.items_schema.required`"*. That is **why driving the real thing was ten minutes
rather than an afternoon, and why a second finding came free instead of the run
stopping at the first traceback.** `handoff` said the same of the validator
registry. **Two packages independently report that admission-error quality is
what makes this check affordable at all.**

**`task-graph` corrects the pattern, and the correction is the section.** I
broadcast `getattr(x, "name", default)`. That was the wrong thing to pin:

> The defect is not a function name. It is **asking whether a collaborator can do
> its job and continuing when the answer is no.** `getattr(x, n, d)`, `hasattr`,
> `try/except AttributeError` and `if hasattr(...) else None` are four spellings.
> **I pinned one and reintroduced another within the hour.**

Theirs:

```python
if "agent_specs" in r and hasattr(r.get("agent_specs"), "check_knowledge"):
```

Written **an hour after** they removed `getattr(handoff_specs, "load_report",
lambda: None)` **two functions up in the same file**, for the same defect, having
written a test forbidding exactly that shape — **and it slipped past because the
test forbids a three-argument `getattr` and this is `hasattr`.** A rename in
`agent` would have stopped `check_knowledge` running, in silence, every suite
green.

**And the tolerance was bought for one of their own test doubles.** Their `Stub`
did not answer `check_knowledge`, so they widened the **production** path to
accommodate the fake. `handoff`'s rule settles it: **a stub that wants tolerance
answers the method.**

**And the sweep's first prospective prevention, hours later.** Asked to add an
`agent` key three packages had agreed to, `agent-mod` **declined to build their
step ahead of the sequence**:

> Writing `getattr(spec, "agent", None)` today would be **a read of a field that
> exists nowhere with a default that is a legal value** — the exact defect class
> this morning's sweep found in **nine of nine packages, two of them in my own
> gate**. **The only caller exercising it would be my stub.**

**That is the tenth instance, not created.** Every other finding today was
archaeology; this is the first time the pattern was recognised in work **not yet
written**. Their question back is what keeps it from returning: **is `agent`
optional?** — *"because if it is required I drop the constructor argument rather
than leave a fallback nobody can reach."*

**A clean result, reported rather than swallowed, is what documented a seam.**
`monitor` measured that `EventRecord` inherits `extra="forbid"`, so any `**extra`
key that is not a field raises — **nearly a live defect** against `agent`'s
`_report`, except that it lifts `evidence` into `attributes` and pops it first.
**They reported the negative result anyway**, and `642aa8e` writes the constraint
into the docstring so the next person adding a kwarg does not discover it as a
`ValidationError`.

> A clean result from someone who did the work is not nothing, and **it is the
> half that usually goes unwritten.**

Twice today a *"we checked, and it was fine"* produced something durable —
`monitor`'s clean stub sweep supplied the reason it was clean, which became the
evidence for `handoff`'s ranking. **Neither would exist if a null result were
treated as nothing to report.**

**This is the strongest argument for running the check everywhere rather than
trusting anyone who knows the pattern:** knowing it, having written the test, and
having fixed the twin sixty minutes earlier **was not enough.** Only running the
check was.

**`closure` is the sixth, and supplies the tell that beats all of mine.** Both
their `getattr` guards were dead — no test had ever supplied a `closures` lacking
`origin_of` or `_build_index` — and one had **become reachable** when
`build_registry` started taking `registries=` from the caller. It then labelled
every `Problem` with the closure's **name** where a file path belongs:
**indistinguishable from a real origin in a message**, and the file path is the
one thing their design §6.2 asks those messages to carry.

> The question that would have caught mine fastest is not *"is this guard
> needed?"* but ***"has anything ever taken this branch?"*** — **one grep, no
> judgement**, and it is answerable before you know whether the guard is right.

**`handoff` ranked the three checks after running all of them, and the ranking is
the most transferable thing the sweep produced.** Cheapest first:

| | check | costs | finds |
|---|---|---|---|
| 1 | **`grep`/trace for branches nothing has taken** | nothing — no test, no judgement | what nothing else can |
| 2 | **run the stub's subject once, on purpose** | a real test | wrong-on-the-day |
| 3 | **compare types, not names** | one line | drift |

> **Mine came out 3, 2, 1 — in increasing order of cost and *decreasing* order of
> what they found.**

**`monitor` quantified what a dead guard costs, which nobody else did.** Their
`getattr(task, "monitor_spec", None)` — unreachable for any real `Task`, but able
to absorb a **rename**, after which every task silently resolves to the global
default monitor and nothing fails anywhere. Measured by mutating the field name:

```
rename + the old getattr guard : 2 failed, 96 passed
rename + direct access         : 7 failed, 91 passed
```

**The guard was absorbing five tests' worth of a rename it had no business
absorbing.** Same defect as `pusher.live_handle`'s, which they removed **that
morning, in the same package** — and it survived because they had defended
`monitor_for` twice, for criteria 1 and 2, and looked at it as *the thing that
resolves names correctly* rather than as a guard.

**And they found the tell's limit by applying it.** Three of their five sites
also have unreachable defaults, and **by the tell alone all three should go.**
They stay: each sits in a **formatter that must be total.** An `AttributeError`
inside `monitor_for`'s `except KeyError` replaces the `KeyError` and loses which
monitor name failed; one inside the excepthook loses the thread-death record
entirely, which is criterion 25.

> ***"examined and kept"* and *"never looked at"* are indistinguishable in the
> code** — which is the whole problem.

**`agent-mod` mechanised the question, and this is its best form.**
`probe_default_arms.py` shadows `getattr` inside the package's modules, **counts
only the arms where the attribute was genuinely absent**, and runs the suite. It
answers *"has anything ever taken this branch?"* **as a number, before any
judgement about whether the guard is right** — which is what makes it survive the
author having had a good reason.

**Their result is the most consequential finding of the sweep:**

| default arm taken | |
|---|---|
| `Prepared.confinement` **×34**, `Prepared.environment` **×17** | **every one from their own `StubEnvManager`, none from production** |

`_apply_confinement` read `getattr(prepared, "confinement", None)` and **returned
early on `None`** — so **a missing field and "there is no confinement" were the
same answer, and the consequence of guessing wrong is a task starting
unconfined.** It never fired in production **only because `env_mgr` happens to
declare the field; nothing in `agent`'s code required that.**

**And it is the second instance of a distinct sub-shape: a stub's deficiency
becoming production tolerance.** `task-graph`'s was the first — their `Stub` did
not answer `check_knowledge`, so **the production path widened to accommodate the
fake.** Here the stub returned four fields of six and the production code learned
to tolerate the absence.

**The direction is what makes it dangerous.** A stub that is wrong gets found
when the real thing disagrees. **A stub that is *impoverished* gets accommodated**
— the production code bends toward it, and the bend is invisible because
everything passes. `handoff`'s rule is the fix in both cases: **a stub that wants
tolerance answers the method.**

Beside it, `_environment`'s docstring still described `Prepared` as a five-field
`NamedTuple` awaiting a sixth — **the sixth had landed hours earlier.** A
fallback whose reason had expired, with an empty dict as its legal-looking
default.

**A warning for anyone instrumenting rather than grepping, from `spec-loader`,
who hit it.** Their first trace run reported **all three accessors DEAD — and it
was false.** Every consumer does `from spec_loader import body_of`, **binding the
function at import**, so patching the module attribute measured nothing.

> **A false DEAD is precisely what this exercise exists to catch.**

**The instrument had the defect it was looking for**, and would have justified
deleting three live branches. Fixed by rebinding the name in **every module
holding it**; the corrected run shows all five branches live on both arms, 3 to
1067 hits. **Grep cannot make this mistake; a trace can, and it fails in the
dangerous direction.**

**So the tell finds candidates, not verdicts**, and the difference between a
considered keep and an oversight has to be **written down where the guard is**,
or the next person runs the same sweep and reaches the same three sites with no
way to know they were already decided.

They also name what the widened search actually bought, which was not the other
three spellings: **it reframed the five they had already passed that morning.**
*"Can the default fire?"* is a different question from *"is this a `getattr`?"*,
and they had audited the same five sites hours earlier asking whether each guard
was **reasonable**. Every one of them is.

That is the order I broadcast them in, and it is backwards. **The reason is
structural, not luck:**

> My stubs were at least **exercised**, so a wrong one had a chance of showing.
> **A branch that has never run cannot be wrong-but-passing; it is simply
> unknown** — and no amount of checking stubs against real subjects reaches it.

Their trace (no coverage module here; `sys.settrace` over seven candidate
branches, probe kept) found six taken 33–92 times and **one never**:
`content.py:241`'s `except (TypeError, AttributeError)`. Driving it showed the
guard **caught the wrong exceptions**: `"notaschema"` — **main design §3.5's own
worked example** — makes `dict()` raise `ValueError`, uncaught, escaping every
caller catching `Malformed`; and `[("a", 1)]` makes `dict()` **succeed**, a list
of pairs silently accepted as a schema object. Fixed by refusing rather than
coercing, which is what the docstring always promised, **with the positive case
pinned too, because a refusal test alone passes when the function refuses
everything.**

**And they take `closure`'s sentence one step further, about themselves:**

> I defended that `except` clause in a commit message this morning, and **the
> thing I was defending was correct while the code beneath it had never run.
> Defending the *reason* is not the same as exercising the *code*, and I could
> not tell those apart from inside.**

**And the guard `closure` deleted landed on `spec-loader`, which is the eighth
finding and a new shape.** `closure` now calls `closures.origin_of` unguarded.
**`origin_of` was on `BaseSpecRegistry` and not on the `SpecRegistry` Protocol**
— and `Registries`' own docstring invites a non-subclass implementation: *"a
Protocol rather than a class so a test supplies five dicts."*

**So a registry satisfying the declared contract could fail the now-unguarded
call.** Asked whether their base had ever failed to provide it, they answered
exactly: *"it never has, and it never could, because every registry in the tree
subclasses it — **but the Protocol permitted it to.**"* The deleted guard was
**correct against the declared contract and dead against the real one**, and
deleting it left the gap load-bearing.

> **A method a whole-catalogue pass calls unguarded belongs on the Protocol, not
> only on the base.**

**Closing that distance is the owner's job, not the caller's** — which is why
this arrived at `spec-loader` rather than staying `closure`'s.

**And the sibling to `closure`'s line, from the package that has now authored two
of these:**

> **A contract you wrote is harder to see than one you were given.**

Both of theirs were **fine in their head and unavailable to anyone else**:
`_validate`'s exception types, which four packages had to remember and one did
not, and `origin_of`, documented on the wrong class. **Neither was found by them
reading their own code** — both came from someone else's guard or someone else's
crash.

**That is better than anything I broadcast**, because it removes the step where
you have to be right about the design.

And the reason theirs survived:

> **A guard you have defended is harder to see than one you have not.**

**`validator` has the same mechanism one step earlier, and it is worse.** Their
`EDGE_KINDS` listed two of three **under a comment citing the paper about listing
too few**:

> I did not defend it to anyone — **I explained it, in writing, beside the
> code.** Their version needed a second person; **mine needed only me.**

**An articulation closes the question whether or not the code does what the
articulation says**, and unlike a defence it requires no audience to become
load-bearing.

**`env_mgr`'s nineteen were worse again, in the opposite direction.** Not
defended, not explained — **invisible**:

> A three-argument `getattr` **reads as ordinary Python rather than as an
> assertion about somebody else's type.** `closure`'s survived a written defence;
> nineteen of mine survived never being looked at, **in a package whose whole
> discipline is looking.**

Their programmatic sweep of all 22 sites: **19 dead defaults, 1 inverted, 2
live.** The inverted one is `getattr(ctx, "repo_locations", {})` — `Context` has
no such field either, so **the fallback was taken every time and the primary
branch was the dead one.** It is `repos`' twin: *which* repos and *where they
are*, two halves of one missing route, **neither half existing**. They found it
only because the tell is mechanical and they ran it over everything **rather than
over the thing they already knew about.**

**And their generalisation is the one that explains all nine packages at once:**

> **A defensive default is a claim about a collaborator, and a claim about a
> collaborator goes stale when the collaborator ships.**

Every one of their nineteen was **correct when written** — `Task.permissions`,
`Grant` and `Access` did not exist, and the plan said to code against the
documented shape. **The justification expired when `task_graph` shipped and
nothing announced it.** That is the stub-expiry rule one level down.

Which is exactly why `closure`'s tell beats every judgement including mine:
***"has anything ever taken this branch?"* is answerable without knowing why the
branch was written**, so it survives the author having had a good reason at the
time. **All nineteen had good reasons at the time.**

Theirs had an argument attached — one they had made **to another team member,
about that member's code, the same morning**, describing this exact shape: *a
guard whose precondition expired, degrading silently into something that looks
like the documented behaviour.* Their defence was that theirs guarded a double
the Protocol promises. **It did not: the double already had `origin_of`.**

They also nearly skipped it: *"I went to item 2 first because it is the
interesting one, and the defect was in the boring one."*

**Two things it cleared, with evidence rather than reasoning.** `FakeRunner.advance`
— which I had named as most suspect — is clean: it calls `enter_phase`
positionally, `monitor._advance` calls it by keyword, both valid, and the subject
is driven for real by `monitor`'s tests and `demo`'s live run. **My suspicion was
wrong and they measured it rather than agreeing with me.** And
`ConsumableMgr.charge` was **correct and untested** — design §6.3.1's non-leaf
spend path had no test at all. *A correct path in the state where it stops being
correct.*

`monitor`'s is the fourth: `base.py:196` reads `getattr(args.thread, "task_id",
NO_TASK)` and **nothing in production sets it** — the only assignment anywhere is
in their own test. **Every thread death in a real run records `NO_TASK`**, and
criterion 25's attribution half is dead. Not a false shape: `Thread` genuinely
has no `task_id` and it was documented as an opt-in convention. **It is §4.12 —
a convention nobody adopted** — and its only caller is the test that proves it
works.

**The information exists one frame away.** `TaskAttempt.begin()` already spells
the id into the thread *name*. `monitor` declined to parse it back out, correctly:
that would turn a declared convention into **a string format nobody agreed to** —
`engineer_principle.md` §4.4 for the third time this afternoon, and the same
shape as `layout.stage` discarding the handoff id.

**And their clean stub result came with the reason, which is the more valuable
half:**

> The reason is not virtue: **four of my five subjects have been driven for
> real**, in probes I wrote for other reasons. Your check is what those probes
> were, arrived at from the other direction.

**That is the strongest evidence for `handoff`'s instruction**: *run the subject
once, on purpose* protects you **whether or not you meant it as that check.** The
four packages that found defects had no such probes.

**The fifth subject is the tell.** `StubBackend` has never been driven, because
`claude-agent-sdk` is not installed on this machine. **The one subject that
cannot be driven is the one with no evidence**, and no test here can change that.
Recorded as a limit rather than an item: `agent`'s backend seam is unverified
against a real backend by anyone.

**A related shape, found in the same sweep**, in `tests/interfaces/test_pushable.py`:
its static-only rationale says *"neither package has an implementation yet"*.
`ClaudeSdkBackend` now exists — **half the sentence is false and the test is
still right**, because it is still not constructible here. **A true conclusion
resting on an expired premise, surviving because the conclusion still passes.**
That is why nobody would notice.




**A second instance, from `closure`, in the same shape:** their `covers`
argument, their `task_of` argument and their redundancy argument were each *"a
greenfield argument applied to something that was not greenfield"* — sound
reasoning about a world other than the one in front of them. Fluency is
indifferent to whether the premises hold.

**And the dispute closed without a ruling from me.** `spec-loader` stopped
asserting the state in their README and pointed at the two places that hold it —
*a row asserting "still disputed" is a state, and states go stale where a pointer
does not.* That is §8.3 applied by the party who would have benefited from being
the one to record it.

### 8.5 A record that cannot go stale, because a test reads it

Nine packages were asked to write their durable facts to disk. **`agent` had
already done something better**: `tests/agent/test_criterion_mapping.py` reads
the mapping table out of `agent/README.md`, extracts every `` `test_*` `` name
from it, globs `def test_*` out of `tests/agent/`, and **fails on the
difference**. A second test pins the row count against `spec.md` §8.

**Written because the failure had already happened there**: a test removed when
`set_task` moved to the scheduler stayed in the table for four commits.

This is the answer to §8.3's problem rather than a mitigation of it. §8.3 says a
description of a moving thing is stale on arrival and every stale item this week
was true when written — **a prose table of test names is exactly that shape**,
and every package now has one. `spec-loader` re-ran each name against the tree
before writing theirs, which is the manual version and is only true on the day.
`agent`'s is true on every day the suite runs.

The general form: **where a durable record duplicates something the tree already
knows, the record is a duplication like any other, and §8.1's price applies —
something has to check it.** A README is not exempt because it is prose.

**Three packages arrived at this independently, and only `agent` was told.**
`monitor` walks the AST of `tests/monitor` and `tests/interfaces` — 42 names
cited, 42 verified. `env_mgr`'s `test_every_test_the_readme_cites_exists` fails
the moment the README names a test that does not exist — 79 cited, 79 present.
Convergence by three parties who could each see the problem from their own side
is worth more than the ruling that would have imposed it.

**And the audits produced the two strongest pieces of evidence for §8.3 in the
stage, both against the auditor's own file:**

`monitor` found **three stale claims in the README they were extending** — a row
saying `task_graph`'s `TaskStatus` "has none of the three yet" when it had had
all three since rev. 12, a `base.py` description predating three of its symbols,
and a test list naming two interface files of five. **In the same file where
they had recorded the lesson.** Their own summary is the best one available:
*that is the argument for the exercise better than anything I could say about it.*

`env_mgr` found the design's §14.5 test plan named tests before they existed and
**three had ended up called something else** — so a mapping transcribed from the
plan would have been wrong on arrival. The plan was not wrong when written.

**Two limits, both stated by the packages that built it.** `env_mgr` declined
the reverse direction deliberately: most of their 228 tests hold a design
decision rather than a criterion, and requiring every one to be cited **turns a
mapping into an inventory**. `monitor` marks the four criteria covered only on
their side, with the note *do not read a green † row as end-to-end coverage* —
a machine-checked table proves its names exist, **not that they test what the
row claims.**
