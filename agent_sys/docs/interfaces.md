# agent_sys — the module interface contract

| | |
|---|---|
| Status | **Normative for what crosses a module boundary**, and for nothing else |
| Version | 6 |
| Updated | 2026-09-15 |
| Summary | What crosses a module boundary: the composition root, the shared vocabulary, and each package's exports, permitted imports and run-time resolutions. |
| Companion | `spec_loader/`, `handoff/`, `validator/`, `agent/`, `closure/`, `env_mgr/`, `monitor/` — each a `protocols.py` plus its `.pyi`, the same contract importable and type-checkable |

---

## 1. What this file is for, and how binding it is

Eight design documents exist and every one of them specifies interfaces. What
none of them can specify is **the other side of its own seams**, and the cost
of leaving that unwritten is concrete: four documents each writing a different
part of one composition root leaves five of the eight modules unwireable as
written, with components resolved by name and registered by nobody.

So this file exists to hold exactly what no single module can hold:

- **§2 — the composition root.** One listing. Every other document defers to it.
- **§3 — the shared vocabulary.** The types more than one module names.
- **§4 — the seams**, module by module: what each exports, what it may import,
  and what it resolves at run time.
- **§7 — the dependency declaration.**
- **§8 — the importable half**, and what it must agree with.

**Open questions do not live here.** A seam this file leaves undecided is a
near-term decision ([`TODO.md`](TODO.md)) or a subsystem
([`ROADMAP.md`](ROADMAP.md)). This file states what *is* agreed; the two lists
state what is not, and there is no third place.

### 1.1 This contract is reportable, not sacred

**If implementing something shows an interface here to be wrong, say so and
propose the change.** That is not a failure of the contract; it is the contract
working. A design document is a prediction about code that does not exist yet,
and this one is a prediction about eight of them at once.

What is asked instead is narrow: **do not change a cross-module signature
quietly.** A seam has two sides and only one of them is in front of you. Raise
it, name both sides, and let it be changed in one place — which is how the
fifteen contradictions that pass turned up would each have been caught
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
| `monitor:<name>` | `Monitor` | `monitor` | `Task.monitor_spec`, by name. Absent takes the default. **Rev. 4: a name that will not resolve is no longer merely an unwatched task — it is a task that never advances a phase** (`monitor` spec §5.3). **Who calls `set_task` is held by §2.7 and by the code. This row does not restate it** — it has carried four different answers, and the last two were wrong while reading as settled, one of them costing `task_graph` a build. A row asserting a position is a state; states go stale where a pointer does not. This row sanctioned the edge from the beginning and nothing had ever traversed it |
| `budget` | `Budget` | `monitor` | `agent.Runner`, at the completeness gate. One global value: nobody yet knows what a normal task costs, so a per-task limit would be authored out of numbers no one has |
| `validator_executor` | `Executor` | `agent` | `validator.phase` resolves it for an agent-bodied validator. **Registered by nobody — added to this table rev. 5 so integration reads it rather than discovers it.** It is C1's third instance and the least bad one: `validator` raises naming the component and citing `agent` O6, so it fails loudly. Whoever closes O6 registers the name |
| `scheduler` | `Scheduler` | `task_graph` | every `Task` transition, via `_registry` |
| `agent_spec:subgraph` | a **name**, not a spec | `task_graph` | `AgentMgr.is_registered` at `Scheduler.submit`, and `instantiate` at dispatch. **Added by the UI stage, and it is a name nobody authored.** A non-leaf no longer carries an `agent` (main spec §4.8, leaf-only), but `submit` gates every task on the name resolving and `_dispatch_pass` writes `agent.id` into a required `Execution.agent_id` — so removing the field would have reached persistence. `SUBGRAPH_AGENT_SPEC = "subgraph"` is registered unconditionally by `_bridge_agent_specs`, **before** the `agent_specs` loop, because a graph can be built with no agent specs admitted at all and a non-leaf in one still has to pass the gate. **No document is invented**: `AgentMgr`'s table is `dict[str, dict]` and the one reader that would need a real spec, `runner.agent_spec_of`, is unreachable for a non-leaf — `_main` releases and returns at `agent/runner.py`, before `_deploy`. An authored spec of the same name warns rather than colliding |

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

**The site is `Scheduler._dispatch_pass`, immediately before `runner.start`.**

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

**`FakeRunner` is not a counterexample to that.** `advance()` calls
`task.enter_phase` directly and never routes through a monitor, so it escapes
the obligation by not having the collaborator at all — and **a double that does
not participate is not evidence that a real implementation should not be
obliged**.

It also means `tests/task_graph`, the dry-run and the G3 gate do **not** hit
`ScopeViolation` on the first advance: nothing in `task_graph` raises, because
`FakeRunner.advance` bypasses the monitor. The `ScopeViolation` appears only
where `monitor._advance` is the route, which is the CLI with a loop running.

**What the site buys, precisely:** *the watch set is populated wherever a task is
dispatched, whatever runner is installed* — **not** that the planned channel is
exercised end to end, which no current configuration does.

`task_graph/docs/design.md`'s *"Nothing in this module runs a monitor"* holds:
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
  catalogue.** A `skip=` origin/name mismatch, or `check_graph` reading the
  wrong shape, is invisible in exactly this way.
- **A skipped *effect* pass leaves a system that fails later and elsewhere.** An
  `agent_mgr` left empty after a successful load surfaces as `unknown agent spec
  'collect'` at submit — **three layers from its cause, and the signature that
  gets reported as somebody else's bug.**

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

`put` raises and names the wiring rather than publishing something
half-checked. **Loud and unwired beats quiet and wrong**, and the composition
root meets it on day one, which is the point.

**Check the names in this listing against the shipped code before using them** —
`engineer_principle.md` §5.2. `KindSource` is a Protocol with `kind_for`; the
registry accessor is `kind_of`; there is no `type_of`.

**All three fail loudly.** A `KindSource`
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
who builds the root `Task` is `cli`'s, knowingly — its design records the deviation.

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

**`runner` is added for `validator` spec §8.2's `producer` row, and `agent` built the field it
reads.** `validator` spec §8.2 gives output validation *"the producer's — the
task that just ran"* configuration, which would otherwise be a discarded local
of `_deploy`. It is `TaskAttempt.environment`, a read-only
`Mapping[str, str]`, reached as `attempt_of(task.id).environment`. **A resolve,
not an import** — `test_import_rules.py`'s `ALLOWED` is unchanged and `validator`
still may not import `agent`.

**Two facts a reader of that row needs.** The component registered as `runner` is
**not one protocol**: `task_graph/bootstrap.py` registers the shipped
`FakeRunner`, which has `start` and `stop` and no attempts, so the capability is
checked rather than assumed. And **`TaskAttempt.environment` is `{}` until
`_deploy`** — which for a **non-leaf** is forever, since the scheduler runs its
main phase by unfolding — so `validator` reads empty as *absent* and falls
through to the global row. §8.2's row is *the configuration already resolved*,
and a task that resolved none has not got one.

**`consumer` stays unreachable, and in principle rather than for want of a
field.** `env.prepare` has one call site, `agent/runner.py`, inside
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

**Without `closures` a closure's declared phase validators never run.** A
`PhaseRunner._select` that builds a phase's set from the **handoff kind's** list
leaves the closure's `validators` list read by nothing in the tree.
`closure.schema.json` says why the kind cannot carry them:

> *"They are a property of the task rather than of any one handoff kind, **which
> is why the handoff specs cannot carry them**."*

So a closure declaring `validators: ['check_grounded']` ran **nothing**, and a
handoff kind declaring one ran it in both phases of every task touching that
kind. Two behaviours; the specs describe the first.

**The fix is not "read the closure's list as well" — it is "ask `closures` for
the set and stop deriving it here."** `closure/query.py` already computes the
union and `closure` spec §3 states it: *"Every validator that will run, phase
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
The wide reading blocks `examples/ok.filetree_grounded_report.4/closures/main.jsonnet` — `outputs: []`,
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
| **Resolves** | `agent_specs`, `task_specs`, `env_mgr`, `phase_runner`, `handoff_store`, `budget`, and **`handoff_mgr`** — the last because the runner is the agent-facing write path's only production caller. `task_specs` is §5.1b's route — `task.closure` → the task spec |

**`TaskAttempt`, `Runner.attempt_of` and `Runner.carry_on` are declared — rev. 5,
and this paragraph is the fifth §4 row to have trailed a settled decision.**
Measured rather than restated:

```
agent/protocols.py   class TaskAttempt(Protocol)
agent/protocols.py   class Runner(Protocol)
agent/protocols.py       def attempt_of(task_id) -> TaskAttempt | None
agent/protocols.py       def carry_on(task_id) -> str
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
the **edge kind** — `closure:collect_trace` beside `kind:trace` — where the
earlier plan was to union two answers at the composition root. **The union never carried
*how* a validator is reached, and "how" is what *what breaks if I change this*
actually wants.** Two indexes over one fact was the shape this revision spent the
day removing.

**The enumeration question closes with it**: one owner and one
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
| **Exports** | **`EnvManager`** — `prepare` (**checks; no longer confines** — `spawn` applies), `prepare_validation`, `place_zone`; `Prepared` (**eleven** fields, plus **`wrap_argv`** and **`spawn`** — §5.15), **`ValidationZone`**, **`Zone`**, **`Availability`**, `Zone`, `Confinement`, `Policy`, `Granted`, `Mode`, `Context`; `contained`; `NoConfinement`, `UnresolvedGrant`, `PrepareRefused` |
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

**`Prepared.tools` is the eleventh field — added for change (C), and it is the
route criterion 18 never had.** `env_mgr/remote/tools.py` has defined
`env_remote_run` / `_push` / `_pull` since it was written and **no agent could
reach any of them**: `Prepared` had no field to carry a `ToolDef` and
`Assignment` had no field to receive one, so spec §5.5's *"the whole remote↔local
surface is exposed to agents as tool calls"* had no implementation on the
delivery side. §4.12's family, and the missing half was invisible from
`env_mgr`'s side because a tool surface with no consumer looks exactly like a
tool surface.

**It is a field and not a third `EnvManager` method**, on this row's own
precedent: `wrap_argv` is on the returned value for the same reason, and
`test_env_manager_exposes_exactly_these` pins the component's method set at two
so a third is a decision rather than a drift.

**`tools()` gained a `remote_root` parameter in the same change, and that was a
defect rather than a widening.** It took `(conn, zone)` and passed `zone.root` —
a *local* path — as the working directory of a command run on another machine.
The only configuration in which that appears to work is a **strong** mapping,
where the two paths are equal by definition, which is the worst way for a defect
to hide.

**`agent_spec` is the third parameter — added rev. 5, and without it four spec
keys have no consumer.** `agent` spec §3.1's `env` and `agent` design §3.4's
`rules` / `hooks` / `skills` all name this module as the thing that reads them,
and `prepare` had no parameter they could arrive through. That is exactly the
mechanical check this file exists for — *a document says X
consumes Y; check whether X's signature can accept Y* — and it failed. The
default keeps every existing two-argument call working.

**`Prepared` carries the deployed environment, and the field is functional
rather than tidy.** `material.deploy` computes `CLAUDE_CONFIG_DIR`,
`CLAUDE_CODE_TMPDIR` and the spec's own `env`; without a field for them `prepare`
drops them and the runner cannot see them. Measured, and the reason this is not
cosmetic: **with `~/.claude` granted, a confined agent reads the *operator's
personal* `CLAUDE.md` and obeys its language rule.** Pointing
`CLAUDE_CONFIG_DIR` into the zone is what removes the `$HOME` grant entirely. The
rejected alternative was `agent` calling `material.deploy` itself, which puts an
environment decision in the runner — the thing *one method, and it stays one*
exists to prevent.

**One caller obligation, because it fails silently.** On the bubblewrap rung
`apply()` confines nothing: **bwrap *is* the exec**, so whoever starts the
executor must wrap the command line — **`prepared.wrap_argv(argv)`**, above. Skip
it and `prepare` succeeded, the task ran, and there was no sandbox. `bwrap` is
absent on the development machine, so nothing local catches it, and
[`ROADMAP.md`](ROADMAP.md) §6.1 is the half of this that is still undecided.

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

**The suite stays green across these.** Three things changed inside the package
and each is small: `Grant.handoff: HandoffId` became `Grant.kind: str`, the
cascade puts its reason in the report rather than on the task, and
`resume_system` rebuilds `OrderedIdSet` pools.

**The scheduler never names a spec registry**, and `test_authority.py` enforces
it. The narrower rule that actually holds: *the scheduler never reads a spec; a
`Task` transition may read the catalogue it came from.* `unfold` and
`replace_with` resolve `closures`, and that adds no scheduler edge.

### 4.8 `cli`

| | |
|---|---|
| **Exports** | nothing. **Nothing may import `cli`**, and `tests/interfaces/test_import_rules.py::test_nothing_imports_the_cli` walks every component package's AST to prove it |
| **Imports** | anything of ours. It is the composition root, so it is the one package with no restriction |
| **Resolves** | via `build_registry` |

`cli/build.py` holds `root_task`, `handoff_ids` and `wire`. `closure` declined
them because a helper returning a `Task` would make it import `task_graph`; `cli`
took them knowingly and its design records the deviation.

**The rule is keyed on the package name, so the name is itself checked.** A
name in this row that no directory carries degrades the enforcement silently
rather than failing it: the test intersects a file's imports with a set of our
package names, and a set holding a name no package has can never contain it, so
the assertion passes against every possible tree.
`test_every_allowed_package_exists` is the cheap guard against the class — a rule
whose subject does not exist checks nothing.

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

## 7. The dependency declaration

**Landed** — `handoff` owns the block and all eight entries are in
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
