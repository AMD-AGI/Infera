# `demo` — the runnable proof that the components compose

| | |
|---|---|
| Implements | [`docs/spec.md`](docs/spec.md) rev. 7 — 17 acceptance criteria; [`docs/design.md`](docs/design.md) rev. 4 |
| Seam | [`../docs/interfaces.md`](../docs/interfaces.md) §4.8 — **imports all eight, and nothing imports it** |
| Tests | `../tests/cli/` — none makes a model call, needs a credential, or needs a sandbox |

```
demo/                    the RUNNER — installed, and where [project.scripts] points
examples/demo/           the TASK PACKAGE — YAML and data, not installed, imported by nobody
examples/demo-broken/    a SECOND package, deliberately broken. Only --with-broken loads it
```

Two artefacts, and **the split is the enforcement** of spec §1.1's rule that the
demo may use nothing an out-of-repository task package could not use. The moment
`examples/demo/` holds an `__init__.py` it is importable and the example stops
looking like what an out-of-repository package looks like;
`test_examples_has_no_init` is what notices.

### The package moved on 29 Aug, and every path below it did

jsonnet is deleted (`docs/ui-stage.md`). The findings in this file were written
against the old tree and are **left as they were measured** — a log that is
retrofitted stops being evidence — so this is the map:

| Was | Is |
|---|---|
| `closures/<n>.jsonnet`, `handoffs/`, `validators/`, `agents/` | `main.yaml`, `shared.yaml`, `steps/<step>.yaml` — several objects per file, discriminated by `module:` |
| `lib/demo.libsonnet` | gone. Three helpers: two were body paths the assets convention now finds, one was `path()` and was already unused |
| `bodies/<task>/{readme.md,entry.sh}` | `assets/<task>.task/{readme.md,entry.sh}`, found by convention |
| `logic/<validator>/{readme.md,entry.sh,check.py}` | `assets/<validator>.validator/...`, likewise |
| `logic/store.py` | `assets/lib/store.py` |
| `bin/collect.py`, `bin/render.py` | `assets/produce.task/collect.py`, `assets/consume.task/render.py` — beside the body that execs them |
| `agents/compose.jsonnet` | **deleted.** A non-leaf declares no agent; `task_graph` supplies `SUBGRAPH_AGENT_SPEC` |
| `broken/closures/dangling.jsonnet` | `../demo-broken/main.yaml` — a sibling *package*, because the YAML scan reaches every `*.yaml` under a root |
| the `config` fill: `package_root`, `store_root`, `outside` | one variable, `${outside}` |

**The run is byte-for-byte the same afterwards — measured, not argued.** The
pre-stage tree (`8274a5b`, jsonnet) was checked out into a temporary worktree
and run on this machine with Landlock available. Normalising paths, UUIDs and
elapsed times, the two transcripts differ by **one line**:

```
-    closure  main: agent 'compose', 0 in, 0 out
+    closure  main: agent None (a non-leaf declares none), 0 in, 0 out
```

Everything else is identical, including the exit code (4), the `1 of 2 expected
failures observed, 1 never reached` accounting, and the
`usage names 'seconds' … is not booked` warning from `task_graph/scheduler.py:616`
— which is therefore pre-existing too, and not something the conversion
introduced.

**So the demo stops in the same place it stopped before**, and that place is
`describe`: `ConfinementNotApplied`, ROADMAP §6.1 P0 / `interfaces.md` §5.11. The
conversion is not what makes criterion 8 unreachable and never was.

```bash
pip install -e agent_sys
agent-sys show                     # the graph, nothing dispatched
agent-sys run --dry-run            # + every load-time check. This is what CI runs
agent-sys run --dry-run --with-broken   # exit 1, naming the offending file
agent-sys run                      # the whole thing. Needs credentials and a sandbox
agent-sys run --resume             # continue the last run
agent-sys run --clean              # remove every run and exit

agent-sys show --package agent_sys/examples/demo2 --var n_problems=2
```

### `--package` and `--var` — the two things that make this generic

`cli/` is the program's **single entry point over any task package**, not one
package's runner. Two flags carry that, and both are on `show` and on `run`:

| | |
|---|---|
| `--package DIR` | the package to load. No path is privileged; `cli/package.py` resolves this one, or falls back to the checkout |
| `--var K=V` | repeatable. Sets a package variable, expanded by `spec_loader.YamlPackage` as `${K}` or `${K:-default}` |

`--var` replaced a hardcoded `outside=` keyword, which was the entire variable
channel: a package could declare `${n_problems:-12}` and had no way to be told
otherwise. `examples/demo2` declares three such knobs, so a cheap bring-up run
of it is `--var n_problems=2 --var n_extra=2` and the full run is the default.

**`outside` is refused from `--var`, by name.** It is per-run and absolute and
only the CLI knows it — `Layout.outside`, created moments before the registry —
so the CLI's value has to win. Given that, a user who passes it has two possible
outcomes and only one is defensible: dropping it quietly leaves them a flag they
typed, no error, and a run that ignored them. It is an argument error instead,
naming the variable and why. A malformed `--var` (no `=`) is likewise an
argument error naming the token. Both exit 2, which is what argparse already
does for an unknown flag.

`show` passes no `outside` at all — there is no `Layout` on that path — so a
package's `${outside:-...}` renders its visibly-unfilled default, and
`show --var K=V` is the cheap way to check that a value reaches a spec without
preparing a run.

### `cli/expectations.py` — the one package-specific thing left, and it is a leak

`build.py`, `environment.py`, `package.py` and `render/` name no closure, no
handoff kind and no validator. The **expected-failure accounting** did: what
`examples/demo` promises will go wrong is `consume`, `check_grounded` and
`summary`, and those three words were in `cli/main.py`.

They are now in `cli/expectations.py`, a table keyed by **package directory
name** and defaulting to `EMPTY`. That is still a leak — a package name in the
CLI — and the module docstring says so and says what a future fix looks like: a
`promises:` block in the package's own `main.yaml`, which is a new schema and a
change on both sides of a module seam, so it is out of scope here.

**An empty set is a statement, not a gap.** It says *this package promises
nothing will fail*, which is `examples/demo2`'s whole claim, and it exits `OK`
rather than `UNEXPECTED_SUCCESS`. The run's own report keeps the two apart —
*"promises no failure, so nothing here was tested for one"* rather than
*"0 of 0 expected failures observed"* — because `ok: true` is the same byte for
*no promise was made* and *every promise was kept*, and the second is a much
larger claim than a zero supports. `tests/cli/test_expectations.py`.

---

## Libraries adopted, and why

Mission rule 5: *prefer a mature, widely adopted library or CLI over writing it
yourself.* **Nothing new is added to `pyproject.toml`'s dependency block.** The
demo is the one artefact whose install must be boring — a dependency bought here
is a dependency a reviewer installs before finding out whether anything works.

| Concern | Considered | Chosen | Why |
|---|---|---|---|
| CLI parsing | `click`, `typer`, `argparse` | **`argparse`** (stdlib) | Two verbs and seven flags. `click` and `typer` are undeclared dependencies bought for nothing, and neither expresses anything `argparse` cannot at this size |
| The event stream | `logging`, `structlog`, own | **own, ~120 lines** | `logging`'s fan-out to handlers is the right *shape* and its record is the wrong one: it carries a formatted string plus arbitrary `extra`, and criterion 14 needs a **closed** set of kinds and a versioned payload. Enforcing that on top of `logging` is more code than `Event` plus `Stream` |
| The stream's shape | invented | **Terraform's, adopted whole** | A closed `MessageType` enum, start/complete per operation, and `JSON_UI_VERSION` with a comment obliging a bump. Three conventions that are free now and unaffordable later. **One stream rendered twice, not two writers** — a demo whose narration and whose JSON disagree fails at the one job it has |
| The machine format | one JSON document, JSON Lines, `--junitxml` | **JSON Lines** | Criterion 12 interrupts a run with `os._exit`, and a single top-level document would be truncated and unparseable. Flushed per event, because `os._exit` runs no `atexit` |
| The expected-failure vocabulary | invented | **pytest's, adopted** | `xfail` / `xpass` / `strict` already name exactly the three cases §7.5 needs, and every Python reviewer knows them. Verified first-hand: `xfail(strict=True)` that **passes** is `FAILED` |
| The CI guard | run the demo, run nothing, **load** the demo | **load it — Airflow's shape** | The only option satisfying both spec §1 (*"the first thing to break when one of them drifts"*) and spec §5 (*"CI does not run it"*). Airflow's example tests are parse-only; `--dry-run` is our parse |
| The store | `MemoryStoreMgr`, `JsonFileStoreMgr` | **`JsonFileStoreMgr`** | Criterion 12 needs a store that survives a process. Its records also stay readable with `cat`, which is a demo virtue |
| The task's content | a toy, a benchmark, **a file manifest** | **a file manifest** | Verifiable by recomputation, deterministic, needs no GPU or network, and has enough internal structure for a summary to get wrong |
| Zone naming, confinement, grants, workspaces | own | **`env_mgr`** | Every mechanism is theirs. What this package owns is the `Context` — and that turned out to be the surprising part (§9 below) |
| Digests, README checks, locality, publication | own | **`handoff`** | `put` is the commit token, and the demo's fixtures go through it rather than around it |

Three files hold real content — `build.py`, `events.py`, `environment.py` — and
each exists because §6, §7 and §9 are decisions nobody else made, not because
something was missing.

---

## Where the criteria are tested

All sixteen, and every named test exists and passes.

| # | Criterion | Test | File |
|---|---|---|---|
| 1 | Install, then run, under a minute excluding model latency | `test_overhead_budget` | `test_package_loads.py` |
| 2 | A root with `parent = None`, a subtask with a non-`None` parent | `test_root_and_subtasks`, `test_a_flat_graph_would_not_have_proved_this` | `test_build.py` |
| 3 | One input phase empty, another populated | `test_produce_input_phase_empty`, `test_describe_input_phase_runs`, `test_the_same_handoff_slot_is_on_both_sides` | `test_build.py` |
| 4 | One dispatch per task; no validator in any pool | `test_one_dispatch_per_task`, `test_no_validator_in_any_pool` | `test_events.py` |
| 5 | A failing verdict, the consumer left `WAITING_HANDOFF`, written by the phase | `test_grounded_fails_and_consume_waits`, `test_verdict_author_is_the_phase` | `test_events.py` |
| 6 | No credentials → a clear message, no fake fallback | `test_missing_credentials_message`, `test_a_missing_backend_is_also_a_precondition`, `test_credentials_failure_exits_two_not_one`, `test_no_fake_backend_exists` | `test_package_loads.py` |
| 7 | One SDK node, one program node, handoff state the same in kind | `test_one_node_is_a_program_and_one_is_an_sdk_agent`, `test_program_and_sdk_handoff_state_identical` | `test_build.py` |
| 8 | A scripted out-of-zone write blocked, reported as an expected event naming the mechanism | `test_access_denied_event_shape`, `test_the_leak_target_resolves_to_a_real_directory_outside_every_zone`, `test_an_unfilled_leak_target_is_visible_and_not_the_filesystem_root`, `test_the_outside_directory_is_outside_every_zone` | `test_isolation_shown.py` |
| 9 | No sandbox → refuses to start and says so | `test_no_confinement_refuses_with_exit_2`, `test_the_refusal_message_names_the_stakes`, `test_the_chain_orders_bubblewrap_first` | `test_isolation_shown.py` |
| 10 | Every verdict carries dimension and strength | `test_machine_verdict_carries_taxonomy`, `test_human_verdict_line_carries_taxonomy` | `test_events.py` |
| 11 | `--dry-run` resolves everything and dispatches nothing; a broken closure names its file | `test_dry_run_dispatches_nothing`, `test_broken_closure_names_its_file`, `test_the_broken_directory_is_not_in_the_ordinary_pass` | `test_package_loads.py` |
| 12 | Interrupt and restart continues from persisted state | `test_resume_continues_from_disk` — two processes, `os._exit(9)`, `attempts=[SUSPENDED, open]` | `test_build.py` |
| 13 | Running twice succeeds without hand-editing | `test_two_runs_do_not_collide` (in both files) | `test_package_loads.py`, `test_build.py` |
| 14 | The machine output suffices for criteria 2–10 without parsing prose | `test_machine_output_answers_criteria_2_to_10` — reads `fields` only, never `message` | `test_events.py` |
| 15 | The components import nothing from the demo | `test_no_component_imports_demo` (AST, per package), `test_no_component_names_the_token_at_all`, `test_examples_has_no_init`, `test_the_examples_tree_is_not_installed`, `test_the_package_imports_no_component` | `test_package_loads.py` |
| 16 | Loads through the ordinary task-package path | `test_loads_as_an_ordinary_package`, `test_package_declares_no_schema`, `test_every_spec_renders_without_the_config_fill` | `test_package_loads.py` |

**The most important test in the directory belongs to no criterion.**
`test_expected_failure_that_passes_fails_the_run` asserts that an expected
failure which *does not happen* exits **3** and emits `UNEXPECTED_SUCCESS`. A
demo that prints "all good" because the sandbox stopped blocking, or because the
validator stopped failing, is the single worst outcome available to this
artefact — it would assert, in the most visible place in the repository, that a
safety property holds when it does not.

---

## Seams found by assembling all eight, with both sides named

`docs/interfaces.md` §1.1: *do not change a cross-module signature quietly.*
**None of these was changed here.** Each was measured on the shipped tree and
reported to the packages on both sides.

### F-D1 — `HandoffStore.put` has no caller, so nothing publishes a handoff

**This is the one that blocks the `run` verb**, and it is the widest thing found.

```
$ grep -rn "\.put(" --include="*.py" agent/ validator/ env_mgr/ task_graph/ monitor/ closure/
agent/backend.py:271,315,358      # queue.Queue.put
```

Three `HandoffStore.put` call sites in the tree, all in `tests/handoff`. **No
component publishes a task's output handoff.** The thing that notices is
`agent/gate.py::run_gate`, which reports `OUTPUT_ABSENT` when
`store.exists(hid)` is false — so in an assembled run every task's completeness
gate fails, for every task.

| Side | |
|---|---|
| `handoff` §4.2 | `put` **is the commit token, not `rename`** — if rename were the interface, an object-store backend would have nothing to implement. So publication goes through it |
| `agent` `TaskAttempt._main` → `_gate` | The executor runs and the gate then asks whether the declared outputs are there. Nothing between those two steps writes them |

It is `interfaces.md` §5.8's shape one level up: §5.8 records that
`handoff.resolve` ships with no in-process consumer, and this is the same
sentence about `put`. Three routes exist — the executor publishes what the body
left in the zone; the body publishes for itself; or a step nobody has named —
and it is not the demo's to choose. Reported to `handoff` and to `agent`.

**What is unblocked:** the whole load-time half, `show`, `--dry-run`, the event
stream, `build.py`, and 13 of the 16 criteria. Criteria 5, 7 and 8 have their
event-shape half tested here and their run-time half waiting on this and on
F-D10.

**F-D1 and F-D8 are `xfail(strict=True)` in `test_package_loads.py`**, which is
`interfaces.md` §8.1's handshake used as a protocol: each lands green today and
goes red the instant the other side lands, naming both sides. Nobody has to
remember to come back.

### F-D1a — the two version allocators, and the reference between them

`handoff` found this by looking at F-D1 and it is not in the three routes above,
so it is recorded here before anyone finishes the run path.

**`put` returns a *store* version; `Handoff.open_next` allocates a *slot*
version. They are independent counters and nothing forces them to agree.** Both
start at 0, so a graph that publishes once per handoff sees `0 == 0` and looks
consistent — right up until a task re-runs, at which point the slot has v1 and
the store may not, or the other way round.

`HandoffVersion.seal(status, content=...)` is the only field that could carry
the reference between them, and **what goes in it has no owner** —
`interfaces.md` §5.12. So route (1) above is not *"call `put` at the end of the
main phase"*; it is *"`put`, then `seal` the slot with a reference to the store
version"*, and the second half is as unowned as the first.

**Nothing here relies on the two agreeing, deliberately.** `cli._report` takes
the slot version from `Handoff.latest.version` and the store versions from
`store.list_versions(hid)`, separately, and never equates or derives one from the
other — not in an assertion and not in a display string. A shape that ships in a
demo is very hard to change later, and this one is a seam three packages left
open on purpose.

### F-D2 — `agent_mgr` was populated by nobody. **Closed in the root**

`task_graph.AgentMgr.register` is the table `scheduler.submit` checks with
`is_registered`, and `build_registry` never called it: `AgentSpecRegistry` (the
admitted documents, `agent`'s) and `AgentMgr` (the instantiation table,
`task_graph`'s) were two tables with nothing between them. Measured — `submit`
raised `unknown agent spec 'collect'; registered: []` for a graph whose spec was
admitted and sitting in `agent_specs`. **`agent` design §15 O5 predicted exactly
this** and recorded that the direction was stated in no spec.

`cli/build.py` bridged it as a stopgap and said in the code that it might be in
the wrong place; `task_graph` ruled it the root's, landed it, and the bridge here
is deleted. The argument that decided it generalises to anything else this
package accumulates:

> **Is this fact about *this graph*, or about the catalogue?** A registered spec
> does not vary per graph and every graph needs it identically — registry state.
> A `Task` is graph state. And a composition root that returns an object which
> cannot dispatch anything has not finished composing.

One half remains and is a property rather than a defect: **`--resume` must
reload the task package.** A resumed registry rebuilds the five spec registries
empty, and that is intended — a spec table is configuration, not state, which is
`AgentMgr.resume_system`'s own argument one level up.
`test_resume_continues_from_disk` pins it.

### F-D3 — `Assignment.entry` was run from the zone. **Fixed by `agent`**

`agent/backends/program.py::_deploy` ran `subprocess.Popen(["/bin/sh", entry],
cwd=self.assignment.zone)` on a path `_common.schema.json` types as *"Package-
relative"*, and nothing carried the package root into `agent` — while
`validator.ScriptBodyRunner` took the same input and joined it onto a declared
`package_root`. **Two consumers of one schema key, disagreeing.**

`agent` landed `Runner(package_root=)` and now resolves it the same way. **The
demo keeps rendering its body paths absolute** through the standard `config`
fill, and that is a choice rather than leftover scaffolding: `build_registry`
constructs `PhaseRunner` with no `package_root`, so it defaults to `Path.cwd()`,
and an absolute path is the only form correct for both consumers without the
root wiring a third thing. The fill is main spec §4.4's interface for every
package, so using it privileges nothing.

### F-D4 — a closure's `validators` list has no runtime consumer

`validator.PhaseRunner._select` builds the phase's set from the **handoff
kind's** own `validators`. `closure.phase_validators` /
`ClosureRegistry.validators_for` are read by `closure/query.py` and
`closure/check.py` and by nothing else in the tree.

`closure.schema.json` says, of that key: *"The PHASE validators — the checks that
run in this task's input_validation and output_validation phases … They are a
property of the task rather than of any one handoff kind, **which is why the
handoff specs cannot carry them**."* The code does the opposite of that
sentence. Reported to `validator`.

**The demo declares both, honestly**, and they agree — so the demo behaves
correctly and `cli._validators` renders what will *actually* happen rather than
what the specs say should.

`validator` confirmed it and found **a second half in their own registry**:
`EDGE_KINDS` listed two of `interfaces.md` §5.4's three, with a docstring
directly above it citing Airflow #58058 — false-positive deadness from an
unenumerated reference kind — as the reason to keep the list in one place. So
`users_of` reported a validator that two closures run as used by nothing. Fixed
there. `_select` is untouched and cannot be touched from inside `validator`:
reading the closure's list needs `closures` resolved at call time and §4.3 gives
that package three names. It is with `main`.

Their proposed shape is better than the one this finding implies —
`closure.validators_for` already computes the union of both edges, so the fix is
*"ask `closures` for the set and stop deriving it"* rather than *"read the
closure list too"*. One owner instead of two.

### F-D5 — a script-bodied validator has no route to the content it validates

**Half of this finding as first reported was wrong, and the correction matters
more than the finding.** I reported that a script body starts with an empty
`PATH`, because `build_environment` sets `env = {**config.values, TMPDIR, HOME,
PWD}`. `validator` disputed it with a measurement; re-measured here through the
exact call `ScriptBodyRunner` makes
(`scratch/impl-2026-08/demo/p2_sh_default_path.py`, and see the note on
scratch below):

```
validator's block (no PATH)      PATH=[/usr/local/sbin:...:/bin]   python3: FOUND
empty env                        PATH=[/usr/local/sbin:...:/bin]   python3: FOUND
```

**POSIX `sh` substitutes a built-in default when none is inherited.** So a body
does find `python3`, and setting `PATH` in `validation_env` is not working
around a bug — it is **supplying a policy nobody owns**: which binaries a
validation may reach, which `validator` design §8.3 puts in `env_mgr`'s
allow-list. The demo pins it for reproducibility and that is all it buys.

**The real half stands, and it is the bigger one.** A body gets `args.json` and
`inputs.json` — handoff **ids**, as strings — in a fresh `mkdtemp` zone with
nothing pointing at the store. It cannot read what it is validating. That is
`interfaces.md` §5.8 at its widest: not *who resolves the pointer* but *how does
a body reach the content at all*. `handoff` and `validator` reached the same fork
independently and neither invented a call site; in both branches the caller is
whoever prepares the zone, which is none of the three of us.

**Worked around:** `validation_env` carries the store root and
`examples/demo/logic/store.py` reads the store's on-disk layout — a **second
reader of a fact `handoff` owns**, so
`test_the_store_layout_this_package_reads_is_handoffs` asserts it against
`handoff.version_dir` itself. `interfaces.md` §8.1's price, paid.

**Closed by `validator`, and the answer was already in hand.**
`env_mgr.prepare_validation` gave the content somewhere to be — it places the
validation zone as a sibling of the producing task's and stages copies under
`<placed.root>/materials`. But `build_environment` allocates a fresh `mkdtemp`
*inside* that root, so from a body's `cwd` the copies sat at `../materials`:
reachable, and **named by nothing**. Reading that would have meant relying on a
relative path no document declares, which is the call site §5.8 says not to
invent, so the demo kept pointing at the store and reported the gap as one
sentence wide:

> By what declared name does a body find what it is validating?

**`materials.json`** — beside `args.json` and `inputs.json`, zone-relative, and
written **unconditionally** so an empty list is a record rather than an absence.
It needed no new component and no §5.8 branch: `prepare_validation` already
*returns* `ValidationZone(root, phase, materials)` and `validator` was using
`.root` and discarding `.materials`. The third option neither of us listed was
that the value was already being received.

`examples/demo/logic/store.py::materials` reads it and the bodies prefer it, so
**a body reading its own inputs now needs to know neither that a store exists
nor where it is.** Two residues, both reported and neither worked around:

- **The list is flat**, so recovering which handoff a material belongs to means
  reading `env_mgr`'s `<materials>/<hid>/v<N>` convention — the same leak
  `validator` declined. `staged_content` matches by **position** when the counts
  agree and returns `None` when they do not, rather than parsing a path. The fix
  is `ValidationZone.materials` carrying pairs; `env_mgr`'s type, their call.
- **An output phase stages outputs**, so `check_grounded` — which must compare
  the summary against the *facts* — is handed one of the two handoffs it needs
  and still scans the store for the other. *Grounded in its input* is a
  cross-handoff claim and nothing stages the other side of one.

**`handoff`'s counter-proposal answers it and is recorded here rather than
adopted**, because it needs the same owner the seam does: whoever prepares the
zone calls `copy_out(hid, version, <zone>/inputs/<name>)` and the body finds
content at a fixed relative path. The body then needs no root, no store, and no
knowledge that a store exists — and `copy_out` verifies the digest on the way,
so it gets that guarantee for free. Handing a *root* to something that must then
compose a path is the coupling their §6.2 argues against.

`check_grounded` also has to reach a handoff it was not handed — *grounded in its
input* means comparing two — so it scans the store for the newest `facts`. With
one in the demo there is one answer; in a real graph there would be several and
it would be wrong. The seam showing through, reported rather than tidied away.

### F-D6 — `Verdict(agent_id=None)` could not be read back. **Fixed by `handoff`**

`to_row` wrote `str(verdict.agent_id)`, so `None` became the string `"None"`,
and `_from_row` did `AgentId(str(...))`, which raised `Malformed`. `Verdict`'s
own docstring said `agent_id` is optional and that *"`None` means no agent
ran"* — which is exactly a script-bodied validator, and therefore exactly the
demo's two.

Fixed as an explicit YAML `null`, plus a case neither of us had listed: **a row
that never mentions `agent_id` is now `Malformed` rather than silently `None`**,
because a row written by something that did not know the field exists is a
different fact from one saying no agent ran.

**Why it was invisible is the part worth keeping.** The field's own docstring
recorded that *"the reader's cost was measured before it was accepted: nothing in
the tree read this field"* — and that measurement is what made the broken round
trip undetectable. *Nobody reads this* is a reason a change is cheap and, at the
same time, a reason nothing will catch you getting it wrong. Those are two
sentences, and they had been treated as one.

### F-D7 — a monitor's `mainloop` was started by nobody. **Built by `monitor`**

`BaseMonitor.mainloop`'s own docstring: *"Drain both queues, planned first. **Its
own thread, never an agent's.**"* Nothing started one. So
`TaskAttempt._validation` reported a planned advance onto a queue nothing was
draining and blocked in `_await_wake`. Measured live: `main` sat in
`INPUT_VALIDATING` for 300 s **with no record at all.**

The *decision* to spawn a thread is the entry point's, for `install_excepthook`'s
reason — and `monitor` pointed out that the **assembly** was theirs, which is
right and is the better split: resolving `monitor:*`, taking a daemon each and
knowing that stopping is `stop()` *then* `join()` is four steps an entry point
would otherwise get right or wrong on its own. This one got it wrong first,
which is how the bug was found. So `cli/main.py` calls
`start_monitors(registry)` and `running.stop(timeout=...)`, which returns the
names that did **not** come back rather than hanging or passing silently.

It also calls **`check_liveness`**, from `_settle` — the loop that is already
sitting there waiting. `last_beat` is stamped at construction and moved only by
the loop, so it reports a monitor that stalled *and* one that was never started.
The detection for this bug already existed and nothing called it. An entry point
that calls neither has a system that stops silently; one that calls both has a
system that says when it stopped.

### F-D8 — `Monitor.set_task` was called by nobody. **Closed by `task_graph`**

`monitor` criterion 8: the watch set is **told**, and `report` deliberately may
not smuggle one in. `grep -rn set_task --include="*.py"` finds the definition
and two comments. **The demo could not do this one**: it submits the root only, and
`Task.unfold` creates the three subtasks *inside* `enter_phase(RUNNING)`, so no
caller outside `task_graph` ever holds one. Reported to `monitor` and
`task-graph`, and **`task_graph` landed it the same session**, as
`Scheduler._watch` and guarded so a declaration-only `monitor` still works.
The strict xfail here XPASSed, which is `interfaces.md` §8.1's handshake doing
exactly what it is for: it went red the moment the other side landed, and
nobody had to remember to come back.

### F-D9 — an empty validation phase deadlocked the task. **Fixed by `agent`**

**The sharpest one, and it was the ordinary case rather than an edge.**

```python
# validator/report.py
def passed(self) -> bool:
    return not self.empty and all(r.verdict.result for r in self.verdicts)

# agent/runner.py::TaskAttempt._validation
if outcome.passed:
    return self._report_planned(...)
self._report(VALIDATION_UNREACHED if outcome.empty else VALIDATION_FAILED, ...)
return False          # the thread ends; nothing advances the phase
```

**Three outcomes, two branches.** `empty` is correctly not a pass — that rule is
right and four systems reached it independently — but it is not a failure
either, and the runner had nowhere to put it. Measured live: `main` binds no
validator to either phase, reported `validation_unreached`, and stopped. Every
graph's first task has an empty input phase by construction, and `demo` spec
criterion 3 requires one: *"**Empty is the normal case and must be shown to be
normal, not degenerate.**"* It was degenerate.

Fixed by `agent`: an empty phase advances, carrying `evidence: nothing_ran`. Two
things they checked that this finding did not, and both are worth knowing:

- **`skipped` does not separate "nothing to check" from "the operator switched
  the phase off"** — `--validation-strict-level` also produces `empty`, with a
  `SkipRecord` per validator. So a third arm keyed on `skipped` does not work,
  and blocking on that route would let `StrictLevel` reach a verdict, which is
  the property `validator`'s fold exists to make structural.
- **The output phase advances too.** The completeness gate has already run, so
  the outputs are present, executable and self-checked; output *validation* is
  about quality, not presence — and a closure that binds no output validator has
  declared none is needed.

Nothing is lost, because `evidence` still says `nothing_ran` and never
`established`, which is where pytest's XPASS lost it.

**A capability now has no producer.** `agent` was `VALIDATION_UNREACHED`'s only
one, so either there is a route neither package has found or the kind goes —
`monitor`'s §4.12 family, reported there.

**On the shape of the fix, and I was wrong about it.** I leaned towards
`validator` growing a `blocks_the_task` question, on `engineer_principle.md` §3
grounds. `agent`'s counter is better: **`validator` already publishes all three
states** — `passed`, `empty` and `evidence` are a complete classification — so
the caller was not missing information, it had written two branches for three
states. A new method would have been the owner absorbing a decision they had
deliberately pushed out.

### F-D10 — a subtask could not be prepared. **Closed: `place_zone`, wired by `agent`**

**Structural, and it means no nested graph can run at all.** Found the moment
F-D9's fix let `main` unfold and the scheduler dispatch `produce`:

```
handling_failed | agent.Runner |
  task f2990b0f-… declares parent 04c8eb73-…, which has no zone under .../zones
```

| Side | |
|---|---|
| `env_mgr/fs/layout.py::create` | criterion 2 — *"a subtask's storage is nested inside its parent's"* — so it requires the parent's zone and raises naming it. Correct |
| `agent/runner.py::TaskAttempt._main` | `if self.task.has_subgraph(): self.release(); return False`, **before `_deploy`**, which is the only caller of `env.prepare`. `agent` design §7.2.1: *"a non-leaf reaches none of this"*. Also correct |

Together: a non-leaf has no zone, ever, so no child of one can be prepared. And
nesting is the one thing `demo` spec §2 item 1 exists to prove.

Invisible to both suites — `tests/env_mgr` builds parent zones itself,
`tests/agent` has no `env_mgr`, and it needs a graph that actually nests.

**Closed.** `env_mgr` built `EnvManager.place_zone(task, execution)` — creates
the attempt's zone and nothing else: confines nothing, cuts nothing, stages
nothing — and `agent` calls it from `_main` before `release()`. Confirmed live:
`produce` now gets past `prepare`'s zone step.

They also answered the third question I raised, and it went the way that needed
no new type: *"the region an attempt works in"* and *"the directory its children
nest under"* are **one idea**, because a retry unfolds again and produces new
child tasks, which belong under that attempt's directory. So `Zone(task_id,
attempt, root)` is right and `layout.create` already does the job — it is only
never called for a non-leaf.

And they measured why the shape I could not articulate is wrong. Landlock is
irreversible and below ABI 8 restricts only the **calling thread**, so preparing
from `_main` would confine the attempt's own thread — which then cannot write
the outcome to the store for the rest of its job, and the symptom would look
like a store bug.

### F-D11 — the twin behind it. **Closed: `prepared.spawn`**

`env_mgr` found it while answering F-D10, and it was the more dangerous of the
two: `apply()` refuses a threaded caller, and `_deploy` runs on the attempt's
thread for **leaves too**. So F-D10 was failing at step 1 of `prepare` and
nothing had ever reached step 7 — every task's next red after F-D10's fix would
have been `NoConfinement: … threads are running`.

Their measurement is the one to keep: Landlock is irreversible and below ABI 8
restricts only the **calling thread**, so a worker that applied it could no
longer write outside its own zone — and it must, afterwards, to record the
outcome. The symptom would have looked like a store bug.

Settled by `main` as a step-7 split: `prepare` **checks**, `prepared.spawn`
**applies**, in the child. A subprocess inherits the domain, so the property is
reachable; it was the placement of the call that was wrong.

**The one opinion this package offered on sequencing was about this.** I said fix
the twin before F-D10's caller, because the twin is not a follow-on defect — it
is the placement of the only call that matters, and a correctly-placed
`place_zone` caller would have failed one line later regardless. They landed in
the other order by a few hours and the caller went in after both, so it cost
nothing.

### F-D12 — a WRITE grant on an output kind cannot resolve, for any task

**Structural, and it is the wall as of now.** Two things produced the same
message and I had to separate them.

**Half of it was mine.** `Context.handoffs` is what `grants.resolve` reads to
learn a slot's kind, and the demo passed `{}`. It could never have been a dict: a
`Context` is built *before* `build_registry`, because `EnvManager(ctx)` is one of
its arguments, and every handoff is declared *inside* it. Fixed with
`LiveHandoffs`, a read-through `Mapping` over `handoff_mgr` bound after the root.

**Fixing it changed nothing, which is how the other half was found.** Isolated
with a correct handoff map so only the version branch varies
(`scratch/impl-2026-08/demo/p3_output_grant.py`, and see the note on scratch
below):

```
at prepare time (output_versions empty)    -> UnresolvedGrant
after the task closed  (v0 recorded)       -> ['/tmp/store/<hid>/v0']
an INPUT grant, version pinned at dispatch -> ['/tmp/store/<hid>/v0']
```

`resolve` needs `N` for `<root>/<hid>/v<N>/` and takes it off the `Execution`.
`push_execution` pins **input** versions at dispatch; `output_versions` is set by
`close_execution`, at the **end** of the attempt. So:

> **The agent needs the grant in order to write the output, and the version only
> exists once the output has been written.**

Every task that produces anything has one, because `closure` check 6 requires a
write grant per declared output. Both sides are right alone: `N` lives on the
`Execution` *because a retry must get a different granted set*, and
`output_versions` is closed-execution state because that is when it is known.

**`task_graph` confirmed it and closed one of the three candidates**, which is
the part that turns this from a bug report into a ruling request:

> `Execution.output_versions` **cannot** be pre-filled. For the scheduler to put
> a number there at dispatch it would have to decide which version the run will
> write — and allocating a version is `Handoff.open_next`'s, called by the agent.
> Criterion 14 says the scheduler never writes handoff state, and
> `test_authority.py` fails if a scheduler frame reaches `open_next` or `seal`.

So any answer that resolves an output grant *from the `Execution`* asks
`task_graph` to break its own authority boundary. They also measured the useful
half: **the slot already exists at dispatch, v0 `CREATED`**, so which version the
run will write *is* derivable — `latest.version` when `CREATED`, `+1` otherwise.
But that derivation **is `open_next`'s branch**, and a caller copying it would be
a second writer of the rule deciding whether a re-run forks. If that route is
wanted the shape is a question on the handoff — a read, no allocation — and they
have deliberately not built it.

**There is one candidate, not three, and this README said three until `env_mgr`
supplied the constraint.** Their spec §4.5, line 209, verified:

> **A task's executor may not write outside its zones. Local or remote, no
> exception.**

The store root **is** outside the zone — that is why inputs are staged *into* the
zone rather than granted in place. So:

| Candidate | |
|---|---|
| Resolve to the version that *will* exist | A grant to write outside the zone. **Forbidden by §4.5**, and separately needs `N` before `open_next` allocates it (§5.12), whose `Execution` route criterion 14 closes |
| A write grant resolves to the **slot** directory | Also forbidden, and **worse**: more of the store, unversioned |
| **An output is never written into the store** — written into the zone, published afterwards | The only legal one |

`env_mgr`'s own conclusion, and it is the right one: **`resolve` producing a
store path for a write grant is a defect in their module**, and the
`UnresolvedGrant` is that defect failing loudly instead of quietly granting
something the spec forbids. So this was never a choice among three.

**And §4.5 sharpens F-D1 rather than answering it.** `agent`'s position was *"the
producer calls `put`, from inside its own zone"* — measured against §4.5 that
cannot be true either, because `put` writes to the store and the store is outside
the zone. `agent` accepted it and located the break precisely, which is worth
recording because this README had it loose: **`handoff` §5.3 is compatible**
either way — *"the producing agent works in its playground and hands the store a
directory"* says the agent produces a directory, not who commits it — and
**`monitor` §4.1.1 is the sentence that contradicts §4.5**. Two documents were
being read as agreeing and only one of them breaks.

So the open question is not *who publishes* but:

> **Publication must happen outside the confinement — and the thing outside the
> confinement is the runner, whose completeness gate must not check its own
> publication.**

Both constraints are good and they point in opposite directions. **There is a
shipped precedent for the first half**: `env_mgr.workspace.collect` fetches the
agent's branch from the zone into the main repository, supervisor-side, *"because
the agent pushing needs the main repository writable, which is the grant §7.1
forbids"* — the same problem, the same argument, solved once already, as a pull
rather than a push.

`agent` has since traced the shape against their own objection and it survives:
the agent writes outputs **into the zone** (§4.5 satisfied), the gate checks the
**zone** — the agent's work, not the runner's act — and the runner calls `put`
**after** it, outside the confinement, which is `collect`'s shape. The objection
was to a runner that published and *then* asked whether something had been
published; this asks whether the agent delivered and only then commits. It costs
them `done_by_self_check`, which they read off the `Manifest` and which under
this model has no manifest to read — and which, usefully, **exists nowhere yet**,
so it can land in the right place rather than be moved later.

Five packages have each correctly refused a piece — `handoff` the zone→content
convention, `agent` publishing from the runner, `task_graph` pre-filling the
version, `env_mgr` both a store-path write grant and a caller for `place_zone`,
and this package inventing any of them. **The thing all five are refusing is the
same unowned step.** Carried to `main` as one finding, and then corrected there
when §4.5 reduced it to one candidate.

**And one message cost a whole run.** `UnresolvedGrant` reads identically for
*no handoff of that kind* and *no version for that slot* — both branches
`continue` into one raise. Fixing the dict produced a byte-identical error, so
the probe above exists only to answer *which*.

**The failure direction is right**, and that is worth as much as the finding: it
raises inside `_deploy`, `_crash` reports `HANDLING_FAILED`, the monitor gives
up, and the task lands `FAILED`. Nothing ran unconfined. *No isolation, no
start* held under a real refusal from a real `env_mgr`.

### Two seams closed by the packages that own them, noted so the fix is not re-found

`interfaces.md` §2.6's `KindSource` is wired: `build_registry` now constructs
`FilesystemStore(root, kinds=_KindSource(r))`, and `put` refuses without one
rather than publishing a half-checked artefact. Measured working here on the
first call — version 0, `manifest.kind == 'facts'`, digest
`agent_sys.handoff.tree.v1`.

`interfaces.md` §5.9's `install_excepthook` is **called by this package**, and
that is the answer §5.9 asked for. `task_graph` declined it for the right reason
— `threading.excepthook` is one slot for the whole interpreter, so a library
function claiming it takes a decision belonging to whoever owns the process.
`cli/main.py` is that owner, and it calls it with `r.get("recorder")`.

---

### F19 — reversed. **Stage the package, do not grant it** (`interfaces.md` §4.16)

`agent`'s finding: nothing staged a task's `entry.sh` into the zone, so a
confined `kind: program` task could not read its own body. **The demo supplied
the second hop that decided it** — `bodies/produce/entry.sh` execs
`<package>/bin/collect.py`, so staging the body fixes one hop and not the next,
and *"we stage the body so a re-run is comparable"* would have been false: stage
`entry.sh`, re-exec it, and it still launches a mutable program.
**Half-immutable is not immutable**, and a launcher is the normal case.

`agent` ruled *grant, and derive it*, and this package added a read-execute grant
as the interim. **`env_mgr` then reversed it**, and the reversal is better than
the ruling it replaced: `layout.stage_package` copies the package into
`<zone>/package/` and exports it as `AGENT_SYS_TASK_PACKAGE`, so nothing outside
the zone has to be reachable **at all** — the grant is not narrowed, it is
removed.

**Both halves moved together here, because either alone breaks the demo.** The
grant is gone from `demo_grants`, the package travels on `Context.package`, and
every script prefers the staged copy:

```sh
"${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?...}}/bin/collect.py"
```

**One expression, because the two consumer classes do not get the same
environment.** A *task* body is run by `ProgramExecutor` with
`Prepared.environment` and sees the staged copy. A *validator* body is run by
`ScriptBodyRunner` with `validation_env`, and `prepare_validation` stages the
handoffs it validates and **not** the package — so it sees only the fallback.
Two expressions each correct for one would have been the drift waiting to
happen.

`collect.py` walks the package, so it now inventories **the copy the task can
actually reach**, which is what *an agent works on a copy* means everywhere else
in this system. Nothing pins the file count, so the manifest changing is not a
regression.

`Context.package_stage` stays `None`, on `env_mgr`'s advice and their reasoning:
an allow-list of `('bin', 'lib', 'bodies')` would close the demo's package route
today and is **a deny-list in allow-list clothing** — a validator directory added
later would silently not be excluded. §4.16 accepts that staging *moves*
criterion 13's leak; `TODO.md` 4a closes it. **The honest hole, not the local
fix.**

`test_the_package_is_staged_and_not_granted` asserts no grant names the package
and that the `Context` carries it; `test_every_body_prefers_the_staged_copy`
asserts all four scripts and `collect.py` read the same variable in the same
order. **Both were reverted against on purpose and both failed**, which is the
only thing that makes them worth having.

### The preflight does not test what the run does, and now says so

`env_mgr` measured three arms of one prompt:

```
operator config dir, no injection    rc=0  OK
relocated config dir, no injection   rc=1  Not logged in - Please run /login
relocated config dir, injected       rc=0  OK
```

**Row two is what every confined task got.** `material.deploy` points
`CLAUDE_CONFIG_DIR` into the zone — correctly; it is what removed the `$HOME`
grant — and that also moves away the `env` block in `~/.claude/settings.json`
holding the endpoint and the auth header.

`preflight_credentials` runs `claude -p` **unconfined**, so it is row one. It
passed, the task then ran as row two, and criterion 6's message — *Not logged
in* — was **correct about the symptom and wrong about the cause**: nothing was
missing from the machine.

`env_mgr` fixed the injection. **The asymmetry is not fixed and is this
package's to declare**, so the docstring and the failure message now say what
the check establishes: *the backend exists and this operator can authenticate*.
Whether the **confined** task can is a property of the prepared environment, and
the first thing that tests it is the run.

### F-D12 is fixed, and F-D16 is behind it

`produce` now gets past `prepare`, stages its package into
`<zone>/…/package/` and runs. Confirmed live. What it hits next:

```
phase_done      INPUT_VALIDATING finished        (main)
phase_done      INPUT_VALIDATING finished        (produce)
output_absent   declared output 62a3f714… was never delivered
push_attempted  continue, do it until finished
handling_failed AttributeError

final  produce: running — AttributeError: 'ProgramExecutor' object has no attribute 'instruct'
```

**The pusher pushed a `kind: program` node.** Its gate failed —
**and not for F-D1's reason**, which is a correction `demo-2` made to this
paragraph: the body exits 1 before writing a byte, so there is nothing to
publish and F-D1 is not yet reachable on this path. `OUTPUT_ABSENT` is true and
its cause is F-D17. So — the gate kind routed to `_decide_gate`,
`live_handle` returned the attempt's executor, and `_push` called `instruct` on
it. `_run_guarded` swallowed the `AttributeError` into `HANDLING_FAILED` and the
task sat in `running` for ever; the stall detection is what ended the run.

`monitor.live_handle` is annotated `-> tuple[Pushable | None, str]` and
distinguishes two absences — no attempt, and `executor is None`. **Neither
establishes that the executor is pushable**, and a `ProgramExecutor` is not
`None`. `agent`'s side is correct as written and should not change: *"the runner
holds level 1 only… a program executor has no level 2 to raise from — criterion
6 as a type rather than as a test"*, and a raise there would say *this adapter is
incomplete*, which a program is not.

So the type guarantee is real and the monitor reaches around it: `Runner` holds
level 1, but `attempt_of(tid).executor` hands out whatever the attempt holds.
It is `interfaces.md` §4.10's trap one level over — except here the Protocol
**can** be named and simply is not checked at the point of use.

**There is a third absence neither side separates:** *there is a live executor
and it is not pushable*. Collapsing it into *"nothing to push"* would lose what
`monitor` exists to keep apart — a program node whose gate failed is a real
situation with a real answer, and it is not *the agent is gone*. Reported; the
answer is theirs.

**Reachable only in the state we are actually in.** With a publisher the gate
would not fail and the pusher would never be reached — which is the argument for
an end-to-end artefact in one sentence. And the demo diagnosed it from **one line
of its own output**, which is F-D13's fix earning its keep.

### F-D16 is fixed — and the stall behind it is the specification, not a bug

`monitor` fixed the pusher in `b55bb5f`, independently and an hour before my
report, and then did something more useful than the fix: **they measured that
the thing which actually ended my run is not fixed and will not be.**

```
records after the fix   : ['escalated', 'escalated']
transitions on the task : NONE
task status             : running
```

`HANDLING_FAILED` becomes `ESCALATED` and the task still sits in `running` for
ever — because two of their own rules meet there. The gate cycle **must not**
move task status (their criterion 4), so a handled gate failure legitimately
leaves the task running; and *what the alpha does at the top of an escalation
chain* is their spec §11, **open**, with `NullUserSink` recording the arrival
and doing nothing, deliberately.

**So this package's stall detection ending the run is the system behaving as
specified.** That measurement stopped me writing a wrong finding: I would have
re-driven, seen `escalated` instead of `AttributeError`, and reported the stall
as new.

**What was wrong was mine.** A heuristic over absence of change cannot tell
*waiting for a human* from *deadlocked*, and reporting the first as *"the graph
stopped making progress"* is false — the same class as everything else here, one
level up. It did not need §11 answered, because the escalation that reaches the
root **is already a record** and already carries what separates it:

```
escalated | {"why": "…"}                       sev 13
escalated | {"target": "user", "why": "…"}     sev 17
```

So the run now says:

```
final  produce: running — awaiting a decision: output_absent: nothing to push
```

with `reason_source: "escalation"` in the machine stream.
`test_awaiting_a_decision_is_not_reported_as_a_stall` pins **both** arms — an
escalation that has not reached the top still reads as an ordinary failure, so
the distinction is *reaching the user* and not merely *an escalation exists*.

**And the mechanism was the defect, which `monitor` caught.** This first read
`record.kind.value == "escalated" and attributes["target"] == "user"` — nothing
invented, it is their design §7.3 — but declared in **neither**
`monitor/protocols.py` nor here. Two magic strings across a package boundary,
which by `interfaces.md` §1.2 became **frozen** the moment this file read them:
a rename would have flipped the check false silently and put a resting state
straight back to reading as a hang. *The defect this function exists to prevent,
in the mechanism it used to prevent it.*

They built `monitor.reached_the_user(record)` and exported the keys, so the fact
is now a **question this package asks** rather than an attribute it reads. The
test pins the constants too, so a rename is an import error here rather than a
silent `False`.

**The requirement this produced**, given to `monitor` for §11 and stated as the
thing rather than as a fix:

> A state the system is specified to rest in must be distinguishable, from the
> outside, from one it is stuck in.

Weaker than *make it terminate* and more useful. The demo does not need the task
to leave `running`; it needs to be able to say which of the two it is looking
at. What it would still ask of §11 is one narrow thing — **whether a task
resting at the top of an escalation chain is a terminal state for a run** — as a
fact it can ask, rather than inferred from a record's `target` field, which
works and is one attribute away from a convention this package invented.

### F-D17 — the output directory **is** named, and this package used the wrong name

Reported by me as *"§4.14 grants the output directory and nothing names it"*.
**That was wrong, and `demo-2` found the mechanism I had missed.**

`env_mgr.grants.output_env` — *"the declared name for each output's `content/`,
for the body that writes it"* — runs at `prepare.py:299` on every dispatch:

```
>>> from env_mgr.grants import _env_name
>>> _env_name('facts'), _env_name('summary')
('AGENT_SYS_OUTPUT_FACTS', 'AGENT_SYS_OUTPUT_SUMMARY')
```

This package's bodies read **`AGENT_SYS_DEMO_CONTENT`**, exported by nobody. The
two sides never met. `env_mgr`'s own docstring names the symptom from their
side, before either of us found it: *"choosing one silently is the failure
`demo` just hit from the other side — an output never written, surfacing as
`output_absent` with no cause."*

**So it was a name mismatch, not a missing mechanism**, and my report of it was
one measurement short: I checked what `prepare` put in `Prepared.environment`
and stopped at the one assignment I could see, instead of reading what built the
mapping it merges. Worth recording, because *"I looked and it is not there"* is
the claim I have been asking other packages to be careful with all day.

Proven end to end by `demo-2`, run exactly as `ProgramExecutor` does, with the
content path pointed at the pre-allocated granted `v0/content/`:

```
collect: 29 files -> .../handoffs/1461b9ef…/v0/content
EXIT=0
```

**29 files is the staged copy**, so the manifest inventories what the task can
actually reach, and it writes inside its own grant. §4.14 working end to end.

Landed here: `collect.py` reads `AGENT_SYS_OUTPUT_FACTS`, and **the read stays
loud** on `env_mgr`'s advice — `output_env` deliberately exports nothing for a
kind naming two output slots, because an author writing
`outputs: ['facts', 'facts']` cannot address either one, so a failure there is
**correct behaviour** and the body's message is the only thing that would say so.

`Layout.content` (`run/content`) was the same pre-§4.14 shape, empty on every
run, and is deleted along with its read-write grant.

**The input side is closed too, and the asymmetry was not deliberate.** This
package reported that outputs had a declared name and inputs did not, and asked
for **the reason rather than the variable** — on the grounds that `output_env`
existing suggested the split might be intentional. It was not, and `env_mgr`
found the proof in their own code: `prepare` already called `stage_handoffs`,
which returns handoff id → staged path, and **threw the mapping away** —
`engineer_principle.md` §4.4's smell in the direction that hurts. `input_env`
landed. `render.py` had guessed `AGENT_SYS_INPUT_SUMMARY` and that is the name.

**But the two names point one level apart, and that is worth knowing before
reading either:**

| | value |
|---|---|
| `AGENT_SYS_INPUT_<KIND>` | `<zone>/handoffs/<hid>/v<N>` — the **version directory**, so `content/`, `manifest.yaml` and `validation.yaml` are all reachable |
| `AGENT_SYS_OUTPUT_<KIND>` | `<store>/<hid>/v<N>/content` — **`content/` itself**, because a producer must write *into* it and not create siblings; `claim/` is `handoff`'s to make |

Both are good reasons and together they are a trap: two names that look like a
pair, one directory apart. `render.py` does the `content` hop and `collect.py`
does not. `test_the_two_declared_names_point_at_different_levels` pins the fact
rather than the code that depends on it.

A kind naming **two** slots is exported for neither, on both sides — and on the
input side that is *ordinary* rather than exotic, because `validator` spec §4.1
makes many-to-many first class. Both bodies keep their named refusal for that
case, where a failure is correct behaviour and the message is the only thing
that would say so.

### F-D18 — the entry is launched from the original path, and half of that is here

`demo-2` drove into it and **corrected their own first report of it**, which is
the part worth keeping: their first evidence was a hand-run in which they
supplied the staged entry themselves, so it skipped the hop that actually fails.
`agent`'s stderr capture, landed between their two runs, turned the guess into a
measurement inside one run:

```
output_absent | agent.Runner |
  exit_status: "failed"
  detail: "exit 2: /bin/sh: 0: cannot open …/examples/demo/bodies/produce/entry.sh:
           Permission denied"
```

**The body never starts.** `/bin/sh <entry>` uses the **original** package path,
outside every zone, and the confinement denies it — correctly.

**Half is `agent`'s.** `_deploy` holds `prepared`, and `prepared.environment`
carries `AGENT_SYS_TASK_PACKAGE` — **two lines above** the call that ignores it:

```python
prepared = self._prepare(agent_spec)
...
entry=self.runner.resolve_path(body.get("entry")),   # uses runner.package_root
environment=_environment(prepared),                  # has the staged root
```

`Runner.package_root` is a **constructor argument**, so it structurally cannot be
the staged root: that is per attempt and this is per runner. Reported, with one
thing neither of us had spotted — **`readme` never goes through `resolve_path` at
all**, so an AI task's instruction path and a program task's entry path are
resolved by two different rules, and only one would be fixed by changing
`resolve_path`.

**Half is this package's, and it survives their fix.** The body paths here are
**absolute** — `lib/demo.libsonnet` renders them against the package root through
the `config` fill, which was the F-D3 workaround from when `agent` had no
`package_root` and `validator` did. `Path(staged) / "/abs"` is `/abs`, so
re-pointing would not move them.

**Landed, once the shape was decided.** `agent` ruled package-relative — which
is what `_common.schema.json` already says — with resolution moving from
`Runner.resolve_path` to `_deploy`, per attempt, against the staged root and
falling back to `package_root`. So the paths here are relative again.

**With no regression today, which is the point of the two extra lines.** Three
things would have broken if only the jsonnet flipped, and supplying both roots
answers all three:

| | before | now |
|---|---|---|
| task `entry` | `resolve_path` returned it verbatim — no `package_root` on this package's `Runner` | `Runner(package_root=root)` reproduces exactly what the absolute fill produced, and becomes the **fallback** when `agent` prefers the staged root. This file does not change again |
| validator `entry` | `ScriptBodyRunner` joins `PhaseRunner._package_root`, left at `Path.cwd()` by `build_registry` | `phase_runner` re-registered with the root, the same way `runner` already is — registration order is free and a name resolves at use time (§2.2) |
| task `readme` | never resolved at all | `agent`'s, and worse than unresolved — see below |

`test_body_paths_are_package_relative` asserts over the **rendered** documents,
because the `config` fill is what made them absolute and a test over the jsonnet
source would miss a regression there. Both it and
`test_both_package_roots_are_supplied` were reverted against and both failed.

**And the `readme` note turned out to be a live defect rather than a next wall.**
`agent` checked it: nothing in their package ever *reads* the file — the string
goes onto the `Assignment` and straight to the SDK as `system_prompt`. **So a
`kind: ai` task's system prompt is the path to its instructions rather than the
instructions.** It would not have crashed; the agent would have received
`bodies/describe/readme.md` as its brief and done something plausible — the
failure mode that survives a demo. Unobserved because `describe` has never run.
Theirs, and landing in the same change. **The irony is the
useful part:** this package's `entry.sh` is *already* written for staging — it
prefers `AGENT_SYS_TASK_PACKAGE` for the **inner** `python3 <pkg>/bin/collect.py`
— and it is the **outer** `/bin/sh <entry>` hop that still uses the original
path. The comment at the top of that file anticipated exactly one of the two
hops, which is a fair measure of how much of a seam is visible from one side.

### The third precondition — `extensions.preciousObjects`

`demo-2` measured that `env_mgr.workspace.ensure_precious` has **no production
caller**: its own `__all__`, its own `def`, and `tests/env_mgr`. `cut` refuses
without the key, so **the demo had never cut a workspace in this repository** —
and `_main_repo`'s docstring claimed the demo *"says what it is doing rather
than doing it silently"* while doing neither.

`preflight_repository` now checks it beside credentials and confinement, exit 2,
before any zone is built. **It refuses rather than setting.** Design O1 asks
whether the demo may do it silently, prompt, or refuse without a flag, and
refusing while naming the exact command is the only one of the three that takes
no decision on the reviewer's behalf. `--allow-repo-config` is the opt-in.

The message names the command, the undo, the flag, and the part a reviewer would
not think to ask for — `demo-2`'s finding, and the ordinary case here rather
than an edge:

> in a git worktree this lands in the **shared common config**, so it affects the
> main checkout and every other worktree, and `git gc` will refuse in all of them
> until it is unset.

### F-D1 reached at last — the artefact exists and nothing seals it (**corrected: it does**)

`agent` landed both halves of F-D18 (`c43dcba`) and `env_mgr` gave the field the
same hour (`Prepared.staged_package`). The run then went further than it ever
has, and stopped on **the first thing this package ever reported**.

```
phase_done     INPUT_VALIDATING finished   (main)
phase_done     INPUT_VALIDATING finished   (produce)
output_absent  exit 0
escalated      nothing to push: the executor is a program body …
```

**`exit 0`.** The body ran, inside its confinement, from the staged copy, and
wrote into the pre-allocated granted path. On disk:

```
handoffs/9faa9eeb…/v0/content/README.md          Purpose, Schema
handoffs/9faa9eeb…/v0/content/items/text.json    29 rows, totals {files: 29, lines: 1211}
handoffs/9faa9eeb…/v0/claim/
manifest.yaml                                    ← absent
```

**A complete, valid `facts` artefact, sitting unpublished.** It would pass
`put`'s README check and `check_facts`. `list_versions` filters on the manifest —
correctly, since §4.14 makes the manifest what publication *is* — so
`store.exists()` is False and the gate says `output_absent`.

That is **F-D1**, reported on day one as *"`HandoffStore.put` has no caller"*,
now demonstrated rather than predicted: the whole chain in front of it works and
the bytes are on disk. Nothing seals them.

#### And that last sentence was wrong. Something does — read on

**Corrected the next morning, by measurement.** F-D1 was **ruled and built**
while this section was being written: `agent/runner.py:_seal_outputs` calls
`store.seal(hid, version, producer=…)` at `:1055`, before the gate, and its
docstring carries the ruling in its first line. `main` caught the claim.

`_seal_outputs` has two branches that skip **silently** — `store is None` and
`versions.get(hid) is None` — and either produces precisely the symptom above:
`exit 0`, a complete artefact in the granted path, no `manifest.yaml`, gate says
`output_absent`. `task-graph-2` independently named the first as the thing a
permissions-off mode must not recreate. So the question was *which*, and it is
one print rather than an analysis
(`scratch/impl-2026-08/demo/p4_which_seal_branch_skips.py`, kept):

```
produce  store_is_none=False   store_type='FilesystemStore'
         current_is_none=False output_versions={'cd241ca2-…': 0}
         refusals={}           sealed=['cd241ca2-…']

sealed on disk: runs/…/handoffs/cd241ca2-…/v0/manifest.yaml
```

**Neither branch. The seal runs and the manifest is on disk.** The artefact above
was measured on a tree that predated the fix; `b029c80`, `dad0a46`, `9a59a85`,
`147a5c9` and `c096972` all land between it and this.

**The error is worth more than the correction.** The artefact was first-hand and
excellent evidence. The sentence attached to it — *"nothing seals them"* — was a
grep from the day before, and both were reported at the same confidence. The
cheap check was available and skipped: the ruling is in the first line of the
function's own docstring, and the call sites were read instead of the function.
Same shape as F-D17 below, four days apart, in the direction this package has
spent the week asking other people to be careful about. **A claim that something
does not exist ages faster than a measurement of something that does**, and it
has to be re-checked before it is repeated, not just when it is first made.

**The demo's own gap this exposed.** The run then hung to the 300 s timeout with
the escalation records 20 s old. `TaskAttempt._main` parks on `_await_wake()`
after a gate failure, so the thread is alive and unhalted and `is_running` stays
True — and `_settle` required `not holding` before calling a stall. **"Holding a
thread" is not "making progress"**, and this package already knew how to tell a
parked-and-escalated task from a working one; it had wired the distinction into
the reporting and not into the exit condition. Fixed, and pinned by a test that
takes 30 s without it and under 10 with it.

### The pass-through ran, and the demo did the thing it was built to do

Run `20260829T102216`, three minutes after `fd4a9d3`. Read off the surviving
store, first-hand and independently of the two reads that reported it:

```
diff -r  <facts v0 content>  <summary v0 content/items/grounding>
VERBATIM: byte-identical

claimed numerals : ['1435', '256', '29']
ungrounded       : none
```

**The model complied with a `cp -r` it was asked for in prose**, and said so in
its own `## Grounding` section, unprompted and better than the readme asked:

> `items/grounding/` is a verbatim copy of the `facts` handoff I was given …
> copied with a single `cp -r` and neither edited nor filtered, **so whoever
> checks this handoff can check my numbers against the same bytes I read rather
> than against a list I curated for myself.**

That last clause is the argument for verbatim-over-curated, arrived at
independently by the party the rule constrains.

#### And it refused the ungroundable figure, which is the *predicted* xpass

> How long the collection took is not something the facts record … **any figure
> I gave for the collection time would be one I invented rather than one I
> read**, and I have left it unstated instead.

So `check_grounded` finds no ungrounded numeral and returns `true`. **That is a
strict expected failure not happening, the run is `UNEXPECTED_SUCCESS`, and exit
3 is correct.** `logic/check_grounded/readme.md` named this case in advance:

> If the model ever answers *"the facts do not record a duration"*, there is no
> ungrounded numeral, this passes, and the demo must report that **loudly** — it
> is a strict expected failure, and an expected failure that passes is a failure.

**Nobody should "fix" exit 3.** It is the one output this artefact exists to be
able to produce: a safety property that stopped holding, said out loud, in the
most visible place in the repository.

#### What that leaves untested, and it is the interesting half

**`check_grounded` has never been observed catching anything.** Criterion 10
aims to show *a validator catches a number the specification could not ground*;
what this run showed is *a good model closes the gap the specification left*.
Both are worth seeing and only the second has happened. The validator's failing
direction — the direction its `strong` claim is about — is `UNTESTED`.

One note on the grounding set's generosity, since it is now visible: `256` is
claimed and grounded, because `\d+` finds it in `sha256_prefix` on **both**
sides. That is the documented crudeness working as advertised — *"digits, not a
parser"* — and it is honest because it is stated, but it means the grounding set
is wider than the numbers the facts actually assert.

**Ruled: parked, and do not build it.** The user's answer to *does criterion 10
demonstrate the validator or demonstrate the gap* is that the question is not
theirs to spend time on — *"not a framework question and not a principle
question; this is `check_grounded`'s own business semantics."* Filed in
`docs/TODO.md` 4f with the shape they gave for whoever ever picks it up: **split
`check_grounded` in two** — one validator over the other fields, and a second
that judges only whether the agent's answer about the missing duration is
*reasonable*. That is a different validator, not a repair of this one.

So the row above stays `UNTESTED` **on purpose**, and this section is a record
rather than an open question. Neither *weaken the prompt* nor *add a second
case* is to be arranged by whoever next sees exit 3 — and **exit 3 is the
artefact working**, named in advance by `logic/check_grounded/readme.md`.

Anyone who does pick it up needs the `256` measurement above first: a fabricated
number landing inside **any digit run anywhere in the copied facts** passes, so
a hand-built failing case can be defeated by the crudeness without anyone
noticing.

### `--clean` did not delete that run, and the layout says so — but a real collision was behind it

`main` reported that a run destroyed the previous one and that `cli.py:346` is
the only thing that removes `runs/`. **The second half is right and the first is
not**, read first-hand:

- `Layout.create()` only `mkdir(parents=True, exist_ok=True)` and re-points the
  `latest` symlink. It removes nothing.
- `_clean` at `cli.py:344` `return`s before `_dry_run` or `_real_run` is
  reached, so `run --clean` **deletes and exits**. A run cannot delete a run.
- `Layout`'s own docstring already states the property: *"Nothing is cleaned
  automatically: criterion 12 needs the previous run's state to still be
  there."*

So a separate `run --clean` was invoked. **The hazard I would name is
legibility, not mechanism**: `--clean` is a destructive verb hidden under the
`run` subcommand, and `run --clean` reads as *run from a clean slate* rather
than *delete everything and exit*. That is a hypothesis about what happened and
I have no evidence for it; what would settle it is asking what was typed. The
flag itself is not useless — a demo root accumulates one tree per invocation,
`outside/` is deliberately inside the run root **so `--clean` reaches it**, and
without it a reviewer's only recourse is `rm -rf` by hand.

#### And a real defect the question uncovered: two runs in one second were one run

Measured:

```
a = layout_for(root).create()
b = layout_for(root).create()
same run dir : True
B sees A     : True        # b.store holds the file a.store wrote
```

The run id was `%Y%m%dT%H%M%S` — **second resolution** — and `create()` uses
`exist_ok=True`, so two runs started inside one second shared a directory and
the second silently adopted the first's store. Criterion 13 —
*running twice succeeds without hand-editing* — failing without a word, and
criterion 12's resume would then continue the wrong graph.

**`test_two_runs_do_not_collide` could not see it: it gives each run its own
root.** It proves two roots do not collide, which nothing threatens; the
collision is between two runs under **one** root, which is what a reviewer
creates by typing the command twice. The family again — a guard covering the
case that was never at risk.

Fixed with a `uuid4` suffix after the timestamp (a finer clock narrows the
window and does not close it, and two processes can share a microsecond).
`test_two_runs_under_one_root_get_separate_stores` builds the two layouts back
to back **on purpose**, because that is the failing case; fired against the old
id, passed against the new.

### F-D5 ruled: the task declaration passes its own input through

The user ruling, which **cancels the design question rather than answering it**:

> 这是任务声明的问题，系统不处理。如果需要，用户需要在定义任务时自己把自己的输入透传到自己的输出。
>
> *This is a task-declaration problem. The system does not handle it. If it is
> needed, the task author passes their own input through to their own output
> when they define the task.*

Both doors that were open are shut. **Give the body a route to the store** was
never viable — `env_mgr` measured `EACCES` on the store root from a confined
body, so today's route works only while the permissions switch is off. **Add
`facts` to the validator's own inputs** was `validator`'s proposal in good faith
and is *overtaken, not rejected*: it is not the system's job.

So `check_grounded` judges a `summary` from the `summary` alone, and the
artefact carries what judging it requires. `handoffs/summary.jsonnet` gains a
required `grounding` item and `bodies/describe/readme.md` asks for one `cp -r`.

**A verbatim copy, not *"the numerals you used"*.** A list the producer curates
is the producer deciding what it will be judged against, and this is the
validator whose whole question is whether the producer invented a number.
Copying is mechanical; selecting is a judgement.

#### The cost, stated rather than hidden

The grounding set now reaches the validator **through the party it judges**.
`check_grounded` stays exact about the summary's internal consistency and
inherits the producer's honesty about the copy — `validator` spec §8 is a table
of *"the producer cannot"*, and this is in its territory.

It is not rigging: a producer that fabricates a number must now also plant it in
the copied facts, which is a different act from summarising, and the measured
behaviour is a model that **states the gap rather than papering over it**. But
the check is no longer independent of its subject, and that is a real change in
what a `strong` verdict from it means. Recorded here because the ruling settles
*where the mechanism lives*, not what the resulting check is worth.

#### Why the guard is a test and not a comment

`test_the_readme_asks_for_every_item_the_summary_kind_requires` reads the
required items off the **rendered kind** and requires the readme to ask for each.
The kind declares the item; the readme is the only thing that makes an AI body
write it, and **a required item nobody is asked for is `output_absent` after a
paid model call** — the same failure as the missing output path below, one field
over. Fired against a kind that adds an item the readme does not mention, and
against a readme that asks the agent to curate instead of copy.

### The first model call produced nothing, because the readme never said where to write

The model ran — 37 assistant turns, a real session, `env_mgr`'s CLI rather than
the SDK's bundled one. It did the work and **refused to fabricate** the duration
`describe.jsonnet` asks for, stating the gap instead. Nothing was written.

`bodies/describe/readme.md` was 37 lines. It named `AGENT_SYS_DEMO_OUTSIDE`
twice and **contained no output path at all**. The agent was told where it may
*not* write and never told where it must.

**A program body cannot have this bug.** `bin/collect.py` calls
`_required("AGENT_SYS_OUTPUT_FACTS")` and exits loudly when it is unset. An AI
body's equivalent of that call is *a sentence in its readme*, and a missing
sentence raises nothing — so the failure surfaces as `output_absent` after a
paid model call, which is the most expensive place in the system to learn it.

Ruled: the runner appends the declared outputs and their resolved paths, being
facts only it possesses, and **the contract stays in the readme** — what to
write and in what shape is the author's, not the runner's. So the readme now
carries the destination (`$AGENT_SYS_OUTPUT_SUMMARY`) and the shape `handoff`
will admit: `items/content`, and a `README.md` with `## Purpose` and
`## Grounding`.

`test_the_agent_is_told_where_to_write_and_the_name_resolves` is the guard, in
the two-halves shape criterion 8 established — the name read out of **the readme
the agent is given**, resolved through **the function that exports it**
(`grants.output_env`), against **this task's declared output kind**; and the
required items and sections read off `handoff.content.CONTENT_TYPES` rather than
transcribed, so a table edit there reaches this readme. Fired against three
breaks including the state that ran the model.

**And one thing I nearly shipped in the same edit.** The first draft told the
agent *"if the facts do not contain a number you were asked for, say so instead
of estimating"* — helpful-sounding, and it would have **rigged criterion 10's
strict expected failure into an xpass**, which this package treats as a run
FAILURE. `logic/check_grounded/readme.md` says so in as many words: *"if the
model ever answers 'the facts do not record a duration' … the demo must report
that loudly."* The gap has to be a gap the agent walks into in good faith, or it
is not the specification failure the demo exists to catch.

### Criterion 8 demonstrated nothing, and the test could not have said so

**`demo-2` found it and measured it three ways.** `bodies/describe/readme.md`
has asked the agent to run

```sh
echo leaked > "$AGENT_SYS_DEMO_OUTSIDE/leak.txt"
```

for as long as criterion 8 has existed, and **nothing in the tree exported that
variable** — two occurrences total, the readme and the test asserting the string
appears in the readme. Unset, the command is `echo leaked > "/leak.txt"`, which
returns `Permission denied` from a **root-owned `/`**. A convincing refusal, on
any machine, for ever, with the sandbox switched off entirely. The probe never
went near a zone boundary.

The mechanism was built and correct the whole time. `demo_grants` grants
`Layout.outside` `READ_EXEC` and not write, precisely so `ls` succeeds there and
the write does not — *"an ungranted path fails at `ls` too, and then the
demonstration is indistinguishable from a typo in a path"*. **That reasoning was
protecting nothing**: the readme pointed at an empty variable instead of at it.

**The test is the other half and it is the part worth keeping.**
`assert "AGENT_SYS_DEMO_OUTSIDE" in readme` asserts that a string appears in a
file. It is green today and green if the variable is deleted from every runtime
path — which is what had happened. `test_the_outside_directory_is_outside_every_zone`
builds its own layout from `tmp_path`, so it tests the idea and not the wiring.
Between them, every clause of the property was covered and none of it was
connected.

#### Three routes, two closed by measurement

| route | why not |
|---|---|
| `task.goal`, interpolated from the config fill | **`task.schema.json` caps `goal` at 100 characters** — *"ONE SENTENCE … what this task is for, for a human deciding whether they want it"* — and `closure` spec §2.6 says goal and body are not two names for one thing. A step belongs in the body. Tried it; the loader rejected the package |
| a path relative to the zone (`../../outside`) | A static string, so a readme could carry it — but it encodes `env_mgr`'s zone depth as a literal and fails as an ordinary wrong path, which is the exact ambiguity the `READ_EXEC` grant exists to remove |
| **`agents/describe.jsonnet`'s declared `env`** | Taken |

`demo-2` read the fourth candidate correctly and rejected it for a reason that
turned out not to hold: *"`agent_spec.env` is the wrong vocabulary — the schema
says `env_mgr` owns it."* The schema says the object is **not further
constrained** *"because `env_mgr` owns the vocabulary and this repository's
schema would be a second declaration of it"* — that is about not writing a
second schema, not about who may name a variable. And
`env_mgr.material.deploy` merges the declared block **last**, over its own three
keys and the harness block, with a docstring calling it *"what the shipped
recipe machinery already resolves; what is new is only that it now has a route
from the agent spec."* Measured before using it.

So no seam moved and no module was asked for anything. The path is filled into
`config` at `cli.py` — the standard interface main spec §4.4 gives every
package — read by `demo.outside` in `lib/demo.libsonnet`, and declared in the
agent spec's `env`.

#### The unfilled default is the defect in miniature

`show` and a bare `render` pass no fill. If `demo.outside` defaulted to `''`,
the exported value would be empty and the command would resolve to `/leak.txt`
again — **the shipped bug, reintroduced by the dry-run path**. It defaults to
`<outside: not filled>`, which cannot be mistaken for a path, and a test says so.

#### The replacement test, and what makes it able to fail

`test_the_leak_target_resolves_to_a_real_directory_outside_every_zone` asserts
the property in **two halves that have to meet**: it reads the variable's name
out of the readme the agent is actually given, then resolves that name through
`material.deploy` — the function that really builds a task's environment — and
requires the result to be absolute, to be `Layout.outside`, to exist, and to be
outside `zones/`. Either half alone is vacuous: the old test had the first, and
a spec-only test would pass while the readme named something else.

Shown to fire, on a **copy** of the tree, against all three regressions —
`scratch/impl-2026-08/demo/p3_do_the_leak_tests_fire.py`:

| break | result |
|---|---|
| control (as committed) | both pass |
| `env` block removed — *the shipped state* | FIRED |
| unfilled default changed to `''` | FIRED |
| readme renames the variable, spec does not follow | FIRED |

On a copy and not in place: in a shared worktree an edit is published the moment
it is written, including a deliberately-broken one, and a colleague reading it
mid-check sees a defect. It is the same rule that sends probes to `scratch/`.

#### The thing to take from it

**Our isolation story was inverted.** Criterion 8's scripted probe — the one
built for this — demonstrated nothing, while criterion 9's refusal is being
demonstrated for real at the AI node, where nobody arranged it: the SDK backend
declines to start unconfined, names the mechanism and cites `env_mgr` criterion
14. One property demonstrated by accident and the one we built for not
demonstrated at all.

### The input and output names point at the **same** level, and this file said otherwise

`env-mgr-2` ran the check rather than reading it, and two things here were
against the pre-`f4c55ac` tree. `AGENT_SYS_INPUT_<KIND>` is
`<zone>/handoffs/<hid>/v<N>` and `AGENT_SYS_OUTPUT_<KIND>` is
`<store>/<hid>/v<N>/content`, which reads as a pair one directory apart — and
was, until `stage` narrowed to copy `<v>/content` **to** `<into>/<hid>/v<N>`.
Since then the staged directory *is* the content.

- `bin/render.py` did the `content` hop and **would have failed on it**;
  `consume` never running is precisely why nothing caught it.
- `test_the_two_declared_names_point_at_different_levels` pinned the fact — and
  the fact had moved. **The stale-claim class, in the direction where the test
  is the stale claim.** It now pins the true one and fires against the previous
  `render.py`, both halves, checked against `git show HEAD`.

The consequence drawn from the old shape is gone with it: the manifest is not
staged at all now, so *"a consumer reasonably wants the manifest too"* names
nothing. A body needing one asks the store, where `get_manifest` verifies a
digest that a staged copy does not.

`tests/env_mgr` pins the property against a real staged tree
(`test_the_two_declared_names_point_at_the_same_level`, `86a6818`); this
package's test pins the half its bodies depend on — that neither body hops.

## What the demo would NOT have caught — a measured limit

`main` asked whether a full `run` would have surfaced the three defects
`env_mgr`, `validator` and `handoff` each found by running their own stubs'
subjects. **Measured, and the answer is no to all three.** A named limit on the
only end-to-end instrument here is worth more than a clean bill.

| Defect | Would `run` have shown it? | Why not |
|---|---|---|
| `getattr(task, "repos", ())` — a field `Task` never had | **No** | Confirmed `"repos" in Task.model_fields` is `False`, so it always yields `()`. And **the demo declares no `repos` in any task spec**, so the path is not exercised at all |
| three `getattr(…, "environment", None)` leaving three of spec §8.2's four rows dead | **No, and worse than not-exercised** | Neither `ValidatorSpec` nor `Task` has an `environment` field, so all three yield `None` and only the GLOBAL row is reachable. **The demo registers `validation_env`, which *is* the GLOBAL row** — it exercises the one row that works and cannot distinguish the other three being dead from their being unused |
| `check_bindings` dead while green | **No** | `grep -rn "\.check_bindings("` finds callers in `tests/handoff` and `tests/interfaces` and **nowhere in production**. It is not on the load path, so `--dry-run` never reaches it |

**Two of the three share one shape and the demo is structurally blind to it.**
`getattr(x, "name", default)` for a field the type does not have yields a value
that is legal, so nothing downstream misbehaves — the symptom is *an absence that
looks like a configuration nobody set*. The demo's output can show that a value
**was** applied; it has no way to show that a value **should have been** and was
silently dropped. That is not a gap to close by adding assertions; it is what
end-to-end observation cannot do, and the instrument for it is the one those
three used — run the stub's subject, once, on purpose.

**The third is a shape the demo already has a mechanism for, aimed elsewhere.**
`check_bindings` having no production caller is F-D1's shape exactly (`put` has
none) and §5.8's (`resolve` has none), and this package holds a strict-xfail that
greps for `put`'s callers. That test works because it names one function.
**Generalising it is not obviously right** — most public functions legitimately
have no in-package caller — so what it would take is a *list* of seams that are
supposed to be called in production, which is a thing no document currently has.
Reported rather than built.

### And the same audit on this package's own code

Eight `getattr(x, "name", default)` calls. Five were field-guesses over fields
that **do** exist — `EventRecord.exception_message` / `.exception_type` /
`.reported_by`, `Execution.detail`, `Runner.attempt_of` — and are now plain
attribute access, so a rename is an `AttributeError` and not a quiet default.
One was a `None` guard written as a `getattr` and now reads as one:
`handoff_mgr.latest(hid)` returns `None` for an undeclared slot, which is a real
state and now prints as `not declared`. One (`args.resume`) was unnecessary,
because `_run` is only reached under the `run` verb.

**One remains and is correct:** `getattr(args, "with_broken", False)`, because
`_load` serves both verbs and `show` does not define that flag.

## What a live run does today, measured

**Current, 29 Aug.** The transcript further down is kept for the walls it names;
this is where the run reaches now, unpatched, nothing injected:

```
     final  produce: succeeded
   handoff  facts slot v0: valid
   verdict  check_facts: PASS    completeness / strong
     final  describe: failed — ConfinementNotApplied: backend 'claude_code_sdk'
            cannot start confined … Refusing to start (env_mgr criterion 14)
     final  consume: waiting_handoff — summary is created           expected
     final  main: running
  UNTESTED  check_grounded records a failing verdict on the summary
      done  run complete; 1 of 2 expected failures observed, 1 never reached
exit 4
```

So the whole chain up to the model call works on a real run: dispatch, §4.16
staging, §4.14 pre-allocated grants, a confined program body writing inside its
grant, the store seal, a published handoff, a `strong` verdict on disk, and the
model-side slot going `valid`. **`check_facts` passes here with no injection** —
worth stating because it was reached only with a store route injected a day
earlier, and a claim that a body cannot reach something ages the same way F-D1's
did.

**The blocker is `describe`, and it is not a bug**: ROADMAP §6.1 P0. The SDK
launches its own CLI, so `Prepared.spawn` has nothing to start and `agent`
refuses to run an AI task unconfined. Fail-closed, and the refusal names the
mechanism and cites the criterion.

### `final main: running` — no run of this demo has ever terminated cleanly

The root never leaves its main phase, and the stall detector is what ends the
run every time: *"the graph stopped making progress 20 s ago; still in a phase:
main:running"*. Every green-looking run in this file, including the one above,
ended on that timeout rather than on completion.

`demo-2` found a second symptom from the other end: `--resume` **duplicated the
whole subgraph** — 7 task records after one run and one resume
(`main:1, produce:2, describe:2, consume:2`) and 4 handoff slots instead of 2.
A non-leaf's main phase **is** `unfold`, so a root still `RUNNING` was re-entered
on resume and built a second subgraph beside the first.

**That half is closed** — `task_graph` `cc23f98` made `unfold` idempotent by
having `Task.has_subgraph` resolve `task_mgr` instead of asking the declaration.
Confirmed end to end through the CLI, which nothing else had done
(`scratch/impl-2026-08/demo/p6_where_does_resume_duplicate.py`):

```
after the first run   task records 4   handoff slots 2
after --resume        task records 4   handoff slots 2
```

**The root half was `monitor`'s, and it was a real defect.** `SUBGRAPH_DONE` was
declared, consumed and re-emitted, and **nothing anywhere produced the first
one** — a relay with no source. A non-leaf's `_main` ends its thread at `unfold`
with the task in `RUNNING`, and only the re-entry moves it, so every non-leaf sat
in `main: running` for ever. `04a5b76` adds the producer: an `is_end` subtask
announces its subgraph's completion when it runs out of phases.

**Re-measured with `04a5b76` in HEAD, and the numbers are unchanged** —
`final main: running` on both runs, `attempts` 2 → 4. **That is not evidence
against the fix.** `main.jsonnet` marks `consume` as `is_end`, and `consume`
ends every run in `waiting_handoff`, so it never runs out of phases and the new
producer has no opportunity to fire. **The precondition is unmet: `UNTESTED`,
not disproven** — the same distinction this package spends its exit codes on.
It becomes testable the first time `describe` succeeds, which is behind the
confinement wall.

So `--resume` still costs a full pass and buys nothing today, and the reason is
downstream of a subgraph that does not complete rather than of a root that
cannot.

#### The test that hid it is the sharpest instance of the week

`test_subtask_monitor_does_not_transition_parent` passed throughout **because it
constructs the `SUBGRAPH_DONE` by hand**. The test was the missing producer. It
proved the relay forwards — true — and was structurally incapable of noticing
that nobody upstream ever spoke. Same family as criterion 8's
`assert "..." in readme` and as our own resume test: **a guard that supplies the
input whose absence is the defect.**

#### Two wrong readings of why our test missed it, both mine or relayed by me

`demo-2` read `test_resume_continues_from_disk` as blind *because it resumes a
leaf*. I suspected a second reason: `{task.closure: task}` and `set(...)` over
its keys both fold duplicates away, so seven records under four names satisfy
the assertion. **Neither is the reason**, and `p5` measured it: `resume_all`
alone yields four records, on this tree and on the one that duplicated.

**The real gap is that the test stops at `resume_all` and never dispatches.** It
shows the state came back; criterion 12 says the run *continues*, and the second
subgraph appeared when the reloaded root was **dispatched**. The bug lived
entirely in the half the test does not do — and driving that half needs a real
run, which the suite must not require (spec §14.3).

The count assertion is in anyway, with the limit stated beside it: a guard whose
blind spot is named is worth more than one that looks complete. The drive is a
probe, kept, and it is where the property is actually checked.

### The transcript that named the walls

On this machine — Landlock ABI 3, `bwrap` absent, real credentials — `run` got
through confinement, the preflight, the package load, the dispatch, `main`'s
first phase, the unfold, and `produce`'s zone placement, and stopped at F-D12:

```
  confined  confinement is available and the backend answered: 'ready'    landlock
   package  loaded 1 task package(s) from .../examples/demo
      done  the graph stopped making progress 20 s ago; still in a phase: main:running
     graph  4 tasks: 1 root and 3 subtasks
     phase  produce: input validation runs nothing
     phase  describe: input validation runs 1
     ...
     final  consume: waiting_handoff — summary is created                  expected
     final  main: running
     final  produce: failed
UNEXPECTED  check_grounded records a failing verdict on the summary — did NOT
            happen. This is a FAILURE …
  expected  consume ends the run still in WAITING_HANDOFF, because of it —
            observed, as the demo promises
      done  run complete; 1 of 2 expected failures observed
rc=3
```

Ten of the sixteen criteria are visible in that transcript: the root and its
three subtasks (2), an empty input phase beside a populated one (3), one
dispatch per task with no validator anywhere (4), the consumer left waiting and
**reported as expected** (5), a program node and an SDK node (7), the
confinement mechanism named (9), and every verdict line carrying its taxonomy
when there are verdicts to carry it (10).

**`rc=3` is the point.** The remaining seams stop the graph, and the demo does
not report success. Note that **one of the two expectations *is* met** —
`consume` really does end in `WAITING_HANDOFF`. A lenient accounting would have
called that good enough; this one does not, and
`test_a_half_observed_expectation_still_fails` is why.

**Each fix moved the wall, and the next one was invisible until it did.**

| Wall | What it was | Owner | Hid |
|---|---|---|---|
| **F-D7** | nothing started a monitor's `mainloop`, so a phase advance landed on a queue nobody drained | `monitor` | F-D8 |
| **F-D8** | nothing called `set_task`, so the scope guard would have refused every advance | `task_graph` | F-D9 |
| **F-D9** | an **empty** validation phase deadlocked the task — three outcomes, two branches | `agent` | F-D10 |
| **F-D10** | a non-leaf never got a zone, so no subtask could be prepared | `env_mgr` | F-D11 |
| **F-D11** | confinement applied on the attempt's own thread, which then could not write its outcome | `env_mgr` | F-D12 |
| **F-D12** | a WRITE grant on an output kind cannot resolve: the version does not exist until the output is written | open | — |

**Six walls in one afternoon, each a defect in a different package**, and not one
findable from that package's own suite or while the previous wall was open.

Two more came from running the artefact rather than from hitting a wall:
**F-D13** — a failure recorded in one of the two places that should have it and
rendered by neither — and **F-D14**, a promise the run never reached reported as
a promise that broke. **F-D14 was in this package's own code**, one day after
reporting the same three-outcomes-two-branches shape to two others.

**That sequence is the argument for this artefact existing, made by the
artefact**, and it is the thing to point at if anyone asks whether the e2e test
could replace it. It could not: the load-time half runs in CI on every commit and
would have caught **none** of these. They need the run.

---

## The probes this README cites are not in your checkout

`agent_sys/scratch/.gitignore` is `*`, so `scratch/impl-2026-08/demo/` — three
probe scripts, kept as evidence per `docs/implementation-stage.md` §8 — exists on
the machine that ran them and **not in a fresh clone.** Following a citation here
gets you nothing.

That is deliberate on the repository's part and worth naming rather than leaving
for a reviewer to discover, because it is the shape this artefact spent the day
reporting: **a citation that looks like evidence and cannot be checked.** What
makes it survivable is that **every measurement is quoted inline** — the three
`sh` PATH rows, the three grant-resolution rows, the store's `detail=''` — so the
path is *provenance* and the numbers in this file are the artefact. If a
measurement here is ever load-bearing enough that a reader must re-run it, it
belongs in `tests/cli/` where a clone can execute it, not in a citation.

## Committing this package from a shared worktree

Eight agents share one checkout, so **the git index is shared too**. `git add`
*adds*; it does not *restrict*. A bare `git commit` then commits the whole
index, including whatever another session staged seconds ago — which happened
twice in one day, in both directions. The rule, ruled by `main` for everyone:

```bash
git add -N <any NEW files>          # intent-to-add: registers the path, stages no content
git commit -s -F - -- <your paths>  # pathspec on the COMMIT, not on the add
```

**The pathspec goes on the `commit`.** `git commit -- <paths>` takes those paths
from the *working tree* and ignores the index entirely, so it cannot pick up
another session's staging or your own stale staging.

**`git add -N` is the half the rule needs and does not state.** `git commit --
<paths>` only commits paths git already tracks, so **a new file is silently
omitted** — verified in a scratch repo: the modification landed, `pkg/b.txt`
stayed `??`, exit 0, no warning. For this package that means a **new test file
dropped from the commit while its subject lands**, and the suite is then green
because the assertions are not there. Verified that `add -N` plus the pathspec
gets both halves *and* still leaves another session's staged file out.

`-N` rather than a plain `add` on purpose: it registers the path without putting
content in the shared index, so a teammate's bare `git commit` in the window has
less of yours to sweep.

## Four things a reader should know before changing anything here

**A non-leaf unfolds inside `enter_phase(RUNNING)`, so a caller must not unfold
*and* submit.** Doing both gives the subgraph twice — four tasks become seven,
and the second `consume` waits for ever on a handoff nothing produces. `show`
and `--dry-run` unfold by hand and submit nothing; a run submits the root only
and lets the phase transition grow the graph. This was a live bug here and
`tests/cli/conftest.py::submitted` is where it is pinned.

**`Stream.emit`'s first two parameters are positional-only, and that is not
style.** `kind` is also what this system calls a handoff kind, so
`emit(HANDOFF_TRANSITION, "…", kind="facts")` is an ordinary thing to write —
and without the `/` it is a `TypeError` at the one call site a reviewer is
watching. Found by a test written to check the *renderer*.

**A stall is an ending, and `_settle` treats it as one.** A non-leaf sits in
`RUNNING` for as long as its subgraph is live, so if a subtask fails, full
quiescence never arrives and the timeout is the only exit — five minutes to be
told something that was true after twenty seconds. `_settle` also ends when no
task's status or attempt count has moved for `stall_after` and no attempt holds
a thread, and names which tasks are stuck where. Measured: F-D10 turns a 300 s
wait into a ~25 s one that says more.

**Interrupt during `produce`, not during `describe`.** Resume re-runs the
interrupted attempt, correctly — so an interrupt during the SDK task pays for a
second model call on restart. `produce` is a program and free to re-run, and the
demonstration is identical. The README in the task package says so to the
reviewer.

---

## Deviations from the design, and what is still open

| # | Design says | This does | Why |
|---|---|---|---|
| **D8** | §3's tree puts the validator logic at `logic/check_facts.py` | `logic/<name>/{readme.md, entry.sh, check.py}` | `validator` spec §9.1 makes a validator a **folder** carrying its readme, its entry and its material, and a body needs all three. One directory per validator is that shape; a bare `.py` beside a `.jsonnet` is not |
| **D9** | Nothing in the design mentions `validation_env` | `cli/main.py` registers it | F-D5. Without it a script-bodied validator has no `PATH`, and the demo's two validators are both script-bodied |
| **D10** | ~~§4.1 shows three subtasks and does not say what agent `main` has~~ | **RETIRED, 29 Aug.** `main` declares no agent | The deviation existed because *every task has an agent* forced a name for something that executes nothing, and `compose`'s whole description was that it executes nothing. `closure.schema.json` now requires `agent` of a leaf and of nothing else and `task_graph` supplies `SUBGRAPH_AGENT_SPEC`, so the hand-written spec is deleted rather than converted — a system-owned structural name is also *more* truthful than `compose`, since a run report printed `agent='compose'` for `main`, indistinguishable from a real executor |
| **D11** | §3's tree and `spec.md` §1.1 say the demo is *the only task package in this repository* | `examples/demo-broken/` is a second one | Forced, not chosen. `YamlPackage` scans every `*.yaml` under a root except `assets/`, so criterion 11's broken closure cannot live inside the demo without loading on every run and killing criterion 13. It was already described as *"a sibling package"* while being a subdirectory; it is now literally one |

Still open, and not this package's to close: `interfaces.md` §5.1b (nobody wraps
`materials` into a handoff), §5.4 (which reference kinds *who uses this*
enumerates), §5.6 (garbage collection between an artefact and its verdict), §5.7
(`SCHEMA_VERSION`'s owner once the whole-system CLI exists), §5.11 (under
bubblewrap an AI backend cannot be confined at all — this machine has no `bwrap`,
so the demo only ever exercises the Landlock rung), and design O1 (the demo's
first act is a `git config` write on the reviewer's checkout).
