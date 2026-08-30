# `agent` — what was adopted, what was built, and what is reported

| | |
|---|---|
| Implements | [`docs/spec.md`](docs/spec.md) rev. 6, criteria 1–16; [`docs/design.md`](docs/design.md) rev. 7 |
| Seam | [`../docs/interfaces.md`](../docs/interfaces.md) §4.4 |
| Tests | `../tests/agent/` — 131 passed, 1 xfailed |

---

## 1. Libraries adopted, and why

The rule is to prefer a mature, widely adopted library or CLI over writing it
yourself. Here is every third-party thing this package uses and the reason.

| Piece | Decision | Why |
|---|---|---|
| **pydantic v2** | **Adopt** | Already a declared dependency and `task_graph`'s own base. `AgentSpec` is nine keys with `extra="forbid"`, which is what makes criteria 5 and 14 structural rather than checked — there is nowhere to put a permission or a configuration knob |
| **`importlib.metadata`** | **Adopt** | `backend_entry`'s entry-point form (§6.3). The stdlib mechanism for "discover what exists"; nothing else is needed for it |
| **`claude-agent-sdk`** | **Adopt, as an extra** | The first AI backend. 376 MB installed and ~1.3 s to import, so it is `[project.optional-dependencies] claude` and is imported inside the constructor, never at module scope (§8.1). `tests/agent/test_claude_sdk.py::test_the_extra_is_not_imported_at_module_scope` walks the AST of every file in the package to keep it that way |
| **`asyncio`, `queue`, `threading`, `subprocess`** | **Adopt** | Stdlib. The loop, the message queue, the attempt's thread and the program executor's child |
| The selection chain | **Build** — ~40 lines, shaped by keyring | No library expresses a three-source chain returning a *structured* result. keyring, matplotlib, virtualenv and fsspec each supplied one property; §6 records which |
| The two protocols and the status enum | **Build** | Spec §4.3 fixes the six names and §1.1 fixes the split |
| `ExecutorBase` — the sugar layer | **Build**, ~120 lines | Spec §4.3's "every synchronous verb is sugar this layer wraps" is one rule, not a per-method convention, so it is one base class and not a mixin per verb |
| The three-phase loop and `TaskAttempt` | **Build** | It is `TaskRunner`'s whole job, and no prior art has our phase model. §7.5 records why there is no thread pool |
| The completeness gate | **Build**, `gate.py` | No prior art exists for it — every supervision system surveyed is crash-shaped — which is why it is this mechanical |
| The transform helper | **Neither** | Design §3.5: an independent module whose release cadence is the union of the harnesses'. `rulesync`, which does exactly this job for ~30 harnesses, has 299 releases and seven major versions inside two weeks |

**No dependency was added that was not already declared or already an extra.**
`pyproject.toml` gains one line — the `claude` extra — which
`docs/implementation-stage.md` §7 assigns to this package's §8.1.

### 1.1 One departure from the design's own file list

- **`gate.py`** — the completeness gate. `monitor` spec §4.1.0 names it as a
  thing, `monitor` design §8 assigns it here, and putting it in `runner.py`
  would bury a spec-level concept inside a 400-line file.
**Both of the modules that existed only because a seam was unsettled are gone.**
`events.py` went when `monitor` exported `event()`; `body.py` went when
`spec_loader` exported `Body` and `body_of`. Neither needed a decision in the
end — the other side closed each one, which is what reporting them was for.

---

## 2. What was decided where the design left a choice

| | |
|---|---|
| **`Selection.backend` is typed `Executor`, not `AgentBackend`** | A `kind: program` spec selects a `ProgramExecutor`, which has no level 2 by construction, and the runner holds level 1 only. `agent/protocols.py:191` narrows it to `AgentBackend`; that is §3's finding F5 |
| **`select_backend` gained `assignment=`, and it is required** | Design §7.2.1 lists the four things the executor gets and never says through which call. They arrive at construction, because the probe *is* the constructor (§6.4). **No default** — `interfaces.md` §4.11: a caller that omitted it would build an agent with no instruction, no entry and no zone, one that starts and does nothing (F9) |
| **Every adapter's constructor is `(key, config, assignment)`** | One contract, so `_probe` needs no per-backend knowledge |
| **`MonitorUnresolved` is raised** | `docs/interfaces.md` §2.1 rev. 4 says such a task "never advances a phase". It fails visibly instead: a task that hangs for ever and a task that says why are the same outcome told two ways, and only one is debuggable |
| **A required knowledge ref is fatal in both modes** | The only reading under which criterion 1 ("an unresolvable knowledge handoff is rejected at load") and criterion 2 ("missing knowledge warns by default") are both true. It is also the only reader `KnowledgeRef.required` has |
| **An executor that cannot start confined refuses the task** | The caller applies the confinement, and running anyway is the silent no-op the whole isolation chain exists to prevent. "No isolation, no start" reads the same for the caller's half of it — F7, F8 |
| **`TaskAttempt`, `carry_on` and `attempt_of` are declared in `protocols.py`** | `monitor` was duck-typing `attempt.executor` against a seam with one checkable side. Declaring them costs nothing and makes `tests/interfaces/test_stub_agreement.py` cover them. `resume` was declared too and has since been removed — F17 |
| **`AgentSpecRegistry` subclasses `BaseSpecRegistry`** | Landed by `spec_loader`, so the four methods rev. 1 wrote are deleted. Check 2 goes in `_validate` (before anything is stored) and the parsed model is cached in `_admitted` (only on the branch that stores) — `_validate` also runs on a byte-identical re-registration that then no-ops, so caching there would parse twice and cache a spec a later collision check rejected |
| **The default-monitor rule is `monitor`'s, and this delegates** | Two implementations disagreed about what an absent `monitor_spec` means — `monitor:default` there, "registered first" here. Latent under `build_registry`'s wiring and live the moment `monitors=[...]` names another first. The two are the ends of one conversation: this picks who a phase is *reported to*, `monitor`'s picks who an escalation is *sent to*, so disagreeing gives one task two watchers. Only the exception type stays ours |

### 2.2 `carry_on` — the operation, offered instead of the predicate

`monitor`'s `_advance` read `is_running` for exactly one purpose: choosing
between `wake()` and `resume()`. That is `engineer_principle.md` §3's stated
symptom — *a caller that reads `a.b.c`, branches on it, and acts* — and §4.4's
instruction is to offer the computation instead.

**The shape argument stands with the race removed entirely, and that is
`monitor`'s own correction to the position they first took.** They had declined
it as machinery for an interleaving nobody could reproduce; it is not machinery
for the race, it is a simplification that happens to close its dangerous half.

It returns `"woken"` or `"resumed"` so the caller can put it in the record.
**Reading it to record is not branching on it to decide.** Their design §6.1
argued against this collapse — *"it would hide, at the one place it matters,
which of the two shapes a task is"* — and they have withdrawn it, for a reason
worth keeping: **the branch never revealed the shape.** It revealed
thread-liveness, a *proxy* for leaf-versus-non-leaf, and it is that proxy which
was already wrong once.

**A plain string, not an enum**, and that is F3 applied one seam early: the only
caller may not import `agent`, so a member would reach them as a value anyway,
and a bare string compared against a member with `is` takes the wrong branch
silently.

`is_running` stays. Nothing of `monitor`'s breaks, and `carry_on` is declared on
`Runner` only — the attempt's own is internal, because nothing crosses the seam
reaching for it.

### 2.1 §5.5's attribution mechanism — **chosen: one fresh executor per phase**

`docs/interfaces.md` §5.5 split the question: `validator` §8.2 owns the
*requirement* (a phase must carry an `agent_id`) and this module's **O6** owns
the *mechanism*. It was reported open, and then it closed — not by testing the
four SDK candidates but because **the question turned out to be a different
one**.

Asking `validator` *whose* `agent_id` their assertion is about settled it. It is
**ours**, not the SDK's: `assert_attributable` takes any non-empty string and
nothing in that package touches an SDK field. So the mechanism needs no SDK
feature at all.

`agent/validator_executor.py` mints **one fresh, unbound `Agent` per phase**.
Better than the three SDK-flavoured candidates on three counts: it needs no SDK
feature, it works for a `kind: program` body as well as an AI one, and the
identity is real rather than derived.

**And asking the question found a defect in the other side.** `validator`'s
interim id was `f"{producing_agent_id}:{kind}"` — distinct per phase and *not a
distinct agent*, which their own §8.1 says is what criterion 10 wants. They
named the weakness in `40764ea` rather than leaving it implied, and it retires
when they take the id from this executor.

**What stays open is smaller and is not the mechanism.** Whether
`fork_session=True` leaves the main phase's session interruptible is still
unmeasured, and it matters only if the assertion ever becomes about the SDK's
`agent_id`. Nothing here precludes it: `ClaudeSdkBackend.session_ref` is still
the one place the SDK-session ↔ `AgentId` correspondence is recorded.

One candidate stays ruled out and the adapter shows why: `interrupt()` takes no
submission identifier and acts on the whole connection, so several `session_id`s
on one client cannot be it.

### 2.3 Which agent runs a validator — the spec's, then the wiring's

`ValidatorExecutor.agent_spec` used to be the only answer, fixed at the
composition root, so **every agent-bodied validator in the system ran as one
agent spec**. `validator` found it while checking their own criterion 9 and
`main` ruled it a decision separate from their §8.2 row 1, which arrives at the
same key by coincidence of representation. I assented to both.

`_agent_spec` is the whole of it — `spec.agent or self.agent_spec` — and each
half of that line was corrected by somebody before it was written:

- **`spec.agent`, not `getattr(spec, "agent", None)`.** `validator`'s
  correction. The field is declared, so a default could not fire; worse, a
  future removal would fall through silently and read as *nobody declared an
  agent* rather than raising.
- **`or` falls back on absent, never on unresolvable.** `closure`'s
  distinction, which `validator` adopted after first conflating them. Absent is
  a declaration — take the default. A name that is present and does not resolve
  reaches `_mint` and raises with the candidates listed, because falling back
  there would hand the author a **working** run in an environment they did not
  configure. `minLength: 1` in `spec-loader`'s schema is what makes the `or`
  unable to reach that case.

The constructor argument survives as a real default: an absent `agent` is legal
and is the ordinary case. Four packages in sequence, and I went last on purpose —
writing the read before `validator`'s field existed would have made my own stub
its only caller, which is §6.1.1's failure class exactly.

---

## 3. Interface problems found, with both sides named

`docs/interfaces.md` §1.1: *do not change a cross-module signature quietly.*
None of these was changed here.

| | Side A | Side B |
|---|---|---|
| **F1 — `EnvManager.prepare` arity** | `docs/interfaces.md` §4.6 and `env_mgr/protocols.py:261`: `prepare(task, execution)` | `env_mgr/docs/design.md` §11.1 and §11.4: `prepare(task, execution, agent_spec)`, which §11.5's `material.deploy(agent_spec, zone)` needs |
| | **Resolved, and neither side had to change.** `env-mgr` shipped `prepare(self, task, execution, agent_spec=None)`, so the frozen two-argument call still works. The runner passes the spec — without it, `agent` spec §3.1's `env` and design §3.4's `rules` / `hooks` / `skills` have **no consumer at all**, which is four spec keys an author can write and nothing ever reads. `test_prepare_is_given_the_agent_spec` | |
| **F2 — `closure.body_of` is unreachable** | `agent/docs/design.md` §7.2.1 walks the route as `closure.body_of(closure.task_of(spec))` | `closure/protocols.py` exports six accessors and **no `body_of`**, though `closure/docs/design.md` §3.5 declares it; and `docs/interfaces.md` §4.4 plus `tests/interfaces/test_import_rules.py::ALLOWED` forbid `agent` importing `closure` at all |
| | **Closed.** `Body`, `body_of` and `subgraph_of` are in `spec_loader`; `agent/body.py` is deleted and `runner.py` imports the leaf's. The argument that decided it was `closure`'s: `_common.schema.json` already held **one** `$defs.body` that both `task.schema.json` and `validator.schema.json` `$ref`, against three Python types. One consequence worth knowing — `body_of` returns the mapping **as written**, so a task with no body is `{}` (falsy) rather than `Body(readme="")` (truthy, and reports a body that is present and empty) | |
| **F3 — `validator.PhaseKind` cannot be named** | The seam is `run_phase(kind: PhaseKind, task, registry)` | `agent` may not import `validator` |
| | **The value is passed instead.** `PhaseKind` is a `(str, Enum)`, so a `run_phase` comparing by `==` works and one comparing by `is` does not. `validator` should be told | |
| **F4 — `EventRecord` had no factory** | `monitor` design §8 requires the runner to build and `report()` a record at every phase boundary | It was written against a `monitor` that was declaration-only, so `agent/events.py` resolved the class by name and fell back to a local model of §3.3's shape |
| | **Closed the same day, by the other side.** `monitor` now exports `event(kind, task_id, **fields)` and `EventRecord` self-fills its fingerprint, so `agent/events.py` and its drift test are deleted and the runner calls `monitor.event`. Recorded because the shape recurs: a wave-1 package that has to *build* a neighbour's value type needs the neighbour to export a factory, and the seam file listed only the Protocol | |
| **F5 — `Selection.backend`'s type** | `agent/protocols.py:191` and design §6.1: `AgentBackend` | Criterion 15 and design §9 require a `ProgramExecutor`, which is `Executor` and deliberately not `AgentBackend` |
| | The concrete `Selection` annotates `Executor`. The `.pyi` was not edited | |
| **F6 — `Recorder` is registered by no name** | `monitor` design §9.2 assigns `Recorder.open(task, attempt)` at attempt start to `agent` | `docs/interfaces.md` §2 registers no `recorder` |
| | **Closed, and `monitor` supplied the decisive argument**: §2 line 129's `install_excepthook(recorder=...)` is a literal `...`, so the root must build a `Recorder` anyway and registering it is finishing an unfinished line rather than adding a row. My alternative — open-on-first-use — is withdrawn: `Recorder.write` already calls `open()`, so the only case left uncovered is the attempt that reports *nothing*, which is the sole case the marker exists for. The skip is now a raise | |
| **F7 — `Prepared` drops the deployed environment, and cannot apply rung 1** | `env_mgr` design §11.5's `material.deploy` computes `CLAUDE_CONFIG_DIR`, `CLAUDE_CODE_TMPDIR`, `TMPDIR` and the agent spec's `env`; and on rung 1 bubblewrap **is** the exec, so `apply()` confines nothing and the caller must run `bwrap_argv(policy, availability, argv)` | `env_mgr/protocols.py:247` — `Prepared` is a five-field frozen `NamedTuple` carrying neither. `bwrap_argv` also needs an `Availability`, which is on neither `Prepared` nor anything `agent` may import |
| | **Both fields shipped (`b96fb80`), and the runner uses them.** `wrap_argv` is on `Prepared` rather than `Confinement`, which is right: building a command line needs the policy too, and the `bwrap` binary is resolved at exec rather than remembered from probe time. **What remains is the AI half, and it is not closable by either package** — see the row below. Original finding: |
| | **Both halves matter and one of them fails silently.** First: measured, with `~/.claude` granted a confined demo agent read the *operator's personal* `CLAUDE.md` and obeyed its language rule, so `CLAUDE_CONFIG_DIR` pointing into the zone is what removes the `$HOME` grant. Second: skip the wrap on a bwrap machine and `prepare` succeeded, the task ran, and there was no sandbox. **The runner reads both defensively and refuses to start when a bwrap confinement has no wrapper** (`ConfinementNotApplied`), so the silent case is loud. Two fields to agree with `env-mgr`: `Prepared.environment` and `Prepared.wrap_argv`. `bwrap` is absent here, so nothing local would have caught it | |

| **F8 — an executor that cannot start confined must not start, and it is now every mechanism** | `interfaces.md` split step 7 (`b846c3c`): `prepare` **checks** a mechanism exists, `prepared.spawn(argv, **kw)` **applies** it in the child | `ClaudeSDKClient` spawns the `claude` CLI itself, so **there is no child of ours for `spawn` to start** — and since `prepare` no longer confines the runner's process, that CLI is not confined by inheritance either |
| | **I first reported the split as widening this, and `env_mgr` corrected it: no capability was lost.** Pre-split, Landlock confining the runner's thread did mean a self-spawned CLI inherited the domain — but only from a single-threaded caller, and `apply()` refused above one thread, which `agent.Runner` always is. **So in any configuration that could actually run, an AI task was never confined.** The outcome is unchanged either side of the split — the task does not start — and only the message differs. F8 did not get worse; it got honest. |
| | **And it is soluble, which "unconfinable" wrongly implied.** `env_mgr` measured that **a grandchild inherits**: a child confined the way `spawn` confines, spawning a grandchild itself with no wrapper, is denied (`rc=13`) — which is exactly the shape of an SDK spawning its own CLI. So the real statement is *an AI task cannot be confined **in-process***. Running the harness inside a `spawn`-ed child confines the CLI with no shim, no argv interception and no cooperation from the SDK. **The cost is level 2**: `interrupt`, `instruct` and `query` are built on an in-process `ClaudeSDKClient`, and `monitor`'s `Pushable` is that handle — out-of-process, all of it needs an IPC channel that does not exist. Roadmap, not alpha. `test_an_ai_agent_cannot_be_confined_under_any_mechanism` still asserts today's architecture honestly and is not weakened | |
| | **Three versions, and the first two were wrong in the same direction — worth recording because the shape recurs.** (1) *"Does a wrapper exist?"* refused the honest case (no wrapper at all) and admitted the dishonest one (a wrapper this executor cannot apply); `wrap_argv` landing on `Prepared` silently disarmed it, because a bound method is truthy. (2) *"Is `AgentSpec.kind` ai?"* — but the kind is a **proxy** for the executor, and a CLI override resolves a backend entry in its own right (design D6, deliberately), so a `kind: program` spec pinned to an AI backend passed and ran unwrapped. **(3) Ask the executor.** `ExecutorBase.accept_confinement` refuses by default and `ProgramExecutor` overrides it — which is spec §3.3.1's own shape (an unimplementable method raises) and **not** the capability matrix §3.3.1 forbids, a distinction the docstring states because the two look alike. Found by `closure`'s review; measured at `scratch/impl-2026-08/agent/probe_r1_override.py`. `interfaces.md` §5.11 still holds whether bubblewrap is meant to cover AI tasks at all | |
| **F9 — `select_backend`'s fourth parameter was undeclared, with a default** | `agent/protocols.py` declared `(spec, *, override, config_order)` | The implementation had `assignment: Assignment \| None = None`, and the probe *is* the constructor — so a caller using the declared signature built an agent with no instruction, no entry point and no zone, and **it would start and do nothing** |
| | **Declared, and the default dropped** — `interfaces.md` §4.11 verbatim, and the fourth instance of a fallback whose reason had expired. `closure` found it and named the reason their own signature test had missed the same shape: *a signature test that compares parameter names and not defaults is not a signature test; the default is the drift* | |

| **F10 — a package-relative `entry` did not resolve** | `_common.schema.json` types `entry` as a *"package-relative path to the entry.sh"*, and `validator.ScriptBodyRunner` joins it against a `package_root` its constructor takes | `ProgramExecutor` ran it with `cwd` set to the zone and nothing carried the package root into this package, so **the two consumers of one schema key disagreed** |
| | **Fixed here**, the way `validator` already did it: `Runner(package_root=)`, joined in `resolve_path`. An absolute path is unaffected, so a package that renders its body paths absolute — which `demo` does — keeps working. Found by `demo` F-D3 while assembling all eight | |

| **F11 — one attempt could hold two threads, and `is_running` is read unsynchronised** | `monitor` checks `attempt.is_running`, then either `wake()`s or `resume()`s | `begin()` had no guard, so `resume` on a running attempt started a **second** thread — both would run phases, report, and call `on_done`. And the predicate is mutated on three threads (`release` on the attempt's, `halt` on the scheduler's, `begin` on the monitor's) and read on a fourth |
| | **The two halves are different in kind and only one is fixed.** The second thread was **not a race**: an unconditional missing precondition on a public verb, firing every time — `begin()` now raises `ThreadAlreadyHeld`, and the attempt's own `RLock` makes `is_running` read `_thread` and `_halted` together instead of torn. **The check-then-act race is not closed**, and neither `closure`, `monitor` nor I can construct an interleaving: a parked leaf is waiting on the very wake in question, and a non-leaf released at `_main` long before `SUBGRAPH_DONE` arrives. What would close it is a whole operation rather than a guard — see §5 |

| **F12 — an empty validation phase deadlocked the task** | `PhaseOutcome` distinguishes three states: passed, empty, failed. `empty` is correctly **not** a pass — four systems reached that independently | `_validation` wrote **two arms**, so empty fell into the failure arm: it reported `VALIDATION_UNREACHED` and ended the thread with the task still in its phase, for ever |
| | **Fixed: an empty phase advances, carrying `evidence: nothing_ran`.** Found by `demo` on the first live run of all eight — `main` sat in `INPUT_VALIDATING` for 300 s, and the monitor was correctly escalating a report that should not have been made. **It is the ordinary case, not an edge**: every graph's first task has no inputs. Every route to `empty` that `validator` can produce is a legitimate advance and I checked rather than assumed — no validator bound, or the phase switched off by `--validation-strict-level`, which emits a `SkipRecord` per validator. Blocking on the second would make the knob reach a *verdict*, which is the property `validator`'s fold exists to prevent. **My own test had encoded the defect** and was green throughout; it is rewritten, not repaired |
| **F13 — `VALIDATION_UNREACHED` had the wrong producer, then none** | `monitor` spec §2.1: the kind is for *"no verdict **reachable** — its `entry.sh` crashed, its agent died"* | I had mapped it onto `PhaseOutcome.empty`, which is *"there was nothing to check"*. F12 removed the wrong producer and left the right one unwritten |
| | **Neither of the two readings I offered was the case — `monitor` supplied a third.** `validator/phase.py` raises in seven places, nothing here caught it, so it reached `_crash` as `HANDLING_FAILED`; their pusher routes that to `GiveUp` where the right kind routes to `Escalate`. **A crashed validator died at its own monitor instead of reaching the user** — the quietest possible dead branch, which is the defect §2.1 exists to close — and `HANDLING_FAILED` (*the monitor's handler raised*) was a wrong value flowing on besides. Fixed by catching **any** exception out of `run_phase`: that is §2.1's sentence exactly, needs no answer to *"is `ValidatorInvalid` the complete set"*, and escalating an unexpected error is the safer direction. `_crash` keeps `HANDLING_FAILED` for everything else | |
| **F15 — I was told to import `validator` and may not** | `monitor` proposed `except ValidatorInvalid` for F13 | `docs/interfaces.md` §4.4 and `test_import_rules.py` give `agent` `{spec_loader, task_graph, monitor}`. `validator` is not in it |
| | Not imported, and **not string-matched on a class name either**, which would have been worse than both. Catching the behaviour rather than the type is better than the suggestion regardless of the rule. **This is F3's shape a second time** — a neighbour reasoning about my imports from their own permissions — and it is worth a rule rather than a third instance | |

| **F18 — a non-leaf never got a zone, so its children could not be placed** | A subtask's storage nests inside its parent's — `env_mgr` criterion 2, and `layout.create` raises naming the parent | `_main` returns at `has_subgraph()` **before `_deploy`**, which is `prepare`'s only caller, so a non-leaf had no zone ever and **no nested graph could run** |
| | **Fixed with `env_mgr.place_zone`** (`6fa6a6e`), which they built when I asked: `prepare`'s first step and none of the rest. Calling full `prepare` was the other candidate and is wrong three times — a `git clone --shared` for a task that never executes; a refusal, since `apply()` rejects more than one live thread; and **`env_mgr` measured the one I could not**, that the confined thread can no longer write outside the zone, so the attempt could not record its own outcome. Found by `demo` live; nesting is the one thing their spec §2 exists to prove | |
| **F14 — nobody called `Monitor.set_task`** | `set_task` is the only way a monitor learns what it watches, and `_run_guarded` raises `ScopeViolation` for a task it was never given | Nothing anywhere called it, so **every planned advance raised** and no task advanced a phase. `demo` found it on the live run |
| | **Not here — `Scheduler._dispatch_pass`** (`interfaces.md` §2.7, committed at `5f8c0f3`). Four positions, three rounds of both packages having built it, and safe every time only because `_Watch.add` dedupes — which neither of us designed for. **The argument is about neither site**: `TaskRunner` declares `start` and `stop` and says nothing about monitoring, so a `set_task` in a runner implementation is a per-implementation obligation of a contract that does not state it. `FakeRunner` is not a counterexample — *a double that does not participate escapes the obligation by not having the collaborator*, which says nothing about whether a real implementation should be obliged. Mine is removed and `test_the_runner_does_not_call_set_task` is what stops a fourth round; the two sentences worth keeping — `set_task` is idempotent, and an unresolvable `monitor_spec` raises because that is `monitor_for`'s to say — are carried in a comment at their call site | |

| **F16 — `AgentSpecRegistry.check_knowledge` has no caller** | Spec §3.6 checks 3 and 4, and **criterion 2's whole mechanism**: missing knowledge warns by default and is fatal under the run-config flag | Nothing in any package calls it. My unit tests call it directly, which is why it is green and why nobody noticed |
| | **So the knowledge pass never runs in an assembled system.** Same class as `demo` F-D8 — code that exists, is tested, and is invoked by nobody — and found by applying `monitor`'s own stub audit to my public surface. It needs a caller at the composition root, after `load_package`, with the handoff registry in hand; `build_registry` is `task_graph`'s and I have not edited it. Reported | |
| **F17 — `Runner.resume` and `TaskAttempt.wake` lost their only caller** | Both are declared in `protocols.py` and were `monitor`'s route for the non-leaf re-entry | `carry_on` subsumed them, and `monitor`'s Protocol shrink then stopped requiring them. Nothing outside `agent`'s own tests calls either |
| | **Removed**, jointly with `monitor`, who enumerated all twelve of their spec §7.1 actions and both open questions: **no path wants a bare re-entry without the wake-or-resume decision** — a replacement monitor adopting a dead one's tasks wants `carry_on` *most*, being the caller least able to know whether a thread is parked. Removed on the plain ground that it was a published promise nobody used; the stronger-sounding argument — that publishing it invites a caller to decide which shape a task is — is bounded by `ThreadAlreadyHeld`, and `monitor` said so **against their own case**. `wake()` and `is_running` stay: `carry_on` is built on them. **Two `resume`s in this system and only one is gone** — the claude-agent-sdk session resume, ~5.5 s warm and losing `permission_mode` / `--mcp-config` / `--settings` / `--add-dir`, is untouched | |

| **F19 — a confined `kind: program` task cannot read its own `entry.sh`** | `Assignment.entry` is a **package-relative** path joined against `Runner(package_root=)` — the task package, which is not the zone | Nothing stages the task **body** into the zone. `env_mgr` stages handoffs and `material.deploy` places `rules`/`hooks`/`skills` under `<zone>/config/`; `closure` spec §2.6's `readme.md` and `entry.sh` are placed by nobody |
| | **Measured, end to end, first run of the real seam** — `scratch/impl-2026-08/agent/probe_end_to_end_spawn.py`, real `EnvManager`, real `prepare`, real `spawn`, real `ProgramExecutor`, Landlock ABI 3: `entry.sh` outside the zone → `cannot open …: Permission denied`; inside the zone → runs, and `home-read=DENIED root-write=denied home-write=denied zone-write=yes`. **So the confinement works and the body is unreachable** — same shape as M3, one artefact over. Either the package root is granted read-execute or the body is staged into the zone like a handoff; the second is more consistent, since the zone is what an agent reaches. Not mine alone: the grant is `env_mgr`'s and the body is `closure`'s | |

| **F20 — every failed task had an empty `Execution.detail`** | The field is documented *"from the runner; for a human"*, `Scheduler.on_task_done` has always taken `detail`, and `_crash` has the exception in hand | `OnDone` was `Callable[[TaskId, TaskStatus, dict[str, float]], None]` — **the declared type could not express the argument the implementation accepted** |
| | Measured by `demo` on a live run: `detail=''` while the same exception sat complete in the monitor's record. **Not fixed by passing the keyword**, which would have worked in production and broken every conforming callback — F3's shape and F9's, a third time. `task_graph` widened `OnDone` to a Protocol (`f1faf74`) and I pass `f"{type(exc).__name__}: {exc}"`. That form is measured, not preferred: `str(KeyError('agent'))` is `"'agent'"`, a bare quoted word in the field a human reads first — and the joined form is exactly `exception_type` + `exception_message` from the recorder, so the two renderings are one fact | |

---

## 4. Criterion → test

Every one of the 16, and each test exists and passes.

| # | Test | File |
|---|---|---|
| 1 | `test_load_rejects_unknown_backend`, `test_load_rejects_unresolvable_knowledge` | `test_registry.py` |
| 2 | `test_knowledge_missing_warns_then_fatal` | `test_registry.py` |
| 3 | `test_selection_precedence` (parametrised over the three sources), `test_cli_override_does_not_fall_through`, `test_first_available_in_declared_order_wins` | `test_selection.py` |
| 4 | `test_unsupported_method_raises`, `test_a_program_executor_has_nothing_to_raise_from` | `test_backend.py` |
| 5 | `test_agent_spec_has_no_permissions_field`, `test_same_spec_two_tasks_two_reaches` | `test_spec.py` |
| 6 | `test_backend_is_not_a_runner`, `test_runner_holds_level_one_only`, `test_runner_unchanged_across_backends` | `test_runner.py` |
| 7 | `test_start_async_returns_before_started`, `test_start_equals_async_plus_wait` | `test_backend.py` |
| 8 | `test_status_sequence`, `test_task_status_is_superset` | `test_backend.py` |
| 9 | `test_interrupt_drains_before_next_query`, `test_the_drain_is_bounded` | `test_claude_sdk.py` |
| 10 | `test_instruct_does_not_end_run` | `test_claude_sdk.py` |
| 11 | `test_query_history_session_matches_agent_id`, `test_the_session_ref_is_learned_from_messages_not_from_the_client`, `test_query_before_anything_ran_is_empty_not_an_error` | `test_claude_sdk.py` |
| 12 | `test_no_interface_reaches_a_subagent` | `test_backend.py` |
| 13 | `test_material_stored_canonically`; **`test_transform_lossless` — `xfail(strict=True)`, O1** | `test_spec.py`, `test_transform.py` |
| 14 | `test_backend_has_no_configuration_method` | `test_backend.py` |
| 15 | `test_swap_backend_same_handoff_state` | `test_runner.py` |
| 16 | `test_records_hold_no_prompt_text` | `test_records.py` |
| — | **`validator_executor`**, which belongs to no criterion of ours: `test_each_phase_runs_as_a_different_agent_from_the_producer`, `test_the_phase_agent_is_unbound` | `test_validator_executor.py` |

**Criteria 9 and 11 do not need the extra.** The adapter takes its client from
`config["client"]`, which is the same seam a third party uses to pin a
pre-configured handle — so the two tests the design marked `claude` and
`skipif` run everywhere instead, against a fake transport. Nothing in
`pytest agent_sys` needs a credential, a network, or a sandbox.

**Criterion 13's `xfail` is `strict=True`**, so a green result is a failure.
O1 is not answered by making the test pass.

### 4.1 Four tests that belong to no criterion, and are here anyway

Each guards a seam obligation that **fails silently** if it is skipped — the
category `materials/00-architecture.md` §7 names, where a document says X
consumes Y and nobody checks that X's signature can accept Y.

| Test | What would otherwise be silent |
|---|---|
| `test_prepare_is_given_the_agent_spec` | `rules` / `hooks` / `skills` / `env` deployed by nobody: four spec keys an author can write and nothing reads |
| `test_prepare_is_called_once_per_attempt_with_the_execution` | A retry reusing the first attempt's granted set, because `N` in `<root>/<hid>/v<N>/` lives on the `Execution` |
| `test_a_refused_environment_never_reaches_an_executor` | A `try` that logged a `NoConfinement` and continued: the system reports the agent is sandboxed while it runs with the operator's privileges |
| `test_the_runner_refuses_a_bwrap_confinement_it_cannot_apply`, `test_an_ai_agent_under_bwrap_refuses_to_start` | Rung 1 applied by nobody. `bwrap` is absent on this machine, so these are the only things that would catch it |
| `test_a_missing_recorder_is_loud` | Criterion 14's empty-versus-missing distinction voided by a wiring gap, silently |
| `test_is_running_tells_a_parked_leaf_from_a_released_non_leaf` | A non-leaf's re-entry stalling for ever, because `wake()` on a released attempt sets an `Event` nobody waits on |
| `test_a_phase_that_raises_is_unreached_not_a_handler_failure` | A crashed validator routed to `GiveUp` instead of `Escalate`, so the branch it killed was the quietest kind of dead branch |
| `test_a_task_the_scheduler_watched_can_report`, `test_the_runner_does_not_call_set_task` | Every planned advance raising `ScopeViolation`, so no task advances a phase — `demo` F-D8, invisible to eight unit suites because every stub `set_task` was a no-op. The call is `Scheduler._dispatch_pass`'s; these assert the fixture models it and that a fourth round of building it here is a red test |
| `test_the_schema_and_the_model_both_admit_it` / `..._reject_it` | The schema and `AgentSpec` drifting apart. `spec_loader` owns `agent.schema.json` and this package owns its content, so they are two records of one shape — and the first version invented `{type, handoff}` for a knowledge reference against the model's `{kind, knowledge_type, required}`, which under `additionalProperties: false` would have rejected **every knowledge-bearing agent spec in the system**. Caught by `spec-loader` asking; this makes the next one mechanical |

### 4.2 The first run against a real backend, 2026-08-29

`claude-agent-sdk` 0.2.148 was installed on this machine and the adapter was
driven against the live gateway for the first time. Probes kept in
`scratch/impl-2026-08/agent/`, run with every `ANTHROPIC_*` variable scrubbed
so nothing is an artifact of the operator's own shell.

**It works.** `ClaudeSdkBackend.start()` returned `FINISHED` / `'success'`,
`duration_ms=5178`, `num_turns=2`, real cost. **Three defects were invisible
until it did**, and all three share one cause: *the adapter guessed the SDK's
surface and the test double ratified the guess.*

| | measured | fixed |
|---|---|---|
| `session_ref` was `None` on **every** real run | `ClaudeSDKClient` has no `session_id` attribute at all — `hasattr` is `False`. The id is on the messages | recorded as messages stream past, one writer |
| `query()` raised `AttributeError` on **every** real run | `get_session_messages` is a **module-level function**, synchronous, taking `(session_id, directory=...)` — not a client method | called as the function it is; the zone is the project directory |
| a failed run recorded `detail='success'` | the SDK sets `subtype='success'` even with `is_error=True`; `terminal_reason='api_error'` is what distinguishes the cases | `_detail_of` consults `terminal_reason` **on failures only**, so successes still read `'success'` |

The old `FakeClient` defined `session_id = "sess-abc"` and an async
`get_session_messages`. **Neither exists.** The suite was green against a
fiction, which is why the double is now forbidden to re-invent them
(`test_the_session_ref_is_learned_from_messages_not_from_the_client` asserts
`not hasattr(client, "session_id")`).

**Credentials need no threading, measured.** The CLI reads
`~/.claude/settings.json` itself: with `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`,
`ANTHROPIC_CUSTOM_HEADERS` and `ANTHROPIC_MODEL` all absent from the
environment, a query still returned in 0.79 s. And
`ClaudeAgentOptions.env` is **merged over** `os.environ`
(`subprocess_cli.py:809`), not a replacement — so passing
`Assignment.environment` as `env` strips nothing the CLI needs.

### 4.3 What a session transcript cannot tell you

**Three people concluded a defect from a `.jsonl` on 2026-08-29 and all three
were wrong in the same direction.** Recorded here because the file looks
authoritative, is easy to grep, and answers a narrower question than anyone
assumed. Probe:
`scratch/impl-2026-08/agent/probe_transcript_records_system_prompt.py`.

| grepped for | in the transcript | so a zero means |
|---|---|---|
| the **system prompt** | **never recorded** | nothing. The brief may have arrived and worked |
| an **environment variable** | **never recorded** | nothing. The agent may have had it |
| **idle since timestamp T** | writes stop before the turn does | nothing. *Transcript idle is not conversation over* |

**The control is what makes the first two conclusive.** A system prompt of
*"whatever you are asked, reply with exactly the single word BANANA"*, asked
`What is 2+2?`, answered `BANANA` — so the prompt demonstrably arrived and took
effect — **and the marker appears nowhere in the file**, across six line types
(`queue-operation`, `attachment`, `user`, `last-prompt`, `atis-latch`,
`assistant`). Reproduced three times.

The third is `agent-mod`'s, from the process table rather than a probe: a CLI
alive at 14 minutes with its transcript quiet for 5. It nearly became a
`DRAIN_SECONDS` defect report against this package.

> **A negative grep is indistinguishable from a grep for the wrong name** — and
> here the file itself is the wrong name. The process table answered all three
> questions the transcript could not.

---

## 5. What is not built, and why

| | |
|---|---|
| The transform helper | §1's last row, design §3.5. Criterion 13's second half is O1 and is not testable as written |
| ~~Resolving a body's paths against the **staged** package~~ | **Built, `c43dcba`.** §4.16 left the original tree outside every grant, so `/bin/sh <entry>` was denied — `demo` measured `exit 2: … Permission denied`, visible only because `detail` travels now (`8beb4f9`). `Runner.resolve_path` could not be fixed in place: `package_root` is a constructor argument and the staged copy is per attempt, so resolution moved into `_deploy` against `Prepared.staged_package` (`env_mgr` `086c12e`), with `resolve_path` as the unprepared fallback. **`readme` was the worse half**: the schema calls it a path, nothing here read the file, and `claude_sdk.py:221` handed the string to the SDK as `system_prompt` — a `kind: ai` task's brief was the path to its brief. It never crashed and no AI task had run. Now read; a missing file raises rather than falling back (§4.11) |
| A `human` executor | Spec §9. `kind: human` loads and fails at selection, which is the honest outcome |
| Mid-run backend fallback | O7. §3.3's "pins the whole run" implies none, and every surveyed project except LiteLLM agrees |
| Relocating the SDK's transcript | O3. `~/.claude/projects/<encoded-cwd>/*.jsonl` lands outside the confinement zone; three levers exist and the choice is `env_mgr`'s |
| Choosing which `claude` CLI runs | **O2 — closed 2026-08-29, and it was a real defect.** Measured: `_find_cli()` returns the SDK's bundled executable *before* it ever calls `shutil.which`, and the two are **different binaries at different versions** — `_bundled/claude` is 2.1.251, the `PATH` one `env_mgr` installs plugins into is 2.1.246. Both work, so the run succeeded and the agent silently lacked its own recipe's plugins. `env_mgr.Prepared.agent_cli` reports the CLI the environment was provisioned for — **declared** on the `Context`, not re-resolved, because `PATH` can differ between the process that ran the recipe and the process that runs the agent. The runner carries it onto `Assignment.agent_cli`; this backend pins it, and **refuses** when a prepared run reports none rather than falling back (`interfaces.md` §4.11). `tests/interfaces/test_agent_cli_seam.py` pins the field from both sides. **The first version of this row described an env var, `AGENT_SYS_CLAUDE_CLI`, that `env_mgr` never published** — and its guard asserted the literal against a copy of itself, so it stayed green while `material.deploy`'s `CLAUDE_CONFIG_DIR` made every prepared AI run take the refusal branch. `ROADMAP.md` §9.2 |
| ~~Closing F11's check-then-act race~~ | **Built** — `Runner.carry_on(task_id)`, §2.2 |

---

## 6. What is owed, and to whom

**Everything this package needs from elsewhere is a ruling, not work.** Nothing
here is blocked; each item costs one line or less on this side. Recorded because
a decision waiting in somebody's context is a decision that gets rediscovered.

| Owed | To / from | State |
|---|---|---|
| **F-D1 — who publishes a handoff** | three packages, `main` rules | Nothing calls `HandoffStore.put`, so the completeness gate reports `OUTPUT_ABSENT` for every task in an assembled run. My original position — *the producer calls `put` from inside its own zone* — is **dead**: `env_mgr` spec §4.5 forbids an executor writing outside its zones, and `put` writes to the store. The answer is `env_mgr.workspace.collect`'s shape — **the agent writes into the zone, the gate checks the zone, the runner publishes after** — which keeps my objection that a publishing runner must not check its own publication. Requires `monitor` §4.1.1's sentence to be withdrawn |
| **F19 — the task body under confinement** | `env_mgr`, `main` rules | A confined `kind: program` task cannot read its own `entry.sh` (measured). **Recommendation: grant the package root read-execute, derived**, not stage — staging is the *less safe* option, because a staged copy lands in the writable zone while a grant is read-only, and `demo`'s launcher body proved the immutability argument for staging does not hold anyway |
| **F17 — `Runner.resume` removed** | `main` | `interfaces.md` line 697 still names it as *"owed and undeclared"*; it is neither |
| ~~**`ValidatorSpec.agent`**~~ | **closed** | Landed as step 4 of four, after `spec-loader` `fe9fd55` and `validator` `ec5fbba`. See §2.3 |
| **`minLength: 1` on `validator.agent`** | `spec-loader` holds it; shared with `validator` | A **known edge, deliberately not defended against.** `_agent_spec`'s `or` is safe from falling back on a *present* name only because the schema forbids an empty one; `validator`'s `_bound_environment` returns `None` for absent and raises for unresolvable for the same reason. Relax the clause and `""` becomes "absent" in **both** packages, quietly. Neither suite would notice. The schema is the enforcement point and duplicating it here is how two copies drift — so it stays a recorded edge rather than a defensive `if`. `validator` is telling `spec-loader` with both halves, so they hear one dependency and not two footnotes |

### 6.1 `done_by_self_check` — the deadline is gone and the blocker moved

**The field exists nowhere.** `monitor` spec §4.1.2 specifies it and §9 carries it
as an unbuilt `handoff` propagation item; `agent/gate.py::_self_check` reads it
defensively and reports only when it is **present and false**, so the check is
written and dormant. `tests/agent/conftest.py::StubManifest` is the only place in
the tree it appears as a field.

**Two corrections to what this section used to say**, both from `handoff` and
both worth keeping visible rather than overwriting:

1. It claimed *"there is no manifest when the gate runs"*. Measurably wrong, and
   contradicted by my own code — `gate.py:90` calls `store.get_manifest`. I had
   asserted F-D1's proposed shape as though it were settled. It is not.
2. The deadline is gone. It read *"the window closes when `handoff` builds it on
   the manifest"*; `handoff` has now declined to build it there, and for a
   better reason than the one either of us had.

**The blocker is that no producer has a channel to claim it.** `handoff`'s
enumeration is complete: `False` blocks every existing producer, `True` makes
the check lie, `None` makes this file's tolerance clause permanently untrue so
the guard could never be tightened. Their answer is `seal(..., done_by_self_check:
bool)` as a **required** keyword, on the agent-produced path only.

**My side of that: no such channel exists.** `AgentResult` is `status` / `usage`
/ `detail`, and `detail` is a human string — routing a claim through it would be
parsing prose. The SDK's `ResultMessage.structured_output` would be typed, but
`ProgramExecutor` has no equivalent, so building on it gives a capability one
executor has and the other does not, which is the matrix spec §3.3.1 forbids.
**A zone artefact is the only candidate that works for both bodies.**

**And a required `bool` relocates the default rather than removing it** — the
caller must answer for *the agent wrote nothing* too. The resolution is that
`seal` never sees that case, because the gate blocks first, and the failure kind
already exists for it: absent → `SELF_CHECK_UNSET`, present-and-false →
`SELF_CHECK_UNSET`, present-and-true → the gate passes and the runner has a real
`True` to hand `seal`. Under that shape `_self_check`'s tolerance clause is one I
**delete** rather than one that rots.

**Who reads the artefact and calls `seal` is F-D1, still unruled.** Nothing here
calls `put`, `allocate` or `seal`. I am not writing a reader for a file nothing
writes — §6.1.1 is what that costs. The open interface is the artefact's **name
and shape**, which belongs to three packages and currently has no owner.

#### The shape, settled 2026-08-29 — and then deliberately not built

**Recorded so the next person does not re-derive it.** Three agents have now
spent real effort here; the design question is answered and the *build* is
stopped, and those are two different states.

`main` first ruled a **producer branch**: absent is a fault for an agent-bodied
producer and an inapplicable question for a `kind: program` one, reusing
`interfaces.md` §5.13 — a script body has no agent, which is why
`Verdict.agent_id` became `AgentId | None`.

**That ruling required an input this package does not have.** Measured:

| | |
|---|---|
| `Manifest` | `digest` / `algorithm` / `kind` / `producer` / `created_at` |
| `Manifest.producer` | **`TaskId`, not `AgentId`** — `handoff/protocols.py:126` |
| `seal(hid, version, *, producer: TaskId)` | `handoff/store.py:296`, same |
| `run_gate(outputs, usage, *, store, budget)` | no producer-class parameter |

So *"there was no agent to claim"* is **not computable in `_self_check`**.

**Accepted instead:** the claim artefact carries `agent_id: AgentId | None`, the
way `Verdict` already does. §5.13 route (a) applied properly —
*the record says "no agent" by having no agent.*

| artefact | gate |
|---|---|
| absent | **fault, for everyone** |
| present, `agent_id: null` | program body — inapplicable, passes |
| present, `agent_id` set, claim true | passes |

**Why this is better than the branch it replaces, and it is not a preference.**
The distinction was always *a fact about the producer, not a branch in the
executor* — and under this shape **there is no branch anywhere**, in the gate or
the executor. It needs nothing that exists only on the SDK path, so §3.3.1's
capability matrix is untouched.

**Still blocking, and it is why nothing was built:** something must write the
artefact on the `kind: program` path. If the *producer* writes it, a program that
fails to write it is again indistinguishable from an agent that did not claim —
so the writer is probably the runner, which is **F-D1, unruled, and the user's.**

**And the tolerance clause must survive until then.** Deleting it early makes
every `kind: program` output report `SELF_CHECK_UNSET` on every attempt, and **a
signal that always fires for a whole producer class is one a monitor learns to
discount** — which would cost the check on the AI path too, the only path it was
ever worth anything on. That is `interfaces.md` §4.13's family from the other
side: not a real value that never arrives, but **a value that arrives so
reliably it stops carrying information.** The clause is in `gate.py::_self_check`
with this reasoning attached, so it reads as a decision and not an oversight.

### 6.1.1 What the wrong-on-the-day sweep found (`interfaces.md` §8.7)

**One shipped defect, and it was next door to the item above.**
`tests/agent/conftest.py::StubManifest` carried an `items` mapping. The real
`handoff.protocols.Manifest` is `digest` / `algorithm` / `kind` / `producer` /
`created_at` and **has never had one**, so `gate.py`'s
`getattr(manifest, "items", None)` returned `None` for every handoff ever
published, `_executable` returned early, and **`OUTPUT_NOT_EXECUTABLE` was
unreachable in production while its unit tests were green.** The dead branch
also had a second bug nothing could observe: it passed a `TemporaryDirectory`
straight to `copy_out`, which refuses a destination that exists.

`Manifest.digest` is a whole-tree digest, not a per-item map, and the Protocol
has no listing call — so the keys can only come from `copy_out`, and the fix
gives up the "the common case costs nothing" claim this file used to make.

`tests/agent/test_gate_against_the_real_store.py` drives the real
`FilesystemStore` on `tmp_path` — no credentials, no network, no sandbox — and
is the only thing that could have caught this: the check the code does not use
is the expensive half, which is why it had not been written.

**Everything else in the sweep was clean, checked rather than assumed:**
`env_mgr.Prepared` (`confinement` / `spawn` / `environment` / `zone`, and
`Zone.root`), `task_graph.Task.closure`, `task_graph.AgentMgr.get(spec_name)`
minting an unbound `Agent`, `validator.report.PhaseOutcome.blocks_the_task` and
`.evidence`, `spec_loader.Body` (`readme` / `entry` / `materials`),
`validator.spec.ValidatorSpec.brief`, `ValidationEnvironment.cwd` / `.env`.
Two notes on that list: `validator.protocols.PhaseOutcome` declares neither
`blocks_the_task` nor `evidence` while the concrete `report.PhaseOutcome`
has both, so the *declaration* is the thing that is behind — `validator`'s to
say, not mine; and `_evidence`'s `getattr` fallback is now a fallback whose
reason has expired, kept only because a `PhaseRunner` stub predating the field
is still a legal caller.

**The widened sweep — four spellings, and the grep that beats judgement.**
`task-graph` widened the shape from `getattr` to *"asking whether a collaborator
can do its job and continuing when the answer is no"*, and `closure` supplied
the question that is answerable before you know whether a guard is right:
**has anything ever taken this branch?** `scratch/impl-2026-08/agent/probe_default_arms.py`
answers it mechanically — it shadows `getattr` inside the package's modules,
counts the arms where the attribute was genuinely *absent*, and runs the suite:

| Default arm taken | Verdict |
|---|---|
| `Prepared.confinement` ×34, `Prepared.environment` ×17 | **Fixed.** Every one from `StubEnvManager`, none from production — the real `Prepared` has all six fields. `_apply_confinement` now reads `prepared.confinement` directly, because a `getattr` default there answers *"no confinement"* to a **missing field**, and its consequence is a task starting unconfined. `_environment`'s docstring was also an expired premise: it still claimed `Prepared` was five fields awaiting a sixth, which Ruling 2 landed |
| `ClaudeSdkBackend._connected` ×4 | **Fixed.** "Have I connected?" is a question about my own state; initialised in `__init__` |
| `Manifest.done_by_self_check` ×3 | **Kept**, and it is §6.1's item — the field exists nowhere, and this is the one guard that must tolerate absence |
| `test_claude_sdk.Message.*` ×8 | **Kept.** My own SDK-message double is thinner than the real thing, and see below |

`hasattr` and `except AttributeError` appear once between them in the package —
`backends/__init__.py:64`, which **raises** `BackendUnsupported` rather than
continuing, so it is not the shape.

**One line owed to `monitor`, now paid.** `monitor/base.py`'s excepthook reads
`thread.task_id` and nothing in production set it, so every real thread death
recorded `NO_TASK` and criterion 25's attribution half was dead. `begin()` sets
it beside the thread name, before `start`; `tests/agent/test_runner.py` drives
`monitor`'s real `install_excepthook` and asserts the record, with an
unattributed thread as the control.

**One place the check cannot be applied.** `backends/claude_sdk.py` reads
`terminal_reason`, `subtype`, `is_error`, `api_error_status` and `session_id`
off SDK message objects. `claude-agent-sdk` is an extra and is not installed
here, and driving it needs credentials, which `implementation-stage.md` forbids
in the suite. Those five reads are **unverified against their subject** and
this is the honest state of them, not a claim that they are fine.

### 6.2 What another package owes nobody, but should know

`interfaces.md` §4.13 records the general form of the `OnDone` finding, and three
others share its shape — **a failure that is recorded somewhere is not the same as
a failure that is reported**. All four were found by a consumer rather than a
producer, which is the argument for `demo`'s assembly existing at all.
