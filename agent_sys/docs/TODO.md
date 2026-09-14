# agent_sys — TODO

Near-term work: decisions to make and pieces to build inside the alpha. Long-term
subsystems live in [`ROADMAP.md`](ROADMAP.md).

Each item names where it came from and what would close it.

**Checked against the tree on 2026-08-28, after the implementation stage.** Three
items closed by being built; one changed shape. Marked inline rather than deleted,
because an item that was closed by construction is worth a reader knowing about.

**And nine seams opened that are not here.** Implementation surfaces questions a
design stage cannot, and they live in `interfaces.md` §5 rather than being copied
— that file is normative and this one is not. §5.8 (who materialises a Pointer's
value), §5.12 (two version allocators with no join), §5.13 (a script body has no
agent and `Verdict.agent_id` is required) and §5.14 (what publishes a handoff, and
from where) are the four that block something real.

---

## Decisions still open

| # | Item | What closes it |
|---|---|---|
| 1 | **What the demo's task actually does** | Package content, not a system decision (demo spec §1.1): the demo package can change its task without a spec changing. Still wanted — something small, verifiable, and not contrived |
| 2 | **What tasks the e2e test picks** | Settled that it is a separate artefact modelled on the demo's shape with its own tasks (demo spec §5). Which tasks is a system-level implementation-stage choice |
| 3 | ~~**Cross-handoff input validation input naming**~~ **— closed by construction** | Lookup is by handoff **uuid**, and `handoff/pointer.py` is RFC 6901 over `python-jsonpath`, the one library of six that separates malformed / missing / null. Two inputs of one kind are two uuids, so the ambiguity cannot arise. `tests/handoff/test_pointer.py` |
| 4 | ~~**Digest canonicalisation**~~ **— closed** | `handoff/digest.py`: `ALGORITHM = "agent_sys.handoff.tree.v1"`, sha256 git-shaped over the subtree, and `canonical()` is RFC 8785 JCS via `rfc8785` — wrapped by an encoder that **rejects `-0.0`**, which the library emits as `0` (measured). Written down rather than decided by accident |

| 4a | **A package's layout must separate a task's `bin` from the validators' `bin`** — *user-owned, and it is the precondition for staging* | **F19 reversed to staging** (`interfaces.md` §4.16), so a task gets a copy of what it needs rather than a grant on the package root. That only closes `env_mgr` criterion 13 if **a task's executable set can be named without dragging `validators/` along** — which is a package-layout guarantee, not something `env_mgr` can enforce. The user owns it. Until it holds, staging moves the leak rather than closing it |

| 4j | **A run killed by a signal leaks its servers, and nothing ever reaps the registry** — *recorded with `run_server`, 2026-09-04* | `env_mgr/servers.py` guarantees exactly *"stopped on normal and handled-error exit"* — whenever `owned_servers` unwinds. It does **not** unwind on `SIGTERM` (no handler is installed) or on `SIGKILL`. **What it costs to leave undone**: a crashed or timed-out run leaves a **listening process** and a registry file that no later run reads, so the port stays taken for as long as the machine is up, and the only thing that will ever notice is the *next* run's port check — which by then can only report a conflict, because the holder is a stranger to it. On a shared host the leak is somebody else's problem before it is ours. **Two separable pieces, and the first is the cheap one.** (1) A **sweep at start-up**: read registries left under earlier run roots and stop anything whose `starttime` still matches, which needs no new mechanism and reuses `stop_all` unchanged. (2) Closing the `SIGKILL` case itself, which the sweep does not do. `prctl(PR_SET_PDEATHSIG, SIGTERM)` was **measured to work** on this host — with it a child died when its parent was `kill -9`'d, without it the child survived — but it is the wrong tool at the site that spawns these: the spawning process is the recipe child, which exits within seconds, so the server would die the moment its own install finished. It would need the supervisor to be the direct parent, which is an architecture change and not a flag. **Deliberately not built with `run_server`**: a sweep that reaps the wrong thing is worse than a leak, and deciding which run roots it may reach is an owner's call about scope, not an implementation detail |

| 4k | **Installs run unconfined, and `env_mgr` §4 does not say so** — *measured 2026-09-04, recorded only; nothing changed and nothing should be* | `env_mgr`'s design makes confinement the load-bearing property, and **installs are an exception to it.** Measured while siting the server registry: `agent_assets.py::_run_cmd` is `subprocess.run(list(argv), capture_output=True, text=True, env=dict(environ), timeout=timeout)` — **no `preexec_fn`, no Landlock ruleset, and the full inherited environment** — so `python -m env_mgr bootstrap <recipe>`, and every `run:` string an installer shells from it, executes with the supervisor's own reach. Nothing about that is accidental or obviously wrong: **an install writes outside every zone by definition**, which is what installing is, and confining it to a zone would defeat the purpose rather than harden it. The gap is documentary. §4 reads as though confinement is universal within `env_mgr`, and the one path that is deliberately outside it is named nowhere, so the next reader meets the exception by discovering it in `_run_cmd` rather than by being told. **What would close it: a sentence in §4 naming the install path as out of scope for confinement, and why.** Not a code change — recorded here rather than acted on, because the behaviour is pre-existing, arguably required, and this round is not the place to relitigate it |

| 4l | **One fact, two readers, two different fields — and the disagreement was silent** — *instance fixed, class unrecorded until now, 2026-09-04* | *"Does this zone have a far side, and where?"* was answered in two places from two fields. `prepare.py:645 _remote_tools` read **`far_roots`**; the `paths.zone_env` call site forty lines above read **`ctx.mapping`**, which is **weak-only** because it is `sync`'s input and strength answers *must bytes be copied*. A **strong** mapping still has a far side and its `remote_root` is not in `ctx.mapping` at all. **Result: the agent was handed `env_remote_run`/`push`/`pull` pointed at a far side, and not one `AGENT_SYS_*_REMOTE` variable saying where it is.** The comment at `prepare.py:528-538` records that this was the configuration the accepted remote run used — **live, not latent**. The instance is fixed. **What is recorded here is the class**, because the fix was one call site and nothing prevents the third: a question with two answer sources, where one source is *nearly* right, fails by omission rather than by raising, and omission is what `AGENT_SYS_*_REMOTE` does — no variable, no error, an agent that improvises a path. **What would close it**: `_far_side(ctx)` at `prepare.py:621` already exists and reads *both* fields; if every consumer of "where is the far side" went through it, there would be one reader. Nobody has checked whether any consumer still does not |

## Unowned, reported more than once, recorded so they do not go stale silently

Each of these has been raised by a package that does not own it and has stayed
unclaimed. **Not blocked on anyone — nobody has them.**

| # | Item | Why it is not already fixed |
|---|---|---|
| 4b | **A typo'd `kind` in `Task.kinds` is caught by nothing at runtime** | `_participates` turns it into a no-op, and §4.16's narrowing removed the last place it would have raised. Probably `closure` check 6, at load time. **Reported twice by `env_mgr`, still unowned** |
| 4c | **P0 — a validator cannot reach the artefact its target was produced *from*, so five task packages scan the store instead** — *and the route this row previously proposed does not exist* | Was scoped to `examples/demo/logic/store.py`; the file is now `examples/demo/assets/lib/store.py` and **five copies** of it (`demo`, `demo2`, and three under `examples/llm_e2e_performance_optimization/`). See below |
| 4d | **`test_a_gate_failure_does_not_deadlock_the_next_dispatch` fails 2 runs in 4** | Green alone and green in its own file. **No cause offered** — and the day's rule applies: a red suite in a shared worktree is not evidence about anyone's change. With `agent-mod-2`. **2026-08-29, end of day: 4 full-suite runs, 4 green** (`1905 passed, 3 skipped, 4 xfailed`, ~64 s each). **Not "fixed" — the worktree was quiet, so the trigger may simply have been absent**, which is the converse of the rule above and the same instrument problem. Running it alone proves nothing and was already known not to; recorded so the next person starts from four data points rather than repeating the isolated run |
| 4e | **A hole in the store has no reaper** | §4.14 makes holes permanent and never renumbered by design. Whether they should ever be collected is undecided, not deferred |
| 4f | **`check_grounded` has never been observed catching anything** — *ruled parked 2026-08-29, deliberately not worked* | Criterion 10 aims to show a validator catching an ungrounded number; three end-to-end runs showed a good model **declining to fabricate one** instead, so the validator's **failing** direction — what its `strong` claim is about — has never executed. **The user's ruling: not a framework question and not a principle question, this is `check_grounded`'s own business semantics, and it is not worth the time.** The shape they suggested if anyone ever picks it up: **split it in two** — one validator over the other fields, and a second that judges only whether the agent's answer about the missing duration is *reasonable*, passing if it is. **Two measurements bear on any such build:** `check_grounded` matches `\d+`, *"digits, not a parser"*, so `256` reads as grounded via `sha256_prefix` — the grounding set is **wider than what the facts assert**, and a fabricated number landing inside any digit run in the copied facts passes anyway. And `logic/check_grounded/readme.md` named the `UNEXPECTED_SUCCESS`/exit-3 outcome in advance, so **exit 3 is the artefact working, not a fault to repair** |

| 4g | **The backend's `claude` child processes do not exit when their task completes** — *first report, 2026-09-04, measured not inferred* | Seen while watching an `examples/demo2` run for an unrelated reason. Nine `claude` CLI processes alive at once, one per agent task, **elapsed 6 to 26 minutes and holding 5–11 seconds of CPU each, all sleeping (`S`)**. The oldest was `directions`, which the run log showed completing 26 minutes earlier; `ps -o pid=,stat=,etime=,time=` is the whole measurement. So they are not working and not being reaped — they accumulate for the life of a run, one per task. Harmless on `demo2`; a package with many tasks, or a long-lived supervisor, is where it stops being harmless. **Whose it is, is the open part**: it could be `claude-agent-sdk` not closing its transport, or `agent/backends/claude_sdk.py` not disposing the client after the result arrives. Nothing narrows it yet, and nobody has claimed it. What would close it: run one AI task, capture the child pid, and watch whether it exits when the SDK returns — if it does, the leak is in how the runner holds the client, not in the CLI |

| 4n | **The `assets/` mechanism resolves entry points and pretends to be a resource mechanism** — *user-owned, small-scope refactor wanted (PR 155 review, 2026-09-04)* | `spec_loader/assets.py` finds **one file per role**: `body.readme` and `body.entry`, by filename convention, scoped by an optional `<name>[.<type>]/` folder. **Every other file an object needs is carried by nothing.** They arrive because `layout.stage_package(include=None)` copies the **whole package** into the zone, and a body reaches them by hand-built path — `exec python3 "$AGENT_SYS_TASK_PACKAGE/assets/check_x.validator/check.py"` is the pattern in every shipped validator. So a validator's `check.py`, its `readme.md` and any shared `assets/lib/*.py` ride along on a copy that is not the assets mechanism, and `body.entry` is a **pointer, not a manifest**. Two consequences: the object's resource set is never named anywhere, so `TODO.md` 4a (naming a task's executable set) cannot be answered from the assets index; and each body re-derives the same path string, so a layout change breaks them one by one at run time rather than at load. **The user's ruling: the mechanism is implemented badly and wants a small-scope refactor** — not a rewrite. Surfaced when `agent` needed its *directory* rather than a file and `fill_body` had no answer, which is `assets.py`'s own recorded gap (*"two of the four kinds have no body — a gap, not an omission"*), closed for `agent` by `resolve_folder` while leaving the resource question untouched |
| 4m | **The stdio MCP servers hand-roll the protocol instead of using `mcp.server`** — *asked for in the PR 155 review, 2026-09-04, not built* | `env_mgr/addons/envchk-baseline/.claude/servers/envchk_baseline_server.py` and the package's `envchk_stdio.mcp.py` speak JSON-RPC over stdin/stdout directly. The review asked for the standard library instead, and that is the right shape. **What blocks it is the interpreter, not the code**: these servers are launched as `python3 <script>` from inside a zone, so the interpreter is whichever `python3` the zone's `PATH` selects — a system one that does not have the `mcp` package. **Measured**: `/usr/bin/python3 -c 'import mcp'` raises `ModuleNotFoundError`. **Inferred, not run for this case**: that the server would then exit at import and be reported as **zero tools rather than an error**. What the tree does establish is the neighbouring property for a stdio server that is missing or broken (`env_mgr/addons/envchk-baseline/README.md`); whether an `ImportError` at start-up presents the same way has not been driven. Closing it means either declaring `mcp` as an install the recipe layer performs into the zone and pinning the server's interpreter to it, or launching servers through a resolved interpreter rather than a bare name. Both are wider than a rewrite of the two files, which is why this is recorded rather than done |
| 4h | **`env_mgr` spec §9.1's shared root has no constant in this tree** — *waiting on PR 154 (`dev.yihou.aiopt.more.demo`), 2026-09-04* | §9.1 states the rule — a declared install lands in one shared root, and only a `.claude/` tree is per-agent — and names PR 154's `AGENT_SYS_HOME` (`~/.infera_agent_sys`, `bin/ share/ state/ run/`) as its single owner. **That module is not in this branch**, so today's installs pin their destinations one variable at a time (`UV_TOOL_DIR`, `UV_TOOL_BIN_DIR`, `UV_CACHE_DIR`, and serena's `SERENA_HOME`), each into a scratch path chosen by the caller. **Deliberately not re-implemented here**: a second root would be exactly the parallel mechanism `engineer_principle.md` §2 forbids, and the two would drift over which one is authoritative. When 154 merges: take the path from its module, repoint those four pins under `<root>`, and delete the per-caller choice. Nothing about the rule changes — only where the string comes from |
| 4i | **User-level AI material outlives the run that declared it** — *surfaced 2026-09-04, needs an owner decision, not a patch* | `env_mgr` spec §9.1 sends a `.claude/` tree declared in `main.yaml`/`default.yaml` to **user level**, i.e. the agent_sys root's Claude config. PR 154 puts that root **deliberately outside any run root**, because a resident daemon has to outlive a single run. Both decisions are right on their own and their product is that **a task package's skills, hooks and MCP declarations persist into the next run of a different package.** Nobody chose that; it fell out. Three shapes are available and they are not equivalent — scope the material to the run and give up daemon-visible continuity; keep it and accept cross-run bleed as the meaning of *user level*; or add a third scope between them, which is the parallel hierarchy `engineer_principle.md` §2 exists to prevent. **Recorded before implementation rather than discovered after**, and it is not worked around silently |

### 4c in full — why the store scan exists, and why declaring the input would not remove it

**This row said the declared route was `materials.json`, and that reaching a
non-target artefact was a matter of *declaring* it — `inputs: ['summary',
'facts']`. Read against the code, that fix does not work.** A validator's
`inputs` is a **filter over the task's slots on this phase's side, not a
request**. `validator/phase.py:731`:

```python
return list(task.inputs if kind is PhaseKind.INPUT else task.outputs)
```

and `phase.py:657` selects from exactly that: `mine = [t for t in targets if
self._kind_of(t, registry) in spec.inputs]`. `env_mgr/prepare.py:691-695` stages
the same set. So a kind the task does not hold **on that side** is not a target,
is not staged, and cannot be declared into existence.

**The concrete case.** `check_problems` must verify that the problem set cites a
direction that exists — i.e. that the artefact is faithful to what its producer
consumed. `directions` is on the producing task's **input** side; the validator
runs on that task's **output** phase (and again on the students' input phases,
where the candidate set is `[problems]` too). The two sides never meet in any
phase, so there is no phase in which `directions` is reachable. Its declaration
is `inputs: [problems]` (`examples/demo2/steps/problems.yaml:48`) while its body
reads `directions` — so the schema's own promise for that field, *"DECLARED
rather than discovered, so a reviewer can answer 'what does this actually read'
without running it"*, is already false here. Same shape in
`demo/check_grounded`.

**What the packages do instead.** `lib/store.py` reads `handoff`'s on-disk
layout through `AGENT_SYS_DEMO_STORE` (`cli/main.py:825`) and scans for *the
newest artefact of that kind anywhere in the store*. Its own docstring calls
that crude and wrong in a graph with more than one producer; it happens to be
right in these packages because there is exactly one. The ~30
`staged_content(hid) or content_dir(hid)` sites are a different thing and not
this problem — each is commented as the fallback for a run with **no `env_mgr`
wired**, i.e. a validator run standalone.

**Why it is P0 and why it is quiet.** The scan is only alive because two things
are switched off: `prepare_validation` *"does not confine anything"*
(`env_mgr/prepare.py:686`, and `EnvManager.prepare_validation` at `:751` records
that who confines a validation body *"is a third question that this ruling did
not settle"*), and `AGENT_SYS_NO_PERMISSIONS` defaults to on
(`prepare.py:80-95`). Either one landing kills the route — `env_mgr`'s `p11`
measured **EACCES on the store root from a confined body**, and `store_root()`
is `os.environ[...]` rather than `.get`, so the body dies *before*
`write_verdict` and `PhaseRunner` gets **no `verdict.json` at all** rather than
a `False`. So confining validations — ROADMAP §6.1's P0 — silently converts a
grounding check into a missing file. **Not measured:** whether a confined
*validation* body fails the same way an agent body does; that needs a policy
applied to one validation zone and a run.

**The fix that removes the knowledge rather than moving it.** Give the output
phase read access to the producer's inputs — stage `task.inputs` read-only
alongside `task.outputs` in `prepare_validation`, and let `inputs:` select from
the union. Then the declaration becomes true, `declared_dir` is the only route a
body needs, and `versions` / `content_dir` / `kind_of` / `latest_of_kind` /
`handoff_dir` delete from all five copies. **A design question for `validator`
and `env_mgr` jointly, not a patch** — it widens what an output validation may
see, which is a criterion-13 (anti-gaming) question and must be argued there
before it is built.

Raised again 2026-09-04 while labelling runtime directories (PR #156), which is
what made the five duplicated readers visible in one diff.


| 4o | **A non-leaf pins a store version it can never fill, and the dead directory is permanent** — *characterised 2026-09-08 from a passing `demo-2task-ai` run; nothing changed, and this round changed nothing under `task_graph/`* | `scheduler.py::_pin_outputs` is `{hid: store.allocate(hid) for hid in task.outputs}` — unconditional, one store version per declared output **per dispatched task**, allocated at dispatch so `env_mgr`'s kind-named write grant has a directory to resolve against. It keys off `task.outputs` alone and does not distinguish a leaf from a pass-through parent, so a non-leaf that declares an output — `demo-2task-ai`'s `main` declares `outputs: [problems]`, `env_checker`'s declares `[env_report]` — reserves `v<N>/` with `content/` and `claim/` and never writes into it. **The parent's allocation is provably dead, and that is measured, not inferred**: `store.seal` has exactly one production caller (`agent/runner.py:1123`, inside `_seal_outputs`); `_seal_outputs` has exactly one (`agent/runner.py:741`, inside `_main`'s loop); `_main` has exactly one (`agent/runner.py:614`); and `_main`'s **first statement** is `if self.task.has_subgraph(): self.release(); return False`, which returns before `_deploy`, before `_open_outputs` and before the loop — its own docstring says *"A non-leaf reaches none of this."* So no path lets a non-leaf seal its declared output directly. **What it costs to leave undone is small and entirely legibility.** Nothing breaks: `list_versions` filters on the manifest, so the reservation is invisible to every store reader — measured against the real `FilesystemStore`, `problems` is `['v0','v1']` on disk but `list_versions → [1]`, `latest → 1`, `exists(hid, 0) → False`. The cost is that the on-disk numbering of a package with a declaring root is permanently offset by one, holes are never compacted by design, and a reader who compares the tree against the log meets `handoff problems slot v0: valid` next to an empty `v0/` — two different counters (`HandoffMgr`'s slot advances on every agent write, the store's on every dispatch), which is exactly the shape of thing that gets filed as a bug a year later by someone who has neither docstring open. **What would close it**: either `_pin_outputs` skips tasks for which `has_subgraph()` is true — the allocation is dead for precisely that set, so the predicate is already the right one and needs no new concept — or, if the reservation is wanted so that a future non-leaf publish route inherits a pinned version, one sentence in `_pin_outputs` saying the parent's version is deliberately dead and why, which costs nothing and removes the misreading. **Inferred, not measured**: which of the two is wanted. That is a design call belonging to whoever owns `task_graph`, and the argument is recorded here rather than acted on |

| 4p | **Nothing owns *which* interpreter runs a body, so a validator resolves `python3` by luck** — *found 2026-09-09 when `--docker` removed the luck; nothing changed under `validator/` or `cli/*, and the defect is not container-specific* | `entry.sh` in every demo validator is `exec "${AGENT_SYS_DEMO_PYTHON:-python3}" check.py` — use the interpreter you were told to use, else whatever `python3` resolves to. **`AGENT_SYS_DEMO_PYTHON` exists in exactly one place in the tree**: `cli/main.py:996`, `sys.executable`, on the `global_` row. But `validator/environment.py::choose_configuration` is spec §8.2's **selection chain, not a merge** — *bound, else the consumer's for input validation, else the producer's for output validation, else a predefined global one* — and a validator on an **output** phase always has a producer environment, because the task that produced the artefact had one. So the one row that names the interpreter is the one row that is never consulted, and the fallback fires on every output phase, on every host. `build_environment` then *"never inherits `os.environ`"*, so nothing downstream can repair it. **The division of labour is clean except for the last step, which nobody holds.** `env_mgr` owns *can this interpreter be reached*: `interpreter_grants` grants the interpreter's own prefix, because with it under `$HOME` — conda, pyenv, uv, venv, so every ordinary install — `subprocess` fails in the parent naming the interpreter rather than the sandbox; and `executable_path` **projects `PATH` from the granted set rather than choosing it**, whose stated invariant is that `PATH` can never name a directory the kernel will refuse. Neither is a claim about *which* interpreter, and `executable_path`'s own docstring already records the consequence, reported by `validator`, who *"measured it and correctly declined to pick a value"*: a process handed no `PATH` gets the shell binary's built-in default, so *"two hosts resolve `python3` differently and nothing in the run record says which"*. **What it costs**: on a developer host the first `python3` on the derived `PATH` happens to be the same interpreter that is running `agent_sys`, so the fallback is invisible. In the `agent-sys` container it is not: `/usr/local/bin/python3` (position 2) shadows `/opt/venv/agent/bin` (position 7), and `check_directions` dies at `store.py:39` with `ModuleNotFoundError: No module named 'yaml'` — exit 5, no `verdict.json`. Note the two obvious fixes are both wrong: the venv **is** already on the dispatch `PATH`, so adding it is a no-op, and putting it first would work by shadowing rather than by saying which interpreter is meant. **What would close it**: either `choose_configuration` merges the global row's run-facts into whichever row it selects, or `_validation_env`'s facts belong on every row rather than only the last. **Inferred, not measured**: which of the two is wanted — a design call for whoever owns `validator/` — and that the other five packages sharing this `entry.sh`/`store.py` shape (`demo`, `demo2`, three `demo-runtime-*`) are exposed; they have never been run in a container. **A second defect, independent of the code**: `entry.sh`'s comment and `validator/phase.py:325`'s docstring both assert the input-phase story and are silent on output — and that docstring already records having given `demo` a wrong answer once. The same wrong belief has now cost two investigations |

## Blocked on another change landing

| # | Item | What closes it |
|---|---|---|


## To build in the alpha

| # | Item | Note |
|---|---|---|
| 5 | **A whole-system CLI** | Receives a global task, a config YAML, and some CLI options, and runs the whole thing. Currently only the demo has an entry point |
| 6 | **`env_mgr` submodule that sets up the Claude Code SDK** | Installs and configures the SDK from the API key and endpoint supplied in config, so a fresh machine can run an agent without hand-setup |
| 7 | **A skill / rule / hook set per handoff content type** — *the delivery mechanism now exists*: `env_mgr/material.py` deploys `rules`, `hooks` and `skills` into the zone (`MATERIAL_KEYS`). What is missing is the four sets themselves | Every content type — reproducible, code, structured text, text — needs its own agent skill set to produce it correctly. Four sets, not one generic one |
| 8 | **The `--validation-strict-level` CLI switch** | Controls whether a validation phase may be skipped: by config, or because something else already validated the handoff |
| 9 | **The mandatory-knowledge CLI option** | Knowledge parts are strongly suggested with a warning by default; this flag makes them mandatory |
| 10 | **Agent-harness format transform helper** | Converts rules, hooks, and skills between harness formats. Claude Code's format is the canonical stored form. An independent module — `agent_harness_backend_transform_helper` or similar |
| 11 | **Remote↔local operations as agent tool calls** | Not a natural-language description of a procedure. An agent should call a tool, so it can actually use it reliably |
| 12 | ~~**Task-agent env reuse**~~ **— the alpha half is built** | The general mechanism stays deferred (roadmap §6). The validation-phase answer is `validator/environment.py::choose_configuration`, spec §8.2's chain: the bound env, else the consumer's for input validation and the producer's for output validation, else a predefined global one |
