# agent_sys — TODO

Near-term work: decisions to make and pieces to build inside the alpha.
Long-term subsystems live in [`ROADMAP.md`](ROADMAP.md).

**Every item carries the same four things**: what it is, why it is not done,
what would close it, and who owns it. An item with no owner is not blocked on
anyone — nobody has it.

| Status | Meaning |
|---|---|
| **OPEN** | Wanted, not started, and nothing prevents it |
| **BLOCKED** | Cannot proceed until a named decision or change lands |
| **PARKED** | Deliberately not worked, with a reason. Not a backlog item |
| **CLOSED** | Settled. Kept because an item closed by construction is worth knowing about |

The seams that implementation opened are here, as items 25–30. A separate
register of open seams makes sense only while nobody knows who owns what; in a
built system an open question is either a near-term decision (here) or a
subsystem ([`ROADMAP.md`](ROADMAP.md)), and there is no
third place.

---

## Index

| # | Item | Status | Owner |
|---|---|---|---|
| 1 | What the demo's task actually does | OPEN | package |
| 2 | A package's layout must separate a task's `bin` from the validators' `bin` | OPEN | user |
| 3 | The `assets/` mechanism resolves entry points and pretends to be a resource mechanism | OPEN | user |
| 4 | Five packages scan the store instead of declaring a pass-through | OPEN | package authors |
| 5 | Nothing owns *which* interpreter runs a body | OPEN | `validator` |
| 6 | A non-leaf pins a store version it can never fill | OPEN | `task_graph` |
| 7 | User-level AI material outlives the run that declared it | OPEN | user |
| 8 | A run killed by a signal leaks its servers, and nothing reaps the registry | OPEN | — |
| 9 | The backend's `claude` child processes do not exit when their task completes | OPEN | — |
| 10 | A typo'd `kind` in `Task.kinds` is caught by nothing at runtime | OPEN | — |
| 10a | A closure's `validators` list has no runtime consumer | OPEN | `validator` + `closure`, jointly |
| 10b | `handoff` spec §5.3's escape hatch has no way in | OPEN | `cli` |
| 11 | `test_a_gate_failure_does_not_deadlock_the_next_dispatch` is intermittent | OPEN | — |
| 12 | The per-caller install pins are not repointed at the shared root | OPEN | `env_mgr` |
| 13 | Installs run unconfined, and `env_mgr` spec §4 does not say so | OPEN | `env_mgr` |
| 14 | One fact with two readers and two fields — recorded as a class | OPEN | `env_mgr` |
| 15 | The stdio MCP servers hand-roll the protocol instead of using `mcp.server` | BLOCKED | — |
| 16 | A hole in the store has no reaper | PARKED | — |
| 17 | `check_grounded` has never been observed catching anything | PARKED | — |
| 18 | A whole-system CLI | OPEN | — |
| 19 | An `env_mgr` submodule that sets up the Claude Code SDK | OPEN | `env_mgr` |
| 20 | A skill / rule / hook set per handoff content type | OPEN | — |
| 21 | The `--validation-strict-level` CLI switch | OPEN | `cli` |
| 22 | The mandatory-knowledge CLI option | OPEN | `cli` |
| 23 | Agent-harness format transform helper | OPEN | — |
| 24 | Remote↔local operations as agent tool calls | OPEN | — |
| 25 | `materials` is declared by two schemas and read by nothing | OPEN | `closure` + `validator` |
| 26 | `handoff.resolve` has no caller | OPEN | `validator` |
| 27 | Two version allocators for one artefact, and nothing joins them | OPEN | `handoff` + `task_graph` |
| 28 | A non-leaf may declare an output no entry can produce | OPEN | `task_graph` |
| 29 | `ValidatorId` — a fourth typed id the spec asks for and nothing needs | OPEN | — |
| 30 | `SCHEMA_VERSION` will have two owners once the whole-system CLI exists | BLOCKED | `cli` |
| 31 | The task's whole environment is passed on `argv`, so its credential is world-readable | OPEN | `agent` |
| 32 | Two of the four strict xfails may encode shipped bugs, not unbuilt features | OPEN | — |
| 33 | An unknown `--var` is accepted silently | OPEN | `cli` |
| 34 | `give_back` clamps overspend, while `charge` says spend is never refused | OPEN | `task_graph` |
| 35 | The Claude SDK adapter's loop has one driver at a time | OPEN | `agent` |
| 36 | A program body's grandchildren outlive `_terminate` outside bwrap | PARKED | `agent` |
| C1–C5 | Closed | CLOSED | — |

---

## Open

### 1 — What the demo's task actually does

Package content, not a system decision: the demo package can change its task
without a spec changing (`cli` spec §1.1). Still wanted — something small,
verifiable, and not contrived.

**Closes when:** somebody picks a task and it lands.

### 2 — A package's layout must separate a task's `bin` from the validators' `bin`

`interfaces.md` §4.16 reversed F19 to **staging**, so a task gets a copy of what
it needs rather than a grant on the package root. That only closes `env_mgr`
criterion 13 if **a task's executable set can be named without dragging
`validators/` along** — which is a package-layout guarantee, not something
`env_mgr` can enforce.

**Closes when:** the user fixes the layout rule. Until it holds, staging moves
the leak rather than closing it.

### 3 — The `assets/` mechanism resolves entry points and pretends to be a resource mechanism

`spec_loader/assets.py` finds **one file per role** — `body.readme` and
`body.entry` — by filename convention, scoped by an optional `<name>[.<type>]/`
folder. **Every other file an object needs is carried by nothing.** They arrive
because `layout.stage_package(include=None)` copies the *whole* package into the
zone, and a body reaches them by a hand-built path:
`exec python3 "$AGENT_SYS_TASK_PACKAGE/assets/check_x.validator/check.py"` is the
pattern in every shipped validator. So `body.entry` is a **pointer, not a
manifest**.

Two consequences: an object's resource set is named nowhere, so item 2 cannot be
answered from the assets index; and each body re-derives the same path string, so
a layout change breaks them one by one at run time rather than at load.

**Closes when:** a small-scope refactor gives an object a resource manifest. Not
a rewrite — the user has ruled it small.

### 4 — Five packages scan the store instead of declaring a pass-through

A validator's `inputs` is a **filter over the task's slots on this phase's side,
not a request**. `validator/phase.py`:

```python
return list(task.inputs if kind is PhaseKind.INPUT else task.outputs)
```

and the selection is `mine = [t for t in targets if self._kind_of(t, registry) in
spec.inputs]`. `env_mgr/prepare.py` stages the same set. So a kind the task does
not hold **on that side** is not a target, is not staged, and cannot be declared
into existence.

**The concrete case.** `check_problems` must verify that the problem set cites a
direction that exists. `directions` is on the producing task's **input** side;
the validator runs on that task's **output** phase. The two sides never meet in
any phase, so there is no phase in which `directions` is reachable. Its
declaration is `inputs: [problems]` while its body reads `directions` — so the
schema's own promise for that field, *"DECLARED rather than discovered, so a
reviewer can answer 'what does this actually read' without running it"*, is
already false there.

**What the packages do instead.** `assets/lib/store.py` reads `handoff`'s on-disk
layout through `AGENT_SYS_DEMO_STORE` and scans for *the newest artefact of that
kind anywhere in the store*. Its own docstring calls that crude and wrong in a
graph with more than one producer; it is right in these packages only because
there is exactly one.

**Why it is P0 and why it is quiet.** The scan is alive only because two things
are off: `prepare_validation` does not confine anything, and
`AGENT_SYS_NO_PERMISSIONS` defaults to on. Either one landing kills the route —
a confined body gets `EACCES` on the store root, and `store_root()` is
`os.environ[...]` rather than `.get`, so the body dies *before* `write_verdict`
and `PhaseRunner` gets **no `verdict.json` at all** rather than a `False`. So
confining validations — [`ROADMAP.md`](ROADMAP.md) §6.1's P0 — silently converts
a grounding check into a missing file.

**Settled, and the ruling cancels the design question rather than answering
it:**

> 这是任务声明的问题，系统不处理。如果需要，用户需要在定义任务时自己把自己的输入透传到自己的输出。
>
> *This is a task-declaration problem. The system does not handle it. If a task
> needs its input visible to its own output validation, the task author passes
> that input through to their own output when they declare the task.*

**So what remains open is not the mechanism — it is the five packages that have
not been rewritten to obey the ruling.** Each still carries a `lib/store.py`
that scans, and each still declares an `inputs:` its body contradicts. Until
they pass their inputs through, the declaration stays false and the scan stays
load-bearing.

**Closes when:** the five packages declare pass-through outputs and their
`lib/store.py` copies delete. No framework change; the framework side is
settled.

**Not measured:** whether a confined *validation* body fails the same way an
agent body does. That needs a policy applied to one validation zone and a run.

### 5 — Nothing owns *which* interpreter runs a body

`entry.sh` in every demo validator is `exec "${AGENT_SYS_DEMO_PYTHON:-python3}"
check.py` — use the interpreter you were told to use, else whatever `python3`
resolves to. **`AGENT_SYS_DEMO_PYTHON` is set in exactly one place**, on the
`global_` row. But `validator/environment.py::choose_configuration` is spec
§8.2's **selection chain, not a merge** — bound, else the consumer's for input
validation, else the producer's for output validation, else a predefined global
one — and a validator on an **output** phase always has a producer environment.
So the one row that names the interpreter is the one row that is never consulted,
and the fallback fires on every output phase, on every host. `build_environment`
then never inherits `os.environ`, so nothing downstream can repair it.

**The division of labour is clean except for the last step, which nobody holds.**
`env_mgr` owns *can this interpreter be reached*: `interpreter_grants` grants the
interpreter's own prefix, and `executable_path` **projects `PATH` from the
granted set rather than choosing it**. Neither is a claim about *which*
interpreter. A process handed no `PATH` gets the shell binary's built-in default,
so two hosts resolve `python3` differently and nothing in the run record says
which.

**What it costs.** On a developer host the first `python3` on the derived `PATH`
happens to be the interpreter running `agent_sys`, so the fallback is invisible.
In a container it is not: a system `python3` earlier on `PATH` shadows the venv,
and a validator dies with `ModuleNotFoundError: No module named 'yaml'` — exit 5,
no `verdict.json`. Both obvious fixes are wrong: the venv **is** already on the
dispatch `PATH`, so adding it is a no-op, and putting it first would work by
shadowing rather than by saying which interpreter is meant.

**Closes when:** either `choose_configuration` merges the global row's run-facts
into whichever row it selects, or the validation environment's facts belong on
every row rather than only the last. Which of the two is wanted is a design call
for whoever owns `validator/`.

**A second defect, independent of the code:** `entry.sh`'s comment and
`validator/phase.py`'s docstring both assert the input-phase story and are silent
on output, so a reader of either arrives at the wrong belief.

### 6 — A non-leaf pins a store version it can never fill, and the dead directory is permanent

`scheduler.py::_pin_outputs` is `{hid: store.allocate(hid) for hid in
task.outputs}` — unconditional, one store version per declared output per
dispatched task, allocated at dispatch so `env_mgr`'s kind-named write grant has
a directory to resolve against. It keys off `task.outputs` alone and does not
distinguish a leaf from a pass-through parent, so a non-leaf that declares an
output reserves `v<N>/` with `content/` and `claim/` and never writes into it.

**The parent's allocation is provably dead.** `store.seal` has one production
caller, inside `_seal_outputs`; `_seal_outputs` has one, inside the runner's
`_main` loop; and `_main`'s **first statement** is `if self.task.has_subgraph():
self.release(); return False`, which returns before `_deploy`, before
`_open_outputs` and before the loop. No path lets a non-leaf seal its declared
output directly.

**What it costs is small and entirely legibility.** Nothing breaks:
`list_versions` filters on the manifest, so the reservation is invisible to every
store reader. The cost is that the on-disk numbering of a package with a
declaring root is permanently offset by one, holes are never compacted by design,
and a reader comparing the tree against the log meets `handoff problems slot v0:
valid` next to an empty `v0/` — two different counters, which is the shape of
thing that gets filed as a bug a year later.

**Closes when:** either `_pin_outputs` skips tasks for which `has_subgraph()` is
true — the allocation is dead for precisely that set, so the predicate is already
the right one — or one sentence in `_pin_outputs` says the parent's version is
deliberately dead and why. Which of the two is wanted is a design call for
whoever owns `task_graph`.

### 7 — User-level AI material outlives the run that declared it

`env_mgr` spec §9.1 sends a `.claude/` tree declared in `main.yaml`/`default.yaml`
to **user level**, i.e. the agent_sys root's Claude config. That root is
deliberately outside any run root, because a resident daemon has to outlive a
single run. Both decisions are right on their own and their product is that **a
task package's skills, hooks and MCP declarations persist into the next run of a
different package.** Nobody chose that; it fell out.

**Closes when:** the owner picks one of three, and they are not equivalent —
scope the material to the run and give up daemon-visible continuity; keep it and
accept cross-run bleed as the meaning of *user level*; or add a third scope
between them, which is the parallel hierarchy `engineer_principle.md` §2 exists
to prevent.

### 8 — A run killed by a signal leaks its servers, and nothing ever reaps the registry

`env_mgr/servers.py` guarantees exactly *"stopped on normal and handled-error
exit"* — whenever `owned_servers` unwinds. It does **not** unwind on `SIGTERM`
(no handler is installed) or on `SIGKILL`.

**What it costs:** a crashed or timed-out run leaves a **listening process** and a
registry file no later run reads, so the port stays taken for as long as the
machine is up. The only thing that will notice is the *next* run's port check,
which by then can only report a conflict, because the holder is a stranger to it.
On a shared host the leak is somebody else's problem before it is ours.

**Two separable pieces, and the first is the cheap one.**
1. A **sweep at start-up**: read registries left under earlier run roots and stop
   anything whose `starttime` still matches. Needs no new mechanism and reuses
   `stop_all` unchanged.
2. Closing the `SIGKILL` case itself, which the sweep does not do.
   `prctl(PR_SET_PDEATHSIG, SIGTERM)` was **measured to work** — with it a child
   died when its parent was `kill -9`'d, without it the child survived — but it is
   the wrong tool at the site that spawns these: the spawning process is the
   recipe child, which exits within seconds, so the server would die the moment
   its own install finished. It would need the supervisor to be the direct
   parent, which is an architecture change and not a flag.

**Deliberately not built**: a sweep that reaps the wrong thing is worse than a
leak, and deciding which run roots it may reach is an owner's call about scope.

### 9 — The backend's `claude` child processes do not exit when their task completes

Measured, not inferred: nine `claude` CLI processes alive at once, one per agent
task, elapsed 6 to 26 minutes and holding 5–11 seconds of CPU each, all sleeping.
The oldest corresponded to a task the run log showed completing 26 minutes
earlier. So they are not working and not being reaped — they accumulate for the
life of a run, one per task.

Harmless on a small package; a package with many tasks, or a long-lived
supervisor, is where it stops being harmless.

**Whose it is, is the open part**: it could be `claude-agent-sdk` not closing its
transport, or `agent/backends/claude_sdk.py` not disposing the client after the
result arrives.

**Closes when:** somebody runs one AI task, captures the child pid, and watches
whether it exits when the SDK returns. If it does, the leak is in how the runner
holds the client, not in the CLI.

### 10 — A typo'd `kind` in `Task.kinds` is caught by nothing at runtime

`_participates` turns it into a no-op, and `interfaces.md` §4.16's narrowing
removed the last place it would have raised.

**Closes when:** it becomes a load-time check — probably `closure` check 6.
Reported twice by `env_mgr` and still unowned.

### 10a — A closure's `validators` list has no runtime consumer

`validator`'s phase runner builds a phase's validator set from the **handoff
kind's** own `validators`. A closure's `validators` key — and the accessors over
it, `closure.phase_validators` and `ClosureRegistry.validators_for` — are read by
`closure`'s own query and load-check modules and **by nothing else in the tree**.

`closure.schema.json` describes that key as *the phase validators, the checks
that run in this task's input and output validation phases*. Nothing runs them.
So a package author can declare a phase validator on the closure, load cleanly,
and have it silently never execute.

One load-time reader now exists: `validator.separation.check_catalogue` (review
R09) checks a closure's phase validators against its task's permissions. That is
a separation check, not execution — the list still never runs.

**Closes when:** either the phase runner consults the closure's list as well as
the kind's, or the key is withdrawn from the schema and the accessors with it.
The two are opposite answers to *who owns the phase's validator set*, and that
question belongs to `validator` and `closure` jointly.

### 10b — `handoff` spec §5.3's escape hatch has no way in

§5.3 permits a kind with no validator **for bring-up and debugging**, off by
default and reporting every kind it lets through. The mechanism exists —
`HandoffSpecRegistry` takes `allow_no_validator` — and **nothing exposes it**:
the composition root builds the registry with the default, so the only callers
that can reach the hatch are tests.

So the rule holds today without exception, which is the safe direction. What is
missing is the *debugging* half the spec promises: bringing up a package whose
validators are not written yet means editing the composition root.

**Closes when:** the CLI grows the flag, off by default, and the run record names
every kind it admitted without a validator. Small, and it is the CLI's because
the registry side is already built.

### 11 — `test_a_gate_failure_does_not_deadlock_the_next_dispatch` is intermittent

Observed failing 2 runs in 4, then green in 4 consecutive full-suite runs. **No
cause offered.** Running it alone proves nothing and was already known not to.

**Closes when:** somebody reproduces it deliberately rather than waiting for it.

### 12 — The per-caller install pins are not repointed at the shared root

`env_mgr` spec §9.1 states the rule — a declared install lands in one shared
root, and only a `.claude/` tree is per-agent — and `AGENT_SYS_HOME`
(`~/.infera_agent_sys`, `bin/ share/ state/ run/`) is its single owner. That
constant exists in `env_mgr/prefix.py`, but the installs still pin their
destinations one variable at a time (`UV_TOOL_DIR`, `UV_TOOL_BIN_DIR`,
`UV_CACHE_DIR`, and serena's `SERENA_HOME`), each into a scratch path chosen by
the caller.

**A second root must not be introduced**: that is exactly the parallel mechanism
`engineer_principle.md` §2 forbids, and the two would drift over which is
authoritative.

**Closes when:** those four pins are repointed under `<root>` and the per-caller
choice is deleted. Nothing about the rule changes — only where the string comes
from.

### 13 — Installs run unconfined, and `env_mgr` spec §4 does not say so

`env_mgr`'s design makes confinement the load-bearing property, and **installs are
an exception to it.** Measured: `agent_assets.py::_run_cmd` is
`subprocess.run(list(argv), capture_output=True, text=True, env=dict(environ),
timeout=timeout)` — **no `preexec_fn`, no Landlock ruleset, and the full inherited
environment** — so `python -m env_mgr bootstrap <recipe>`, and every `run:` string
an installer shells from it, executes with the supervisor's own reach.

Nothing about that is obviously wrong: **an install writes outside every zone by
definition**, which is what installing is, and confining it to a zone would defeat
the purpose rather than harden it. **The gap is documentary.** §4 reads as though
confinement is universal within `env_mgr`, and the one path deliberately outside
it is named nowhere, so the next reader meets the exception by discovering it in
`_run_cmd` rather than by being told.

**Closes when:** §4 gains a sentence naming the install path as out of scope for
confinement, and why. Not a code change.

### 14 — One fact, two readers, two different fields — and the disagreement was silent

*"Does this zone have a far side, and where?"* was answered in two places from two
fields. `prepare.py::_remote_tools` read **`far_roots`**; the `paths.zone_env`
call site read **`ctx.mapping`**, which is **weak-only** because it is `sync`'s
input and strength answers *must bytes be copied*. A **strong** mapping still has
a far side and its `remote_root` is not in `ctx.mapping` at all. Result: the agent
was handed `env_remote_run`/`push`/`pull` pointed at a far side, and not one
`AGENT_SYS_*_REMOTE` variable saying where it is. That was the configuration the
accepted remote run used — live, not latent.

The instance is fixed. **What is recorded here is the class**, because the fix was
one call site and nothing prevents the third: a question with two answer sources,
where one source is *nearly* right, fails by omission rather than by raising, and
omission is what `AGENT_SYS_*_REMOTE` does — no variable, no error, an agent that
improvises a path.

**Closes when:** every consumer of "where is the far side" goes through
`_far_side(ctx)`, which already exists and reads *both* fields. Nobody has checked
whether any consumer still does not.

### 15 — The stdio MCP servers hand-roll the protocol instead of using `mcp.server` — BLOCKED

`env_mgr/addons/envchk-baseline/.claude/servers/envchk_baseline_server.py` and the
package's `envchk_stdio.mcp.py` speak JSON-RPC over stdin/stdout directly. The
standard library is the right shape.

**What blocks it is the interpreter, not the code**: these servers are launched as
`python3 <script>` from inside a zone, so the interpreter is whichever `python3`
the zone's `PATH` selects — a system one that does not have the `mcp` package.
**Measured**: `/usr/bin/python3 -c 'import mcp'` raises `ModuleNotFoundError`.
**Inferred, not run**: that the server would then exit at import and be reported
as **zero tools rather than an error**.

This is item 5 wearing a different hat.

**Closes when:** either `mcp` is declared as an install the recipe layer performs
into the zone and the server's interpreter is pinned to it, or servers are
launched through a resolved interpreter rather than a bare name.

### 16 — A hole in the store has no reaper — PARKED

`interfaces.md` §4.14 makes holes permanent and never renumbered by design.
Whether they should ever be collected is **undecided, not deferred**.

### 17 — `check_grounded` has never been observed catching anything — PARKED

Criterion 10 aims to show a validator catching an ungrounded number; three
end-to-end runs showed a good model **declining to fabricate one** instead, so the
validator's **failing** direction — what its `strong` claim is about — has never
executed.

**Settled: not a framework question and not a principle question.** This is
`check_grounded`'s own business semantics and it is not worth the time. The
shape suggested if anyone picks it up: **split it in two** — one validator over
the other fields, and a second that judges only whether the agent's answer about
the missing value is *reasonable*, passing if it is.

**Two measurements bear on any such build:** `check_grounded` matches `\d+`,
*"digits, not a parser"*, so `256` reads as grounded via `sha256_prefix` — the
grounding set is **wider than what the facts assert**, and a fabricated number
landing inside any digit run in the copied facts passes anyway. And the
validator's own readme named the `UNEXPECTED_SUCCESS`/exit-3 outcome in advance,
so **exit 3 is the artefact working, not a fault to repair**.

### 18 — A whole-system CLI

Receives a global task, a config YAML, and some CLI options, and runs the whole
thing. Today only `cli/` has an entry point, and it is the composition root for
one package at a time. This is also where the control surface — abort and
instruct — will land (see [`../docs/spec.md`](spec.md) §10).

### 19 — An `env_mgr` submodule that sets up the Claude Code SDK

Installs and configures the SDK from the API key and endpoint supplied in config,
so a fresh machine can run an agent without hand-setup.

### 20 — A skill / rule / hook set per handoff content type

*The delivery mechanism exists*: `env_mgr/material.py` deploys `rules`, `hooks`
and `skills` into the zone (`MATERIAL_KEYS`). What is missing is the four sets
themselves. Every content type — reproducible, code, structured text, text —
needs its own agent skill set to produce it correctly. **Four sets, not one
generic one.**

### 21 — The `--validation-strict-level` CLI switch

Controls whether a validation phase may be skipped: by config, or because
something else already validated the handoff.

### 22 — The mandatory-knowledge CLI option

Knowledge parts are strongly suggested with a warning by default; this flag makes
them mandatory.

### 23 — Agent-harness format transform helper

Converts rules, hooks, and skills between harness formats. Claude Code's format is
the canonical stored form. An independent module.

### 24 — Remote↔local operations as agent tool calls

Not a natural-language description of a procedure. An agent should call a tool, so
it can actually use it reliably.

### 31 — The task's whole environment is passed on `argv`, so its credential is world-readable

`agent` builds the backend invocation with `--mcp-config <json>`, where the JSON
is a **command-line argument** carrying each server's `env` block inline. That
block is the task's environment, and on this deployment it contains
`ANTHROPIC_CUSTOM_HEADERS`, whose value is the API subscription key.

`/proc/<pid>/cmdline` is world-readable on Linux. So **every user on the host can
read the credential for as long as a task is running**, with no privilege and no
race — one `ps -eo args` is enough. Measured on `smc300x-ccs-aus-a16-19`
during a live `--docker` run of `ok.agent_capabilities.2` on 2026-09-16; four
other tenants' containers were running on the same box at the time.

**`ANTHROPIC_API_KEY` in that block is the literal string `dummy`,** so the header
is not a secondary detail — it *is* the credential this deployment authenticates
with, and it is the one on `argv`.

Two things this is **not**:

- It is not specific to `--docker`. The container inherits nothing here; the
  exposure is wherever the backend is spawned, which is every run on every host.
- It is not the fault of the secret's storage. `~/.claude/settings.json` is
  correct at rest (mode-protected, never copied into an image); the value is
  read from there legitimately and then published by how it is *passed*.

**Why it survived review.** Nothing in the codebase prints it — the value reaches
no log, artefact, fixture or commit, and the leader of that stage confirmed as
much. The publication happens in the kernel's process table, which no amount of
reading `agent_sys`'s own output would reveal. It was found by an unrelated `ps`
run while looking for the o11y daemon's pid.

**Closes when:** the MCP configuration stops being an `argv` value. The candidate
shapes, in the order they should be considered:

1. Write the config to a file inside the zone (which already exists, is already
   mode-restricted, and is already deleted with the zone) and pass the path. The
   backend CLI accepts a path where it accepts JSON; whether it accepts one for
   *every* invocation shape this code builds is unverified and is the first thing
   to check.
2. Pass the secret-bearing names through the child's **environment** instead of
   its `argv`. `/proc/<pid>/environ` is `0400` and owner-only, unlike `cmdline`.
   Weaker than (1) — same-uid processes still read it — but strictly better than
   today and a much smaller change.

Not attempted. Both touch how `agent` spawns a backend, which is a cross-module
surface (`docs/interfaces.md` §1.1), and neither should be chosen without reading
the backend's own argument handling first.

**Not measured, and worth measuring before (2) is chosen:** whether any *other*
`env` block this code builds — validation environments, the remote side — reaches
a child's `argv` by a different route. The finding above is one call site, found
once, and nobody has swept for siblings.


### 32 — Two of the four strict xfails may encode shipped bugs, not unbuilt features

PR #167 review R42 (PLAUSIBLE). `xfail(strict=True)` is this repository's marker
for *specified and not yet built*. The review argued that two of the four encode
confirmed defects in code that already ships, which the marker then hides as
planned work. It did not settle which two. The four:

- `tests/agent/test_transform.py::test_transform_lossless`
- `tests/cli/test_package_loads.py::test_something_publishes_a_handoff`
- `tests/env_mgr/test_layout.py::test_a_staged_package_still_carries_the_validators`
- `tests/task_graph/test_versioning.py::test_a_consumer_is_pinned_to_the_version_its_producer_actually_published`

**Closes when:** an owner classifies each one, and any that is a shipped bug is
tracked here as a defect rather than carried as an xfail.

### 33 — An unknown `--var` is accepted silently

PR #167 review R10. `--var K=V` with a `K` no spec references exits 0, so a
typo'd name runs the package on its defaults and nothing says so.

**Why not done:** the CLI does not know which names a package references. That
needs `spec_loader.substitute` to report the names it was asked for, carried out
of `YamlPackage.documents()` and checked on both the `show` and `run` paths —
about two dozen lines, judged over budget for the review pass.

**Closes when:** both paths warn (or refuse) on a supplied name no spec asked for.

### 34 — `give_back` clamps overspend, while `charge` says spend is never refused

PR #167 review R24. `ConsumableMgr.give_back` settles `min(actual, amount)`, as
`task_graph/docs/design.md` specifies and
`test_a_consumable_clamps_actual_to_the_reservation` asserts — so spend beyond
the reservation is dropped. `charge`'s own docstring says *refusing to book spend
that has already happened is the one thing a budget must never do*. The two
rules disagree about the same tokens.

**Closes when:** the owner picks one: book the excess through `charge`, or state
in the design why a reservation caps what is recorded.

### 35 — The Claude SDK adapter's loop has one driver at a time

PR #167 review R01, mitigated rather than fixed. `_run` holds `_loop_lock` for a
whole response; `interrupt` and `_terminate` now send the control request onto
the running loop instead of waiting for the lock
(`_interrupt_in_flight`), so a long turn no longer holds off a stop.

**What remains:** anything else that needs the loop mid-response still waits.
The structural fix is a loop on its own thread with every caller submitting
through `run_coroutine_threadsafe`, which reorders the SDK's shutdown and was
judged too wide for the review pass.

**Closes when:** the loop runs on a dedicated thread.

### 36 — A program body's grandchildren outlive `_terminate` outside bwrap — PARKED

PR #167 review R02. `ProgramBackend._terminate` signals the direct child only.
Under bwrap, `--unshare-pid --die-with-parent` reaps the whole tree; under
Landlock or with confinement off, a body's servers survive it.

**Parked, by owner decision:** what a body starts (sglang, a bench, a container)
is the task's to clean up, not the executor's. A process-group kill would also
take the body out of the terminal's foreground group, so Ctrl-C would orphan its
children unless the CLI handled `SIGINT`.

---

## Closed

Kept because an item closed by construction is worth a reader knowing about.

### C1 — What tasks the e2e test picks

`examples/as.e2e.test.yaml` is the artefact, and
`examples/run_simple_e2e_test.sh` runs it. An entry there is a claim that running
that package exercises the system end to end **and the outcome is checkable
without a human reading prose** — so a case belongs only when its expected result
is a specific exit code and, where that would be ambiguous, a specific string the
run must emit.

### C2 — Cross-handoff input validation input naming

Lookup is by handoff **uuid**, and `handoff/pointer.py` is RFC 6901 over
`python-jsonpath` — the one library of six that separates malformed / missing /
null. Two inputs of one kind are two uuids, so the ambiguity cannot arise.
`tests/handoff/test_pointer.py`.

### C3 — Digest canonicalisation

`handoff/digest.py`: `ALGORITHM = "agent_sys.handoff.tree.v1"`, sha256 git-shaped
over the subtree, and `canonical()` is RFC 8785 JCS via `rfc8785` — wrapped by an
encoder that **rejects `-0.0`**, which the library emits as `0` (measured).
Written down rather than decided by accident.

### C4 — Task-agent env reuse, the alpha half

The general mechanism stays deferred ([`ROADMAP.md`](ROADMAP.md) §6). The
validation-phase answer is `validator/environment.py::choose_configuration`, spec
§8.2's chain: the bound env, else the consumer's for input validation and the
producer's for output validation, else a predefined global one. **Item 5 is what
that chain still gets wrong.**

### C5 — The import graph is enforced rather than intended

`interfaces.md` §4 gives each package the set it may import and
`tests/interfaces/test_import_rules.py` walks every file's AST against it. The
rule is keyed on the package *name*, so a name that matches no package can never
appear in an import set and the assertion passes against every possible tree —
vacuous rather than failing. `test_every_allowed_package_exists` is the guard
against that class.

### 25 — `materials` is declared by two schemas and read by nothing

A task spec carries a `materials` key — *things this task may need for itself* —
and a validator body carries the same key. **Nothing reads either.** Every
occurrence in the design set is a reference to the key existing.

The intent is that the system eventually wraps what is declared there into a
handoff, while leaving the author free about its content. Until something does,
a package that declares `materials` gets silence.

**Closes when:** either a consumer is built, or the key is withdrawn from both
schemas. Those are opposite answers and the choice belongs to `closure` and
`validator` jointly.

### 26 — `handoff.resolve` has no caller

`resolve` is the RFC 6901 addressing into a handoff's content, and **no module
calls it.** Three documents say `validator` consumes it, two of them
`validator`'s own.

The reason is structural rather than an oversight: a validator's implementation
is a **body** now, not a registered Python callable, and a body reads its
staged inputs from disk. So the in-process addressing surface has no caller by
construction.

**Closes when:** either a body is given a route that goes through it, or
`resolve` is recorded as a library function for package authors rather than an
internal seam.

### 27 — Two version allocators for one artefact, and nothing joins them

One artefact has two independent version numbers: the **slot** version, advanced
by the handoff record, and the **store** version, the directory the bytes land
in. Nothing reconciles them, and they diverge whenever a dispatch allocates a
version the task never fills (item 6 is one way that happens).

**What it costs is legibility, not correctness** — every store reader filters on
the manifest, so a hole is invisible to them. The cost is a reader comparing the
log against the tree and finding two counters.

**Closes when:** one of the two is derived from the other, or the two are named
differently everywhere so nobody reads them as the same number.

### 28 — A non-leaf may declare an output no entry can produce

A parent's outputs are wired to its subgraph **through the end entry alone**. A
kind the parent declares and the end entry does not produce therefore gets a
fresh, unconnected handoff: the parent's declaration is satisfied by something
nothing writes into.

**Nothing catches it.** It is legal at load and silent at run time.

**Closes when:** the graph check rejects a parent output that no entry produces.
`task_graph`'s, and the check does not exist.

### 29 — `ValidatorId` — a fourth typed id the spec asks for and nothing needs

The whole-system spec asks for a typed id joining `TaskId` / `AgentId` /
`HandoffId`. Three designs key validators **by name** instead, consistently, and
the reason is good: a validator is not instantiated per run the way an agent is,
so what it needs is a unique vocabulary entry, not a per-object identity.

**Nothing is blocked.** Build on names.

**Closes when:** either the spec drops the request, or something appears that
genuinely needs per-instance validator identity — at which point the id is an
addition rather than a migration.

### 30 — `SCHEMA_VERSION` will have two owners once the whole-system CLI exists — BLOCKED

`cli`'s event stream is a versioned interface because an acceptance criterion
asserts over it. The whole-system CLI (item 18) will want the same stream, and
then two artefacts share one constant with no bump policy.

**Fine until item 18 lands.** Closes with it.
