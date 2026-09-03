# `examples/env_checker` — does the environment an agent is promised actually arrive?

One leaf, one handoff kind, two validators, and a subject that is `agent_sys`
itself: **seven Claude Code capabilities, installed per-agent, across three
install levels**, each one exercised and each one reported with a token that a
run which did not install it cannot produce.

```
main                        non-leaf: readme, no entry.sh, NO agent
│                           inputs [] · outputs [env_report]
│
└── probe_env               is_end · ai: env_probe
                            out: env_report                [structured_text]
                              check_env_report_shape     seconds · strong · completeness
                              check_capabilities_genuine minutes · strong · trustworthiness
```

| # | capability | level | delivered by |
|---|---|---|---|
| 1 | skill | L3 | `assets/env_probe.agent/.claude/skills/envchk-probe/` — auto-detected |
| 2 | hook | L3 | `.claude/settings.json` → `hooks/envchk_session_start.py`, at `SessionStart` |
| 3 | plugin | L3 | `.claude/plugins/` — a local marketplace, `claude plugin install` |
| 4 | external MCP server | **L2** | `agent_sys/components/envchk-baseline/.claude/.mcp.json` |
| 5 | bundled stdio MCP server | L3 | `.claude/tools/envchk_stdio.mcp.py` — location is the declaration |
| 6 | in-process `ToolDef` | L3 | `.claude/tools/envchk_inproc.tooldef.py` → `mcp__env_mgr__envchk_echo_token` |
| 7 | serena | **L1** | `recipes: [serena]` — the real thing, over the network |

## Run it

```sh
agent-sys run --package agent_sys/examples/env_checker \
  --var nonce="$(python3 -c 'import secrets;print(secrets.token_hex(16))')" \
  --var uv_root=/tmp/$USER/agentsys_uv
```

Both variables are **required and have no default**; `steps/check.yaml` argues
each at the agent's `env` block. `nonce` because a constant nonce would make the
first published handoff contain every later run's answers; `uv_root` because
serena's L1 install is `uv tool install`, whose defaults write `~/.local/share/uv`
— host state outside any zone.

**Accept by opening the handoff, not by reading the exit code.**
[`ACCEPTANCE.md`](ACCEPTANCE.md) is the criteria, written before the run: the
exact invocation, one row per capability naming the file to open and the
condition that **fails**, what a PASS does not prove, how the run id is pinned
on a shared box, the abort conditions, and an eight-item pre-flight.

## Why tokens

Every capability carries `sha256(f"{salt}:{label}:{nonce}")[:12]`, where the
salt lives in exactly one place — that capability's own artefact — and the nonce
is per-run. There is **no table of salts anywhere**, including in
`assets/lib/envchk.py`: one file listing all seven would let a single read
produce all seven tokens.

This exists because of `.claude/CLAUDE.md`'s first principle. A previous stage
in this repository reported fourteen tasks and ten validators PASS over a run in
which every result was zero. The general form of that failure is a producer
being believed about its own environment, and a token is the cheapest thing that
cannot be produced by belief.

**What that buys, stated exactly**: an agent cannot report seven tokens if the
seven capabilities were not installed into its zone. It does **not** prove the
agent obtained each token through the capability rather than by reading the
file — four of the seven artefacts are readable files, and the agent and the
artefacts are in the same zone by construction, because putting them there is
the thing being measured.
`assets/check_capabilities_genuine.validator/readme.md` closes that gap for the
three capabilities that are processes — it starts both MCP servers itself and
imports the ToolDef module — and states the residual for the rest, per
capability, in its *What it cannot catch* section.

## Layout

```
main.yaml                                       root non-leaf
steps/check.yaml                                agent + 2 validators + handoff kind + task
assets/
  main.task/readme.md
  probe_env.task/readme.md                      the brief: use all seven, report evidence
  env_probe.agent/
    README.md                                   maps the .claude/ tree and argues settings.json
    serena_probe.py                             section 7's subject — NOT a capability
    .claude/…                                   L3: six of the seven
  lib/zone.py                                   the four body-facing zone files
  lib/envchk.py                                 the token scheme and the capability register
  check_env_report_shape.validator/
  check_capabilities_genuine.validator/
```

`agent_sys/components/envchk-baseline/` is L2 and lives outside this package on
purpose: a component that only one package can reach is not a component.

## Known gaps

Written down rather than left to be discovered.

1. **`$AGENT_SYS_COMPONENTS_ROOT` is expected and not yet exported.** A task
   package is staged into a zone, so `agent_sys/components/` — a *repository*
   path — has no relative route from it. `check_capabilities_genuine` takes that
   variable first and searches upwards from the package second; in a staged run
   only the variable can answer, and when neither does, the L2 capability is
   reported unverifiable **by name**. That is a fault, not a shrug, and the
   variable is what closes it.

2. **`$AGENT_SYS_INSTALL_REPORT` is expected and not yet exported.** The brief
   tells the agent to look for it first and to fall back to a search under
   `$AGENT_SYS_MY_LOGS`. Without a defined location, `install_report` depends on
   the agent finding a file whose path nobody promised — and
   `check_env_report_shape` fails a report that omits it, which is the correct
   verdict and an avoidable one.

3. **`.mcp.json` values need `${VAR}` expansion at load.** The component writes
   `"args": ["${CLAUDE_CONFIG_DIR}/servers/envchk_baseline_server.py"]` because
   an absolute path in a component is one machine's answer. Whoever loads
   `.mcp.json` into `Prepared.mcp_servers` has to expand it; unexpanded, the
   server does not start and the symptom is a server with no tools rather than
   an error.

4. ~~`$UV_TOOL_BIN_DIR` has to reach `PATH`.~~ **Closed, by not using `PATH`.**
   serena's MCP entry names the binary absolutely —
   `"command": "${UV_TOOL_BIN_DIR}/serena"` — which is the form measured working
   (probe E) and needs no search path, no grant and no ordering change.

   Kept as a numbered entry rather than deleted, because the reason `PATH` was
   never going to work is a fact about this system worth having written down:
   a task body's `PATH` is derived from the policy
   (`isolation/policy.py::executable_path`), the policy is composed at `prepare`
   step **2**, and `material.deploy` does not install anything until step **6b**
   — so the directory does not exist when `PATH` is computed and would not
   appear on it even unconfined. It is the same reasoning that put
   `${CLAUDE_CONFIG_DIR}/servers/…` in the component: the variable, not the
   literal, and not the search path.

5. **No `resources` block.** A leaf may declare a pool; nothing here needs one,
   and `cli/build.py:85` — the only reader — declares no pools anyway.

## Deferred, on purpose — the follow-up list

**Items with triggers, not a wish list.** Each says what would make it worth
doing; an item nobody can tell has become due is a wish.

### 1. `6b` can become genuinely independent

Today `6b` compares `proof.raw.path` against a path the validator derives
itself, and cross-checks it against the install report — but that cross-check is
**consistency only**, because the report reaches the validator through
`payload["install_report"]`, a field the *agent* supplies.

**Trigger: already met.** Runs 3 and 4 established by measurement that a real
validation zone **does** carry the environment — `check_capabilities_genuine`
re-derived the L2 row, which requires `AGENT_SYS_COMPONENTS_ROOT`, and the
upward-search fallback cannot fire from a zone. So the validator can read
`$AGENT_SYS_INSTALL_REPORT` **itself** and the comparison stops depending on the
agent. Three lines in a body that already reads that file. Not done this round
only because the round closed.

### 2. The placeholder regex in four other files

Described in full under *Out of scope, recorded* above. **Trigger: whoever next
runs one of those four packages**, or a decision to build the shared helper.
The repair wants to be **one shared change**, which is why it did not happen as
four copies here.

### 3. `agent_sys`'s in-process tool factory

Row 6 works around a contract, it does not fix one: an in-process `ToolDef` runs
in the supervisor and cannot see `Prepared.environment`, so a tool needing
per-run values has no supported route to them. `core-impl`'s axis argument is
the specification — **an argument binds per call, a closure per construction,
the environment per process, and the process outlives the attempt.**

**Trigger: the second package that wants a per-run value in an in-process
tool.** One package working around it is a workaround; two is a missing
feature, and the second one is the evidence that justifies changing the
contract.

### 4. A verdict is not attributable to a named validator from a run's artefacts

`scribe`'s escalation, and the largest gap between what this round could verify
and what it recorded. A run's artefacts say *a validation failed*; recovering
**which validator** and **why** required re-running the validator by hand — it
cost time twice on 2026-09-03, once at the worst possible moment, when the
reason for run 2's FAIL was not recoverable from the files at all.

**Trigger: already met, twice.** This is the one item on this list that is not
about `examples/env_checker`; it is about `validator`/`monitor` recording enough
to answer *which check said no*. Noted here because this package is where it
kept biting.

## Deferred, on purpose — the runtime declaration check

`selftest/run.py`'s case 2 catches **expected-but-declared-nowhere** before a
run: every capability reached through MCP has an artefact declaring its surface,
the brief states that surface verbatim, and every declared surface is claimed.
Three assertions, each demonstrated red.

It does **not** catch **declared-but-did-not-arrive** — a mis-set `${VAR}`, an
unplaced file, a component whose install failed. That check exists in design:
compare `Capability.surface` against the names `env_mgr` now records in
`$AGENT_SYS_INSTALL_REPORT` (`agent_assets` records `names` for the external
route, `server` for the bundled one, and `server` + `tools` for the in-process
one, as of `9a9fdff`). It is three lines in `check_capabilities_genuine`, which
already reads that file.

**It is deliberately not built for run 2**, and the reason is worth keeping:
**run 2 is itself the empirical check for that class.** A declared server that
does not arrive makes its capability fail, and the acceptance table says so. A
cheaper detector for something the expensive detector is about to run anyway
buys a tree move, a pre-flight re-run and a fresh mutation baseline, and the
tree moved six times on 2026-09-03.

If run 2 surfaces a declared-but-absent server, that is the evidence for
building it — and its shape will come from a real failure rather than from a
design. Written here rather than left in a thread, because *a comment is not a
declaration* and neither is a mailbox.

## One lesson from building the instruments, kept where the next author will look

Six checks in this package and its scratch tooling turned out to be **checks
that could not fail**, and the pattern in every one was the same: *the check
tested a proxy for the property, and the proxy was the right proxy — it just was
not the property.* `command -v` in a shell the subprocess does not run in; a
probe against the SDK's bundled CLI rather than the pinned one; a pre-flight row
importing a module and printing three constants, which would have passed before
the export code existed; a capability row keyed on a literal server name, green
with the tool renamed.

**Half of them were in the instruments rather than in the package**, and
instruments get less adversarial scrutiny precisely because they are the thing
doing the measuring. The way to know is to make the check go red: delete the
fix, rename the thing, point it at a copy.

Two corollaries, both bought on 2026-09-03 and both about *this* package's
tooling rather than about `agent_sys`:

- **A gate whose only self-test is a live launch will be tested by launching.**
  The dirty-tree gate in `selftest/launch.sh` was verified by running the
  script; the gate passed and the script then launched, starting a third run
  that overwrote two of run 2's logs. The careful action and the destructive one
  were the same command. Every gate now needs a `--check` that runs it and
  exits.
- **Fixing one instance of a class does not inoculate you against the class.**
  `preflight.sh` erased its own hand-written verdict on every run; that was
  found, argued and fixed by generating the verdict instead. The *same* bug — a
  fixed filename for a per-run artefact — was then written into `launch.sh`, the
  neighbouring script, and destroyed run 2's launch log. **The second instance
  arrives in the file nobody is looking at**, and having just fixed the first is
  what makes you not look.

## Out of scope, recorded: the same placeholder defect is in four other files

Run 3's `check_env_report_shape` failed on **correct input** — the agent
documented the token's format in its `## Schema` section, in backticks, and the
`<…>` placeholder rule could not tell *documenting* a placeholder from *leaving*
one. This package's copy is narrowed (code spans and fenced blocks are excluded
from the angle rule; `TODO`/`TBD`/`FIXME`/`XXX` still match everywhere).

**The same pattern is in four files this round did not touch:**

```
examples/single_real_task/assets/check_packup_shape.validator/check.py
examples/llm_e2e_performance_optimization/analyze-demo/assets/check_identity_resolved.validator/check.py
examples/llm_e2e_performance_optimization/deploy-demo/assets/check_deploy_kit.validator/check.py
examples/llm_e2e_performance_optimization/integration-demo/assets/lib/patchkit.py
```

Verified rather than assumed: `single_real_task`'s copy was run against the exact
line run 3 died on, and **it flags it too**. Nobody has hit it there only because
no author has yet written an angle-bracketed placeholder inside a code span.

**Not fixed here, and the reason is not etiquette.** Three of those are
`llm_e2e_performance_optimization`, a different deliverable with its own
acceptance history; one is `single_real_task`, the template. Changing a
validator in a package we are not running changes an acceptance criterion for
work that was accepted under the old one, **without re-running it** — the same
hazard as a drive-by edit to shared configuration. The right end state is one
shared helper rather than five copies drifting apart, and that touches four
files this round may not change.

**The mechanism is the part worth carrying.** This defect **propagates by
copy**: the regex was lifted from another validator that already had it, without
re-reading what it matched. **A defect that spreads by copying gets more
entrenched with every reuse, and each copy arrives carrying the authority of the
file it came from.** It is also a different species from the checks-that-cannot-
fail catalogued above — it **fails on correct input** — and the repair points the
opposite way: narrow it, never delete it. A real check that misfired once is
still a real check.

## Measured, so not assumed

Six probes on 2026-09-03, first-hand, written up at
`/tmp/yihou/agentsys_envchecker_20260903/logs/PROBES.md`. Every capability here
has a mechanism behind it that was run rather than read about:

| | measured |
|---|---|
| A | `claude plugin marketplace add` / `install` honour `CLAUDE_CONFIG_DIR`; `~/.claude` untouched |
| A' | they **merge** into an existing `settings.json` rather than clobbering it |
| B' | a `SessionStart` command hook in `$CLAUDE_CONFIG_DIR/settings.json` **fires for an SDK-started session** |
| C' | an external `mcp_servers` entry reaches the model and a real `tools/call` returns; the working shape carries `"type": "stdio"` |
| D | `uv tool install "git+https://github.com/oraios/serena"` returns rc 0 |
| E | the installed serena serves 21 MCP tools, `Serena 1.28.1` |
| F' | a plugin installed into the zone config **is visible to the session** — and loads from the marketplace **source** directory, not from a copy |

**B', C' and F' carry primes because the originals were about the wrong build.**
They were first measured through `ClaudeAgentOptions` with no `cli_path`, and
the SDK's `_find_cli` returns its own *bundled* binary — **2.1.251** — before it
consults `PATH`, while the run pins **2.1.246** through `Prepared.agent_cli`.
Re-measured on the pinned build, all three still hold; A and A' hold on it by
direct repetition. A probe is evidence about the build it ran on, and that
applies to probes about the harness exactly as it applies to probes about
serena. `ACCEPTANCE.md` pre-flight row **6b** is what keeps this true: it pins
the version, and a mismatch stops the run for two reasons at once — an
uncharacterised CLI, and evidence that no longer applies to it.

F is why `.claude/plugins/` sits inside the agent asset directory that gets
staged: a marketplace pointed anywhere else installs cleanly and then fails to
load under confinement with nothing naming the cause. It is also why nothing the
plugin ships may be a symlink or need a build step — those files are read from
the staged source path at run time.

## Related

- `examples/single_real_task/` — the AI-task template this package's shape,
  validator layout and `assets/lib/zone.py` come from.
- `examples/demo/main.yaml` — the non-leaf root.
- `agent_sys/components/README.md` — the L2 contract.
