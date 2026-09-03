# Acceptance criteria for the `env_checker` run

**Written before the run.** `.claude/CLAUDE.md` principle 2 asks for that, and
principle 1 says why: a stage in this repository once reported *fourteen tasks
and ten validators PASS* over a run in which every result was zero. Criteria
written afterwards are criteria fitted to whatever happened.

Every claim below names **a file to open** and **a condition that fails**.
Nothing here is satisfied by an exit code.

---

## 1. The invocation

```sh
cd /home/yihou/dev/git.16-19/infera.aiopt.real.task_package/agent_sys

ENVCHK_RUN_NONCE="$(python3 -c 'import secrets;print(secrets.token_hex(16))')"
echo "nonce for this run: ${ENVCHK_RUN_NONCE}"      # keep it; the validators need it

python3 -m cli.main run \
  --package examples/env_checker \
  --demo-root /tmp/yihou/agentsys_envchecker_20260903/runroot \
  --json      /tmp/yihou/agentsys_envchecker_20260903/runroot/stream.json \
  --var nonce="${ENVCHK_RUN_NONCE}" \
  --var uv_root=/tmp/yihou/agentsys_envchecker_20260903/uv \
  --timeout 3600
```

**`python3 -m cli.main`, from inside our own `agent_sys/`, never bare
`agent-sys`.** The globally installed console script in `~/miniconda3/bin`
resolves to a **different worktree** —
`/home/yihou/dev/git.16-19/infera.aiopt.all/agent_sys` — which another Claude
session is working in. Running it would load this package with somebody else's
`spec_loader`: a schema error that is not our bug, or worse, a pass that means
nothing.

**`--demo-root` on local disk.** Runs, handoffs, playground and workspace all
relocate together under it. `/tmp/yihou/...` is local, is ours, and is where
this task's scratch belongs.

### Why each variable is required rather than defaulted

| `--var` | why no default |
|---|---|
| `nonce` | Every token is `sha256(f"{salt}:{label}:{nonce}")[:12]`. A default is a constant; a constant nonce makes every run's seven tokens identical, and the **first published handoff then contains the answers to every run after it**. `${nonce}` with no `:-` is a load-time fault naming file and line — measured: `steps/check.yaml:75:5` |
| `uv_root` | serena's L1 install is `uv tool install`, whose defaults write `~/.local/share/uv`, `~/.local/bin` and `~/.cache/uv` — host state outside every zone, on a box we share, **and it succeeds while doing it**. A default here would ship one machine's scratch path, with a username in it, as everyone's |

### The fourth pin: the package must be in a commit

**`selftest/launch.sh` refuses to launch when `git status --porcelain` is
non-empty for `examples/env_checker` or `components/`.**

Run 2 is why. Its pins recorded `env_mgr` at `9a9fdff`, which was true and
**not sufficient**: the serena wiring, self-test case 2 and the validator that
produced the verdict were all working-tree edits. Anyone reproducing from
`9a9fdff` gets a different package than the one that ran, and **no pin recorded
after the fact can repair that** — the run is simply not attributable.

Three pins were never enough. A run is attributable to a **pair** of SHAs: the
package that ran and the `env_mgr` tree it ran against.

A gate and not a recorded `git diff`, deliberately. The diff would be evidence
and would still let an unattributable run happen. Refusing is the only version
that cannot be skipped by judging a change small — which is precisely the
judgement this gate exists to not depend on, the same argument as re-running the
pre-flight after every `env_mgr` commit.

### The pin

`.claude/CLAUDE.md` principle 3: *"`ls -1td runs/ | head -1` is not 'my run' —
this box is shared."* Two things pin it, and both happen **at launch**:

1. **`--json <path>`.** The stream file is named by us before the run starts, so
   the run it describes is ours by construction. Its first records carry the run
   id. This is the pin; the rest is convenience.
2. **`echo` the nonce.** Independently of any id, our run is *the run whose
   handoff carries `nonce_digest == sha256("nonce:"+ENVCHK_RUN_NONCE)[:12]`*.
   That is checkable against a handoff on its own, with no run directory, and it
   is what settles an argument about which of two directories is ours.

Record both in the run log before reading any artefact:

```sh
python3 -c 'import hashlib,os;print("expect nonce_digest:",
  hashlib.sha256(("nonce:"+os.environ["ENVCHK_RUN_NONCE"]).encode()).hexdigest()[:12])'
grep -m1 -o '"run_id"[^,]*' /tmp/yihou/agentsys_envchecker_20260903/runroot/stream.json
```

---

## 2. What is accepted, per capability

The artefact is the handoff's content directory. Under `--demo-root`, that is
the published `env_report` version's `content/`; open it and read
**`items/text.json`**.

Shorthand below: `T(label)` is
`"ENVCHK-" + label.upper() + "-" + sha256(f"{salt}:{label}:{nonce}")[:12]`,
where `salt` is the single `ENVCHK_SALT: <32 hex>` tag in the named artefact and
`nonce` is `$ENVCHK_RUN_NONCE`.

| # | capability | level | the file to open | **the condition that fails** |
|---|---|---|---|---|
| 1 | skill | L3 | `items/text.json` → `capabilities.skill` | `.token != T("skill")` computed from the salt in `assets/env_probe.agent/.claude/skills/envchk-probe/SKILL.md`; or `.level != "L3"`; or `.status != "ok"` |
| 2 | hook | L3 | same → `capabilities.hook` | `.token != T("hook")` from the salt in `.claude/hooks/envchk_session_start.py`; **or** `.proof.record.payload.session_id` is absent; **or** `.proof.record.payload.hook_event_name != "SessionStart"`; **or** `.proof.record.token != .token` |
| 3 | plugin | L3 | same → `capabilities.plugin` | `.token != T("plugin")` from the salt in `.claude/plugins/envchk-plugin/skills/envchk-plugin-skill/SKILL.md`; or `.proof.plugin_list` empty. **A `.token` equal to section 1's is a fail even if well-formed** — two routes, two salts |
| 3b | plugin **source path** | L3 | the zone's `config/settings.json` | its marketplace entry's `{"source":"directory","path": …}` is **not** under the run root — e.g. it points at `/home/yihou/dev/...`. Probe F measured that a plugin loads from its marketplace *source* directory rather than from a copy, so a marketplace outside the zone **installs cleanly and then fails to load under confinement with nothing naming the cause**. This is its own row and not a footnote precisely because it is a condition that must fail loudly rather than be checked if someone remembers |
| 4 | external MCP | **L2** | same → `capabilities.mcp_external` | `.token` differs from what the validator gets by **starting** `agent_sys/components/envchk-baseline/.claude/servers/envchk_baseline_server.py` and calling `tools/call`; or `.proof.raw.token != .token`; or `.level != "L2"` |
| 5 | bundled stdio MCP | L3 | same → `capabilities.mcp_stdio` | `.token` differs from what the validator gets by **starting** `.claude/tools/envchk_stdio.mcp.py`; or `.proof.raw.token != .token` |
| 6 | in-process ToolDef | L3 | same → `capabilities.tooldef` | `.token` differs from what the validator gets by **importing** `.claude/tools/envchk_inproc.tooldef.py` and calling `TOOLS[0].call()`; or `.proof.raw.token != .token` |
| 7 | serena | **L1 install + L2 declaration** | same → `capabilities.serena` | **the package declares no serena MCP server** — `components:` must name `serena`, whose `.claude/.mcp.json` registers it; an install without a declaration gives the agent `No such tool available` and is how run 1 failed. Then: `.status == "ok"` and `.token != T("serena")` from the salt in `assets/env_probe.agent/serena_probe.py`; **or** `.status == "unavailable"` and `install_report` carries no non-`ok` entry mentioning serena; **or** `.proof.raw` is not a `find_symbol` response whose hit for `envchk_serena_token` carries `name_path`, `kind`, `relative_path` naming `serena_probe.py`, a `body_location` with integer `start_line`/`end_line`, and a `body` **containing the salt**. That schema was measured against Serena 1.28.1 on this host on 2026-09-03, not remembered — see *What a PASS does not prove* for what it is and is not worth |
| — | install report | — | same → `install_report` | fewer than **2** entries; or `install_report_source` empty. It must be the `outcomes` **array** out of `$AGENT_SYS_INSTALL_REPORT` (`agent_assets.install.json`), verbatim, `ok` entries included |
| — | the whole report | — | same → `nonce_digest` | `!= sha256("nonce:" + $ENVCHK_RUN_NONCE)[:12]`. This one condition invalidates every token in the file: the report was produced against a different nonce, so none of it is about this run |
| — | the README | — | `README.md` | any of `## Purpose`, `## Schema`, `## Method`, `## Limits` missing; fewer than 10 content lines; any `TODO`/`TBD`/`FIXME`/`XXX`/`<…>` placeholder |

Additionally, for every section: `how` under **80 non-whitespace characters**
fails, and `status` other than `ok` fails for anything but `serena`.

### And both validators must be green *on the artefact*

```sh
# inside the run root, the verdicts the two bodies wrote
grep -r 'check_env_report_shape\|check_capabilities_genuine' \
     /tmp/yihou/agentsys_envchecker_20260903/runroot/stream.json
```

**Green validators are not the acceptance; the table above is.** They are
checked because a green validator over a report that fails a row in that table
would mean the *validator* is broken, and that is worth knowing separately.
`check_capabilities_genuine`'s own body recomputes rows 1–7, so in practice the
table and the verdict should agree — and a disagreement is a finding, not a
tie-break.

---

## 3. What a PASS does **not** prove

Read this before deciding the run succeeded. Stated per capability in
`assets/check_capabilities_genuine.validator/readme.md`; here in one place.

- **Four of the seven artefacts are files an agent with `Read` can open** — the
  two `SKILL.md`s, `serena_probe.py`, and the hook script. An agent that opened
  them and reported the tokens passes. This is not closable in-band: the agent
  and the artefacts share the zone **by construction**, because putting them
  there is the thing being measured.
- **What the tokens do buy, in full**: an agent cannot report seven tokens if
  the seven capabilities were not installed into its zone, because the salts
  exist nowhere else — not in the brief, not in `assets/lib/envchk.py`, not in
  either validator. A run where `env_mgr` silently delivered nothing produces no
  salts and therefore no tokens, however confident the narrative. **That** is
  the failure this package exists to catch.
- **Three are stronger than the other four.** Rows 4, 5 and 6 are re-derived by
  the validator *running the capability*, so their tokens have no file-read
  path. Row 2's payload (`session_id`, `hook_event_name`) can only come from the
  harness invoking the hook.
- **Freshness is not established.** `pid` and `at` are checked for shape, not
  value: the validator has no run window, and a freshness rule would be a number
  with no basis.
- **A forged hook payload is not caught.** An agent that found its own session
  id and wrote the file itself passes row 2.
- **Serena being *used* is not established**, only that it was installed and
  that its salt was reachable. Row 7 checks `proof.raw` against the response
  **shape** Serena 1.28.1 actually returns — measured on this host on
  2026-09-03 by driving the installed binary, not remembered — and that is a
  **forgery-cost increase, not a closure**: it moves the bar from *read one
  file* to *read the file **and** know serena's response schema*. **A model that
  has seen serena's schema can still fabricate it.** Two further consequences,
  stated rather than discovered: a serena release that changes that schema turns
  an honest run red, which is a validator update and not a capability failure
  (the failure message says so); and the salt sits inside the function body
  rather than at module scope purely so that `find_symbol`'s body carries it —
  without that, an agent could call serena correctly and still have no salt.
- **One green run is not three.** The lead's acceptance for this round is
  written + one real run; this file does not claim reproducibility.

---

## 4. Abort conditions

Stop the run rather than let it finish and read the wreckage:

| condition | why | what to do |
|---|---|---|
| **`prepare` exceeds ~10 minutes with no output** | **The operator's rule, and it is deliberately tighter than the machine's.** `installers/base.py::run_cmd` still cannot bound itself — it is `subprocess.run(shell=True)` with no timeout — but the *parent* now does: `env_mgr/agent_assets.py::RECIPE_TIMEOUT_SECONDS` is **20 minutes**, so a hung recipe ends by itself as a named failure rather than hanging forever. The two do not conflict; they answer different questions. **10 min: a human decides this run is not going to teach us anything and stops it. 20 min: the machine guarantees it stops regardless.** A bound is not a reason to stop watching — a networked install that is going to fail usually shows it long before either number | interrupt; re-run with `recipes:` temporarily removed to separate an L1 hang from everything else |
| **anything is written under `$HOME` outside `/home/yihou/dev/...`** — in particular `~/.local/share/uv`, `~/.local/bin`, `~/.claude/plugins` | host state on a shared box, and the whole point of `${uv_root}` and the relocated `CLAUDE_CONFIG_DIR` | stop immediately, report, do **not** clean up by deleting |
| **a second `agent-sys` run appears under our `--demo-root`** | the other Claude session on this box; two runs interleaving in one root makes every artefact ambiguous | stop, re-pin, do not delete the other run |
| **the agent starts editing anything under `$AGENT_SYS_AGENT_ASSETS` or `$CLAUDE_CONFIG_DIR`** | those are the subject of the measurement; a salt edited to make a token match makes the whole deliverable worthless | stop, keep the transcript, report it as the finding it is |
| **`ENVCHK_NONCE` appears anywhere in the handoff** | the brief forbids it, and a published nonce lets the next run's report be computed from this one's deliverable | stop, do not publish |

Not abort conditions — let these finish and read the result:

- serena failing to install. That is a `warn` in the install report and a
  legitimate `unavailable`, and it is a row this design is built to accept.
- an agent reporting a capability as failed. That is a result.

---

## 5. Pre-flight

Run in order, from inside our `agent_sys/`. Each item is a command and the
output that means *proceed*.

**The results go on disk, not into a message.** `selftest/preflight.sh` runs
every row below and writes the command **and its actual output** to
`/tmp/yihou/agentsys_envchecker_20260903/logs/PREFLIGHT.md`, with the git HEAD
and worktree state it ran against. A reported 10/10 is a claim; that file is the
artefact — which is the standard §2 applies to the run, turned on the gate that
precedes it. Re-run it after any commit that touches `env_mgr`. **Do not run any of these while `core-impl` is in
`env_mgr/`** — a failure seen then is a failure that cannot be attributed.

```sh
cd /home/yihou/dev/git.16-19/infera.aiopt.real.task_package/agent_sys
```

| # | command | proceed when |
|---|---|---|
| 1 | `python3 -m cli.main show --package examples/env_checker --var nonce=x --var uv_root=/tmp/yihou/x` | `2 tasks in the graph`, and both `output validation runs 2` |
| 2 | `python3 -c 'from env_mgr import paths; print(paths.COMPONENTS_ROOT_ENV_VAR, paths.INSTALL_REPORT_ENV_VAR, paths.AGENT_ASSETS_ENV_VAR)'` | three names print. An `AttributeError` means the constants do not exist. **This row is about the names, not about the things the names refer to** — see below |
| 3 | `python3 -c 'from env_mgr.recipe import load_recipe; print(len(load_recipe("env_mgr/recipes/serena.yaml")[1]), "items")'` | `3 items`. A `FileNotFoundError` means the L1 recipe is not placed yet and section 7 cannot pass |
| 4 | `python3 -m cli.main run --package examples/env_checker --demo-root /tmp/yihou/agentsys_envchecker_20260903/dryrun --var nonce=x --var uv_root=/tmp/yihou/x --dry-run` | completes, dispatches nothing, no `REJECTED` |
| 5 | `claude plugin validate examples/env_checker/assets/env_probe.agent/.claude/plugins` | `Validation passed` |
| 6 | `command -v uv; command -v claude; claude --version` | all three answer. `uv` is what the L1 recipe installs serena with; `claude` is what installs the L3 plugin. **This row proves the binaries exist on this host and nothing about which build the run uses** — that is 6b |
| 6b | `python3 -c "import shutil,subprocess;p=shutil.which('claude');print(p);print(subprocess.run([p,'--version'],capture_output=True,text=True).stdout.strip())"` | prints a path and **`2.1.246`**. That is the build the run pins **and** the build every probe conclusion now holds on — see *Measured on the pinned build* below. A different version is a **stop**, and it is a stop for two reasons at once: the run would invoke a CLI nobody characterised, **and** the probe evidence would no longer apply to it. The discharge is re-measuring the probes on the new build, not proceeding carefully |
| 7 | `python3 /tmp/yihou/agentsys_envchecker_20260903/selftest/run.py` | `ALL OK` — 20 cases, both validators driven through their real `entry.sh` over a synthetic handoff. Case 2 is `every-capability-is-declared`: every capability reached through MCP has an artefact declaring its server name, which is run 1's failure catchable with no run at all. Case 1 is `salts-are-isolated`: seven `ENVCHK_SALT:` tags across the package and the components registry, all distinct, **each value in exactly one authored file**, and none of them in the three documents that describe the scheme. That is the property every other case assumes, so it is checked mechanically rather than argued |
| 8 | `git --git-dir=/home/yihou/dev/git.16-19/infera/.git config --get extensions.preciousObjects` | prints `true`. **If it prints nothing, stop and ask the user** — see below |
| 9 | `df -h /tmp \| tail -1` | room to spare. Measured 2026-09-03: `/tmp` is on the 7 TB NVMe with ~3 TB free, and the workspace is a `git clone --shared`, so the main repo's 129 MB of objects are reached through alternates rather than copied. Disk is not a constraint for this run; the row exists so that a full disk is ruled out rather than assumed |

**Row 2 is weaker than it looks, and the weakness is named rather than fixed.**
Printing three constants proves the constants exist. It would have passed
*before any code assigned them*, and it still passes if nothing ever does — the
check is about the **name**, not about the thing the name refers to. That is the
same defect as `command -v` in the wrong shell and a probe on the wrong binary,
in a third costume.

Both exports are in fact **conditional** (`agent_assets.install`):
`AGENT_SYS_COMPONENTS_ROOT` is set only when the agent declares `components:`,
and `AGENT_SYS_INSTALL_REPORT` only when `logs_dir` is not `None`. Both
conditions hold here — this agent declares `components: [envchk-baseline]`, and
`material.deploy` always passes `logs_dir=<zone>/logs` — but **that was
established by reading the code, not by row 2**, and a reader who trusts the row
has not learned it.

What actually discharges it is the run: `$AGENT_SYS_INSTALL_REPORT` resolving to
a real file is what the `install_report` acceptance row reads, and the L2 salt
being re-derivable is what `check_capabilities_genuine` reports by name if
`AGENT_SYS_COMPONENTS_ROOT` is absent. A stronger pre-flight row would have to
call `agent_assets.install` for real, which runs the L1 recipe — too heavy for a
gate, and it would be testing `env_mgr`'s own code, which has its own tests.

**Row 6b reproduces the resolution the launcher actually performs**, rather than
asking a shell. `cli/environment.py::build_context` sets `agent_cli=shutil.which(BACKEND)`
with `BACKEND = "claude"` — resolved in the **launching process**, so 6b must be
run from the same shell that will run the launch, which is what a pre-flight is
for. It is deliberately *not* `command -v`: measured 2026-09-03, this host
carries two builds — `/usr/local/bin/claude` 2.1.197 (root, npm) and
`/home/yihou/.local/bin/claude` 2.1.246 — and which one answers depends on whose
`PATH` is asking.

**Why a version equality and not merely a path.** Every claim this package makes
about plugins, hooks and MCP rests on the probes, and **a probe is evidence
about the build it ran on**. Carrying it to another build is the same error as
encoding an unmeasured schema, one layer up. So 6b pins the version and a
mismatch stops the run; re-measuring the probes on the new build is the
discharge, not a shrug.

### Measured on the pinned build

That rule caught the probes themselves. B, C and F built `ClaudeAgentOptions`
**without `cli_path`**, and `SubprocessCLITransport._find_cli` returns the SDK's
*bundled* binary before it ever consults `PATH` — so those three ran on
**2.1.251**, not on the 2.1.246 the run pins. Three of the six conclusions this
package rests on were about a build the run would not use, and **no artefact
would have said so**.

They were re-measured on 2026-09-03 with `cli_path` pinned to 2.1.246, one
session doing all three (`probes/probe_bcf_on_pinned_cli.py`), and all three
came back positive: **B'** the `SessionStart` hook fires and its token matches;
**C'** an external `mcp_servers` entry reaches the model, returning a value only
the server process holds; **F'** the plugin is visible and again names the
**marketplace source path** as its skill base directory. That run's plugin
install also went through the pinned CLI, so A/A' hold on 2.1.246 by repetition
rather than by inference.

**Cite B'/C'/F', not B/C/F.** The originals are not wrong; they are about
2.1.251, which is not the build this run uses.

The comment on that assignment, inside `build_context`, had already written the
hazard down — *"execute a different build from the one `env_mgr` installed plugins into
and succeed without them"* — which is exactly the failure the row-6 caveat
surfaced. The row exists so that a known-shaped risk is checked rather than
remembered.

**Item 8 is the one that would have killed the run before anything we wrote
ran.** `env_mgr/workspace.py::cut` refuses to cut a workspace unless the main
repository has `extensions.preciousObjects` set — *"refusing to cut a workspace
that a `git gc` in the main repository would silently corrupt"*. That is
`prepare` **step 4**, so the failure would land before a single component was
installed, with an error about git rather than about this package.

It is already `true` on `/home/yihou/dev/git.16-19/infera/.git`, set by some
earlier `agent_sys` run. **So `--allow-repo-config` is neither needed nor
permitted here** and must not appear in the invocation: that flag exists to let
the demo set the config *itself*, and writing config into a repository shared by
eleven worktrees is exactly the host-state change this task may not make without
asking. If item 8 ever comes back empty, that is a question for the user, not a
flag.

**Item 5 checks the manifests, not the path.** `claude plugin validate` cannot
see where the marketplace will resolve *at run time*, and that is the condition
that actually matters — so it is **acceptance row 3b** above, with a file and a
failing condition, rather than a note here to be remembered afterwards.

---

## 6. One fact about timing, measured rather than assumed

**L1 runs during `prepare`, not during the agent's session.**
`env_mgr/material.py::deploy` calls `agent_assets.install(...)`, and that
function runs every recipe in its `_recipe_paths(...)` loop — **before the agent
exists**. So a slow `uv tool install` costs wall-clock on the *preparation*, the
agent's own clock is unaffected, it never waits for serena and cannot mistake a
slow install for a broken one. Nothing in the brief needs to tell it to wait.

The cost is moved rather than removed, which is why *"`prepare` exceeds ten
minutes"* is an abort condition above and not a footnote.

> **Symbols, not line numbers.** This section cited `material.py:155` and
> `agent_assets.py:280` until 2026-09-03, and both were wrong within hours of
> being written — the real call sites moved to `material.py:194` and
> `agent_assets.py:380` when `base_env` was threaded through. A line number in a
> document that outlives the code is a maintenance trap that reads as precision.
> Cite the **function**; if a line helps a reader find it, give it as a hint and
> expect it to rot. The same applies to every other reference in this file.
