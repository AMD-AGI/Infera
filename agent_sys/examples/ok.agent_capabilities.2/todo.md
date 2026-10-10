# `ok.agent_capabilities.2` — deferred work

What this package has not closed, and what would close each item. The criteria
themselves are in [`ACCEPTANCE.md`](ACCEPTANCE.md); what the package is and how
to run it is in [`README.md`](README.md).

---

## Known gaps

Written down rather than left to be discovered.

1. ~~`$AGENT_SYS_ADDONS_ROOT` is expected and not yet exported.~~ **Closed, by
   the variable being deleted rather than exported.** A task package is staged
   into a zone, so `agent_sys/env_mgr/addons/` — a *repository* path — had no
   relative route from it, and `check_capabilities_genuine` took the variable
   first and searched upwards from the package second. Both are gone: the
   package's own recipe layer **copies** section 4's server into the zone, so its
   artefact is one directory from the staged package and no variable names
   anything outside the zone. `env_mgr/docs/spec.md` §6.6 deleted the variable
   along with the declaration key it served.

   Kept as a numbered entry for gap 4's reason — the argument is worth having
   written down. `AGENT_SYS_ADDONS_ROOT` was the only exported path pointing
   outside a zone, and it needed a `READ_EXEC` grant to be usable at all; an
   exported-but-ungranted path is `paths.py`'s named defect. Copy-into-the-zone
   removes both halves rather than balancing them.

2. **`$AGENT_SYS_INSTALL_REPORT` is expected and not yet exported.** The brief
   tells the agent to look for it first and to fall back to a search under
   `$AGENT_SYS_MY_LOGS`. Without a defined location, `install_report` depends on
   the agent finding a file whose path nobody promised — and
   `check_env_report_shape` fails a report that omits it, which is the correct
   verdict and an avoidable one.

3. **`.mcp.json` values need `${VAR}` expansion at load.** The agent's
   `.claude/.mcp.json` writes
   `"args": ["${CLAUDE_CONFIG_DIR}/servers/envchk_baseline_server.py"]` because
   an absolute path is one machine's answer. Whoever loads
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
   and `cli/build.py` — the only reader — declares no pools anyway.

6. ~~`recipes: [serena]` does not resolve from a wheel install.~~ **Closed, and
   measured closed.** `agent_sys:serena` resolves against
   `agent_sys/env_mgr/recipes/`, which `pyproject.toml` did not ship as package
   data, so section 7 installed from a checkout and refused from a wheel. Both
   recipe layers are now `package-data`; the wheel was built, its members
   counted, and the reference resolved through `_recipe_paths` against an
   installed venv.

   Kept as a numbered entry because the *class* is the durable part: the same
   comment in `pyproject.toml` had already described this failure mode for
   `spec_loader/schemas`, and this was its third instance. The sweep that found
   it — and `default.env_recipe.yaml`, whose absence would have been **silent** —
   was `find env_mgr -type f ! -name '*.py'`.

## Deferred, on purpose — the follow-up list

**Items with triggers, not a wish list.** Each says what would make it worth
doing; an item nobody can tell has become due is a wish.

### 1. `6b` can become genuinely independent

Today `6b` compares `proof.raw.path` against a path the validator derives
itself, and cross-checks it against the install report — but that cross-check is
**consistency only**, because the report reaches the validator through
`payload["install_report"]`, a field the *agent* supplies.

**Trigger: met, and the subject is gone.** Runs 3 and 4 established by
measurement that a real validation zone **does** carry the environment —
`check_capabilities_genuine` re-derived the `mcp_external` row, which then
required `AGENT_SYS_ADDONS_ROOT`, and the upward-search fallback cannot fire
from a zone. That measurement stands and is what makes reading
`$AGENT_SYS_INSTALL_REPORT` from a validator body viable. **6b itself no longer
exists**, so the item survives only as the general point: any comparison this
package makes against `payload["install_report"]` is consistency and not
corroboration, because both sides come from the agent.

### 2. The placeholder regex in other files

Described in full under *Out of scope, recorded* above. **Trigger: whoever next
runs one of those packages**, or a decision to build the shared helper. The
repair wants to be **one shared change**, which is why it did not happen as
several copies here.

### 3. ~~`agent_sys`'s in-process tool factory~~ — closed by deletion

The item was: an in-process `ToolDef` runs in the supervisor and cannot see
`Prepared.environment`, so a tool needing per-run values has no supported route
to them, and row 6 worked around that rather than fixing it. **The route for
component-supplied tools is deleted** (`env_mgr/docs/spec.md` §9.2), so there is
nothing to give a per-run value to.

Kept, because the analysis outlives the feature and because one in-process
server remains — `env_mgr/remote/tools.py`, §6's standing exception, injected as
a live object and therefore subject to the same limit. `core-impl`'s axis
argument is still the specification: **an argument binds per call, a closure per
construction, the environment per process, and the process outlives the
attempt.**

### 4. A verdict is not attributable to a named validator from a run's artefacts

`scribe`'s escalation, and the largest gap between what this round could verify
and what it recorded. A run's artefacts say *a validation failed*; recovering
**which validator** and **why** required re-running the validator by hand — it
has cost time twice, once at the worst possible moment, when the reason for a
FAIL was not recoverable from the files at all.

**Trigger: already met, twice.** This is the one item on this list that is not
about `examples/ok.agent_capabilities.2`; it is about `validator`/`monitor` recording enough
to answer *which check said no*. Noted here because this package is where it
kept biting.

## Deferred, on purpose — the runtime declaration check

`selftest/run.py`'s case 2 catches **expected-but-declared-nowhere** before a
run: every capability reached through MCP has an artefact declaring its surface,
the brief states that surface verbatim, and every declared surface is claimed.
Three assertions, each demonstrated red.

It does **not** catch **declared-but-did-not-arrive** — a mis-set `${VAR}`, an
unplaced file, a recipe item whose install failed. That check exists in design:
compare `Capability.surface` against the names `env_mgr` records in
`$AGENT_SYS_INSTALL_REPORT` (`agent_assets` records `names` for entries read out
of `.mcp.json` and `server` for the bundled one). It is three
lines in `check_capabilities_genuine`, which already reads that file.

**It is deliberately not built**, and the reason is worth keeping: **the run is
itself the empirical check for that class.** A declared server that
does not arrive makes its capability fail, and the acceptance table says so. A
cheaper detector for something the expensive detector is about to run anyway
buys a tree move, a pre-flight re-run and a fresh mutation baseline, and the
tree moves often enough already.

If a run surfaces a declared-but-absent server, that is the evidence for
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

Two corollaries, both about *this* package's tooling rather than about
`agent_sys`:

- **A gate whose only self-test is a live launch will be tested by launching.**
  The dirty-tree gate in `selftest/launch.sh` was verified by running the
  script; the gate passed and the script then launched, starting a third run
  that overwrote two earlier logs. The careful action and the destructive one
  were the same command. Every gate now needs a `--check` that runs it and
  exits.
- **Fixing one instance of a class does not inoculate you against the class.**
  `preflight.sh` erased its own hand-written verdict on every run; that was
  found, argued and fixed by generating the verdict instead. The *same* bug — a
  fixed filename for a per-run artefact — was then written into `launch.sh`, the
  neighbouring script, and destroyed an earlier launch log. **The second instance
  arrives in the file nobody is looking at**, and having just fixed the first is
  what makes you not look.

## Out of scope, recorded: the same placeholder defect is in other files

`check_env_report_shape` can fail on **correct input** — an agent that documents
the token's format in its `## Schema` section, in backticks, trips the `<…>`
placeholder rule, which cannot tell *documenting* a placeholder from *leaving*
one. This package's copy is narrowed (code spans and fenced blocks are excluded
from the angle rule; `TODO`/`TBD`/`FIXME`/`XXX` still match everywhere).

**The same pattern is in files this round did not touch, among them:**

```
examples/ok.sglang_real_model.2/assets/check_packup_shape.validator/check.py
```

Verified rather than assumed: `ok.sglang_real_model.2`'s copy was run against the exact
line that triggered it here, and **it flags it too**. Nobody has hit it there
only because no author has yet written an angle-bracketed placeholder inside a
code span.

**Not fixed here, and the reason is not etiquette.** Three of those are
`llm_e2e_performance_optimization`, a different deliverable with its own
acceptance history; one is `ok.sglang_real_model.2`, the template. Changing a
validator in a package we are not running changes an acceptance criterion for
work that was accepted under the old one, **without re-running it** — the same
hazard as a drive-by edit to shared configuration. The right end state is one
shared helper rather than several copies drifting apart, and that touches files
this round may not change.

**The mechanism is the part worth carrying.** This defect **propagates by
copy**: the regex was lifted from another validator that already had it, without
re-reading what it matched. **A defect that spreads by copying gets more
entrenched with every reuse, and each copy arrives carrying the authority of the
file it came from.** It is also a different species from the checks-that-cannot-
fail catalogued above — it **fails on correct input** — and the repair points the
opposite way: narrow it, never delete it. A real check that misfired once is
still a real check.

## Measured, so not assumed

Six probes, first-hand. Every capability here has a mechanism behind it that was
run rather than read about:

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
