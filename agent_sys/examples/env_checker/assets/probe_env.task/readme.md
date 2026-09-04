# probe_env — use all six capabilities, then report what actually happened

Your environment was assembled by `env_mgr` through **two install routes**: a
**recipe** declares something and `env_mgr` installs it, or your own
`.claude/` tree is **copied** into your session's configuration directory. Your
job is to find out whether all six capabilities it was supposed to give you
actually arrived, by **using** each one, and to hand back a report that a
program can check.

**The sections are numbered 1-5 and 7.** Six was an in-process tool and the route
that carried it was deleted; the number is left unused rather than closed up, so
that you can tell a missing section from a renumbered one.

**Using is the whole task.** A section that says a capability is present because
you saw its file is not the deliverable; it is the thing this package exists to
catch. Every section below has one route that produces its token and the token
is the only part a validator believes.

## Where your output goes

Write into the directory named by **`$AGENT_SYS_OUTPUT_ENV_REPORT`**. That is
your output handoff's content directory; it already exists and you are granted
write on it. Do not create anything beside it — `claim/` and `manifest.yaml`
are the system's to write, and the manifest is what makes the version published.

```
README.md          with ## Purpose, ## Schema, ## Method, ## Limits
items/text.json    the report
```

**`items/text.json` and nothing else under `items/`.** The kind's `items_schema`
is closed (`steps/check.yaml`), so a second item is rejected before anyone reads
it. If you have a log worth keeping, quote the part that matters inside the
report rather than adding a file for it.

## The token scheme, once

Every capability carries a token:

```
token = "ENVCHK-" + LABEL.upper() + "-" + sha256(f"{salt}:{label}:{nonce}").hexdigest()[:12]
```

- `nonce` is **`$ENVCHK_NONCE`** in your environment. Check it is set before you
  start. If it is not, stop and report that — everything below derives from it
  and six tokens computed from an empty string are six wrong answers that
  look right.
- `salt` is a 32-hex constant that lives **only inside that one capability's own
  artefact**. There is no list of them. If you have not reached the capability,
  you do not have its salt, and that is the point.

**Never write `$ENVCHK_NONCE`'s value into the report or into any file.** The
report records `nonce_digest`, which is:

```sh
python3 -c 'import hashlib,os; print(hashlib.sha256(("nonce:"+os.environ["ENVCHK_NONCE"]).encode()).hexdigest()[:12])'
```

## What you are given

Facts, not instructions. None of them is the recipe.

| variable | what it is |
|---|---|
| `$ENVCHK_NONCE` | This run's nonce. Required; the package refuses to load without it |
| `$UV_TOOL_BIN_DIR` | Where serena's binary was installed. **You do not launch it** — the harness did, from that absolute path. It is here so that, if `mcp__serena__*` is absent from your tool list, you can say in section 7 whether the binary exists rather than only that the tools do not |
| `$AGENT_SYS_AGENT_ASSETS` | Your agent's asset directory, staged. Its `README.md` maps the `.claude/` tree, and `serena_probe.py` beside it is section 7's subject |
| `$CLAUDE_CONFIG_DIR` | Your session's config directory, inside the zone. Both install routes end here: what a recipe placed and what was copied from your own `.claude/` tree sit side by side, and nothing in the directory records which was which |
| `$AGENT_SYS_MY_LOGS` | The zone's logs directory. Section 2's hook writes here |
| `$AGENT_SYS_OUTPUT_ENV_REPORT` | Where the report goes |

**Do not hard-code a path that is not one of these.** Every one of them differs
between runs, and a report that names `/tmp/something` is a report about one
machine.

---

# The six sections

One per capability, in this order. The order is `assets/lib/envchk.py`'s
`CAPABILITIES` tuple and the validators use the same one; each row there carries
its own section number, which is why 6 can be missing.

## 1. skill — copied

`envchk-probe` is a skill in your own `.claude/skills/`. It arrived because the
directory exists at `$AGENT_SYS_AGENT_ASSETS/.claude/skills/`, **not** because
anything declared it — that is what the copy route means.

**Invoke it as a skill.** Its `SKILL.md` carries the salt and the command.
Opening the file with `Read` gets you the same string and is a different claim;
say in `how` which you did, and if the skill is not in your skill list, say
*that* — an honest `status` of a capability that did not arrive is the most
valuable line in the whole report.

## 2. hook — copied

A `SessionStart` hook, declared in `.claude/settings.json`, which is the only
file Claude Code reads hooks from. It has **already fired** if it fired at all —
before your first turn — and it wrote:

```
$AGENT_SYS_MY_LOGS/envchk-hook.json
```

Read that file and put its **entire parsed contents** in `proof.record`. The
token is in it. So is a `payload` object holding the `session_id`,
`transcript_path`, `cwd` and `hook_event_name` Claude Code handed the hook on
stdin, and that payload is the strongest single piece of evidence in this report:
a file cannot be read into existence, and the payload is only populated when the
harness invoked the hook.

If the file is absent, the hook did not fire. Do not run the script yourself and
report the result — it would produce a token and an empty payload, the validator
reports the empty payload, and you will have spent the turn to fail more
confusingly. Report `status` honestly instead and say in `how` that the file was
absent.

## 3. plugin — copied

A plugin called `envchk-plugin`, installed from a **local marketplace** at
`$CLAUDE_CONFIG_DIR/plugins`. It ships a skill, `envchk-plugin-skill`, and that
skill carries a **different salt** from section 1's — reporting one token for
both is the specific mistake this pair is here to catch.

Two things go in this section:

- the token, from the plugin's skill;
- in `proof.plugin_list`, the output of `claude plugin list`. That is the
  install record, and it is the half of this capability a file read cannot
  produce.

## 4. an MCP server a recipe installed — recipe

`envchk_baseline`. **Two halves, and they come from different places**: the
server file is shipped by `agent_sys` under
`env_mgr/addons/envchk-baseline/` and copied into
`$CLAUDE_CONFIG_DIR/servers/` by this package's own recipe layer
(`assets/main.env_recipe.yaml`); the entry that registers it is in **your own**
`.claude/.mcp.json`. If this section fails, one of those two halves failed, and
saying which in `how` is worth more than the token.

Call `mcp__envchk_baseline__envchk_report`. It takes no arguments. Put the
tool's whole result object in `proof.raw` — `token`, `label`, `installed_by`,
`pid`, `at`. The validator starts the same server itself and compares.

## 5. bundled stdio MCP server — copied

`envchk_stdio`, from `.claude/tools/envchk_stdio.mcp.py`. **Nothing declares
it** — `env_mgr` registered it because of the file's location and suffix.

That is the whole difference between this section and section 4, and it is
narrower than it used to be: both servers are now `type: stdio`, both are spawned
by the harness, and both entries reach it through your own `.claude/`. Section 4
is *declared explicitly and installed by a recipe*; this one is *declared by
where the file sits and installed by the copy*. Nothing else separates them.

Call `mcp__envchk_stdio__envchk_report`, same shape, `proof.raw` again.

## 6. — deleted, and this heading is the record of it

There is no section 6. It was an **in-process `ToolDef`** — a
`.claude/tools/*.tooldef.py` whose module-level `TOOLS` `agent_sys` imported into
its own supervisor process and published as `mcp__env_mgr__envchk_echo_token`.

That route is gone (`agent_sys/docs/spec.provisioning.md` §6): it ran
third-party code inside the process that supervises every agent, with its
memory, file descriptors and credentials, and no boundary that could fail
closed. An add-on now ships a server that runs on its own, which is what
sections 4 and 5 measure.

**Do not report a `tooldef` section.** The report's capability set is closed and
an extra key fails the shape check by name. If you find such a tool in your tool
list, that is a finding worth putting in `## Limits`, because it should not be
there.

## 7. serena — recipe

The real serena. **Two halves**, and knowing which is which is what makes an
honest report possible here: the binary is installed by an `env_mgr` recipe
(`recipes: [serena]` on the agent spec), and it is registered as an MCP server by
an entry in **your own** `.claude/.mcp.json` — the same arrangement as section 4.
Its tools are `mcp__serena__*`.

If those tools are absent from your tool list, say so plainly — and check
`$UV_TOOL_BIN_DIR` so you can report **which half is missing**: a binary that
exists with no tools is a declaration problem, and no binary is an install
problem. Run 1 of this package was the first: installed, undeclared, and every
call returned `No such tool available`.

There is no serena-shaped token to ask for, so the token is planted in a file
serena has to find:

```
$AGENT_SYS_AGENT_ASSETS/serena_probe.py     symbol: envchk_serena_token
```

Call it exactly like this — measured working against Serena 1.28.1 on this host
on 2026-09-03:

```
mcp__serena__find_symbol
  name_path="envchk_serena_token"
  relative_path="env_probe.agent/serena_probe.py"
  include_body=true
```

**`relative_path` is relative to serena's project root, which is your
workspace — not to `$AGENT_SYS_AGENT_ASSETS`.** Your asset directory is copied
*into* the workspace as a subdirectory, so the file is one level down and
`relative_path="serena_probe.py"` finds nothing. `env_probe.agent/` is this
package's asset directory name; if `ls $AGENT_SYS_MY_WORKSPACE` shows a
different one, use that and say so in `how`.

**Supply `relative_path`. Do not omit it.** Measured on this host on
2026-09-03 against a copy of this repository's tracked tree — 1458 files, 794
of them Python: with `relative_path`, the call answered in **2.3 s** cold; with
it omitted, **no response came back at all** within 6.3 s and the language
server shut down. Why is not established and this brief does not guess; what is
established is that the bounded call works and the unbounded one did not.

The salt is a **local inside that function**, so the body serena returns
contains it. Take the salt from the returned body and derive the token.

Put the tool's response in `proof.raw`, **unedited** — the array of hits, or the
JSON string of it, exactly as the tool gave it to you. The validator checks it
carries `name_path`, `kind`, `relative_path`, `body_location` with integer
`start_line`/`end_line`, and a `body` containing the salt. Do not reformat it,
do not drop `kind` because it looks redundant, and do not paste the token in
place of the response — the token is derived and is not in what serena returns.

**This is the one section that may be `"status": "unavailable"`**, and only on
one condition: `install_report` must carry a non-`ok` outcome naming serena. The
validator checks the two against each other. An `unavailable` beside a clean
install report is a FAIL, and it is a worse outcome than an honest `ok` you had
to work for or an honest failure you could point at.

---

# The report

`items/text.json`, exactly this shape:

```json
{
  "nonce_digest": "<12 hex, from the command above>",
  "capabilities": {
    "skill":        {"installed_by": "copied", "status": "ok", "token": "ENVCHK-SKILL-...", "how": "...", "proof": {...}},
    "hook":         {"installed_by": "copied", "status": "ok", "token": "ENVCHK-HOOK-...", "how": "...", "proof": {"record": {...}}},
    "plugin":       {"installed_by": "copied", "status": "ok", "token": "ENVCHK-PLUGIN-...", "how": "...", "proof": {"plugin_list": "..."}},
    "mcp_external": {"installed_by": "recipe", "status": "ok", "token": "ENVCHK-MCP_EXTERNAL-...", "how": "...", "proof": {"raw": {...}}},
    "mcp_stdio":    {"installed_by": "copied", "status": "ok", "token": "ENVCHK-MCP_STDIO-...", "how": "...", "proof": {"raw": {...}}},
    "serena":       {"installed_by": "recipe", "status": "ok", "token": "ENVCHK-SERENA-...", "how": "...", "proof": {"raw": [{"name_path": "envchk_serena_token", "kind": "Function", "relative_path": "serena_probe.py", "body_location": {"start_line": 0, "end_line": 0}, "body": "..."}]}}
  },
  "install_report": [ ... ],
  "install_report_source": "how you obtained it"
}
```

| field | rule |
|---|---|
| `installed_by` | `recipe` or `copied`, and it must be the one this brief gives for that section. It is not a guess — it says which of the two install routes you are claiming worked. **The key used to be `level` with `L1`/`L2`/`L3`; those levels no longer exist.** A report still carrying `level` fails on a missing key, which is the intended message |
| `status` | `ok` or `unavailable`. `unavailable` is permitted for `serena` only, under the condition in section 7 |
| `token` | the string, or `null` when `status` is not `ok`. Never a token you did not obtain |
| `how` | at least **80 characters** of non-whitespace: the tool or command you used, and how you know it was that route rather than a file read. This is the field a human reads when a token mismatches |
| `proof` | the required key for that section, per the table above. `raw` means *the tool's response, unedited* — do not summarise it, do not reformat the numbers |

## `install_report`

`env_mgr` records an outcome per recipe item and per file it placed —
a `level`, a `message` and `details`. The levels are **`ok`, `info`, `warn`,
`fail`** — that tuple is owned by `env_mgr.outcome.LEVELS` and it is the
whole set; there is no `refused`. Copy the levels through verbatim and do
not re-word them.
It writes them to **`$AGENT_SYS_INSTALL_REPORT`**, a JSON file whose contents
are an object:

```json
{"outcomes": [{"level": "...", "message": "...", "details": {...}}, ...]}
```

**Copy the `outcomes` array — the array, not the object around it** — into
`install_report`, verbatim and unedited, and put the path you read it from in
`install_report_source`. If `$AGENT_SYS_INSTALL_REPORT` is not set, look for
`agent_assets.install.json` under `$AGENT_SYS_MY_LOGS`.

**If you cannot find it, say so in `install_report_source` and leave
`install_report` as `[]`.** That will fail `check_env_report_shape`, and failing
there is correct: a run that cannot show what was installed cannot support
section 7's exemption either, and a report that quietly omits it would let a
missing install pass as a present one.

Do not summarise it and do not drop the entries that say `ok`. A validator
reading this array has to be able to see the whole install, including the parts
that worked — an array containing only the failures is a different document
that happens to look like this one.

## `README.md`

Four sections, all required, all checked for being non-empty and not a
placeholder:

- **`## Purpose`** — what this handoff is: a per-capability record of which
  Claude Code capabilities installed for one agent, and by which of the two
  install routes.
- **`## Schema`** — the shape of `items/text.json`. Describe the fields; a
  reader should not have to open the file to know what is in it.
- **`## Method`** — what you actually did, per capability. Which tool, which
  command, in which order, and anything that went wrong on the way.
- **`## Limits`** — what this run did **not** establish. Be specific and be
  hard on yourself: which tokens you obtained through the capability and which
  you could have obtained by reading a file; anything you retried; anything you
  are unsure of. An honest `Limits` is worth more than a confident one, and a
  short vague one is the section a reviewer will notice first.

---

# Rules

- **Do not report a token you did not obtain.** Both validators recompute; two
  capabilities are re-run from scratch by `check_capabilities_genuine`, which
  starts both MCP servers itself and speaks the protocol to them. A fabricated
  token is not a risk you are taking, it is a fault you are writing down.
- **A capability that did not arrive is a result.** `status` other than `ok`,
  `token: null`, and a `how` that says what you tried. That is a complete and
  useful answer, and it is the answer this package was built to be able to
  receive.
- **Never print or copy the value of an environment variable.** Naming one is
  fine; its value is not. `$ENVCHK_NONCE` above all — the report carries its
  digest and that is deliberate.
- **Do not modify anything under `$AGENT_SYS_AGENT_ASSETS` or
  `$CLAUDE_CONFIG_DIR`.** They are the subject of the measurement. Editing a
  salt to make a token match is the one action that would make the whole
  deliverable worthless.
- **Do not install anything.** If a capability is missing, that is the finding.

---

**A note for whoever edits this file next.** Several rules above are written as
*"doing this costs you more than not doing it"* rather than as *"do not do
this"* — section 2's is the clearest: running the hook script by hand is not
merely forbidden, it produces a token with an empty payload that the validator
reports, so the shortcut costs a turn and fails more confusingly than the honest
answer. That construction is deliberate and it should survive editing. A
prohibition leaves the incentive in place and relies on compliance; showing that
the shortcut is worse removes the incentive. **A rule an agent has a reason to
follow survives pressure that a rule it has merely been given does not** — and
an agent under time pressure is exactly the condition this package is meant to
report honestly from.

---

**This is a `readme.md` and there is no `entry.sh` beside it.** That one file's
difference is the whole of what "an agent task" versus "a program task" means in
this system.
