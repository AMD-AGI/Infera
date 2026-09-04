# `env_probe.agent/` — the agent's own asset directory

Found by the folder convention `assets/<name>.<type>/`
(`spec_loader/assets.py:_folder_names`), the same one that binds
`assets/probe_env.task/readme.md` to the task called `probe_env`. Two things
happen to this directory and they are different:

| | |
|---|---|
| `.claude/` | **installed** — auto-detected, never declared, and copied into the zone's `$CLAUDE_CONFIG_DIR`. This is the one copy route `agent_sys/docs/spec.provisioning.md` §3 leaves |
| the whole directory | **copied** to `<zone>/workspace/<dirname>/`, and named by `$AGENT_SYS_AGENT_ASSETS` |

So `serena_probe.py` beside this file is *not* a capability: it is a subject,
placed in the workspace for a code-analysis server to find. Anything that is a
capability goes under `.claude/` and anything that is not, must not.

## The `.claude/` tree, file by file

```
.claude/
├── settings.json                     hooks — the only file Claude Code reads them from
├── hooks/envchk_session_start.py     capability 2, fired at SessionStart
├── skills/envchk-probe/SKILL.md      capability 1
├── plugins/                          capability 3 — a local marketplace
│   ├── .claude-plugin/marketplace.json
│   └── envchk-plugin/
│       ├── .claude-plugin/plugin.json
│       └── skills/envchk-plugin-skill/SKILL.md
├── tools/
│   └── envchk_stdio.mcp.py           capability 5 — a bundled stdio MCP server
└── .mcp.json                         the entries for capabilities 4 and 7
```

**Capability 6 was `tools/envchk_inproc.tooldef.py` and is deleted.** It measured
the in-process `ToolDef` route, which `spec.provisioning.md` §6 removed for
component-supplied tools; nothing replaces it, and the number is left unused so
that the deletion is visible rather than smoothed over by renumbering.

**Capabilities 4 and 7 are declared here and installed elsewhere**, and that
split is the point of both:

| | declared | installed |
|---|---|---|
| 4 `envchk_baseline` | the `.mcp.json` above | `assets/main.env_recipe.yaml` copies the server out of `agent_sys/env_mgr/addons/envchk-baseline/` |
| 7 `serena` | the `.mcp.json` above | `recipes: [serena]` on the agent spec → `agent_sys/env_mgr/recipes/serena.yaml` |

Run 1 (2026-09-03) is why the split is written down: serena installed cleanly and
every `mcp__serena__*` call returned `No such tool available`, because nothing
declared it. **An install is not a declaration**, in either direction.

## Why `settings.json` carries no explanation

It is JSON and JSON has no comments, so the argument for every key in it lives
here.

- **`hooks.SessionStart`** — one entry, `type: command`. `SessionStart` and not
  `PreToolUse` because it fires exactly once, before the agent has done
  anything, so the file it writes is evidence about the *harness* rather than
  about a tool call the agent chose to make.
- **`python3 "$CLAUDE_CONFIG_DIR/hooks/..."`** and not an absolute path. The
  variable is expanded by the shell Claude Code runs the command in, using the
  value `env_mgr` set (`material.py:63`), so the same file works in every zone.
  An absolute path here would be one machine's answer, and a hook whose command
  does not resolve **does not fail the session** — it is a hook that quietly
  never fires, which is the exact failure this package is built to detect.
- **`timeout: 20`** — a hook that hangs holds up session start. Twenty seconds
  is far past what writing one small file costs and far short of a wait anybody
  would sit through wondering whether the run had begun.
- **No `matcher`.** `SessionStart` has no tool to match on.

## Why `.mcp.json` is here, when it used to be somewhere else

Both entries in it were previously carried by add-ons under
`agent_sys/env_mgr/addons/{serena,envchk-baseline}/.claude/.mcp.json`, reached by
an `agent_plugins:` key on the agent spec. That key is deleted and this is the
one retained copy route, so the declarations moved here **byte for byte** —
serena's keeps `--project ${AGENT_SYS_MY_WORKSPACE}`, which is the reason it
could not have stayed shared: the directory serena indexes differs per agent and
has no environment-variable equivalent.

**What this package stopped proving by that move.** Section 4 used to
demonstrate that a `.claude/` tree *this repository ships* could be installed for
an agent by name. There is no such route now, so nothing here measures it —
correctly, because it does not exist. What section 4 measures instead is the
route that replaced it: a recipe places a payload `agent_sys` ships, and the
agent declares it. That is a weaker claim about ownership and an equally strong
one about the server working, and both halves still fail loudly — the recipe item
is `required`, and a missing declaration is run 1's `No such tool available`.
