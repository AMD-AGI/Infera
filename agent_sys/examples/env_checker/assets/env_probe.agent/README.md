# `env_probe.agent/` — the agent's own asset directory

Found by the folder convention `assets/<name>.<type>/`
(`spec_loader/assets.py:_folder_names`), the same one that binds
`assets/probe_env.task/readme.md` to the task called `probe_env`. Two things
happen to this directory and they are different:

| | |
|---|---|
| `.claude/` | **installed** — it is L3, auto-detected, never declared, and merged into the zone's `$CLAUDE_CONFIG_DIR` |
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
└── tools/
    ├── envchk_stdio.mcp.py           capability 5 — a bundled stdio MCP server
    └── envchk_inproc.tooldef.py      capability 6 — an in-process ToolDef
```

Capability 4 (an external MCP server) is **not** here: it is L2, and it comes
from `agent_sys/components/envchk-baseline/`, named by the agent spec's
`components:` key. Capability 7 (serena) is L1, named by `recipes:`. One run
therefore crosses all three levels, which is the only reason this package needs
seven capabilities rather than two.

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

## Why there is no `.mcp.json` here

There could be, and leaving it out is a decision. If this directory declared an
external MCP server too, capability 4 would be provable from L3 alone and the
run would never need `agent_sys/components/` — which would leave L2 declared and
untested. The one external server in this run comes from the component
precisely so that its absence is a failure.
