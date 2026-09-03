# `agent_sys/components/` — the components this repository ships

A **component** is a bundle of Claude Code capabilities that more than one task
package would otherwise carry a private copy of. Naming one from an agent spec
is **L2** of the three-level install hierarchy:

| level | what | declared how |
|---|---|---|
| **L1** | industry components — serena, a marketplace plugin, an apt/pip tool | `recipes: [<name>]` on the agent spec → an `env_mgr` recipe |
| **L2** | components this repository ships, normalised | `components: [<name>]` on the agent spec → a directory here |
| **L3** | components a task package carries for one agent | **not declared** — auto-detected at `<agent asset dir>/.claude/` |

**L2 and L3 have the same on-disk shape**, and that is the point of the split
rather than an accident of it: a component is promoted from a package to this
registry by moving a directory, and demoted by moving it back. Nothing is
converted, nothing is re-declared, and there is no second format to keep in
step with the first.

## The contract

```
agent_sys/components/<name>/
├── README.md            what this component gives an agent, and what it costs
├── recipe.yaml          OPTIONAL — anything that must be installed first (L1 machinery)
└── .claude/             REQUIRED — Claude Code's own canonical layout
    ├── settings.json        hooks, and any other user-scope setting
    ├── skills/<skill>/      SKILL.md
    ├── plugins/             a local marketplace: .claude-plugin/marketplace.json
    ├── .mcp.json            {"mcpServers": {...}} — external MCP servers
    └── tools/
        ├── *.mcp.py         a bundled stdio MCP server, auto-registered
        └── *.tooldef.py     module-level `TOOLS` -> in-process `mcp__env_mgr__<tool>`
```

Two rules, and both exist because a component is read by a machine before it is
read by a person:

1. **`.claude/` is Claude Code's format, not ours.** A file is placed, not
   parsed — `env_mgr/material.py`'s own words. A component that needs a
   conversion step is a component in the wrong format.
2. **`recipe.yaml` is for what must exist *before* `.claude/` means anything** —
   the binary an `.mcp.json` entry names, the language server a skill assumes.
   It is the same recipe format `env_mgr/recipes/*.yaml` uses
   (`env_mgr/recipe.py`: `target` + `items`, each item naming an `installer`, an
   `importance` and a `layer`).

## Paths inside a component

A component is copied into the zone before it is used, so **nothing in it may
name a path outside itself**. Where a component has to point at one of its own
files — an `.mcp.json` naming the server it ships — it does so through
`${CLAUDE_CONFIG_DIR}`, which `env_mgr` has already redirected at the zone's
`config/` directory (`env_mgr/material.py:63`). A component that hard-codes
`/home/<someone>/...` works on exactly one machine and fails silently on the
next, because an MCP server that cannot start is reported as a server with no
tools rather than as an error.

## What is here

| component | gives an agent | costs |
|---|---|---|
| [`envchk-baseline`](envchk-baseline/) | one **external MCP server**, declared through `.mcp.json`, whose single tool returns a nonce-derived token | one `python3` subprocess for the life of the session; no network |

`envchk-baseline` exists to be L2's worked example, and it is a real one rather
than a stub: `examples/env_checker` declares it, runs it, and its
`check_capabilities_genuine` validator re-starts the server itself and compares
the token the agent reported against the token the server actually produces.

## Adding one

1. `mkdir agent_sys/components/<name>/.claude` and put the capability in it, in
   Claude Code's layout.
2. Write `README.md`: what an agent gets, what it costs to install, and what the
   component does **not** do. The third is the one a reader cannot reconstruct.
3. Add a row to the table above.
4. If it needs something installed first, add `recipe.yaml` beside `.claude/`.

There is no registry file to edit and no name to register: the directory name
**is** the name `components: [<name>]` resolves.
