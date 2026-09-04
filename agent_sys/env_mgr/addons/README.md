# `agent_sys/env_mgr/addons/` — the agent plugins this repository ships

An **agent plugin** is a bundle of Claude Code capabilities — plugins, hooks,
tools, an MCP server — that this repository defines once, so that more than one
task package does not carry a private copy. An agent asks for one by name:

```yaml
  agent_plugins: [envchk-baseline]
```

The name is the directory name here. There is no registry file and nothing to
register.

## Where this sits, and what changed

**There is no "L2" any more.** This directory used to be the middle rung of a
three-level install hierarchy (L1 industry / L2 repository / L3 package), and the
levels were a vocabulary three documents each restated slightly differently. What
actually differs between the rungs is **who owns the directory**, so that is what
is named now:

| what | who owns it | how an agent asks |
|---|---|---|
| something the industry ships — serena, a marketplace plugin, an apt/pip tool | upstream | `recipes:` → an `env_mgr` recipe |
| something **this repository** ships | us | `agent_plugins:` → a directory here, **or** an `env_mgr` recipe carrying `tags: [internal]` |
| something **one task package** carries for one agent | that package | nothing — auto-detected at `<agent assets>/.claude/` |

The last two have the **same on-disk shape**, and that is the point of the split
rather than an accident of it: a bundle is promoted from a package to this
directory by moving it, and demoted by moving it back. Nothing is converted and
there is no second format to keep in step.

### Internal origin, in a recipe, as a tag

Where the thing to install is an *install* rather than a `.claude/` tree, an
`env_mgr` recipe declares that it is ours with an ordinary tag:

```yaml
items:
  - installer: uv
    importance: required
    tags: [internal]
    tool: ...
```

Nothing was added for this. Verified first-hand against the tree, not recalled:
`Item.tags` exists (`env_mgr/recipe.py`), `tags` is in `_CLI_KEYS` so it is
excluded from `Item.spec` and cannot leak into an installer's arguments,
`--tag` is already a CLI flag (`env_mgr/cli.py`), and `env_mgr/runner.py`
selects on tag intersection. So `env-mgr install --tag internal` works today
with no schema change.

**What a tag does not do:** it marks provenance and nothing else. It does not
place a `.claude/` tree, it does not change grants, and no installer reads it.
An agent plugin in this directory is still reached by `agent_plugins:`.

## The contract

```
agent_sys/env_mgr/addons/<name>/
├── README.md            what this gives an agent, and what it costs
├── recipe.yaml          OPTIONAL — anything that must be installed first
└── .claude/             REQUIRED — Claude Code's own canonical layout
    ├── settings.json        hooks, and any other user-scope setting
    ├── skills/<skill>/      SKILL.md
    ├── plugins/             a local marketplace: .claude-plugin/marketplace.json
    ├── .mcp.json            {"mcpServers": {...}} — external MCP servers
    └── tools/
        ├── *.mcp.py         a bundled stdio MCP server, auto-registered
        └── *.tooldef.py     module-level `TOOLS` -> in-process `mcp__env_mgr__<tool>`
```

Two rules, and both exist because this is read by a machine before it is read by
a person:

1. **`.claude/` is Claude Code's format, not ours.** A file is placed, not
   parsed — `env_mgr/material.py`'s own words. Anything that needs a conversion
   step is in the wrong format.
2. **`recipe.yaml` is for what must exist *before* `.claude/` means anything** —
   the binary an `.mcp.json` entry names, the language server a skill assumes.
   It is the same recipe format `env_mgr/recipes/*.yaml` uses
   (`env_mgr/recipe.py`: `target` + `items`, each item naming an `installer` and
   an `importance`).

## Paths inside one

The `.claude/` tree is copied into the zone before it is used, so **nothing in it
may name a path outside itself**. Where it has to point at one of its own files —
an `.mcp.json` naming the server it ships — it does so through
`${CLAUDE_CONFIG_DIR}`, which `env_mgr` has already redirected at the zone's
`config/` directory (`env_mgr/material.py`). A hard-coded `/home/<someone>/...`
works on exactly one machine and fails silently on the next, because an MCP
server that cannot start is reported as a server with no tools rather than as an
error.

**Copying is the default and the exceptions are enumerated**
(`env_mgr/agent_assets.py::_place_tree` and `_NOT_PLACED`). Every member of
`.claude/` is copied into the zone config directory except three, and each of
those is *read* or *relocated* rather than skipped: `settings.json` is **merged**
into the zone's own, `.mcp.json` is **read** and its `${VAR}`s expanded against
the zone environment — it does **not** land in the zone at all — and `plugins/`
is **relocated** to `marketplaces/` because `claude plugin install` writes
`<config>/plugins/` itself. So `servers/`, `hooks/`, `skills/` and `tools/` land
where a `${CLAUDE_CONFIG_DIR}`-relative path expects them, and the two files that
configure rather than ship are not there at all.

**Do not reference `${AGENT_SYS_ADDONS_ROOT}` from inside one.** That path
is outside the zone; a server registered at it installs cleanly, reports success,
and then cannot be read under confinement. Copy-into-the-zone is one rule for a
repository plugin and a package's own material alike — a package's
`tools/*.mcp.py` used to be registered at its source path and worked only because
that path happened to lie inside the staged package.

## What is here

| plugin | gives an agent | costs |
|---|---|---|
| [`envchk-baseline`](envchk-baseline/) | one **external MCP server**, declared through `.mcp.json`, whose single tool returns a nonce-derived token | one `python3` subprocess for the life of the session; no network |
| [`serena`](serena/) | the **declaration** half of serena — the `.mcp.json` entry that registers the MCP server. It does **not** install serena; that is `recipes: [serena]` | nothing on its own; useless without the install |

`envchk-baseline` exists to be this directory's worked example, and it is a real
one rather than a stub: `examples/env_checker` declares it, runs it, and its
`check_capabilities_genuine` validator re-starts the server itself and compares
the token the agent reported against the token the server actually produces.

## Adding one

1. `mkdir agent_sys/env_mgr/addons/<name>/.claude` and put the capability in it,
   in Claude Code's layout.
2. Write `README.md`: what an agent gets, what it costs to install, and what it
   does **not** do. The third is the one a reader cannot reconstruct.
3. Add a row to the table above.
4. If it needs something installed first, add `recipe.yaml` beside `.claude/`.

There is no registry file to edit and no name to register: the directory name
**is** the name `agent_plugins: [<name>]` resolves.
