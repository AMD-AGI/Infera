# `agent_sys/env_mgr/addons/` — what this repository ships for agents

An **add-on** is a capability `agent_sys` defines once — an MCP server, a skill,
a hook — so that more than one task package does not carry a private copy.

**There is no declaration key, and that is the design rather than a gap.**
`env_mgr/docs/spec.md` §6.6 is normative: an add-on is installed by
**declaring it in a recipe**, like everything else that is not the agent's own
`.claude/` tree. A recipe finds this directory by importing `env_mgr` —

```yaml
  - installer: embed
    importance: required
    name: envchk-baseline-server
    tags: [internal, envchk]
    run: |
      set -eu
      src="$(python3 -c 'import env_mgr, os; print(os.path.join(os.path.dirname(env_mgr.__file__), "addons"))')"
      ...
```

— which works from a git checkout and from a wheel alike, because `addons/` is
`package-data` **inside** `env_mgr` (`pyproject.toml`) and because
`agent_assets._child_env` pins `PYTHONPATH` to the package root, which
`installers/base.py::run_cmd` inherits.

## What this directory is not

**There is no `agent_plugins: [<name>]` key on the agent spec.** A key naming a
directory here and copying its whole `.claude/` tree into the zone would need a
grant, and there is none: no JSON-schema property, no
`isolation/policy.py::addon_grants`, no exported `AGENT_SYS_ADDONS_ROOT`.

**Not granting is the point.** `AGENT_SYS_ADDONS_ROOT` would be the only path
`env_mgr` exported that pointed *outside* the zone, and it needed a `READ_EXEC`
grant to be usable at all. A recipe needs neither: installs run at `prepare`
step 6b, before any confinement is applied, so a recipe reads this directory
unconfined and **copies what it needs into the zone**. Nothing the confined body
touches is outside it.

| what | how an agent asks | where it lands |
|---|---|---|
| something the industry ships — serena, a marketplace plugin, an apt/pip tool | `recipes:`, or the package / default recipe layer | wherever the installer puts it |
| something **this repository** ships — a directory here | the same: a recipe, whose item carries `tags: [internal]` to mark it ours | wherever that recipe copies it |
| something **one task package** carries for one agent | nothing — auto-detected at `<agent assets>/.claude/` and copied | the zone's `config/` |

Only the third row is a tree copy. `env_mgr/docs/spec.md` §9.1.

### `tags: [internal]` marks provenance and does nothing else

Nothing was added for it. Verified first-hand against the tree, not recalled:
`Item.tags` exists (`env_mgr/recipe.py`), `tags` is in `_CLI_KEYS` so it is
excluded from `Item.spec` and cannot leak into an installer's arguments,
`--tag` is already a CLI flag (`env_mgr/cli.py`), and `env_mgr/runner.py`
selects on tag intersection. So `env-mgr install --tag internal` works today
with no schema change.

**What the tag does not do:** it does not place a file, it does not change
grants, and no installer reads it.

## The contract

```
agent_sys/env_mgr/addons/<name>/
├── README.md            what this gives an agent, what it costs, and what it does NOT do
└── .claude/             the payload, in Claude Code's own canonical layout
    ├── servers/*.py         a server a recipe copies into the zone
    ├── skills/<skill>/      SKILL.md
    └── hooks/               …
```

**`.claude/` is Claude Code's format, not ours.** A file here is placed by a
recipe, not parsed — `env_mgr/material.py`'s own words. Anything needing a
conversion step is in the wrong format. Keeping the harness's layout is what
lets the recipe's `cp` be a `cp`.

**An `.mcp.json` does not belong here.** A stdio server is spawned by the
harness from an entry in the *agent's* `.claude/.mcp.json`
(`env_mgr/docs/spec.md` §6.7), and that entry names `--project`, `HOME` and other
values that differ per agent. Both add-ons' `.mcp.json` files were moved into
`examples/ok.agent_capabilities.2/assets/env_probe.agent/.claude/.mcp.json` for exactly that
reason. What stays here is the **payload** the entry points at.

**There is no `recipe.yaml` here either.** Nothing finds an add-on by name, so
there is nothing for a recipe beside `.claude/` to hang off; an add-on's
prerequisites are declared in whichever recipe installs it.

## Paths inside one

A payload is copied into the zone before it is used, so **nothing in it may name
a path outside itself**. Where a declaration has to point at one of these files
it does so through `${CLAUDE_CONFIG_DIR}`, which `env_mgr` has already redirected
at the zone's `config/` (`env_mgr/material.py`). A hard-coded `/home/<someone>/…`
works on exactly one machine and fails silently on the next, because an MCP
server that cannot start is reported as **a server with no tools** rather than as
an error.

## What is here

| add-on | gives an agent | costs |
|---|---|---|
| [`envchk-baseline`](envchk-baseline/) | one **stdio MCP server** whose single tool returns a nonce-derived token | one `python3` subprocess for the life of the session; no network |

`envchk-baseline` exists to be this directory's worked example, and it is a real
one rather than a stub: `examples/ok.agent_capabilities.2` installs it from
`assets/main.env_recipe.yaml`, declares it in its agent's `.mcp.json`, runs it,
and its `check_capabilities_genuine` validator re-starts the server itself and
compares the token the agent reported against the token the server produces.

`serena/` was here too and is deleted: it held only the `.mcp.json` that
registers the server, and that moved to the agent that wants it. The **install**
is `env_mgr/recipes/serena.yaml`, which is a recipe and not an add-on.

## Adding one

1. `mkdir agent_sys/env_mgr/addons/<name>/.claude` and put the payload in it, in
   Claude Code's layout.
2. Write `README.md`: what an agent gets, what it costs to install, and what it
   does **not** do. The third is the one a reader cannot reconstruct.
3. Add a row to the table above.
4. Write the recipe item that installs it, and **check `pyproject.toml`'s
   `package-data`** — a leading dot is not matched by `*`, so a new dot-file at
   the leaf of `.claude/` ships only if a glob names it. Count the wheel's
   members; do not read the build's exit code.
