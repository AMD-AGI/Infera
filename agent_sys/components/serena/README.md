# `serena` — the declaration half of serena, as an L2 component

## What an agent gets

The `serena` MCP server, and with it serena's tool surface —
`mcp__serena__find_symbol`, `get_symbols_overview`,
`find_referencing_symbols`, `list_memories` and the rest. Measured 2026-09-03
on this host: **21 tools**, `Serena 1.28.1`.

## What it does not do — and this is the whole point of the component

**It does not install serena.** There is no `recipe.yaml` beside `.claude/`, and
its absence is deliberate rather than an omission. The install is L1, through an
agent spec's `recipes: [serena]` → `env_mgr/recipes/serena.yaml` → `uv tool
install`. This component is the **declaration**, and it is useless without the
install.

So an agent that wants serena declares **both**:

```yaml
  recipes: [serena]        # L1 — installs the binary
  components: [serena]     # L2 — registers it as an MCP server
```

That looks redundant and is not. It is the shape of the bug this component was
written in response to.

## The bug, because a component that exists because of one should say so

`examples/env_checker`'s first real run, 2026-09-03: serena **installed
cleanly** — the install report says `ok | ran: uv tool install
git+https://github.com/oraios/serena` — and the agent reported

> *"No `mcp__serena__*` tool exists in this session. I called
> `mcp__serena__find_symbol` … and got 'Error: No such tool available'."*

The invocation had been written down — in a trailing comment in
`env_mgr/recipes/serena.yaml`, beside the install, saying *"the entry below
belongs to whichever component wants serena"*. **A comment is not a
declaration.** The intent was documented well enough that several readers took
it for implemented.

Install and declaration are two halves and neither implies the other. This
component is the half that had no home.

## The entry, and the four things in it that are not cosmetic

```json
"command": "${UV_TOOL_BIN_DIR}/serena",
"args": ["start-mcp-server", "--context", "claude-code",
         "--project", "${AGENT_SYS_MY_WORKSPACE}", "--transport", "stdio"],
"env": {"HOME": "${TMPDIR}"}
```

**`${UV_TOOL_BIN_DIR}/serena`, absolutely, never bare `serena`.** A task body's
`PATH` is derived from the policy (`isolation/policy.py::executable_path`) and
the policy is composed at `prepare` step 2, while the install happens at 6b — so
the directory does not exist when `PATH` is computed and would not appear on it
even unconfined. The variable is expandable here because `material.deploy`
passes the agent's declared `env` block into `agent_assets.install`'s `environ`,
which is what `${VAR}` in an `.mcp.json` resolves against; the recipe that
installed the binary and the entry that launches it therefore cannot disagree.

**`--project ${AGENT_SYS_MY_WORKSPACE}`, the whole workspace, measured not
assumed.** The workspace is a `git clone --shared` of the repository, so the
question was whether a first-run scan is slow enough for an agent to conclude
serena is broken. Cold `find_symbol` against a copy of the tracked tree — 1458
files, 794 Python — answered in **2.3 s**, against 3.0 s for a one-file
directory. Tree size is not the cost: a bounded `find_symbol` opens the named
file rather than scanning. Narrowing this would buy nothing and would cost the
agent any view of the rest of its own workspace.

**`HOME` scoped to this subprocess and nowhere else.** serena writes its state
under `$HOME`; `${TMPDIR}` is the zone's own temp directory, so that state stays
inside the zone and dies with it. **Do not hoist this into the agent's `env:`
block** — `HOME` there is Claude Code's `HOME` too, and moving it moves the
session's own state and history. It looks tidier and it is a much larger change
wearing the same four characters.

**A caller must supply `UV_TOOL_DIR` / `UV_TOOL_BIN_DIR` / `UV_CACHE_DIR`.**
Unset, `uv tool install` writes `~/.local/share/uv` and `~/.local/bin` — host
state outside every zone — **and succeeds while doing it**. The L1 recipe's
first item refuses rather than letting that happen, so an agent that declares
`recipes: [serena]` without those three gets a named non-`ok` outcome in the
install report instead of a silent modification of somebody's home directory.

## Known limits

- **serena writes `.serena/` into the project directory** — 48 KB of config and
  symbol cache. Here that is the workspace clone, inside the run root, and it
  dies with the run. It is never written into a checkout.
- **Supply `relative_path` to `find_symbol`.** Measured on the same tree: with
  it, 2.3 s; with it omitted, **no response within 6.3 s** and the language
  server shut down. Why is not established and this file does not guess.
- **The tool surface is serena's, and it changes with serena.** Nothing here
  pins a version; `Serena 1.28.1` is what was measured, not what is required.
