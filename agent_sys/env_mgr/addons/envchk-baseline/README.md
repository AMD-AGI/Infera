# `envchk-baseline` — one stdio MCP server, shipped as an add-on

## What an agent gets

One MCP server, `envchk_baseline`, carrying one tool:

| tool | takes | returns |
|---|---|---|
| `mcp__envchk_baseline__envchk_report` | nothing | a JSON object: `token`, `label`, `level`, `pid`, `at` |

`token` is derived from this component's own salt and the per-run nonce in
`$ENVCHK_NONCE`:

```
token = "ENVCHK-" + LABEL.upper() + "-" + sha256(f"{salt}:{label}:{nonce}")[:12]
```

The salt is the one literal in `.claude/servers/envchk_baseline_server.py`
tagged `ENVCHK_SALT:`, and it exists nowhere else in this repository. That is
the whole of what the token proves and it is worth stating exactly: **a caller
that never reached this server cannot produce the string**, because it does not
have the salt. It does not prove the caller reached the server *rather than*
reading the file — see `examples/env_checker`'s
`check_capabilities_genuine.validator/readme.md`, which states that limit in
full and closes it for this component by re-running the server itself.

## What it costs

One `python3` subprocess for the life of the session. No network, no wheel, no
`pip install`: the server speaks JSON-RPC over stdin/stdout out of the standard
library, so it starts in milliseconds and cannot fail on a package index being
unreachable.

**Installing it is one `cp`**, and a recipe does it —
`examples/env_checker/assets/main.env_recipe.yaml`, an `embed` item that locates
this directory by importing `env_mgr` and copies `servers/envchk_baseline_server.py`
into `$CLAUDE_CONFIG_DIR/servers/`. The item is `required`, because the agent's
`.mcp.json` names that exact path and a missing server is reported by Claude Code
as a server with **no tools** rather than as an error.

## What it does not do

- **It is not a general-purpose toolbelt.** One tool, one string. Anything else
  an agent wants from an MCP server is a different component.
- **It does not read or write anything outside its own process.** No filesystem
  access, no environment beyond `ENVCHK_NONCE`, no state between calls. Two
  calls in one session return the same `token` and different `pid`/`at` only if
  the harness restarted the server.
- **It does not check that `ENVCHK_NONCE` is set.** Unset reads as the empty
  string and the token is then a well-formed string derived from nothing, which
  the consumer's validator catches by recomputing it. Failing here instead would
  turn a diagnosable mismatch into an MCP server that reports no tools, which is
  the harder of the two to read.

## Who declares it, and why not this directory

**The declaration lives with the agent, not with the payload.** This server
reaches a session through the *agent's own* `.claude/.mcp.json` →
`Prepared.mcp_servers` → `ClaudeAgentOptions.mcp_servers`, which is the route
`spec.provisioning.md` §5 assigns to every stdio server: the harness spawns it,
so the harness has to be told about it in a file it reads.

There used to be a copy of that entry in `.claude/.mcp.json` **here**, reached by
an `agent_plugins: [envchk-baseline]` key. Both are gone. What is left in this
directory is the payload and nothing that configures anything —
`examples/env_checker/assets/env_probe.agent/.claude/.mcp.json` is the one
declaration.

Shipping the server ourselves is what makes a run of `examples/env_checker`
hermetic — no registry, no network, nothing to be unavailable on the day — and
it changes nothing about the route being exercised.

## The path the declaration names

```json
"type": "stdio",
"args": ["${CLAUDE_CONFIG_DIR}/servers/envchk_baseline_server.py"]
```

`"type": "stdio"` is written explicitly rather than left to a default: it is the
entry shape measured working through `ClaudeAgentOptions.mcp_servers` on
2026-09-03 (`/tmp/yihou/agentsys_envchecker_20260903/logs/PROBES.md`, probe C),
and a component is not the place to rely on a default that a probe did not
cover.

`${CLAUDE_CONFIG_DIR}` and not an absolute path, because the recipe **copies**
`servers/envchk_baseline_server.py` into the zone before the server is started,
and `env_mgr` has already pointed that variable at the zone's `config/`
(`env_mgr/material.py`). An absolute path here would work on the machine it was
written on and fail on the next one **as a server with no tools rather than as
an error**, which is the failure mode this whole example package exists to make
impossible.

**The variable is what makes the recipe and the declaration agree.** Both name
`$CLAUDE_CONFIG_DIR/servers/envchk_baseline_server.py` — the recipe as the
destination of its `cp`, the agent's `.mcp.json` as the argument of its
`python3`. Two writers of one path is the risk, and the variable is what keeps
them from drifting onto two different machines' answers; the item being
`required` is what turns a mismatch into a named failure instead of a silent
one.
