# `envchk-baseline` — one external MCP server, as an L2 component

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
unreachable. There is no `recipe.yaml` beside `.claude/` for the same reason —
there is nothing to install first.

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

## Why "external"

**"External" names the declaration route, not the vendor.** This server reaches
the session through `.claude/.mcp.json` → `Prepared.mcp_servers` →
`ClaudeAgentOptions.mcp_servers`, which is the same route
`npx -y @modelcontextprotocol/server-filesystem` would take and a different one
from `.claude/tools/*.mcp.py`, where the file's location is the declaration.
Shipping the server ourselves is what makes a run of `examples/env_checker`
hermetic — no registry, no network, nothing to be unavailable on the day — and
it changes nothing about the route being exercised.

## The path in `.mcp.json`

```json
"type": "stdio",
"args": ["${CLAUDE_CONFIG_DIR}/servers/envchk_baseline_server.py"]
```

`"type": "stdio"` is written explicitly rather than left to a default: it is the
entry shape measured working through `ClaudeAgentOptions.mcp_servers` on
2026-09-03 (`/tmp/yihou/agentsys_envchecker_20260903/logs/PROBES.md`, probe C),
and a component is not the place to rely on a default that a probe did not
cover.

`${CLAUDE_CONFIG_DIR}` and not an absolute path, because a component is **copied
into the zone** before it is used and `env_mgr` has already pointed that
variable at the zone's `config/` (`env_mgr/material.py:63`). An absolute path
here would work on the machine it was written on and fail on the next one **as a
server with no tools rather than as an error**, which is the failure mode this
whole example package exists to make impossible.
