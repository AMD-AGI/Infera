# main — does the environment an agent is promised actually arrive?

This package's subject is `agent_sys` itself. Its subgraph is a single entry,
`probe_env`, which is also its `is_end` — so `env_report`, that entry's output,
is the one kind that leaves this graph.

```
main                        non-leaf: readme, no entry.sh, NO agent
│                           inputs [] · outputs [env_report]
│
└── probe_env               is_end · ai: env_probe
                            L1 recipes:    [serena]
                            L2 agent_plugins: [envchk-baseline]
                            L3 auto:       assets/env_probe.agent/.claude/
                            out: env_report                [structured_text]
                              check_env_report_shape     program · seconds · strong
                              check_capabilities_genuine program · minutes · strong
```

Seven capabilities, three levels, one run:

| # | capability | level | delivered by |
|---|---|---|---|
| 1 | skill | L3 | `.claude/skills/envchk-probe/` — auto-detected, **not** declared |
| 2 | hook | L3 | `.claude/settings.json` → `hooks/envchk_session_start.py`, fired at `SessionStart` |
| 3 | plugin | L3 | `.claude/plugins/` — a local marketplace, installed with `claude plugin install` |
| 4 | external MCP server | **L2** | `agent_sys/agent_plugins/envchk-baseline/.claude/.mcp.json` |
| 5 | bundled stdio MCP server | L3 | `.claude/tools/envchk_stdio.mcp.py` — the file's location is the declaration |
| 6 | in-process `ToolDef` | L3 | `.claude/tools/envchk_inproc.tooldef.py` → `mcp__env_mgr__envchk_echo_token` |
| 7 | serena | **L1** | `recipes: [serena]` — the real thing, over the network |

`main` is a **non-leaf**: its work *is* its subgraph, so it carries a readme and
no `entry.sh`, and it names no agent — `closure.schema.json` requires one of a
leaf and of nothing else.

The one thing `main` owns is the **grant**: `handoffs/env_report`, write.
Permissions are inherited downwards, so the root is the one place that has to
know the whole vocabulary, and `write` covers `read` for whatever consumer
arrives later.

## Why one leaf

There is nothing to sequence and nothing to parallelise, and a second step would
be a second way for the run to fail for a reason that has nothing to do with the
question being asked. The question is single: *did the seven arrive?* A graph
that could fail before reaching it would make a red run ambiguous, and an
ambiguous red run is the thing this package's own validators are written to
avoid.

## How to run it, and how to accept it

```sh
agent-sys run --package agent_sys/examples/env_checker \
  --var nonce="$(python3 -c 'import secrets;print(secrets.token_hex(16))')" \
  --var uv_root=/tmp/$USER/agentsys_uv
```

The nonce is **required and has no default** — `steps/check.yaml` argues why at
the `env` block, and the short form is that a constant nonce would make the
first published handoff contain the answers to every run after it.

**Accept by opening the handoff, not by reading the exit code.** Seven sections,
seven tokens, both validators PASS. This repository has already had a stage
report fourteen tasks and ten validators green over a run in which every result
was zero (`.claude/CLAUDE.md`, principle 1), and that is the failure mode the
whole token scheme is a response to.
