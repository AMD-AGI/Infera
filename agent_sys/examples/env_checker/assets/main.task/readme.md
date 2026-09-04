# main — does the environment an agent is promised actually arrive?

This package's subject is `agent_sys` itself. Its subgraph is a single entry,
`probe_env`, which is also its `is_end` — so `env_report`, that entry's output,
is the one kind that leaves this graph.

```
main                        non-leaf: readme, no entry.sh, NO agent
│                           inputs [] · outputs [env_report]
│
└── probe_env               is_end · ai: env_probe
                            recipe  default → assets/main.env_recipe.yaml
                                            → recipes: [serena]
                            copied  assets/env_probe.agent/.claude/  (auto-detected)
                            out: env_report                [structured_text]
                              check_env_report_shape     program · seconds · strong
                              check_capabilities_genuine program · minutes · strong
```

Six capabilities, two install routes, one run. `agent_sys/docs/
spec.provisioning.md` owns the routes and there are exactly two: a **recipe**
declares something and `env_mgr` installs it, or the agent's own `.claude/` tree
is **copied** into the zone config.

| # | capability | installed by | delivered by |
|---|---|---|---|
| 1 | skill | copied | `.claude/skills/envchk-probe/` — auto-detected, **not** declared |
| 2 | hook | copied | `.claude/settings.json` → `hooks/envchk_session_start.py`, fired at `SessionStart` |
| 3 | plugin | copied | `.claude/plugins/` — a local marketplace, installed with `claude plugin install` |
| 4 | an MCP server a recipe installed | **recipe** | `assets/main.env_recipe.yaml` copies `env_mgr/addons/envchk-baseline/`'s server in; the agent's `.claude/.mcp.json` declares it |
| 5 | bundled stdio MCP server | copied | `.claude/tools/envchk_stdio.mcp.py` — the file's location is the declaration |
| 7 | serena | **recipe** | `recipes: [serena]` — the real thing, over the network; the agent's `.claude/.mcp.json` declares it |

**Section 6 is absent and the number is not reused.** It was an in-process
`ToolDef`, published as `mcp__env_mgr__envchk_echo_token`, and
`spec.provisioning.md` §6 deleted that route for component-supplied tools.
Renumbering serena to 6 would leave a reader to infer that a capability was
never there.

**What this package stopped proving.** Until 2026-09-04 section 4 measured a
distinct third thing: that a `.claude/` tree **this repository ships** could be
installed for an agent by naming it (`agent_plugins: [envchk-baseline]`). That
declaration key is deleted, so no such route exists to measure. Sections 4 and 5
are consequently no longer separated by *who owns the declaring directory* —
both entries are now the agent's own. What still separates them is stated at
each: 4 is declared explicitly and its payload installed by a recipe; 5 is
declared by where its file sits and installed by the copy.

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
question being asked. The question is single: *did the six arrive?* A graph
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

**Accept by opening the handoff, not by reading the exit code.** Six sections,
six tokens, both validators PASS. This repository has already had a stage
report fourteen tasks and ten validators green over a run in which every result
was zero (`.claude/CLAUDE.md`, principle 1), and that is the failure mode the
whole token scheme is a response to.
