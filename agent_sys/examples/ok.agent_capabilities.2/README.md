# `examples/ok.agent_capabilities.2` — does the environment an agent is promised actually arrive?

One leaf, one handoff kind, two validators, and a subject that is `agent_sys`
itself: **six Claude Code capabilities, installed per-agent by the two install
routes `env_mgr/docs/spec.md` §6 and §9 defines**, each one exercised and
each one reported with a token that a run which did not install it cannot
produce.

```
main                        non-leaf: readme, no entry.sh, NO agent
│                           inputs [] · outputs [env_report]
│
└── probe_env               is_end · ai: env_probe
                            out: env_report                [structured_text]
                              check_env_report_shape     seconds · strong · completeness
                              check_capabilities_genuine minutes · strong · trustworthiness
```

| # | capability | installed by | delivered by |
|---|---|---|---|
| 1 | skill | copied | `assets/env_probe.agent/.claude/skills/envchk-probe/` — auto-detected |
| 2 | hook | copied | `.claude/settings.json` → `hooks/envchk_session_start.py`, at `SessionStart` |
| 3 | plugin | copied | `.claude/plugins/` — a local marketplace, `claude plugin install` |
| 4 | an MCP server a recipe installed | **recipe** | `assets/main.env_recipe.yaml` places `env_mgr/addons/envchk-baseline/`'s server; the agent's `.claude/.mcp.json` declares it |
| 5 | bundled stdio MCP server | copied | `.claude/tools/envchk_stdio.mcp.py` — location is the declaration |
| 7 | serena | **recipe** | `recipes: [agent_sys:serena]` — the real thing, over the network; the agent's `.claude/.mcp.json` declares it |

**There are two install routes and exactly two**, and the table's second column
is which one: a **recipe** declares something and `env_mgr` installs it, or the
agent's own `.claude/` tree is **copied** into the zone's config directory.

**Section 6 is absent and the number is not reused.** It was an in-process
`ToolDef`, published as `mcp__env_mgr__envchk_echo_token`, and
`env_mgr/docs/spec.md` §9.2 deleted that route for component-supplied tools.
Renumbering serena to 6 would leave a reader to infer a capability was never
there.

### What this package proves now, and what it stopped proving

The table once had a third column's worth of meaning: three *install
levels*, where L2 was **a `.claude/` tree this repository ships, installed for an
agent by naming it** (`agent_plugins: [envchk-baseline]`). That declaration key
is deleted, so:

- **Stopped proving:** that a shipped `.claude/` tree can be installed by
  declaration. No such route exists. Nothing here measures it and nothing should.
- **Stopped proving:** that sections 4 and 5 differ by **who owns the declaring
  directory**. Both entries are now in the agent's own `.claude/.mcp.json`, so
  that distinction is gone. What still separates them is stated at each: 4 is
  declared explicitly and its payload installed by a recipe; 5 is declared by
  where its file sits and installed by the copy.
- **Stopped proving:** `env_mgr`'s *load the placed copy, not the source*
  isolation property for in-process tools. That was row 6b, the widest claim in
  the package, and `ACCEPTANCE.md` records that it has to come back if the route
  does.
- **Still proves, unchanged:** that each of the six capabilities reached the
  agent's zone, by a token the agent could not compute without doing so; that
  serena's *install* and *declaration* are two halves and neither implies the
  other; and that a marketplace registered outside the run root installs
  cleanly and then fails to load (row 3b).
- **Newly proves:** that a recipe can install something `agent_sys` ships into an
  agent's zone with **no exported path pointing outside it** — section 4's
  server is located by importing `env_mgr` from inside the recipe child.

## Run it

```sh
agent-sys run --package agent_sys/examples/ok.agent_capabilities.2 \
  --var nonce="$(python3 -c 'import secrets;print(secrets.token_hex(16))')" \
  --var uv_root=/tmp/$USER/agentsys_uv
```

Both variables are **required and have no default**; `steps/check.yaml` argues
each at the agent's `env` block. `nonce` because a constant nonce would make the
first published handoff contain every later run's answers; `uv_root` because
serena's install is `uv tool install`, whose defaults write `~/.local/share/uv`
— host state outside any zone.

**Accept by opening the handoff, not by reading the exit code.**
[`ACCEPTANCE.md`](ACCEPTANCE.md) is the criteria, written before the run: the
exact invocation, one row per capability naming the file to open and the
condition that **fails**, what a PASS does not prove, how the run id is pinned
on a shared box, the abort conditions, and an eight-item pre-flight.

## Why tokens

Every capability carries `sha256(f"{salt}:{label}:{nonce}")[:12]`, where the
salt lives in exactly one place — that capability's own artefact — and the nonce
is per-run. There is **no table of salts anywhere**, including in
`assets/lib/envchk.py`: one file listing all six would let a single read
produce all six tokens.

This exists because of `.claude/CLAUDE.md`'s first principle. A previous stage
in this repository reported fourteen tasks and ten validators PASS over a run in
which every result was zero. The general form of that failure is a producer
being believed about its own environment, and a token is the cheapest thing that
cannot be produced by belief.

**What that buys, stated exactly**: an agent cannot report six tokens if the
six capabilities were not installed into its zone. It does **not** prove the
agent obtained each token through the capability rather than by reading the
file — four of the six artefacts are readable files, and the agent and the
artefacts are in the same zone by construction, because putting them there is
the thing being measured.
`assets/check_capabilities_genuine.validator/readme.md` closes that gap for the
two capabilities that are processes — it starts both MCP servers itself — and
states the residual for the rest, per capability, in its *What it cannot catch*
section.

## Layout

```
main.yaml                                       root non-leaf
steps/check.yaml                                agent + 2 validators + handoff kind + task
assets/
  main.task/readme.md
  probe_env.task/readme.md                      the brief: use all six, report evidence
  env_probe.agent/
    README.md                                   maps the .claude/ tree and argues settings.json
    serena_probe.py                             section 7's subject — NOT a capability
    .claude/…                                   the copied tree: five of the six, plus the .mcp.json entries for 4 and 7
  main.env_recipe.yaml                          the package recipe layer: places section 4's server
  lib/zone.py                                   the four body-facing zone files
  lib/envchk.py                                 the token scheme and the capability register
  check_env_report_shape.validator/
  check_capabilities_genuine.validator/
```

`agent_sys/env_mgr/addons/envchk-baseline/` lives outside this package on
purpose: an add-on only one package can reach is not an add-on. It is installed
from here by `assets/main.env_recipe.yaml`, which locates it by importing
`env_mgr` rather than by any path this package knows.

## Deferred

[`todo.md`](todo.md) carries the known gaps, the two deliberate
deferrals, and the measurements this package rests on. [`ACCEPTANCE.md`](ACCEPTANCE.md)
is the criteria, and it says to open the handoff rather than read the code.

## Related

- `examples/ok.sglang_real_model.2/` — the AI-task template this package's shape,
  validator layout and `assets/lib/zone.py` come from.
- `examples/ok.filetree_grounded_report.4/main.yaml` — the non-leaf root.
- `agent_sys/env_mgr/addons/README.md` — the add-on contract.
- `env_mgr/docs/spec.md` §6 and §9 — normative for the two install routes.
