# `o11y` — observe and nudge a fleet of interactive Claude Code sessions

One **main** Claude Code session, driven by the user from tmux, coordinates **child** sessions it starts
(a KernelForge campaign is the usual one). This package keeps the main session pushing, records what the
fleet did at the level a person needs, and shows where the optimisation stands — without any service the
user has to run.

| | |
|---|---|
| Specification | [`docs/spec.md`](docs/spec.md) — the properties and the acceptance criteria |
| Design | [`docs/design.md`](docs/design.md) — layout, protocol, trigger, analysis, display, test plan |
| Protocol | [`protocol/schemas/`](protocol/schemas/) — one JSON Schema per exchanged file |
| Tests | [`tests/`](tests/) — `tests/mods/` (the two mods), the Python suites beside it |

## What it does

| Part | Runs as | Does |
|---|---|---|
| **Trigger** | mod in plugin `agent-sys-o11y`, main session only | timed kick and packup prompts, analysis requests by counting tool calls and turns, the fleet registry, one question per unregistered session |
| **Agent door** | MCP server `o11y` in the same plugin, plus `python -m o11y_core` | the run ledger and the other agent-facing operations, every record schema-checked |
| **Backstop** | PostToolUse hook in the same plugin | reports a ledger write that bypassed the door |
| **Analysis** | `o11y_analyze`, a one-shot process the trigger starts | levelled records (info / warning / error) and the facts the display shows, reading each history incrementally |
| **Display** | mod in plugin `agent-sys-o11y-dash`, main session only | a pane (general / detail / KernelForge / log; narrow or wide), a one-line band, config forms |

The parts never import one another; they exchange only the files of [`protocol/`](protocol/), under the
fleet root `$AGENT_SYS_HOME/state/o11y/fleets/<slug(launch dir)>/` (design §4, §5).

## Who uses it

The user installs it once and starts the main session with `AGENT_SYS_O11Y=true` (or simply uses an
agent_sys skill in it). Agents in the fleet use the door's tools (`ledger_append`, `ledger_query`,
`event_report`, `session_register`, `fleet_status`, `ledger_schema`) through the
`agent-sys-o11y-writer` skill (`agent_sys/tools/skill/agent-sys-o11y-writer/`). Nothing in `agent_sys`
imports this package.

## Install

```bash
agent_sys/o11y/install.sh              # venv in $AGENT_SYS_HOME/share/o11y/venv, then prints the user steps
agent_sys/o11y/install.sh --register   # also registers both plugins in the current Claude config dir
```

`install.sh` never edits settings or keybindings files. The steps it prints are the user's:

1. register the plugins (`claude plugin marketplace add agent_sys/o11y`, then
   `claude plugin install agent-sys-o11y@agent-sys-local` and `agent-sys-o11y-dash@agent-sys-local`) —
   or run the script with `--register`;
2. add the allow rule `mcp__plugin_agent-sys-o11y_o11y__*` to `permissions.allow`;
3. merge `plugins/agent-sys-o11y-dash/keybindings.snippet.json` into `keybindings.json`.

Development: `claude --plugin-dir agent_sys/o11y/plugins/agent-sys-o11y --plugin-dir
agent_sys/o11y/plugins/agent-sys-o11y-dash` (overrides an installed copy; the mods reload on save, once
the current turn has ended).

## Use

| Environment variable | Meaning |
|---|---|
| `AGENT_SYS_O11Y` | `true` = this session is a fleet's main; `child` = a child; any other value = never activate; unset = become main on the first agent_sys skill use |
| `AGENT_SYS_O11Y_FLEET_DIR` | the fleet root, overriding the default under the prefix |
| `AGENT_SYS_O11Y_PARENT_SESSION_ID` | set for a child by `session_register` |
| `AGENT_SYS_HOME` | env_mgr's prefix (default `~/.infera_agent_sys`); must be absolute and canonical |

Children are launched with the command `session_register` returns, which carries the child variables.

Keys (after merging the snippet): `ctrl+q`, then `q` menu · `g` general · `d` detail · `k` KernelForge ·
`l` log · `m` narrow/wide · `b` band · `t` / `a` / `v` trigger / analysis / display configuration. The
same actions are slash commands: `/o11y`, `/o11y-general`, `/o11y-detail`, `/o11y-kf`, `/o11y-log`,
`/o11y-mode`, `/o11y-band`, `/o11y-cfg-trigger`, `/o11y-cfg-analysis`, `/o11y-cfg-display`.

Configuration lives in `<fleet root>/config/{trigger,analysis,display}.json` (omitted keys take the
schema defaults); the configuration forms write it.

## Build

The two mods are TypeScript bundled by esbuild into each plugin's single hooks module. **The bundles
(`plugins/*/mods/dist/*.js`) are committed**: a directory-marketplace install runs the source tree, so a
user needs no node. After changing `plugins/*/mods/src/` or `protocol/ts/`:

```bash
cd agent_sys/o11y && npm ci && npm run build && npm run check && npm test
```

`npm test` fails when a committed bundle differs from a fresh build.

## What is inside

| Path | What |
|---|---|
| `.claude-plugin/marketplace.json` | directory marketplace `agent-sys-local` listing the two plugins |
| `protocol/schemas/` | the protocol: JSON Schema 2020-12, one file per exchanged document or record |
| `protocol/test_vectors/` | vectors shared by the Python and TypeScript tests (fleet-directory slug) |
| `protocol/ts/` | the mods' protocol layer: schema-subset validator, defaults, write rules, fleet root and slug |
| `plugins/agent-sys-o11y/` | trigger mod (`mods/`), MCP server, CLI and analyzer (`server/`), backstop (`hooks/`) |
| `plugins/agent-sys-o11y-dash/` | display mod (`mods/`), keybindings snippet |
| `plugins/*/protocol` | symlink to `../../protocol` |
| `package.json`, `build.mjs` | the esbuild build of both mods |
| `install.sh` | venv in the prefix; prints, and on request runs, the plugin registration |
| `tests/mods/` | protocol-layer unit tests, drift tests (prefix, slug, schema defaults), bundle freshness |
| `docs/` | specification and design |
