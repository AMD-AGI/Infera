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
| **Ledger door** | plugin `agent-sys-o11y-door`, every fleet session: MCP server `o11y`, CLI, backstop hook | the run ledger and the other agent-facing operations, every record schema-checked; reports a ledger write that bypassed the door |
| **Trigger** | mod in plugin `agent-sys-o11y`, main session only | timed kick and packup prompts, the fleet registry, one question per unregistered session, analysis requests by counting tool calls and turns |
| **Analysis** | jobs in plugin `agent-sys-o11y` (`server/o11y-run --kind extract\|log\|facts`), one-shot processes the trigger starts | ledger records extracted from histories, levelled records (info / warning / error), and the facts the display shows, reading each history incrementally |
| **Display** | mod in plugin `agent-sys-o11y-dash`, main session only | a pane (general / detail / KernelForge / log; narrow or wide), a one-line band, configuration forms |

The parts never import one another; they exchange only the files of [`protocol/`](protocol/), under the
fleet root `$AGENT_SYS_HOME/state/o11y/fleets/<slug(launch dir)>/`, pinned per session at activation
(design §4, §5).

## Who uses it

The user installs it once and starts the main session with `AGENT_SYS_O11Y=true` (or simply uses an
agent_sys skill in it). Agents in the fleet use the door's tools (`ledger_append`, `ledger_query`,
`session_register`, `fleet_status`) through the
`agent-sys-o11y-writer` skill (`agent_sys/tools/skill/agent-sys-o11y-writer/`). Nothing in `agent_sys`
imports this package.

## Install

The Python runtime is an env_mgr recipe (a venv in the prefix):

```bash
cd agent_sys && uv run env-mgr install env_mgr/recipes/o11y.yaml --path "$AGENT_SYS_HOME"
```

The remaining steps change your Claude Code configuration, so they are yours:

1. Register the plugins in your Claude config dir (`$CLAUDE_CONFIG_DIR`, else `~/.claude`):
   ```bash
   claude plugin marketplace add agent_sys/o11y
   for p in agent-sys-o11y-door agent-sys-o11y agent-sys-o11y-dash; do claude plugin install "$p@agent-sys-local"; done
   ```
2. Let the door's tools run without a prompt each time: add `mcp__plugin_agent-sys-o11y-door_o11y__*` to
   `permissions.allow` in your user settings.
3. Shortcuts (prefix `ctrl+q`): merge the `bindings` block of
   `plugins/agent-sys-o11y-dash/keybindings.snippet.json` into your `keybindings.json`.

Development: `claude --plugin-dir agent_sys/o11y/plugins/agent-sys-o11y-door --plugin-dir
agent_sys/o11y/plugins/agent-sys-o11y --plugin-dir agent_sys/o11y/plugins/agent-sys-o11y-dash` (overrides an
installed copy; the mods reload on save, once the current turn has ended).

## Use

| Environment variable | Meaning |
|---|---|
| `AGENT_SYS_O11Y` | `true` = this session is a fleet's main; `child` = a child; any other value = never activate; unset = become main on the first agent_sys skill use |
| `AGENT_SYS_O11Y_FLEET_DIR` | the fleet root, overriding the default under the prefix |
| `AGENT_SYS_O11Y_PARENT_SESSION_ID` | set for a child by `session_register` |
| `AGENT_SYS_O11Y_OTEL_DIR` | the optional collector's working directory on local disk |
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
| `.claude-plugin/marketplace.json` | directory marketplace `agent-sys-local` listing the three plugins |
| `protocol/schemas/` | the protocol: JSON Schema 2020-12, one file per exchanged document or record; `common.v1` holds shared definitions |
| `protocol/test_vectors/` | vectors shared by the Python and TypeScript tests |
| `protocol/ts/` | the mods' protocol layer: validator for a declared keyword subset, defaults, config reader, write rule, fleet root, slug, sanitize |
| `plugins/agent-sys-o11y-door/` | MCP server, CLI and backstop hook |
| `plugins/agent-sys-o11y/` | trigger mod (`mods/src/trigger/`: registry, schedule, counters, spawn, otel_health, index), analysis jobs (`server/`), collector launcher (`bin/o11y-launch`) |
| `plugins/agent-sys-o11y-dash/` | display mod (`mods/src/display/`: frame, one renderer per tab, forms, index), keybindings snippet |
| `plugins/*/protocol` | symlink to `../../protocol` |
| `package.json`, `build.mjs` | the esbuild build of both mods |
| `../env_mgr/recipes/o11y.yaml` | the venv recipe |
| `tests/mods/` | protocol-layer and mod unit tests, drift tests (prefix, slug, schema keywords and defaults), bundle freshness |
| `docs/` | specification and design |
