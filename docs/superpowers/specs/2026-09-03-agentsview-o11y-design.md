# AgentsView as an o11y sub-component of `agent_sys` — design

| | |
|---|---|
| Date | 2026-09-03 |
| Upstream | https://github.com/kenn-io/agentsview — **used as an external dependency; its code is never modified** |
| Mission | repo-root `mission.md`: (1) bring agentsview up standalone, (2) auto-start it whenever `agent_sys` is deployed, default port 18888, CLI-configurable, warn-and-skip when the port is taken, and scope the panel to `agent_sys`'s own sessions, (3) prove the integration on `examples/demo2` |

---

## 1. What AgentsView is, and the two facts the design rests on

AgentsView is a Go binary that discovers coding-agent session transcripts on the
local filesystem, syncs them into a local SQLite archive, and serves a web UI.
For Claude Code it reads `~/.claude/projects/**/*.jsonl`, one JSONL per session.

Two measured facts from its docs drive everything below:

- **Its session roots are configurable by environment variable.**
  `CLAUDE_PROJECTS_DIR` overrides the Claude Code root, `AGENTSVIEW_DATA_DIR`
  overrides the archive location, and `disabled_agents` in `config.toml` removes
  providers from local scanning entirely.
- **`agentsview serve` takes `--port` (default 8080) and auto-discovers a free
  port when the requested one is busy.** That auto-discovery is *not* what the
  mission asks for, so the port decision is made by `agent_sys` before launch,
  never delegated to agentsview.

`agent_sys` reaches its AI backend through `claude-agent-sdk`, which spawns the
`claude` CLI. `agent/backends/claude_sdk.py` already forwards
`assignment.environment` into the child as `ClaudeAgentOptions.env` — that
existing seam is the whole write-side of the isolation story, so no new plumbing
is invented for it.

## 2. Install prefix and environment variables

`agent_sys` gets a user-level prefix of its own, laid out like `~/.local`, owned
by `env_mgr`:

```
~/.infera_agent_sys/            $AGENT_SYS_HOME
├── bin/                        $AGENT_SYS_BIN          # the agentsview binary
├── share/                      $AGENT_SYS_SHARE        # static assets
├── state/                      $AGENT_SYS_STATE
│   ├── agentsview/             $AGENTSVIEW_DATA_DIR    # sessions.db, config.toml, serve.log
│   └── claude/                 $AGENT_SYS_CLAUDE_HOME  # CLAUDE_CONFIG_DIR for agent children
│       └── projects/           →  agentsview's CLAUDE_PROJECTS_DIR
└── run/                        $AGENT_SYS_RUN          # pid / port record
```

The variable names join the `AGENT_SYS_*` family that `env_mgr/paths.py` already
defines (`AGENT_SYS_MY_ZONE`, `AGENT_SYS_TASK_PACKAGE`, …). The directory name
is `~/.infera_agent_sys` as specified.

**`$AGENT_SYS_BIN` is appended to `PATH` only inside `agent_sys`'s own process
tree.** `env_mgr/prepare.py` already derives `environment = {"PATH":
executable_path(policy)}` rather than reading the ambient one; the prefix joins
that derivation. No shell rc file is written and no host state outside the
prefix is touched.

**`state/claude` is deliberately not under a run root.** The daemon outlives any
single run, so `CLAUDE_PROJECTS_DIR` must be a stable path. Sessions from every
run accumulate there, which is exactly the requested scope: the panel shows
`agent_sys`'s sessions and not the user's.

## 3. Session isolation — four gates

1. **Write side.** `CLAUDE_CONFIG_DIR=$AGENT_SYS_CLAUDE_HOME` is added to
   `assignment.environment`, so the `claude` CLI that `agent_sys` spawns writes
   its JSONL under the prefix. The variable is placed in the child's environment
   only — `os.environ` of the `agent_sys` process is never mutated — so a Claude
   Code the user runs in their own terminal still reads and writes `~/.claude`
   and is entirely unaffected.
2. **Read side.** `CLAUDE_PROJECTS_DIR=$AGENT_SYS_CLAUDE_HOME/projects` — the
   one root agentsview scans.
3. **Every other provider off.** `disabled_agents` in the prefix's
   `config.toml` lists every provider except Claude Code, so a stray
   `~/.codex`, `~/.gemini` or Cursor database is never read.
4. **Separate archive.** `AGENTSVIEW_DATA_DIR=$AGENT_SYS_STATE/agentsview`. A
   pre-existing `~/.agentsview` belonging to the user is never opened.

**Consequence that must be handled, not discovered later:** moving
`CLAUDE_CONFIG_DIR` also moves where the child looks for credentials and
settings. `env_mgr` populates the prefix by **symlinking** the user's existing
credential and settings files read-only rather than copying them, so no second
copy of a token lands on disk.

**Open question, deliberately not answered by reasoning:** whether
`claude-agent-sdk` honours `CLAUDE_CONFIG_DIR` for *credential lookup* as well
as for transcript placement. The experiment that settles it is one real run with
the variable set, then `find $AGENT_SYS_CLAUDE_HOME/projects -name '*.jsonl'`
and a check that `~/.claude/projects` gained nothing. This is the first
experiment of Phase B and no code depends on a guessed answer.

## 4. Lifecycle, port, and failure semantics

**Trigger.** The `env_mgr` deploy path, at its end, calls a new
`env_mgr/o11y/agentsview.py`:

- `ensure_installed()` — idempotent; satisfied by the recipe item in §5.
- `ensure_running(port)` — reuse a live daemon, otherwise
  `agentsview serve --background --no-browser --port <port>`.

The daemon then **persists** across runs. `agent-sys run` only health-checks it
and emits the URL into the run's event stream.

**Port.** Default `18888`. Resolution order, highest first: CLI flag
(`--agentsview-port`) → `AGENTSVIEW_PORT` → `18888`. `--no-agentsview` disables
the component outright.

**A taken port is a warning and a skip, never a failure.** `agent_sys` probes
`127.0.0.1:<port>` itself before launching; if the bind fails it logs exactly one
warning naming the port and the reason, and deployment continues. The probe
exists specifically because agentsview's own behaviour — silently moving to
another port — contradicts the requirement.

**Every failure mode is warning-level**: binary absent, install failed, daemon
did not come up, health check timed out. o11y is a bystander; it may never fail
an `agent_sys` deploy or run. The recipe item carries
`importance: suggested`, so `installers/base.level_for_missing` already downgrades
install failure to a warning — the existing mechanism is used rather than a
parallel one.

**Binding.** `127.0.0.1` only. `--require-auth` is not enabled and the UI is not
published beyond loopback.

## 5. Where the code goes

No new installer class. `installers/bin.py` already means "install one
executable with one command, probe it with `check_cmd`, skip when satisfied",
which is the shape agentsview needs. It becomes one recipe item:

```yaml
- installer: bin
  importance: suggested
  layer: system
  name: agentsview
  check_cmd: "agentsview --version"
  install: "<one command that lands the binary in $AGENT_SYS_BIN>"
  tags: [o11y]
```

Whether the official `install.sh` accepts an install prefix is unverified. If it
does, the `install:` command is that script with the prefix variable set; if it
does not, the same field becomes "download the release tarball and unpack into
`$AGENT_SYS_BIN`". Neither needs a new installer class. Settling this is the
first experiment of Phase A.

| File | Responsibility |
|---|---|
| `env_mgr/paths.py` | prefix resolution and the `AGENT_SYS_*` name constants |
| `env_mgr/recipes/agentsview.o11y.yaml` | the `bin` item above |
| `env_mgr/o11y/agentsview.py` | `ensure_installed` / `ensure_running` / `probe_port` / `health` |
| `env_mgr/prepare.py` | `PATH` and `CLAUDE_CONFIG_DIR` into `environment`; call `ensure_running` at the end of deploy |
| `env_mgr/cli.py`, `cli/main.py` | `--agentsview-port`, `--no-agentsview` |
| `tests/env_mgr/test_agentsview_o11y.py` | the assertions below |

## 6. Testing

Unit tests, none of which start a real agentsview:

1. Port already bound → skip is returned, exactly one warning is logged, **no
   exception propagates**.
2. Install failure / daemon start failure / health-check timeout → same three
   properties.
3. `--no-agentsview` → not a single external call is made.
4. The injected `environment` carries `CLAUDE_CONFIG_DIR` pointing into the
   prefix, **and the process's own `os.environ` is unchanged** — this is the
   assertion that guards "the user's Claude Code is unaffected".
5. Port resolution order: flag beats env var beats `18888`.

## 7. End-to-end acceptance

One full `agent-sys run` over `examples/demo2`. Each criterion names a file to
open and a condition that fails:

| # | Check | How it fails |
|---|---|---|
| 1 | `curl -sf http://127.0.0.1:18888/` returns 200 | non-200 or connection refused |
| 2 | `$AGENT_SYS_CLAUDE_HOME/projects/**/*.jsonl` contains this run's session | zero matching files |
| 3 | The panel lists that session and **no others** | any session from another source appears |
| 4 | `~/.claude/projects` gained no entry during the run | `find -newermt` before/after differs |
| 5 | The run's exit code is what it would be with `--no-agentsview` | codes differ |
| 6 | With 18888 pre-bound, the run still succeeds and logs one warning | run fails, or warning missing/duplicated |

Criterion 3 is judged by reading the panel's session list, not by an exit code.

## 8. Out of scope

Semantic search, PostgreSQL/DuckDB mirrors, remote hosts, `--require-auth`,
exposing the UI beyond loopback, and any modification to agentsview itself.
