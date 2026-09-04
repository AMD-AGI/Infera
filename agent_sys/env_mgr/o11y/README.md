# The o11y panel: AgentsView

`agent_sys` ships an observability panel — [AgentsView](https://github.com/kenn-io/agentsview),
an external Go binary — that reads the session transcripts a run produces and
serves search, analytics and token-cost views over them at
**`http://127.0.0.1:18888`**. Deploying `agent_sys` starts it; it stays resident
across runs, and a later `agent-sys run` only health-checks it and prints the
URL. It binds loopback only and is not published beyond it.

Design: `../docs/design.md` §17.1.

**AgentsView's own code is never modified.** Every knob turned is one it already
publishes — `--port`, `CLAUDE_PROJECTS_DIR`, `AGENTSVIEW_DATA_DIR`, and
`disabled_agents` in a `config.toml` written into our own data directory.

## The prefix, and the variable that names each part of it

The binary and all its state live under one directory `agent_sys` owns, laid out
like `~/.local`. Nothing is installed anywhere else. The layout itself is
`env_mgr/prefix.py` — an `env_mgr` concern that o11y is merely the first
consumer of.

| variable | path | holds |
|---|---|---|
| `AGENT_SYS_HOME` | `~/.infera_agent_sys` | the root. Set it to relocate all of the below |
| `AGENT_SYS_BIN` | `<root>/bin` | the `agentsview` binary |
| `AGENT_SYS_SHARE` | `<root>/share` | static assets |
| `AGENT_SYS_STATE` | `<root>/state` | — |
| `AGENT_SYS_CLAUDE_HOME` | `<root>/state/claude` | `CLAUDE_CONFIG_DIR` for agent children; `projects/` beneath it is the panel's `CLAUDE_PROJECTS_DIR` |
| `AGENTSVIEW_DATA_DIR` | `<root>/state/agentsview` | `sessions.db`, our `config.toml`, `serve.log`, and AgentsView's own `daemon.<pid>.json` |
| `AGENT_SYS_RUN` | `<root>/run` | run-time scratch |

`state/claude` is deliberately **not** under a run root: the daemon outlives any
single run, so the directory it reads has to be a stable path.

`AGENT_SYS_BIN` is not added to a child's `PATH` by default. `PATH` there is
derived from the granted set so that it cannot name a directory the kernel will
refuse, and the prefix is under `$HOME`, which the default grants do not cover.
Nothing in a child needs to *exec* the binary — the panel is started once by the
CLI, not per task.

## Flags

| | |
|---|---|
| `--agentsview-port N` | the port. Resolution order: this flag, then `AGENTSVIEW_PORT`, then `18888`. An unusable value warns and falls back |
| `--no-agentsview` | do not start the panel. Not one external call is made |

`--dry-run` and `--clean` are exempt for the same reason: a dry run that leaves a
resident daemon behind has broken its only promise.

## Every failure is one warning and a skip

**o11y may never fail the thing it observes.** Each of these logs exactly one
warning, returns a status, and lets the deployment continue — none of them
raises, and there is a test per mode holding that line:

| case | warning says |
|---|---|
| the port is in use by something else | which port, and to pass `--agentsview-port` |
| the port is held by *our own* wedged daemon | how to stop it — a different fix, so a different message |
| the binary is not installed | where it looked, and the recipe to run |
| `serve --background` exits non-zero | the exit code and its stderr |
| the launch subprocess hangs | it timed out |
| the daemon starts but never answers | the URL and how long we waited |
| anything unforeseen inside the supervisor | the exception, and that the run continues without a panel |

A port already in use is a **warning and a skip, never a relocation.**
AgentsView's own `serve` would quietly move to the next free port; we probe the
port ourselves before launching precisely so it cannot, because a panel that
silently moved to 18889 is a panel nobody knows the address of.

## Four things a nervous reader should be told plainly

**Your own `~/.claude` is never read, written, or reconfigured.**
`CLAUDE_CONFIG_DIR` is placed in the agent child's environment dictionary and
never in `agent_sys`'s own `os.environ`, so a Claude Code you start in your own
terminal inherits nothing from us. Measured, not asserted: a real redirected turn
left `~/.claude/projects` at 801 files with an empty `diff`, and
`test_agent_environment_does_not_touch_this_process` is the standing guard.

**A pre-existing AgentsView of yours is never adopted.** If something is already
listening on the port, reuse requires *two* gates: it must answer
`GET /api/v1/agents` with a 200 and a body that parses as JSON, **and**
`$AGENTSVIEW_DATA_DIR` must hold a live `daemon.<pid>.json` naming that port.
That record is AgentsView's own, written into whichever data directory the daemon
was configured with — so a daemon you started writes it somewhere else and
structurally cannot appear in ours. It is removed on a clean `serve stop`; only
an unclean death leaves a stale one, and the recorded pid is checked for
liveness. If you were expecting reuse and got the "port in use" warning, this is
why.

**The binary is pinned and lands only in the prefix.** The recipe installs a
named release (`v0.42.0`) and verifies its published `SHA256SUMS`, rather than
tracking "latest" — two machines running different panels with no record of which
is not an observability story. Upstream's `install.sh` is *not* used: it hardcodes
`/usr/local/bin`, falling back to `~/.local/bin` and then to `sudo`, with no
override point anywhere in the script. Nothing of ours is written to either path.
The recipe is amd64-only.

**The panel cannot fail your run.** Beyond the table above, the single call site
wraps the supervisor in a `try` of its own, so even a bug we have not thought of
inside it degrades to a warning. Install failure is covered by the same rule
through the recipe's `importance: suggested`.

## What the panel calls a "project"

Claude Code writes one directory per *working directory*, named by slugifying
that path, with one JSONL per session inside. AgentsView shows each of those as a
project. `agent_sys` runs every attempt in its own zone, so the list is roughly
one entry per attempt rather than one per package.
