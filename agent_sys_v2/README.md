English | [中文](README.zh-CN.md)

# agent_sys v2 — an agent workbench for Infera performance work

One engineer steers a group of Claude Code agents that find and remove the bottlenecks of an Infera deployment —
topology and parallelism, engine flags, CPU paths, collectives, KV transfer or kernels — with every claim backed by a
measured result, and every agent visible and steerable in a web console.

- **A task is a prompt and a workspace.** Agents decide what to investigate; skills give them reliable tools. The
  task graph grows during the work: an agent drafts subtasks, you approve them (one click for a whole plan).
- **Evidence, not procedure, is checked.** Acceptance checks that the harness runs when an agent says it is done (a
  failure goes back to the agent); results in one schema feed a ledger with the baseline, the best and the curves.
- **Observable and controllable.** Live output of every run and what it was given, cost, what it waits for; message,
  stop, take over in a terminal, raise the budget, review. A machine is held by one task at a time.

It stands beside [`../agent_sys`](../agent_sys) (v1, the task-package runtime), which it does not touch; both can
be installed in one environment (`agent-sys` is v1's command, `agent-sys-v2` this one's). The design — product, web
UI, back end and the rules they follow — is [`docs/design.zh-CN.md`](docs/design.zh-CN.md), in Chinese.

## Requirements

- Linux, Python 3.10+, `tmux`.
- A working Claude Code configuration in `~/.claude/settings.json` (credentials, gateway, model mapping). Runs use the
  `claude` CLI that comes with `claude-agent-sdk` (pinned); no other setup.

Agents do not run in your `~/.claude`: they get a Claude Code configuration directory of their own,
`$AGENT_SYS_HOME/claude`, made again from your settings on every start with only what they need to log in and reach
their models — the login and routing variables of the `env` block (`ANTHROPIC_*`, cloud provider credentials, proxies),
`modelOverrides`, `modelSettings` and a few more (the full list, and why, is in `docs/design.zh-CN.md` §5.9). Your MCP
servers, plugins, CLAUDE.md, skills and hooks are not loaded: an agent runs in `bypassPermissions`, and with your
Atlassian MCP server, say, it could write to Jira and Confluence without asking. A project that needs an MCP server or a
plugin names it in `agent.extra_args` (`--mcp-config <file>`, `--plugin-dir <dir>`). Session records stay in
`~/.claude/projects` (the agents' `projects/` links there), and a Claude OAuth login in `~/.claude/.credentials.json`
is used where it is. `claude_config: inherit` in `agent_sys.yaml` goes back to running agents in your own `~/.claude`.

## Install

From the root of the Infera checkout, in a virtual environment of your choice:

```bash
python3 -m venv ~/.venvs/agent-sys-v2 && . ~/.venvs/agent-sys-v2/bin/activate
pip install -e agent_sys_v2            # `uv pip install -e agent_sys_v2` works too; add [dev] for the tests
agent-sys-v2 --version
```

Install it from a checkout of its own (or a wheel), not from the Infera checkout a project's tasks change: a task's
acceptance checks run agent_sys_v2's own modules and record their digests when a run starts, so a `git checkout` or
`pull` in that checkout during a run fails the checks.

## Run

```bash
agent-sys-v2 serve
```

- The server goes into a tmux window (`tmux -L agent_sys`, window `server`) and listens **only on a Unix socket**,
  `/tmp/agent_sys-<uid>/agent_sys.sock`, in a directory only you can enter. It prints where it is and the next line.
- On your desktop, forward a port to the socket and open the console:

  ```bash
  ssh -N -L 8765:/tmp/agent_sys-<uid>/agent_sys.sock <you>@<this host>
  # then open http://localhost:8765
  ```

- Logs: `tmux -L agent_sys attach -t agent_sys:server`. Stop: Ctrl-C in that window, or
  `tmux -L agent_sys kill-window -t agent_sys:server`. Restarting the server does not touch running tasks: each run
  is a process of its own in the same tmux, and the server picks them up again from their files.
- `--foreground` runs it in the current terminal; `--home DIR` (or `AGENT_SYS_HOME`) picks another home, `--sock PATH`
  another socket, and `AGENT_SYS_TMUX=<name>` another tmux server (for a second server next to your real one).

Everything lives in `AGENT_SYS_HOME`, default `~/agent_sys_home`, created on first use:

```
agent_sys.yaml  rules/  knowledge/                     global settings, rules and lessons
projects/<p>/project.yaml                             goal, machines, ledger metric, skills, agent defaults
projects/<p>/rules/project.md                         the project's background, injected into every run
projects/<p>/tasks/<type>/<name>.md                   a task: its prompt (and frontmatter)
projects/<p>/workspaces/<type>/<name>/                progress.md, report.md, results/*.json, .agent_sys/runs/
projects/<p>/repos/                                   code the agents work on (not scanned)
```

A home made by an earlier version keeps its projects in `campaigns/`, which agent_sys no longer reads. Stop its
server and rename the directory once: `mv <home>/campaigns <home>/projects`.

## A first project

1. On the home page, **New project** with the template **Infera 性能优化** (`infera-perf`); or
   `agent-sys-v2 new glm52-agentx --template infera-perf --title "GLM-5.2 4P4D AgentX"`.
   The template comes with round one queued: `intake/r1` → `survey/r1` ∥ `baseline/r1` → `profile/r1` → `plan/r1`.
   Clone Infera into `projects/<p>/repos/Infera`: the agents work from the project directory and the skills refer
   to `repos/Infera/examples/<kit>`.
2. Nothing starts yet: the board shows **Template to review** with the five tasks in the order they run. Open each,
   check (and edit) its prompt, machines and checks, and look over the project's rules and settings; then **Start
   project** (or **Start** one task by hand). From then on the tasks follow their dependencies.
3. `intake/r1` starts: it looks around (weights, InferenceX, the nodes) and asks you a prefilled
   questionnaire (**Answer N questions**; a row you leave empty takes its suggestion). With your answers it writes the
   project's `rules/project.md`, the goal, machines and working point in `project.yaml`, and a one-page brief you
   approve. Nothing downstream starts before that.
4. The survey (InferenceX reference curves, no GPU) and the baseline (the v0 curve, noise, calibration) run, then
   the profile (cheap layers first, `candidates.yaml`; you approve it). Follow a task in **Run output**; write to
   the agent in the box at the bottom (even while it runs); **Stop**; **Take over in terminal**.
5. `plan/r1` ranks the candidates, splits the round's GPU budget over the machines and drafts the experiments
   (subtasks) and `close/r1`. Approve the plan, then **Join them all** in its drawer: the experiments and the close
   are queued together. Each experiment holds its machines; the close waits (`kind: children`) until the plan's
   every subtask is done or cancelled, integrates what was accepted, measures the curve, seals v1 if it beats v0,
   and drafts `intake/r2`: the next round's questionnaire, prefilled with its advice.
6. The **Ledger** tab draws every series as a Y–X curve (ours, the baseline, the references) with its value and Δ at
   the working point. A task that is done but fails its checks comes back to you under **Needs you** with their output.

A task file (`depends_on` kinds: `input` and `code` wait for the upstream to be done, `children` for the upstream and
all of its subtasks; `review` and `subtask` do not block):

```markdown
---
title: SPF scheduling at C64
depends_on: [baseline/b0]
machines: [smc-a16-19]            # held while it runs and while it waits for you
checks:                           # run when the agent marks progress.md done; a failure goes back to it
  - test -s "$AGENT_SYS_WORKSPACE/report.md"
  - '"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger verify "$AGENT_SYS_WORKSPACE/results"'
check_retries: 2
agent: {model: opus, effort: high, budget_usd: 50}
---
## Task
...
```

A result is `results/<label>.json` in the `infera.result/v1` schema (see
[`src/agent_sys_v2/skills/infera-bench/SKILL.md`](src/agent_sys_v2/skills/infera-bench/SKILL.md)); the A/B verdict comes from
`src/agent_sys_v2/skills/infera-experiment/scripts/ab_judge.py`, on results converted from AgentX output only, and
`ledger verify` recomputes it from that output.

## Look around without a model

```bash
python agent_sys_v2/tools/demo.py /tmp/$USER/agent_sys_demo
AGENT_SYS_TMUX=asys-demo agent-sys-v2 serve --home /tmp/$USER/agent_sys_demo --sock /tmp/agent_sys-$(id -u)/demo.sock
ssh -N -L 8766:/tmp/agent_sys-<uid>/demo.sock <you>@<this host>     # on your desktop; open http://localhost:8766
```

A project with three machines, tasks in most states, a ledger and a drafted plan. Its `project.yaml` sets
`agent.replay`, so every run plays a recorded session back (`tests/fixtures/events/`) instead of calling a model.
Stop it with `tmux -L asys-demo kill-server`. `tools/demo.py` only replaces a directory it made itself: it refuses a
directory that is not empty and not a demo home made before.

## Develop

```bash
pip install -e "agent_sys_v2[dev]"
cd agent_sys_v2 && python -m pytest tests -q     # 3-4 minutes, needs tmux; no test calls a model
node --test src/agent_sys_v2/web/test/*.test.js  # the web UI's pure functions and its English dictionary
ruff format --check . && ruff check .            # the repository's settings
```

| Path | What |
|---|---|
| `src/agent_sys_v2/server.py`, `cli.py` | REST + SSE (scans the files into memory, pushes changes, serves the UI); the `agent-sys-v2` command |
| `src/agent_sys_v2/model.py`, `scan.py`, `ops.py` | the files as a model (tasks, edges, state.json); scanning a project and watching it change; the write operations |
| `src/agent_sys_v2/sched.py`, `machines.py` | the scheduler: dependencies, slots and machines |
| `src/agent_sys_v2/runs.py`, `runner.py`, `launch.py`, `tmux.py` | one Claude Code session per task, one runner process per round in a tmux window; what a run is given (system.md, the first message, launch.md) |
| `src/agent_sys_v2/events.py`, `procs.py`, `replay.py` | a run's events as the UI reads them; its processes; recorded sessions played back without a model |
| `src/agent_sys_v2/checks.py`, `ledger.py`, `perfcheck.py`, `skills.py` | acceptance checks; the results ledger; the field checks of the `infera-perf` template; the skills index |
| `src/agent_sys_v2/knowledge.py`, `questions.py`, `files.py`, `i18n.py` | rules and lessons; questions.md as a form; workspace files; English |
| `src/agent_sys_v2/web/` | the UI: Preact + htm, no build step ([README](src/agent_sys_v2/web/README.md)) |
| `src/agent_sys_v2/templates/` | the protocol injected into every run (zh / en) and the project templates `blank`, `infera-perf` |
| `src/agent_sys_v2/skills/` | the Infera performance skills ([README](src/agent_sys_v2/skills/README.md)) |
| `tests/` | pytest; runs replay recorded sessions |
| `tools/demo.py` | the demo home above |
| `docs/design.zh-CN.md` | the design (in Chinese) |

## When something is off

- *the socket's directory can be entered by other users*: point `--sock` at a directory only you can enter.
- *already has a server running*: one server per `AGENT_SYS_HOME` (two schedulers would start a task twice).
- A queued task that does not start says why on its card: the project is archived, an error in its prompt's
  frontmatter or checks (it starts once fixed), upstream tasks (for a `children` dependency also the upstream's
  subtasks), a template waiting for your review (**Start project** on the board), auto start switched off, machines
  held by another task, or slots.
- Runs take their settings from `~/.claude/settings.json` (into `$AGENT_SYS_HOME/claude`, see Requirements); the server
  drops `CLAUDE*` / `ANTHROPIC*` variables from the shell it was started in, so a server started inside a Claude Code
  session still uses your settings. A run's **What this run was given** (launch.md) lists the keys it got and those it
  did not (names only).
- *`$AGENT_SYS_HOME/claude/projects` is a directory with files in it*: runs refuse to start rather than delete it;
  move the session records in it to `~/.claude/projects`, delete it, and start again.
- The v1 runtime also reads `AGENT_SYS_HOME` (as its state prefix); if you set it for v1, v2 makes its directories
  there too, side by side.
