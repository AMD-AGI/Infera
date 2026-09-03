# Task — wire **AgentsView** into `agent_sys` as its o11y panel

Three deliverables, in order, from the repo-root `mission.md`:

1. **Stand it up alone.** Get [`kenn-io/agentsview`](https://github.com/kenn-io/agentsview)
   running on this box and serving its web UI.
2. **Make it a sub-component of `agent_sys`'s o11y.** Deploying `agent_sys`
   starts it automatically. Default port **18888**, configurable on the command
   line, and **a taken port is a warning and a skip, never a failure**.
   The panel must show **only the sessions `agent_sys` itself produced**, not
   every session on the user's machine.
3. **Prove it** end to end on `examples/demo2`.

**AgentsView's own code is never modified.** It is an external dependency, used
the way its documentation says to use it. Every knob we turn is one it already
publishes: `--port`, `CLAUDE_PROJECTS_DIR`, `AGENTSVIEW_DATA_DIR`,
`disabled_agents`.

The full design — approved 2026-09-03, read it before writing code — is
[`docs/superpowers/specs/2026-09-03-agentsview-o11y-design.md`](../docs/superpowers/specs/2026-09-03-agentsview-o11y-design.md).

## Background

`agent_sys` is the decoupled multi-agent task-flow system in this repo. It
reaches its AI backend through `claude-agent-sdk`, which spawns the `claude`
CLI, which writes one JSONL transcript per session. AgentsView is a Go binary
that reads exactly those transcripts and turns them into search, analytics and
token-cost views. The two fit together with no glue on either side — the whole
integration is *where the transcripts land* and *which directory the panel
reads*.

## Context — the four decisions already made, and why

| | |
|---|---|
| **Install form** | A native binary under a dedicated prefix `~/.infera_agent_sys`, laid out like `~/.local` (`bin/ share/ state/ run/`). `env_mgr` owns the prefix and injects `PATH` for `agent_sys`'s process tree only — no shell rc is written, no host state outside the prefix is touched. |
| **Env vars** | The root and every key subdirectory is named by a variable, joining the existing `AGENT_SYS_*` family in `env_mgr/paths.py`: `AGENT_SYS_HOME`, `AGENT_SYS_BIN`, `AGENT_SYS_SHARE`, `AGENT_SYS_STATE`, `AGENT_SYS_RUN`, `AGENT_SYS_CLAUDE_HOME`. |
| **Session scoping** | `CLAUDE_CONFIG_DIR=$AGENT_SYS_CLAUDE_HOME` is injected **into the agent child process only**, via the `assignment.environment` seam that `agent/backends/claude_sdk.py:358` already forwards. Transcripts land in the prefix; agentsview reads that one root; other providers are switched off with `disabled_agents`; the archive is a separate `AGENTSVIEW_DATA_DIR`. |
| **Lifecycle** | Started at the end of the `env_mgr` deploy path and **left resident** across runs. `agent-sys run` only health-checks it and prints the URL. |

**The one constraint that outranks the feature: the user's own Claude Code must
be untouched.** `CLAUDE_CONFIG_DIR` goes into the child's environment dict and
never into this process's `os.environ`. There is a unit test whose only job is
to hold that line.

## Key references

- `agent_sys/agent/backends/claude_sdk.py` — `_options()`, line ~358: the
  `options.setdefault("env", dict(self.assignment.environment))` seam. The
  write-side of session scoping is this one line's existing behaviour.
- `agent_sys/env_mgr/prepare.py:468` — `environment = {"PATH":
  executable_path(policy)}`. PATH is *derived from policy, never inherited*;
  the new prefix joins that derivation.
- `agent_sys/env_mgr/installers/bin.py` — "install one executable with one
  command, probe with `check_cmd`, skip when satisfied". **No new installer
  class is needed**; agentsview is one recipe item. `importance: suggested`
  makes install failure a warning through `level_for_missing`, which is the
  existing mechanism for "this is optional".
- `agent_sys/env_mgr/recipes/sglang.repo.yaml` — the recipe format to copy.
- `agent_sys/env_mgr/paths.py` — where the `AGENT_SYS_*` constants live.
- AgentsView docs: `docs/configuration.md` (env vars, `disabled_agents`,
  `[[session_sources]]`), `docs/commands.md` (`serve --port`, `--background`,
  `--no-browser`, daemon lifecycle).

## Core principles

1. **Read the artefact, not the exit code.** Acceptance criterion 3 — "the panel
   shows this run's session and no others" — is judged by reading the session
   list. A green run proves nothing about what is on the panel.
2. **o11y may never break the thing it observes.** Binary missing, install
   failed, port taken, daemon dead, health check timed out: every one of these
   is one `log.warning` and a skip. There is a test per failure mode asserting
   no exception escapes.
3. **Suspend, don't conclude.** Two things are unverified and must not be
   reasoned into: whether `install.sh` accepts an install prefix, and whether
   `claude-agent-sdk` honours `CLAUDE_CONFIG_DIR` for *credential* lookup and
   not just transcript placement. Each has a named experiment in the spec.
   Write no code that depends on a guessed answer.
4. **Every identifier is a parameter.** Port, prefix, data dir, config dir:
   `: "${VAR:=…}"`, never a literal buried in a function.
5. All temporary activity lives in **`/home/yihou/ws.agentsview_o11y/`**. The
   repo receives the deliverable and nothing else.
6. **Do not change host state** beyond `~/.infera_agent_sys` and our own
   directories. Never delete anything outside a `yihou/` directory.
7. Work in English; report to the user in Chinese.

## Other notable details

### The agent team this mission mandates

`mission.md` requires the work to run as an agent team: a leader polling
teammates **every 10 minutes** (first sighting of a problem is *recorded only*;
intervene on the second poll if it is still unresolved), plus a dedicated
reporter appending a progress section to `work.checkpoint.summary.md` **every 30
minutes**. That file's required sections are enumerated in `mission.md` — percent
complete, elapsed and projected time, reliability of the estimate, current
progress, code problems fixed and unfixed, non-code problems, undetermined
questions, and new commits with one line each.

The previous task's `CLAUDE.md` and checkpoint log are preserved as
`.claude/CLAUDE.e2e_deploy.20260903-0736.md.bak` and
`work.checkpoint.summary.e2e_deploy.20260903-0736.md.bak`.

### DCO sign-off is required on every commit

CI blocks any PR containing a commit without a `Signed-off-by:` trailer.

```bash
git commit -s -m "..."
git config user.name && git config user.email   # check before the first commit
```

Sign off **as yourself** — never a colleague's line, never a bot identity.
Cherry-picks do not inherit the trailer (`git cherry-pick -s`). To repair a
range without duplicating trailers:

```bash
git log --format='%h %s | %(trailers:key=Signed-off-by,valueonly)' origin/main..HEAD
git rebase --signoff <last-already-signed-commit>
git push --force-with-lease origin <branch>
```

`Co-Authored-By` is separate, is not a substitute, and must not be added when
contributing to third-party upstreams.

### Branch

`dev.yihou.aiopt.more.demo`. Changes are confined to `agent_sys/env_mgr/`,
`agent_sys/cli/`, `agent_sys/tests/env_mgr/`, and the design doc.
