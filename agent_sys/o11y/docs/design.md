# o11y — Design

| | |
|---|---|
| Status | Normative for how the fleet observability package is built |
| Version | 2 |
| Updated | 2026-11-23 |
| Summary | The two plugins, the one-shot analyzer, the agent door, the file protocol under the prefix, and the libraries adopted, with the criterion→test map. |
| Implements | [`spec.md`](spec.md) |

---

## 1. Scope

This document builds [`spec.md`](spec.md): the trigger (T), analysis (A), display (D), the agent door
(ledger writer offered as MCP tools and a CLI) and the protocol (P) that is their only contract.

### 1.1 What this package owns
- The fleet registry, the run ledger, the level records and the display facts of each fleet.
- Two Claude Code plugins, the Python core / analyzer they run, the optional collector launcher, and the
  `agent-sys-o11y-writer` skill.

### 1.2 What it does not
`monitor` (task-run exceptions), `env_mgr/o11y` (AgentsView), `env_mgr` prefix layout, and every source
format (Claude Code, KernelForge, machine pool). Spec §11.

## 2. Layout and import graph

```
agent_sys/o11y/
  README.md  docs/spec.md  docs/design.md  .gitignore   # ignores .venv, node_modules, .claude-plugin/types
  .claude-plugin/marketplace.json     # directory marketplace "agent-sys-local": the two plugins
  protocol/
    schemas/*.v1.schema.json          # P — JSON Schema 2020-12, the only shared contract
    ts/                               # the mods' protocol layer: schema-subset validator, defaults,
                                      #   write rules, fleet-root and slug resolution (bundled into each mod)
  plugins/
    agent-sys-o11y/                   # T + agent door
      .claude-plugin/plugin.json      # hooks module → mods/dist/trigger.js; mcpServers.o11y (alwaysLoad)
      hooks/hooks.json                # modules + PostToolUse backstop
      hooks/ledger_backstop.py
      server/o11y_core/               # validate, append (flock), query, config, registries, CLI, MCP server
      server/o11y_analyze/            # one-shot analyzer (spawned by T)
      server/py.sh                    # runs the prefix venv's python
      bin/o11y-launch                 # fleet launcher (optional OTel copy; child launch helper)
      mods/src/trigger/  mods/dist/trigger.js
      protocol -> ../../protocol
    agent-sys-o11y-dash/              # D only
      .claude-plugin/plugin.json      # hooks module → mods/dist/display.js
      mods/src/display/  mods/dist/display.js
      keybindings.snippet.json
      protocol -> ../../protocol
  package.json  build.mjs             # one esbuild build, two entry points
  install.sh                          # venv in the prefix; prints and (on request) runs the registration
agent_sys/tools/skill/agent-sys-o11y-writer/SKILL.md   # canonical skill; installed like its siblings
```

- A plugin carries exactly one hooks module, and an event registered without a matcher may be registered
  once per plugin; hence one plugin per mod.
- The `protocol` symlinks resolve: an installed directory-marketplace plugin runs from the source tree
  (`$.plugin.root`, `CLAUDE_PLUGIN_ROOT`), and the install cache's copy dereferences the link.
- The Python venv lives at `$AGENT_SYS_HOME/share/o11y/venv` (installed artefacts, prefix `share`), never in
  a plugin directory (the install cache copies plugin directories per version).

### 2.1 The import graph

```
                 protocol/schemas (JSON)
            ▲            ▲              ▲
 protocol/ts│     loaded │              │protocol/ts
   ┌────────┴──┐  ┌──────┴─────┐  ┌─────┴─────┐
   │ trigger   │  │ o11y_core  │  │ display   │
   │ (TS mod)  │  │ (python)   │  │ (TS mod)  │
   └───────────┘  └──▲──────▲──┘  └───────────┘
                     │      │
           o11y_analyze   MCP server · CLI · backstop · o11y-launch
```
- trigger, display and o11y_core never import one another; o11y_analyze imports only o11y_core's public
  api. Nothing here imports agent_sys's Python packages, and no agent_sys package imports this one, so
  `docs/interfaces.md` §7 is unaffected.
- Duplicated facts with drift tests (engineer_principle §5.1): the prefix rule (`env_mgr/prefix.py`), the
  slug rule (TS and Python), and the schema defaults the TS layer applies.

## 3. Architecture

```
 tmux ───────────────────────────────────────────────────────────────────────────────────────
 ┌ MAIN claude (AGENT_SYS_O11Y=true) ────────────────────────────┐   ┌ CHILD claude ───────────┐
 │ trigger mod  ─ tick 30 s: registry · kick/packup · slow timer   │   │ AGENT_SYS_O11Y=child    │
 │   │ counters: tool.call (no agentId), turn.complete            │   │ mods dormant            │
 │   │ $.prompt.submit("[o11y automated kick] …") → conversation  │   │ MCP door: ledger_*,     │
 │   ├─► P: registry/ triggers/ queue/ state/                     │   │   event_report …        │
 │   └─ $.clock.after → detached spawn ──► o11y_analyze (one-shot)│   │ backstop hook           │
 │ display mod ◄── P: facts/ logs/ registry/ config/              │   └───────────┬─────────────┘
 │ MCP server o11y (stdio child) · backstop hook (PostToolUse)    │               │
 └───────────────────────┬───────────────────────────────────────┘               │
                         ▼                                                        ▼
  $AGENT_SYS_HOME/state/o11y/fleets/<slug(launch dir)>/  ◄── o11y_core: sole JSONL writer
    config/ registry/ ledger/runs.jsonl logs/events.jsonl triggers/ queue/ facts/ state/
                         ▲
  o11y_analyze reads: $CLAUDE_CONFIG_DIR/sessions/*.json · histories (+subagents) · machine pool ·
                      KernelForge forge_experiments/ · e2e-flow files · [OTel JSONL]
  [optional] o11y-launch → claude --settings otel.json ─OTLP─► otelcol (docker) ─► user's endpoint(s)
                                                                          └─► $AGENT_SYS_O11Y_OTEL_DIR
```

## 4. Names, locations, environment

| Variable | Meaning | Default |
|---|---|---|
| `AGENT_SYS_O11Y` | `true` = main; `child` = child; any other set value = never activate; unset = activate on first agent_sys skill use | unset |
| `AGENT_SYS_O11Y_FLEET_DIR` | the fleet's data root | `$AGENT_SYS_HOME/state/o11y/fleets/<slug(launch dir)>` |
| `AGENT_SYS_O11Y_PARENT_SESSION_ID` | the main session that launched this child | — |
| `AGENT_SYS_O11Y_OTEL_DIR` | local-disk directory for the collector (the container runtime cannot mount NFS homes) | `/var/tmp/$USER/infera_agent_sys/o11y/otel` |
| `AGENT_SYS_HOME` | env_mgr's prefix | `$HOME/.infera_agent_sys` |

- **Launch directory** = the session's `$.session.root()` / `CLAUDE_PROJECT_DIR`, which is the directory
  claude was started in (not the git root). Every component resolves the fleet root from it the same way.
- **Slug** = Claude Code's project-directory rule: every non-alphanumeric character → `-`. One TS and one
  Python implementation, drift-tested.
- **`AGENT_SYS_HOME` must be absolute and canonical** (the mods cannot resolve symlinks); the drift test
  covers a symlinked value against `env_mgr.prefix.Prefix.resolve`.
- **Claude configuration directory** = `$CLAUDE_CONFIG_DIR`, else `~/.claude`, per session; recorded in the
  registry and passed to the analyzer.
- A child receives `AGENT_SYS_O11Y=child`, `AGENT_SYS_O11Y_FLEET_DIR`, `AGENT_SYS_O11Y_PARENT_SESSION_ID`
  from `session_register` (and the `o11y-launch` command line when OTel is on).

## 5. Protocol (P)

| Path under the fleet root | Schema | Writer | Readers | Write |
|---|---|---|---|---|
| `config/{trigger,analysis,display}.json` | `config.*.v1` | display forms, the user | T / A / D | whole file |
| `registry/sessions.json` | `registry.sessions.v1` | T | A, D, door | whole file |
| `registry/launches.jsonl` | `registry.launch.v1` | o11y_core (`session_register`) | T, A | append |
| `ledger/runs.jsonl` | `runs.v1` (13 kinds) | o11y_core | A, D, door | append |
| `logs/events.jsonl` | `event.v1` | o11y_core | D, A | append |
| `triggers/<seq>.json` | `trigger_fire.v1` | T | A, D | one file per record |
| `queue/<ts>-<kind>.json` | `queue.v1` | T | A (removes drained items) | one file per item |
| `facts/{general,detail,kernelforge,usage}.json` | `facts.*.v1` | A | D | whole file |
| `state/active/<sid>.json` | private to T | T | T | whole file |
| `state/trigger.json` | private to T (schedule, pending kick) | T | T | whole file |
| `state/analyze/` | private to A (cursors, dedup) | A | A | — |
| `state/backstop/` | private to the backstop (cursor per ledger) | backstop | backstop | — |

- Whole files: tmp + fsync + rename (Python) / tmp + `mv -f` (mods, which have no rename).
- JSONL only via `o11y_core.append`: flock on the target file, one `write()` per line ≤ 64 KiB, fsync; a torn
  tail left by a dead writer is closed before the next line.
- `v`, `id`, `ts`, `writer{session_id, role, source, from[]}` are set by o11y_core; `role` ∈ main | child |
  analyzer | other.
- Files 0600, directories 0700.
- Readers skip invalid lines (`o11y.protocol.invalid_line`, once per file per pass); whole documents that
  fail validation are treated as absent (`o11y.protocol.invalid_file`).
- Configs default omitted keys and reject unknown ones. A new major version is a new schema file; readers
  accept both during the transition.
- **Event catalogue** (§8.4) is owned by this document.

## 6. The agent door — `o11y_core`

- **Library**: `schemas` (jsonschema; explicit RFC 3339 pattern; one-line problems for required / anyOf /
  enum / if-then / unknown fields with a did-you-mean, the last only once the record is otherwise valid),
  `store` (append, tolerant reader, atomic write), `api` (whole operations: `ledger_append`,
  `ledger_query` collapsing superseded records via the public `identity()`, `event_report`,
  `session_register`, `fleet_status`, config load), `claude_sessions` (tolerant reader of
  `<config dir>/sessions/<pid>.json`; procStart mismatch = dead).
- **Writer identity**: session id from the parent process's (`ppid`) registry entry — an MCP server keeps a
  stale `CLAUDE_CODE_SESSION_ID` after `/clear`, and `CLAUDE_PID` is not set for MCP servers. Role: the
  fleet registry if the session is listed, else `AGENT_SYS_O11Y`, else `other`.
- **Fleet root**: `AGENT_SYS_O11Y_FLEET_DIR`, else from `CLAUDE_PROJECT_DIR` (cwd can change).
- **MCP server** (`mcp>=2,<3`, low-level `Server`): `ledger_append {kind, record}` with a compact per-kind
  description, `ledger_schema(kind)` (read-only), `ledger_query`, `event_report`, `session_register`,
  `fleet_status`. No top-level combinators, refs inlined, every input validated server-side; results
  `{ok, id}` or `{ok:false, problems[]}`. Measured cost: +1,146 input tokens per request for all six.
- **Selective listing** (spec §10.4): `tools/list` returns the six tools only when the session is a fleet
  member — `AGENT_SYS_O11Y` ∈ {true, child} at start, or `state/active/<sid>.json` exists for the parent
  session; otherwise an empty list. The server checks membership on each request and once per tick-period
  (30 s); when membership appears it sends `notifications/tools/list_changed`. (Client handling of
  `list_changed` is to be verified, test 23.)
- **CLI** `python -m o11y_core …` (`--fleet-dir`) over the same api.
- **Backstop** PostToolUse `Write|Edit|Bash` with no `if` (indirect writers have no path on the command
  line): stat → unchanged exits 0; changed → validate new bytes, or the whole file after an in-place edit →
  invalid / in-place edit / direct Write-Edit → exit 2 with the problems and a pointer to `ledger_append`.
  A background command's write is caught at the next tool call.

## 7. Trigger — the `agent-sys-o11y` mod

- **Activation**: `AGENT_SYS_O11Y=true` at `session.start`; or unset + first `tool.call` with
  `tool=='Skill'`, no `agentId`, skill matching `^(infera-agent-sys|agent-sys-.+)$`, or `command.run` with
  such a command. Any other value vetoes. Persisted in `state/active/<sid>.json`; re-armed on reload; on
  `/clear` rewritten for the new session id with the old one in lineage.
- **Never registered** in either plugin: `skill.prompt`, `classic.*`, `prompt.*` (the host's guard skips
  them and injects a notice into the conversation).
- **Tick** (`$.clock.every`, 30 s): registry normalisation → `registry/sessions.json` when changed (last good
  value kept for torn records; `active`, `fleet_dir`, config dir filled); kick/packup due check against
  `state/trigger.json`; slow log timer.
- **Kick**: `$.prompt.submit("[o11y automated kick] " | "[o11y automated packup] " + render(template))`;
  interpolated values sanitised; the submitted text is stored in the `trigger_fire` record. Pending
  (`submitted_at`) is persisted in `state/trigger.json` and cleared when a turn starts or the submit
  resolves; after a reload a pending entry younger than one interval still blocks. Refused (Esc) → retried
  after the next `turn.complete` with `isAborted == false`.
- **Counters**: `tool.call` (no `agentId`) and `turn.complete`; every `refresh.every_n` / `log.every_n` →
  `queue/` item. Spawning is scheduled with `$.clock.after` (work started inside an event handler is aborted
  by Esc) as `sh -c '"$@" >>"$1" 2>&1 </dev/null' sh <log> setsid -f <pluginRoot>/server/py.sh -m
  o11y_analyze --kind <k>` with `AGENT_SYS_O11Y_FLEET_DIR` and `CLAUDE_CONFIG_DIR` in its env; every value
  is an argument, nothing is interpolated into the script, and config cannot name an executable.
- **Unregistered sessions** (spec §3.5): on each tick, a live session in the same tmux server, launched under
  the fleet directory after the main started, and absent from `registry/launches.jsonl`, triggers one
  automated prompt `"[o11y automated question] Session <name> (tmux <target>, cwd <dir>) is not registered.
  Is it yours, and what is it for? Register it with session_register, or declare it not part of the fleet."`;
  asked once per session id (recorded in `state/trigger.json`); same one-pending rule as the kick.
- **Exit**: on `session.end`, a detached `o11y-launch --sweep` stops the collector if no fleet session is
  alive; the same sweep runs at every launch and analyzer start (covers a SIGKILLed last session).

## 8. Analysis — `o11y_analyze`

### 8.1 One invocation
Lock per kind (non-blocking; a second spawn exits) → claim queue items → run → re-scan the queue and loop
while items of its kind exist, within the budget (default 300 s) → release → exit. It exits early if the
main session it serves is gone. Drained items are removed only when their resolved path lies inside the
fleet root's `queue/`.

### 8.2 Ingest and cursors
- One cursor per history file for each consumer (ingest, log model pass, display model pass): (inode,
  offset of last full line, sha of last line).
- Shrink / mismatch / new inode → ingest re-scans from 0 (detectors are idempotent) and writes
  `o11y.protocol.history_reset`; the model cursors jump to the end.
- First sight of a history: the model reads only the last `lookback_bytes` (config `log.lookback_bytes`,
  `display.lookback_bytes`); token totals take everything as the baseline.
- Usage deduplicated by `message.id`; totals include cache reads; subagent files included.
- Cursors advance only after outputs are durably written; a failed model call leaves its cursor.

### 8.3 Detectors (code)
| Event | Level | Rule |
|---|---|---|
| `o11y.session.started/ended/cleared` | INFO | registry diff (cleared = same pid, new sid) |
| `o11y.session.crashed` | ERROR | registry file present, pid dead or procStart mismatch |
| `o11y.session.permission_wait` | WARN | `waiting_for` ∈ {permission prompt, sandbox request, worker request} for > 10 min |
| `o11y.session.idle` | WARN | idle beyond `detectors.idle_warn_s` |
| `o11y.tokens.crossed` | INFO | new tokens (input + cache creation + output), per session and fleet, once per pass that crossed ≥ 1 step |
| `o11y.compaction.done` | INFO | `compact_boundary` in history (or OTel compaction) |
| `o11y.api.error` | ERROR | first occurrence of a fatal kind (rate_limit quota, billing_error, authentication_failed, …) |
| `o11y.api.error_repeated` | ERROR | ≥ K consecutive transient API errors |
| `o11y.machine.requested/granted/released` | INFO | pool outbox `submitted` / `granted` / `released` (`ended` only when no `released` follows) |
| `o11y.machine.lost/bad_node/idle_released` | WARN | pool outbox, fleet requests only |
| `o11y.campaign.stale` | WARN | live status running, `events.jsonl` unchanged > `detectors.campaign_stale_s` |
| `o11y.source.schema_mismatch` | ERROR | required field absent / wrong type; repeated only after the file changes |
| `o11y.protocol.invalid_line/invalid_file/history_reset` | WARN/WARN/INFO | §5, §8.2 |

### 8.4 Event catalogue
Detector events above; model-pass events `o11y.run.*`, `o11y.task.conclusion`, `o11y.task.service`,
`o11y.incident.reported`; agent-reported `o11y.agent.*` via `event_report`; analyzer self-reports
`o11y.analyzer.llm_failed`. Every event carries `attributes.o11y.dedup_key`.

### 8.5 Model passes
`claude -p --bare --json-schema <schema> --model <cfg>` with thinking off, all tools disallowed, a neutral
cwd, `AGENT_SYS_O11Y=off`; `is_error` is checked. Log pass: new slices of main and child histories + the
ledger tail → events with evidence and missing ledger records (`writer.source=extracted`, deduplicated
against the ledger), written through o11y_core. Display pass: only when the ledger lacks the task, current
run or plan.

### 8.6 Facts
From the ledger (task, best, runs, plan, machines), usage totals, the machine pool, KernelForge
(`run_state.json`, `events.jsonl`, `llm_usage.json`, `dashboard.json` per the AKLR branch of Hyperloom;
roots from ledger `campaign` records), e2e-flow files (`sources.e2e_flow_globs`), and the task book
(`sources.task_book`). Wall time starts at the task record, else the fleet's earliest history. Validated
against `facts.*.v1` before an atomic write.

## 9. Display — the `agent-sys-o11y-dash` mod

- Resolves the fleet root exactly as T does; dormant on its own env check for children; active when
  `registry/sessions.json` lists this session as an alive, active main (it never reads T's code or store).
- Polls fleet-root mtimes each second (configurable); writes only `config/*.json` via forms.
- Pane: narrow / wide (reopened at another size; a pane opened unasked needs ≥ 144 columns, 110 after the
  user opened it once); tabs general / detail / kernelforge / log; band line; age and stale marks per tab;
  an invalid facts file shows its problem and recovers on the next poll.
- Config forms are generated from the config schemas (range and type checked before saving).
- Commands `/o11y` (menu) and one command per sub-key; `keybindings.snippet.json` binds `ctrl+q` + `q g d k
  l m b t a v` (Chat context; the user merges it). Chords do not fire while a dialog is open.

## 10. Optional OTel copy — `o11y-launch`

- Reads the user's effective OTel settings. If they set per-signal endpoints or headers, mirrors them into
  `--settings` and the collector config; if it cannot mirror them, starts plain `claude` (spec §9.3).
- Idempotent `docker start || docker run -d --restart unless-stopped --network host --user $(id -u)` of the
  official `otel/opentelemetry-collector-contrib` with an otlp receiver → otlphttp exporter(s) to the
  user's endpoint(s) (sending queue on, no processors) and a file exporter (size rotation, `*.jsonl`) to
  `$AGENT_SYS_O11Y_OTEL_DIR`; health check; then `exec claude --settings o11y-otel.json` whose only key is
  the local endpoint. Unhealthy → plain `claude`.
- T's tick checks collector health while active and records `o11y.otel.collector_down` (ERROR) per outage.
- A reads `$AGENT_SYS_O11Y_OTEL_DIR/*.jsonl` (the rotated set, by name) when present.

## 11. Build versus adopt

| Need | Adopted | Why |
|---|---|---|
| Schema language and Python validation | JSON Schema 2020-12 + `jsonschema` (already an agent_sys dependency) | language-neutral; agent_sys's own enforcement point |
| Mod-side validation | a JSON-Schema-subset validator in `protocol/ts`, reading the bundled schemas | one source of truth; zod would restate every schema in TS |
| Bundling | esbuild (40k★) | mods may import only bundled files |
| Agent door | MCP python-sdk (24.9k★), stdio, plugin-bundled | standard tool calls; lifetime follows the session |
| Model calls | `claude -p --bare --json-schema` | validated structured output through the user's gateway; no SDK dependency |
| Record shape | OpenTelemetry Log Data Model names | standard; matches Claude Code's own events |
| Collector | official OpenTelemetry Collector contrib | official; content-identical forwarding measured |
| Display host | Claude Code Mods | in-session, no extra process |
| Written here | the two mods, o11y_core, o11y_analyze, o11y-launch, schemas, skill | no adoptable component does the fleet logic or the domain views |

## 12. Test plan

| Criterion | Test |
|---|---|
| 1, 2, 3 | isolated interactive session (tmux, `--plugin-dir`, 1-min kick, packup every 2): submitted text == recorded text, marker per kind; queued kick → skip record, also after a hot reload; packup cadence |
| 4, 5 | child with `AGENT_SYS_O11Y=child` invoking an agent_sys skill stays dormant; unset env + model-invoked and user-typed skill both activate; `/clear` keeps main |
| 6, 8 | model requests before/after 100 ticks and counters; analyzer on unchanged history makes no `claude -p` call |
| 7 | 200 scripted tool calls incl. failures, parallel batches, Esc; requests == thresholds; all served; model windows contiguous and disjoint |
| 9 | fixture histories with repeated usage per `message.id` and subagent files; one milestone per crossing pass |
| 10, 11, 12 | fixture registry with a waiting status aged 11 min; dead child pid file; dead main + new main start; machine-pool outbox lines |
| 13, 14, 15 | MCP live call (default model) with a missing and an unknown field → refusal → fixed; `echo >>` to the ledger → backstop message at next tool call; 4 processes × 150 appends on the prefix path |
| 16 | KernelForge fixture missing `best` → one error naming file and field, none on the next pass, one after the file changes |
| 17, 18 | facts with old `generated_at` → stale mark; appearance within one tick; chords idle and busy |
| 19 | `ps` after `/exit` and after SIGKILL of the main; analyzer exits within budget |
| 20 | checksums of the config dir and project `.claude/` before and after a session (installation excluded) |
| 21 | forwarded vs direct decode comparison against a local stand-in; collector down → direct export |
| 22 | kill the MCP server, the analyzer and the mod worker mid-run: the session continues |
| 23 | first-request input tokens in a fleet session, a non-fleet session, and a session before and after skill activation (`list_changed` delivered without restart) |
| 24 | a hand-started claude in the fleet's tmux server and directory → one question; `session_register` → no repeat |
| drift | prefix (incl. symlinked `AGENT_SYS_HOME`) vs `env_mgr.prefix.Prefix.resolve`; slug TS vs Python; TS defaults vs schema defaults |

## 13. Deviations from the spec

None.

## 14. New open questions

1. A live fleet run with the real default model and a multi-hour soak at the real 30-minute cadence have
   not been done.
