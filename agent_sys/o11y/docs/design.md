# o11y — Design

| | |
|---|---|
| Status | Normative for how the fleet observability package is built |
| Version | 3 |
| Updated | 2026-11-23 |
| Summary | The three plugins, the analysis jobs, the ledger door, the file protocol under the prefix, the libraries adopted, and the criterion→test map. |

Implements [`spec.md`](spec.md).

---

## 1. Scope

Builds the five subsystems of spec §1 — trigger (T), analysis (A), display (D), ledger door, protocol (P) —
for the scope of spec §1.1; what it does not own is spec §11.

## 2. Layout and import graph

```
agent_sys/o11y/
  README.md  docs/{spec,design}.md  .gitignore      # ignores .venv, node_modules, .claude-plugin/types
  .claude-plugin/marketplace.json     # directory marketplace "agent-sys-local": three plugins
  package.json  build.mjs             # one esbuild build, two entry points (trigger, display)
  protocol/
    schemas/*.v1.schema.json          # the contract (JSON Schema 2020-12); common.v1 holds shared defs
    events.json                       # event catalogue: names and levels
    test_vectors/                     # slug vectors; per-schema valid/invalid documents (both validators)
    py/                               # o11y_protocol: validate, append (flock), atomic write, fleet root, slug
    ts/                               # mod-side: schema-subset validator, defaults, atomic write, fleet root, slug, sanitize
  plugins/
    agent-sys-o11y-door/              # LEDGER DOOR — every fleet session (main and children)
      .claude-plugin/plugin.json      # mcpServers.o11y; hooks/hooks.json (PostToolUse backstop)
      server/o11y_door/               # api (ledger_append/query, session_register, fleet_status), mcp_server, cli
      hooks/ledger_backstop.py
      bin/py                          # runs the prefix venv's python
      protocol -> ../../protocol
    agent-sys-o11y/                   # TRIGGER + ANALYSIS — active only in a main session
      .claude-plugin/plugin.json      # hooks module → mods/dist/trigger.js
      mods/src/trigger/{registry,schedule,counters,spawn,otel_health,index}.ts
      server/o11y-run                 # the launch contract (spec §8.6)
      server/o11y_job/                # one-shot job mechanics: lock, request loop, budget, llm call
      server/o11y_sources/            # read-only adapters + pure functions (read_new, usage, attribution)
      server/o11y_extract/            # extract job: histories → ledger records (source=extracted)
      server/o11y_log/                # log job: detectors + log model pass → events
      server/o11y_facts/              # facts job: model-free projection → facts/*.json
      bin/o11y-launch                 # collector owner: launch, health, --sweep
      otel/collector.template.json
      protocol -> ../../protocol
    agent-sys-o11y-dash/              # DISPLAY — main session only
      .claude-plugin/plugin.json      # hooks module → mods/dist/display.js
      mods/src/display/{frame,general,detail,kernelforge,log,forms,index}.ts
      keybindings.snippet.json
      protocol -> ../../protocol
agent_sys/env_mgr/recipes/o11y.yaml   # venv (share/o11y/venv) + checks; modelled on agentsview.o11y.yaml
agent_sys/tools/skill/agent-sys-o11y-writer/SKILL.md
```

- A plugin carries exactly one hooks module, and an event without a matcher may be registered once per
  plugin; hence one plugin per mod. The door is its own plugin so children keep it even if the trigger
  plugin is disabled.
- The analysis packages live in the trigger plugin because only a main session launches them; they share no
  code with the trigger mod.
- `protocol` symlinks resolve: an installed directory-marketplace plugin runs from the source tree, and the
  install cache's copy dereferences the link.
- The venv is `$AGENT_SYS_HOME/share/o11y/venv`, created by the env_mgr recipe; never in a plugin directory.

### 2.1 The import graph

```
                 protocol/schemas · events.json · test_vectors
          ┌───────────────┬──────────────┴─────────────┬───────────────┐
      protocol/ts     protocol/py ◄─────────────┬──────┴───────┐   protocol/ts
          │               │                     │              │       │
   trigger mod     o11y_door (api)        o11y_job      o11y_sources  display mod
   (registry,      mcp · cli · backstop      ▲   ▲   ▲         ▲      (frame + one
    schedule,                                │   │   │         │       renderer
    counters,                       o11y_extract o11y_log o11y_facts   per tab)
    spawn, otel_health)                 (each imports o11y_job, o11y_sources, o11y_door.api;
                                         never one another)
   trigger ──launch contract (o11y-run --kind k)──► analysis
```
- trigger, display, door and the three analysis jobs never import one another. The jobs reach the ledger
  and events only through `o11y_door.api` (the single writer). The one call across subsystems is the launch
  contract.
- `o11y_job` exists because neither `o11y_sources` (read-only) nor the door (records only) can host the job
  mechanics every analysis kind repeats (engineer_principle §2).
- Duplicated facts with drift tests (engineer_principle §5.1): the prefix rule (vs `env_mgr/prefix.py`,
  incl. a symlinked `AGENT_SYS_HOME`), the slug rule (TS vs Python, `test_vectors/slug.json`), the schema
  defaults and validator verdicts (TS vs Python, `test_vectors/<schema>.json`). The prefix rule is copied
  rather than imported because this package sits outside agent_sys's import graph.

## 3. Architecture

```
 tmux ─────────────────────────────────────────────────────────────────────────────────────────
 ┌ MAIN claude (AGENT_SYS_O11Y=true) ───────────────────────────────┐  ┌ CHILD claude ────────────┐
 │ trigger mod ─ registry tick 30 s · schedule · counters · spawn    │  │ AGENT_SYS_O11Y=child      │
 │   │ $.prompt.submit("[o11y automated kick] …")                    │  │ door plugin only:         │
 │   ├─► registry/sessions.json · triggers/ · requests/analysis.json │  │   MCP ledger_* · register │
 │   │   otel/health.json                                            │  │   backstop hook           │
 │   └─ $.clock.after → o11y-run --kind extract|facts|log (detached) │  └─────────────┬────────────┘
 │ display mod ◄── facts/ · logs/events.jsonl · triggers/ · registry/ · config/     │
 │ door: MCP o11y (stdio) · backstop (PostToolUse)                   │                │
 └──────────────────────┬───────────────────────────────────────────┘                │
                        ▼                                                             ▼
  $AGENT_SYS_HOME/state/o11y/fleets/<slug(launch dir)>/   ◄── o11y_door.api: sole JSONL writer
    config/ registry/ ledger/runs.jsonl logs/events.jsonl triggers/ requests/ facts/ otel/ state/
                        ▲
  o11y_sources reads: <config dir>/sessions/*.json · histories (+subagents) · machine pool ·
                      KernelForge forge_experiments/ · e2e-flow files · task book · [OTel JSONL]
  [optional] o11y-launch → claude --settings ─OTLP+bearer─► otelcol (docker) ─► user's endpoint(s)
                                                                       └─► $AGENT_SYS_O11Y_OTEL_DIR
```

## 4. Names, locations, environment

| Variable | Meaning | Default |
|---|---|---|
| `AGENT_SYS_O11Y` | `true` = main; `child` = child; any other set value = never activate; unset = activate on first agent_sys skill use | unset |
| `AGENT_SYS_O11Y_FLEET_DIR` | the fleet's data root | `$AGENT_SYS_HOME/state/o11y/fleets/<slug(launch dir)>` |
| `AGENT_SYS_O11Y_PARENT_SESSION_ID` | the main session that launched this child | — |
| `AGENT_SYS_O11Y_OTEL_DIR` | collector working directory on local disk (the container runtime cannot mount NFS homes) | `/var/tmp/$USER.infera_agent_sys/o11y/otel` (a top-level directory of our own under the sticky `/var/tmp`, so every component can be checked) |
| `AGENT_SYS_HOME` | env_mgr's prefix | `$HOME/.infera_agent_sys` |

- **Root pin.** At activation T writes `$AGENT_SYS_HOME/state/o11y/sessions/<sid>.json` with the fleet root
  derived from the launch directory (`$.session.root()` at that moment), and carries it over on `/clear`.
  T, D, the door (via ppid → sid) and the backstop read the pin first, then `AGENT_SYS_O11Y_FLEET_DIR`, then
  derive. `$.session.root()` alone is not used later because it follows directory changes.
- **Slug**: every non-alphanumeric character → `-`; longer than 200 characters → the first 200 + `-` +
  sha256[:12]. Used only for the fleet directory name.
- **History location**: from the session record / hook input, else `<config dir>/projects/*/<sid>.jsonl`,
  cached per session id — never derived from a slug.
- **Claude configuration directory**: `$CLAUDE_CONFIG_DIR`, else `~/.claude`, per session; recorded in the
  registry and passed through the launch contract.
- `AGENT_SYS_HOME` must be absolute and canonical (mods cannot resolve symlinks).

## 5. Protocol (P)

| Path under the fleet root | Schema | Writer | Readers | Write |
|---|---|---|---|---|
| `config/{trigger,analysis,display}.json` | `config.*.v1` | display forms (the user may hand-edit) | T / A / D | whole file |
| `registry/sessions.json` | `registry.sessions.v1` | T | A, D, door | whole file |
| `registry/launches.jsonl` | `registry.launch.v1` | door (`session_register`) | T | append |
| `ledger/runs.jsonl` | `runs.v1` (12 kinds) | door | A, door (`ledger_query`) | append |
| `logs/events.jsonl` | `event.v1` | door (`api.event_report`, called by A and the door) | D, A | append |
| `triggers/<seq>.json` | `trigger_fire.v1` (kick, packup, question) | T | D | one file per record |
| `requests/analysis.json` | `requests.v1` (per-kind `seq`, `ts`, `reason`) | T | A | whole file |
| `otel/health.json` | `otel_health.v1` (outage edges with a counter) | T | A | whole file |
| `facts/{general,detail,kernelforge}.json` | `facts.*.v1` (general holds `spend`) | o11y_facts | D | whole file |
| `state/trigger.json`, `state/active/` | private to T | T | T | whole file |
| `state/{extract,log,facts}/` | private to each job (served seq, cursors, dedup) | that job | that job | — |
| `state/backstop/` | private to the backstop | backstop | backstop | — |

- Whole files: tmp + fsync + rename (Python) / tmp + `chmod 600` + `mv -f` (mods). JSONL: `o11y_protocol.append`
  only (flock on the target file, one `write()` per line ≤ 64 KiB, fsync, a torn tail closed first).
- `v`, `id`, `ts`, `writer{session_id, role, source, from[]}` set by the door; role ∈ main | child | analyzer | other.
- Files 0600, directories 0700. Shared definitions in `common.v1` (time, id, writer, facts header), resolved
  with `referencing.Registry` (Python) and a small cross-file resolver (TS); a schema handed to
  `claude --json-schema` is bundled self-contained by a drift-tested generator.
- Readers skip invalid lines and report them on stderr; invalid configuration is handled per spec §8.4
  (last valid kept, one report with did-you-mean).

## 6. Ledger door — plugin `agent-sys-o11y-door`

- **api** (whole operations): `ledger_append` (validate with one-line problems for required / anyOf / enum /
  if-then / unknown fields with did-you-mean, the last only once the record is otherwise valid; server sets
  `v`, `id`, `ts`, `writer`), `ledger_query` (supersede by `identity()`), `session_register` (`decision:
  child | not_in_fleet`, `purpose`, join keys `name` and/or tmux target; returns the child's launch env and
  argv), `fleet_status` (read-only), `event_report` (internal, used by A).
- **Writer identity**: session id from `<config dir>/sessions/<ppid>.json` (`session_of_pid`, one read, no
  liveness judgement); role from the registry if listed, else `AGENT_SYS_O11Y`, else `other`. Every launch
  wrapper `exec`s claude so ppid is claude.
- **MCP server** (`mcp>=2,<3`, low-level `Server`, no `alwaysLoad`): four tools — `ledger_append {kind,
  record}` (compact per-kind description), `ledger_query`, `session_register`, `fleet_status`. A refused
  append returns the problems plus that kind's full schema. Every input validated server-side.
- **Selective listing**: tools listed only for fleet members — `AGENT_SYS_O11Y ∈ {true, child}` at start, or
  `registry/sessions.json` lists the parent session as an active member. Claude Code 2.1.325 speaks the
  2026-07-28 MCP wire, where an unsolicited `notifications/tools/list_changed` is ignored unless the server
  serves `subscriptions/listen`; the server serves it (SDK `ListenHandler`) and keeps the legacy
  notification for older clients. Checked each request and every 30 s.
- **CLI** `bin/py -m o11y_door …` (`--fleet-dir`) over the same api.
- **Backstop** PostToolUse `Write|Edit|Bash` (no `if`: an indirect writer has no path on the command line):
  a stdlib-only stat precheck exits at once when the ledger is unchanged or the session is not a member;
  otherwise validate new bytes (or the whole file after an in-place edit) → exit 2 with the problems and a
  pointer to `ledger_append`. A valid appended record is accepted.

## 7. Trigger — plugin `agent-sys-o11y` (mod)

Modules: `registry` (normaliser, the only judge of alive / crashed / status / lineage), `schedule` (kick,
packup, question), `counters`, `spawn`, `otel_health`, `index` (event wiring). They pass plain values.
- **Activation**: env `true` at `session.start`; or unset + first `tool.call` (`tool=='Skill'`, no `agentId`,
  skill `^(infera-agent-sys|agent-sys-.+)$`) or `command.run` with such a command; any other value vetoes.
  On activation: write the root pin and `state/active/<sid>.json`, and rewrite `registry/sessions.json` at once.
  Re-armed on reload; carried to the new sid on `/clear`.
- **Never registered** in any plugin: `skill.prompt`, `classic.*`, `prompt.*`.
- **Registry tick** (`registry_interval_s`, default 30): Claude session records + `launches.jsonl` → sessions
  with role, parent, status, status_since, alive (pid + procStart), lineage, last good value for torn records;
  a dead former main stays (active=false) for `dead_retention_s`. Claude Code deletes a dead process's
  session record when another claude starts in the same configuration directory, so a former main whose
  record vanished is kept from the last snapshot (crashed if it was alive there); on a clean `session.end`
  T marks its own entry ended (not crashed).
- **Schedule**: absolute due times in `state/trigger.json`; `$.prompt.submit(marker + render(template))`;
  interpolated values pass `sanitize` (charset `[A-Za-z0-9._:/@%+-]`, others → `_`, cap 200, quoted
  `name="…"`; fails closed). One pending (persisted with `submitted_at`; cleared when a turn starts or the
  submit resolves; after a reload a persisted pending entry blocks until the next `turn.start`, at most 3
  intervals — a reload is deferred while a turn or dialog is in progress); refused → one retry after a `turn.complete` with `isAborted == false`. Questions:
  candidates = live sessions in the same tmux server, launched under the fleet directory after the main
  started, absent from `launches.jsonl`; asked once per sid (mark in `state/trigger.json`).
- **Counters**: `tool.call` (no `agentId`) and `turn.complete` → bump `requests/analysis.json` seq for
  `extract`, `facts`, `log` at their every-N; slow timer bumps `log`.
- **Spawn** (`$.clock.after`, because work started inside an event handler is aborted by Esc):
  `sh -c 'log=$1; shift; setsid -f "$@" >>"$log" 2>&1 </dev/null' sh <log> <pluginRoot>/server/o11y-run
  --kind <k>` with the fleet dir and config dir in env; every value is an argument; config cannot name an
  executable.
- **otel_health** (only when the session was launched through `o11y-launch`): sample the collector each
  tick, write outage edges to `otel/health.json`, run `o11y-launch --sweep` each tick and, detached, on
  `session.end` other than `clear`/`resume`.

## 8. Analysis — `o11y-run`, `o11y_job`, `o11y_sources`, `o11y_extract`, `o11y_log`, `o11y_facts`

### 8.1 Launch contract and job mechanics
`o11y-run --kind extract|log|facts` maps the kind to its job module. `o11y_job.run_consumer(kind, handler,
budget)`: non-blocking fcntl lock per kind (busy → exit 0) → target = `requests/analysis.json[kind].seq` →
handler → record `served` in `state/<kind>/` → loop while `seq > served`, within `budget_s` → exit 0; any
failure → non-zero exit and a log line. Exits early when the main session is gone (the crash is the log
job's to report). `o11y_job.llm`: `claude -p --bare --json-schema <bundled schema> --model <cfg>
--no-session-persistence --tools ""` with thinking off, a stripped environment (gateway variables, PATH,
HOME, the user's normal telemetry), a neutral cwd and `AGENT_SYS_O11Y=off`; `is_error` checked.

### 8.2 Sources (read-only)
Adapters with required-field checks returning values and problems: registry file, histories
(`read_new(cursor) → lines, cursor'`, cursor = inode, offset of last full line, sha of last line; shrink /
mismatch / new inode → reset flag), machine pool (fleet attribution: requester user = current user, and the
full notify socket path from the registry), KernelForge (`run_state.json`, `events.jsonl`, `llm_usage.json`,
`dashboard.json`, AKLR branch of Hyperloom; roots only from ledger `campaign` records that are absolute,
exist, contain `forge_experiments/run_state.json` and are at depth ≥ 3; bounded-depth discovery), e2e-flow
files (`sources.e2e_flow_globs`), task book, OTel JSONL (owner/mode/symlink-checked directory; rotated files
by inode). Pure function `usage(messages)` → new and cache-read totals, deduplicated by `message.id`.

### 8.3 Jobs
| Job | Reads | Writes | Model |
|---|---|---|---|
| `o11y_extract` (frequent) | new slices of main and child histories (lookback on first sight; reset → jump to end) + ledger tail | ledger records, source=extracted, deduplicated against the ledger (task, run_start with machines, run_end, plan, campaign, kernel, hotops) | cheap model |
| `o11y_log` | registry snapshots, history slices, pool, KernelForge, `otel/health.json`, `facts/general.json` `spend`, ledger | events (§8.4) | log model pass for task-level meaning only |
| `o11y_facts` | ledger, registry, sources, task book | `facts/{general,detail,kernelforge}.json` incl. usage totals (`spend`: new, cache_read, per session) | none |
Detectors are code in `o11y_log`; session events are diffs of registry snapshots; tokens.crossed diffs
`spend.new` against the last reported step; a kept `run_end` never overrides a recorded `best`.

### 8.4 Event catalogue (`protocol/events.json`)
| Event | Level | Rule |
|---|---|---|
| `o11y.session.crashed` | ERROR | registry: alive=false without a clean end (children; a former main on the next main's start) |
| `o11y.session.permission_wait` | WARN | `waiting_for` ∈ {permission prompt, sandbox request, worker request} > 10 min |
| `o11y.session.idle` | WARN | idle beyond `detectors.idle_warn_s` |
| `o11y.tokens.crossed` | INFO | per session and fleet, new tokens, once per pass that crossed ≥ 1 step |
| `o11y.compaction.done` | INFO | `compact_boundary` (or OTel compaction) |
| `o11y.api.error` | ERROR | first billing_error / authentication_failed / account_on_hold / verification_required |
| `o11y.api.error_repeated` | ERROR | ≥ K consecutive transient errors (rate_limit included) |
| `o11y.machine.requested/granted/released` | INFO | pool `submitted` / `granted` / `released` (`ended` only without a later `released`) |
| `o11y.machine.lost/bad_node/idle_released` | WARN | pool, fleet requests only |
| `o11y.campaign.stale` | WARN | live status running, `events.jsonl` unchanged > `detectors.campaign_stale_s` |
| `o11y.source.schema_mismatch` | ERROR | per file signature |
| `o11y.otel.collector_down` | ERROR | outage edge in `otel/health.json` |
| `o11y.files.removed_externally` | ERROR | log pass: a path the session used fails not-found and the fleet's own calls contain no rm/mv of it |
| `o11y.task.conclusion`, `o11y.task.service`, `o11y.run.*` | INFO | log pass |
| `o11y.agent.incident` / `o11y.agent.finding` | per severity | code: ledger `incident` / `finding` records written by sessions, deduplicated by record id |
| `o11y.analyzer.llm_failed` | WARN | `o11y_job.llm` failure |
Every event carries `attributes.o11y.dedup_key`.

## 9. Display — plugin `agent-sys-o11y-dash` (mod)

- Resolves the fleet root like T (pin first); dormant on its own env veto; active when the registry lists this
  session as an alive, active main.
- Reads only `facts/*`, `logs/events.jsonl`, `triggers/*`, `registry/sessions.json`, `config/*`; polls mtimes
  (`poll_ms`, default 1000); writes only `config/*.json` via forms.
- Frame + one renderer per tab (general / detail / kernelforge / log); narrow / wide (reopened at another
  size; a pane opened unasked needs ≥ 144 columns, 110 after the user opened it once); band; age and stale
  marks; an invalid facts file shows its problem and recovers on the next poll.
- Forms generated from the config schemas. Commands `/o11y` (menu) and one per sub-key;
  `keybindings.snippet.json` binds `ctrl+q` + `q g d k l m b t a v` (Chat context; chords do not fire while a
  dialog is open).

## 10. Optional OTel copy — `o11y-launch`

- Reads the effective OTel settings (managed > user > shell); mirrors per-signal endpoints, headers,
  protocol and compression; falls back to plain `claude` (reason on stderr) when it cannot mirror (managed
  pinning, headers helper, mTLS, beta tracing, caller's own `--settings`), when docker is missing, or when
  health fails.
- Working directory `$AGENT_SYS_O11Y_OTEL_DIR`: no symlinks on the path; our components owned by the user with mode 0700; root-owned components sticky or not group/other-writable; any other owner refused (one rule, shared with the OTel source adapter);
  files created `O_CREAT|O_EXCL|O_NOFOLLOW` 0600 then renamed; otherwise fall back.
- `docker run -d --pull never --restart unless-stopped --network host --user uid:gid` of the official
  `otel/opentelemetry-collector-contrib` (container labelled as ours; config hash label): otlp receiver with
  the `bearertokenauth` extension (random token, stored 0600) → `otlp_http` exporter(s) to the user's
  endpoint(s) (sending queue + retry, no processors) and a size-rotated file exporter. Health = `docker
  inspect` shows our running container AND the health endpoint answers.
- `exec claude --settings` with per-signal local endpoints and `OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer
  <token>`; privacy switches untouched. The docker restart policy satisfies spec §9.5's restart; `--sweep`
  stops the container when no launched session is alive.

## 11. Build versus adopt

Adoption rule: a dependency is adopted when it has ≥ 20K GitHub stars, or is the reference implementation of
a standard, or is a de-facto standard by usage, or is already an agent_sys dependency; exceptions are listed.

| Need | Adopted | Rule | Why |
|---|---|---|---|
| Schema language, Python validation | JSON Schema 2020-12 + `jsonschema` (+ `referencing`) | existing dependency; reference impl. | language-neutral; agent_sys's own enforcement point |
| Mod-side validation | a JSON-Schema-subset validator in `protocol/ts` (supported keywords listed; a test fails on any other keyword) | written here | one source of truth; ajv (14.9k★) is below the bar; zod would restate every schema |
| Bundling | esbuild | ≥ 20K | mods import only bundled files |
| Tool calls | MCP python-sdk | ≥ 20K | standard tool calls; lifetime follows the session |
| Model calls | `claude -p --bare --json-schema` | the platform itself | validated structured output through the user's gateway |
| Record model | OpenTelemetry Log Data Model names | standard | matches Claude Code's own events |
| Collector | otel/opentelemetry-collector-contrib | reference impl. (user-approved) | content-identical forwarding |
| Python env | env_mgr recipe (as `agentsview.o11y.yaml`) | existing mechanism | agent_sys already installs into its prefix this way |
| Display host | Claude Code Mods | the platform | in-session, no extra process |
| Scheduling | **not** Claude Code CronCreate / `/loop` | rejected | a mod cannot call CronCreate; fires up to 15 min late; recurring tasks expire after 7 days; loses a run after Esc; no skip records |
| Written here | the mods, the door, the jobs, o11y-launch, schemas, skill | — | no adoptable component does the fleet logic or the domain views |

Size target (checked at review with `wc -l` over non-test sources, after formatting; set to the measured
scope, which every criterion needs): trigger ≤ 500, display ≤ 600, protocol/ts ≤ 250, protocol/py ≤ 250,
door ≤ 600, o11y_job ≤ 500, o11y_sources ≤ 1400, extract + log + facts ≤ 1300, o11y-launch ≤ 570 lines.

## 12. Test plan

| Criterion | Test |
|---|---|
| 1–3 | isolated interactive session (tmux, `--plugin-dir`, 1-min kick, packup every 2): submitted == recorded, marker per kind; queued kick → skip record also after a hot reload; packup cadence with a question in between |
| 4, 5 | child with `AGENT_SYS_O11Y=child` using an agent_sys skill stays dormant; unset env + model- and user-invoked skill activate; `/clear` keeps main and the root pin |
| 6, 8 | model requests before/after 100 ticks; a job on unchanged history makes no `claude -p` call |
| 7 | 200 scripted tool calls incl. failures, parallel batches, Esc; requests == thresholds; all served; model windows contiguous and disjoint |
| 9 | fixtures with repeated usage per `message.id` and subagent files; ccusage `session --json --offline` as an independent oracle |
| 10–12 | fixture registry waiting 11 min; dead child; dead main + new main start (with T's registry present); pool outbox incl. another user's records |
| 13–15 | MCP live call (default model) missing / unknown field → refusal → fixed; `echo` of an invalid record → backstop at next call, valid one accepted; 4 × 150 appends on the prefix path |
| 16, 25, 27 | fixture catalogue per §6.1 level; KernelForge fixture missing `best` → one error until the file changes; torn/invalid records skipped |
| 17, 18 | facts with old `generated_at` → stale mark; band within one period + poll; chords idle and busy for every sub-key |
| 19, 22 | `ps` after `/exit` and SIGKILL of the main; kill the MCP server, a job and the mod worker mid-run |
| 20 | two identical scripted sessions (with/without plugins, incl. a model-calling job) → config-dir and project `.claude/` diff limited to Claude's own per-session files; `strace -f -e openat,rename,unlink` of the out-of-process parts |
| 21, 30 | forwarded vs direct decode comparison against a local stand-in; collector down at launch → direct; sender without token refused; killed collector restarted, outage recorded; non-fleet session export unchanged |
| 23 | first-request input tokens and startup time: fleet / non-fleet / before-and-after mid-session activation |
| 24 | hand-started claude in the fleet's tmux server and directory → one question; answer or no answer → no repeat |
| 26 | fixture history: a file used then missing without an rm in the fleet's calls → one error; with the session's own rm → none |
| 28 | config with a misspelt key → one report with did-you-mean, previous config in effect |
| 29 | file modes; `--plugin-dir` overrides an installed copy; installer output lists every user step |
| 31 | history text naming a path and a command → no read of that path, no command; job runs with `--tools ""` |
| drift | prefix vs `env_mgr.prefix.Prefix.resolve` (incl. symlinked home); slug TS vs Python; validator verdicts and defaults TS vs Python |

## 13. Deviations from the spec

None.

## 14. New open questions

1. A live fleet run with the default model and a multi-hour soak at the real 30-minute cadence are pending.
2. A SIGKILLed last fleet session leaves the collector running until the next launch's sweep.
3. Whether docker's restart policy restarts the collector after a daemon restart without a live session (to
   be measured before relying on `on-failure` instead).
