# o11y — Specification

| | |
|---|---|
| Status | Normative for observing and nudging a fleet of interactive Claude Code sessions |
| Version | 3 |
| Updated | 2026-11-23 |
| Summary | What the fleet observability package must guarantee: who is observed, what triggers what, what is recorded at which level, what is displayed, the one protocol between its parts, and what may never happen to the user's work or configuration. |

Source: [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §1 (Observability is its own package) — this package
delivers its interactive-fleet part; task-run, per-spec and per-task metrics stay open there. Part of
[`../../docs/spec.md`](../../docs/spec.md).

---

## 1. Purpose

agent_sys work is driven from tmux by one **main** Claude Code session — the user's entry point and the
coordinator — which starts **child** sessions on demand (a KernelForge campaign is the usual one).
Nothing today keeps that fleet moving when the main session goes quiet, records what it did at the
level a person needs, or shows where the optimisation stands. This package does those things for one
fleet at a time, as five subsystems:

| | |
|---|---|
| **Trigger** | Keep the main session pushing (timed kick, periodic packup, questions about unknown sessions), maintain the fleet registry, and decide, by counting, when analysis should run |
| **Analysis** | Turn what the fleet's sessions did into levelled records (info / warning / error), extract structured facts into the ledger, and project the facts the display needs — reading each session's history incrementally |
| **Display** | Show the task, its progress, the run chain, the machines and the KernelForge campaigns inside the main session |
| **Ledger door** | The single validated writer of the fleet's shared records, offered to agents as tool calls and to programs as a command, with a backstop for writes that bypass it |
| **Protocol** | The only way the others exchange anything: schema-described files, plus one declared launch contract from trigger to analysis |

### 1.1 In scope
- Interactive Claude Code sessions on one host that belong to one fleet: one main, any number of children.
- The run ledger: the fleet's structured record of tasks, runs, results, plans, ideas, decisions,
  campaigns, hot operators, kernels, incidents and findings, written by sessions and by analysis.
- An optional copy of Claude Code's own OpenTelemetry stream as an additional source (§9).

### 1.2 Out of scope
- agent_sys task-graph runs and their agents (`monitor`, `env_mgr/o11y`); task-run, per-spec and
  per-task metrics (ROADMAP §1). §11 states the boundary.
- `claude -p` fleets, Desktop and IDE surfaces, more than one host.
- Changing how any observed session does its work. This package nudges the main session; it never edits
  a session's files, settings or history.

### 1.3 Limits that follow from living in the session
- Every always-on part runs inside the main session (§2 principle 4). **The death of the main session's
  own process is not observed while it is dead**: it is reported when a main session of the same fleet
  next starts. Deaths of children are observed while the main session lives.
- After the user interrupts the main session, Claude Code refuses automated prompts until the user's next
  input; a due prompt waits for that (§4.3).

## 2. Design principles

0. **Modular and decoupled above all; adopt before building; least code.** Every subsystem and every
   module inside one has one responsibility and a stated interface, and depends on others only through
   it. A mature, maintained industry component is preferred to anything written here; what is written here
   is the least code that the remaining requirements need.
1. **Mechanics cost no tokens.** Counting, timing, registering, deciding that something is due, and
   every detection a program can make are done by code. A model is called only where judgement is
   needed: reading new history for task-level meaning, and extracting structured facts into the ledger.
2. **One protocol.** Subsystems never import one another. They exchange only schema-described files, except
   for one declared launch contract by which the trigger starts analysis (§8.6).
3. **One writing component per file.** Every protocol file has exactly one writing component; appends to
   shared records go through the ledger door, which validates before it writes. The user may hand-edit
   configuration.
4. **Lifetime follows the session.** Every always-on part lives inside the main session and ends with
   it. Work that outlives a call is a bounded one-shot process. No component asks the user to operate a
   service by hand. The optional OTel collector (§9) is the one exception; it is started by the fleet's
   launch and stopped when no fleet session is alive.
5. **o11y cannot hurt the work.** A failure of any part of this package degrades observation only: it
   never blocks or fails an observed session, and its added latency stays within §12's bounds. §9.5 states
   the one cost the optional OTel copy can impose on the user's own telemetry.
6. **The user's configuration is the user's.** This package's code never writes the Claude configuration
   directory or any project's `.claude/`; Claude Code processes this package starts for model calls keep only
   Claude Code's own per-process files. Registering the plugins with Claude Code is a step the user runs;
   every other user-side setting is documented (README) for the user to apply.
7. **Incremental by construction.** History already analysed is never analysed again; a pass reads only
   what was appended since the last pass, and history older than the first pass is read only within a
   configured lookback.
8. **Every record names its evidence.** A record says which session, which history position or file it
   came from, and whether it was written by a session or extracted by analysis.
9. **Storage follows agent_sys's layout.** Persistent state and installed artefacts live under agent_sys's
   own prefix (`env_mgr`'s prefix), never in `$HOME` directly and never inside a project checkout. The one
   exception is the optional collector's working directory (§9), which must be on local disk.

## 3. The fleet

3.1 **Membership.** A fleet is one main session and the children it started. A session is the main of
a fleet when it was launched with o11y enabled, or when it first uses an agent_sys skill and was not
launched as a child or vetoed. A main stays main across `/clear` and reload of the same process. A session
launched as a child never becomes a main.

3.2 **Identity of a fleet.** A fleet is identified by the directory its main session was launched in, fixed
at activation; later directory changes of the session do not move it. Two mains launched in the same
directory share the fleet.

3.3 **Registry.** The package maintains, for its fleet, the list of sessions with their identity (including
earlier identities of the same session after `/clear`), role, parent, status (busy / idle / waiting, and on
what), how long they have been in that status, and whether their process is alive. A main whose process
died stays listed, marked dead, for a retention period.

3.4 **Sources of truth.** Session existence, status and waiting reason come from Claude Code's own
per-process session records; parentage comes from the fleet's own launch records. A malformed or torn
record does not erase a session: the registry keeps the last good value and marks it. The registry has one
judge; every other part reads it.

3.5 **Sessions outside the convention.** A live session that is not in the fleet's launch records but
plausibly belongs to it (same tmux server, launched under the fleet's directory, after the main started) is
not guessed into the fleet: the main session is asked, once per such session, whether the session is its
own and what it is for. The answer — a child with its purpose, or not part of the fleet — is recorded,
visible in the registry, and not asked again; an unanswered question is not repeated either.

3.6 **Children are reached through the main session only.** The package addresses prompts to the main
session; the main session relays to its children.

## 4. Triggers

4.1 **Kick.** On a configurable interval (default 30 minutes) the main session receives a configurable
prompt (default: keep pushing, re-read the task book, check for drift, record runs in the ledger, relay
to children). Every automated prompt starts with a fixed marker naming its kind; values interpolated from
sessions are confined to a safe character set, length-capped and quoted as data; the exact text handed to
Claude Code is recorded.

4.2 **Packup.** On its own interval (default 60 minutes) or every N kicks (default 3), whichever comes
first, the kick is replaced by a packup prompt (pack up experiments since the last packup, complete the
ledger, relay to children). Questions (§3.5) neither count toward nor reset this cadence.

4.3 **At most one pending.** If an automated prompt is still queued, a newly due one is skipped and
recorded as skipped; this holds across a reload of the package. A prompt removed from the queue or refused
is recorded as refused, and one is retried after the user's next completed (not aborted) turn. A missed
interval is not replayed more than once.

4.4 **Frequency triggers.** Tool calls and turns of the main session are counted; every N (configurable
per purpose) an extraction, a facts refresh or a log analysis is requested. A slow timer (configurable,
default 15 minutes) requests log analysis even when the main session is idle, so registry-only events (a
dead child, a pending permission) are not missed. The registry is refreshed on a fixed short period (30
seconds).

4.5 **Requests coalesce, none is lost.** At most one analysis of each kind runs at a time; a request made
while one runs is served by that run before it ends, or by the next.

## 5. The run ledger and the ledger door

5.1 **Content.** The ledger records, per fleet: task definitions (metric, unit, direction, baseline,
target), decisions, run starts and ends (theme, kind of point, machines, status, value, evidence), the
current best, plans, ideas, KernelForge campaigns, hot operators and their share, per-kernel stage including
end-to-end validation, incidents (including service up/down and conclusions worth logging), and findings.
Corrections are new records that supersede earlier ones; nothing is edited in place.

5.2 **One door.** Main sessions, children and analysis all write the ledger through the same writer,
offered to agents as standard tool calls and to programs as a command. The writer validates every record,
refuses unknown fields, and answers an invalid record with the specific missing, wrong or misplaced fields
and the full description of that record kind.

5.3 **Backstop.** An invalid record, an in-place edit, or a whole-file write of the ledger that bypasses the
door is detected and reported, with the reason and a pointer to the door, at the next tool call of a fleet
session. A valid record appended outside the door is accepted.

5.4 **Agent-facing operations** beyond writing the ledger: read the ledger, register a child launch or
answer a question about a session (which returns how to launch the child as a child), and read the fleet's
status.

5.5 **Writer identity.** Every record names the session that wrote it and that session's role (main,
child, analyzer, or other when the session belongs to no fleet).

5.6 **Offered only where needed.** The door's tools are offered only to sessions that belong to a fleet; a
session that joins mid-way gains them without a restart.

## 6. Analysis

6.1 **Levels.**
- `info` is task-level only — an analysis finished, a service started or stopped, a conclusion reached —
  plus resource milestones: new-token consumption crossing a further step (§6.2), a compaction, a machine
  requested, granted or released.
- `warning` is an unplanned event: a session waiting more than 10 minutes on a permission prompt, a
  sandbox request or a worker request; a machine lost or reclaimed; a session idle beyond a threshold; a
  KernelForge campaign reported running whose state has not changed beyond a threshold.
- `error` is anything that stops a managed session or the system: an API error that is fatal by kind
  (billing, authentication, account hold, verification) on its first occurrence; transient API errors
  (including rate limits) repeated K times; a dead child; files a session used that were removed by
  someone other than the fleet; a major incident; a source whose format no longer matches; a collector
  outage (§9.5).

6.2 **Tokens.** A session's tokens are summed over the unique API messages recorded in its history and its
subagents' histories, kept as two totals: new tokens (input, cache creation, output) and cache reads. The
milestone counts new tokens: one record per analysis pass in which a session's or the fleet's new-token
total crossed one or more steps (default 1M); history that existed before the first pass is the baseline.
Both totals are displayed. There is one accumulator.

6.3 **Detection before judgement.** Every event a program can detect from the registry, the history, the
machine pool or KernelForge's state is detected by code. The model handles only what needs reading, and
reads only main and child histories (subagent histories feed tokens and detectors).

6.4 **Incremental.** Each session's history is read from where the previous pass stopped. A history that
shrank or was replaced is re-scanned by the detectors (which are idempotent); the model resumes from the
end, never re-reading what it has read.

6.5 **One extractor.** Structured facts that sessions did not record themselves (task, current run, plan,
campaign roots, hot operators, kernel stages, end-to-end validation) are extracted by one frequent, cheaper
model pass into the ledger, marked as extracted and superseded by anything a session writes. Display facts
are a model-free projection of the ledger and the structured sources.

6.6 **Source drift is loud.** When a source file lacks a field this package relies on, or a value has the
wrong type, analysis records one `error` naming the file and the field, repeats it only after the file
changes, and skips that source.

6.7 **Untrusted text stays data.** Text read from histories never chooses a file-system location, a
command, or a tool for this package; model passes run without tools.

## 7. Display

7.1 Shown only in the main session of a fleet; never in a child. Its band appears within one registry
period plus one display poll of activation; its pane opens when the terminal has room.
7.2 Two modes: narrow (brief) and wide (detailed). The band is visible whenever active and no dialog is open.
7.3 Tabs: **general** (task metric, baseline, target, current best as value, % vs baseline and % of target,
new tokens and cache reads, wall time since the task started, progress per hour and per 1M new tokens),
**detail** (current run and its point, the last n runs (default 3), the next planned point, machines and
what each runs), **KernelForge** (hot operators and share, campaigns with age, gain and gain per hour,
per-operator stage and end-to-end validation), **log** (the latest N records, filterable by level).
7.4 Every tab says how old its facts are and marks them stale past their declared freshness.
7.5 A master key followed by one key reaches every tab, mode, the band, and the configuration of trigger,
analysis and display, whenever no dialog is open.

## 8. Protocol

8.1 Every exchanged file has one published schema and one writing component; every record carries a
schema version.
8.2 Readers never see a torn whole-file document; appended records are whole lines.
8.3 Readers skip invalid records, report them, and never stop on them.
8.4 A configuration document with an unknown or invalid key is reported once (with the likely intended key)
and not applied; the last valid configuration stays in effect, else the defaults. The work is never blocked.
8.5 Level records follow a standard log record model and carry a key identifying the same underlying fact
across passes. Event names and levels are published with the protocol.
8.6 **Launch contract.** The trigger starts analysis through one declared command per analysis kind; the
command's environment names the fleet and the Claude configuration; its exit status says drained (or
already running) versus failed; its runtime is bounded.
8.7 Files are readable by their owner only.

## 9. Optional copy of Claude Code's OpenTelemetry stream

9.1 When enabled, fleet sessions send their OpenTelemetry stream to a local collector that forwards it to
the endpoint(s) the user already configured — identical in content after decoding, not byte-identical —
and keeps a local copy for analysis.
9.2 Sessions outside the fleet are unaffected; the user's export configuration is not edited.
9.3 If the local collector is not healthy at launch, or the user's export configuration uses settings the
launcher cannot mirror, the session starts with its normal export.
9.4 Content-privacy switches are never turned on by this package. The collector accepts only this user's
sessions.
9.5 **The cost of the copy:** while the collector is down mid-session, that session's telemetry is lost for
the user's endpoint too. The collector is restarted automatically while a fleet session is alive, and every
outage is recorded as an error.

## 10. Installation, activation, configuration

10.1 Installed as Claude Code plugins from a marketplace inside this repository, registered by a step the
user runs; a development mode loads the same plugins from the working tree and takes precedence over an
installed copy.
10.2 Activation and role are carried by `AGENT_SYS_`-prefixed environment variables; the data location
follows agent_sys's prefix and may be overridden by an `AGENT_SYS_`-prefixed variable.
10.3 Every user-side step the package needs (registering the plugins, a permission rule, key bindings) is
documented for the user to apply; nothing applies it for them.
10.4 In a fleet session the door's tool definitions cost at most 1,000 input tokens per request; elsewhere
nothing. Installing the plugins adds at most the start latency of criterion 32 to a session outside every
fleet.
10.5 What the user configures: kick and packup timing, frequency thresholds, the slow timer, models,
effort, lookback, prompts, the task book, and display mode, row count and refresh. Everything else is fixed.

## 11. What this module does not own

- **agent_sys task runs.** `monitor` owns a task's exceptions and the pusher inside a run; ROADMAP §1's
  task-run, per-spec and per-task metrics remain open there.
- **The transcript panel.** `env_mgr/o11y` deploys AgentsView over agent_sys-run transcripts.
- **The prefix layout.** `env_mgr` owns the prefix; this package uses its `share` and `state` directories.
- **KernelForge, the machine pool, Claude Code.** They are sources; their formats are theirs.

## 12. Acceptance criteria

1. With o11y enabled and no user interrupt since the last input, an idle main session receives the kick
   within one registry period of its due time; the text handed to Claude Code equals the recorded text and
   starts with the kick marker.
2. With a kick queued and not yet run, a newly due kick is skipped and recorded — also after a reload of the
   plugin; never two queued.
3. Every N-th kick (default 3) is a packup, or earlier if the packup interval elapsed first; questions do not
   affect the count.
4. A session launched as a child, or any session whose activation variable has a value other than enabled,
   never shows the display and never receives automated prompts, even after using an agent_sys skill.
5. A session with no activation variable becomes the main on its first agent_sys skill use, whether the
   model or the user invoked the skill, and remains main after `/clear`.
6. Counting, triggering, timing and registering cause no model request.
7. With N=10, after 200 tool calls including failed calls, parallel batches and an interrupted turn, every
   threshold crossed produces one analysis request, every request is served, and no history window is given
   to the model twice.
8. Analysis on unchanged history makes no model request.
9. Per session, the new-token and cache-read totals equal the sums over unique API messages recorded in its
   history and its subagents'; a pass in which a new-token total crosses steps writes exactly one milestone.
10. A child or main waiting more than 10 minutes on a permission prompt produces one warning within one
    slow-timer period after the 10 minutes elapse.
11. A child killed with SIGKILL produces one error within one slow-timer period; a main killed with SIGKILL
    produces one error when a main of the same fleet next starts.
12. A machine-pool `lost` event for a fleet request produces one warning; one grant and one release produce
    one info each; another user's pool records produce nothing.
13. A ledger record missing a required field, or carrying an unknown one, is refused with a message naming
    the field; an agent given the message fixes it on the next call (observed with the default model).
14. An invalid ledger record appended outside the door is reported to the writing session at its next tool
    call; a valid one is accepted.
15. Two sessions appending to the ledger concurrently on the prefix's file system never produce a torn or
    interleaved line.
16. A KernelForge state file missing a relied-upon field produces one error naming the file and field until
    the file changes; other sources continue.
17. Each display tab shows its contents (§7.3), its facts' age and a stale mark past the declared freshness;
    the band appears within one registry period plus one poll of activation.
18. The master key followed by each sub-key reaches the intended tab, mode, band toggle or configuration
    form, while the main session is idle and while it is busy, with no dialog open.
19. After the main session exits normally or is killed, no process started by this package remains except
    an analysis run, which exits within its budget, and the optional collector while another fleet session
    is alive.
20. Between two identical scripted sessions with and without the plugins (one including an analysis pass that
    calls the model), the Claude configuration directory and the project's `.claude/` differ only in files
    Claude Code itself rewrites every session; no settings, plugin, skill or CLAUDE.md file changes.
21. With OTel enabled, the content forwarded to the user's endpoint is identical after decoding to what the
    session would have sent directly; with the collector down at launch, the session exports directly; a
    sender without the fleet's credential is refused by the collector.
22. Disabling or crashing any part of this package leaves every observed session's work unaffected.
23. The door's tool definitions cost at most 1,000 input tokens per request in a fleet session and none
    elsewhere; a session that activates mid-way gains the tools without a restart.
24. A live session not in the launch records, in the fleet's tmux server and directory, causes exactly one
    question to the main session; after a child or not-in-fleet answer, or no answer, no further question.
25. Levels: a fixture set covering each §6.1 case produces exactly the listed records at the listed levels.
26. A file used by a fleet session and removed by someone else produces one error; a file the session
    removed itself produces none.
27. Torn or invalid protocol records and source records are skipped, reported, and do not stop any part;
    the registry keeps the last good value of a torn session record.
28. A configuration document with a misspelt key is reported once with the likely intended key, and the
    previous valid configuration stays in effect.
29. Protocol files are created owner-only; the development plugins override installed ones; the README
    documents every user-side step.
30. With OTel enabled, sessions outside the fleet keep their direct export; privacy switches stay off; a
    killed collector is restarted while a fleet session is alive and the outage is recorded as an error.
31. Text in a history that names a path or a command never causes this package to read that path or run that
    command; model passes have no tools.
32. Added latency, measured by median over at least 10 runs (100 for per-call figures) on the reference host:
    outside every fleet, a `-p` session starts at most 0.4 s and an interactive session at most 0.25 s later
    than without the plugins; the backstop adds at most 10 ms per tool call outside a fleet, 80 ms in a fleet
    session when the ledger is unchanged, and 600 ms on a call that changed the ledger.

## 13. Open questions

1. Whether a session declared "not part of the fleet" should still have its permission waits and crashes
   reported, since it shares the user's machine and tmux server.
