# AgentsView as `agent_sys`'s o11y panel — design

Component design for `env_mgr/o11y/agentsview/`. The module-level design refers
here from `../../docs/design.md` §17.

**The rule that outranks every feature below:** o11y may never fail the thing it
observes. Every failure is one `log.warning` and a skip, with a test per mode.

## 1. The panel

[AgentsView](https://github.com/kenn-io/agentsview) is an external Go binary
that reads Claude Code's JSONL transcripts and serves search, analytics and
token-cost views. `agent_sys` reaches its backend through `claude-agent-sdk`,
which spawns the `claude` CLI, which writes exactly those transcripts — so the
two fit with no glue on either side. The whole integration is *where the
transcripts land* and *which directory the panel reads*.

**AgentsView's own code is never modified.** Every knob is one it publishes:
`--port`, `CLAUDE_PROJECTS_DIR`, `AGENTSVIEW_DATA_DIR`, `disabled_agents`.

### The prefix

`~/.infera_agent_sys`, laid out like `~/.local` (`bin/ share/ state/ run/`),
owned by `env_mgr` and named by the `AGENT_SYS_*` family in `prefix.py`. It
exists because the two obvious alternatives are both wrong: `/usr/local/bin` is
host state we promised not to touch, and `~/.local/bin` is the user's. Upstream's
`install.sh` is not used — it hardcodes `/usr/local/bin` with no override point.
The recipe (`recipes/agentsview.o11y.yaml`, `installer: bin`,
`importance: suggested`) installs a pinned release and verifies its published
`SHA256SUMS`.

`state/claude` is deliberately not under a run root: the daemon outlives any
single run, so the directory it reads must be a stable path.

### Session scoping — five gates

The panel must show only the sessions `agent_sys` produced. **The constraint
that outranks the feature: the user's own Claude Code must be untouched.**

| | |
|---|---|
| 1 | `CLAUDE_CONFIG_DIR=$AGENT_SYS_CLAUDE_HOME` in the **child's** environment dict, never in our `os.environ`. `material.deploy` sets its own per-attempt value, so `<zone>/config/projects` is symlinked into the prefix — credentials and settings stay the zone's, only the output is shared |
| 2 | `CLAUDE_PROJECTS_DIR` points the panel at that one root |
| 3 | `disabled_agents` switches off the other 60 providers. Pinned and hand-maintained; `check_disabled_agents` warns when it has drifted from the installed binary **in either direction** — a name the binary dropped breaks `serve` loudly, a name it gained leaks silently |
| 4 | `AGENTSVIEW_DATA_DIR` is ours, so a user's own archive and settings are untouched |
| 5 | `HOME` is redirected into the prefix for the binary's own subprocesses. AgentsView derives every provider's *default* root from `HOME`, so this needs no list and cannot go stale when upstream adds a provider we have never heard of. Found while measuring, and stronger than gate 3 |

### Lifecycle, port, failure

Started at the end of the `env_mgr` deploy path and left **resident**
(`daemon_idle_timeout = "0s"`; the default 20m would empty the panel for anyone
opening the URL after their run). Default port `18888`; resolution order is
`--agentsview-port`, then `AGENTSVIEW_PORT`, then the default, and an unusable
value falls back with a warning rather than failing. `--no-agentsview`,
`--dry-run` and `--clean` make no external call at all.

**A taken port is a warning and a skip, never a relocation.** We bind-probe
before launching precisely because `serve` would otherwise move quietly to the
next free port, and a panel on 18889 is a panel nobody knows the address of.
`--replace` is passed for the same reason: without it, `serve --background
--port N` silently attaches to any daemon already alive for this data directory
and reports *its* port, exit 0, `N` ignored.

**Reuse requires proof of ownership.** A live AgentsView on the port is not
evidence it is ours — a user's own daemon lists every session on the machine.
Two gates: it answers `/api/v1/agents` with 200 and JSON (a status code is not
an identity), **and** a live `daemon.<pid>.json` in our data directory names
that port. That record is AgentsView's own artefact, read never written; because
the data directory is ours alone, one found there was written by a daemon we
configured. It is removed on a clean stop, so only an unclean death leaves a
stale one, and that is caught by checking the pid.

**No validation path may start a daemon.** Measured: `health`, `projects` and
`session list` all autostart one on a port AgentsView picks — the delegation
this component exists to prevent, happening where nobody is watching.
`doctor sync` does not, and answers the same question. `AGENTSVIEW_NO_DAEMON=1`
does not rescue them; it makes them refuse outright.

**Success goes to the event stream** (`EventKind.O11Y_PANEL`), not `logging`:
this package never configures `logging`, so an info record reaches nobody while
`log.warning` still reaches stderr through `lastResort`.

### Known limitation

The zone symlink's behaviour under an enforcing policy is **untested, because
currently untestable**: `agent_sys` refuses to start any AI task under
`AGENT_SYS_NO_PERMISSIONS=0` today, before the executor runs, so nothing ever
traverses the link. That refusal predates this feature — measured with paired
arms differing in one file, both failing identically. If a confined child ever
does follow it, the prefix is under `$HOME`, which `DEFAULT_SYSTEM_SET` does not
grant; the likely repair is a grant on `$AGENT_SYS_CLAUDE_HOME/projects`, which
is a permissions decision and is deliberately not taken here. Read this as
"untested", never as "safe".

## 2. One project per run

AgentsView derives a project from the session's **deepest** path segment. Every
agent attempt runs in its own zone, and zones nest, so one run's sessions arrive
as several unrelated projects — measured on a real nested fixture, four sessions
of one run as `0_11e34171`, `0_f6daeb1b`, `task.main.b869ddf0_…` and
`task.solve_a.8c8fb4c1_…`.

**Renaming the directories cannot fix this, and that is the whole reason this
section exists.** PR #156 put the closure name into every runtime directory,
which made those strings readable; it did not join them, because a nested child
task fragments off from its parent however prettily both are named. The only
filesystem fix would be putting every attempt of a run under one directory,
which is precisely what zone isolation exists to prevent.

So `env_mgr/o11y/mapping.py` posts **one `explicit` mapping over the run root**
at run start, through AgentsView's own settings API. The dependency stays
unmodified.

### What was measured before any of it was written

Every property below came from a container probe against a real v0.42.0, not
from the API's shape or a field's name. Three of them changed the code.

| | |
|---|---|
| **`explicit` is the only usable layout** | The other legal value, `repo_dot_worktrees`, matched **zero** sessions across thirteen prefix shapes — including a genuine `<repo>/.worktrees/<name>` tree the archive had correctly identified. Why is **unsettled**; upstream source at `ff8fb4e8` would settle it. We do not use it |
| **`explicit` is depth-independent** | One mapping over `runs/<A>` caught that run's sessions at depth 3 *and* 5, and only that run's. Nesting is a non-issue — which is what makes the whole design work |
| **`Origin` is mandatory** | Without it a mutating call answers a plain-text `403 Forbidden`, not the JSON error shape. It reads exactly like a missing endpoint |
| **`machine` must be read, never assembled** | A mapping written with a `machine` the daemon does not recognise matches nothing and says nothing. It comes from the daemon's own `local_machine`. The recon itself ran in a container whose hostname was not the host's — which is how a hard-coded `gethostname()` would have shipped broken |
| **Names normalise `-` → `_`** | Everything else round-trips, including `@ : / + # % .`, spaces and non-ASCII; there is no length cap. We normalise before posting so the string we send is the string the panel shows |
| **No `apply`, `preview`, `reclassify` or token** | A mapping that exists *before* ingest labels the session at sync time. Those calls are only for sessions already in the archive |
| **`409` is success** | `POST` is not idempotent; uniqueness is `(machine, path_prefix)`, not including layout. A re-run of the same run id conflicts with its own row |
| **Prefix boundaries are segment-safe, longest wins** | `run-AAA` does not capture `run-AAAX`; a specific row beats a general one |
| **Classification reads the recorded cwd, not the disk** | A session whose directory never existed classified and mapped normally; one whose directory was moved away survived a `sync --full`. **Zone teardown after a run is harmless to the panel** |
| **AgentsView does not look downward** | A zone is a non-git directory containing `git clone --shared` at `<zone>/workspace` whose alternates point outside it. Byte-for-byte identical classification to a bare plain directory: `repository_path=''`, `worktree_relationship='unknown'`. It never reaches `alternates` |

### Retention, which is a policy and not hygiene

One row per run, and **no garbage collection** — deliberately. Deleting a row
does not disturb the panel on `apply`, but the next `sync --full` re-derives
labels from the mapping table and that run's sessions revert to their directory
names. **Keeping the label is keeping the row.** So any GC is a decision about
how long old runs stay named, not a tidiness measure, and a GC that silently
un-names last month's runs is worse than a table of small rows.

**Measured, and it is why no GC is needed.** The cost is `O(rows × sessions)`,
not `O(rows)` — visible only by repeating the whole curve at a different session
count, where the same row counts cost 3.5–4× more. It works out at ~0.75 µs per
(row × session), holding across a 20× range of the product: ~2.6 s added to a
`sync --full` at 5000 rows over 800 sessions, and nothing measurable below ~1000
rows.

**But `sync --full` is not the path the daemon runs.** Paired arms over an
identical 800-session archive differing only in the mapping table: idle 0.991 s
against 1.023 s, one new session 1.150 s against 1.189 s. **+32 ms and +39 ms at
5000 rows**, both inside the spread within either arm. Incremental sync — file
watch and poll timer — is unaffected.

Since runs and sessions grow together the mapping term is quadratic in run
count, but it loses to the linear re-parse cost of a full sync until roughly
**15 000 runs**, by which point a full sync takes ~19 minutes for reasons that
have nothing to do with mappings. If a GC is ever wanted, the trigger is
full-sync latency, not table size.

**One thing left unsettled, and it is the number a GC would turn on:** whether
`enabled=false` rows still cost a scan. Disabling rather than deleting is the
attractive shape for a GC — deleting un-names the run — and that measurement is
what would decide it. Re-run the scaling harness after flipping the rows via
`PUT /{id}`.
