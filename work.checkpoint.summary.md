# Checkpoint summary — AgentsView as `agent_sys`'s o11y panel

Append-only. One section per 30 minutes of wall clock. Earlier sections are
never revised, including their wrong estimates — the record over time *is* the
value of this file.

Effort start (T+0) taken as **2026-09-03 07:36 UTC**, the minute the workspace
`/home/yihou/ws.agentsview_o11y/` was created.

The previous effort's log is preserved at
`work.checkpoint.summary.e2e_deploy.20260903-0736.md.bak`.

Reporter reads, cheapest first: `git log --oneline` in the worktree;
`/home/yihou/ws.agentsview_o11y/recon/PHASE0.md` and `ACCEPTANCE.md`;
the plan's checkboxes in `docs/superpowers/plans/2026-09-03-agentsview-o11y.md`;
`python -m pytest tests/env_mgr tests/cli -q` in `agent_sys/`.

---

## T+0 — 2026-09-03 07:50 UTC (baseline)

### 1. Progress

**Effort: ~8 %.** Elapsed 14 minutes. Estimated remaining: 4–6 hours.
**Reliability of that estimate: low** — it rests on two unmeasured facts (see
§5), and if either goes the wrong way the design changes rather than the
schedule.

Done: research on AgentsView's configuration surface; design agreed with the
user across four sections and written to
`docs/superpowers/specs/2026-09-03-agentsview-o11y-design.md`; task `CLAUDE.md`
rewritten (old one backed up); the seven-phase implementation plan written.

Not started: all code.

### 2. Current state

Two commits, no code yet. Phase 0 (the two experiments) and Phases 1–2 (prefix
+ supervisor, both pure unit-testable) are independent and start in parallel.

### 3. Code problems — fixed / unfixed

None yet.

### 4. Non-code problems

One design constraint discovered while reading, worth recording because it
shaped the plan: `env_mgr/prepare.py:683` documents a test
(`test_env_manager_exposes_exactly_these`) that pins `EnvManager`'s method set,
so the side-car could not be hung off that class without a deliberate decision.
It is called once from `cli/main.py` instead — which is also the correct
cardinality, since `prepare()` runs per task and the daemon starts per
deployment.

### 5. Undetermined questions

Both are named experiments in Phase 0 of the plan; neither is guessed at, and no
code depends on an assumed answer.

1. **Does the official `install.sh` accept an install prefix?** If not, the
   recipe's `install:` line becomes a release-tarball unpack. Either way no new
   installer class is needed — `installers/bin.py` already has the right shape.
2. **Does `claude-agent-sdk` honour `CLAUDE_CONFIG_DIR` for credential lookup,
   or only for transcript placement?** If only the latter, credentials get
   symlinked (never copied) into the prefix — Task 4.2. If transcripts are *not*
   redirected at all, the isolation design fails and comes back to the user.

### 6. New commits

| commit | what |
|---|---|
| `40be959` | approved design doc + rewritten task `CLAUDE.md`; previous task's `CLAUDE.md` and checkpoint log preserved as `.bak` |
| `fc8dc74` | the implementation plan: Phase 0 experiments, Phases 1–5 TDD, Phase 6 the six-check acceptance on `demo2` |

### 7. Other

The rule that governs every judgement call in this effort: **an o11y side-car
may never fail the thing it observes.** Every failure path is one `log.warning`
and a skip, there is a unit test per path, and `cli/main.py` wraps the whole
call in a bare `except Exception` as a structural backstop.

The second rule, equal in weight: **the user's own Claude Code must be
untouched.** `CLAUDE_CONFIG_DIR` is placed in the child's environment dict and
never in this process's `os.environ`; one test exists whose only job is to hold
that line.

---

## T+1 — 2026-09-03 08:18 UTC

### 1. Progress

**Effort: ~45 %.** Elapsed 42 minutes since T+0. Projected remaining:
2.5–4 hours. **Reliability: medium** — this is the first checkpoint with real
signal (11 commits, both Phase 0 unknowns resolved favourably), which is much
better than the T+0 guess, but the projection still assumes Phase 6 acceptance
goes cleanly on first try, which is unverified.

Done since T+0: both Phase 0 experiments resolved (§5, both closed out); Phase
1 (prefix + env var names); Phase 2 (port resolution + fail-open
`ensure_running`, including the "only reuse a daemon that is ours" check);
Phase 3 (recipe installer); Phase 4's transcript-scoping piece
(`CLAUDE_CONFIG_DIR`); Phase 5 (CLI flags, call site, `--dry-run`/`--clean`
exemption). That is 11 commits across 5 of 7 phases.

Not started: Phase 6 (acceptance run on `demo2`) — no `ACCEPTANCE.md` exists
yet in the workspace.

### 2. Current state

Working tree has **uncommitted changes** on top of the 11 commits: `cli/main.py`,
`env_mgr/o11y/agentsview.py`, `env_mgr/recipes/agentsview.o11y.yaml`, and both
test files. Reading the failing tests (not guessing): they reference
`cli_main._was_freshly_installed` and `cli_main.ensure_installed`, neither of
which exists yet in `agent_sys/cli/main.py` — this reads as a red-state TDD
step (test written, implementation not yet written) refining how a fresh
install is distinguished from an already-satisfied one, on top of already-committed
Phase 3/5 work. Consistent with the plan's own step order
("write the failing test" → "watch it fail" → "implement").

### 3. Code problems — fixed / unfixed

**Fixed** (read from commit messages, not verified independently beyond the
test suite below): a stale `PATH` claim in the agentsview recipe comment
(`3a39b08`); a daemon-reuse check now also confirms the found process is both
agentsview *and* ours before reusing it (`dffecb4`).

**Unfixed, currently red:** `python -m pytest tests/env_mgr tests/cli -q` in
`agent_sys/` → **6 failed, 618 passed, 2 skipped, 2 xfailed**. All 6 failures
are the in-progress install-messaging tests described in §2
(`test_agentsview_flags.py::test_a_fresh_install_says_so_exactly_once` and 5
siblings, plus `test_imports.py::test_nothing_new_imports_the_installer_machinery[agentsview.py]`).
No failure outside that cluster.

### 4. Non-code problems

None observed this interval.

### 5. Undetermined questions — resolutions

Both questions open at T+0 are now closed, per
`/home/yihou/ws.agentsview_o11y/recon/PHASE0.md` (measured, not inferred):

1. **Install prefix knob**: `install.sh` has none — hardcodes `/usr/local/bin`
   or `$HOME/.local/bin` with no env var override anywhere in the 214-line
   script. Fell back to the release-tarball path per the plan's own
   contingency; sha256 verified against the GitHub release's `SHA256SUMS`.
2. **`CLAUDE_CONFIG_DIR` behaviour**: transcripts redirect, credentials resolve
   natively, and `$HOME/.claude/projects` was diffed before/after (801 files,
   empty diff) and confirmed untouched. The symlink fallback (Task 4.2) is
   **not needed** and the plan itself says to skip it under this verdict.

One thing PHASE0.md flags that the design didn't anticipate: a real `claude -p`
run under the redirected var also created `.claude.json`, `backups/`, and
`sessions/` directly under the prefix root (not just `projects/`), lazily on
first run — noted as a documentation gap in the design's tree diagram, not a
blocker.

### 6. New commits

| commit | what |
|---|---|
| `4fdc454` | the `~/.infera_agent_sys` prefix and its env var names |
| `890679d` | publish the prefix env var names from `paths.py` |
| `489197a` | agentsview side-car port resolution and bind probe |
| `314ceaa` | `ensure_running` warns and skips on every failure path |
| `dffecb4` | only reuse an agentsview daemon that is agentsview and is ours |
| `4cc1d34` | scope agent transcripts to the prefix via `CLAUDE_CONFIG_DIR` |
| `a7d30ac` | document the agentsview o11y side-car |
| `7fc1cb6` | `--agentsview-port` / `--no-agentsview` and the o11y call site |
| `557a350` | agentsview o11y recipe, installed into the prefix |
| `3a39b08` | fix: stale `PATH` claim in the agentsview recipe comment |
| `a997b31` | fix: no o11y daemon for `--dry-run` or `--clean` |

### 7. Other

The plan file's checkboxes (`docs/superpowers/plans/2026-09-03-agentsview-o11y.md`)
are **all still unchecked** despite 5 of 7 phases having landed commits — the
checkbox state is not being kept live and is not a usable progress signal from
outside; commit messages and the test suite are the only reliable read this
interval.

`work.checkpoint.summary.md` in the working tree (85 lines pre-this-edit) is
shorter than what's committed at `HEAD` (2690 lines) — checked before writing
here: the 2690-line version is the *previous* task's log, byte-identical to
`work.checkpoint.summary.e2e_deploy.20260903-0736.md.bak` (both 2690 lines).
This commit will be the first to record the fresh T+0/T+1 file at `HEAD`; not
an incident, just flagging why `git diff --stat` on this file looks alarming
(2657 deletions) if read without checking the `.bak`.

---

## T+2 — 2026-09-03 08:49 UTC

### 1. Progress

**Effort: ~55 %.** Elapsed 73 minutes since T+0. Projected remaining:
2–3 hours. **Reliability: medium-high** — Phase 6 has now actually run once
end-to-end and produced a real, dated verdict document (not a guess); the
projection's main risk is that the verdict names 3 concrete follow-up items
(§ below) whose fix effort is unmeasured.

Since T+1: the red test cluster from that checkpoint is fixed (3 commits:
`bfec099`, `255b087`, `fac0e0d`); full suite is green again (**626 passed, 2
skipped, 2 xfailed**, up from 618 passed / 6 failed). Phase 6 acceptance was
then run for real against `examples/demo2` and **written up in
`ACCEPTANCE.md`** — read in full this interval, not skimmed.

### 2. Current state

**Phase 6 acceptance verdict: FAIL.** Read directly from
`/home/yihou/ws.agentsview_o11y/recon/ACCEPTANCE.md` (dated 2026-09-03, run
08:15:23–08:23:17 UTC, `EXIT=5`): checks 1, 2, 4 fail; checks 3, 5, 6 not
executed because the task's own instruction was to stop at the first
non-empty check-4 diff. This is the correct way to read a "PASS"-shaped 6/6
checklist that is actually 0/6 — the document says so itself, plainly, in its
own first line.

There is now uncommitted work in progress on `env_mgr/o11y/agentsview.py` and
its test (+223/−14 lines) — reads as the fix for the acceptance doc's item 1
below (currently in flight, not yet committed).

### 3. Code problems — fixed / unfixed

**Fixed:** the readiness-probe transcript leak into `~/.claude/projects`
(`fac0e0d`) — this was acceptance item 2 in last run's list, already landed
before this checkpoint even though `ACCEPTANCE.md` predates the commit by 6
minutes.

**Unfixed, found by Phase 6 and named for a specific phase in
`ACCEPTANCE.md`'s own words:**
1. `OTHER_PROVIDERS` in `agentsview.py::write_config` is a hand-written list of
   31 provider names; the real `agentsview v0.42.0` binary rejects the first
   unrecognised one (`"claude-cowork"`) and `serve --background` exits 1 —
   meaning **the panel has never once actually started** in any run so far.
   The fail-open contract held (one warning, run continued), so this did not
   break demo2, it just means checks 1 and 3 have no panel to test against.
   Uncommitted work in progress looks aimed at this.
2. Check 4's own method is unusable on a box where this agent team works
   inside the repo under test: 3 of 4 new `~/.claude/projects` entries during
   the acceptance run were this team's own security-review sessions on
   `agentsview.py`, not a leak.
3. The plan document itself has a wrong command:
   `python -m cli.main run examples/demo2` (positional) should be
   `--package examples/demo2`.

### 4. Non-code problems

The acceptance run also surfaced an unrelated demo2 content failure
(`solutions_c: invalid`, `grade: waiting_handoff`) with no connection to o11y —
flagged in `ACCEPTANCE.md` as observed-not-investigated, and I'm passing that
framing through unchanged rather than re-diagnosing it myself.

### 5. Undetermined questions

One new one, opened by check 2 in `ACCEPTANCE.md` rather than closed: zero
transcripts reached the prefix during the demo2 run, and zero reached the
zone's `config/projects` either — the predicted mechanism
(`prepare.py:480` set, then `prepare.py:533`→`material.py:89` overwrite) is
named as the *suspected* cause but the doc itself says "does not match that
prediction cleanly... open, not settled." I have not attempted to settle it;
it is not mine to resolve from the outside.

### 6. New commits

| commit | what |
|---|---|
| `bfec099` | `ensure_installed` for the agentsview recipe item |
| `255b087` | install the o11y panel on first run, with a one-line notice |
| `fac0e0d` | fix: the readiness probe writes its transcript into the prefix |

### 7. Other

Full `pytest tests/env_mgr tests/cli -q` in `agent_sys/` this interval: **626
passed, 2 skipped, 2 xfailed in 110.14s** — clean, and notably slower than
T+1's 18s run (18 s vs 110 s), consistent with the acceptance run's real
`agentsview` binary and `claude -p` subprocess calls now being exercised by
the suite rather than the fake shell-script binary alone; not confirmed by
reading which specific tests grew, just flagging the wall-clock jump.

---

## T+3 — 2026-09-03 09:22 UTC

### 1. Progress

**Effort: ~65 %.** Elapsed 106 minutes since T+0. Projected remaining:
1.5–2.5 hours. **Reliability: medium** — the acceptance-item-1 bug (§ T+2) is
now closed with unusually thorough measurement (below), but there is no
updated `ACCEPTANCE.md` yet, so I cannot say checks 1/2/4 now pass — only that
the specific cause named for check 1 is fixed and re-verified against the
real binary.

Since T+2: 3 commits (`09e4b97`, `dde3720`, `5bd5ade`), all on the single
`OTHER_PROVIDERS` problem from `ACCEPTANCE.md` item 1. Full suite green:
**643 passed, 2 skipped, 2 xfailed in 18.9 s** (up from 626, and back to
T+1's fast wall-clock — the slow 110 s run at T+2 was the acceptance run
itself sharing the interval, not a new steady-state cost). Working tree is
clean (`git status --short` empty).

### 2. Current state

The `OTHER_PROVIDERS` fix went through **three iterations in one interval**,
each documented in `/home/yihou/ws.agentsview_o11y/recon/PHASE0.md` §0.3–0.5
(read in full, not skimmed):

1. `09e4b97` — measured the 60 provider slugs the real `agentsview doctor
   sync` accepts (not the docs table); found 5 of the original 31 guessed
   slugs wrong (e.g. `claude-cowork` → `cowork`); cross-checked one-name-at-a-
   time and all-at-once against the real binary, including a negative-control
   bogus name to confirm the validator isn't a no-op.
2. `dde3720` — went further: derive `disabled_agents` live from the binary at
   install time, since a hardcoded list can also *miss* a provider silently
   (gate-3 leak with no error). **Team lead rejected this design** (recorded
   verbatim in PHASE0.md): a silent upstream rename would then silently change
   what the panel shows with no commit to review.
3. `5bd5ade` — reverted to the pinned list, kept the discovery mechanism only
   as a two-directional completeness check (rename-or-removal, the original
   direction, plus addition-and-never-listed, the new direction) that names
   every offending slug in one warning.

This is not just a fix landing — it's a **design decision reverted by review
mid-flight and replaced with a stricter check**, worth recording precisely
because it changes what "done" means for this item.

### 3. Code problems — fixed / unfixed

**Fixed:** the wrong-slug bug that stopped `serve` from ever starting
(acceptance item 1) — closed as of `5bd5ade`, verified end-to-end against the
real installed v0.42.0 binary (`check_disabled_agents` → `()`, clean, both
directions).

**Still open, unchanged from T+2:** acceptance item 2 (transcript leak
mechanism between `prepare.py:480` and `material.py:89` — "open, not
settled"); item 3 (check 4's method can't distinguish a leak from this team's
own sessions); item 4 (plan doc's wrong CLI invocation). I see scratch logs
this interval (`demo2_noagentsview.log`, `demo2_portbusy.log`, an `runA`
before/after diff of `~/.claude/projects`) that look like probes toward
checks 5/6, but **no new `ACCEPTANCE.md`** — I am not reporting these as
passed or failed since I have no written verdict to read, only raw logs.

### 4. Non-code problems

None new this interval.

### 5. Undetermined questions

Unchanged from T+2 — the transcript-routing mechanism for check 2 is still
open per the last written verdict.

### 6. New commits

| commit | what |
|---|---|
| `09e4b97` | measure `OTHER_PROVIDERS` against the real binary, not the docs table |
| `dde3720` | derive `disabled_agents` from the installed binary, not a hardcoded list (later reverted) |
| `5bd5ade` | pin `OTHER_PROVIDERS`, `check_disabled_agents` both directions |

### 7. Other

`docs/superpowers/plans/2026-09-03-agentsview-o11y.md` checkboxes: still 0
checked, same observation as T+2 — not a live signal from outside.

---

## T+4 — 2026-09-03 09:54 UTC

### 1. Progress

**Effort: ~70 %.** Elapsed 138 minutes since T+0. Projected remaining: 1–2
hours. **Reliability: medium-low this interval** — no code commits landed
(zero, `git log a094baa..HEAD` empty, tree clean), but `ACCEPTANCE.md` was
substantially rewritten with three real acceptance runs (B/A/C) against three
different commits, which is more acceptance work than any prior interval, so
"no commits" here does not mean idle.

### 2. Current state

`ACCEPTANCE.md` (re-read in full, dated 2026-09-03, no new time stamp on the
verdict line but file mtime is within this interval) now reports a **6-row
table** instead of last time's partial one: checks 1, 2, 4 = **FAIL**; check 3
= **NOT EXECUTED**; checks 5, 6 = **PASS, qualified**. Its own top line: **"The
feature does not pass."** Three separate runs (B: default/panel path, exit 5,
demo2-content failure unrelated to o11y; A: `--no-agentsview`, exit 0; C:
default with 18888 pre-bound, exit 0) were run across three different
commits, because teammates fixed two of the found bugs (`fac0e0d`, and the
`09e4b97`/`dde3720`/`5bd5ade` chain from T+3) while runs A and C were still in
flight. The document is explicit that **neither fix has been re-measured
end-to-end** and calls for "a clean re-run against one pinned commit... before
anyone calls this accepted."

### 3. Code problems — fixed / unfixed

No new fixes this interval (no commits). One **self-correction** recorded
inside `ACCEPTANCE.md` itself, worth flagging because it's the kind of thing
this log exists to catch: an earlier version of the document claimed check
2's transcripts were in neither the prefix nor the zone (searched under the
wrong root — the *repository* directory instead of the run root
`~/.local/state/agent-sys-demo`). Re-measured against the correct tree: **9
task-agent transcripts are in `<zone>/<task>/config/projects/`**, confirming
`material.py:89`'s post-prefix overwrite of `CLAUDE_CONFIG_DIR` does win in
the real pipeline, exactly as originally predicted before that wrong
measurement muddied it.

**Noting an apparent inconsistency I have not resolved, not read past:** the
document's own closing "what goes back" list (item 4) says "where demo2's
task-agent transcripts actually land is still unknown. Neither the prefix nor
the zone holds them" — which reads as the *superseded* claim, sitting below
the corrected section that says the opposite. I am reporting both passages as
written rather than picking one; this is the document's own internal
consistency to fix, not mine to referee from outside.

Check 4's acceptance criterion was **rewritten** (old: "`~/.claude/projects`
gained no entry" — fails on any teammate's unrelated Claude Code session in
this shared repo; new: "no new file has a `cwd` inside the run's zone tree,
and none is a session `agent_sys` spawned" — provenance-based, immune to
concurrent teammates, and still correctly FAILs on the one real leak,
`cli/environment.py:332`'s readiness probe, itself already fixed by `fac0e0d`
but not yet re-measured against the rewritten criterion).

### 4. Non-code problems

None new.

### 5. Undetermined questions

**Newly closed, with a caveat:** check 2's mechanism, open at T+2/T+3, is now
called "settled" by the document's corrected measurement (transcripts land in
the zone, not the prefix) — but see §3's inconsistency note; I would not
treat this as fully closed until that self-contradiction in the same
document is resolved.

**New, opened by this acceptance round:** gate 1 of the original design (agent
children's transcripts always land in the prefix) is now reported to **not
hold in the real pipeline** — only the Phase 0 isolated probe and the
readiness probe honour it; real demo2 task agents get `<zone>/config`
instead, because `material.py:89` applies after `prepare.py:480`. The document
frames this as "a design conflict, not a coding slip" and says deciding
between the two writers is "above this report" — i.e. explicitly a decision
for the user/team lead, not something resolved this interval.

### 6. New commits

None this interval.

### 7. Other

Full suite still green: **643 passed, 2 skipped, 2 xfailed in 110.4 s** —
same pass count as T+3's fast run, but back to T+2's slow wall-clock (110 s
vs T+3's 19 s), which lines up with real acceptance-style runs (the ones
behind `ACCEPTANCE.md`) sharing this interval rather than a steady-state
regression; not independently confirmed which tests are slow.

---

## T+5 — 2026-09-03 10:27 UTC

### 1. Progress

**No measurable change this interval — reporting that plainly rather than
estimating a number.** `git log f578599..HEAD` is empty, the tree is clean,
and `find /home/yihou/ws.agentsview_o11y -mmin -70 -type f` (I widened the
window past the normal 35 minutes to be sure) returns the **same file list as
T+4's check**, nothing newer. I have no artefact from this interval at all,
which means I genuinely cannot tell "stuck" from "thinking/discussing
something that hasn't produced output yet" from outside — I am not going to
guess a percentage on top of T+4's 70 %, because there is nothing to move it.

### 2. Current state

Unchanged from T+4: `ACCEPTANCE.md` still says "the feature does not pass";
still calls for a clean re-run against one pinned commit; the design conflict
between the prefix's `CLAUDE_CONFIG_DIR` and the zone's later overwrite in
`material.py:89` is still, in the document's own words, "above this report" —
i.e. a decision for the user/team lead. Plausible reading: this interval's
silence is that decision being made or discussed rather than coded, but this
is a guess, not something I measured, and I'm labelling it as such.

### 3. Code problems — fixed / unfixed

None to report — no commits, no test run diff.

### 4. Non-code problems

None new.

### 5. Undetermined questions

Unchanged from T+4, including the unresolved internal inconsistency in
`ACCEPTANCE.md` flagged there (its own closing list still contradicts its
corrected check-2 section) — I re-checked the file this interval and it has
not been edited since T+4's read, so that inconsistency still stands
unaddressed in the document itself.

### 6. New commits

None this interval.

### 7. Other

None.

---

## T+6 — 2026-09-03 10:57 UTC

### 1. Progress

**Effort: ~78 %.** Elapsed 201 minutes since T+0. Projected remaining:
1–1.5 hours. **Reliability: medium** — the design conflict named as "above
this report" at T+4/T+5 now has a concrete implementation in flight (not yet
committed), which fits my T+5 guess that the silence was decision-making
rather than idleness, but I did not know that at the time and am not
crediting myself for it — it is confirmed only now, by new files.

Since T+5: activity resumed. Working tree has an **uncommitted** change to
`agent_sys/env_mgr/material.py` (+59 lines) plus a new, currently untracked
test file `agent_sys/tests/env_mgr/test_material.py` (171 lines). Full suite
green: **653 passed, 2 skipped, 2 xfailed in 19.2 s** (up from 643 — 10 new
passing tests, consistent with `test_material.py` landing green even though
the source file it tests is still uncommitted).

### 2. Current state

Read `material.py`'s diff directly. The uncommitted change is a resolution to
exactly the "check 2" design conflict `ACCEPTANCE.md` flagged and declined to
resolve itself (T+4/T+5): a new `_share_projects()` function, called from
`deploy()`, that **symlinks `<zone>/config/projects` to the o11y prefix's own
`projects/`** — keeping every other zone-scoped path (credentials, settings,
`sessions/`, `backups/`) exactly as isolated as before, while making the one
subdirectory the panel actually reads (transcripts, Claude Code's own output)
a shared physical location. This reads as an attempt to satisfy both writers
`ACCEPTANCE.md` said were "both [there] for a reason" rather than picking one.

Notable defensive details in the diff, read directly rather than taken on
faith: never raises (a bare `except (OSError, KeyError)` → one `log.warning`,
attempt proceeds); idempotent (an existing symlink pointing at the same
target is left alone); and it refuses to `os.rmdir()` a non-empty directory
that might hold real transcripts rather than force-replacing it — the comment
states the reasoning explicitly ("losing a zone from the panel is cheaper
than deleting somebody's evidence").

A workspace scratch run (`zonelink/demo2.log`) was in flight but not
concluded at the time I checked it — mid-run phase transitions, no
`EXIT=` line yet. Not reporting a result from it.

### 3. Code problems — fixed / unfixed

**In progress, not yet committed:** the check-2 transcript-routing conflict
(zone vs. prefix `CLAUDE_CONFIG_DIR`) — see §2. Cannot call it fixed until it
is committed and re-measured against `ACCEPTANCE.md`'s own check 2.

No other changes this interval.

### 4. Non-code problems

None new.

### 5. Undetermined questions

Whether the symlink approach in §2 actually makes the o11y panel show the
right sessions once check 1 also passes end-to-end — not measured this
interval, only the mechanism read from source.

### 6. New commits

None this interval (work is uncommitted, in progress).

### 7. Other

None.

---

## T+7 — 2026-09-03 11:29 UTC

### 1. Progress

**Effort: ~85 %.** Elapsed 233 minutes since T+0. Projected remaining:
30–60 minutes, **conditional on port 18888 being free** (see §3) — the last
missing acceptance check (a live panel) is now blocked by something mundane
rather than a design question. **Reliability: medium-high** — this interval
has first-hand, direct filesystem evidence for its main claim, not just a
commit message.

Since T+6: `_share_projects()` committed as `84027bd`
("share only `<zone>/config/projects` with the o11y prefix"). Suite still
green: **653 passed, 2 skipped, 2 xfailed in 109.3 s** (same pass count as
T+6, slow wall-clock again — a real run shared the interval, as at T+2/T+4).

### 2. Current state

**Verified the zonelink fix directly against a real run, not from a log
claim.** Latest run root `runs/20260903T104807-76274e`
(`~/.local/state/agent-sys-demo/`): every task zone's `config/projects` is
now a **symlink** to `/home/yihou/.infera_agent_sys/state/claude/projects`
(checked 3 of them with `ls -la`, all identical target). And
`find /home/yihou/.infera_agent_sys/state/claude/projects -name '*.jsonl'
-mmin -45` returns **10** files — matching demo2's 9 declared `kind: ai`
agents plus 1 probe. This is the first time in this effort that a demo2 run's
task-agent transcripts have been directly confirmed landing in the prefix,
first-hand, from the filesystem rather than inferred.

**But the panel still didn't come up in this same run**, for a third, new
reason: `grep -i agentsview zonelink/demo2.log` shows only
`"port 18888 is in use by something else; skipping the o11y panel."` — the
port-busy fail-open path from check 6, not the provider-slug bug from check 1.
Something (unclear what — not investigated, this is a reporter not a
debugger) is holding 18888 on this host right now. So check 1 is **still
unverified end-to-end**, for a third distinct cause across the three runs
this effort has produced (provider slug → then port busy → next attempt
needed with the port actually free).

### 3. Code problems — fixed / unfixed

**Fixed, and now confirmed first-hand:** check 2's transcript routing
(`84027bd`, verified against the real run above).

**Still open:** check 1 (live panel) has never once been demonstrated in this
entire effort, across all commits, for three different reasons in three
different runs. Not a code defect this time — a busy port on a shared host is
exactly the failure mode check 6 is designed to tolerate, and it did (the
run completed). It just means nobody has yet seen the panel come up.

### 4. Non-code problems

Port 18888 is occupied on this host at report time, for reasons not
investigated (could be a leftover from an earlier probe in this same effort,
e.g. `recon/binder.pid`'s test binder, or something unrelated). Whoever
attempts the next full acceptance pass will need to free it first or use
`--agentsview-port`.

### 5. Undetermined questions

Whether the panel, once actually reachable, shows exactly demo2's 9 sessions
and no others (check 3) — still cannot be tested without a live daemon.

### 6. New commits

| commit | what |
|---|---|
| `84027bd` | share only `<zone>/config/projects` with the o11y prefix |

### 7. Other

None.

---

## T+8 — 2026-09-03 12:02 UTC

### 1. Progress

**Effort: ~87 %.** Elapsed 266 minutes since T+0. Projected remaining:
30–60 minutes, unchanged from T+7's window — the blocker moved, not shrank.
**Reliability: medium** — check 2 stands confirmed and no code regressed, but
check 1 has now failed for a **fourth distinct reason** across the runs this
effort has produced, which is a pattern (something about this host or this
binary's startup path is unreliable for `serve --background`) rather than a
string of unrelated one-offs, and I have not seen anyone name that pattern yet.

Since T+7: **no new commits** (`git log 5df203d..HEAD` empty, tree clean).
Suite unchanged: **653 passed, 2 skipped, 2 xfailed in 109.1 s**. A second
full acceptance attempt ("acc2") was run: `logs/acc2_demo2.log`,
`recon/acc2_before.txt` / `acc2_pfx_before.txt` / `acc2_runs_before.txt`,
timestamped `20260903T115423Z`.

### 2. Current state

**Check 1 failed again, differently.** `grep -i agentsview
logs/acc2_demo2.log`: `"agentsview: started but did not answer
http://127.0.0.1:18888 within 30s; skipping the o11y panel."` — this time the
process actually launched (unlike the provider-slug exit-1 at T+2, unlike the
port-busy skip at T+7) but never became ready inside its 30 s window. Curl
against 18888 right now also returns `000`. Across this effort's runs, check 1
has now failed for provider-slug rejection, port-already-bound, **and**
readiness-timeout — three different mechanisms, zero successes. I'm not
diagnosing the timeout myself (out of scope for this role), but naming the
pattern: **no run in this effort has ever seen a live panel**, and the causes
keep being different, which argues against "just retry" as the fix.

The acc2 demo2 run itself again did not finish on its own terms — `EXIT=5`,
`check_solvable: FAIL` — a different demo2 content failure than T+4's run B
(`solutions_c: invalid` there vs. `check_solvable: FAIL`/`problems: invalid`
here), reinforcing that this package's non-determinism, not o11y, is what's
producing non-zero exits.

Check 4 (rewritten criterion): diffed `~/.claude/projects` before/after — 4
new files, all under a **different git checkout entirely**
(`infera.aiopt.real.task_package`), none with a `cwd` in this run's zone tree
and none `agent_sys`-spawned by this run. Under the rewritten check-4
criterion from `ACCEPTANCE.md` (T+4), these are correctly not a leak — first
time this interval's noise has been cleanly excluded by the new wording
rather than requiring a manual read of each file, which is itself a small
confirmation the rewrite works as intended.

### 3. Code problems — fixed / unfixed

No code changes this interval. Check 2's fix (`84027bd`) is not contradicted
by anything seen this run — did not re-verify the symlinks this time to avoid
duplicating T+7's check, but nothing suggests regression.

**Newly named, not yet attributed to any commit:** check 1's readiness
timeout. Whether this is a slow host, a slow `agentsview` daemon warm-up, a
race in `ensure_running`'s probe logic, or something else — not established.

### 4. Non-code problems

Same host-load caveat as always on a shared box: a 30 s readiness window
timing out is exactly the kind of thing that could be host contention rather
than a bug, and I have no measurement that distinguishes those.

### 5. Undetermined questions

New: why `serve --background` doesn't answer within 30 s when it isn't being
rejected by config or blocked by a busy port. Not mine to resolve.

### 6. New commits

None this interval.

### 7. Other

None.

---

## T+9 — 2026-09-03 12:35 UTC

### 1. Progress

**Effort: ~88 %.** Elapsed 299 minutes since T+0. Projected remaining:
30–45 minutes. **Reliability: low-medium** — I see an investigation of T+8's
readiness-timeout in progress (a controlled repro under
`scratch/port_repro/`) but no written verdict yet, so I don't know whether
it's close to resolved or has opened something new.

Since T+8: still **no new commits** (tree clean, `git log d592250..HEAD`
empty). Suite unchanged: **653 passed, 2 skipped, 2 xfailed**, and this run
took **18.6 s** (fast, vs. T+8's 109 s) — no acceptance-scale run competed for
CPU this time, consistent with the acc2 round having finished before this
check.

### 2. Current state

Read directly rather than concluded: `scratch/port_repro/prefixA/state/
agentsview/serve.log` shows a **controlled, isolated** `agentsview serve
--background` starting against a small scratch prefix (2 sessions to sync,
64 directories watched, 120 roots polled) and coming up in **844 ms** —
nowhere near the 30 s timeout seen in the acc2 run. This looks like a
deliberate differential-comparison step (a clean, minimal prefix vs. the real
one) rather than a coincidence, but I have not found a written conclusion
comparing the two, so I am not calling the cause found.

Also new: `logs/acc2_demo2_noav.log` (a `--no-agentsview` companion run to
acc2, mid-flight when checked, no `EXIT=` line yet) and a full second
checkout of the repository under `zonelink/control/` — the latter looks like
a clean-tree reference copy for the same kind of differential comparison,
not something I'm reading further into without a stated conclusion to check
against.

Check 4's rewritten criterion, re-applied to acc2: `acc2_new.txt` lists 5 new
`~/.claude/projects` files, all under `infera.aiopt.real.task_package` (a
different checkout), none in this run's zone — consistent with T+8's read,
now backed by the tool's own diff output rather than my manual cross-check.

### 3. Code problems — fixed / unfixed

None to report — no commits this interval.

### 4. Non-code problems

None new beyond T+8's readiness-timeout, which is now under active
investigation (§2) rather than newly discovered.

### 5. Undetermined questions

Same as T+8. The 844 ms vs. 30 s-timeout contrast is suggestive of "prefix
size / session count affects `serve` startup time" but I have not seen this
stated as the conclusion anywhere, so I'm reporting the two numbers, not the
inference.

### 6. New commits

None this interval.

### 7. Other

None.
