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
