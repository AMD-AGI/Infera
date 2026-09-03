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
