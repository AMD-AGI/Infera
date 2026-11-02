# Daily PR handoff maintenance

The public entry point is [PR_HANDOFF.md](../PR_HANDOFF.md). It contains all
authored PRs in repositories GitHub explicitly reports as public, plus a separate
external-dependency list. [OWNED_PR_INDEX.md](../OWNED_PR_INDEX.md) is the narrower
four-repository inventory. [PR_STATUS.md](../PR_STATUS.md) has a generated current
section followed by dated maintenance history.

The main-based team documentation lives in
[`docs/glm52-pr-tracking/`](https://github.com/AMD-AGI/Infera/tree/dev/yihou/glm52-pr-tracking/docs/glm52-pr-tracking).
Its README starts with 48k deployment priorities and separates Ready and Draft
queues. `export_team_tracker.py` exports only public PR records and pins pack-up
references to the archive commit; main does not need to carry the experiment
branch's files. The local `team_tracker` config supplies its dedicated worktree,
branch, directory and open PR URL.

The account-wide handoff, including restricted repositories, is generated at
`runtime/daily-maintenance/PR_HANDOFF_ALL.md` and is intentionally ignored by Git.
Share it only with authorized colleagues. The public files contain no restricted
repository names/URLs. The internal ledger is self-contained and uses public web
links for the reproduction documents, so it can be copied to an authorized
handoff location without its original directory tree.

## What is maintained daily

At **09:00 UTC / 17:00 Asia/Shanghai**, the installed user cron entry runs
`run_daily.py` on the configured maintenance host. It:

1. Discovers every PR authored by `dorado269`, including newly opened PRs; checks
   search completeness and repository visibility.
2. Reads current PR state, exact head/base, review decision, requested reviewers,
   assignees, all paginated inline review threads and GitHub's check rollup.
   Refreshes external dependencies and the two issue-only reports as well.
3. Preserves curated purpose, priority, evidence, owner and Ready/merge actions
   from `handoff_notes.json`; flags changed heads and newly discovered PRs for
   triage. Old tests do not automatically qualify a changed head.
4. Updates the full internal Markdown and public tables/snapshot. The frozen
   reproduction snapshot and source checksums remain historical pins.
5. Pulls with rebase, commits the four generated public files with `git commit -s`
   under the configured yihou identity, rebases concurrent remote updates and
   pushes normally to `dev/pd_opt/glm_5.2_agentx`.
6. If a team tracking PR is configured and open, exports the same audit into its
   five Markdown files, checks them, signs a documentation-only commit and pushes
   that owned branch. It checks the PR lifecycle and branch identity before
   publication. A dirty worktree is preserved for manual reconciliation. Per
   owner instruction, **Infera #228 is permanently Draft, for view and track
   only: never promote or merge it**. With `permanent_draft: true`, the updater
   checks and restores its Draft flag before collection and publication.

If the tracker is closed, the team publisher reports `TRACKER_CLOSED` and
preserves its branch/artifacts for the owner to decide how to resume. The
underlying account collector continues daily. The tracker is not intended to
merge, and the scheduler never pushes directly to main. `run_status.json`
records this handoff state explicitly.

Publication uses an isolated worktree below this inventory's ignored `worktrees/`
directory. It never checks out branches, stages files or stashes changes in the
experiment workspace. That workspace's local tracked copy is a snapshot until
its next manual pull; the GitHub branch link and runtime internal ledger are the
daily-updated entry points. No force push or deletion is performed.

The timer **refreshes the ledger**, not an autonomous code-review service. It
does not mutate runtime PRs, post comments, mark Ready, merge, allocate GPUs or certify
new evidence. A human/agent delegate must assess new reviews/CI failures and
update the curated actions. No external messaging is configured. The task depends
on this host being online and its GitHub/SSH credentials remaining valid.

## Handoff workflow

Use the priority table to assign a named delegate. Edit the PR's `owner` field in
`handoff_notes.json` (or the ignored `runtime/daily-maintenance/private_notes.json`
for restricted work). Record what was actually done, exact tested head, commands,
run IDs, evidence limits and next actor. Refresh after every substantive change,
in addition to the daily timer. A new `TRIAGE` row means no curated review exists.

For each active PR, distinguish:

- **To Ready:** remaining implementation/evidence/scope work owned by us. Reuse
  matching archived hardware runs; request a pool node only for concrete gaps.
- **To merge:** maintainer review, executed required CI, dependency order and
  rebases. Missing `run-ci` or a draft gate is not a failed model regression.
- **Already Ready:** do not invent a new promotion gate. Resolve actual current
  comments and checks; Ready does not mean approved or safe to merge.

Do not modify other people's PR branches. Keep S13 attribution agreement,
issue-only workarounds and excluded patches as recorded. Never use the live
experiment nodes crsuse2-m2m-135/136 for validation.

## Operation and recovery

Machine-specific config and credentials are not committed. The installed config
is `runtime/daily-maintenance/config.json`; all cache/temp/log paths are under the
same NFS-backed yihou task directory. The runtime config supplies the dedicated
worktree, Python/pre-commit paths, target branch and checked commit identity.

```sh
# From the experiment repository root; output/logs stay inside the task tree.
python yihou/glm52.patch-inventory_20261027/maintenance/run_daily.py \
  --config yihou/glm52.patch-inventory_20261027/runtime/daily-maintenance/config.json
crontab -l
cat yihou/glm52.patch-inventory_20261027/runtime/daily-maintenance/run_status.json
cat yihou/glm52.patch-inventory_20261027/runtime/daily-maintenance/last_publication.json
```

`run_status.json` records RUNNING/SUCCESS/FAILED and the exact log path;
`last_publication.json` preserves the last successful push. API failures, a dirty
publisher checkout, changed identity or rebase conflicts stop publication and
preserve files for inspection. A lock prevents overlapping runs. The last
successful refresh is printed in every table; **older than 36 hours is stale**.
Cron misses are not replayed automatically; run the command above after recovery.

The initial end-to-end refresh and signed push were verified on November 2,
including authentication without interactive token or SSH-agent environment
variables. The local `scheduler.json` records the installed cron schedule and
verified publication. On Git 2.34, initialize cone mode with
`git sparse-checkout init --cone` before `git sparse-checkout set` when creating
the dedicated checkout; ensure root lint/config files are present before running.

Changing schedule: edit only the cron block named `yihou-pr-ledger-daily`, keeping
other entries. Moving hosts: disable that block on the former host, create a
dedicated clean worktree on suitable NFS/data storage, update local config and
credentials, run one refresh/publish, then install the new host's timer. Cron and
local runtime files do not travel through Git. Back up/transfer private notes
only through an authorized internal channel.

## Validation

The updater uses Python's standard library and authenticated `gh`. Its regression
tests cover visibility filtering, missing visibility, changed-head evidence,
new PR triage, null check rollups and draft/merged/closed distinctions. Run in
this directory with bytecode/cache paths under the checked yihou runtime area:

```sh
python -B -m unittest discover -s yihou/glm52.patch-inventory_20261027/maintenance -p 'test_*.py'
```

Public generation is also checked against every restricted repository identifier
before publication. The repo's applicable pre-commit hooks run on generated
files before each signed commit. The generator never claims that a failed or
skipped CI gate means its downstream functional tests ran.
