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

## Two-hour autonomous Base CI maintenance

The separate `yihou-glm52-ci-watch` cron block runs `run_ci_watch.py` at
`0 */2 * * *` in UTC. It monitors exactly SGLang #33059, #46820, #46830, #45524,
#46918, #47276, #47278, #47897 and #47872. The last two externally authored
fixes in that list are #45524/#46918; their branches remain read-only. #47276
is optional for the frozen llr37 deployment, but is explicitly included.

This worker is authorized to inspect current-head Base and relevant AMD logs,
request justified bounded retries, and make small, attributable, validated fixes
to owned PR branches. It preserves existing review and evidence constraints.
It cannot merge, close, open, or change Ready/Draft state. Infera #228 remains
permanently Draft. The daily 09:00 collector remains installed separately.

The worker uses a dedicated sparse worktree and the saved Codex/GitHub identity.
Its sparse paths must include the repository `scripts/` directory so the
required `check-author-email.sh` pre-commit hook remains available. Never
skip a required hook to compensate for an incomplete checkout.
It does not run in, stash, clean, or stage the experiment checkout or its
uncommitted pack-ups. Artifacts and logs use the verified NFS task directory.
`codex exec` supports this scripted execution and structured result format:
[official non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode).
No API credentials are copied into configuration or published files.

The launcher has its own nonblocking lock and a 90-minute agent deadline. An
active previous cycle prevents overlap. The retry helper additionally reserves
actions before sending, checks current head/attempt and existing CI labels,
enforces at least two hours between requests and at most two automated retries
per unchanged head, and never retries the same attempt twice. Queued/running
jobs are not retried. Unknown outcomes retain their reservation for inspection.
An expired upstream author cooldown uses the `stale_gate` path, with fresh
policy and account-wide Base-run checks before sending. The logged deadline
is conditional on no newer counted run. Such retries are sequential and never
use the `transient` path to avoid a cooldown refusal.
Actual deterministic failures require diagnosis; repeated/unclear failures are
handed off instead of repeatedly retried. Bot retries are PR-wide, as described
below. Source fixes use isolated PR checkouts, focused validation and signed
commits, with a fresh remote-head check before pushing.

The current reports are [CI_48K_WATCH.md](../CI_48K_WATCH.md) and
[CI_48K_PENDING.md](../CI_48K_PENDING.md). The launcher updates only the scoped
curated CI actions, retaining ownership, evidence, priority and readiness notes,
then invokes the existing daily publisher to refresh the permanent-Draft tracker.
A failed publication or agent run is recorded locally and is not called success.

Machine configuration is ignored by Git at `runtime/ci-watch-20261103/config.json`.
Logs, immutable cycle results and retry history live beside it. A stable
`NEXT_SESSION.md` and `next_session.json` retain unresolved handoff items.
The user's scoped Codex AGENTS instruction reads that notice at the next
interactive project session. Acknowledging a displayed notice does not resolve
or remove its open findings. No email, Slack or other notification is sent.
The host, NFS, network and saved credentials must remain available.

```sh
python maintenance/run_ci_watch.py --config /path/containing/yihou/config.json
# Inspect status before interpreting an old report as current.
cat runtime/ci-watch-20261103/run_status.json
cat runtime/ci-watch-20261103/NEXT_SESSION.md
# Only after the pending findings have actually been reported to the user:
python maintenance/run_ci_watch.py --config /path/containing/yihou/config.json --acknowledge
```

To pause this task, set `enabled` to false in its local configuration. Preserve
unrelated cron entries and the daily ledger task. Before moving hosts, disable
the old host's watcher to avoid duplicate actions; transfer configuration and
retry history without sharing credentials or deleting preserved artifacts.

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

## Retrying CI as the PR author

Verified November 3 on SGLang #47276, unchanged head `74c576b7`:

- Direct `gh run rerun 41467493468 --repo sgl-project/sglang --failed` was
  rejected with `Must have admin rights to Repository` for the current account.
  The `run-ci` label does not grant repository Actions write permission.
- The repository's supported `/rerun-failed-ci` PR comment grants authors the
  handler's rerun capability. Our [request](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6597067522)
  produced a successful [handler run](https://github.com/sgl-project/sglang/actions/runs/41590296733)
  and [Base attempt 2](https://github.com/sgl-project/sglang/actions/runs/41467493468/attempts/2).
  Both previously cancelled H100 shards were requeued; passing results remained.
- This command operates on **all unsuccessful current-head workflows**, not only
  Base. This request retried Base, Extra, AMD Extra and MLU failed jobs; the skipped
  Mori AMD workflow received a full rerun. Do not describe it as a two-job-only API.
- At the audited handler revision, `/rerun-base-ci` without arguments aliases a
  broad command. A tier argument additionally requires CI-label permission even
  if the label exists. Use a maintainer's Actions access when strict Base-only
  scope is required. Recheck the current handler before relying on these details.
- Before posting, verify the unchanged PR head, existing CI labels and latest
  attempt: if another actor has already started the retry, do not duplicate it.
  A successful handler or requeued job does not mean the tests have passed.

Source: [official handler at cbfc0932](https://github.com/sgl-project/sglang/blob/cbfc0932e3be191e8bbb0873cac35c8ceb65313b/scripts/ci/utils/slash_command_handler.py).

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
