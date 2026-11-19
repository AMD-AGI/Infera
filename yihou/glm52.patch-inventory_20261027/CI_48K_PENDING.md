# Pending handoff for the next user session

Updated: **2026-11-19T03:12:47.886755+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #45524: Historical/unrefreshed; outside the resumed scope. Base passed 27 functional jobs. Same AMD shard 6 exited 255 after starting tests; historical log retrieval returned HTTP 404. First exception, source equivalence and historical AWQ/PP review remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: External author eaglewangao-us/maintainers retrieve runner logs, recover coverage and complete source-equivalence review; preserve frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Historical/unrefreshed; outside the resumed scope. Base has 36 functional successes and two shards cancelled without tests. One annotation identifies github-actions[bot]; finish log confirms cancellation caused the aggregate failure. Cancellation policy and original-PR integration remain unresolved. Scoped validation at 9763ef56 does not qualify current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: External author yu-depend/maintainers own original-PR conflict, remaining coverage and review decisions, including the unexplained cancellation policy. Advance owned continuation #49732 separately; preserve frozen workaround and source-specific MI355X evidence at 9763ef56.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Attempt36 repeats FlashInfer MLA EAGLE draft graph mismatch: 1/6 elements, abs 43 at (1,2) ; 10/36 files passed, one failed, 25 unexecuted. Exact tested-merge test/runner blobs match PR head. Verified sent reservation/comment and successful handler; attempt 37 queues B0 with no runner/steps and retains 22 functional passes. Required-main ancestry is eligible. Main reproduction, output component/dtype, patch causality and source/overlap/numerical qualifications remain unknown.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow queued attempt 37 without duplication. If terminal unsuccessful, inspect its first concrete error and use the helper no earlier than 2026-11-19T04:02:38.962693+00:00, subject to fresh config/head/account-wide eligibility. Attention/runner owners need output-component/dtype, replay-state evidence and exact-main controls; no matching repair or patch causality is established. Maintainer review, #45307 overlap coordination and exact-source/numerical qualifications remain. Preserve the optional llr37-disabled role.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/37)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6998376699)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44448615037)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/135233161717)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...3d739b80766dd8145d1e798edb9d202aba091d47)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Fresh audit confirms current-head AMD attempt 1 passed 17 functional stages before this cycle. Historical MI355X evidence still lacks verified exact-source/run archiving and BF16/DCP/LSE/combined-stack qualification. Inherited CPU evidence and new artifact-only checks do not establish numerical/current-head e2e A/B.

Tracking key: `pr47278-amd-validation-source`.

Next action: Owner/kernel reviewers archive exact source/run evidence and arrange approved MI355X current-head numerical/e2e A/B using cycles/20261119T030001Z/PR47278_GPU_VALIDATION.json. Qualify later reconciliation separately.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42947083003/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/commit/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335)

## #47897: Historical/unrefreshed; outside the resumed scope. Base passed 27 functional jobs. Matching inherited logs identify AMD MIOpen SQLite database-open error and PP KVTransferError; Xeon package/checkout mismatch remains. ARM64 passed. Five later MIOpen files, two PP files and 16/23 Xeon files were unexecuted.

Tracking key: `pr47897-architecture-api-ci`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: CI maintainers repair database access, investigate PP peers and align installed/test revisions; preserve passing Base and numerical qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Historical/unrefreshed; outside the resumed scope. Run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K does not establish long-context DSA accuracy.

Tracking key: `pr47897-source-long-context`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: Archive exact evidence and complete unequal paired-row/table-length, dispatch/fallback and long-context review.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Historical/unrefreshed; outside the resumed scope. OWNER PAUSED pending redesign. Historical passing CI and source/numerical qualifications remain archived and were not queried.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: Owner-paused; resume only on owner instruction. Preserve opt-in/fallback/tie, measured 32k regression and exact-source/long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204156/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Historical/unrefreshed; outside the resumed scope. Matching Xeon failure cannot import _paged_experts_num_resident from the installed package. Checkout/package alignment remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Candidate f5fc9afe versus main 5a0e808c59b82eb7d435bcef63a7cf9c1b4bc2c9 and merge base d565013acdcb4590d6b88ffb7557a6f83f3a4793 still produces nine kernel-file conflicts; GitHub reports CONFLICTING. Kernel/test blobs are unchanged. Preserve return_lse/bf16_dot/natural-LSE/empty-row/head-mask with scaled-KV/GROUP_W, BF16 partials, DCP/LSE and matching launches/reducers, including live external helper/default contracts. No reconciliation or changed-source qualification occurred.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Owner/kernel reviewer uses cycles/20261119T030001Z/PR47278_CURRENT_MAIN_CONFLICTS.md and PR47278_REVIEW_ANALYSIS.md to supply a reviewed reconciliation. Qualify numerical/e2e behavior before publication; preserve current passing head.

- [Evidence](https://github.com/sgl-project/sglang/commit/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a0e808c59b82eb7d435bcef63a7cf9c1b4bc2c9/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1326)
- [Evidence](https://github.com/sgl-project/sglang/blob/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_decode.py#L252)

## #49732: Historical/unrefreshed; outside the resumed scope. Base attempt 3 passed all 31 functional jobs; CI gap recovered. Current-head maintainer review and integrated-image qualification remain. Isolated configuration checks and archived 9763ef56 GPU evidence do not establish full module/model coverage for this candidate.

Tracking key: `pr49732-base-review-validation`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: Maintainers review and merge owned continuation preserving yu-depend/#46918 credit. Complete integrated-image qualification and exact-source numerical evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42974439450/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/49732)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)

## #47278: Reviewer still requests non-target geometry coverage/removal, unreachable-variant cleanup, seq15/16/128 coverage, compiler evidence, warmup/comment cleanup and current-head e2e A/B; no new feedback. Refreshed inherited drafts and two-phase GPU handoff. Boundary patch prepares 10 numerical cases; seq15 exact-baseline runner remains unexecuted beyond syntax/stdlib plan. Prior 14 CPU checks, 113 mocked launches and five contract checks are inherited. Live external decode helpers/defaults must remain. No production edit, GPU access, issue or reply.

Tracking key: `pr47278-review-20261117`.

Next action: Owner/kernel reviewer must choose geometry removal or H32/H128/H96 numerical and benchmark coverage; remove only unreachable variants while preserving live external decode helpers/defaults. Use cycles/20261119T030001Z/PR47278_SEQ15_AB_DRAFT.md for prepared seq15 exact-baseline A/B and execute seq16/128 reference tests. Reconcile nine main conflicts preserving return_lse/bf16_dot/natural-LSE/empty-row/head-mask, scaled-KV/GROUP_W, BF16 partials, DCP/LSE and matching launches/reducers. Preserve passing f5fc9afe until a reviewed candidate is supplied. Owner must arrange approved MI355X current-head numerical/e2e A/B, compiler evidence, warmup measurements and exact-source/run archiving using cycles/20261119T030001Z/PR47278_GPU_VALIDATION.json; qualify later reconciliation separately. Issue/review publication requires owner authorization. Preserve historical combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/blob/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L343)
- [Evidence](https://github.com/sgl-project/sglang/blob/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_decode.py#L252)
- [Evidence](https://github.com/sgl-project/sglang/blob/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_decode.py#L282)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a0e808c59b82eb7d435bcef63a7cf9c1b4bc2c9/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1326)
