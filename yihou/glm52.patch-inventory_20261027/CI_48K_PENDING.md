# Pending handoff for the next user session

Updated: **2026-11-18T09:16:36.956625+00:00**.

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

## #47276: Attempt 30 failed on wait-for-base-b GitHub HTTP 502, followed by B0 health-check fail-fast; no new functional tests ran. Helper request, durable reservation and matching successful handler are verified; attempt 31 is queued with 22 retained functional passes and no new test steps. Required-main ancestry is eligible. Historical attempt-29 FlashInfer graph mismatch remains: 1/6 elements, difference 43 at (1,2); 10/36 B0 files passed, one failed, 25 unexecuted. Main reproduction, patch causality and source/overlap/numerical qualifications remain unknown.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow queued Base attempt 31 without duplicating it. If terminal unsuccessful, inspect the new error and use the helper after 2026-11-18T10:08:37.563723+00:00, subject to fresh head/account-wide eligibility. CI maintainers should investigate the wait-for-base-b HTTP 502 if it recurs. Attention/runner owners still need output-component/dtype and replay-state evidence for attempt 29; main reproduction and patch causality remain unknown. Preserve optional llr37-disabled role, #45307 overlap coordination and exact-source/numerical qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/31)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6974042372)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44273158464)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/134652581632)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/134652583472)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...3d739b80766dd8145d1e798edb9d202aba091d47)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/29)

## #47278: Current-head AMD audit again confirms 17 functional stages, finish and separate JIT/benchmark passed before this cycle. Historical 33-test MI355X claim still lacks verified exact-source/run archiving and BF16/DCP/LSE/combined-stack qualification. Twelve CPU checks do not qualify the requested current-head numerical/e2e A/B.

Tracking key: `pr47278-amd-validation-source`.

Next action: Owner/kernel reviewers archive matching candidate evidence and arrange approved MI355X numerical/current-head e2e A/B using cycles/20261118T090002Z/PR47278_GPU_VALIDATION.json. Preserve source/environment separation for the later reconciled candidate; CPU routing checks and passing AMD do not resolve these qualifications.

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

## #47278: Exact candidate f5fc9afe versus main 3a41a599192906eee6aace241798136a6db2b9ad with merge base d565013a yields nine kernel-file conflicts. Source anchors and excerpts are saved; no reconciliation was attempted. Main return_lse/bf16_dot/natural-LSE/empty-row/head-mask contracts must coexist with candidate BF16 partials, scaled-KV/group-width, DCP/LSE and actual matching launch/reducers. Passing Base/AMD does not resolve integration or changed-source validation.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Owner/kernel agent reviews cycles/20261118T090002Z/PR47278_CURRENT_MAIN_CONFLICTS.md and supplies a reconciled candidate preserving all interface/layout contracts. Repeat numerical/e2e qualification before publishing performance-sensitive repairs; preserve current passing head until that handoff.

- [Evidence](https://github.com/sgl-project/sglang/commit/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335)
- [Evidence](https://github.com/sgl-project/sglang/blob/3a41a599192906eee6aace241798136a6db2b9ad/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L904)
- [Evidence](https://github.com/sgl-project/sglang/blob/3a41a599192906eee6aace241798136a6db2b9ad/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1155)
- [Evidence](https://github.com/sgl-project/sglang/blob/3a41a599192906eee6aace241798136a6db2b9ad/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1326)

## #49732: Historical/unrefreshed; outside the resumed scope. Base attempt 3 passed all 31 functional jobs; CI gap recovered. Current-head maintainer review and integrated-image qualification remain. Isolated configuration checks and archived 9763ef56 GPU evidence do not establish full module/model coverage for this candidate.

Tracking key: `pr49732-base-review-validation`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: Maintainers review and merge owned continuation preserving yu-depend/#46918 credit. Complete integrated-image qualification and exact-source numerical evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42974439450/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/49732)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)

## #47278: Reviewer requests non-target geometry coverage/removal, unreachable-variant cleanup, seq15/16/128 tests, compiler evidence, warmup/comment cleanup and current-head e2e A/B. Twelve CPU checks passed, including 113 tuned launches and negative controls. Local review-only seq16/128 numerical additions and seq15/128/129 selector assertions were prepared; only CPU assertions/syntax ran. No kernel edit, numerical/e2e validation, issue or review reply was published.

Tracking key: `pr47278-review-20261117`.

Next action: Owner/kernel agent must choose geometry removal or H32/H128/H96 numerical/benchmark coverage, review the local boundary-test draft, remove unreachable variants with matching lowering validation, and reconcile nine exact current-main kernel conflicts. Preserve return_lse/bf16_dot/natural-LSE/empty-row/head-mask contracts, BF16 partials, scaled-KV/GROUP_W, DCP/LSE and matching launch/reducers. Preserve passing f5fc9afe until a reviewed candidate is supplied. Owner must arrange approved MI355X current-head numerical/e2e A/B and exact-source archiving; authorize any review reply or Triton issue separately. Historical combined-stack limits remain.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/blob/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L343)
- [Evidence](https://github.com/sgl-project/sglang/blob/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1122)
- [Evidence](https://github.com/sgl-project/sglang/blob/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1756)
- [Evidence](https://github.com/sgl-project/sglang/blob/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1811)
