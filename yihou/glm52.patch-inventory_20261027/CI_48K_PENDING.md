# Pending handoff for the next user session

Updated: **2026-11-18T07:55:48.761026+00:00**.

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

## #47276: Attempt 29 repeated the same FlashInfer MLA graph mismatch: 1/6 elements, difference 43 at (1,2); 10/36 B0 files passed, one failed, 25 unexecuted. Existing helper request/handler verified; attempt 30 B0 is queued with 22 retained functional passes and no new tests. Required-main ancestry is eligible. Current-main harness differs; no matching merged correction, baseline reproduction or patch causality established. Source/overlap/numerical qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow queued Base attempt 30 without duplicate requests. If terminal unsuccessful, inspect its first error and use the helper no earlier than 2026-11-18T08:38:31Z, subject to fresh author-wide eligibility. Attention/runner owners must identify the mismatching output component, dtype and graph-replay state; main reproduction and patch causality remain unknown. Preserve optional llr37-disabled role, #45307 overlap coordination and exact-source/numerical qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/30)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/134652583472)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6972253608)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44260477081)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/131659136748)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...3d739b80766dd8145d1e798edb9d202aba091d47)
- [Evidence](https://github.com/sgl-project/sglang/blob/4dfbb953eb55c56e5c592cebce1a1db0d04fe6bc/python/sglang/test/kits/attention_unittest/runner_modes/speculative_draft_runner.py)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Fresh current-head AMD audit confirms 17 functional stages, finish and separate JIT/benchmark passed before this cycle. Historical 33-test MI355X claim still lacks verified exact-source/run archiving and BF16/DCP/LSE/combined-stack qualification. Current-head e2e A/B requested by reviewer remains unexecuted here.

Tracking key: `pr47278-amd-validation-source`.

Next action: Owner/kernel reviewers archive matching candidate evidence and arrange approved MI355X numerical/current-head e2e A/B validation using cycles/20261118T073828Z/PR47278_GPU_VALIDATION.json; CPU routing checks and passing AMD do not resolve these qualifications.

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

## #47278: Supplied d5fb0d33/f5fc9afe reconciliation remains preserved, but fresh GitHub state is mergeable=false/dirty against main. Main 4dfbb953eb55c56e5c592cebce1a1db0d04fe6bc adds return_lse/bf16_dot, natural-log empty-row LSE and non-power-of-two-head fallback contracts. Exact current conflict hunks were not constructed. Scaled-KV/group-width/launch/reducer and source/run qualifications remain.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Owner/kernel agent must reconcile latest main with exact source anchors before publishing a changed candidate, preserving BF16 partials, DCP/LSE, scaled-KV/GROUP_W, cache/async-copy and matching reducers. Repeat numerical/e2e validation; preserve passing head until reviewed reconciliation is supplied.

- [Evidence](https://github.com/sgl-project/sglang/commit/d5fb0d33dd664193bee70c21025f185766e7f492)
- [Evidence](https://github.com/sgl-project/sglang/commit/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335)
- [Evidence](https://github.com/sgl-project/sglang/blob/4dfbb953eb55c56e5c592cebce1a1db0d04fe6bc/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1326)
- [Evidence](https://github.com/sgl-project/sglang/blob/4dfbb953eb55c56e5c592cebce1a1db0d04fe6bc/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L841)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #49732: Historical/unrefreshed; outside the resumed scope. Base attempt 3 passed all 31 functional jobs; CI gap recovered. Current-head maintainer review and integrated-image qualification remain. Isolated configuration checks and archived 9763ef56 GPU evidence do not establish full module/model coverage for this candidate.

Tracking key: `pr49732-base-review-validation`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: Maintainers review and merge owned continuation preserving yu-depend/#46918 credit. Complete integrated-image qualification and exact-source numerical evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42974439450/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/49732)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)

## #47278: Reviewer requests non-target geometry coverage or removal, unreachable-variant cleanup, seq=15/16/128 tests, compiler-issue evidence, warmup/comment cleanup and current-head e2e A/B. Seven CPU checks confirm H32/H128 and additional H96 scope, disabled ROTATE_IDX and its guarded dormant unmasked load, and DQ reduction bypassing generic knobs. No kernel edit, GPU validation, issue or review reply was published.

Tracking key: `pr47278-review-20261117`.

Next action: Owner/kernel agent must address the November 17 review: choose geometry removal or H=32/H=128 coverage (CPU also identifies H=96); remove unreachable variants carefully, add seq=15/16/128 coverage, and document compiler/warmup constraints. Reconcile current-main conflicts while preserving new return_lse/bf16_dot/natural-LSE contracts, BF16 partials, scaled-KV/GROUP_W, DCP/LSE and matching launch/reducers. Preserve passing f5fc9afe until a reviewed candidate is supplied. Owner must arrange approved MI355X current-head e2e A/B and numerical validation, archive exact sources/runs, and authorize any review reply or Triton issue. Existing combined-stack limits remain.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/blob/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L343)
- [Evidence](https://github.com/sgl-project/sglang/blob/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1122)
- [Evidence](https://github.com/sgl-project/sglang/blob/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1756)
- [Evidence](https://github.com/sgl-project/sglang/blob/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1811)
