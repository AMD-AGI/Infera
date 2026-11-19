# Pending handoff for the next user session

Updated: **2026-11-19T21:13:06.600436+00:00**.

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

## #47276: Current e95ec5ab Base attempt 3 remains active: 32 functional passes, including 12 retained, and three runnerless queues (B1/C2). All seven JIT jobs passed, including one retained. No new failure; maintainer review, #45307 overlap, historical FlashInfer causality and exact-source/numerical limits remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Preserve active Base attempt 3 and queue position; no duplicate retry. On terminal unsuccessful Base, freshly verify config/head/attempt and promptly use the helper if eligible. Required-main ancestry passes; this cycle's author-wide reservation next allows a request at 22:02:31 UTC, subject to fresh checks. Keep the passing gate; no new two-hour gate wait applies. Maintainers still review and coordinate #45307 overlap; archive exact source/runs and retain numerical limits. Optional and disabled in frozen llr37. Current-head dual-image results remain author-reported; if FlashInfer mismatches recur, obtain matching controls before assigning causality.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-7010787687)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44535642771)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...e95ec5abc8b4a9bf27b3dae160e42dc5beeb5f6e)

## #47278: Current cabe5837 Base attempt 1 failed after CUDA allocation before attention and 17 pre-test health-check failures. Helper recovery/handler verified; attempt 2 has 13 retained functional passes, three real functional tests running and 14 queued, with no new failure. AMD remains 15 passes/two queues. Inherited 45-case/12-boundary reference passes do not cover current-head e2e or all exact-parent/layout/replay controls.

Tracking key: `pr47278-amd-validation-source`.

Next action: Preserve cabe5837, active Base attempt 2 and AMD attempt 1; follow real test outcomes without duplication. On terminal unsuccessful Base, inspect the first new failure and use the helper after fresh eligibility checks; the current author-wide reservation next permits a request at 22:02:31 UTC, subject to later reservations. No deterministic source repair is established. Owner/kernel reviewers still need current-head e2e A/B against exact parent 8d72d2ea11901b49d694bcea543f1cffccf27629, exact-parent seq15/non-target evidence, extended layout/replay qualification and raw source/run mapping. Reuse the inherited 12 boundary-reference passes within scope. Compiler reproducer/authorized Triton issue and warmup 13.3s/26.9s audit remain follow-ups; compiler issue was reviewer-nonblocking. Use cycles/20261119T210002Z/PR47278_REVIEW_HANDOFF.md and PR47278_GPU_VALIDATION.json. Owner must use an approved allocation workflow; no GPU route or issue/review posting is authorized here. Preserve BF16 partials and historical combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7022138354)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44614750603)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/job/135546036028)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908317/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7008488632)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/job/135585592844)

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

## #49732: Historical/unrefreshed; outside the resumed scope. Base attempt 3 passed all 31 functional jobs; CI gap recovered. Current-head maintainer review and integrated-image qualification remain. Isolated configuration checks and archived 9763ef56 GPU evidence do not establish full module/model coverage for this candidate.

Tracking key: `pr49732-base-review-validation`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: Maintainers review and merge owned continuation preserving yu-depend/#46918 credit. Complete integrated-image qualification and exact-source numerical evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42974439450/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/49732)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)

## #47278: No new reviewer response. Current cabe5837 host gate/dead-variant/reducer checks and existing pure boundary/page-offset regressions pass. Inherited MI35x CI covers 12 boundary references. Current-head e2e A/B, exact-parent seq15/non-target numerical/timing evidence, raw run/source mapping and warmup audit remain. Standalone Triton 3.8.0 reproducer/authorized issue remains reviewer-nonblocking follow-up; LEAN_EPILOGUE nonreproduction is author-reported. Local drafts remain unpublished; no worker GPU execution.

Tracking key: `pr47278-review-20261117`.

Next action: Preserve cabe5837, active Base attempt 2 and AMD attempt 1; follow real test outcomes without duplication. On terminal unsuccessful Base, inspect the first new failure and use the helper after fresh eligibility checks; the current author-wide reservation next permits a request at 22:02:31 UTC, subject to later reservations. No deterministic source repair is established. Owner/kernel reviewers still need current-head e2e A/B against exact parent 8d72d2ea11901b49d694bcea543f1cffccf27629, exact-parent seq15/non-target evidence, extended layout/replay qualification and raw source/run mapping. Reuse the inherited 12 boundary-reference passes within scope. Compiler reproducer/authorized Triton issue and warmup 13.3s/26.9s audit remain follow-ups; compiler issue was reviewer-nonblocking. Use cycles/20261119T210002Z/PR47278_REVIEW_HANDOFF.md and PR47278_GPU_VALIDATION.json. Owner must use an approved allocation workflow; no GPU route or issue/review posting is authorized here. Preserve BF16 partials and historical combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7008488632)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/job/135585592844)
- [Evidence](https://github.com/sgl-project/sglang/blob/cabe5837e956276e5507244e6c5235c1f5f5d707/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
