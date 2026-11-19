# Pending handoff for the next user session

Updated: **2026-11-19T18:10:29.035802+00:00**.

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

## #47276: Current e95ec5ab Base attempt 3 remains active: 27 functional passes, three runnerless queued B jobs, C not materialized; 12 CPU passes retained. All seven JIT jobs passed, including one retained. No new failure; historical FlashInfer causality, maintainer review, #45307 overlap and exact-source/numerical qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow active Base attempt 3 without duplication; when terminal unsuccessful, freshly verify head/config/attempt and promptly use owner_requested through the helper if eligible. Required-main ancestry passes and the saved hourly interval has elapsed; the active run currently blocks recovery. Retain the passing gate without a new two-hour wait. Preserve maintainer review, #45307 overlap coordination, exact-source/run archival and numerical limits; this optimization is optional and disabled in frozen llr37. Current-head dual-image results remain author-reported. If historical FlashInfer mismatches recur, obtain matching controls before attributing them to main or the patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-7010787687)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44535642771)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...e95ec5abc8b4a9bf27b3dae160e42dc5beeb5f6e)

## #47278: Current cabe5837 Base remains active: 13 functional passes, 15 pre-test health-check failures, two queued B jobs and skipped C placeholders. First JIT error is CUDA allocation before attention; the five newly audited failures cascade from it. AMD has 15 functional passes and two queued jobs. Inherited exact-source MI35x suite passed 45 cases, including 12 boundary references, with one additive main DCP test. Current-head e2e, exact-parent/non-target comparisons, extended layout/replay, compiler reproduction and raw author-evidence mapping remain incomplete.

Tracking key: `pr47278-amd-validation-source`.

Next action: Preserve cabe5837 and active Base/AMD; when Base becomes terminal unsuccessful, recheck config/head/attempt and promptly use the helper if eligible. No deterministic source repair is established for the CUDA allocation error. Retain the inherited 12 passing boundary-reference cases and checked-in scaled-KV/DCP/LSE/workspace coverage. Owner/kernel reviewers still need current-head e2e A/B versus exact parent 8d72d2ea11901b49d694bcea543f1cffccf27629, exact-parent seq15 equality, non-target numerical/timing evidence, extended layout/replay qualification and raw source/run mapping. Prepare the Triton 3.8.0 decoupled-dot/async-copy standalone reproducer and authorized issue/link; audit reported warmup 13.3s/26.9s. Use cycles/20261119T180002Z/PR47278_REVIEW_HANDOFF.md and PR47278_GPU_VALIDATION.json. GPU allocation/access through another route is unauthorized; issue/review posting needs owner authorization. Preserve BF16 partials and historical combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/job/135585592844)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/job/135546036028)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7008488632)
- [Evidence](https://github.com/sgl-project/sglang/blob/cabe5837e956276e5507244e6c5235c1f5f5d707/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L344)
- [Evidence](https://github.com/sgl-project/sglang/blob/cabe5837e956276e5507244e6c5235c1f5f5d707/test/registered/kernels/ops/attention/test_triton_sparse_mla_hip.py#L168)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/job/135551545932)

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

## #47278: No new reviewer response found. Preserved cabe5837/reply; fresh source checks confirm H64 scope, dead-variant removal, parent-identical generic reducer and boundary parameters. Inherited MI35x CI passed all 12 boundary references but does not prove exact-parent seq15 equality. Current-head e2e A/B, independent non-target numerical/timing evidence, standalone Triton 3.8.0 reproducer/authorized issue link, and audited warmup/raw source mapping remain. LEAN_EPILOGUE nonreproduction is author-reported. Local drafts remain unpublished; no worker GPU execution.

Tracking key: `pr47278-review-20261117`.

Next action: Preserve cabe5837 and active Base/AMD; when Base becomes terminal unsuccessful, recheck config/head/attempt and promptly use the helper if eligible. No deterministic source repair is established for the CUDA allocation error. Retain the inherited 12 passing boundary-reference cases and checked-in scaled-KV/DCP/LSE/workspace coverage. Owner/kernel reviewers still need current-head e2e A/B versus exact parent 8d72d2ea11901b49d694bcea543f1cffccf27629, exact-parent seq15 equality, non-target numerical/timing evidence, extended layout/replay qualification and raw source/run mapping. Prepare the Triton 3.8.0 decoupled-dot/async-copy standalone reproducer and authorized issue/link; audit reported warmup 13.3s/26.9s. Use cycles/20261119T180002Z/PR47278_REVIEW_HANDOFF.md and PR47278_GPU_VALIDATION.json. GPU allocation/access through another route is unauthorized; issue/review posting needs owner authorization. Preserve BF16 partials and historical combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7008488632)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/job/135585592844)
- [Evidence](https://github.com/sgl-project/sglang/blob/cabe5837e956276e5507244e6c5235c1f5f5d707/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
