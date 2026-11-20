# Pending handoff for the next user session

Updated: **2026-11-20T06:08:50.719865+00:00**.

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

## #47276: Current e95ec5ab Base attempt 3 remains fully passed: 35 functional jobs (12 retained/23 rerun) and seven JIT jobs (one retained). All execution predates this cycle. CI recovered; maintainer review, #45307 overlap, final source/run archival and independent numerical qualifications remain under this mixed key. Success does not establish historical FlashInfer mismatch causality. Optional and disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Maintainers review passing e95ec5ab and coordinate #45307 overlap. Preserve current-head Base success; no freshness rerun/rebase. Archive exact source/run mapping and retain independent numerical qualifications. Optional and disabled in frozen llr37. Author-reported dual-image evidence remains source-specific; obtain matching controls before assigning historical FlashInfer mismatches to main or this patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-7010787687)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44535642771)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...e95ec5abc8b4a9bf27b3dae160e42dc5beeb5f6e)

## #47278: Current cabe5837 Base attempt 2 advanced to 34 functional passes (13 retained/21 rerun) and seven JIT passes; one B300 C job remains runnerless without test steps, with no finish. AMD has 16 functional passes/one MI35x disaggregation queue. Prior request/handler verified; no new upstream failure or functional/JIT test execution this cycle. Inherited 45-case/12-boundary numerical evidence does not cover requested current-head e2e A/B or all exact-parent/layout/replay qualifications.

Tracking key: `pr47278-amd-validation-source`.

Next action: Preserve cabe5837 and active Base attempt 2/AMD attempt 1; follow the runnerless Base B300 job and AMD MI35x disaggregation queue without duplicate retries. On terminal unsuccessful Base, inspect the first new error and promptly use the helper after fresh config/head/attempt/cooldown checks. Owner/kernel reviewers provide current-head e2e A/B against exact parent 8d72d2ea11901b49d694bcea543f1cffccf27629 and raw run/source mapping; retain exact-parent seq15/non-target, layout/replay, BF16-partial and DCP/LSE qualifications. See cycles/20261120T060002Z/PR47278_REVIEW_HANDOFF.json and PR47278_GPU_VALIDATION.json. Owner must arrange approved GPU allocation; no machine-pool skill or alternate GPU route is available here. Compiler follow-up is reviewer-nonblocking; issue/reply requires owner authorization. Warmup 13.3s/26.9s is author-reported; historical whole-stack throughput is separate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7022138354)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44614750603)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/job/135546036028)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908317/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7008488632)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/job/135585592844)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/job/136045091836)

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

## #47278: No changed comments/reviews. Exact-source/static checks confirm H64 routing scope, dead-variant removal and parent-identical generic reducer/launch; previous CPU boundary/page-offset and MI35x reference passes are inherited. Current-head e2e A/B, raw run/source mapping and exact-parent/non-target/layout/replay qualifications remain. Compiler follow-up is reviewer-nonblocking: source and author report corruption on Triton 3.8.0, exact output on 3.7.0; LEAN_EPILOGUE nonreproduction and warmup 13.3s/26.9s remain author-reported. Local handoff/drafts updated without posting.

Tracking key: `pr47278-review-20261117`.

Next action: Preserve cabe5837 and active Base attempt 2/AMD attempt 1; follow the runnerless Base B300 job and AMD MI35x disaggregation queue without duplicate retries. On terminal unsuccessful Base, inspect the first new error and promptly use the helper after fresh config/head/attempt/cooldown checks. Owner/kernel reviewers provide current-head e2e A/B against exact parent 8d72d2ea11901b49d694bcea543f1cffccf27629 and raw run/source mapping; retain exact-parent seq15/non-target, layout/replay, BF16-partial and DCP/LSE qualifications. See cycles/20261120T060002Z/PR47278_REVIEW_HANDOFF.json and PR47278_GPU_VALIDATION.json. Owner must arrange approved GPU allocation; no machine-pool skill or alternate GPU route is available here. Compiler follow-up is reviewer-nonblocking; issue/reply requires owner authorization. Warmup 13.3s/26.9s is author-reported; historical whole-stack throughput is separate.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7008488632)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/job/135585592844)
- [Evidence](https://github.com/sgl-project/sglang/blob/cabe5837e956276e5507244e6c5235c1f5f5d707/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
