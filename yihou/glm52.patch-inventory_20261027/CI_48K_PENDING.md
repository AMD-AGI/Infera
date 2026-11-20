# Pending handoff for the next user session

Updated: **2026-11-20T18:15:31.987143+00:00**.

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

## #47276: Current e95ec5ab Base attempt 3 remains fully passed: 35 functional jobs and seven separate JIT jobs; 12 functional/one JIT retained by inherited matching-step accounting. No new test execution or review activity. Maintainer review, #45307 overlap, final source/run archival and independent numerical qualifications remain. Success does not establish historical FlashInfer mismatch causality. Optional and disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Maintainers review passing e95ec5ab and coordinate #45307 overlap. Preserve current-head Base success; no freshness rerun/rebase. Archive exact source/run mapping and retain independent numerical qualifications. Optional and disabled in frozen llr37. Author-reported dual-image evidence remains source-specific; obtain matching controls before assigning historical FlashInfer mismatches to main or this patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-7010787687)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44535642771)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...e95ec5abc8b4a9bf27b3dae160e42dc5beeb5f6e)

## #47278: New 9602092a Base attempt 1 has 32 functional/seven JIT passes and one four-GPU GB300 job queued without steps; AMD attempt 1 has 16 functional/two JIT passes and one eight-GPU disaggregation job queued without steps. New MI35x sparse-MLA log confirms 45 cases passed, including 12 boundary/two LSE cases, on a matching production blob. No current-head functional failure is observed. Previous cabe5837 Base success, Qwen3-8B startup OOM and unexplained cancellation policy remain historical; new queues do not prove their recovery. Author supplied current-code e2e A/B, but raw source/image/run archival, reviewer assessment and numerical/layout/replay qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Follow current 9602092a Base/AMD queues without duplicate requests; preserve the author update. Maintainers assess the newly supplied current-code e2e A/B and complete review. Owner archives exact-parent 8d72d2ea source/image/raw run records and retains seq15/non-target numerical, BF16-partial, scaled-KV, DCP/LSE and layout/replay qualifications. The new report uses two servers/four measured runs per arm and establishes no TPOT or throughput gain. Retrieve existing evidence before requesting additional GPU work; any required hardware work needs owner-approved allocation. Compiler issue is now linked but identical causality is unconfirmed and reviewer-nonblocking. No issue/review reply without owner authorization. See cycles/20261120T180002Z/PR47278_REVIEW_HANDOFF.json and PR47278_GPU_VALIDATION.json; this required source-family port is not a byte-identical frozen-overlay replacement.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473623/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136428385564)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7033217328)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44694630806)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/job/136148493015)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/job/135554671697)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908317/attempts/3)
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

## #47278: Author supplied current-code e2e A/B in issuecomment-7043695032 and updated head to 9602092a; direct executable AST equality confirms only a compiler-issue comment differs from cabe5837. New CPU metadata checks confirm H64 scope, removed variants, exact-parent generic reducer/launch and split boundaries; current-head tested merge passed 45 sparse-MLA cases. The new report shows ITL 10.3% lower with TPOT/throughput overlap; reviewer acceptance and immutable raw evidence remain. Seq15/non-target exact-parent numerical, BF16-partial, scaled-KV/DCP/LSE and layout/replay qualifications are separate. Existing closed Triton issue 985 is now linked as a suspected cause, not proven identical; follow-up remains reviewer-nonblocking. Warmup 13.3s/26.9s remains author-reported. No new review decision; drafts remain local.

Tracking key: `pr47278-review-20261117`.

Next action: Follow current 9602092a Base/AMD queues without duplicate requests; preserve the author update. Maintainers assess the newly supplied current-code e2e A/B and complete review. Owner archives exact-parent 8d72d2ea source/image/raw run records and retains seq15/non-target numerical, BF16-partial, scaled-KV, DCP/LSE and layout/replay qualifications. The new report uses two servers/four measured runs per arm and establishes no TPOT or throughput gain. Retrieve existing evidence before requesting additional GPU work; any required hardware work needs owner-approved allocation. Compiler issue is now linked but identical causality is unconfirmed and reviewer-nonblocking. No issue/review reply without owner authorization. See cycles/20261120T180002Z/PR47278_REVIEW_HANDOFF.json and PR47278_GPU_VALIDATION.json; this required source-family port is not a byte-identical frozen-overlay replacement.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136428385564)
- [Evidence](https://github.com/sgl-project/sglang/blob/9602092ac908c2077a400abe7aa3d1a1b26a9220/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
- [Evidence](https://github.com/ROCm/triton/issues/985#issuecomment-6617640183)
