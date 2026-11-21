# Pending handoff for the next user session

Updated: **2026-11-21T21:07:22.731992+00:00**.

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

## #47276: Current e95ec5ab Base attempt 3 remains fully passed: 35 functional/seven JIT jobs, including inherited 12 functional/one JIT retained results. Sources, completed-test timestamps and reviews unchanged. Maintainer review, #45307 overlap, exact source/run archival and independent numerical qualifications remain. Historical FlashInfer mismatch causality is unproven. Optional and disabled in frozen llr37. Fresh REST mergeability is null/unknown; no conflict inferred.

Tracking key: `pr47276-base-overlap-source`.

Next action: Maintainers review passing e95ec5ab and coordinate #45307 overlap. Preserve current-head Base success; no freshness retry/rebase. Owner archives exact source/run mapping and retains independent numerical qualifications. Optional and disabled in frozen llr37. Author-reported dual-image evidence remains source-specific; matching controls are needed before assigning historical FlashInfer mismatches to main or this patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-7010787687)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44535642771)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...e95ec5abc8b4a9bf27b3dae160e42dc5beeb5f6e)

## #47278: Current 9602092a Base attempt 1 and AMD attempt 1 remain fully passed: Base 33 functional/seven JIT and AMD 17 functional/two JIT jobs. All completed-test timestamps match prior evidence. Inherited matching logs record 45 sparse-MLA passes and four disaggregation files/13 cases. Prior-head OOM/cancellation remain historical without demonstrated causal recovery. Reviewer assessment, source/image/raw-run archival and independent numerical/layout/replay qualifications remain. Fresh REST mergeability is null/unknown; no conflict inferred.

Tracking key: `pr47278-amd-validation-source`.

Next action: Preserve passing 9602092a Base and AMD; no freshness retry/rebase. Maintainers assess supplied current-code e2e A/B and complete review. Owner archives exact-parent 8d72d2ea source/image/raw runs; retain seq15/non-target numerical, BF16-partial, scaled-KV, DCP/LSE and layout/replay qualifications. Author reports 10.3% lower ITL from two servers/four measured runs per arm; TPOT/throughput overlap. Retrieve existing evidence before any necessary owner-approved GPU allocation; no machine-pool skill is available here. Compiler link remains suspected, unconfirmed and reviewer-nonblocking. No issue/review reply without owner authorization. Local follow-up: cycles/20261121T210002Z/PR47278_REVIEW_HANDOFF.json and PR47278_GPU_VALIDATION.json. Required source-family port; not a byte-identical frozen-overlay replacement.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473623/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473623/job/136437583246)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136396984786)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136428385564)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/blob/9602092ac908c2077a400abe7aa3d1a1b26a9220/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
- [Evidence](https://github.com/ROCm/triton/issues/985#issuecomment-6617640183)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7033217328)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44694630806)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/job/136148493015)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/job/135554671697)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908317/attempts/3)
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

## #47278: Source and reviews unchanged at 9602092a; no new reviewer decision. Author supplied current-code e2e A/B reporting 10.3% lower ITL; TPOT/throughput overlap. Base and AMD fully passed. Inherited CPU controls verify H64 scope, removed variants, exact-parent generic reducer/launch and split boundaries; matching CI passed 45 sparse-MLA cases. Reviewer acceptance and immutable raw evidence remain, alongside seq15/non-target exact-parent numerical, BF16-partial, scaled-KV/DCP/LSE and layout/replay qualifications. Existing Triton issue 985 remains a suspected, unconfirmed, reviewer-nonblocking link. Warmup 13.3s/26.9s is author-reported; draft stays local.

Tracking key: `pr47278-review-20261117`.

Next action: Preserve passing 9602092a Base and AMD; no freshness retry/rebase. Maintainers assess supplied current-code e2e A/B and complete review. Owner archives exact-parent 8d72d2ea source/image/raw runs; retain seq15/non-target numerical, BF16-partial, scaled-KV, DCP/LSE and layout/replay qualifications. Author reports 10.3% lower ITL from two servers/four measured runs per arm; TPOT/throughput overlap. Retrieve existing evidence before any necessary owner-approved GPU allocation; no machine-pool skill is available here. Compiler link remains suspected, unconfirmed and reviewer-nonblocking. No issue/review reply without owner authorization. Local follow-up: cycles/20261121T210002Z/PR47278_REVIEW_HANDOFF.json and PR47278_GPU_VALIDATION.json. Required source-family port; not a byte-identical frozen-overlay replacement.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136428385564)
- [Evidence](https://github.com/sgl-project/sglang/blob/9602092ac908c2077a400abe7aa3d1a1b26a9220/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
- [Evidence](https://github.com/ROCm/triton/issues/985#issuecomment-6617640183)
