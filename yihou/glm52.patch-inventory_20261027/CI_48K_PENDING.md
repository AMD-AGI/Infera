# Pending handoff for the next user session

Updated: **2026-11-11T03:06:48.949631+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #46820: Base and required NVIDIA Extra fully passed, including torch-compile and FDFO. AMD/Xeon/ARM64 passed. Maintainer merge assessment and socket-contract/default-unbounded-HWM/source qualifications remain separate.

Tracking key: `pr46820-base-review`.

Next action: Maintainers assess approval and merge; preserve exact-source and socket-contract qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42765405574/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820)

## #46830: Base/AMD remain successful; reviewer approved November 10. Maintainer merge assessment and historical consumer/deadlock, CUDA/ROCm, skipped-case and source qualifications remain.

Tracking key: `pr46830-base-review`.

Next action: Maintainers complete merge assessment while preserving passing head and existing qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42422218071/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830)

## #45524: Base passed 27 functional jobs. AMD shard 6 exited 255 after starting tests; fresh log retrieval remains HTTP 404. First exception, source equivalence and historical AWQ/PP review remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers retrieve runner logs, recover affected coverage and complete source-equivalence review; preserve frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Base retains 36 functional successes and two shards cancelled without tests. Cancellation policy remains unknown; fresh GitHub mergeability is CONFLICTING/DIRTY. Scoped validation at 9763ef56 does not qualify the current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers resolve conflicts, recover coverage, complete review and stock-image integration; preserve scoped MI355X evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Attempt 15 graph mismatch remains inherited failure evidence. Successful recovery handler created attempt 16, still queued with 22 retained functional passes and no new test steps. Numerical/source evidence and overlap review remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow active attempt 16. If mismatch recurs, runner owner captures component/dtype, tensors and replay state. Preserve optional llr37-disabled role, #45307 overlap and exact-source review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/16)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6785457161)

## #47278: Current candidate Base remains active: 12 functional passes (A11/B1), two B test steps running, additional B/JIT work queued or executing; C not yet created. Old-head cancellations are historical, not current-head failures.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Follow active Base without duplicate recovery; retain separate numerical/source qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42947082865/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47278: Current-head AMD has three functional stages and JIT unit tests passed; six functional test steps running and four queued. No complete current-head AMD result yet. Inherited 33-test MI355X report still needs exact-source/run archiving and BF16/DCP/LSE/combined-stack qualification.

Tracking key: `pr47278-amd-validation-source`.

Next action: Follow active AMD; owner/reviewers archive and verify matching candidate evidence before qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42947083003/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47897: Base passed 27 functional jobs. Matching AMD MIOpen/PP failures and Xeon checkout/package mismatch remain; ARM64 passed. Five later MIOpen files, two PP files and 16/23 files in the failing Xeon suite remain unexecuted.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers repair database access, investigate PP peers and align installed/test revisions; preserve passing Base and separate numerical qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K does not establish long-context DSA accuracy.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact evidence and complete unequal paired-row/table-length, dispatch/fallback and long-context review.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: OWNER PAUSED pending redesign. Historical passing CI and source/numerical qualifications remain archived and were not queried this cycle.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Owner-paused; resume only on owner instruction. Preserve opt-in/fallback/tie, measured 32k regression and exact-source/long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204156/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Matching Xeon failure cannot import _paged_experts_num_resident from the installed package. Checkout/package alignment remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Author supplied reconciliation d5fb0d33 and candidate f5fc9afe, reporting scaled-KV compatibility, a 16-split cap and 33 MI355X test passes. Source/run archival verification, scaled-KV/group-width/launch/reducer review and current-head e2e qualification remain. Changed head alone does not resolve these qualifications.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Owner/reviewers verify supplied evidence and preserve BF16 partials, DCP/LSE, GROUP_W, scaled-KV/cache/async-copy and matching launch/reducer semantics. Archive exact source/runs and retain combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/commit/d5fb0d33dd664193bee70c21025f185766e7f492)
- [Evidence](https://github.com/sgl-project/sglang/commit/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)
