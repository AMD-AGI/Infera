# Pending handoff for the next user session

Updated: **2026-11-11T07:21:08.480110+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #46820: Base and required NVIDIA Extra passed, including torch-compile and FDFO; AMD/Xeon/ARM64 passed. Maintainer merge assessment and socket-contract/default-unbounded-HWM/source qualifications remain.

Tracking key: `pr46820-base-review`.

Next action: Maintainers assess approval and merge; preserve exact-source and socket-contract qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42765405574/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820)

## #46830: Base/AMD passed; reviewer approved November 10. Maintainer merge assessment and consumer/deadlock, CUDA/ROCm, skipped-case and source qualifications remain.

Tracking key: `pr46830-base-review`.

Next action: Maintainers complete merge assessment while preserving passing head and existing qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42422218071/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830)

## #45524: Base passed 27 functional jobs. AMD shard 6 exited 255 after starting tests; fresh log retrieval remains HTTP 404. First exception, source equivalence and historical AWQ/PP review remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers retrieve runner logs, recover coverage and complete source-equivalence review; preserve frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Base retains 36 functional successes and two shards cancelled without tests. Cancellation policy unknown; current merge state is dirty. Scoped validation at 9763ef56 does not qualify current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers own original-PR conflict, coverage and review decisions. Progress owned continuation #49732 separately; preserve scoped MI355X evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Attempt 17 failed on GitHub HTTP 500 in the wait job; H100 tests skipped. Accepted recovery created queued attempt 18 with 22 retained functional passes and no new test execution. Historical FlashInfer MLA EAGLE graph mismatch and source/overlap qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow attempt 18 without duplication. Next author slot 2026-11-11T08:14:42Z, subject to fresh eligibility. If graph mismatch recurs, runner owner captures comparison component/dtype, tensors and replay state. Preserve optional llr37-disabled role, #45307 overlap and exact-source/numerical review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/18)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/130355315129)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6789589676)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42978072736)

## #47278: Base now has 28 functional passes (A11/B15/C2), but GB300 setup lacked yaml; two other C shards failed health checks. Three functional suites never ran. Helper refused recovery due to hourly author interval.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Retry through helper at/after 2026-11-11T08:14:42Z after fresh eligibility checks; CI maintainers investigate missing PyYAML setup. Preserve numerical/source constraints.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42947082865/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42947082865/job/130373088272)

## #47278: Current-head AMD has 14 functional stages and JIT unit tests passed; three functional jobs queued without test steps. Inherited 33-test MI355X report requires exact-source/run archiving and BF16/DCP/LSE/combined-stack qualification.

Tracking key: `pr47278-amd-validation-source`.

Next action: Follow active AMD; owner/reviewers archive and verify matching candidate evidence before qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42947083003/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47897: Base passed 27 functional jobs. Matching AMD MIOpen/PP failures and Xeon checkout/package mismatch remain; ARM64 passed. Five later MIOpen files, two PP files and 16/23 files in the failing Xeon suite remain unexecuted.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers repair database access, investigate PP peers and align installed/test revisions; preserve passing Base and numerical qualifications.

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

## #47278: Author supplied reconciliation d5fb0d33 and candidate f5fc9afe, reporting scaled-KV compatibility, a 16-split cap and 33 MI355X passes. Source/run archival verification, scaled-KV/group-width/launch/reducer review and current-head e2e qualification remain. Changed head alone does not resolve these qualifications.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Owner/reviewers verify supplied evidence and preserve BF16 partials, DCP/LSE, GROUP_W, scaled-KV/cache/async-copy and matching launch/reducer semantics. Archive exact source/runs and retain combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/commit/d5fb0d33dd664193bee70c21025f185766e7f492)
- [Evidence](https://github.com/sgl-project/sglang/commit/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #49732: Maintainer run-ci authorization is present; Base attempt 2 passed gate and 11 actual A jobs, with B pending. Local isolated configuration tests pass, but full CI, review and main/image qualification remain. Archived 9763ef56 MI355X evidence applies only to that original source.

Tracking key: `pr49732-base-review-validation`.

Next action: Follow active Base and maintainer review; preserve original #46918 credit. Historical MI355X evidence at 9763ef56 does not qualify this main-based candidate or stock image. Full registered-module collection previously lacked pybase64; CI must cover imports/tests. Retain integrated-image qualification.

- [Evidence](https://github.com/sgl-project/sglang/pull/49732)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42974439450/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)
- [Evidence](https://github.com/AMD-AGI/Infera/tree/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/validation/b8_49732)
