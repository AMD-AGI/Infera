# Pending handoff for the next user session

Updated: **2026-11-10T01:12:50.677665+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #46820: Base passed; HaiShaw approved, but REVIEW_REQUIRED remains. AMD PP retains transfer/HTTP errors before missing answer: basic 6/6 and PP 2/3 passed, two later files unexecuted. Recorded launches disable changed send. Socket/source review and default-unbounded-HWM limits remain.

Tracking key: `pr46820-base-review`.

Next action: Maintainers complete common-code/socket review and merge assessment. AMD CI owners correlate transfer peers and compare the same revision/environment; preserve passing Base and avoid speculative metrics changes.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659241/job/127907742328)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820#pullrequestreview-5832462115)

## #46830: Current head passed 30 Base and 17 AMD stage jobs. B200 DSA has 9 passes and 8 skips. Consumer/deadlock, CUDA/ROCm review and source qualifications remain.

Tracking key: `pr46830-base-review`.

Next action: Maintainers assess affected-path coverage and skipped B200 cases, then complete review. Preserve maintainer commits, fixture repair and passing runs.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42422218071/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42422217998/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830#pullrequestreview-5781043791)

## #45524: Base passed 27 functional jobs. AMD retains 12 stage successes, one failed shard and four skipped stages. Shard 6 exited 255 after starting tests; logs return 404. First exception, source equivalence and historical AWQ/PP review remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers retrieve runner logs and recover coverage; complete source-equivalence review while preserving passing Base/disaggregation and the frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903737/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Base has 36 functional successes and two shards cancelled without tests. Bot annotation is known; cancellation policy is not. Main conflict remains. Validation at 9763ef56 does not qualify current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers resolve conflicts, recover cancelled coverage, complete review and stock-image integration; preserve frozen workaround and scoped MI355X evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/job/125361049407)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Attempt 8 repeated the H100 FlashInfer MLA EAGLE graph mismatch after ten passing files; 25/36 files and C did not execute. Recovery is verified through successful handler and attempt 9, now queued with 22 functional passes retained and no new tests. Test/harness blobs match main; no established correction. Optional deployment, overlap and numerical/source review remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow queued Base without duplication. Next local author slot is 2026-11-10T02:04:07.238558Z, subject to intervening requests. Runner owner captures output tuple/dtype, tensors and replay state if failure recurs. Preserve optional llr37-disabled role, #45307 coordination and exact numerical/source/run review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/9)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/129519810316)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6755879335)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42733050889)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 8 retains 37 functional passes, four JIT suites, one benchmark and one empty job; B2 was cancelled before tests. Fresh helper refusal confirms required main 50322ac7 is absent. Historical FDFO left 31 files unexecuted. Four production conflicts remain against eafc4914.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer integrates required main with numerical validation, preserving BF16 partials and DCP/LSE; resume helper recovery afterward.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127656329036)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

## #47278: Inherited AMD attempt 2 passed 18 stage jobs. BF16/DCP/LSE review, exact run-to-source archiving and combined-stack qualifications remain; existing evidence does not qualify a reconciled kernel.

Tracking key: `pr47278-amd-validation-source`.

Next action: Yihou/maintainers finish review and source/run archiving; validate numerical behavior after conflict reconciliation.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base passed 27 functional jobs. Inherited AMD MIOpen/PP failures and Xeon checkout/package mismatch remain; ARM64 passed. MIOpen ran zero cases, leaving five later files unexecuted; PP left two files unexecuted. Failing Xeon suite left 16/23 files unexecuted.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers repair database access, investigate PP peers and align CPU revisions; preserve passing Base and separate source/numerical qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K does not establish equivalence or long-context DSA accuracy.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact evidence and complete unequal paired-row/table-length, dispatch/fallback and long-context review.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Current head has 23 functional Base passes; one B test executes, one B job is queued and C has not started. B0 passed 38/38 files, including seven FlashInfer MLA tests and the historical failing graph method. AMD/Xeon remain active; ARM64 passed. Full CI and source-specific numerical/long-context qualifications remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Follow active Base and affected architecture workflows without duplication. Maintainers review opt-in behavior, min(length,2048), exact fallback and valid ties; preserve measured 32k regression, exact source/run mapping and MI355X/long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204156/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204156/job/129523220766)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204152/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Matching inherited failure remains; checkout/package alignment is unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Required main 50322ac7 remains absent. Fresh read-only preview against eafc49148eb9515546a3ff64c0407d221f90e900 confirms four production conflicts: scaled-KV loading, block sizing, fused launch and split-K launch. Test file merges cleanly; numerical compatibility remains unvalidated.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles required main with numerical validation. Preserve BF16 partials, DCP/LSE, GROUP_W, cache/async-copy semantics, exact source/run mapping and combined-stack limits; resume helper recovery after safe integration.

- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/compare/a5d8b3af95d7915910547073b1771825981ccb98...eafc49148eb9515546a3ff64c0407d221f90e900)
