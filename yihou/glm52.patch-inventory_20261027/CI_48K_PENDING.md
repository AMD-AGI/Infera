# Pending handoff for the next user session

Updated: **2026-11-10T00:12:17.246999+00:00**.

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

## #47276: Base attempt 8 remains queued with 22 functional successes retained and no new functional test step started. Prior B0 graph comparison failed after ten passing files, leaving 25/36 files and C unexecuted. Previous request/handler verified again. Optional deployment, overlap and numerical/source review remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker follows active Base; runner owner captures output tuple/dtype, tensors and replay state if the assertion recurs. Preserve optional llr37-disabled role, #45307 coordination and exact numerical/source/run review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/129164387670)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722473799)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 8 retains 37 functional successes, four JIT suites, one benchmark and one empty job; B2 was cancelled before tests. Fresh helper refusal confirms required main 50322ac7 remains absent. Historical FDFO left 31 files unexecuted. Four production integration conflicts persist against current main 4a5c6b8f.

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

## #47872: At published dccb51e0, Base gate and 13 functional jobs passed (A11/B2); three B test steps execute, nine B jobs are queued and C is unstarted. Three JIT suites and one benchmark passed separately. AMD remains active; Xeon has two test-job passes and one queued, ARM64 passed. No old-head results retained. Historical graph failure is not proven fixed; source-specific numerical and long-context qualifications remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker follows active Base and affected architecture CI without duplication; diagnose any recurring graph failure without speculative kernel changes. Maintainers review opt-in behavior, min(length,2048), exact fallbacks and valid ties; preserve measured 32k regression, exact source/run mapping and MI355X/long-context limits. Integration clears the text conflict, not CI or numerical qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204156/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/commit/dccb51e0f9f3d97dc01aea7aa39db1c0cfe3df34)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/18)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Matching inherited failure remains; checkout/package alignment is unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Required main 50322ac7 remains absent. Fresh read-only preview against 4a5c6b8f632f1bb9ef1fb7724da9fda771dda6be confirms four production conflicts: scaled-KV loading, block sizing, fused launch and split-K launch. Test file merges cleanly; numerical compatibility remains unvalidated.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles required main with numerical validation. Preserve BF16 partials, DCP/LSE, GROUP_W, cache/async-copy semantics, exact source/run mapping and combined-stack limits; resume helper recovery after safe integration.

- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/compare/a5d8b3af95d7915910547073b1771825981ccb98...4a5c6b8f632f1bb9ef1fb7724da9fda771dda6be)
