# Pending handoff for the next user session

Updated: **2026-11-09T16:09:09.623352+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base fully passed 28 functional jobs. This mixed key remains for REVIEW_REQUIRED.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Maintainers complete review and merge assessment; preserve native MI300/FlashInfer historical follow-up and source-specific MI355X limits without adding a hardware gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059/files)

## #33059: Native MI300 coverage remains unestablished; existing approval supplies no MI300 evidence. Historical follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers assess native MI300 follow-up alongside source-specific MI355X evidence and required review.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base passed. AMD PP retains transfer/HTTP errors preceding missing answer; basic 6/6 and PP 2/3 passed, with two later files unexecuted. Launches disable the changed send. Socket/source review and default-unbounded-HWM limits remain.

Tracking key: `pr46820-base-review`.

Next action: Maintainers complete common-code review; AMD CI owners correlate transfer peers and compare the same revision/environment. Preserve passing Base and avoid speculative metrics changes.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659241/job/127907742328)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820)

## #46830: Current head passed 30 Base and 17 AMD stage jobs. B200 DSA has 9 passes and 8 skips. Consumer/deadlock, CUDA/ROCm review, exact-source and affected-path qualifications remain.

Tracking key: `pr46830-base-review`.

Next action: Maintainers assess successful CI and skipped B200 cases, then complete review and merge assessment. Preserve updated commits and passing runs.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42422218071/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42422217998/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42422218071/job/128555170421)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830#pullrequestreview-5781043791)

## #45524: Base passed 27 functional jobs. AMD retains 12 stage successes, one failed shard and four skipped stages. Shard 6 exited 255 after starting tests; logs return 404 and the inherited archive omits it. First exception, source equivalence and historical AWQ/PP review remain unresolved.

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

## #47276: Attempt 7 retains 22 functional successes but B0 failed FlashInfer graph comparison after ten passing files, leaving 25/36 unexecuted; C skipped. Fresh helper refusal confirms the hourly interval remains occupied by the previous #47872 request. No matching main correction established; optional deployment, overlap and numerical/source review remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker retries no earlier than 2026-11-09 16:38:08 UTC through the helper after fresh checks. Capture tuple position, tensors and replay metadata; preserve optional llr37-disabled role, #45307 coordination and numerical/source qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/7)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/129164387670)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 8 retains 37 functional successes, four JIT suites, one benchmark and one empty job; B2 was cancelled before tests. Helper refusal at 16:02:43 UTC confirms required main 50322ac7 absent. Aggregate failed on cancellation; historical FDFO left 31 files unexecuted. Numerical conflicts remain.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer integrates required main with numerical conflict validation, preserving BF16 partials and DCP/LSE; resume helper recovery afterward.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127656329036)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

## #47278: Inherited AMD attempt 2 passed 18 stage jobs. BF16/DCP/LSE review, exact run-to-source archiving and combined-stack qualifications remain; existing evidence does not qualify a reconciled kernel.

Tracking key: `pr47278-amd-validation-source`.

Next action: Yihou/maintainers finish review and source/run archiving; validate numerical behavior after conflict reconciliation.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base passed 27 functional jobs. Matching inherited AMD MIOpen/PP failures and Xeon checkout/package mismatch remain; ARM64 passed. MIOpen ran zero cases, leaving five later files unexecuted; PP left two files unexecuted. The failing Xeon suite left 16/23 files unexecuted.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers repair database access, investigate PP peers and align CPU revisions; preserve successful Base and separate source/numerical qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K does not establish equivalence or long-context DSA accuracy.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact evidence and complete unequal paired-row/table-length, dispatch/fallback and long-context review.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Historical attempt 10 failed FlashInfer MLA EAGLE graph comparison (1/6 mismatched; maximum absolute difference 43), leaving 25/36 files unexecuted. Previous durable request and successful PR-wide handler verified. Attempt 11 retains 22 functional successes; B0 Run test started at 15:48:41 UTC and remains active, with no new completed functional result. Main conflict and source/long-context qualifications remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Follow running attempt 11 without duplication. Runner owner captures tuple index/dtype, tensors and replay metadata if failure recurs. Yihou/kernel maintainer reconciles main and completes source/long-context review, preserving opt-in behavior, min(length,2048), exact fallbacks, valid ties, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/11)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/129261917096)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872#issuecomment-6744154635)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42646663792)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/129226381332)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Matching inherited failure remains; checkout/package alignment is unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Fresh helper refusal confirms required main 50322ac7 absent; live PR remains conflicting. Inherited scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflicts require numerical reconciliation beyond routine CPU-only repair.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer integrates main with numerical validation, preserving BF16 partials and DCP/LSE; retain source/run and combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
