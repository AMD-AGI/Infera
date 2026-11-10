# Pending handoff for the next user session

Updated: **2026-11-10T06:35:27.306288+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #46820: Base has 11 A passes, three B test steps running and 12 B jobs queued; C unexecuted. Required NVIDIA Extra has five GPU stage passes, two simulator passes, three GPU tests running, 19 GPU jobs and one Xeon Extra job queued. Neither run is complete. AMD/Xeon coverage, socket review and default-unbounded-HWM/source limitations remain. Old-head AMD PP attribution remains historical.

Tracking key: `pr46820-base-review`.

Next action: Worker follows active Base and required Extra; recover eligible terminal failures through the helper. Maintainers finish common-code/socket review and merge assessment. Next local request slot 2026-11-10T07:06:36.969177+00:00, subject to fresh checks.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42765405451/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42765405574/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42765405233/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)

## #46830: Current head passed 30 Base and 17 AMD stage jobs. B200 DSA has nine passes and eight skips. Consumer/deadlock, CUDA/ROCm review and source qualifications remain.

Tracking key: `pr46830-base-review`.

Next action: Maintainers assess affected-path coverage and skipped B200 cases, then complete review. Preserve maintainer commits, fixture repair and passing runs.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42422218071/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42422217998/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830#pullrequestreview-5781043791)

## #45524: Base passed 27 functional jobs. AMD retains 12 stage successes, one failed shard and four skipped stages. Shard 6 exited 255 after starting tests; fresh log request returned HTTP 404. First exception, source equivalence and historical AWQ/PP review remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers retrieve runner logs and recover coverage; complete source-equivalence review while preserving passing Base/disaggregation and the frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903737/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Base has 36 functional successes and two shards cancelled without tests. Bot annotation is known; cancellation policy is not. Main conflict remains. Validation at 9763ef56 does not qualify current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers resolve conflicts, recover cancelled coverage, complete review and stock-image integration; preserve frozen workaround and scoped MI355X evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Attempt 14 remains queued with 22 retained functional successes and no new test steps. Historical attempt-13 H100 FlashInfer MLA EAGLE mismatch left 25 files unexecuted; test/harness blobs still match main fa703d685adc2465a86b53fb1929cfc870707756. Root cause, numerical/source evidence and overlap review remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow attempt 14 without duplication. Runner owner captures output component/dtype, tensors and replay state if mismatch recurs. Preserve optional llr37-disabled role, #45307 overlap coordination and exact numerical/source/run review. Next local request slot 2026-11-10T07:06:36.969177+00:00, subject to fresh priority checks.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/14)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6760378662)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42766307113)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/129643229661)

## #47278: Attempt 8 retains 37 functional passes, four JIT suites, one benchmark and one empty job; B2 cancelled before tests. Fresh helper refusal confirms required main 50322ac7 absent. Historical FDFO left 31 files unexecuted. Fresh merge preview finds four production conflicts against main fa703d68.

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

## #47872: Base passed 30 functional jobs, five JIT suites and two benchmarks; AMD passed all 17 stages and finish, and Xeon/ARM64 passed. Separate review, source/run mapping and numerical/long-context qualifications remain; graph-test success does not establish patch causality.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Maintainers review opt-in behavior, min(length,2048), exact fallback and valid ties; retain measured 32k regression, exact source/run mapping and MI355X/long-context limits. Preserve passing CI.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204156/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204152/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Matching inherited failure remains; checkout/package alignment unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Required main 50322ac7 remains absent. Fresh read-only preview against main fa703d685adc2465a86b53fb1929cfc870707756 finds four production conflicts: scaled-KV loading, block sizing, fused launch and split-K launch. Test file merges cleanly; numerical compatibility unvalidated.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles required main with numerical validation. Preserve BF16 partials, DCP/LSE, GROUP_W, scaled-KV/cache/async-copy semantics, exact source/run mapping and combined-stack limits; resume helper recovery after safe integration.

- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/compare/a5d8b3af95d7915910547073b1771825981ccb98...fa703d685adc2465a86b53fb1929cfc870707756)
