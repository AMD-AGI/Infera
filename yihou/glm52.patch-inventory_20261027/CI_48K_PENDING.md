# Pending handoff for the next user session

Updated: **2026-11-10T09:09:48.250644+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #46820: Base has 27 functional passes, two C tests running and two C jobs queued. Required NVIDIA Extra has 20 GPU test passes, two simulator passes, four GPU tests running and three queued. Remaining AMD/Xeon coverage, common-code/socket review and default-unbounded-HWM/source qualifications remain.

Tracking key: `pr46820-base-review`.

Next action: Follow active Base/Extra and recover eligible terminal failures through the helper. Maintainers complete review and merge assessment. Next author slot 10:03:02.484224 UTC, subject to fresh checks.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42765405451/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42765405574/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)

## #46830: Current head passed 30 Base and 17 AMD stages. B200 DSA has nine passes and eight skips. Consumer/deadlock, CUDA/ROCm review and source qualifications remain.

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

## #47276: Attempt 14 repeated the FlashInfer MLA EAGLE graph mismatch (1/6, maximum difference 43), leaving 25 files unexecuted. Helper recovery produced queued attempt 15 with 22 retained functional successes and no new functional test steps. Test/harness blobs match current main; root cause, numerical/source evidence and overlap review remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow attempt 15 without duplication. Runner owner captures output component/dtype, tensors and replay state if mismatch recurs. Preserve optional llr37-disabled role, #45307 overlap coordination and exact numerical/source/run review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/15)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6763321034)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42787444375)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/129663638835)

## #47278: Attempt 8 retains 37 functional passes and a cancelled B2 shard without tests. Historical FDFO left 31 files unexecuted. Prior helper refused missing required main; owner escalated four kernel conflicts for reconciliation. No candidate or new numerical validation supplied.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Owner supplies a reconciled kernel candidate before integration or recovery. Preserve BF16 partials, DCP/LSE, GROUP_W, scaled-KV/cache/async-copy and matching launch/reducer semantics.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127656329036)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

## #47278: Inherited AMD attempt 2 passed 18 stages. Fresh PR head unchanged; BF16/DCP/LSE review, exact source/run archiving and combined-stack qualifications remain. Existing evidence does not qualify a reconciled kernel.

Tracking key: `pr47278-amd-validation-source`.

Next action: Owner supplies kernel reconciliation; reviewers preserve exact-source qualifications and obtain matching numerical/performance validation before recovery.

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

## #47872: OWNER PAUSED pending redesign. Historical, unverified this cycle: Base passed 30 functional jobs, five JIT suites and two benchmarks; AMD passed 17 stages and finish; Xeon/ARM64 passed. Separate review, source/run mapping and numerical qualifications remain archived.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Owner-paused; resume only on owner instruction. Preserve historical opt-in/fallback/tie, measured 32k regression and exact-source/long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204156/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204152/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Matching inherited failure remains; checkout/package alignment unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: November 10 conflict handoff found four production conflicts against main 73bf84e90f1cc14ccf65de37b57f4d3d8653ea85: scaled-KV loading, tile sizing and fused/split-K launch/reducer semantics. Fresh PR head unchanged. Clean test-file merge does not validate numerical compatibility.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Owner supplies a kernel-agent-reconciled candidate with numerical/performance evidence. Preserve BF16 partials, DCP/LSE, GROUP_W, scaled-KV/cache/async-copy semantics, exact source/run mapping and combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/reviews/PR47278_CONFLICT_HANDOFF_20261110.md)
