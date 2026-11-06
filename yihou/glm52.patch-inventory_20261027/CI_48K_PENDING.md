# Pending handoff for the next user session

Updated: **2026-11-06T00:16:30.555461+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Required main ancestry is now integrated. Fresh Base passed its gate and started nine CPU shards; full coverage remains pending. Historical FlashInfer failure is not proven fixed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Preserve active Base, then recover terminal failures until full success; maintainers review the integrated head.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/commit/a609bd2dc345e96132a53b3b46b13f6df03bdf11)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this remains historical follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers assess native MI300 follow-up alongside source-specific MI355X evidence and review.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Maintainer head 4890a5aa has executing Base: five CPU shards passed, remaining A jobs active, later suites unexecuted at audit. Socket-contract review and default-unbounded-HWM evidence limits remain.

Tracking key: `pr46820-base-review`.

Next action: Preserve active CI and recover terminal Base failures; maintainers finish shared-code/socket-contract review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)

## #46830: Base/AMD remain gate-only failures without tests. Helper preflight was hourly-blocked; new author Base implies conditional recovery no earlier than November 6 02:07:05 UTC. Maintainer history, fixture fix and resolved comments remain preserved.

Tracking key: `pr46830-base-review`.

Next action: Recover this oldest-created failed priority run with stale_gate after fresh helper checks; maintainers finish DSA consumer/deadlock review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/1)

## #45524: Inherited Base has 35 functional successes. AWQ timed out, leaving 27/30 files unexecuted; three siblings failed health checks. Scoped AMD PP/MLA failures, CPU revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers diagnose AWQ/PP, align CPU revisions, recover coverage and finish review. Preserve frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base shards were cancelled without steps after 36 functional successes. Cancellation policy and main conflicts remain unresolved. Historical 9763ef56 validation does not qualify the current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers reconcile main, recover coverage and finish review/stock-image integration; preserve scoped evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Current Base remains cooldown-only with zero tests. Conditional recovery is no earlier than November 6 02:07:05 UTC. Historical CPU evidence does not qualify current GPU numerics.

Tracking key: `pr47276-base-overlap-source`.

Next action: Recover after priority PRs and fresh helper checks; retain optional llr37-disabled role, coordinate #45307, finish numerical review and archive source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/1)

## #47278: Verified attempt 7 queues B2 without steps. All 37 functional successes and executed JIT results are retained. Attempt 6 failed required main ancestry; historical FDFO failure left 31 files unexecuted. Numerical integration conflicts remain.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Preserve active attempt. Yihou/kernel maintainer reconciles required main with BF16-partial and DCP/LSE validation before further source recovery.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/7)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6672007077)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42128118272)

## #47278: Inherited AMD attempt 2 passed 18 stage jobs, 14 retained. BF16/DCP/LSE review, source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou finish review and archive exact evidence. Passing AMD does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base remains successful with 27 functional jobs, four executed JIT unit suites and one benchmark; another successful JIT job ran 0/0 tests. Scoped AMD MIOpen/PP failures and Xeon revision mismatch persist; ARM64 passed.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers repair database access, compare PP baselines and align CPU revisions. Preserve passing Base and separate review qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact evidence; preserve unequal paired-row, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base retains 10 A successes, two executed JIT unit successes and one benchmark. CUDA-device failure left 79/80 files unexecuted; nine B jobs and another JIT job failed health checks. Four B and two JIT jobs queue. AMD remains active. Xeon SPR source mismatch is resolved, but GNR queues. Review/source qualifications remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Preserve active attempts; recover terminal Base after priority PRs. Retain opt-in, exact fallbacks, valid ties, min(length,2048), 32k performance limits and source-specific MI355X evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127585215489)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124400918)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124400875/job/127584974808)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. This PR's checkout/package mismatch remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers build/install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Main conflict and absent mandatory ancestry persist. Established conflicts span scaled-KV, block sizing, GROUP_W, cache/async-copy and split-K. Attempt 7 does not reconcile these numerical changes.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles preserved conflicts, retains BF16 partials and DCP/LSE, and validates affected numerical cases before rewriting.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127416482108)
