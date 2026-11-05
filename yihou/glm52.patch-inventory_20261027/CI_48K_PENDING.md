# Pending handoff for the next user session

Updated: **2026-11-05T23:42:17.759612+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base-only attempt 2 has 27 inherited functional successes; B0 FlashInfer graph comparison failed with eight files unexecuted, B1 failed health checks and B2 remains runnerless. Mandatory main floor 50322ac7 is absent.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: preserve active B2, then safely integrate required main and recover full current-head Base. Maintainers: finish review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/job/127254428521)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate. Owner hold is lifted.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers: assess native MI300 follow-up alongside source-specific MI355X evidence and approval; worker continues Base recovery.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Current 2292a90a Base/AMD/Xeon/ARM64 remain gate-only failures without functional tests. Local slot is occupied until November 6 00:14:39.190 UTC; earliest conditional upstream recovery is 00:47:11 UTC. Inherited CPU evidence and historical failure attribution do not establish fresh Base success.

Tracking key: `pr46820-base-review`.

Next action: Worker: recheck eligibility and recover full Base in priority order. Maintainers: finish shared-code/socket-contract review and retain exact-source/default-unbounded-HWM reproduction limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125423985/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125423985/job/127588454073)

## #46830: Current c12719f6 Base/AMD remain gate-only failures with no tests. Helper dry-run is hourly-blocked; earliest conditional upstream recovery is November 6 00:47:11 UTC. Maintainer merge history, fixture fix and resolved comments remain preserved; fresh coverage and DSA review remain.

Tracking key: `pr46830-base-review`.

Next action: Worker: prioritize this oldest-created eligible failed priority run after fresh cooldown checks. Maintainers: complete DSA consumer/deadlock and cross-platform review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/job/127587305756)

## #45524: Inherited Base has 35 functional successes. VLM ultimately passed; AWQ timed out after 1800 seconds, leaving 27/30 files unexecuted. Three siblings failed health checks without tests. Scoped AMD PP/MLA failures, CPU revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: inspect AWQ startup and PP baselines, align CPU revisions, recover coverage and finish review. Preserve frozen producer patch; worker remains read-only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base B shards were cancelled without steps after 36 functional successes; finish failure reflects cancellation. Cancellation policy remains unknown. Mergeability is CONFLICTING/DIRTY. Historical 9763ef56 validation does not qualify current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: reconcile main, recover coverage, finish review and validate stock-image integration. Preserve scoped MI355X evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Current 3d739b80 Base failed only cooldown; zero functional/JIT tests. Earliest conditional upstream recovery is November 6 00:47:11 UTC. Historical parser results and CPU launch evidence do not qualify current GPU numerics.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: recover stale_gate after actionable priority PRs. Maintainers/yihou: retain optional llr37-disabled role, coordinate #45307, finish numerical review and archive exact source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/127591306865)

## #47278: Prior request and handler are verified: attempt 7 queues B2 with zero steps. All 37 functional successes, four executed JIT unit suites and one benchmark are retained; another successful JIT job ran 0/0 tests. Attempt 6 failed mandatory-main ancestry; historical FDFO failure left 31 files unexecuted. Missing main and numerical conflicts remain.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Preserve active attempt 7. Yihou/kernel maintainer must reconcile numerical conflicts and integrate required main with BF16-partial and DCP/LSE validation; worker resumes recovery afterward.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/7)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6672007077)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42128118272)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

## #47278: Inherited AMD attempt 2 passed 18 stage jobs, 14 retained; C passed 3/3 files. BF16/DCP/LSE review, source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou: finish BF16 partials and DCP/LSE review and archive source/run evidence. Passing AMD does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base remains successful with 27 functional jobs, four executed JIT unit suites and one benchmark, all predating this cycle. A fifth successful JIT job ran 0/0 tests. Shared architecture changes retain AMD/Xeon/ARM64 scope; AMD MIOpen/PP failures and Xeon revision mismatch persist, ARM64 passed.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers: repair database access, compare PP baselines and align CPU revisions. Preserve passing Base and separate review/provenance/long-context qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base has 10 A successes, two executed JIT unit successes and one benchmark; H200 ran 0/0 tests. CUDA-device failure left 79/80 files unexecuted. Six B jobs and another JIT job failed health checks; seven B and two JIT jobs queue, C skipped. AMD progressed to seven stage successes; Xeon is executing but incomplete.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Preserve active attempts, then recover terminal Base through the helper after priority PRs. Maintainers/yihou: preserve opt-in, min(length,2048), exact fallbacks, valid ties, 32k regression and source-specific MI355X limits; finish review/source mapping.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127585215489)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124400918)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124400875)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from installed package. Checkout/package mismatch remains separate from repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Current Xeon attempt 1 has one SPR unit-test job executing, a second building and a third queued. No completed functional result establishes resolution of the historical package/test revision mismatch. ARM64 success is separate.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: Worker: inspect completed current-head Xeon test_server_args results; CI maintainers align package/test revisions if mismatch recurs. Do not backport unrelated parser changes.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124400875/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124400875/job/127584974681)

## #47278: Fresh CONFLICTING/DIRTY status and absent mandatory main floor persist. Established conflicts span scaled-KV, block sizing, GROUP_W, cache/async-copy and split-K. Attempt 7 does not reconcile these numerical changes; no kernel edit occurred.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer: reconcile preserved conflicts, retain BF16 partials and DCP/LSE, and validate affected numerical cases before rewriting. Existing AMD success does not validate conflict resolution.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127416482108)
