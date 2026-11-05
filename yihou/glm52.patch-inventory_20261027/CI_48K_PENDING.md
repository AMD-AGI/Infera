# Pending handoff for the next user session

Updated: **2026-11-05T23:14:40.609484+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base-only scope. Attempt 2 has 27 functional successes, none retained from attempt 1. B0 failed FlashInfer graph comparison with eight files unexecuted; B1 failed health checks without tests; B2 remains runnerless. Full current-head Base remains required.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: preserve B2, then reconcile the minimum-base requirement if applicable and recover terminal failures. Maintainers: finish review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/job/127254428521)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/job/127254428584)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate. Owner hold is lifted.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers: assess native MI300 follow-up alongside source-specific MI355X evidence and approval; worker continues Base recovery.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: New head 2292a90a integrates required main; eight real ZMQ CPU cases passed. Base and scoped AMD/Xeon/ARM64 stopped at cooldown gates; no functional tests or retained results. Historical FlashInfer failure remains unproven as patch-caused; 17 old launches disabled KV events.

Tracking key: `pr46820-base-review`.

Next action: Worker: recheck stale_gate eligibility at 2026-11-06 00:47:11 UTC and recover full Base in priority order. Maintainers: finish socket-contract/source review and retain default-unbounded-HWM reproduction limits.

- [Evidence](https://github.com/sgl-project/sglang/commit/2292a90adc3e9f4080e1e88af1eb01547aa07918)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125423985/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125423985/job/127588454073)

## #46830: New head c12719f6 preserves maintainer merge history, fixture fix and resolved comments; six CPU tests plus two subtests passed. Base/AMD stopped at cooldown gates with no functional execution. Historical non-DSA PP transfer failure left two files unexecuted; fresh coverage and DSA review remain required.

Tracking key: `pr46830-base-review`.

Next action: Worker: prioritize this oldest-created newly integrated priority run when stale_gate becomes eligible after 2026-11-06 00:47:11 UTC. Maintainers: complete DSA consumer/deadlock and cross-platform review.

- [Evidence](https://github.com/sgl-project/sglang/commit/c12719f69d0b4291d572322da30b7347d818b4ce)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820535/job/127140798942)

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

## #47276: Required-main integration published as 3d739b80 with unchanged kernel/loader blobs and passing CPU launch-selection check. New Base failed only cooldown; no functional or JIT tests executed. Historical parser failure and retained old-head successes do not qualify this head.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: recover stale_gate when eligible after priority PRs. Maintainers/yihou: retain optional llr37-disabled role, coordinate #45307, finish numerical review and archive exact source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/commit/3d739b80766dd8145d1e798edb9d202aba091d47)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/127591306865)

## #47278: Attempt 6 failed required-main ancestry before B2 tests. Gate, 37 functional successes, five JIT units and one benchmark are retained. Historical FDFO threshold failure left 31 files unexecuted. Main contains its correction, but numerical integration conflicts remain unqualified.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer: reconcile numerical conflicts and integrate required main with BF16-partial and DCP/LSE validation. Worker resumes Base recovery afterward; retain review/source archiving.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/6)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127416482108)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

## #47278: Inherited AMD attempt 2 passed 18 stage jobs, 14 retained; C passed 3/3 files. BF16/DCP/LSE review, source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou: finish BF16 partials and DCP/LSE review and archive source/run evidence. Passing AMD does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base remains successful with 27 functional jobs, five JIT units and one benchmark, all predating this cycle. Shared architecture changes justify AMD/Xeon/ARM64 scope. AMD MIOpen/PP failures and Xeon revision mismatch persist; ARM64 passed.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers: repair database access, compare PP baselines and align CPU revisions. Preserve passing Base and separate review/provenance/long-context qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: New 2f78eb19 Base passed its gate and 10 A jobs this cycle. One JIT unit and one benchmark passed; another unit failed CUDA-device allocation, leaving 79/80 files unexecuted. B0 then failed health checks without tests; 12 B jobs remain queued. ARM64 passed; AMD and Xeon remain active. No old-head successes are retained.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker: preserve active attempts, then recover terminal Base through the helper while honoring priority PRs. Maintainers/yihou: finish review/source mapping; preserve opt-in, min(length,2048), exact fallbacks, valid ties, 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/commit/2f78eb197d76359bfb054618038ef1054bb060b3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127585215489)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124400918/attempts/1)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from installed package. Checkout/package mismatch remains separate from repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Historical Xeon attempt 3 mixed branch package and merge-revision tests, causing a missing ValueError and missing imports. Integration published 2f78eb19, but current-head Xeon 42124400875 has not executed its functional suite. The old mismatch is not yet demonstrated resolved. ARM64 success does not establish Xeon coverage.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: Worker: follow new Xeon run and inspect actual test_server_args execution. CI maintainers: align package/test revisions if mismatch recurs; no unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124400875/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294696)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294654)

## #47278: Fresh mergeability remains CONFLICTING/DIRTY. Established conflicts span scaled-KV, block sizing, GROUP_W, cache/async-copy and split-K. They now also block mandatory MIN_BASE_SHA integration; no numerical reconciliation or kernel edit occurred.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer: reconcile preserved conflicts, retain BF16 partials and DCP/LSE, and validate affected numerical cases before rewriting. Existing AMD success does not validate conflict resolution.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127416482108)
