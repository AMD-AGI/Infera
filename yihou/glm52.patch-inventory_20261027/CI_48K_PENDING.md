# Pending handoff for the next user session

Updated: **2026-11-06T21:10:37.888015+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base fully recovered: 28 functional successes, 15 retained; final B2 passed 9/9 files. This mixed item remains for REVIEW_REQUIRED.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Maintainers complete review and merge assessment; preserve native MI300/FlashInfer historical follow-up and source-specific MI355X limits without adding a hardware gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059/files)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers assess native MI300 follow-up alongside source-specific MI355X evidence and required review.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base recovered: attempt 4 passed all 28 functional jobs, retaining 27; B2 passed 9/9 files after the prior installation-lock failure. The same key remains for shared-code AMD PP coverage and socket/source review. AMD retains 16 successes with PP queued; default-unbounded-HWM limits remain.

Tracking key: `pr46820-base-review`.

Next action: Maintainers complete socket/source review and merge assessment while preserving current-head Base success and default-unbounded-HWM limitations. Follow the active AMD PP job without duplication; obtain transfer-peer evidence if its prior transfer/KeyError failure repeats.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/128048607214)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659241/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820)

## #46830: Inherited request and successful five-workflow handler are verified. Base attempt 5 retains 22 functional successes; B0 queues without steps and C remains unexecuted. Attempt 4 repeated the FlashInfer eager/graph mismatch (1/6, maximum difference 43), leaving 25/36 files unexecuted. Tested-merge test/runner match current main adeec96c. AMD retains 16 successes with PP queued. B200/source/deadlock review remains.

Tracking key: `pr46830-base-review`.

Next action: Worker follows B0 and downstream C plus AMD PP; use the helper after terminal unsuccessful Base and eligibility, no earlier than November 6 21:39:24 UTC absent another reservation. Attention-test maintainers capture actual/expected tensors, top-k indices and replay metadata at the failing comparison; no DSA causality or small CPU-validatable repair is established. Maintainers complete consumer/deadlock and CUDA/ROCm review, retaining B200 and exact-source qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/5)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830#issuecomment-6694984320)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42290190788)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/job/128016347235)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086232/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/blob/f69d354dd2649fc5f9962f5fe6ceec877637c1c0/test/registered/attention/unittests/mla/test_flashinfer.py#L240)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830/files)

## #45524: Base passed 27 functional jobs. AMD disaggregation passed 13 tests before this cycle, bringing stage successes to 12. Another AMD shard failed with exit 255; saved individual-log query returned 404 and full archive omitted it. AMD C skipped; Xeon/ARM64 passed. Historical AWQ/PP and source-equivalence review remain.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers retrieve the missing runner log, recover failed/skipped AMD coverage and complete source-equivalence review. Preserve passing Base/disaggregation and frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903737/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649515)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Two B shards cancelled without tests after 36 successes. Annotation names github-actions[bot]; policy unknown. Finish failed on cancellation; prior main conflict remains unresolved; fresh mergeability is UNKNOWN. Validation at 9763ef56 does not qualify current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers reconcile conflict, recover cancelled shards, complete review and stock-image integration. Preserve frozen workaround and source-specific MI355X evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/job/125361049407)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Attempt 3 has 21 functional successes: 20 retained plus B1 9/9 files. B0 repeated FlashInfer eager/graph mismatch (1/6 elements, maximum difference 43), leaving 25/36 files unexecuted. B2 queues without steps and C is unexecuted. Five JIT unit suites/two benchmarks passed, one JIT job ran 0/0. Optional deployment, overlap and numerical/source qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker follows B2 and recovers terminal Base when eligible, behind actionable priority PRs. CI attention-test owner captures eager/graph outputs and replay metadata for the repeated FlashInfer mismatch. Preserve optional llr37-disabled role, coordinate #45307 overlap, complete numerical review and archive exact source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/127983092114)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/127983092335)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 8 cancelled before B2 started; 37 functional successes and six JIT conclusions retained, including one empty job. Finish failed on cancellation; policy unknown. Required main remains absent; historical FDFO shard left 31 files unexecuted.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer integrates required main with numerical conflict validation, preserving BF16 partials and DCP/LSE. Resume helper recovery after safe integration; retain source/run and combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127656329036)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

## #47278: Inherited AMD attempt 2 passed 18 stage jobs, 14 retained. BF16/DCP/LSE review, source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou finish review and archive exact evidence. Passing AMD does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base passed 27 functional jobs, four JIT unit suites and one benchmark; another JIT job ran 0/0. Inherited AMD MIOpen database/PP failures and Xeon checkout/package mismatch remain; ARM64 passed.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers repair database access, compare PP baselines and align CPU revisions. Preserve passing Base and separate review qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact evidence; preserve unequal paired-row/table-length, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Attempt 3 has 21 functional successes, including B1 9/9 files and 20 retained jobs. B0 repeated FlashInfer graph mismatch (1/6, difference 43), leaving 25/36 files unexecuted. B2 queues, C unexecuted; eight JIT conclusions retained including one empty job. AMD/Xeon/ARM64 passed previously. Prior main conflict and source/long-context review remain; fresh mergeability is UNKNOWN.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Follow B2; request eligible recovery after Base becomes terminal, behind priority PRs. Capture eager/graph tensors and replay metadata if repeated. Reconcile main conflict without interrupting active CI; preserve opt-in, min(length,2048), exact fallbacks, valid ties, 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127950269984)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127950270263)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Fresh helper dry-run remains blocked/required_main_missing: required 50322ac7ba85df18111ec79d18a4c5c554df60c9 is absent and ancestry diverges. Fresh mergeability is UNKNOWN; inherited scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflict evidence remains unresolved and requires numerical reconciliation beyond routine CPU-only repair.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer integrates required main and reconciles scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflicts with numerical validation. Preserve BF16 partials and DCP/LSE. Resume helper recovery after safe integration; archive source/runs and retain combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
