# Pending handoff for the next user session

Updated: **2026-11-06T20:07:29.669183+00:00**.

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

## #46820: Attempt 4 retains gate and 27 functional successes; B2 queues without steps. Prior uv cache-lock failure occurred before tests; installer matches inspected main db5cd6ba. AMD PP queues after transfer/KeyError failure. Full Base and socket/source review remain.

Tracking key: `pr46820-base-review`.

Next action: Follow B2 and AMD PP without duplication. If installation repeats, runner owner identifies competing cache-lock activity; obtain PP transfer-peer evidence if repeated. Maintainers finish socket/source review and retain default-unbounded-HWM limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42263230817)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/127907757589)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659241/attempts/2)

## #46830: Attempt 4 retains 22 functional successes; B0 queues without steps and C is unexecuted. Prior FlashInfer mismatch repeated (1/6, maximum difference 43), leaving 25/36 files unexecuted. AMD queues PP with 16 successes retained. B200/source/deadlock review remains.

Tracking key: `pr46830-base-review`.

Next action: Worker follows B0/downstream C and AMD PP without duplication. CI attention-test owner captures eager/graph output tensors, top-k indices and replay metadata if the mismatch repeats; the test explicitly selects FlashInfer, not the changed DSA backend. Obtain PP transfer-peer evidence. Maintainers complete DSA consumer/deadlock and CUDA/ROCm review, preserving B200 and exact-source qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42253965740)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/job/127869572862)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086232/attempts/3)

## #45524: Base passed 27 functional jobs. AMD disaggregation passed 13 tests before this cycle, bringing stage successes to 12. Another AMD shard failed with exit 255; saved individual-log query returned 404 and full archive omitted it. AMD C skipped; Xeon/ARM64 passed. Historical AWQ/PP and source-equivalence review remain.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers retrieve the missing runner log, recover failed/skipped AMD coverage and complete source-equivalence review. Preserve passing Base/disaggregation and frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903737/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649515)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Two B shards cancelled without tests after 36 successes. Annotation names github-actions[bot]; policy unknown. Finish failed on cancellation; main conflict persists. Validation at 9763ef56 does not qualify current head/image.

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

## #47872: Attempt 3 has 21 functional successes, including B1 9/9 files and 20 retained jobs. B0 repeated FlashInfer graph mismatch (1/6, difference 43), leaving 25/36 files unexecuted. B2 queues, C unexecuted; eight JIT conclusions retained including one empty job. AMD/Xeon/ARM64 passed previously. Main conflict and source/long-context review remain.

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

## #47278: Fresh helper dry-run remains blocked/required_main_missing. Required commit 50322ac7ba85df18111ec79d18a4c5c554df60c9 remains absent; ancestry diverges and live main conflict persists. Inherited scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflict evidence requires numerical reconciliation, not a routine CPU-only repair.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer integrates required main and reconciles scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflicts with numerical validation. Preserve BF16 partials and DCP/LSE. Resume helper recovery after safe integration; archive source/runs and retain combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
