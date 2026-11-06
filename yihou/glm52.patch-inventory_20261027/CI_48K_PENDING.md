# Pending handoff for the next user session

Updated: **2026-11-06T16:11:44.181130+00:00**.

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

## #46820: Base retains 27 functional successes, 24 from prior attempts; B2 still queues. AMD retains 16 stage successes with PP queued after transfer timeout/KeyError: answer; two PP files unexecuted and KV events disabled. Full Base and socket/source review remain.

Tracking key: `pr46820-base-review`.

Next action: Worker follows queued B2 and AMD PP; full current-head Base success remains required. Recover terminal unsuccessful Base through the helper after the local slot opens at 16:35:16 UTC, subject to fresh checks. Obtain transfer-peer evidence if PP repeats. Maintainers finish socket/source review, preserving default-unbounded-HWM limitations.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/127907757356)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659241/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659241/job/127627233721)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)

## #46830: Base attempt 3 newly failed with 22 functional successes, 21 retained. H100 FlashInfer graph comparison repeated the same 1/6 mismatch and maximum difference 43; 10/36 files passed, one failed and 25 were unexecuted; C never ran. Test/runner match main and dispatch uses FlashInfer MLA, not the changed DSA backend. Helper refused recovery because the hourly interval has not elapsed. AMD PP and B200/source/deadlock review remain.

Tracking key: `pr46830-base-review`.

Next action: Worker retries terminal Base through the helper no earlier than 16:35:16 UTC after fresh priority/head/attempt checks. If the H100 mismatch repeats, capture eager/graph output tensors and replay metadata for the upstream attention-test owner; do not weaken the test or infer a DSA regression. Obtain AMD PP transfer-peer evidence. Maintainers complete DSA consumer/deadlock and CUDA/ROCm review, retaining B200 and exact-source qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/job/127869572862)
- [Evidence](https://github.com/sgl-project/sglang/blob/c12719f69d0b4291d572322da30b7347d818b4ce/python/sglang/srt/layers/attention/attention_registry.py#L80)
- [Evidence](https://github.com/sgl-project/sglang/blob/c12719f69d0b4291d572322da30b7347d818b4ce/test/registered/attention/unittests/mla/test_flashinfer.py#L219)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086232/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830/files)

## #45524: Base fully passed 27 functional jobs; final B2 completed 9/9 files before this cycle. AMD exit-255 first exception remains unknown after another HTTP 404; disaggregation queues. Xeon/ARM64 passed. Historical AWQ/PP and source-equivalence review remain.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers preserve passing Base, retrieve missing AMD runner logs and follow queued disaggregation. Finish source-equivalence review, preserving historical AWQ/PP evidence and the frozen producer patch. Old CPU mismatch is not a current Xeon failure.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903737/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903737/job/127744404616)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Two B shards cancelled without tests after 36 successes. Annotation names github-actions[bot]; policy unknown. Finish failed on cancellation; main conflict persists. Validation at 9763ef56 does not qualify current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers reconcile conflict, recover cancelled shards, complete review and stock-image integration. Preserve frozen workaround and source-specific MI355X evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/job/125361049407)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Previously authorized request and successful handler produced active Base attempt 3, retaining 20 functional successes. Three B shards and one JIT benchmark still queue; C unexecuted. Seven retained JIT conclusions include one empty job. Prior FlashInfer failure left 25/36 files unexecuted; two B siblings never tested after health failures. Optional deployment, overlap and numerical/source qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker follows active Base without duplication; next author-wide request no earlier than 16:35:16 UTC. Capture eager/graph outputs if FlashInfer repeats. Preserve optional llr37-disabled role, coordinate #45307 overlap, complete numerical review and archive exact source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6688655255)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42244435986)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/127759704234)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/127759704220)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 8 cancelled before B2 started; 37 functional successes and six JIT conclusions retained, including one empty job. Finish failed on cancellation; policy unknown. Required main remains absent; historical FDFO shard left 31 files unexecuted.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer integrates required main with numerical conflict validation, preserving BF16 partials and DCP/LSE. Resume recovery after safe integration; retain source/run and combined-stack limits.

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

## #47872: Base attempt 3 retains 20 functional successes and eight JIT conclusions: five unit suites, two benchmarks and one empty job. Three B shards queue; C unexecuted. Prior FlashInfer mismatch left 25/36 files unexecuted and two B siblings skipped tests after health failures. AMD/ARM64/Xeon passed previously. Main conflict and source/long-context review remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker follows queued B jobs and downstream C without duplication. Recover future terminal Base only when eligible after 16:35:16 UTC and after priority PRs; investigate repeated FlashInfer mismatch. Yihou/maintainers reconcile main conflict without interrupting active CI and complete source/long-context review. Preserve opt-in, min(length,2048), exact underflow/overflow fallback, valid ties, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872#issuecomment-6687290656)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42234946547)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127735962144)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Required-main directive 50322ac7ba85df18111ec79d18a4c5c554df60c9 persists despite issue 21065 being closed. Fresh ancestry remains diverged and main conflicts. Prior helper refusal retained; scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflicts need numerical validation.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles required main with numerical validation, preserving BF16 partials and DCP/LSE. Resume recovery after safe integration; archive exact source/runs and retain combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
