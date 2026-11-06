# Pending handoff for the next user session

Updated: **2026-11-06T13:41:27.970987+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Current-head Base is fully recovered: attempt 2 passed 28 functional jobs, with 15 retained; final B2 passed 9/9 files. The remaining part of this mixed handoff is maintainer review (REVIEW_REQUIRED), not incomplete Base CI.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Maintainers review and merge the current-head patch now that full Base has passed. Preserve native MI300/FlashInfer historical follow-up and source-specific MI355X evidence limits without adding a hardware gate; reviewDecision remains REVIEW_REQUIRED.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/job/127716852500)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059/files)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers assess native MI300 follow-up alongside source-specific MI355X evidence and required review.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base attempt 3 has 26 functional successes, including 24 retained. H100 C passed Mooncake EP and Eagle DP (3/3 files), resolving that coverage gap without establishing the old JIT failure cause. B2 and GB300 C still queue. AMD retains 16 successes and queues PP after transfer timeout/auxiliary-send failure/KeyError: answer; two files remain uncovered. KV events were disabled. Full Base, shared-platform coverage and socket/source review remain.

Tracking key: `pr46820-base-review`.

Next action: Worker follows queued Base attempt 3 and AMD attempt 2; no duplicate request. Local retry interval ends at 14:03:35 UTC, subject to fresh eligibility. Complete full Base and shared-platform coverage; if AMD repeats, capture transfer-peer errors. Maintainers finish socket review, preserving exact-source and default-unbounded-HWM limitations.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/127907757539)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/127907757209)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820#issuecomment-6685453289)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42222439678)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659241/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659241/job/127627233721)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)

## #46830: Base attempt 3 retains 21 functional successes; two B shards queue, C unexecuted, JIT skipped. Prior FlashInfer eager/graph mismatch left 25/36 files unexecuted; test and runner harness match current main, with no established merged repair. AMD PP auxiliary-transfer failure/KeyError: answer left two files unexecuted and bypassed changed DSA code. B200/source/deadlock review remains.

Tracking key: `pr46830-base-review`.

Next action: Worker follows active Base and uses the helper after terminal failure when eligible. If repeated, CI/FlashInfer owners capture eager/graph outputs; CI/Mooncake owners compare PP baselines and transfer-peer errors. Complete DSA consumer/deadlock and CUDA/ROCm review, retaining B200/source limits and maintainer/fixture commits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/job/127685657167)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086232/job/127683946037)
- [Evidence](https://github.com/sgl-project/sglang/blob/f69d354dd2649fc5f9962f5fe6ceec877637c1c0/test/registered/amd/disaggregation/test_disaggregation_pp.py)

## #45524: Base has 26 inherited functional successes and one B2 shard queued. AMD exit-255 runtime error remains unknown because the previous log request returned HTTP 404 BlobNotFound; AMD disaggregation queues. Xeon GNR tests are executing. Historical AWQ unexecuted coverage, PP/MLA/health-check failures, CPU fork_worker_preload mismatch and source equivalence remain pending.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers follow active Base/GNR, obtain the missing AMD runner log and recover uncovered suites. Complete review/source-equivalence assessment; retain historical AWQ/PP/CPU-revision findings and the frozen producer patch. Worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903737/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two B shards cancelled without steps after 36 functional successes. Annotation names github-actions[bot]; cancellation policy unknown. Finish failed on cancellation; main conflict persists. Validation at 9763ef56 does not qualify current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers reconcile main conflict, recover cancelled shards, finish review and stock-image integration. Preserve frozen workaround and source-specific MI355X evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/job/125361049407)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Base has 20 functional successes. FlashInfer failure left 25/36 files unexecuted; another B failed only its sibling-health gate, one B queues, C unexecuted. JIT: five executed unit suites, one benchmark, one empty job and one gate-blocked benchmark. Optional deployment, overlap and numerical/source qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker retries terminal Base when eligible after actionable priority PRs. CI/FlashInfer owners capture eager/graph outputs if repeated. Preserve optional llr37-disabled role, coordinate #45307 overlap, complete numerical review and archive exact source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/127759704234)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/127759704308)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 8 cancelled before B2 started; 37 functional successes and six JIT conclusions retained, including one empty job. Finish failed on cancellation; policy unknown. Required main remains absent, prior helper refusal persists, and historical FDFO shard left 31 files unexecuted.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer integrates required main with numerical validation of kernel conflicts, preserving BF16 partials and DCP/LSE. Resume recovery after safe integration and retry eligibility; retain source/run and combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127656329036)

## #47278: Inherited AMD attempt 2 passed 18 executed stage jobs, 14 retained. BF16/DCP/LSE review, exact source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou finish review and archive exact evidence. Passing AMD does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base passed 27 functional jobs, four executed JIT unit suites and one benchmark; another JIT job ran 0/0. Verified inherited AMD MIOpen database/PP failures and Xeon checkout/package mismatch remain; ARM64 passed.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers repair database access, compare PP baselines and align CPU revisions. Preserve passing Base and separate review qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact evidence; preserve unequal paired-row/table-length, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base has 20 functional successes, ten retained. FlashInfer failure left 25/36 files unexecuted; another B failed its sibling-health gate, one B queues, C unexecuted. JIT: five executed unit suites, two benchmarks and one empty job, four conclusions retained. AMD/ARM64/Xeon passed. Main conflict and source/long-context review remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker retries terminal Base when eligible after actionable priority PRs. CI/FlashInfer owners investigate eager/graph mismatch. Yihou/maintainers reconcile main conflict safely and finish review/source mapping, preserving opt-in, min(length,2048), exact underflow/overflow fallback, valid ties, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127735962144)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127735962110)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Required-main directive 50322ac7ba85df18111ec79d18a4c5c554df60c9 persists despite issue 21065 being closed; fresh ancestry remains diverged and main conflicts. Prior helper refusal retained. Scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflicts require numerical validation.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles required main and kernel conflicts with numerical validation, preserving BF16 partials and DCP/LSE. Worker resumes recovery after safe integration; archive exact source/runs and retain combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
