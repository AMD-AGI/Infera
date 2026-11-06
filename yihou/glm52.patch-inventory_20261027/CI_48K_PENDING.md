# Pending handoff for the next user session

Updated: **2026-11-06T08:12:50.173972+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base has 20 functional successes, 15 retained; three B jobs queue. C unexecuted; JIT skipped. Full current-head Base and review remain required.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker preserves active Base; maintainers require full current-head Base and review. Retain native MI300/FlashInfer historical follow-up and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/2)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers assess native MI300 follow-up alongside source-specific MI355X evidence and required review.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base has 20 functional successes, 15 retained; three B jobs queue. C unexecuted; JIT skipped. AMD has three queued jobs; ARM64/Xeon passed. Full Base and shared socket-contract review remain.

Tracking key: `pr46820-base-review`.

Next action: Worker follows active coverage; maintainers require full current-head Base and socket-contract review. Preserve exact-source/default-unbounded-HWM reproduction limits; historical FDFO/VLM attribution does not waive Base.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/127697566668)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)

## #46830: Base has 20 functional successes, none retained; one B job executes tests and two queue. C unexecuted; JIT skipped. AMD has three queued jobs. Inherited B200 skips and platform/deadlock review remain.

Tracking key: `pr46830-base-review`.

Next action: Preserve active Base/AMD, fixture and maintainer commits. Maintainers complete DSA consumer/deadlock and CUDA/ROCm review using scoped evidence; unrelated Qwen TTFT is not a DSA repair task.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086232/attempts/2)

## #45524: Base has 18 functional successes and four B jobs queued; C/JIT unexecuted. AMD shard 6 exits 255 and its log still returns HTTP 404; first exception/test unknown. AMD has 11 stage successes and one queued shard; dependent stages skipped. ARM64/Xeon SPR passed; GNR queues. Historical AWQ timeout with 27/30 files unexecuted, health-check failures, PP/MLA errors, CPU fork_worker_preload mismatch and historical-source equivalence remain pending fresh coverage.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers obtain AMD shard 6 runner logs and recover coverage, then complete review/source-equivalence assessment. Preserve historical findings and frozen producer patch. Worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903737/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649574)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two B shards remain cancelled without steps after 36 functional successes. Cancellation policy remains unknown; fresh mergeability confirms main conflict. Validation at 9763ef56 does not qualify this head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers reconcile conflict, recover coverage, finish review and stock-image integration. Preserve frozen workaround and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Base has 15 functional successes (A10/B5), two B jobs executing and six queued; C unexecuted. Four JIT unit suites and one benchmark passed; one success was 0/0 and two JIT jobs queue. No functional/JIT results retained from gate-only attempt 1. Optional deployment role, overlap coordination and numerical/source qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker follows active Base. Preserve optional llr37-disabled role, coordinate #45307 overlap, complete numerical review and archive exact source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6678963481)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42176388426)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 8 cancelled before B2 started; 37 functional successes and six JIT conclusions retained, including one empty job. Required-main failure and historical 31 unexecuted FDFO-shard files remain unresolved.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer reconciles required main and numerical conflicts before fresh CI. Worker resumes recovery after safe integration; preserve source and combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127597692465)

## #47278: Inherited AMD attempt 2 remains successful with 18 executed stage jobs, 14 retained. BF16/DCP/LSE review, exact source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou finish review and archive exact evidence. Passing AMD does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base remains successful with 27 functional jobs, four executed JIT unit suites and one benchmark; another JIT success ran 0/0. Identity-verified inherited AMD MIOpen database/PP failures and Xeon checkout/package revision mismatch remain; ARM64 passed.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers repair database access, compare PP baselines and align CPU revisions. Preserve passing Base and separate review qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact evidence; preserve unequal paired-row/table-length, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base has 19 functional successes, ten retained; four B jobs queue. C unexecuted. Six JIT success conclusions include one empty job; four retained and two jobs queue. AMD has three queued jobs. Inherited SM120, B200 and underflow evidence does not resolve review/source qualifications.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker follows active coverage. Yihou/maintainers finish review and exact-source mapping, preserving opt-in, min(length,2048), exact underflow/overflow fallback, valid ties, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127735962640)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127735962613)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Current main c117e31f9f6b6984fddaae3fe574eb0580d62d61 contains required 50322ac7ba85df18111ec79d18a4c5c554df60c9; this PR does not. MIN_BASE_SHA persists despite issue 21065 being closed, and the inherited executed health check enforced it. Fresh mergeability is CONFLICTING; numerical conflicts require validated reconciliation.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflicts with numerical validation, preserving BF16 partials and DCP/LSE. Worker resumes recovery after safe integration; retain exact-source and combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127597692465)
