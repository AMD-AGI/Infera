# Pending handoff for the next user session

Updated: **2026-11-06T09:12:30.502087+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base has 20 functional successes, 15 retained; one B shard executes tests and two queue without steps. C unexecuted; JIT skipped. Full current-head Base and review remain required.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker preserves active Base. Maintainers require full current-head Base and review; retain native MI300/FlashInfer historical follow-up and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/2)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers assess native MI300 follow-up alongside source-specific MI355X evidence and required review.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base has 21 functional successes, 15 retained; two B shards queue. C unexecuted; JIT skipped. Inherited H100 B-large log passed 36/36 files. AMD has three queued jobs; ARM64/Xeon passed. Full Base and socket-contract review remain.

Tracking key: `pr46820-base-review`.

Next action: Worker follows active coverage; maintainers require full current-head Base and socket-contract review. Preserve exact-source/default-unbounded-HWM reproduction limits; historical FDFO/VLM attribution does not waive Base.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/127697566770)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)

## #46830: Base has 21 functional successes and one failed H100 FlashInfer MLA EAGLE draft CUDA Graph comparison: 1/6 mismatch, maximum absolute difference 43; 10 passed files, one failed, 25/36 unexecuted. One B shard queues; C unexecuted, JIT skipped. Test/runner blobs still match main c06e39c6; backend difference only relocates an import, not an established repair. Fixture selects FlashInfer rather than changed DSA backend. Inherited same-case pass on #46820 does not establish flakiness or root cause. AMD has three queued jobs; B200/source limits and platform/deadlock review remain.

Tracking key: `pr46830-base-review`.

Next action: Preserve active Base; use owner_requested recovery once terminal and helper-eligible. If repeated, CI/FlashInfer owners capture eager versus graph outputs for runner_eagle_draft_decode_mla_cuda_graph_chain, topk=1. Preserve fixture and maintainer commits; complete DSA consumer/deadlock and CUDA/ROCm review without speculative kernel edits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/job/127685657167)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/127697566770)
- [Evidence](https://github.com/sgl-project/sglang/blob/f69d354dd2649fc5f9962f5fe6ceec877637c1c0/test/registered/attention/unittests/mla/test_flashinfer.py#L216)

## #45524: Base has 19 functional successes and three B jobs queued; C/JIT unexecuted. AMD shard 6 exits 255 and fresh log retrieval still returns HTTP 404; first exception/test unknown. AMD has 11 stage successes and one queued shard; dependent stages skipped. ARM64/Xeon SPR passed; GNR queues. Historical AWQ timeout with 27/30 files unexecuted, health-check failures, PP/MLA errors, CPU fork_worker_preload mismatch and historical-source equivalence remain pending fresh coverage.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers obtain AMD shard 6 runner logs and recover coverage, then complete review/source-equivalence assessment. Preserve historical findings and frozen producer patch. Worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903737/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two B shards remain cancelled without steps after 36 functional successes. Cancellation policy remains unknown; fresh mergeability confirms main conflict. Validation at 9763ef56 does not qualify this head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers reconcile conflict, recover coverage, finish review and stock-image integration. Preserve frozen workaround and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Base has 19 functional successes (A10/B9), one B test running and three queued; C unexecuted. Four JIT unit suites and one benchmark passed; one success was 0/0 and two JIT jobs queue. No functional/JIT results retained from gate-only attempt 1. Optional deployment role, overlap coordination and numerical/source qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker follows active Base. Preserve optional llr37-disabled role, coordinate #45307 overlap, complete numerical review and archive exact source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 8 cancelled before B2 started; 37 functional successes and six JIT conclusions retained, including one empty job. Required-main failure and historical 31 unexecuted FDFO-shard files remain unresolved. Fresh helper refused required_main_missing; no retry was posted.

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

## #47872: Base has 20 functional successes, ten retained; one B job prepares and two queue. C unexecuted. Six inherited JIT successes include one empty job; four retained, one JIT test running and one queued. AMD has 15 stage successes and two queued. Inherited SM120, B200 and underflow evidence does not resolve review/source qualifications.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker follows active coverage. Yihou/maintainers finish review and exact-source mapping, preserving opt-in, min(length,2048), exact underflow/overflow fallback, valid ties, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124400918/job/127624252011)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Current main c06e39c6ca27c3d78b1696bb25d23b57ce5720ff contains required 50322ac7ba85df18111ec79d18a4c5c554df60c9; this PR does not. MIN_BASE_SHA persists despite issue 21065 being closed; inherited health check enforced it. Fresh helper refused required_main_missing before reservation/comment. Mergeability remains CONFLICTING; scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflicts need numerical validation.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles these conflicts with numerical validation, preserving BF16 partials and DCP/LSE. Worker resumes recovery after safe integration; retain exact-source and combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127597692465)
