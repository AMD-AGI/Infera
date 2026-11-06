# Pending handoff for the next user session

Updated: **2026-11-06T01:42:11.573010+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Current head has 15 passing Base functional jobs (A10/B5); one B job installs dependencies and seven queue. C has not executed; JIT skipped. Full Base and historical FlashInfer coverage remain pending.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: preserve active Base and recover terminal failures until full current-head success. Maintainers: complete required review with current approval; retain native MI300 follow-up as historical context and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/1)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this remains historical follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers assess native MI300 follow-up alongside source-specific MI355X evidence and review.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Maintainer head 4890a5aa has 15 passing Base functional jobs, two executing B suites and six queued B jobs. Full Base, shared socket-contract review and exact-source/default-unbounded-HWM evidence limits remain.

Tracking key: `pr46820-base-review`.

Next action: Worker: preserve active runs and recover terminal Base failures. Maintainers: complete shared-code/socket-contract review and retain exact-source/default-HWM limits. Full current-head Base remains required.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)

## #46830: Current Base/AMD remain gate-only failures with zero tests. Fresh helper refusal requires waiting until November 6 02:07:05 UTC, citing Base 42134452467; no reservation/comment/new attempt. Maintainer history, fixture and resolved review thread remain preserved.

Tracking key: `pr46830-base-review`.

Next action: Worker: prioritize stale_gate after the conditional deadline and fresh checks. Maintainers: finish DSA consumer/deadlock and CUDA/B200/ROCm review; preserve resolved comments and fixture. Unrelated Qwen tokenizer TTFT is not a DSA repair task.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086232/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/1)

## #45524: Inherited Base has 35 functional successes. AWQ timeout leaves 27/30 files unexecuted; three siblings fail health checks. Scoped AMD PP/MLA failures, CPU fork_worker_preload API mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: diagnose AWQ startup and PP failures, align installed/test CPU revisions, recover coverage and finish review. Preserve frozen producer patch until historical-source equivalence is established.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two B shards were cancelled without steps after 36 functional successes. One annotation identifies github-actions[bot]; policy remains unknown. Fresh GraphQL CONFLICTING/DIRTY confirms the main conflict. Validation at 9763ef56 does not qualify this head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: reconcile main conflict, recover cancelled coverage, finish review and stock-image integration. Preserve frozen workaround and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Current Base remains cooldown-only with zero tests. Fresh helper refusal requires waiting until November 6 02:07:05 UTC, citing Base 42134452467; no reservation/comment. Historical CPU evidence does not qualify current GPU numerics.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: recover after actionable priority PRs and fresh cooldown checks. Maintainers/yihou: retain optional llr37-disabled role, coordinate #45307 overlap, complete numerical review and archive exact source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/1)

## #47278: Attempt 7 completed with failure: B2 passed health checks, then failed required-main ancestry before tests. All 37 functional successes and six JIT conclusions are retained; one JIT job was empty. Historical FDFO failure left 31 files unexecuted. Required integration and numerical conflicts remain.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer: reconcile required main and numerical conflicts, preserve BF16 partials and DCP/LSE, and validate affected cases before rewriting. Worker: recover fresh CI after safe integration. Archive exact source/runs and preserve combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/7)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127597692465)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

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

## #47872: Base has ten A successes, two executed JIT unit successes and one benchmark; H200 ran 0/0. CUDA-device failure leaves 79/80 files unexecuted. Eleven B jobs and two other JIT jobs failed health checks; two B jobs and one JIT job queue. C skipped. AMD has 14 stage successes and three queued; ARM64/SPR passed, GNR queues. Review/source qualifications remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker: preserve active attempts and recover terminal Base after priority PRs. Maintainers/yihou: finish review/source mapping; retain opt-in, min(length,2048), exact underflow/overflow fallbacks, valid ties, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127585215489)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127585215483)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124400918/attempts/1)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. This PR's checkout/package mismatch remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers build/install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Fresh GraphQL confirms CONFLICTING/DIRTY. Main includes required 50322ac7ba85; PR comparison is diverged and attempt 7 failed that prerequisite before tests. Previously established conflicts span scaled-KV, block sizing, GROUP_W, cache/async-copy and split-K; an unchanged retry cannot reconcile them.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles preserved conflicts, retains BF16 partials and DCP/LSE, and validates affected numerical cases before rewriting. Worker resumes CI recovery after safe integration.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127597692465)
