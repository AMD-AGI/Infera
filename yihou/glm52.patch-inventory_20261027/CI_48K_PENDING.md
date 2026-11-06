# Pending handoff for the next user session

Updated: **2026-11-06T05:51:16.954901+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base has 17 functional successes, 15 retained; six B jobs queue, C/JIT unexecuted. Full Base and required review remain.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker preserves active coverage; maintainers require full current-head Base and review, retaining source-specific evidence limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/2)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers assess native MI300 follow-up alongside source-specific MI355X evidence and required review.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base has 19 functional successes, 15 retained; four B jobs queue, C/JIT unexecuted. Shared platform coverage and socket-contract review remain.

Tracking key: `pr46820-base-review`.

Next action: Worker follows active coverage. Maintainers require full Base and shared-code review; retain exact-source/default-unbounded-HWM reproduction limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)

## #46830: Base has 17 functional successes, three B suites executing and three queued; C unexecuted, JIT skipped. AMD has three queued jobs. Inherited B200 evidence includes eight skipped DSA cases; platform/deadlock review remains.

Tracking key: `pr46830-base-review`.

Next action: Preserve active Base/AMD, fixture and maintainer commits. Maintainers complete DSA consumer/deadlock and CUDA/ROCm review using scoped evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086232/attempts/2)

## #45524: Inherited Base has 35 functional successes. AWQ timeout leaves 27/30 files unexecuted; three siblings failed health checks. AMD PP/MLA failures, CPU fork_worker_preload mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers diagnose startup/PP failures, align CPU revisions, recover coverage and finish review. Preserve frozen producer patch pending source equivalence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two B shards cancelled without steps after 36 functional successes. Cancellation policy remains unknown; main conflict persists. Validation at 9763ef56 does not qualify this head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers reconcile conflict, recover coverage, finish review and stock-image integration. Preserve frozen workaround and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Base remains cooldown-gate-only with zero functional/JIT execution. Hourly slot is occupied until 06:43:59 UTC. Optional deployment role, overlap coordination and numerical/source qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker considers stale_gate after priority PRs and fresh cooldown checks. Preserve llr37-disabled role, coordinate #45307, complete numerical review and archive exact evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 8 cancelled before B2 started; 37 functional successes and six JIT conclusions retained, including one empty job. Required-main failure and historical 31 unexecuted FDFO-shard files remain unresolved.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer reconciles required main and numerical conflicts before fresh CI. Worker resumes recovery after safe integration; preserve source and combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127597692465)

## #47278: Inherited AMD attempt 2 passed 18 stage jobs, 14 retained. BF16/DCP/LSE review, exact source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou finish review and archive exact evidence. Passing AMD does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base remains successful with 27 functional jobs, four executed JIT unit suites and one benchmark; another JIT success ran 0/0. Scoped AMD MIOpen/PP failures and Xeon revision mismatch persist; ARM64 passed.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers repair database access, compare PP baselines and align CPU revisions. Preserve passing Base and separate review qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact evidence; preserve unequal paired-row/table-length, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Verified recovery produced Base attempt 2, retaining ten A successes and four JIT conclusions. B200 functional tests started; one JIT job installs dependencies, 12 B/three JIT jobs queue. No new pass yet. Historical CUDA-device failure and 79/80 unexecuted files remain unresolved; AMD queues and review/source qualifications remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker follows active coverage without duplication. Maintainers/yihou preserve opt-in, min(length,2048), exact fallback, valid ties, measured 32k regression and source-specific validation limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47872#issuecomment-6677932126)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170148805)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127735961988)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127585215489)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Current main f9436ef6449a272f4e3abdf6b8c263ef3eab1185 contains required 50322ac7ba85, but the PR does not. Runner rebase requirement and main conflict persist; numerical conflicts require validated reconciliation.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflicts, preserving BF16 partials and DCP/LSE with numerical validation.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127597692465)
