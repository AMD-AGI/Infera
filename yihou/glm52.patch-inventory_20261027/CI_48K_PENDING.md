# Pending handoff for the next user session

Updated: **2026-11-06T02:11:57.628092+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base was bulk-cancelled after 15 functional successes. Two B suites were interrupted, six never started; C/JIT skipped. Full current-head Base and historical FlashInfer coverage remain incomplete. Helper refused the hourly interval.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: recover after November 6 02:42:11 UTC, subject to priority ordering and fresh checks, until full current-head Base succeeds.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42145643976/job/127656339337)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers assess native MI300 follow-up alongside source-specific MI355X evidence and required review.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base was bulk-cancelled after 15 functional successes; two B suites were interrupted and six never started. Full Base, shared socket-contract review and exact-source/default-HWM evidence limits remain.

Tracking key: `pr46820-base-review`.

Next action: Worker: recover after November 6 02:42:11 UTC subject to priority ordering and fresh checks. Maintainers: require full current-head Base and finish shared socket-contract review; retain default-unbounded-HWM reproduction limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42145643976/job/127656339337)

## #46830: Base/AMD remain gate-only failures with zero tests. Helper refused the local hourly interval until at least 02:42:10.260739 UTC. Upstream eligibility must be rechecked. Maintainer history, fixture and resolved thread remain preserved.

Tracking key: `pr46830-base-review`.

Next action: Worker: prioritize stale_gate at the next eligible slot. Maintainers: finish DSA consumer/deadlock and CUDA/B200/ROCm review; preserve fixture and resolved comments. Do not expand the patch into unrelated tokenizer optimization.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086232/attempts/1)

## #45524: Inherited Base has 35 functional successes. AWQ timeout leaves 27/30 files unexecuted; three siblings fail health checks. AMD PP/MLA failures, CPU fork_worker_preload API mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: diagnose AWQ startup and PP failures, align installed/test CPU revisions, recover coverage and finish review. Preserve frozen producer patch pending source-equivalence evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two B shards were cancelled without steps after 36 functional successes. One annotation identifies github-actions[bot]; historical cancellation policy remains unknown. Main conflict remains CONFLICTING/DIRTY. Validation at 9763ef56 does not qualify this head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: reconcile main conflict, recover coverage, finish review and stock-image integration. Preserve frozen workaround and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Base remains cooldown-only with zero tests. Helper refuses the local hourly interval until at least 02:42:10.260739 UTC; upstream eligibility remains conditional. Optional deployment role, overlap coordination and numerical/source evidence remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: recover after actionable priority PRs and fresh cooldown checks. Maintainers/yihou: retain optional llr37-disabled role, coordinate #45307, complete numerical review and archive exact source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Previous launcher request and successful five-workflow handler are verified. Attempt 8 was bulk-cancelled before B2 started; 37 functional successes and six JIT conclusions were retained, including one empty JIT job. No new tests ran. Attempt 7 required-main failure and historical 31 unexecuted FDFO-shard files remain unresolved.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer: safely reconcile required main and numerical conflicts before fresh CI. Worker: recover after integration; preserve exact-source and combined-stack evidence limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42145588500)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127597692465)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42145643976/job/127656339337)

## #47278: Inherited AMD attempt 2 passed 18 stage jobs, 14 retained. BF16/DCP/LSE review, source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou finish review and archive exact evidence. Passing AMD does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base remains successful with 27 functional jobs, four executed JIT unit suites and one benchmark; another successful JIT job ran 0/0. Scoped AMD MIOpen/PP failures and Xeon revision mismatch persist; ARM64 passed.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers repair database access, compare PP baselines and align CPU revisions. Preserve passing Base and separate review qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact evidence; preserve unequal paired-row, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base is now bulk-cancelled: ten A successes, eleven B health-check failures, two B cancellations without steps, C skipped. JIT CUDA-device failure leaves 79/80 files unexecuted; three JIT jobs failed health checks. Two unit suites and one benchmark passed; H200 ran 0/0. Platform queues and review/source qualifications remain. Helper refused the hourly interval.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker: recover after priority PRs and the next eligible slot. Maintainers/yihou: retain opt-in, min(length,2048), exact underflow/overflow fallback, valid ties, measured 32k regression and source-specific MI355X limits; complete review/source mapping.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127585215489)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127585215431)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42145643976/job/127656339337)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers build/install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Fresh GraphQL remains CONFLICTING/DIRTY. Required 50322ac7ba85 is on main but absent from the tested PR. Attempt 7 failed that prerequisite; attempt 8 was cancelled before tests. Existing scaled-KV, block sizing, GROUP_W, cache/async-copy and split-K conflicts remain; unchanged retries cannot integrate main.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles preserved conflicts, retains BF16 partials and DCP/LSE, and validates affected numerical cases before rewriting. Worker resumes recovery after safe integration.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127597692465)
