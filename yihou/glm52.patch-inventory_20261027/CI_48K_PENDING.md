# Pending handoff for the next user session

Updated: **2026-11-06T03:48:29.984810+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base remains cancelled after 15 functional successes. Two B suites were interrupted and six never started; C/JIT skipped. Complete current-head Base remains required.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: retry after 2026-11-06 04:43:14 UTC with fresh eligibility checks. Maintainers: require full Base and review; preserve historical MI300/FlashInfer context and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42145643976/job/127656339337)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers assess native MI300 follow-up alongside source-specific MI355X evidence and required review.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base attempt 2 retains the passing gate and 15 functional successes; eight B jobs queue without steps. No new tests started; C/JIT unexecuted. Full Base and shared socket-contract review remain pending.

Tracking key: `pr46820-base-review`.

Next action: Worker follows active attempt 2. Maintainers require complete current-head Base and socket-contract review; preserve exact-source/default-unbounded-HWM reproduction limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820#issuecomment-6676279407)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42158237565)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)

## #46830: Base attempt 2 has ten executed A successes and 13 B jobs queued without steps; C unexecuted and JIT skipped. AMD has eight stage successes, four running and one queued. Full coverage and review remain pending.

Tracking key: `pr46830-base-review`.

Next action: Preserve active attempts. Maintainers finish DSA consumer/deadlock and CUDA/B200/ROCm review, preserving the fixture and resolved comments. Unrelated Qwen TTFT is not a DSA repair task.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086232/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830#issuecomment-6675426030)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42152065630)

## #45524: Inherited Base has 35 functional successes. AWQ timeout leaves 27/30 files unexecuted; three siblings fail health checks. AMD PP/MLA failures, CPU fork_worker_preload API mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers diagnose startup/PP failures, align installed/test CPU revisions, recover coverage and finish review. Preserve the frozen producer patch pending source equivalence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two B shards cancelled without steps after 36 functional successes. Historical cancellation policy remains unknown. Main conflict persists. Validation at 9763ef56 does not qualify this head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers reconcile main conflict, recover coverage, finish review and stock-image integration. Preserve frozen workaround and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Base remains cooldown-gate-only with zero functional/JIT tests. Priority recovery occupies the hourly slot. Optional deployment role, overlap coordination and numerical/source evidence remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker considers stale_gate after priority PRs and 2026-11-06 04:43:14 UTC, with fresh upstream checks. Maintainers/yihou retain optional llr37-disabled role, coordinate #45307 and complete numerical review/source archiving.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 8 cancelled before B2 started; 37 functional successes and six JIT success conclusions are retained, including one empty job. No new tests executed. Required-main failure and historical 31 unexecuted FDFO-shard files remain unresolved.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer reconciles required main and numerical conflicts before fresh CI. Worker resumes recovery after integration; preserve exact-source and combined-stack limits.

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

## #47872: Base retains ten A successes, eleven B health-check failures, two cancellations without steps and skipped C. JIT CUDA-device failure leaves 79/80 files unexecuted; three JIT jobs failed health checks. Two unit suites and one benchmark passed; another job ran 0/0. Platform queues and review/source qualifications remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker recovers after priority PRs and 2026-11-06 04:43:14 UTC with fresh checks. Maintainers/yihou finish review/source mapping; preserve opt-in, min(length,2048), exact underflow/overflow fallback, valid ties, 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127585215489)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127585215431)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42145643976/job/127656339337)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers build/install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Freshly queried main c843608d390a contains required 50322ac7ba85, but the tested PR does not. Runner rebase requirement and CONFLICTING state persist. Preserved numerical conflicts require validated reconciliation; unchanged retries cannot integrate the prerequisite.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflicts, retains BF16 partials/DCP/LSE and validates numerical cases before rewriting. Worker resumes recovery after safe integration.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127597692465)
