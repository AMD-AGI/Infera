# Pending handoff for the next user session

Updated: **2026-11-05T04:09:47.801447+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base attempt 3 retains 25 successes; B1/B2 remain runnerless without steps. Full coverage and historical LLaVA recovery remain unestablished. Excluded checks are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Follow active B1/B2 without duplication. If LLaVA repeats, collect result-wait stacks and runner resource evidence; preserve separate review/source qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/3)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. This is historical evidence/review follow-up, not an established new maintainer hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Maintainers/yihou: assess native MI300 coverage alongside source-specific MI355X evidence and finish review. Allocate hardware only for a concrete coverage need.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Base attempt 5 retains 37 successes while B2 waits without a runner. Historical LLaVA submit timeout bypassed the changed metrics send because KV events were disabled. Full coverage and shared-code review remain.

Tracking key: `pr46820-base-review`.

Next action: Follow B2 without duplication. Obtain blocked-submit stacks and runner resource evidence if failure repeats. Maintainers: finish socket-contract review using current-head approval and inherited CPU evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/5)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)

## #46830: Base attempt 4 retains 26 successes; B2 remains runnerless. Prior handler success is not suite success. AMD has 16 stage successes, 12 retained, and two queued jobs. Review/source qualifications remain.

Tracking key: `pr46830-base-review`.

Next action: Follow active Base/AMD jobs. Reconsider terminal recovery after 04:33:16 UTC. Maintainers: finish DSA consumer/deadlock review; unrelated Qwen TTFT is not a DSA repair task.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41962323161)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/attempts/3)

## #45524: Base LLaVA/AWQ failures, fail-fast siblings, AMD PP/MLA failures, runner-wait cancellation, CPU revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: compare failing baselines, align revisions, recover coverage and finish review. Preserve frozen producer patch; worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base jobs remain cancelled without execution; fresh mergeable=false/DIRTY reports a main conflict. Source-specific 9763ef56 validation does not qualify the entire current head/image. Excluded AMD CI is not claimed repaired.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: reconcile main, recover Base coverage, finish review and validate image integration. Preserve scoped MI355X evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Attempt 3 remains cancelled with 25 retained Base successes and six retained JIT results; B1/B2 never executed. Review, #45307 overlap, numerical qualifications and source/run archiving remain. Optional path stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Recover through helper after 04:33:16 UTC and actionable priority PRs. Maintainers/yihou: coordinate overlap/landing order, finish review and archive exact source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Base attempt 3 remains cancelled after runnerless B2 cancellation, with 37 successes, 36 retained, and six retained JIT results. Helper refused recovery because the author-wide interval has not elapsed.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Recover through helper after 04:33:16 UTC, after actionable priority PRs and fresh eligibility checks. Incidental PR-wide retries do not expand diagnostic scope.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/126987873769)

## #47278: AMD has 17 stage successes, 14 retained; C job 126709688128 is executing Run test without a result. BF16/DCP/LSE review and source/run qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Follow executing AMD C without duplication. Maintainers/yihou: preserve BF16 partials, DCP/LSE behavior, review, exact source/run archiving and combined-stack qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base passed 27 jobs and six JIT jobs. AMD MIOpen/PP failures and Xeon mismatch remain in shared-import scope; review/source/numerical qualifications remain.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers: repair MIOpen access, compare PP baselines and align Xeon revisions. Preserve passing Base and separate review qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base attempt 2 remains cancelled with 19 retained Base and three retained JIT successes. B0 began a test without a result; other cancelled test jobs never executed. AMD PP B remains queued; Xeon mismatch and review/source/correctness/performance limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Recover Base through helper after 04:33:16 UTC and actionable priority PRs; follow AMD PP B. Align Xeon revisions. Preserve opt-in, min(length,2048) selections, exact fallback, valid ties, archival mapping, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/126847867405)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/126847836075)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover the shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon failed on missing req_types and disaggregation.mode; a cancelled sibling failed _paged_experts_num_resident. Logs show merge-checkout tests versus PR-branch installed package. Main retains the workflow mismatch.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same commit, then recover Xeon. Do not backport unrelated modules or rebase solely to refresh CI.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/126847836075)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/126847835959)
- [Evidence](https://github.com/sgl-project/sglang/blob/8083ca7ad7fdcaf2388a88196559b870e70e762a/.github/workflows/pr-test-xeon.yml)

## #47278: Fresh mergeable=false/DIRTY confirms conflicts. Main 493c857736725a402fa42e1adc3f7e67ff300375 adds no sparse-MLA change since the prior audit. Seven inherited conflicts involving scaled-KV loads, block sizing, GROUP_W, cache/async-copy settings and split-K parameters remain numerically unqualified.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer: reconcile preserved conflicts, retain BF16 partials and DCP/LSE paths, then validate affected numerical cases before rewriting the branch.

- [Evidence](https://github.com/sgl-project/sglang/compare/dc65932d2ba1090fae016ada6b0b54d4f254b4b9...493c857736725a402fa42e1adc3f7e67ff300375)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
