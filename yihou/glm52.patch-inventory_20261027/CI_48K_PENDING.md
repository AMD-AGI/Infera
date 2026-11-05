# Pending handoff for the next user session

Updated: **2026-11-05T02:43:16.887841+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base attempt 3 retains 25 Base successes and its passing gate; B1/B2 remain runnerless with no steps. Recovery request and successful PR-wide handler are verified, but coverage and historical LLaVA recovery remain unestablished. Excluded checks are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: follow attempt 3 without duplication. If LLaVA repeats, collect submit stacks, worker count/quota and parent memory; preserve review/source qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6648090936)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41955932043)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. This is historical evidence/review follow-up, not an established new maintainer hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Maintainers/yihou: assess native MI300 coverage alongside source-specific MI355X evidence and finish review. Allocate hardware only for a concrete coverage need.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Prior retry chain remains verified. Base attempt 5 retains 37 successes while B2 waits without a runner. Historical LLaVA submit timed out with kv_events_config=None, bypassing the modified metrics send. Full Base coverage and upstream shared-code review remain.

Tracking key: `pr46820-base-review`.

Next action: Worker: follow B2 without duplication; if LLaVA repeats, collect blocked-submit stacks, worker count/quota and parent memory. Maintainers: complete shared socket-contract review using current-head approval and inherited CPU evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/5)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820#issuecomment-6647135942)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41949224712)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/126714183803)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)

## #46830: Base attempt 3 ended after runnerless B2 cancellation; 26 successes, 22 retained. AMD has 16 stage successes, 12 from the previous attempt, with two queued jobs. Full coverage and review/source qualifications remain.

Tracking key: `pr46830-base-review`.

Next action: Worker: helper refused recovery until 2026-11-05 03:32:53 UTC; recheck then with priority ordering and follow queued AMD jobs. Maintainers: finish DSA deadlock/consumer review. Historical Qwen TTFT is not a DSA repair task.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/job/126987490650)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/attempts/3)

## #45524: Base B0 first logged a LLaVA image-result timeout and image-text assertion before the terminating AWQ timeout; fail-fast siblings, AMD PP/MLA failures, runner-wait cancellation, CPU revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: compare VLM/AWQ/PP/MLA baselines, align revisions, recover coverage and finish review. Preserve frozen producer patch; worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base jobs remain cancelled without execution. Source-specific 9763ef56 validation does not qualify the entire current head or integrated image. Excluded AMD CI is not claimed repaired.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover Base coverage, finish review and validate image integration. Preserve scoped MI355X evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Attempt 3 ended cancelled with 25 retained Base successes and six retained JIT results; B1/B2 never executed. Review, #45307 overlap, numerical qualifications and source/run archiving remain; optional path stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: retry cancelled Base through the helper no earlier than 2026-11-05 03:32:53 UTC, after actionable priority PRs, with fresh eligibility checks. Maintainers/yihou: finish review, coordinate #45307 overlap/landing order, archive exact source/run mappings and retain numerical qualifications and the optional role, disabled in frozen llr37.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6642701354)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41917456597)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Base attempt 3 ended after runnerless B2 cancellation, with 37 Base successes, 36 retained, and six retained JIT results. Inherited B1 passed 39/39 files. Full coverage remains incomplete.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Recover through the helper no earlier than 2026-11-05 03:32:53 UTC, after actionable priority PRs. Incidental PR-wide retries do not expand diagnostic scope.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/126987873769)

## #47278: AMD has 17 stage successes, 14 retained, and one runnerless C job. BF16/DCP/LSE review and source/run qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Follow AMD C coverage. Maintainers/yihou: preserve BF16 partials, DCP/LSE behavior, review, exact source/run archiving and combined-stack qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)

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

## #47872: Base attempt 2 ended cancelled with 19 retained Base and three retained JIT successes. B0 started test_spec_eagle_fa3.py but produced no result before cancellation; two other B and three JIT jobs never executed, and C skipped. AMD PP B remains queued; Xeon mismatch and review/source/correctness/performance limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker: recover cancelled Base no earlier than 2026-11-05 03:32:53 UTC, after actionable priority PRs; follow AMD PP B. CI maintainers: align Xeon installed/tested revisions. Maintainers/yihou: finish review; preserve opt-in, exactly min(length,2048) selections, underflow/overflow fallback, valid ties, source/run archiving, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/126847867405)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/126847836075)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover the shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon attempt 2 failed on missing req_types and disaggregation.mode; a cancelled sibling failed _paged_experts_num_resident. Logs confirm merge-checkout tests versus PR-branch installed package. Main retains the same workflow.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same commit, then recover Xeon shards. Recover Base through the helper when eligible; do not backport unrelated modules or rebase solely to refresh CI.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/126847836075)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/126847835959)
- [Evidence](https://github.com/sgl-project/sglang/blob/8083ca7ad7fdcaf2388a88196559b870e70e762a/.github/workflows/pr-test-xeon.yml)

## #47278: Fresh mergeability remains false. Main was audited at 242d15673ec74f5ae783334a246cb7969c1c9e30; its new model-support increment does not touch sparse MLA. Inherited seven numerical-kernel conflicts involving scaled-KV loads, block sizing, GROUP_W, cache/async-copy settings and split-K parameters remain unqualified.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer: reconcile preserved conflicts, retain BF16 partials and DCP/LSE paths, then validate affected numerical cases before rewriting the branch. No speculative merge or GPU access performed.

- [Evidence](https://github.com/sgl-project/sglang/compare/bf40ca98f4287347ec8d19662aa390eacc253a1e...242d15673ec74f5ae783334a246cb7969c1c9e30)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
