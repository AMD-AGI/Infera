# Pending handoff for the next user session

Updated: **2026-11-05T00:39:36.880199+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Attempt 2 has 25 Base successes, 23 retained. B0 passed 38/38 files but omitted the historical failing VLM file. B1/B2 remain runnerless. Full Base and LLaVA recovery remain unestablished; excluded checks are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Follow B1/B2 without duplication. If LLaVA repeats, obtain blocked-thread stacks, worker count, CPU quota and parent memory. Preserve review/source qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/job/126744927116)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. This is historical evidence/review follow-up, not an established new maintainer hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Maintainers/yihou: assess native MI300 coverage alongside source-specific MI355X evidence and finish review. Allocate hardware only for a concrete coverage need.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Attempt 4 has 37 Base successes, 36 retained; B2 remains runnerless. Historical failing VLM launches bypassed the changed KV-events send. Shared-code review remains.

Tracking key: `pr46820-base-review`.

Next action: Follow B2. If LLaVA repeats, obtain submission stacks, worker count, CPU quota and parent memory. Maintainers: finish socket-contract review with inherited CPU evidence and existing approval.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/126714183803)

## #46830: Base attempt 3 has 26 successes, 22 retained. Inherited B1 passed 39/39 files; B2 remains runnerless. AMD has 16 successes, 12 retained, and two queued jobs. Full coverage and review/source qualifications remain.

Tracking key: `pr46830-base-review`.

Next action: Follow Base B2 and two AMD jobs without duplication. Maintainers: finish DSA deadlock/consumer review and preserve source evidence. Historical Qwen TTFT is not a DSA repair task or new hardware gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/job/126779404433)

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

## #47276: Attempt 3 retains 25 Base successes and six JIT results; two B jobs remain runnerless. Review, #45307 overlap, numerical qualifications and source/run archiving remain; optional path stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow queued jobs without duplication. Maintainers/yihou: finish review, coordinate overlap/landing order, archive exact source/run mappings and preserve numerical limits and optional deployment role.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6642701354)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41917456597)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Base attempt 3 now has 37 Base successes, 36 retained, plus six retained JIT results. B1 newly passed 39/39 files at 00:31 UTC; B2 remains runnerless. Prior request and successful handler are verified; full coverage is incomplete.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Follow queued B2 without duplication. Previous incidental Extra, AMD Extra, MLU and AMD Mori retries do not expand diagnostic scope.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6640175098)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41898283965)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/126813860175)

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

## #47872: Base retains 19 Base and three JIT successes; three B and three JIT jobs await runners and C has not run. AMD has 17 successes, 16 retained, with PP B queued; earlier retry C passed 6/6 files. Xeon mismatch and review/source/correctness/performance limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Follow Base/AMD PP coverage and align Xeon revisions. Finish review while preserving opt-in, exact selection count, underflow/overflow fallback, valid ties, source/run archiving, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/126847791613)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/126847836075)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover the shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon attempt 2 failed on missing req_types and disaggregation.mode; a cancelled sibling failed _paged_experts_num_resident. Logs confirm merge-checkout tests versus PR-branch installed package. Main retains the same workflow.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same commit, then recover Xeon shards. Preserve active Base; do not backport unrelated modules or rebase solely to refresh CI.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/126847836075)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/126847835959)
- [Evidence](https://github.com/sgl-project/sglang/blob/8083ca7ad7fdcaf2388a88196559b870e70e762a/.github/workflows/pr-test-xeon.yml)

## #47278: Fresh GitHub recheck reports mergeable=false. Main advanced to 1d062d2476f8144ef95533898f55c32829ecb257 without changing sparse MLA. Inherited comparison found seven numerical-kernel conflicts involving scaled-KV loads, block sizing, GROUP_W, cache/async-copy settings and split-K parameters. Compatibility remains unverified.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer: reconcile preserved conflicts, retain BF16 partials and DCP/LSE paths, then validate affected numerical cases before rewriting the branch. No speculative merge or GPU access performed.

- [Evidence](https://github.com/sgl-project/sglang/compare/a5d8b3af95d7915910547073b1771825981ccb98...1d062d2476f8144ef95533898f55c32829ecb257)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
