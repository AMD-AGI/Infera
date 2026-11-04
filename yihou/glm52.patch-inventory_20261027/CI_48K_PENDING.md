# Pending handoff for the next user session

Updated: **2026-11-04T19:10:50.956019+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Attempt 2 has 24 Base successes; three B jobs remain runnerless. Historical LLaVA/cancelled B coverage remains unresolved. Excluded checks are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Follow queued B jobs without duplication; inspect new failures and obtain worker stacks/count/quota/memory if LLaVA repeats. Preserve review/source qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/2)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. This is historical evidence/review follow-up, not an established new maintainer hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Maintainers/yihou: assess native MI300 coverage alongside source-specific MI355X evidence and finish review. Allocate hardware only for a concrete coverage need.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Attempt 4 retains 36 Base successes and its gate; two B jobs await runners. Historical VLM failure bypassed the changed KV-events send. Shared-code review remains.

Tracking key: `pr46820-base-review`.

Next action: Follow actual B execution; obtain submission stacks/count/quota/memory if repeated. Maintainers: finish socket-contract review with inherited CPU evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41868496790)

## #46830: Base attempt 3 has 25 successes: 22 retained plus three recovered C jobs. Newly inspected H100 Mooncake and GB300 logs pass; AMD MLA also passes, bringing AMD to 13 successes with 12 retained. Two Base B jobs await runners and one AMD B test runs. Earlier compiler/MLA causes remain unestablished; historical Qwen TTFT and DSA review remain separate.

Tracking key: `pr46830-base-review`.

Next action: Worker: follow the two queued Base B jobs and active AMD B test; inspect any new failure. Maintainers: finish DSA deadlock/consumer review. Preserve inherited source evidence; the historical Qwen tokenizer TTFT failure is not a DSA optimization or new hardware gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/job/126779403597)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/job/126779403846)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/job/126779423737)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/attempts/3)

## #45524: Base has 35 successes, AWQ timeout and health-check failures. AMD PP/MLA failures, runner-wait cancellation, CPU revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: compare baselines, align revisions, recover coverage and finish review. Preserve frozen producer patch; worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base jobs remain cancelled without execution. Source-specific 9763ef56 validation does not qualify the entire current head or integrated image. Excluded AMD CI is not claimed repaired.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover Base coverage, finish review and validate image integration. Preserve scoped MI355X evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Base remains cancelled with 25 successes and two unexecuted B jobs. Review, #45307 overlap, numerical qualifications and source/run archiving remain. Optional path stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Recover coverage after 19:34:17 UTC and eligible priority PRs. Maintainers/yihou: coordinate landing order, review and exact source/run archiving.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Base attempt 2 remains cancelled with 36 Base and six JIT successes retained; two B jobs never executed. This cycle helper refused retry_interval_not_elapsed without posting. Coverage remains incomplete.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Reevaluate owner_requested after 19:34:17 UTC and eligible priority PRs; inspect PR-wide side effects and preserve existing successes.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/2)

## #47278: AMD has 17 stage successes, including 14 retained and three earlier retry successes. One C job remains runnerless. BF16/DCP/LSE review and source/run qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Follow remaining AMD C coverage. Maintainers/yihou: preserve BF16 partials, DCP/LSE coverage, review, exact source/run archiving and combined-stack qualification.

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

## #47872: Base attempt 1 has 19 successes but remains failed with LLaVA/health-check failures and skipped C. AMD MIOpen/PP and Xeon mismatch persist. Review/source/correctness/performance limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Reevaluate owner_requested after 19:34:17 UTC and eligible priority PRs. Repair environment alignment; retain opt-in, min(length,2048) selections, exact fallbacks, valid ties, source/run archiving and measured performance limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125868991469)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125893452204)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover the shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon cannot import _paged_experts_num_resident; its cancelled sibling also failed req_types and disaggregation.mode imports. Checkout/package mismatch remains unresolved.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover the shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853412984)
