# Pending handoff for the next user session

Updated: **2026-11-04T16:14:14.753208+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base ended cancelled: 23 successes, two unexecuted B cancellations, LLaVA worker-result failure and C HTTP 500. Helper refused hourly cooldown. Patch causality is unestablished; excluded non-Base tests are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: reevaluate helper recovery after 16:32:53 UTC, oldest eligible priority run first. Runner owner: obtain worker stacks/count/quota/memory if repeated. Preserve review and source-specific evidence constraints.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/job/126392603595)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/job/126683827365)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. This is historical evidence/review follow-up; no new maintainer hardware gate was established.

Tracking key: `pr33059-native-mi300-review`.

Next action: Maintainers/yihou: assess native MI300 coverage alongside author-reported MI355X results and finish review. Preserve Ready and source-specific limits; allocate hardware only for concrete additional coverage.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Attempt 3 retains 36 Base successes with no new Base execution. Both retry B jobs cancelled; finish failed while job/run API remains active. Historical VLM submission failure is unrecovered and its KV events were disabled. Shared-code review remains.

Tracking key: `pr46820-base-review`.

Next action: Worker: refresh terminal status and prioritize this oldest Base run after 16:32:53 UTC. Runner owner: capture submission-thread stacks/count/quota/memory if repeated. Retain socket-contract evidence without speculative VLM repairs.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/126685148590)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820#issuecomment-6633167088)

## #46830: Base ended cancelled with 22 successes, two B cancellations, Mooncake nvcc exit 127 and two C health-check failures. Compiler stderr/version remain unknown. AMD MLA startup -9, Qwen TTFT failures and one queue persist; affected launches bypass changed DSA behavior.

Tracking key: `pr46830-base-review`.

Next action: Worker: reevaluate recovery after the hourly deadline and older eligible priority runs. Runner owner: capture actual compiler diagnostics and compare passing peer. Preserve historical DSA review evidence; no tokenizer optimization, threshold relaxation or gratuitous rebase.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/job/126577528175)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/attempts/2)

## #45524: Base has 35 successes, AWQ timeout and three health-check failures. AMD has 13 successes, PP/MLA failures and a cancellation after 24h awaiting a runner. CPU revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: compare baselines, align installed/tested revisions, recover coverage and finish review. Preserve frozen producer patch; worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base H100 jobs remain cancelled before execution. Source-specific 9763ef56 validation does not qualify the entire head or integrated image. Previously excluded AMD CI is not claimed repaired.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover Base coverage, finish review and validate image integration. Preserve scoped MI355X evidence and frozen workaround; worker has no retry/branch authority.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Base ended cancelled with 25 successes and two unexecuted B jobs; all five JIT units and one benchmark passed. Review, #45307 overlap, numerical qualifications and source/run archiving remain; optional path stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: recover cancelled coverage after eligible priority PRs and the hourly deadline. Maintainers/yihou: coordinate overlap/landing order, finish review and archive exact source/run mappings; retain numerical limits and optional deployment role.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/job/126683783457)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Prior successful six-workflow handler verified. Base attempt 2 retains 36 Base and six JIT successes without new Base execution. Two B jobs report queued without runners/steps, but finish log reports cancelled B and exit 1. Recovery remains unverified.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Worker: refresh Base job/run consistency and actual B execution; do not duplicate the active attempt. Reevaluate unsuccessful termination under the hourly budget, priority PRs first.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/126685109215)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6634763693)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41858886910)

## #47278: New AMD retry log confirms 15/15 files passed, including previously failing test_vision_chunked_prefill.py in 328s. Fourteen other stage successes are retained. Job/workflow still reports active; terminal aggregation, BF16/DCP/LSE review and source/run qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Worker: verify AMD terminal result. Maintainers/yihou: preserve BF16 partials, DCP/LSE coverage, review, exact source/run archiving and combined-stack qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126675266178)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47897: Base passed 27 jobs, five JIT units and one benchmark. AMD MIOpen/PP failures and Xeon revision mismatch remain in shared-import scope. Review and source/numerical qualifications remain.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers: repair MIOpen access, compare PP baselines and align Xeon revisions. Preserve passing Base. Finish review with paired-row, dispatch/fallback and source qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base remains failed with 19 successes, LLaVA worker-result/startup failure, two health-check failures and skipped C. Three JIT units pass; two units/benchmark failed health checks. AMD MIOpen/PP and Xeon mismatch persist. Hourly deadline is 16:32:53 UTC; priority PRs precede this run. Review/source/correctness/performance limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker: reevaluate owner_requested after eligible priority PRs and the hourly deadline; disclose PR-wide effects. CI maintainers: repair environment/source alignment. Maintainers/yihou: retain opt-in, min(length,2048) selections, exact fallbacks, valid ties, source/run archiving and measured performance limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125868991469)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125893452204)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover the shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon cannot import _paged_experts_num_resident; its cancelled sibling also failed req_types and disaggregation.mode imports. Checkout/package mismatch remains unresolved by the architecture API repair.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover the shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853412984)
