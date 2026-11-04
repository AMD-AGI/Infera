# Pending handoff for the next user session

Updated: **2026-11-04T15:44:01.468623+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base remains active with 23 successes, two B queues, LLaVA worker-result failure and separate C HTTP 500. Later submission timeout/port conflicts are secondary. Same-source peer passed; this PR remains unrecovered. Previously excluded non-Base tests are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Follow the active attempt; after unsuccessful termination use eligible owner_requested recovery. Runner owner: capture worker stacks/count/quota/memory if repeated. Preserve review and source-specific evidence constraints.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/job/126392603595)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/job/126434007693)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. This is historical evidence/review follow-up; no new maintainer hardware gate was established.

Tracking key: `pr33059-native-mi300-review`.

Next action: Maintainers/yihou: assess native MI300 coverage alongside author-reported MI355X results and finish review. Preserve Ready and source-specific limits; allocate hardware only for concrete additional coverage.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Attempt 3 retains 36 Base successes; two B jobs still have no runner/steps. Existing PR-wide request and successful handler verified. Historical submission timeout remains unrecovered; KV events were disabled. AMD/Xeon/ARM64 pass; shared-code review remains.

Tracking key: `pr46820-base-review`.

Next action: Follow actual B execution without duplication. If terminal unsuccessful, reevaluate hourly recovery. Capture submit-thread stacks, worker count/quota and memory if repeated; retain socket-contract evidence without speculative VLM repairs.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820#issuecomment-6633167088)

## #46830: Base remains active: 22 successes, two B queues, Mooncake nvcc exit 127 and two C health-check failures. Passing peer matches source/package versions, but actual compiler version/stderr is unknown. AMD MLA startup -9, Qwen TTFT failure and one queue persist; failing launches bypass changed DSA behavior.

Tracking key: `pr46830-base-review`.

Next action: Follow active attempts and use eligible recovery after termination. Runner owner: capture actual compiler diagnostics and compare passing peer. Retain bounded MLA/TTFT follow-up and historical DSA review evidence; no tokenizer optimization, threshold relaxation or gratuitous rebase.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/job/126577528175)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/job/126578054895)

## #45524: Base has 35 successes, AWQ timeout and three health-check failures. AMD has 13 successes, PP/MLA failures and a cancellation after 24h awaiting a runner. CPU revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: compare baselines, align installed/tested revisions, recover missing coverage and finish review. Preserve frozen producer patch; worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base H100 jobs remain cancelled before execution. Source-specific 9763ef56 validation does not qualify the entire head or integrated image. Previously excluded AMD CI is not claimed repaired.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover Base coverage, finish review and validate image integration. Preserve scoped MI355X evidence and frozen workaround; worker has no retry/branch authority.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Base has 25 successes; two B jobs await runners. Four JIT units and one benchmark passed; the remaining JIT unit now executes tests. Review, #45307 overlap, numerical evidence and source/run archiving remain; optional path stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: follow two B queues and the running JIT unit. Maintainers/yihou: finish review, coordinate #45307 overlap/landing order and archive exact source/run mappings; preserve numerical qualifications and keep this optional path disabled in frozen llr37.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Recovery request accepted: successful six-workflow handler and Base attempt 2 verified. All 36 Base and six JIT successes are retained; two formerly cancelled H100 B jobs are queued without runners/steps. Coverage remains incomplete; no new Base failure or pass.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Worker: follow actual Base B execution without duplication. Reevaluate only after unsuccessful termination and the author-wide budget; next slot no earlier than 16:32:53 UTC, priority PRs first.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6634763693)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41858886910)

## #47278: AMD attempt 2 now executes tests for the previous VLM pool-submission/startup failure; 14 stage successes are retained. No new AMD test outcome yet. BF16/DCP/LSE and exact-source constraints remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Worker: inspect actual AMD test execution and any first new failure. Runner owner: compare repeated VLM startup failure with an executed matching baseline. Maintainers/yihou: preserve BF16 partials, DCP/LSE coverage, review, source/run archiving and combined-stack qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)
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

## #47872: Base remains failed with 19 successes, LLaVA worker-result/startup failure, two health-check failures and skipped C. Three JIT units pass; remaining JIT jobs failed health checks. AMD MIOpen/PP and Xeon mismatch persist. #47278 used the available hourly slot; next local deadline is 16:32:53 UTC. Review/source correctness/performance limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker: reevaluate owner_requested after 16:32:53 UTC, checking priority PRs and older eligible runs first; disclose PR-wide effects. CI maintainers: align Xeon revisions and repair MIOpen/PP environment issues. Maintainers/yihou: retain opt-in, exactly min(length,2048) selections, underflow/overflow fallback, valid boundary ties, source/run archiving, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125868991469)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125893452204)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover the affected shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon cannot import _paged_experts_num_resident; its cancelled sibling also failed req_types and disaggregation.mode imports. Checkout/package mismatch remains unresolved by the architecture API repair.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover the affected shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853412984)
