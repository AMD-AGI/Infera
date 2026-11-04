# Pending handoff for the next user session

Updated: **2026-11-04T17:45:53.668112+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Recovery request and six handler-triggered reruns verified. Base attempt 2 retains 23 real passes; three B jobs and one C job await runners with no new test steps. Handler log reaches cleanup but run/check APIs still report queued. Historical LLaVA timeout, GitHub HTTP 500 and cancelled coverage remain unresolved; excluded non-Base tests are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: follow the four queued Base jobs without duplication and reconcile handler status; inspect the first new failure. Runner owner: obtain LLaVA worker stacks/count/quota/memory if repeated. Preserve separate review/source qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6637510859)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41878556259)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/job/126392603595)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. This is historical evidence/review follow-up; no new maintainer hardware gate was established.

Tracking key: `pr33059-native-mi300-review`.

Next action: Maintainers/yihou: assess native MI300 coverage alongside source-specific author MI355X results and finish review. Preserve Ready; allocate hardware only for a concrete coverage need.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Base attempt 4 is verified after successful five-workflow handler. All 36 Base successes and the gate are retained; two B jobs await runners without new test steps. Historical VLM failure bypassed the changed KV-events send. Shared-code review remains.

Tracking key: `pr46820-base-review`.

Next action: Worker: follow actual B execution without duplication. Runner owner: capture submission stacks/count/quota/memory if repeated. Maintainers: finish shared socket-contract review with inherited CPU evidence.

- [Evidence](https://github.com/sgl-project/sglang/pull/46820#issuecomment-6636121578)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41868496790)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/4)

## #46830: Base remains cancelled: 22 real Base passes, two unexecuted B cancellations, Mooncake nvcc exit 127 and two C health-check failures. Helper refused retry until 18:33:42 UTC. Tested dependency installer matches current main and already selects Mooncake 0.3.14; no missing integration repair established. AMD MLA startup -9, Qwen TTFT failures and one queued job remain; compiler stderr/version are unknown. Failing launches bypass the changed DSA path.

Tracking key: `pr46830-base-review`.

Next action: Worker: reconsider recovery after 18:33:42 UTC, oldest eligible priority run first. Runner owner: capture actual nvcc version/child stderr and compare the passing peer. Maintainers: finish DSA deadlock/consumer review. Do not relax thresholds or turn Qwen TTFT into tokenizer work.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/job/126577528175)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/attempts/2)

## #45524: Base has 35 successes, AWQ timeout and three health-check failures. AMD has 13 successes, PP/MLA failures and runner-wait cancellation. CPU revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: compare baselines, align installed/tested revisions, recover coverage and finish review. Preserve frozen producer patch; worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base jobs remain cancelled before execution. Source-specific 9763ef56 validation does not qualify the entire head or integrated image. Previously excluded AMD CI is not claimed repaired.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover Base coverage, finish review and validate image integration. Preserve scoped MI355X evidence and frozen workaround; worker has no retry/branch authority.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Base ended cancelled with 25 successes and two unexecuted B jobs; five JIT units and one benchmark passed. Review, #45307 overlap, numerical qualifications and source/run archiving remain. Optional path stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: recover coverage after 18:33:42 UTC and eligible priority PRs. Maintainers/yihou: coordinate landing order, finish review and archive exact source/run mappings; retain numerical limits and optional deployment role.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Base attempt 2 remains cancelled: 36 Base and six JIT successes retained, two B jobs cancelled without execution, finish failed. Coverage remains incomplete.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Worker: reevaluate helper recovery after 18:33:42 UTC and eligible priority PRs; preserve successful results and verify actual B execution.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/126685109215)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6634763693)

## #47278: AMD has 16 successful stage jobs: 14 retained plus recovered VLM B and first new C shard. Two C jobs remain queued. Full completion, BF16/DCP/LSE review and source/run qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Worker: follow remaining AMD C jobs. Maintainers/yihou: preserve BF16 partials, DCP/LSE coverage, review, exact source/run archiving and combined-stack qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126675266178)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688068)

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

## #47872: Base remains failed with 19 successes, LLaVA failure, two health-check failures and skipped C. Three JIT units passed; two units/benchmark failed health checks. AMD MIOpen/PP and Xeon mismatch persist. Review/source/correctness/performance limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker: reevaluate owner_requested after 18:33:42 UTC and eligible priority PRs; disclose PR-wide effects. CI maintainers: repair environment/source alignment. Maintainers/yihou: retain opt-in, min(length,2048) selections, exact fallbacks, valid ties, source/run archiving and measured performance limits.

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
