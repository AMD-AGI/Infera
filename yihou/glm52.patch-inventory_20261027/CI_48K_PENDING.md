# Pending handoff for the next user session

Updated: **2026-11-04T13:13:04.232932+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base has 23 actual successes, two B jobs queued, a new VLM worker-result timeout and the previous C GitHub HTTP 500 before tests. FA3 LLaVA does not execute the changed Aiter/DSA path. Pool methods match current main; #47120 is not an established remedy. The inspected green main Base candidate skipped all functional coverage. Integrated dLLM correction remains; excluded non-Base CI is not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Follow the active attempt without duplication. CI maintainer: obtain worker stacks, submission/processing timings, worker count/quota and memory evidence before classifying recovery. Reassess the helper only after termination with a justified reason and disclosed PR-wide effects; finish review while preserving MI355X/native-MI300 evidence limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/job/126392603595)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/job/126523051344)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41797955547/attempts/1)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. This is a historical evidence/review follow-up; no new maintainer hardware gate was established.

Tracking key: `pr33059-native-mi300-review`.

Next action: Maintainers/yihou: assess the historical native MI300 request alongside author-reported MI355X results and finish review. Preserve Ready and source-specific limits; use allocated hardware only for concrete additional coverage.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Base retains 36 earlier-attempt successes but LLaVA CPU-pool submission times out; startup/1800-second timeout and a health-check cascade follow. kv_events_config=None disables the modified send. Timeout/pool methods still match main, including after #47120. No verified repair or transience is established. The inherited simplified probe did not reproduce the real workload; the inspected green main Base run skipped functional tests. AMD/Xeon/ARM64 pass; review remains.

Tracking key: `pr46820-base-review`.

Next action: CI maintainer: capture submission-thread stacks, worker count/quota and parent memory to distinguish startup cost from a lock stall. Keep recovery separate from speculative KV publisher/VLM repairs. Maintainers: finish shared-code review and preserve socket-contract evidence without adding GPU or throughput gates.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869468)
- [Evidence](https://github.com/sgl-project/sglang/blob/ac205f829173029ce17c1898fd9c8d9ab2f861fd/python/sglang/srt/managers/scheduler_components/kv_events_publisher.py#L75)
- [Evidence](https://github.com/sgl-project/sglang/commit/7828f5670110271c9d5c6653ebeac28e160e9d42)

## #46830: Base has 21 actual successes, two executing jobs and four queued jobs. AMD retains 11 successes, two failures and one queued job: MLA startup exit -9 after compilation/watchdog stalls and Qwen TTFT 138.109/135.305 ms versus 130 ms. Those Aiter/non-speculative launches do not use the changed DSA path. Inherited baselines are not controlled runner comparisons.

Tracking key: `pr46830-base-review`.

Next action: Follow active Base/AMD attempts. CI maintainer: assess MLA runner/compiler evidence and controlled Qwen timing baselines. Maintainers: finish review while preserving historical deadlock/consumer evidence; do not impose unrelated tokenizer optimization, threshold changes or fresh-CI rebases.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/job/126434463729)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/job/126434463415)

## #45524: Base has 35 successes, AWQ timeout and three health-check failures. AMD has 13 successes, PP transfer/MLA startup failures and a job cancelled after 24 hours awaiting a runner. CPU revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: diagnose matching baseline failures, align CPU installed/tested revisions, recover missing coverage and finish review. Preserve the frozen producer patch; current-PR equivalence remains uncertified. Worker stays read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541289)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541168)

## #46918: Two Base H100 shards remain cancelled before execution. Source-specific 9763ef56 validation does not qualify the entire head or integrated image. AMD transfer CI was previously excluded without claiming repair.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover cancelled Base coverage, finish review and validate image integration. Preserve source-specific MI355X evidence and the frozen workaround; worker has no retry or branch authority.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Base has 21 actual successes and six queued jobs; three JIT units passed and three JIT jobs remain queued. The dLLM correction is integrated. Review, #45307 overlap and exact source/run archiving remain; this optional path stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow active coverage without duplication. Maintainers: finish review, coordinate #45307 landing order and preserve numerical/source evidence. Keep the optional fused-indexer path disabled in frozen llr37.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Two Base H100 shards remain cancelled. The inherited actual_failure_requires_triage refusal matches the current head, attempt and AMD failure; its recorded scope covered six workflows.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Maintainer: use scoped Actions access to recover cancelled Base coverage, or finish AMD VLM triage before another helper evaluation. Respect the refusal and supported helper's PR-wide effects.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)

## #47278: AMD VLM timeout and skipped C remain in scope because shared split-K/reduce interfaces serve decode/DCP callers. Transience is unestablished. BF16/DCP/LSE and source-mapping constraints remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: CI owners: compare actual baseline execution. Maintainers/yihou: preserve BF16 partials and DCP/LSE coverage through review, exact source/run archiving and combined-stack qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47897: Base passed 27 jobs, five JIT units and one benchmark. AMD retains MIOpen/PP failures; Xeon revision mismatch remains in shared-import scope. Review and source/numerical qualifications remain.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI owners: compare PP baseline behavior, repair MIOpen access and align Xeon revisions. Preserve passing Base. Maintainers: finish review while retaining paired-row, dispatch/fallback and source qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base has 19 actual successes, none carried from earlier attempts, plus LLaVA worker-result/startup failure, two health-check cascades and skipped C. Three JIT units passed; remaining JIT failed health checks. AMD has 16 stage successes with MIOpen/PP failures. Xeon revision mismatch, review and source-specific correctness/performance limits persist.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: CI maintainers: recover VLM/PP coverage using matching baselines, repair MIOpen access and align Xeon revisions. Maintainers/yihou: finish review; preserve opt-in behavior, exactly min(length,2048) selections, underflow/overflow fallback, valid boundary ties, source/run archiving, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125868991469)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125893452204)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125860057676)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed fork package. The checkout/package mismatch remains separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon cannot import _paged_experts_num_resident; its cancelled sibling also failed req_types and disaggregation.mode imports. Checkout/package mismatch remains; the architecture API repair does not resolve it.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853412984)
