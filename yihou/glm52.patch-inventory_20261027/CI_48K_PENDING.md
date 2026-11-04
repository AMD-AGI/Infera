# Pending handoff for the next user session

Updated: **2026-11-04T14:46:22.511819+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base has 23 actual successes, two queued B jobs, a LLaVA worker-result timeout and a separate C GitHub HTTP 500 before tests. FA3/non-speculative VLM does not exercise the modified path. Newly inspected peer job 126434007693 passed identical test/LLaVA source; root cause and this PR recovery remain unresolved. Integrated dLLM correction remains; excluded non-Base tests are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: follow the active attempt without duplication. After termination, assess bounded recovery using the executed peer comparison; CI maintainer should compare worker stacks, processing timings, worker count/quota and memory. The separate C HTTP 500 does not explain B. Maintainers: finish review and assess historical native MI300 coverage without inventing a hardware gate; preserve Ready and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/job/126392603595)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/job/126523051344)
- [Evidence](https://github.com/sgl-project/sglang/blob/201d823b3c6a41f44684d639d44f9535e53cb786/python/sglang/srt/multimodal/processors/llava.py#L167)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41797955547/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/job/126434007693)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. This is a historical evidence/review follow-up; no new maintainer hardware gate was established.

Tracking key: `pr33059-native-mi300-review`.

Next action: Maintainers/yihou: assess the historical native MI300 request alongside author-reported MI355X results and finish review. Preserve Ready and source-specific limits; use allocated hardware only for concrete additional coverage.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Pre-existing PR-wide request produced attempt 3: 36 retained Base successes and two queued B jobs. Attempt 2 submission timeout is historical and not yet resolved. KV events were disabled, bypassing the patch; relevant methods match current main. AMD/Xeon/ARM64 pass; shared-code review remains. No new retry by this cycle.

Tracking key: `pr46820-base-review`.

Next action: Worker: follow attempt 3 without duplication and inspect its eventual B results. If submission timeout recurs, CI maintainer should capture submission-thread stacks, worker count/quota and parent memory before choosing recovery. Maintainers: finish shared-code review and preserve socket-contract evidence; do not expand this into speculative VLM/KV publisher repair or new GPU/throughput requirements.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869468)
- [Evidence](https://github.com/sgl-project/sglang/blob/ac205f829173029ce17c1898fd9c8d9ab2f861fd/python/sglang/srt/managers/scheduler_components/kv_events_publisher.py#L75)
- [Evidence](https://github.com/sgl-project/sglang/commit/7828f5670110271c9d5c6653ebeac28e160e9d42)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41797955547/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820#issuecomment-6633167088)

## #46830: Base now has 22 actual successes, two queued B jobs, Mooncake EP nvcc exit 127 during class startup and two C health-check cascades. Same-source/package peer H100 coverage passed; failing-run compiler stderr and runner/toolchain cause remain unknown. AMD retains MLA startup -9, Qwen TTFT threshold failure and one queued job. These launches bypass the modified DSA path.

Tracking key: `pr46830-base-review`.

Next action: Worker: preserve active Base/AMD attempts and passed jobs. CI maintainer: compare the failing H100 runner with the passing peer, capture Mooncake compiler stderr and verify the invoked nvcc executable/toolchain before selecting recovery; exit 127 alone does not prove a missing compiler. Retain MLA runner/compiler and controlled Qwen TTFT follow-up separately. Maintainers: finish review and preserve historical DSA deadlock/consumer evidence; no tokenizer optimization, threshold relaxation or fresh-CI rebase.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/job/126434463729)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/job/126434463415)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/job/126577528175)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/job/126577528163)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/job/126577528440)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/job/126578054895)

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

## #47276: Base has 24 actual successes including all five C jobs; one B job executes and two await runners. Three JIT units passed, one unit and benchmark execute, one unit awaits a runner. Review, #45307 overlap, numerical evidence and exact source/run archiving remain; optional path stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: follow active coverage without duplication. Maintainers: finish review and coordinate #45307 overlap/landing order. Preserve numerical evidence and exact source/run archiving; keep this optional fused-indexer path disabled in frozen llr37.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
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
