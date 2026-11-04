# Pending handoff for the next user session

Updated: **2026-11-04T08:44:23.228064+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base remains incomplete: 19 actual successes, three B jobs queued without runners; C not started and JIT skipped. Prior upstream-test integration is verified. Excluded tests are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: follow the active Base attempt. Maintainers: complete review. Yihou/reviewers: supply native MI300 DSA/padded-row coverage; preserve Ready state and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/blob/201d823b3c6a41f44684d639d44f9535e53cb786/test/registered/dllm/test_dllm_batching_fdfo.py#L95)

## #33059: Native MI300 DSA/padded-row reviewer coverage remains unestablished. Unrelated collective tests do not supply it.

Tracking key: `pr33059-native-mi300-review`.

Next action: Yihou/reviewers: supply native MI300 coverage using allocated hardware if needed and finish review; preserve Ready state.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Base retains 36 prior-attempt successes but fails first at the 10-second LLaVA CPU-pool submit deadline, then VLM warmup/1800-second timeout and sibling health check. Bounded 1/8-worker CPU probe completed 24 submissions in Python 3.13 (first submits 3–28 ms) without reproducing CI Python 3.12 failure; actual CI worker count/quota and blocking stack remain unknown. Current main offers no verified processor timeout fix. Shared AMD/Xeon/ARM64 pass; review remains.

Tracking key: `pr46820-base-review`.

Next action: Yihou/CI maintainer: capture actual CI worker count, CPU quota, process memory and submit-thread stack at the 10-second deadline; distinguish worker-launch cost from lock blocking on the CI Python 3.12 workload before repair/retry. The bounded probe did not reproduce failure. Recover skipped coverage and complete shared-code review; preserve socket-contract evidence without GPU/throughput claims.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869468)
- [Evidence](https://github.com/sgl-project/sglang/blob/ac205f829173029ce17c1898fd9c8d9ab2f861fd/python/sglang/srt/multimodal/processors/base_processor.py#L538)
- [Evidence](https://github.com/sgl-project/sglang/blob/ac205f829173029ce17c1898fd9c8d9ab2f861fd/python/sglang/srt/multimodal/processors/llava.py#L157)

## #46830: Base attempt 2 now has 19 successes and three B jobs queued. AMD has 11 stage successes, two failures and one queued job: MLA compilation/watchdog startup and tokenizer TTFT 138.109/135.305 ms above 130 ms. Same-day main ROCm10 passed the identical tokenizer test at 120.59 ms and MLA in 340 s, but runner/environment equivalence is unestablished. Integrated dLLM correction does not resolve remaining coverage/review.

Tracking key: `pr46830-base-review`.

Next action: Worker: follow active Base/AMD attempts. Yihou/CI maintainer: compare runner/environment and compilation traces with the successful main baseline, then isolate MLA startup and tokenizer latency before recovery. Do not relax thresholds or infer transience. Maintainers: finish review; preserve historical DSA deadlock/consumer evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/job/126434463729)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/job/126434463415)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41752157708/job/126319697462)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41752157708/job/126319697481)

## #45524: Base has 35 successes, AWQ timeout and three health-check failures. AMD completed/failure with 13 successes, PP transfer and MLA startup failures, plus one job cancelled after 24 hours awaiting a runner. CPU revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: diagnose against matching baselines, recover cancelled/skipped coverage and finish review. Preserve the frozen producer patch and external ownership.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541289)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541168)

## #46918: Two Base H100 shards remain cancelled before execution. Source-specific 9763ef56 validation does not qualify the entire head or integrated image. AMD transfer CI was previously excluded without claiming repair.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover Base coverage, finish review and validate image integration. Preserve source-specific MI355X evidence and the frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Base attempt 2 now has 19 actual successes and three B jobs queued; three JIT units passed, two units and benchmark queued. dLLM correction is integrated. Full coverage, review, #45307 overlap and source/run archiving remain; optional path stays disabled in llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: follow active coverage without duplication. Maintainers: finish review and coordinate #45307 overlap/landing order. Preserve numerical evidence and exact source/run archiving; keep the optional fused-indexer path disabled in frozen llr37.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6624386421)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41786705555)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Two Base H100 shards remain cancelled. The inherited actual_failure_requires_triage refusal matches the current head, attempt and AMD failure; recorded scope covered six workflows.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Maintainer: recover Base with scoped Actions access, or triage AMD before another helper evaluation. Respect the refusal.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)

## #47278: AMD VLM timeout and skipped C remain in scope because shared split-K/reduce interfaces serve decode/DCP callers. Transience is unestablished. BF16/DCP/LSE and source-mapping constraints remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: CI owners: compare actual baseline execution. Maintainers/yihou: preserve BF16 partials and DCP/LSE coverage through review, exact source/run archiving and combined-stack qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47897: Base passed 27 jobs, five JIT units and one benchmark. AMD retains MIOpen and PP transfer failures; Xeon revision mismatch remains in shared-import scope. Review and source/numerical qualifications remain unresolved.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI owners: compare AMD PP with baseline, investigate MIOpen and align Xeon revisions. Base needs no retry. Maintainers: finish review while preserving paired-row, dispatch/fallback and source requirements.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base retains 19 successes, VLM failure, two health-check cascades and skipped C. Three JIT units passed; remaining JIT failed health checks. AMD completed/failure with 16 successes after the queued job passed; MIOpen and PP failures remain. Xeon revision mismatch, review and source-specific validation limits persist.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: CI maintainers: compare VLM/PP failures with matching baselines, recover skipped coverage, repair MIOpen access and align Xeon revisions. Maintainers/yihou: finish review; preserve opt-in behavior, exactly min(length,2048) selections, underflow/overflow fallback, valid boundary ties, source/run archiving, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125868991469)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125893452204)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125860057676)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/126460775278)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed fork package. Matching logs retain the merge-checkout/package mismatch, separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon cannot import _paged_experts_num_resident; its cancelled sibling also failed req_types and disaggregation.mode imports. Matching logs retain checkout/package mismatch; the architecture API repair does not resolve it.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853412984)
