# Pending handoff for the next user session

Updated: **2026-11-04T07:11:36.173880+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Prior upstream-test integration is verified. Base attempt 1 has 19 actual successes and three B jobs queued without runners or steps. C has not started; JIT skipped. Full Base coverage remains incomplete; excluded tests are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: follow the active attempt without duplication. Maintainers: complete review. Preserve native MI300 follow-up and inherited MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/blob/201d823b3c6a41f44684d639d44f9535e53cb786/test/registered/dllm/test_dllm_batching_fdfo.py#L95)

## #33059: Native MI300 DSA/padded-row reviewer coverage remains unestablished. Unrelated collective tests do not supply it.

Tracking key: `pr33059-native-mi300-review`.

Next action: Yihou/reviewers: supply native MI300 coverage using allocated hardware if needed and finish review; preserve Ready state.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Base remains unsuccessful: 36 retained successes, VLM timeout and a health-check cascade. Shared-code AMD/Xeon/ARM64 pass; review and controlled baseline diagnosis remain.

Tracking key: `pr46820-base-review`.

Next action: CI owners: compare VLM preprocessing/warmup with baseline execution before retrying and recover skipped coverage. Maintainers: complete review. Preserve CPU socket-contract evidence without GPU or throughput claims.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869468)

## #46830: Earlier retry cleared the gate. Base attempt 2 has 10 successes, three running and nine queued. AMD has three successes, six running, four queued and a new MLA startup compilation/watchdog failure. No controlled baseline establishes transience. The dLLM correction is integrated; coverage and review remain incomplete.

Tracking key: `pr46830-base-review`.

Next action: Worker: follow active Base/AMD attempts without duplication. CI owners: compare MLA compilation/startup failure with a matching baseline. Maintainers: finish review and recover coverage; preserve historical DSA evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/job/126434463729)

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

## #47276: Earlier retry cleared the gate. Attempt 2 has 10 Base successes, three running and nine queued; three JIT units passed, two units and benchmark queued. The dLLM correction is integrated. Full CI, review, #45307 overlap and source/run archiving remain. Optional path stays disabled in llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: follow active coverage without duplication. Maintainers: finish review and coordinate overlap/landing order; preserve numerical evidence, exact source/run mapping and optional deployment role.

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

## #47872: Base retains 19 successes, VLM failure and two health-check cascades; C skipped. Three JIT units passed; remaining JIT coverage blocked by health checks. AMD retains 15 successes, one queued job, MIOpen and PP failures. Xeon mismatch and source-specific validation limits persist.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: CI owners: compare failures with matching baselines, recover coverage, investigate MIOpen and align Xeon revisions. Let queued work run. Maintainers: finish review; preserve opt-in behavior, exact selection count, underflow/overflow fallback, boundary ties, source archiving and measured 32k regression.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125868991469)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125893452204)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125860057676)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed fork package. Matching logs retain the merge-checkout/package mismatch, separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon cannot import _paged_experts_num_resident; its cancelled sibling also failed req_types and disaggregation.mode imports. Matching logs retain checkout/package mismatch; the architecture API repair does not resolve it.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853412984)
