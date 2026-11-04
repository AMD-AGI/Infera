# Pending handoff for the next user session

Updated: **2026-11-04T05:40:36.747050+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Prior upstream-test integration is verified at 201d823b3c6a; the old >250 failure belongs to a superseded run. Current Base attempt 1 has 15 actual successes, two B jobs running and five queued without runners. C has not run; JIT is skipped. Full Base coverage remains incomplete. Excluded tests are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: follow the active attempt without duplication. Maintainers: complete review. Preserve the separate native MI300 follow-up and inherited MI355X evidence limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/blob/201d823b3c6a41f44684d639d44f9535e53cb786/test/registered/dllm/test_dllm_batching_fdfo.py#L95)

## #33059: Native MI300 DSA/padded-row reviewer coverage remains unestablished. Unrelated MI300 collective tests do not supply it.

Tracking key: `pr33059-native-mi300-review`.

Next action: Yihou/reviewers: supply native MI300 coverage using allocated hardware if needed and finish review; preserve Ready state.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Base remains unsuccessful: 36 retained successes, one VLM timeout and one health-check cascade with tests skipped. Shared-code AMD/Xeon/ARM64 pass; review and controlled baseline diagnosis remain.

Tracking key: `pr46820-base-review`.

Next action: CI owners: compare VLM preprocessing/warmup with baseline execution before retrying and recover skipped coverage. Maintainers: complete shared-code review. Preserve CPU socket-contract evidence without GPU or throughput claims.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869468)

## #46830: The merged upstream dLLM correction is verified in 76b75ed6ed9b; source reconciliation is complete. Current Base and AMD attempts fail only at the author cooldown gate; no functional tests ran. Logged boundary 06:31:19 UTC is conditional on fresh account-wide evidence. Executed CI and review remain pending.

Tracking key: `pr46830-base-review`.

Next action: Worker: evaluate this older-created run first after the deadline using only stale_gate with fresh policy/head/attempt/account-wide checks; inspect PR-wide scope and outcome. Maintainers: obtain Base/AMD coverage and finish review while preserving historical DSA evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/job/126390009695)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/attempts/1)

## #45524: Base attempt 2 failed with 35 successful jobs, AWQ timeout and three health-check failures that skipped tests. AMD has 13 stage successes and one queued job, PP transfer failure and MLA compilation/watchdog stall. CPU fork_worker_preload mismatch persists; historical-source equivalence remains uncertified.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: diagnose failures against matching baselines, recover skipped coverage and finish review. Let queued AMD work run. Preserve the frozen producer patch and external ownership; worker stays read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541289)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base H100 shards remain cancelled before execution. Source-specific 9763ef56 validation does not qualify the entire current head or integrated image. AMD transfer CI was previously excluded under Base-only scope without claiming repair.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover Base coverage, finish review and validate image integration. Preserve source-specific MI355X evidence and the frozen workaround until reviewed integration.

- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: The merged upstream dLLM correction is verified in 674ad71976e9; the old >250 failure is historical. Current Base attempt 1 is gate-only under the author cooldown, with zero Base/JIT tests executed. CI, review, #45307 overlap coordination and exact source/run archiving remain. The optional path stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: evaluate after the deadline and older #46830, rechecking account-wide cooldown and PR-wide helper scope. Maintainers: complete review and overlap coordination; preserve numerical/source evidence and the optional deployment role.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/job/126390711532)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Two Base H100 shards remain cancelled. The inherited actual_failure_requires_triage refusal still matches the head, attempt and scoped AMD failure. Its recorded scope was six PR-wide workflows. No new helper request was sent.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Maintainer: recover Base with scoped Actions access, or triage the AMD failure before another helper evaluation. Do not bypass the recorded refusal.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)

## #47278: AMD VLM preprocessing/warmup timeout and skipped C remain in scope because shared split-K/reduce interfaces serve other decode/DCP callers. No controlled baseline establishes transience. BF16/DCP/LSE and source-mapping constraints remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: CI owners: compare actual baseline execution. Maintainers/yihou: preserve BF16 partials and DCP/LSE coverage through review, exact source/run archiving and combined-stack qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47897: Base attempt 2 remains successful with 27 executed Base jobs, five JIT units and one benchmark. AMD remains failed with 16 stage successes, MIOpen database failure and PP auxiliary-data/KV-transfer failure. Xeon revision mismatch remains within shared-import scope. Scoped CI, review and numerical/source qualifications remain unresolved.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI owners: compare AMD PP failure with baseline execution, investigate MIOpen access and align Xeon revisions. Base needs no retry. Maintainers: finish review while preserving paired-row, dispatch/fallback and source evidence requirements.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage through integration; reviewers assess remaining long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base remains failed: 19 successful jobs, one VLM startup/preprocessing failure and two health-check cascades; C skipped. Three JIT units passed; two units and the benchmark were blocked by health checks. AMD has 15 stage successes, one queued job, MIOpen failure and PP transfer failure. Xeon mismatch persists; inherited MI355X evidence does not qualify the whole head.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: CI owners: compare Base VLM and AMD PP failures with matching baselines, recover skipped coverage, investigate MIOpen access and align Xeon revisions. Let queued AMD work run. Maintainers: finish review; preserve opt-in behavior, exact selection count, underflow/overflow fallback, boundary ties, source/run archiving, the measured 32k regression and source-specific validation limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125868991469)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125893452204)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125860057676)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed fork package. Matching logs retain the inherited merge-checkout/package revision mismatch, separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected Xeon shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed fork package. The cancelled sibling also failed imports of sglang.srt.managers.req_types and sglang.srt.disaggregation.mode. Matching logs retain the inherited checkout/package revision mismatch; the architecture API repair does not resolve it.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected Xeon shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853412984)
