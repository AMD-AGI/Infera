# Pending handoff for the next user session

Updated: **2026-11-03T16:11:54.709022+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base has 37 successes (20 retained, 17 executed in attempt 3) and one unstarted H100 shard queued. Scope reduced to Base; historical AMD failure is not a fix/pass claim.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: CI owners: finish the queued Base shard; retain separate native MI300 review follow-up.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41423254776/attempts/3)

## #33059: Native MI300 DSA/padded-row reviewer coverage remains unestablished. Unrelated MI300 collective tests do not supply it.

Tracking key: `pr33059-native-mi300-review`.

Next action: Yihou/reviewers: supply native MI300 coverage using allocated hardware if needed and finish review; preserve Ready state.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Base remains unsuccessful: 36 retained successes, one VLM timeout and one health-check cascade with tests skipped. Shared-code AMD/Xeon/ARM64 pass; review and controlled baseline diagnosis remain.

Tracking key: `pr46820-base-review`.

Next action: CI owners: compare VLM preprocessing/warmup with baseline execution before retrying; recover the fail-fast sibling. Maintainers: complete shared-code review and required CI. Preserve CPU socket-contract coverage without adding GPU or throughput qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869468)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/2)

## #46830: The last queued Base shard failed the dLLM speed assertion twice (235.376/236.910 versus >250). Current main uses >200, but that was not the executed test. Base has 37 successes; AMD passes. Source reconciliation and review remain.

Tracking key: `pr46830-base-review`.

Next action: CI owners: verify source alignment with the upstream threshold change before integration/retry; do not weaken tests or assume transience. Maintainers: finish review and preserve historical DSA deadlock/consumer evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467026948/job/125761043114)
- [Evidence](https://github.com/sgl-project/sglang/blob/4387a94cf218ba870474d7a2a40a632bc4975798/test/registered/dllm/test_dllm_batching_fdfo.py#L95)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467026948/attempts/2)

## #45524: Base has 35 attempt-2 successes, two queued shards, AWQ timeout and one new health-check cascade. Scoped AMD compiler/watchdog and CPU fork_worker_preload failures remain; historical-source equivalence is unproven.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: diagnose the failures, recover skipped coverage and complete review. Preserve the separate frozen PP producer patch; current-PR equivalence remains uncertified. Worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190843)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541289)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758584/job/125761894917)

## #46918: Two Base H100 shards remain cancelled before execution. Source-specific 9763ef56 validation does not qualify the entire current head or integrated image. AMD transfer CI was removed from this mixed finding under Base-only scope, without claiming repair.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover Base coverage, finish review and validate image integration. Preserve source-specific MI355X evidence and the frozen workaround until reviewed integration.

- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Base remains at 37 successes: 36 retained and one executed in attempt 2; one shard queues. Review, #45307 overlap coordination and exact source/run archiving remain. The optional fused-indexer path stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Finish Base and review; coordinate landing order and archive exact numerical evidence. Keep the optional fused-indexer path disabled in frozen llr37.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467493468/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Two Base H100 shards remain cancelled. The helper refused actual_failure_requires_triage on an AMD stage-B job. Its six-workflow PR-wide scope is verified; no retry was sent.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Maintainer: recover Base with scoped Actions access, or triage the AMD failure before another helper evaluation. Do not bypass the recorded refusal.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)

## #47278: AMD VLM preprocessing/warmup timeout and skipped C remain in scope because shared split-K/reduce interfaces serve other decode/DCP callers. No controlled baseline establishes a transient cause. BF16/DCP/LSE and source-mapping constraints remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: CI owners: compare actual baseline execution. Maintainers/yihou: preserve BF16 partials and DCP/LSE coverage through review, exact source/run archiving and combined-stack qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47897: Base progressed to A9/B11/C5=25 successes and five JIT units plus benchmark pass; two Base shards queue. Shared-import scope retains AMD MIOpen and Xeon revision mismatch. Inherited API repair does not qualify full CI or numerics.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI owners: finish Base, investigate AMD database access and align Xeon revisions. Complete review while preserving unequal paired-row/table/length coverage, dispatch/fallback gates and exact source/run mapping.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage through integration; reviewers assess remaining long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base has 19 successes, two queued shards and a new VLM startup/preprocessing failure. Two JIT jobs fail-fast before tests; three units pass and benchmark queues. Scoped AMD MIOpen and Xeon revision mismatch remain. Inherited MI355X evidence does not qualify the whole head.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: CI owners: diagnose VLM against baseline, recover JIT coverage, investigate AMD database access and align Xeon sources. Complete review; preserve opt-in behavior, exactly min(length,2048) selections, underflow/overflow fallback and boundary ties. Retain source-specific validation and the measured 32k regression.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125868991469)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125854248511)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125854248530)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125893452204)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)

## #47897: Xeon test_server_args.py imports _paged_experts_num_resident from the installed fork package. Inherited source comparison identifies a merge-checkout/package revision mismatch, separate from the repaired architecture API; the matching failure persists.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected Xeon shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon test_server_args.py imports _paged_experts_num_resident from the installed fork package. Inherited source comparison identifies a merge-checkout/package revision mismatch, separate from the repaired architecture API; the matching failure persists.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected Xeon shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
