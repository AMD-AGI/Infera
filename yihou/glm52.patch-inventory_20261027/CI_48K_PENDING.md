# Pending handoff for the next user session

Updated: **2026-11-03T20:40:32.376808+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base attempt 3 completed unsuccessfully: 37 successes (20 retained, 17 executed), and the last shard failed dLLM speed twice at 224.739/224.875 versus >250. Executed merge 456e2e3486 retains >250 while current main uses >200. Base-only scope remains; excluded non-Base checks are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: CI owners: reconcile the executed and upstream test revisions before integration/retry; do not weaken tests or assume a transient cause. Native MI300 reviewer coverage remains a separate pending item.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41423254776/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41423254776/job/125745684582)
- [Evidence](https://github.com/sgl-project/sglang/blob/71ef675ff584d01758d790e0c436f93e44f4097b/test/registered/dllm/test_dllm_batching_fdfo.py#L95)

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
- [Evidence](https://github.com/sgl-project/sglang/blob/71ef675ff584d01758d790e0c436f93e44f4097b/test/registered/dllm/test_dllm_batching_fdfo.py#L95)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467026948/attempts/2)

## #45524: Base retains 35 attempt-2 successes, two queued shards, AWQ timeout and a health-check cascade. AMD now additionally fails the PP GSM8K test after aux-data send/KV retrieval failures and HTTP 500; 12 stage jobs passed and two remain queued. Existing AMD compiler/watchdog and CPU fork_worker_preload mismatches persist. Historical-source equivalence is unproven.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: diagnose PP aux-data/KV-transfer failure against a matching baseline, AWQ timeout, AMD compiler/watchdog failure and CPU test/package mismatch; recover skipped coverage and finish review. Preserve the frozen PP producer patch; current-PR equivalence remains uncertified. Worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190843)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541289)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758584/job/125761894917)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)

## #46918: Two Base H100 shards remain cancelled before execution. Source-specific 9763ef56 validation does not qualify the entire current head or integrated image. AMD transfer CI was previously removed from this mixed finding under Base-only scope, without claiming repair.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover Base coverage, finish review and validate image integration. Preserve source-specific MI355X evidence and the frozen workaround until reviewed integration.

- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Base remains failed: 37 successes (36 retained, one executed in attempt 2), and dLLM speed 222.256/222.160 below the executed >250 assertion. Fresh main still uses >200; source reconciliation, review, #45307 overlap coordination and exact source/run archiving remain. Optional fused indexer stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: CI owners: reconcile the executed >250 assertion with upstream main >200 and verify the test revision before integration/retry; do not weaken tests or assume transience. Maintainers: finish review, coordinate overlapping #45307 and rebase whichever lands second. Archive exact source/run and numerical evidence; keep the optional fused-indexer path disabled in frozen llr37.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467493468/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467493468/job/125765401572)
- [Evidence](https://github.com/sgl-project/sglang/blob/71ef675ff584d01758d790e0c436f93e44f4097b/test/registered/dllm/test_dllm_batching_fdfo.py#L95)

## #47278: Two Base H100 shards remain cancelled. The previous cycle helper refusal actual_failure_requires_triage remains applicable to the unchanged AMD stage-B failure. Its recorded scope was six PR-wide workflows; no helper request or retry was sent this cycle.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Maintainer: recover Base with scoped Actions access, or triage the AMD failure before another helper evaluation. Do not bypass the recorded refusal.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)

## #47278: AMD VLM preprocessing/warmup timeout and skipped C remain in scope because shared split-K/reduce interfaces serve other decode/DCP callers. No controlled baseline establishes a transient cause. BF16/DCP/LSE and source-mapping constraints remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: CI owners: compare actual baseline execution. Maintainers/yihou: preserve BF16 partials and DCP/LSE coverage through review, exact source/run archiving and combined-stack qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47897: Base remains at A9/B11/C5=25 successes and five JIT units plus benchmark pass; two Base shards queue. Shared-import scope retains AMD MIOpen and Xeon revision mismatch. Inherited API repair does not qualify full CI or numerics.

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

## #47872: Base remains at 19 successes, two queued shards and VLM startup/preprocessing failure. Three JIT units passed; two units and the benchmark failed health checks before execution. Scoped AMD MIOpen and Xeon source mismatch remain. Inherited MI355X evidence does not qualify the whole head.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: CI owners: diagnose Base VLM startup/preprocessing against baseline and recover the two JIT units and benchmark blocked by health checks; investigate AMD MIOpen access and align Xeon package/test revisions. Complete review; preserve opt-in behavior, exactly min(length,2048) selections, underflow/overflow fallback and valid boundary ties. Archive exact source/run evidence; retain the measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125868991469)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125854248511)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125854248530)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125893452204)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125854248810)

## #47897: Xeon test_server_args.py imports _paged_experts_num_resident from the installed fork package. Inherited source comparison identifies a merge-checkout/package revision mismatch, separate from the repaired architecture API; the matching failure persists.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected Xeon shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed fork package. The cancelled sibling also failed imports of sglang.srt.managers.req_types and sglang.srt.disaggregation.mode. These matching logs retain the inherited merge-checkout/package revision mismatch; the architecture API repair does not resolve it.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected Xeon shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853412984)
