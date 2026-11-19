# Pending handoff for the next user session

Updated: **2026-11-19T12:15:25.628300+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #45524: Historical/unrefreshed; outside the resumed scope. Base passed 27 functional jobs. Same AMD shard 6 exited 255 after starting tests; historical log retrieval returned HTTP 404. First exception, source equivalence and historical AWQ/PP review remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: External author eaglewangao-us/maintainers retrieve runner logs, recover coverage and complete source-equivalence review; preserve frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Historical/unrefreshed; outside the resumed scope. Base has 36 functional successes and two shards cancelled without tests. One annotation identifies github-actions[bot]; finish log confirms cancellation caused the aggregate failure. Cancellation policy and original-PR integration remain unresolved. Scoped validation at 9763ef56 does not qualify current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: External author yu-depend/maintainers own original-PR conflict, remaining coverage and review decisions, including the unexplained cancellation policy. Advance owned continuation #49732 separately; preserve frozen workaround and source-specific MI355X evidence at 9763ef56.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Current e95ec5ab Base attempt 2 remains queued with 12 previously observed CPU passes. New JIT failure is CUDA initialization at torch.randn before the attention kernel; A GPU and three other JIT failures are cascades, B/C skipped, two JIT jobs still queue. Helper refused active Base, not cooldown or main eligibility. Historical FlashInfer mismatch causality remains unknown and is not demonstrated on this head. Review, #45307 overlap and exact-source/numerical qualification remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow active attempt 2 without duplication; once terminal unsuccessful, freshly inspect head/config/attempt and promptly use the configured owner_requested helper if eligible. The saved local hourly interval has elapsed, but active Base currently blocks recovery; upstream/account-wide eligibility must be rechecked at request time. CI maintainers need CUDA context/runner evidence if initialization failures recur; no patch causality is established. Preserve source/run archiving, numerical limits, maintainer review, #45307 overlap coordination and optional llr37-disabled role. Historical FlashInfer causality still needs matching controls if it recurs.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/job/135445585001)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/job/135445585062)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...e95ec5abc8b4a9bf27b3dae160e42dc5beeb5f6e)

## #47278: Published cabe5837 supersedes aed07ab6 and must be qualified separately. Current Base has 0 functional passes/13 running/0 queued; one JIT shard failed at CUDA input allocation before attention, while Base and AMD remain active. Author reports 172 passes/5 skips per image, 108-case comparisons and gfx942 controls; raw exact-source/run mapping is not independently verified. CPU routing/source controls pass, but current-head numerical/e2e, scaled-KV, DCP/natural-LSE, replay and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Follow active current-head Base/AMD without duplication; recover terminal unsuccessful Base through the helper after fresh eligibility checks. Preserve cabe5837 and author work. Geometry/dead-code/boundary source requests are addressed; retain independent numerical qualification, exact-parent seq=15 equality, BF16 partials, scaled-KV, DCP/natural-LSE and replay controls. Owner/kernel reviewers must archive the new exact-source GPU results, complete current-head e2e A/B against parent 8d72d2ea, and provide a minimal Triton 3.8.0 async-copy/decoupled-dot reproducer and authorized issue/link. Use cycles/20261119T120002Z/PR47278_REVIEW_HANDOFF.md and PR47278_GPU_VALIDATION.json. Warmup 13.3s/26.9s is author-reported. No machine-pool skill/GPU route is authorized here; posting an issue/review reply still needs owner authorization. Historical whole-stack performance does not qualify this head.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/job/135546036028)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7008488632)
- [Evidence](https://github.com/sgl-project/sglang/blob/cabe5837e956276e5507244e6c5235c1f5f5d707/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1475)

## #47897: Historical/unrefreshed; outside the resumed scope. Base passed 27 functional jobs. Matching inherited logs identify AMD MIOpen SQLite database-open error and PP KVTransferError; Xeon package/checkout mismatch remains. ARM64 passed. Five later MIOpen files, two PP files and 16/23 Xeon files were unexecuted.

Tracking key: `pr47897-architecture-api-ci`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: CI maintainers repair database access, investigate PP peers and align installed/test revisions; preserve passing Base and numerical qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Historical/unrefreshed; outside the resumed scope. Run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K does not establish long-context DSA accuracy.

Tracking key: `pr47897-source-long-context`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: Archive exact evidence and complete unequal paired-row/table-length, dispatch/fallback and long-context review.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Historical/unrefreshed; outside the resumed scope. OWNER PAUSED pending redesign. Historical passing CI and source/numerical qualifications remain archived and were not queried.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: Owner-paused; resume only on owner instruction. Preserve opt-in/fallback/tie, measured 32k regression and exact-source/long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204156/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Historical/unrefreshed; outside the resumed scope. Matching Xeon failure cannot import _paged_experts_num_resident from the installed package. Checkout/package alignment remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #49732: Historical/unrefreshed; outside the resumed scope. Base attempt 3 passed all 31 functional jobs; CI gap recovered. Current-head maintainer review and integrated-image qualification remain. Isolated configuration checks and archived 9763ef56 GPU evidence do not establish full module/model coverage for this candidate.

Tracking key: `pr49732-base-review-validation`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: Maintainers review and merge owned continuation preserving yu-depend/#46918 credit. Complete integrated-image qualification and exact-source numerical evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42974439450/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/49732)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)

## #47278: The author published cabe5837 and replied to the reviewer. CPU/source inspection confirms geometry removal, H64-only gating, dead-variant removal, baseline-identical generic reducer/launch and seq=15/16/128 test parametrization. These review subitems progressed; no new worker numerical pass is claimed. Remaining requests: current-head e2e A/B, exact-parent seq=15 numerical equality, standalone Triton 3.8.0 async-copy/decoupled-dot reproducer/issue link and verified warmup/raw evidence. The old LEAN_EPILOGUE miscompile is now author-reported nonreproducing; do not perpetuate the old version attribution as a fresh fact.

Tracking key: `pr47278-review-20261117`.

Next action: Follow active current-head Base/AMD without duplication; recover terminal unsuccessful Base through the helper after fresh eligibility checks. Preserve cabe5837 and author work. Geometry/dead-code/boundary source requests are addressed; retain independent numerical qualification, exact-parent seq=15 equality, BF16 partials, scaled-KV, DCP/natural-LSE and replay controls. Owner/kernel reviewers must archive the new exact-source GPU results, complete current-head e2e A/B against parent 8d72d2ea, and provide a minimal Triton 3.8.0 async-copy/decoupled-dot reproducer and authorized issue/link. Use cycles/20261119T120002Z/PR47278_REVIEW_HANDOFF.md and PR47278_GPU_VALIDATION.json. Warmup 13.3s/26.9s is author-reported. No machine-pool skill/GPU route is authorized here; posting an issue/review reply still needs owner authorization. Historical whole-stack performance does not qualify this head.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7008488632)
- [Evidence](https://github.com/sgl-project/sglang/blob/cabe5837e956276e5507244e6c5235c1f5f5d707/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L344)
- [Evidence](https://github.com/sgl-project/sglang/blob/cabe5837e956276e5507244e6c5235c1f5f5d707/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1315)
- [Evidence](https://github.com/sgl-project/sglang/blob/cabe5837e956276e5507244e6c5235c1f5f5d707/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
- [Evidence](https://github.com/sgl-project/sglang/blob/cabe5837e956276e5507244e6c5235c1f5f5d707/test/registered/kernels/ops/attention/test_triton_sparse_mla_hip.py#L168)
