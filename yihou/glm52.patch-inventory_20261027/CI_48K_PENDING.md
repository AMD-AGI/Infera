# Pending handoff for the next user session

Updated: **2026-11-19T09:19:13.763320+00:00**.

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

## #47276: New author-published e95ec5ab has active Base attempt 2: gate and 12 CPU shards pass; A GPU and JIT queue, B/C unexecuted. The author recovered a new-head cooldown-only attempt through an existing request and successful handler; this worker sent no duplicate. Old 3d739b80 FlashInfer graph mismatch and its causality/main controls remain historical unknowns, not a demonstrated failure or fix on the new head. Review, #45307 overlap and exact-source/numerical qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow active current-head Base without duplication; B/C and GPU coverage remain incomplete. If it becomes terminal unsuccessful, inspect its new first error and use the configured helper after fresh head/config/account-wide checks; the recent author request alone excludes another same-PR request before 09:47:45 UTC. Preserve new launch tuning and maintainer/author changes. Maintainers still owe review, #45307 overlap coordination and exact-source/run archiving; retain numerical qualifications and optional llr37-disabled role. Historical graph causality is unknown; seek matching current-head/main controls if it recurs.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-7004263880)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44490656161)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/38)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: New aed07ab6 integrates previously conflicting main interfaces. Current Base has 16 functional passes, one running and 12 queued; AMD has 15 passes and two queued. Old-head full success is historical. New PR-description claims include ROCm10/7.2 166 passes/5 skips, 108-case comparisons, gfx942 controls and kernel timings; exact-source raw evidence is not independently verified. Return_lse/bf16_dot routing, BF16 partials and natural-LSE source wiring have CPU/source controls, not new worker numerical qualification. Current-head e2e, DCP/scaled-KV/replay and combined-stack qualification remain; these preserve the semantic constraints formerly included in the resolved text-conflict item.

Tracking key: `pr47278-amd-validation-source`.

Next action: Owner/kernel reviewers archive and verify the new candidate's author-reported exact-source/run bundle, complete current-head numerical/e2e A/B against parent 8d72d2ea using cycles/20261119T090001Z/PR47278_GPU_VALIDATION.json, and follow active Base/AMD without duplication. Preserve source-specific BF16/DCP/LSE and combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44474121414/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44474120786/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/commit/aed07ab6f7f0773fbeb8676bcd7a67e18b9ad15e)
- [Evidence](https://github.com/sgl-project/sglang/blob/aed07ab6f7f0773fbeb8676bcd7a67e18b9ad15e/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1591)

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

## #47278: Reviewer requests remain on published aed07ab6: geometry affects H32/H128 and H96; ROTATE_IDX and tuned generic-reducer options remain unreachable; checked-in tuned numerical test still uses seq48/60/96. New return_lse tests do not supply seq15/16/128 coverage. Current-head CPU checks/metadata sweeps, ten-case boundary draft, pinned seq15 plan and compiler/reply drafts are prepared locally. Compiler reproduction/issue evidence, warmup and current-head e2e A/B remain unexecuted by this worker. Old text conflicts are cleared separately; GPU and review constraints are not resolved.

Tracking key: `pr47278-review-20261117`.

Next action: Follow active Base and AMD; preserve aed07ab6 and concurrent author work. Owner/reviewer must decide geometry removal versus H32/H128 numerical/benchmark coverage, account for H96, and remove only globally unreachable variants while preserving live external defaults and LSE/reducer contracts. Use cycles/20261119T090001Z/PR47278_REVIEW_HANDOFF.md and PR47278_GPU_VALIDATION.json for seq15 exact-parent A/B, seq16/128 reference tests, compiler evidence, warmup and approved MI355X current-head numerical/e2e A/B against 8d72d2ea. Archive/verify the new author-reported source/run evidence; retain BF16 partials, scaled-KV, DCP/natural-LSE and historical combined-stack limits. Reviewer/Triton publication still needs owner authorization; no GPU route is authorized here.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/blob/aed07ab6f7f0773fbeb8676bcd7a67e18b9ad15e/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L344)
- [Evidence](https://github.com/sgl-project/sglang/blob/aed07ab6f7f0773fbeb8676bcd7a67e18b9ad15e/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1734)
- [Evidence](https://github.com/sgl-project/sglang/blob/aed07ab6f7f0773fbeb8676bcd7a67e18b9ad15e/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1855)
- [Evidence](https://github.com/sgl-project/sglang/blob/aed07ab6f7f0773fbeb8676bcd7a67e18b9ad15e/test/registered/kernels/ops/attention/test_triton_sparse_mla_hip.py)
