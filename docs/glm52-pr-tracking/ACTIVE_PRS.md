# Active PR handoff

GitHub refresh: **2026-11-11T12:11:24.231017+00:00**. [Overview and priorities](README.md).

## Ready for review

### [sgl-project/sglang #49732](https://github.com/sgl-project/sglang/pull/49732) — Continue #46918 under an owned branch: cap ROCm absorb-BMM compile-time wave budget.

- **State / responsibility:** READY; yihou; original implementation yu-depend. GitHub assignees: none. Requested reviewers: Fridge003, ispobock, fzyzcjy, ch-wan, wisclmy0611, zijiexia, JustinTong0323, sogalin.
- **Base / head:** `main` @ `eef6d87111ca94083f2840821de02d037463c174` / `yihou/fix-rocm-absorb-bmm-waves-followup-46918` @ `da22f562c7339c2680bcd6dbf1a8db59a4b81386`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 8, NEUTRAL 1, QUEUED 4, SKIPPED 55, SUCCESS 101. Reviews: claude: COMMENTED (2026-11-11T06:43:32Z).
- **Last curated action:** Automated 3-hour audit 2026-11-11T12:10:20.838177+00:00: Verified current source scope, ownership, preserved yu-depend credit and authorization. Downloaded first HTTP 502 failure and health-check cascade logs. owner_requested helper refused retry_interval_not_elapsed; no retry posted or code change.
- **Latest GitHub activity:** [Latest comment 2026-11-11T07:05:19Z](https://github.com/sgl-project/sglang/pull/49732#issuecomment-6789435999); PR updated 2026-11-11T10:53:53Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-11; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** Original PR https://github.com/sgl-project/sglang/pull/46918; original commit 9763ef56. Historical MI355X evidence: validation/b8_46918/RESULTS.md. Current config AST matches original; focused CPU 9 tests/259 subtests and pre-commit pass. Full registered module collection lacked pybase64. Reproducible CPU config checker: https://github.com/AMD-AGI/Infera/tree/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/validation/b8_49732
- **To Ready:** Ready; initial run-ci label needs maintainer.
- **To merge / dependency:** Base audit 2026-11-11T12:10:20.838177+00:00: Base only: ROCm-specific absorb-BMM configuration and environment declarations. Maintainer run-ci present; gate and 25 actual functional jobs passed (A11/B14/C0). wait-for-base-b failed on GitHub HTTP 502 after 14/15 B jobs completed; remaining B job failed health check and skipped tests. C suites skipped; finish failed. Initial missing-label attempt executed no tests. Inherited isolated configuration check: 9 tests/259 subtests; full registered-module local collection lacked pybase64. Historical 9763ef56 MI355X evidence does not qualify current head/image. Worker retries terminal Base no earlier than 2026-11-11T13:03:23Z, subject to fresh eligibility and older failed-run priority. Maintainers review owned continuation, preserving #46918 credit. Complete current-head CI and integrated-image qualification; historical 9763ef56 GPU evidence and isolated config checks do not establish full module/model coverage.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/42974438453/job/130378339018); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/42974439250/job/130378325100); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/42974439072/job/130378424430); [pr-test-amd-extra-finish](https://github.com/sgl-project/sglang/actions/runs/42974439250/job/130378377514); [pr-test-extra-finish](https://github.com/sgl-project/sglang/actions/runs/42974439072/job/130378471648); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q -p no:cacheprovider test/registered/unit/models/test_rocm_mla_absorb_fusion_gate.py
```


### [sgl-project/sglang #47276](https://github.com/sgl-project/sglang/pull/47276) — Speed up gfx950 fused indexer logits/top-k for MTP verify rows.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me, DarkSharpness, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `32b7b66572d76bb5b7042e5792fee31d9de60286` / `yihou/kf-fused-indexer-gfx950` @ `3d739b80766dd8145d1e798edb9d202aba091d47`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 5, IN_PROGRESS 1, QUEUED 2, SKIPPED 58, SUCCESS 97. Reviews: claude: COMMENTED (2026-10-29T13:35:23Z).
- **Last curated action:** Automated 3-hour audit 2026-11-11T12:10:20.838177+00:00: Used owner_requested helper after fresh config/head/attempt checks; verified durable reservation, posted request, successful handler and queued Base attempt 19. Handler retried Base, NVIDIA Extra, AMD Extra, Xeon and MLU failed jobs and fully reran skipped Mori. No code change.
- **Latest GitHub activity:** [Latest comment 2026-11-11T12:03:23Z](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6795283862); PR updated 2026-11-11T12:04:28Z.
- **Evidence freshness:** HEAD CHANGED: re-audit evidence from 674ad71976. Notes last assessed 2026-11-11; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** PR-reported bit-identical logits/top-2048/K-write controls and real-acceptance TP8 GSM8K; isolated whole-call 1.20–1.36x; serving ITL used simulated acceptance 3.61. Historical [glm52.llrepro-kf-ar-topk.packup_20261024](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-kf-ar-topk.packup_20261024/README.md) is a related source family, not a verified final-port run mapping.
- **To Ready:** Already Ready. Preserve independent numerical evidence and coordinate #45307; bind final source/run IDs to a public result bundle when available.
- **To merge / dependency:** Base audit 2026-11-11T12:10:20.838177+00:00: Base only: gfx950 kernels and HIP-guarded dispatch. Attempt 18 executed the H100 shard: FlashInfer MLA EAGLE draft CUDA Graph comparison failed, 1/6 elements mismatched, absolute difference 43 at (1,2), tolerance .03. Nine files passed, the tenth failed and 25/35 did not execute. Attempt 19 retains gate and 22 functional passes (A10/B12/C0), five JIT suites, two benchmarks and one inherited 0/0 JIT job. Identical step timestamps confirm retention. Retry shard queued with no steps; no new functional execution. Follow active attempt 19 without duplication. If comparison fails again, runner owner captures tuple component/dtype, tensors and replay state; current comparator omits component labels. Preserve optional llr37-disabled role, #45307 overlap and exact-source/numerical review.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/42126240330/job/130510878651); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/42126248059/job/130510854062); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/42126247972/job/130510836237); [pr-test-amd-extra-finish](https://github.com/sgl-project/sglang/actions/runs/42126248059/job/130510911126); [pr-test-extra-finish](https://github.com/sgl-project/sglang/actions/runs/42126247972/job/130510897283).

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
# GPU: use a pool node; exact architecture/flags are in the PR
PYTHONPATH=python python -m pytest -q test/registered/amd/test_dsa_gfx950_fused_indexer.py
```


### [sgl-project/sglang #47278](https://github.com/sgl-project/sglang/pull/47278) — Speed up gfx950 FP8 sparse MLA decode with a 64-head split-K plan.

- **State / responsibility:** READY; yihou; kernel delegate requested from owner. GitHub assignees: none. Requested reviewers: DarkSharpness, HaiShaw, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `d565013acdcb4590d6b88ffb7557a6f83f3a4793` / `yihou/kf-sparse-mla-gfx950` @ `f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 11, IN_PROGRESS 1, SKIPPED 63, SUCCESS 98. Reviews: claude: COMMENTED (2026-10-29T13:43:32Z).
- **Last curated action:** Automated 3-hour audit 2026-11-11T12:10:20.838177+00:00: Downloaded and inspected current Base setup/cascade failures; verified AMD completion. Fresh-head owner_requested helper refused retry_interval_not_elapsed after #47276 recovery. No retry posted or source change; preserved supplied reconciliation candidate.
- **Latest GitHub activity:** [Latest comment 2026-11-11T02:36:21Z](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6785328484); PR updated 2026-11-11T10:06:07Z.
- **Evidence freshness:** HEAD CHANGED: re-audit evidence from a5d8b3af95. Notes last assessed 2026-11-11; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** PR reports 22/22 kernel tests, FP32-reference SNR >35 dB and real-acceptance GSM8K mean .942 baseline / .941 PR; isolated kernel 2.4–3.2x. Serving ITL used simulated acceptance 3.61. Final-port pack-up mapping still needs audit. Conflict handoff: [risk and exact source anchors](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/reviews/PR47278_CONFLICT_HANDOFF_20261110.md).
- **To Ready:** Already Ready. Preserve the BF16-partial accuracy change and DCP/LSE coverage; raw run-to-source archiving remains a handoff follow-up.
- **To merge / dependency:** Base audit 2026-11-11T12:10:20.838177+00:00: Base plus AMD: shared sparse-MLA load/launch/reduction interfaces. Base gate and 28 functional jobs passed (A11/B15/C2). GB300 setup failed importing yaml; two other C shards failed health checks, so three functional suites never ran. Six JIT jobs and two benchmarks succeeded separately. AMD now passed gate, finish, all 17 functional stages, JIT unit tests and benchmark; skipped optional suites are not coverage. Inherited 33-test MI355X report remains source/run and combined-stack limited. Worker retries failed Base no earlier than 2026-11-11T13:03:23Z, subject to fresh helper eligibility; runner owner addresses missing PyYAML. Owner/kernel reviewer verifies supplied reconciliation and exact runs: scaled-KV/GROUP_W, BF16 partials, DCP/LSE, cache/async-copy and launch/reducer semantics; retain e2e/combined-stack limits.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/42947078594/job/130282820673); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/42947082637/job/130282808344); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/42947082953/job/130282796327); [stage-a-unit-test-mlx](https://github.com/sgl-project/sglang/actions/runs/42947082798/job/130282507728); [pr-test-amd-extra-finish](https://github.com/sgl-project/sglang/actions/runs/42947082637/job/130282848931); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
# GPU: use a pool node; preserve PR architecture and dtype matrix
PYTHONPATH=python python -m pytest -q test/registered/kernels/ops/attention/test_triton_sparse_mla_hip.py
```


### [sgl-project/sglang #47872](https://github.com/sgl-project/sglang/pull/47872) — Provide opt-in long-context top-k on compact page tables with exact sampling fallbacks.

- **State / responsibility:** READY; yihou; active tracking paused by owner pending leader-requested redesign. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me, DarkSharpness, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `e1cd0c55e553b3f9440f23835e9ac2777f2bfe92` / `yihou/kf-triton-topk-gfx950` @ `dccb51e0f9f3d97dc01aea7aa39db1c0cfe3df34`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 9, SKIPPED 61, SUCCESS 101. Reviews: claude: COMMENTED (2026-11-02T07:47:39Z).
- **Last curated action:** Automated 3-hour audit 2026-11-11T12:10:20.838177+00:00: Owner-paused: retained historical evidence without lifecycle, CI or review queries; no retry, rebase or modification.
- **Latest GitHub activity:** [Latest comment 2026-11-09T22:40:57Z](https://github.com/sgl-project/sglang/pull/47872#issuecomment-6753411506); PR updated 2026-11-10T06:07:45Z.
- **Evidence freshness:** HEAD CHANGED: re-audit evidence from 1061c29838. Notes last assessed 2026-11-11; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** [Historical indexer source](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/patches/K01-kf-decode-kernel-overlay.NEWFILES.diff); [PR evidence](https://github.com/sgl-project/sglang/pull/47872). Author reports exact overflow fallback, valid nondeterministic boundary ties and 1.5–1.9x isolated speedup at 400k contexts/72+ rows, but about 1.3% slower serving at 32k with simulated acceptance 3.61. Real-acceptance GSM8K .953 from one server is a smoke test. Maintenance reproduced candidate underflow in the actual GPU pipeline: original 256 valid slots instead of 2048; fixed 2048 unique valid slots and all 5 GPU tests pass without skips. Run kf47872-20261102T1038Z-topk-underflow used MI355X and the public v0.5.22 ROCm10 image, Torch 2.11.0+rocm10.0.0 / Triton 3.8.0. [Pinned source, CPU/GPU records and reproduction scripts](https://github.com/AMD-AGI/Infera/tree/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/validation/kf_47872). The GPU runner uses the exact module/tests with namespace and CI-registration shims for the older image; no new server/model/performance result is claimed.
- **To Ready:** Ready restored after the underflow repair, import-guard regression and exact-source MI355X validation. The temporary correctness Draft is resolved; retain opt-in behavior and the historical performance limits.
- **To merge / dependency:** Base audit 2026-11-11T12:10:20.838177+00:00: Owner-paused; no lifecycle, CI or review queries. Historical head dccb51e0f9f3d97dc01aea7aa39db1c0cfe3df34 and attempt 1 are inherited only: Base had 30 functional passes, five JIT suites and two benchmarks; AMD/Xeon/ARM64 passed. Hold is neither a failure nor a pass. Owner-paused pending leader-requested redesign; resume only on owner instruction. Preserve historical source/run, opt-in/fallback/tie and 32k/long-context qualifications.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/42722198134/job/129518670251); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/42722204121/job/129518691419); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/42722204209/job/129518889601); [stage-a-unit-test-mlx](https://github.com/sgl-project/sglang/actions/runs/42722203849/job/129518870666); [stage-a-test-1-gpu-xpu](https://github.com/sgl-project/sglang/actions/runs/42722203841/job/129518929361); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/amd/test_dsa_triton_topk_transform.py
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_triton_topk_gate.py
```


### [sgl-project/sglang #47897](https://github.com/sgl-project/sglang/pull/47897) — Speed up gfx950 decode logits while preserving paired-row tails and fallback dispatch.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: DarkSharpness, HaiShaw, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `a133231d3140e081424d833caa0187f15d40dc3c` / `yihou/kf-gluon-indexer-logits-gfx950` @ `8a9f215cb422ab507f00537c695ecccf1cfbe329`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: CANCELLED 2, FAILURE 9, SKIPPED 63, SUCCESS 94. Reviews: claude: COMMENTED (2026-11-02T09:31:07Z).
- **Last curated action:** Automated 3-hour audit 2026-11-11T12:10:20.838177+00:00: Freshly verified passing Base/ARM64 and affected AMD/Xeon failures. Downloaded matching logs and confirmed MIOpen SQLite access, PP peer-transfer errors and package/checkout import mismatch. No justified Base retry.
- **Latest GitHub activity:** [Latest comment 2026-11-03T09:39:38Z](https://github.com/sgl-project/sglang/pull/47897#issuecomment-6600214639); PR updated 2026-11-04T03:50:39Z.
- **Evidence freshness:** HEAD CHANGED: re-audit evidence from fcfa508799. Notes last assessed 2026-11-11; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** [Historical indexer source](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/patches/K01-kf-decode-kernel-overlay.NEWFILES.diff); [PR evidence](https://github.com/sgl-project/sglang/pull/47897). The c2cacf5f candidate matches the frozen indexer module; the PR fixes its fused-pair tail omission. Author reports 2.67–3.06x isolated long-context logits, numerical/top-k checks, and real-acceptance GSM8K means .934 main/.927 PR with overlapping ranges. GSM8K is a short-context smoke test. Serving timings used simulated acceptance 3.61 in a single-node setup, not the frozen PD/AgentX setup. Local PR_NOTES.md and working_process.md identify validation scripts; raw results are not yet a public archived bundle.
- **To Ready:** Already Ready. Preserve existing hardware evidence and the paired-row regression; archived raw run-to-source bundle is still a handoff follow-up.
- **To merge / dependency:** Base audit 2026-11-11T12:10:20.838177+00:00: Base plus AMD/Xeon/ARM64: shared architecture import and CUDA norm dispatch. Gate, finish and 27 functional jobs passed (A9/B13/C5), four JIT suites and one benchmark; another JIT job ran 0/0. AMD retains 16 passing stages with MIOpen database-access and PP failures; five later MIOpen-shard files and two PP files unexecuted. Xeon package/checkout mismatch left 16/23 files unexecuted and two jobs cancelled. ARM64 passed. CI maintainers repair MIOpen database access, investigate PP peers and align installed/test Xeon revisions. Preserve passing Base. Yihou/reviewers archive exact source/runs and complete unequal paired-row/table-length, dispatch/fallback and long-context numerical review; short-context GSM8K is insufficient.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41601312319/job/125839021635); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41601317697/job/125838910785); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41601317776/job/125838953662); [build-test (xeon-spr, spr, stage-a-test-cpu-intel,stage-b-test-cpu-intel, spr1, 0, 1, 4)](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259); [pr-test-amd-extra-finish](https://github.com/sgl-project/sglang/actions/runs/41601317697/job/125838974185); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/amd/test_dsa_gluon_paged_mqa_logits.py
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_mqa_logits_chunking.py
```


### [sgl-project/sglang #45484](https://github.com/sgl-project/sglang/pull/45484) — Preserve the Aiter is_shuffled weight tag through offload/onload.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `449617bc94a5d4ae23ad62d0da2f842e711ca983` / `fix/offloader-keep-is-shuffled` @ `43dec26b032c08f4e9eee0c772db9f44d3afc275`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: CANCELLED 2, FAILURE 16, SKIPPED 90, SUCCESS 29. Reviews: none.
- **Last curated action:** 2026-11-03: confirmed unchanged head, Ready state, passing lint and missing run-ci. The Base gate is authorization failure, not a model or kernel failure. No mutation by this audit.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:21:05Z](https://github.com/sgl-project/sglang/pull/45484#issuecomment-6568974915); PR updated 2026-11-02T04:21:05Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-03; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.v0521-offloaderv2.packup_20261015](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.v0521-offloaderv2.packup_20261015/README.md) ofs0c/ofg3c/ofg2c 23/23 each; AST-equivalent port, CPU 3+2 and allocated MI355X two-onload-cycle test. Full-model rerun is not an automatic Ready gate.
- **To Ready:** No remaining author-side promotion step; CPU and exact-head MI355X onload regression already exist.
- **To merge / dependency:** Ready; latest lint passes but run-ci remains absent. [Latest Base](https://github.com/sgl-project/sglang/actions/runs/40471208281) stops at the missing-label gate; functional tests were not executed. Maintainer: add run-ci and trigger required CI; yihou/delegate: handle actual test/review findings.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40471208261/job/122078486436); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40471208251/job/122078485647); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40471208207/job/122078579497); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40471208218/job/122078580449); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40471208196/job/122078578924); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/utils/test_offloader.py
# Existing GPU command; allocate a pool node if rerunning
PYTHONPATH=python python test/registered/model_loading/test_offloader_shuffled_weights.py
```


### [sgl-project/sglang #46822](https://github.com/sgl-project/sglang/pull/46822) — Skip HIP DSA indexing when an eager DP rank has no tokens.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me.
- **Base / head:** `main` @ `a6f5bd397be420c12d0690c07e16e241124269e5` / `yihou/rocm-dsa-eager-idle-20261027` @ `11262e1199d914ac1946bdcf3d18f56cc3f45612`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: CANCELLED 1, FAILURE 17, SKIPPED 90, SUCCESS 25. Reviews: none.
- **Last curated action:** 2026-11-03: confirmed current Ready state at 11262e1199d9 and passing lint. Historical Draft-gate result needs a fresh authorized attempt, not another Ready conversion. No code change or retry by this audit.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:21:06Z](https://github.com/sgl-project/sglang/pull/46822#issuecomment-6568975144); PR updated 2026-11-02T04:21:06Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-03; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.dcp4-pd.packup_20261022](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.dcp4-pd.packup_20261022/README.md) R16/R17 fault, R18 5/5 +40/40; later separate R24/R25 row-list bug covered by #45796; current CPU 4 tests +20 subtests.
- **To Ready:** No remaining author-side promotion step; retain historical production equivalence.
- **To merge / dependency:** Ready, latest lint passes, run-ci absent. [Latest Base](https://github.com/sgl-project/sglang/actions/runs/40492640946) is the old October 27 Draft=true rejection and does not reflect the current Ready state. Maintainer: add run-ci and trigger a fresh attempt; functional tests have not run in that old workflow.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492640626/job/122150134982); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492640595/job/122150134832); [Check PR title](https://github.com/sgl-project/sglang/actions/runs/40492640193/job/122150139430); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492640629/job/122150249116); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492640511/job/122150226596); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_indexer_idle.py
```


### [AMD-AGI/Infera #237](https://github.com/AMD-AGI/Infera/pull/237) — Migrate agent_sys Forge workflows to Hyperloom and public AKLR knowledge

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: xiaobochen-amd, jiejingzhangamd, limou102.
- **Base / head:** `main` @ `5e253783fb318fb9d66d552d193d8aadb93b5e07` / `dev/yihou/agent_sys/new_kf` @ `14be5a14c8444889cf7b626a6c6454175b745d48`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 14 unresolved inline threads. Checks: FAILURE 1, QUEUED 7, SUCCESS 8. Reviews: dorado269: COMMENTED (2026-11-11T07:39:11Z), dorado269: COMMENTED (2026-11-11T07:43:50Z), dorado269: COMMENTED (2026-11-11T07:46:09Z), dorado269: COMMENTED (2026-11-11T07:49:15Z), dorado269: COMMENTED (2026-11-11T07:51:41Z), dorado269: COMMENTED (2026-11-11T07:53:06Z), dorado269: COMMENTED (2026-11-11T07:53:55Z), dorado269: COMMENTED (2026-11-11T07:55:10Z), dorado269: COMMENTED (2026-11-11T07:57:18Z), dorado269: COMMENTED (2026-11-11T07:58:52Z), dorado269: COMMENTED (2026-11-11T08:00:19Z), dorado269: COMMENTED (2026-11-11T08:01:37Z), dorado269: COMMENTED (2026-11-11T08:02:22Z), dorado269: COMMENTED (2026-11-11T08:04:27Z), dorado269: COMMENTED (2026-11-11T08:34:52Z), dorado269: COMMENTED (2026-11-11T08:35:13Z), dorado269: COMMENTED (2026-11-11T08:35:15Z), dorado269: COMMENTED (2026-11-11T08:35:17Z), dorado269: COMMENTED (2026-11-11T08:35:19Z), dorado269: COMMENTED (2026-11-11T08:35:22Z), dorado269: COMMENTED (2026-11-11T08:35:24Z), dorado269: COMMENTED (2026-11-11T08:35:26Z), dorado269: COMMENTED (2026-11-11T08:35:28Z), dorado269: COMMENTED (2026-11-11T08:35:30Z), dorado269: COMMENTED (2026-11-11T10:12:13Z), dorado269: COMMENTED (2026-11-11T10:15:22Z), dorado269: COMMENTED (2026-11-11T10:18:14Z), dorado269: COMMENTED (2026-11-11T10:20:51Z), dorado269: COMMENTED (2026-11-11T10:33:34Z), dorado269: COMMENTED (2026-11-11T11:01:22Z), dorado269: COMMENTED (2026-11-11T11:01:25Z).
- **Last curated action:** Discovered by the daily author-wide query.
- **Latest GitHub activity:** PR updated 2026-11-11T11:16:54Z; no issue comments.
- **Evidence freshness:** NOT AUDITED. Notes last assessed unrecorded; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** No curated evidence yet; inspect the PR.
- **To Ready:** New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.
- **To merge / dependency:** Resolve current review and required CI; no automatic merge certification.

- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4535211786) in `agent_sys/tools/skill/agent-sys-kernel-forge/references/aklr.md` (outdated=True); inspect the current diff before resolving.
- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4535246425) in `agent_sys/tools/skill/agent-sys-kernel-forge/references/sources.md` (outdated=True); inspect the current diff before resolving.
- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4535264102) in `agent_sys/tools/skill/agent-sys-kernel-forge/references/sources.md` (outdated=True); inspect the current diff before resolving.
- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4535291677) in `agent_sys/tools/skill/agent-sys-kernel-forge/references/sources.md` (outdated=True); inspect the current diff before resolving.
- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4535311477) in `agent_sys/tools/skill/agent-sys-kernel-forge/references/sources.md` (outdated=True); inspect the current diff before resolving.
- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4535322058) in `agent_sys/tools/skill/agent-sys-kernel-forge/references/sources.md` (outdated=True); inspect the current diff before resolving.
- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4535338477) in `agent_sys/examples/03_kernelforge_one_kernel/README.md` (outdated=True); inspect the current diff before resolving.
- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4535376285) in `agent_sys/tests/examples/test_forge_migration.py` (outdated=True); inspect the current diff before resolving.
- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4535386318) in `agent_sys/tools/skill/agent-sys-kernel-forge/references/campaign.md` (outdated=True); inspect the current diff before resolving.
- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4535408010) in `agent_sys/tools/skill/agent-sys-kernel-forge/references/aklr.md` (outdated=False); inspect the current diff before resolving.
- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4536584305) in `agent_sys/examples/03_kernelforge_one_kernel/assets/fixture/workset/items/codes/workset.yaml` (outdated=True); inspect the current diff before resolving.
- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4536614081) in `agent_sys/examples/03_kernelforge_one_kernel/README.md` (outdated=False); inspect the current diff before resolving.
- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4536641514) in `agent_sys/examples/README.md` (outdated=False); inspect the current diff before resolving.
- Unresolved [review thread](https://github.com/AMD-AGI/Infera/pull/237#discussion_r4536665385) in `agent_sys/tools/skill/agent-sys-kernel-forge/SKILL.md` (outdated=True); inspect the current diff before resolving.
- Red-check entry points: [rust](https://github.com/AMD-AGI/Infera/actions/runs/43009954148/job/130487258682).


## Draft

### [AMD-AGI/Infera #221](https://github.com/AMD-AGI/Infera/pull/221) — Publish six KernelForge families and their historical evidence.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `dev/yihou/kernel-registry` @ `c80d852fce60ecee5706fd7cd86d847e6d4643b3` / `dev/yihou/glm52-kernel-catalog` @ `b8b3938d6b7c8713711a7f0859e3f5c7bf548935`.
- **Review / CI:** no decision; mergeability MERGEABLE / CLEAN; 0 unresolved inline threads. Checks: SUCCESS 3. Reviews: none.
- **Last curated action:** 2026-11-02: reconciled current GitHub state and archived evidence for handoff.
- **Latest GitHub activity:** PR updated 2026-10-28T10:55:40Z; no issue comments.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** PR reports 741 payload hashes, 113 generated records, 103 measurements, six CPU semantic references, Ruff/Pyright; 239 payloads unchanged and 502 changed by redaction. See agent_sys/kernel_registry/catalogs/glm52/{README,REDACTION,EVIDENCE}.md in the PR.
- **To Ready:** Retarget/reconcile with main after merged #220. Resolve 20 unavailable external fixtures; validate the 502 redacted derivatives before presenting them as tested kernels, or scope the PR explicitly to archival evidence. Re-run catalog validation/CPU checks after retarget.
- **To merge / dependency:** Yihou/delegate: scope and evidence decision; maintainers: catalog review and CI on main. Historical results apply to original bytes, not renamed derivatives.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
# Follow the catalog README validation commands at the exact PR head.
```


### [kvcache-ai/Mooncake #5005](https://github.com/kvcache-ai/Mooncake/pull/5005) — Offer opt-in HIP copy-stream priority without changing the default.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `72e267f41d7eef241b641b9d132de431adf71ef8` / `yihou/hip-copy-stream-priority-20261027` @ `8313cccbaae8f3c9d7e0a29b830ce6a8a892dd91`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 3, SKIPPED 1, SUCCESS 31. Reviews: none.
- **Last curated action:** 2026-11-02: signed rebase onto 72e267f4; packaging test/lint pass; contributed CUDA build failure evidence to #5144.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:18:39Z](https://github.com/kvcache-ai/Mooncake/pull/5005#issuecomment-6568944195); PR updated 2026-11-02T04:28:10Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/README.md) R039/R043 microbench; allocated MI355X HIP build +4/4 CTest before rebase, stream/test sources unchanged; source packaging regression1pass. #5030 fixed previous wheel failure; new missing mooncake_te_build_config.h is covered by #5144.
- **To Ready:** Yihou must perform the human line-by-line review required by Mooncake AGENTS.md. Reconcile the current base CUDA build regression with #5144 and rerun affected CI. Existing HIP tests count; no blanket new serving benchmark.
- **To merge / dependency:** Follow external #5144; after merge rebase and rerun CUDA build checks. #4953 source compatibility was checked but combined runtime was not. Maintainers review after human submitter review.

- Red-check entry points: [CTest unit tests (Python 3.12)](https://github.com/kvcache-ai/Mooncake/actions/runs/41387680143/job/125092088899); [tent-ci (cuda-on)](https://github.com/kvcache-ai/Mooncake/actions/runs/41387680143/job/125092276800); [CI Gate](https://github.com/kvcache-ai/Mooncake/actions/runs/41387680143/job/125102510508).

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
python -m pytest -q python/tests/packaging/test_project.py::test_structured_object_storage_has_one_authoritative_source
# HIP build/CTest commands and original node evidence: PR body and reviews/M02-followup-body-20261102.md
```


### [sgl-project/sglang #46819](https://github.com/sgl-project/sglang/pull/46819) — Cache DCP gather metadata instead of rebuilding it per transfer.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `8a455f1b9005a05091ad1825f0f3493bfd74d27d` / `yihou/glm52-dcp-pack-cache-20261027` @ `aed9bc6953d0a95c879cff5151ee20627e9937b3`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 18, SKIPPED 87, SUCCESS 24. Reviews: none.
- **Last curated action:** 2026-11-03: verified unchanged head and Draft state; newest lint passes. No new functional CI result or evidence resolving the existing readiness blockers.
- **Latest GitHub activity:** [Latest comment 2026-10-27T11:02:43Z](https://github.com/sgl-project/sglang/pull/46819#issuecomment-6440276807); PR updated 2026-11-02T04:22:19Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-03; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.dcpx-pack-fix-best.packup_20261023](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.dcpx-pack-fix-best.packup_20261023/README.md) dcpx8c/9/10; current-head MI355X 3 tests +16 subtests, CPU 8+4; isolated cache medians 39.57→17.15 /39.09→17.07 /130.42→107.52 us. dcpx10 also removed a wait; its combined speedup does not transfer to this PR.
- **To Ready:** CPU-first option: narrow to the independently measured cache and update tests/body. If retaining pinned-index/priority changes, qualify their lifetime, interference and PP ordering on allocated hardware. Preserve producer wait.
- **To merge / dependency:** Still Draft with the recorded Ready blockers; latest lint passes. [Latest Base](https://github.com/sgl-project/sglang/actions/runs/40491188455) rejects Draft and skips functional tests. Yihou/delegate: complete the scoped Ready action first; maintainers then enable required CI and review. Related #44724/#39768/#43782 are overlap coordination, not established mandatory prerequisites.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40491188646/job/122145248512); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40491188142/job/122145248715); [Check PR title](https://github.com/sgl-project/sglang/actions/runs/40491187276/job/122145248409); [Check PR title](https://github.com/sgl-project/sglang/actions/runs/40492975001/job/122151238195); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40491188159/job/122145348637); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m pytest -q test/registered/unit/disaggregation/test_dcp_pack_metadata.py test/registered/unit/disaggregation/test_dcp_pack.py
# GPU only when qualifying the retained stream/lifetime changes
PYTHONPATH=python python -m pytest -q test/registered/kernels/ops/kvcache/test_pd_dcp_gather.py
```


### [sgl-project/sglang #46824](https://github.com/sgl-project/sglang/pull/46824) — Relayout DCP-sharded draft KV together with target KV.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `e0c7992d0d48c8085655752de7a9f43bf1862bbc` / `yihou/pd-dcp-sharded-draft-20261027` @ `1bcfc52149011aae08502e9da0620b0325492799`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 17, SKIPPED 88, SUCCESS 23. Reviews: none.
- **Last curated action:** 2026-11-03: verified unchanged head and Draft state; newest lint passes. No new functional CI result or evidence resolving the existing readiness blockers.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:20:34Z](https://github.com/sgl-project/sglang/pull/46824#issuecomment-6568968279); PR updated 2026-11-02T04:21:11Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-03; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.dcp4-pd.packup_20261022](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.dcp4-pd.packup_20261022/README.md) pdBe-r1 math29×2, passkey8K/34K/96K; rebase production delta unchanged; 43 CPU tests +61 subtests exercise merged sharding policy.
- **To Ready:** Map pdBe-r1 to current supported combinations; resolve or explicitly reject the known DP2×DCP2 graph-garbling case. Test only uncovered MTP/prefix/PP cases on allocated hardware as needed. #44322 has merged; old backend hold is cleared.
- **To merge / dependency:** Still Draft with the recorded Ready blockers; latest lint passes. [Latest Base](https://github.com/sgl-project/sglang/actions/runs/41387906434) rejects Draft and skips functional tests. Yihou/delegate: complete the scoped Ready action first; maintainers then enable required CI and review. #46824 must land before #46826 enables the configuration.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387906323/job/125092720788); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387903680/job/125092712814); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387906431/job/125092720718); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387906168/job/125092787241); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387906229/job/125092786601); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/disaggregation/test_dsa_pd_dcp.py test/registered/unit/disaggregation/test_dcp_pack.py test/registered/unit/disaggregation/test_mooncake_transfer_batching.py
```


### [sgl-project/sglang #46826](https://github.com/sgl-project/sglang/pull/46826) — Allow DCP decode behind DCP1 prefill when transport/backend support it.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `e0c7992d0d48c8085655752de7a9f43bf1862bbc` / `yihou/rocm-dcp-pd-config-20261027` @ `21f8e688faa155a1f5abaa53ba3ca03c20900d7c`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 17, SKIPPED 88, SUCCESS 23. Reviews: none.
- **Last curated action:** 2026-11-03: verified unchanged head and Draft state; newest lint passes. No new functional CI result or evidence resolving the existing readiness blockers.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:22:46Z](https://github.com/sgl-project/sglang/pull/46826#issuecomment-6568996226); PR updated 2026-11-02T04:23:15Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-03; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.dcp4-pd.packup_20261022](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.dcp4-pd.packup_20261022/README.md) pdBe-r1; current 12 CPU tests +13 subtests cover implicit Triton and a2a Q-rep defaults. Combined graph/coverage gap remains.
- **To Ready:** Complete the supported-configuration coverage with #46824; preserve rejection of unqualified graph/layout combinations. Backend/default policy is already tested; do not wait again for merged #44322.
- **To merge / dependency:** Still Draft with the recorded Ready blockers; latest lint passes. [Latest Base](https://github.com/sgl-project/sglang/actions/runs/41388093572) rejects Draft and skips functional tests. Yihou/delegate: complete the scoped Ready action first; maintainers then enable required CI and review. #46824 must land before #46826 enables the configuration.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388093425/job/125093288690); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388091310/job/125093282853); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388093499/job/125093288867); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388093826/job/125093355564); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388093534/job/125093360115); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/test_dsa_pd_dcp_config.py test/registered/unit/test_dsa_dcp_backend_defaults.py
```


### [AMD-AGI/Infera #228](https://github.com/AMD-AGI/Infera/pull/228) — Permanent Draft tracker for 48k deployment priorities, implementation PRs and experiment references.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: xiaobochen-amd, jiejingzhangamd, limou102.
- **Base / head:** `main` @ `8847491c2856d5bc43e230f413c62afe68147e46` / `dev/yihou/glm52-pr-tracking` @ `327b4cc9d1e8672a59cd460838485cae58229c06`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: SKIPPED 8, SUCCESS 4. Reviews: none.
- **Last curated action:** 2026-11-02: owner requested permanent Draft for view/track; converted #228 to Draft and added a daily Draft-state guard.
- **Latest GitHub activity:** PR updated 2026-11-11T09:01:35Z; no issue comments.
- **Evidence freshness:** HEAD CHANGED: re-audit evidence from 2cb681fc1e. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** Five Markdown files from main 8847491c; applicable pre-commit hooks and diff check passed; 21 pinned archive references and local links checked; restricted repository identifiers excluded. Separate publisher safety tests: 14 CPU tests.
- **To Ready:** PERMANENT DRAFT: for view and track only. Never promote this documentation PR to Ready.
- **To merge / dependency:** DO NOT MERGE. Maintain the open Draft as the team coordination surface; review/merge priorities apply to linked implementation PRs only.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
pre-commit run --files docs/glm52-pr-tracking/*.md
git diff --check
```


## External dependencies and equivalent work

These PRs belong to their original authors. Contribute evidence there; do not open duplicates or edit their branches.

| PR | Purpose | State / head | Dependency / next action |
|---|---|---|---|
| [AMD-AGI/Infera #210](https://github.com/AMD-AGI/Infera/pull/210) | v0.5.22 integrated image recipe | MERGED / `8859b247ce` | Merged image history; reproduce pinned fdbafb60 source rather than substitute arbitrary main. |
| [SemiAnalysisAI/InferenceX #4629](https://github.com/SemiAnalysisAI/InferenceX/pull/4629) | Publish the Infera SGLang AgentX PD recipe. | CLOSED / `69be557c8c` | Upstream #33059/#46830/#46820; progress externally owned absorb-BMM #46918 for issue #46829, now covered by scoped MI355X validation (CI/review/merge and image integration remain); select an image with these plus merged #21194 and remove engine patches. Coordinate #630/#47117, diagnose platform RDMA bring-up, complete sweep and review. |
| [sgl-project/sglang #21194](https://github.com/sgl-project/sglang/pull/21194) | Fix GLM PP construction on stages without an embedding. | MERGED / `2059478e21` | Validate a pinned image carrying the merged fix. No duplicate filing; preserve historical source and attribution. |
| [sgl-project/sglang #34465](https://github.com/sgl-project/sglang/pull/34465) | Earlier PP non-first-rank hidden-size proposal | CLOSED / `62cd16df78` | Closed unmerged. Third-party #21194 subsequently merged on October 29 and guards the GLM-5.2 PP construction failure. Validate image inclusion; no new duplicate S13 filing. Preserve archived source and attribution. |
| [sgl-project/sglang #35233](https://github.com/sgl-project/sglang/pull/35233) | HiCache host-pointer alias resolution | MERGED / `c29e6054f5` | Merged into base lineage; do not reapply old #33968 workaround. |
| [sgl-project/sglang #41395](https://github.com/sgl-project/sglang/pull/41395) | Early-send consumer completion | MERGED / `9fc5f30621` | Merged in frozen base; preserve separate producer patch. |
| [sgl-project/sglang #42553](https://github.com/sgl-project/sglang/pull/42553) | Host-allocation pointer handling | MERGED / `b2de502825` | Merged upstream; retain correct frozen-source behavior. |
| [sgl-project/sglang #45524](https://github.com/sgl-project/sglang/pull/45524) | Producer event before non-final PP KV send | READY / `ef18975a8a` | Base audit 2026-11-11T12:10:20.838177+00:00: Base plus AMD/Xeon/ARM64: shared PP producer/send synchronization. Gate, finish and 27 functional Base jobs passed (A9/B13/C5); JIT skipped. AMD has 12 passing stages, one failed shard and four skipped stages. Shard 6 started tests and exited 255; fresh log retrieval returned HTTP 404. Xeon/ARM64 passed. External author eaglewangao-us/maintainers retrieve runner logs and recover AMD coverage; complete source-equivalence review while preserving passing Base/disaggregation, historical AWQ/PP evidence and frozen producer patch. |
| [sgl-project/sglang #46918](https://github.com/sgl-project/sglang/pull/46918) | Keep absorb-BMM hand tuning while limiting the waves-per-EU budget that may trigger the gfx950 compiler abort. | READY / `bb5ae4af94` | Base audit 2026-11-11T12:10:20.838177+00:00: Base only: ROCm-guarded absorb-BMM tuning. Gate and 36 functional jobs passed (A19/B12/C5). Two B shards cancelled without test steps; finish failed on cancellation. Inherited evidence identifies github-actions[bot]; cancellation policy remains unknown. JIT skipped. External author yu-depend/maintainers own original-PR conflict, coverage and review decisions; progress owned continuation #49732 separately. Preserve frozen workaround and MI355X evidence at 9763ef56, which does not qualify current main/image. |
| [AMD-AGI/Infera #222](https://github.com/AMD-AGI/Infera/pull/222) | Advertise the KV-events port allocated by srt-slurm. | MERGED / `f668fbf773` | Use a release carrying this fix; verify allocated endpoints with colocated prefills. |
| [NVIDIA/srt-slurm #630](https://github.com/NVIDIA/srt-slurm/pull/630) | Launch Infera frontends with etcd-registered SGLang workers. | READY / `5bf86633c6` | Complete frontend integration review; reconcile the recipe backport when merged. |
| [kvcache-ai/Mooncake #4540](https://github.com/kvcache-ai/Mooncake/pull/4540) | Select destination-local GPU write rails. | DRAFT / `6fccbba168` | Coordinate fabric-specific validation and review; not a first-version universal prerequisite. |
| [kvcache-ai/Mooncake #4541](https://github.com/kvcache-ai/Mooncake/pull/4541) | Make RDMA ACK timeout configurable. | DRAFT / `faa24ab499` | Coordinate fabric-specific validation and review; do not attribute RDMA registration EINVAL to ACK timeout without evidence. |
| [sgl-project/sglang #45307](https://github.com/sgl-project/sglang/pull/45307) | Overlapping gfx950 fused indexer optimization | READY / `1255a0a4ed` | Coordinate overlap with our #47276; rebase whichever lands second. |
| [sgl-project/sglang #47117](https://github.com/sgl-project/sglang/pull/47117) | Document Infera GLM-5.2 PD on MI355X. | DRAFT / `3e89cdc460` | Original author llying-001 owns code and promotion; coordinate cookbook review/merge with InferenceX #4629. Maintainer approval is needed for Mintlify fork preview. Lint passes; Base is green only through documentation filtering with functional suites skipped. Extra is blocked by Draft, not a test assertion. Yihou supports evidence/command review without taking over the branch. |
| [ROCm/aiter #6697](https://github.com/ROCm/aiter/pull/6697) | Related upstream MoE sort-into-GEMM1 experiment; not our five-hook KF port. | READY / `9a20161613` | External author: resolve current main conflicts and respond to the design objection about merging sorting into GEMM1. Track relevance; do not duplicate or modify the author branch. Our separate five-patch MoE port remains held until an active path and current-baseline gain are demonstrated. |
| [SemiAnalysisAI/InferenceX-app #1542](https://github.com/SemiAnalysisAI/InferenceX-app/pull/1542) | Display the Infera SGLang framework name in the dashboard. | DRAFT / `4aa976240e` | Dashboard review/merge after recipe coordination. |
| [kvcache-ai/Mooncake #4340](https://github.com/kvcache-ai/Mooncake/pull/4340) | HIP IPC suballocation relocation | READY / `d70e5eabea` | salexspb owns root fix; our R018/R019 evidence is posted. Not used in llr37 RDMA recipe. |
| [kvcache-ai/Mooncake #4953](https://github.com/kvcache-ai/Mooncake/pull/4953) | Asynchronous HIP copy submission | MERGED / `41112e77f4` | Track interaction with #5005; source API unchanged, combined runtime not qualified. |
| [kvcache-ai/Mooncake #5030](https://github.com/kvcache-ai/Mooncake/pull/5030) | Wheel structured-ranges packaging fix | MERGED / `a068e70bef` | Merged and included in #5005 rebase; prior wheel blocker cleared. |
| [kvcache-ai/Mooncake #5144](https://github.com/kvcache-ai/Mooncake/pull/5144) | CUDA guard target generated-header include path | MERGED / `39c861c81e` | Existing fix for current #5005 inherited CUDA build failure; follow merge then rebase, no duplicate. |
| [sgl-project/sglang #39768](https://github.com/sgl-project/sglang/pull/39768) | PP prefill/DCP transfer work | READY / `b892b1f0e3` | Overlap coordination for #46819; do not infer required dependency. |
| [sgl-project/sglang #42163](https://github.com/sgl-project/sglang/pull/42163) | Shard transfer queues across rooms | DRAFT / `294113a01f` | YAMY1234 owns the draft. Our dcpx/g4tune data is posted; mixed-shard TPOT regression disclosed. Not used in llr37. |
| [sgl-project/sglang #43299](https://github.com/sgl-project/sglang/pull/43299) | Non-CUDA/NIXL producer synchronization | MERGED / `b5d0769169` | Merged; distinct from open #45524. |
| [sgl-project/sglang #43782](https://github.com/sgl-project/sglang/pull/43782) | DCP gather kernel changes | READY / `dfdfc5502a` | Overlap coordination for #46819; compare final source if it lands. |
| [sgl-project/sglang #44322](https://github.com/sgl-project/sglang/pull/44322) | ROCm DCP backend and sharding policy | MERGED / `b8a0a1ac70` | Merged; old D10/D11 backend dependency cleared. Do not keep waiting for this PR. |
| [sgl-project/sglang #44724](https://github.com/sgl-project/sglang/pull/44724) | DCP transfer staging changes | READY / `2ae31e9a73` | Overlap coordination for #46819; not an established merge prerequisite. |
| [sgl-project/sglang #45156](https://github.com/sgl-project/sglang/pull/45156) | Initialize the active idle attention backend | READY / `c02b88d488` | WQW-Git owns the fix; our duplicate #45483 is closed. Maintainer CI/review; no exact-head hardware claim. |
| [sgl-project/sglang #45337](https://github.com/sgl-project/sglang/pull/45337) | Validate prefill_complete using effective attention DP | READY / `910c16cb4e` | nvjullin owns the fix; address performance-review concerns. Historical on_complete item; llr37 uses completion, not prefill_complete. |
| [sgl-project/sglang #45453](https://github.com/sgl-project/sglang/pull/45453) | Agree across ranks before deferred KV release | MERGED / `2647c3761e` | Merged release hunk; residual empty-queue polling guard has no current reproducer. |
| [sgl-project/sglang #45796](https://github.com/sgl-project/sglang/pull/45796) | Copy host lengths before graph padding | MERGED / `e122674f23` | Merged; covers R24/R25 later row-count fault. |
| [sgl-project/sglang #46424](https://github.com/sgl-project/sglang/pull/46424) | Sparse output batch-major padding | MERGED / `e536a1144f` | Merged; no duplicate D23. Not the baked page-table row patch. |
| [sgl-project/sglang #46861](https://github.com/sgl-project/sglang/pull/46861) | Shared fused-KV allocation workaround on ROCm | MERGED / `b24937a833` | Merged workaround for #33059 CI baseline issue; not Mooncake root-fix completion. |
| [sgl-project/sglang #47180](https://github.com/sgl-project/sglang/pull/47180) | Q replication with per-tensor FP8 scales | MERGED / `bb767aa490` | Merged; original D22 enablement retired. Separate graph/numerical qualification remains. |
| [sgl-project/sglang #47184](https://github.com/sgl-project/sglang/pull/47184) | Rebase the original PP placeholder-attribute repair. | CLOSED / `ce3087c8bb` | Original author: verify a nightly carrying #21194 and retire this redundant PR. |
| [sgl-project/sglang #45233](https://github.com/sgl-project/sglang/pull/45233) | Speculative penalty accounting proposal | DRAFT / `612b93311b` | Closed unmerged; recheck replacements before reopening work. |

## Issue-only work and patches without a PR

| Issue | State | Updated |
|---|---|---|
| [[Bug] ROCm absorb-BMM hand config aborts Triton compilation at M=256 on gfx950](https://github.com/sgl-project/sglang/issues/46829) | OPEN | 2026-10-27T11:15:22Z |
| [[Bug] ROCm DCP-PD allocator reserve growth precedes HIP resource aborts under burst load](https://github.com/sgl-project/sglang/issues/46831) | OPEN | 2026-10-30T13:42:15Z |

Required unpublished S13, A01, the full historical K01 overlay, I07 and inherited image patches are recorded in [deployment requirements](README.md). Remaining exclusions and reasons are in [PR_PACKUP_MAP.md](PACKUP_MAP.md).
