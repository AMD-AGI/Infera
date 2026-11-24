# Active PR handoff

GitHub refresh: **2026-11-24T00:08:22.315296+00:00**. [Overview and priorities](README.md).

## Ready for review

### [sgl-project/sglang #47276](https://github.com/sgl-project/sglang/pull/47276) — Speed up gfx950 fused indexer logits/top-k for MTP verify rows.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me, DarkSharpness, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `33a03ba42ee4c6036d75c22e71b9bb8600c79355` / `yihou/kf-fused-indexer-gfx950` @ `e95ec5abc8b4a9bf27b3dae160e42dc5beeb5f6e`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: CANCELLED 1, FAILURE 1, SKIPPED 78, SUCCESS 102. Reviews: claude: COMMENTED (2026-10-29T13:35:23Z).
- **Last curated action:** Automated 3-hour audit 2026-11-24T00:07:14.229187+00:00: Freshly verified authorized dorado269 fork/branch, unchanged e95ec5ab, Ready/run-ci, current checks, paginated diff/reviews/comments, six diff blobs and required-main ancestry. Base attempt 3 remains fully successful; executed-step timestamps match prior evidence, with no tests started this cycle. No new feedback, retry or rebase. MERGEABLE/BLOCKED/REVIEW_REQUIRED; passing head preserved.
- **Latest GitHub activity:** [Latest comment 2026-11-19T13:39:28Z](https://github.com/sgl-project/sglang/pull/47276#issuecomment-7010787687); PR updated 2026-11-20T09:02:55Z.
- **Evidence freshness:** HEAD CHANGED: re-audit evidence from 674ad71976. Notes last assessed 2026-11-24; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** PR-reported bit-identical logits/top-2048/K-write controls and real-acceptance TP8 GSM8K; isolated whole-call 1.20–1.36x; serving ITL used simulated acceptance 3.61. Historical [glm52.llrepro-kf-ar-topk.packup_20261024](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-kf-ar-topk.packup_20261024/README.md) is a related source family, not a verified final-port run mapping.
- **To Ready:** Already Ready. Preserve independent numerical evidence and coordinate #45307; bind final source/run IDs to a public result bundle when available.
- **To merge / dependency:** Base audit 2026-11-24T00:07:14.229187+00:00: Base only: guarded gfx950 fused-indexer kernels/dispatch do not demonstrate shared-backend behavior change. Attempt 3 gate, finish and all 35 functional jobs passed (A13/B17/C5), plus seven separate JIT jobs. Inherited accounting: 12 functional/one JIT passes retained from earlier attempts. Fresh executed-step timestamps match prior evidence; zero tests started this cycle. Twelve auxiliary jobs skipped. Historical FlashInfer causality and independent review/source/numerical qualifications remain. Maintainers review passing e95ec5ab and coordinate #45307 overlap. Preserve passing head; no freshness retry/rebase. Owner archives exact source/run mapping and independent numerical evidence. Optional and disabled in frozen llr37. Historical FlashInfer causality remains unproven without matching controls. Only a future cooldown-gate failure requires an authorized helper-policy audit.

- Red-check entry points: [pr-test-amd-finish](https://github.com/sgl-project/sglang/actions/runs/44478753078/job/136190090344).

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
# GPU: use a pool node; exact architecture/flags are in the PR
PYTHONPATH=python python -m pytest -q test/registered/amd/test_dsa_gfx950_fused_indexer.py
```


### [sgl-project/sglang #47278](https://github.com/sgl-project/sglang/pull/47278) — Speed up gfx950 FP8 sparse MLA decode with a 64-head split-K plan.

- **State / responsibility:** READY; yihou; kernel delegate requested from owner. GitHub assignees: none. Requested reviewers: DarkSharpness, HaiShaw, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `459b85fb6884b980a69c376ba2c4513ff9b0c71f` / `yihou/kf-sparse-mla-gfx950` @ `9602092ac908c2077a400abe7aa3d1a1b26a9220`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: SKIPPED 78, SUCCESS 102. Reviews: claude: COMMENTED (2026-10-29T13:43:32Z), kkHuang-amd: APPROVED (2026-11-23T05:37:43Z).
- **Last curated action:** Automated 3-hour audit 2026-11-24T00:07:14.229187+00:00: Freshly verified authorized unchanged 9602092a, exact parent 8d72d2ea, both diff blobs, Ready/run-ci and unchanged reviews. Base/AMD remain fully passed; kkHuang-amd APPROVED/LGTM binds to this head. Aggregate remains MERGEABLE/BLOCKED/REVIEW_REQUIRED. Refreshed local review/validation handoffs; CPU/GPU/e2e results remain inherited. No source change or retry.
- **Latest GitHub activity:** [Latest comment 2026-11-20T15:40:00Z](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032); PR updated 2026-11-23T05:37:43Z.
- **Evidence freshness:** HEAD CHANGED: re-audit evidence from a5d8b3af95. Notes last assessed 2026-11-24; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** PR reports 22/22 kernel tests, FP32-reference SNR >35 dB and real-acceptance GSM8K mean .942 baseline / .941 PR; isolated kernel 2.4–3.2x. Serving ITL used simulated acceptance 3.61. Final-port pack-up mapping still needs audit. Conflict handoff: [risk and exact source anchors](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/reviews/PR47278_CONFLICT_HANDOFF_20261110.md).
- **To Ready:** Already Ready. Preserve the BF16-partial accuracy change and DCP/LSE coverage; raw run-to-source archiving remains a handoff follow-up.
- **To merge / dependency:** Base audit 2026-11-24T00:07:14.229187+00:00: Base plus AMD: common raw-FP8 fused-launch lowering and sparse-MLA split-K contracts justify AMD coverage; tuned plan is gfx950/H64 guarded. Base attempt 1 gate/finish and 33 functional jobs (A12/B16/C5) plus seven JIT jobs passed; 12 auxiliary jobs skipped. AMD attempt 1 gate/finish and 17 functional jobs (A1/B13/C3) plus two JIT jobs passed; five auxiliary jobs skipped. No previous-attempt retention. Executed-step timestamps match prior evidence; zero new tests this cycle. Inherited matching logs record 45 sparse-MLA passes and four disaggregation files/13 cases. Author current-code e2e A/B and exact-head human approval remain separate from raw provenance and aggregate review. Preserve passing 9602092a Base/AMD; no freshness retry/rebase. Maintainers assess remaining REVIEW_REQUIRED/BLOCKED requirements despite exact-head approval. Owner archives exact-parent 8d72d2ea source/image/raw evidence. Retain seq15/non-target numerical, BF16-partial, scaled-KV, DCP/LSE and layout/replay qualifications. Author reports current-code A/B with 10.3% lower ITL; TPOT/throughput overlap. Compiler causality is suspected/unconfirmed and reviewer-nonblocking; warmup is author-reported. Retrieve existing evidence before necessary authorized GPU work; no machine-pool skill here. No issue/review reply without owner authorization. Local follow-up: cycles/20261124T000002Z/PR47278_REVIEW_HANDOFF.json and PR47278_GPU_VALIDATION.json. Required source-family port, not byte-identical frozen overlay.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
# GPU: use a pool node; preserve PR architecture and dtype matrix
PYTHONPATH=python python -m pytest -q test/registered/kernels/ops/attention/test_triton_sparse_mla_hip.py
```


### [sgl-project/sglang #47872](https://github.com/sgl-project/sglang/pull/47872) — Provide opt-in long-context top-k on compact page tables with exact sampling fallbacks.

- **State / responsibility:** READY; yihou; active tracking paused by owner pending leader-requested redesign. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me, DarkSharpness, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `e1cd0c55e553b3f9440f23835e9ac2777f2bfe92` / `yihou/kf-triton-topk-gfx950` @ `d60269391fbb071a74e190ff7acabf0819aa5546`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 15, SKIPPED 80, SUCCESS 83. Reviews: claude: COMMENTED (2026-11-02T07:47:39Z).
- **Last curated action:** Automated 3-hour audit 2026-11-24T00:07:14.229187+00:00: Outside the resumed scope; historical/unrefreshed row copied without lifecycle, CI or review queries, retry, rebase or modification. Owner-paused pending redesign.
- **Latest GitHub activity:** [Latest comment 2026-11-19T01:46:32Z](https://github.com/sgl-project/sglang/pull/47872#issuecomment-6997141745); PR updated 2026-11-19T20:56:29Z.
- **Evidence freshness:** HEAD CHANGED: re-audit evidence from 1061c29838. Notes last assessed 2026-11-24; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** [Historical indexer source](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/patches/K01-kf-decode-kernel-overlay.NEWFILES.diff); [PR evidence](https://github.com/sgl-project/sglang/pull/47872). Author reports exact overflow fallback, valid nondeterministic boundary ties and 1.5–1.9x isolated speedup at 400k contexts/72+ rows, but about 1.3% slower serving at 32k with simulated acceptance 3.61. Real-acceptance GSM8K .953 from one server is a smoke test. Maintenance reproduced candidate underflow in the actual GPU pipeline: original 256 valid slots instead of 2048; fixed 2048 unique valid slots and all 5 GPU tests pass without skips. Run kf47872-20261102T1038Z-topk-underflow used MI355X and the public v0.5.22 ROCm10 image, Torch 2.11.0+rocm10.0.0 / Triton 3.8.0. [Pinned source, CPU/GPU records and reproduction scripts](https://github.com/AMD-AGI/Infera/tree/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/validation/kf_47872). The GPU runner uses the exact module/tests with namespace and CI-registration shims for the older image; no new server/model/performance result is claimed.
- **To Ready:** Ready restored after the underflow repair, import-guard regression and exact-source MI355X validation. The temporary correctness Draft is resolved; retain opt-in behavior and the historical performance limits.
- **To merge / dependency:** Base audit 2026-11-24T00:07:14.229187+00:00: Historical/unrefreshed evidence copied from the November 13 result; outside the November 18 resumed scope. Owner-paused; no lifecycle, CI or review queries. Historical head dccb51e0f9f3d97dc01aea7aa39db1c0cfe3df34 and Base run 42722204156/attempt 1 remain inherited evidence only. The hold is neither a failure nor a pass. Outside the resumed scope; historical/unrefreshed constraints only, requiring explicit owner resumption before active work: Owner-paused pending leader-requested redesign; resume only on owner instruction. Preserve historical source/run, opt-in/fallback/tie and 32k/long-context qualifications.

- Red-check entry points: [base-b-disaggregation-test-large-8-gpu-amd-mi35x (rocm10, linux-mi35x-gpu-8.fabric)](https://github.com/sgl-project/sglang/actions/runs/44479273883/job/135413156752); [pr-test-amd-finish](https://github.com/sgl-project/sglang/actions/runs/44479273883/job/135876488181); [wait-for-base-b](https://github.com/sgl-project/sglang/actions/runs/44479274015/job/135426496637); [call-jit-kernel-tests / jit-kernel-unit-test-1-gpu-large](https://github.com/sgl-project/sglang/actions/runs/44479274015/job/135407737053); [call-jit-kernel-tests / jit-kernel-benchmark-test-1-gpu-large](https://github.com/sgl-project/sglang/actions/runs/44479274015/job/135407737136); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/amd/test_dsa_triton_topk_transform.py
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_triton_topk_gate.py
```


### [sgl-project/sglang #47897](https://github.com/sgl-project/sglang/pull/47897) — Speed up gfx950 decode logits while preserving paired-row tails and fallback dispatch.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: DarkSharpness, HaiShaw, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `2544c9ac34be4447bdffa612c37015a3f8da9e84` / `yihou/kf-gluon-indexer-logits-gfx950` @ `f08bde1f0ad9fea26a9cb629dfc512a540ad9fc0`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: SKIPPED 1, SUCCESS 5. Reviews: claude: COMMENTED (2026-11-02T09:31:07Z).
- **Last curated action:** Automated 3-hour audit 2026-11-24T00:07:14.229187+00:00: Followed unchanged interactive f08bde1f. Freshly verified author/fork/branch, Ready/run-ci, three diff blobs, six source snapshots, two freshly fetched callers, unchanged reviews, required-main ancestry and zero Base jobs/check-runs. Fresh public page explicitly confirms maintainer approval pending. Read-only helper target_blocker returns base_run_not_unsuccessful; no retry CLI, request, reservation, handler or new attempt. Refreshed local handoff; no edit/rebase/push or GPU execution. Launcher recovery check (owner_requested): blocked / base_run_not_unsuccessful.
- **Latest GitHub activity:** [Latest comment 2026-11-03T09:39:38Z](https://github.com/sgl-project/sglang/pull/47897#issuecomment-6600214639); PR updated 2026-11-23T06:21:57Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-24; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** [Historical indexer source](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/patches/K01-kf-decode-kernel-overlay.NEWFILES.diff); [PR evidence](https://github.com/sgl-project/sglang/pull/47897). The c2cacf5f candidate matches the frozen indexer module; the PR fixes its fused-pair tail omission. Author reports 2.67–3.06x isolated long-context logits, numerical/top-k checks, and real-acceptance GSM8K means .934 main/.927 PR with overlapping ranges. GSM8K is a short-context smoke test. Serving timings used simulated acceptance 3.61 in a single-node setup, not the frozen PD/AgentX setup. Local PR_NOTES.md and working_process.md identify validation scripts; raw results are not yet a public archived bundle.
- **To Ready:** Already Ready. Preserve existing hardware evidence and the paired-row regression; archived raw run-to-source bundle is still a handoff follow-up.
- **To merge / dependency:** Base audit 2026-11-24T00:07:14.229187+00:00: Base only: three-file diff adds a gfx950-only Gluon module and guarded Aiter dispatch/tests; no shared norm/architecture edits. Interactive rebase onto 2544c9ac remains f08bde1f. Base 45123546744 attempt 1 is completed/action_required; freshly fetched public page explicitly awaits maintainer approval. Zero Base jobs/check-runs, gates, functional/JIT tests or retained results. Historical Base success covers a different head. Source hashes/callers match the recorded implementation; inherited CPU controls were not rerun. Serving engagement and numerical/performance qualifications remain separate. Launcher recovery check (owner_requested): blocked / base_run_not_unsuccessful. Upstream maintainer approves existing Base 45123546744; run-ci is present. Follow actual execution and use guarded PR-wide helper for eligible terminal failures. Preserve rebased f08bde1f; no redundant rebase or freshness retry. Maintainers review source-family port; owner/kernel reviewer establishes serving Gluon engagement because scheduled generic and qualifying verify calls route to Aiter. Preserve schedules, unequal paired-row/table-length, common-prefix/tail, fallback and long-context numerical qualifications; archive exact source/image/raw runs. Hand off performance-sensitive dispatch changes before publication. Short-context GSM8K and frozen whole-stack throughput do not qualify this patch. Local handoff: cycles/20261124T000002Z/PR47897_REVIEW_HANDOFF.json; no alternate GPU access.

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


### [SemiAnalysisAI/InferenceX #5193](https://github.com/SemiAnalysisAI/InferenceX/pull/5193) — musespark mi355x: append TP4/EP4 concurrency 58 / musespark mi355x：追加 TP4/EP4 并发 58

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: billishyahao, chunfangamd, seungrokj, 1am9trash, zijiecode.
- **Base / head:** `main` @ `1f8cf90a7ba3b44e5586d542e8820187914b7bae` / `yihou/musespark-mi355x-c58` @ `21c83acacb9fd41116474b92b1da2889b0dec085`.
- **Review / CI:** no decision; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: NEUTRAL 1, SKIPPED 5, SUCCESS 5. Reviews: claude: COMMENTED (2026-11-23T13:48:55Z).
- **Last curated action:** Discovered by the daily author-wide query.
- **Latest GitHub activity:** [Latest comment 2026-11-23T13:45:19Z](https://github.com/SemiAnalysisAI/InferenceX/pull/5193#issuecomment-7102981140); PR updated 2026-11-23T13:48:55Z.
- **Evidence freshness:** NOT AUDITED. Notes last assessed unrecorded; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** No curated evidence yet; inspect the PR.
- **To Ready:** New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.
- **To merge / dependency:** Resolve current review and required CI; no automatic merge certification.


### [sgl-project/sglang #51760](https://github.com/sgl-project/sglang/pull/51760) — [Doc] [AMD] cookbook: Muse Spark Open MI355X concurrency 58 from InferenceX #5155

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: wisclmy0611, zijiexia, JustinTong0323, sogalin.
- **Base / head:** `main` @ `e183c415bdfc5995f6288b1651e1c320c6a74015` / `muse-mi355x-cookbook-c58-v2` @ `304a99d1c6d2e0efba5e6afcdf4dec534dd2eed6`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: NEUTRAL 1, SKIPPED 90, SUCCESS 32. Reviews: claude: COMMENTED (2026-11-20T07:22:23Z).
- **Last curated action:** Discovered by the daily author-wide query.
- **Latest GitHub activity:** [Latest comment 2026-11-20T07:22:24Z](https://github.com/sgl-project/sglang/pull/51760#issuecomment-7033197153); PR updated 2026-11-20T07:23:25Z.
- **Evidence freshness:** NOT AUDITED. Notes last assessed unrecorded; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** No curated evidence yet; inspect the PR.
- **To Ready:** New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.
- **To merge / dependency:** Resolve current review and required CI; no automatic merge certification.


### [uppinote20/claude-dashboard #118](https://github.com/uppinote20/claude-dashboard/pull/118) — feat: add calibrate-pricing command for contracted gateway rates

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `24458736e52f900a34f2fd000bc11c629ed77e7b` / `feat/calibrate-pricing` @ `99a8a7bcc61c5fb9e5ff2d5eea742a14e441fe50`.
- **Review / CI:** CHANGES_REQUESTED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 1, SKIPPED 2. Reviews: uppinote20: CHANGES_REQUESTED (2026-11-16T11:40:47Z).
- **Last curated action:** Discovered by the daily author-wide query.
- **Latest GitHub activity:** [Latest comment 2026-11-12T11:52:54Z](https://github.com/uppinote20/claude-dashboard/pull/118#issuecomment-6822855403); PR updated 2026-11-16T11:40:47Z.
- **Evidence freshness:** NOT AUDITED. Notes last assessed unrecorded; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** No curated evidence yet; inspect the PR.
- **To Ready:** New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.
- **To merge / dependency:** Resolve current review and required CI; no automatic merge certification.

- Red-check entry points: [Vercel](https://vercel.com/git/authorize?team=uppinote&slug=uppinote&teamId=team_D2tUDJGTLb40ESjOzznqmYrg&type=github&job=%7B%22headInfo%22%3A%7B%22sha%22%3A%2299a8a7bcc61c5fb9e5ff2d5eea742a14e441fe50%22%7D%2C%22id%22%3A%22QmUipTWKzh3SbYyEEVZU1CvmvHQQgUn1czs2zbg1ARvRTJ%22%2C%22org%22%3A%22uppinote20%22%2C%22prId%22%3A118%2C%22repo%22%3A%22claude-dashboard%22%7D).


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
- **Base / head:** `main` @ `8847491c2856d5bc43e230f413c62afe68147e46` / `dev/yihou/glm52-pr-tracking` @ `bd90d08b2a1b8484eed6a539a44796cbf38755eb`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: SKIPPED 8, SUCCESS 4. Reviews: none.
- **Last curated action:** 2026-11-02: owner requested permanent Draft for view/track; converted #228 to Draft and added a daily Draft-state guard.
- **Latest GitHub activity:** PR updated 2026-11-23T21:09:19Z; no issue comments.
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


### [AMD-AGI/Infera #252](https://github.com/AMD-AGI/Infera/pull/252) — agent_sys/o11y: fleet observability — spec and design

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `962845a65c03e1195dab41ae9a2d97940a415a63` / `dev/yihou/agent_sys/o11y_001` @ `3f247ebf17d3dacfeb36534fb08c9c0e0506d517`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 1, SKIPPED 7, SUCCESS 4. Reviews: none.
- **Last curated action:** Discovered by the daily author-wide query.
- **Latest GitHub activity:** PR updated 2026-11-23T17:11:41Z; no issue comments.
- **Evidence freshness:** NOT AUDITED. Notes last assessed unrecorded; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** No curated evidence yet; inspect the PR.
- **To Ready:** New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.
- **To merge / dependency:** Resolve current review and required CI; no automatic merge certification.

- Red-check entry points: [lint](https://github.com/AMD-AGI/Infera/actions/runs/45224964091/job/137869193439).


## External dependencies and equivalent work

These PRs belong to their original authors. Contribute evidence there; do not open duplicates or edit their branches.

| PR | Purpose | State / head | Dependency / next action |
|---|---|---|---|
| [AMD-AGI/Infera #210](https://github.com/AMD-AGI/Infera/pull/210) | v0.5.22 integrated image recipe | MERGED / `8859b247ce` | Merged image history; reproduce pinned fdbafb60 source rather than substitute arbitrary main. |
| [SemiAnalysisAI/InferenceX #4629](https://github.com/SemiAnalysisAI/InferenceX/pull/4629) | Publish the Infera SGLang AgentX PD recipe. | CLOSED / `89594d7f51` | Upstream #33059/#46830/#46820; progress externally owned absorb-BMM #46918 for issue #46829, now covered by scoped MI355X validation (CI/review/merge and image integration remain); select an image with these plus merged #21194 and remove engine patches. Coordinate #630/#47117, diagnose platform RDMA bring-up, complete sweep and review. |
| [sgl-project/sglang #21194](https://github.com/sgl-project/sglang/pull/21194) | Fix GLM PP construction on stages without an embedding. | MERGED / `2059478e21` | Validate a pinned image carrying the merged fix. No duplicate filing; preserve historical source and attribution. |
| [sgl-project/sglang #34465](https://github.com/sgl-project/sglang/pull/34465) | Earlier PP non-first-rank hidden-size proposal | CLOSED / `62cd16df78` | Closed unmerged. Third-party #21194 subsequently merged on October 29 and guards the GLM-5.2 PP construction failure. Validate image inclusion; no new duplicate S13 filing. Preserve archived source and attribution. |
| [sgl-project/sglang #35233](https://github.com/sgl-project/sglang/pull/35233) | HiCache host-pointer alias resolution | MERGED / `c29e6054f5` | Merged into base lineage; do not reapply old #33968 workaround. |
| [sgl-project/sglang #41395](https://github.com/sgl-project/sglang/pull/41395) | Early-send consumer completion | MERGED / `9fc5f30621` | Merged in frozen base; preserve separate producer patch. |
| [sgl-project/sglang #42553](https://github.com/sgl-project/sglang/pull/42553) | Host-allocation pointer handling | MERGED / `b2de502825` | Merged upstream; retain correct frozen-source behavior. |
| [sgl-project/sglang #45524](https://github.com/sgl-project/sglang/pull/45524) | Producer event before non-final PP KV send | READY / `5818b0bf46` | Base audit 2026-11-24T00:07:14.229187+00:00: Historical/unrefreshed evidence copied from the November 13 result; outside the November 18 resumed scope. Base plus AMD/Xeon/ARM64: shared PP producer/send synchronization. Gate, finish and 27 functional Base jobs passed (A9/B13/C5); JIT skipped. AMD has 12 passing stages, one failed shard and four skipped stages. Matching inherited shard-6 evidence records exit 255 after tests started; historical log retrieval returned HTTP 404. First exception remains unavailable. Xeon/ARM64 passed. Execution predates this cycle. Outside the resumed scope; historical/unrefreshed constraints only, requiring explicit owner resumption before active work: External author eaglewangao-us/maintainers retrieve runner logs and recover AMD coverage; complete source-equivalence review while preserving passing Base/disaggregation, historical AWQ/PP evidence and frozen producer patch. |
| [sgl-project/sglang #46918](https://github.com/sgl-project/sglang/pull/46918) | Keep absorb-BMM hand tuning while limiting the waves-per-EU budget that may trigger the gfx950 compiler abort. | READY / `bb5ae4af94` | Base audit 2026-11-24T00:07:14.229187+00:00: Historical/unrefreshed evidence copied from the November 13 result; outside the November 18 resumed scope. Base only: ROCm-guarded absorb-BMM tuning. Gate and 36 functional jobs passed (A19/B12/C5); two B shards cancelled without test steps. One cancellation annotation identifies github-actions[bot]; no actor annotation exists for the other shard. Inherited matching finish log confirms base-b-test cancellation caused exit 1; cancellation policy remains unknown. JIT skipped. Execution predates this cycle. Outside the resumed scope; historical/unrefreshed constraints only, requiring explicit owner resumption before active work: External author yu-depend/maintainers own original-PR conflict, remaining coverage and review decisions, including the unexplained cancellation policy. Advance owned continuation #49732 separately; preserve frozen workaround and source-specific MI355X evidence at 9763ef56. |
| [AMD-AGI/Infera #222](https://github.com/AMD-AGI/Infera/pull/222) | Advertise the KV-events port allocated by srt-slurm. | MERGED / `f668fbf773` | Use a release carrying this fix; verify allocated endpoints with colocated prefills. |
| [NVIDIA/srt-slurm #630](https://github.com/NVIDIA/srt-slurm/pull/630) | Launch Infera frontends with etcd-registered SGLang workers. | MERGED / `290bc0de4b` | Complete frontend integration review; reconcile the recipe backport when merged. |
| [kvcache-ai/Mooncake #4540](https://github.com/kvcache-ai/Mooncake/pull/4540) | Select destination-local GPU write rails. | MERGED / `2ae1f3dfed` | Coordinate fabric-specific validation and review; not a first-version universal prerequisite. |
| [kvcache-ai/Mooncake #4541](https://github.com/kvcache-ai/Mooncake/pull/4541) | Make RDMA ACK timeout configurable. | READY / `0b29369693` | Coordinate fabric-specific validation and review; do not attribute RDMA registration EINVAL to ACK timeout without evidence. |
| [sgl-project/sglang #45307](https://github.com/sgl-project/sglang/pull/45307) | Overlapping gfx950 fused indexer optimization | READY / `1255a0a4ed` | Coordinate overlap with our #47276; rebase whichever lands second. |
| [sgl-project/sglang #47117](https://github.com/sgl-project/sglang/pull/47117) | Document Infera GLM-5.2 PD on MI355X. | CLOSED / `3e89cdc460` | Original author llying-001 owns code and promotion; coordinate cookbook review/merge with InferenceX #4629. Maintainer approval is needed for Mintlify fork preview. Lint passes; Base is green only through documentation filtering with functional suites skipped. Extra is blocked by Draft, not a test assertion. Yihou supports evidence/command review without taking over the branch. |
| [ROCm/aiter #6697](https://github.com/ROCm/aiter/pull/6697) | Related upstream MoE sort-into-GEMM1 experiment; not our five-hook KF port. | READY / `9a20161613` | External author: resolve current main conflicts and respond to the design objection about merging sorting into GEMM1. Track relevance; do not duplicate or modify the author branch. Our separate five-patch MoE port remains held until an active path and current-baseline gain are demonstrated. |
| [SemiAnalysisAI/InferenceX-app #1542](https://github.com/SemiAnalysisAI/InferenceX-app/pull/1542) | Display the Infera SGLang framework name in the dashboard. | MERGED / `f79b6e9545` | Dashboard review/merge after recipe coordination. |
| [kvcache-ai/Mooncake #4340](https://github.com/kvcache-ai/Mooncake/pull/4340) | HIP IPC suballocation relocation | READY / `6e8885249f` | salexspb owns root fix; our R018/R019 evidence is posted. Not used in llr37 RDMA recipe. |
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
| [sgl-project/sglang #45337](https://github.com/sgl-project/sglang/pull/45337) | Validate prefill_complete using effective attention DP | READY / `2995ee29e0` | nvjullin owns the fix; address performance-review concerns. Historical on_complete item; llr37 uses completion, not prefill_complete. |
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
