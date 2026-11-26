# Public PR handoff and daily status

Last successful GitHub refresh: **2026-11-26T06:11:01.832093+00:00**. Author: `dorado269`.

**72 PRs: 16 open (7 ready / 9 draft), 35 merged, 21 closed unmerged.** Includes historical work outside GLM-5.2; external authors are listed separately.

Ready means non-draft, not maintainer approval or passed functional CI. GitHub mergeability is a snapshot, not a merge recommendation.

Daily refresh: 09:00 UTC (17:00 Asia/Shanghai), on the configured maintenance host. A timestamp older than 36 hours requires operator investigation. Raw CI counts can include multiple workflows; skipped/draft/permission gates do not establish executed test coverage.

Task owner is yihou until an explicit delegate is recorded. Requested reviewers are not assumed to have accepted work. Edit curated notes to record a named delegate, evidence and the next action; the updater preserves those notes and flags changed heads.

Priority P0 = historical frozen-image behavior / required reproduction inputs; P1 = related current-main kernel ports and evidence packaging; P2 = other GLM fixes; P3 = unrelated or comparison work. See [deployment requirements](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/LLR37_DEPLOYMENT_PRIORITY.md).

## Active work

| Priority | PR / purpose | State / head | Historical 47k-class deployment | Ready action | Merge action / waiting on |
|---|---|---|---|---|---|
| P1 | [AMD-AGI/Infera #221](https://github.com/AMD-AGI/Infera/pull/221) — Publish six KernelForge families and their historical evidence. | DRAFT / `b8b3938d6b` | EVIDENCE/TOOLING: useful for handoff; not a serving dependency or an Infera core inference fix. | Retarget/reconcile with main after merged #220. Resolve 20 unavailable external fixtures; validate the 502 redacted derivatives before presenting them as tested kernels, or scope the PR explicitly to archival evidence. Re-run catalog validation/CPU checks after retarget. | Yihou/delegate: scope and evidence decision; maintainers: catalog review and CI on main. Historical results apply to original bytes, not renamed derivatives. |
| P1 | [sgl-project/sglang #47276](https://github.com/sgl-project/sglang/pull/47276) — Speed up gfx950 fused indexer logits/top-k for MTP verify rows. | READY / `d334464b72` | OPTIONAL K01 FAMILY PORT: the added fused-indexer path is off in llr37, and later llr42 activation was not retained. | Already Ready. Already Ready. Preserve independent numerical evidence and coordinate #45307; bind final source/run IDs to a public result bundle when available. | Base audit 2026-11-26T06:09:55.358909+00:00: Base only. Launch, logits, and top-k changes are confined to the guarded gfx950 fused indexer. The head contains minimum base 30e076fa. The gate job succeeded and reported missing commit approval. Its cooldown step and 18 downstream jobs skipped. The required finish job failed its tests-ran assertion. Zero functional, JIT, or build jobs executed or were retained. The inherited Base pass applies to e95ec5ab. Launcher recovery check (owner_requested): blocked / missing_current_head_or_base_run. An authorized maintainer runs /approve-ci on d334464b7225 while it is current. Verify the bot approval status, handler outcome, and actual Base execution. Maintainers review and coordinate #45307. The owner archives exact source, raw runs, and independent numerical evidence. The optimization is optional and disabled in frozen llr37. |
| P1 | [sgl-project/sglang #47872](https://github.com/sgl-project/sglang/pull/47872) — Provide opt-in long-context top-k on compact page tables with exact sampling fallbacks. | READY / `d60269391f` | K01 REQUIRED FAMILY PORT: historical top-k belongs to the active llr37 indexer; this opt-in replacement has a different implementation and is slower in the measured 32k case. | Already Ready. Ready restored after the underflow repair, import-guard regression and exact-source MI355X validation. The temporary correctness Draft is resolved; retain opt-in behavior and the historical performance limits. | Base audit 2026-11-26T06:09:55.358909+00:00: Historical and unrefreshed. Owner-paused pending redesign. The archived passing Base attempt 1 applies to dccb51e0f9f3d97dc01aea7aa39db1c0cfe3df34. Owner-paused pending the leader-requested redesign. On owner-directed resumption, preserve opt-in behavior, fallback and tie coverage, the 32k serving regression, and exact-source and long-context limits. |
| P1 | [sgl-project/sglang #47897](https://github.com/sgl-project/sglang/pull/47897) — Speed up gfx950 decode logits while preserving paired-row tails and fallback dispatch. | READY / `d6cb51b650` | K01 REQUIRED FAMILY PORT: historical indexer logits are active in llr37; this corrected main-based implementation is not byte-identical to the frozen overlay. | Already Ready. Already Ready. Preserve existing hardware evidence and the paired-row regression; archived raw run-to-source bundle is still a handoff follow-up. | Base audit 2026-11-26T06:09:55.358909+00:00: Base only. The diff adds a guarded, unscheduled gfx950 Gluon route and preserves Aiter fallback for persistent schedules. The head contains minimum base 30e076fa. The gate job succeeded and reported missing commit approval. Its cooldown step and 18 downstream jobs skipped. The required finish job failed its tests-ran assertion. Zero functional, JIT, or build jobs executed or were retained. Launcher recovery check (owner_requested): blocked / missing_current_head_or_base_run. An authorized maintainer runs /approve-ci on d6cb51b6500f while it is current. Verify the bot approval status, handler outcome, and actual Base execution. Maintainers review. The owner and kernel reviewer archive exact source, image, and raw runs and validate production Gluon dispatch, persistent-schedule fallback, unequal paired tables and lengths, common prefixes, tails, and long-context numerics on allocated hardware. The frozen llr37 throughput measures the whole deployment. |
| P2 | [sgl-project/sglang #45484](https://github.com/sgl-project/sglang/pull/45484) — Preserve the Aiter is_shuffled weight tag through offload/onload. | READY / `71e55ddc2c` | NOT IN LLR37: OffloaderV2 weight offload is not HiCache. | Already Ready. No remaining author-side promotion step; CPU and exact-head MI355X onload regression already exist. | Maintainer /tag-and-rerun-ci; run-ci is absent. Recover current-head Base after admission; retain source/numerical/review qualifications. |
| P2 | [sgl-project/sglang #46819](https://github.com/sgl-project/sglang/pull/46819) — Cache DCP gather metadata instead of rebuilding it per transfer. | DRAFT / `74116b2c05` | NOT IN LLR37: DCP metadata optimization; the peak uses DPA4/DCP1. | CPU-first option: narrow to the independently measured cache and update tests/body. If retaining pinned-index/priority changes, qualify their lifetime, interference and PP ordering on allocated hardware. Preserve producer wait. | Maintainer /tag-and-rerun-ci; run-ci is absent. Keep Draft until existing runtime/performance/graph qualification is resolved; Draft skips tests. |
| P2 | [sgl-project/sglang #46822](https://github.com/sgl-project/sglang/pull/46822) — Skip HIP DSA indexing when an eager DP rank has no tokens. | READY / `6d7219a952` | NOT ADDED TO LLR37: separate DCP/eager-idle fault repair. | Already Ready. No remaining author-side promotion step; retain historical production equivalence. | Maintainer /tag-and-rerun-ci; run-ci is absent. Recover current-head Base after admission; retain source/numerical/review qualifications. |
| P2 | [sgl-project/sglang #46824](https://github.com/sgl-project/sglang/pull/46824) — Relayout DCP-sharded draft KV together with target KV. | DRAFT / `1bcfc52149` | NOT IN LLR37: DCP1-to-DCP-N transport. | Retire/supersede original production patch after mapping unique tests; do not advance this old implementation to Ready. | Retirement candidate: current main 5b4ddbb609 has generic collapsed-draft registration/owner mapping from #48741 and packed collapsed draft from #51455. Actual main owner-plan passed 24 CPU cases. Retire the old production diff; map unique regression/byte-copy coverage before closure. No PR closure or transport/e2e qualification claimed. |
| P2 | [sgl-project/sglang #46826](https://github.com/sgl-project/sglang/pull/46826) — Allow DCP decode behind DCP1 prefill when transport/backend support it. | DRAFT / `dba5922c0a` | NOT IN LLR37: constrained ROCm DCP enablement. | Complete the supported-configuration coverage with #46824; preserve rejection of unqualified graph/layout combinations. Backend/default policy is already tested; do not wait again for merged #44322. | Maintainer /tag-and-rerun-ci; run-ci is absent. Keep Draft until existing runtime/performance/graph qualification is resolved; Draft skips tests. |
| P3 | [AMD-AGI/Infera #228](https://github.com/AMD-AGI/Infera/pull/228) — Permanent Draft tracker for 48k deployment priorities, implementation PRs and experiment references. | DRAFT / `565826f19e` | COORDINATION ONLY: the main-based 48k deployment/PR handoff; not runtime code. | PERMANENT DRAFT: for view and track only. Never promote this documentation PR to Ready. | DO NOT MERGE. Maintain the open Draft as the team coordination surface; review/merge priorities apply to linked implementation PRs only. |
| TRIAGE | [AMD-AGI/Infera #252](https://github.com/AMD-AGI/Infera/pull/252) — agent_sys/o11y: fleet observability — spec and design | DRAFT / `a94eb6d53e` | Unclassified: inspect before adding to a deployment | New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes. | Resolve current review and required CI; no automatic merge certification. |
| TRIAGE | [AMD-AGI/Infera #254](https://github.com/AMD-AGI/Infera/pull/254) — docs(agent-sys): plan for a unified Claude Code install of skills and o11y plugins | DRAFT / `e551d0f8d2` | Unclassified: inspect before adding to a deployment | New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes. | Resolve current review and required CI; no automatic merge certification. |
| TRIAGE | [SemiAnalysisAI/InferenceX #5193](https://github.com/SemiAnalysisAI/InferenceX/pull/5193) — musespark mi355x: append TP4/EP4 concurrency 58 / musespark mi355x：追加 TP4/EP4 并发 58 | READY / `61873d9a6f` | Unclassified: inspect before adding to a deployment | Already Ready. New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes. | Resolve current review and required CI; no automatic merge certification. |
| TRIAGE | [uppinote20/claude-dashboard #118](https://github.com/uppinote20/claude-dashboard/pull/118) — feat: add calibrate-pricing command for contracted gateway rates | READY / `4b4e51933f` | Unclassified: inspect before adding to a deployment | Already Ready. New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes. | Resolve current review and required CI; no automatic merge certification. |
| TRIAGE | [vllm-project/vllm #69540](https://github.com/vllm-project/vllm/pull/69540) — [KV Offload] Remove O(n) heap compaction stalls in CPU LRU policy | DRAFT / `b018a28528` | Unclassified: inspect before adding to a deployment | New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes. | Resolve current review and required CI; no automatic merge certification. |
| TRIAGE | [vllm-project/vllm #69541](https://github.com/vllm-project/vllm/pull/69541) — [ROCm][Perf] Skip chunked-context metadata in ROCM_AITER_FA builders whose layers are all sliding-window | DRAFT / `2b8492ddcc` | Unclassified: inspect before adding to a deployment | New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes. | Resolve current review and required CI; no automatic merge certification. |

## Handoff details

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

### [sgl-project/sglang #47276](https://github.com/sgl-project/sglang/pull/47276) — Speed up gfx950 fused indexer logits/top-k for MTP verify rows.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me, DarkSharpness, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `5f241eb8ba6cb43b5dde2d0e57f3ab1c482202ef` / `yihou/kf-fused-indexer-gfx950` @ `d334464b722580ff01a29a5b83e7b2a10c94c46e`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 1, SKIPPED 34, SUCCESS 10. Reviews: claude: COMMENTED (2026-10-29T13:35:23Z).
- **Last curated action:** Automated 3-hour audit 2026-11-26T06:09:55.358909+00:00: Rechecked the fork head, required checks, attempt jobs and logs, and commit statuses. GitHub reports MERGEABLE, BLOCKED, and REVIEW_REQUIRED. The hourly slot is open and this attempt is ineligible. The check-suite view of the attempt keeps the legacy launcher from selecting the gate-denied run. Launcher recovery check (owner_requested): blocked / missing_current_head_or_base_run.
- **Latest GitHub activity:** [Latest comment 2026-11-19T13:39:28Z](https://github.com/sgl-project/sglang/pull/47276#issuecomment-7010787687); PR updated 2026-11-25T05:41:16Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-26; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** PR-reported bit-identical logits/top-2048/K-write controls and real-acceptance TP8 GSM8K; isolated whole-call 1.20–1.36x; serving ITL used simulated acceptance 3.61. Historical [glm52.llrepro-kf-ar-topk.packup_20261024](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.llrepro-kf-ar-topk.packup_20261024/README.md) is a related source family, not a verified final-port run mapping.
- **To Ready:** Already Ready. Preserve independent numerical evidence and coordinate #45307; bind final source/run IDs to a public result bundle when available.
- **To merge / dependency:** Base audit 2026-11-26T06:09:55.358909+00:00: Base only. Launch, logits, and top-k changes are confined to the guarded gfx950 fused indexer. The head contains minimum base 30e076fa. The gate job succeeded and reported missing commit approval. Its cooldown step and 18 downstream jobs skipped. The required finish job failed its tests-ran assertion. Zero functional, JIT, or build jobs executed or were retained. The inherited Base pass applies to e95ec5ab. Launcher recovery check (owner_requested): blocked / missing_current_head_or_base_run. An authorized maintainer runs /approve-ci on d334464b7225 while it is current. Verify the bot approval status, handler outcome, and actual Base execution. Maintainers review and coordinate #45307. The owner archives exact source, raw runs, and independent numerical evidence. The optimization is optional and disabled in frozen llr37.

- Red-check entry points: [pr-test-finish](https://github.com/sgl-project/sglang/actions/runs/45555151383/job/139005424922).

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
# GPU: use a pool node; exact architecture/flags are in the PR
PYTHONPATH=python python -m pytest -q test/registered/amd/test_dsa_gfx950_fused_indexer.py
```

### [sgl-project/sglang #47872](https://github.com/sgl-project/sglang/pull/47872) — Provide opt-in long-context top-k on compact page tables with exact sampling fallbacks.

- **State / responsibility:** READY; yihou; active tracking paused by owner pending leader-requested redesign. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me, DarkSharpness, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `e1cd0c55e553b3f9440f23835e9ac2777f2bfe92` / `yihou/kf-triton-topk-gfx950` @ `d60269391fbb071a74e190ff7acabf0819aa5546`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 15, SKIPPED 80, SUCCESS 83. Reviews: claude: COMMENTED (2026-11-02T07:47:39Z).
- **Last curated action:** Automated 3-hour audit 2026-11-26T06:09:55.358909+00:00: Carried forward the historical record. Owner-paused and read-only.
- **Latest GitHub activity:** [Latest comment 2026-11-19T01:46:32Z](https://github.com/sgl-project/sglang/pull/47872#issuecomment-6997141745); PR updated 2026-11-19T20:56:29Z.
- **Evidence freshness:** HEAD CHANGED: re-audit evidence from 1061c29838. Notes last assessed 2026-11-26; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** [Historical indexer source](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/patches/K01-kf-decode-kernel-overlay.NEWFILES.diff); [PR evidence](https://github.com/sgl-project/sglang/pull/47872). Author reports exact overflow fallback, valid nondeterministic boundary ties and 1.5–1.9x isolated speedup at 400k contexts/72+ rows, but about 1.3% slower serving at 32k with simulated acceptance 3.61. Real-acceptance GSM8K .953 from one server is a smoke test. Maintenance reproduced candidate underflow in the actual GPU pipeline: original 256 valid slots instead of 2048; fixed 2048 unique valid slots and all 5 GPU tests pass without skips. Run kf47872-20261102T1038Z-topk-underflow used MI355X and the public v0.5.22 ROCm10 image, Torch 2.11.0+rocm10.0.0 / Triton 3.8.0. [Pinned source, CPU/GPU records and reproduction scripts](https://github.com/AMD-AGI/Infera/tree/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/validation/kf_47872). The GPU runner uses the exact module/tests with namespace and CI-registration shims for the older image; no new server/model/performance result is claimed.
- **To Ready:** Ready restored after the underflow repair, import-guard regression and exact-source MI355X validation. The temporary correctness Draft is resolved; retain opt-in behavior and the historical performance limits.
- **To merge / dependency:** Base audit 2026-11-26T06:09:55.358909+00:00: Historical and unrefreshed. Owner-paused pending redesign. The archived passing Base attempt 1 applies to dccb51e0f9f3d97dc01aea7aa39db1c0cfe3df34. Owner-paused pending the leader-requested redesign. On owner-directed resumption, preserve opt-in behavior, fallback and tie coverage, the 32k serving regression, and exact-source and long-context limits.

- Red-check entry points: [base-b-disaggregation-test-large-8-gpu-amd-mi35x (rocm10, linux-mi35x-gpu-8.fabric)](https://github.com/sgl-project/sglang/actions/runs/44479273883/job/135413156752); [pr-test-amd-finish](https://github.com/sgl-project/sglang/actions/runs/44479273883/job/135876488181); [wait-for-base-b](https://github.com/sgl-project/sglang/actions/runs/44479274015/job/135426496637); [call-jit-kernel-tests / jit-kernel-unit-test-1-gpu-large](https://github.com/sgl-project/sglang/actions/runs/44479274015/job/135407737053); [call-jit-kernel-tests / jit-kernel-benchmark-test-1-gpu-large](https://github.com/sgl-project/sglang/actions/runs/44479274015/job/135407737136); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/amd/test_dsa_triton_topk_transform.py
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_triton_topk_gate.py
```

### [sgl-project/sglang #47897](https://github.com/sgl-project/sglang/pull/47897) — Speed up gfx950 decode logits while preserving paired-row tails and fallback dispatch.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: DarkSharpness, HaiShaw, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `5f241eb8ba6cb43b5dde2d0e57f3ab1c482202ef` / `yihou/kf-gluon-indexer-logits-gfx950` @ `d6cb51b6500fb5033b5a0fb8789ebd04acee59a7`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 1, SKIPPED 34, SUCCESS 10. Reviews: claude: COMMENTED (2026-11-02T09:31:07Z).
- **Last curated action:** Automated 3-hour audit 2026-11-26T06:09:55.358909+00:00: Rechecked the fork head, required checks, logs, and commit statuses. GitHub reports MERGEABLE, BLOCKED, and REVIEW_REQUIRED. The check-suite view of the attempt prevents an ineligible launcher retry. The last durable request targeted superseded head 5a06f9e7 and the bot reported zero reruns. The two production dispatch functions match main 6387ceaf in a local AST comparison. Launcher recovery check (owner_requested): blocked / missing_current_head_or_base_run.
- **Latest GitHub activity:** [Latest comment 2026-11-24T06:21:16Z](https://github.com/sgl-project/sglang/pull/47897#issuecomment-7123846157); PR updated 2026-11-25T05:41:41Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-26; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** [Historical indexer source](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/patches/K01-kf-decode-kernel-overlay.NEWFILES.diff); [PR evidence](https://github.com/sgl-project/sglang/pull/47897). The c2cacf5f candidate matches the frozen indexer module; the PR fixes its fused-pair tail omission. Author reports 2.67–3.06x isolated long-context logits, numerical/top-k checks, and real-acceptance GSM8K means .934 main/.927 PR with overlapping ranges. GSM8K is a short-context smoke test. Serving timings used simulated acceptance 3.61 in a single-node setup, not the frozen PD/AgentX setup. Local PR_NOTES.md and working_process.md identify validation scripts; raw results are not yet a public archived bundle.
- **To Ready:** Already Ready. Preserve existing hardware evidence and the paired-row regression; archived raw run-to-source bundle is still a handoff follow-up.
- **To merge / dependency:** Base audit 2026-11-26T06:09:55.358909+00:00: Base only. The diff adds a guarded, unscheduled gfx950 Gluon route and preserves Aiter fallback for persistent schedules. The head contains minimum base 30e076fa. The gate job succeeded and reported missing commit approval. Its cooldown step and 18 downstream jobs skipped. The required finish job failed its tests-ran assertion. Zero functional, JIT, or build jobs executed or were retained. Launcher recovery check (owner_requested): blocked / missing_current_head_or_base_run. An authorized maintainer runs /approve-ci on d6cb51b6500f while it is current. Verify the bot approval status, handler outcome, and actual Base execution. Maintainers review. The owner and kernel reviewer archive exact source, image, and raw runs and validate production Gluon dispatch, persistent-schedule fallback, unequal paired tables and lengths, common prefixes, tails, and long-context numerics on allocated hardware. The frozen llr37 throughput measures the whole deployment.

- Red-check entry points: [pr-test-finish](https://github.com/sgl-project/sglang/actions/runs/45555280464/job/139005836787).

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/amd/test_dsa_gluon_paged_mqa_logits.py
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_mqa_logits_chunking.py
```

### [sgl-project/sglang #45484](https://github.com/sgl-project/sglang/pull/45484) — Preserve the Aiter is_shuffled weight tag through offload/onload.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: merrymercy, Ying1123, Fridge003, ispobock, HaiShaw, ch-wan, BBuf, Edwardf0t1.
- **Base / head:** `main` @ `b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7` / `fix/offloader-keep-is-shuffled` @ `71e55ddc2c0ce424e48ff6d159d8c8053293329c`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 1, SKIPPED 87, SUCCESS 50. Reviews: none.
- **Last curated action:** 2026-11-25 next-turn: rebased onto 5f241eb8ba6c, signed 8e6eca78f9a2 pushed with exact lease; scoped hooks and 3 checked-in CPU tests pass with optional imports isolated. No new GPU/e2e claim.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:21:05Z](https://github.com/sgl-project/sglang/pull/45484#issuecomment-6568974915); PR updated 2026-11-25T18:21:44Z.
- **Evidence freshness:** HEAD CHANGED: re-audit evidence from 8e6eca78f9. Notes last assessed 2026-11-25; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** [glm52.v0521-offloaderv2.packup_20261015](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.v0521-offloaderv2.packup_20261015/README.md) ofs0c/ofg3c/ofg2c 23/23 each; AST-equivalent port, CPU 3+2 and allocated MI355X two-onload-cycle test. Full-model rerun is not an automatic Ready gate.
- **To Ready:** No remaining author-side promotion step; CPU and exact-head MI355X onload regression already exist.
- **To merge / dependency:** Maintainer /tag-and-rerun-ci; run-ci is absent. Recover current-head Base after admission; retain source/numerical/review qualifications.

- Red-check entry points: [pr-test-finish](https://github.com/sgl-project/sglang/actions/runs/45572742572/job/139063258479).

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/utils/test_offloader.py
# Existing GPU command; allocate a pool node if rerunning
PYTHONPATH=python python test/registered/model_loading/test_offloader_shuffled_weights.py
```

### [sgl-project/sglang #46819](https://github.com/sgl-project/sglang/pull/46819) — Cache DCP gather metadata instead of rebuilding it per transfer.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `5f241eb8ba6cb43b5dde2d0e57f3ab1c482202ef` / `yihou/glm52-dcp-pack-cache-20261027` @ `74116b2c056d539e8d1fb92b1ed4e1bdf74becb0`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 1, SKIPPED 34, SUCCESS 10. Reviews: none.
- **Last curated action:** 2026-11-25 next-turn: rebased onto 5f241eb8ba6c, signed 74116b2c056d pushed with exact lease; scoped hooks and 2 checked-in CPU tests pass with optional imports isolated. No new GPU/e2e claim.
- **Latest GitHub activity:** [Latest comment 2026-10-27T11:02:43Z](https://github.com/sgl-project/sglang/pull/46819#issuecomment-6440276807); PR updated 2026-11-25T05:37:10Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-25; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** [glm52.dcpx-pack-fix-best.packup_20261023](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.dcpx-pack-fix-best.packup_20261023/README.md) dcpx8c/9/10; current-head MI355X 3 tests +16 subtests, CPU 8+4; isolated cache medians 39.57→17.15 /39.09→17.07 /130.42→107.52 us. dcpx10 also removed a wait; its combined speedup does not transfer to this PR.
- **To Ready:** CPU-first option: narrow to the independently measured cache and update tests/body. If retaining pinned-index/priority changes, qualify their lifetime, interference and PP ordering on allocated hardware. Preserve producer wait.
- **To merge / dependency:** Maintainer /tag-and-rerun-ci; run-ci is absent. Keep Draft until existing runtime/performance/graph qualification is resolved; Draft skips tests.

- Red-check entry points: [pr-test-finish](https://github.com/sgl-project/sglang/actions/runs/45554785515/job/139004196957).

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m pytest -q test/registered/unit/disaggregation/test_dcp_pack_metadata.py test/registered/unit/disaggregation/test_dcp_pack.py
# GPU only when qualifying the retained stream/lifetime changes
PYTHONPATH=python python -m pytest -q test/registered/kernels/ops/kvcache/test_pd_dcp_gather.py
```

### [sgl-project/sglang #46822](https://github.com/sgl-project/sglang/pull/46822) — Skip HIP DSA indexing when an eager DP rank has no tokens.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me.
- **Base / head:** `main` @ `7f00b8e55bac8f644e31e6de8d56c38bbba36d5a` / `yihou/rocm-dsa-eager-idle-20261027` @ `6d7219a95214fdcb8321a06989a08e220d5910e7`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 3, QUEUED 1, SKIPPED 91, SUCCESS 42. Reviews: none.
- **Last curated action:** 2026-11-25 next-turn: rebased onto 5f241eb8ba6c, signed 16f18d9a04e1 pushed with exact lease; scoped hooks and 4 checked-in CPU tests pass with optional imports isolated. No new GPU/e2e claim.
- **Latest GitHub activity:** [Latest comment 2026-11-25T09:50:04Z](https://github.com/sgl-project/sglang/pull/46822#issuecomment-7158863110); PR updated 2026-11-26T03:02:04Z.
- **Evidence freshness:** HEAD CHANGED: re-audit evidence from 16f18d9a04. Notes last assessed 2026-11-25; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** [glm52.dcp4-pd.packup_20261022](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.dcp4-pd.packup_20261022/README.md) R16/R17 fault, R18 5/5 +40/40; later separate R24/R25 row-list bug covered by #45796; current CPU 4 tests +20 subtests.
- **To Ready:** No remaining author-side promotion step; retain historical production equivalence.
- **To merge / dependency:** Maintainer /tag-and-rerun-ci; run-ci is absent. Recover current-head Base after admission; retain source/numerical/review qualifications.

- Red-check entry points: [base-b-test-1-gpu-small-amd (rocm10, linux-mi300-1gpu-sglang, 5)](https://github.com/sgl-project/sglang/actions/runs/45746224501/job/139684689305); [wait-for-base-b-amd](https://github.com/sgl-project/sglang/actions/runs/45746224501/job/139684689116); [pr-test-finish](https://github.com/sgl-project/sglang/actions/runs/45746222965/job/139662920859).

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_indexer_idle.py
```

### [sgl-project/sglang #46824](https://github.com/sgl-project/sglang/pull/46824) — Relayout DCP-sharded draft KV together with target KV.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `e0c7992d0d48c8085655752de7a9f43bf1862bbc` / `yihou/pd-dcp-sharded-draft-20261027` @ `1bcfc52149011aae08502e9da0620b0325492799`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 17, SKIPPED 88, SUCCESS 23. Reviews: none.
- **Last curated action:** 2026-11-25 current-main source/control audit; no PR mutation. Retirement candidate: current main 5b4ddbb609 has generic collapsed-draft registration/owner mapping from #48741 and packed collapsed draft from #51455. Actual main owner-plan passed 24 CPU cases. Retire the old production diff; map unique regression/byte-copy coverage before closure. No PR closure or transport/e2e qualification claimed.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:20:34Z](https://github.com/sgl-project/sglang/pull/46824#issuecomment-6568968279); PR updated 2026-11-02T04:21:11Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-25; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.dcp4-pd.packup_20261022](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.dcp4-pd.packup_20261022/README.md) pdBe-r1 math29×2, passkey8K/34K/96K; rebase production delta unchanged; 43 CPU tests +61 subtests exercise merged sharding policy.
- **To Ready:** Retire/supersede original production patch after mapping unique tests; do not advance this old implementation to Ready.
- **To merge / dependency:** Retirement candidate: current main 5b4ddbb609 has generic collapsed-draft registration/owner mapping from #48741 and packed collapsed draft from #51455. Actual main owner-plan passed 24 CPU cases. Retire the old production diff; map unique regression/byte-copy coverage before closure. No PR closure or transport/e2e qualification claimed.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387906323/job/125092720788); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387903680/job/125092712814); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387906431/job/125092720718); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387906168/job/125092787241); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387906229/job/125092786601); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/disaggregation/test_dsa_pd_dcp.py test/registered/unit/disaggregation/test_dcp_pack.py test/registered/unit/disaggregation/test_mooncake_transfer_batching.py
```

### [sgl-project/sglang #46826](https://github.com/sgl-project/sglang/pull/46826) — Allow DCP decode behind DCP1 prefill when transport/backend support it.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `5f241eb8ba6cb43b5dde2d0e57f3ab1c482202ef` / `yihou/rocm-dcp-pd-config-20261027` @ `dba5922c0a46508df00507a04c38bd80306e5404`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 1, SKIPPED 34, SUCCESS 10. Reviews: none.
- **Last curated action:** 2026-11-25 next-turn: rebased onto 5f241eb8ba6c, signed dba5922c0a46 pushed with exact lease; scoped hooks and 7 checked-in CPU tests pass with optional imports isolated. No new GPU/e2e claim.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:22:46Z](https://github.com/sgl-project/sglang/pull/46826#issuecomment-6568996226); PR updated 2026-11-25T05:39:41Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-25; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** [glm52.dcp4-pd.packup_20261022](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.dcp4-pd.packup_20261022/README.md) pdBe-r1; current 12 CPU tests +13 subtests cover implicit Triton and a2a Q-rep defaults. Combined graph/coverage gap remains.
- **To Ready:** Complete the supported-configuration coverage with #46824; preserve rejection of unqualified graph/layout combinations. Backend/default policy is already tested; do not wait again for merged #44322.
- **To merge / dependency:** Maintainer /tag-and-rerun-ci; run-ci is absent. Keep Draft until existing runtime/performance/graph qualification is resolved; Draft skips tests.

- Red-check entry points: [pr-test-finish](https://github.com/sgl-project/sglang/actions/runs/45554983648/job/139004862719).

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/test_dsa_pd_dcp_config.py test/registered/unit/test_dsa_dcp_backend_defaults.py
```

### [AMD-AGI/Infera #228](https://github.com/AMD-AGI/Infera/pull/228) — Permanent Draft tracker for 48k deployment priorities, implementation PRs and experiment references.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: xiaobochen-amd, jiejingzhangamd, limou102.
- **Base / head:** `main` @ `8847491c2856d5bc43e230f413c62afe68147e46` / `dev/yihou/glm52-pr-tracking` @ `565826f19e6a11e325c674a3b0dc62f55858fe5b`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: SKIPPED 8, SUCCESS 4. Reviews: none.
- **Last curated action:** 2026-11-02: owner requested permanent Draft for view/track; converted #228 to Draft and added a daily Draft-state guard.
- **Latest GitHub activity:** PR updated 2026-11-26T03:07:35Z; no issue comments.
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
- **Base / head:** `main` @ `962845a65c03e1195dab41ae9a2d97940a415a63` / `dev/yihou/agent_sys/o11y_001` @ `a94eb6d53ee0f78783461933fe737ceef8dc0e2e`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: SKIPPED 3, SUCCESS 10. Reviews: dorado269: COMMENTED (2026-11-25T06:26:46Z), dorado269: COMMENTED (2026-11-25T06:30:29Z), dorado269: COMMENTED (2026-11-25T06:38:14Z), dorado269: COMMENTED (2026-11-25T06:39:33Z), dorado269: COMMENTED (2026-11-25T09:37:40Z), dorado269: COMMENTED (2026-11-25T09:37:42Z), dorado269: COMMENTED (2026-11-25T09:37:44Z).
- **Last curated action:** Discovered by the daily author-wide query.
- **Latest GitHub activity:** PR updated 2026-11-25T14:13:09Z; no issue comments.
- **Evidence freshness:** NOT AUDITED. Notes last assessed unrecorded; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** No curated evidence yet; inspect the PR.
- **To Ready:** New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.
- **To merge / dependency:** Resolve current review and required CI; no automatic merge certification.

### [AMD-AGI/Infera #254](https://github.com/AMD-AGI/Infera/pull/254) — docs(agent-sys): plan for a unified Claude Code install of skills and o11y plugins

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `3528a242d9b04d0c503f415b9dc603cc02c244bd` / `dev/yihou/agent_sys/install_unify` @ `e551d0f8d2ab2474c9212707cb9efd7b8d24dbe3`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: SKIPPED 8, SUCCESS 4. Reviews: none.
- **Last curated action:** Discovered by the daily author-wide query.
- **Latest GitHub activity:** PR updated 2026-11-25T08:32:18Z; no issue comments.
- **Evidence freshness:** NOT AUDITED. Notes last assessed unrecorded; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** No curated evidence yet; inspect the PR.
- **To Ready:** New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.
- **To merge / dependency:** Resolve current review and required CI; no automatic merge certification.

### [SemiAnalysisAI/InferenceX #5193](https://github.com/SemiAnalysisAI/InferenceX/pull/5193) — musespark mi355x: append TP4/EP4 concurrency 58 / musespark mi355x：追加 TP4/EP4 并发 58

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: billishyahao, chunfangamd, seungrokj.
- **Base / head:** `main` @ `049fd0ee7762d3affd5a18860983dc1468dbc81a` / `yihou/musespark-mi355x-c58` @ `61873d9a6fb5fc16914246b86d070eae344001fb`.
- **Review / CI:** no decision; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: CANCELLED 10, SKIPPED 44, SUCCESS 21. Reviews: claude: COMMENTED (2026-11-23T13:48:55Z), zijiecode: APPROVED (2026-11-26T03:29:01Z), 1am9trash: APPROVED (2026-11-26T06:04:20Z).
- **Last curated action:** Discovered by the daily author-wide query.
- **Latest GitHub activity:** [Latest comment 2026-11-26T06:09:40Z](https://github.com/SemiAnalysisAI/InferenceX/pull/5193#issuecomment-7184150777); PR updated 2026-11-26T06:09:40Z.
- **Evidence freshness:** NOT AUDITED. Notes last assessed unrecorded; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** No curated evidence yet; inspect the PR.
- **To Ready:** New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.
- **To merge / dependency:** Resolve current review and required CI; no automatic merge certification.

### [uppinote20/claude-dashboard #118](https://github.com/uppinote20/claude-dashboard/pull/118) — feat: add calibrate-pricing command for contracted gateway rates

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `24458736e52f900a34f2fd000bc11c629ed77e7b` / `feat/calibrate-pricing` @ `4b4e51933fa178f674e4ecbcf8f85eb06beeabc2`.
- **Review / CI:** CHANGES_REQUESTED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 1, SKIPPED 2. Reviews: uppinote20: CHANGES_REQUESTED (2026-11-16T11:40:47Z).
- **Last curated action:** Discovered by the daily author-wide query.
- **Latest GitHub activity:** [Latest comment 2026-11-12T11:52:54Z](https://github.com/uppinote20/claude-dashboard/pull/118#issuecomment-6822855403); PR updated 2026-11-26T04:35:32Z.
- **Evidence freshness:** NOT AUDITED. Notes last assessed unrecorded; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** No curated evidence yet; inspect the PR.
- **To Ready:** New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.
- **To merge / dependency:** Resolve current review and required CI; no automatic merge certification.

- Red-check entry points: [Vercel](https://vercel.com/git/authorize?team=uppinote&slug=uppinote&teamId=team_D2tUDJGTLb40ESjOzznqmYrg&type=github&job=%7B%22headInfo%22%3A%7B%22sha%22%3A%224b4e51933fa178f674e4ecbcf8f85eb06beeabc2%22%7D%2C%22id%22%3A%22Qmcw8mdUdb5uqbzRR6SdrzgndG1WFAiMYwF5wf3LKFbmqJ%22%2C%22org%22%3A%22uppinote20%22%2C%22prId%22%3A118%2C%22repo%22%3A%22claude-dashboard%22%7D).

### [vllm-project/vllm #69540](https://github.com/vllm-project/vllm/pull/69540) — [KV Offload] Remove O(n) heap compaction stalls in CPU LRU policy

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `341b2b31f3219701e6f309426b481e5e13a6ad7e` / `kv-offload-lru-no-heap-compaction` @ `b018a28528c50a8b48602fbf2dd61c02fe704689`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: SKIPPED 1, SUCCESS 4. Reviews: none.
- **Last curated action:** Discovered by the daily author-wide query.
- **Latest GitHub activity:** [Latest comment 2026-11-24T04:25:48Z](https://github.com/vllm-project/vllm/pull/69540#issuecomment-7122010998); PR updated 2026-11-25T07:59:33Z.
- **Evidence freshness:** NOT AUDITED. Notes last assessed unrecorded; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** No curated evidence yet; inspect the PR.
- **To Ready:** New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.
- **To merge / dependency:** Resolve current review and required CI; no automatic merge certification.

### [vllm-project/vllm #69541](https://github.com/vllm-project/vllm/pull/69541) — [ROCm][Perf] Skip chunked-context metadata in ROCM_AITER_FA builders whose layers are all sliding-window

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `341b2b31f3219701e6f309426b481e5e13a6ad7e` / `rocm-aiter-fa-skip-swa-chunk-meta` @ `2b8492ddcccbf9cfec9609f65372271c6be9819e`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: SKIPPED 2, SUCCESS 6. Reviews: none.
- **Last curated action:** Discovered by the daily author-wide query.
- **Latest GitHub activity:** [Latest comment 2026-11-24T04:25:55Z](https://github.com/vllm-project/vllm/pull/69541#issuecomment-7122012632); PR updated 2026-11-24T04:25:55Z.
- **Evidence freshness:** NOT AUDITED. Notes last assessed unrecorded; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** No curated evidence yet; inspect the PR.
- **To Ready:** New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.
- **To merge / dependency:** Resolve current review and required CI; no automatic merge certification.

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
| [sgl-project/sglang #45524](https://github.com/sgl-project/sglang/pull/45524) | Producer event before non-final PP KV send | READY / `5818b0bf46` | Base audit 2026-11-26T06:09:55.358909+00:00: The gate and finish passed. Base ran and passed 35 functional jobs (A13/B17/C5), including both PP producer-event CPU cases. Thirteen jobs skipped, including JIT. The shared CUDA and ROCm PP ordering change warrants AMD disaggregation coverage. AMD attempt 1 passed 16 functional jobs, with seven jobs skipped across the workflow. Its failing shard passed the PP file. The NIXL prefill server exited with code -9 at PP size 1 in setUpClass before any accuracy cases. The changed branch requires PP size greater than 1. The termination cause is unknown. External author eaglewangao-us and the requested PD maintainers own review. Preserve AWQ and PP qualification and compare with the frozen producer patch. If NIXL startup recurs, the AMD runner owner can obtain process or cgroup termination evidence. |
| [sgl-project/sglang #46918](https://github.com/sgl-project/sglang/pull/46918) | Keep absorb-BMM hand tuning while limiting the waves-per-EU budget that may trigger the gfx950 compiler abort. | READY / `bb5ae4af94` | Base audit 2026-11-26T06:09:55.358909+00:00: Historical and unrefreshed. The gate and 36 functional jobs passed (A19/B12/C5). Two B shards were cancelled before tests. An annotation identifies github-actions[bot], and the finish log attributes the aggregate failure to cancellation. The cancellation policy remains unknown. JIT skipped. Outside active scope. External author yu-depend and maintainers own the original conflict, coverage, and review. Preserve attribution, the frozen workaround, and the scoped 9763ef56 MI355X evidence alongside #49732. |
| [AMD-AGI/Infera #222](https://github.com/AMD-AGI/Infera/pull/222) | Advertise the KV-events port allocated by srt-slurm. | MERGED / `f668fbf773` | Use a release carrying this fix; verify allocated endpoints with colocated prefills. |
| [NVIDIA/srt-slurm #630](https://github.com/NVIDIA/srt-slurm/pull/630) | Launch Infera frontends with etcd-registered SGLang workers. | MERGED / `290bc0de4b` | Complete frontend integration review; reconcile the recipe backport when merged. |
| [kvcache-ai/Mooncake #4540](https://github.com/kvcache-ai/Mooncake/pull/4540) | Select destination-local GPU write rails. | MERGED / `2ae1f3dfed` | Coordinate fabric-specific validation and review; not a first-version universal prerequisite. |
| [kvcache-ai/Mooncake #4541](https://github.com/kvcache-ai/Mooncake/pull/4541) | Make RDMA ACK timeout configurable. | MERGED / `0b29369693` | Coordinate fabric-specific validation and review; do not attribute RDMA registration EINVAL to ACK timeout without evidence. |
| [sgl-project/sglang #45307](https://github.com/sgl-project/sglang/pull/45307) | Overlapping gfx950 fused indexer optimization | READY / `1255a0a4ed` | Coordinate overlap with our #47276; rebase whichever lands second. |
| [sgl-project/sglang #47117](https://github.com/sgl-project/sglang/pull/47117) | Document Infera GLM-5.2 PD on MI355X. | CLOSED / `3e89cdc460` | Original author llying-001 owns code and promotion; coordinate cookbook review/merge with InferenceX #4629. Maintainer approval is needed for Mintlify fork preview. Lint passes; Base is green only through documentation filtering with functional suites skipped. Extra is blocked by Draft, not a test assertion. Yihou supports evidence/command review without taking over the branch. |
| [ROCm/aiter #6697](https://github.com/ROCm/aiter/pull/6697) | Related upstream MoE sort-into-GEMM1 experiment; not our five-hook KF port. | READY / `9a20161613` | External author: resolve current main conflicts and respond to the design objection about merging sorting into GEMM1. Track relevance; do not duplicate or modify the author branch. Our separate five-patch MoE port remains held until an active path and current-baseline gain are demonstrated. |
| [SemiAnalysisAI/InferenceX-app #1542](https://github.com/SemiAnalysisAI/InferenceX-app/pull/1542) | Display the Infera SGLang framework name in the dashboard. | MERGED / `f79b6e9545` | Dashboard review/merge after recipe coordination. |
| [kvcache-ai/Mooncake #4340](https://github.com/kvcache-ai/Mooncake/pull/4340) | HIP IPC suballocation relocation | READY / `6e8885249f` | salexspb owns root fix; our R018/R019 evidence is posted. Not used in llr37 RDMA recipe. |
| [kvcache-ai/Mooncake #4953](https://github.com/kvcache-ai/Mooncake/pull/4953) | Asynchronous HIP copy submission | MERGED / `41112e77f4` | Track interaction with #5005; source API unchanged, combined runtime not qualified. |
| [kvcache-ai/Mooncake #5030](https://github.com/kvcache-ai/Mooncake/pull/5030) | Wheel structured-ranges packaging fix | MERGED / `a068e70bef` | Merged and included in #5005 rebase; prior wheel blocker cleared. |
| [kvcache-ai/Mooncake #5144](https://github.com/kvcache-ai/Mooncake/pull/5144) | CUDA guard target generated-header include path | MERGED / `39c861c81e` | Existing fix for current #5005 inherited CUDA build failure; follow merge then rebase, no duplicate. |
| [sgl-project/sglang #39768](https://github.com/sgl-project/sglang/pull/39768) | PP prefill/DCP transfer work | READY / `b892b1f0e3` | Overlap coordination for #46819; do not infer required dependency. |
| [sgl-project/sglang #42163](https://github.com/sgl-project/sglang/pull/42163) | Shard transfer queues across rooms | READY / `c65c81cb3d` | YAMY1234 owns the draft. Our dcpx/g4tune data is posted; mixed-shard TPOT regression disclosed. Not used in llr37. |
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

Required unpublished S13, A01, the full historical K01 overlay, I07 and inherited image patches are recorded in [deployment requirements](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/LLR37_DEPLOYMENT_PRIORITY.md). Remaining exclusions and reasons are in [PR_PACKUP_MAP.md](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/PR_PACKUP_MAP.md).

## Complete authored history

| PR | Purpose / title | State | Last GitHub update | Next action |
|---|---|---|---|---|
| [AMD-AGI/Infera #4](https://github.com/AMD-AGI/Infera/pull/4) |  feat(image): support DeepSeek V4 on MI325X (gfx942) for atom + sglang | MERGED | 2026-07-22T10:18:58Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #7](https://github.com/AMD-AGI/Infera/pull/7) | Upgrade vLLM base image to stable v0.25.1 (digest-pinned) | MERGED | 2026-07-21T11:52:51Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #15](https://github.com/AMD-AGI/Infera/pull/15) | Upgrade SGLang base image to v0.5.15.post1 (official release repo) | MERGED | 2026-07-22T09:53:34Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #18](https://github.com/AMD-AGI/Infera/pull/18) | (WIP)Fix MoRIIO MLA page-length PD corruption on v0.25.1 + GLM-5.1 regression test | CLOSED | 2026-07-23T08:09:00Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [AMD-AGI/Infera #19](https://github.com/AMD-AGI/Infera/pull/19) | feat(sglang): unified Mooncake image — one engine.so, every PD path runtime-decided | MERGED | 2026-07-28T01:54:32Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #25](https://github.com/AMD-AGI/Infera/pull/25) | Feature/Fix: support GLM for all engines mix and vllm moriio pd & adding all to test | MERGED | 2026-07-28T01:53:25Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #29](https://github.com/AMD-AGI/Infera/pull/29) | Yihou.dev.sglang.mooncake.experiment | CLOSED | 2026-07-28T04:35:31Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [AMD-AGI/Infera #31](https://github.com/AMD-AGI/Infera/pull/31) | fix: persist e2e worker logs to shared NFS — and fix the first crash they exposed | MERGED | 2026-07-30T10:41:29Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #33](https://github.com/AMD-AGI/Infera/pull/33) | feat(preflight): mooncake KV-registration mode selector | MERGED | 2026-08-02T04:44:09Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #34](https://github.com/AMD-AGI/Infera/pull/34) | sglang DSA: enable PD + DP-attention + EAGLE MTP on GLM-5.2 / gfx950 (4 patches) | CLOSED | 2026-07-31T11:00:34Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [AMD-AGI/Infera #58](https://github.com/AMD-AGI/Infera/pull/58) | sglang DSA: enable PD + DP-attention + EAGLE MTP on GLM-5.2 / gfx950 (3 patches, rebased onto main) | MERGED | 2026-08-01T10:34:56Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #59](https://github.com/AMD-AGI/Infera/pull/59) | [wip]fix(kvaware+kvd): two bugs blocking KV-aware routing + kvd together on SGLang, with image, docs and repro | CLOSED | 2026-08-03T10:28:28Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [AMD-AGI/Infera #79](https://github.com/AMD-AGI/Infera/pull/79) | fix(glm5.2): the remaining fixes for agentic serving — DSA indexer, ROCm hicache, PD/kvd wiring | MERGED | 2026-08-04T02:55:57Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #84](https://github.com/AMD-AGI/Infera/pull/84) | docs(examples): add a GLM-5.2 1P1D SGLang deployment example | MERGED | 2026-08-04T10:35:48Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #87](https://github.com/AMD-AGI/Infera/pull/87) | docs(sglang): record the MTP penalty-accounting defect we filed but do not patch | CLOSED | 2026-08-21T08:40:25Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [AMD-AGI/Infera #94](https://github.com/AMD-AGI/Infera/pull/94) | chore(examples): drop the GLM-5.2 results/ directory | MERGED | 2026-08-07T07:47:10Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #95](https://github.com/AMD-AGI/Infera/pull/95) | docs(patches): give every patch a schema-checked upstream status record | CLOSED | 2026-08-21T08:40:32Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [AMD-AGI/Infera #123](https://github.com/AMD-AGI/Infera/pull/123) | feat(agent_sys): add env_mgr layered environment manager | MERGED | 2026-08-20T03:26:57Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #124](https://github.com/AMD-AGI/Infera/pull/124) | feat(agent_sys): task_graph — task-management substrate for the agent loop | MERGED | 2026-08-25T09:08:00Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #132](https://github.com/AMD-AGI/Infera/pull/132) | feat(agent_sys): the agent work system — specs, designs, and a running implementation | CLOSED | 2026-10-12T03:36:15Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [AMD-AGI/Infera #142](https://github.com/AMD-AGI/Infera/pull/142) | [draft][agent_sys] task package init | CLOSED | 2026-09-14T09:19:34Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [AMD-AGI/Infera #147](https://github.com/AMD-AGI/Infera/pull/147) | [draft]Dev.yihou.aiopt.kernel opt module | MERGED | 2026-09-02T08:07:58Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #150](https://github.com/AMD-AGI/Infera/pull/150) | Dev.yihou.aiopt.kernel opt module | CLOSED | 2026-09-03T04:49:11Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [AMD-AGI/Infera #151](https://github.com/AMD-AGI/Infera/pull/151) | GLM-5.3 series: engine image, example kits, and parked e2e cases | CLOSED | 2026-09-10T06:52:17Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [AMD-AGI/Infera #154](https://github.com/AMD-AGI/Infera/pull/154) | feat(agent_sys): AgentsView as the o11y panel, one project per run | MERGED | 2026-09-04T15:01:09Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #155](https://github.com/AMD-AGI/Infera/pull/155) | [Draft] agent_sys with agent config | CLOSED | 2026-09-07T05:42:01Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [AMD-AGI/Infera #156](https://github.com/AMD-AGI/Infera/pull/156) | feat(agent_sys): label every runtime directory with the name it alrea… | MERGED | 2026-09-04T09:41:44Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #157](https://github.com/AMD-AGI/Infera/pull/157) | [draft]Dev.yihou.aiopt.task package.concat | CLOSED | 2026-09-08T08:59:57Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [AMD-AGI/Infera #163](https://github.com/AMD-AGI/Infera/pull/163) | GLM-5.3 (glm_moe_dsa): mixed, 1P1D, and a Kubernetes recipe — plus overlay support for SGLang bases | MERGED | 2026-09-11T09:15:20Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #164](https://github.com/AMD-AGI/Infera/pull/164) | Keep an alternative fast-benchmark script for comparison. | CLOSED | 2026-11-02T12:18:15Z | Closed unmerged; comparison-only benchmark tooling. |
| [AMD-AGI/Infera #167](https://github.com/AMD-AGI/Infera/pull/167) | feat(agent_sys): an agent work system, and the five-stage LLM e2e optimisation flow | MERGED | 2026-10-19T08:50:38Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #208](https://github.com/AMD-AGI/Infera/pull/208) | Create, pack, run and share agent_sys tasks from Claude Code. | MERGED | 2026-11-02T07:29:14Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #218](https://github.com/AMD-AGI/Infera/pull/218) | Avoid SIGPIPE false negatives in Mooncake artifact checks. | MERGED | 2026-10-30T06:56:00Z | Merged October 30; optional rebuild backport, no runtime throughput claim. |
| [AMD-AGI/Infera #220](https://github.com/AMD-AGI/Infera/pull/220) | agent_sys: add simple kernel artifact catalogs and skill | MERGED | 2026-10-29T02:18:53Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #221](https://github.com/AMD-AGI/Infera/pull/221) | Publish six KernelForge families and their historical evidence. | DRAFT | 2026-10-28T10:55:40Z | See active handoff above |
| [AMD-AGI/Infera #223](https://github.com/AMD-AGI/Infera/pull/223) | ci(release): build the Rust router once and bundle it in a Linux x86_64 wheel | MERGED | 2026-10-29T07:41:06Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #228](https://github.com/AMD-AGI/Infera/pull/228) | Permanent Draft tracker for 48k deployment priorities, implementation PRs and experiment references. | DRAFT | 2026-11-26T03:07:35Z | See active handoff above |
| [AMD-AGI/Infera #237](https://github.com/AMD-AGI/Infera/pull/237) | Migrate agent_sys Forge workflows to Hyperloom and public AKLR knowledge | MERGED | 2026-11-12T06:28:11Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #238](https://github.com/AMD-AGI/Infera/pull/238) | feat(agent-sys): offer claude-dashboard and link kernel_registry skills in the skill installer | MERGED | 2026-11-26T04:35:32Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #252](https://github.com/AMD-AGI/Infera/pull/252) | agent_sys/o11y: fleet observability — spec and design | DRAFT | 2026-11-25T14:13:09Z | See active handoff above |
| [AMD-AGI/Infera #254](https://github.com/AMD-AGI/Infera/pull/254) | docs(agent-sys): plan for a unified Claude Code install of skills and o11y plugins | DRAFT | 2026-11-25T08:32:18Z | See active handoff above |
| [AMD-AGI/Primus-Turbo #387](https://github.com/AMD-AGI/Primus-Turbo/pull/387) | Dev/yihou/bench | MERGED | 2026-08-24T04:04:02Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Primus-Turbo #391](https://github.com/AMD-AGI/Primus-Turbo/pull/391) | bench: FlashInfer B200 sweep aligned with aiter reference | MERGED | 2026-06-23T09:34:51Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Primus-Turbo #394](https://github.com/AMD-AGI/Primus-Turbo/pull/394) | bench: FlashInfer B200 sweep aligned with aiter reference | MERGED | 2026-06-24T04:11:17Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Primus-Turbo #400](https://github.com/AMD-AGI/Primus-Turbo/pull/400) | Dev.microbench.b200.gptoss | MERGED | 2026-08-24T04:03:56Z | Merged: retain provenance; no active merge work |
| [SemiAnalysisAI/InferenceX #5155](https://github.com/SemiAnalysisAI/InferenceX/pull/5155) | musespark mi355x: append TP4/EP4 concurrency 58 / musespark mi355x：追加 TP4/EP4 并发 58 | CLOSED | 2026-11-24T04:04:47Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [SemiAnalysisAI/InferenceX #5193](https://github.com/SemiAnalysisAI/InferenceX/pull/5193) | musespark mi355x: append TP4/EP4 concurrency 58 / musespark mi355x：追加 TP4/EP4 并发 58 | READY | 2026-11-26T06:09:40Z | See active handoff above |
| [kvcache-ai/Mooncake #5005](https://github.com/kvcache-ai/Mooncake/pull/5005) | Offer opt-in HIP copy-stream priority without changing the default. | MERGED | 2026-11-12T22:25:03Z | Merged: retain provenance; no active merge work |
| [sgl-project/sglang #33059](https://github.com/sgl-project/sglang/pull/33059) | Make real query rows and padded Aiter metadata agree. | MERGED | 2026-11-09T20:42:39Z | Merged: retain provenance; no active merge work |
| [sgl-project/sglang #33643](https://github.com/sgl-project/sglang/pull/33643) | Speculative penalty token accounting | CLOSED | 2026-10-27T11:08:13Z | Closed; external #45233 also closed unmerged. Search main/replacements before resuming. |
| [sgl-project/sglang #33968](https://github.com/sgl-project/sglang/pull/33968) | Historical HiCache pointer aliases | CLOSED | 2026-10-27T08:06:24Z | Closed as superseded by merged #35233/#42553. |
| [sgl-project/sglang #33970](https://github.com/sgl-project/sglang/pull/33970) | Historical early KV-send completion | CLOSED | 2026-10-27T08:06:44Z | Closed as superseded by merged #41395; separate producer work #45524 remains. |
| [sgl-project/sglang #33973](https://github.com/sgl-project/sglang/pull/33973) | Historical DSA eager host-sync removal | CLOSED | 2026-10-27T11:15:53Z | Closed old draft; follow replacement #46830. |
| [sgl-project/sglang #45483](https://github.com/sgl-project/sglang/pull/45483) | Historical eager-idle backend initialization | CLOSED | 2026-10-27T07:09:42Z | Closed duplicate; follow external #45156. |
| [sgl-project/sglang #45484](https://github.com/sgl-project/sglang/pull/45484) | Preserve the Aiter is_shuffled weight tag through offload/onload. | READY | 2026-11-25T18:21:44Z | See active handoff above |
| [sgl-project/sglang #46819](https://github.com/sgl-project/sglang/pull/46819) | Cache DCP gather metadata instead of rebuilding it per transfer. | DRAFT | 2026-11-25T05:37:10Z | See active handoff above |
| [sgl-project/sglang #46820](https://github.com/sgl-project/sglang/pull/46820) | Keep a blocked metrics consumer from stopping the scheduler. | MERGED | 2026-11-11T10:24:26Z | Merged: retain provenance; no active merge work |
| [sgl-project/sglang #46822](https://github.com/sgl-project/sglang/pull/46822) | Skip HIP DSA indexing when an eager DP rank has no tokens. | READY | 2026-11-26T03:02:04Z | See active handoff above |
| [sgl-project/sglang #46824](https://github.com/sgl-project/sglang/pull/46824) | Relayout DCP-sharded draft KV together with target KV. | DRAFT | 2026-11-02T04:21:11Z | See active handoff above |
| [sgl-project/sglang #46826](https://github.com/sgl-project/sglang/pull/46826) | Allow DCP decode behind DCP1 prefill when transport/backend support it. | DRAFT | 2026-11-25T05:39:41Z | See active handoff above |
| [sgl-project/sglang #46828](https://github.com/sgl-project/sglang/pull/46828) | D22: optional BF16 kv_b retention | CLOSED | 2026-11-02T04:17:56Z | Closed after merged #47180 supplied FP8 Q-rep enablement; no isolated current-main benefit established. |
| [sgl-project/sglang #46830](https://github.com/sgl-project/sglang/pull/46830) | Remove rank-divergent host syncs that can deadlock DSA eager execution. | MERGED | 2026-11-11T10:22:28Z | Merged: retain provenance; no active merge work |
| [sgl-project/sglang #47276](https://github.com/sgl-project/sglang/pull/47276) | Speed up gfx950 fused indexer logits/top-k for MTP verify rows. | READY | 2026-11-25T05:41:16Z | See active handoff above |
| [sgl-project/sglang #47278](https://github.com/sgl-project/sglang/pull/47278) | Speed up gfx950 FP8 sparse MLA decode with a 64-head split-K plan. | MERGED | 2026-11-25T05:25:08Z | Merged: retain provenance; no active merge work |
| [sgl-project/sglang #47872](https://github.com/sgl-project/sglang/pull/47872) | Provide opt-in long-context top-k on compact page tables with exact sampling fallbacks. | READY | 2026-11-19T20:56:29Z | See active handoff above |
| [sgl-project/sglang #47897](https://github.com/sgl-project/sglang/pull/47897) | Speed up gfx950 decode logits while preserving paired-row tails and fallback dispatch. | READY | 2026-11-25T05:41:41Z | See active handoff above |
| [sgl-project/sglang #49732](https://github.com/sgl-project/sglang/pull/49732) | Continue #46918 under an owned branch: cap ROCm absorb-BMM compile-time wave budget. | MERGED | 2026-11-17T10:03:39Z | Merged: retain provenance; no active merge work |
| [sgl-project/sglang #51739](https://github.com/sgl-project/sglang/pull/51739) | [Doc] [AMD] cookbook: Muse Spark Open MI355X concurrency 58 settings | CLOSED | 2026-11-20T07:22:28Z | Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements |
| [sgl-project/sglang #51760](https://github.com/sgl-project/sglang/pull/51760) | [Doc] [AMD] cookbook: Muse Spark Open MI355X HiCache frees out-of-window SWA slots | MERGED | 2026-11-25T22:45:26Z | Merged: retain provenance; no active merge work |
| [uppinote20/claude-dashboard #118](https://github.com/uppinote20/claude-dashboard/pull/118) | feat: add calibrate-pricing command for contracted gateway rates | READY | 2026-11-26T04:35:32Z | See active handoff above |
| [vllm-project/vllm #69540](https://github.com/vllm-project/vllm/pull/69540) | [KV Offload] Remove O(n) heap compaction stalls in CPU LRU policy | DRAFT | 2026-11-25T07:59:33Z | See active handoff above |
| [vllm-project/vllm #69541](https://github.com/vllm-project/vllm/pull/69541) | [ROCm][Perf] Skip chunked-context metadata in ROCM_AITER_FA builders whose layers are all sliding-window | DRAFT | 2026-11-24T04:25:55Z | See active handoff above |

The historical llr37-c104 result is 48,713 total tokens/s/GPU (input + output, eight GPUs), using simulated MTP acceptance. PP overlap and decode custom all-reduce have known real-acceptance correctness failures. This table makes no individual-patch speedup or production-correctness claim.
