# Active PR handoff

GitHub refresh: **2026-11-02T06:31:58.643163+00:00**. [Overview and priorities](README.md).

## Ready for review

### [sgl-project/sglang #46820](https://github.com/sgl-project/sglang/pull/46820) — Keep a blocked metrics consumer from stopping the scheduler.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: merrymercy, Ying1123, hnyls2002, xiezhq-hermann.
- **Base / head:** `main` @ `8a455f1b9005a05091ad1825f0f3493bfd74d27d` / `yihou/kv-metrics-nonblocking-20261027` @ `82176e70284c8d67b41de74b8e8005d148e57c8b`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: CANCELLED 1, FAILURE 17, SKIPPED 90, SUCCESS 25. Reviews: none.
- **Last curated action:** 2026-11-02: reviewed unchanged head and posted CI/review follow-up.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:21:04Z](https://github.com/sgl-project/sglang/pull/46820#issuecomment-6568974604); PR updated 2026-11-02T04:21:04Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/README.md); c44a to c44b; four real-ZMQ CPU tests pass; no model-throughput claim.
- **To Ready:** No remaining author-side promotion step; no GPU required for the socket contract.
- **To merge / dependency:** Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492222914/job/122148710598); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492222830/job/122148710150); [Check PR title](https://github.com/sgl-project/sglang/actions/runs/40492222320/job/122148710151); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492222963/job/122148809423); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492222921/job/122148809840); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/managers/test_kv_metrics_backpressure.py
```


### [sgl-project/sglang #46830](https://github.com/sgl-project/sglang/pull/46830) — Remove rank-divergent host syncs that can deadlock DSA eager execution.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: merrymercy, Fridge003, ispobock, Qiaolin-Yu, hebiao064, HaiShaw.
- **Base / head:** `main` @ `a6f5bd397be420c12d0690c07e16e241124269e5` / `fix-dsa-decode-dp-host-sync-deadlock` @ `8bd9fe8c3b1612519e81891a9dc12c5237664eec`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: CANCELLED 1, FAILURE 16, SKIPPED 90, SUCCESS 25. Reviews: none.
- **Last curated action:** 2026-11-02: checked unchanged production/head and requested maintainer CI/review.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:21:07Z](https://github.com/sgl-project/sglang/pull/46830#issuecomment-6568975398); PR updated 2026-11-02T04:21:08Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.llrepro-pp4-dpa4.packup_20261024](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-pp4-dpa4.packup_20261024/README.md); original MI355X deadlock and MI300X masking investigation; production delta matches #33973; CPU 3 tests + 2 subtests and consumer audit.
- **To Ready:** No remaining author-side promotion step; retain scoped historical evidence.
- **To merge / dependency:** Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40494799953/job/122157416408); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40494799791/job/122157414920); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40494799805/job/122157511026); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40494799794/job/122157505439); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40494799718/job/122157520879); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_eager_metadata_no_host_sync.py
```


### [sgl-project/sglang #47276](https://github.com/sgl-project/sglang/pull/47276) — Speed up gfx950 fused indexer logits/top-k for MTP verify rows.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me, DarkSharpness, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `68e57f40057175ff22a17ea0bdbe7ecbc69d9d91` / `yihou/kf-fused-indexer-gfx950` @ `23568fdf24f3c70b057f71ba0ca3aee94ab594b7`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 17, SKIPPED 88, SUCCESS 22. Reviews: claude: COMMENTED (2026-10-29T13:35:23Z).
- **Last curated action:** 2026-11-02: reconciled current GitHub state and archived evidence for handoff.
- **Latest GitHub activity:** PR updated 2026-10-29T13:36:12Z; no issue comments.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** PR-reported bit-identical logits/top-2048/K-write controls and real-acceptance TP8 GSM8K; isolated whole-call 1.20–1.36x; serving ITL used simulated acceptance 3.61. Historical [glm52.llrepro-kf-ar-topk.packup_20261024](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-kf-ar-topk.packup_20261024/README.md) is a related source family, not a verified final-port run mapping.
- **To Ready:** Already non-draft. Audit final October 29 run IDs/source hashes against pack-ups and preserve real-acceptance versus simulated-performance boundaries.
- **To merge / dependency:** Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result. Coordinate overlapping #45307 and rebase whichever lands second.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40883318678/job/123480324817); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40883318704/job/123480326889); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40883318628/job/123480324022); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40883318636/job/123480477565); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40883318702/job/123480456202); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
# GPU: use a pool node; exact architecture/flags are in the PR
PYTHONPATH=python python -m pytest -q test/registered/amd/test_dsa_gfx950_fused_indexer.py
```


### [sgl-project/sglang #47278](https://github.com/sgl-project/sglang/pull/47278) — Speed up gfx950 FP8 sparse MLA decode with a 64-head split-K plan.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: DarkSharpness, HaiShaw, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `390acf84e31d3b63dac64e021911d94770e55a8c` / `yihou/kf-sparse-mla-gfx950` @ `375f052eda199e2b7fc56ae4adc61b79b685d384`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 16, SKIPPED 89, SUCCESS 22. Reviews: claude: COMMENTED (2026-10-29T13:43:32Z).
- **Last curated action:** 2026-11-02: reconciled current GitHub state and archived evidence for handoff.
- **Latest GitHub activity:** PR updated 2026-10-29T13:44:24Z; no issue comments.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** PR reports 22/22 kernel tests, FP32-reference SNR >35 dB and real-acceptance GSM8K mean .942 baseline / .941 PR; isolated kernel 2.4–3.2x. Serving ITL used simulated acceptance 3.61. Final-port pack-up mapping still needs audit.
- **To Ready:** Already non-draft. Bind final tests/runs to source hashes and archive locations; do not substitute for the complete K01 overlay.
- **To merge / dependency:** Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40884568078/job/123484718637); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40884567805/job/123484718203); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40884567975/job/123484716191); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40884568111/job/123485006308); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40884567988/job/123484870281); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
# GPU: use a pool node; preserve PR architecture and dtype matrix
PYTHONPATH=python python -m pytest -q test/registered/kernels/ops/attention/test_triton_sparse_mla_hip.py
```


### [sgl-project/sglang #45484](https://github.com/sgl-project/sglang/pull/45484) — Preserve the Aiter is_shuffled weight tag through offload/onload.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `449617bc94a5d4ae23ad62d0da2f842e711ca983` / `fix/offloader-keep-is-shuffled` @ `43dec26b032c08f4e9eee0c772db9f44d3afc275`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: CANCELLED 2, FAILURE 16, SKIPPED 90, SUCCESS 29. Reviews: none.
- **Last curated action:** 2026-11-02: reconciled current GitHub state and archived evidence for handoff.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:21:05Z](https://github.com/sgl-project/sglang/pull/45484#issuecomment-6568974915); PR updated 2026-11-02T04:21:05Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.v0521-offloaderv2.packup_20261015](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.v0521-offloaderv2.packup_20261015/README.md) ofs0c/ofg3c/ofg2c 23/23 each; AST-equivalent port, CPU 3+2 and allocated MI355X two-onload-cycle test. Full-model rerun is not an automatic Ready gate.
- **To Ready:** No remaining author-side promotion step; CPU and exact-head MI355X onload regression already exist.
- **To merge / dependency:** Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result.

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
- **Last curated action:** 2026-11-02: reconciled current GitHub state and archived evidence for handoff.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:21:06Z](https://github.com/sgl-project/sglang/pull/46822#issuecomment-6568975144); PR updated 2026-11-02T04:21:06Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.dcp4-pd.packup_20261022](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.dcp4-pd.packup_20261022/README.md) R16/R17 fault, R18 5/5 +40/40; later separate R24/R25 row-list bug covered by #45796; current CPU 4 tests +20 subtests.
- **To Ready:** No remaining author-side promotion step; retain historical production equivalence.
- **To merge / dependency:** Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492640626/job/122150134982); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492640595/job/122150134832); [Check PR title](https://github.com/sgl-project/sglang/actions/runs/40492640193/job/122150139430); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492640629/job/122150249116); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492640511/job/122150226596); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_indexer_idle.py
```


### [AMD-AGI/Infera #208](https://github.com/AMD-AGI/Infera/pull/208) — Create, pack, run and share agent_sys tasks from Claude Code.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: xiaobochen-amd, jiejingzhangamd, limou102, JohnQin-llm.
- **Base / head:** `main` @ `8847491c2856d5bc43e230f413c62afe68147e46` / `dev.yihou.agentsys.skills` @ `3bd26aa4afec447a2e69dfae2969ddbf0c7bf3a5`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 1, SKIPPED 7, SUCCESS 4. Reviews: none.
- **Last curated action:** 2026-11-02: reconciled current GitHub state and archived evidence for handoff.
- **Latest GitHub activity:** PR updated 2026-11-02T05:06:44Z; no issue comments.
- **Evidence freshness:** HEAD CHANGED: re-audit evidence from bfe1c68024. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** PR body reports 234 tools tests and full suite 2675 passed/18 skipped/4 xfailed, plus dogfooding. These are author-reported at an earlier revision; the November 2 head changed during this handoff audit.
- **To Ready:** Already non-draft. Re-audit the new head against the earlier tests and inspect current lint failure.
- **To merge / dependency:** Inspect the current lint formatting diff (run 41392116339, job 125105681579, including projection/test files); fix or attribute it against base, re-run affected agent_sys tests, request maintainer review. No GPU prerequisite established.

- Red-check entry points: [lint](https://github.com/AMD-AGI/Infera/actions/runs/41392116339/job/125105681579).

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
python -m pytest -q agent_sys/tests/tools
# Follow the repository lint command for the exact new head.
```


### [AMD-AGI/Infera #228](https://github.com/AMD-AGI/Infera/pull/228) — Track 48k deployment priorities, Ready/Draft work and immutable experiment references.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: xiaobochen-amd, jiejingzhangamd, limou102.
- **Base / head:** `main` @ `8847491c2856d5bc43e230f413c62afe68147e46` / `dev/yihou/glm52-pr-tracking` @ `2cb681fc1e09a65f77434c8fe8d8373fa25fc4bc`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: PENDING 1, SKIPPED 8, SUCCESS 3. Reviews: none.
- **Last curated action:** 2026-11-02: created Ready PR #228 from main with signed commit 2cb681fc; connecting daily exports to the open documentation branch.
- **Latest GitHub activity:** PR updated 2026-11-02T06:29:59Z; no issue comments.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** Five Markdown files from main 8847491c; applicable pre-commit hooks and diff check passed; 21 pinned archive references and local links checked; restricted repository identifiers excluded. Separate publisher safety tests: 14 CPU tests.
- **To Ready:** Documentation is already Ready. Recheck references/privacy when curated source links or generator logic change. No GPU validation needed.
- **To merge / dependency:** Maintainers: review documentation and required checks. After merge, create a main-based follow-up PR for subsequent team documentation updates; do not push the updater directly to main.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
pre-commit run --files docs/glm52-pr-tracking/*.md
git diff --check
```


## Draft

### [sgl-project/sglang #33059](https://github.com/sgl-project/sglang/pull/33059) — Make real query rows and padded Aiter metadata agree.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me.
- **Base / head:** `main` @ `0abac089e5d038aef55c814b8f9f6222da81a75f` / `fix-dsa-indexer-hip-aiter-dp-padding` @ `ea88d74bfc2c934159e4fc73a258861f3c5d96e2`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 17, SKIPPED 88, SUCCESS 22. Reviews: none.
- **Last curated action:** 2026-11-02: signed rebase onto acd6f5570; 12 CPU tests and scoped lint pass; replied with remaining model-matrix gap.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:24:48Z](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6569022080); PR updated 2026-11-02T04:26:02Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.llrepro-pp4-dpa4.packup_20261024](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-pp4-dpa4.packup_20261024/README.md); historical GLM DP8/EP8 checks 4/4 and 256/256; current port 12 CPU tests. GLM runs do not establish the requested DeepSeek matrix.
- **To Ready:** Reuse historical GLM evidence; fulfill HaiShaw's specific DeepSeek-V3.2 DP/stage-c accuracy matrix, diagnose failures, link exact heads/results, then promote. CPU/source work can prepare it; missing model coverage needs an allocated GPU node.
- **To merge / dependency:** Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result. Reviewer: HaiShaw. The shared Qwen fused-KV issue has merged workaround #46861; do not continue listing it as an unexplained patch regression.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388353179/job/125094083960); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388352914/job/125094083570); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388352859/job/125094082345); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388353041/job/125094148325); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388353037/job/125094147038); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_aiter_paged_padding.py
# GPU: allocate a pool node first
python test/registered/amd/accuracy/mi30x/test_deepseek_v32_dp_eval_amd.py
python test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py
python test/registered/amd/test_deepseek_v32_basic.py
```


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
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 3, SKIPPED 1, SUCCESS 31. Reviews: none.
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
- **Last curated action:** 2026-11-02: reconciled current GitHub state and archived evidence for handoff.
- **Latest GitHub activity:** [Latest comment 2026-10-27T11:02:43Z](https://github.com/sgl-project/sglang/pull/46819#issuecomment-6440276807); PR updated 2026-11-02T04:22:19Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.dcpx-pack-fix-best.packup_20261023](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.dcpx-pack-fix-best.packup_20261023/README.md) dcpx8c/9/10; current-head MI355X 3 tests +16 subtests, CPU 8+4; isolated cache medians 39.57→17.15 /39.09→17.07 /130.42→107.52 us. dcpx10 also removed a wait; its combined speedup does not transfer to this PR.
- **To Ready:** CPU-first option: narrow to the independently measured cache and update tests/body. If retaining pinned-index/priority changes, qualify their lifetime, interference and PP ordering on allocated hardware. Preserve producer wait.
- **To merge / dependency:** Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result. Related #44724/#39768/#43782 are overlap coordination, not established mandatory prerequisites.

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
- **Last curated action:** 2026-11-02: signed rebase onto acd6f5570; actual canonical-policy regression added, 43 CPU tests +61 subtests and lint pass.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:20:34Z](https://github.com/sgl-project/sglang/pull/46824#issuecomment-6568968279); PR updated 2026-11-02T04:21:11Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.dcp4-pd.packup_20261022](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.dcp4-pd.packup_20261022/README.md) pdBe-r1 math29×2, passkey8K/34K/96K; rebase production delta unchanged; 43 CPU tests +61 subtests exercise merged sharding policy.
- **To Ready:** Map pdBe-r1 to current supported combinations; resolve or explicitly reject the known DP2×DCP2 graph-garbling case. Test only uncovered MTP/prefix/PP cases on allocated hardware as needed. #44322 has merged; old backend hold is cleared.
- **To merge / dependency:** Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result. This transport must land before #46826 enables the configuration. #45524 producer ordering remains separate.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387906323/job/125092720788); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387903680/job/125092712814); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387906431/job/125092720718); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387906168/job/125092787241); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41387906229/job/125092786601); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/disaggregation/test_dsa_pd_dcp.py test/registered/unit/disaggregation/test_dcp_pack.py test/registered/unit/disaggregation/test_mooncake_transfer_batching.py
```


### [sgl-project/sglang #46826](https://github.com/sgl-project/sglang/pull/46826) — Allow DCP decode behind DCP1 prefill when transport/backend support it.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `e0c7992d0d48c8085655752de7a9f43bf1862bbc` / `yihou/rocm-dcp-pd-config-20261027` @ `21f8e688faa155a1f5abaa53ba3ca03c20900d7c`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 17, SKIPPED 88, SUCCESS 23. Reviews: none.
- **Last curated action:** 2026-11-02: signed rebase onto acd6f5570; default-selection tests added; 12 CPU tests +13 subtests and lint pass.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:22:46Z](https://github.com/sgl-project/sglang/pull/46826#issuecomment-6568996226); PR updated 2026-11-02T04:23:15Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.dcp4-pd.packup_20261022](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.dcp4-pd.packup_20261022/README.md) pdBe-r1; current 12 CPU tests +13 subtests cover implicit Triton and a2a Q-rep defaults. Combined graph/coverage gap remains.
- **To Ready:** Complete the supported-configuration coverage with #46824; preserve rejection of unqualified graph/layout combinations. Backend/default policy is already tested; do not wait again for merged #44322.
- **To merge / dependency:** Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result. Merge after #46824; review may proceed before that merge with the dependency explicit.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388093425/job/125093288690); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388091310/job/125093282853); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388093499/job/125093288867); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388093826/job/125093355564); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41388093534/job/125093360115); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/test_dsa_pd_dcp_config.py test/registered/unit/test_dsa_dcp_backend_defaults.py
```


### [AMD-AGI/Infera #164](https://github.com/AMD-AGI/Infera/pull/164) — Keep an alternative fast-benchmark script for comparison.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `6492209bcae7b23c52f68b36e2889813049c0e5d` / `dev.yihou.sglang.bench.fast.script` @ `4676ec9087bcbca766b0e84d0b70ea3e7c81641e`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 2, SKIPPED 6, SUCCESS 3. Reviews: none.
- **Last curated action:** 2026-11-02: reconciled current GitHub state and archived evidence for handoff.
- **Latest GitHub activity:** PR updated 2026-09-16T04:01:29Z; no issue comments.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** No submission-ready evidence audit; this is not the llr37 AgentX harness.
- **To Ready:** Decide whether to close the comparison draft or define supported behavior, fill the placeholder body and add regression evidence before promotion.
- **To merge / dependency:** Yihou/delegate: scope decision first; then fix failing checks and request review.

- Red-check entry points: [dco](https://github.com/AMD-AGI/Infera/actions/runs/35054006399/job/104660219861); [lint](https://github.com/AMD-AGI/Infera/actions/runs/35054006396/job/104660293965).


## External dependencies and equivalent work

These PRs belong to their original authors. Contribute evidence there; do not open duplicates or edit their branches.

| PR | Purpose | State / head | Dependency / next action |
|---|---|---|---|
| [AMD-AGI/Infera #210](https://github.com/AMD-AGI/Infera/pull/210) | v0.5.22 integrated image recipe | MERGED / `8859b247ce` | Merged image history; reproduce pinned fdbafb60 source rather than substitute arbitrary main. |
| [sgl-project/sglang #34465](https://github.com/sgl-project/sglang/pull/34465) | Earlier PP non-first-rank hidden-size proposal | CLOSED / `62cd16df78` | Closed unmerged; no proof main fixes S13. Obtain llying attribution agreement before any new filing. |
| [sgl-project/sglang #35233](https://github.com/sgl-project/sglang/pull/35233) | HiCache host-pointer alias resolution | MERGED / `c29e6054f5` | Merged into base lineage; do not reapply old #33968 workaround. |
| [sgl-project/sglang #41395](https://github.com/sgl-project/sglang/pull/41395) | Early-send consumer completion | MERGED / `9fc5f30621` | Merged in frozen base; preserve separate producer patch. |
| [sgl-project/sglang #42553](https://github.com/sgl-project/sglang/pull/42553) | Host-allocation pointer handling | MERGED / `b2de502825` | Merged upstream; retain correct frozen-source behavior. |
| [sgl-project/sglang #45524](https://github.com/sgl-project/sglang/pull/45524) | Producer event before non-final PP KV send | READY / `14c8591329` | Frozen image carries a related historical producer patch; eaglewangao-us owns current #45524. Review/CI and compatibility audit, no duplicate. #43299 fixes a distinct path. |
| [sgl-project/sglang #45307](https://github.com/sgl-project/sglang/pull/45307) | Overlapping gfx950 fused indexer optimization | READY / `1255a0a4ed` | Coordinate overlap with our #47276; rebase whichever lands second. |
| [kvcache-ai/Mooncake #4340](https://github.com/kvcache-ai/Mooncake/pull/4340) | HIP IPC suballocation relocation | READY / `9b4b049d9c` | salexspb owns root fix; our R018/R019 evidence is posted. Not used in llr37 RDMA recipe. |
| [kvcache-ai/Mooncake #4953](https://github.com/kvcache-ai/Mooncake/pull/4953) | Asynchronous HIP copy submission | READY / `35c488e938` | Track interaction with #5005; source API unchanged, combined runtime not qualified. |
| [kvcache-ai/Mooncake #5030](https://github.com/kvcache-ai/Mooncake/pull/5030) | Wheel structured-ranges packaging fix | MERGED / `a068e70bef` | Merged and included in #5005 rebase; prior wheel blocker cleared. |
| [kvcache-ai/Mooncake #5144](https://github.com/kvcache-ai/Mooncake/pull/5144) | CUDA guard target generated-header include path | READY / `39c861c81e` | Existing fix for current #5005 inherited CUDA build failure; follow merge then rebase, no duplicate. |
| [sgl-project/sglang #39768](https://github.com/sgl-project/sglang/pull/39768) | PP prefill/DCP transfer work | READY / `b892b1f0e3` | Overlap coordination for #46819; do not infer required dependency. |
| [sgl-project/sglang #42163](https://github.com/sgl-project/sglang/pull/42163) | Shard transfer queues across rooms | DRAFT / `294113a01f` | YAMY1234 owns the draft. Our dcpx/g4tune data is posted; mixed-shard TPOT regression disclosed. Not used in llr37. |
| [sgl-project/sglang #43299](https://github.com/sgl-project/sglang/pull/43299) | Non-CUDA/NIXL producer synchronization | MERGED / `b5d0769169` | Merged; distinct from open #45524. |
| [sgl-project/sglang #43782](https://github.com/sgl-project/sglang/pull/43782) | DCP gather kernel changes | READY / `dfdfc5502a` | Overlap coordination for #46819; compare final source if it lands. |
| [sgl-project/sglang #44322](https://github.com/sgl-project/sglang/pull/44322) | ROCm DCP backend and sharding policy | MERGED / `b8a0a1ac70` | Merged; old D10/D11 backend dependency cleared. Do not keep waiting for this PR. |
| [sgl-project/sglang #44724](https://github.com/sgl-project/sglang/pull/44724) | DCP transfer staging changes | READY / `2ae31e9a73` | Overlap coordination for #46819; not an established merge prerequisite. |
| [sgl-project/sglang #45156](https://github.com/sgl-project/sglang/pull/45156) | Initialize the active idle attention backend | READY / `c02b88d488` | WQW-Git owns the fix; our duplicate #45483 is closed. Maintainer CI/review; no exact-head hardware claim. |
| [sgl-project/sglang #45337](https://github.com/sgl-project/sglang/pull/45337) | Validate prefill_complete using effective attention DP | READY / `60fe4e6a88` | nvjullin owns the fix; address performance-review concerns. Historical on_complete item; llr37 uses completion, not prefill_complete. |
| [sgl-project/sglang #45453](https://github.com/sgl-project/sglang/pull/45453) | Agree across ranks before deferred KV release | MERGED / `2647c3761e` | Merged release hunk; residual empty-queue polling guard has no current reproducer. |
| [sgl-project/sglang #45796](https://github.com/sgl-project/sglang/pull/45796) | Copy host lengths before graph padding | MERGED / `e122674f23` | Merged; covers R24/R25 later row-count fault. |
| [sgl-project/sglang #46424](https://github.com/sgl-project/sglang/pull/46424) | Sparse output batch-major padding | MERGED / `e536a1144f` | Merged; no duplicate D23. Not the baked page-table row patch. |
| [sgl-project/sglang #46861](https://github.com/sgl-project/sglang/pull/46861) | Shared fused-KV allocation workaround on ROCm | MERGED / `b24937a833` | Merged workaround for #33059 CI baseline issue; not Mooncake root-fix completion. |
| [sgl-project/sglang #47180](https://github.com/sgl-project/sglang/pull/47180) | Q replication with per-tensor FP8 scales | MERGED / `bb767aa490` | Merged; original D22 enablement retired. Separate graph/numerical qualification remains. |
| [sgl-project/sglang #45233](https://github.com/sgl-project/sglang/pull/45233) | Speculative penalty accounting proposal | CLOSED / `612b93311b` | Closed unmerged; recheck replacements before reopening work. |

## Issue-only work and patches without a PR

| Issue | State | Updated |
|---|---|---|
| [[Bug] ROCm absorb-BMM hand config aborts Triton compilation at M=256 on gfx950](https://github.com/sgl-project/sglang/issues/46829) | OPEN | 2026-10-27T11:15:22Z |
| [[Bug] ROCm DCP-PD allocator reserve growth precedes HIP resource aborts under burst load](https://github.com/sgl-project/sglang/issues/46831) | OPEN | 2026-10-30T13:42:15Z |

Required unpublished S13, A01, the full historical K01 overlay, I07 and inherited image patches are recorded in [deployment requirements](README.md). Remaining exclusions and reasons are in [PR_PACKUP_MAP.md](PACKUP_MAP.md).
