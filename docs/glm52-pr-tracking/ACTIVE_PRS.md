# Active PR handoff

GitHub refresh: **2026-11-02T10:50:37.930363+00:00**. [Overview and priorities](README.md).

## Ready for review

### [sgl-project/sglang #33059](https://github.com/sgl-project/sglang/pull/33059) — Make real query rows and padded Aiter metadata agree.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me.
- **Base / head:** `main` @ `b45a36531c3673048aeaf35322c3df7f18edd038` / `fix-dsa-indexer-hip-aiter-dp-padding` @ `795488cd19aaa3cc852d6c87ffe36b88af20ca23`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 18, SKIPPED 88, SUCCESS 21. Reviews: claude: COMMENTED (2026-11-02T09:04:20Z).
- **Last curated action:** 2026-11-02: author published the [MI355X model results](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447), marked Ready, and advanced the branch to 795488cd by merging main b45a3653. Status audit confirmed tested/current DSA and test blobs match. [Latest lint](https://github.com/sgl-project/sglang/actions/runs/41423254991/job/125203622542) passed at 09:49:54 UTC. Base now stops at account cooldown, superseding the earlier Draft-gate and XPU-lint failures; no branch rewrite is needed for lint.
- **Latest GitHub activity:** [Latest comment 2026-11-02T09:41:37Z](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447); PR updated 2026-11-02T09:48:02Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [Author GPU report](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447): 8x MI355X DeepSeek-V3.2 DP GSM8K 0.957; basic TP in AMD CI mode 0.958; DP + EAGLE MTP with eager padded rows 0.951 versus unpatched main crashing with lengths must have one entry per score row. Runs used ab4407a57c; independently checked identical production DSA and regression-test Git blobs at current head 795488cd (306d4b5f619fc37d8760d2e22dc07ede2e1d280e and 7a7e8c9b759c80f6666fe80fe793e4cabaacb746). These are author-reported model runs, not rerun by this status audit. The MI300 configuration failed on MI355X both with and without the patch; native MI300 coverage is still absent. Existing 12 CPU regression tests and historical [GLM pack-up](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-pp4-dpa4.packup_20261024/README.md) remain separate evidence.
- **To Ready:** Already Ready by the author/user decision; preserve that state. New MI355X DP and padded MTP evidence supersedes the earlier Draft/missing-matrix assessment. Native MI300 coverage remains a reviewer follow-up, not a Draft promotion step.
- **To merge / dependency:** Current head 795488cd is Ready and lint passes. The existing run-ci label is present; latest [Base gate](https://github.com/sgl-project/sglang/actions/runs/41423254776/job/125204140539) sees Draft=false but hits the 120-minute account cooldown after a counted run at 09:07:56 UTC. Nominal eligibility is November 2 at 11:07:56 UTC if no newer counted run intervenes; expiry does not automatically rerun CI. Rerun required checks when policy permits, address review findings, and obtain native MI300 validation through an equipped CI owner. Reviewer: HaiShaw. The inherited XPU logger error is fixed upstream by #47837 and already included; the shared Qwen fused-KV workaround #46861 is also inherited.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41423254739/job/125204221758); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41423255282/job/125203628111); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41423254861/job/125203622111); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41423254778/job/125203731177); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41423254677/job/125203728571); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_aiter_paged_padding.py
# GPU: allocate a pool node first
python test/registered/amd/accuracy/mi30x/test_deepseek_v32_dp_eval_amd.py
python test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py
python test/registered/amd/test_deepseek_v32_basic.py
```


### [sgl-project/sglang #46820](https://github.com/sgl-project/sglang/pull/46820) — Keep a blocked metrics consumer from stopping the scheduler.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: merrymercy, Ying1123, hnyls2002, xiezhq-hermann.
- **Base / head:** `main` @ `b45a36531c3673048aeaf35322c3df7f18edd038` / `yihou/kv-metrics-nonblocking-20261027` @ `783c1a2192ecb1b3ab65e2cd87656919482e0034`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 18, SKIPPED 88, SUCCESS 21. Reviews: none.
- **Last curated action:** 2026-11-02: rebased/pushed to 783c1a2192ec on main b45a36531c36, including upstream logger fix #47837. Full-tree Ruff, scoped repository hooks and 4 real-ZMQ CPU tests passed; range-diff unchanged. [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41424697358/job/125208358328); [Base gate](https://github.com/sgl-project/sglang/actions/runs/41424697709/job/125208623306) sees Ready; missing maintainer-controlled `run-ci`. No new GPU measurement was performed. [PR follow-up](https://github.com/sgl-project/sglang/pull/46820#issuecomment-6574434420).
- **Latest GitHub activity:** [Latest comment 2026-11-02T10:07:30Z](https://github.com/sgl-project/sglang/pull/46820#issuecomment-6574434420); PR updated 2026-11-02T10:07:30Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/README.md); c44a to c44b; four real-ZMQ CPU tests pass; no model-throughput claim.
- **To Ready:** No remaining author-side promotion step; no GPU required for the socket contract.
- **To merge / dependency:** Ready; the inherited XPU logger F821 is resolved by upstream #47837 and [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41424697358/job/125208358328). [Base gate](https://github.com/sgl-project/sglang/actions/runs/41424697709/job/125208623306) sees Ready; missing maintainer-controlled `run-ci`. Maintainers: add `run-ci`, trigger required CI and review/approve. Yihou/delegate: respond to functional CI or review findings. This authorization gate is not a functional-test failure.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41424697608/job/125208359299); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41424689584/job/125208334701); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41424697443/job/125208357115); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41424697445/job/125208513553); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41424697428/job/125208467350); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/managers/test_kv_metrics_backpressure.py
```


### [sgl-project/sglang #46830](https://github.com/sgl-project/sglang/pull/46830) — Remove rank-divergent host syncs that can deadlock DSA eager execution.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: merrymercy, Fridge003, ispobock, Qiaolin-Yu, hebiao064, HaiShaw.
- **Base / head:** `main` @ `b45a36531c3673048aeaf35322c3df7f18edd038` / `fix-dsa-decode-dp-host-sync-deadlock` @ `625403853ab69c90697080466bc06d5bfcad7fbe`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 18, SKIPPED 88, SUCCESS 21. Reviews: none.
- **Last curated action:** 2026-11-02: rebased/pushed to 625403853ab6 on main b45a36531c36, including upstream logger fix #47837. Full-tree Ruff, scoped repository hooks and 3 CPU tests plus 2 subtests passed; range-diff unchanged. [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41425300647/job/125210347134); [Base gate](https://github.com/sgl-project/sglang/actions/runs/41425300984/job/125210545797) sees Ready; missing maintainer-controlled `run-ci`. No new GPU measurement was performed. [PR follow-up](https://github.com/sgl-project/sglang/pull/46830#issuecomment-6574436019).
- **Latest GitHub activity:** [Latest comment 2026-11-02T10:07:34Z](https://github.com/sgl-project/sglang/pull/46830#issuecomment-6574436019); PR updated 2026-11-02T10:07:34Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.llrepro-pp4-dpa4.packup_20261024](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-pp4-dpa4.packup_20261024/README.md); original MI355X deadlock and MI300X masking investigation; production delta matches #33973; CPU 3 tests + 2 subtests and consumer audit.
- **To Ready:** No remaining author-side promotion step; retain scoped historical evidence.
- **To merge / dependency:** Ready; the inherited XPU logger F821 is resolved by upstream #47837 and [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41425300647/job/125210347134). [Base gate](https://github.com/sgl-project/sglang/actions/runs/41425300984/job/125210545797) sees Ready; missing maintainer-controlled `run-ci`. Maintainers: add `run-ci`, trigger required CI and review/approve. Yihou/delegate: respond to functional CI or review findings. This authorization gate is not a functional-test failure.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425300726/job/125210348279); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425097208/job/125209688503); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425300693/job/125210348093); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425300622/job/125210454303); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425300703/job/125210467277); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_eager_metadata_no_host_sync.py
```


### [sgl-project/sglang #47276](https://github.com/sgl-project/sglang/pull/47276) — Speed up gfx950 fused indexer logits/top-k for MTP verify rows.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me, DarkSharpness, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `a133231d3140e081424d833caa0187f15d40dc3c` / `yihou/kf-fused-indexer-gfx950` @ `598ba3f41f53e1c1e2a332a4d369210f283deb5f`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 18, SKIPPED 88, SUCCESS 21. Reviews: claude: COMMENTED (2026-10-29T13:35:23Z).
- **Last curated action:** 2026-11-02: rebased and pushed 598ba3f41f53 on main a133231d3140. Full-tree Ruff, scoped hooks and CPU launch-selection assertions passed; 7 GPU tests skipped; range-diff unchanged. [Lint passed](https://github.com/sgl-project/sglang/actions/runs/41430249327/job/125226720255); [Base attempt 1](https://github.com/sgl-project/sglang/actions/runs/41430249649/job/125226908540) sees Ready; missing maintainer-controlled `run-ci`. [PR follow-up](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6575193133).
- **Latest GitHub activity:** [Latest comment 2026-11-02T10:45:45Z](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6575193133); PR updated 2026-11-02T10:45:45Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** PR-reported bit-identical logits/top-2048/K-write controls and real-acceptance TP8 GSM8K; isolated whole-call 1.20–1.36x; serving ITL used simulated acceptance 3.61. Historical [glm52.llrepro-kf-ar-topk.packup_20261024](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-kf-ar-topk.packup_20261024/README.md) is a related source family, not a verified final-port run mapping.
- **To Ready:** Already Ready. Preserve independent numerical evidence and coordinate #45307; bind final source/run IDs to a public result bundle when available.
- **To merge / dependency:** Ready, [Lint passed](https://github.com/sgl-project/sglang/actions/runs/41430249327/job/125226720255). [Base attempt 1](https://github.com/sgl-project/sglang/actions/runs/41430249649/job/125226908540) sees Ready; missing maintainer-controlled `run-ci`. Maintainers: add `run-ci`, trigger required functional CI and review/approve. Yihou/delegate: handle concrete CI/review findings and maintain source/evidence mappings. Coordinate overlapping #45307; its author must resolve current main conflicts.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41430249431/job/125226722287); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41430246213/job/125226712876); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41430249470/job/125226721326); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41430249500/job/125226865580); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41430249463/job/125226853315); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
# GPU: use a pool node; exact architecture/flags are in the PR
PYTHONPATH=python python -m pytest -q test/registered/amd/test_dsa_gfx950_fused_indexer.py
```


### [sgl-project/sglang #47278](https://github.com/sgl-project/sglang/pull/47278) — Speed up gfx950 FP8 sparse MLA decode with a 64-head split-K plan.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: DarkSharpness, HaiShaw, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `a133231d3140e081424d833caa0187f15d40dc3c` / `yihou/kf-sparse-mla-gfx950` @ `73ee22beaac2989182a7c077c2a9a125dbde4652`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 17, SKIPPED 89, SUCCESS 21. Reviews: claude: COMMENTED (2026-10-29T13:43:32Z).
- **Last curated action:** 2026-11-02: rebased and pushed 73ee22beaac2 on main a133231d3140. Full-tree Ruff, scoped hooks and 3 CPU contracts passed; 22 GPU tests skipped; touched-source change is only inherited main #47782 architecture lookup. [Lint passed](https://github.com/sgl-project/sglang/actions/runs/41430645537/job/125228065040); [Base attempt 1](https://github.com/sgl-project/sglang/actions/runs/41430646049/job/125228291012) sees Ready; missing maintainer-controlled `run-ci`. [PR follow-up](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6575194570).
- **Latest GitHub activity:** [Latest comment 2026-11-02T10:45:50Z](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6575194570); PR updated 2026-11-02T10:45:50Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** PR reports 22/22 kernel tests, FP32-reference SNR >35 dB and real-acceptance GSM8K mean .942 baseline / .941 PR; isolated kernel 2.4–3.2x. Serving ITL used simulated acceptance 3.61. Final-port pack-up mapping still needs audit.
- **To Ready:** Already Ready. Preserve the BF16-partial accuracy change and DCP/LSE coverage; raw run-to-source archiving remains a handoff follow-up.
- **To merge / dependency:** Ready, [Lint passed](https://github.com/sgl-project/sglang/actions/runs/41430645537/job/125228065040). [Base attempt 1](https://github.com/sgl-project/sglang/actions/runs/41430646049/job/125228291012) sees Ready; missing maintainer-controlled `run-ci`. Maintainers: add `run-ci`, trigger required functional CI and review/approve. Yihou/delegate: handle concrete CI/review findings and maintain source/evidence mappings.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41430645677/job/125228066599); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41430644367/job/125228064833); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41430645628/job/125228066044); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41430645669/job/125228253228); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41430645650/job/125228172278); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
# GPU: use a pool node; preserve PR architecture and dtype matrix
PYTHONPATH=python python -m pytest -q test/registered/kernels/ops/attention/test_triton_sparse_mla_hip.py
```


### [sgl-project/sglang #47872](https://github.com/sgl-project/sglang/pull/47872) — Provide opt-in long-context top-k on compact page tables with exact sampling fallbacks.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me, DarkSharpness, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `a133231d3140e081424d833caa0187f15d40dc3c` / `yihou/kf-triton-topk-gfx950` @ `1061c298389facbdeaafced9891a5c7c9b399331`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: CANCELLED 1, FAILURE 18, SKIPPED 91, SUCCESS 25. Reviews: claude: COMMENTED (2026-11-02T07:47:39Z).
- **Last curated action:** 2026-11-02: rebased and pushed 1061c298389f on main a133231d3140. Full-tree Ruff, scoped hooks and 32 CPU tests passed; actual-kernel interpreter controls passed; original GPU underflow reproduced and all 5 repaired MI355X GPU tests passed with zero skips. [Lint passed](https://github.com/sgl-project/sglang/actions/runs/41429823944/job/125225314296); [Base attempt 2](https://github.com/sgl-project/sglang/actions/runs/41429823967/job/125228691842) sees Ready; missing maintainer-controlled `run-ci`. Repaired candidate underflow and deferred optional Gluon imports behind platform/flag guards; restored Ready after GPU validation and retriggered Base with /tag-and-rerun-ci. The new gate supersedes the old Draft failure. [PR follow-up](https://github.com/sgl-project/sglang/pull/47872#issuecomment-6575197280).
- **Latest GitHub activity:** [Latest comment 2026-11-02T10:45:58Z](https://github.com/sgl-project/sglang/pull/47872#issuecomment-6575197280); PR updated 2026-11-02T10:45:58Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [Historical indexer source](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/patches/K01-kf-decode-kernel-overlay.NEWFILES.diff); [PR evidence](https://github.com/sgl-project/sglang/pull/47872). Author reports exact overflow fallback, valid nondeterministic boundary ties and 1.5–1.9x isolated speedup at 400k contexts/72+ rows, but about 1.3% slower serving at 32k with simulated acceptance 3.61. Real-acceptance GSM8K .953 from one server is a smoke test. Maintenance reproduced candidate underflow in the actual GPU pipeline: original 256 valid slots instead of 2048; fixed 2048 unique valid slots and all 5 GPU tests pass without skips. Run kf47872-20261102T1038Z-topk-underflow used MI355X and the public v0.5.22 ROCm10 image, Torch 2.11.0+rocm10.0.0 / Triton 3.8.0. [Pinned source, CPU/GPU records and reproduction scripts](https://github.com/AMD-AGI/Infera/tree/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/validation/kf_47872). The GPU runner uses the exact module/tests with namespace and CI-registration shims for the older image; no new server/model/performance result is claimed.
- **To Ready:** Ready restored after the underflow repair, import-guard regression and exact-source MI355X validation. The temporary correctness Draft is resolved; retain opt-in behavior and the historical performance limits.
- **To merge / dependency:** Ready, [Lint passed](https://github.com/sgl-project/sglang/actions/runs/41429823944/job/125225314296). [Base attempt 2](https://github.com/sgl-project/sglang/actions/runs/41429823967/job/125228691842) sees Ready; missing maintainer-controlled `run-ci`. Maintainers: add `run-ci`, trigger required functional CI and review/approve. Yihou/delegate: handle concrete CI/review findings and maintain source/evidence mappings.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41429823784/job/125228757968); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41429818252/job/125228785971); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41429823788/job/125228768045); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41429823668/job/125228751586); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41429823831/job/125228735765); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/amd/test_dsa_triton_topk_transform.py
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_triton_topk_gate.py
```


### [sgl-project/sglang #47897](https://github.com/sgl-project/sglang/pull/47897) — Speed up gfx950 decode logits while preserving paired-row tails and fallback dispatch.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: DarkSharpness, HaiShaw, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `a133231d3140e081424d833caa0187f15d40dc3c` / `yihou/kf-gluon-indexer-logits-gfx950` @ `fcfa508799f1463dd0e91c278fe80fc6fd9ad0e1`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 17, SKIPPED 89, SUCCESS 21. Reviews: claude: COMMENTED (2026-11-02T09:31:07Z).
- **Last curated action:** 2026-11-02: rebased and pushed fcfa508799f1 on main a133231d3140. Full-tree Ruff, scoped hooks and 4 CPU tests passed; 4 GPU tests skipped; range-diff unchanged. [Lint passed](https://github.com/sgl-project/sglang/actions/runs/41428923887/job/125222275790); [Base attempt 1](https://github.com/sgl-project/sglang/actions/runs/41428924194/job/125222524471) sees Ready; missing maintainer-controlled `run-ci`. [PR follow-up](https://github.com/sgl-project/sglang/pull/47897#issuecomment-6575195946).
- **Latest GitHub activity:** [Latest comment 2026-11-02T10:45:54Z](https://github.com/sgl-project/sglang/pull/47897#issuecomment-6575195946); PR updated 2026-11-02T10:45:54Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [Historical indexer source](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/patches/K01-kf-decode-kernel-overlay.NEWFILES.diff); [PR evidence](https://github.com/sgl-project/sglang/pull/47897). The c2cacf5f candidate matches the frozen indexer module; the PR fixes its fused-pair tail omission. Author reports 2.67–3.06x isolated long-context logits, numerical/top-k checks, and real-acceptance GSM8K means .934 main/.927 PR with overlapping ranges. GSM8K is a short-context smoke test. Serving timings used simulated acceptance 3.61 in a single-node setup, not the frozen PD/AgentX setup. Local PR_NOTES.md and working_process.md identify validation scripts; raw results are not yet a public archived bundle.
- **To Ready:** Already Ready. Preserve existing hardware evidence and the paired-row regression; archived raw run-to-source bundle is still a handoff follow-up.
- **To merge / dependency:** Ready, [Lint passed](https://github.com/sgl-project/sglang/actions/runs/41428923887/job/125222275790). [Base attempt 1](https://github.com/sgl-project/sglang/actions/runs/41428924194/job/125222524471) sees Ready; missing maintainer-controlled `run-ci`. Maintainers: add `run-ci`, trigger required functional CI and review/approve. Yihou/delegate: handle concrete CI/review findings and maintain source/evidence mappings.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41428923967/job/125222277028); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41428919961/job/125222266334); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41428923814/job/125222275548); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41428924043/job/125222431779); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41428923807/job/125222392390); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/amd/test_dsa_gluon_paged_mqa_logits.py
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_mqa_logits_chunking.py
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
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 17, SKIPPED 88, SUCCESS 23. Reviews: none.
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


### [AMD-AGI/Infera #228](https://github.com/AMD-AGI/Infera/pull/228) — Permanent Draft tracker for 48k deployment priorities, implementation PRs and experiment references.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: xiaobochen-amd, jiejingzhangamd, limou102.
- **Base / head:** `main` @ `8847491c2856d5bc43e230f413c62afe68147e46` / `dev/yihou/glm52-pr-tracking` @ `34977c9346a6cd09f33265d0b3cf6583f60980cb`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: SKIPPED 8, SUCCESS 4. Reviews: none.
- **Last curated action:** 2026-11-02: owner requested permanent Draft for view/track; converted #228 to Draft and added a daily Draft-state guard.
- **Latest GitHub activity:** PR updated 2026-11-02T10:09:55Z; no issue comments.
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
| [SemiAnalysisAI/InferenceX #4629](https://github.com/SemiAnalysisAI/InferenceX/pull/4629) | Publish the Infera SGLang AgentX PD recipe. | READY / `a7aed81b6b` | Upstream #33059/#46830/#46820; progress externally owned absorb-BMM #46918 for issue #46829, now covered by scoped MI355X validation (CI/review/merge and image integration remain); select an image with these plus merged #21194 and remove engine patches. Coordinate #630/#47117, diagnose platform RDMA bring-up, complete sweep and review. |
| [sgl-project/sglang #21194](https://github.com/sgl-project/sglang/pull/21194) | Fix GLM PP construction on stages without an embedding. | MERGED / `2059478e21` | Validate a pinned image carrying the merged fix. No duplicate filing; preserve historical source and attribution. |
| [sgl-project/sglang #34465](https://github.com/sgl-project/sglang/pull/34465) | Earlier PP non-first-rank hidden-size proposal | CLOSED / `62cd16df78` | Closed unmerged. Third-party #21194 subsequently merged on October 29 and guards the GLM-5.2 PP construction failure. Validate image inclusion; no new duplicate S13 filing. Preserve archived source and attribution. |
| [sgl-project/sglang #35233](https://github.com/sgl-project/sglang/pull/35233) | HiCache host-pointer alias resolution | MERGED / `c29e6054f5` | Merged into base lineage; do not reapply old #33968 workaround. |
| [sgl-project/sglang #41395](https://github.com/sgl-project/sglang/pull/41395) | Early-send consumer completion | MERGED / `9fc5f30621` | Merged in frozen base; preserve separate producer patch. |
| [sgl-project/sglang #42553](https://github.com/sgl-project/sglang/pull/42553) | Host-allocation pointer handling | MERGED / `b2de502825` | Merged upstream; retain correct frozen-source behavior. |
| [sgl-project/sglang #45524](https://github.com/sgl-project/sglang/pull/45524) | Producer event before non-final PP KV send | READY / `14c8591329` | Original author: rebase on current main; CI-authorized maintainer: add run-ci and rerun. Review the narrowed non-final PP-chunk producer-event fix and unit test; cached-prefix change was dropped on October 29. No duplicate; #43299 fixes a distinct path. Configuration scope: relevant for PP communication overlap, async depth > 0, or pure-chunk output-communication skipping. Not a first-version #4629 prerequisite with its current default PP settings. |
| [sgl-project/sglang #46918](https://github.com/sgl-project/sglang/pull/46918) | Keep absorb-BMM hand tuning while limiting the waves-per-EU budget that may trigger the gfx950 compiler abort. | READY / `9763ef56bf` | Maintainers: enable run-ci and review/merge; original author: rebase if requested and rerun affected checks. Our MI355X evidence covers Q-absorb at this exact head; other architectures and model integration are not claimed. Select an image containing a reviewed fix before removing the historical workaround. Do not duplicate or modify the original author's branch. [MI355X evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md); [upstream discussion](https://github.com/sgl-project/sglang/pull/46918#issuecomment-6572803085). |
| [AMD-AGI/Infera #222](https://github.com/AMD-AGI/Infera/pull/222) | Advertise the KV-events port allocated by srt-slurm. | MERGED / `f668fbf773` | Use a release carrying this fix; verify allocated endpoints with colocated prefills. |
| [NVIDIA/srt-slurm #630](https://github.com/NVIDIA/srt-slurm/pull/630) | Launch Infera frontends with etcd-registered SGLang workers. | READY / `cb38af8275` | Complete frontend integration review; reconcile the recipe backport when merged. |
| [kvcache-ai/Mooncake #4540](https://github.com/kvcache-ai/Mooncake/pull/4540) | Select destination-local GPU write rails. | DRAFT / `6fccbba168` | Coordinate fabric-specific validation and review; not a first-version universal prerequisite. |
| [kvcache-ai/Mooncake #4541](https://github.com/kvcache-ai/Mooncake/pull/4541) | Make RDMA ACK timeout configurable. | DRAFT / `faa24ab499` | Coordinate fabric-specific validation and review; do not attribute RDMA registration EINVAL to ACK timeout without evidence. |
| [sgl-project/sglang #45307](https://github.com/sgl-project/sglang/pull/45307) | Overlapping gfx950 fused indexer optimization | READY / `1255a0a4ed` | Coordinate overlap with our #47276; rebase whichever lands second. |
| [sgl-project/sglang #47117](https://github.com/sgl-project/sglang/pull/47117) | Document Infera GLM-5.2 PD on MI355X. | DRAFT / `3e89cdc460` | Coordinate cookbook review/merge with the #4629 publication plan. |
| [ROCm/aiter #6697](https://github.com/ROCm/aiter/pull/6697) | Related upstream MoE sort-into-GEMM1 experiment; not our five-hook KF port. | READY / `9a20161613` | External author: resolve current main conflicts and respond to the design objection about merging sorting into GEMM1. Track relevance; do not duplicate or modify the author branch. Our separate five-patch MoE port remains held until an active path and current-baseline gain are demonstrated. |
| [SemiAnalysisAI/InferenceX-app #1542](https://github.com/SemiAnalysisAI/InferenceX-app/pull/1542) | Display the Infera SGLang framework name in the dashboard. | DRAFT / `4aa976240e` | Dashboard review/merge after recipe coordination. |
| [kvcache-ai/Mooncake #4340](https://github.com/kvcache-ai/Mooncake/pull/4340) | HIP IPC suballocation relocation | READY / `9b4b049d9c` | salexspb owns root fix; our R018/R019 evidence is posted. Not used in llr37 RDMA recipe. |
| [kvcache-ai/Mooncake #4953](https://github.com/kvcache-ai/Mooncake/pull/4953) | Asynchronous HIP copy submission | READY / `35c488e938` | Track interaction with #5005; source API unchanged, combined runtime not qualified. |
| [kvcache-ai/Mooncake #5030](https://github.com/kvcache-ai/Mooncake/pull/5030) | Wheel structured-ranges packaging fix | MERGED / `a068e70bef` | Merged and included in #5005 rebase; prior wheel blocker cleared. |
| [kvcache-ai/Mooncake #5144](https://github.com/kvcache-ai/Mooncake/pull/5144) | CUDA guard target generated-header include path | MERGED / `39c861c81e` | Existing fix for current #5005 inherited CUDA build failure; follow merge then rebase, no duplicate. |
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
| [sgl-project/sglang #47184](https://github.com/sgl-project/sglang/pull/47184) | Rebase the original PP placeholder-attribute repair. | READY / `ce3087c8bb` | Original author: verify a nightly carrying #21194 and retire this redundant PR. |
| [sgl-project/sglang #45233](https://github.com/sgl-project/sglang/pull/45233) | Speculative penalty accounting proposal | CLOSED / `612b93311b` | Closed unmerged; recheck replacements before reopening work. |

## Issue-only work and patches without a PR

| Issue | State | Updated |
|---|---|---|
| [[Bug] ROCm absorb-BMM hand config aborts Triton compilation at M=256 on gfx950](https://github.com/sgl-project/sglang/issues/46829) | OPEN | 2026-10-27T11:15:22Z |
| [[Bug] ROCm DCP-PD allocator reserve growth precedes HIP resource aborts under burst load](https://github.com/sgl-project/sglang/issues/46831) | OPEN | 2026-10-30T13:42:15Z |

Required unpublished S13, A01, the full historical K01 overlay, I07 and inherited image patches are recorded in [deployment requirements](README.md). Remaining exclusions and reasons are in [PR_PACKUP_MAP.md](PACKUP_MAP.md).
