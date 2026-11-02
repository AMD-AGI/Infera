# Public PR handoff and daily status

Last successful GitHub refresh: **2026-11-02T10:09:36.958069+00:00**. Author: `dorado269`.

**60 PRs: 16 open (9 ready / 7 draft), 26 merged, 18 closed unmerged.** Includes historical work outside GLM-5.2; external authors are listed separately.

Ready means non-draft, not maintainer approval or passed functional CI. GitHub mergeability is a snapshot, not a merge recommendation.

Daily refresh: 09:00 UTC (17:00 Asia/Shanghai), on the configured maintenance host. A timestamp older than 36 hours requires operator investigation. Raw CI counts can include multiple workflows; skipped/draft/permission gates do not establish executed test coverage.

Task owner is yihou until an explicit delegate is recorded. Requested reviewers are not assumed to have accepted work. Edit curated notes to record a named delegate, evidence and the next action; the updater preserves those notes and flags changed heads.

Priority P0 = historical frozen-image behavior / required reproduction inputs; P1 = related current-main kernel ports and evidence packaging; P2 = other GLM fixes; P3 = unrelated or comparison work. See [deployment requirements](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/LLR37_DEPLOYMENT_PRIORITY.md).

## Active work

| Priority | PR / purpose | State / head | Historical 47k-class deployment | Ready action | Merge action / waiting on |
|---|---|---|---|---|---|
| P0 | [sgl-project/sglang #33059](https://github.com/sgl-project/sglang/pull/33059) — Make real query rows and padded Aiter metadata agree. | READY / `795488cd19` | BAKED: historical Aiter query-row repair; current-main PR differs from image script. | Already Ready. Already Ready by the author/user decision; preserve that state. New MI355X DP and padded MTP evidence supersedes the earlier Draft/missing-matrix assessment. Native MI300 coverage remains a reviewer follow-up, not a Draft promotion step. | Current head 795488cd is Ready and lint passes. The existing run-ci label is present; latest [Base gate](https://github.com/sgl-project/sglang/actions/runs/41423254776/job/125204140539) sees Draft=false but hits the 120-minute account cooldown after a counted run at 09:07:56 UTC. Nominal eligibility is November 2 at 11:07:56 UTC if no newer counted run intervenes; expiry does not automatically rerun CI. Rerun required checks when policy permits, address review findings, and obtain native MI300 validation through an equipped CI owner. Reviewer: HaiShaw. The inherited XPU logger error is fixed upstream by #47837 and already included; the shared Qwen fused-KV workaround #46861 is also inherited. |
| P0 | [sgl-project/sglang #46820](https://github.com/sgl-project/sglang/pull/46820) — Keep a blocked metrics consumer from stopping the scheduler. | READY / `783c1a2192` | BAKED: historical nonblocking KV metrics send. | Already Ready. No remaining author-side promotion step; no GPU required for the socket contract. | Ready; the inherited XPU logger F821 is resolved by upstream #47837 and [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41424697358/job/125208358328). [Base gate](https://github.com/sgl-project/sglang/actions/runs/41424697709/job/125208623306) sees Ready; missing maintainer-controlled `run-ci`. Maintainers: add `run-ci`, trigger required CI and review/approve. Yihou/delegate: respond to functional CI or review findings. This authorization gate is not a functional-test failure. |
| P0 | [sgl-project/sglang #46830](https://github.com/sgl-project/sglang/pull/46830) — Remove rank-divergent host syncs that can deadlock DSA eager execution. | READY / `625403853a` | BAKED: historical dsa_dp_sync.diff; replaces closed #33973. | Already Ready. No remaining author-side promotion step; retain scoped historical evidence. | Ready; the inherited XPU logger F821 is resolved by upstream #47837 and [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41425300647/job/125210347134). [Base gate](https://github.com/sgl-project/sglang/actions/runs/41425300984/job/125210545797) sees Ready; missing maintainer-controlled `run-ci`. Maintainers: add `run-ci`, trigger required CI and review/approve. Yihou/delegate: respond to functional CI or review findings. This authorization gate is not a functional-test failure. |
| P1 | [AMD-AGI/Infera #221](https://github.com/AMD-AGI/Infera/pull/221) — Publish six KernelForge families and their historical evidence. | DRAFT / `b8b3938d6b` | EVIDENCE/TOOLING: useful for handoff; not a serving dependency or an Infera core inference fix. | Retarget/reconcile with main after merged #220. Resolve 20 unavailable external fixtures; validate the 502 redacted derivatives before presenting them as tested kernels, or scope the PR explicitly to archival evidence. Re-run catalog validation/CPU checks after retarget. | Yihou/delegate: scope and evidence decision; maintainers: catalog review and CI on main. Historical results apply to original bytes, not renamed derivatives. |
| P1 | [sgl-project/sglang #47276](https://github.com/sgl-project/sglang/pull/47276) — Speed up gfx950 fused indexer logits/top-k for MTP verify rows. | READY / `23a8de201b` | K01 FAMILY PORT: prioritize upstream adoption; not the exact historical overlay and not a new patch needed atop llr37. | Already Ready. Already non-draft. Audit final October 29 run IDs/source hashes against pack-ups and preserve real-acceptance versus simulated-performance boundaries. | Ready; the inherited XPU logger F821 is resolved by upstream #47837 and [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41425379090/job/125210599208). [Base gate](https://github.com/sgl-project/sglang/actions/runs/41425379155/job/125210861058) sees Ready; missing maintainer-controlled `run-ci`. Maintainers: add `run-ci`, trigger required CI and review/approve. Yihou/delegate: respond to functional CI or review findings. This authorization gate is not a functional-test failure. Coordinate overlapping #45307 and rebase whichever lands second. |
| P1 | [sgl-project/sglang #47278](https://github.com/sgl-project/sglang/pull/47278) — Speed up gfx950 FP8 sparse MLA decode with a 64-head split-K plan. | READY / `d3a13d1cf8` | K01 FAMILY PORT: prioritize upstream adoption; BF16 split partials differ from the frozen overlay. | Already Ready. Already non-draft. Bind final tests/runs to source hashes and archive locations; do not substitute for the complete K01 overlay. | Ready; the inherited XPU logger F821 is resolved by upstream #47837 and [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41425638344/job/125211454851). [Base gate](https://github.com/sgl-project/sglang/actions/runs/41425638677/job/125211664882) sees Ready; missing maintainer-controlled `run-ci`. Maintainers: add `run-ci`, trigger required CI and review/approve. Yihou/delegate: respond to functional CI or review findings. This authorization gate is not a functional-test failure. |
| P2 | [kvcache-ai/Mooncake #5005](https://github.com/kvcache-ai/Mooncake/pull/5005) — Offer opt-in HIP copy-stream priority without changing the default. | DRAFT / `8313cccbaa` | NOT IN LLR37: the peak uses cross-host RDMA with HIP transport disabled. | Yihou must perform the human line-by-line review required by Mooncake AGENTS.md. Reconcile the current base CUDA build regression with #5144 and rerun affected CI. Existing HIP tests count; no blanket new serving benchmark. | Follow external #5144; after merge rebase and rerun CUDA build checks. #4953 source compatibility was checked but combined runtime was not. Maintainers review after human submitter review. |
| P2 | [sgl-project/sglang #45484](https://github.com/sgl-project/sglang/pull/45484) — Preserve the Aiter is_shuffled weight tag through offload/onload. | READY / `43dec26b03` | NOT IN LLR37: OffloaderV2 weight offload is not HiCache. | Already Ready. No remaining author-side promotion step; CPU and exact-head MI355X onload regression already exist. | Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result. |
| P2 | [sgl-project/sglang #46819](https://github.com/sgl-project/sglang/pull/46819) — Cache DCP gather metadata instead of rebuilding it per transfer. | DRAFT / `aed9bc6953` | NOT IN LLR37: DCP metadata optimization; the peak uses DPA4/DCP1. | CPU-first option: narrow to the independently measured cache and update tests/body. If retaining pinned-index/priority changes, qualify their lifetime, interference and PP ordering on allocated hardware. Preserve producer wait. | Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result. Related #44724/#39768/#43782 are overlap coordination, not established mandatory prerequisites. |
| P2 | [sgl-project/sglang #46822](https://github.com/sgl-project/sglang/pull/46822) — Skip HIP DSA indexing when an eager DP rank has no tokens. | READY / `11262e1199` | NOT ADDED TO LLR37: separate DCP/eager-idle fault repair. | Already Ready. No remaining author-side promotion step; retain historical production equivalence. | Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result. |
| P2 | [sgl-project/sglang #46824](https://github.com/sgl-project/sglang/pull/46824) — Relayout DCP-sharded draft KV together with target KV. | DRAFT / `1bcfc52149` | NOT IN LLR37: DCP1-to-DCP-N transport. | Map pdBe-r1 to current supported combinations; resolve or explicitly reject the known DP2×DCP2 graph-garbling case. Test only uncovered MTP/prefix/PP cases on allocated hardware as needed. #44322 has merged; old backend hold is cleared. | Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result. This transport must land before #46826 enables the configuration. #45524 producer ordering remains separate. |
| P2 | [sgl-project/sglang #46826](https://github.com/sgl-project/sglang/pull/46826) — Allow DCP decode behind DCP1 prefill when transport/backend support it. | DRAFT / `21f8e688fa` | NOT IN LLR37: constrained ROCm DCP enablement. | Complete the supported-configuration coverage with #46824; preserve rejection of unqualified graph/layout combinations. Backend/default policy is already tested; do not wait again for merged #44322. | Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result. Merge after #46824; review may proceed before that merge with the dependency explicit. |
| P3 | [AMD-AGI/Infera #164](https://github.com/AMD-AGI/Infera/pull/164) — Keep an alternative fast-benchmark script for comparison. | DRAFT / `4676ec9087` | NOT IN LLR37: comparison-only benchmark tooling. | Decide whether to close the comparison draft or define supported behavior, fill the placeholder body and add regression evidence before promotion. | Yihou/delegate: scope decision first; then fix failing checks and request review. |
| P3 | [AMD-AGI/Infera #228](https://github.com/AMD-AGI/Infera/pull/228) — Permanent Draft tracker for 48k deployment priorities, implementation PRs and experiment references. | DRAFT / `a1f30364ce` | COORDINATION ONLY: the main-based 48k deployment/PR handoff; not runtime code. | PERMANENT DRAFT: for view and track only. Never promote this documentation PR to Ready. | DO NOT MERGE. Maintain the open Draft as the team coordination surface; review/merge priorities apply to linked implementation PRs only. |
| TRIAGE | [sgl-project/sglang #47872](https://github.com/sgl-project/sglang/pull/47872) — [Perf] [AMD] dsa: Opt-in single-pass Triton top-k for long-context decode rows | READY / `a899828aaf` | Unclassified: inspect before adding to a deployment | Already Ready. New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes. | Resolve current review and required CI; no automatic merge certification. |
| TRIAGE | [sgl-project/sglang #47897](https://github.com/sgl-project/sglang/pull/47897) — [Perf] [AMD] dsa: In-tree gfx950 paged MQA logits for decode / MTP-verify rows | READY / `d6876dbdfc` | Unclassified: inspect before adding to a deployment | Already Ready. New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes. | Resolve current review and required CI; no automatic merge certification. |

## Handoff details

### [sgl-project/sglang #33059](https://github.com/sgl-project/sglang/pull/33059) — Make real query rows and padded Aiter metadata agree.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me.
- **Base / head:** `main` @ `b45a36531c3673048aeaf35322c3df7f18edd038` / `fix-dsa-indexer-hip-aiter-dp-padding` @ `795488cd19aaa3cc852d6c87ffe36b88af20ca23`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 18, SKIPPED 88, SUCCESS 21. Reviews: claude: COMMENTED (2026-11-02T09:04:20Z).
- **Last curated action:** 2026-11-02: author published the [MI355X model results](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447), marked Ready, and advanced the branch to 795488cd by merging main b45a3653. Status audit confirmed tested/current DSA and test blobs match. [Latest lint](https://github.com/sgl-project/sglang/actions/runs/41423254991/job/125203622542) passed at 09:49:54 UTC. Base now stops at account cooldown, superseding the earlier Draft-gate and XPU-lint failures; no branch rewrite is needed for lint.
- **Latest GitHub activity:** [Latest comment 2026-11-02T09:41:37Z](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447); PR updated 2026-11-02T09:48:02Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [Author GPU report](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447): 8x MI355X DeepSeek-V3.2 DP GSM8K 0.957; basic TP in AMD CI mode 0.958; DP + EAGLE MTP with eager padded rows 0.951 versus unpatched main crashing with lengths must have one entry per score row. Runs used ab4407a57c; independently checked identical production DSA and regression-test Git blobs at current head 795488cd (306d4b5f619fc37d8760d2e22dc07ede2e1d280e and 7a7e8c9b759c80f6666fe80fe793e4cabaacb746). These are author-reported model runs, not rerun by this status audit. The MI300 configuration failed on MI355X both with and without the patch; native MI300 coverage is still absent. Existing 12 CPU regression tests and historical [GLM pack-up](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.llrepro-pp4-dpa4.packup_20261024/README.md) remain separate evidence.
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
- **Evidence / pack-up:** [glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/README.md); c44a to c44b; four real-ZMQ CPU tests pass; no model-throughput claim.
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
- **Evidence / pack-up:** [glm52.llrepro-pp4-dpa4.packup_20261024](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.llrepro-pp4-dpa4.packup_20261024/README.md); original MI355X deadlock and MI300X masking investigation; production delta matches #33973; CPU 3 tests + 2 subtests and consumer audit.
- **To Ready:** No remaining author-side promotion step; retain scoped historical evidence.
- **To merge / dependency:** Ready; the inherited XPU logger F821 is resolved by upstream #47837 and [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41425300647/job/125210347134). [Base gate](https://github.com/sgl-project/sglang/actions/runs/41425300984/job/125210545797) sees Ready; missing maintainer-controlled `run-ci`. Maintainers: add `run-ci`, trigger required CI and review/approve. Yihou/delegate: respond to functional CI or review findings. This authorization gate is not a functional-test failure.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425300726/job/125210348279); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425097208/job/125209688503); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425300693/job/125210348093); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425300622/job/125210454303); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425300703/job/125210467277); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_eager_metadata_no_host_sync.py
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

### [sgl-project/sglang #47276](https://github.com/sgl-project/sglang/pull/47276) — Speed up gfx950 fused indexer logits/top-k for MTP verify rows.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me, DarkSharpness, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `b45a36531c3673048aeaf35322c3df7f18edd038` / `yihou/kf-fused-indexer-gfx950` @ `23a8de201bd5c157aed0ab8949c8dc57057e2bad`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 18, SKIPPED 88, SUCCESS 21. Reviews: claude: COMMENTED (2026-10-29T13:35:23Z).
- **Last curated action:** 2026-11-02: rebased/pushed to 23a8de201bd5 on main b45a36531c36, including upstream logger fix #47837. Full-tree Ruff, scoped repository hooks and CPU launch-selection assertions passed; 7 GPU tests skipped; range-diff unchanged. [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41425379090/job/125210599208); [Base gate](https://github.com/sgl-project/sglang/actions/runs/41425379155/job/125210861058) sees Ready; missing maintainer-controlled `run-ci`. No new GPU measurement was performed. [PR follow-up](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6574437744).
- **Latest GitHub activity:** [Latest comment 2026-11-02T10:07:39Z](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6574437744); PR updated 2026-11-02T10:07:40Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** PR-reported bit-identical logits/top-2048/K-write controls and real-acceptance TP8 GSM8K; isolated whole-call 1.20–1.36x; serving ITL used simulated acceptance 3.61. Historical [glm52.llrepro-kf-ar-topk.packup_20261024](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.llrepro-kf-ar-topk.packup_20261024/README.md) is a related source family, not a verified final-port run mapping.
- **To Ready:** Already non-draft. Audit final October 29 run IDs/source hashes against pack-ups and preserve real-acceptance versus simulated-performance boundaries.
- **To merge / dependency:** Ready; the inherited XPU logger F821 is resolved by upstream #47837 and [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41425379090/job/125210599208). [Base gate](https://github.com/sgl-project/sglang/actions/runs/41425379155/job/125210861058) sees Ready; missing maintainer-controlled `run-ci`. Maintainers: add `run-ci`, trigger required CI and review/approve. Yihou/delegate: respond to functional CI or review findings. This authorization gate is not a functional-test failure. Coordinate overlapping #45307 and rebase whichever lands second.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425379445/job/125210601468); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425373838/job/125210585791); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425379420/job/125210600443); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425379038/job/125210716928); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425379193/job/125210734722); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
# GPU: use a pool node; exact architecture/flags are in the PR
PYTHONPATH=python python -m pytest -q test/registered/amd/test_dsa_gfx950_fused_indexer.py
```

### [sgl-project/sglang #47278](https://github.com/sgl-project/sglang/pull/47278) — Speed up gfx950 FP8 sparse MLA decode with a 64-head split-K plan.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: DarkSharpness, HaiShaw, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `b45a36531c3673048aeaf35322c3df7f18edd038` / `yihou/kf-sparse-mla-gfx950` @ `d3a13d1cf87fa33107d015c1d122ee48cacd9888`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 17, SKIPPED 89, SUCCESS 21. Reviews: claude: COMMENTED (2026-10-29T13:43:32Z).
- **Last curated action:** 2026-11-02: rebased/pushed to d3a13d1cf87f on main b45a36531c36, including upstream logger fix #47837. Full-tree Ruff, scoped repository hooks and 3 CPU contracts passed; 22 GPU tests skipped; range-diff unchanged. [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41425638344/job/125211454851); [Base gate](https://github.com/sgl-project/sglang/actions/runs/41425638677/job/125211664882) sees Ready; missing maintainer-controlled `run-ci`. No new GPU measurement was performed. [PR follow-up](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6574439522).
- **Latest GitHub activity:** [Latest comment 2026-11-02T10:07:44Z](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6574439522); PR updated 2026-11-02T10:07:44Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** PR reports 22/22 kernel tests, FP32-reference SNR >35 dB and real-acceptance GSM8K mean .942 baseline / .941 PR; isolated kernel 2.4–3.2x. Serving ITL used simulated acceptance 3.61. Final-port pack-up mapping still needs audit.
- **To Ready:** Already non-draft. Bind final tests/runs to source hashes and archive locations; do not substitute for the complete K01 overlay.
- **To merge / dependency:** Ready; the inherited XPU logger F821 is resolved by upstream #47837 and [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41425638344/job/125211454851). [Base gate](https://github.com/sgl-project/sglang/actions/runs/41425638677/job/125211664882) sees Ready; missing maintainer-controlled `run-ci`. Maintainers: add `run-ci`, trigger required CI and review/approve. Yihou/delegate: respond to functional CI or review findings. This authorization gate is not a functional-test failure.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425638802/job/125211457799); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425636676/job/125211453931); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425638432/job/125211456020); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425638528/job/125211640493); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41425638567/job/125211581108); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
# GPU: use a pool node; preserve PR architecture and dtype matrix
PYTHONPATH=python python -m pytest -q test/registered/kernels/ops/attention/test_triton_sparse_mla_hip.py
```

### [kvcache-ai/Mooncake #5005](https://github.com/kvcache-ai/Mooncake/pull/5005) — Offer opt-in HIP copy-stream priority without changing the default.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `72e267f41d7eef241b641b9d132de431adf71ef8` / `yihou/hip-copy-stream-priority-20261027` @ `8313cccbaae8f3c9d7e0a29b830ce6a8a892dd91`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 3, SKIPPED 1, SUCCESS 31. Reviews: none.
- **Last curated action:** 2026-11-02: signed rebase onto 72e267f4; packaging test/lint pass; contributed CUDA build failure evidence to #5144.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:18:39Z](https://github.com/kvcache-ai/Mooncake/pull/5005#issuecomment-6568944195); PR updated 2026-11-02T04:28:10Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/README.md) R039/R043 microbench; allocated MI355X HIP build +4/4 CTest before rebase, stream/test sources unchanged; source packaging regression1pass. #5030 fixed previous wheel failure; new missing mooncake_te_build_config.h is covered by #5144.
- **To Ready:** Yihou must perform the human line-by-line review required by Mooncake AGENTS.md. Reconcile the current base CUDA build regression with #5144 and rerun affected CI. Existing HIP tests count; no blanket new serving benchmark.
- **To merge / dependency:** Follow external #5144; after merge rebase and rerun CUDA build checks. #4953 source compatibility was checked but combined runtime was not. Maintainers review after human submitter review.

- Red-check entry points: [CTest unit tests (Python 3.12)](https://github.com/kvcache-ai/Mooncake/actions/runs/41387680143/job/125092088899); [tent-ci (cuda-on)](https://github.com/kvcache-ai/Mooncake/actions/runs/41387680143/job/125092276800); [CI Gate](https://github.com/kvcache-ai/Mooncake/actions/runs/41387680143/job/125102510508).

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
python -m pytest -q python/tests/packaging/test_project.py::test_structured_object_storage_has_one_authoritative_source
# HIP build/CTest commands and original node evidence: PR body and reviews/M02-followup-body-20261102.md
```

### [sgl-project/sglang #45484](https://github.com/sgl-project/sglang/pull/45484) — Preserve the Aiter is_shuffled weight tag through offload/onload.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `449617bc94a5d4ae23ad62d0da2f842e711ca983` / `fix/offloader-keep-is-shuffled` @ `43dec26b032c08f4e9eee0c772db9f44d3afc275`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: CANCELLED 2, FAILURE 16, SKIPPED 90, SUCCESS 29. Reviews: none.
- **Last curated action:** 2026-11-02: reconciled current GitHub state and archived evidence for handoff.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:21:05Z](https://github.com/sgl-project/sglang/pull/45484#issuecomment-6568974915); PR updated 2026-11-02T04:21:05Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.v0521-offloaderv2.packup_20261015](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.v0521-offloaderv2.packup_20261015/README.md) ofs0c/ofg3c/ofg2c 23/23 each; AST-equivalent port, CPU 3+2 and allocated MI355X two-onload-cycle test. Full-model rerun is not an automatic Ready gate.
- **To Ready:** No remaining author-side promotion step; CPU and exact-head MI355X onload regression already exist.
- **To merge / dependency:** Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40471208261/job/122078486436); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40471208251/job/122078485647); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40471208207/job/122078579497); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40471208218/job/122078580449); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40471208196/job/122078578924); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/utils/test_offloader.py
# Existing GPU command; allocate a pool node if rerunning
PYTHONPATH=python python test/registered/model_loading/test_offloader_shuffled_weights.py
```

### [sgl-project/sglang #46819](https://github.com/sgl-project/sglang/pull/46819) — Cache DCP gather metadata instead of rebuilding it per transfer.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `8a455f1b9005a05091ad1825f0f3493bfd74d27d` / `yihou/glm52-dcp-pack-cache-20261027` @ `aed9bc6953d0a95c879cff5151ee20627e9937b3`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 18, SKIPPED 87, SUCCESS 24. Reviews: none.
- **Last curated action:** 2026-11-02: reconciled current GitHub state and archived evidence for handoff.
- **Latest GitHub activity:** [Latest comment 2026-10-27T11:02:43Z](https://github.com/sgl-project/sglang/pull/46819#issuecomment-6440276807); PR updated 2026-11-02T04:22:19Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.dcpx-pack-fix-best.packup_20261023](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.dcpx-pack-fix-best.packup_20261023/README.md) dcpx8c/9/10; current-head MI355X 3 tests +16 subtests, CPU 8+4; isolated cache medians 39.57→17.15 /39.09→17.07 /130.42→107.52 us. dcpx10 also removed a wait; its combined speedup does not transfer to this PR.
- **To Ready:** CPU-first option: narrow to the independently measured cache and update tests/body. If retaining pinned-index/priority changes, qualify their lifetime, interference and PP ordering on allocated hardware. Preserve producer wait.
- **To merge / dependency:** Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result. Related #44724/#39768/#43782 are overlap coordination, not established mandatory prerequisites.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40491188646/job/122145248512); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40491188142/job/122145248715); [Check PR title](https://github.com/sgl-project/sglang/actions/runs/40491187276/job/122145248409); [Check PR title](https://github.com/sgl-project/sglang/actions/runs/40492975001/job/122151238195); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40491188159/job/122145348637); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m pytest -q test/registered/unit/disaggregation/test_dcp_pack_metadata.py test/registered/unit/disaggregation/test_dcp_pack.py
# GPU only when qualifying the retained stream/lifetime changes
PYTHONPATH=python python -m pytest -q test/registered/kernels/ops/kvcache/test_pd_dcp_gather.py
```

### [sgl-project/sglang #46822](https://github.com/sgl-project/sglang/pull/46822) — Skip HIP DSA indexing when an eager DP rank has no tokens.

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me.
- **Base / head:** `main` @ `a6f5bd397be420c12d0690c07e16e241124269e5` / `yihou/rocm-dsa-eager-idle-20261027` @ `11262e1199d914ac1946bdcf3d18f56cc3f45612`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: CANCELLED 1, FAILURE 17, SKIPPED 90, SUCCESS 25. Reviews: none.
- **Last curated action:** 2026-11-02: reconciled current GitHub state and archived evidence for handoff.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:21:06Z](https://github.com/sgl-project/sglang/pull/46822#issuecomment-6568975144); PR updated 2026-11-02T04:21:06Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.dcp4-pd.packup_20261022](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.dcp4-pd.packup_20261022/README.md) R16/R17 fault, R18 5/5 +40/40; later separate R24/R25 row-list bug covered by #45796; current CPU 4 tests +20 subtests.
- **To Ready:** No remaining author-side promotion step; retain historical production equivalence.
- **To merge / dependency:** Maintainers: enable required CI (SGLang run-ci where absent), review and approve; us: address concrete review findings, rebase if needed, then rerun affected checks. A permission/draft gate is not a functional-test result.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492640626/job/122150134982); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492640595/job/122150134832); [Check PR title](https://github.com/sgl-project/sglang/actions/runs/40492640193/job/122150139430); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492640629/job/122150249116); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/40492640511/job/122150226596); remaining checks are in the linked PR and JSON snapshot.

Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):

```sh
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_indexer_idle.py
```

### [sgl-project/sglang #46824](https://github.com/sgl-project/sglang/pull/46824) — Relayout DCP-sharded draft KV together with target KV.

- **State / responsibility:** DRAFT; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: none requested.
- **Base / head:** `main` @ `e0c7992d0d48c8085655752de7a9f43bf1862bbc` / `yihou/pd-dcp-sharded-draft-20261027` @ `1bcfc52149011aae08502e9da0620b0325492799`.
- **Review / CI:** REVIEW_REQUIRED; mergeability UNKNOWN / UNKNOWN; 0 unresolved inline threads. Checks: FAILURE 17, SKIPPED 88, SUCCESS 23. Reviews: none.
- **Last curated action:** 2026-11-02: signed rebase onto acd6f5570; actual canonical-policy regression added, 43 CPU tests +61 subtests and lint pass.
- **Latest GitHub activity:** [Latest comment 2026-11-02T04:20:34Z](https://github.com/sgl-project/sglang/pull/46824#issuecomment-6568968279); PR updated 2026-11-02T04:21:11Z.
- **Evidence freshness:** Recorded head unchanged; evidence retains its stated limits. Notes last assessed 2026-11-02; new comments/reviews still require human triage.
- **Activity triage:** GitHub activity timestamp matches the curated audit.
- **Evidence / pack-up:** [glm52.dcp4-pd.packup_20261022](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.dcp4-pd.packup_20261022/README.md) pdBe-r1 math29×2, passkey8K/34K/96K; rebase production delta unchanged; 43 CPU tests +61 subtests exercise merged sharding policy.
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
- **Evidence / pack-up:** [glm52.dcp4-pd.packup_20261022](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.dcp4-pd.packup_20261022/README.md) pdBe-r1; current 12 CPU tests +13 subtests cover implicit Triton and a2a Q-rep defaults. Combined graph/coverage gap remains.
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
- **Base / head:** `main` @ `8847491c2856d5bc43e230f413c62afe68147e46` / `dev/yihou/glm52-pr-tracking` @ `a1f30364ce0e561faf62e35dcfe7682d2f110dcc`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: SKIPPED 8, SUCCESS 4. Reviews: none.
- **Last curated action:** 2026-11-02: owner requested permanent Draft for view/track; converted #228 to Draft and added a daily Draft-state guard.
- **Latest GitHub activity:** PR updated 2026-11-02T09:01:30Z; no issue comments.
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

### [sgl-project/sglang #47872](https://github.com/sgl-project/sglang/pull/47872) — [Perf] [AMD] dsa: Opt-in single-pass Triton top-k for long-context decode rows

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: 1am9trash, hubertlu-tw, kkHuang-amd, HaiShaw, Fridge003, YAMY1234, rainj-me, DarkSharpness, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `960e30769e8af4618ca0f7a6f1d85464176d881b` / `yihou/kf-triton-topk-gfx950` @ `a899828aaf4fa1ee2013c072a19be9115b773c09`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 18, SKIPPED 88, SUCCESS 22. Reviews: claude: COMMENTED (2026-11-02T07:47:39Z).
- **Last curated action:** Discovered by the daily author-wide query.
- **Latest GitHub activity:** PR updated 2026-11-02T09:31:23Z; no issue comments.
- **Evidence freshness:** NOT AUDITED. Notes last assessed unrecorded; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** No curated evidence yet; inspect the PR.
- **To Ready:** New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.
- **To merge / dependency:** Resolve current review and required CI; no automatic merge certification.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41408590510/job/125156177835); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41408590617/job/125156184075); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41408590394/job/125156176865); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41408590616/job/125156264608); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41408590374/job/125156270189); remaining checks are in the linked PR and JSON snapshot.

### [sgl-project/sglang #47897](https://github.com/sgl-project/sglang/pull/47897) — [Perf] [AMD] dsa: In-tree gfx950 paged MQA logits for decode / MTP-verify rows

- **State / responsibility:** READY; yihou; delegate unassigned. GitHub assignees: none. Requested reviewers: DarkSharpness, HaiShaw, BBuf, celve, HydraQYH, yuan-luo.
- **Base / head:** `main` @ `1ced9a9eb0224e5c12c9ed236b188868a635cd5c` / `yihou/kf-gluon-indexer-logits-gfx950` @ `d6876dbdfcc6ed3ff26680405d19b1b16f02ab12`.
- **Review / CI:** REVIEW_REQUIRED; mergeability MERGEABLE / BLOCKED; 0 unresolved inline threads. Checks: FAILURE 17, SKIPPED 89, SUCCESS 21. Reviews: claude: COMMENTED (2026-11-02T09:31:07Z).
- **Last curated action:** Discovered by the daily author-wide query.
- **Latest GitHub activity:** PR updated 2026-11-02T09:32:05Z; no issue comments.
- **Evidence freshness:** NOT AUDITED. Notes last assessed unrecorded; new comments/reviews still require human triage.
- **Activity triage:** NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes.
- **Evidence / pack-up:** No curated evidence yet; inspect the PR.
- **To Ready:** New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.
- **To merge / dependency:** Resolve current review and required CI; no automatic merge certification.

- Red-check entry points: [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41421313847/job/125197250372); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41421313141/job/125197249169); [call-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41421313782/job/125197249825); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41421313716/job/125197351105); [pr-gate / pr-gate](https://github.com/sgl-project/sglang/actions/runs/41421313832/job/125197356900); remaining checks are in the linked PR and JSON snapshot.

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
| [AMD-AGI/Infera #164](https://github.com/AMD-AGI/Infera/pull/164) | Keep an alternative fast-benchmark script for comparison. | DRAFT | 2026-09-16T04:01:29Z | See active handoff above |
| [AMD-AGI/Infera #167](https://github.com/AMD-AGI/Infera/pull/167) | feat(agent_sys): an agent work system, and the five-stage LLM e2e optimisation flow | MERGED | 2026-10-19T08:50:38Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #208](https://github.com/AMD-AGI/Infera/pull/208) | Create, pack, run and share agent_sys tasks from Claude Code. | MERGED | 2026-11-02T07:29:14Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #218](https://github.com/AMD-AGI/Infera/pull/218) | Avoid SIGPIPE false negatives in Mooncake artifact checks. | MERGED | 2026-10-30T06:56:00Z | Merged October 30; optional rebuild backport, no runtime throughput claim. |
| [AMD-AGI/Infera #220](https://github.com/AMD-AGI/Infera/pull/220) | agent_sys: add simple kernel artifact catalogs and skill | MERGED | 2026-10-29T02:18:53Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #221](https://github.com/AMD-AGI/Infera/pull/221) | Publish six KernelForge families and their historical evidence. | DRAFT | 2026-10-28T10:55:40Z | See active handoff above |
| [AMD-AGI/Infera #223](https://github.com/AMD-AGI/Infera/pull/223) | ci(release): build the Rust router once and bundle it in a Linux x86_64 wheel | MERGED | 2026-10-29T07:41:06Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Infera #228](https://github.com/AMD-AGI/Infera/pull/228) | Permanent Draft tracker for 48k deployment priorities, implementation PRs and experiment references. | DRAFT | 2026-11-02T09:01:30Z | See active handoff above |
| [AMD-AGI/Primus-Turbo #387](https://github.com/AMD-AGI/Primus-Turbo/pull/387) | Dev/yihou/bench | MERGED | 2026-08-24T04:04:02Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Primus-Turbo #391](https://github.com/AMD-AGI/Primus-Turbo/pull/391) | bench: FlashInfer B200 sweep aligned with aiter reference | MERGED | 2026-06-23T09:34:51Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Primus-Turbo #394](https://github.com/AMD-AGI/Primus-Turbo/pull/394) | bench: FlashInfer B200 sweep aligned with aiter reference | MERGED | 2026-06-24T04:11:17Z | Merged: retain provenance; no active merge work |
| [AMD-AGI/Primus-Turbo #400](https://github.com/AMD-AGI/Primus-Turbo/pull/400) | Dev.microbench.b200.gptoss | MERGED | 2026-08-24T04:03:56Z | Merged: retain provenance; no active merge work |
| [kvcache-ai/Mooncake #5005](https://github.com/kvcache-ai/Mooncake/pull/5005) | Offer opt-in HIP copy-stream priority without changing the default. | DRAFT | 2026-11-02T04:28:10Z | See active handoff above |
| [sgl-project/sglang #33059](https://github.com/sgl-project/sglang/pull/33059) | Make real query rows and padded Aiter metadata agree. | READY | 2026-11-02T09:48:02Z | See active handoff above |
| [sgl-project/sglang #33643](https://github.com/sgl-project/sglang/pull/33643) | Speculative penalty token accounting | CLOSED | 2026-10-27T11:08:13Z | Closed; external #45233 also closed unmerged. Search main/replacements before resuming. |
| [sgl-project/sglang #33968](https://github.com/sgl-project/sglang/pull/33968) | Historical HiCache pointer aliases | CLOSED | 2026-10-27T08:06:24Z | Closed as superseded by merged #35233/#42553. |
| [sgl-project/sglang #33970](https://github.com/sgl-project/sglang/pull/33970) | Historical early KV-send completion | CLOSED | 2026-10-27T08:06:44Z | Closed as superseded by merged #41395; separate producer work #45524 remains. |
| [sgl-project/sglang #33973](https://github.com/sgl-project/sglang/pull/33973) | Historical DSA eager host-sync removal | CLOSED | 2026-10-27T11:15:53Z | Closed old draft; follow replacement #46830. |
| [sgl-project/sglang #45483](https://github.com/sgl-project/sglang/pull/45483) | Historical eager-idle backend initialization | CLOSED | 2026-10-27T07:09:42Z | Closed duplicate; follow external #45156. |
| [sgl-project/sglang #45484](https://github.com/sgl-project/sglang/pull/45484) | Preserve the Aiter is_shuffled weight tag through offload/onload. | READY | 2026-11-02T04:21:05Z | See active handoff above |
| [sgl-project/sglang #46819](https://github.com/sgl-project/sglang/pull/46819) | Cache DCP gather metadata instead of rebuilding it per transfer. | DRAFT | 2026-11-02T04:22:19Z | See active handoff above |
| [sgl-project/sglang #46820](https://github.com/sgl-project/sglang/pull/46820) | Keep a blocked metrics consumer from stopping the scheduler. | READY | 2026-11-02T10:07:30Z | See active handoff above |
| [sgl-project/sglang #46822](https://github.com/sgl-project/sglang/pull/46822) | Skip HIP DSA indexing when an eager DP rank has no tokens. | READY | 2026-11-02T04:21:06Z | See active handoff above |
| [sgl-project/sglang #46824](https://github.com/sgl-project/sglang/pull/46824) | Relayout DCP-sharded draft KV together with target KV. | DRAFT | 2026-11-02T04:21:11Z | See active handoff above |
| [sgl-project/sglang #46826](https://github.com/sgl-project/sglang/pull/46826) | Allow DCP decode behind DCP1 prefill when transport/backend support it. | DRAFT | 2026-11-02T04:23:15Z | See active handoff above |
| [sgl-project/sglang #46828](https://github.com/sgl-project/sglang/pull/46828) | D22: optional BF16 kv_b retention | CLOSED | 2026-11-02T04:17:56Z | Closed after merged #47180 supplied FP8 Q-rep enablement; no isolated current-main benefit established. |
| [sgl-project/sglang #46830](https://github.com/sgl-project/sglang/pull/46830) | Remove rank-divergent host syncs that can deadlock DSA eager execution. | READY | 2026-11-02T10:07:34Z | See active handoff above |
| [sgl-project/sglang #47276](https://github.com/sgl-project/sglang/pull/47276) | Speed up gfx950 fused indexer logits/top-k for MTP verify rows. | READY | 2026-11-02T10:07:40Z | See active handoff above |
| [sgl-project/sglang #47278](https://github.com/sgl-project/sglang/pull/47278) | Speed up gfx950 FP8 sparse MLA decode with a 64-head split-K plan. | READY | 2026-11-02T10:07:44Z | See active handoff above |
| [sgl-project/sglang #47872](https://github.com/sgl-project/sglang/pull/47872) | [Perf] [AMD] dsa: Opt-in single-pass Triton top-k for long-context decode rows | READY | 2026-11-02T09:31:23Z | See active handoff above |
| [sgl-project/sglang #47897](https://github.com/sgl-project/sglang/pull/47897) | [Perf] [AMD] dsa: In-tree gfx950 paged MQA logits for decode / MTP-verify rows | READY | 2026-11-02T09:32:05Z | See active handoff above |

The historical llr37-c104 result is 48,713 total tokens/s/GPU (input + output, eight GPUs), using simulated MTP acceptance. PP overlap and decode custom all-reduce have known real-acceptance correctness failures. This table makes no individual-patch speedup or production-correctness claim.
