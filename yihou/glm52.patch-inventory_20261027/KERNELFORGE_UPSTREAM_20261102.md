# KernelForge upstream PRs: source, evidence and maintenance

This November 2, 2026 audit connects four existing SGLang PRs and one unfiled
MoE candidate to the historical GLM-5.2 kernel overlay. Sections 1–5 preserve
source provenance and historical author measurements. Section 6 records current
heads, CI, review actions and the new #47872 GPU regression result; the live
work queue is [PR_HANDOFF.md](PR_HANDOFF.md).
[Infera #228](https://github.com/AMD-AGI/Infera/pull/228) remains permanently Draft
for viewing and tracking; that restriction does not apply to implementation PRs.

## 1. What the frozen 48k deployment actually requires

The frozen llr37 deployment uses the complete
[K01 overlay](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/patches/K01-kf-decode-kernel-overlay.NEWFILES.diff).
Its [site configuration](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-c104-conc.packup_20261027/sites/llr37.site.env)
enables sparse MLA, indexer logits/top-k and MoE, with
`SGLANG_OPT_USE_TOPK_V2=false`. The later fused-indexer alias path is not enabled.
The [frozen reproduction guide](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/PUBLIC_REPRODUCE.md#32-files-added-or-replaced-at-launch)
explains the complete source and image requirements.

| Kernel family / original candidate | Role in exact llr37 | Existing upstream work |
| --- | --- | --- |
| `sparse_mla.best.iter14.64fbeaa2.py` / `kf_sparse_mla.py` | Required historical overlay family | [SGLang #47278](https://github.com/sgl-project/sglang/pull/47278), reworked with BF16 split-K partials and current DCP semantics |
| `indexer.best.iter15.c2cacf5f.py` / `kf_indexer.py`: logits | Required historical overlay family | [SGLang #47897](https://github.com/sgl-project/sglang/pull/47897), in-tree Gluon implementation with a paired-row correctness repair |
| Same indexer candidate: top-k | Required historical overlay family; selected through the old v1 hook | [SGLang #47872](https://github.com/sgl-project/sglang/pull/47872), compact-page-table port with exact underflow/overflow fallback and an explicit opt-in flag |
| `moe_mxfp4.best.iter14.9cd51e60.py` / `kf_moe.py` | Required historical overlay family | No submitted PR; current-baseline benefit was not established; see section 3 |
| `fused_indexer.best.1748c11.py` / `kf_fused_indexer.py` | Present in the source bundle, but the added fused-indexer path is not selected in llr37 | [SGLang #47276](https://github.com/sgl-project/sglang/pull/47276), later scoped optimization; not an extra prerequisite for exact llr37 |

The [llr5–llr7 archive](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-kf-ar-topk.packup_20261024/README.md)
records initial sparse/indexer/MoE/top-k engagement. Its older overlay is
superseded by the complete K01 source for llr37. The later llr42 fused-indexer
experiment is separate and was not retained in the frozen recipe.

These PRs are re-engineered ports, not byte-identical replacements for the full
overlay. Merging all four does not establish that the historical score or
combined-stack correctness transfers to a new image. llr37 uses EAGLE 3/1/4 and
simulated acceptance 2.99; the PR speed experiments use EAGLE 5/1/6, simulated
acceptance 3.61 and single-node TP8 or attention-DP8 serving, not PD/AgentX.
The historical approximately 48k measurements are not correctness-qualified:
the real-acceptance investigation associated garbling with PP communication
overlap and decode custom all-reduce. Do not use that throughput as an isolated
patch-performance claim.

## 2. Evidence and review scope of the four PRs

The evidence below is reported in the source notes and linked PRs. This audit
checked source correspondence without repeating those historical model and
performance experiments. Measurements stay attached to their reported revisions
and setup. The separate new #47872 correctness validation is recorded in section 6.

| PR | Concrete change | Reported evidence | Remaining review focus |
| --- | --- | --- | --- |
| [#47276](https://github.com/sgl-project/sglang/pull/47276) | Share KV reads across six sibling MTP rows; retain main's per-row kernel for larger ungrouped batches; tune short-row dispatch and top-k histogram work | Seven GPU tests; bit-identical logits, top-2048 slot sets and K-write bytes against the baseline. Separate real-acceptance TP8 GSM8K smoke: .953 baseline / .963 earlier PR revision with the same kernels. Simulated-acceptance serving ITL 4.96–5.00 ms versus 5.06–5.10 ms at 120k input, concurrency 8 | Coordinate [#45307](https://github.com/sgl-project/sglang/pull/45307): both change `paged_mqa_logits.cuh` and its launch signature. Rebase whichever lands second. Grouping is specialized to six rows/request; other draft counts use fallback kernels |
| [#47278](https://github.com/sgl-project/sglang/pull/47278) | Use one 64-head block per row and a split-K plan for FP8 sparse MLA; preserve current DCP/LSE behavior and BF16-cache async handling | 22 GPU tests. FP32-reference SNR 37–38 dB with BF16 partials, versus approximately 31.5 dB for KF's FP8 partials on the cited larger shapes. Isolated kernel 2.4–3.2x; real-acceptance GSM8K means .942 baseline / .941 PR across three servers each. Simulated-acceptance 32k serving ITL 8.80–8.84 ms versus 9.97–10.00 ms | Preserve the accuracy-motivated BF16 partials and both reduce branches' LSE stores during rebases. Main's historical low SNR on some shapes and intermittent chunked-prefill watchdog failures were observed in controls; they are not established PR regressions |
| [#47897](https://github.com/sgl-project/sglang/pull/47897) | Add guarded gfx950 in-tree Gluon paged-MQA logits; fix the candidate's paired-row tail omission by making all splits use a consistent common-prefix decision | Compare valid-column logits against Aiter, max absolute difference 7.6e-6 from FP32 summation order; top-k sets agree on the recorded workset. Isolated call 2.67–3.06x at 385k–462k context. Real-acceptance GSM8K means .934 baseline / .927 PR, with overlapping observed ranges. Simulated-acceptance 32k serving ITL approximately 0.9% lower | Keep coverage for plain decode and paired rows with different tables and lengths, not only MTP siblings. Preserve the gfx950/preshuffle/page/head/row gate and Aiter fallback. Overlapping GSM8K ranges do not prove model equivalence or long-context DSA accuracy |
| [#47872](https://github.com/sgl-project/sglang/pull/47872) | Add opt-in Triton top-k on compact page tables; add exact whole-row fallback when the 8192-entry candidate buffer overflows | Reference tests include overflow, short and all-equal rows. Top-k sets agree with v2 on the recorded synthetic and model worksets. About 1.5–1.9x at 400k context with 72+ rows, but slower at 32k. Simulated-acceptance 32k serving ITL 14.11–14.15 ms versus 13.93–13.95 ms, a 1.3% regression. One real-acceptance GSM8K run .953 is a smoke test | Preserve `SGLANG_DSA_TOPK_TRITON=1` opt-in and exact overflow fallback. Boundary ties permit different valid selections across calls. Long-context serving gain was not measured; do not generalize the kernel result to serving throughput |

Short GSM8K prompts often select all tokens because they are below top-k=2048;
these model runs have limited power to validate DSA selection. Numerical kernel
tests and adversarial inputs provide separate evidence. PR3's paired-row repair
and PR4's overflow fallback fix defects in the source KF candidates; they must
not be described as bugs demonstrated in the unmodified upstream baseline.

The early October 29 `analysis_result.md` statement that GPU validation was
needed before opening is superseded by the later recorded tests and filings.
Likewise, the missing-overflow-fallback hazard in PR_NOTES section 6 describes
the original candidate; section 7 records its repair in #47872. Preserve the
remaining tie-order and workload-performance limitations.

## 3. MoE remains unfiled for a concrete reason

The local candidate hooks five Aiter/FlyDSL paths and hard-codes the GLM-5.2
MXFP4 shape family: 257 experts, top-k 9, hidden size 6144, per-1x32 scaling.
The historical 1.105x gain was measured on Aiter `e7d2453`. On the later Aiter
`2bb549` environment, the recorded four-weight-set graph benchmark measured
0.2326 ms for Aiter's own path and 0.2365 ms for both the KF candidate and port.
An engagement probe found that none of the five hooks fired: the stage-2
configuration lacked `kernelName2`, and the sort hook was not called.

The local `yihou/kf-mxfp4-moe-decode-gfx950` branch is therefore not a submitted
PR and should remain an explicit held item. Related external
[ROCm/aiter #6697](https://github.com/ROCm/aiter/pull/6697) concerns folding MoE
sort into GEMM1; inspect its current scope before proposing overlapping work.
A useful next step is to identify whether any relevant current execution path
still needs these hooks. Filing requires an active implementation and isolated
benefit on that baseline. The source notes record the user's earlier preference
for an SGLang-first port; this audit does not silently redirect or submit it.

## 4. Reproducible validation commands and evidence limits

Run each command from its own selected PR checkout with that revision's
dependencies. The GPU commands require an allocated pool node and the
appropriate ROCm/gfx950 environment. These generic commands do not imply that
every suite was rerun during maintenance. Section 6 and its linked evidence bundle
identify the exact #47872 module/test runner that was executed on GPU. The PR
bodies contain historical server/benchmark commands and image/import adaptations.

```bash
# #47276: one gfx950 GPU
PYTHONPATH=python python3 test/registered/amd/test_dsa_gfx950_fused_indexer.py -v

# #47278: ROCm sparse MLA numerical and contract coverage
PYTHONPATH=python python3 -m pytest -q \
  test/registered/kernels/ops/attention/test_triton_sparse_mla_hip.py

# #47897: one gfx950 GPU; Aiter reference and paired-row coverage
PYTHONPATH=python python3 test/registered/amd/test_dsa_gluon_paged_mqa_logits.py -v

# #47872: top-k reference, overflow and ties
PYTHONPATH=python python3 test/registered/amd/test_dsa_triton_topk_transform.py -v
```

On a CPU host, skipped vendor tests do not count as numerical passes. The
existing #47276 launch-selection assertion and three #47278 CPU contracts can
be invoked independently without GPU fixtures; their invocation is recorded
in the [48k rebase audit](FOLLOWUP_20261102_48K_REBASE.md). Repository lint and
CPU checks are useful rebase controls, while existing matching hardware
evidence should retain its source and environment boundaries.

The source notes refer to scripts under `validate/`, including
`validate_pr1.yihou.py`, `validate_pr2.yihou.py`, `bench_pr34.yihou.py`,
`topk_rows_bench.yihou.py`, and `pr5/engage_check.yihou.py`. Later model scripts
retain tags such as `r12_g_pr3_b` and `r14_g_main_1` / `r14_g_pr3_1`.
Their recorded raw-output destination is node-local
`/data/yihou/upstream_pr/e2e/<tag>/`; this document does not claim those raw
results are publicly archived. A source path or command is not a published
result bundle.

## 5. Source-note provenance

These local notes were read without modification. They are named to make this
audit traceable, not offered as public download links. Public implementation
references are the PRs above; public historical artifacts are the immutable
Infera links in section 1.

| Source | SHA256 at this audit |
| --- | --- |
| `/home/yihou/kf-glm52-yihou/upstream_pr_yihou_20261029/PR_NOTES.md` | `96c251b84cdeeafe84773bd22c70071729a8c0fa98799c57ad88a541afb7bce1` |
| `/home/yihou/kf-glm52-yihou/build/patch.upstream.tracking.md` | `4202461595bfc5375de02ecc9a29e0949355eba494688ab790517f046af261c5` |

An independent in-memory reconstruction of K01 confirmed byte equality between
its indexer, MoE and fused-indexer files and the named KF candidates. Its sparse
MLA file differs only by the import-path compatibility fallback between
`kvcache.cache_ops` and `attention.metadata`; the kernel body is unchanged.
This establishes candidate-to-overlay provenance, not equality between those
candidates and the later PR implementations.

## 6. Maintenance results

Audited at 2026-11-02T10:45:43.141452+00:00. All four owned kernel PRs are Ready and rebased onto main `a133231d3140e081424d833caa0187f15d40dc3c`; all four exact-head upstream lint jobs pass. Each branch update retained a local backup and used an exact old-head lease.

| PR | Current head / state | Validation | CI / next action |
|---|---|---|---|
| [#47276](https://github.com/sgl-project/sglang/pull/47276) | `598ba3f41f53` / Ready | Full-tree Ruff, scoped hooks and CPU launch-selection assertions passed; 7 GPU tests skipped; range-diff unchanged | [Lint passed](https://github.com/sgl-project/sglang/actions/runs/41430249327/job/125226720255); [Base attempt 1](https://github.com/sgl-project/sglang/actions/runs/41430249649/job/125226908540) sees Ready; missing maintainer-controlled `run-ci`. |
| [#47278](https://github.com/sgl-project/sglang/pull/47278) | `73ee22beaac2` / Ready | Full-tree Ruff, scoped hooks and 3 CPU contracts passed; 22 GPU tests skipped; touched-source change is only inherited main #47782 architecture lookup | [Lint passed](https://github.com/sgl-project/sglang/actions/runs/41430645537/job/125228065040); [Base attempt 1](https://github.com/sgl-project/sglang/actions/runs/41430646049/job/125228291012) sees Ready; missing maintainer-controlled `run-ci`. |
| [#47897](https://github.com/sgl-project/sglang/pull/47897) | `fcfa508799f1` / Ready | Full-tree Ruff, scoped hooks and 4 CPU tests passed; 4 GPU tests skipped; range-diff unchanged | [Lint passed](https://github.com/sgl-project/sglang/actions/runs/41428923887/job/125222275790); [Base attempt 1](https://github.com/sgl-project/sglang/actions/runs/41428924194/job/125222524471) sees Ready; missing maintainer-controlled `run-ci`. |
| [#47872](https://github.com/sgl-project/sglang/pull/47872) | `1061c298389f` / Ready | Full-tree Ruff, scoped hooks and 32 CPU tests passed; actual-kernel interpreter controls passed; original GPU underflow reproduced and all 5 repaired MI355X GPU tests passed with zero skips | [Lint passed](https://github.com/sgl-project/sglang/actions/runs/41429823944/job/125225314296); [Base attempt 2](https://github.com/sgl-project/sglang/actions/runs/41429823967/job/125228691842) sees Ready; missing maintainer-controlled `run-ci`. |
| MoE candidate, no PR | Held | Current hooks did not engage; no gain on Aiter 2bb549 | Yihou/delegate: establish an active, beneficial current-baseline port before filing. Related external #6697 remains separate. |

#47872 was temporarily Draft after a confirmed sampler-underflow bug. The fix uses the existing exact whole-row fallback when candidates are insufficient or overflow, and moves optional Gluon imports behind the ROCm/opt-in guards. The new guard tests fail on the original implementation and pass after the repair. CPU interpreter and actual MI355X GPU checks reproduce the original 256/2048 result and verify the repaired 2048/2048 result. [Published reproduction and exact-source evidence](validation/kf_47872/README.md) records the image digest, source hashes, run ID, scripts and scope. Its occurrence in the historical llr37 inputs was not assessed.

Ready was restored after hardware validation. `/tag-and-rerun-ci` produced a fresh Base attempt that explicitly sees `Draft: false`; all four Base gates now need the maintainer-controlled `run-ci` label. A successful rerun request or passing lint does not mean functional CI ran.

The GPU check reused the earlier task's verified idle standing pool grant and only one GPU; the temporary duplicate request was released and the task container exited. No other allocation or container was changed. The remaining three rebases use CPU/lint/source controls and their existing hardware evidence; skipped GPU suites are not new numerical passes. No new serving-performance result is claimed.

Related external work remains separate: #45307 has main conflicts and overlaps the optional fused indexer; ROCm/aiter #6697 has main conflicts and an unanswered design objection. Their authors own those changes. Our MoE port remains held for lack of current-baseline benefit. Infera #228 remains permanently Draft for tracking.
