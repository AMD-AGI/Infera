# PR priority for the historical 47k-class deployment

The **47.9k stack** is [llr35/llr35r at C96](../glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/README.md):
47,843 / 47,950 total tokens/s/GPU, rounded mean **47,897**. The subsequently
requested **llr37-c104** archive uses the same patch stack with concurrency 104:
**48,713.28739 total tokens/s/GPU** (48,310.33559 input + 402.95180 output), across
eight MI355X GPUs. The priority and required-source lists below apply to both.

The [final C104 archive](../glm52.llrepro-c104-tail.packup_20261028/README.md)
records three baseline runs, 48,713 / 48,103 / 47,795, mean **48,204**. Its later
llr51 changes decode memory fraction to 0.96 and reports 48,806 once, without a
repeat. That configuration candidate does not introduce another required PR or
replace the requested frozen llr37 reference.

This run used **simulated MTP acceptance 2.99**. Real-acceptance investigations
found garbling associated with PP communication overlap and decode custom
all-reduce. Reproducing its throughput is separate from qualifying correctness.
Do not attribute the whole-stack number to any individual patch.

For the separate stock-image publication path, see [InferenceX first version versus 48k](INFERENCEX_V1_VS_48K.md). The shared owned unmerged engine PRs are #33059/#46830/#46820; #21194 already covers the GLM PP construction issue on main. **November 2 correction:** issue #46829 has external candidate [#46918](https://github.com/sgl-project/sglang/pull/46918), opened October 28; the previous no-fix-PR status missed it. The exact candidate now has [MI355X validation](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md): the original M=256 compiler abort reproduces on three seeds, while cap=2 and config=None pass all 12 capped/workaround cases with bitwise-identical outputs; M=128/129/257 controls also pass. This covers the scoped Q-absorb reproducer, not full-model accuracy or a serving speedup. Maintainer CI/review/merge and image integration remain.

Use [PR_HANDOFF.md](PR_HANDOFF.md) for live states, exact heads, CI, review and
assigned work; [PUBLIC_REPRODUCE.md](PUBLIC_REPRODUCE.md) for source pins and
commands; [PR_PACKUP_MAP.md](PR_PACKUP_MAP.md) for the wider experiment history.
The [KernelForge provenance and maintenance map](KERNELFORGE_UPSTREAM_20261102.md)
connects all four kernel PRs and the unfiled MoE candidate to the frozen overlay.
The frozen image and current-main PRs are different source revisions. Applying
all open PRs on top of the historical image is not the reproduction procedure.

## Priority PRs

| Priority | PR / purpose | Why it matters to this deployment | Handoff action |
|---|---|---|---|
| **P0** | SGLang [#33059](https://github.com/sgl-project/sglang/pull/33059): align real and padded Aiter query rows | Historical equivalent is **baked** as `patch_dsa_indexer_hip_dp_padded_rows.py`. Required behavior in the frozen recipe; current PR is a different main-based port. | Ready at `795488cd`; preserve the user decision. [Author MI355X DP/MTP results](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447) now cover the padded-row failure; tested/current DSA and test blobs match. [Latest lint passed](https://github.com/sgl-project/sglang/actions/runs/41423254991/job/125203622542). Follow up native MI300 coverage and required CI/review; the current Base gate is account cooldown, not Draft. |
| **P0** | SGLang [#46830](https://github.com/sgl-project/sglang/pull/46830), replacing closed #33973: remove divergent host syncs | Historical implementation is **baked** as `dsa_dp_sync.diff`; prevents the DSA eager path from hanging. | Already Ready at the handoff audit; prioritize maintainer CI and review. |
| **P0** | SGLang [#46820](https://github.com/sgl-project/sglang/pull/46820): nonblocking KV metrics | Historical implementation is **baked**; prevents an auxiliary metrics queue from stopping scheduling. | Already Ready; prioritize maintainer CI/review. Socket contract has CPU coverage; no GPU needed to qualify it. |
| **P0, external author** | SGLang [#45524](https://github.com/sgl-project/sglang/pull/45524): wait for PP producer before early KV send | Frozen image has a **related historical producer-event patch**, plus merged consumer fix #41395. Current PR is not certified byte-equivalent. | Follow eaglewangao-us's existing PR and contribute evidence; preserve the frozen producer patch for reproduction. No duplicate. |
| **P1, optional for llr37** | SGLang [#47276](https://github.com/sgl-project/sglang/pull/47276): gfx950 fused indexer/top-k | A reworked K01-family optimization. The added fused-indexer path is **off in llr37**; its later llr42 activation was not retained. | Track CI/review, coordinate #45307 overlap, and bind final run IDs to source/pack-ups. Do not enable the later fused-indexer alias path in the frozen llr37 recipe. |
| **P1, required source family** | SGLang [#47278](https://github.com/sgl-project/sglang/pull/47278): gfx950 sparse MLA split-K | Sparse MLA is enabled in the frozen K01 overlay. This current-main port uses BF16 split partials and is not byte-identical to that candidate. | Prioritize CI/review and exact source/evidence mapping; qualify a new combined stack before claiming the historical score on current main. |
| **P1, required source family** | SGLang [#47897](https://github.com/sgl-project/sglang/pull/47897): gfx950 Gluon paged-MQA logits | Indexer logits are enabled in the frozen overlay. This reworked in-tree port also repairs the original candidate's paired-row tail omission. | Track CI/review, paired rows with unequal tables/lengths, dispatch gates and fallback coverage. Historical short-context GSM8K is a smoke test, not proof of long-context DSA accuracy. |
| **P1, required source family** | SGLang [#47872](https://github.com/sgl-project/sglang/pull/47872): opt-in Triton top-k | Indexer top-k is enabled through the old v1 overlay hook in llr37. This compact-page-table port is separate and intentionally opt-in because the measured 32k case regresses. | Follow the live handoff for candidate-underflow repair and validation; require exactly `min(length, 2048)` valid selections as well as overflow correctness. Preserve opt-in behavior and review tie semantics. Ready/Draft state follows the current evidence, not this static mapping. |
| **P1, evidence only** | Infera [#221](https://github.com/AMD-AGI/Infera/pull/221): KernelForge catalog | Helps another team inspect sources and experiments; **not required at runtime**. Redacted derivatives are not identical to historical hardware-tested bytes. | Retarget after merged #220, clarify fixture availability and derivative validation. Keep archival evidence distinct from a qualified kernel release. |

Priority P0 means preserve and upstream behavior present in the historical
recipe. It does not imply each patch has an isolated measured throughput gain,
or that an open PR must merge before the already-patched frozen image can run.

## Changes after the 47.9k baseline: no additional required PR

Rechecked November 2 against the original site files, late-run ledger and code
diffs. The 48k deployment retains the required patch families listed above.

| Run / result (total tokens/s/GPU) | Change from its baseline | PR disposition |
|---|---|---|
| llr37 / llr37r / llr37r2: 48,713 / 48,103 / 47,795; mean 48,204 | llr35 concurrency 96 to 104 | Existing configuration; no new code patch or additional PR. |
| llr51: 48,806, unreplicated | llr37 decode `--mem-fraction-static` 0.95 to 0.96 | The site-file diff contains only this effective setting change and its comment. No additional PR; repeat before selecting it over the replicated baseline. |
| llr41: 48,027; llr50: 48,010 | Existing local-control-broadcast flag / INT4 quick-reduce environment option | Neutral trials; not new required patches. llr50 was not carried forward. |
| llr42: 47,674 | [Overlay module alias and double-patch guard](../glm52.llrepro-garblefree-sweep.packup_20261028/kfov_fused/module_path_alias.diff) activate the historical fused indexer | Not kept after the slower end-to-end result. No dedicated owned alias PR; later #47276 is a reworked current-main kernel port, not proof this historical trial belongs in the best deployment. |
| llr52: 47,402 | [P18 DP-controller patch](../glm52.llrepro-c104-tail.packup_20261028/dpc_poll/dpc_poll.diff): wait up to 1 s for new tokenizer input after draining requests | CPU busy polling fell, but no serving improvement was established and the change was not carried. No dedicated owned PR. Any future CPU-efficiency proposal needs current-main overlap, wake-up/watchdog/latency tests and its own evidence. |

These late trials are archived in the [sweep](../glm52.llrepro-garblefree-sweep.packup_20261028/README.md)
and [final tail](../glm52.llrepro-c104-tail.packup_20261028/README.md).
The later kernel PRs #47276/#47278/#47897/#47872 are scoped upstream work, not
additional requirements introduced by the 47.9k-to-48k configuration changes.
The sparse-MLA/logits/top-k families were already present in the overlay;
the added fused-indexer path was not selected. None of these historical
simulated-acceptance scores establishes a standalone patch speedup.

## Required inputs that are not open PRs

These must be handed over alongside the priority PRs. A list of PR URLs alone
cannot reconstruct the deployment.

| Item | Exact public artifact / pack-up | Status / responsible next action |
|---|---|---|
| **S13: PP non-first-rank hidden size** | [Base kit model source](../glm52.llrepro-pp4-dpa4.packup_20261024/kit/sglang_2p1d_glm5.2/patches/deepseek_v2.py), mounted on prefill | Preserve the mount for the frozen image. Third-party [#21194](https://github.com/sgl-project/sglang/pull/21194) merged October 29 and covers the GLM-5.2 PP construction failure on main; verify an image carrying it. No new duplicate filing. Original author can retire alternative [#47184](https://github.com/sgl-project/sglang/pull/47184) after image validation; retain historical attribution. |
| **Full K01 historical overlay** | [Complete NEWFILES diff](patches/K01-kf-decode-kernel-overlay.NEWFILES.diff); [llr5–llr7 evidence](../glm52.llrepro-kf-ar-topk.packup_20261024/README.md) | Required source reconstruction/mount. #47278/#47897/#47872 port the sparse-MLA/logits/top-k families; #47276 is an optional fused-indexer port. None is a byte-identical substitute for the full overlay or a qualified combined release. |
| **K01 MoE: historical candidate, no current port PR** | `kf_moe.py` in the complete K01 diff above | Keep the historical source for llr37. The proposed current port remains held: on Aiter `2bb549` none of its five hooks engaged, and candidate/port 0.2365 ms did not beat native 0.2326 ms. The older 1.105x result used `e7d2453`; it does not justify filing an inert current port. Recheck active paths and related [Aiter #6697](https://github.com/ROCm/aiter/pull/6697) before proposing new work. |
| **A01: both Aiter GEMM CSVs** | [GEMM pack-up](../glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/README.md): 75-row DPA/K3 and 49-row PTPC CSVs | Required historical mounts. No Aiter PR; correctness and current-build retuning remain. The second CSV alone did not establish a speedup. |
| **I07: worker/common hooks and site config** | [C104 pack-up](../glm52.llrepro-c104-conc.packup_20261027/README.md), `scripts/{worker,common}.sh`, `sites/llr37.site.env` | Required role-specific environments, multiple mounts and launch options. Example/benchmark tooling; not an Infera core bugfix/feature. |
| **S02 absorb-BMM workaround** | Frozen image `patch_mla_absorb_bmm_default_config.py`; [issue #46829](https://github.com/sgl-project/sglang/issues/46829); external [#46918](https://github.com/sgl-project/sglang/pull/46918) | Preserve the frozen workaround. We did not submit its blanket-disable diff. `yu-depend`'s candidate is Open/non-draft at `9763ef56bf57d03f8af4f9adc9eebb717cd0d9b1`, proposing a `waves_per_eu` cap of 2 and fallback controls. The exact candidate now has [MI355X validation](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md): the original M=256 compiler abort reproduces on three seeds, while cap=2 and config=None pass all 12 capped/workaround cases with bitwise-identical outputs; M=128/129/257 controls also pass. This covers the scoped Q-absorb reproducer, not full-model accuracy or a serving speedup. Maintainer CI/review/merge and image integration remain. |
| **Inherited page-table, HiCache and Responses patches** | [Frozen Dockerfile and patches](https://github.com/AMD-AGI/Infera/tree/fdbafb6065dae6d1c53936c7d2cf1f5022a514d0/deploy/docker) | Preserve exact sources. Page-table rows are not #33059 or D23; HiCache staged write-back is not OffloaderV2 #45484. These are not silently attributed to yihou. |
| **Inherited Mooncake RDMA rail/ACK-timeout patches** | Same frozen image: `rdma_dest_local_rail.diff`, `rdma_ack_timeout.diff`; `MC_IB_TIMEOUT=18` | Preserve llying's historical patches and attribution. Related external drafts are [#4540](https://github.com/kvcache-ai/Mooncake/pull/4540) and [#4541](https://github.com/kvcache-ai/Mooncake/pull/4541). Fabric-dependent; omitted from the InferenceX first-version recipe. Neither is HIP stream-priority #5005 or HIP IPC #4340. |
| **Integrated Infera image / wrapper / Rust router** | Frozen Infera `fdbafb6065dae6d1c53936c7d2cf1f5022a514d0`, official SGLang v0.5.22 base, pinned Mooncake source | Build from the public pinned recipe. `infera-sglang:v0522-fdbafb60` is a historical local tag, not a public pull URL. Merged #210 is image history; merged #218 is an optional build-check backport, not runtime performance. |

The base already contains the relevant merged HiCache/early-send fixes
(#35233/#42553/#41395 lineage). Do not reopen superseded #33968/#33970, or use a
newer arbitrary image without auditing which frozen modifications it retains.

## Other open work does not gate this reproduction

| PR family | Why it is separate |
|---|---|
| #46819, #46824, #46826; merged #44322/#47180; closed #46828 | DCP gather/transport/backend/Q-rep work. llr37 uses **DPA4 with DCP1**, not the separate DCP decode recipe. |
| #45484 | OffloaderV2 weights are not offloaded in llr37; this is different from HiCache. |
| #46822 | Separate eager-idle repair established in the DCP fault chain; not an additional llr37 mount. |
| External #42163 | Transfer queue sharding was tested in other pack-ups; it was not selected in llr37. |
| External #45337 | Historical `on_complete` item is the `prefill_complete` validation path; llr37 uses completion. |
| Mooncake #5005 / #4340 / #4953 | HIP transport work; llr37 is cross-host RDMA with HIP transport disabled. |
| Infera #164 / #208 and other account projects | Comparison or general tooling, not the frozen AgentX serving recipe. |

Suggested handoff order: preserve #33059 as Ready, follow up native MI300
validation, and complete required CI/review for #33059/#46830/#46820. For #33059,
[Base's latest gate](https://github.com/sgl-project/sglang/actions/runs/41423254776/job/125204140539)
already sees Ready and the existing `run-ci` label; its 120-minute account
cooldown nominally ends November 2 at 11:07:56 UTC if no newer counted run
intervenes. Expiry does not automatically trigger CI. Track #45524, then progress
#47278/#47897/#47872 according to their current review and correctness work;
track optional #47276 separately and preserve the unfiled MoE disposition.
Deliver the exact K01/A01/S13 artifacts alongside that work. Keep the original
source pins intact; validate correctness and throughput separately for any new
combined image.
