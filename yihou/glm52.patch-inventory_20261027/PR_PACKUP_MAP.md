# PR ↔ patch ↔ experiment pack-up map

Status checked: **2026-10-28T05:02:45.143954+00:00**. Entry point: [PUBLIC_REPRODUCE.md](PUBLIC_REPRODUCE.md).
Full immutable PR heads/bases are in [PR_SNAPSHOT.json](repro/PR_SNAPSHOT.json).
**11 owned active core PRs: 5 ready for review, 6 draft.** Ready means available
for review, not merged or validated as a combined model-serving stack.

`BAKED` means the frozen fdbafb60 image already carries a historical implementation;
`MOUNTED` means additional source/data must be mounted; `BUILD-ONLY` cannot explain
runtime throughput; `NOT IN LLR37` must not be added merely to accumulate fixes.
An experiment archive links evidence and source, not proof that a current PR head
was the binary used by that experiment. All historical serving-throughput numbers
used simulated MTP acceptance; only isolated measurements support patch-level claims.

## Owned core PRs

| Patch and purpose | PR, state, head | Original experiment / source / port | Role in llr37 | Remaining work / evidence boundary |
|---|---|---|---|---|
| S01: metrics backpressure must not freeze scheduling | [#46820](https://github.com/sgl-project/sglang/pull/46820) **READY FOR REVIEW**; `82176e7028` | [c44 archive](../glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/README.md); c44a→c44b; [archived diff](patches/S01-kv-events-metrics-send-noblock.v0521.diff); [prepared port diff](reviews/S01-rebased.diff) | BAKED: historical metrics patch in fdbafb60. No extra llr37 mount. | Current real-ZMQ CPU tests pass; maintainer CI/review. No isolated model speedup claim. |
| Earlier DSA query-row repair | [#33059](https://github.com/sgl-project/sglang/pull/33059) **DRAFT**; `d0062f8bce` | [llr image/kit](../glm52.llrepro-pp4-dpa4.packup_20261024/README.md); original MI355X GLM DP8/EP8 report; DCP patch 0001; [prepared port diff](reviews/PR33059-rebased.diff) | BAKED: historical indexer-row script, not identical to the current-main PR. | Map historical coverage to HaiShaw’s DeepSeek-V3.2 DP/stage-c request. New Qwen3 PD CI accuracy failure remains unattributed. |
| Former #33973: remove rank-divergent host syncs | [#46830](https://github.com/sgl-project/sglang/pull/46830) **READY FOR REVIEW**; `8bd9fe8c3b` | [llr image/kit](../glm52.llrepro-pp4-dpa4.packup_20261024/README.md); original MI355X deadlock/MI300X masking evidence; DCP patch 0002; [prepared port diff](reviews/PR33973-rebased.diff) | BAKED: dsa_dp_sync.diff. Closed #33973 is replaced by this PR. | Production delta matches original; current CPU/consumer audit passed. Maintainer CI/review. |
| S10: cache DCP pack metadata | [#46819](https://github.com/sgl-project/sglang/pull/46819) **DRAFT**; `aed9bc6953` | [DCP pack archive](../glm52.dcpx-pack-fix-best.packup_20261023/README.md); dcpx8c/9/10; [archived diff](patches/S10-dcp-pack-cached-metadata-priority-stream.ov93034eba91.diff); [prepared port diff](reviews/S10-rebased.diff) | NOT IN LLR37: DPA4 decode, no DCP relayout. | Current GPU byte tests and cache microbench pass. Narrow to cache or isolate pinned-index/priority/PP-ordering behavior; current PR retains producer wait, archived experiment removed it. |
| D27: skip eager-IDLE DSA indexer on HIP | [#46822](https://github.com/sgl-project/sglang/pull/46822) **READY FOR REVIEW**; `11262e1199` | [DCP archive](../glm52.dcp4-pd.packup_20261022/README.md); R16–R18 and separate R19–R25 fault chain; [archived diff](patches/D-dcp-branch/0027-Fix-dsa-Skip-the-indexer-on-an-eager-idle-DP-rank-on.patch); [prepared port diff](reviews/D27-rebased.diff) | NOT ADDED TO LLR37: separate DCP fault repair. | Historical production delta and current CPU tests checked; maintainer CI/review. |
| D10 + D13/D16/D29: transport sharded draft KV | [#46824](https://github.com/sgl-project/sglang/pull/46824) **DRAFT**; `bd4c8e5ef6` | [DCP archive](../glm52.dcp4-pd.packup_20261022/README.md); pdBe-r1 math/passkey and host-byte tests; [archived diff](patches/D-dcp-branch/0010-PD-dcp-Relayout-a-DCP-sharded-DSA-draft-KV-like-the-.patch); [prepared port diff](reviews/D10-rebased.diff) | NOT IN LLR37: separate DCP1-to-DCP-N recipe. | Author confirmed #44322 layout predicate. Port to its latest head; resolve or narrow known DP2×DCP2 graph gap. |
| D11: permit constrained ROCm PD DCP decode | [#46826](https://github.com/sgl-project/sglang/pull/46826) **DRAFT**; `d25dbd8bbe` | [DCP archive](../glm52.dcp4-pd.packup_20261022/README.md); pdBe-r1; [archived diff](patches/D-dcp-branch/0011-PD-dsa-Allow-a-ROCm-DCP-decode-behind-a-DCP1-prefill.patch); [prepared port diff](reviews/D11-rebased.diff) | NOT IN LLR37: no DCP decode guard needed. | Needs working #44322 backend and #46824 relayout before enablement; check actual defaults/unsupported combinations. |
| D22: retain BF16 kv_b for replicated Q | [#46828](https://github.com/sgl-project/sglang/pull/46828) **DRAFT**; `a15ee201da` | [DCP archive](../glm52.dcp4-pd.packup_20261022/README.md); isolated benchrun T10; [archived diff](patches/D-dcp-branch/0022-AMD-dsa-Add-SGLANG_DSA_DCP_KEEP_KV_B_BF16-to-keep-GL.patch); [prepared port diff](reviews/D22-rebased.diff) | NOT IN LLR37: DCP-only option, default off. | Align final #44322 Q-projection/dtype contract; reuse T10 and supplement actual numerical/memory gaps. |
| S06 = D21: preserve is_shuffled during weight onload | [#45484](https://github.com/sgl-project/sglang/pull/45484) **READY FOR REVIEW**; `43dec26b03` | [OffloaderV2 archive](../glm52.v0521-offloaderv2.packup_20261015/README.md); ofs0c/ofg3c/ofg2c each 23/23; [archived diff](patches/S06-offloader-keep-is_shuffled.v0521.diff); [prepared port diff](reviews/S06-rebased.diff) | NOT IN LLR37: no OffloaderV2 weight offload; HiCache is different. | Current-head MI355X regression and CPU controls passed; maintainer CI/review. No historical unpatched e2e control. |
| I05c: prevent artifact-check SIGPIPE false negatives | [#218](https://github.com/AMD-AGI/Infera/pull/218) **READY FOR REVIEW**; `5cee57e093` | [c44 archive](../glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/README.md); image-recipe artifacts; [archived diff](patches/I05c-docker-mc4340-pipefail-check-fix.patch); [prepared port diff](reviews/I05c-ready.diff) | BUILD-ONLY: optional rebuild backport; not a runtime llr37 optimization. | 11 CPU cases pass; full image rebuild not performed. Remaining integration CI/review. |
| M02: opt-in HIP copy-stream priority | [#5005](https://github.com/kvcache-ai/Mooncake/pull/5005) **DRAFT**; `13df6a68da` | [c44 archive](../glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/README.md); R039/R043 peer-copy microbench; [archived diff](patches/M02-mooncake-hip-copy-stream-priority.diff); [prepared port diff](reviews/M02-rebased.diff) | NOT IN LLR37: different Mooncake build; llr37 selects cross-host RDMA and disables HIP transport. | HIP build/4 CTests passed. Human submitter review; Python 3.10 wheel missing-module CI failure; check #4953 interaction. No serving speedup claim. |

## Existing upstream work: no competing PR

Ownership remains with the original authors. Our contribution is the linked
historical evidence/coordination, not a new implementation claim.

| Patch family / purpose | Current upstream disposition | Pack-up evidence and original patch | Role in llr37 / next action |
|---|---|---|---|
| S03/S11: distribute transfer queues | [#42163](https://github.com/sgl-project/sglang/pull/42163) **DRAFT**; `294113a01f` | [g4tune](../glm52.v0521-g4tune-xfer.packup_20261020/README.md); [DCP pack archive](../glm52.dcpx-pack-fix-best.packup_20261023/README.md); [archived diff](patches/S11-mooncake-dcp-hashed-room-shard.ov93034eba91.diff) | NOT IN LLR37. YAMY1234 owns the draft; historical mixed-shard data includes a TPOT regression. [our evidence](https://github.com/sgl-project/sglang/pull/42163#issuecomment-6440117565) |
| S05: initialize the active idle backend | [#45156](https://github.com/sgl-project/sglang/pull/45156) **READY FOR REVIEW**; `96903930a0`; our #45483 CLOSED duplicate | [MAF archive](../glm52.v0521-idxshare-maf-fix.packup_20261021/README.md); idxf1a/b/c; [archived diff](patches/S05-eager-idle-init-active-attn-backend.v0521.diff) | No extra llr37 mount. Follow WQW-Git/review; no exact-head hardware claim. |
| S07: validate prefill_complete with effective attention DP | [#45337](https://github.com/sgl-project/sglang/pull/45337) **READY FOR REVIEW**; `164718dbfa` | [pc48/pc48b](../glm52.v0521-prefill-complete.packup_20261014/README.md); [archived diff](patches/S07-pd-prefill-complete-attn-dp-validation.v0521.diff) | NOT IN LLR37: router uses completion, not prefill_complete. Follow nvjullin; this is the historical “on_complete” item. |
| D17: PP producer event before early KV send | [#45524](https://github.com/sgl-project/sglang/pull/45524) **READY FOR REVIEW**; `376701038e`; related [#43299](https://github.com/sgl-project/sglang/pull/43299) **READY FOR REVIEW**; `b5d0769169` | [DCP archive](../glm52.dcp4-pd.packup_20261022/README.md); [archived diff](patches/D-dcp-branch/0017-PD-pp-Gate-non-final-prefill-chunk-sends-on-the-forw.patch) | Separate current PP/non-CUDA work. The frozen image carries its own non-final-chunk patch plus upstream #41395; do not equate revisions. |
| D14/15/18/24/30/31: ROCm DCP indexer/backend | [#44322](https://github.com/sgl-project/sglang/pull/44322) **READY FOR REVIEW**; `b8a0a1ac70`; #42618 CLOSED replacement history | [DCP archive](../glm52.dcp4-pd.packup_20261022/README.md); T15/T16 isolated long-context timing, with short-context regressions; D-dcp-branch patches 0014/0015/0018/0024/0030/0031 | NOT IN LLR37. EricKing626 confirmed #44322 is the replacement and draft-sharding policy is unchanged. Coordinate final port. |
| D26: misleading idle DCP warning | [#44322](https://github.com/sgl-project/sglang/pull/44322) **READY FOR REVIEW**; `b8a0a1ac70` | [DCP archive](../glm52.dcp4-pd.packup_20261022/README.md); [archived diff](patches/D-dcp-branch/0026-Fix-dsa-Do-not-warn-on-an-idle-DP-rank-s-DCP-prepare.patch) | NOT IN LLR37; no standalone PR. Earlier fold-in request was on closed #42618; recheck applicability. |
| D23: batch-major sparse output padding | [#46424](https://github.com/sgl-project/sglang/pull/46424) **MERGED**; `e536a1144f`; #44262 CLOSED superseded | [DCP archive](../glm52.dcp4-pd.packup_20261022/README.md); mixrun R13; [archived diff](patches/D-dcp-branch/0023-Fix-attention-Pad-a-batch-major-padded-extend-output.patch) | No new PR; current main covers it. NOT the baked dsa_page_table_rows image patch. |
| D28: copy host length lists before graph padding | [#45796](https://github.com/sgl-project/sglang/pull/45796) **MERGED**; `e122674f23` | [DCP archive](../glm52.dcp4-pd.packup_20261022/README.md); R24/R25; [archived diff](patches/D-dcp-branch/0028-Fix-forward-batch-Pad-extend_seq_lens_cpu-into-a-new.patch) | No new PR; merged upstream. No extra llr37 mount. |
| D19 release: synchronize deferred-KV release | [#45453](https://github.com/sgl-project/sglang/pull/45453) **MERGED**; `2647c3761e` | [DCP archive](../glm52.dcp4-pd.packup_20261022/README.md); [archived diff](patches/D-dcp-branch/0019-PD-decode-Release-deferred-KV-and-skip-transfer-poll.patch) | Release hunk covered upstream; residual empty-queue polling guard has no established current reproducer. NOT IN LLR37. |
| M01: HIP IPC suballocation relocation | [#4340](https://github.com/kvcache-ai/Mooncake/pull/4340) **READY FOR REVIEW**; `9b4b049d9c` | [c44 archive](../glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/README.md); R018/R019; [archived diff](patches/M01-mooncake-pr4340-port-to-faae8dd4.diff) | NOT IN LLR37 fdbafb60 build. salexspb owns the fix; the separate v0521-mc4340/destpin images carried our port. Historical production interior-count was zero. |
| Earlier HiCache pointer alias patch | [#35233](https://github.com/sgl-project/sglang/pull/35233) **MERGED**; `c29e6054f5`; [#42553](https://github.com/sgl-project/sglang/pull/42553) **MERGED**; `b2de502825`; our #33968 CLOSED | [early integration archive](../glm52-mtp-garbled-decode-rootcause.packup_20260918/README.md) | Base already uses upstream host-pointer handling. Do not reapply the old host-allocation workaround. |
| Earlier early-send completion patch | [#41395](https://github.com/sgl-project/sglang/pull/41395) **MERGED**; `9fc5f30621`; our #33970 CLOSED | [llr image/kit](../glm52.llrepro-pp4-dpa4.packup_20261024/README.md); baked producer patch documented in PUBLIC_REPRODUCE.md | Consumer fix baked in base; distinct producer/PP follow-ups remain. Do not reopen the superseded broad patch. |
| Speculative penalty accounting | [#45233](https://github.com/sgl-project/sglang/pull/45233) **CLOSED, unmerged**; `612b93311b`; our #33643 CLOSED | [real-acceptance archive](../glm52-agentx-c40-realacc.packup_20260918/README.md); [coverage requirements](reviews/PR45233-penalty-coordination.md) | NOT an active upstream fix: author closed the stale external draft on Oct 27. Recheck main/replacements before reopening or filing. |

## Required llr37 artifacts without a ready PR

| Item | Public source and experiment | Why no PR / reproduction action |
|---|---|---|
| S13: non-first PP rank hidden size | [llr image/kit](../glm52.llrepro-pp4-dpa4.packup_20261024/README.md); [archived diff](patches/S13-deepseek-v2-pp-hidden-size.ab7a898b77.patch) | **MOUNTED.** Use the archived kit source. llying’s earlier implementation requires attribution agreement before a new PR; #34465 is closed unmerged. |
| K01: KF decode kernel overlay | [archived diff](patches/K01-kf-decode-kernel-overlay.NEWFILES.diff); [llr5–llr7 evidence](../glm52.llrepro-kf-ar-topk.packup_20261024/README.md); c104 site | **MOUNTED.** Full source is public in the diff; source-only reconstruction is supplied. Numerical/integration qualification and clean kernel integration are unfinished; no PR. |
| A01: two Aiter GEMM CSVs | [llr32/llr35](../glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/README.md); [archived diff](patches/A01-aiter-glm52-bf16-gemm-tuned-csv.NEWFILES.diff) | **MOUNTED.** Use both exact CSVs from that archive; no PR pending correctness and current-Aiter retuning. |
| I07: role environment/multi-mount/custom-AR hooks | [c104 scripts/site](../glm52.llrepro-c104-conc.packup_20261027/README.md); [archived diff](patches/I07-kit-2p1d-role-env-mount-customar-hooks.diff) | **LAUNCH TOOLING.** Copy c104 worker/common onto the complete base kit. This is example/benchmark maintenance, not an Infera core feature. |

The image also inherits llying's page-table alignment, staged HiCache write-back,
Responses argument reconciliation and two Mooncake RDMA patches. Their precise
frozen paths are listed in PUBLIC_REPRODUCE.md; none is silently attributed to
yihou or represented as covered by a superficially similar PR.

## Issues, excluded experiments and retired bundles

| Item | State / archive | Reason / next action |
|---|---|---|
| S02 absorb-BMM config | [issue #46829](https://github.com/sgl-project/sglang/issues/46829); [c44 archive](../glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/README.md) | Workaround is BAKED into the frozen image; issue-only upstreaming. Current MI355X compiler failure reproduced; isolate root cause. |
| S12 allocator reserve growth | [issue #46831](https://github.com/sgl-project/sglang/issues/46831); [S2e/S2k/S2n/S2m](../glm52.dcp4-decode-gmu.packup_20261023/README.md) | NOT IN LLR37. Current-main root cause/reproduction pending; no periodic-empty_cache PR. |
| S03 layer-group parallelism / M03 batch HIP kernel | [c44 archive](../glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/README.md); R034/R035 | No demonstrated layer-parallel gain; M03 never built. Neither is part of llr37. |
| S04/S08/S09/S14 | [inventory](INDEX.md); S14 trial llr34 in the GEMM pack-up | Explicitly excluded experiments, regressions or unsupported accuracy claims. llr37 uses FP8 draft flag, not S14 MXFP4. |
| D12/D25 and T01–T05/T07 | [DCP archive](../glm52.dcp4-pd.packup_20261022/README.md); [inventory](INDEX.md) | A/B controls/probes/logging, not independent production fixes. |
| I01/I02/I03/I06 | [inventory](INDEX.md); earlier same-node/benchmark pack-ups | Old example/bench adaptations; do not count as Infera core fixes or apply wholesale to llr37. |
| I04 router affinity port | [inventory](INDEX.md) | Primarily llying-owned behavior; no independent current-core yihou fix established. Frozen Rust router comes from fdbafb60. |
| I05a/b/d v0.5.21 image bundles | [#210](https://github.com/AMD-AGI/Infera/pull/210) **MERGED**; `8859b247ce` | Retired in favor of v0.5.22 lineage; #218 is a separate build-check fix. No old recipe PR needed. |
| T06 fast-bench comparison | [#164](https://github.com/AMD-AGI/Infera/pull/164) **DRAFT**; `4676ec9087` | Comparison-only tooling outside 11 core PRs. Not the llr37 AgentX harness. |
| Earlier Infera integration | [#58](https://github.com/AMD-AGI/Infera/pull/58) **MERGED**; `c91db7601d` (replaces #34); [#79](https://github.com/AMD-AGI/Infera/pull/79) **MERGED**; `9380f07310` (replaces #59) | Historical downstream integration, not proof all embedded patches are upstream. |
| Earlier Infera examples/docs | #84/#94 MERGED examples; #87/#95 CLOSED documentation | Historical maintenance; use this map and frozen llr37 inputs for reproduction. |

## Updating this map

Refresh PR state and exact head/base, then update PR_SNAPSHOT.json, this table and
PR_STATUS.md together. Preserve original experiment IDs and file hashes. A newer
PR with a narrower synchronization policy or changed backend is a new validation
target; traceable old hardware evidence still counts where the production behavior
is equivalent. Do not make fresh GPU reruns a blanket requirement.

All PR bodies and port diffs under `reviews/` are dated preparation artifacts;
the current PR page and this timestamped map supersede their older draft prose.
Runtime environments, downloaded CI logs and local worktrees are intentionally
not published. Public PR comments contain the shared test commands/results.
