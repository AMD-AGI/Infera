# GLM-5.2 on MI355X: what ATOM (Infera PR #209) and InferenceX SGLang+Dynamo do that our SGLang PD stack could try

Date: 2026-10-23. Scope: read-only analysis of code, recipes and published numbers. **Nothing in this report was run on
our stack.** No item below is claimed to be "already optimal" or "not a factor": anything that was not measured on our
stack is listed under §5 *Possible improvements*, with the evidence that motivates it and the measurement that would
decide it.

Path short-hand: SGLang files cited without a prefix inside the tables are relative to [SGL] (`forward_mla_rocm.py` = `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py`; `dsa_indexer.py` = `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`; `parallel_state.py` = `python/sglang/srt/distributed/parallel_state.py`; `memory_pool.py` = `python/sglang/srt/mem_cache/memory_pool.py`; `comm.py` = `python/sglang/srt/layers/dcp/comm.py`; `conn.py` = `python/sglang/srt/disaggregation/mooncake/conn.py`); ATOM files without a prefix are relative to [ATOM].

Evidence tags: **[F]** first-hand (code read at the pinned commit, recipe text, published number); **[I]** inference;
**[U]** unknown (what would settle it is given).

Working notes behind this report (reader outputs, adversarial reviews):
`analysis.md` (ATOM) and `inferencex_sglang_dynamo.md` (Dynamo), both in this directory; `pr209_body.md` /
`pr209_files.tsv` are copies of the PR #209 description and file list. Paths starting with `yihou_ws/` are in the
local workspace `/home/yihou/dev/git.16-19/sglang.pd.pp.dcp/yihou_ws/` (not in this repo).

---

## 0. References

| Alias | Repository / object | Pinned at | Link |
|---|---|---|---|
| **[INF]** | AMD-AGI/Infera, PR #209 "feat(atom): optimize atom pd performance" (merged) | `adbb9dc51f9d1d0116b38eb4dd9f103082b83802` | https://github.com/AMD-AGI/Infera/pull/209 · `https://github.com/AMD-AGI/Infera/blob/adbb9dc5/<path>` |
| **[ATOM]** | ROCm/ATOM (the PR #209 image pins this commit, `rocm/atom-dev:nightly_202610131505`) | `661ef7661eef0d804d2b52d81ed41d80aeaf9fdb` | `https://github.com/ROCm/ATOM/blob/661ef766/<path>` |
| **[AITER]** | ROCm/aiter (pinned by the same image) | `9205e1059` | https://github.com/ROCm/aiter |
| **[IX]** | SemiAnalysisAI/InferenceX | `ae21092e6379f68e7a5891748deea918823d0b72` | `https://github.com/SemiAnalysisAI/InferenceX/blob/ae21092e/<path>` |
| **[IX#4300]** | InferenceX PR #4300, GB300 GLM-5.2 NVFP4 Dynamo+SGLang PP4-prefill disagg (commit `bc6a0446`) | merged 2026-10-20 | https://github.com/SemiAnalysisAI/InferenceX/pull/4300 |
| **[IX#4045]** | InferenceX PR #4045, GLM-5.2 MI355X ATOM PD points | open | https://github.com/SemiAnalysisAI/InferenceX/pull/4045 |
| **[IX#4064]** | InferenceX PR #4064, GLM-5.2 MI355X SGLang DCP4 (results "TBD", needs sglang #44322) | open | https://github.com/SemiAnalysisAI/InferenceX/pull/4064 |
| **[IX#4398]** | InferenceX PR #4398, GB200 PP8-prefill (not yet on the dashboard) | open | https://github.com/SemiAnalysisAI/InferenceX/pull/4398 |
| **[IX-API]** | InferenceX public results API `https://inferencex.semianalysis.com/api/v1/benchmarks?model=GLM-5.2` | dump 2026-10-23 | local `/var/tmp/yihou/ix_api/glm52_latest.dec.json` |
| **[SRT]** | NVIDIA/srt-slurm (InferenceX submodule `inferencex-e2e/utils/srt-slurm`) | `ae3d6877` | `https://github.com/NVIDIA/srt-slurm/blob/ae3d6877/<path>` |
| **[SGL]** | our SGLang, branch `dev/yihou/glm52_dcp_pd` (origin/main `fda84f80b7` + sgl-project/sglang#44322 + our commits; local only) | `ab7a898b77` | `/home/yihou/dev/git.16-19/sglang.pd.pp.dcp/<path>`; patch series in Infera `yihou/glm52.dcp4-pd.packup_20261022/patches/` |
| **[KIT]** | our PD kit (llying v0522 kit + our edits) | — | `yihou_ws/glm52_dcp_pd_20261021/llkit/kit/`, rendered argv `llkit/render/S{1,2}.argv.txt`; pack-up `Infera yihou/glm52.llying-dcp-s1s3.packup_20261022/` |
| **[KD]** | our matched-load eager kernel diff, DPA4 vs DCP4 | — | `yihou_ws/glm52_dcp_pd_20261021/design/kernel_diff_dpa_vs_dcp.md` |
| **[GMU]** | our decode mem-fraction work | — | Infera `yihou/glm52.dcp4-decode-gmu.packup_20261023/` (commit `2ecf34d3`) |

Related upstream SGLang PRs: sgl-project/sglang#44322 (ROCm triton sparse-MLA DCP), #43735 (CUDA SM100 DSA DCP),
#39378 (cited by IX#4300 for one PP4 prefill worker per node).

---

## 1. Baselines being compared

| System | Hardware / GPUs | Topology | MTP / acceptance | Window / client | tok/s/GPU |
|---|---|---|---|---|---|
| **Ours S1** | MI355X, 8 (one node) | P: PP4 TP1 HiCache · D: TP4 + DPA4, EP1, mem 0.92 | EAGLE 5/1/6, simulated AL 3.61 | 1800 s, IX client `918524ff`, C80 | 34,509 (S1m rerun 34,792) |
| **Ours S2** | MI355X, 8 | same P · D: TP4 EP4 DCP4, mem 0.85 | same | same | 25,941; 27,685 at mem 0.88 + allocator fix [GMU] |
| **ATOM (PR #209 kit)** | MI355X, 8 (one node) | P: PP4 TP1 (eager), Mooncake Store CPU tier · D: TP4 + DCP4, MoE TP4, mem 0.94 | MTP K3, forced AL 2.99 | 3600 s, IX CI client `288471b3` | C56 32,925 · C72 40,075 · C96 46,234 · C112 48,154 [F, INF PR body §5] |
| **ATOM (IX#4045)** | MI355X | 1P1D | — | IX sweep | 39,245 sweep; ATOM CI 46.1K at C112 [F, IX#4045 body] |
| **IX SGLang+Dynamo GB300** | GB300 NVFP4, 12 P + 8 D (5 nodes, MNNVL) | 3 × PP4 prefill · 1 × DEP8 decode | EAGLE 5/1/6, simulated AL 3.61 | ~3630 s, aiperf | C216 55,488 · C264 63,647 [F, IX-API, IX#4300] |
| **IX SGLang+Dynamo B300** | B300 NVFP4, 4 | agg TP4 + DP4-attn | same | ~3600 s | C48 42,047 [F, IX-API run 38356572105] |
| **IX SGLang MI355X (agg)** | MI355X, 4 | agg TP4 EP4, HiCache | — | — | C16 18,327 [F, IX-API; recipe `[IX] inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/glm5.2/sglang/mi355x-fp4-mtp/agentic.yaml:138-149`] |

**Metric** [F]: throughput = (Σ input incl. cache-hit + Σ output) / (last end − first start) / all GPUs
(`[IX] inferencex-e2e/infx/results/agentic/request_metrics.py:213-241`). Same definition as ours (check: 1,272,944 / 20 = 63,647).

**Not apples-to-apples** [F]: hardware and precision (GB300 NVFP4 vs MI355X MXFP4), GPU count (20 vs 8), window
(3600 vs 1800 s), client version, concurrency, MTP K and acceptance (ATOM K3 AL 2.99 vs ours K5 AL 3.61; SGLang draws
one simulated AL per batch `[SGL] python/sglang/srt/speculative/spec_utils.py:456-470`, ATOM per token
`[ATOM] atom/model_ops/rejection_sampler.py:238-241`). InferenceX SGLang points use the same three simulate variables
as ours (`[IX] inferencex-e2e/infx/srt_slurm/synthetic_acceptance.py:36-38,213-228`; golden AL file
`[IX] inferencex-e2e/infx/golden_al_distribution/glm5.2_mtp.yaml`). Correctness: ATOM reports GSM8K 0.975 (200 q) with
real MTP; none of the throughput numbers above is a correctness result.

---

## 2. ATOM / Infera PR #209 — what it changes

### 2.1 Patches in the image (`[INF] deploy/docker/patches/atom_nightly/`), published effects [F]

| Patch | Change | Published effect (MI355X, GLM-5.2 decode TP4+DCP4) |
|---|---|---|
| `patch_atom_api_gc_stalls.py` | finished sequences no longer queued on an unread `outputs_queue`; per-request context gauges opt-in | API process spent ~45 % wall time in gen-2 GC; C64 +1.9 %, TTFT 2.85 → 2.04 s |
| `patch_atom_dcp_topk_custom_ag.py` | DCP top-k exchange via aiter IPC all-gather (`dcp_all_gather`, `[ATOM] atom/model_ops/dcp_ops.py:35-80`) instead of pynccl | 1.3 MB/rank 54 → 31 µs, 24 per step |
| `patch_aiter_mla_bmm_configs.py` | per-M Triton configs (M ≤ 512) for the MLA absorbed bmm (q-up N512/K192, v-up N256/K512) | graph 40: q-up 19.1 → 8.4 µs, v-up 15.2 → 8.1 µs, ≈ −1.4 ms/step |
| `patch_aiter_custom_alltoall.py` (1st half) + `patch_atom_dcp_custom_a2a.py` | aiter `alltoall_vec` on the custom-AR registered buffers, graph-capturable; DCP merge uses it | bit-identical to RCCL; 5.3 MB/rank 56 → 34 µs; −1.0…−1.6 ms/step |
| `patch_aiter_glm52_decode_gemm_rows.py` | 27 tuned FP8 GEMM rows for M 144–240 | qkv_a M160 14.9 → 11.5 µs; −0.9 ms/step at B33/B36 |
| `patch_aiter_custom_alltoall.py` (2nd half) + `patch_atom_dcp_custom_a2a_combine.py` | one kernel: read peers' heads, LSE-merge, per-token FP8 quant | 160 rows 44.7 → 38.1 µs/layer; −0.56 ms/step avg |
| check `check_dcp_merge.py` | GPU check of the two kernels vs RCCL | 24/24 on 4 and 8 GPUs |

Total: decode step 42.77 → ~40 ms at B40, ~101K context (first five rows), −0.2…−0.6 ms more from the fused combine.

### 2.2 Router changes (`[INF] infera/router/disagg_protocols/atom_mooncake.py`, `rust/router/src/protocol.rs`) [F]
- ATOM decode reuses the prefill's `prompt_token_ids` instead of re-tokenizing (~0.25 s at 130K tokens): C64 +0.5 %, TTFT p50 −0.09 s.
- Frontend `/metrics` no longer scrapes ATOM workers.

### 2.3 Kit defaults (`[INF] examples/atom_4p4d_glm5.2/{env.sh,engine/worker.sh,engine/up.sh,engine/store.sh,README.md}`) [F]
- Decode: mem 0.94, `--max-num-batched-tokens 2048`, `AITER_REUSE_IDENTICAL_COMM_GROUPS=1` → KV ~160K → ~198.5K blocks (~12.7M tokens over 4 DCP ranks).
- `--enable_prefix_caching` on both legs (`engine/worker.sh:148`); online `ptpc_fp8` quant of non-expert linears (`QUANT_CONFIG`, `worker.sh:19-20,148`).
- `AITER_QUICK_REDUCE_QUANTIZATION=INT4`, `AITER_USE_FLYDSL_MOE_SORTING=1` (`worker.sh:23`); `TOPK_FORCE_PATH=one`; `--timeout-keep-alive 900`.
- Mooncake `MC_NUM_QP_PER_EP=1`, 16 send workers; Mooncake Store CPU tier (2 × 768 GiB owners, async RDMA loads); LMCache alternative.
- Decode graph sizes 1,2,4,8 then every 4 up to 2×CONC; max-num-seqs 2×CONC.
- README: with LMCache (synchronous loads) C56/C72 = 33,392/40,326, C96/C112 drop to 30.7K/23.3K.

### 2.4 ATOM engine design points observed in the code (`[ATOM]` @ 661ef766) [F]
| Area | ATOM | Our SGLang S2 (DCP4) |
|---|---|---|
| Index-K cache under DCP | sharded: non-owned tokens get slot −1 (`atom/model_ops/attentions/aiter_mla.py:2515-2525`, `:1300-1330`); index keys transferred per DCP shard (`atom/kv_transfer/disaggregation/index_staging.py`) | replicated: index buffer sized for all logical tokens on every rank (`[SGL] python/sglang/srt/mem_cache/memory_pool.py:5346-5352`; [GMU] `results/memory_map.md`) |
| Indexer scoring + top-k | local-shard `deepgemm_fp8_paged_mqa_logits`, local top-2048 with scores, all-gather of **fp32 scores only**, `flydsl_dcp_topk_merge` emits owned compacted slots (`atom/model_ops/dcp_ops.py:1085-1121,1147-1307,1284-1289,1315-1316`) | split indexer packs int32 `[2,rows,topk]` (`[SGL] python/sglang/srt/layers/attention/dsa/dsa_dcp_split_indexer_triton.py:62`) via `group.all_gather` (`python/sglang/srt/layers/attention/dsa/dsa_indexer.py:1654`); aiter AG dtype gate accepts fp32/fp16/bf16 only (`[SGL] python/sglang/srt/distributed/parallel_state.py:1312`); merge then `transform_index_page_table_dcp` |
| Indexer prep | `wk`+`weights_proj` one GEMM (`atom/models/deepseek_v2.py:2087,3397-3401`); fused `indexer_qk_rope_quant_and_cache` (`:1702-1722`) | 3 GEMMs + separate prep kernels; fused gfx950 path only ≤ 48 rows [KD] |
| Rope/cat/KV write | `fused_qk_rope_concat_and_cache_mla` under DCP (`atom/model_ops/attention_mla.py:3138-3163`) | fused path gated on `dcp_size == 1` (`[SGL] python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py:447`); [KD] ~+3.0 ms eager |
| Absorb / projections precision | fp8 Triton bmm with QREP (`attention_mla.py:1473-1514`), online ptpc_fp8 linears | `SGLANG_DSA_DCP_KEEP_KV_B_BF16=1` → bf16 `torch.bmm` (`forward_mla_rocm.py:246,316-322`); bf16 GEMMs for q_a/kv_a/q_b/o_proj [KD] |
| DCP merge | project-before-merge (W_V before a2a, 256-wide; `attention_mla.py:826-836,1337-1373`, default on `atom/config.py:1718`) + IPC a2a + fused combine/quant | 512-wide latent a2a + LSE via pynccl/RCCL (`[SGL] python/sglang/srt/layers/dcp/comm.py:474-537`, `parallel_state.py:1208-1216`), no graph buffers passed (`forward_mla_rocm.py:1030-1036`) |
| Attention kernel | aiter asm `mla_decode_fwd`, persistent, fp8 Q/KV, page 1 (`attention_mla.py:2650-2800`) | Triton sparse MLA |
| MoE | TP4, aiter, shared expert fused (`atom/utils/envs.py:265,789`) | EP4 FlyDSL (untuned shape [KD]) |
| CUDA graphs | FULL mode by default (`atom/model_engine/arg_utils.py:335-343`, `model_runner.py:3970-4120`) | full decode graphs, max bs 96 |
| Spec decode | MTP K3 (4 verify rows/request) | EAGLE 5/1/6 (6 verify rows) |
| Decode prefix reuse | consumer probes local prefix cache, skips leading blocks in transfer (`atom/model_engine/scheduler.py:2004-2050`; `atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py:539-562,1518-1528`) | `--disaggregation-decode-enable-radix-cache` exists, default False (`[SGL] python/sglang/srt/arg_groups/fields/disagg.py:85-88`; path `disaggregation/decode.py:840,1477-1490`, `prefill.py:433-444`, DCP alignment `disaggregation/common/utils.py:150-197`); not set in [KIT] |
| KV transfer packing (DCP) | per-shard staging + landing pool (`atom/kv_transfer/disaggregation/mooncake/mla_landing.py:1-40`) | `build_dcp_token_transfer_plan` (`disaggregation/common/utils.py:183-197`), queue sharding by dst ports (`disaggregation/mooncake/conn.py:3112-3117`) |
| Prefill in-flight prefix coalescing | yes (`atom/model_engine/scheduler.py:768-790,2024-2035`) | [U] not checked |

Same on both sides [F]: prefill TP1×PP4 with partition 20,20,20,18 and chunk 8192; FP8 indexer cache (ATOM FP4 only
with `--index-cache-dtype fp4`, `atom/model_ops/sparse_indexer_fp4.py:79`); query replication (SGLang
`--dcp-replicate-q-proj`, default on for a2a, `[SGL] python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py:41`).

---

## 3. InferenceX SGLang + Dynamo — where ~50K comes from

### 3.1 Points [F, IX-API]
| Point | GPUs (P+D) | tok/s/GPU | TTFT | TPOT | GPU / CPU cache hit | decode KV use | Source |
|---|---|---|---|---|---|---|---|
| GB300 3×PP4 + DEP8, C216 | 12+8 | 55,488 | 3.15 s | 10.8 ms | 0.69 / 0.26 | 1.0 | run 39187209967, image `nightly-dev-20261016-a93f16f0`, Dynamo 1.6.0.dev20261016, NIXL |
| GB300 3×PP4 + DEP8, C264 | 12+8 | 63,647 | 4.34 s | 11.1 ms | 0.64 / 0.31 | 1.0 | same |
| GB300 4×PP4 + DEP16, C256 | 16+16 | 44,453 | 2.42 s | 8.8 ms | 0.72 / 0.23 | 0.96 | same |
| B300 agg TP4 DP4-attn, C48 | 4 | 42,047 | 6.41 s | 19.0 ms | 0.52 | n/a | run 38356572105, image `nightly-dev-cu13-20260901` |

Recipes: `[IX] inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/glm5.2/sglang/gb300-fp4/agentx/disagg-variants.yaml`
(base L3-157; `override_3p1d_pp4_dep8_c216` L223-247; `_c264` L249-278); master rows
`[IX] inferencex-e2e/configs/nvidia-master.yaml:7549-7666`; B300 `.../b300-fp4/agentx/agg-tp4-variants.yaml:13-196`.
The C216 → C264 step changes concurrency, prefill chunk (16384 → 8192), prefill mem (0.83 → 0.88), HiCache size
(unset → 185 GB) and decode mem (0.90 → 0.94) at once.

### 3.2 Recipe vs our S1 [F]
| Item | GB300 C264 | Ours S1 ([KIT] `render/S1.argv.txt`) |
|---|---|---|
| P:D GPUs | 12:8, 3 independent PP4 prefill workers (one per node) | 4:4, 1 PP4 prefill |
| Decode parallelism | DEP8: TP8 DP8 EP8 dp-attention, `moe-a2a-backend flashinfer`, `moe-runner-backend flashinfer_trtllm`, across 2 nodes over MNNVL (L267-278) | TP4 + DP4, EP1 (argv L333) |
| Decode per-rank limits | max-running 256 total (32/rank), `cuda-graph-max-bs-decode 32` | max-running 256 (64/rank), graph bs 256 (L343, L372) |
| Decode mem | 0.90 (0.94 at C264) | 0.92 |
| Decode chunk | 64 | 32768 (L341) |
| Decode extras | `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8` (L95; read by PD decode admission `[SGL] python/sglang/srt/disaggregation/decode.py:141,483-498,1540-1543,1984-1990`), `stream-interval 30`, `skip-tokenizer-init`, `disaggregation-decode-extra-slots 0`, `disable-radix-cache` | unset; extra-slots 0; decode radix off |
| Prefill | max-running 16, CUDA graph off, overlap default, policy default, HiCache io `direct`, chunk 8192, mem 0.88, `modelopt_fp4`, `flashinfer_cutedsl` (L54-86, 258-261) | max-running 256, `--disable-overlap-schedule`, `shortest-prefill-first`, HiCache io `kernel` + `page_first`, chunk 8192, mem 0.88 (L141-195) |
| KV transfer | NIXL over MNNVL (`MC_FORCE_MNNVL=1`, `NCCL_MNNVL_ENABLE=1`) | Mooncake (same node) |
| Router | Dynamo frontend(s) (`python3 -m dynamo.frontend`, `[SRT] src/srtctl/frontends/dynamo.py:225`), up to 10 behind nginx with session-header hashing (`[SRT] src/srtctl/core/schema.py:2591,2630-2634`, `src/srtctl/templates/nginx.conf.j2:18-38`), `router-mode kv`, `router-replica-sync`, prefill session TTL 3600 s, decode active-request weight 8192, busy thresholds None (L23-35, 80-83, 120-124) | Infera router: kv-aware (prefill overlap 20, decode 2), session affinity prefill, `--pd-prefill-guard-release completion`, `--decode-input-demand on` ([KIT] `engine/up.sh:46-52`) |
| KV events | GB300 recipe sets none (srtctl passes `--kv-events-config` only if set: `[SRT] src/srtctl/backends/sglang.py:422-427`, `src/srtctl/backends/base.py:140-148`) | prefill on, decode off |
| Host memory | `dram-utilization 0.85`, HiCache 185 GB (C264) | HiCache 250 GB/rank, ratio 1.5 |
| Client sampling | temperature 1, top_p 0.95, thinking on (L136) | per our InferenceX client defaults [U] |

B300 C48 (closest single-node comparator) adds `enable-dp-lm-head`, `enable-prefill-delayer` (60 passes),
max-running 128 (32/rank), graph bs 64, chunk 32768, HiCache 270 GB `page_first_direct`, `bf16-gemm-backend cutedsl`
(`agg-tp4-variants.yaml:184-196`). Both flags exist in [SGL] (`python/sglang/srt/arg_groups/fields/parallel.py:199`,
`schedule.py:228`).

---

## 4. Observed differences that may explain the gap (none measured on our stack)

Grouped; each is an observation with its source. Their individual contribution is **[U]** for all of them.

1. **Hardware / precision**: GB300 NVFP4 with flashinfer/cutedsl kernels vs MI355X MXFP4 (IX); ATOM uses the same MI355X.
2. **Topology**: IX 12:8 P:D over 5 nodes; ATOM and ours 4:4 on one node.
3. **Spec decode**: ATOM K3 AL 2.99 (4 verify rows) vs ours K5 AL 3.61 (6 rows); IX SGLang same as ours.
4. **Decode KV capacity**: ATOM ~12.7M tokens (DCP4, sharded index-K, mem 0.94) vs ours 8.87M (S2 0.85) / 9.46M (0.88).
5. **Decode prefix reuse**: ATOM on (decode hit 92/84/53/25 % at C56–C112); IX GB300 50K points off; ours off.
6. **DCP decode kernels/comm** (ATOM vs S2): top-k payload and transport, fused indexer prep, fused rope/KV write, fp8 absorb/projections, PBM merge + IPC a2a, asm MLA decode, MoE TP vs EP4.
7. **Allocator behaviour**: our decode reserve grows 15–17 GiB above live tensors under load ([GMU]); ATOM / IX behaviour [U].
8. **Prefill scheduling**: IX max-running 16, overlap on, default policy; ours 256, overlap off, shortest-prefill-first. ATOM eager PP4.
9. **CPU tier**: ATOM Store async loads; ours HiCache L2 (blocking behaviour [U]).
10. **Front end**: API-process GC (ATOM fixed), decode tokenization (ATOM reuses ids; IX `skip-tokenizer-init`), `stream-interval 30` (IX).
11. **Protocol**: window, client, concurrency, warm-up.

---

## 5. Possible improvements (all uncertain; ordered by cost to try)

Columns: what to change · where it comes from · status in our stack · how to measure. "Config" = launch flag/env only.

### 5.1 Config-only
| # | Change | Source | Our status | Measure |
|---|---|---|---|---|
| P1 | Decode radix / prefix reuse: `--disaggregation-decode-enable-radix-cache` (S2 first; S1 needs DP-rank-aware routing, warning in `[SGL] python/sglang/srt/arg_groups/pd_disaggregation_hook.py:~106`) | ATOM `--enable_prefix_caching` + consumer skip | flag exists, default off; never run with DSA+DCP+EAGLE | TTFT, transfer bytes, decode `cached_tokens`, tok/s |
| P2 | `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8` on decode | IX GB300 L95 | env read by PD decode admission | admission queue, running, tok/s |
| P3 | `--stream-interval 30` on decode | IX GB300 | supported | TPOT, tok/s |
| P4 | Decode max-running / graph bs: 32 per rank (DPA) or graph sizes every 4 up to 2×CONC (ATOM) | IX, ATOM | supported | step time by bucket, tok/s |
| P5 | Decode chunked-prefill-size 64 | IX GB300 | supported | tok/s |
| P6 | Prefill max-running 16; overlap schedule on; default schedule policy (three separate runs) | IX GB300 | supported | prefill queue, TTFT |
| P7 | HiCache io `direct` / `page_first_direct` | IX GB300 / B300 | supported (`arg_groups/fields/memory.py:128`) | per-stage load vs forward time |
| P8 | `--enable-dp-lm-head`, `--enable-prefill-delayer` | IX B300 C48 | fields exist | tok/s |
| P9 | `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` | ATOM `AITER_QUICK_REDUCE_QUANTIZATION=INT4` | knob exists (`[SGL] python/sglang/srt/distributed/device_communicators/quick_all_reduce.py:175`), unset | sgl-eval accuracy first, then step time |
| P10 | MTP 3/1/4 (K3) vs 5/1/6 at the matching golden AL | ATOM K3 | supported | step time, accepted tokens/s |
| P11 | S2 MoE EP1 (TP4) instead of EP4 | ATOM, S1 | supported | step time |
| P12 | Decode mem-fraction with the allocator fix (0.88 measured 27,685; 0.92 measured 22,540 once) | ATOM 0.94; [GMU] | fix exists as experiment patch | repeat runs for spread; count `empty_cache` calls |
| P13 | Protocol parity: 3600 s window; read our cache-hit, KV-use, output-ratio metrics; rescore one ATOM/IX JSON with our analyzer | IX, ATOM | — | — |

### 5.2 Small code changes
| # | Change | Source | Our code point | Measure |
|---|---|---|---|---|
| P14 | DCP top-k pack: `.view(torch.float32)` before `group.all_gather`, back to int32 after, so aiter IPC all-gather can take it; or gather scores only | ATOM `dcp_ops.py:35-80,1284-1289` | `dsa_indexer.py:1654`, gate `parallel_state.py:1312` (size gate `:1300-1301`) | trace which kernel the gather lands on; step p50 |
| P15 | MLA absorbed-bmm per-M configs (pass a config instead of `None`) | INF `patch_aiter_mla_bmm_configs.py` | `forward_mla_rocm.py:156-160` (q `:231-241`, v `:292-301`; fp8 path only, i.e. S1 or S2 without KEEP_KV_B_BF16) | per-kernel trace |
| P16 | Pass graph-safe buffers to the DCP a2a; route `_all_to_all_single` to aiter `custom_all_to_all` once the aiter patch is in the image | INF a2a patches | `forward_mla_rocm.py:1030-1036`, `parallel_state.py:1208-1216` | merge time per layer |
| P17 | Avoid decode re-tokenization in PD (forward token ids / `skip-tokenizer-init`) | INF router change, IX decode | Infera SGLang protocol posts identical bodies to both legs (`[INF] infera/router/disagg_protocols/sglang_bootstrap.py:26,52-62`) [U] whether on the critical path | decode tokenize time per request |
| P18 | API / tokenizer-manager GC behaviour | INF GC patch | SGLang has gc freeze (`[SGL] python/sglang/srt/entrypoints/http_server.py:2485-2529` (`/freeze_gc` endpoint)), `gc_threshold`, `--gc-warning-threshold-secs` | run with GC warnings on; py-spy |

### 5.3 Larger engine work (DCP4 decode)
| # | Change | Source | Our code point |
|---|---|---|---|
| P19 | Enable the fused rope/cat/KV-write under DCP | ATOM `attention_mla.py:3138-3163` | gate `forward_mla_rocm.py:447` |
| P20 | Fused indexer prep without the 48-row cap; one GEMM for `wk`+`weights_proj` | ATOM `deepseek_v2.py:1702-1722,2087` | [KD] indexer rows; gfx950 fused indexer |
| P21 | Shard index-K across DCP ranks (memory + write work) | ATOM `aiter_mla.py:2515-2525`, `index_staging.py` | `memory_pool.py:5346-5352` |
| P22 | Project-before-merge (256-wide a2a) + fused combine/FP8 quant | ATOM `attention_mla.py:826-836,1302-1373`; INF combine patch | `layers/dcp/comm.py:474-537` |
| P23 | fp8 absorb / projections (drop `KEEP_KV_B_BF16`, online fp8 linears) + tuned GEMM rows | ATOM ptpc_fp8; INF GEMM-rows patch | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:73-86`; why bf16 was chosen: `design/qrep_bf16_kvb.md` (not re-read here) |
| P24 | Fix draft-pool over-charge (3.57 GiB) | [GMU] | `[SGL] python/sglang/srt/model_executor/pool_configurator.py:303-304` |
| P25 | Allocator reserve control as a proper option (threshold, period) | [GMU] | `disaggregation/decode.py` event loops |
| P26 | Prefill CPU tier with asynchronous loads | ATOM Store | HiCache L2 path [U] |
| P27 | DCP KV transfer packing (landing pool) and in-flight prefix coalescing on prefill | ATOM `mla_landing.py`, `scheduler.py:768-790` | `conn.py:1436-1462,3112-3117` |
| P28 | Topology: P:D ratio > 1:1, DEP decode with `mori` a2a on ROCm (`[SGL] python/sglang/srt/layers/moe/utils.py:41`) | IX GB300 | needs multi-node / launch test |

---

## 6. Unknowns and what would settle them
- Per-kernel ATOM decode times at our shapes → an ATOM decode torch trace at B40/~101K, set against [KD].
- Hardware share of the IX gap → same recipe (B300 agg C48) on MI355X.
- IX unset defaults (kv dtype, page size, DSA backend) and the srt-slurm SHA → `server_args` log and `srt-slurm-sha.txt` of run 39187209967.
- Dynamo KV routing without KV events; DP-rank selection inside DEP8 → Dynamo 1.6.0.dev20261016 router source / frontend logs.
- Whether aiter routes our all-reduce sizes to quick-reduce; which MoE kernel ATOM runs → traces.
- Why ATOM CI (46.1K) and the IX sweep (39.2K) differ → IX#4045 run logs.
- Accuracy of fp8 online quant / INT4 AR vs our bf16 → same-harness sgl-eval.
