# SYNTHESIS: DeepSeek-V4.1-Flash on MI355X, vLLM, PD disaggregation for InferenceX AgentX

Date: 2026-11-19. Phase 1 (research only, no GPU runs). Inputs: `research/{model_arch,inferencex,vllm,sglang,colleagues,pd_method}.md`
including their adversarial Verification sections (corrections applied below; where a dimension file and its
verification disagree, the verification wins).

Evidence tags (kept from the dimension files): **[FH: file:line | API row | PR#]** first-hand code/file/API/PR read;
**[2H]** second-hand (PR author's local run, colleague claim, recipe comment for another SKU); **[INF]** our arithmetic or
inference, unverified. Paths: `V` = `source/vllm` (vLLM main 7a62ca0666, 2026-11-19); `IX` = `source/InferenceX`
(db3fc796, 2026-11-18); `M` = `/perf_apps/chaox/models/DeepSeek-V4.1-Flash`; `RCP` = `V/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`.
All AgentX numbers: metric `tput_per_gpu` (defined in section 3.2), dataset `semianalysis_cc_traces_weka_062126`, FP4 checkpoint
(MXFP4 routed experts, FP8 dense), DSpark 5 drafts with synthetic acceptance length 3.51, 3600 s, unless stated.

---

## 0. Bottom line (decisions rest on these)

1. **The bar on MI355X is vLLM colocated, not PD: 427,451 tok/s/GPU** (2 GPUs, TP1xDP2 attention + TP MoE, fp8_e4m3 KV,
   UMBP host-DRAM KV offload, c384, p90 interactivity 15.8; InferenceX CI 2026-11-18, PR #5036) [FH: API row 448242].
   +50% on this bar = ~641k tok/s/GPU [INF]. Whether the +50% target is measured against this bar or against our own
   reproduction is an open question for the user (section 6, Q1).
2. **No vLLM-to-vLLM PD run of DSv4.1 exists on any AMD GPU** (published, recipe, open PR, or colleague NFS) [FH: searches in
   inferencex.md s8, vllm.md s7, colleagues.md s10]. All high-throughput PD results are NVIDIA Dynamo (GB300 SGLang 1.07M,
   GB300 vLLM 1.01M, B300 vLLM 828k; B300 PD best = 1.32x B300 colocated best) [FH: API rows].
3. **On stock vLLM main, PD for DSv4.1 on ROCm is not runnable with MoRIIO** (rejects the compressor `CircularBufferManager`
   group) [FH: V/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:650-681]; the fix vllm#68513 is an
   open draft tested only on MI325X xGMI. MooncakeConnector and NixlConnector have DSv4.1-specific code on main but no ROCm
   DSv4.1 run was found.
4. **Any PD connector turns off the AITER Gluon sparse-MLA fast path for DSv4.1 on ROCm** (UMBP with default key_namespace is
   the only exception) [FH: V/vllm/models/deepseek_v41/amd/rocm.py:95-109, 597, 624-626; kv_connector/v1/base.py:236-244;
   umbp/connector.py:110-117]. The cost of this has not been measured anywhere. It must be measured before PD is judged.
5. **The wire is not the expected bottleneck**: per-token KV to move is 1.63 KB analytic (vLLM ROCm, MXFP4 indexer) vs 2.47 KB
   engine-reported slope (gap unexplained); a 173k-token request moves 0.29-0.43 GB, ~20-30x less than GLM-5.2 [FH + INF:
   pd_method.md s3]. openai-cluster NIC inventory is unknown and must be measured.
6. **The workload is >99% prefix-cached input**: tokens scored are mostly cache hits (GPU hit 0.88-0.98); output is <1% of the
   score [FH: API rows]. The CI best point's gain came from prefix-cache capacity (UMBP host offload raised c384 by ~2x in the
   author's local runs) [2H: PR #5036]. Any PD design must preserve prefix-cache capacity and affinity on the prefill side.
7. Phase 2 order: reproduce the colocated bar, measure what binds at the knee and the cost of the connector-forced kernel
   path, then bring up 1P1D for correctness, then size P:D (section 5).

---

## 1. State of the art (DSv4.1-Flash AgentX, as of 2026-11-19)

**MI355X.** vLLM colocated leads: DP2+UMBP c384 427,451 (p90i 15.8), c256 401,896 (23.7), DP2 c128 no offload 271,329 (45.1),
image `vllm/vllm-openai-rocm:nightly-60bd6dd2edd1...`, 2026-11-18 [FH: API rows 448240-448242, PR #5036]; the superseded TP2
c64 monokernel point was 186,109 (p90i 62.7, `nightly-4fab5a07`, 2026-11-09) [FH: as-of 2026-11-15 snapshot, row 447802].
ATOM colocated TP2 DEP2 DPA c256 389,350 (p90i 37.8, 2026-11-13) [FH: row 448099]. SGLang colocated TP2 DP2-attn c64 181,052
(p90i 73.3, `v0.5.23-rocm10-mi35x-20261111`, 2026-11-17) [FH: row 448180]; unpublished author-local SGLang DP4+EP4 c256
229,660 @ 78.4 (#5034) and DP2 c128 239.8k @ 50.5 (#5096) [2H]. The only MI355X PD row is TileRT (vLLM TP4 prefill + TileRT TP8
decode over Mooncake), c1, 2,829 tok/s/GPU on 12 GPUs [FH: row 447684, PR #4756]. Leaderboard at interactivity tier 125:
MI355X 150,079 (SGLang) [FH: rankings CSV; interpolation method not verified]. Local colleague bests: SGLang TP2+DPA c64
182,689 (2026-10-26); vLLM TP2 c32 110.1-110.5k (2026-11-04); official CI vLLM TP2 c32 107,989, reproduced 107,583 by chaox
[FH: colleagues.md s8]. Author-local vLLM DEP8 c256 182.7k (InferenceX #5056, open) vs vllm#67377 DEP8 c256 26,714 - a ~7x
gap neither PR explains [2H]; do not use either until re-measured.

**NVIDIA.** PD (all Dynamo, many DEP2 prefill workers + one wide DEP decode, prefill host-DRAM KV offload, concurrency in the
thousands): GB300 SGLang 1,071,944 (9P DEP2 + D DEP16, 34 GPUs, c7480, p90i 105.6, 11-13); GB300 vLLM 1,011,514 (12P DEP2 + D
DEP16, 40 GPUs, c9888, 10-20); B300 vLLM 828,209 (5P DEP2 + D DEP8, 18 GPUs, c3584, p90i 83.8, 11-17); vr200 vLLM up to 2,993,171
(run hosted in NVIDIA/nv-perf-sa-review; official status not checked). Colocated: GB300 vLLM TP4 DPA c1024 863,068; B300 vLLM
DEP8 c1280 625,966 [FH: API rows, inferencex.md s1-2]. Tier-125 ranking: GB300 1,048,780 > B300 726,160 > GB200 372,362 >
B200 353,905 > MI355X 150,079 [FH].

| HW | engine | mode | best tok/s/GPU | point | p90i | GPUs | date | source |
|---|---|---|---:|---|---:|---:|---|---|
| MI355X | vLLM | colocated | 427,451 | TP1xDP2 attn, TP MoE, UMBP, c384 | 15.8 | 2 | 11-18 | row 448242, #5036 |
| MI355X | ATOM | colocated | 389,350 | TP2 DEP2 DPA c256 | 37.8 | 2 | 11-13 | row 448099 |
| MI355X | SGLang | colocated | 181,052 | TP2 DP2-attn c64 | 73.3 | 2 | 11-17 | row 448180 |
| MI355X | TileRT (vLLM P) | PD | 2,829 | 1P TP4 + 1D TP8, c1 | 1333 | 12 | 11-07 | row 447684 |
| GB300 | Dynamo SGLang | PD | 1,071,944 | 9P DEP2 + D DEP16, c7480 | 105.6 | 34 | 11-13 | row 448095 |
| GB300 | Dynamo vLLM | PD | 1,011,514 | 12P DEP2 + D DEP16, c9888 | 75.5 | 40 | 10-20 | row 445572 |
| B300 | Dynamo vLLM | PD | 828,209 | 5P DEP2 + D DEP8, c3584 | 83.8 | 18 | 11-17 | row 448173 |
| B300 | vLLM | colocated | 625,966 | DEP8 c1280 | 35.9 | 8 | 11-10 | row 447890 |

---

## 2. Model-structure facts that matter for serving

| Item | Fact | Evidence |
|---|---|---|
| Layers | 40 = 20-layer causal encoder + 20-layer decoder (CED); + 3 DSpark draft layers (SWA only, 128 experts top-3) | [FH: CFG; TR p.7-9] |
| Attention | every layer: own Q + 128-token SWA ring; layers 2-19 CSA2 ratio 2, 20-39 ratio 1; compressed KV owned only by 2/8/14/20; top-512 indexer (layer 20 builds a 2048x8 candidate pool, 24/28/32/36 re-index in it) | [FH: CFG kv_source_layer_ids, MP:614-621, 498-611] |
| MoE | 384 routed top-6 + 1 shared, FP4 E2M1 experts | [FH: CFG] |
| Active params | ~8B/token prefill (encoder only on all tokens; decoder on last 128), ~16B/token decode | [FH: TR p.1,7,9; shape count 7.9B/16.1B INF] |
| Weights | 510.3 GB total = Engram 202.8 GB + routed FP4 288.8 GB + rest; excl. Engram+vision 285.5 GiB -> does not fit one GPU at util 0.9 (259.2 GiB) | [FH: tensors.json; INF] |
| Measured TP2 weights | 145.15 GiB/GPU with Engram on host; 31.88 GiB non-torch; KV 66.44 GiB = 28,905,651 tokens | [FH: chaox conc32 server.log:188,1144-1146,1394] |
| Engram | 2 lookup tables (layers 1, 14); vLLM default `cpu_offload: true` (pinned host, UVA); 12.7 KB read/token | [FH: V/vllm/config/engram.py:44; INF] |
| KV, official format | 890 B/token growing (2.5 entries x (288 + 68) B) | [FH: README; INF reproduces] |
| KV, vLLM ROCm | record 584 B (`fp8_ds_mla`, default) or 512 B (`fp8_e4m3`, #66036); FP4 record unavailable on ROCm | [FH: attention.py:197-210, 612-624; amd/rocm.py:538-545] |
| Indexer K | 68 B MXFP4 only when `indexer_kv_dtype=mxfp4` (recipe sets it); engine default resolves to FP8 132 B | [FH: config/attention.py:102; mla/indexer.py:67-76; RCP:290-300] |
| Cell per token | 1,630 B (fp8_ds_mla + MXFP4 idx), 1,450 (fp8_e4m3 + MXFP4), 1,790 / 1,610 with FP8 idx | [INF from FH] |
| Engine slope | 2,468 B/token-equivalent from the TP2 log (older image, fp8_ds_mla, `--no-swa-bounded-replay`); 838 B gap to analytic unexplained | [FH log; INF] |
| Bounded state | SWA ring 43 layers x 128 slots (~3-4 MB/request, block-32 rounding not quantified); compressor fp32 ring (`CircularBufferManager`, not prefix-cacheable) | [FH: sparse_swa.py:120-147; compressor.py:250-270] |
| TP and KV | KV replicated on every TP rank (`max_tp_shards=1`); TP adds KV capacity only by freeing weight memory; DP-attention gives each rank its own pool | [FH: attention.py:1309; kv_cache_interface.py:458-460] |
| Bounded replay | `swa_bounded_replay` default True, ROCm CDNA3/4 only; off if the PP stage with layer 20 does not hold all later layers, SP, PCP>1, ubatching, Engram at/after 20, drafter window>128 | [FH: cache.py:381; amd/model.py:1128-1151; attention.py:637-650] |
| PP | ROCm path refuses PP cuts inside a kv-sharing group -> only PP2 at layer 20 (NVIDIA path supports in-group cuts) | [FH: attention.py:553-563, 686-699; nvidia/pipeline.py:44-120] |
| Decode compute | ~32.3 GFLOP GEMM + 3.2 GFLOP attention + 1.6/2.6/4.6 GFLOP indexer per token at 50k/100k/200k | [INF] |
| Per-request PD bytes | fp8_ds_mla + MXFP4: 85 / 166 / 329 MB at ISL 50k/100k/200k (FP8 idx: ~182 MB at 100k); fp8_e4m3: 75/148/293 MB | [INF] |

Capacity note: the 75M/117M/147M-token decode bounds in model_arch.md and the ~320-440 request TP4 estimate in pd_method.md use
inputs that are not MI355X measurements or omit non-torch memory; both were marked unverified by verification [FH: verification
tables]. Real values need startup logs (M2 below).

---

## 3. Recipes to reproduce first (ranked), and the harness contract

### 3.1 Ranked candidate recipes (all vLLM, all colocated; no PD recipe exists to reproduce)

Common env (all CI vLLM MI355X runs) [FH: reproduce_448242.json]:
`AITER_TRITON_LOG_LEVEL=ERROR OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_USE_RUST_FRONTEND=1`.
Common flags: `--attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' --block-size 128
--engram-config '{"cpu_offload":true}' --gpu-memory-utilization 0.9 --language-model-only --max-cudagraph-capture-size 1024
--max-model-len 1048576 --max-num-batched-tokens 16384 --moe-backend aiter --reasoning-parser deepseek_v41
--tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice --disable-uvicorn-access-log
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","enable_adaptive_verification":false,"synthetic_acceptance_length":3.51}'`.
Full command lines: inferencex.md s4a/4b.

| Rank | Recipe | Image | Topology + specific flags | Expected (source) | Why this rank |
|---|---|---|---|---|---|
| R1 | DP2 c128, no offload | `vllm/vllm-openai-rocm:nightly-60bd6dd2edd175a1c55fd3488f871e4c3d2ef27d` | 2 GPUs, `--tensor-parallel-size 1 --data-parallel-size 2 --kv-cache-dtype fp8_e4m3 --prefill-schedule-interval 11`, no monokernel; `vllm-router 0.1.14 --intra-node-data-parallel-size 2 --policy consistent_hash` | 271,329, p90i 45.1 [FH: row 448240] | Same image/shape as the bar without the UMBP service: isolates base engine |
| R2 | DP2 + UMBP c384 (and c256) | same | R1 + UMBP standalone server (`UMBP_DRAM_CAPACITY=449000000000`, `/dev/shm` at `/umbp-host`) + `--kv-transfer-config '{"kv_connector":"UMBPStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"mode":"standalone","endpoint":"unix:///umbp-host/umbp/standalone.grpc.sock"}}'` | 427,451 c384 / 401,896 c256 [FH: rows 448242/448241] | The bar to beat; needs 449 GB host DRAM per 2-GPU instance |
| R3 | TP2 c64 monokernel | `...:nightly-4fab5a07475822c6eaa971dceb2e2220d0446808` | `--tensor-parallel-size 2 --monokernel`, default fp8_ds_mla KV | 186,109, p90i 62.7 [FH: row 447802, 2026-11-09] | Candidate decode-side shape (monokernel needs TP, rejects DP) |
| R4 | TP2 c32 monokernel, `max-num-seqs 8, max-num-batched-tokens 8192` | `...:nightly-rocm100-7d0b4e57` (as CI job 114309765517) | args in `/perf_apps/chaox/DeepSeek_V4.1_Flash/conc32/baseline/launch_server.sh` (`--no-swa-bounded-replay`) | 107,989 CI / 107,583 local repro [FH: colleagues.md s8] | Only point already reproduced on our cluster: harness sanity check and differential reference |
| R5 (later) | DEP8 (TP1xDP8 + EP8 MegaMoE over MoRI), c256 | per InferenceX #5056 (open) | Engram in HBM, fp8_e4m3 | 182.7k author-local [2H]; vllm#67377 reports 26,714 for a similar shape [2H] | Wide-decode candidate; numbers conflict, needs our own measurement |

PD connector bring-up template (different model): InferenceX #5090 (open), MiniMax-M3.1-Flash, MI355X, vLLM 1P TP4 + 2D TP4 over
MooncakeConnector, vllm-router 0.1.16, needs vllm-project/recipes#1262 and vllm#67857; author-local +11-28% over single node at
equal p90 interactivity [2H]. Use for launcher/router wiring only.

### 3.2 AgentX harness contract to use

| Item | Value | Evidence |
|---|---|---|
| Client | `agentx-harness==1.0.14`, `agentx profile` (AIPerf preset); record the pin (colleague runs used 1.0.12) | [FH: IX docs/agentx-standalone.md; colleagues.md s7] |
| Dataset | `--public-dataset semianalysis_cc_traces_weka_062126` (HF `semianalysisai/cc-traces-weka-062126` revision `23f152f6...`), uncapped (dsv41flash not in CAPPED_FAMILIES) | [FH: IX infx/bench/agentic/traces.py:7-18; pd_method.md s2.1] |
| Duration | 3600 s per point (`validation.py:21`); 1200 s "fast" mode only for screening, never for comparison | [FH] |
| Warmup | preset: 10 requests per lane; the count actually used in the 11-18 CI run is not verified (read its aiperf.log) | [FH doc; gap] |
| Seed / restart | seed 42; fresh server per concurrency point; fresh output dir | [FH] |
| Metric | `total_tput_tps = (total_input incl. cached prefix + total_output) / duration` [IX request_metrics.py:250], divided by (prefill GPUs + decode GPUs) [IX __init__.py:132, 237-241] | [FH] |
| GPU denominator gotcha | for vLLM DP layouts the `TP` env must equal the server's total GPUs or per-GPU is wrong | [2H: Muse agentx_harness.md s3.9] |
| Synthetic acceptance | `"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51` in `--speculative-config`, `enable_adaptive_verification:false`; golden thinking_on table 1.81/2.43/2.90/3.26/3.51 for 1-5 drafts; PD: set on decode (and any draft-carrying role) | [FH: golden AL yaml; synthetic_acceptance.py:289-322] |
| Verify AL after run | AL = 1 + d(`vllm:spec_decode_num_accepted_tokens`)/d(`vllm:spec_decode_num_drafts`) | [2H: Muse] |
| Server metrics | `--server-metrics <engine>/metrics` with `AIPERF_REQUIRED_SERVER_METRIC_PREFIX='vllm:'`; KV capacity only from the `GPU KV cache size` log line | [FH + 2H] |
| Router | `vllm-router --policy consistent_hash --intra-node-data-parallel-size N --request-timeout-secs 14400 --disable-retries` | [FH: reproduce_448242.json] |

---

## 4. PD feasibility for DSv4.1 on vLLM ROCm

### 4.1 First-principles pros (mechanisms; magnitudes unmeasured)
- **Transfer is small**: 0.29-0.43 GB/request at the served mean ISL 173k; at +50% on 16 GPUs the deployment moves 4.2-6.3 GB/s,
  ~1.4-2.1% of 8 crusoe ionic rails (38.3 GB/s each measured); openai NICs unknown [INF on FH inputs: pd_method.md s3.2].
- **Decode can pull only the uncached suffix**: vLLM D side computes `remote_tokens - num_computed_tokens` and fetches only
  unhashed blocks when its prefix cache is on [FH: mooncake_connector.py:1180-1250]. SGLang PD decode resends the whole context.
- **Interference removal**: colocated runs chunked prefill in decode steps; cold first turns reach ~0.99M tokens (new-token p99
  41k) [FH: trace stats]. At the TP2 c32 anchor rank-0 device time was decode 78% / prefill 22% [2H: chaox README].
- **Independent shapes**: prefill can use DP/large chunks and host prefix offload; decode can run TP+monokernel (<=48 rows) or
  DP+EP at larger batch with decode-only graphs. CED makes prefill ~half the per-token GEMM of decode (8B vs 16B active) [FH: TR].

### 4.2 Cons and costs
- **Kernel path loss**: any non-UMBP connector disables the AITER sparse-MLA fast path and `extra_cache_nan_free`
  (the latter also needs gfx950 + fp8_ds_mla, so it is off with fp8_e4m3 regardless) [FH: rocm.py:80-109]; open mega-attn
  prefill #65146 also excludes KV connectors.
- **Prefix capacity is what wins today**: the bar's lift came from UMBP host offload on a colocated server. A PD prefill tier
  must keep that. `MultiConnector` keeps the fast path only if all children are UMBP [FH: multi_connector.py:274-280], so
  Mooncake/MoRIIO + UMBP on the prefill side loses the fast path.
- **Weight replication**: each instance needs >=2 GPUs (285.5 GiB weights); min PD = 4 GPUs, each extra instance re-pays
  ~145 GiB/GPU at TP2 [FH log].
- **Closed-loop TTFT coupling**: PD adds handshake + transfer + queueing to TTFT, which directly lowers qps per lane [2H: handbook].
- **Monokernel rejects DP** [FH: mono_decode.py:316-317]; a DP+EP decode loses it; a TP decode is capped at 48 rows for mono.
- **PCP excludes KV transfer** on ROCm [FH: amd/model.py:197-216]; PP only 20/20 on ROCm.
- **Bounded replay at P->D boundary**: D loads N-1 tokens and recomputes the last [FH: mooncake_connector.py:1107-1114]; the
  correctness interplay with DSpark draft state is not checked.
- **Muse lesson**: a KV connector changed the DSpark draft attention path and broke fp8 draft KV (workaround draft
  `kv_cache_dtype auto`) [2H: Muse working_process.md].

### 4.3 Connector choice (ranked for bring-up)
| Rank | Connector | Status for DSv4.1 on ROCm | Needed |
|---|---|---|---|
| 1 | MooncakeConnector | main handles CircularBuffer + SWA clip (`cdiv(window,block)+1`) + bounded replay [FH: mooncake_connector.py:1075-1114, 2430-2445]; no ROCm DSv4.1 run found; vLLM nightly's PyPI Mooncake sends cross-node writes over HIP IPC and its ionic provider lacks kernel ABI 4 [2H: TileRT recipe yaml:19-20]; precedent: vLLM PD on MI355X for MiniMax (#5090) | custom multi-protocol Mooncake build (recipes#1262 / vllm#67857 path) |
| 2 | MoRIIOConnector | main rejects DSv4.1 [FH: moriio_connector.py:650-681] | cherry-pick vllm#68513 (draft; P was TP1xDP2 -> D TP4 monokernel, MI325X xGMI READ, accuracy TODO) + likely #68621, #60783, #68072, #65563 |
| 3 | NixlConnector | DSv4.1 "CSA-linear" handling, ROCm UCX wheel stage in `docker/Dockerfile.rocm:36-137`; pull mode rejects CSA-linear under PP>1 and hetero TP [FH: nixl/base_worker.py:734-800, 995-1008, 1224-1232] | whether the nightly-rocm100 image ships NIXL: not checked |
| - | UMBPStoreConnector | offload only (not PD); keeps fast path | as prefill-side prefix tier, only if not combined with a PD connector (see 4.2) |

### 4.4 Blockers and unknowns that decide feasibility
- B1: MoRIIO rejection (above) - code-level blocker on main.
- B2: fast-path loss under connectors - perf blocker of unknown size.
- B3: Mooncake ROCm transport build - environment blocker (custom build).
- U1: openai-node RDMA inventory (rails, link rate, GPU-NIC NUMA).
- U2: real KV pool sizes and the 2,468 vs 1,630 B/token gap.
- U3: what binds at the colocated knee (prefix capacity, decode step, prefill interference).

---

## 5. Phase-2 plan (ordered; each step has a hypothesis and a discriminating result)

Machines: openai `smci355` MI355X nodes via pool-manager `machines-and-clusters-b1` (start only on "granted"). Model read-only
from `/perf_apps/chaox/models/DeepSeek-V4.1-Flash` (copy at `/perf_apps/data/models/DeepSeek-V4.1-Flash`). Work dirs
`/perf_apps/yihou/dsv41_vllm_pd_20261119`, node-local `/data/yihou/dsv41_vllm_pd_20261119`, containers `yihou-dsv41-*`,
one AITER JIT dir per server. Needs: E0-E4 one node (up to 4 x 2-GPU servers in parallel if host DRAM allows UMBP 449 GB each;
Muse saw host OOM above ~1 TiB per node on its cluster [2H]); E5-E6 one node then two nodes; E7-E8 two to four nodes.

| # | Experiment | Hypothesis | Discriminating result |
|---|---|---|---|
| E0 | Node facts: host DRAM, `ibv_devices`, link rate, rail isolation, GPU<->NIC NUMA, per-rail byte-verified probe (M6) | openai node can host UMBP and has RDMA rails comparable to crusoe | DRAM >= 449 GB x instances; per-rail GB/s number |
| E1 | Reproduce R4 (TP2 c32) then R1 (DP2 c128) at 3600 s | our harness matches CI | within ~2-3% of 107,989 / 271,329; else debug harness before anything else |
| E2 | Reproduce R2 (DP2+UMBP c256, c384) | the 427k bar is reproducible on openai nodes | within noise of 401,896 / 427,451; record GPU/CPU hit, KV usage |
| E3 | Colocated sweep around the knee (DP2 with/without UMBP: c128-c544; TP2/TP4: c32-c128) with per-point prefill vs decode device time, TTFT/TPOT, running count, KV usage, hit (M1) | knee is set by prefix-cache capacity, not decode step | if score tracks GPU+CPU hit and KV usage saturates -> capacity binds; if TPOT/step grows while hit is flat -> compute/interference binds |
| E4 | Kernel-path cost: R1 c128 with `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=0` (emulates the connector gate) [INF: same function, not confirmed equivalent], plus a colocated run with a non-UMBP connector in kv_both | connector-forced path costs X% | measured X; if X is large, PD needs a fix (override `only_restores_canonical_kv` safely) before it can compete |
| E5 | Startup KV dump on every candidate leg (M2): `Available KV cache memory`, `GPU KV cache size`, per-group specs | analytic 1,630 B/token holds once SWA/ring/draft groups are separated | explain the 2,468 vs 1,630 gap; real decode capacity per shape |
| E6 | Decode-only step vs batch (8-256 rows) and context (16k-512k), TP2/TP4 monokernel vs DP+EP (M3); prefill-only new-tok/s at cached prefix 64k/173k/550k, chunk 8k-32k (M4) | P:D by GPU is prefill-heavy (~1.5:1 [INF]) | measured req/s per prefill GPU and per decode GPU -> predicted PD score at E3's knee concurrency |
| E7 | 1P1D bring-up: MooncakeConnector first, MoRIIO (+#68513) second; same-node then cross-node; correctness gate (GSM8K + long-context needle, real acceptance) then transfer latency/bytes vs prediction (M5); decode prefix cache on/off (M8) | DSv4.1 PD runs correctly on ROCm with one connector | pass/fail per connector; bytes/request vs 1.63-2.47 KB x ISL; TTFT delta |
| E8 | Go/no-go: PD at equal total GPUs with P:D from E6 and router affinity, swept around E3's knee | PD beats the reproduced colocated best by more than the noise band | if yes -> Phase 3 PD optimisation (handbook levers: concurrency, P:D, decode DEP/TP, prefill chunking + host prefix tier, router, DSpark depth, kernels); if no -> report and decide with the user |
| E9 | Real DSpark acceptance on AgentX-like prompts per depth vs 3.51 (M9) | synthetic 3.51 is conservative/aggressive | only for labelling and depth choice; never mix synthetic and real in one comparison |

Note on the perf-phase rule (memory: no correctness gates during throughput pushes): E7's correctness gate is for PD
bring-up only, because the connector path is new code; once a PD stack passes, perf iterations proceed without gates.

---

## 6. Open questions

- Q1 (user): is the +50% target relative to the published vLLM MI355X bar (427,451 -> ~641k) or to our reproduced vLLM baseline,
  and at which interactivity (the bar sits at p90i 15.8)?
- Q2: openai-node host DRAM and RDMA inventory (E0); whether UMBP's 449 GB per 2-GPU share fits our node (CI used
  `dram-utilization 0.60`, implying ~3 TB/node [INF]).
- Q3: why the CI 3600 s run (427k c384) is ~1.47x the author's 1200 s cold-cache local run (291,619) - different image and KV
  dtype (fp8_e4m3 vs fp8_ds_mla) and duration; not explained [2H vs FH].
- Q4: cost of the connector-forced sparse-MLA path (E4), and whether `only_restores_canonical_kv` can be overridden for a PD
  connector without breaking correctness (not read).
- Q5: does MooncakeConnector/MoRIIO send one KV copy per decode TP rank or one copy + local broadcast (KV is replicated per rank)?
- Q6: does vLLM PD ship SWA rings or rebuild them on decode (Mooncake clips SWA to the last window; bounded replay N-1 on D);
  interplay with the DSpark draft SWA ring.
- Q7: the 838 B/token gap between engine slope and analytic cell (E5).
- Q8: DEP8 numbers: 182.7k (#5056) vs 26,714 (vllm#67377) at c256 - metric or workload difference not identified.
- Q9: "DCP not on main" for DSv4.1 ROCm: unverifiable (generic DCP plumbing exists; no ROCm DSv4.1 run).
- Q10: whether the infera MoRIIO DSv4 patches (verified on vLLM 0.23.x, DSv4-Pro) are still needed on Nov-2026 nightlies.
- Q11: rankings API interpolation for tier 125 (MI355X 150,079 vs highest measured SGLang point at p90i >= 125 of 110,744).
- Q12: HF transformers `deepseek_v41` exists only in open PR huggingface/transformers#50919; not reviewed beyond config layer types.

---

## Critic additions (2026-11-19)

Scope: gaps closable from code/API/web within one pass. Paths as above (`V` = `source/vllm` 7a62ca0666). New scratch copies in
`notes/critic/`.

### C1. Connector fast-path gate (Q4, B2) - code read, exact mechanism of the gate
- The gate is one class method, `KVConnectorBase_V1.only_restores_canonical_kv()` (default `False`) [FH: V/vllm/distributed/kv_transfer/kv_connector/v1/base.py:236-244],
  read through `kv_connector_may_load_foreign_kv()` [FH: V/.../kv_connector/utils.py:103-114]. Only `UMBPStoreConnector` (key_namespace "auto")
  and `MultiConnector` (all children) override it [FH: grep over V/vllm; umbp/connector.py:110-117; multi_connector.py:274-280].
  Mooncake, MoRIIO, NIXL, Mooncake-store, LMCache, offloading, and **`DecodeBenchConnector`** do not override it.
- Every DSv4.1 ROCm use of the flag goes through two functions: `_aiter_sparse_mla_enabled()` (needs env on AND no foreign writer) and
  `_trust_dsv4_extra_cache_nan_free()` (needs gfx950 AND `fp8_ds_mla` AND no foreign writer) [FH: V/vllm/models/deepseek_v41/amd/rocm.py:80-109, 383-385, 597, 1254-1258].
  The stated reason is NaNs in unwritten slots: "The aiter kernel does not scrub NaNs" [FH: rocm.py:99-100].
- Consequence for E4 [FH-derived]: with `--kv-cache-dtype fp8_e4m3` (the bar's config), `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=0`
  selects exactly the same two gate outcomes as attaching a non-UMBP connector (nan_free is already off for fp8_e4m3). With
  `fp8_ds_mla` the env-var emulation is NOT equivalent (nan_free stays on without a connector). So E4 should run on fp8_e4m3.
- Consequence for E6 [FH]: `DecodeBenchConnector` (V/.../v1/decode_bench_connector.py), the tool the vLLM team used for decode
  measurement, would itself switch DSv4.1 ROCm to the Triton sparse path. Decode-only numbers taken with it measure the PD-side
  kernel path, not the colocated one; label them so.
- Whether overriding `only_restores_canonical_kv=True` for Mooncake/MoRIIO is safe (i.e. whether these connectors only write whole
  blocks produced by an identical layout, including the last partial block) is NOT established; not read.

### C2. Mooncake TP mapping: one KV copy per decode TP rank (Q5) - code read
- Each consumer (decode) TP rank handshakes with `handshake_target_ranks()`: `[tp_rank // ratio]` when D TP <= P TP, else
  `[tp_rank*k + i for i<k]` [FH: V/.../kv_connector/utils.py:683-706]. When the producer cache is replicated (MLA-style,
  `local_replicates_kv_cache`), only one producer rank sends to a given consumer rank (`local_tp_rank % tp_ratio == 0`), other
  pairs `continue` [FH: mooncake_connector.py:288-342, 2228-2238, 3146-3147].
- So every decode TP rank receives its own full copy; wire bytes per request = per-request KV bytes x decode TP size (DP ranks are
  separate requests). A TP2 decode doubles the bytes in pd_method.md s3.2 [FH-derived].
- Re-scaled demand at the bar's +50% target (641k tok/s/GPU, 16 GPUs, served mean ISL 173,006 + OSL 1,023.5): 58.9 qps x
  0.29-0.43 GB = 17-25 GB/s with a TP1 decode, 34-51 GB/s with a TP2 decode [INF]; vs 8 crusoe rails ~306 GB/s measured
  [2H: knowledge repo]. pd_method.md's 4.2-6.3 GB/s used the old 107k anchor and one copy; superseded by this line.

### C3. Image contents for PD on ROCm (Section 4.3 "NIXL: not checked") - Dockerfile read
- vLLM main's ROCm Dockerfile builds NIXL (ai-dynamo/nixl @1683cf3b) + UCX (@2ff0b47c) as `nixl-rocm` wheel and installs it into the
  `runtime` stage that `vllm-openai` derives from [FH: V/docker/Dockerfile.rocm:36-137, 248, 319-326, 594]. MoRI v1.2.3.post1 with
  UMBP is in the base [FH: V/docker/Dockerfile.rocm_base:15-16, 168-188]. Mooncake comes from PyPI `mooncake-transfer-engine-rocm==0.3.14`
  [FH: V/requirements/rocm.txt:31].
- Whether the published `vllm/vllm-openai-rocm:nightly-*` tags are built from this Dockerfile and contain these wheels: [INF], not
  checked inside an image. Check with `pip list | grep -i -E "nixl|mooncake|mori"` in the R1 image at E0.

### C4. vllm-project/recipes repo (previously "not searched")
- Upstream `models/deepseek-ai/DeepSeek-V4.1-Flash.yaml` last changed 2026-10-17 (#1137 GB300 PD) [FH: GitHub commits API]; the
  in-tree `V/recipes/...` copy is newer (date_updated 2026-11-19, includes `mi355x_agentx_dp2_umbp` preset from IX #5036)
  [FH: diff, `notes/critic/recipes_repo_DeepSeek-V4.1-Flash.yaml`]. Keep using the in-tree copy.
- Recipe PD statement: the only verified vLLM PD layout is 1P1D GB200 NVL4 TP4/TP4 over NIXL with `vllm-router --vllm-pd-disaggregation`,
  and "With Speculative decoding on, DSpark runs in both pools so the transferred KV stays compatible" [FH: RCP:2076-2086]. Implication for
  E7: launch the prefill pool with the same `--speculative-config` as decode.
- Open recipes PRs touching V4.1 on AMD: #1262 (Mooncake PD connector on AMD with the ROCm wheel; open, updated 11-18), #962/#965/#966
  (Sept image alignment, open/stale), #1185/#1197 (open "Add recipe") [FH: gh search]. Merged #1254 UMBP offload option [FH].

### C5. ROCm/vllm fork (previously "not searched")
- DSv4.1 work there is gfx1250 enablement/perf (#1336 merged, #1360 merged, #1386 open, #1317 open "initial enablement and gates
  for dsv4.1f") and older DSv4 AITER decode stacks (#901-#903, May). No PR matching "disaggregated" or "MoRIIO"; one branch
  `akaratza_rocm_pd_accuracy` (contents not read) [FH: gh search/branches API, 2026-11-19]. Nothing to cherry-pick for MI355X PD found.

### C6. vLLM blog 2026-10-23 "Serving DeepSeek-V4.1-Flash at Scale" (PD + Wide-EP, GB300) - not in any dimension file
Source: https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep (copy `notes/critic/vllm_blog_2026-10-23_dsv41_pd_wide_ep.md`). All NVIDIA; ROCm applicability unverified.
- Claims [2H: vLLM team, GB300/GB200]: decode MoE-bound (DEP4, 128K ctx, 64 req/rank: MoE 62% of target step, attention+indexer 13%);
  DEP32 target forward 8.86 -> 9.53 ms from 64K to 855K context; prefill DEP2 > DEP4 per GPU (64.4K vs 53.3K tok/s/GPU, AgentX
  prefill-only c256), DP-imbalance loss 5.9% (DEP2) vs 13.6% (DEP4) by mirror/lag replay; new-token prefill at 64K new on 128K cached
  71.2K -> 102.5K tok/s/GPU after three fixes; 8K prefill 87.9 ms (no context) vs 189.4 ms (512K cached), two GB300 GPUs.
- PD CPU-side levers listed with PR status checked today [FH: gh pr view]: NIXL push + packed DSv4.1 MLA rows #61121 (OPEN; blog: 141K -> 440K
  tok/s/GPU, GB300 8P1D c3072); `max_async_kv_loads_per_step` #61725 (MERGED 11-10; present in V/vllm/config/scheduler.py:90; blog:
  460K -> 516K with two other fixes, 7P1D c4800); recursive-doubling CPU all-reduce for DP sync #61767 (MERGED); prompt int32 buffers
  #62702, SWA null-head skip #61724, chunked staged writes #62698 (MERGED); #59908 CLOSED.
- Methods directly reusable for E6 [FH: PRs merged]: `vllm bench forward` #61339 (V/vllm/benchmarks/forward), `DecodeBenchConnector`
  (see C1 caveat on ROCm), mirror/lag replay for DP imbalance, real-vs-uniform routing for EP balance.

### C7. InferenceX harness / AMD PD templates
- Warmup: "Official runs use 10 warmup requests per lane and a one-hour profile"; `agentx-fast` = 1 warmup/lane + 20-min profile
  [FH: IX .github/AGENT_OPERATIONS.md:55; IX .github/workflows/benchmark-tmpl.yml:64]. Closes the s3.2 warmup gap for official rows
  (the 11-18 aiperf.log itself still not read).
- IX ships an agent skill for live AgentX/disagg debugging: `IX/.agents/skills/debug-agentx-runs/SKILL.md` (phase markers, per-role
  log checks; "Attention DP inside an aggregate deployment is not disaggregation") [FH]. Use for E7/E8 babysitting.
- MI355X PD launcher templates in IX main (other models/engines): GLM-5.2 ATOM PD with per-NIC `mooncake_store` prefix tier on prefill,
  `MC_ENABLE_DEST_DEVICE_AFFINITY=1` for rail-isolated fabric, DCP4 decode [FH: IX/inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/glm5.2/atom/mi355x-fp4/agentx/disagg-variants.yaml:1-160];
  Kimi-K3.1 SGLang PD over Mooncake [FH: .../kimik3.1/sglang/mi355x-fp4/agentx/disagg-variants.yaml:16,66-68]. No vLLM MI355X disagg recipe in IX main;
  #5090 (MiniMax vLLM Mooncake PD) still open, last commit 11-19 07:31 [FH: gh].

### C8. HF transformers PR #50919 (Q12) - cache-layer read
- Still OPEN (updated 2026-11-18) [FH: gh]. `DeepseekV41CacheLayer` per-request state: SWA window keeping `sliding_window-1` keys,
  `compressed_kv` + `index_keys` on compressed layers, a pooling buffer (`buffer_kv`/`buffer_gate`) holding the tokens of an
  unfinished compression group, and an Engram `token_history` on the first layer [FH: source/transformers_pr50919/modeling_deepseek_v41.py:50-120].
  This is the reference list of state a P->D handoff must reproduce or recompute; vLLM's equivalents are the SWA ring, compressed
  KV, indexer K and `CircularBufferManager` (SYNTHESIS s2). Whether vLLM's Mooncake path transfers or recomputes the pooling-buffer
  state: not established (bounded replay recomputes the last token only [FH: mooncake_connector.py:1107-1114]).

### C9. AMD blogs (re-searched)
- WebSearch 2026-11-19 "DeepSeek-V4.1-Flash MI355X vLLM disaggregated ... AMD blog": no AMD/ROCm blog on DSv4.1 PD or vLLM serving;
  index shows a DeepSeek-V4 (Flash) post and an "AMD Disaggregated Inference" series post whose topic was truncated (not opened).

---

## Remaining gaps

- G1: Safety of overriding `only_restores_canonical_kv` for Mooncake/MoRIIO (partial last block, foreign bytes) - needs a read of
  the connectors' write paths plus a correctness test (E4/E7).
- G2: Whether published `vllm-openai-rocm` nightlies contain nixl-rocm / mooncake / mori-UMBP wheels - check in-image at E0.
- G3: Whether vLLM's PD path transfers or recomputes the compressor pooling buffer and SWA rings, and interplay with DSpark draft
  KV (Q6) - code not read end to end.
- G4: openai-cluster host DRAM and NIC/RDMA inventory (U1, Q2) - needs a node (E0).
- G5: The 2,468 vs 1,630 B/token gap (Q7) - needs a startup log (E5).
- G6: DEP8 conflict 182.7k vs 26,714 (Q8); CI-vs-local 1.47x (Q3) - needs our own runs.
- G7: Rankings tier-125 interpolation (Q11) - not pursued.
- G8: The 11-18 CI run's aiperf.log (actual warmup count / errors) - not downloaded.
- G9: ROCm/vllm branch `akaratza_rocm_pd_accuracy`; the AMD "Disaggregated Inference" blog series post; NIXL push connector blog
  (2026-10-14) and #61121 diff - not read.
- G10: Colleague dirs not inspected in depth: `/perf_apps/chaox/{GEAK,agentx-kernel-extraction-standalone,mono_k2_r2}`,
  `/perf_apps/jihhe/dsv41di/eplb-dg`; yanfwang vLLM idxshard image source.
- G11: ROCm applicability of the vLLM-blog NVIDIA findings (decode MoE-bound, context-insensitive step, prefill DEP2 optimum) -
  to be measured in E6 with `vllm bench forward`.

---

## Update 2026-11-19: Dynamo+vLLM study

Full write-up: `research/dynamo_vllm.md` (inputs `research/dynamo/{recipes,vllm_side,framework,perf_anatomy}.md`).
- Template: B300 Dynamo+vLLM PD (8-GPU HGX, 4-5x DEP2 P + one DEP8 D, KV over RDMA push): 828,209 @ c3584 (row 448173) and
  2-node 738,119 @ c2048 (row 448172) [FH]. Same-SKU PD gain 1.323x (B300), 1.172x (GB300) [FH rows]. The 828k row ran without
  the async-KV admit cap (env ignored), draft EPLB, and Engram FABRIC [FH logs].
- New planning budget (dynamo_vllm.md s2.3) [INF]: +50% = 5.59 req/s/GPU -> c/p + o/g <= 0.179 GPU-s per request
  (c computed prompt tokens/request 4.4k-6.8k, p prefill tok/s per P GPU, o = 906, g output tok/s per D GPU). Validated on B300
  (0.164 -> 6.11 req/s/GPU). p and g on MI355X are unmeasured and now gate PD.
- Changed: E0 (+host DRAM, connector wheels in image, NUMA map), E2 (per-rank logs), E3 (+KV hint, P/D device split), E4 (hard
  gate), E6 -> split into E12/E13 with the budget as go/no-go, E7 (P = DP2 worker, D = DP4/DEP4, decode prefix cache off,
  NIXL push as 3rd connector, same spec config both pools), E8 (B300 topology ladder: same node -> 4xDP2 P node + DEP8 D node -> 3 nodes).
- New: E10 colocated knob A/B (long-prefill-token-threshold 8192, prefix-cache-retention-interval 32768, gpu-mem 0.95 /
  kv-cache-memory, NVIDIA colocated interval 96 + mbt 65536); E11 router A/B (consistent_hash vs vllm-router 0.1.17 kv_aware vs
  infera kv_event_aware); E12 prefill rate p (DP2 vs DEP2 vs DP4, indexer top-k share); E13 decode rate g with wide EP (DEP4/DEP8
  AITER MegaMoE, Engram HBM, FULL_DECODE_ONLY on gfx950, EP balancedness, #5056 vs #67377); E14 PD admission/slot sweep
  (max-num-seqs at equilibrium, mbt 10x, --max-async-kv-loads-per-step 2/4/7, inflight = seqs, D prefix cache on/off);
  E15 prefill host tier under PD (UMBP vs SimpleCPUOffload); E16 router/frontend CPU at high c; E17 ROCm code items (fast-path
  override, DSv4.1 ROCm EPLB redundancy, FULL_DECODE_ONLY, NIXL push on nixl-rocm).
- Order on the one assigned node: E0 -> E1 -> E2(+E11) -> E4 -> E10+E3 -> E12+E13 (budget gate) -> E7 -> E14+E15 -> E8 stage 1;
  second node needed for E8 stage 2. 3600 s runs only.
- Not transferable: NVL72 domain, Grace C2C, CUDA kernels (MegaAttn, MegaMoE, DeepSelect, specialized decode), symm_mem EPLB,
  nvfp4 KV record, FABRIC Engram, torch.compile graphs on ROCm DSv4.1, raw FP4 peak.
