# Dynamo + vLLM (and Dynamo + SGLang / TRT-LLM) DSv4.1-Flash AgentX recipes, and what transfers to MI355X vLLM

Date: 2026-11-19. Research only (no GPU, no docker). Tags: **[FH]** first-hand (file:line, API row id, PR read by me,
log line read by me); **[2H]** second-hand (PR author's / blog's own claim, not reproduced); **[INF]** my arithmetic or
inference, unverified.

Sources and paths used below:
- `IX` = `source/InferenceX` @ db3fc796 (2026-11-18); fetched origin/main 6ed3774 (2026-11-19): no change under
  `benchmarks/multi_node/srt-slurm-recipes/dsv41flash` between the two [FH: git diff --stat].
- `V` = `source/vllm` @ 7a62ca0666 (2026-11-19). `RCP` = `V/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`.
- `DYN` = `source/dynamo` sparse shallow clone @ beb1a41 (2026-11-19), docs + kv_router only.
- API = `https://inferencex.semianalysis.com/api/v1/{benchmarks,reproduce,server-log}`. Saved:
  `notes/dynamo/configs/reproduce_<row>.json` and the **verbatim run config** `notes/dynamo/configs/row_<row>_config.yaml`
  (the `config.yaml` that srtctl actually launched, incl. the synthetic-AL rewrite of `--speculative-config`);
  log excerpts `notes/dynamo/logs/448173_*` (B300 best); PR bodies `notes/dynamo/prs/pr_<n>.json`;
  derived per-GPU table `notes/dynamo/derived_pd_rows.txt`.
- All rows: DSv4.1-Flash FP4 checkpoint (TRT-LLM: NVFP4), DSpark 5 drafts, synthetic AL 3.51, AgentX dataset
  `semianalysis_cc_traces_weka_062126`, 3600 s, metric `tput_per_gpu` = (all input incl. cached + output) / duration / (P+D GPUs).

---

## 0. Summary (what the evidence shows)

1. Every top NVIDIA PD entry has the same shape: **N x DEP2 prefill workers (TP1, DP2, EP2) + ONE wide DEP decode worker
   (DEP8/16/32)**, NIXL **push** transfer, **host-DRAM KV tier on prefill only** (vLLM `SimpleCPUOffloadConnector`, lazy,
   60-170 GiB per prefill rank), **decode prefix caching off**, Dynamo KV-aware router with 14400 s session affinity and
   prefill-only (no conditional local prefill on decode) [FH: row configs 448173, 445572, 448128, 448144].
2. Best per HW (tok/s/GPU): GB300 SGLang 1,071,944 (9x DEP2 + DEP16, 34 GPUs, c7480, p90i 105.6, row 448095); GB300 vLLM
   1,011,514 (12x DEP2 + DEP16, 40 GPUs, c9888, p90i 75.5, row 445572, 2026-10-20); B300 vLLM 828,209 (5x DEP2 + DEP8,
   18 GPUs, c3584, p90i 83.8, row 448173); GB200 vLLM 817,333 (5x DEP4 + DEP16, 36 GPUs, c5600, p90i 125.6, row 448128);
   vr200 vLLM 2,993,171 (8x DEP2 + DEP16, 32 GPUs, c15800, p90i 180.8, row 448150, run in private NVIDIA repo, no config
   published). GB300 TRT-LLM: open PR #4991, internal runs up to 1,492,133 (5x DEP2 + DEP8, c7000) [2H].
3. Per-GPU request rate is the lever: B300 best serves 6.11 req/s per GPU (110 req/s / 18) vs MI355X bar 3.73 (7.46 / 2);
   +50% on MI355X (641k at mean input 114k) needs ~5.6 req/s/GPU [FH metrics; INF arithmetic].
4. Mechanisms the logs show on B300 best (448173): decode runs AT its `max-num-seqs` cap (1340-1342 running of 168x8=1344,
   619-639 waiting, KV fetch queue 254-355) with only 19-20% decode KV used -> decode-slot-bound at the best point;
   prefill ranks report 98% prefix hit, GPU KV used 7-9% [FH: decode/prefill logs 21:58 / 21:44].
5. P:D is chosen by Little's law, not by a ratio rule: keep prefill saturated, size decode `max-num-seqs` to the
   equilibrium batch (running = request rate x decode time per request), concurrency from measured think time [2H: IX #4337,
   #4673]. Best points land at P:D GPU = 10:8, 24:16, 18:16, 20:16, 16:16 (1.0-1.5 : 1) [FH].
6. Decode tuning pattern (all vLLM rows): FULL_DECODE_ONLY graphs in steps of 6 (=1+5 drafts) up to 6 x max-num-seqs,
   `max-num-batched-tokens = 10 x max-num-seqs`, max-num-seqs on "MegaAttn wave boundaries", 2-7 async KV loads admitted per
   step, EPLB with 16-32 redundant experts [FH configs; 2H #4337].
7. Prefill tuning pattern: `max-num-seqs 64`, `max-num-batched-tokens 16384/32768`, `long-prefill-token-threshold 8192`,
   `prefill-schedule-interval 2`, `prefix-cache-retention-interval 32768`, `gpu-memory-utilization 0.95`,
   Engram tables on host for prefill [FH configs].
8. ROCm transfer status (code read on V): NIXL push + DSv4.1 CSA-linear: present; SimpleCPUOffload has a HIP batch-copy
   path; scheduler knobs (`max_async_kv_loads_per_step`, `max_num_inflight_kv_loads`, `long_prefill_token_threshold`,
   `prefix_cache_retention_interval`) are platform-generic; **EPLB redundancy absent on the ROCm DSv4.1 runner**;
   **DeepSelect top-k is SM100/SM103 only**; MegaMoE/MegaAttn/FlashMLA are NVIDIA kernels; any non-UMBP connector turns the
   ROCm sparse-MLA fast path off (SYNTHESIS C1) [FH file:line in s8].
9. Measured-on-NVIDIA claims that cannot be assumed for MI355X: decode MoE-bound, decode step insensitive to context,
   DEP2 > DEP4 for prefill [2H: vLLM blog 2026-10-23]. Each needs an MI355X measurement (E6).

---

## 1. Inventory: every Dynamo DSv4.1-Flash AgentX PD entry (published API, 2026-11-19)

Derived columns: qps/GPU = mean_qps / all GPUs; req/s/Pgpu = mean_qps / prefill GPUs; out/Dgpu = output tok/s / decode GPUs;
uncached/Pgpu = input_tput x (1 - gpu_hit - cpu_hit) / prefill GPUs [INF arithmetic on FH API fields;
`server_{gpu,cpu}_cache_hit_rate` semantics in PD not verified - B300 and small vr200 points show a near-constant 0.48/0.50
split, which looks like a metric artifact; IX #4336 links a "CPU KV metadata caveats" audit #4399, not read].

| row | HW | engine (image) | P | D | GPUs | conc | tok/s/GPU | p90i | qps/GPU | req/s/Pgpu | out/Dgpu | gpu/cpu hit | TTFT mean s | date |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 448150 | vr200 | dynamo-vllm (`vllm-cu134-vr1111f-b1`, mooncake host tier) | 8xDEP2 | DEP16 | 32 | 15800 | 2,993,171 | 180.8 | 20.48 | 40.97 | 42,432 | 0.098/0.877 | 2.5 | 11-13 |
| 448148 | vr200 | same | 6xDEP2 | DEP16 | 28 | 11500 | 2,632,993 | 259.7 | 17.77 | 41.45 | 32,349 | 0.084/0.884 | 2.5 | 11-13 |
| 448149 | vr200 | same | 6xDEP2 | DEP32 | 44 | 11000 | 1,695,282 | 411.5 | 11.28 | 41.37 | 16,192 | 0.111/0.855 | 2.4 | 11-13 |
| 448147 | vr200 | same | 3xDEP4 | DEP32 | 44 | 6600 | 1,097,953 | 561.8 | 7.19 | 26.37 | 10,315 | 0.824/0.141 | 0.9 | 11-13 |
| 448144 | vr200 | same (vllm-simple host tier) | 1xDEP2 | 3xTP4 | 14 | 310 | 167,331 | 840.3 | 1.10 | 7.67 | 1,331 | 0.481/0.500 | 1.5 | 11-13 |
| 448145 | vr200 | same | 1xDEP2 | 4xTP4 | 18 | 260 | 112,598 | 1030.9 | 0.72 | 6.51 | 843 | 0.481/0.500 | 1.2 | 11-13 |
| 448095 | gb300 | dynamo-sglang (`lmsysorg/sglang:nightly-dev-cu13-20261111-314bce7c`) | 9xDEP2 | DEP16 | 34 | 7480 | 1,071,944 | 105.6 | 7.68 | 14.50 | 16,686 | 0.786/0.165 | 6.0 | 11-13 |
| 448094 | gb300 | same | 7xDEP2 | DEP16 | 30 | 5540 | 992,956 | 161.8 | 6.99 | 14.98 | 13,510 | 0.819/0.135 | 5.3 | 11-13 |
| 448093 | gb300 | same | 6xDEP2 | DEP32 | 44 | 3018 | 474,138 | 321.5 | 3.16 | 11.59 | 4,493 | 0.905/0.056 | 1.2 | 11-13 |
| 445572 | gb300 | dynamo-vllm (`vllm/vllm-openai:cu134-nightly-aarch64-dev-20261019`) | 12xDEP2 | DEP16 | 40 | 9888 | 1,011,514 | 75.5 | 7.45 | 12.42 | 18,820 | n/a | 7.8 | 10-20 |
| 445571 | gb300 | same | 9xDEP2 | DEP16 | 34 | 7480 | 993,521 | 145.6 | 7.13 | 13.47 | 15,653 | n/a | 10.5 | 10-20 |
| 445573 | gb300 | same | 16xDEP2 | DEP32 | 64 | 10888 | 954,146 | 196.5 | 6.62 | 13.25 | 13,732 | n/a | 4.3 | 10-20 |
| 445574 | gb300 | same | 6xDEP2 | DEP32 | 44 | 3018 | 461,635 | 331.1 | 3.09 | 11.32 | 4,404 | n/a | 1.9 | 10-20 |
| 448128 | gb200 | dynamo-vllm (`...cu134-nightly-aarch64-01fba62-1111d`) | 5xDEP4 | DEP16 | 36 | 5600 | 817,333 | 125.6 | 5.79 | 10.41 | 13,354 | 0.702/0.261 | 4.0 | 11-13 |
| 448127 | gb200 | same | 4xDEP4 | DEP16 | 32 | 4096 | 761,086 | 191.9 | 5.24 | 10.48 | 10,848 | 0.727/0.247 | 3.1 | 11-13 |
| 448125 | gb200 | same | 3xDEP4 | DEP16 | 28 | 3018 | 662,528 | 229.9 | 4.53 | 10.57 | 8,212 | 0.727/0.237 | 3.2 | 11-13 |
| 448126 | gb200 | same | 3xDEP4 | DEP32 | 44 | 3018 | 444,467 | 327.9 | 3.00 | 11.00 | 4,288 | 0.804/0.159 | 3.0 | 11-13 |
| 448173 | b300 | dynamo-vllm (`vllm/vllm-openai:cu134-nightly-96710245...`, vLLM 0.34.0rc1.dev142) | 5xDEP2 | DEP8 | 18 | 3584 | 828,209 | 83.8 | 6.11 | 11.00 | 13,908 | 0.466/0.510 | 8.5 | 11-17 |
| 448172 | b300 | same | 4xDEP2 | DEP8 | 16 | 2048 | 738,119 | 120.5 | 5.16 | 10.32 | 10,527 | 0.478/0.501 | 1.4 | 11-17 |
| 448171 | b300 | same | 3xDEP2 | DEP8 | 14 | 1280 | 576,607 | 175.4 | 3.94 | 9.19 | 7,081 | 0.480/0.500 | 1.1 | 11-17 |
| 448170 | b300 | same | 2xDEP2 | DEP8 | 12 | 768 | 420,185 | 214.1 | 2.86 | 8.59 | 4,420 | 0.480/0.500 | 0.9 | 11-17 |
| 448169 | b300 | same | 2xDEP2 | DEP8 | 12 | 256 | 154,021 | 274.7 | 1.01 | 3.03 | 1,543 | 0.481/0.500 | 0.3 | 11-17 |
| 448242 | **mi355x** | vllm colocated (`vllm/vllm-openai-rocm:nightly-60bd6dd2...`) | - | TP1xDP2 | 2 | 384 | **427,451** | 15.8 | 3.73 | (3.73) | (3,378) | 0.883/0.057 | 2.2 | 11-18 |

Not published but relevant [2H]:
- GB300 Dynamo + TRT-LLM (open IX #4991, jh-nv, image `nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc4.dev202611160007`, Dynamo
  1.7.0.dev20261110, internal cluster, "no public run links"): c7000 5xDEP2 + DEP8 18 GPUs 1,492,133 total tok/s/GPU at median
  87.7 tok/s/user (first 3,068 s only); c11718 10xDEP2 + DEP16 1,447,284; c7900 7xDEP2 + DEP16 1,404,065; c5100 1,172,727;
  c7600 7xDEP2 + DEP32 827,879 [2H: PR #4991 validation table]. Interactivity there is median (1000/p50 ITL), not p90.
- GB300 vLLM #5015 (open; points 5P1D DEP32 c3200, 7P1D DEP16 c6600, 12P1D DEP32 c10288; GSM8K 0.976/0.975/0.973; sweep run
  43089248484) - throughput numbers not in the PR body I read [FH: PR body; RCP:1760-1766].
- SGLang GB300 local 3600 s runs: 9xDEP2 BS192 c7480 1,117,063 @ p90 97.4; 7xDEP2 c5540 1,033,494 @ 157.5 [2H: IX #4673].

## 2. Score curve vs concurrency and how P:D was chosen

Curves (tok/s/GPU, p90i) [FH rows above]:
- B300 vLLM (fixed decode DEP8): c256 2P 154k @ 275 -> c768 2P 420k @ 214 -> c1280 3P 577k @ 175 -> c2048 4P 738k @ 121 ->
  c3584 5P 828k @ 84. Each step adds one DEP2 prefill worker AND raises decode `max-num-seqs` 18 -> 40 -> 72 -> 132 -> 168
  [FH: RCP:2353; row configs]. tok/s per added GPU stays positive to 18 GPUs; no point beyond 5P was published.
- GB300 vLLM: DEP16 decode beats DEP32 decode at the high-throughput end (12P+DEP16 1.01M @ 75 vs 16P+DEP32 954k @ 197); DEP32
  serves the high-interactivity end (6P+DEP32 462k @ 331).
- GB200 vLLM: DEP4 prefill (GB200 NVL4 tray); 3P->5P DEP4 at DEP16: 663k -> 761k -> 817k.

How P:D was chosen [2H: PR authors]:
- IX #4337 (GB300 vLLM, Juntian777): "on run 39207089292, 16P1D DEP32 c10888 reached 954K total tok/s/GPU with prefill at 53K
  uncached tok/s/GPU and decode near its max-num-seqs 76 cap. Scaling prefill and decode capacity from that run and DEP32
  decode-only step times, 20P1D at max-num-seqs 152 balances prefill and decode; conc is sized by Little's law from that run's
  think time." and "max-num-seqs sits on MegaAttn wave boundaries; batched tokens are max-num-seqs x 10 and CUDA graph sizes go up
  to max-num-seqs x 6 in steps of 6".
- IX #4673 (GB300 SGLang): "Prefill stays saturated at every point. Concurrency is sized so prefill always has queued work; the
  decode batch is whatever the prefill rate supports (Little's law: running requests = request rate x decode time per request)."
  "Binding decode cap (c5540) ... Throughput is the same as with a high cap, because prefill sets the request rate"; at c7480
  "cap 192 with total_requests balancing lets decode take every request prefill delivers. total_requests keeps the per-rank
  running counts within 2-7 of each other (11-26 with total_tokens); the step is paced by the busiest rank."
- IX #4997 (SGLang): DEP32 decode keeps "each rank at about 18 running requests, so the decode step stays short" -> one DEP32 point
  covers the middle of the curve.
- IX #4673 also states a "1200 s -> 3600 s factor (about 3.8x)" used to project fast runs [2H]; our SYNTHESIS Q3 saw a 1.47x
  CI/local gap on MI355X. Do not compare 1200 s runs to 3600 s rows.

What the B300 best-point logs show [FH: notes/dynamo/logs/late_*decode*.txt, mid_*prefill*.txt, 448173]:
- Decode (DEP8) at 21:58 (about 80 min into the job; run started routing 20:39): `Reqs Running: 1340, Waiting: 619, Deferred: 365,
  KV fetch: 254 waiting to start ... 365 completed waiting, GPU KV cache used: 20.1%, Prefix cache hit rate: 0.0%, External prefix
  cache hit rate: 100.0%`, generation 116,892 tok/s; DSpark mean AL 3.50. -> running = 1340 of the 1344 cap.
- Prefill w0 (DEP2) at 21:44: `Reqs Running: 3-12, Waiting: 3-13, GPU KV cache used: 7.2-9.2%, Prefix cache hit rate: 87.5-98.5%`,
  prompt throughput 34k-156k tok/s per 1-s sample.
- KV capacity per rank: prefill DEP2 `Available KV cache memory: 88.54 GiB`, `GPU KV cache size: 47,894,993 tokens`; decode DEP8
  `192.39 GiB`, `188,403,438 tokens`; KV format `nvfp4_ds_mla`; host tier `SimpleCPUOffloadConnector: role=WORKER, per_rank=70.00 GB,
  world_size=1, mode=lazy` = `593133 CPU blocks (70.00 GB)` per DP rank; weights 149.21 GiB/GPU prefill (DEP2), 46.94 GiB/GPU decode
  (DEP8); Engram `94.42 GiB shared across DP replicas via /dev/shm` (pinned host) on both roles.
  Derived [INF]: ~1,985 B/token prefill, ~1,096 B/token decode (bytes / reported tokens).

## 3. Recipes (verbatim excerpts; full launched configs in `notes/dynamo/configs/row_<row>_config.yaml`)

### 3.1 B300 Dynamo + vLLM, 5P1D DEP2/DEP8 c3584 (row 448173, run 43769338888 attempt 2, IX head a358d7d7, recipe
`IX/.../dsv41flash/vllm/b300-fp4/agentx/disagg-variants.yaml` override `override_5p1d_dep2_dep8_c3584`, merged IX #4585)

Image `vllm/vllm-openai:cu134-nightly-96710245ece8fe57a0df3ea18cdc79cbb8ce93cf@sha256:6743719...` (vLLM `0.34.0rc1.dev142+g96710245e`),
Dynamo `1.6.0.dev20261010`, 3 nodes x 8 B300 (Xeon 8559C), EFA [FH: reproduce_448173.json launch/fingerprints].

Frontend (on first decode node) [FH: row_448173_config.yaml]:
```yaml
frontend:
  type: dynamo
  numa_bind: true
  args:
    router-mode: kv
    router-session-affinity-ttl-secs: 14400
    active-decode-blocks-threshold: None
    active-prefill-tokens-threshold: None
    active-prefill-tokens-threshold-frac: None
  env:
    DYN_ROUTER_TEMPERATURE: '0'
    DYN_ROUTER_QUEUE_THRESHOLD: None
    DYN_TOKENIZER: fastokens
    DYN_TOKENIZER_CACHE: '1'
    DYN_TOKENIZER_CACHE_BYTES: '8000000000'
    DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS: '3600'
    DYN_TCP_CHANNEL_BUFFER: '128'
    DYN_TCP_REQUEST_TIMEOUT: '60'
dynamo: {request_plane: tcp, event_plane: zmq, sidecar: true,
         sidecar_args: [--grpc-startup-deadline-secs, '9000', --dyn-tool-call-parser, deepseek_v41, --dyn-reasoning-parser, deepseek_v41]}
benchmark.env: {AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING: '0', AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID: 'true',
                AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS: '14400'}
```
Prefill role (5 workers x 2 GPUs; env common to both roles):
```yaml
env: {VLLM_USE_V2_MODEL_RUNNER: '1', VLLM_ENGINE_READY_TIMEOUT_S: '7200', VLLM_RPC_TIMEOUT: '600000', VLLM_LOG_STATS_INTERVAL: '1',
      VLLM_SERVER_DEV_MODE: '1', VLLM_ALLREDUCE_USE_SYMM_MEM: '0', VLLM_RANDOMIZE_DP_DUMMY_INPUTS: '1', NCCL_CUMEM_ENABLE: '1',
      UCX_TLS: cuda_copy,cuda_ipc,srd,tcp, UCX_MEMTYPE_CACHE: n, UCX_MEMTYPE_REG_WHOLE: n, UCX_RNDV_PIPELINE_ERROR_HANDLING: y,
      NIXL_UCX_SGL_ENABLE: '0', FI_PROVIDER: efa, FI_EFA_USE_DEVICE_RDMA: '1', FI_EFA_FORK_SAFE: '1'}
args:
  kv-transfer-config: '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"transfer_mode":"push","backends":["LIBFABRIC"],"enforce_handshake_compat":false,"kv_lease_duration":120}},{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":75161927680,"lazy_offload":true}}]}}'
  language-model-only: true
  tokenizer-mode: deepseek_v41
  reasoning-parser: None
  kv-cache-dtype: fp8
  max-model-len: 1048576
  tensor-parallel-size: 1
  data-parallel-size: 2
  enable-expert-parallel: true
  enable-cumem-allocator: false
  enable-ep-weight-filter: true
  cpu-distributed-timeout-seconds: 7200
  attention-config: '{"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'
  engram-config: '{"cpu_offload":true,"embedding_across_dp":true,"use_thp":false}'
  speculative-config: '{"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic", "rejection_sample_method": "synthetic", "enable_adaptive_verification": false, "synthetic_acceptance_length": 3.51}'
  prefill-schedule-interval: 2
  gpu-memory-utilization: 0.95
  kernel-config: '{"enable_flashinfer_autotune":true,"moe_backend":"deep_gemm_mega_moe","sparse_indexer_topk_backend":"deep_select"}'
  compilation-config: '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2046],"decoder_replay_cudagraph_capture_sizes":[6,...,1536,2048],"max_cudagraph_capture_size":2046}'
  max-num-seqs: 64
  max-num-batched-tokens: 16384
  long-prefill-token-threshold: 8192
  prefix-cache-retention-interval: 32768
  numa-bind: true
```
Decode role (1 worker x 8 GPUs; prefill env + `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP: '4'`):
```yaml
args:
  kv-transfer-config: '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"transfer_mode":"push","backends":["LIBFABRIC"],"enforce_handshake_compat":false,"kv_lease_duration":120}}'
  (same model/attention/kernel/engram/spec/kv-cache-dtype/gpu-mem 0.95/prefill-schedule-interval 2 as prefill)
  data-parallel-size: 8
  enable-eplb: true
  eplb-config: '{"communicator":"symm_mem","num_redundant_experts":16,"window_size":1000,"step_interval":3000,"use_async":true,"log_balancedness":true,"log_balancedness_interval":3000}'
  no-enable-prefix-caching: true
  numa-bind-nodes: [0, 0, 0, 0, 1, 1, 1, 1]
  max-num-seqs: 168
  max-num-batched-tokens: 1680
  compilation-config: '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":1008,"cudagraph_capture_sizes":[6,12,18,...,1002,1008]}'
```
Notes [FH]: `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` is not in V main (`grep ASYNC_KV vllm` = 0 hits); main has the merged flag
`--max-async-kv-loads-per-step` (vllm#61725, `V/vllm/config/scheduler.py:90-101`) which GB200 and #5015 use. RCP:2346-2347 says the
env and `push_as_nixl` came from dev image `nightly-x86_64-dev-20261012-59e75f4`. Decode log warns "EPLB is disabled for the DSpark
draft model" and "EP weight filter disabled" in this image.

### 3.2 GB300 Dynamo + vLLM, 12P1D DEP2/DEP16 c9888 (row 445572, run 39207089292 attempt 12, 2026-10-20, IX #4266 append)

Differences vs 3.1 (everything else identical in `row_445572_config.yaml`) [FH]:
| item | GB300 445572 | B300 448173 |
|---|---|---|
| nodes | 11 x 4 GB300 (aarch64, MNNVL); frontend on a dedicated node | 3 x 8 B300, frontend on first decode node |
| transport env | `UCX_TLS=cuda_copy,cuda_ipc,tcp`, `UCX_CUDA_IPC_ENABLE_MNNVL=y`, `NIXL_UCX_SGL_ENABLE=1`, `NCCL_MNNVL_ENABLE=1`, `NCCL_NVLS_ENABLE=1`, `VLLM_RS_TOKEN_IDS_AS_BYTES=1` | EFA LIBFABRIC, `NIXL_UCX_SGL_ENABLE=0` |
| NIXL connector | `kv_connector_module_path: ...v1.nixl.push_as_nixl`, `backends:["UCX"]` | `transfer_mode: push`, `LIBFABRIC` |
| host KV per prefill rank | `cpu_bytes_to_use` 85,899,345,920 (80 GiB) | 75,161,927,680 (70 GiB) |
| prefill `max-num-batched-tokens` | 32768 | 16384 |
| `enable-cumem-allocator` | true | false (EFA cannot register VMM memory, RCP:2371) |
| decode Engram | `cpu_offload: false` (HBM) | `cpu_offload: true` |
| decode `max-num-seqs` / mbt / graph max | 253 / 2530 / 1518 | 168 / 1680 / 1008 |
| async KV admits per step | 6 | 4 |
| frontend env | + `DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT: '10000000'` | - |
| health/timeouts | `VLLM_ENGINE_READY_TIMEOUT_S 3600`, sidecar deadline 3600 | 7200 / 9000 |

Current GB300 recipe in IX main (#4336, 9P/16P/6P overrides) additionally sets frontend `no-kv-events: true`,
`DYN_ROUTER_SESSION_AFFINITY_MODE: soft`, decode `worker_selection: dynamo-two-tier-cost-fn {cache_threshold: 1.0}`, base host
KV 60 GiB/rank (70 GiB at 9P), comment "Sharing prefill hosts exhausted RAM at c5120 with Engram and host KV offload" for the
dedicated frontend node [FH: IX/.../vllm/gb300-fp4/agentx/disagg-variants.yaml:39-73, 105, 239]. The latest GB300 preset (#5015,
open; mirrored in RCP:1624-1800) adds: prefill EPLB with 2 redundant experts; prefill Engram tables in HBM split over all prefill
GPUs (`peer_group_size` = total prefill GPUs, needs CUDA FABRIC/IMEX); host KV `cpu_bytes_to_use` 182,536,110,080 (170 GiB "per prefill
worker" per runbook); `linear_backend_per_quant {"mxfp8":"torch"}` on prefill; decode `additional-config {"dspark_fused_markov":true}`;
`--max-num-inflight-kv-loads` = max-num-seqs and `--max-async-kv-loads-per-step 3`; gpu-mem 0.92; `VLLM_DEEP_GEMM_WARMUP=relax`;
spec-config `disable_eagle_block_drop: true` [FH: RCP lines cited].

### 3.3 GB200 Dynamo + vLLM, 5P1D DEP4/DEP16 c5600 (row 448128, run 43142466340 attempt 2, IX #4618)

vs 3.2 [FH: row_448128_config.yaml]: prefill DEP4 (`gpus: 4`, `data-parallel-size: 4`), host KV 96,636,764,160 (90 GiB) per rank,
prefill gpu-mem 0.9, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `numa-bind-nodes [0,0,1,1]` on prefill; decode gpu-mem 0.92,
`max-num-seqs 112`, `max-num-batched-tokens 1120`, **`max-async-kv-loads-per-step: 7`, `max-num-inflight-kv-loads: 112`** (flags, not
env), decode Engram in HBM. Frontend = #4336 style (soft affinity, no-kv-events, two-tier decode selector, decode active-request
weight 1e7). Image `vllm/vllm-openai:cu134-nightly-aarch64-01fba62-1111d@sha256:9aa51d63...` (vLLM main 01fba62 + vllm#66552).

### 3.4 vr200 Dynamo + vLLM (rows 448144-448150, run in private `NVIDIA/nv-perf-sa-review` 43515830826)

Best row 448150 has `logs: null` in the reproduce API (no config published); API fields: `kv_offload_backend: mooncake` 0.3.13,
`kv_p2p_transfer: nixl`, Dynamo `1.6.0.dev20261015`, GPU hit 0.098 / CPU hit 0.877, decode KV used 93.6% [FH: reproduce_448150.json, API
metrics]. `gh repo view NVIDIA/nv-perf-sa-review` -> not resolvable (private) [FH]. The published low-concurrency config (row 448144,
1xDEP2 + 3xTP4) shows vr200 deltas: prefill `FLASHINFER_MLA_SPARSE_DSV41` backend, `stream-interval: 20`, `kv_lease_duration: 900`,
host KV 159,987,531,776 (149 GiB), `engram {"cpu_offload":true,"use_thp":true,"dp_shared_memory":false}`, `safetensors-load-strategy lazy`,
`VLLM_USE_DEEP_GEMM=1`; decode env `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP=16`, `VLLM_SD_POSTFWD_WITH_CONNECTOR=1`,
`VLLM_ASYNC_OUTPUT_SPIN_WAIT=1`, `VLLM_W57_*`; frontend `DYN_TOKENIZER_CACHE_EXTEND=1`, `DYN_TCP_POOL_SIZE=2048` [FH:
row_448144_config.yaml]. The 2.99M point's recipe is therefore NOT known; the vr200 hardware generation is not comparable to MI355X.

### 3.5 GB300 Dynamo + SGLang, 9P1D DEP2/DEP16 c7480 (row 448095, run 43029773811 attempt 2, IX #4997)

[FH: row_448095_config.yaml; IX/.../sglang/gb300-fp4/agentx/disagg-dep16-hicache-variants.yaml]
- Frontend: Dynamo kv router, 14400 s affinity, thresholds None (same as vLLM lanes), router 1.7.0.dev20261110.
- Prefill (9 x TP2/DP2/EP2 attention-DP): `--disaggregation-transfer-backend mooncake` (RDMA over `mlx5_0..3`,
  `MC_ENABLE_DEST_DEVICE_AFFINITY=1`, `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True`), **HiCache** on prefill (`--enable-hierarchical-cache
  --hicache-ratio 1.2 --hicache-mem-layout page_first_direct --hicache-io-backend direct --hicache-write-policy write_back`),
  `--chunked-prefill-size 65536 --max-prefill-tokens 32768 --max-running-requests 128 --mem-fraction-static 0.85`,
  `--swa-prefix-tails 2048 --enable-decoder-swa-bounded-replay --enable-encoder-swa-bounded-replay`, `--moe-a2a-backend megamoe
  --dsv4-attn-backend flashmla_mega`, `--stream-interval 32`, Engram host table per rank, `SGLANG_OPT_DSV41_BF16_INDEX_LOGITS=1`,
  `SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=1`, `SGLANG_ENABLE_PREFILL_WAR_READ_DONE=1`.
- Decode (DEP16): `--router-mode round-robin --load-balance-method total_requests`, `--max-running-requests 3072
  --cuda-graph-max-bs-decode 192 --disaggregation-decode-extra-slots 256 --mem-fraction-static 0.75`, EPLB 16 redundant
  (`--eplb-rebalance-num-iterations 15000`), `--enable-fp32-lm-head`, `SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS=1`, stream-interval 32.
- PR #4303 replaced prefill HiCache with Mooncake external linking (80 GB prefill / 140 GB decode segments) and #4595/#4673 moved back to
  prefill HiCache; the current lane is "disagg-dep16-hicache" [2H: PR bodies].

### 3.6 GB300 Dynamo + TensorRT-LLM (open IX #4991; no published row)

[2H: PR body] DEP2 prefill + DEP8/16/32 decode for the throughput points; MegaMoE CuTe DSL target MoE, CUTLASS draft MoE; FP8 KV;
host KV tier on prefill 111-125 GiB per rank bound to GPU NUMA; decode `stream_interval: 20`; DSpark draft 3 at the throughput points
(5 at DEP32); `DYN_ENGINE_CONV_AFFINITY_DP_RANK_SOURCE=dynamo` (start a sub-agent's first turn on the parent's attention-DP rank) "raised the
first-turn prefix hit at c7600 from 0.29 to 0.42 and output throughput at c7000 by 4.9%"; `DYN_TOKEN_IDS_AS_BYTES=1`; per-request perf
metrics in the decode handler cost "39.5% at c7900, 30.9% at c7600, 18.8% at c5100 and about 13% at c7000" when on. Short TEP8/TP4 decode
points use least-loaded decode routing without session affinity ("Session affinity pinned each session to one small-batch decode worker
and left others idle").

## 4. Cross-recipe comparison (best point per HW/engine)

| knob | B300 vLLM 448173 | GB300 vLLM 445572 | GB200 vLLM 448128 | GB300 SGLang 448095 | MI355X vLLM R2 448242 |
|---|---|---|---|---|---|
| mode | PD 5P1D | PD 12P1D | PD 5P1D | PD 9P1D | colocated |
| prefill shape | DEP2 (TP1 DP2 EP2) | DEP2 | DEP4 | TP2 DP2-attn EP2 | (TP1 DP2 attn, TP MoE) |
| decode shape | DEP8 | DEP16 | DEP16 | DEP16 | same server |
| P:D GPUs | 10:8 | 24:16 | 20:16 | 18:16 | 2 |
| KV p2p | NIXL push LIBFABRIC/EFA | NIXL push (push_as_nixl) UCX MNNVL | NIXL push UCX | Mooncake RDMA | none |
| host KV tier | SimpleCPUOffload lazy 70 GiB/rank, prefill only | 80 GiB/rank | 90 GiB/rank | HiCache ratio 1.2 prefill | UMBP standalone 449 GB / 2 GPUs |
| decode prefix cache | off | off | off | (SGLang decode, n/a) | on |
| KV dtype (resolved) | fp8 -> `nvfp4_ds_mla` [FH log] | fp8 | fp8 | (SGLang `v41_new` layout) | fp8_e4m3 (512 B record) |
| prefill mnbt / seqs | 16384 / 64 | 32768 / 64 | 32768 / 64 | chunk 65536, max-prefill 32768 / 128 | 16384 / default |
| long-prefill-token-threshold | 8192 | 8192 | 8192 | - | 0 (default) |
| prefill-schedule-interval | 2 | 2 | 2 | - | 11 |
| prefix-cache-retention-interval | 32768 | 32768 | 32768 | swa-prefix-tails 2048 | 0 (default) |
| decode max-num-seqs / rank | 168 | 253 | 112 | max-running 3072 / 16 = 192 | colocated, default |
| decode graphs | FULL_DECODE_ONLY, steps of 6 to 1008 | to 1518 | to 672 | cuda-graph-max-bs 192 | breakable cudagraph, max capture 1024 |
| async KV admits/step | 4 (env) | 6 (env) | 7 (flag) + inflight 112 | decode-extra-slots 256 | n/a |
| EPLB decode | 16 redundant, symm_mem | 16 | 16 | 16 | n/a (not on ROCm runner) |
| MoE kernel | deep_gemm_mega_moe | same | same | megamoe | aiter (CK a8w4), TP |
| attention | FLASHMLA_MEGA_ATTN_DSV41 | same | same | flashmla_mega | AITER Triton/Gluon sparse MLA |
| indexer | mxfp4 + sparse logits, deep_select top-k | same | same | BF16 index logits | mxfp4 + sparse logits |
| Engram | host, embedding_across_dp, /dev/shm shared | prefill host, decode HBM | prefill host, decode HBM | host table per rank / decode peer shards | host (cpu_offload) |
| gpu-mem util | 0.95 / 0.95 | 0.95 / 0.95 | 0.90 / 0.92 | mem-frac 0.85 / 0.75 | 0.9 |
| router | Dynamo kv, affinity 14400 (hard default) | Dynamo kv + decode active-request weight | Dynamo kv, soft affinity, two-tier decode | Dynamo kv; decode round-robin + total_requests | vllm-router 0.1.14 consistent_hash |
| parsers | in Dynamo frontend (`reasoning-parser None` on engines) | same | same | auto | in vLLM |
| concurrency | 3584 | 9888 | 5600 | 7480 | 384 |

## 5. Flag-by-flag diff: best B300 / GB300 vLLM PD vs MI355X R2 (`configs/r2-dp2-umbp-c384.env`)

R2 flags from `configs/r2-dp2-umbp-c384.env:31-73` (= CI row 448242). "Status on ROCm" is a code read on V unless stated.

| flag / env | B300 448173 | GB300 445572 | MI355X R2 | Status on ROCm vLLM main (evidence) |
|---|---|---|---|---|
| deployment | 5x DEP2 P + 1x DEP8 D, Dynamo | 12x DEP2 + DEP16 | 1 server TP1xDP2 | PD connector paths exist; MoRIIO rejects DSv4.1 (SYNTHESIS B1); NIXL/Mooncake untested on ROCm DSv4.1 |
| `--tensor-parallel-size` | 1 | 1 | 1 | same |
| `--data-parallel-size` | 2 (P) / 8 (D) | 2 / 16 | 2 | DP works (R2) |
| `--enable-expert-parallel` | true | true | off (TP MoE) | EP exists; AITER MegaMoE over MoRI is open-PR territory (IX #5056/#5098, 2H) |
| `--enable-eplb` + eplb-config | D: 16 redundant, symm_mem | same | none | **ROCm DSv4.1 runner: "no redundant EPLB copies"** [FH: V/vllm/models/deepseek_v41/amd/model.py:167-173] |
| `--enable-ep-weight-filter` | true | true | - | not checked on ROCm |
| `--kv-cache-dtype` | fp8 -> nvfp4_ds_mla | fp8 | fp8_e4m3 | FP4 record unavailable on ROCm (SYNTHESIS s2) |
| `--attention-config` backend | FLASHMLA_MEGA_ATTN_DSV41 | same | (default ROCm sparse MLA) | NVIDIA kernel; ROCm uses AITER path |
| indexer_kv_dtype / sparse_logits | mxfp4 / true | same | mxfp4 / true | same |
| `--kernel-config` moe_backend | deep_gemm_mega_moe | same | `--moe-backend aiter` | NVIDIA kernel |
| sparse_indexer_topk_backend | deep_select | same | default | **"deep_select: SM100a/SM103a only"** [FH: V/vllm/config/kernel.py:391] |
| enable_flashinfer_autotune | true | true | - | NVIDIA only |
| `--engram-config` | `{"cpu_offload":true,"embedding_across_dp":true,"use_thp":false}` | P host, D `cpu_offload:false` | `{"cpu_offload":true}` | `embedding_across_dp`, `dp_shared_memory` (ROCm via `VLLM_ROCM_ENGRAM_HIP_VMM=1`) exist [FH: V/vllm/config/engram.py:52-66]; `peer_group_size` needs CUDA FABRIC [FH: :72-77] |
| `--speculative-config` | dspark 5, synthetic 3.51, adaptive off | same | same | same |
| `--gpu-memory-utilization` | 0.95 | 0.95 | 0.9 | tunable; non-torch memory on MI355X TP2 was 31.88 GiB (SYNTHESIS s2) |
| `--max-num-seqs` | P 64 / D 168 | P 64 / D 253 | default | generic |
| `--max-num-batched-tokens` | P 16384 / D 1680 (=10x seqs) | P 32768 / D 2530 | 16384 | generic |
| `--long-prefill-token-threshold` | 8192 (P) | 8192 | 0 | generic [FH: V/vllm/config/scheduler.py:104-110] |
| `--prefill-schedule-interval` | 2 (both roles) | 2 | 11 | generic [FH: scheduler.py:239-245] |
| `--prefix-cache-retention-interval` | 32768 (P) | 32768 | 0 | generic core [FH: V/vllm/config/cache.py:286-292; v1/core/kv_cache_coordinator.py:30-60]; whether ROCm DSv4.1 SWA prefix-cache replay honours it: **not verified** (IX changelog #3690-era said prefix-cache SWA replay forced off on ROCm) |
| `--no-enable-prefix-caching` | D only | D only | on | generic |
| `--compilation-config` | P: VLLM_COMPILE FULL_AND_PIECEWISE + decoder_replay sizes; D: FULL_DECODE_ONLY steps of 6 | same | `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `--max-cudagraph-capture-size 1024` | ROCm DSv4.1 not torch-compiled; FULL_AND_PIECEWISE aborted at MI355X bring-up (IX #2962, 2H) |
| `--max-cudagraph-capture-size` | P 2046 / D 6 x seqs | same | 1024 | - |
| `--kv-transfer-config` P | Multi[NIXL push + SimpleCPUOffload lazy 70 GiB] | Multi[NIXL push UCX + SimpleCPUOffload 80 GiB] | UMBPStoreConnector standalone | NIXL push present on main incl. DSv4.1 CSA-linear; needs identical P/D block sizes and cache-group layouts [FH: nixl/push_worker.py:150-163, 640-680; factory.py:126-130]; SimpleCPUOffload has a HIP batch-copy path (ROCm 7.1+, chunks of 8192 descriptors on gfx950) [FH: v1/simple_kv_offload/cuda_mem_ops.py:53-118]; any non-UMBP child loses the ROCm sparse-MLA fast path (SYNTHESIS C1) |
| `--kv-transfer-config` D | NIXL push kv_consumer | same | - | as above |
| kv_load_failure_policy / lease | fail / 120 s | fail / 120 s | recompute | IX #3950 author flagged `fail`+lease 120 as a risk under deep router queues [2H] |
| async KV admit cap | env 4 | env 6 | - | flags `--max-async-kv-loads-per-step`, `--max-num-inflight-kv-loads` on main [FH: scheduler.py:78-101] |
| `--enable-cumem-allocator` | false | true | default | not checked on ROCm |
| `--numa-bind` / nodes | true / [0,0,0,0,1,1,1,1] | true / [0,0,1,1] | - | generic option; MI355X GPU-NUMA map must be measured (E0) |
| `--reasoning-parser` | None (frontend parses) | None | deepseek_v41 | moving parsing to the router only possible if router supports it |
| `--language-model-only`, tokenizer-mode | yes | yes | yes | same |
| `VLLM_USE_V2_MODEL_RUNNER=1` | yes | yes | - | not checked whether ROCm DSv4.1 path uses V2 runner |
| `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1` | yes | yes | - | generic env [FH: V/vllm/envs.py:1638-1641] |
| `VLLM_ALLREDUCE_USE_SYMM_MEM=0`, `NCCL_CUMEM_ENABLE=1` | yes | yes | - | NVIDIA comm envs |
| `VLLM_ROCM_*`, `AITER_*`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `VLLM_USE_RUST_FRONTEND=1` | - (Rust frontend used: log "RustFrontend") | - | yes | ROCm-specific |
| router | Dynamo 1.6.0 kv mode, affinity 14400 s, conditional disagg off, temperature 0 | + decode active-request weight 1e7 | vllm-router consistent_hash, intra-node DP 2 | Dynamo frontend is CPU/Rust; Dynamo has ROCm import-compat fixes "from AMD MI355X PoC" (ai-dynamo/dynamo#9929 merged 2026-05-31) [FH: PR title]; Dynamo + vLLM ROCm sidecar not tested by us |
| tokenizer | Dynamo fastokens + 8 GB tokenizer cache | same | vLLM Rust frontend | - |
| concurrency | 3584 | 9888 | 384 | - |

## 6. Dynamo router semantics that the recipes rely on [FH: DYN docs @ beb1a41]

- Cost (default selector): `prefill_load_scale * adjusted_prefill_blocks + potential_decode_blocks + decode_active_request_weight *
  active_requests`; `overlap_score_credit` (default 1.0) credits device-local prefix overlap against prefill load
  [DYN/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md:150-165, 700-722].
  GB300's `DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT=10000000` therefore makes decode selection effectively least-active-requests [INF].
- `dynamo-two-tier-cost-fn`: "Ranks on two tiers ... active-request load first, then device-KV prefix overlap"; `cache_threshold` is the
  fraction of blocks that must be device-resident before the cache tier applies; GB300/GB200 set 1.0 for decode [same file:120, 220-235].
- Session affinity: binds session -> worker **and data-parallel rank**; `hard` (default) or `soft`; parent-session binding for
  sub-agents (experimental, v1.6) [same file:478-525]. Frontend log of 448173 routes Prefill to `(worker_id, dp_rank)` and Decode to a
  `dp_rank` per request [FH: notes/dynamo/logs/448173_*frontend_0.out.txt:351-380].
- Conditional disaggregation off: `active-*-threshold None` so "every request is prefilled on a prefill rank and never locally on one of
  the 16 lockstep MegaMoE decode ranks" [2H: IX #3950].

## 7. Levers transferable to MI355X vLLM (each with ROCm status and the experiment that decides it)

Ranked by how directly the evidence ties them to throughput. "Gate" = measurement needed before claiming any gain.

| # | Lever (source) | MI355X status | Gate / experiment |
|---|---|---|---|
| L1 | **Shape: N x small prefill workers + 1 wide decode, P:D by Little's law** (all rows; #4337, #4673) | prefill = R2's TP1xDP2 server is the natural P worker; decode must be DP-wide; ROCm lacks EPLB redundancy, MegaMoE, monokernel-with-DP | E6: decode step vs batch at DP4/DP8 (+EP if usable) and prefill req/s per GPU at the R2 shape; then size P:D and concurrency from measured request rate x decode time |
| L2 | **Decode slot cap + admission control**: `max-num-seqs` = equilibrium batch, `max-num-batched-tokens = 10 x seqs`, FULL_DECODE_ONLY graphs in steps of 6 up to 6 x seqs, `--max-async-kv-loads-per-step` 2-7, `--max-num-inflight-kv-loads = max-num-seqs` (#61725) | scheduler flags generic on main; ROCm uses breakable cudagraph, so the graph-size list must be checked against what ROCm captures | E7/E8 sweep of seqs x admits; log `Reqs Running/Waiting/Deferred/KV fetch` as in 448173 |
| L3 | **Prefill-only host KV tier**, sized 70-170 GiB per prefill rank (SimpleCPUOffload lazy) or HiCache; decode prefix cache off | UMBP (fast path kept only alone); SimpleCPUOffload HIP path exists; under a PD connector the fast path is lost anyway (C1), so UMBP-vs-SimpleCPUOffload becomes a pure capacity/latency choice | E4 (fast-path cost) + A/B UMBP vs SimpleCPUOffload on the P worker; record CPU hit and `External prefix cache hit rate` |
| L4 | **Prefill scheduler knobs**: `long-prefill-token-threshold 8192`, `prefill-schedule-interval 2` on P (R2 uses 11 colocated), `prefix-cache-retention-interval 32768`, `max-num-seqs 64`, mbt 16k-32k | generic; retention-interval behaviour for ROCm DSv4.1 SWA groups not verified | also testable colocated on R2 now (cheap): R2 + `--long-prefill-token-threshold 8192` / `--prefix-cache-retention-interval 32768` |
| L5 | **KV-aware, DP-rank-granular routing with session affinity** (Dynamo kv mode, TTL 14400, soft; decode by least active requests; sub-agent first turn on parent's DP rank: +4.9% on TRT-LLM c7000, 2H) | vllm-router consistent_hash today; Dynamo frontend could front ROCm vLLM via the gRPC sidecar (untested); vllm-router has a PD mode (`--vllm-pd-disaggregation`, RCP:2080) | E7: run both routers on 1P1D; compare prefill prefix hit and per-rank running-count spread |
| L6 | **NIXL push (WRITE from P) + one region per packed MLA row** (#61121 open; blog: 141K -> 440K tok/s/GPU on GB300 8P1D c3072, 2H) | NIXL push on main with DSv4.1 CSA-linear; ROCm NIXL+UCX wheel stage in Dockerfile.rocm (C3); not run on ROCm | E7 connector bring-up: Mooncake first (SYNTHESIS 4.3) but add NIXL push as a candidate since the NVIDIA lanes all use it |
| L7 | **Decode DP load balance by request count** (SGLang `total_requests`: running spread 2-7 vs 11-26 with total_tokens; vLLM blog: step varies little with context so least-load by request count) | needs MI355X evidence that decode step is context-insensitive (blog numbers are GB300) | E6 decode step vs context on MI355X |
| L8 | **Frontend/CPU overheads**: parsers once in the router, Rust frontend + fastokens + tokenizer cache, token ids as bytes, recursive-doubling CPU all-reduce for DP sync (#61767 merged), prompt int32 buffers (#62702 merged), skip SWA null heads (#61724 merged), no per-request perf metrics (TRT-LLM -13..-39%, 2H) | merged items are on main; whether the R2 image (60bd6dd2) contains them: check `git merge-base` per PR | profile CPU time per step at c384+ on R2 and on PD decode |
| L9 | **gpu-memory-utilization 0.95 / Engram host tables shared across DP** | R2 at 0.9; Engram shared host tables on ROCm via `VLLM_ROCM_ENGRAM_HIP_VMM=1` | E5 KV size dump at 0.9 vs 0.95 |
| L10 | **DEP2 over DEP4 for prefill** (blog: DEP4 = 83% of DEP2 per GPU; imbalance loss 13.6% vs 5.9%, GB300, 2H) | matches R2's DP2; untested at DP4 on MI355X | mirror/lag replay method if DP4 is considered |
| L11 | **Benchmark discipline**: 3600 s only (fast-run factor ~3.8x in #4673, 1.47x on MI355X), dedicated client/frontend node when host RAM is shared with Engram + host KV (GB300 OOM at c5120) | applicable directly | - |
| L12 | **Kernel-side items behind the NVIDIA numbers** (layer-20 bounded replay #59894 +19.9% at 32K; DeepSelect prefill top-k #62240 +15.6%; Engram 2 MB pages #62359 +2.8%; MegaMoE; EPLB symm_mem #62009; DSpark heterogeneous EPLB #61395) | #59894 generic replay logic (ROCm uses bounded replay); DeepSelect SM100-only; EPLB not on ROCm runner | kernel work items, measured with `vllm bench forward` (#61339) |

Not transferable as-is (NVIDIA-only): FLASHMLA_MEGA_ATTN_DSV41, deep_gemm_mega_moe, deep_select, flashinfer autotune, symm_mem EPLB
communicator, MNNVL/IMEX Engram peer groups, nvfp4_ds_mla KV, EFA/LIBFABRIC envs.

## 8. Gaps (not established)

- G1: vr200 2.99M recipe (private repo); GB300 vLLM #5015 throughput numbers; TRT-LLM numbers are internal (2H).
- G2: semantics of `server_{gpu,cpu}_cache_hit_rate` in PD rows (constant 0.48/0.50 on B300) and IX #4399 caveats - not read.
- G3: whether `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` (env, B300/GB300 rows) and `--max-async-kv-loads-per-step` (flag) behave identically.
- G4: ROCm behaviour of `prefix-cache-retention-interval` for DSv4.1 SWA/compressor groups; whether ROCm DSv4.1 uses the V2 model runner.
- G5: Dynamo + vLLM ROCm sidecar never run by us; MI355X PoC (#9929) contents not read.
- G6: why B300 decode at the best point was cap-bound (1340/1344) while TTFT mean was 8.5 s: no measurement separates router queueing
  from prefill queueing from KV fetch wait; the logs show `Waiting 619-639`, `KV fetch 254-355 waiting to start`.
- G7: the B300 prefill per-rank token capacity (47.9M) vs decode (188.4M) implies different bytes/token (1,985 vs 1,096 B); cause not read.

## Verification (adversarial)

Verifier pass 2026-11-19. Every source re-opened: API `/api/v1/benchmarks?model=DeepSeek-V4.1-Flash&sequence=agentic-traces`
re-fetched (`notes/dynamo/verify/bench.json`), `/api/v1/reproduce?id=448145|448148|448149|448150` (`notes/dynamo/verify/rep_*.json`),
row configs, saved server logs, PR bodies, `gh api` on run 43769338888 and on vllm-project/vllm @96710245, vLLM main clone @7a62ca0.

| # | Claim | Verdict | Correct fact / caveat (evidence) |
|---|---|---|---|
| V1 | B300 828,209, 5xDEP2+DEP8, 18 GPUs, c3584, p90i 83.8, TTFT 8.5 s, 11-17, image 96710245, vLLM 0.34.0rc1.dev142, Dynamo 1.6.0.dev20261010 | **confirmed** | API row 448173: tput_per_gpu 828,209.3, p90_intvty 83.82, mean_ttft 8.508, mean_qps 109.95, P 10 GPU/5 workers TP2 EP2, D 8 GPU TP8 EP8 [FH]. Version string `0.34.0rc1.dev142+g96710245e.d20261110` and `ai_dynamo-1.6.0.dev20261010` in decode log [FH]. IX #4585 merged 2026-11-17T01:41Z [FH]. Note: the run (attempt 2) actually executed 2026-11-16T18:15Z -> 2026-11-17T00:43Z [FH: gh api runs/43769338888/attempts/2]; API `created_at` 01:41 is ingestion time, so the 11-16 log timestamps are consistent. |
| V2 | GB300 1,011,514, 12xDEP2+DEP16, 40 GPUs, c9888, p90i 75.5, 10-20; host KV 80 GiB/P rank; D max-num-seqs 253; 6 admits | **confirmed** | Row 445572: 1,011,513.5, p90i 75.47, qps 298.1, mean TTFT 7.78 s, image `cu134-nightly-aarch64-dev-20261019` (vLLM `0.32.0rc1.dev281+g49cb2bbf7`) [FH API; gb300_decode_head.txt]. `cpu_bytes_to_use` 85,899,345,920 = 80 GiB exactly; `max-num-seqs: 253`; `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP: '6'` [FH row_445572_config.yaml:99,182,152]. Run 39207089292 attempt 12 [FH reproduce_445572.json]. GB300 decode log does NOT flag the env as unknown (it flags VLLM_RPC_TIMEOUT, VLLM_RS_TOKEN_IDS_AS_BYTES, VLLM_TORCH_PROFILER_DIR) [FH], so on that dev image the env was registered. |
| V3 | All NVIDIA vLLM PD recipes: Multi[NIXL push + SimpleCPUOffload lazy] on P only, NIXL consumer + no prefix cache on D, Dynamo kv mode, TTL 14400, thresholds None | **corrected** | True for every row with a published config (448172/448173 B300, 445571-445573 GB300, 448128 GB200, 448144 vr200) [FH configs]. Two corrections: (a) push is selected two different ways: GB300 rows + vr200 448144 use the out-of-tree module `...nixl.push_as_nixl` (from dev image `nightly-x86_64-dev-20261012-59e75f4` per vLLM's own recipe, `source/vllm/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml:2346-2348`), B300/GB200 use `"transfer_mode":"push"` which main's factory maps to NixlPushConnector [FH factory.py:126-130]; `push_as_nixl` is absent from main [FH grep]. NIXL backend is LIBFABRIC on B300 (EFA), UCX on GB300/GB200. (b) "All" is not established: vr200 rows 448148, 448149, 448150 (incl. the 2,993,171 best) report `kv_offload_backend: mooncake`, not vllm-simple, and publish no config/logs [FH rep_448148..448150.json]; vr200 448144/448145 decode is 3-4 x TP4 without DP attention, not DEP [FH]. GB200/GB300/vr200 additionally set `DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT=10000000`; B300 does not [FH]. |
| V4 | B300 best point decode at cap 1340/1344, Waiting 619, KV fetch 254, KV 20.1%, AL 3.50; prefill 87.5-98.5% hit, 7-9% KV | **corrected (minor) + caveat** | Decode lines 11-16 21:58:40/41: Running 1340/1342, Waiting 619/639, Deferred 365/284, KV fetch 254/355 waiting, GPU KV 20.1%/19.3%, Mean acceptance length 3.50; 168x8=1344 [FH late_*decode*.txt]. Only **2** decode stat lines were sampled; treat as a snapshot, not a time-average. Prefill w0 over 13 lines (21:44): prefix hit **87.5-98.9%**, GPU KV **3.9-9.2%** [FH mid_*prefill*.txt]; one prefill worker of five sampled. |
| V5 | B300 capacity: P 88.54 GiB = 47.9M tok, D 192.39 GiB = 188.4M tok, nvfp4_ds_mla, 70 GB CPU per DP rank (593,133 blocks), P weights 149.21 GiB/GPU | **confirmed with wording fix** | All numbers present in logs [FH]. Fixes: config says `kv-cache-dtype: fp8` [row_448173_config.yaml:106,162]; the runtime switches to "DeepSeek's nvfp4_ds_mla KV cache format" (attention.py:231 / cache.py:512 log lines) [FH]. 149.21 GiB is "Model loading took 149.21 GiB memory" (P, EP2) vs 46.94 GiB (D, EP8), not a pure weight figure; P profile reports 157.2 GiB "weights + non-torch" [FH]. CPU tier logs "593133 CPU blocks (70.00 GB)"; configured 75,161,927,680 B = 70 GiB [FH]. |
| V6 | Little's-law sizing; D mbt = 10 x seqs, graphs steps of 6 up to 6 x seqs, seqs on MegaAttn wave boundaries | **confirmed [2H]** | IX #4337 body: "max-num-seqs sits on MegaAttn wave boundaries; batched tokens are max-num-seqs x 10 and CUDA graph sizes go up to max-num-seqs x 6 in steps of 6"; "conc is sized by Little's law from that run's think time". IX #4673: "Prefill stays saturated at every point ... (Little's law: running requests = request rate x decode time per request)" and "decode CUDA graph cap sits at the per-rank equilibrium batch" (SGLang) [2H PR bodies]. |
| V7 | MI355X 3.73 req/s/GPU vs B300 6.11; 641k needs ~5.6 | **confirmed** | 448242 mean_qps 7.4639 / 2 = 3.73; 448173 109.954 / 18 = 6.11 [FH API]. Check: 427,451 / (113,745 in + 906 out) = 3.73, so 641,177 / 114,651 = **5.59** req/s/GPU [INF arithmetic, assumes the same token mix]. B300's mean input is 134,602 vs MI355X 113,745 (traces differ per run), so req/s is not apples-to-apples across rows. |
| V8 | ROCm DSv4.1 runner has no redundant EPLB experts; DeepSelect SM100a/SM103a only | **confirmed** | amd/model.py:167-173 comment + `n_redundant_experts = 0`; `update_physical_experts_metadata` asserts local count unchanged (:1444-1449) [FH]. kernel.py:391 docstring; enforcement `has_deep_select()` requires `is_cuda()` and capability family 100 [FH sparse_mqa_logits.py:46-51]. |
| V9 | NIXL push on main handles CSA-linear, requires identical block sizes and cache-group layouts; SimpleCPUOffload HIP batch path ROCm 7.1+, 8192 chunk on gfx950 | **confirmed** | push_worker.py:150-163 (`NotImplementedError` on block size mismatch), :651-655 (group layout), :673 `_is_csa_linear` [FH]; cuda_mem_ops.py: `hipMemcpyBatchAsync`, "requires ROCm 7.1+", `_ROCM_DEFAULT_MAX_BATCH_DESCRIPTORS = 8192` "(MI350X/gfx950)", override `VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS`, optional `hipMemcpyFlagExtPreferCE` (SDMA, HIP 7.16+) [FH]. Not run on ROCm by us. |
| V10 | Scheduler knobs generic on main; #61725 merged 11-10; `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` absent from main | **confirmed, plus a new refutation of the recipe table** | #61725 MERGED 2026-11-10T03:06Z, merge dc74b8c; image commit 96710245 is 18 commits ahead of it [FH gh]. Flags at scheduler.py (offsets 78-110), used at v1/core/sched/scheduler.py:1076,1095,2803 [FH]. Env absent from main `vllm/` [FH grep]. **New:** the B300 best-run decode log prints `Unknown vLLM environment variable detected: VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` [FH 448173 decode_w0 log, 11-16 19:32:10], the 96710245 envs.py/scheduler.py contain no reader [FH gh api], and the B300 config sets no `--max-async-kv-loads-per-step` [FH row_448173_config.yaml]. So the "async KV admit cap env 4" row of section 5 did not take effect on B300 448173 (cap effectively unset). GB200 448128 uses the real flags (`max-async-kv-loads-per-step: 7`, `max-num-inflight-kv-loads: 112`) [FH row_448128_config.yaml:191,200]. G3 is thereby partly answered. |
| V11 | GB300 SGLang 1,071,944: Mooncake RDMA, P HiCache 1.2 write_back, D round-robin + total_requests, extra-slots 256, max-running 3072, EPLB 16; 2-7 vs 11-26 spread | **confirmed with provenance caveat** | Row 448095: 1,071,943.7, c7480, p90i 105.6, TTFT 5.96 s, 9xDEP2 + DEP16, Dynamo 1.7.0.dev20261110 [FH API/reproduce]. Config: `MOONCAKE_PROTOCOL: rdma`, `hicache-ratio: 1.2`, `hicache-write-policy: write_back`, decode `router-mode: round-robin` (SGLang DP router inside the decode worker; the Dynamo frontend is `router-mode: kv`), `load-balance-method: total_requests`, `disaggregation-decode-extra-slots: 256`, `max-running-requests: 3072`, `cuda-graph-max-bs-decode: 192`, `ep-num-redundant-experts: 16` [FH row_448095_config.yaml:56,82,107,116-119,168-199]. The 2-7 vs 11-26 statement is in IX #4673 (20261030 nightly) whereas row 448095 is from IX #4997 (20261111 nightly) [2H]. |
| V12 | TRT-LLM GB300 PD internal 5xDEP2+DEP8 c7000 1,492,133; sub-agent first turn on parent's DP rank 0.29->0.42, +4.9%; per-request metrics cost 13-39.5% | **corrected** | IX #4991 OPEN, created **2026-11-10**, last updated 2026-11-16 (not authored 11-16) [FH gh]. c7000 1,492,133 is median-interactivity 87.7 tok/s/user (not p90), decode batch 320, **DSpark draft 3** (vLLM rows use 5), and covers only the **first 3,068 s** (job hit the 2-hour limit; from per-request export) [2H PR body]. Prefix-hit 0.29 -> 0.42 was measured at **c7600** and +4.9% output at **c7000**, via `DYN_ENGINE_CONV_AFFINITY_DP_RANK_SOURCE=dynamo` [2H]. The 13-39.5% loss is TRT-LLM `record_stats` on earlier Dynamo builds (off by default since ai-dynamo/dynamo#18170) - engine-specific, no vLLM measurement [2H]. Host KV per DEP2 P rank 125 GiB at DEP8/DEP16 points [2H]. |

Net changes to the body above: section 5 row "async KV admit cap ... env 4" should read "env 4 set but inert (unknown env, no flag)" for B300;
section 0/7 statements of "all NVIDIA vLLM PD rows" should exclude vr200 448148-448150 (Mooncake offload, no config); L5's "+4.9% on TRT-LLM
c7000" is from a c7000 run truncated at 3,068 s.
