# Dynamo + vLLM for DSv4.1-Flash AgentX: what to absorb into the MI355X vLLM PD plan

Date: 2026-11-19. Synthesis of `research/dynamo/{recipes,vllm_side,framework,perf_anatomy}.md` including their adversarial
Verification sections (where a body and its verification disagree, the verification wins and is applied here). Research only:
no GPU, no docker.

Tags: **[FH]** first-hand (API row id, launched config, server log line, code file:line, PR read); **[2H]** second-hand
(PR author / vLLM blog claim on NVIDIA hardware, not re-measured); **[INF]** our arithmetic or inference, unverified.
Paths: `V` = `source/vllm` (main 7a62ca0666, 2026-11-19); `IX` = `source/InferenceX` (db3fc796; origin/main 6ed3774 has no
dsv41flash recipe change); `N` = our image's vLLM commit 60bd6dd2 (main 2026-11-16, `vllm/vllm-openai-rocm:nightly-60bd6dd2...`);
`RCP` = `V/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`. Launched NVIDIA configs: `notes/dynamo/configs/row_<row>_config.yaml`;
logs: `notes/dynamo/logs/`, `notes/dynamo_perf/logs/`. All rows: AgentX `semianalysis_cc_traces_weka_062126`, FP4 checkpoint,
DSpark 5 drafts, synthetic AL 3.51, 3600 s, metric tok/s/GPU over (P+D) GPUs.

Rows used throughout [FH: InferenceX API, re-fetched in `notes/dynamo/verify/bench.json`, `notes/dynamo_perf/verify/bench_refetch.json`]:

| Row | HW | Mode / shape | GPUs P+D | conc | tok/s/GPU | p90i | TTFT / TPOT | image (vLLM) | date |
|---|---|---|---|---:|---:|---:|---|---|---|
| 448242 | MI355X | vLLM colocated TP1xDP2 + UMBP (**the bar**) | 2 | 384 | 427,451 | 15.8 | 2.20 s / 50.0 ms | rocm nightly-60bd6dd2 | 11-18 |
| 448173 | B300 | Dynamo+vLLM 5x DEP2 P + DEP8 D | 10+8 | 3584 | 828,209 | 83.8 | 8.51 s / 11.2 ms | cu134-nightly-96710245 (0.34.0rc1.dev142), Dynamo 1.6.0.dev20261010 | run 11-16, ingested 11-17 |
| 448172 | B300 | same, 4x DEP2 P (1 node) + DEP8 D (1 node) | 8+8 | 2048 | 738,119 | 120.5 | 1.41 s / 7.4 ms | same | same run |
| 447890 | B300 | vLLM colocated DEP8 | 8 | 1280 | 625,966 | 35.9 | 4.53 s / 21.5 ms | cu134-nightly-a2bcb565 | 11-10 |
| 447896 | GB300 | vLLM colocated DEP4 | 4 | 1024 | 863,068 | 26.1 | 4.42 s / 29.6 ms | - | - |
| 445572 | GB300 | Dynamo+vLLM 12x DEP2 P + DEP16 D | 24+16 | 9888 | 1,011,514 | 75.5 | 7.78 s / 12.0 ms | cu134-nightly-aarch64-dev-20261019 (main 49cb2bbf7 + dev patches) | 10-20 |
| 448128 | GB200 | Dynamo+vLLM 5x DEP4 P + DEP16 D | 20+16 | 5600 | 817,333 | 125.6 | 3.99 s / 7.5 ms | cu134-nightly-aarch64-01fba62-1111d | 11-13 |
| 448095 | GB300 | Dynamo+SGLang 9x DEP2 P + DEP16 D | 18+16 | 7480 | 1,071,944 | 105.6 | 5.96 s / - | sglang nightly-dev-cu13-20261111 | 11-13 |

vr200 2,993,171 (row 448150) has no published config or logs (private `NVIDIA/nv-perf-sa-review`; reports Mooncake host
offload, not SimpleCPUOffload) [FH: `notes/dynamo/verify/rep_448150.json`] and is not used as a template. GB300 TRT-LLM
1,492,133 is an open PR (IX #4991) with internal numbers, median (not p90) interactivity, DSpark draft 3, first 3,068 s only [2H].

---

## 1. What Dynamo + vLLM does for DSv4.1 that we do not (ranked by measured / estimated contribution)

Ranking is by the size of the published evidence; the magnitudes come from different rows, hardware and images and are
**not additive**. No NVIDIA source publishes a per-knob ablation of a PD row [FH: framework.md O1].

| # | What they do | Size of evidence | Tag |
|---|---|---|---|
| 1 | **Disaggregate: N x DEP2 prefill workers + ONE wide DEP decode**, P:D sized by Little's law (best points at P:D GPU 1.0-1.5:1) | same-SKU PD / colocated: B300 828,209 / 625,966 = **1.323x**; GB300 1,011,514 / 863,068 = **1.172x**. B300 per-request decode rate colocated ~46-52 tok/s -> PD decode ~84-89 tok/s (**~1.9x**) | [FH rows 447890/448173/447896/445572; logs, verified V3a/V3c in perf_anatomy.md] |
| 2 | **Wide-EP decode (DEP8/16) to free HBM for KV, with EPLB (16 redundant experts, symm_mem)** | B300 D DEP8: 46.94 GiB weights, **192.39 GiB KV per GPU**; MI355X bar DP2: 150.82 GiB weights, **61.02 GiB KV per GPU** (3.15x). EPLB on GB300 decode: **+12.0% DEP16, +25.0-26.2% DEP32** | [FH startup logs 448173 / 448242]; [2H vllm#62009 body] |
| 3 | **CUDA kernels behind the colocated gap** (FlashMLA MegaAttn, DeepGEMM MegaMoE, DeepSelect top-k, torch.compile FULL_AND_PIECEWISE graphs) | colocated vs colocated: B300 DEP8 / MI355X = **1.464x**, GB300 DEP4 / MI355X = **2.019x**, confounded with shape, scheduler, interactivity | [FH rows; INF split not possible] |
| 4 | **Prefill kernel fixes**: layer-20 bounded replay #59894, Engram 2 MB pages #62359, DeepSelect prefill top-k #62240 | new-token prefill **+35.6-43.9%** (4K-64K new on 128K cached; 64K: 71.2K -> 102.5K tok/s/GPU), GB300 | [2H vLLM blog 2026-10-23] |
| 5 | **Prefill at DEP2, not DEP4** | per-GPU prefill **64.4K vs 53.3K tok/s/GPU** (+21%); DP-imbalance loss 5.9% vs 13.6% | [2H blog, GB300, AgentX prefill-only c256] |
| 6 | **CPU-path fixes for wide DP decode with a KV connector** (async KV admit cap #61725, recursive-doubling DP all-reduce #61767, int32 prompts #62702, SWA null-head skip #61724, chunked staged writes #62698) | **460K -> 516K** tok/s/GPU (+12%, three fixes together, GB300 7P1D c4800); DEP32 DP sync 10.8 -> 1.5 ms/step | [2H blog; PR bodies] |
| 7 | **NIXL push with packed DSv4.1 MLA rows** (#61121, still OPEN) | **141K -> 440K** tok/s/GPU on GB300 8P1D c3072 (fixes a transfer path, not a general gain); per 100k transfer 179.2 -> 103.6 MB, 4.56 -> 2.83 ms | [2H blog; #61121 body] |
| 8 | **Prefill-only host-DRAM KV tier** (SimpleCPUOffload lazy, **70 GiB per prefill GPU** on B300, 80 GiB on GB300) + decode prefix caching OFF | B300 PD hit GPU 0.466 + CPU 0.510 (API; the constant 0.48/0.50 split across B300 points may be a metric artifact) | [FH row 448173; log `per_rank=70.00 GB, world_size=1`]; contribution not isolated |
| 9 | **KV-event-aware router at (worker, DP-rank) granularity + session affinity TTL 14400 s; decode by fewest active requests** | no vLLM ablation; TRT-LLM: sub-agent first turn on the parent's DP rank raised first-turn prefix hit 0.29 -> 0.42 (c7600) and output +4.9% (c7000) | [FH configs]; [2H IX #4991] |
| 10 | **Decode slot cap at the equilibrium batch** (`max-num-seqs`), mbt = 10 x seqs, FULL_DECODE_ONLY graphs in steps of 6 up to 6 x seqs | B300 best point: decode Running 1340-1342 of cap 1344, Waiting 619-639, KV used 19-20% (2 log lines, snapshot) | [FH 448173 decode log]; [2H IX #4337/#4673 sizing method] |
| 11 | **Prefill scheduler knobs**: max-num-seqs 64, mbt 16K-32K, long-prefill-token-threshold 8192, prefill-schedule-interval 2, prefix-cache-retention-interval 32768, gpu-memory-utilization 0.95 | no isolated number | [FH configs] |
| 12 | **Frontend CPU path**: Rust frontend with fastokens + 8 GB tokenizer cache, parse reasoning/tools once in the frontend, token ids as bytes, frontend on a dedicated node/NUMA | TRT-LLM per-request decode metrics cost 13-39.5% (engine-specific); GB300 recipe notes host RAM exhaustion at c5120 when the frontend shared prefill hosts | [2H IX #4991; FH IX gb300 yaml comment] |

What the best B300 row did **without** [FH: 448173 decode/prefill logs, verified in recipes.md V10 and vllm_side.md]:
the async KV admit cap (`VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP=4` printed as "Unknown vLLM environment variable", no
`--max-async-kv-loads-per-step` flag), EPLB on the DSpark draft (#61395 newer than the image), Engram cuMem FABRIC (fell back
to /dev/shm). So 828k on B300 does not depend on those three levers. Also unused by every NVIDIA DSv4.1 run: KVBM, the Dynamo
planner, conditional disaggregation, DBO/microbatching [FH: run configs].

What is already in our image N (2026-11-16) [FH: GitHub compare API, vllm_side.md s0]: all 13 merged PRs the vLLM blog links
(#61725, #61767, #62702, #61724, #62698, #59894, #62359, #62240, #62009, #61395, #61868, #61869, #61339). Not in N: #61121
(open), #65595 (ROCm HIP-VMM Engram, merged 11-18), #66795 (NIXL push stall fix, merged 11-19). The gap is therefore not
"missing main commits" in general; it is (a) code that only runs on CUDA and (b) recipe-level choices we do not run.

---

## 2. Gap decomposition

### 2.1 Throughput identity [FH rows; identity is definitional]
tok/s/GPU = (conc/GPU) x (mean ISL+OSL) / (conc/qps). B300 PD / MI355X bar = 1.037 (conc/GPU 199 vs 192) x 1.577 (per-lane
cycle 51.4 s -> 32.6 s) x 1.183 (mean request 114,651 -> 135,614 tokens) = 1.934. GB300 PD / bar = 1.287 x 1.548 x 1.184 = 2.36.
- **Concurrency per GPU is not where NVIDIA wins**; per-request speed is: TPOT 50.0 ms (bar) vs 11.2 ms (B300 PD); e2e 47.2 s vs 19.7 s.
- The cycle falls less than e2e (1.58x vs 2.4x) because time out of flight per request is 4.2 s on the bar vs 10-17 s on every
  other row (AgentX replays recorded inter-turn delays). Why the bar point shows only 4.2 s is open (perf_anatomy Q-A).
- The 1.18x larger mean request on NVIDIA rows is not explained (perf_anatomy Q-B); traces are sampled per run.

### 2.2 Ratio chain (measured row ratios; what each contains is confounded) [FH rows; INF chain]
| Step | Ratio | Contains |
|---|---:|---|
| A. MI355X colocated -> B300 colocated DEP8 | 1.464 | hardware (FP4 1.34x, scale-up link 1.68x, HBM parity, FP8 0.89x [2H specs]), CUDA kernels, DEP8 shape (47.6 vs 150.8 GiB weights/GPU), scheduler (interval 96 + mbt 65536 vs 11 + 16384), p90i 35.9 vs 15.8 |
| B. B300 colocated -> B300 PD | 1.323 | PD isolation, prefill host tier, Dynamo router, 18 vs 8 GPUs, newer image |
| C. B300 PD -> GB300 PD | 1.221 | NVL72 DEP16, MNNVL KV path, Grace host, 40 GPUs, older image |
| A' MI355X colocated -> GB300 colocated DEP4 | 2.019 | as A plus Grace/C2C, 4 GPUs |

Of the B300 PD gap (1.94x), 1.46x already exists between colocated single servers and 1.32x is PD on the same SKU. Which
resource binds on MI355X colocated (prefix capacity, decode step, prefill interference) is not measured (SYNTHESIS U3, E3).

### 2.3 Per-request GPU-time budget for the +50% target (new planning tool) [INF arithmetic on FH inputs]
For a PD deployment whose P and D GPU counts are balanced, req/s per GPU r satisfies 1/r = c/p + o/g, where c = computed
(uncached) prompt tokens per request, p = computed prompt tok/s per prefill GPU, o = output tokens per request, g = output tok/s
per decode GPU. Cross-check on B300 448173: c = 5 workers x 112.2k (prefill w0 window mean, assumed equal for w1-w4) / 109.95
qps = 5.10k; p = 56.1k; o = 13,908 x 8 / 109.95 = 1,012; g = 13,908 (API out/Dgpu) -> 1/r = 0.091 + 0.073 = 0.164 GPU-s -> r = 6.11,
matching the API 6.11 req/s/GPU.

Target: 641,177 tok/s/GPU at the bar's token mix (113,745 in + 906 out) = **5.59 req/s/GPU -> budget 0.179 GPU-s per request**
(on one 8-GPU node 44.7 req/s; on 16 GPUs 89.5 req/s). With c between 4.4k (engine prompt counter, 3.9% computed) and 6.8k (API,
6.0% uncached) - the two disagree, perf_anatomy s3.4:

| decode g (out tok/s per D GPU) | o/g (s) | required prefill p (computed tok/s per P GPU), c = 4.4k | c = 6.8k |
|---:|---:|---:|---:|
| 13.9k (B300 PD level) | 0.065 | >= 38.7k | >= 59.8k |
| 10k | 0.091 | >= 49.9k | >= 77.1k |
| 7k | 0.129 | >= 89.1k | >= 137.7k |

For reference, the bar's colocated server runs ~17k prompt tok/s + ~3.6k output tok/s per GPU at the same time (log window means)
[FH perf_anatomy s3.1/3.3]; neither p nor g alone has been measured on MI355X. Consequences for the plan [INF]: (i) **p and g
are the two gating measurements** (E12, E13); (ii) the target is reachable by PD only if measured p and g put c/p + o/g <= 0.179
plus transfer/router overhead; (iii) reducing c (prefix hit: router + host tier, levers L7/L8) relaxes p one-for-one. Caveats:
ignores transfer time, router queueing, and integer P/D granularity; assumes the bar's token mix.

Decode seats implied (Little's law) [INF]: at 44.7 req/s on 8 GPUs and B300-like 84 tok/s per request, decode holds
~44.7 x 906 / 84 = ~480 running requests.

---

## 3. Lever table

Effort: S = flags only; M = bring-up / new config / image swap; L = vLLM code work; XL = kernel work.
"Expected effect" is a hypothesis with its reasoning, never a prediction of size [INF].

| ID | Lever | Dynamo/vLLM evidence (number, source) | ROCm / MI355X status (code-cited) | Effort | Expected effect for us [INF] | Exp |
|---|---|---|---|---|---|---|
| L1 | PD shape: N x DP2 (TP1) prefill workers + one wide DP decode; P:D by Little's law, 1.0-1.5:1 GPUs | B300 1.323x over own colocated; 2-node 4P+1D 738,119 @ p90i 120.5 = 1.73x the bar [FH rows 448172/448173]; sizing method [2H IX #4337/#4673] | R2's TP1xDP2 server is the natural P worker; no DSv4.1 vLLM PD run on any AMD GPU (SYNTHESIS s0.2); MoRIIO rejects DSv4.1 on main (`moriio_connector.py:650-681`), fix #68513 open | M-L | only if E12/E13 rates satisfy the s2.3 budget | E7, E8 |
| L2 | Wide-EP decode DEP4/DEP8 (MegaMoE over MoRI) to raise KV/GPU | B300 D 192.4 GiB KV/GPU vs bar 61.0 [FH logs]; blog: decode MoE 62% of step at DEP4 [2H] | AITER MegaMoE requires gfx950, TP=1, PP=1, EP, `--all2all-backend mori` [FH `V/vllm/models/deepseek_v4/amd/model.py:611-641`], #66644 in N; DEP8 c256 182.7k (IX #5056) vs 26,714 (vllm#67377) conflict [2H]; open #68385 (A4W4, MXFP4 combine) +22.2% at DEP8 [2H] | M | more KV per GPU and larger decode batch per GPU; step cost at DEP8 on MI355X unknown | E13 |
| L3 | EPLB with redundant experts on decode | +12.0% DEP16, +25-26% DEP32 decode (GB300) [2H #62009]; balancedness 0.35 -> 0.94 [2H blog] | **rejected**: "AITER MegaMoE does not support EPLB" [FH `deepseek_v4/amd/model.py:639-640`]; `n_redundant_experts = 0` [FH `deepseek_v41/amd/model.py:167-173`]; symm_mem communicator raises off CUDA [FH `V/vllm/distributed/eplb/eplb_communicator.py:761-762`]; #61176 open: one MI355X DP8/EP8 c512 run -2.2% (4354 -> 4256 tok/s), GSM8K 0.970 -> 0.9475 [2H #61176 comment] | L | gain at DEP8 depends on measured imbalance; the only MI355X datapoint is negative | E13 (balancedness), E17 |
| L4 | Decode slot sizing: max-num-seqs = equilibrium batch, mbt = 10 x seqs, FULL_DECODE_ONLY graphs in steps of 6 up to 6 x seqs | B300 D 168/rank, GB300 253, GB200 112 [FH configs]; B300 best at cap [FH log] | flags generic; ROCm sparse-MLA builders report UNIFORM_BATCH [FH `V/vllm/models/deepseek_v41/amd/rocm.py:335-360`]; FULL_DECODE_ONLY used on gfx942 with breakable graphs off [FH RCP:262-289]; gfx950 + MegaMoE + DSpark untested | S (+ verify) | moves queueing from decode step time to TTFT (IX #4673 rationale [2H]) | E13, E14 |
| L5 | Async KV admission cap: `--max-async-kv-loads-per-step` 2-7, `--max-num-inflight-kv-loads` = max-num-seqs | one schedule() with 140K requests 174.3 ms -> 5.8 ms (limit 2) [2H #61725]; GB200 uses 7 / 112 [FH row_448128]; B300 828k ran with it inert [FH log] | in N as CLI flags [FH `V/vllm/config/scheduler.py:90-101`; `v1/core/sched/scheduler.py:1076`]; env var is not read upstream | S | reduces per-rank schedule() spikes when many long KV loads land at once | E14 |
| L6 | Decode prefix caching OFF | all NVIDIA vLLM PD rows [FH configs] | generic; with Mooncake, D prefix cache saves only compressed-KV/indexer suffix, SWA and ring are pulled every time [FH g3 s0.6] | S | trade: more free KV and simpler allocation vs fewer transferred bytes; A/B | E14 |
| L7 | Prefill-only host KV tier (70-80 GiB per prefill GPU) | B300 CPU hit 0.510 [FH API]; GB300 SGLang moved from external L3 back to local HiCache because layer-wise L3 reads stalled prefill [2H IX #4595] | UMBP keeps the AITER sparse-MLA fast path only alone or with all-UMBP MultiConnector children [FH `umbp/connector.py:110-117`, `multi_connector.py:274-280`]; SimpleCPUOffload has a HIP batch-copy path (ROCm 7.1+, 8192 descriptors on gfx950) [FH `V/vllm/v1/simple_kv_offload/cuda_mem_ops.py:53-118`] but no `only_restores_canonical_kv` override; host DRAM on openai node unknown | M | keeps the bar's prefix-capacity win (UMBP ~2x at c384 in author runs [2H #5036]) inside PD; cost of the lost fast path is E4 | E4, E15 |
| L8 | KV-event router at DP-rank granularity + session affinity (TTL 14400, soft) | Dynamo `router-mode kv`, temperature 0, thresholds None [FH configs]; +4.9% from parent-rank sub-agent placement (TRT-LLM) [2H] | bar uses vllm-router 0.1.14 `consistent_hash` = hard hash of `x-correlation-id`, no load or KV term [FH framework.md s0.4]; vllm-router 0.1.17 `kv_aware` (KV events, (worker, rank), host credit 0.8, PD over NIXL/Mooncake) [FH `VR/src/policies/kv_aware.rs:366,794-900`]; infera `kv_event_aware` per-rank [FH `INF/rust/router/src/kv_event.rs`]; neither run on DSv4.1 ROCm; whether ROCm DSv4.1 KV events cover the hybrid groups: not checked | M | higher prefix hit and less per-rank imbalance than consistent_hash, if the bar loses either today (unmeasured) | E11, E8 |
| L9 | Decode DP-rank choice by fewest active requests | GB300/GB200 `DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT=1e7` [FH configs]; SGLang total_requests: running spread 2-7 vs 11-26 by tokens [2H IX #4673] | vllm-router 0.1.17 decode minimum-occupancy [FH VR docs]; valid only if MI355X decode step is context-insensitive (blog shows it on GB300 only [2H]) | S-M | lower lockstep skew across decode ranks | E13, E14 |
| L10 | Prefill scheduler knobs: long-prefill-token-threshold 8192, prefix-cache-retention-interval 32768, prefill-schedule-interval 2 (P-only), max-num-seqs 64, mbt 16K-32K | all NVIDIA PD rows [FH configs]; R2 uses 0 / 0 / 11 / default / 16384 | generic [FH `scheduler.py:104-110, 239-245`; `V/vllm/config/cache.py:286-292`]; retention-interval on ROCm DSv4.1 SWA groups not verified | S | first two testable colocated now; retention-interval may raise SWA-group prefix hit on multi-turn | E10, E14 |
| L11 | NVIDIA colocated scheduler: prefill-schedule-interval 96 + mbt 65536 + capped max-num-seqs + explicit kv-cache-memory-bytes | B300 colocated 625,966 [FH row 447890 config] | generic | S | unknown direction on MI355X; cheap A/B | E10 |
| L12 | gpu-memory-utilization 0.95 / explicit `--kv-cache-memory` | NVIDIA 0.95 [FH]; bar 0.9, its log says 72.57 GiB KV possible vs 61.02 used [FH 448242 log, perf_anatomy V4] | generic | S | +19% KV GiB per GPU on the bar shape [INF: 72.57/61.02]; whether capacity binds is E3 | E10, E5 |
| L13 | DEP2 prefill (EP2 MoE) vs DP2 with TP MoE | blog DEP2 > DEP4 [2H]; all NVIDIA P workers use EP [FH] | MegaMoE "~2x slower/layer at 2 ranks" in earlier local work [2H IX #5098]; untested at DSv4.1 DP2+EP2 | S-M | unknown; measure | E12 |
| L14 | Engram: `embedding_across_dp`, HBM tables on decode, HIP-VMM host tables | GB300/GB200 decode Engram in HBM [FH]; #5056 DEP8 HBM 182.7k vs host 175.6k [2H] | in-node peer reads with 1-hop xGMI, TP1, cpu_offload=false [FH `V/vllm/models/deepseek_v41/common/engram.py:1092-1112`, `platforms/rocm.py:1030-1042`]; #66500 in N; #65595 (`VLLM_ROCM_ENGRAM_HIP_VMM=1`) not in N, startup-time only [2H] | S | frees host DRAM for the KV tier; lookup gain not shown | E13 |
| L15 | Connector transfer: NIXL push (prefill WRITEs into decode) alongside Mooncake / MoRIIO | all NVIDIA vLLM PD rows [FH]; packed rows #61121 open | push on main incl. DSv4.1 CSA-linear, needs identical P/D block sizes and group layouts [FH `nixl/push_worker.py:150-163, 640-680`]; nixl-rocm + UCX built in `V/docker/Dockerfile.rocm:36-137` (published image contents unchecked); any non-UMBP connector disables the fast path [FH SYNTHESIS C1]; Mooncake/MoRIIO do not verify layout identity for MLA [FH g1 s0.3] | M-L | a third connector candidate if Mooncake/MoRIIO fail | E7 |
| L16 | Same speculative config on both pools; `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1`; `--enable-ep-weight-filter`; `--numa-bind` | all NVIDIA PD rows [FH]; RCP:2085-2086 | generic; env in N [FH `V/vllm/envs.py:1638-1641`] | S | correctness of transferred draft KV; idle-rank dummy batches | E7 |
| L17 | CPU/frontend: Rust frontend, parse once in router, router on a separate host/NUMA, never per-request decode metrics | GB300 host RAM exhaustion at c5120 [FH IX comment]; TRT-LLM 13-39.5% [2H] | `VLLM_USE_RUST_FRONTEND=1` already set [FH reproduce_448242]; vllm-router 0.1.17 gRPC frontend [FH release notes]; router CPU at c >= 1000 unmeasured | S-M | removes a CPU ceiling if one exists at high concurrency | E16 |
| L18 | Prefill indexer top-k cost | DeepSelect +15.6% prefill at 32K [2H blog] | DeepSelect SM100a/SM103a only [FH `V/vllm/config/kernel.py:391`; `V/vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py:46-51`]; ROCm uses AITER `indexer_top_k_prefill` / `_C.top_k_per_row_prefill` [FH `V/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:153-162, 1624-1640`]; cost at 128K+ unmeasured | XL | only if the profile shows top-k is a large share of prefill | E12, E17 |
| L19 | Benchmark discipline: 3600 s only; dedicated client/router host | IX #4673 fast-run factor ~3.8x [2H]; MI355X CI/local 1.47x (SYNTHESIS Q3) | applies directly | S | avoids false comparisons | all |
| L20 | Measurement methods: `vllm bench forward` (#61339), mirror/lag replay for DP imbalance, real-vs-uniform routing for EP balance | blog methods [2H] | in N; profiler hooks CUDA/NVTX only [FH `V/vllm/benchmarks/forward/profile.py:152-173`]; `DecodeBenchConnector` switches ROCm DSv4.1 to the Triton sparse path [FH SYNTHESIS C1] | S-M | gives p and g (s2.3) without a full PD stack | E12, E13 |

---

## 4. Changes to the Phase-2/3 plan (SYNTHESIS s5, E0-E9)

### 4.1 Machine context
The user assigned one machine to this project on 11-19 (the other goes to the Muse vLLM session) [user request]. So everything up to
E14 is planned for **one 8-GPU MI355X node**: up to four 2-GPU colocated servers in parallel (host DRAM permitting UMBP), or
one same-node PD stack. Two-node steps (E8 stage 2) need a second node. Model read-only from `/perf_apps/chaox/models/DeepSeek-V4.1-Flash`.

### 4.2 Changed existing experiments
| # | Change | Why (lever) |
|---|---|---|
| E0 | add: host DRAM total (NVIDIA uses 70-80 GiB per prefill GPU + ~94 GiB Engram per node; bar uses UMBP 449 GB per 2 GPUs); `pip list | grep -iE "nixl|mooncake|mori"` in the R1 image; GPU-NUMA map for `--numa-bind-nodes` | L7, L15, L16 |
| E2 | log per DP rank: Running/Waiting, prefix hit, external hit, KV used (as in the 448173 logs) so router effects are visible | L8, L9 |
| E3 | add the vLLM-reported max KV (`--kv-cache-memory` hint line) and a device-time split prefill vs decode at c256/c384 | L12; s2.2 |
| E4 | unchanged, promoted to a hard gate before any PD connector choice (run on `fp8_e4m3`, SYNTHESIS C1) | L7, L15 |
| E6 | split into E12 (prefill rate p) and E13 (decode rate g, wide EP); add the s2.3 budget as the go/no-go: proceed to PD only if measured c/p + o/g <= 0.179 GPU-s (plus measured transfer/TTFT overhead) | s2.3 |
| E7 | P = DP2 (TP1) worker with the same `--speculative-config` as D; D = DP4 (or DEP4) on the same node; decode `--no-enable-prefix-caching` as the default; connector candidates in order Mooncake, MoRIIO + #68513, **NIXL push** (nixl-rocm); `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1` on every DP>1 role | L1, L6, L15, L16 |
| E8 | topology ladder copied from B300: stage 1 same node (2x DP2 P + DP4/DEP4 D, P:D 1:1; or 3x DP2 P + DP2 D if p is the limiter); stage 2 two nodes = 4x DP2 P node + DEP8 D node, concurrency by Little's law from E12/E13 rates and measured think time (B300 analogue 448172: 738,119 @ c2048); stage 3 three nodes (5P over 2 nodes + DEP8). Router: vllm-router PD mode with `kv_aware` prefill + min-occupancy decode, vs infera router | L1, L2, L8, L9 |

### 4.3 New experiments
| # | Experiment | Hypothesis | Discriminating result | Nodes |
|---|---|---|---|---|
| E10 | Colocated knob A/B on R2 (DP2+UMBP, c384 and c256, 3600 s, one 2-GPU server per arm, in parallel): (a) `--long-prefill-token-threshold 8192`; (b) `--prefix-cache-retention-interval 32768`; (c) `--gpu-memory-utilization 0.95` or explicit `--kv-cache-memory` (72.57 GiB hint); (d) NVIDIA-colocated scheduler `--prefill-schedule-interval 96 --max-num-batched-tokens 65536` + capped `--max-num-seqs` | NVIDIA's generic scheduler/cache knobs change the colocated bar | tok/s/GPU outside the run-to-run noise band (noise band from repeated E2 arms) | 1 |
| E11 | Router A/B on R2: vllm-router 0.1.14 `consistent_hash` (bar) vs 0.1.17 `kv_aware` (engine KV events, `--intra-node-data-parallel-size 2`, session TTL 14400 soft) vs infera `kv_event_aware` | consistent_hash loses prefix hit or DP balance at c384 | per-rank prefix hit and running-count spread; tok/s/GPU | 1 |
| E12 | Prefill-only rate p on MI355X at the AgentX mix: DP2 with TP MoE vs DEP2 (`aiter_mega_moe`, EP2) vs DP4; `vllm bench forward` at 8K-64K new on 64K-512K cached; profile the indexer top-k share | DEP2 or DP2 reaches the p required by s2.3 | measured p (computed tok/s/GPU); indexer top-k % of prefill step | 1 |
| E13 | Decode-only rate g and step vs batch / context: DP2 (bar shape), DEP4, DEP8 with AITER MegaMoE over MoRI; Engram HBM + `embedding_across_dp`; FULL_DECODE_ONLY graphs in steps of 6 on gfx950; `log_balancedness` for EP imbalance; resolve #5056 vs #67377 | wide EP raises KV/GPU and g; step is (or is not) context-insensitive on MI355X | KV GiB/GPU per shape; g at 100-200 rows/rank; step(ctx) slope; balancedness | 1 |
| E14 | PD admission/slot sweep on the first working PD stack: decode max-num-seqs at the equilibrium batch, mbt = 10 x seqs, `--max-async-kv-loads-per-step` {2,4,7}, `--max-num-inflight-kv-loads` = seqs, decode prefix cache on/off, prefill-schedule-interval {2,11} and retention-interval on P | NVIDIA's decode admission settings carry over | Running vs cap, Waiting, KV-fetch queue, per-rank schedule() time, TTFT/TPOT, tok/s/GPU | 1 (then 2) |
| E15 | Prefill host tier under PD: PD connector alone vs MultiConnector[PD + UMBP] vs MultiConnector[PD + SimpleCPUOffload lazy, 70 GiB per prefill GPU] | a host tier keeps the bar's prefix-capacity win inside PD; with any PD connector the fast path is already off, so UMBP vs SimpleCPUOffload is a capacity/latency choice | external/CPU hit, prefill p, tok/s/GPU | 1 |
| E16 | CPU/frontend at high concurrency: router and API-server CPU, per-rank schedule() time, router on a separate NUMA node or host | a CPU ceiling appears above c1000 | CPU saturation vs throughput plateau | 1-2 |
| E17 | ROCm code items (no GPU until a patch exists): (a) safe `only_restores_canonical_kv` override for Mooncake/MoRIIO/NIXL (g1_fastpath_override.md s2-3); (b) redundant-expert EPLB for the DSv4.1 ROCm runner (cherry-pick #61176, measure); (c) FULL_DECODE_ONLY for DSv4.1 on gfx950; (d) NIXL push with nixl-rocm on DSv4.1 | each removes one ROCm-only blocker found in this study | patch + correctness test + the E-number that measures it | 0-1 |

### 4.4 Order (one node)
E0 -> E1 -> E2 (+E11 arms in parallel) -> E4 (gate) -> E10 + E3 (parallel 2-GPU arms) -> E12 + E13 (gate: s2.3 budget) ->
E7 (same-node 1P1D, connector order Mooncake / MoRIIO+#68513 / NIXL push) -> E14 + E15 -> E8 stage 1 -> (second node) E8 stage 2
-> E16. E5 and E9 unchanged and folded into E3/E13 startup logs. E17 items run as code work in parallel and land in the experiment
that measures them. All comparisons at 3600 s.

### 4.5 Recipe changes to carry into the first PD stack (each still A/B'd, none assumed)
- P worker: TP1 DP2 (EP2 if E12 favours it), `--max-num-seqs 64 --max-num-batched-tokens 16384|32768 --long-prefill-token-threshold 8192
  --prefill-schedule-interval 2 --prefix-cache-retention-interval 32768 --gpu-memory-utilization 0.95`, host KV tier per E15,
  same `--speculative-config` as D, `--numa-bind`.
- D worker: DP4/DEP4 (one node) or DEP8 (own node), `--no-enable-prefix-caching`, `--max-num-seqs` = equilibrium batch per rank,
  `--max-num-batched-tokens` = 10 x seqs, `--max-async-kv-loads-per-step 4`, `--max-num-inflight-kv-loads` = seqs,
  `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1`, Engram per E13.
- Router: prefill KV-aware at (worker, rank) with session TTL 14400 soft; decode least-active; no router queueing or retries;
  timeouts 14400 s; client sends the correlation id.

---

## 5. What cannot transfer, and why

| Item | Why | Evidence |
|---|---|---|
| 72-GPU NVLink domain (DEP16/DEP32 decode, KV over MNNVL cuda_ipc) | MI355X scale-up domain is 8 GPUs; beyond 8 the EP all-to-all crosses RDMA. The B300 template (8-GPU HGX, DEP8 decode in one node, KV over EFA) is the transferable class | [FH configs; 2H specs] |
| Grace / NVLink-C2C host memory | x86 + PCIe host on MI355X | [2H specs] |
| FlashMLA MegaAttn, DeepGEMM MegaMoE, DeepSelect top-k, specialized decode #61868/#61869, FlashInfer autotune | CUDA kernels; ROCm equivalents are AITER Gluon/Triton sparse MLA, AITER CK a8w4 / AITER MegaMoE, persistent/per-row top-k, monokernel (TP only) | [FH `kernel.py:391`; `amd/mono_decode.py:316-317`] |
| symm_mem EPLB communicator | raises "symm_mem EPLB requires CUDA" | [FH `eplb_communicator.py:761-762`] |
| `nvfp4_ds_mla` KV record | FP4 KV record unavailable on ROCm (fp8_ds_mla 584 B or fp8_e4m3 512 B) | [FH SYNTHESIS s2] |
| cuMem FABRIC Engram peer groups (`peer_group_size`) | needs CUDA FABRIC/IMEX; ROCm has in-node xGMI peer reads and HIP-VMM host tables instead | [FH `V/vllm/config/engram.py:72-77`] |
| torch.compile `FULL_AND_PIECEWISE` + decoder-replay CUDA graphs on prefill | DSv4.1 not torch-compiled on ROCm, breakable graphs; no `decoder_replay_cudagraph.py` under `amd/`; FULL_AND_PIECEWISE aborted at MI355X bring-up | [FH RCP:227-231; 2H IX #2962] |
| `push_as_nixl` module, `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` env | dev-image only / ignored upstream; use `transfer_mode: push` and `--max-async-kv-loads-per-step` | [FH RCP:2346-2348; 448173 log] |
| EFA/LIBFABRIC and UCX MNNVL envs, `enable-cumem-allocator` choices | NVIDIA fabric specifics; MI355X fabric (ionic / openai NICs) is unmeasured (E0) | [FH configs] |
| Raw FP4 peak 1.34-1.49x per GPU | hardware | [2H specs] |
| Dynamo itself (optional) | no official ROCm image (#12537 closed stale 11-11); KVBM excluded for ROCm; one MI355X PoC (vLLM+RIXL, MiniMax-M2.5, 730 tok/s aggregate on 2 nodes) and an unresolved DSR1 `ibv_reg_mr` failure. The features the NVIDIA runs used have AMD-side equivalents (vllm-router 0.1.17, infera router) | [FH gh; 2H PoC README] |

---

## 6. Open (not established)
- Cause of B300's 8.5 s mean TTFT at the best point (router vs prefill queue vs KV-fetch wait); the decode-at-cap reading rests on 2 log lines.
- API GPU/CPU hit split in PD rows (constant 0.48/0.50 on B300) vs engine prompt counters (3.8-3.9% computed on both B300 and the bar).
- Why NVIDIA logs show ~1.1 KB per token-slot on decode vs ~2.0 KB on prefill with the same KV format.
- Whether ROCm DSv4.1 prefill steps are graph-captured; whether FULL_DECODE_ONLY works on gfx950 with MegaMoE and DSpark.
- Whether vLLM ROCm DSv4.1 KV events expose the compressed/SWA groups in a form KV-aware routers index.
- MI355X values of p and g (s2.3), decode step vs context, and EP imbalance - all measurement items (E12, E13).

---

## Critic additions (completeness pass, 2026-11-19 ~11:00)

Raw evidence: `notes/dynamo/critic/` (API refetch `bench_20261119_critic.json`, PR bodies `pr/`, Dynamo docs page `web/`,
vLLM PR search `vllm_pr_search_pd.txt`).

**A1. Row coverage re-checked [FH: API refetch, 227 rows, no new ids vs `notes/dynamo/verify/bench.json`].** Best per
(HW, framework, disagg) for dsv41flash AgentX. Rows not used anywhere in this document:
| Row | HW | Mode | GPUs | conc | tok/s/GPU | p90i | image | date |
|---|---|---|---:|---:|---:|---:|---|---|
| 447344 | B200 | vLLM colocated DEP8 | 8 | 576 | 388,471 | 106.7 | cu134-nightly-40941f2c | 11-05 |
| 447768 | B200 | SGLang colocated DEP4 (offload on) | 4 | 256 | 364,592 | 110.4 | sglang nightly-dev-cu13-20261107 | 11-09 |
| 447977 | B300 | SGLang colocated DEP2 (offload on) | 2 | 256 | 503,971 | 65.9 | sglang nightly-dev-cu13-20261107 | 11-11 |
| 448099 | MI355X | ATOM colocated TP2 DEP2 DPA | 2 | 256 | 389,350 | 37.8 | rocm/atom-dev nightly_202611101510 | 11-13 |
- **No B200 or MI355X Dynamo PD row is published.** B200 Dynamo+vLLM PD (IX #3039, 1P DEP8 / 1D DEP16, NIXL, no host tier)
  closed unmerged 10-12 after the launcher refactor; B200 Dynamo+SGLang 2P1D (IX #4328) closed stale 11-09 after an NVIDIA
  reviewer said "Perf is worse than B200 SGLang" (colocated) [FH: `pr/pr_3039.json`, `pr/pr_4328.json`]. So on the 8-GPU-HGX class
  without NVL72, only B300 has a PD win on record; B200 has a reported PD loss with no number [2H].
- B200 is the closest NVIDIA peer to MI355X (8-GPU HGX, no Grace); its best published colocated point is 388k at p90i 106.7,
  i.e. not pushed to the bar's p90i 15.8 [FH row 447344]. Its config was not read in this pass.
- ATOM 448099 is an existing MI355X DEP2 (EP2 MoE) colocated point at 389k [FH row]; it is a reference for L13 (DEP2 on MI355X)
  that E12 can reuse (different engine).

**A2. NVIDIA Dynamo's own DSv4.1 recipe page (docs.nvidia.com/dynamo/dev/recipes/deepseek-v4-1-flash) was not cited before**
[FH: fetched 11-19, `web/dynamo_docs_dsv41.html`]. Synthetic 64K in / 400 out, 8 GPUs, not AgentX:
| SKU | Agg 2xTP4 out tok/s/GPU (conc) | PD 1P TP4 + 1D TP4 (conc) | PD/agg | TTFT p90 agg -> PD |
|---|---|---|---:|---|
| B200 | 990.57 (c168) | 1,087.71 (c184) | 1.098 [INF] | 3.07 s -> 90.08 s |
| GB200 | 953.08 (c168) | 1,154.87 (c168) | 1.212 | 3.52 s -> 57.12 s |
| H200 | 315.45 (c56) | 359.41 (c64) | 1.139 [INF] | 3.45 s -> 35.04 s |
Config facts on the page: EPLB with communicator **`torch_gloo`** (not symm_mem) on B200/GB200; DSpark **3** drafts; long-prefill
threshold 4096 on P; prefix-cache retention interval 1024; P mbt 32768; decode graph cap 1024. Relevance: this is the only
published single-node (8-GPU) PD-vs-agg comparison for DSv4.1, the shape of our one-node E7/E8 stage 1; the PD gain comes with a
15-30x TTFT p90 increase. Workload differs from AgentX, so the ratio is not transferable as a number.

**A3. EPLB communicator correction to L3 / s5 [FH `V/vllm/distributed/eplb/eplb_communicator.py:761-776, 851-868`].** Backends are
`torch_nccl`, `torch_gloo`, `pynccl`, `nixl`, `symm_mem`; only symm_mem is CUDA-only, and its own error text says "use nixl or
torch_gloo instead"; the nixl backend has a ROCm branch at :455. So the communicator is **not** the ROCm blocker; the blocker is
the model-level rejection "AITER MegaMoE does not support EPLB" (`deepseek_v4/amd/model.py:639-640`) and `n_redundant_experts = 0`
in the DSv4.1 AMD model. E17(b) should target the model/MegaMoE side and use torch_gloo/nixl/torch_nccl as communicator.

**A4. GB300 recipe moved on; the vLLM recipe now carries it [FH: vllm#68556 MERGED 2026-11-19 03:14Z, in V (compare: V is 33
ahead); `RCP:1623-1760`].** Source: IX #5015 (OPEN, no published numbers; image `cu134-nightly-aarch64-01fba62-1111d`, 5P1D DEP32
c3200 / 7P1D DEP16 c6600 / 12P1D DEP32 c10288). Facts not in the lever table:
- **EPLB on prefill too**: P DEP2 `num_redundant_experts: 2`, `log_balancedness: true`; D 16 (DEP16) or 32 (DEP32).
- Prefill host tier **SimpleCPUOffload 182,536,110,080 B (170 GiB) per DEP2 worker** (85 GiB/GPU), lazy; gpu-mem-util **0.92**.
- Prefill torch.compile `VLLM_COMPILE` + `FULL_AND_PIECEWISE` and `decoder_replay_cudagraph_capture_sizes` (s5 row "cannot transfer").
- Decode: `max-num-seqs` **18 (c3200 DEP32), 128 (c6600 DEP16), 72 (c10288 DEP32)** per rank; mbt = 10 x seqs;
  `--max-async-kv-loads-per-step 3`; `--max-num-inflight-kv-loads` 40/128/72; graph sizes multiples of 6 because 6 = 1 + 5 DSpark
  tokens (recipe comment, RCP ~1760) - this explains L4's "steps of 6" and on our side the step should be 1 + our draft count.
- Decode `--additional-config '{"dspark_fused_markov":true}'` - lives in `V/vllm/models/deepseek_v41/nvidia/dspark.py` only (CUDA path) [FH grep].
- Engram on prefill: `cpu_offload:false, peer_group_size` = all prefill GPUs (cuMem FABRIC; not transferable, s5).
- `--safetensors-load-strategy prefetch`, `--enable-cumem-allocator`, `--numa-bind-nodes` per rank.
- RCP:2340-2389 (B300 guide prose, `push_as_nixl`, `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP`) describes IX #4231, which IX **reverted in
  #4351** (10-20); the 828k B300 row came from IX #4585. Do not take B300 settings from that prose. A B300 preset matching #4585 is
  vllm#68045 (OPEN 11-17).

**A5. Already matched by the bar (checked, no lever):** `--attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'`
is set in the bar [FH `notes/ix/reproduce_448242.json`] and on every NVIDIA row; ROCm requires mxfp4 for sparse logits
[FH `V/vllm/models/deepseek_v41/attention.py:1547-1552`; `V/vllm/model_executor/layers/rocm_paged_mxfp4_indexer.py:1-9`]; it is
not auto-enabled on ROCm (`V/vllm/model_executor/models/config.py:770-790` requires `is_cuda()`), so any new PD role config must
set it explicitly.

**A6. Unread vLLM PRs that touch the plan [FH: PR bodies in `pr/vllm_*.json`; effects are author claims, 2H].**
| PR | State | What | Plan impact |
|---|---|---|---|
| #68511 | OPEN 11-18 | SimpleCPUOffloadConnector declares `only_restores_canonical_kv`; body: DSv4.1 on ROCm then keeps the AITER sparse-MLA path ("follows from the hook and was not measured") | contradicts L7's "no override" once merged; makes SimpleCPUOffload a fast-path-preserving prefill tier alone or with UMBP; MultiConnector with NIXL/Mooncake child still non-canonical. Add as E15 arm and to E17(a) |
| #67562 | OPEN 11-13 | session-based prefix-cache retention, `--session-cache-ttl` 600 s default, all KV groups | AgentX multi-turn retention; complements L10 retention-interval; E10 arm once merged |
| #67593 | OPEN 11-14 (stacked on #67203) | NIXL chunked prefill push for hybrid models incl. DSv4.1; TTFT p50 up to -27% vs whole-request push | L15 follow-on; affects TTFT not GPU-time |
| #65584 | OPEN 11-04 | backoff last prompt token for SWA in P/D (NIXL + Mooncake) - decoder reads null_block at N-W otherwise (#65550); cites existing DSv4.1 bounded-replay backoff | correctness check for E7 (DSv4.1 has SWA); verify DSv4.1 path is already covered |
| #65237 | MERGED 11-05 | lazy SimpleCPUOffload stores at the GPU eviction end; recomputed blocks 67.5% -> 59.7% (simulation, DSv4.1-like SWA) | in N? not checked (N = 11-16 main, likely yes [INF]) |
| #65373 | MERGED 11-12 | DSv4.1 CPU KV shared across TP ranks (H100 TP8: 6.02x req/s, synthetic) | only for TP>1 roles; our P is TP1 |
| #65146 | OPEN 11-02 | ROCm fused mega-attention prefill kernel for MI355X | direct ROCm analogue of FlashMLA MegaAttn prefill (L18/E12); cited in 2 notes but not in the lever table |
| #68045 | OPEN 11-17 | B300 Dynamo AgentX presets from IX #4585 | the authoritative B300 template once merged |

**A7. IX PRs on DSv4.1 Dynamo not read before [FH `pr/pr_*.json`]:** #4351 (revert of B300 #4231), #4328 / #3039 (B200, A1),
#4345 (GB300 prefill Engram THP per-rank tables, closed same day), **#5100** (GB200 **llm-d-router** + vLLM PD, closed same minute
as opened): 2x DP4/EP4 prefill + DEP8 or DEP4 decode, precise prefix-cache routing from per-rank KV events, token-load +
active-request scorers, NIXL lease 600 s, **EPLB 32 redundant experts on decode**, 64 GiB CPU KV per prefill rank; author's source
runs at c512: 2P1D DEP8 2.94M input tok/s on 16 GPUs [2H]. llm-d-router is a non-Dynamo KV-event router alternative for L8
(ROCm support not checked). #3941/#4071/#3950/#4266 (earlier GB300 points/frontend) are summarised in recipes.md only via #4303.

## Remaining gaps
- G-a: B200 colocated 447344 config and why B200 PD was "worse than colocated" (#4328 has no number) - unread/unpublished.
- G-b: vr200 2.99M (448150) recipe: private repo, Mooncake host offload; still unknown.
- G-c: GB300 TRT-LLM (#4991) and TokenSpeed (#5105) internals: not opened in this pass beyond recipes.md V12.
- G-d: #5015 has no published numbers; the GB300 preset in RCP is unvalidated by an API row.
- G-e: llm-d ROCm support and Dynamo community `rocm` fork contents: not opened.
- G-f: #68511 effect on DSv4.1 ROCm fast path is asserted by its author, not measured; needs E15 arm after merge or cherry-pick.
- G-g: whether N (60bd6dd2) contains #65237 / #65373: not checked (`gh api compare` per PR).
- G-h: SYNTHESIS C6/C7 were not re-read in this critic pass; consistency of A3 (EPLB communicator) with SYNTHESIS statements not checked.
- G-i: Dynamo docs table (A2) is synthetic 64K/400, DSpark 3; no AgentX single-node PD row exists for any NVIDIA SKU.
