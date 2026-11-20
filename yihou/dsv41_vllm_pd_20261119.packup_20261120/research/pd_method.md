# PD-disaggregation method and AgentX priors for DeepSeek-V4.1-Flash on MI355X (vLLM)

Date: 2026-11-19. Phase 1 (research only, no GPU runs). Author: pd-method research leaf.

Evidence tags: **[F: file:line]** first-hand code/config/log read; **[O: URL/PR#]** official upstream source;
**[A]** arithmetic on first-hand inputs (formula shown); **[2nd]** second-hand (a colleague's file/claim I read but
did not reproduce, or a PR author's claim); **[I]** inference, unverified. Every number states its source.

Local source snapshots used (all inside this workspace):
- vLLM main `7a62ca0666e9153bd67b6ac24c4e10b1f0f6fa64` (2026-11-19), files fetched with `gh api` into
  `notes/pd_method_src/` (`vllm_models_deepseek_v41_attention.py` = `vllm/models/deepseek_v41/attention.py`,
  `vllm_models_deepseek_v41_compressor.py`, `vllm_v1_attention_backends_mla_indexer.py`,
  `vllm_v1_attention_backends_mla_sparse_swa.py`, `mooncake_connector.py`, `moriio_connector.py`).
- InferenceX `db3fc79642b9` (2026-11-18) at `source/InferenceX`.
- Reference model code `/perf_apps/chaox/models/DeepSeek-V4.1-Flash/inference/model.py` and `config.json`.
- Trace statistics computed by `scripts/pd_method/trace_stats.py` -> `notes/pd_method_src/trace_stats.json`.

---

## 0. Decision summary (what Phase 2 decisions rest on)

1. **The PD wire is not a plausible bottleneck for this model.** Per-token KV that must move is ~1.63 KB
   (analytic, vLLM ROCm layout) to ~2.47 KB (vLLM engine capacity slope); a served-mean request (ISL 173k) moves
   ~0.29-0.43 GB, ~20-30x less than GLM-5.2 (47.7 KB/token). At 1.5x the current vLLM MI355X number
   (161k tok/s/GPU) on 16 GPUs the whole deployment moves 4.2-6.3 GB/s, ~1.4-2 % of 8 rails x 38.3 GB/s
   (measured on crusoe ionic; openai-cluster NICs not documented). [A, inputs in section 3]
2. **Decode KV capacity is unlikely to bind first.** vLLM TP2 reports 28.9M tokens in 66.44 GiB per GPU;
   a TP4 decode would hold ~55M tokens (~320 requests at ISL 173k; ~440 with Engram tables on host),
   x4 under DP attention. [F log + A]
3. **In the only vLLM MI355X AgentX anchor (TP2, C32, colocated, 107,583 tok/s/GPU) decode-phase kernels took
   ~78 % of rank-0 device time and prefill ~22 %.** Per request: ~0.19 s prefill vs ~0.65 s decode device time
   at a decode batch of ~8. That is the room PD (or higher concurrency) can reclaim. [2nd: chaox README; A]
4. **vLLM DSv4.1 PD on ROCm is not proven upstream.** MoRIIO on main rejects DSv4.1's compressor
   `CircularBufferManager` group; the fix vllm#68513 is OPEN (2026-11-18, validated by its author on MI325X
   XGMI, P TP2+DP / D TP4). MooncakeConnector on main has DSv4.1-specific code (SWA clipping, bounded-replay
   N-1, circular-buffer regions) but I found no ROCm DSv4.1 PD validation. The only MI355X DSv4.1 PD recipe in
   InferenceX is TileRT (vLLM prefill + TileRT decode over Mooncake), not pure vLLM. [O, F]
5. **vLLM PD differs from SGLang PD in two ways that matter here:** (a) the vLLM decode can keep a prefix cache
   and then pulls only the unhashed suffix blocks (code read; SGLang PD decode keeps none by default);
   (b) PP is constrained: vLLM refuses PP splits inside a DSv4.1 kv-sharing group, and layers 20-39 form one group,
   so only a 20/20 PP2 split is possible (no PP4 prefill as in GLM-5.2). [F]
6. **Phase 2 must first run a colocated vLLM concurrency sweep** (the closed-loop knee and what binds there) and a
   decode-only step-time-vs-batch curve, then bring PD up for correctness. PD is worth it only if the colocated knee
   is set by prefill/decode interference or decode step at small batch, and the predicted PD score at equal GPU
   count clears the colocated best by more than the noise band. Section 7 lists the measurements.

---

## 1. Handbook (pd-agentx-tuning) applied to DSv4.1-Flash

Arch tags of DSv4.1-Flash from `config.json` + `inference/model.py`:

| tag | DSv4.1-Flash fact | source |
|---|---|---|
| MoE | 384 routed experts, top-6, 1 shared, moe_intermediate 2304, FP4 experts (block 32 e8m0) | [F: config.json `text_config`, `quantization_config`] |
| MLA-like latent KV | 1 KV head, head_dim 512 (rope 64), q_lora 1280, grouped low-rank output (o_groups 8, o_lora 1024) | [F: config.json; model.py:614-662] |
| SWA | every layer attends a 128-token sliding window ring; layers 0,1 and the 3 draft layers are SWA-only | [F: model.py:78-81, 701-721; config `compress_ratios`] |
| KV compression + sharing | compressed KV only on kv-source layers 2,8,14 (ratio 2) and 20 (ratio 1); layers 3-7/9-13/15-19/21-39 read their source's cache | [F: config `kv_source_layer_ids`; model.py:614-621, 655-679, 740-766] |
| DSA indexer | 32 heads x 128, top-512 compressed positions; computed on 8 index-source layers (2,8,14,20,24,28,32,36); index-K owned by the 4 kv sources; two-level candidate pre-filter from layer 20 (2048 blocks of 8) | [F: config; model.py:488-611] |
| MTP | DSpark draft: 3 layers (`num_nextn_predict_layers` 3), block 5, own MoE 128 experts top-3, reads target layers 37-39 | [F: config; model.py:127-149] |
| Engram | n-gram hash tables at layers 1 and 14: 188.8 GiB of the 475.2 GiB checkpoint (fp8 rows of 256 + scales) | [F: safetensors JSON headers of the 48 shards read 2026-11-19 (inline python, header-only); model.py:296-345] |
| hyper-connections | hc_mult 4 residual copies (compute only, no KV) | [F: config] |

Handbook step / lever applicability (handbook: `~/.claude/skills/pd-agentx-tuning`):

| handbook item | applies to DSv4.1-Flash on vLLM? | note |
|---|---|---|
| Step 0 model intake (I1-I10) | yes; **re-derive the KV cell from the vLLM log**, the analytic cell (1,630 B) and the engine slope (2,468 B/token-equivalent) differ by 51 % and the gap is not explained (section 3.1) | SKILL.md step 0; model-intake.md I3 |
| Step 1 objective + real-acceptance gate | yes; InferenceX vLLM MI355X runs use `rejection_sample_method: synthetic`, `synthetic_acceptance_length 3.51` [F: chaox `baseline/launch_server.sh`] | waiver must be labelled |
| Step 3 reproduce strongest recipe | yes: colocated vLLM TP2 C32 107,583 tok/s/GPU is reproduced locally by chaox [2nd] | differential reference |
| Step 4 concurrency sweep | yes, first-class [agnostic]; peak C not known for vLLM MI355X beyond C32-C64 recipe points | |
| Step 5 bottleneck localisation | yes; vLLM names differ from SGLang (no `#prealloc-req` / `#inflight-req`); use `vllm:` metrics and connector stats | bottleneck-diagnosis.md §1-4 are SGLang-named |
| Placement same- vs cross-node | re-measure; GLM-5.2's cross-node win came from a slow same-node HIP-IPC copy path with 5-8 GB per request; with 0.3-0.4 GB per request the copy path matters less [I] | levers-catalog §1 |
| P:D ratio / instance count | **top structural lever** (output is ~0.6 % of tokens; decode needs few GPUs per prefill GPU, section 4) | levers-catalog §1 |
| PP prefill (GLM-5.2 PP4) | **mostly blocked**: only PP2 split at layer 20 is legal in vLLM (section 5.3) | |
| Prefill HiCache / host tier | SGLang-only in the handbook; vLLM analogue = `SimpleCPUOffloadConnector` via `MultiConnector` (used by the GB300 vLLM disagg recipe, 60-70 GiB/worker) | [F: gb300 disagg-variants.yaml:110] |
| SPF / prefill scheduling | vLLM has no SPF policy in main per the Muse map (P8 in Muse SYNTHESIS) [2nd]; candidate only if prefill queue binds | |
| Router R1 / R2 / session affinity | yes [agnostic] as ideas; implementation depends on the router (vllm-router `consistent_hash`, Dynamo `kv` + soft session affinity) | [O: vllm#68513 body; F: gb300 recipe] |
| Sender-queue shard, DCP pack fix | SGLang-code specific; DCP pack fix [layout:DCP] not applicable (no DCP decode planned) | |
| DCP decode | **low priority**: KV is replicated per TP rank (`max_tp_shards=1`), but the KV is tiny, so DCP's capacity motive is weak; DP attention removes replication instead | [F: attention.py:1306-1323] |
| Decode mem-fraction / max-num-seqs / graph sizes | yes [agnostic]; vLLM `--gpu-memory-utilization`, `--max-num-seqs`, capture sizes in multiples of (1+5 drafts) | GB300 recipe uses multiples of 6 |
| MTP depth (K3 vs K5) | [arch:MTP] yes: DSpark `num_speculative_tokens` 5 in every recipe; depth A/B only with measured real acceptance per depth | |
| index_share / OffloaderV2 / HiCache E1/E2 | SGLang-specific; no vLLM equivalent identified | |
| Kernel overlay / GEMM retune | yes after step 5; chaox extraction already ranks MoE a8w4 (FlyDSL), mxfp8 block32 GEMM, sparse MLA (Gluon), mono decode layer, quick-allreduce for vLLM TP2 C32 | [2nd: chaox conc32 README] |
| Expert-truncation capacity probe | [arch:MoE] usable as a probe idea; KV capacity is unlikely to bind (section 3.3) so lower value | |
| FP4 KV | ROCm: not available — `nvfp4_ds_mla` requires the V4.1 MXFP8 records that only SM100 decodes | [F: attention.py:612-625] |

---

## 2. AgentX workload priors

### 2.1 Dataset and harness facts
- Dataset `semianalysisai/cc-traces-weka-062126@23f152f6f0f9...`, 393 traces, 56,798 main turns, 1,697 sub-agent
  groups, 42,029 sub-agent inner requests, 98,827 requests, 21.64e9 input / 1.065e8 output tokens.
  [F: dataset README/stats.txt in `/perf_apps/chaox/DeepSeek_V4.1_Flash/conc32/hf_home/...`]
- `in` is counted in 64-token hash blocks of the *trace's* tokenizer (Claude), capped at 990,016. [F: dataset README]
- Harness rules: trajectories start at 25-75 % of each trace, first-turn prefix cache-bust, idle gap cap 300 s,
  3600 s window, 10 warm-up requests per lane. [2nd: Muse `research/sglang_lever_map.md:142-144`]
- Score = (sum ISL incl. cache hits + sum OSL) / wall / all GPUs. [2nd: handbook measurement.md §1; Muse agentx_harness.md:12]

### 2.2 Whole-trace statistics (my computation, `scripts/pd_method/trace_stats.py`)
Prefix hit = leading hash blocks already seen earlier in the same trace (chained block ids). [F/A]

| metric | all requests | main-agent | sub-agent |
|---|---:|---:|---:|
| n | 98,827 | 56,798 | 42,029 |
| ISL mean / p50 / p90 / p99 / max | 218,922 / 142,016 / 549,504 / 863,488 / 989,824 | 313,543 / 254,080 / 651,008 / 897,920 | 91,051 / 64,000 / 196,480 / 455,808 |
| OSL mean / p50 / p90 / p99 | 1,077 / 444 / 2,659 / 9,311 | | |
| new (uncached) tokens mean / p50 / p90 / p99 / max | 3,702 / 1,600 / 5,888 / 40,960 / 949,440 | | |
| main turns per trace mean / p50 / p90 / max | 144.5 / 65 / 331 / 2,977 | | |
| prefix hit, token-weighted / request-mean | 0.983 / 0.900 | | |
| ISL > 128k / > 256k / > 512k | 53.1 % / 30.0 % / 11.1 % | | |
| input share of tokens | 0.9951 (OSL/ISL = 0.0049) | | |

### 2.3 Served statistics (what the closed loop actually delivers)
- vLLM DSv4.1 MI355X TP2 C32 (InferenceX CI job 114309765517, reproduced by chaox 2026-10-14): mean ISL 173,006,
  OSL 1,023.5, 4,476 requests in 3,606.7 s (qps 1.241), GPU prefix hit 0.960, TTFT p50 0.26 s, TPOT p50 6.11 ms,
  total 107,583 tok/s/GPU, synthetic acceptance 3.51, `--no-swa-bounded-replay`, Engram on host.
  [2nd: `/perf_apps/chaox/DeepSeek_V4.1_Flash/conc32/baseline/BASELINE_REPRO.md`, `launch_server.sh`, result JSON
  `per_gpu.total_tput_tps 107583.43`]
- GLM-5.2 SGLang PD (other tokenizer): served ISL p50 72k, mean 117-142k, falling with concurrency; prefix hit
  96.1-96.5 %. [2nd: handbook measurement.md §2]
- Closed-loop cycle at the vLLM C32 anchor: C / qps = 32 / 1.241 = 25.8 s per lane = idle gap g + TTFT + D with
  D = OSL x TPOT = 6.25 s, TTFT 0.26 s -> g ~19 s. [A] Implication [I]: per-lane throughput is dominated by the
  trace idle time; the score scales with lanes per GPU until an engine stage saturates. GB300 vLLM PD recipes run
  C3018 on 44 GPUs and C10888 on 64 GPUs (69 and 170 lanes/GPU) vs 16 lanes/GPU at the MI355X C32 anchor.
  [F: gb300 disagg-variants.yaml override names/shapes]

---

## 3. First principles for PD on DSv4.1-Flash

### 3.1 KV cell (per token) and per-request transfer bytes

vLLM ROCm layout facts [F, vLLM main 7a62ca0]:
- ROCm keeps the **V4 KV record**: 448 fp8 NoPE + 64 bf16 RoPE + 7 UE8M0 scales + pad = **584 B per stored row**,
  576-B page alignment; the 528-B V4.1 MXFP8 record is used only on SM100 / Ampere
  (`attention.py:196-211`, `:610-620`).
- Compressed cache: owned only by kv-source layers, one row per `compress_ratio` tokens
  (`attention.py:1290-1323`, `tokens_per_state=self.compress_ratio`); replicated per TP rank (`max_tp_shards=1`).
- Index-K cache: owned only by kv-source layers; MXFP4 row = 64 + 4 = **68 B** when `indexer_kv_dtype=mxfp4`
  (InferenceX MI355X vLLM recipe sets it), else fp8 128 + 4 = 132 B (`attention.py:185-193`, `:537-551`;
  `indexer.py:67-92`; recipe [F: single_node/.../vllm/mi355x-fp4-mtp/agentic.yaml:37]).
- SWA cache: per attention layer, block 32, 584 B/row, window 128 (`attention.py:649-665`; `sparse_swa.py:77-147`).
- Compressor tail state: fp32 circular buffer (`CircularBufferManager`), rows = (drafts + 2) rounded to a power of
  two >= 8 (`compressor.py:240-270`).

Analytic cell [A]:
- compressed: 3 x 584 / 2 + 1 x 584 = **1,460 B/token**
- index-K (mxfp4): 3 x 68 / 2 + 68 = **170 B/token** (fp8 index-K: 330)
- **full cell = 1,630 B/token** (1,790 with fp8 index-K)
- SWA per request: (128/32 + 1) blocks x 32 x 584 = 93,440 B/layer -> 3.74 MB (40 layers) / 4.02 MB (43 incl. draft)
  [I: number of SWA layers in vLLM incl. DSpark draft not confirmed]; the Mooncake connector clips SWA groups to
  `cdiv(window, block)+1` blocks before transfer (`mooncake_connector.py:1075-1106`), consistent with this count.

Engine-reported cross-checks:
- vLLM TP2 (chaox log, 2026-10-14 image): `Available KV cache memory: 66.44 GiB`, `GPU KV cache size: 28,905,651
  tokens` -> **2,468 B per token-equivalent** [F: `/perf_apps/chaox/.../conc32/baseline/server.log:1144-1146`; A].
  The 838-B gap to the analytic 1,630 B is **not explained** (candidates: SWA / compressor-ring / draft groups
  sized from the same pool, page padding, an older image's layout) — measure, do not assume.
- SGLang (yanfwang, MI355X TP2/DPA2, 2026-10-28): `bytes_per_full_token=1673.31` plus a separate fixed SWA pool
  36.71 GB [2nd: `/perf_apps/yanfwang/dsv41_kvlen_bench_1028/logs/` server log lines "DSV4 memory calculation"].

Per-request bytes [A] (analytic = 1,630 x ISL + 4.02 MB; engine-slope = 2,468 x ISL; GLM-5.2 = 47,700 x ISL for scale):

| ISL | analytic | engine slope | GLM-5.2 cell |
|---:|---:|---:|---:|
| 64,000 (sub-agent p50) | 108 MB | 158 MB | 3.05 GB |
| 142,016 (trace p50) | 236 MB | 350 MB | 6.77 GB |
| 173,006 (served mean, vLLM C32) | 286 MB | 427 MB | 8.25 GB |
| 218,922 (trace mean) | 361 MB | 540 MB | 10.4 GB |
| 989,824 (max) | 1.62 GB | 2.44 GB | 47.2 GB |

### 3.2 Wire bandwidth demand vs NICs

Formula [A]: qps = S x N / (ISL_mean + OSL_mean); wire bytes/s = qps x bytes_per_request(ISL_mean)
(whole context resent, decode prefix cache off). With ISL 173,006, OSL 1,023.5:

| score S (tok/s/GPU) | GPUs N | qps | analytic GB/s | engine-slope GB/s |
|---:|---:|---:|---:|---:|
| 107,583 (vLLM C32 anchor) | 8 | 4.95 | 1.41 | 2.11 |
| 107,583 | 16 | 9.89 | 2.83 | 4.22 |
| 161,375 (+50 %) | 16 | 14.84 | 4.24 | 6.33 |
| 250,000 | 16 | 22.98 | 6.57 | 9.81 |

NIC side:
- openai cluster (smci355): **NIC inventory not found** in `~/.claude/skills/machine-pool/clusters/openai.md`
  (searched for nic/rdma/ionic/400/rail; none). Must be measured (section 7, M6).
- crusoe MI355X (GLM-5.2 nodes): 8 x ionic (Pensando) rails, physically isolated (ionic_i reaches only ionic_i),
  measured 38.3-38.6 GB/s per rail local-NUMA GPU, 27.5-29.6 GB/s cross-NUMA [2nd: knowledge repo
  `pdx-case-crossnode-g3x1/content/items/env/environment.md:11-12`]. 8 rails ~306 GB/s measured; nominal 8 x 400 Gb/s
  = 400 GB/s.
- Demand at +50 % on 16 GPUs = 4.2-6.3 GB/s = **1.4-2.1 % of 306 GB/s** (one rail alone = 11-17 %). [A]
- Per-request latency floor on one rail at served mean: 0.29-0.43 GB / 38.3 GB/s = 7.5-11 ms [A]; real latency will be
  set by descriptor count / small pages, not bandwidth [I]: at block 128 a ratio-2 compressed page is 64 x 584 =
  37,376 B, a ratio-2 MXFP4 index page 64 x 68 = 4,352 B; ~1,350 blocks per group at ISL 173k x 8 groups -> ~10.8k
  pages per request if not coalesced [A]. Whether vLLM packs a block's compressed + index pages into one region
  ("block stride is the sum of every page in the group", `attention.py:1395-1397`) is not established for the
  connector path.

### 3.3 Decode KV capacity

Inputs [F]: MI355X device 287.98 GiB; vLLM TP2 weights 145.15 GiB/GPU with Engram on host, peak activation
15.71 GiB, CUDA graphs 11.15 GiB at util 0.90 (`server.log:188, 1394-1398`); checkpoint groups from safetensors
headers: routed experts 268.95 GiB, Engram tables 188.83 GiB, draft experts 6.72, attention 4.87, other 2.82,
shared experts 1.42, vision 0.77 GiB.

Estimate for a TP4 decode at util 0.95 [A, assumes even sharding and TP2's activation/graph overheads]:
weights 145.15/2 + 188.83/4 = 119.8 GiB/GPU (Engram on GPU) -> KV 0.95 x 287.98 - 119.8 - 15.7 - 11.2 =
126.9 GiB -> 55.2M tokens at 2,468 B -> **~320 requests at ISL 173k per TP4 instance** (KV replicated per TP rank);
Engram on host (+47.2 GiB) -> 75.8M tokens, ~440 requests. Under DP-attention x4 each rank holds only its own
requests -> ~4x. For comparison the TP2 colocated anchor holds 28.9M tokens (~167 requests at 173k) and ran C32.
Verdict [I]: decode KV capacity binds only at hundreds of resident requests per instance; decode batch will be set by
step time / max-num-seqs / graph capture first.

### 3.4 Prefill vs decode compute split

From the vLLM TP2 C32 colocated anchor (rank 0, folded to the baseline hour): device time DECODE 2,931.0 s,
PREFILL 839.4 s of 3,770.4 s [2nd: chaox `conc32/README.md` "Share basis"]. Per request [A]: prefill 0.1875 s,
decode 0.655 s of TP2-instance device time; mean running decode batch = qps x OSL x TPOT = 1.241 x 1023.5 x
6.11 ms ~ 7.8.

- Prefill-only ceiling (if decode cost were zero and the 0.96 hit held): 1 / (2 x 0.1875) = 2.67 req/s per GPU
  = ~464k tok/s per prefill GPU [A; upper bound, ignores queueing, chunk scheduling, hit loss at higher C].
- Decode cost model (SGLang only, TP2/DPA2 MI355X, yanfwang 2026-10-28): step ms = 12.81 + 0.656 x requests/rank at
  16K context; at B=16/rank the step is flat-to-falling with context (24.5 ms at 16K, 22.3 ms at 262K)
  [2nd: `/perf_apps/yanfwang/dsv41_kvlen_bench_1028/analysis.txt`]. Applying it to vLLM is [I]. With
  acceptance 3.51 it gives 2.4k / 3.3k / 4.1k / 4.6k output tok/s per rank at B = 16 / 32 / 64 / 128, i.e. 2.4-4.5
  requests/s per decode rank (OSL 1,023.5) [A on 2nd].
- Demand: output tok/s = qps x 1,023.5; input-dominated score means decode GPUs needed per prefill GPU is small:
  at 2.67 req/s per prefill GPU and 4.0 req/s per decode rank (B=64), P:D by GPU ~ 1.5:1 [I]. GB300 vLLM PD runs
  6P(DEP2)+DEP32 (12:32 GPUs) to 16P(DEP2)+DEP32 (32:32) [F: gb300 disagg-variants.yaml] - a different hardware
  balance; MI355X ratio must be measured.

### 3.5 What PD can buy vs colocated (mechanisms named; magnitudes unmeasured)
1. **No prefill/decode interference.** Colocated vLLM runs chunked prefill (16,384-token chunks in the recipe) in the
   same steps as decode; cold first turns reach 0.99M tokens (new-token p99 41k, max 949k, section 2.2). PD isolates
   decode TPOT from these. Magnitude at the knee: unknown (M7).
2. **Independent parallelism and batch.** Decode can run DP-attention/EP at large batch with decode-only graphs
   (`FULL_DECODE_ONLY`) and mono kernels; prefill can run larger chunks and its own TP/DP. Note the ROCm mono decode
   kernels only cover decode steps up to 48 rows (recipe comment) [F: mi355x vllm agentic.yaml:50-53].
3. **Decode batch decoupled from KV.** Tiny KV per request lets the decode batch grow to the step-time optimum.
4. **Prefill prefix capacity.** Prefill GPUs no longer hold decode KV; plus host offload of prefix KV
   (SimpleCPUOffloadConnector) and of Engram tables.
Costs: extra weight replicas (145 GiB/GPU at TP2 per instance), transfer/handshake latency added to TTFT (in the
closed loop TTFT lowers qps directly, handbook measurement.md §1), router complexity, unproven ROCm PD path.

---

## 4. vLLM PD specifics that differ from SGLang

| topic | vLLM (main 7a62ca0 unless noted) | SGLang (handbook, GLM-5.2) |
|---|---|---|
| Decode prefix cache | D side computes `count = remote_tokens - num_computed_tokens` and pulls only **unhashed** blocks; a full local hit pulls nothing and just frees P (`mooncake_connector.py:1180-1250`). GB300 vLLM disagg recipe nevertheless sets `no-enable-prefix-caching` on decode [F: gb300 disagg-variants.yaml:166] | decode radix off by default; hybrid-SWA rejected; whole context resent |
| SWA transfer | SWA groups clipped to the last `cdiv(window,block)+1` blocks (`mooncake_connector.py:1075-1106`) | SGLang DSv4 has its own SWA pool sizing (`DSV4 SWA sizing` log) [2nd] |
| Bounded replay | with SWA bounded replay the D side loads N-1 tokens and recomputes the last (`mooncake_connector.py:1107-1114`); the MI355X CI anchor ran `--no-swa-bounded-replay` | n/a |
| Compressor tail state | separate `CircularBufferManager` group (fp32 ring). MoRIIO main rejects it (`moriio_connector.py:655-675`); vllm#68513 (OPEN, 2026-11-18) adds it. Mooncake registers circular-buffer regions unpacked (`mooncake_connector.py:2432-2441`) | n/a |
| PP | refuses PP splits inside a kv-sharing group (`attention.py:553-563`, `:686-699`); groups = layers 2-7, 8-13, 14-19, 20-39 -> only PP2 at layer 20 is balanced [A] | PP4x TP1 prefill was the GLM-5.2 base |
| Connectors on ROCm | MooncakeConnector (HMA, DSv4.1-aware code), MoRIIOConnector (AMD, needs #68513 for DSv4.1), NixlConnector (NVIDIA recipes; ROCm use not checked), TileRT custom connector (prefill side only) | Mooncake |
| torch.compile / graphs | DSv4.1 is not torch-compiled; breakable CUDA graphs needed for FULL_AND_PIECEWISE (`VLLM_USE_BREAKABLE_CUDAGRAPH=1`) [F: recipe env] | — |
| Router | vllm-router `--vllm-pd-disaggregation --kv-connector moriio --policy consistent_hash` (PR #68513 body); NVIDIA recipes use Dynamo kv router with soft session affinity, decode active-request weight 1e7 | SGLang router kv-aware + session affinity + R1/R2 |
| Async admits | `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` 2-6 on GB300 decode [F] | — |

Upstream evidence for DSv4(.1) PD in vLLM: vllm#41957 (merged 2026-05-09, NIXL DSv4-Flash DPEP4/TP4 PD GSM8K
0.95), vllm#45831 (merged 2026-06-17, DSv4 PD 4P4D on 8xH100 after an `is_mla_region` fix), vllm#68513 (OPEN,
MoRIIO DSv4.1 on MI325X), vllm#60808 (merged 2026-10-21, MoRIIO READ must await indexer-cache reads; validated on
GLM-5.3-Flash MI300X). [O]

---

## 5. Applicable-lever shortlist for Phase 3 (ranked by probability x size; all [I] until measured)

1. **Concurrency (lanes/GPU) on the best colocated and PD shapes.** The anchor runs 16 lanes/GPU; GB300 PD runs
   69-170. Expected to be the largest single mover; re-sweep after every structural change.
2. **P:D ratio and instance count** (prefill-heavy by GPU; section 3.4), plus same-node vs cross-node placement
   (re-measure; transfer bytes are ~20-30x smaller than GLM-5.2).
3. **Decode layout: TP vs DP-attention+EP (DEP), max-num-seqs, capture sizes in multiples of 6.** DP removes the
   per-rank KV replication and sets decode batch per rank; mono-kernel coverage stops at 48 rows.
4. **Prefill parallelism and chunking**: TP2/TP4 vs DP+EP, `max-num-batched-tokens`, long-prefill threshold; PP2 at
   layer 20 only.
5. **Router**: prefill session affinity (prefix hits; trace hit 0.98 token-weighted), decode pick by load.
6. **Prefill prefix capacity**: host KV offload connector; Engram tables on host (frees 47 GiB/GPU at TP4) on the
   leg that needs memory.
7. **Decode prefix cache on** (suffix-only pulls) - A/B against off; risk: bounded-replay interplay.
8. **DSpark depth** (5 drafts default), measured real acceptance per depth before any simulated A/B.
9. **Transfer path tuning** only if M5 shows transfer on the critical path (async admits per step, descriptor coalescing).
10. **Kernels** once step 5 says compute binds: MoE a8w4, sparse MLA, mxfp8 GEMM, quick-allreduce, mono decode
    (chaox extraction shares), GEMM retune.

Not applicable / low value: DCP decode (KV tiny), FP4 KV on ROCm (SM100 only), PP4 prefill (illegal split),
SGLang-only levers (index_share, OffloaderV2, HiCache E1/E2, SPF without a vLLM implementation).

---

## 6. Risks

- **ROCm PD path immaturity**: MoRIIO DSv4.1 fix unmerged (#68513); Mooncake DSv4.1 on ROCm unvalidated (not found);
  past bugs: indexer-cache reads not awaited (#60808), backend selection with `fp8_ds_mla` (#62311).
  Correctness gate (GSM8K + long-context NIAH, real acceptance) before any perf number.
- **Cell mismatch** (1,630 analytic vs 2,468 engine): capacity and transfer predictions may be off by ~1.5x.
- **Closed-loop TTFT coupling**: PD adds handshake + transfer + queueing to TTFT; every +1 s TTFT cost ~2.5 % on
  GLM-5.2 at C44 [2nd: handbook measurement.md §1].
- **Small-page transfer overhead** (4.3 KB index pages) could make per-request transfer latency op-bound.
- **Bounded replay N-1** and DSpark draft state at the P->D boundary: correctness interplay not checked.
- **Engram host tables**: host DRAM / PCIe load on shared nodes; GB300 notes RAM exhaustion with Engram + host KV
  offload when frontend shared prefill hosts [F: gb300 disagg-variants.yaml:45].
- **Comparability**: InferenceX MI355X vLLM numbers use synthetic acceptance 3.51; never compare across that
  boundary.

---

## 7. Measurements Phase 2 must take to decide PD feasibility

| id | measurement | decides |
|---|---|---|
| M1 | Colocated vLLM concurrency sweep on the CI recipe (TP2 and TP4, C32 -> knee), 3600 s official mode; record score, qps, mean ISL, prefix hit, TTFT/TPOT p50/p90, decode running count, KV usage, prefill vs decode device-time split at each point | the reference PD must beat; what binds at the knee |
| M2 | Startup KV lines on every candidate leg: `Available KV cache memory`, `GPU KV cache size`, per-group spec dump -> explain the 2,468 vs 1,630 B gap; confirm SWA layer count | capacity + transfer bytes |
| M3 | Decode-only step time vs batch (8..256 rows) and vs context (16k..512k) on vLLM, graph ON, TP4 and DP4+EP (vLLM `decode_bench_connector.py` exists in main) | decode GPUs per qps; mono-kernel 48-row cliff |
| M4 | Prefill-only throughput on the AgentX mix: new tokens/s per GPU at cached-prefix lengths 64k/173k/550k, chunk 8k/16k/32k | prefill GPUs per qps; P:D ratio |
| M5 | PD bring-up on MI355X: MooncakeConnector (main) and MoRIIO (#68513), 1P1D cross-node and same-node; correctness gate; per-request transfer latency, bytes (connector stats) vs prediction, NIC tx per rail | feasibility on ROCm |
| M6 | openai-node RDMA inventory (`ibv_devices`, link rate, rail isolation, GPU<->NIC NUMA map) and a per-rail byte-verified probe | wire budget denominator |
| M7 | Interference in colocated mode: TPOT distribution of running decodes during long cold prefills (bucket by prefill chunk present / absent) | size of the PD "interference" win |
| M8 | Decode prefix cache on vs off under PD: transfer bytes per request, TTFT, score | suffix-only pull lever |
| M9 | Real DSpark acceptance on AgentX-like prompts (per depth) vs synthetic 3.51 | step-1 waiver labelling, depth choice |

Feasibility rule (proposal): proceed with PD optimisation only if M5 passes correctness, and the PD estimate from
M3 + M4 at the M1 knee concurrency (same total GPUs) exceeds the M1 best by more than the measured noise band.

---

## 8. Not found / gaps (where I searched)
- openai cluster NIC/RDMA details: not in `machine-pool/clusters/openai.md`, Muse workspace, or knowledge repo
  (only crusoe ionic data found).
- vLLM DSv4.1 PD on MI355X with any connector: no merged PR or recipe found (`gh search prs` queries "deepseek v4
  disaggregated", "dsv4 PD", "MoRIIO deepseek", "mooncake connector deepseek v4", "DSV41 PD"); only #68513 (MI325X).
- Any measured vLLM colocated-vs-PD A/B for DSv4.1 on any hardware: not found in InferenceX recipes (results are not
  in the repo).
- Served AgentX ISL distribution with the DeepSeek tokenizer beyond the C32 mean (173,006): not found.
- Composition of vLLM's 2,468 B/token-equivalent KV slope: not established.

---

## Verification (adversarial, 2026-11-19)

Independent re-check of 12 claims against their cited sources. Confirmed claims are listed briefly. Refuted or corrected claims give the correct fact and its source.

| # | claim (short) | verdict | note / correct fact + source |
|---|---|---|---|
| 1 | KV record 584 B, compressed cache on kv sources 2/8/14/20, MXFP4 index-K 68 B, cell 1,630 (1,790 fp8), SWA ~93 KB/layer/req | confirmed | `attention.py:185-193` (68 / 132 B), `:198-204, 610-611` (584 B), `:1290-1299` (only kv sources own a cache), `:1433-1440` (non-owning index sources share the kv source's K cache, so 4 index-K owners rather than 8), `:649-665` (SWA block 32 -> (4+1)x32x584 = 93,440 B). config `kv_source_layer_ids` [2,8,14,20], `compress_ratios`. InferenceX mxfp4 setting is at `vllm/mi355x-fp4-mtp/agentic.yaml:38`, not :37. |
| 2 | TP2: 66.44 GiB KV = 28,905,651 tokens = 2,468 B; weights 145.15 GiB; device 287.98 GiB | confirmed | `server.log:188, 1144, 1146, 1394`; Engram on host: `server.log:7, 65` (47.21 GiB per rank offloaded). |
| 3 | Anchor 107,583 tok/s/GPU with ISL 173,006 / OSL 1,023.5 / TPOT p50 6.11 ms / qps 1.241 | **corrected** | The claim mixes two runs. Official CI job 114309765517: **107,988.6** tok/s/GPU total, ISL 173,005.9, OSL 1,023.53, TPOT p50 6.11 ms (mean 7.58), 4,476 req / 3,606.7 s = 1.241 req/s (`conc32/official_ci/bmk_result/dsv41flash_tp2_conc32_..._spec-mtp-d9729896e9d42d7c2537.json`). chaox reproduction 2026-10-14: **107,583.4** tok/s/GPU, ISL 173,269.7, OSL 1,023.2, TPOT p50 6.22 ms (mean 7.53), 4,464 req / 3,616.0 s (`conc32/baseline/results/dsv41flash_tp2_conc32_repro.json`). GPU prefix hit 0.960, synthetic acceptance 3.51, Engram on host and `--no-swa-bounded-replay` hold for both (`launch_server.sh:22-23`). |
| 4 | Device time DECODE 2,931 / PREFILL 839 of 3,770 s; 0.19 / 0.65 s per request; mean decode batch ~7.8 | **corrected (minor)** | The share basis is confirmed (`conc32/README.md:14`, rank 0). The "mean decode batch ~7.8" uses TPOT **p50** (6.11 ms). With the official **mean** TPOT of 7.58 ms, 1.241 x 1,023.5 x 7.58 ms = **~9.6**. Also, per-request device time is rank-0 device time, not TP2-instance time. |
| 5 | Wire 4.2-6.3 GB/s at 161k x 16 GPUs; 1.4-2.1% of 8 x 38.3 GB/s; 0.29-0.43 GB/request vs 8.3 GB for GLM | confirmed (arithmetic) | 161e3x16/174,029.5 = 14.80 req/s; x173,006 x {1,630; 2,468} B = 4.17 / 6.32 GB/s; /306.4 GB/s = 1.37% / 2.06%; 173,006 x 47.7 KB = 8.25 GB. The rail probe is at `pdx-case-crossnode-g3x1/content/items/env/environment.md:12` (crusoe ionic, local-NUMA 38.3-38.6 GB/s). This assumes the whole ISL is sent on every request (no D-side prefix reuse). |
| 6 | Trace corpus: 393 traces, 98,827 requests, ISL mean 218,922 / p50 142,016 / p99 863,488, OSL 1,077, new tokens 3,702, hit 0.983, input share 0.9951, 144.5 main turns | confirmed | Recomputed independently with a separate parser on `.../snapshots/23f152f6.../traces.jsonl`: 393 traces, 98,827 requests (56,798 main + 42,029 sub-agent), ISL mean 218,921.8, p50 142,016, OSL mean 1,077.4, input share 0.99510. Units are the trace's own (64-token hash blocks, Claude tokenizer), not DeepSeek tokens. |
| 7 | MoRIIO rejects CircularBufferManager; #68513 OPEN, MI325X XGMI only; no merged ROCm DSv4.1 PD validation | confirmed | `moriio_connector.py:650-680` (CircularBufferManager is allowed only with mamba + READ mode). `gh pr view 68513`: OPEN, updated 2026-11-18, "Tested on MI325 with XGMI backend". Note that the P command is `--data-parallel-size 2` on 2 GPUs with no TP flag, although its label is "TP2+DPA". D is TP4. `gh search prs "DSv41 PD"` returns only #68513 for PD. |
| 8 | Mooncake SWA clip cdiv(window,block)+1; bounded replay N-1; D pulls only unhashed blocks | confirmed | `mooncake_connector.py:1075-1106` (clipping applies only when `_is_hma_required`), `:1107-1114`, `:1206-1250`. |
| 9 | vLLM refuses PP splits inside a kv-sharing group, so the only balanced PP is PP2 at layer 20 and PP4 is impossible | **corrected (scope)** | This is true only for the shared/ROCm path: `attention.py:553-563, 686-699` raise NotImplementedError, and `amd/model.py` has no pipeline-boundary relay. vLLM main **does** support cuts inside sharing groups on the NVIDIA path: `nvidia/pipeline.py:44-120` (`get_pipeline_boundaries`, `PipelineCache`, `copy_pipeline_cache_rows`), used by `nvidia/model.py:133, 822`. Restate it as "on ROCm at 7a62ca0". |
| 10 | TP4 decode at util 0.95: ~127 GiB KV, ~55M tokens, ~320 req (~440 with Engram on host); DPA ~4x | **corrected** | The estimating method does not reproduce the measured TP2 point. Applied to TP2 at util 0.90 it gives 0.90x287.98 - 145.15 - 15.71 - 11.15 = 87.2 GiB, but the measured value is 66.44 GiB. vLLM's own accounting (`server.log:1394`) is 259.19 - 177.03 (weights + **31.88 GiB non-torch**) - 15.71 activation = 66.45. CUDA-graph memory is not subtracted there, and the 31.88 GiB of non-torch memory is omitted by the estimate. The baseline already runs Engram on host, and the TP4 per-rank Engram size is not established. So the ~320 / ~440 request figures are **unverified**. Re-applying vLLM's accounting with even sharding and unchanged non-torch and activation [inference, untested] gives ~153 GiB, ~67M tokens and ~386 requests at Engram on host. Measure it (M2). `max_tp_shards=1` (KV replicated per TP rank) is confirmed at `attention.py:1309`. |
| 11 | GB300 vLLM PD: NIXL, prefill DEP2 + SimpleCPUOffload 60-70 GiB, decode DEP16/32 with no prefix caching, up to 16P:1D at C10,888; only MI355X PD recipe is TileRT | confirmed | `vllm/gb300-fp4/agentx/disagg-variants.yaml:105, 122, 175, 179, 197-208, 239` (64424509440 B = 60 GiB, 75161927680 B = 70 GiB). `tilert/mi355x-fp4/agentx/disagg-1p1d-tp4-tp8-mtp.yaml:17, 63-67` (vLLM prefill with DsV41TileRTConnector, transport mooncake, engine tilert). Local clone db3fc79 = upstream main HEAD (2026-11-19T06:28Z). The multi_node dsv41flash dirs upstream are sglang / tilert / vllm (b300, gb200, gb300). |
| 12 | SGLang decode step 12.81 + 0.656/req at 16K; B=16 24.5 ms at 16K, 22.3 ms at 262K; bytes_per_full_token 1673.31 | confirmed | `/perf_apps/yanfwang/dsv41_kvlen_bench_1028/analysis.txt`; `logs/server_kv.log:139`; `server.sh:20-22` (TP2, DP2, DP attention, DSPARK block 5); the logs show gfx950. |
