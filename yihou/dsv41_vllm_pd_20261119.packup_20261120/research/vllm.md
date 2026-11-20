# vLLM support and performance for DeepSeek-V4 / V4.1-Flash (focus: ROCm / MI355X, PD disaggregation)

Date: 2026-11-19. Phase 1 research only (no GPU runs).
Evidence tags: [first-hand: file:line @ vllm main 7a62ca0666 (2026-11-19)] = code read in
`source/vllm` (shallow clone in this workspace); [official: PR#/URL] = read from the PR/issue
body or official repo via gh; [second-hand] = third-party page; [inference] = my reasoning, not verified.
All vLLM paths below are relative to `source/vllm/` in this workspace.
Raw PR-body dumps used for this file: `notes/vllm_research_pr_bodies.txt`, `notes/vllm_research_ix_prs.txt`.

---

## 0. Decision-relevant summary

1. **Model code.** DSv4.1-Flash (`DeepseekV41ForCausalLM`, the arch in our checkpoint's config.json) lives in
   `vllm/models/deepseek_v41/` with a dedicated ROCm tree `vllm/models/deepseek_v41/amd/`
   [first-hand: vllm/model_executor/models/registry.py:382-384; /perf_apps/chaox/models/DeepSeek-V4.1-Flash/config.json:2-4].
   Arch shipped in v0.30.0 (#56228); recipe min version 0.31.0; **MI355X needs the `nightly-rocm100`
   image** because `--monokernel` and the paged MXFP4 indexer are on main, not in 0.31.0
   [first-hand: recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml:35-50]. Latest release v0.34.0 (2026-11-16);
   latest ROCm nightly `nightly-rocm100-9c1d45c2…` (2026-11-19) [official: gh release list; Docker Hub tags API].
2. **No vLLM PD result for DSv4.1 on AMD exists in any official source I found.** The only verified vLLM PD lanes
   are NVIDIA (GB200 1P1D TP4+TP4 NIXL; GB300 Dynamo 5P/7P/12P DEP2 prefill + DEP16/32 decode, NIXL push)
   [first-hand: recipe yaml:384-399, 1623-1800]. The only MI355X "PD" item in InferenceX is #4756
   (vLLM TP4 prefill + **TileRT** TP8 decode over Mooncake, c1 only, merged as draft config, "do not start a sweep")
   [official: InferenceX#4756].
3. **MoRIIO (the ROCm-native connector) on main REJECTS DSv4.1.** DSv4.1's compressor state uses
   `CircularBufferManager` [first-hand: vllm/models/deepseek_v41/compressor.py:259-270]; MoRIIO under the hybrid
   KV manager only accepts FullAttention / SlidingWindow / mamba groups, and CircularBuffer only together with
   mamba in READ mode [first-hand: vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:650-681].
   Reproduced error on main and with `--disable-hybrid-kv-cache-manager` ("Failed to promote local KV cache specs
   to one unified type") is in **open PR #68513** (2026-11-18), which lifts the restriction; tested on MI325X,
   xGMI, READ mode, P = TP2+DPA2, D = TP4; accuracy section still "TODO" [official: vllm#68513].
4. **NIXL has explicit DSv4.1 handling on main** ("CSA-linear" path for CircularBuffer groups, DSv4.1 SWA bounded-replay
   prefill backoff) [first-hand: nixl/base_worker.py:734-800, nixl/base_scheduler.py:395-410] and a ROCm NIXL/UCX
   build stage exists in `docker/Dockerfile.rocm:36-137` [first-hand]. No ROCm run of NIXL + DSv4.1 found.
5. **Mooncake** connector also handles CircularBuffer groups and DSv4.1 bounded replay [first-hand:
   mooncake/mooncake_connector.py:1100-1115, 2430-2445]; InferenceX#4756 notes the vLLM nightly's PyPI Mooncake
   "sends cross-node writes over HIP IPC and its ionic provider lacks kernel ABI 4", needing a custom build
   [official: InferenceX main, srt-slurm-recipes/dsv41flash/tilert/mi355x-fp4/agentx/disagg-1p1d-tp4-tp8-mtp.yaml:19-20].
6. **Any PD connector disables the AITER Gluon sparse-MLA fast path on ROCm for DSv4.1.** `_aiter_sparse_mla_enabled`
   returns False when `kv_connector_may_load_foreign_kv` is True [first-hand: vllm/models/deepseek_v41/amd/rocm.py:95-109,
   597, 624-626]; that is True unless the connector class overrides `only_restores_canonical_kv` (base returns False)
   [first-hand: kv_connector/v1/base.py:236-244; kv_connector/utils.py:103-114]. Only UMBP (and MultiConnector of
   UMBP-only children) override it [first-hand: umbp/connector.py:110; multi_connector.py:274-280]. #64715 states
   "Before this PR, any KV connector turned them off" [official: vllm#64715]. Same gate turns off the
   `extra_cache_nan_free` path (`rocm.py:82-92,1254-1258`) and the open mega-attn prefill kernel #65146 requires
   "no KV connector" [official: vllm#65146]. Perf impact of this for MoRIIO/NIXL not measured anywhere I found.
7. **Best published MI355X vLLM AgentX throughput (aggregated, single node):** DP2 (attention DP2, MoE TP2), c256:
   250,435 tok/s/GPU (no offload) -> 275,836 with UMBP; c384: 143,888 -> 291,619 with UMBP (local 1200 s fast runs);
   one-hour local c256 310,739 tok/s/GPU [official: InferenceX#5036, #4992]. Metric = InferenceX "Total throughput
   per GPU" (definition not verified by me; it is far above output throughput, so it counts input/prefix tokens).

---

## 1. Model files and ROCm code path

| Item | Status | Source |
|---|---|---|
| Arch `DeepseekV41ForCausalLM` -> `vllm.models.deepseek_v41` | on main | [first-hand: registry.py:382-384] |
| ROCm model tree | `deepseek_v41/amd/{model.py,rocm.py,mono_decode.py,mono/,dspark.py,engram.py,mhc.py}` (5468 lines .py at top level) | [first-hand] |
| DSv4 (non-.1) | `vllm/models/deepseek_v4/` incl. `amd/mega_moe.py`, `amd/mtp.py` | [first-hand] |
| ROCm attention backend name | `ROCM_FLASHMLA_SPARSE_DSV41` (`DeepseekV41ROCMAiterMLASparseBackend`), kernel block sizes [64,128], `supports_pcp=True`, extra KV dtype `fp8_e4m3` | [first-hand: amd/rocm.py:537-559] |
| SWA backend | `DeepseekV41ROCMAiterSparseSWABackend` | [first-hand: amd/rocm.py:562-569] |
| Other ROCm backends rejected for V4.1 | `ValueError` unless ROCM_FLASHMLA_SPARSE_DSV41 / DSV4 (deprecated) | [first-hand: amd/model.py:175-195] |
| torch.compile | not supported for DeepseekV41ForCausalLM; ROCm sparse SWA backend reports only UNIFORM_BATCH -> needs `VLLM_USE_BREAKABLE_CUDAGRAPH=1` on gfx950 | [first-hand: recipe yaml:216-231] |

### KV cache groups of DSv4.1 (what a PD connector must move)

| Cache | Spec / manager | Source |
|---|---|---|
| Compressed KV (kv-source layers only) | `MLAAttentionSpec` / `FullAttentionManager` | [first-hand: deepseek_v41/attention.py:1290-1301] |
| Indexer K cache | `MLAAttentionSpec` (tokens_per_state=compress_ratio) / `FullAttentionManager` | [first-hand: deepseek_v41/attention.py:1370-1405] |
| SWA cache | `MLAAttentionSpec` / `SlidingWindowManager` (bounded_replay) | [first-hand: v1/attention/backends/mla/sparse_swa.py:120-148] |
| Compressor state ring | `AttentionSpec` fp32 / `CircularBufferManager` (1 block per request, not prefix-cacheable) | [first-hand: deepseek_v41/compressor.py:250-270; v1/core/single_type_kv_cache_manager.py:1630-1660] |

Recipe: global KV "890 bytes per token" (DeepSeek's figure) plus a 128-token SWA window per layer
[first-hand: recipe yaml guide "Memory" section ~line 2110]. #66036: plain fp8_e4m3 = 512 B/token vs 584 B packed
fp8_ds_mla (+12.5% KV capacity) [official: vllm#66036].

---

## 2. Feature matrix (ROCm / MI355X unless stated)

| Feature | Status on ROCm gfx950 | Source |
|---|---|---|
| MXFP4 MoE via AITER | `--moe-backend aiter` (CK a8w4 experts); `aiter_triton_mxfp4_bf16` pins Triton W4A16 | [first-hand: recipe yaml:216-224, guide ~2138] |
| AITER MegaMoE (DP+EP over MoRI) | merged #66644 (2026-11-10); requires `--enable-expert-parallel` | [official: vllm#66644; first-hand: amd/model.py:620-628] |
| MegaMoE A4W4 + MXFP4 combine, overlap | open #68385 (+22.2% output tok/s, DEP8 1k/8k c256, needs aiter PR #8186) | [official: vllm#68385] |
| TP8 MegaMoE | open #63040; its C16 8K/1K test regressed -12% / -13% | [official: vllm#63040] |
| FHMoE DP8 over RCCL | open #54134 (+8.06% output tok/s in its test) | [official: vllm#54134] |
| Mono decode (`--monokernel`) | merged (#60397, #61082 all 40 layers, #64101 TP8); needs TP in {2,4,8}, CDNA3/4, FlyDSL; decode steps <=48 rows; rejects DP (per recipe and #64715 text) | [first-hand: amd/mono_decode.py:404-443; official: vllm#64101, #64715] |
| Gluon/Triton sparse MLA (aiter) | `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1` (#53492); **auto-disabled with any non-UMBP KV connector** | [first-hand: amd/rocm.py:95-109] |
| Mega-attention prefill kernel | open #65146 (prefill 1.15-1.73x op speedup); auto-selection excludes KV connector, PCP, DCP | [official: vllm#65146] |
| Paged MXFP4 sparse indexer | `--attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'`, `--block-size 128` (#58671) | [first-hand: recipe yaml:290-300] |
| KV cache dtype | default `fp8_ds_mla` packed record; `fp8_e4m3` per-tensor (#66036, merged 2026-11-12) | [official: vllm#66036; first-hand: amd/rocm.py:537-544] |
| Prefix caching + SWA bounded replay | on; decoder replay #59039, prefix-cache replay on gfx950 #64094 (+13.5% c128, +12% c256 "best MI355X config") | [official: vllm#64094; first-hand: recipe guide ~2195] |
| CUDA graph | FULL_AND_PIECEWISE via breakable CUDA graph; gfx942 uses FULL_DECODE_ONLY | [first-hand: recipe yaml:226-231, 272-288] |
| Separate prefill/decode capture sizes | merged #63406 | [official: vllm#63406 title] |
| DSpark spec decode (5 drafts) | supported; adaptive verification on ROCm #52362; AMD fused Markov sampling #63891 | [official: PR titles; recipe] |
| DP-attention (DP2, MoE TP2) | verified AgentX lane (InferenceX#5036), vllm-router consistent_hash | [first-hand: recipe yaml:1849-1928] |
| DEP8 (DP8 attn + EP8 MegaMoE) | runs (#67377 benchmark config); InferenceX#5056 open | [official: vllm#67377, InferenceX#5056] |
| Engram tables | host offload (`cpu_offload:true`) at TP2 (94.4 GiB/rank otherwise); resident at TP4 (47.2 GiB/rank); `embedding_across_dp` for DEP | [first-hand: recipe yaml:247-256; InferenceX#5056] |
| PCP (prefill context parallel) | eager PCP merged #66842; on ROCm requires PP=DP=DCP=1, **no spec decode, no KV transfer, no prefix caching, no MegaMoE, no monokernel** | [first-hand: amd/model.py:197-216] |
| DCP (decode context parallel) | **not on main**; open #57239 (ROCm, WIP, test plan TBD), #44573 (NVIDIA, inactive) | [official: vllm#57239, #44573] |
| PP | DSpark+PP open (#63321, #63619); PP split inside an index-sharing group unsupported | [first-hand: amd/rocm.py:615-623] |
| Fused shared experts on AMD V4.1 | NotImplementedError | [first-hand: amd/model.py:629-640 region] |
| KV offload UMBP (MORI host pool) | merged #64715; verified DP2 lane | [official: vllm#64715] |

---

## 3. Perf numbers (every number with source)

Notation: AgentX = InferenceX agentic multi-turn trace (`semianalysis_cc_traces_weka_062126` where stated), DSpark
5 drafts with synthetic acceptance length 3.51 unless stated. "Total tok/s/GPU" = InferenceX metric as reported.

### 3.1 MI355X, vLLM, AgentX (aggregated)

| Source | Date | Engine/image | Config | Conc | Metric | Value |
|---|---|---|---|---|---|---|
| InferenceX#5036 (merged 2026-11-18) | local fast runs (1200 s), `nightly-96710245` + main `595b1514e8`, fp8_ds_mla KV | DP2 (TP1xDP2, MoE TP), interval 11, 16384 budget, Engram host | 128 | Total tok/s/GPU no-UMBP -> UMBP | 193,099 -> 183,984 |
| same | same | same | 256 | same | 250,435 -> 275,836 (+10.1%) |
| same | same | same | 384 | same | 143,888 -> 291,619 |
| same | same | same | 544 | same | 46,641 -> 287,025 |
| same | same | same | 256/384 | p90 interactivity (tok/s/user) | 12.9->16.4 / 6.6->10.4 |
| InferenceX#4992 (closed) | one-hour local run, `nightly-083fbac02` + patched overlay, no FP8 KV | DP2, cadence interval 11 | 256 | tok/s/GPU, p90 intvty | 310,739; 16.6 |
| same | same | delayer N11 W5s (vllm#65655, open) | 256 | same | 317,021; 33.5 |
| vllm#65655 (open) | fast replays, 2x MI355X | DP2, default vs delayer N16 W5000 | 128 | Total tok/s/GPU | 186,066 -> 201,610 (+8.4%) |
| vllm#64715 (merged) | 3600 s, main `51c6dfd8c8` / `77d2790e78` | TP2 / DP2, UMBP | TP2 c128 / TP2 c192 / DP2 c256 | Total tok/s/GPU base -> UMBP | 245.3K->246.8K / 179.3K->262.5K / 270.3K->280.6K |
| InferenceX#5125 (open) | official sweep figure quoted | TP2 | 16 | tok/s/GPU | 58.3k official (40.8k local 900 s) |
| InferenceX#5125 | local 900 s, `nightly-rocm100-8ca76632` | TP4 c16 cap 2/3/4 | 16 | tok/s/GPU; p90 intvty | 20.8k/21.3k/21.3k; 576/454/409 |
| InferenceX#4480 (merged 2026-10-27) | `nightly-e68efc68` | TP4 c1 / TP2 c8 (after caps) | 1 / 8 | Total tok/s/GPU | 7,399 / 31,601 |
| InferenceX#4666 / recipe | `nightly-335dd77b` | TP8 mono c1 | 1 | interactivity p50/p90 | 696.9 / 685.5 tok/s/user, TTFT p50 0.17 s |
| vllm#67377 (merged 2026-11-15) | 8x MI355X, DEP8 (DP8 attn + EP8), aiperf agentic fast | MegaMoE?, interval 96, 32768 budget | 256 | "throughput_per_chip" | 26,714 tok/s/GPU, p90 intvty 4.61 |
| InferenceX#5056 (open) | local one-hour, 8x MI355X | DEP8 MegaMoE+MoRI, Engram in HBM | 256 | tok/s/GPU; p90 intvty | 182.7K; 120.6 (host Engram: 175.6K; 96.4) |

Note (suspended, not concluded): #67377's DEP8 c256 figure (26.7K tok/s/GPU, p90 4.6) and #5056's DEP8 c256
(182.7K, p90 120.6) differ by ~7x for nominally similar shapes; the cause is not stated in either PR. Do not compare
these without re-deriving the metric definition and workload settings.

### 3.2 MI355X, vLLM, synthetic / kernel-level

| Source | Setup | Number |
|---|---|---|
| vllm#64101 | TP8, random 32k/1k, AL 3.87, nightly `5d35ce3934` | c1 step 8.59 -> 5.36 ms with mono (580 tok/s); c4: 1163 tok/s; TP8 c4 is 26% below 2xTP4 (1262 vs ~1706 tok/s) |
| vllm#61082 | AgentX TP4 c1, 3600 s | p90 interactivity 541.6 -> 562.6 tok/s/user |
| vllm#66036 | TP2, DSpark AL 3.51, `vllm bench serve` random | 8K/1K c64 output tok/s 2111.9 (packed) -> 2272.0 (fp8_e4m3, +7.6%) |
| vllm#68385 (open) | DEP8, 1k/8k, c256, DSpark k=6 | output tok/s 32,397 -> 39,605 (+22.2%) |
| vllm#67377 | dense MXFP8 GEMM offline | prefill 189.8 -> 106.7 ms/step; decode 4544 -> 2822 us/step |
| vllm#65146 (open) | prefill op | 1.73x (16 tok) ... 1.15x (2048 tok) |
| vllm#64094 | DEP8, 1 request | TTFT 8k 492->316 ms; 32k 1.84->1.23 s; 128k 7.42->4.83 s |
| vllm#67364 | C1 TP8 step | 4.30 ms step; 716 output tok/s at 32K |
| vllm issue #56506 (closed, 2026-09-11) | TP4, early build | c32 output 469 tok/s (117 /GPU); 14,020 kernel launches per decode step, 85% mHC |

### 3.3 Other hardware (reference)

| Source | Setup | Number |
|---|---|---|
| vllm#61121 (open) | GB200 1P1D TP4+TP4 NIXL pull, 100k-token prompt | descriptors 11,118 -> ~430; MB/transfer 179.2 -> 103.6; xfer 4.56 -> 2.83 ms; TTFT p50 c1 1078 -> 1057 ms |
| InferenceX#5125 | B300 vLLM TP4 p90 interactivity | c1 826, c16 629, c32 469 |
| recipe guide | DGX Station GB300 TP1, `nightly-dee37d89` | 90.5 tok/s (1 stream) ... 416-423 (16 streams), prose |
| GB300 Dynamo PD (InferenceX#5015 open; recipe preset) | 5P1D DEP32 c3200, 7P1D DEP16 c6600, 12P1D DEP32 c10288 | no throughput numbers in PR body or recipe; GSM8K 0.976/0.975/0.973 |

---

## 4. PD connector feasibility for DSv4.1 on ROCm

| Connector | Registered | ROCm transport | DSv4.1 hybrid groups (Full+SWA-MLA+CircularBuffer) on main | Open fixes | Evidence of DSv4.1 run on AMD |
|---|---|---|---|---|---|
| MoRIIOConnector | yes (factory.py:230-234) | MoRI RDMA (default) / xGMI; mori v1.2.3.post1 in Dockerfile.rocm_base:15 | **Fails closed**: CircularBuffer accepted only with mamba in READ (moriio_connector.py:658-681). SWA-MLA allowed in WRITE (681-695) | #68513 (circular buffers for DSv41 PD), #68621 (SWA window anchoring), #60783 (kernel-block-split MLA/indexer views), #68072 (hetero-TP indexer caches), #65563 (unequal P/D DP routing), #68374, many robustness PRs | Only #68513 author: MI325X, xGMI, READ, TP2+DPA2 P -> TP4 D, no numbers |
| NixlConnector (pull/push) | yes (factory.py:200-216) | UCX; ROCm wheel built in Dockerfile.rocm:36-137 | Handles CircularBuffer as "CSA-linear"; pull mode for MLA/CSA-linear with hetero TP restricted ("use NixlPushConnector") (base_worker.py:1007); CSA-linear requires both ends same side of a layout condition (1224-1232); host staging unsupported for CSA-linear (1666) | #61121 (packed rows whole, -45% bytes on GB200) | none found on ROCm |
| MooncakeConnector | yes (factory.py:248-252) | Mooncake TE; ROCm needs multi-protocol build per InferenceX#4756 | handles CircularBuffer + DSv41 bounded replay (mooncake_connector.py:1100-1115, 2430-2445) | - | TileRT lane uses its own `DsV41TileRTConnector` over mooncake, not vLLM's MooncakeConnector |
| UMBPStoreConnector | yes | MORI UMBP host pool (offload, not PD) | supported (#64715), keeps AITER fast paths | - | verified DP2 lane |
| P2pNccl / LMCache | registered | - | not investigated for DSv4.1 | - | not found |

Cross-cutting constraints for any PD connector on ROCm (first-hand):
- AITER sparse-MLA and NaN-free extra-cache fast paths turn off (section 0 item 6).
- Mega-attn prefill (#65146, open) excludes KV connectors.
- PCP excludes `kv_transfer_config` (amd/model.py:197-216).
- Mono decode needs TP in {2,4,8}; DP rejects it (recipe, #64715). A D pool using `--monokernel` must be TP;
  monokernel + kv consumer combination: no evidence either way (not found). #68513's D command does use
  `--monokernel` at TP4 with MoRIIO consumer [official: vllm#68513].
- MoRIIO xGMI single-node: larger decode TP than prefill TP fails (docs/features/moriio_connector_usage.md:163).
  #68513 ran P TP2+DP2 -> D TP4 on xGMI (P per-rank TP=2 < D TP=4) [official]; I have not reconciled this with the doc note.
- MoRIIO PP only on prefill, WRITE, single node (moriio_connector_usage.md:169).

Router/proxy: `vllm-router --vllm-pd-disaggregation --kv-connector moriio --policy consistent_hash
--intra-node-data-parallel-size N` (#68513 command; vllm-router 0.1.14 used by InferenceX#5036 for DP2);
toy proxy `examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py`; docs recommend vllm-router
(moriio_connector_usage.md:81-99) [first-hand].

Feasibility verdict: **vLLM PD for DSv4.1 on MI355X is not runnable on stock main with MoRIIO** (first-hand code +
#68513 log). Paths: (a) cherry-pick #68513 (+#68621/#60783 as needed) for MoRIIO; (b) try NixlConnector on ROCm
(code path exists, no ROCm evidence); (c) Mooncake with a custom multi-protocol build. Whether PD beats the
DP2+UMBP aggregated baseline (~250-310K tok/s/GPU total) is unmeasured.

---

## 5. Docker images seen

| Image | Where | Notes |
|---|---|---|
| `vllm/vllm-openai-rocm:nightly-rocm100-5555a2d8f7116b2b4877f66516f4128176cf722a` | /perf_apps/chaox/run_DeepSeek-V4.1-Flash-vllm.sh:136 | colleague's dev container |
| `nightly-e68efc685ab1…` | recipe mi355x_agentx (InferenceX#4480) | TP2/TP4 lane |
| `nightly-335dd77b0916…` | recipe mi355x_agentx_tp8 (#4666) | TP8 c1 mono |
| `nightly-be47189f37f6…` | recipe mi355x_agentx_dp2_umbp | carries #64715, #66036 |
| `nightly-60bd6dd2edd1…` (2026-11-16) | InferenceX#5036/#5098 | #64715, #66036, #66644, #67377 |
| `nightly-rocm100-8ca766326652…` (2026-11-17) | InferenceX#5125 | #63893, #64883 |
| `nightly-rocm100-9c1d45c2…` (2026-11-19) | Docker Hub latest | - |
| `ghcr.io/tile-ai/tilert-rocm-decode:0.1.6` | InferenceX#4756 TileRT decode | non-vLLM decode |

---

## 6. Open PRs we may need to cherry-pick (vllm-project/vllm, state as of 2026-11-19)

PD / connector:
- #68513 [ROCm][MoRI-IO] Transfer circular buffers for DSv41 PD (blocker for MoRIIO + DSv4.1).
- #68621 [MoRIIO] Transfer hybrid draft sliding-window KV in READ and WRITE modes (draft, no accuracy yet).
- #60783 MoRIIO accept kernel-block-split MLA/indexer KV views.
- #68072 MoRIIO hetero-TP WRITE: replicated indexer caches.
- #65563 MoRIIO proxy routing for unequal P/D DP sizes.
- #68374 MoRIIO skip layers sharing another layer's KV (WRITE hang fix; relevant if DSpark draft shares KV - not verified for DSv4.1).
- #67074, #66008, #62190, #67216, #58962, #63855/#63856 (MoRIIO READ/handshake robustness).
- #61121 (+ base #61074) NIXL packed MLA rows whole for DSv4.1.
- #64785 MoRIIO transfer stats / Prometheus metrics (useful for bottleneck finding).

Perf:
- #65655 cross-DP prefill delayer (+8.4% DP2 c128 in its test).
- #65146 mega-attn prefill (excludes KV connector; would need work for P side).
- #68385 MegaMoE improvements for DEP8 (+ aiter #8186).
- #57239 ROCm DCP for DSv4 (WIP).
- #57756 fuse DSv4.1 decode metadata and final norm (ROCm).
- #54134 FHMoE DP8 over RCCL.

---

## 7. Search log (where I looked)

- `gh search prs --repo vllm-project/vllm` with "DeepSeek-V4", "DeepSeek-V4 ROCm", "DeepSeek V4.1 ROCm", "DSv41",
  "DeepSeek-V4.1-Flash" (merged), "MoRIIO", "MoRI-IO DeepSeek", "NixlConnector DeepSeek-V4", "disaggregated DeepSeek-V4",
  "Mooncake DeepSeek V4", "P/D DeepSeek-V4.1" (the last several returned nothing).
- `gh search prs --repo SemiAnalysisAI/InferenceX` "dsv41", "DeepSeek-V4.1-Flash MI355X vLLM", "dsv41flash-fp4-mi355x-vllm".
- vllm main shallow clone (`source/vllm`, 7a62ca0666): recipes/, vllm/models/deepseek_v4*, kv_connector/v1/{moriio,nixl,mooncake,umbp}, docker/.
- gh release list; Docker Hub tags API for vllm/vllm-openai-rocm.
- WebSearch rocm.blogs.amd.com: no ROCm blog on DSv4.1-Flash vLLM serving found (only a Primus V4-Flash training blog).
- Colleague dirs /perf_apps/yanfwang/dsv41_*, /perf_apps/jihhe/dsv41di, /perf_apps/yangwu/dsv4-3045-clean3p3d: no vLLM-on-AMD PD config found (only GB200 vLLM disagg yamls copied from InferenceX).
- Not searched: vllm-project/recipes repo directly (used the in-tree `recipes/` copy), ROCm/vllm fork.

---

## Verification (adversarial, 2026-11-19)

Independent re-check of 12 claims against source (vllm main clone `source/vllm` @ 7a62ca0666, InferenceX clone `source/InferenceX` @ db3fc79, and `gh pr view`).
Confirmed (no change): MoRIIO CircularBuffer rejection (moriio_connector.py:650-681, compressor.py:259-270); NIXL DSv4.1 handling + ROCm NIXL/UCX wheel stage; InferenceX#5036 numbers (AL 3.51 is from the PR's golden-AL check comment, not the body); InferenceX#4992 numbers; InferenceX#5056 182.7K vs vllm#67377 26,714 (both quoted correctly); monokernel constraints and vllm#64101 numbers (DP rejection is at mono_decode.py:316-317 via `_unsupported_moe`); Mooncake comment (actual path `inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/dsv41flash/tilert/mi355x-fp4/agentx/disagg-1p1d-tp4-tp8-mtp.yaml:19-20`); vllm#66036 / #64715 numbers.

| # | Claim (short) | Verdict | Correct fact + source |
|---|---|---|---|
| 2 | PR #68513 tested P = TP2+DP2 -> D = TP4 | corrected | PR text says "TP2+DPA P", but the posted P command uses `HIP_VISIBLE_DEVICES=0,1 --data-parallel-size 2` with no `--tensor-parallel-size`, i.e. 2 GPUs, TP1 x DP2 (attention DP2); D = TP4 with `--monokernel`. PR is a **draft** (isDraft=true, created 2026-11-18T21:14Z). Other details (MI325, xGMI, read_mode, both error messages, accuracy "TODO") confirmed. [official: vllm#68513] |
| 3 | UMBP overrides `only_restores_canonical_kv` and keeps the fast paths | corrected (conditional) | UMBP returns True only when `kv_connector_extra_config.key_namespace` is `"auto"` (default); an explicit key_namespace returns False and the fast paths turn off. [first-hand: kv_connector/v1/umbp/connector.py:110-117]. Also: `extra_cache_nan_free` additionally requires gfx950 and `kv_cache_dtype == "fp8_ds_mla"` [first-hand: deepseek_v41/amd/rocm.py:80-91], so with `fp8_e4m3` KV it is off regardless of connector. |
| 4 | Verified vLLM PD lanes are only GB200 1P1D and GB300 Dynamo | corrected (incomplete) | The vllm recipe yaml lists only those two (recipe yaml:384-399, 1623-1640) - confirmed. But InferenceX main also carries a **B300** Dynamo+vLLM disagg recipe `inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/dsv41flash/vllm/b300-fp4/agentx/disagg-variants.yaml` (name `dsv41flash-fp4-b300-dynamo-vllm-agentx-disagg`). Still NVIDIA only; no MI355X vLLM-to-vLLM PD recipe found in InferenceX `dsv41flash/` (only `tilert/mi355x-fp4`). InferenceX#4756: merged 2026-11-07, c1, "Do not start a benchmark sweep" - confirmed. |
| 10 | "DCP is not on main" for DSv4.1 | unverifiable | PCP gate (amd/model.py:197-216) and PR states (#57239 open, self-described Draft/WIP; #44573 open, labels stale/needs-rebase) confirmed. But main has generic DCP plumbing reaching the DSv4 path: `v1/attention/backends/mla/compressor_utils.py:143` passes `dcp_world_size`, `v1/attention/backends/mla/indexer.py:566,996` read DCP size; some NVIDIA DSv4.1 indexers explicitly reject CP (nvidia/ops/candidate_indexer.py:194, nvidia/specialized_decode/ampere/indexer.py:640). Whether DCP>1 runs end to end for DSv4.1 on ROCm main was not determined (no ROCm-specific DCP rejection found outside the PCP check). |
| 12 | fp8_e4m3 KV gives +12.5% capacity | confirmed (note) | PR text says "12.5% more KV-cache capacity ... 512 instead of 584 bytes per token" [vllm#66036]; arithmetic 584/512 = 1.141 (+14.1% tokens), 512/584 = 0.877 (-12.3% bytes). The PR's 12.5% figure is quoted faithfully. +7.6% is TP2, DSpark AL 3.51, `vllm bench serve` random 8K/1K c64 (2111.9 -> 2272.0) - confirmed. |
