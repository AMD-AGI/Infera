# DeepSeek-V4.1-Flash: official model structure, weight/KV inventory, PD implications

Date: 2026-11-19. Phase 1 research only (no GPU runs).
Evidence tags: `[FH: file:line]` first-hand code/file read; `[FH-OFF: URL/PR]` official source;
`[SH]` second-hand; `[INF]` my own arithmetic or inference from first-hand facts (formula shown).

Abbreviations used below:
- `M` = `/perf_apps/chaox/models/DeepSeek-V4.1-Flash` (local checkpoint)
- `MP` = `M/inference/model.py`; `CFG` = `M/config.json`; `ICFG` = `M/inference/config.json`
- `TR` = `M/DeepSeek_V41_Tech_Report.pdf` (text extracted to `source/techreport/report.txt`; cited as TR p.N = PDF page)
- `V` = `source/vllm` (vLLM main clone @ `7a62ca0666e9`, 2026-11-19)
- `RCP` = `V/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml` (official vLLM recipe)
- Helper scripts and outputs: `scripts/model_arch/{st_inventory.py,calc.py,kv_calc.py,tensors.json}`

---

## 0. Top facts a decision rests on

1. **Global (growing) KV is tiny: 890 B/token in the official format** (FP4 main KV + MXFP4 indexer K, shared across layers). README states 890 B/token [FH: M/README.md "Compressed Sparse Attention 2" paragraph]; my recomputation from the config gives exactly 2.5 entries/token x (288 + 68) B = 890 B [INF from CFG compress_ratios / kv_source_layer_ids, MP:760-762, MP:546].
2. **vLLM on ROCm does NOT use the FP4 main-KV record.** `nvfp4_ds_mla` (288 B) is SM100-only; on ROCm the default resolves to the V4 `fp8_ds_mla` record = 584 B/entry, or `fp8_e4m3` plain rows = 512 B/entry [FH: V/vllm/models/deepseek_v41/attention.py:197-210, 612-624; amd/rocm.py:538-545, 638-639]. => vLLM-ROCm global KV = **1630 B/token (fp8_ds_mla) or 1450 B/token (fp8_e4m3)**, 1.6-1.8x the official 890 B [INF].
3. **SWA KV is bounded**: 128-token ring on every layer (40 backbone + 3 DSpark) ~= 2.8-3.2 MB/request regardless of context [INF; MP:664-669; vLLM uses SlidingWindowManager, V/vllm/v1/attention/backends/mla/sparse_swa.py:119-145].
4. **Causal Encoder-Decoder (CED)**: decoder (layers 20-39) global KV is projected from the encoder output; prefill only runs the 20 encoder layers on all prompt tokens plus the decoder on the last 128 tokens ("Decoder SWA Bounded Replay"). Active params: **~8B/token prefill, ~16B/token decode** [FH-OFF: TR p.7, p.9, p.20; my shape count gives 7.9B / 16.1B, INF]. vLLM implements decoder replay on AMD (`swa_bounded_replay` default True, CDNA3/4 only, model runner V2 only) [FH: V/vllm/config/cache.py:381-386; V/vllm/models/deepseek_v41/amd/model.py:713-745, 1119-1158; attention.py:626-651].
5. **Engram tables = 202.8 GB of the 510.3 GB checkpoint** (2 tables x ~384M rows x 256 B FP8 + E8M0 scales), pure lookup (12.7 KB read per token). vLLM default `cpu_offload: true` (pinned host memory via UVA) [FH: tensors.json via st_inventory.py; V/docs/features/engram.md "CPU offload"].
6. **Weights without Engram/vision: 307 GB (285.5 GiB)** -> TP1 cannot fit on one 288 GB MI355X; TP2 resident ~145 GiB/rank, TP4 ~81 GiB/rank, TP8 ~36 GiB/rank with Engram offloaded [INF + SH: RCP comments lines ~130-133, ~253-268].
7. **MLA-style KV is replicated across TP ranks** (`num_kv_heads=1, max_tp_shards=1`) [FH: attention.py:1306-1311, 1378-1385]. Per-GPU KV capacity therefore does not grow with TP; DP-attention multiplies it.
8. PD transfer per request (global KV + SWA) at ISL 50k/100k/200k: **official format 47/92/181 MB; vLLM-ROCm fp8_ds_mla 85/166/329 MB; fp8_e4m3 75/148/293 MB** [INF, kv_calc.py].
9. Decode per-token compute: GEMM ~32.3 GFLOP (2 x 16.1B active) + core sparse attention ~3.2 GFLOP + indexer 1.6/2.6/4.6 GFLOP at 50k/100k/200k context [INF, kv_calc.py].

---

## 1. Layer-by-layer structure

### 1.1 Global shapes [FH: CFG text_config; ICFG]
- 40 backbone layers, hidden 5120, vocab 129280, 64 query heads, 1 KV head (MQA latent), head_dim 512 (448 NoPE + 64 RoPE), q_lora_rank 1280, o_lora_rank 1024, o_groups 8, sliding_window 128, rope_theta 10000 (SWA-only layers) / compress_rope_theta 160000 + YaRN factor 16 over 65536 for compressed layers [FH: MP:681-697], max_position_embeddings 1048576.
- MoE every layer: 384 routed experts top-6 + 1 shared, moe_intermediate 2304, scoring `sqrtsoftplus`, `noaux_tc` bias for selection only, norm_topk_prob, routed_scaling 1.5, SwiGLU clamp 10 [FH: CFG; MP:794-829 Gate; MP:832-853 Expert; MP:856-906 MoE].
- Hyper-connections (mHC): hc_mult 4, Sinkhorn 20 iters, eps 1e-6 [FH: CFG; MP:909-996]. Residual stream is 4 x 5120 per token; `hc_*_fn` is [24, 20480] fp32 per sublayer (mix_hc = (2+4)*4 = 24) [FH: MP:940-948; tensors.json `layers.3.hc_attn_fn` F32 [24,20480]]. "Single-Pass mHC": each sublayer uses the mixing coefficients produced by the previous sublayer [FH: MP:978-996; FH-OFF: TR p.12-13].
- rms_norm_eps 1e-20 [FH: CFG].

### 1.2 Attention: SWA + CSA2 (Compressed Sparse Attention 2)
Every layer computes its own Q and its own 128-token SWA KV; layers with `compress_ratio>0` additionally attend to `index_topk=512` entries of a compressed ("main"/global) KV [FH: MP:614-621, 766-791; FH-OFF: TR p.10-11].

`compress_ratios` (43 entries = 40 backbone + 3 DSpark) [FH: CFG]:
- `0` = SWA only (no global KV) - layers 0, 1 and DSpark layers 40-42.
- `2` = main KV pools 2 consecutive tokens into one latent with a learned softmax gate (fp32 pooling) - layers 2-19.
- `1` = one latent per token (no pooling; plain projection) - layers 20-39 [FH: MP:429-485 Compressor; MP:444-462].
- CSA2 vs V4 CSA: no overlapping 2m-window, no absolute position embedding, indexer K projected from main KV [FH-OFF: TR p.10].

Sharing (`kv_source_layer_ids=[2,8,14,20]`, `index_source_layer_ids=[2,8,14,20,24,28,32,36]`) [FH: CFG; MP:82-84, 498-525, 654-663, 723-764, 1169-1183]:
- **Full mode** (kv source + index source): computes its own main KV (Compressor), projects indexer K from it (`wk` 512->128, MXFP4), runs indexer, publishes main KV / index K / top-k idx. Layers 2, 8, 14, 20.
- **Reindex mode** (index source only): reuses main KV + index K of the latest kv-source; runs its own indexer Q and fresh top-512. Layers 24, 28, 32, 36.
- **Reuse mode**: reuses main KV and the latest top-k indices; no indexer. All other CSA2 layers.
- Hierarchical Sparse Indexer: layer 20 (`candidate_source_layer_id`) scores all positions, picks top `candidate_topk_blocks=2048` blocks of `candidate_block_size=8` (16384 candidate positions); Reindex layers 24-36 score only inside this pool [FH: MP:98-101, 570-576, 584-611; FH-OFF: TR p.11-12].
- Indexer: 32 heads x 128 dim, FP4 (MXFP4) Q and K, ReLU scores weighted by `weights_proj`, top-512 [FH: MP:488-581; CFG].
- Core attention: `sparse_attn(q, kv=concat(SWA window, compressed KV), attn_sink, topk_idxs)`; attention sink per head [FH: MP:640, 775-781; M/inference/kernel.py:392-404].
- Output projection: grouped low-rank `wo_a` (8 groups, 4096 -> 8 x 1024, block-diagonal einsum) then `wo_b` 8192 -> 5120 [FH: MP:646-651, 784-791].
- Q path: `wq_a` 5120->1280, RMSNorm, `wq_b` 1280 -> 64x512 [FH: MP:641-643; tensors.json].
- KV path: `wkv` 5120 -> 512, RMSNorm, RoPE on last 64 dims; SWA KV kept FP8 (block 32, UE8M0) [FH: MP:701-709]; compressed KV after RoPE stored FP4 E2M1 with one E4M3 scale per 16 channels [FH: MP:760-762; FH-OFF: TR p.14 "FP4 Main KV Cache"].

### 1.3 Causal Encoder-Decoder (CED) [FH-OFF: TR p.7-9, p.19-20]
- Encoder = layers 0-19, decoder = layers 20-39. Decoder global KV (layer 20's ratio-1 KV, reused by 21-39) is projected from the encoder's final hidden state H_20. In MP this is literally layer 20's Compressor applied to the stream entering layer 20 [FH: MP:740-749, 984-987].
- Decoder SWA KV still comes from each decoder layer's own hidden state -> **Decoder SWA Bounded Replay**: during prefill, run decoder layers only over the last n_win=128 prompt tokens (approximate SWA state, trained-for in post-training) [FH-OFF: TR p.20].
- **Encoder SWA Bounded Replay**: on a prefix hit with missing SWA KV, replay only the last 128 tokens of the cached prefix [FH-OFF: TR p.20]. DeepSeek's own deployment keeps SWA KV out of the persistent KV cache (stored in a short-TTL DRAM pool) [FH-OFF: TR p.19].
- Note: the reference `MP` does NOT implement the prefill skip; it runs every layer on every token [FH: MP:1244-1275; generate.py has no replay logic (grep)].
- vLLM: `cache_config.swa_bounded_replay=True` by default; "Keep the sliding-window KV ... out of prefix caching and rebuild it after a prefix hit by recomputing the hit's last window. The layers past the last KV-source layer then also run eager prefill steps on each request's trailing window only. Requires model runner V2." [FH: V/vllm/config/cache.py:381-386]. AMD model gates: off if PP splits the decoder, sequence parallel, PCP>1, ubatching, Engram at/after layer 20, or drafter window > 128 [FH: V/.../amd/model.py:1119-1158]; ROCm only on CDNA 3/4 [FH: V/.../attention.py:640-651]. Log text says replay runs "in eager prefill steps" [FH: amd/model.py:744-745].

### 1.4 Per-layer table [FH: CFG lists; FH-OFF: TR p.7 Fig.3 and p.21-22 "Model Setups" agree]

| Layer(s) | ratio | CSA2 mode | main KV from | top-k from | Extra |
|---|---|---|---|---|---|
| 0 | 0 | SWA only | - | - | |
| 1 | 0 | SWA only | - | - | Engram #0 |
| 2 | 2 | Full | own (ratio-2) | own indexer (full scan of N/2) | |
| 3-7 | 2 | Reuse | L2 | L2 | |
| 8 | 2 | Full | own | own | |
| 9-13 | 2 | Reuse | L8 | L8 | |
| 14 | 2 | Full | own | own | Engram #1 |
| 15-19 | 2 | Reuse | L14 | L14 | end of encoder |
| 20 | 1 | Full + candidate source | own (ratio-1, from H_20) | own (full scan of N) + 2048x8 candidate pool | start of decoder |
| 21-23 | 1 | Reuse | L20 | L20 | |
| 24 / 28 / 32 / 36 | 1 | Reindex | L20 | own, inside candidate pool | |
| 25-27, 29-31, 33-35, 37-39 | 1 | Reuse | L20 | latest Reindex layer | 37, 38, 39 = DSpark target layers |
| 40-42 (mtp.0-2) | 0 | SWA only (DSpark) | - | - | 128 experts top-3 MoE |

### 1.5 Engram (layers 1 and 14) [FH: MP:296-365; M/inference/engram.py:86-184; CFG; FH-OFF: TR p.13, p.18]
- Per position: n-grams of order 2,3,4 (max_ngram_size 4) over a compressed tokenizer vocab (99092 ids), 8 hash heads per order -> 24 table rows per layer, each row 256 dims FP8 + 8 E8M0 scales [FH: engram.py:129-184; MP:343-345].
- Table rows: 384,006,168 (layer 1) and 384,016,682 (layer 14) x 256 B = 98.3 GB + 3.07 GB scales each; **202.8 GB total** [FH: tensors.json; files model-00047/48 are 101.5 GB each, `ls -la M`].
- `wkv`: 6144 -> 25600 (= 5120 x (4 hc + 1)) FP8, 157M params per layer; gated (signed-sqrt sigmoid) add into each hc copy [FH: MP:345, 350-365].
- Lookup indices depend only on token ids (hash state = token history; only last 3 ids are needed) -> prefetchable from host memory; TR states DeepSeek prefetches from host via RDMA [FH-OFF: TR p.13]; vLLM serves lookups from pinned host memory via UVA by default [FH: V/docs/features/engram.md].
- Bytes read per token: 2 layers x 24 rows x 264 B = 12.7 KB [INF]. At prefill of 100k uncached tokens that is ~1.27 GB read from the table location per request [INF].
- Placement facts on MI355X in vLLM recipe: TP2 offloads to host, TP4 can keep resident (47.2 GiB/rank at TP4, 94.4 GiB/rank at TP2) [FH: RCP ~lines 253-268 comment; SH as to the GiB numbers, which are recipe comments, not my measurement].

### 1.6 MTP / DSpark [FH: MP:1022-1159, 1210-1216, 1264-1286; CFG; FH-OFF: TR p.13-14]
- 3 DSpark blocks (`mtp.0-2`), each a full Block (attention SWA-only window 128, own MoE 128 routed experts top-3 + 1 shared, mHC) [FH: MP:142-149, 1034-1078, 1103-1129].
- Input: mean over hc copies of the attention input of target layers 37, 38, 39, concatenated (15360) -> `main_proj` 15360->5120 + norm [FH: MP:1116-1117, 1267-1270, 1131-1138]. Embedding and LM head are shared with the backbone [FH: MP:1215-1216; M/inference/convert.py:109].
- Draft: one forward computes logits for 5 positions (`dspark_block_size=5`, noise-token padded), Markov head (rank 256) adds token-to-token bias sequentially, confidence head predicts per-position acceptance [FH: MP:1080-1101, 1140-1159].
- Draft KV: each DSpark layer keeps its own 128-slot SWA ring fed from main_x (bounded) [FH: MP:1046-1068].
- vLLM recipe: `--speculative-config {"method":"dspark","num_speculative_tokens":5,...}`; adaptive verification refused on ROCm, so AMD sets `enable_adaptive_verification:false` [FH: RCP `spec_decoding` feature]. InferenceX throughput runs substitute a synthetic acceptance length 3.51 [FH: RCP mi355x_agentx runbook] (this is a benchmark convention, not a measured acceptance).

### 1.7 Vision tower (ignore for text serving)
- 32-layer ViT (dim 1024, 16 heads, patch 14, 3x3 pixel-unshuffle) + 2-layer MLP aligner [FH: CFG vision_config; FH-OFF: TR p.8]. Weights 0.97 GB BF16 [FH: tensors.json]. vLLM `--language-model-only` skips loading it [FH: RCP `text_only` feature].

### 1.8 Reference implementations
- Official reference: `M/inference/model.py` (1312 lines) + `kernel.py` (tilelang kernels) + `engram.py` + `convert.py`.
- HF transformers: **not on main**; `deepseek_v4` exists on main but `deepseek_v41` is only in **open PR huggingface/transformers#50919** "Add DeepSeek-V4.1-Flash" (opened 2026-11-18, head f377a139) [FH-OFF: gh pr view 50919]; earlier #48721 (text) and #48768 (vision) closed unmerged. The PR modeling file (1784 lines) is saved at `source/transformers_pr50919/modeling_deepseek_v41.py`; its config names layer types `sliding_attention / compressed_attention / kv_shared_indexed_attention / kv_shared_attention` [FH: source/transformers_pr50919/configuration_deepseek_v41.py:80-98, 360-374].
- vLLM: `V/vllm/models/deepseek_v41/` (attention.py 1740, compressor.py 463, decoder_replay_layers.py 146, sparse_mla.py 337; `amd/` model.py 1606, rocm.py 1574, mono_decode.py 849, engram.py, mhc.py, dspark.py) [FH: wc -l]. Recipe `min_vllm_version: 0.31.0`, architecture shipped in 0.30.0 (vllm-project/vllm#56228); AMD stays on `vllm/vllm-openai-rocm:nightly-rocm100` because monokernel (#60397, #61082), paged MXFP4 indexer (#58671), Gluon sparse MLA (#53492) are on main but not in 0.31.0 [FH: RCP lines 33-49].
- SGLang: served through the `dsv4` backend, `python/sglang/srt/models/deepseek_v4.py` + `layers/attention/dsv4/dsv41_sparse.py`; cookbook `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` lists MI355X recipes, PD with Mooncake/NIXL (DSpark flags required on both roles), HiCache "not validated on ROCm for this model yet" [FH-OFF: sglang main a735253e, saved at `source/sglang_snippets/DeepSeek-V4_1.mdx` lines 100-110, 213-231]. SGLang KV storage byte format for V4.1 on ROCm: not determined (dsv41_sparse.py uses fake-quant FP4 helpers; mem-pool record size not located).

---

## 2. Weight inventory (from safetensors headers, all 48 shards)

Method: read every shard's JSON header (no tensor data) and sum `data_offsets` [FH: scripts/model_arch/st_inventory.py]. Total 510.286 GB = index `metadata.total_size` exactly [FH: model.safetensors.index.json]. Local checkpoint: HF revision `5254448103b6...` downloaded 2026-10-14 (launch repo; RCP says the 1017 refresh has identical config and index layout) [FH: M/.cache/huggingface/trees; RCP lines 113-122].

| Component | GB | dtype breakdown |
|---|---|---|
| Routed experts, backbone (40 x 384) | 288.78 | I8 (= packed FP4 E2M1x2) 271.79 + E8M0 scales 16.99 |
| Engram tables (L1, L14) | 202.76 | F8_E4M3 196.61 + E8M0 6.14 |
| Attention core (wq_a/wq_b/wkv/wo_a/wo_b, 40 layers) | 5.07 | FP8 E4M3 5.065 + E8M0 0.005 |
| Shared experts (40) | 1.42 | FP8 |
| Embedding / LM head | 1.32 / 1.32 | BF16 |
| Vision + aligner | 0.97 | BF16 |
| Engram wkv etc. | 0.32 | FP8 |
| Router (gate weight + bias + bias_vl) | 0.16 | BF16 / F32 |
| mHC fn/base/scale | 0.16 | F32 |
| Indexer / compressor | 0.045 / 0.037 | FP8+BF16 / BF16 |
| DSpark (mtp.0-2) total | 7.93 | routed 7.22 (FP4), attn 0.38, heads 0.21, shared 0.11, mHC 0.01 |

Formats [FH: MP:26-30, 186-205, 219-233; CFG quantization_config]: dense linears FP8 E4M3 with one UE8M0 scale per 32x32 block; experts FP4 E2M1 with one UE8M0 scale per 32 along K (e.g. `w1.scale` [2304,160]); compressor wkv/wgate BF16 (ratio-2 promoted to fp32 at load) [FH: MP:444-448]; `wo_a` is dequantized to BF16 by convert.py [FH: MP:784-786].

Parameter counts [FH: scripts/model_arch/calc.py over tensors.json]: total 763.2B = backbone ~551.9B (matches "552B backbone") + Engram 196.6B (matches "196B Engram" [FH-OFF: TR p.7]) + DSpark 14.2B + vision 0.49B. One routed expert = 35.39M params.

Active params per token [INF from shapes]: per backbone layer = attention 126.6M + shared expert & router 37.4M + 6 routed x 35.39M = 212.3M + mHC ~1M = ~377M.
- Decode: 40 x 377M + Engram wkv 0.31B + indexers/compressors 0.06B + LM head 0.66B = **~16.1B** (official "16B").
- Prefill (CED, N >> 128): 20 encoder layers 7.55B + Engram wkv 0.31B + layer-20 compressor/index-K projections ~0.01B = **~7.9B** (official "8B"); decoder layers only for the last 128 tokens.
- DSpark draft forward: 3 x (126.6M attn + 3 x 35.39M routed + ~36M shared) + main_proj 79M ~= 0.89B per draft position, plus shared LM head for 5 positions [INF].

---

## 3. Per-token KV / cache cells

### 3.1 Official format (reference code + tech report)
| Cache | Owner layers | Entries/token | Bytes/entry | Bytes/token | Grows with context? |
|---|---|---|---|---|---|
| Main (global) KV, ratio 2 | 2, 8, 14 | 3 x 1/2 | 256 FP4 + 32 E4M3 = 288 | 432 | yes |
| Main (global) KV, ratio 1 | 20 | 1 | 288 | 288 | yes |
| Indexer K (MXFP4, 128 dims) | 2, 8, 14, 20 | 2.5 | 64 + 4 = 68 | 170 | yes |
| **Global total** | | | | **890** | yes |
| SWA KV (FP8 + UE8M0 /32) | every layer 0-39 (+3 DSpark) | ring of 128 | 512 + 16 = 528 | bounded: 128 x 43 x 528 = 2.9 MB/request | no |
| Compressor open-group state | 2, 8, 14 (ratio 2) | ratio x 512 fp32 kv + score | - | bounded: 3 x 2 x 2 x 512 x 4 = 24.6 KB/request | no |
| Engram token history | - | token ids (last 3 needed) | - | negligible | no |

Sources: [FH: MP:452-456, 520-525, 546, 664-680, 760-762; engram.py:155-157; FH-OFF: TR p.14 (FP4 main KV, E4M3 scale per 16 channels; SWA kept FP8), README "890 bytes per token"]. The 890 B matches my computation exactly [INF].

For a 100k-token context: global 89.0 MB + SWA 2.9 MB = **~92 MB per request** [INF].

### 3.2 vLLM main on ROCm (what we will actually run)
- Compressed-KV cache exists only on kv-source layers; consumers read the source's cache (cross-layer sharing implemented) [FH: V/.../attention.py:1295-1301, 669-700].
- Record per compressed entry on ROCm: `fp8_ds_mla` V4 record = 448 fp8 NoPE + 64 bf16 RoPE + 7 UE8M0 scales + pad = **584 B**; `fp8_e4m3` per-tensor plain row = **512 B**; `nvfp4_ds_mla` (288 B) raises unless SM100 [FH: attention.py:197-210, 612-624; amd/rocm.py:538-545, 638-639; RCP umbp preset uses `--kv-cache-dtype fp8_e4m3` via vllm-project/vllm#66036].
- Indexer K: MXFP4 = 68 B/entry when `indexer_kv_dtype=mxfp4` (ROCm paged MXFP4 indexer, vllm-project/vllm#58671, required on ROCm per error text), else FP8 132 B [FH: attention.py:185-193, 1546-1552; RCP mi355x override].
- SWA cache: `MLAAttentionSpec` under `SlidingWindowManager`, block_size 32, record 584 B on ROCm [FH: sparse_swa.py:76-145; attention.py:654-668].
- Compressor state: `CompressorStateCache`, fp32 ring per request, `CircularBufferManager` (bounded) [FH: compressor.py:237-272].
- KV is **not sharded by TP** (`num_kv_heads=1, max_tp_shards=1`) [FH: attention.py:1306-1311, 1378-1385].

| Format (ROCm) | Global B/token | +SWA (bounded) | 100k context total |
|---|---|---|---|
| fp8_ds_mla + MXFP4 indexer (recipe TP default) | 2.5 x (584+68) = 1630 | 128 x 43 x 584 = 3.2 MB | 166 MB |
| fp8_e4m3 + MXFP4 indexer (recipe DP2 UMBP preset) | 2.5 x (512+68) = 1450 | 2.8 MB | 148 MB |
| fp8_ds_mla + FP8 indexer | 1790 | 3.2 MB | 182 MB |
[INF; scripts/model_arch/kv_calc.py]. Page alignment padding (512 B alignments, block 128) not included -> small extra, not quantified.

Which part grows: only compressed main KV + indexer K (2.5 entries/token). Bounded: SWA rings (43 x 128 slots), compressor state, DSpark SWA, Engram history.

---

## 4. Implications for PD disaggregation

### 4.1 Bytes transferred per request (prefill -> decode), global KV for whole prompt + SWA rings [INF, kv_calc.py]
| ISL | Official 890 B/tok | vLLM-ROCm fp8_ds_mla (1630) | vLLM-ROCm fp8_e4m3 (1450) |
|---|---|---|---|
| 50k | 47.4 MB | 84.7 MB | 75.3 MB |
| 100k | 91.9 MB | 166.2 MB | 147.8 MB |
| 200k | 180.9 MB | 329.2 MB | 292.8 MB |
Notes:
- Transfer time = bytes / effective link bandwidth; e.g. at an assumed 50 GB/s effective, 166 MB = 3.3 ms [INF, bandwidth is an assumption, not measured].
- With prefix caching in multi-turn AgentX, only the uncached suffix's KV is new, but whether the vLLM connector ships only new blocks for this model's 4 cache groups (SWA, compressed, indexer, compressor state) is **not verified** (gap).
- Under TP on the decode side every rank holds a full KV copy (replicated, fact 7), so the connector either sends N copies or one copy + local broadcast - connector behaviour not verified (gap).
- SWA rings: either transferred (bounded ~3 MB) or rebuilt on decode by bounded replay of the last 128 tokens; which path vLLM PD takes is not verified (gap).

### 4.2 Decode KV capacity per GPU (upper bound) [INF, kv_calc.py]
Budget = 0.9 x 288 GiB (MI355X 288 GiB per RCP comment; AMD recipe default `--gpu-memory-utilization 0.9` [FH: RCP hardware_overrides.amd]) minus resident weights with Engram offloaded (TP2 ~145, TP4 ~81, TP8 ~36 GiB/rank [SH: RCP comments for MI325X/H100 arms; same checkpoint]). Ignores activations, CUDA-graph pools and the indexer logits workspace (RCP notes a `[max-num-batched-tokens, max-model-len]` x 2 B buffer, 16 GiB at 8192 x 1M on the H100 path [FH: RCP ~lines 133-138]) -> real numbers will be lower.

| Shape | KV budget/rank | tokens (fp8_ds_mla 1630 B) | tokens (fp8_e4m3 1450 B) | tokens (official 890 B, not available on ROCm vLLM) |
|---|---|---|---|---|
| TP1 | weights 285.5 GiB > 259.2 GiB budget: does not fit (even with Engram offloaded) | - | - | - |
| TP2 | 114 GiB | 75.2M (= 752 x 100k) per TP group | 84.6M | 137.8M |
| TP4 | 178 GiB | 117.4M per TP group | 132.0M | 215.0M |
| TP8 | 223 GiB | 147.0M per TP group | 165.3M | 269.3M |
Because KV is replicated across TP ranks, "per TP group" = per rank; DP-attention gives each rank its own pool.

Observed context: the vLLM recipe's MI355X DP2 UMBP preset reports that at concurrency 256/384 the AgentX working set (prefix cache of all live sessions) outgrew GPU KV and host-DRAM offload (UMBP) raised throughput/GPU 10.1% at c256 and 2x at c384 (local agentx-fast replays, earlier nightly, `fp8_ds_mla`) [FH: RCP lines 1908-1925; numbers are the recipe authors' measurements, SH for us]. So for AgentX the binding capacity is prefix-cache retention across sessions, not running-batch KV.

### 4.3 Compute per decode token [INF, kv_calc.py]
- GEMM: 2 x 16.1B = **32.3 GFLOP/token** (FP8 dense + FP4 experts).
- Core sparse attention: 38 CSA2 layers x 64 heads x (128 SWA + 512 top-k) x 512 x 4 + 2 SWA-only layers = **3.2 GFLOP/token**, independent of context once N >= 1024.
- Indexer: layers 2/8/14 full-scan N/2 entries, layer 20 full-scan N, layers 24-36 scan min(16384, N): **1.56 / 2.58 / 4.63 GFLOP** at N = 50k / 100k / 200k; index-K bytes read 13 / 21.5 / 38.5 MB per token per sequence. The encoder Full layers (2, 8, 14) are not restricted by the candidate pool and remain linear in N [FH-OFF: TR p.12 "The first Full Mode layer still scans the entire causally visible range"; FH: MP:564-581].
- DSpark adds ~0.89B active per draft position (5 positions per draft forward) [INF].
- Prefill per new token: ~2 x 7.9B = 15.8 GFLOP GEMM + encoder sparse attention ~1.5 GFLOP + encoder indexers (avg over causal positions) ~0.6 GFLOP at N=100k; decoder layers run on only 128 tokens/request [INF; FH-OFF: TR p.9].

### 4.4 Structural facts relevant to choosing PD vs colocated (observations only, no conclusion)
- Prefill:decode per-token GEMM work ratio is ~1:2 (8B vs 16B active) by design, and TR says CED "nearly halves prefill computation" [FH-OFF: TR p.8-9].
- Engram lookups on the prefill side are proportional to new tokens (12.7 KB/token from wherever the tables live) [INF].
- PP cannot split inside a kv-sharing or index-sharing group in vLLM ("PP splits inside a v4.1 kv-sharing group are not supported") [FH: attention.py:691-700; amd/rocm.py:614-620].
- ROCm sparse-SWA backend reports UNIFORM_BATCH only; torch.compile unsupported for DeepseekV41ForCausalLM; breakable CUDA graphs required [FH: RCP hardware_overrides.amd comments].
- vLLM PD for this model is recipe-verified only on NVIDIA (GB200 1P1D NIXL; GB300 Dynamo disagg, InferenceX #5015) [FH: RCP pd_cluster and gb300 presets]; no MI355X PD preset exists in RCP (searched `disagg|nixl|mooncake|pd_` in RCP).

---

## 5. Gaps (not found / not verified)
- vLLM KV-connector (NIXL/Mooncake/MoRI on ROCm) handling of the V4.1 cache groups (SWA ring, compressed KV, indexer K, compressor state): not read yet; no MI355X PD preset in RCP.
- Whether vLLM PD transfers SWA rings or relies on bounded replay on the decode side.
- Exact per-rank resident weights on MI355X at each TP/DP shape (numbers above are recipe comments for other SKUs with the same checkpoint); real KV pool sizes need a startup log.
- SGLang's on-device V4.1 KV record size on ROCm (mem pool not located).
- HF transformers modeling only in open PR #50919; not reviewed beyond config layer types.
- Tech report figure pages (KV-cache size chart, kernel counts) only text-extracted; figures not viewed (no PDF renderer on host).

---

## Verification (adversarial, 2026-11-19)

Independent re-check of the section 0 claims against the cited sources (vLLM clone `source/vllm` @ `7a62ca0666e9`, checkpoint `M`). Only refuted/corrected items are detailed; all other claims were confirmed.

Confirmed without change: 890 B/token official global KV (2.5 entries x (288+68) B; CFG compress_ratios/kv_source_layer_ids, MP:28 fp4_block_size=32, MP:546, MP:760-762, TR p.1 abstract); CED 8B/16B active (TR p.1, p.4, p.7, p.9; calc.py rerun gives 35.39M/expert, 763.2B total); checkpoint inventory (tensors.json rerun: total 510.286 GB, Engram tables 202.758 GB, routed 288.778 GB, attention 5.070 GB, ex-Engram/vision 285.50 GiB); Engram `cpu_offload: bool = True` (V/vllm/config/engram.py:44, V/docs/features/engram.md:92-97) and 2 x 24 x 264 B = 12.7 KB/token (CFG engram_n_heads 8, max_ngram 4); SWA bounded (SlidingWindowManager, V/vllm/v1/attention/backends/mla/sparse_swa.py:146-147); recipe UMBP numbers (+10.1% c256, 2x c384, -4.7% c128; RCP:1914-1922); no MI355X PD preset (RCP presets: mi355x_agentx, mi355x_agentx_tp8, mi355x_agentx_dp2_umbp; only pd_cluster strategy is gb300_agentx_pd, RCP:1626) and AMD nightly reason (RCP:44-49).

| # | Claim (short) | Verdict | Correct fact + source |
|---|---|---|---|
| V1 | vLLM-ROCm global KV = 1630 or 1450 B/token | corrected | The 584 B (fp8_ds_mla) / 512 B (fp8_e4m3) main-KV records are right (attention.py:612-624, 1302-1321). But 1630/1450 assume the MXFP4 indexer (68 B), which is NOT the engine default: `indexer_kv_dtype` defaults to "auto" (V/vllm/config/attention.py:102) and `dsa_indexer_uses_fp4` resolves auto -> "fp8" (V/vllm/v1/attention/backends/mla/indexer.py:67-76), i.e. 132 B/entry (attention.py:185-193). The MI355X recipe override sets it explicitly: `{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}` plus `--block-size 128` (RCP:290-300). Without that flag: 2.5 x (584+132) = 1790 B/token (fp8_ds_mla) or 2.5 x (512+132) = 1610 B/token (fp8_e4m3). Minor: `nvfp4_ds_mla` is allowed whenever `_use_v41_mxfp8_kv_record()` is true, i.e. SM100 or Ampere (attention.py:203-210), not SM100 only; irrelevant for ROCm (non-CUDA -> raises). |
| V2 | swa_bounded_replay disabled "under PP splits" | corrected | Not every PP split: replay is off only when the pipeline stage holding the last KV-source layer (20) does not also hold every later layer (`start_layer > cut or end_layer < num_hidden_layers`, V/vllm/models/deepseek_v41/amd/model.py:1128-1132). Other gates confirmed: sequence parallel / PCP>1 / ubatching (model.py:1133-1140), Engram layer at/after 20 (1141-1142), drafter window > 128 or non-SWA drafter layers (1143-1151), ROCm CDNA 3/4 only (attention.py:637-650), PCP>1 without prefix caching (attention.py:626-632). "Requires model runner V2" is from the config docstring (V/vllm/config/cache.py:381-386); no enforcing check located. Also relevant for PD (not in the original claim): the NIXL and Mooncake connectors special-case bounded replay (V/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:105,408; mooncake/mooncake_connector.py:1055,1108-1114: D side loads N-1 tokens and recomputes the last one). |
| V3 | "adding TP does not raise per-GPU KV capacity" | corrected | KV specs are `num_kv_heads=1, max_tp_shards=1` and `max_tp_shards` means "TP ranks beyond this hold replicas" (V/vllm/v1/kv_cache_interface.py:458-460), so per-token KV bytes per GPU are NOT divided by TP. But per-GPU KV token capacity does grow with TP because resident weights per rank shrink (the document's own table 4.2: 75M -> 117M -> 147M). Correct wording: TP does not shard KV; it adds KV capacity only through freed weight memory. |
| V4 | Decode KV upper bound TP2/TP4/TP8 = 75M/117M/147M tokens using 145/81/36 GiB resident weights | corrected (inputs) | Arithmetic is correct for the stated inputs (0.9 x 288 = 259.2 GiB minus 145/81/36 GiB, / 1630 B). But the inputs are not MI355X measurements and the recipe contains a conflicting measurement: on MI325X (gfx942) TP4 with Engram offloaded, vllm-project/vllm#60731 measured 98.88 GiB of weights per rank without monokernel and 170.11 GiB with the monokernel weight copies (RCP:2264-2270); "81.2 GiB" is the H200 TP4 measurement (RCP:361-364), "~145 GiB" TP2 and "~36 GiB" TP8 are comments for MI300X/MI325X and H100 (RCP:131-133, 256-271). The MI355X presets pass `--monokernel` (RCP:287, 306 region; RCP:1100), whose weight-copy cost on gfx950 is not stated in RCP. Also RCP's own H200 measurement shows 38.5 GiB KV ~= 20M tokens (RCP:361-364), i.e. ~2.07 KB effective per token, above 1630 B. Treat the 75M/117M/147M figures as unvalidated; the real number needs an MI355X startup log ("Available KV cache memory"). |
| V5 | PD transfer per request 85/166/329 MB (fp8_ds_mla), 75/148/293 MB (fp8_e4m3), 47/92/181 MB (official) | confirmed (arithmetic) | Recomputed: e.g. 100k x 1630 B + 128 x 43 x 584 B = 166.2 MB. Same caveat as V1: assumes `indexer_kv_dtype=mxfp4`; with the engine-default FP8 indexer the fp8_ds_mla figures become 182 MB at 100k. SWA term counts 43 layers incl. 3 DSpark and 128 slots; vLLM allocates SWA in 32-token blocks (sparse_swa.py block_size=32, attention.py:651-656), so the resident SWA per request is block-rounded (not quantified). |

Evidence: all rows first-hand code/file reads by the verifier; no GPU runs.
