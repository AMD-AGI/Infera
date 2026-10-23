# DCP4 decode memory map (workflow wf_5788f636-68c, code read of ab7a898b77, 2026-10-22; nothing run)

# SYNTHESIS

# DCP4 decode leg (GLM-5.2, MTP 5/1/6): memory synthesis for reaching mfs 0.92

Read-only code check on ab7a898b77. I spot-checked three things myself: the wide page table is never dropped on HIP (`layers/attention/dsa_backend.py:1598-1622`), the draft's KV and indexer are charged `×dcp_size` (`model_executor/pool_configurator.py:303-305`), and the index-K buffer is replicated (`mem_cache/memory_pool.py:5346-5352`). The `dcp_replicated` path that skips that replication only switches on when `loc_space_scale > 1` (`mem_cache/kv_cache_configurator.py:2175-2176`), and here `loc_space_scale` is 1. Everything else comes from the four code readers. All "GB" figures are GiB (`utils/common.py:575`).

## 1. Memory budget for one decode rank (TP rank 0)

| Consumer | 0.85 (S2) | Projected at 0.92 | Source | Static budget? |
|---|---|---|---|---|
| Free before weights load (`pre`) | 270.13 | 270.13 | log | n/a (budget baseline, `distributed/bootstrap.py:187-216`) |
| Target weights, incl. bf16 kv_b (~1.1, from `KEEP_KV_B_BF16`) and the 581 MiB gfx950 indexer workspace | 102.57 | 102.57 | log | inside |
| Draft weights | 5.46 | 5.46 | log | inside |
| Target KV pool: 56,016 B/row = 78×576 + 4×21×132 | 115.68 | ~134.6 (inference: +18.9 GiB / 58,848 B per row ≈ 2.56M rows) | log; formula is code-derived and matches to <5 B/row | inside |
| Draft KV pool | 2.28 | ~2.6 | log | inside |
| Draft over-charge: budgeted but never allocated (`pool_configurator.py:303`) | ~3.57 (inference) | ~4.1 | code | budgeted, ends up as free memory |
| `req_to_token` (`mem_cache/memory_pool.py:404-406`) | ~0.46 (inference) | ~0.46 | code | outside |
| Reserve, (1−mfs)×pre (`mem_cache/kv_cache_configurator.py:2834`) | 40.5 | 21.6 | code | — |
| **Free after pool sizing** | **43.56** | **~24.7 (inference)** | log / projection | — |
| Q-rep all-gather + backend init, 79 layers (`model_executor/model_runner.py:1118-1157`) | 5.29 (code estimates 5.86; 0.57 gap unexplained) | ~5.3 | log window | outside |
| Target-verify graph (incl. 2.25 page table) | 5.75 | 5.75 | log | outside |
| Draft-decode graph (incl. 1.50 page table) | 1.92 | 1.92 | log | outside |
| Draft-extend graph (incl. 2.25 page table) | 3.40 | 3.40 | log | outside |
| **`available_gpu_mem` after capture** | **27.20** | **~8.3 (inference)** | log / projection | — |
| Runtime eager peak (sampling, `_transform_table_1_to_real` at ~38 MB per call) | unknown | unknown | — | outside |

S1 reached 10.53 at 0.92 and served stably. That was a different topology, so it is a reference point, not a bound for S2.

## 2. Knobs that cut post-pool usage

| Knob | Expected saving (GiB) | Perf risk | Evidence |
|---|---|---|---|
| `--context-length 262144` | Page tables 6.0 → 1.5, saving ~4.5. Split-indexer logits ~0.45 more. `req_to_token` ~0.35 more. | Rejects requests over 256K; the workload is ~180K. | Code (`dsa_backend.py:1613-1622`, `kv_cache_configurator.py:864-866`). Size is inference. |
| `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1` | 3.40 (measured size of that graph) | Draft extend runs eager every step. | `environ.py:1502`, `eagle_worker_v2.py:583` |
| `--cuda-graph-max-bs-decode 64` | Page tables 6.0 → 4.0 plus some activations | bs > 64 runs eager | `cuda_graph_hook.py:518-530` |
| `--cuda-graph-bs-decode 96` (one size) | Between 0 and ~5.07 of non-page-table memory. Page tables stay at 6.0. | Every batch is padded to 96. | Code. The amount is unknown. |
| `--cuda-graph-backend-decode disabled` | All 11.07 | Large decode slowdown | `eagle_worker_v2.py:464`. Use only as an upper-bound experiment. |
| Move q-rep before pool sizing | 0 net; ~5.3 moves into the static budget | None | Code read. Needs a code change. |
| Make the local q_b / w_kc a view of the gathered tensor | ~1.45 (inference) | Kernel stride compatibility not verified | Code change, inference |
| Drop `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER` | Costs ~1.8 more; keep it on | — | `dsa_dcp_split_indexer.py:233` |
| `--disable-custom-all-reduce` | Does not affect post-pool memory. It may raise "Load weight begin" (observation b), i.e. more KV. | Slower all-reduce | Size unknown; aiter buffer size was not read. |
| Shard index-K | ~17 of pool (KV capacity, not headroom) | Design change | Code (`memory_pool.py:5346-5352`) |
| Fix the draft over-charge | Gains ~3.6 of KV but **costs** ~3.6 of headroom | — | Code. Works against the 0.92 goal. |

## 3. Experiments: decode server only, one variable each

Read these lines each time: "Load weight begin", "Memory pool end", "Capture … end mem usage" for each runner, and `available_gpu_mem`.

1. **E1:** 0.92 with the current S2 flags. Does capture pass? What is the final avail? This tests the ~8.3 projection.
2. **E2:** E1 + `--cuda-graph-bs-decode 96`. Gives the single-size share of the 5.07 GiB of non-page-table graph memory.
3. **E3:** E1 + `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1`, leaving the verify and draft-decode graphs.
4. **E4:** E1 + `--context-length 262144`. Tests the page-table formula 16·B·(ctx+10)·4 B.
5. **E5:** E1 + `--cuda-graph-max-bs-decode 64`.
6. **E6:** E1 + `--cuda-graph-backend-decode disabled`. Ceiling for avail with no graphs.
7. **E7:** E1 + `--disable-custom-all-reduce`. Read only the change in "Load weight begin" (observation b).
8. **E8:** Push mfs further (0.94, 0.95) using the best single-variable result from E2–E5, combined only after each has been measured on its own, to find the maximum that still captures.
9. **Diagnostics (temporary log lines, in a yihou copy):**
   - `torch.cuda.memory_allocated()` before and after `_prepare_replicated_q_proj`, to explain the 0.57 GiB gap.
   - `mem_get_info` after each `init_model_parallel_group`.
   - `available_bytes` and `_cell_size` at `pool_configurator.py:612`, to confirm the 3.57 over-charge.

**Under load (needs the prefill leg):**

10. Best E-config at C80 AgentX with ~180K contexts for ≥10 min. Log `torch.cuda.memory_stats()` `reserved_bytes.all.peak` and `allocated_bytes.all.peak` right after capture and periodically, plus `mem_get_info`. Watch for OOM or retraction.
11. Repeat 10 at the next mfs step up, against 10 as the reference.

## 4. Open questions (measurement only)

- **Final avail at 0.92 with current flags.** E1 answers it.
- **Split of the ~5.07 GiB non-page-table graph memory across batch sizes and runners.** E2 and E3; `--enable-profile-cuda-graph`.
- **The 0.57 GiB gap between the q-rep code estimate (5.86) and the measured window (5.29).** Answered by the `memory_allocated` delta.
- **Size of the aiter custom-AR buffers for the second (DCP) group** (observation b). E7.
- **Runtime eager peak at C80 / 96 running.** Experiment 10. No code bounds it.
- **Whether EP4 with `moe-a2a-backend none` allocates persistent buffers.** Grep the S2 log for buffer lines.
- **`page_size` and `disaggregation_decode_extra_slots`.** Read the server_args log line. These set tail waste and `req_to_token` rows.
- **Capacity (inference):** at 0.92, ~10.2M logical tokens fit about 55 requests of ~184K. A 96-request graph is therefore only reached with shorter contexts, so graph max bs 64 may cover real batches. The C80 batch-size histogram in experiment 10 would show this.

# READER: accounting

# KV pool sizing for the DCP4 + EAGLE decode leg on dev/yihou/glm52_dcp_pd @ ab7a898b77

Only the code was read; nothing was run. The log I used is the DCP decode log `yihou_ws/.../design/tmp_kd/dcp4.decode.head.log`, run at mem 0.85 with max running requests 64. It is not the S2 run.

## 1. How mem_fraction_static becomes the pool size (from the code)

- **Units.** `get_available_gpu_memory` calls `empty_cache`, reads `mem_get_info`, takes the MIN across ranks when `distributed`, and returns GiB (`utils/common.py:445-455, 567-575`). Every "GB" in these logs is GiB.
- **Baseline.** `pre_model_load_memory` is that MIN, measured after parallel init and before weights load (`distributed/bootstrap.py:187-216`, `model_runner.py:1257`).
- **Budget.** `rest = avail_now - pre*(1-mfs)` (`mem_cache/kv_cache_configurator.py:2806, 2834`). Everything already resident is charged against this budget: target weights, draft weights, and the 581 MiB gfx950 indexer workspace allocated during load.
- **Token count.** `tokens = (rest - fixed_overhead) // cell_size`, floored to the page size (`model_executor/pool_configurator.py:612-622`).
- **Target cell size.** MLA dim × layers × elem, plus `indexer_cell × dcp_size` (`pool_configurator.py:375-403`).
- **Draft cell size, added to the same cell.** It is computed as `draft_kv×dcp + draft_idx×dcp` (`pool_configurator.py:279-305`).
- **What the draft pool actually allocates.** On ROCm the DSA draft shards like the target (`configs/model_config.py:214-220`). So `loc_space_scale = 1` and the draft latent is per physical token (`kv_cache_configurator.py:497-502`).
- **The draft is over-charged.** The budget charges 4×576 B per token for the draft latent, but only 576 B is allocated.
- **Arithmetic check (inference).** Assumes 576 B fp8 latent and 132 B index-K per layer:
  - Over-charge: 1728 B × 2,217,280 tokens = 3.57 GiB that is budgeted but never allocated, so it ends up as headroom.
  - The draft pool log matches the sharded layout: 2.28 GiB / 2,217,280 = 1104 B = 576 + 4×132.
  - Predicted post-pool avail at 0.85: 40.51 slack + 3.57 over-charge − 0.46 req_to_token = 43.6. The log shows 43.69.

## 2. Observation (a): S2 stores 52 KB per token vs 44 KB for S1

- The target index-K buffer is sized for logical tokens: `(size + page)*dcp - page` (`mem_cache/memory_pool.py:5346-5352`). Each rank holds index keys for all 4× tokens; only the latent is sharded.
- The numbers fit only if 21 of the 78 layers own index-K. The rest are skip_topk layers (`kv_cache_configurator.py:2177-2182`).
  - S1: 78×576 + 21×132 = 47,700 B, against 47,701 measured.
  - S2: 78×576 + 4×21×132 = 56,016 B, against 56,020 measured.
- Replicated index-K is therefore about 20% of the S2 pool.
- The 21-layer count is inferred, not read. It is settled by `sum(not skip_topk_layers)` or by `index_k_with_scale_buffer` sizes.

## 3. Allocations after pool sizing (observation c)

Order: `alloc_memory_pool` → `_init_post_memory_pool_components` → `init_attention_backends` → `_prepare_replicated_q_proj` (`model_runner.py:964-1118`).

- **dcp_replicate_q_proj.** Turned on by `dcp_comm_backend=a2a` (`arg_groups/model_overrides/deepseek_v2.py:41-42`).
  - For each MLA layer it all-gathers `w_kc_qrep` and `q_b_proj_qrep_weight` (`model_runner.py:1120-1157`). The local shards are kept.
  - Formula: q_b full = 64×256×2048×2 B = 67.1 MB per layer; w_kc full = 64×192×512×2 B = 12.6 MB per layer.
  - 79 layers (78 target + 1 draft) gives 6.30e9 B = 5.86 GiB.
  - The log shows only 5.29 GiB between pool end and capture begin (43.69 → 38.40). That window also contains backend init, so the code estimate is 0.57 GiB above the measurement. I cannot explain this gap; it is unknown.
  - To settle it: log `torch.cuda.memory_allocated()` before and after `_prepare_replicated_q_proj`, and the sum of `numel × element_size` of the two `*_qrep` tensors.
- **KEEP_KV_B_BF16.** This is a pre-pool cost, already inside the 102.57 GB of weights.
  - It keeps kv_b in bf16 (`models/deepseek_common/deepseek_weight_loader.py:83-85`).
  - Per layer, as bf16 local copies: `kv_b_proj.weight` 7.3 MB, w_kc 3.1 MB, w_vc 4.2 MB (`deepseek_weight_loader.py:752-776`). That is about 1.1 GB per GPU in total.
- **DSA backend (HIP).** `kv_indptr`, `kv_indices` of size (req pool + 1) × 2048 × int32, and `_arange_buf` (`layers/attention/dsa_backend.py:639-651`). All under 2 MB. The DCP comm buffers (`layers/dcp/comm.py:506-570`) are allocated per forward, not persistently.
- **req_to_token.** Allocated inside the pool phase but not charged to the budget: (reqs + pre_alloc + 1) × (context_len + extra) × int32 (`memory_pool.py:404-406`). With context_len 1,048,576 this is about 0.46 GiB.
- **Custom all-reduce.** Created before weights load (section 4), not after.
- **Sampling buffers.** Not quantified.

## 4. Observation (b): 7.76 GB less free before "Load weight begin"

- **First-hand:**
  - DCP creates an extra `dcp` group with its own RCCL communicator and custom all-reduce (`distributed/parallel_state.py:2713-2733`, `use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE` at `:2171`).
  - The DCP log shows `AiterCustomAllreduce` constructed twice, for the TP and DCP groups.
  - The DPA log I read has `disable_custom_all_reduce: True` and the line "custom AR disabled".
  - QuickAllReduce stays off unless `ROCM_QUICK_REDUCE_QUANTIZATION` is set (`device_communicators/quick_all_reduce.py:178-189`).
- **Unknown:** the size of the aiter buffers. Aiter is not installed locally.
- **To settle it:** log `mem_get_info` after each `init_model_parallel_group`, or rerun S2 with only `--disable-custom-all-reduce` added and compare the "Load weight begin" line.
- **Effect:** this memory comes off the KV budget, not off the post-pool headroom, because `rest` is measured after it.

## 5. Can these be moved or freed?

- **Move q-rep before sizing (code read).** It only needs the model and the DCP group, so it could run after `load_model`. Its 5.3 GiB would then be charged to KV tokens instead of eating the slack. This creates no new memory: at the same headroom it is equivalent to lowering mfs by about 0.02.
- **Free the duplicate local shards (inference).** The q-rep path is used only for decode and target_verify (`attention_forward_methods/forward_mla.py:140-146`). Extend and draft-extend still read the local q_b and w_kc. Because `all_gather(dim=0)` concatenates the rank shards, the local weight could become a view of the gathered tensor, saving about (16.8 + 3.1) MB × 78 ≈ 1.45 GiB. Whether the kernels accept those strides is unverified.
- **Fix the draft over-charge.** Charge the draft latent without `×dcp` when `dsa_draft_shards_under_dcp` is true. This turns 3.57 GiB of hidden headroom into KV tokens. It reduces headroom, so it does not help the 0.92 goal.
- **Cut index-K replication.** Sharding index-K would save about 8.3 KB per token × 2.2M ≈ 17 GiB of pool. That is a design change, not a flag.
- **Post-capture KV sizing is unavailable here.** It is gated off for `dcp_size != 1` and for MLA (`arg_groups/overrides.py:1790-1793`).

## 6. Projected headroom at 0.92 (inference, unmeasured)

| Step | GiB |
|---|---|
| Slack, 0.08 × 270.07 | 21.6 |
| + draft over-charge | +3.57 |
| − req_to_token | −0.46 |
| Post-pool avail | ≈ 24.7 |
| − q-rep | −5.3 |
| − S2 graphs | −11.07 |
| Left for runtime activations | ≈ 8.3 |

S1 at 0.92 left 10.53 GiB. The real number is settled by the "available_gpu_mem=" line of a 0.92 run.

# READER: graphs

**CUDA graph memory, EAGLE decode path, DCP4 with the triton DSA decode backend (read-only code read of ab7a898b77)**

The largest graph cost comes from the model's 1M-token context length (`context_len=1048576`), not from how many batch sizes are captured. Each runner allocates a page table of `[max_num_tokens, context_len+10]` int32. With the current settings these tables add up to about 6.0 GiB of the 11.07 GiB. Cutting the captured size list leaves them unchanged; lowering `--context-length` or the maximum captured batch size shrinks them.

### 1. How the captured batch-size list is built (first-hand)
- The flags are `--cuda-graph-bs-decode <list>`, `--cuda-graph-max-bs-decode N`, `--cuda-graph-backend-decode {full,breakable,disabled}` and the `--cuda-graph-config` JSON. The JSON wins over the convenience flags (`arg_groups/cuda_graph_hook.py:32-100`; fields at `arg_groups/fields/exec_.py:572-636`). A plain `--cuda-graph-max-bs` does not exist on this branch.
- If no list is given, it is generated from max_bs. With speculative decoding it is 1..8, 10..32 step 2, 40..64 step 4, 72..max step 8 (`cuda_graph_hook.py:518-530`). For max_bs 96 that is 31 sizes, matching the log.
- If a list is given, max_bs is set to its maximum (`memory_hook.py:202-206`).
- `get_batch_sizes_to_capture` then filters the list by alignment and by the request pool size (`base_cuda_graph_runner.py:65-110`).
- All three runners call that same function on the same decode list: target verify (`decode_cuda_graph_runner.py:332`), draft decode (`eagle_draft_cuda_graph_runner.py:156`) and draft extend (`eagle_draft_extend_cuda_graph_runner.py:149`). There is no per-runner override.
- In PD decode mode, prefill graphs are disabled automatically unless set explicitly. That is where `prefill.backend='disabled'` in the log comes from (`cuda_graph_hook.py:557-576`).

### 2. Off switches (first-hand)
- **Draft-extend graph:** env `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1` (`environ.py:1502`, checked at `eagle_worker_v2.py:583`).
- **Draft-decode graph:** no switch. It is captured whenever `speculative_num_steps > 1` (`eagle_worker_v2.py:479`).
- **Everything:** `--cuda-graph-backend-decode disabled` turns off target and draft graphs together (`eagle_worker_v2.py:464`).

### 3. Shared graph pool (first-hand)
- All three runners get their backend from `resolve_decode_backend`, which builds a full-graph backend (`runner_backend/utils.py:48-104`).
- That backend takes the process-wide pool from `get_or_create_global_graph_memory_pool` (`full_cuda_graph_backend.py:128-129`, `runner_utils/pool.py:109-115`, slot at `runtime_context.py:618`). Target and draft share one process, so they share one pool.
- Batch sizes are captured largest first so smaller sizes reuse the pool (`decode_cuda_graph_runner.py:1150-1185`).
- The logits output goes into a shared static buffer (`decode_cuda_graph_runner.py:415-417`).

### 4. Which buffers scale with what (first-hand)
- DSA `init_cuda_graph_state` allocates `page_table = zeros(max_num_tokens, req_to_token.shape[1]) int32`. On HIP, and whenever speculative decoding is on, this wide table is never dropped (`dsa_backend.py:1597-1625`).
- `req_to_token` width is `context_len + 4 + num_draft_tokens` (`kv_cache_configurator.py:864-866, 1606`). Here that is 1,048,586 columns, so each row is 4.0 MiB.
- `max_num_tokens` per runner (the "Capture ... begin/end" lines measure in GiB, `utils/common.py:575`):

| Runner | max_num_tokens | Page tables | Measured total | Remainder |
|---|---|---|---|---|
| Target verify | max_bs × 6 = 576 (`decode_cuda_graph_runner.py:368-372`) | 2.25 GiB | 5.75 | ~3.5 |
| Draft decode | 4 backends (steps−1) × 96 = 384 (`dsa_backend.py:4313-4329`, `eagle_draft_cuda_graph_runner.py:158-163`) | 1.50 GiB | 1.92 | ~0.4 |
| Draft extend | max_bs × 6 = 576 (`eagle_draft_extend_cuda_graph_runner.py:153-158`) | 2.25 GiB | 3.40 | ~1.15 |

- Inference: the remainder is activations in the graph pool plus small metadata. Splitting it per batch size is unknown. It can be settled by `torch.cuda.memory_stats()` around each `capture_one`, or `--enable-profile-cuda-graph`.
- Page-table formula: **GiB ≈ 16·B × (ctx+10) × 4 B / 2³⁰**, where B is the largest captured bs. These tables depend only on B and context length, not on how many sizes are captured.

### 5. Concrete flags and estimates
Estimates are inference; the reference is the 11.07 GiB measured in S2.
1. **Capture only bs 96:** `--cuda-graph-bs-decode 96`. Page tables stay at 6.0 GiB. The total lands between 6.0 and 11.07 GiB; the exact saving needs one run. Batches below 96 are padded to 96 (`eagle_draft_extend_cuda_graph_runner.py:339-340`).
2. **Sparse list:** `--cuda-graph-bs-decode 8 16 32 48 64 80 96`. Same 6.0 GiB floor; somewhat above option 1.
3. **Verify graph only, as far as possible:** `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1` saves the measured 3.40 GiB, leaving about 7.67 GiB. Dropping the draft-decode graph too needs `--speculative-num-steps 1`, which changes the algorithm, or `--cuda-graph-backend-decode disabled`, which removes all graphs.
4. **Biggest lever:** `--context-length 262144` (the workload is about 180K). Page tables become 1536 rows × 1 MiB = 1.5 GiB, saving about 4.5 GiB. It also shrinks the `req_to_token` pool (`(size+pre_alloc+1) × width × 4 B`, `disaggregation/decode.py:189-199`), which is unmeasured.
5. **Lower max bs:** `--cuda-graph-max-bs-decode 64` brings page tables to 4.0 GiB. Batches above 64 then run eager, which needs its own runtime activation memory (unknown).

### 6. Observation (d): why S2 graphs (11.07 GiB) are smaller than S1 (13.2 GiB)
- S1 has a per-DP-rank max bs of 64. That gives 16 × 64 = 1024 rows, about 4.0 GiB of page tables, assuming S1 also ran with ctx 1M (not checked in the S1 log).
- S1's draft decode measured 1.17 vs 1.0 predicted, and draft extend 1.57 vs 1.5 predicted, so the formula fits.
- S2's draft graphs are larger (1.92 / 3.40) because its max bs is 96 instead of 64.
- S1's verify graph is the outlier: 10.46 GiB vs S2's 5.75. Inference: DP attention gathers tokens from all DP ranks before the MoE, so verify activations cover more than the local 384 tokens. This is unconfirmed; comparing S1/S2 `memory_stats` peaks during verify capture would settle it.
- Net: S2 is smaller only because S1's verify graph is so large. S2 carries more page-table memory in its draft graphs.

### Unknowns
- Whether the DCP triton path allocates more inside its graphs (the ~1.15 GiB draft-extend remainder).
- How the ~5.07 GiB non-page-table share splits across the 31 sizes.
- Both need a measured run: options 1 and 3 compared against S2's "Capture ... end mem usage" lines.

# READER: kvbytes

Findings (kvbytes): DSA pool bytes per token and DCP4 decode admission

**1. Pool class and layout (first-hand)**
- The pool is `DSATokenToKVPool`, a subclass of `MLATokenToKVPool` (python/sglang/srt/mem_cache/memory_pool.py:5288). It is built by `_build_dsa_kv_pool` (mem_cache/kv_cache_configurator.py:2148-2190).
- **MLA rows:** each layer has `(size + page_size, 1, kv_cache_dim)` rows (memory_pool.py:4867-4873). On HIP with the triton/tilelang/aiter backends, `kv_cache_dim` is the raw 512 + 64 = 576 bytes in fp8, with no scales and no bf16 rope (kv_cache_configurator.py:3241-3247). Per layer: 576 B per row.
- **Indexer K + scale:** each indexer layer stores 128 + 128/128*4 = 132 B per token as uint8 (index_key_cache.py:48-55; pool_configurator.py:542-549).
  - Layers marked `indexer_types == "shared"` get a 0-row placeholder (index_key_cache.py:63-66; model_config.py:423-427), via `_should_elide_dsa_index_k` (kv_cache_configurator.py:127-141).
  - The GLM-5.2 config lists 21 `"full"` layers out of 78: layers 0-2, then 6, 10, ..., 74. This is second-hand: I read it from an ATOM log of `amd/GLM-5.2-MXFP4`, /var/tmp/yihou/research_glm52_dcp/job_c16.log:1625.
- **DCP behaviour:** the MLA buffer is per-rank, sharded over `size` rows. The indexer buffer is **not** sharded: `index_buf_size = (size + page_size) * dcp_size - page_size` (memory_pool.py:5346-5352). So every rank holds index K for the full logical token space.
  - The env comment confirms this: "scores 1/W of the context pages ... over the replicated index K" (environ.py:1873-1875).
- "KV size" in the log is the MLA buffers plus the indexer buffers (memory_pool.py:5668-5672). Nothing else is counted.

**2. Reconciliation for observation (a)**
`GB` in the log is GiB (memory_pool.py:114).

| | Formula | Expected B/row | Measured |
|---|---|---|---|
| S1 (no DCP) | 78×576 + 21×132 = 44,928 + 2,772 | 47,700 | 126.99 GiB / 2,858,496 = 47,701 |
| S2 (DCP4) | 44,928 + 4×2,772 = 44,928 + 11,088 | 56,016 | 115.68 GiB / 2,217,280 = 56,019 |

- The 52.2 vs 44.4 KB in observation (a) are the same two numbers computed in 1e9 units.
- In S2 the indexer replication is 11,088 B/row, 19.8% of the pool, about 22.9 GiB per GPU. Of that, 3/4 (about 17.2 GiB) is the replicated part.
- Per logical token per rank, DCP4 costs 11,232 (MLA/4) + 2,772 (indexer) = 14,004 B, versus 47,700 B for DPA.
- **Draft pool:** 1 layer, all layers allocated, indexer also ×4.
  - S2: (576 + 528) × 2,217,280 = 2.28 GiB, matching the log.
  - S1: 708 × 2,858,496 = 1.88 GiB, matching the log.

**3. Sizing over-reservation (first-hand code; budget figure is inferred)**
- For EAGLE, the configurator charges the target cell with `draft_kv_size *= dcp_size` (pool_configurator.py:290-305).
- On ROCm the DSA draft is sharded like the target (`dsa_draft_shards_under_dcp`, model_config.py:214-220, so `loc_space_scale = 1`, kv_cache_configurator.py:496-502). Its MLA rows are therefore not ×4, and the measured 2.28 GiB confirms that.
- Sizing cell = 56,016 + 2,304 + 528 = 58,848 B/row. Allocated = 57,120 B/row.
- **Wasted:** 1,728 B × 2,217,280 = **3.57 GiB** that is budgeted but never allocated. It ends up as unused free memory.
  - Inferred cross-check: 115.68 + 2.28 + 3.57 = 121.5 GiB = 2,217,280 × 58,848, consistent.
  - To confirm, add a log of `available_bytes` and `_cell_size` at pool_configurator.py:612.

**4. Is `max_total_num_tokens = 8,869,120` logical capacity? Yes (first-hand).**
- It equals 2,217,280 × 4.
- The DCP allocator is built over `max_total_num_tokens * attn_dcp_size` tokens with `page_size * attn_dcp_size` pages (kv_cache_configurator.py:2724-2735).
- `logical_token_capacity` = rows × dcp (kv_cache_configurator.py:382-391); `logical_max_total_num_tokens` uses it (model_runner.py:1479-1486).

**5. Decode admission under DCP (first-hand)**
- The budget is `allocator.available_size()`, counted in **logical** tokens (free pages × logical page), minus `max(512 × n_active, need_space_for_single_req)` (disaggregation/decode.py:2041-2091).
  - `n_active` = running + in-transfer + waiting requests (decode.py:2000-2013).
  - The default `num_reserved_decode_tokens` is 512 (arg_groups/fields/disagg.py:106-109).
- Per request, admission requires `max(alloc_len_aligned + 512, prefill_len + min(max_new_tokens, 4096) - retractable)` (decode.py:488-497, 573-586; `CLIP_MAX_NEW_TOKEN` = 4096, environ.py:620).
- All of this is in logical tokens, so DCP4 reserves the full context once in logical space. Each rank physically holds 1/4 of the MLA rows. There is **no** ×4 over-reservation in admission.
- **DCP-specific waste:** alignment is to the allocator page, which is `page_size × 4` logical tokens. A request can waste up to `4 × page_size - 1` logical tokens on its tail. `page_size` for S2 is unknown; check the `page_size` field in the server_args log.
- **MTP/EAGLE extras:**
  - Per decode step it allocates `max(steps × topk, draft_tokens)` = max(5, 6) = 6 for topk 1 (mem_cache/allocation_sizing.py:52-53).
  - A 2× double-buffer reserve is applied per step (allocation_sizing.py:64-68).
  - Neither adds anything at admission beyond the 512 tokens per request.
- **Capacity bound (inferred arithmetic):** a 180K-token request needs about 184.1K logical tokens. 8.87M / 184.1K ≈ 48 such requests fit at mem 0.85, not 96. At 0.92 with the same cell size, rows scale by about (avail + 20 GB)/avail, roughly 2.6M rows, about 10.4M logical tokens, about 56 requests.

**6. Things that make DCP4 reserve or spend more than needed (ranked)**
1. **Indexer K replicated ×4 per rank:** 17.2 GiB of the 22.9 GiB indexer is the replicated part (memory_pool.py:5346-5352; pool_configurator.py:401-402).
   - The split decode indexer only scores owned pages, but the prefill/extend, draft-extend and non-split paths still read the full replicated index K (dsa_indexer.py:1557-1580 gates the split path).
   - Whether index K can be sharded is unknown. It needs a code read of every `index_k` write/read path, including the PD transfer's indexer state.
2. **Draft MLA over-charged ×4 in sizing:** 3.57 GiB unused (pool_configurator.py:303). This is the cheapest fix: skip the `*= dcp_size` when `dsa_draft_shards_under_dcp` is true.
3. **Allocator page = page_size × dcp:** tail waste per request, small.
4. **`CLIP_MAX_NEW_TOKEN` 4096 + 512 reserve per request:** this is logical and not DCP-inflated.

Not determined here: the observation (b) offset and the ~5.3 GB allocated after pool sizing (observation c). To settle (c), compare `torch.cuda.memory_stats()` before and after the `dcp_replicate_q_proj` log line.

# READER: runtime

**DCP4 decode leg: runtime GPU memory beyond the static budget (code read at ab7a898b77; nothing was run)**

Most of the memory in question scales with `context_len` (1,048,576 here, because `gmu/decode_only.yihou.sh` sets no `--context-length`), not with the real sequence length. The largest item is a set of wide page tables built just before graph capture. They make up about 6.4 of the 11.07 GB that S2 spent on CUDA graphs. Setting `--context-length 262144` would shrink them about 4x.

**1. Allocations that grow with `context_len` (first-hand)**

- **Wide page table per graph backend, built before capture.**
  - On HIP, `dsa_drop_wide_page_table` is always False (`dsa_backend.py:1597-1599`). So `init_cuda_graph_state` allocates `torch.zeros(max_num_tokens, max_ctx_len, int32)` with `max_ctx_len = req_to_token.shape[1]` (`dsa_backend.py:1613-1622`).
  - `req_to_token` width is `context_len + extra`, where extra is about 4 + 6 (`kv_cache_configurator.py:1651`, `allocation_sizing.py:96-105`).
  - Each runner calls this inside its capture begin/end memory window:
    - target verify: `max_bs*6 = 576` rows (`decode_cuda_graph_runner.py:368-373`)
    - draft decode: called on each of `steps-1 = 4` backends (`dsa_backend.py:4329-4331`, `eagle_draft_cuda_graph_runner.py:163`)
    - draft extend: `max_bs*num_draft_tokens` rows (`eagle_draft_extend_cuda_graph_runner.py:152-158`)
  - Arithmetic (inference): verify 576×1.05M×4 B = 2.42 GB; draft decode 4×96 rows = 1.61 GB, assuming draft decode passes `max_num_token = bs*topk = 96` (not checked); draft extend 2.42 GB. Total about 6.4 GB.
  - That fits the logged 5.75 / 1.92 / 3.40 GB, but the split between page tables and graph activations is unknown.
- **Split decode indexer logits.** `aiter_paged_mqa_logits` does `torch.empty(rows, max_seq_len, fp32)` (`kernels/ops/attention/dsa/paged_mqa_logits.py:110-114`).
  - In the split path, `max_seq_len = owned_pages*64`, with `owned = block_tables[:, r::4]` (`dsa_dcp_split_indexer.py:18-20, 233`).
  - In graphs, `block_tables` comes from the wide table (`dsa_backend.py:1830`), so it is about 16,385 pages, or about 4,097 owned pages, or about 262K columns.
  - Size: 576 rows ≈ 0.60 GB for verify and draft extend; 96 rows ≈ 0.10 GB for draft decode. It is allocated inside the graph pool during capture, so it is fixed in size at replay.
  - Without `SPLIT_DECODE_INDEXER`, the standard path allocates the full width instead (`dsa_indexer.py:1454`, 1265): about 2.4 GB for 576 rows.
  - Nothing caps it by budget or row chunking. The MQA-logits budget (`SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION`, `mqa_logits_utils.py:28-66`) applies only to `_get_topk_ragged` (`dsa_indexer.py:1786`), which is the extend path.
- **gfx950 fused indexer workspace, 581 MiB, allocated before pool sizing.**
  - Its width comes from `config.max_position_embeddings`, not `--context-length` (`dsa_indexer.py:526-545`). Rows are fixed at 48 (`gfx950_fused_indexer.py:31, 82-112`).
  - It is used only for 1 to 48 rows (`dsa_indexer.py:799-801`), and it declines the call if the page table is wider than the workspace (`gfx950_fused_indexer.py:218-233`).
  - Lowering `--context-length` does not shrink it.
- **`req_to_token` itself.** Size is `(max_running + disaggregation_decode_extra_slots + 1) × ~1.05M × 4 B` (`disaggregation/decode.py:191-197`). It is allocated with the pool; at about 4.2 MB per row, the total is unknown until the extra-slots value is known.

**2. Other runtime items**

| Item | Where allocated | Size (inference unless noted) | Source |
|---|---|---|---|
| DCP LSE merge (fp32 `[H, rows, 512]` + LSE all-gather + reduce-scatter) | per layer, in graph pool | ~75 MB at 576 rows if H=64 | `layers/dcp/comm.py:107-123` |
| Split-indexer candidate gather `[8, rows, 2048]` int32 | in graph | ~38 MB | `dsa_dcp_split_indexer.py:233-236` |
| `_transform_table_1_to_real` (`rows × 16,385` int32, fancy-index copy) | eager, torch caching allocator, every replay, per backend | ~38 MB per call; mainly churn | `dsa_backend.py:1086-1094, 1830` |
| `dcp_replicate_q_proj` (`all_gather` of `w_kc` and `q_b_proj` per MLA layer) | eager, after pool sizing, outside the static budget | likely the ~5.3 GB in (c) | `model_runner.py:1117-1158` |
| Verify logits `[576, V]` fp32 | in the verify graph (as long as the logits processor is captured; not verified) | ~0.35 GB at V≈151K | — |
| Eager sampling / acceptance (softmax, top-k/top-p copies) | eager | unknown | not traced |
| MoE EP4 dispatch buffers | — | not determined | see below |
| Mooncake decode-side staging | none on decode | 0 | see below |

- **MoE EP4:** the launch has no `--moe-a2a-backend`, so it is `none` (`arg_groups/fields/exec_.py:802`). Whether `--ep-size 4` with `none` allocates any persistent dispatch buffer is unknown. Settle it by grepping the S2 log for deepep/mori buffer lines.
- **Mooncake:**
  - `SGLANG_DISAGG_STAGING_BUFFER` defaults to False and is refused for MLA (`environ.py:824`, `disaggregation/decode.py:442-450`).
  - The DCP pack buffers are allocated in the handler that receives `decode_kv_args` (`mooncake/conn.py:2982`, `common/dcp_pack.py:93-129`), which reads as prefill-side.
  - Confirm by checking that no "PD DCP pack buffers allocated" line appears in the decode log.

**3. Fragmentation risk (inference)**

- Graph-captured tensors live in the graph private pool, so replay adds nothing new.
- Eager allocations every step: the `_transform_table_1_to_real` copies, the per-layer `all_gather` outputs in `_prepare_replicated_q_proj` (one-time), and sampling. They are small but vary with batch size.
- Measure the real runtime peak by comparing `torch.cuda.memory_stats()["reserved_bytes.all.peak"]` and `["allocated_bytes.all.peak"]` right after graph capture and after about 10 minutes at C80. Also log `mem_get_info` periodically.

**4. Knobs that cap these**

- `--context-length` (for example 262144, which covers ~180K contexts plus margin) shrinks the wide page tables, the split-indexer logits width and `req_to_token`. Estimated saving is about 4.8 GB of graph memory plus about 0.45 GB of logits; inference, not measured.
- `--cuda-graph-max-bs-decode` and `--cuda-graph-bs-decode` set the row count (multiplied by 6 for verify and draft extend).
- `--speculative-num-steps` sets the number of draft-decode backends (steps − 1).
- `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1` keeps the logits at 1/4 width; turning it off would cost about 1.8 GB more in graphs.
- `--disaggregation-decode-extra-slots` sets the extra `req_to_token` rows.
- No env var chunks the decode indexer columns or rows.

**5. Unknown**

- Eager peak in verify sampling. Settle with `memory_stats` around `verify()`.
- The breakdown of each 11.07 GB graph phase. Settle by logging `mem_get_info` before and after each `init_cuda_graph_state`, or by re-running `decode_only.yihou.sh` with `EXTRA="--context-length 262144"` and comparing the capture "mem usage" lines.