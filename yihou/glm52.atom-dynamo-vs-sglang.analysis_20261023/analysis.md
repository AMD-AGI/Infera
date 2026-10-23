# ATOM vs SGLang GLM-5.2 analysis (workflow wf_a4709a8a-cf1, 2026-10-23)

# CRITIC (final)

# Adversarial review of the draft "ATOM PR #209 vs SGLang"

## 1. Claims that are wrong or unsupported

1. **"Every effect number is at B40."** This is wrong for two items.
   - The GEMM rows figure is "−0.9 ms per step at B33/B36" (pr209_body.md, patch table).
   - The GC figure was measured at C64.
2. **A#2 and A#4 assume ATOM's timings carry over to our payloads.** They do not, because our payloads are twice as large.
   - **A#2, top-k gather.** ATOM gathers score only, fp32, 1.3 MB per rank ("Half the payload of the (score, gid) exchange", ATOM 661ef766 `atom/model_ops/dcp_ops.py:1284-1289`). Ours is int32 `[2,rows,topk]`, about 2.6 MB per rank (`dsa_dcp_split_indexer_triton.py:62`).
   - **A#4, merge all-to-all.** ATOM's 5.3 MB per rank is the 256-wide project-before-merge (PBM) payload. PBM is on by default (`atom/config.py:1718`) and is off only when `ATOM_USE_TRITON_MXFP4_BMM=1` (`envs.py:239`, `attention_mla.py:832-836`). Ours is 512-wide.
   - Whether 2.6 MB passes `should_custom_ag` is unknown. The gate limits per-rank size to max_size/(world×2) (`parallel_state.py:1300-1301`), and aiter's `max_size` is unread.
3. **A#7, index-K citation.** The draft cites the wrong branch. `dcp_replicated` is set only when `loc_space_scale>1` (`kv_cache_configurator.py:2175`). The target pool takes `(size+page)*dcp - page` (`memory_pool.py:5346-5352`). The conclusion still holds: every rank stores index-K for all logical tokens (`gmu/memory_map.md:103`).
4. **A#11 / B1#3, "raise decode memory to 0.94".** The draft omits our own contrary data point. S2m at 0.92 with the all-rank empty_cache patch gave 22,540, against 27,685 at 0.88 (`gmu/RESULTS.md:110,116`). That is a single run and the cause was not measured.
5. **A#3 applies to S1 only.** SGLang calls the same aiter function the JSON patches, `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant` (`forward_mla_rocm.py:231,292`), and passes config=None (`:156-160`).
   - In S1, v-up is that call. q-up runs inside `fused_fp8_bmm_rope_cat_and_cache_mla` (`:474-477`); whether that kernel reads the same JSON is unknown.
   - S1's M is rows per DP rank, not 160.
6. **B3, "InferenceX 39–40K vs ATOM CI 46–48K".** The PR #4045 body gives 39,245 for the sweep, 38,905 and 40,499 for single runs, and "ATOM CI reaches 46.1K at C112" (gh PR #4045 body, line 35). 48.2K is Infera's own number, not ATOM CI's.
7. **B2 step level, "ATOM 748 output tok/s per decode GPU".** No source supports this at C80. Linear interpolation of C72 321.9 and C96 387.3 gives about 344 per GPU, so 687 per decode GPU (README table). 748 corresponds to about C91.
8. **Unverified on our side.**
   - S1 requests/s 2.46 and the 57–64 % prefill hit: no source was read.
   - S1 prefill chunk 8192: the kit default is `PP4_CHUNK:-16384` (`llkit/kit/engine/worker.sh`, prefill block). The actual run value is unknown.
9. **B1#6, which client produced which number.**
   - ATOM's headline numbers used the CI client 288471b3; the kit's `agentx.sh` uses 918524ff (ATOM README, around line 152).
   - "Input-only vs total" is roughly right. The README column is input including cache-hit tokens; output is 246–406 tok/s/GPU (about 0.8 %).

## 2. Important items the draft missed

- **INT4 quick all-reduce.** ATOM sets `AITER_QUICK_REDUCE_QUANTIZATION=INT4` on both legs (ATOM kit `engine/worker.sh`, env block). SGLang has the same knob, `ROCM_QUICK_REDUCE_QUANTIZATION` (`quick_all_reduce.py:175`), but our kit does not set it. Accuracy is unmeasured.
- **Decode radix cache under DP-attention.** Enabling it logs "EXPERIMENTAL … requires prefix-aware DP rank routing" (`pd_disaggregation_hook.py:~106`). That matters for S1 (DPA4). Whether the Infera router routes by DP rank is unknown. The DCP prefix transfer requires `decode_prefix_len` aligned to the virtual page (`disaggregation/common/utils.py:166-169`).
- **Decode prefix hit depends on pool size.** The hit falls from C96 because "C × prompt length outgrows the decode's KV pool" (ATOM README, under the table). Items 1 and 11 interact.
- **The CPU tier does not explain the gap at C≤72.** LMCache matched the Store at C56/C72 (33,392 / 40,326); the drop to 30.7K/23.3K is at C96/C112 only (README).
- **S2 runs absorb in bf16.** ATOM runs the absorbed bmm in fp8 under DCP. S2 keeps kv_b in bf16 (`KEEP_KV_B_BF16`), which makes item 3 unusable in S2. Why it was chosen is in `design/qrep_bf16_kvb.md` (not read).
- **Identical choices between the two stacks.** ATOM prefill is the same TP1×PP4 (eager, partition 20,20,20,18) with chunk 8192 and block 16 (ATOM `env.sh`). ATOM decode is EP1 like S1. ATOM's indexer is FP8, not FP4: FP4 needs `--index-cache-dtype fp4` (`sparse_indexer_fp4.py:79`), which the kit does not set.
- **Query replication.** It is on by default in ATOM (`config.py:1717`). SGLang has `--dcp-replicate-q-proj`, defaulted on for a2a (`model_overrides/deepseek_v2.py:41`). Not a gap.

## 3. Corrected A–C (condensed)

### A. Items SGLang could absorb

All effect figures are ATOM's own, at about 101K context; none is reproduced on SGLang.

| # | Item | SGLang status (first-hand) | Applies to | ATOM effect |
|---|---|---|---|---|
| 1 | Decode prefix reuse | Exists, defaults to False (`disagg.py:85`), allowed with EAGLE (`speculative_hook.py:163-171`). DPA needs DP-rank routing. | S1, S2 | Hit 92/84/53/25 % at C56–112; no Δthroughput published |
| 2 | Top-k gather over the custom all-gather | int32 pack is rejected by the dtype gate (`parallel_state.py:1312`); ATOM views it as fp32 (`dcp_ops.py:38-79`) | S2 | 54→31 µs at 1.3 MB; ours is 2.6 MB, so unknown |
| 3 | bmm JSON configs | Same aiter function, config=None | S1, or S2 with fp8 kv_b | q 19.1→8.4 µs, v 15.2→8.1 µs at graph 40 |
| 4 | IPC all-to-all merge | pynccl (`parallel_state.py:1208-1213`); no graph buffers passed (`forward_mla_rocm.py:1030-1036`) | S2 | 56→34 µs at 5.3 MB (256-wide) |
| 5 | Fused rope/cat/KV write under DCP | Gated `dcp_size == 1` (`forward_mla_rocm.py:447`); ATOM runs it (`attention_mla.py:3159-3163`) | S2 | KD eager ≈ +3.0 ms unfused |
| 6 | Fused indexer projections | KD: 3 GEMMs; fused path only at ≤48 rows; ATOM `deepseek_v2.py:2087,1702-1722` | S2 | Not published |
| 7 | Sharded index-K | Full logical size per rank (`memory_map.md:103`) | S2 | Not published |
| 8 | Project-before-merge | Missing | S2 | Halves the a2a payload; Δtime unknown |
| 9 | Fused merge + FP8 quant | Needs #4 and #8 | S2 | 44.7→38.1 µs per layer |
| 10 | FP8 projections + GEMM rows | bf16 (KD) | Both | −0.9 ms at B33/B36 |
| 11 | Quick all-reduce INT4 | Knob exists, unset | Both | Not published; accuracy decision first |
| 12 | Decode memory 0.94 | 0.92 measured worse (22,540) | S2 | 160K→198.5K blocks |
| 13 | Async CPU tier | HiCache L2 blocking unknown | Prefill | Matters at C96+ only |

### B. Why SGLang is slower

**Config and recipe (first-hand):**
- MTP K3 at AL 2.99 vs EAGLE 5/1/6 at AL 3.61, i.e. 0.75 vs 0.60 accepted tokens per verify row.
- Decode prefix reuse is off on our side.
- KV pool: 12.7M (ATOM) vs 8.87M / 9.46M logical tokens (S2).
- MoE layout: S2 is EP4; ATOM and S1 are EP1.
- INT4 quick all-reduce: used by ATOM, not by us.
- Clients: 288471b3 vs 918524ff; measured windows 3600 s vs 1800 s.

None of these has a measured effect on the SGLang side.

**Engine (code read; none of it timed inside ATOM):**
- Top-k payload: 2× ATOM's, over NCCL.
- Top-k merge: ATOM rebuilds globally and emits owned slots directly (`dcp_ops.py:1315-1316`).
- DCP merge all-to-all: 512-wide over NCCL vs 256-wide.
- Fused rope/cat/KV write: gated off under DCP.
- Indexer: 3 GEMMs plus the 48-row cap.
- Index-K: replicated on every rank.
- Projections: bf16 vs fp8.
- Graphs: ATOM's InferenceX recipe sets `cudagraph-mode: FULL` (pr4045 `disagg-variants.yaml:61`).

**Throughput gap at the request level:**
- ATOM requests/s: 2.63 at C72, 3.14 at C96 (requests / 3600 s).
- ATOM TTFT mean: 1.50 / 2.68 s at C72 / C96, against S1's 5.37–5.5 s.
- S1's request rate is unverified, so the per-token vs per-request split of the gap is unknown.

**Not apples-to-apples:**
- SGLang draws one simulated acceptance length per batch (`spec_utils.py:456-470`); ATOM draws per token (`rejection_sampler.py:238-241`).
- Accuracy is not compared on a same-harness score.
- The ATOM CI vs InferenceX gap (46.1K vs 39.2K) is unexplained (PR #4045 body).

### C. Experiments, one variable each

1. Decode radix cache: S2 first, then S1 with DP-rank routing checked. Record TTFT, transfer bytes and cached_tokens.
2. EAGLE 3/1/4 at AL 2.99 vs 5/1/6 at AL 3.61, N=40, about 101K context.
3. S2 top-k gather with the fp32 view. Confirm aiter's all-gather is hit with a trace; record step p50.
4. S1: check the stock aiter JSON buckets, then install the bmm JSON; per-kernel trace.
5. Decode memory sweep 0.85 / 0.88 / 0.92 repeated, since a single run showed 0.92 lower.
6. S2 with EP1.
7. Fused rope/cat/KV write under DCP: eager correctness first.
8. `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` with an sgl-eval accuracy check.
9. S2 with fp8 kv_b plus the bmm JSON.
10. Prefill: HiCache L2 load time vs forward time per PP stage.
11. Rescore one ATOM AgentX JSON with our analysis tool.

# SYNTHESIS (draft)

# ATOM PR #209 vs SGLang GLM-5.2 PD on MI355X: what to absorb, why SGLang is slower, and what to test

Notes on evidence:
- **First-hand** means a code read or a number published by ATOM, Infera or InferenceX. **Inference** means it is reasoned, not measured.
- I re-checked three SGLang citations myself:
  - the `dcp_size == 1` gate at `forward_mla_rocm.py:444-452`;
  - the aiter all-gather dtype gate at `parallel_state.py:1312`;
  - the `disaggregation_decode_enable_radix_cache` flag at `arg_groups/fields/disagg.py:85`.
- Every effect number below is ATOM's published figure at B40, ~101K context, AL 2.99. None has been reproduced on SGLang.

## A. Speedups SGLang could absorb (ranked by effect per unit of effort)

| # | Item | Mechanism | SGLang status | Published effect | Effort | Risk |
|---|---|---|---|---|---|---|
| 1 | Decode prefix reuse | Decode matches its own prefix cache, so only the new suffix of KV is transferred (ATOM `mooncake_connector.py:539-562,1518-1528`) | **Has it, off by default.** `--disaggregation-decode-enable-radix-cache` defaults to False (`disagg.py:85`). Our kit never sets it (`llkit/kit/engine/worker.sh:58-125`). Both decode server-arg dumps show False. | Decode prefix hit 92/84/53/25 % at C56/72/96/112 (Infera README:158-163). No throughput delta published. | Flag | Untested with DSA + DCP4 + EAGLE on ROCm |
| 2 | Top-k exchange over the custom all-gather | View the int32 pack as fp32 so it passes aiter's dtype gate | **Missing.** The int32 `[2,rows,topk]` pack (`dsa_dcp_split_indexer_triton.py:62`) fails the gate at `parallel_state.py:1312` and falls back to pynccl | 54→31 µs per gather, 24 gathers per step | 2 lines | Low (the copy is bit-exact); `should_custom_ag` size check unconfirmed |
| 3 | Absorbed-bmm Triton configs | gfx950 JSON with per-M buckets ≤512 (`patch_aiter_mla_bmm_configs.py`) | **Missing.** `_absorb_bmm_config` returns None (`forward_mla_rocm.py:156`). Applies to S1's fp8 absorb only; S2 keeps kv_b in bf16. | q-up 19.1→8.4 µs, v-up 15.2→8.1 µs, ≈−1.4 ms/step | Data file | Low. Unknown whether our image ships the same stock JSON. |
| 4 | Custom all-to-all for the DCP merge | aiter IPC `alltoall_vec` that works under graph capture | **Missing.** `dcp_a2a_lse_reduce` (`layers/dcp/comm.py:474-537`) goes to pynccl/RCCL (`parallel_state.py:1210-1216`) and allocates buffers on every call (`forward_mla_rocm.py:1030-1036`) | 56→34 µs; −1.0 to −1.6 ms/step | aiter build patch + routing | Medium (image rebuild) |
| 5 | Fused rope + cat + KV write under DCP | ATOM runs `fused_qk_rope_concat_and_cache_mla` with DCP (`attention_mla.py:3138-3160`) | **Missing.** Gated `dcp_size==1` (`forward_mla_rocm.py:447`) | Not in PR #209. Our eager KD measured ≈+3.0 ms for the unfused path (`kernel_diff_dpa_vs_dcp.md`). | Medium | Low/medium |
| 6 | Fused indexer prep without a row cap | wk + weights_proj in one GEMM, plus a fused rope/quant/cache kernel (`deepseek_v2.py:2087,1702-1722`) | **Partial.** The fused gfx950 path stops at 48 rows (KD) | Not published | Medium | Low |
| 7 | Sharded index-K cache | Each DCP rank writes only the index keys it owns | **Missing.** `dcp_replicated` sets index_buf_size = size (`memory_pool.py:5343-5352`) | Not published. Saves index-K memory and writes. | High | Medium |
| 8 | Project-before-merge | Run W_V before the all-to-all, so it carries 256 instead of 512 per head (`attention_mla.py:826-836`) | **Missing** (inference from KD) | Not published | High | Medium |
| 9 | Fused all-to-all + LSE combine + FP8 quant | One kernel | No matching dataflow; SGLang's combine outputs bf16 | 44.7→38.1 µs/layer, −0.56 ms/step | High, needs #4 | Medium |
| 10 | FP8 projections + tuned M144-240 GEMM rows | ATOM quantizes every non-expert linear online with `ptpc_fp8` | **Missing.** Our projections run bf16 (KD:23,25,36) | qkv_a 14.9→11.5 µs, −0.9 ms/step (applies to ATOM's fp8 path only) | Accuracy decision first | Accuracy unmeasured |
| 11 | Decode memory | 0.94 with `AITER_REUSE_IDENTICAL_COMM_GROUPS=1` | S2 runs 0.85/0.88 (8.87M/9.46M tokens) vs ATOM's ~12.7M | ~160K→198.5K blocks | Config | OOM risk |
| 12 | Non-blocking CPU tier on prefill | Mooncake Store loads asynchronously | HiCache L2 write_back; whether it blocks the PP stages is unknown | ATOM: LMCache 30.7K/23.3K vs Store 46.2K/48.2K at C96/C112 | High | — |
| 13 | GC leaks and router token-id reuse | — | Different mechanisms in SGLang; impact unknown | +1.9 %, +0.5 % | Measure first | — |

Items 2, 4, 8 and 9 apply to S2 only. Item 3 applies to S1 only.

## B. Why SGLang is slower than ATOM

### B1. Config and recipe differences (first-hand unless marked)

| # | Difference | Evidence level | Measurement that would confirm its effect |
|---|---|---|---|
| 1 | **MTP depth.** ATOM uses K3 at AL 2.99 (4 verify rows per request, 160 rows at B40). We use EAGLE 5/1/6 at AL 3.61 (6 rows, 240 rows, 5 draft passes). Tokens per verify row: 0.75 vs 0.60. | First-hand config | SGLang EAGLE 3/1/4 at AL 2.99 vs 5/1/6 at 3.61, N=40, ~101K context |
| 2 | **Decode prefix reuse.** ATOM on (84 % hit at C72); ours off. `pd_vs_mixed_decode_diff.md:31` shows the full prefix KV is re-sent every turn. | First-hand | A/B with the flag; record TTFT, transfer bytes and decode cached_tokens |
| 3 | **Decode KV pool.** ~12.7M tokens (ATOM) vs 8.87–9.46M (S2). S2 admission backs up: running 30.8 against a cap of 96, queue mean 12.8. | First-hand. The link to admission is not measured. | Raise memory to 0.94 with shared comm groups, alone |
| 4 | **MoE layout.** ATOM decode EP1/TP4; our S2 is EP4 and hits an untuned FlyDSL shape (KD). | First-hand config. Impact unknown. | S2 with EP1 |
| 5 | **CPU tier.** Mooncake Store (async, ~1.5 TiB) vs HiCache. | First-hand | Per-stage prefill timeline: L2 load time vs forward time |
| 6 | **Concurrency, window and client.** C80 vs C56–112; 1800 s vs 3600 s; client 918524ff vs 288471b3. Input/output ratio 116–118 vs 119–134. | First-hand | Same client and window |
| 7 | **Metric.** ATOM's headline counts input tokens only; ours counts total. The difference is <1 %. Use input vs input: S1 34,215 vs ATOM ~42.1K interpolated at C80 (the interpolation is mine). | First-hand + inference | — |

### B2. Engine and kernel differences

All of these are first-hand from a code read, but **none is timed inside ATOM**.

| # | Difference | Status |
|---|---|---|
| 1 | Top-k payload and transport: score-only fp32 over IPC (ATOM) vs score + position over NCCL, ~4.85 ms/step eager (ours) | Ours is measured; ATOM's is not |
| 2 | Global-rebuild-then-filter merge: ATOM emits owned slots directly | — |
| 3 | DCP merge a2a: 512-wide bf16 over NCCL, 169.8 µs/call eager (ours) vs 256-wide over IPC (ATOM) | Ours measured; ATOM's width from code |
| 4 | Fused rope/cat/KV write disabled under DCP in SGLang | Gate at `forward_mla_rocm.py:447` |
| 5 | Indexer 48-row cap and 3 extra GEMMs in SGLang | From KD |
| 6 | Index-K written on all 4 ranks in SGLang | From code |
| 7 | Attention kernel: aiter asm `mla_decode_fwd` persistent (ATOM) vs Triton sparse-MLA (ours) | KD says DCP4 wins on sparse MLA vs our DPA4; vs ATOM's asm kernel: unknown |
| 8 | Projections: fp8 (ATOM) vs bf16 (ours) | Not measured |
| 9 | ATOM uses FULL cudagraph mode | Whether this matters: unknown |

**Step level (inference, context unmatched).**
- Output tok/s per decode GPU: ATOM 748 vs S1 543 vs S2 461–505.
- S1 TPOT (18.3 ms) is close to ATOM's ITL p50 (16.75–19.88 ms).
- By Little's law, S1's gap sits in requests/s (2.46 vs 2.63–3.14) and TTFT (5.5 s vs 1.5–2.7 s), not in per-token latency. Why is unknown.

### B3. Not apples-to-apples

- Per-row context is unknown on our side.
- K and AL differ.
- ATOM samples acceptance per request; SGLang draws one length per batch (`spec_utils.py:457-470` vs ATOM `rejection_sampler.py:238-241`).
- Client, window and concurrency differ.
- The prefill cache-hit definitions (57–64 % ours vs 94 % ATOM) are unverified to be the same.
- Whether ATOM's ITL equals our TPOT is unknown.
- Accuracy: fp8 + INT4 AR (ATOM) vs bf16 (ours), with no same-harness score.
- InferenceX's own ATOM 1P1D run gave 39–40K at C112, vs ATOM CI's 46–48K on the same Store size. The PR #4045 body leaves that gap unexplained.

## C. Experiments, one variable each, against S1/S2 as baseline

1. Turn on `--disaggregation-decode-enable-radix-cache`: a smoke run, then AgentX C80 on S1 and S2. Record TTFT, transfer bytes and decode cached_tokens.
2. Decodebench N=40, ~101K context: EAGLE 3/1/4 at AL 2.99 vs 5/1/6 at AL 3.61. Read S2's per-row context from its `SGLANG_LOG_DECODE_SEQ_LENS` log first.
3. S2 top-k gather: `.view(float32)` before the gather. Confirm with a trace that it lands on aiter's all-gather; record the step p50.
4. S1: install the PR #209 bmm JSON. First check our image's stock buckets; then a per-kernel trace.
5. Decode memory 0.94 with `AITER_REUSE_IDENTICAL_COMM_GROUPS`-equivalent sharing: record admission queue and running count.
6. S2 with EP1 instead of EP4 (MoE kernel A/B).
7. Enable the fused rope/cat/KV write under DCP (code change; check correctness eagerly first).
8. Route the DCP merge to aiter's custom all-to-all (needs an image with the patch).
9. Prefill: per-stage HiCache L2 load time vs forward time over the S1 window.
10. Rescore one ATOM JSON with our analysis tool to align ITL/TPOT and the token accounting.

Sources (read-only):
- `/home/yihou/dev/git/infera.glm52.pd` (origin/main)
- `/var/tmp/yihou/src/ATOM` @661ef766
- `/var/tmp/yihou/src/InferenceX` branch pr4045
- `/home/yihou/dev/git.16-19/sglang.pd.pp.dcp/yihou_ws/glm52_dcp_pd_20261021/atom_cmp/pr209_body.md`
- `/home/yihou/dev/git.16-19/sglang.pd.pp.dcp/yihou_ws/glm52_dcp_pd_20261021/design/kernel_diff_dpa_vs_dcp.md`

# READER pr209

# PR #209: which ATOM/aiter patches SGLang can absorb (decode TP4/DCP4, GLM-5.2, MI355X)

Infera `origin/main` holds the patch scripts under `deploy/docker/patches/atom_nightly/`. They target ATOM 661ef766 and aiter 9205e1059. Every "effect" number below is what PR #209 claims (first-hand published figures, read from `pr209_body.md` and the patch docstrings). I did not reproduce any of them.

## Patches, ranked by absorbability / effort

### 1. MLA absorbed-bmm Triton configs (`patch_aiter_mla_bmm_configs.py`) — data only, lowest effort
- **What it changes:** it rewrites aiter's gfx950 JSON configs for `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant`, for the shapes N512/K192 (q-up) and N256/K512 (v-up). It adds an `M_BOUNDS` list with one bucket per M up to 512.
- **Claimed effect:** the stock M buckets use BLOCK_M 64, and any M that is not a multiple of 64 takes a masked path at 2–2.5× the time. At graph 40: q-up 19.1→8.4 µs, v-up 15.2→8.1 µs, about −1.4 ms per step.
- **SGLang today:** not present. `_absorb_bmm_config` returns `None` unconditionally (forward_mla_rocm.py:156-160, our commit 4d1d6a647e), so aiter's stock configs are used.
  - v-absorb calls the kernel at :292-301 and q-absorb at :231-241. Both are reached only when `w_kc` is fp8, i.e. S1/DPA.
  - S2 sets `SGLANG_DSA_DCP_KEEP_KV_B_BF16=1`. That keeps kv_b in bf16 (deepseek_weight_loader.py:73-86), so the absorbs fall back to bf16 `torch.bmm` (:246, :316-322) and the patch does not touch them.
  - In S1, the q-absorb may go through the fused `fused_fp8_bmm_rope_cat_and_cache_mla` instead (gate at :444-452, `dcp_size==1`). Whether that kernel reads the same JSON is unknown.
- **Unknown:** whether our base image (`lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020`) ships the same stock JSON. Running the patch script's stock-bucket check inside our container would settle it.

### 2. DCP top-k exchange over the custom all-gather (`patch_atom_dcp_topk_custom_ag.py`) — one line
- **What it changes:** ATOM's top-k gather moves from pynccl to `dcp_all_gather`. That function reinterprets non-float payloads as fp32/fp16 so they pass aiter's dtype gate (ATOM 661ef766 `dcp_ops.py:35-80`).
- **Claimed effect:** 1.3 MB per rank, 54→31 µs, 24 gathers per step.
- **SGLang today:** not present, and for the same reason.
  - The split indexer calls `group.all_gather(pack, dim=0)` (dsa_indexer.py:1654).
  - The pack is `int32 [2, rows, topk]` (dsa_dcp_split_indexer_triton.py:62).
  - SGLang's aiter all-gather gate only accepts fp32/fp16/bf16 (parallel_state.py:1312). An int32 pack therefore falls through to mscclpp or pynccl (:1335+).
- **Fix:** `.view(torch.float32)` before the gather and `.view(torch.int32)` after; the copy stays bit-exact.
- **Unknown:** whether the size check `should_custom_ag` passes at our row counts (8 B per candidate). aiter 9205e1059 defaults `max_size` to 1 GB (`custom_all_reduce.py:876`, via `gh api`), but our image's aiter commit is not confirmed. A one-step trace showing which kernel the gather lands on would settle it.

### 3. DCP merge over a custom all-to-all (`patch_aiter_custom_alltoall.py` first half + `patch_atom_dcp_custom_a2a.py`) — medium effort
- **What it changes:** a new aiter `alltoall_vec` kernel that runs on the registered buffers of aiter's custom all-reduce and inside CUDA graphs. Its Python entry is `custom_all_to_all`; RCCL stays the fallback.
- **Claimed effect:** bit-identical to RCCL; 5.3 MB per rank, 56→34 µs; −1.0 to −1.6 ms per step.
- **SGLang today:** not present.
  - `dcp_a2a_lse_reduce` (layers/dcp/comm.py:474-537) packs the output plus LSE into one byte all-to-all.
  - That call goes to `_all_to_all_single`, which is pynccl, else torch/RCCL (parallel_state.py:1210-1216).
  - forward_mla_rocm.py:1030-1036 calls it without `cuda_graph_buffers`, so it allocates send/recv buffers on every call.
- **How to absorb:** SGLang on ROCm already uses aiter's `CustomAllreduce` (custom_all_reduce.py:392-402). Building Infera's aiter patch into our image would expose `custom_all_to_all` on `dcp_group.ca_comm`. The routing would go in `_all_to_all_single`.
- **Payload difference (inference from code, unconfirmed):** SGLang's payload is B·H·(D+lpd)·sizeof(out_dtype) per rank (comm.py:503-511). If the output is bf16, that is about 2× ATOM's 5.3 MB at the same M; the 2× follows from ATOM's 5.3 MB matching 1-byte elements. Whether our output dtype is bf16 or fp8 is unknown; logging the dtype once would settle it.

### 4. One-kernel all-to-all + LSE combine + FP8 quant (`patch_aiter_custom_alltoall.py` second half + `..._combine.py`) — high effort
- **What it changes:** a single aiter kernel reads this rank's heads straight from the peers' buffers, merges them with the LSE weights, and quantizes per token to FP8.
- **Claimed effect:** 160 rows 44.7→38.1 µs per layer, −0.56 ms per step on average.
- **SGLang today:** no equivalent, and no direct fit. SGLang's combine (`dcp_lse_combine_triton`, comm.py:533) produces bf16 that feeds the bf16 v-bmm. The fused quant only maps if the downstream path moves to fp8. It depends on #3.

### 5. Tuned FP8 GEMM rows for M 144–240 (`patch_aiter_glm52_decode_gemm_rows.py`) — not applicable as is
- **What it changes:** 27 FlyDSL `a8w8_bpreshuffle` rows, added as a new aiter `model_configs` CSV.
- **Claimed effect:** qkv_a at M 160, 14.9→11.5 µs; −0.9 ms per step.
- **Why ATOM needs it:** ATOM quantizes every non-expert linear online with `ptpc_fp8` (Infera kit `worker.sh:19-20`, PR #4045 `online_quant_config`).
- **SGLang today:** our measured traces show bf16 GEMMs for these projections. `kernel_diff_dpa_vs_dcp.md:23,25,36` lists q_a+kv_a on `gemm_a16w16`, q_b on hipBLASLt/triton a16w16, and o_proj on bf16 hgemm/Cijk. aiter's a8w8 table is never consulted for them.
- **This is a structural difference in the "why SGLang is slower" question.** fp8 vs bf16 projections has not been measured. The test is a per-projection kernel A/B at M 144–160.

### 6. API-process GC stalls (`patch_atom_api_gc_stalls.py`) — measure first
- **What it changes:** two leaks in ATOM's API process grew every gen-2 GC pass: finished sequences queued on `outputs_queue` that nothing reads online, and one metric gauge series per request.
- **Claimed effect:** C64 +1.9%, mean TTFT 2.85→2.04 s.
- **SGLang today:**
  - It freezes GC after warmup (http_server.py:2485-2529, `/freeze_gc` at :1229).
  - It has `gc_threshold` (engine.py:1966) and `gc_warning_threshold_secs`.
  - The kit runs `--enable-metrics` (worker.sh:73).
  - I found no metric label keyed on rid, but that grep was narrow.
- **Unknown:** whether SGLang has an equivalent heap growth. The test is a C80 run with `--gc-warning-threshold-secs` on, or py-spy on the decode tokenizer-manager process.

### 7. Router `prompt_token_ids` reuse (router change in PR #209)
- **What it changes:** ATOM PD is serial-pull, so its decode tokenized the prompt again after the prefill returned (~0.25 s at 130K tokens, claimed). The decode now takes the prefill's token ids.
- **Claimed effect:** C64 +0.5%, TTFT p50 −0.09 s.
- **SGLang today:** the router's SGLang protocol is concurrent and posts identical bodies to both legs (`sglang_bootstrap.py:26,52-62`). So yes, the decode tokenizes too, but in parallel with the prefill.
- **Unknown:** whether that tokenization ever lands on the critical path. Decode tokenize time vs prefill latency per request would settle it.

### 8. Not performance
- `patch_atom_dcp_sparse_pad_rows.py` fixes stale indptr offsets in padded eager rows, which caused memory access faults.
- `prebuild_aiter_mha_varlen.py` avoids a ~30 s JIT compile; the README says gfx950 does not load that kernel.
- `patch_lmcache_rocm_pinned_alloc.py` concerns the LMCache tier only.

## Summary (first-hand vs inference)

| # | Item | SGLang today | Effort | Applies to |
|---|---|---|---|---|
| 1 | bmm configs | missing (config=None) | data file | S1 v-absorb; S2 only if kv_b goes fp8 |
| 2 | top-k custom all-gather | missing (int32 dtype gate) | 1–2 lines | S2 |
| 3 | custom all-to-all merge | missing (pynccl) | aiter build patch + routing | S2 |
| 4 | fused combine+quant | no matching dataflow | high | S2, after #3 |
| 5 | FP8 GEMM rows | n/a: SGLang runs bf16 projections | needs an fp8-projection decision | both |
| 6 | GC | different mechanism; growth unknown | measure | both |
| 7 | token-id reuse | concurrent topology; impact unknown | measure | PD |

None of these patches has been measured on SGLang. Every per-step saving above is ATOM's number at its own shapes (B40, ~101K context, AL 2.99).

**Read-only sources:**
- `/home/yihou/dev/git/infera.glm52.pd` (origin/main)
- `/var/tmp/yihou/src/ATOM` @661ef766
- `/var/tmp/yihou/src/InferenceX` branch pr4045
- `/home/yihou/dev/git.16-19/sglang.pd.pp.dcp`

# READER inferencex

# GLM-5.2 MI355X configs in InferenceX: ATOM PD (PR #4045) vs SGLang

## 1. What exists (first-hand)

- **ATOM PD.** Only on branch `pr4045`; PR #4045 is still OPEN and not merged (`gh pr view`: state OPEN, head 9f774fe3).
  - Recipe: `inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/glm5.2/atom/mi355x-fp4/agentx/disagg-variants.yaml`.
  - Master entry: `configs/amd-master.yaml:1489` (pr4045), key `glm5.2-fp4-mi355x-atom-disagg-agentic-mtp`.
  - Image: `rocm/atom-dev:nightly_202610131249`, which is ATOM 6fd1bb1be per the PR body. Our Infera image pins ATOM 661ef766, so the two are different builds.
- **SGLang GLM-5.2 on MI355X: there is no disagg or PD recipe**, on main or on pr4045.
  - The only entry is single-node aggregated: `glm5.2-fp4-mi355x-sglang-agentic-mtp` (`amd-master.yaml:1415`), image `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260930`.
  - Recipe: `benchmarks/single_node/srt-slurm-recipes/glm5.2/sglang/mi355x-fp4-mtp/agentic.yaml`.
  - The SGLang disagg recipes for GLM-5.2 are NVIDIA-only (b200, gb200, gb300, h200). The one MI355X SGLang disagg recipe is for DSv4 (`dsv4/sglang/mi355x-fp4/agentx/disagg-umbp-dspark.yaml`).
- **ATOM single-node.** `glm5.2-fp4-mi355x-atom-agentic-mtp` (`amd-master.yaml:1468`): TP4 at C2–C10, TP8 at C1–C4.

## 2. Side by side

| Item | ATOM 1P1D (InferenceX #4045) | ATOM kit (Infera PR 209, `origin/main`) | SGLang in InferenceX (single-node agg) | Ours (S1 / S2) |
|---|---|---|---|---|
| Prefill | PP4 × TP1, layers 20/20/20/18, `enforce-eager` (yaml:48,64-65) | same (worker.sh:98-99) | n/a | PP4 TP1, EAGLE |
| Decode | TP4 + DCP4, EP1, colocated on GPUs 4-7 (yaml:181,188; master `ep: 1`) | TP4 + DCP4 (worker.sh:113) | TP4 EP4 + HiCache, or TP8 EP1 (agentic.yaml:57-63,158-161) | S1: TP4 + DPA4, EP1. S2: TP4 EP4 DCP4 |
| MTP | K3 at 1P1D, K4 at 1P2D, K5 at 1P3D (yaml:193,286,307) | K3 if CONC≥48, else K4 (env.sh:11) | EAGLE 5/1/6 (agentic.yaml:33-36) | EAGLE 5/1/6 |
| Mem util | P 0.85, D 0.94 (yaml:66,189) | P 0.85, D 0.94 | 0.85 | S1 0.92, S2 0.85 or 0.88 |
| Max batched tokens | P 8192, D 2048 (yaml:68,191) | same | chunked prefill 32768 | chunk 8192 |
| Max seqs | 512 (yaml:190) | P 512, D 2×CONC (worker.sh:103,114) | max-running = 2×CONC | S1 graph bs 256; S2 96 |
| Decode graph sizes | 1,2,4,8,16…256 in steps of 8 (yaml:192) | 1,2,4,8, then 12…2×CONC in steps of 4 (worker.sh:42-44) | max bs = 2×CONC, up to 32 | S1 256, S2 96 |
| KV dtype | fp8, block 16 (yaml:55-56) | fp8, block 16 | fp8_e4m3 | fp8_e4m3 |
| Index cache dtype | not set; for non-V4 models ATOM 661ef766 keeps the KV-dtype default (`config.py:2613-2620`). Inference: fp8 | same | n/a | n/a |
| Weights | MXFP4 experts + online ptpc_fp8 for the rest; MTP layer 78 kept BF16 (yaml:59) | same | MXFP4 checkpoint as shipped | same |
| CPU tier | Mooncake Store on prefill only: 2 × 768 GiB hugepage owners, one shared pool (yaml:95-158,173-179) | Store, 768 GiB owners (env.sh:21-22) | HiCache 180 GB/rank, write_through | HiCache write_back |
| Router | atomesh, round_robin (yaml:17-18) | Rust router | sglang | Infera |
| Concurrency | 56, 72, 96, 112, 128 (master:1507) | 56, 72, 96, 112 | 4–16 (TP4), 1–4 (TP8) | C80 |
| Window | 3600 s default (`infx/matrix/validation.py:20`; `run-sweep.yml:396`) | 3600 s | 3600 s | 1800 s |

## 3. How tok/s/GPU is computed (first-hand)

- Throughput is total tokens over the measured span: `total_tput_tps = (Σinput + Σoutput) / (max request_end − min request_start)` (`infx/results/agentic/request_metrics.py:229-240`).
- Input counts every input token per request, so cache-hit tokens are included.
- The divisor is `num_gpus = prefill GPUs + decode GPUs` (`__init__.py:122-124,232-236`), which is 8 for 1P1D. Our definition matches this.
- The two published ATOM sources use different metrics:
  - PR #4045's body reports **total** tok/s/GPU: C96 = 42,884.
  - PR 209's comparison table reports **input-only** tok/s/GPU (`pr209_body.md:85-92`): #4045 C96 = 42,523; PR 209 C96 = 46,234, with output adding 387.3 on top.
  - The two #4045 numbers differ by about the output rate. Output is about 1% of the total, so the effect is small, but the columns are not the same metric.

## 4. Published results at matching concurrency

- **ATOM, #4045 1P1D (sweep 39028510364, PR body), tok/s/GPU total:**

  | CONC | 56 | 72 | 96 | 112 | 128 |
  |---|---|---|---|---|---|
  | tok/s/GPU | 31,067 | 38,487 | 42,884 | 39,245 | 39,933 |

  TTFT p50 at C112 is 14.78 s.
- **ATOM, PR 209 (input-only):** C56 32,925; C72 40,075; C96 46,234; C112 48,154.
- **SGLang MI355X GLM-5.2:** no published numbers in the repo, in PR #2488 or in PR #2570; I found none at C≥12. To settle this, pull the run artifacts or the dashboard for `glm5.2-fp4-mi355x-sglang-agentic-mtp`.
- There is no SGLang PD point that matches any ATOM PD concurrency.

## 5. How acceptance is forced (first-hand)

- **One source of truth.** The golden AL comes from `infx/golden_al_distribution/glm5.2_mtp.yaml:10-19`: K3 = 2.99, K5 = 3.61. Those values were measured on B300 vLLM, glm-5.2-fp8, SPEED-Bench coding, thinking on.
  - The launcher injects the value per role (`infx/srt_slurm/synthetic_acceptance.py:149-234`). Recipes must not hard-code it (`golden_al_distribution/README.md:152`).
  - ATOM gets `--spec-decode-acceptance-length` (line 215-219); the prefill role gets it too.
  - SGLang gets `SIMULATE_ACC_LEN` with `METHOD=match-expected` and `TOKEN_MODE=real-draft-token` (lines 221-230).
  - Our kit sets the same three SGLang variables (`llkit/kit/engine/worker.sh:107-108`).
- **The averages match.** Both engines target the same minimum-variance floor/ceil distribution:
  - ATOM: `rejection_sampler.py:52-72` @661ef766. Its docstring says it matches SGLang's `match-expected`.
  - SGLang: `spec_utils.py:420-436`.
  - Both emit the real draft tokens.
- **The sampling granularity differs:**
  - **SGLang** draws one length per verify step and applies it to the whole batch (`spec_utils.py:457-470`: one scalar, `sim_accept_index[:, :len]`).
  - **ATOM** draws a uniform random value per token and decides per request (`rejection_sampler.py:238-241`, `torch.rand(num_tokens)`, kernel grid `(batch_size,)`), seeded per step.
- **The depth differs.** ATOM 1P1D runs K3 at AL 2.99 (4 tokens verified per request). Our SGLang runs K5 at AL 3.61 (6 tokens verified per request).
  - So "forced acceptance" is comparable in method, but the two runs differ in MTP depth and target AL.
  - The ATOM vs SGLang gap therefore mixes engine speed with the choice of K.
  - Whether the batch-level vs per-request draw affects step time or tail latency is unknown. A same-engine A/B, or SGLang at K3/2.99 under identical load, would settle it.

## 6. Other differences that remain open

- The MoE layout differs (first-hand): ATOM decode is EP1, MoE over TP4; our S2 is EP4.
- The prefill tier differs (first-hand): ATOM uses a DRAM store of about 1.5 TiB on RDMA with asynchronous loads; we use HiCache.
- Unknown which of these, if any, explains the throughput gap. Measuring it would need per-step decode time at matched batch size and context, plus prefill time per token, for each engine.
- The PR #4045 body itself leaves a gap unexplained: InferenceX measured 39–40K at C112, while ATOM's own CI reports 46.1K on the same Store size.

Files: `/var/tmp/yihou/src/InferenceX` (branch `pr4045`), `/home/yihou/dev/git/infera.glm52.pd` (`origin/main:examples/atom_4p4d_glm5.2/`), `/home/yihou/dev/git.16-19/sglang.pd.pp.dcp/yihou_ws/glm52_dcp_pd_20261021/atom_cmp/pr209_body.md`.

# READER atom_decode

**ATOM 661ef766 vs SGLang DCP4: GLM-5.2 decode path, read from the code**

Abbreviations for ATOM files at 661ef766: dsv2 = `atom/models/deepseek_v2.py`, mla = `atom/model_ops/attention_mla.py`, ops = `atom/model_ops/dcp_ops.py`, amla = `atom/model_ops/attentions/aiter_mla.py`. "KD" means our `design/kernel_diff_dpa_vs_dcp.md`. Read-only copies of the ATOM files are in `/var/tmp/yihou/atom_read/`.

### 1. Launch config (first-hand)

| | ATOM decode (PR209 `worker.sh`; also InferenceX pr4045 `disagg-variants.yaml:61,188-193`) | SGLang S2 (`llkit/kit/engine/worker.sh:117-122`) |
|---|---|---|
| Parallel layout | `-tp 4 --decode-context-parallel-size 4`, no `--enable-expert-parallel`, so the MoE runs TP4 | TP4, EP4, DCP4 |
| Precision | Online `ptpc_fp8` on every non-expert linear (q_a/kv_a, q_b, kv_b, o_proj); experts stay MXFP4 | q_b, kv_b and absorb in bf16 (`KEEP_KV_B_BF16`) |
| Spec decode | MTP K3, so each request sends 4 verify rows | EAGLE 5/1/6, so 6 verify rows |
| CUDA graph | `--cudagraph-mode` defaults to FULL (`arg_utils.py:335-343`): whole forward captured per (bs, q_len) (`model_runner.py:3970-4120`). Sizes 1,2,4,8, then every 4 up to 2×CONC | max bs 96 |
| `--max-num-batched-tokens 2048` | Per-step token budget, charging K+1 tokens per sequence (`scheduler.py:2343`). Capture sizes above `mbt // (K+1)` = 512 are dropped (`model_runner.py:3998-4045`). The PR body says it also freed KV memory (~160K to 198.5K blocks) | — |

### 2. Indexer and top-k under DCP: where ATOM does less work

**Index-K cache is sharded, not replicated (first-hand).**
- ATOM gives slot -1 to tokens a rank does not own (`amla:2515-2525`). That slot mapping drives the indexer K write (`dsv2:1675-1731`).
- The index cache is a region of each rank's own paged pool (`amla:1300-1330`), and blocks are counted per rank (`model_runner.py:2031`).
- Shared (IndexShare) layers get no index slice (`amla:175-210`). SGLang also skips shared layers (`memory_pool.py:5395-5396`).
- SGLang with `dcp_replicated` sizes `index_buf_size = size` (`memory_pool.py:5343-5352`), i.e. the index K is written on every rank.

**Decode scoring (first-hand), `ops:1147-1307`.**
- Each rank scores only its local shard with `deepgemm_fp8_paged_mqa_logits`, using a local context length (`ops:1085-1121`).
- It then runs a local top-2048 that also returns the scores, and all-gathers **only the fp32 scores** `[rows, 2048]`, not score+gid.
- aiter `flydsl_dcp_topk_merge` emits this rank's owned, localized, compacted KV slots and indptr directly. There is no step that rebuilds the global top-k and then filters it.
- PR209 moves that all-gather from pynccl to aiter's IPC all-gather: 54 to 31 µs (PR body).

**SGLang split indexer** (`dsa_dcp_split_indexer.py:46-89`; `dsa_indexer.py:1585-1650`; KD rows "indexer top-k", "candidate all-gather"):
- It packs `[2, rows, topk]` (score + position), so twice ATOM's payload.
- It sends that through `group.all_gather`, which shows up as NCCL in our trace (~4.85 ms/step eager).
- The merge reconstructs global positions, then `transform_index_page_table_dcp` runs separately.
- KD measured `coop_topk` 45.5 + merge 46.2 µs per call.

**Prep (first-hand).**
- ATOM fuses `indexer.wk` and `weights_proj` into one GEMM (`dsv2:2087`, `:3397-3401`).
- It does q/k rope, quant, cache write and the weights scale in one `indexer_qk_rope_quant_and_cache` (`dsv2:1702-1722`).
- KD found SGLang DCP4 running 3 GEMMs plus about 6 separate prep kernels, because the fused gfx950 path is capped at 48 rows.
- I did not find a 48-row cap in ATOM's DCP decode path (`ops:1202-1307`).

**MTP index sharing across draft steps:** disabled under DCP in ATOM (`eagle_proposer.py:94-105`).

### 3. Attention

**QREP (first-hand).**
- q_b is built with effective TP = tp/dcp (`dsv2:2521`, `:2588-2596`; `mla:424-445`), so decode skips the Q all-gather (`mla:1399-1422`).
- The absorb uses `W_K_qrep` through an fp8 Triton bmm (`mla:1473-1514`).
- QREP turns itself off only for MXFP4 BMM (`config.py:1758-1769`).
- SGLang also replicates q (auto for a2a, KD row 6), but in bf16 through hipBLASLt.

**Rope / cat / KV write (first-hand).**
- ATOM does all of it in one `fused_qk_rope_concat_and_cache_mla` with `compute_all_q_rope` under DCP (`mla:3138-3160`). It does not exclude DCP.
- SGLang gates its fused kernel on `dcp_size==1` (`forward_mla_rocm.py:447`), so it falls back to bmm + copy + rope + KV write (KD ~+3.0 ms eager).

**Attention kernel (first-hand).** aiter asm `mla_decode_fwd` in persistent mode, fp8 Q and KV, page size 1, sparse indptr from the indexer's compacted buffer (`mla:2650-2800`). SGLang uses the Triton sparse-MLA kernel.

**Project-before-merge, PBM (first-hand).**
- W_V (fp8 bmm) is applied over the whole group's head set before the merge (`mla:826-836`, `:1337-1373`).
- The a2a therefore carries v_head_dim (256) per head instead of kv_lora_rank (512).
- Inference: SGLang's a2a merges latent outputs and runs v-absorb after (KD "v absorb" row), so its payload per head would be 2×. I have not measured this.
- PR209 adds an IPC `alltoall_vec` (56 to 34 µs) and a combine that merges, then quantizes to per-token FP8 for o_proj (`mla:1302-1335`). SGLang merges through NCCL `all_to_all` (KD: 169.8 µs/call eager, including wait).

**o_proj:** RowParallel with fp8 per-token input. ATOM enables AR+RMSNorm fusion by default (`envs.py:491`). SGLang S2 runs bf16 hgemm with AR fusion on.

### 4. MoE and all-reduce

- **MoE:** TP4 MXFP4 through aiter (non-Triton by default, `envs.py:265`), with the shared expert fused (`envs.py:789`). The exact kernel is **unknown**; a decode trace would settle it. SGLang EP4 hit an untuned FlyDSL shape (KD #3).
- **INT4 quick-reduce:** the kit sets `AITER_QUICK_REDUCE_QUANTIZATION=INT4`. Whether aiter routes the decode all-reduce sizes to quick-reduce is **unknown**; check aiter 9205e1059's dispatch or a trace.

### 5. MTP

- Draft step 0 is not a captured draft graph (`eagle_proposer.py:116-124`, `:760`). Steps 1..K-1 replay a `DraftGraph`.
- Whether step 0 runs inside any capture under FULL mode is unknown; a trace would settle it.

### 6. Where ATOM avoids work SGLang does (from the code; not timed in ATOM)

1. Index-K is written by one rank only, at 1/4 the memory; SGLang writes it on all 4.
2. The top-k exchange is half the payload, over IPC instead of NCCL, and the merge emits owned slots with no global-rebuild-then-filter.
3. No row cap on the fused indexer prep.
4. One fused rope+cat+KV-write under DCP.
5. fp8 q_b, absorb, v-up and o_proj, versus bf16 in our DCP4.
6. The a2a merge carries 256-wide data per head, over IPC, with the o_proj quant fused in.
7. 4 verify rows per request versus our 6. Inference: at forced AL 2.99 vs simulated 3.61 that is 0.75 vs 0.60 accepted tokens per verify row.

### Unknown, and what would settle it

- ATOM per-kernel times at our shape (B40, ~101K context): an ATOM decode torch trace, set against KD's tables.
- Whether quick-reduce INT4 is actually used.
- The ATOM MoE kernel choice.
- Accuracy: ptpc_fp8 online quantization of attention and INT4 AR, versus our bf16. PR209 reports GSM8K 0.975 on 200 questions; there is no same-harness SGLang number to compare.

# READER atom_pd_prefill

# ATOM vs SGLang on the PD side: prefill, KV transfer, decode prefix reuse

## Answer to the key question (code evidence on both sides)

Yes. ATOM's decode reuses prefix KV it already holds. Our SGLang runs did not, even though SGLang has the same mechanism behind a flag we never set.

**ATOM** (661ef766):
- The decode is started with `--enable_prefix_caching` (kit `engine/worker.sh:148`, shared by both legs).
- At admission the consumer probes its local cache with `block_manager.can_allocate(...)` and then allocates the hit blocks (`atom/model_engine/scheduler.py:2004-2050`).
- The Mooncake consumer turns the local hit into `num_computed_blocks = seq.num_cached_tokens // hash_block_size` ("PD incremental: skip leading blocks already in the decode node's prefix cache", `mooncake_connector.py:539-562`). It slices `dst_block_ids = local[off:]` and `src = remote[off*skip_factor:]`, where `skip_factor = dcp_size // remote_dcp_size` (`:1518-1528`). The producer applies the same offset (`:2001-2007`).
- A request with per-request state falls back to a full transfer (`:559`).
- The prefill still computes the whole prompt from its own GPU or Store cache. Only the transfer and the decode KV allocation are reduced.
- Published "decode hit %" (README table, CI client, 3600 s): 92.3 / 84.1 / 52.9 / 25.1 at C56 / 72 / 96 / 112.
- The README attributes the C96+ drop to C × prompt length outgrowing the decode KV pool. That is ATOM's own statement; I have not verified it.

**SGLang**:
- `--disaggregation-decode-enable-radix-cache` exists and defaults to `False` (`arg_groups/fields/disagg.py:85-88`).
- When on, decode matches against its radix tree and locks the prefix (`disaggregation/decode.py:1477-1490`, `_match_prefix_and_lock` at `:840`).
- It then sends `decode_prefix_len`, which the prefill uses as `start_send_idx` (`prefill.py:433-444`).
- The DCP plan already supports a nonzero `decode_prefix_len`, aligned to the virtual page `page × dcp` (`disaggregation/common/utils.py:150-197`).
- The flag is compatible with EAGLE (`speculative_hook.py:162-171`). It is rejected only with HiSparse or the fake backend (`pd_disaggregation_hook.py:98-108`).
- Our kit never passes it (`llkit/kit/engine/worker.sh:58-125`).
- Both decode server-arg dumps show `disaggregation_decode_enable_radix_cache': False` (`design/tmp_kd/dcp4.decode.head.log`, `dpa4.decode.head.log`).
- `design/pd_vs_mixed_decode_diff.md:31` records that every turn re-sends the full prefix KV, with a mean of 4.8 transfers in flight.

**Unknown:**
- Whether SGLang's decode radix path works with DSA + DCP4 + EAGLE on ROCm (index-key pages, draft KV, alignment). I found no code that rejects this combination, and no run has used it.
- To settle it: one smoke run plus AgentX with the flag on DCP4 and DPA4 decode, recording the decode `cached_tokens` per request and transfer bytes.

## Prefill configuration

| | ATOM kit (PR #209) | Our SGLang kit |
|---|---|---|
| Parallelism | TP1 × PP4, layers 20,20,20,18, `--enforce-eager` (`worker.sh:96-99`) | TP1 × PP4, same partition, `--disable-overlap-schedule` (`worker.sh:85-90`) |
| Chunk | 8192, max-num-seqs 512 (`env.sh` `PREFILL_MAX_BATCHED_TOKENS`, `worker.sh:103`) | `PP4_CHUNK`, default 16384 (S1/S2 used 8192), shortest-prefill-first |
| Memory | 0.85 | 0.88 |
| GPU prefix cache | yes | yes (radix) |
| CPU tier | Mooncake Store: one master and 2 × 768 GiB owners shared by the 4 stages over RDMA, 1 load worker, `OFFLOAD_MIN_LOAD_TOKENS=0` (`worker.sh:61-70`) | HiCache L2: 250 GB per rank, ratio 1.5, write_back, kernel I/O, page_first (`worker.sh:98-101`) |

- README states that Store loads do not block the prefill, while LMCache loads synchronously and took about half of each stage's time at C96/C112: 30.7K / 23.3K with LMCache vs 46.2K / 48.2K with the Store. These are published numbers.
- Whether our HiCache L2 loads block the PP stages is unknown. A per-stage timeline of load time vs forward time would settle it.
- ATOM also coalesces in-flight prefixes: a request whose prefix is being computed waits instead of recomputing it (`scheduler.py:768-790, 2024-2035`). I did not check whether SGLang has an equivalent.

## KV transfer layout (DCP striping, index keys)

**ATOM:**
- Each PP stage sends its own layers to every decode DCP rank (README "The recipe").
- The producer gathers the tokens a rank owns into a staging slot (`ATOM_PD_MLA_STAGING`). It RDMA-writes one descriptor per slot of a decode-side landing pool, and the decode scatters it into its paged KV (`mooncake/mla_landing.py:1-40`).
- The docstring of `mla_landing.py` says the per-page staged path caps a NIC at about 10 GB/s with fragmented decode block tables.
- DSA index keys get their own GPU gather onto the DCP shard plan (`index_staging.py` header and `prepare_dcp_index_gather_indices`), so index keys are sharded per DCP rank as well.
- Mooncake settings:
  - `MC_NUM_QP_PER_EP=1`, 16 send workers per stage (default). README gotcha 13: two QPs with 16 workers hung an RDMA write.
  - `ATOM_MOONCAKE_MATCHED_RAILS=auto` and `MC_ENABLE_DEST_DEVICE_AFFINITY=1` (`worker.sh:28-57`).
  - Block size 16 on both legs.
  - The consumer sends a `write_request` to every stage (`mooncake_connector.py:1509-1555`).

**SGLang:**
- `build_dcp_token_transfer_plan` sends the target rows of `(dcp_rank - chunk_start) % dcp_size` stepping by `dcp`, and sends all draft rows unless the draft is sharded (`common/utils.py:183-197`; `mooncake/conn.py:1436-1445`).
- Source rows are packed via `try_pack_dcp_src` (`conn.py:1456-1462`).
- Transfer queues are sharded by the sum of destination session ports (`conn.py:3112-3117`).
- Env: `MC_ENABLE_DEST_DEVICE_AFFINITY=1`, `MC_ENABLE_DEST_LOCAL_RAIL=1`, `MC_IB_TIMEOUT=18`, `MC_MAX_MR_SIZE=1 GiB` (kit `worker.sh:44-50`).
- Not read in this pass: how the DSA index-key buffer enters the SGLang DCP plan, and the QP count and worker count. Both are unknown.

## Router

- **ATOM:** `--router-policy round-robin`, with `--session-affinity` (default prefill). The reference numbers used `SESSION_AFFINITY=off` and `--breaker-failure-threshold 0` (`up.sh:77-81`, README "Reference numbers").
- **Ours:** kv-aware routing (prefill overlap weight 20, decode overlap weight 2), session affinity on prefill, `--pd-prefill-guard-release completion`, `--decode-input-demand on` (`llkit/kit/engine/up.sh:50-51`).
- With 1 prefill and 1 decode, ATOM's README says affinity has nothing to choose (gotcha 8). Policy differences between the two routers should matter little in this topology. That is inference, not measured.

## Decode admission

- **ATOM:** `--max-num-seqs 2×CONC` and CUDA graphs at 1, 2, 4, 8, then every 4 up to 2×CONC (`worker.sh:42-44, 113-114`). Memory 0.94, max-num-batched-tokens 2048, and `AITER_REUSE_IDENTICAL_COMM_GROUPS=1`; README says this gives about 198.5K blocks vs about 160K. Admission stops when `running + parked_remote ≥ max_num_seqs` (`scheduler.py:1992-1997`). KV is allocated before the remote load and the request is parked as `WAITING_FOR_REMOTE_KVS` (`:3102-3115`).
- **Ours, S2:** max-running 96, graph bs 96, memory 0.85.
- **Ours, S1:** DPA4 with max-running 256 and memory 0.92.

## TTFT (published / measured, not comparable clients)

| Run | Concurrency | TTFT mean |
|---|---|---|
| ATOM | C56 | 1.12 s |
| ATOM | C72 | 1.50 s |
| ATOM | C96 | 2.68 s |
| ATOM | C112 | 5.05 s |
| Ours S1 | C80 | 5.5 s |
| Ours S2 | C80 | 17.7 s |

The clients differ: ATOM used InferenceX CI client `288471b3` for 3600 s with forced AL 2.99; ours used `918524ff` for 1800 s with simulated AL 3.61. How much of the TTFT gap comes from the missing decode prefix reuse (full-context transfer every turn) is unknown. A same-client A/B with only `--disaggregation-decode-enable-radix-cache` toggled would measure it, recording TTFT, transfer bytes per request and decode cache hit.

## Absorbable items (from code read, not predicted gains)

1. Enable decode radix cache: flag exists, untested on DSA+DCP.
2. Decode memory: ATOM runs 0.94 with shared communicator groups.
3. A non-blocking CPU tier on the prefill, which ATOM credits for its C96/C112 numbers.
4. Landing-pool packed RDMA for DCP striping, in place of per-page descriptors.
5. In-flight prefix coalescing on the prefill.

Scratch copies of the ATOM README, connector and scheduler are in `/var/tmp/yihou/` (`atom_readme.md`, `atom_mc.py`, `atom_sched.py`).

# READER measured_gap

**GLM-5.2 decode step-time comparison: SGLang (S1 DPA4, S2 DCP4) vs ATOM PR #209**

The comparison is only partly apples-to-apples. Context per row is unknown on our side and the speculative settings differ (ours K5/AL 3.61, ATOM K3/AL 2.99). On the numbers we have:
- **Per decode step, ATOM is faster:** 748 output tok/s per decode GPU vs 543 (S1) and 461–505 (S2).
- **End to end, S1's per-token latency is close to ATOM's:** S1 TPOT 18.3 ms vs ATOM ITL p50 16.75–19.88 ms.
- **The large measured end-to-end differences are elsewhere:** TTFT (S1 5.5 s vs ATOM 1.5–2.7 s), decode prefix hit, and requests per second.

### 1. Sources (first-hand)

**ATOM** (`atom_cmp/pr209_body.md`):
- Step time: 42.77 → ~40 ms at B40, ~101K context (`:24`).
- Per-GPU table (`:87-92`): C56 / C72 / C96 / C112 = input 32,925 / 40,075 / 46,234 / 48,154 and output 246.2 / 321.9 / 387.3 / 406.0 tok/s/GPU. Both are over 8 GPUs, with a 3600 s window and the CI client.
- ITL p50 14.17 / 16.75 / 19.88 / 21.21 ms; decode prefix hit 92.3 / 84.1 / 52.9 / 25.1 % (Infera `origin/main:examples/atom_4p4d_glm5.2/README.md:158-163`).
- Layout: decode TP4 + DCP4 on GPUs 4-7, prefill TP1×PP4 (`README.md:4-5, 78`).
- The headline 48.2K equals the "input" column, so ATOM's headline appears to count input tokens only.

**SGLang** (S1/S2 at C80, 1800 s):
- Pack-up `glm52.llying-dcp-s1s3.packup_20261022/results/{S1-219…,S2-039…}/agentx/agentx_conc80.json`.
- Total / input / output tok/s:
  - S1: 276,074 / 273,723 / 2,351.8
  - S2: 207,525 / 205,778 / 1,747.6
- Mean ISL 111.1K / 112.0K; mean OSL 955 / 951.
- Step p50 by running bucket (`analysis.txt`):
  - S1, per DP rank: 1-8 → 42.8, 9-16 → 67.3, 17-24 → 72.4 ms; all steps 65.8 ms. Mean running is 9.9 per DP rank (about 40 node-wide).
  - S2: 1-8 / 9-16 / 17-24 / 25-32 / 33-40 / 41-64 → 28.4 / 44.6 / 52.6 / 63.1 / 71.5 / 78.9 ms; mean running 30.8.
- The fixed-N A/B (graph mode, ~180K context, mixed rather than PD) is in section 2 (`poll.md:509`).

### 2. Step level: output tokens per step-second per decode GPU (4 decode GPUs)

Formula: batch × AL / step / 4.

| arm | batch | context/row | step ms | AL | tok/s per decode GPU | ms per token per request |
|---|---|---|---|---|---|---|
| ATOM post-patch | 40 | ~101K | ~40 | 2.99 | 748 | 13.4 |
| ATOM pre-patch | 40 | ~101K | 42.77 | 2.99 | 699 | 14.3 |
| S1 DPA4 (all steps p50, mean 39.6 running) | ~40 | unknown | 65.8 | 3.61 | 543 | 18.2 |
| S2 DCP4, bucket 33-40 | 36.5–40 | unknown | 71.5 | 3.61 | 461–505 | 19.8 |
| A/B DPA4, N=24 | 24 | ~180K | 41.9 | ~3.6 | 514 (measured 2,056 / 4) | 11.6 |
| A/B DCP4, N=24 | 24 | ~180K | 55.5 | ~3.6 | 396 (1,584 / 4) | 15.4 |

How far each row can be compared:
- **Batch:** S1 and S2 are matched to B40 only by mean or bucket. The DPA4 step is DP-synchronised, so a per-rank bucket is not a node-level batch.
- **Context:** S1/S2 per-row context is unknown. Earlier PD runs at C40 had ~180K per row (`ledger.benchrun.md:254-257`). How fast step time grows with context is measured only for DCP4: +13 % from 34K to 131K at N=24 (`RESULTS.md:86`).
- **Verify rows and draft passes (not apples-to-apples):**
  - ATOM K3 verifies 4 rows per request per step, i.e. 160 rows at B40; this matches the PR's "160 rows" and M 144–240 tuning.
  - SGLang EAGLE 5/1/6 verifies 6 rows per request, i.e. 240 rows at B40, and runs 5 draft passes instead of 3.
  - Tokens per verify row: 0.75 for ATOM vs 0.60 for SGLang.
  - Both AL values are forced, not real acceptance. ATOM's own CI pairs K5 with AL 3.61 and K3 with AL 2.99 (ATOM 661ef766 `.github/benchmark/models_atomesh.yaml:277,332,571,646`), so the AL scales look the same. Whether ATOM's AL includes the bonus token was not read in code: unknown.
- **Ratio:** ATOM / S2 is 1.48–1.62× and ATOM / S1 is 1.38× at nominal B40. Context is unmatched in both.

### 3. End to end

| | output tok/s per decode GPU | input tok/s/GPU (/8) | requests/s | TTFT mean (s) | TPOT or ITL (ms) | input/output ratio |
|---|---|---|---|---|---|---|
| S1 C80 | 588 | 34,215 | 2.46 | 5.50 | TPOT 18.31 (p50 19.13) | 116.4 |
| S2 C80 | 437 | 25,722 | 1.84 | 17.69 | 20.32 | 117.7 |
| ATOM C72 | 644 | 40,075 | 2.63 | 1.50 | ITL p50 16.75 | 124.5 |
| ATOM C96 | 775 | 46,234 | 3.14 | 2.68 | ITL p50 19.88 | 119.4 |
| ATOM C80, linear interpolation (my inference) | ~688 | ~42.1K | — | — | — | — |

What I infer from this (assumes ATOM's ITL measures the same thing as our TPOT; our ITL mean equals TPOT mean, `analysis.txt:5-6`):
- By Little's law (requests in decode ≈ output rate × per-token time), S1 has 2351.8 × 0.01831 ≈ 43 requests decoding; we measured 39.6.
- ATOM C72 also gives ≈ 43 (2575 × 0.01675), and C96 gives ≈ 62.
- So at C72–C80 the two have a similar number of requests decoding and similar per-token latency.
- The S1 gap of about −19 % against the interpolated C80 sits in requests/s and TTFT, not in TPOT.
- The step-level gap in §2 does not show up end to end for S1. Why is unknown.

### 4. What does not compare directly

- **Concurrency:** ours is C80 only; ATOM has no C80 point. The interpolation is linear and is my assumption.
- **Window:** 1800 s vs 3600 s. ATOM also used the CI client (InferenceX 288471b3), not our 918524ff (`README.md:150-153`). The input/output ratio differs (116–118 vs 119–134), so the workload mix differs.
- **Token accounting:** our 34,509 is total tokens / 8; ATOM's headline is input / 8. That is under 1 % apart, but use input vs input (S1 34,215).
- **Decode prefix cache:** ATOM's decode has one (84 % hit at C72). Our PD decode radix cache is off (`design/perf_knobs.md:12`). The effect on KV transfer and admission is not measured.
- **Prefill cache hit:** ATOM reports 94.4 % total at C72. Our server metric reports 63.7 % (S1) and 57.5 % (S2), with 96 % theoretical. The definitions are not shown to be the same, and an earlier run measured about 3.4 % new tokens (`NEXT_SESSION.md:41`).
- **Decode KV pool:** ATOM ~12.7M tokens at 0.94 (`pr209_body.md:38`) vs S2 8.87M at 0.85 and S2n 9.46M at 0.88 (`gmu/RESULTS.md:111`).

### 5. Largest unexplained gaps and the measurement that settles each

1. **S1 TTFT 5.5 s vs ATOM 1.5–2.7 s, and 2.46 vs 2.63–3.14 requests/s.** Same PP4 TP1 prefill layout, but ATOM runs eager PP. Settle with a prefill-side breakdown for S1: queue, compute, KV-send and new tokens per request, from the prefill logs over the S1 window.
2. **Step 65.8–71.5 ms vs ~40 ms at ~B40.** Two parts:
   - Context is unknown on our side. S2 ran with `SGLANG_LOG_DECODE_SEQ_LENS=1` (`engine-config.env:36`); read context per row from its decode log.
   - K5 vs K3 is unknown. Run an SGLang decodebench at N=40, ~101K context, EAGLE 3/1/4 with SIMULATE_ACC_LEN 2.99, against 5/1/6 with 3.61.
3. **Prefill cache-hit gap, 57–64 % vs 94 %.** Check whether the hit metrics share a definition: compute hits from prefill logs, cached tokens / prompt tokens.
4. **Decode prefix hit on ATOM vs radix off on ours.** Measure KV-transfer bytes per request and S2's admission queue (mean 12.8, max 56) with and without it.
5. **Whether ATOM ITL equals our TPOT.** Read the CI client 288471b3 metric code, or rescore one ATOM JSON with our analysis tool.
6. **S2 vs S1, −25 % end to end.** Admission backs up at running 30.8 against a cap of 96 (`ledger.pdrun.md:493-494`). Cause not measured.