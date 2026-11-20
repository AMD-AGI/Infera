# G3: DSv4.1 per-request state across a P->D handoff (MooncakeConnector, vLLM main)

Date 2026-11-19. Code read only; no GPU, no run. `V` = `source/vllm` @ 7a62ca0666. `MC` =
`V/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`. `HF` =
`source/transformers_pr50919/modeling_deepseek_v41.py`. Tags: [FH] read in code, [INF] inferred from read code,
[OPEN] not established.

## 0. Verdicts

1. **Unit of transfer is the KV-cache group, all groups by default.** Every group with `enable_kv_transfer` (default
   True, only HiSparse sets False) is a transfer group [FH: V/vllm/v1/kv_cache_interface.py:708, 890-896, 916-920;
   V/vllm/v1/hisparse/layout.py:291]. Mooncake registers every layer's cache under its group, including
   `CircularBufferManager` groups (excluded only from the packed-row path, not from registration) [FH: MC:2380-2442,
   2491-2532]. Mooncake declares it restores all transfer groups [FH: MC:869-870].
2. **Transferred (happy path, P and D configured alike):** compressed KV of kv-source layers 2/8/14/20, indexer K of
   the same sources, SWA windows of all target + draft layers (clipped to the last `cdiv(128,32)+1 = 5` blocks), the
   compressor ring block of each ratio-2 source (2/8/14), and DSpark draft-layer context KV [FH + INF, s1].
3. **Recomputed on D:** the last prompt token (P prefills N-1, D loads N-1 and recomputes token N-1), its SWA/compressed/
   indexer/ring rows, its DSpark context KV and target aux hidden states, Engram n-gram lookback (from D's own token
   ids), and the layer-20 candidate block buffer (per-batch scratch) [FH, s2].
4. **Lost:** no per-request model state was found that is neither transferred nor recomputed when both pools share the
   same speculative config and `swa_bounded_replay` setting [INF; not tested]. Hazards are config mismatches and
   partial-block bytes (s4).
5. **HF reference parity:** each of `DeepseekV41CacheLayer`'s five extra states has a vLLM counterpart that is either
   transferred or recomputed (s3). HF has no DSpark state.
6. **D prefix caching changes the pull only for prefix-cacheable groups** (compressed KV, indexer K): D pulls only the
   unhashed suffix. SWA (under bounded replay) and the ring are not prefix-cacheable, so they are pulled every time
   [FH: MC:1241-1253, 2154-2156; single_type_kv_cache_manager.py:1124-1127, 1644-1646].
7. **Recipe:** "With Speculative decoding on, DSpark runs in both pools so the transferred KV stays compatible"
   [FH: V/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml:2085-2086]. Code agrees that group sets must match
   (s4 H1); `prefill_context_kv_only` exists for exactly this P-side role [FH: V/vllm/config/speculative.py:683-690].

## 1. Per-state table

| State | vLLM owner / manager | P->D | Evidence |
|---|---|---|---|
| Main compressed KV (sources 2/8/14/20; consumers read the source cache) | `FullAttentionManager`, MLA spec, `tokens_per_state=ratio` | transferred; with D prefix hit only the unhashed suffix | [FH: V/vllm/models/deepseek_v41/attention.py:1290-1299, 1305-1312; MC:1241-1253, 2154-2156] |
| Indexer K (owned by kv sources; 24/28/32/36 share layer 20's) | `FullAttentionManager` | transferred, suffix only on D hit | [FH: attention.py:537-563, 1405] |
| SWA window, 40 target + 3 DSpark layers | `SlidingWindowManager(sliding_window=128, bounded_replay=...)` | transferred, last 5 blocks (160 tok >= reach 127) on both P and D sides | [FH: V/vllm/v1/attention/backends/mla/sparse_swa.py:146-147; MC:1075-1105, 1248, 1371; single_type_kv_cache_manager.py:1135-1139] |
| Compressor ring (open ratio-2 group rows, fp32 `[kv, score]`) on 2/8/14 | `CircularBufferManager`, 1 block/request, never hashed | transferred (always "unhashed", `blocks_per_sw=0` so not clipped) | [FH: V/vllm/models/deepseek_v41/compressor.py:237-270, 339-345; single_type_kv_cache_manager.py:1630-1700; V/vllm/v1/core/kv_cache_manager.py:113-123] |
| Ratio-1 compressor (layers 20-39) | no state cache | nothing to move | [FH: compressor.py:339-345] |
| Last prompt token N-1 | - | P truncates prompt to N-1 (`max_tokens=1`), D loads N-1 and recomputes N-1 | [FH: MC:1107-1114, 1116-1147, 1149-1156, 1210-1215] |
| Engram token history | `req_states.all_token_ids` gathered into `lookback_token_ids` each step | recomputed on D from prompt ids | [FH: V/vllm/models/deepseek_v41/nvidia/model_state.py:340-366 (used by AMD model: amd/model.py:1559-1561, 1563-1567); common/engram.py:10-17] |
| Engram tables | weights | n/a | - |
| Layer-20 candidate block buffer | `[max_num_batched_tokens, topk_blocks]` scratch, rebuilt each forward | recomputed | [FH: V/vllm/models/deepseek_v41/amd/model.py:654-666] |
| DSpark draft context KV (3 draft layers' SWA) | draft layers are `DeepseekV4DecoderLayer` in the KV config; context KV written from projected target hidden | transferred for positions < N-1 if P runs DSpark; position N-1 recomputed on D | [FH: amd/dspark.py:113-123, 150-175; V/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:298-365] |
| DSpark target aux hidden states for the first decode step | per-step buffer `hidden_states[max_num_tokens]`, no per-request persistence | recomputed from D's forward on N-1 | [FH: spec_decode/dspark/speculator.py:69-77; dflash/speculator.py:490-560] |
| P's sampled token (from N-1 prompt) | output | discarded by design [INF from truncation path] | [FH: MC:1116-1147] |

## 2. Bounded replay at the boundary

- D asks for `N-1 - num_computed_tokens` tokens when `_bounded_replay` (any group with `prefix_replay_tokens > 0`) [FH:
  MC:1055-1057, 1107-1114, 1210-1215]. P drops its last prompt token, guarded by `_p_side_truncated` [FH: MC:1116-1147].
- After the pull, `window_loaded` is True for Mooncake (all replay groups are in its loaded groups), so the scheduler does
  not rewind by the 128-token window and does not round the external hit down to a block [FH: V/vllm/v1/core/sched/
  scheduler.py:1185-1196, 1255-1270, 3615-3622, 3624-3643, 3680-3693]. D then computes only token N-1.
- Ring correctness for the recomputed token: ring slot is `block * capacity + pos % capacity` [FH: compressor.py:62-63,
  74-97]; capacity = `max(8, next_pow2(num_speculative_tokens + 2))` [FH: compressor.py:252-254]. P's ring after N-1
  tokens holds the open row at N-2 if N-1 is odd-length; D's recompute of N-1 reads it from the transferred ring
  ("ordinary decode only needs the latest open row") [FH: compressor.py:129-132]. Kernel-level read of that row at
  the first D step: [OPEN] (the ring read path in the fused compressor kernel was not traced).
- HF equivalent: `update_pooling_buffer` keeps the open-group tail [FH: HF:91-103, 335].

## 3. HF `DeepseekV41CacheLayer` vs vLLM (HF:50-139)

| HF state | vLLM counterpart | P->D |
|---|---|---|
| sliding window, keeps `sliding_window-1` keys (HF:73-85) | SWA group, reach `sliding_window-1+extra_retained` | transferred (5-block clip) |
| `compressed_kv` (HF:105-113, 627) | kv-source compressed groups | transferred |
| `index_keys` | indexer K groups | transferred |
| `buffer_kv`/`buffer_gate` (HF:91-103, 335) | `CompressorStateCache` ring | transferred |
| `token_history` (HF:115-123, 927) | `all_token_ids` lookback gather | recomputed |
| candidates passed per forward (HF:444-464) | candidate block buffer | recomputed |
| DSpark: none (no `dspark`/`mtp` in HF file) [FH: grep] | draft SWA groups | transferred (needs DSpark on P) |

## 4. Hazards (state neither transferred nor recomputed, or misplaced)

- **H1 DSpark off on P.** P's KV config lacks the draft groups; P's sender compares group counts and fails the request
  with "KV group count mismatch" [FH: MC:2096-2110] - an error path, not silent loss. What D does after that failure
  (recompute vs abort) depends on the load-failure policy [OPEN: `_handle_failed_recv` MC:2827-2857 read only in part].
  Remedy per recipe: same `--speculative-config` on P, optionally `prefill_context_kv_only: true` (DFlash-family,
  DSpark included) which keeps `num_speculative_tokens` (so ring capacity and groups match) and only stores context KV
  [FH: speculative.py:683-690, 1531-1547; spec_decode/speculator_config.py:104-116; dflash/config.py:23;
  V/vllm/config/vllm.py:2317-2325 requires a kv producer]. Whether DSv4.1 ROCm runs on the Model Runner V2 path that
  hosts DSpark: [OPEN].
- **H2 `swa_bounded_replay` differs between pools** (e.g. one pool with `--no-swa-bounded-replay`, as in the TP2 log in
  SYNTHESIS s2, or one pool off CDNA3/4 [FH: attention.py:627-648]). Then P sends N tokens' blocks while D requests
  N-1; the sender keeps the *last* `n_remote` blocks when P has more [FH: MC:2154-2156]. If N-1 is a multiple of the
  block size P has one extra block and the suffix mapping is shifted by one block [INF from MC:2140-2160]. Not tested.
  Keep the flag identical on both pools.
- **H3 Partial/unwritten slots.** Blocks are copied whole; tail slots of the last block and of the ring carry whatever
  bytes P had there. This is why non-UMBP connectors disable the AITER sparse-MLA path (`only_restores_canonical_kv`)
  [FH: SYNTHESIS C1; V/vllm/models/deepseek_v41/amd/rocm.py:95-109]. Correctness under the Triton path is not tested
  [OPEN].
- **H4 Logical vs kernel block split.** Non-Mamba groups, ring included, are expanded by
  `_physical_blocks_per_logical_kv_block` [FH: MC:2045-2070, 1595-1621]. Whether DSv4.1 ROCm ever has a split > 1 and
  whether the 1-block ring tolerates it: [OPEN].
- **H5 Draft KV dtype.** Groups must have identical specs on both sides; a draft `kv_cache_dtype` differing between pools
  changes the group layout [INF]. Muse saw a connector change the draft fp8 KV path [2H: SYNTHESIS 4.2].
- **H6 SWA clip uses `sliding_window` only**, not `extra_retained_tokens`; DSpark's prefill lookahead is 1, so
  `extra_retained_tokens = 0` and the clip suffices [FH: MC:1075-1087; V/vllm/v1/core/kv_cache_planning.py:1619-1630;
  speculator_config.py:135-138]. A drafter with lookahead > 1 would need re-check [INF].
- **H7 D-side window alignment with a local hit inside the last window.** Both sides clip to their last 5 SWA blocks; that
  they cover the same positions when D's allocation boundary differs from P's is assumed by the suffix rule, not proven
  [OPEN]. A 1P1D test with prompt lengths around multiples of 32 and partial D prefix hits would decide H2/H7.

## 5. Decode prefix caching

- D presents a block-aligned local hit; Mooncake requests only `N-1 - hit` tokens and only the unhashed blocks of each
  group [FH: scheduler.py:1163-1175; MC:1210-1215, 1241-1253]. P has already computed the full prompt; the sender takes
  the last `n_remote` blocks per group [FH: MC:2154-2156].
- Prefix-cacheable here: compressed KV and indexer K. Not prefix-cacheable: SWA under bounded replay
  (`prefix_cacheable = not bounded_replay`) and the ring (`False`) [FH: single_type_kv_cache_manager.py:1124-1127,
  1644-1646]. So SWA (5 blocks x 43 layers) and ring blocks are pulled on every handoff regardless of D hits [INF].
- After the pull D caches the received blocks, so a later turn can hit them locally [FH: scheduler.py:3677-3679].
- A full local hit (`count <= 0`) loads nothing; `window_loaded` is then False and the 128-token replay rewind applies,
  i.e. D recomputes the window locally [FH: MC:1210-1217, 1241-1253; scheduler.py:1255-1270].

## 6. Not covered

MoRIIO and NIXL paths; failure-policy behavior after a rejected pull; the ring read in the fused compressor kernel;
any measurement.

## Verification (adversarial)

2026-11-19, re-read every cited line in `V` @ 7a62ca0 and `HF`. Code read only; nothing run. Verdict numbers follow the
author's 12-point summary (s0 here covers 1-7; s4 covers 8-10).

| # | Verdict | Result | Evidence / correction |
|---|---|---|---|
| 1 | Per-group transfer, all groups by default; only HiSparse opts out; Mooncake registers ring, restores all groups | **Confirmed** | `enable_kv_transfer: bool = True` [FH: kv_cache_interface.py:708]; `transfer_group_ids` filters on it [FH: :890-896]; the only `enable_kv_transfer=False` in `V/vllm` is hisparse/layout.py:291 [FH: grep]. Registration loop has no ring exclusion; `CircularBufferManager` is excluded only from `use_packed_row` [FH: MC:2380-2442]. `get_loaded_kv_cache_group_ids` returns all `transfer_group_ids` [FH: MC:869-870]. |
| 2 | Compressed KV + indexer K of 2/8/14/20 transferred; 128-token SWA of 43 layers clipped to last 5 blocks on P and D | **Confirmed, with caveat** | Only kv-sources return a `FullAttentionManager` config [FH: attention.py:1290-1299]; indexer K likewise [FH: :1405]; 43 = 40 + 3 DSpark (ratio 0) [2H: model_arch.md:42-44]. Clip `cdiv(sw,bs)+1` [FH: MC:1075-1087]; D side MC:1248, P side MC:1371. Caveat: the clip runs only when `_is_hma_required` [FH: MC:1100], which is False with `--disable-hybrid-kv-cache-manager` [FH: MC:1045-1050]; then nothing is clipped. |
| 3 | Ring always transferred, never hashed, never clipped | **Confirmed** | `prefix_cacheable -> False`, one block per request [FH: single_type_kv_cache_manager.py:1644-1676]; unhashed blocks = `block_hash is None` [FH: kv_cache_manager.py:113-123]; non-SWA groups get `blocks_per_sw=0` -> unclipped [FH: MC:1076-1087, 1102-1105]. Ring lives on kv-source compressors with ratio>1 [FH: attention.py:672-684; compressor.py:339-347]. Ring capacity = `max(8, next_pow2(num_spec+2))` [FH: compressor.py:252-254]: 8 both with spec off and with the recipe's 5 tokens, so this alone does not split the specs. |
| 4 | Bounded replay N-1; D does not rewind the window after a pull | **Confirmed** | MC:1107-1114 (D count), 1116-1147 (P truncation, `max_tokens=1`), 1149-1156; scheduler `_load_restores_replay_window` / `_mark_prefix_replay(window_loaded=True) -> 0` [FH: scheduler.py:3615-3643, 3680-3693]; no block round-down when the window is loaded [FH: :1185-1196]. |
| 5 | Engram lookback, layer-20 candidate buffer, DSpark hidden/context for N-1 recomputed on D | **Confirmed (lookback, buffer); partly supported (DSpark)** | Lookback gathered from `req_states.all_token_ids` each step [FH: nvidia/model_state.py:341-366]; AMD model uses that state class [FH: amd/model.py:1559-1561]. Candidate buffer is `[max_num_batched_tokens, topk_blocks]` scratch [FH: amd/model.py:654-666]. amd/dspark.py:150-175 shows `precompute_and_store_context_kv` writes context KV from `main_x`; that D runs it for token N-1 on its first step is [INF], the call site on D was not traced. |
| 6 | DSpark context KV < N-1 transferred only if P runs DSpark; `prefill_context_kv_only` is the P-side mode | **Partly supported** | `prefill_context_kv_only` text and validation confirmed [FH: speculative.py:683-690, 1531-1547; vllm.py:2317-2325]; DSpark inherits support via `DSparkSpeculatorConfig(DFlashSpeculatorConfig)` [FH: dspark/config.py:31; dflash/config.py:23]; `num_speculative_tokens` is unchanged (only `num_scheduled_spec_tokens` drops to 0) [FH: speculator_config.py:104-116; vllm.py:684-689]. But the cited dflash/speculator.py:298-365 is `set_attn` (draft group discovery), not the P-side write; that P's prefill writes context KV for positions < N-1 is [INF]. |
| 7 | Nothing lost when configs match | **Not refuted; still [INF]** | New gap not covered by the author: on paths where the window is *not* loaded (full D local hit, or a failed pull) D replays the last 128 tokens [FH: scheduler.py:1255-1270, 3680-3693], but the ring is never prefix-cached; whether the replay rewrites the ring's open row was not traced [OPEN]. HF parity holds: `_extra_state_names` = compressed_kv, index_keys, buffer_kv, buffer_gate, token_history [FH: HF:125-126]. Correction to s3 table: HF *does* mention DSpark (HF:1251-1252, "DSpark draft layers are not supported yet", `mtp.*` weights ignored); the conclusion "HF has no DSpark state" stands. |
| 8 | H1: DSpark off on P -> group-count mismatch error, not silent loss | **Not established (downgrade FH -> OPEN)** | MC:2096-2110 errors only if the *group count* differs. The planner buckets layers by manager config and stripes buckets into equal-size groups [FH: kv_cache_planning.py:232-256, 470-532]; draft SWA layers may join the target SWA bucket, and dspark.py itself says "the hybrid manager may place draft layers in different groups" [FH: amd/dspark.py:164-166]. If the 3 draft layers change group membership but not count, region alignment iterates the *producer's* regions and looks each up on D [FH: MC:448-505], so a P without draft layers passes alignment; the remaining guard is the per-region `group_index` equality [FH: MC:476-485]. Which case DSv4.1 hits needs the actual group layout (startup log `kv_cache_groups`) on both pools [OPEN]. Silent loss of D's draft context KV is not excluded by the code read. D's reaction to a failed pull: `_handle_failed_recv` -> `_failed_recv_reqs` under HMA [FH: MC:2827-2865]; scheduler then caches valid tokens and replays [FH: scheduler.py:3654-3693]; full policy path not traced. |
| 9 | H2: `swa_bounded_replay` mismatch -> one-block shift when N-1 is a multiple of 32 | **Confirmed as [INF]** | P off / D on: P does not truncate (needs its own `_bounded_replay`) [FH: MC:1149-1156], D requests N-1 [FH: MC:1107-1114], sender keeps the last `n_remote` blocks [FH: MC:2153-2156] -> shifted mapping when P has one extra block. Reverse (P on / D off) at the same boundary hits "P num blocks less than D" error [FH: MC:2140-2152, 2161-2165]. Extra ways the flag silently flips per pool: PCP>1 without prefix caching, or non-CDNA3/4 [FH: attention.py:627-647]. |
| 10 | H3 whole-block copy; AITER sparse-MLA off with connectors; H4/H7 open | **Confirmed; OPEN items stay OPEN** | Mooncake does not override `only_restores_canonical_kv` (base returns False) [FH: base.py:236-243; utils.py:104-114] -> `_aiter_sparse_mla_enabled` returns False ("does not scrub NaNs") [FH: amd/rocm.py:95-109]. Logical->kernel expansion applies to every non-Mamba group, ring included [FH: MC:2045-2070]; split is set from `select_common_block_size` [FH: MC:1595-1621]; DSv4.1 ROCm value not determined [OPEN]. |
| 11 | D prefix caching narrows only compressed KV / indexer K; window and ring always pulled; full hit pulls nothing | **Confirmed** | SWA `prefix_cacheable = not bounded_replay` [FH: single_type_kv_cache_manager.py:1123-1125]; ring False [FH: :1644-1646]; `num_external_tokens == 0` -> empty pull list [FH: MC:1241-1253]; window not loaded -> replay rewind [FH: scheduler.py:1255-1270]. See row 7 for the untraced ring rebuild on that path. |
| 12 | Recipe "DSpark runs in both pools" | **Quote confirmed; context corrected** | The sentence is at recipe :2084-2086 inside the "Prefill/Decode disaggregation" paragraph describing the verified **GB200 NVL4 1P1D over NIXL** layout [FH: recipe :2075-2086], not a Mooncake / MI355X statement. It is a recipe rule for that preset; applicability to Mooncake on MI355X is [INF]. |

Net: 7 confirmed (1, 2*, 3, 4, 9, 10, 11), 3 partly supported (5, 6, 12-context), 1 downgraded to OPEN (8/H1: error-vs-silent depends
on the group layout), 1 new OPEN gap (ring rebuild on the replay path), 1 minor factual correction (HF does mention DSpark).
Deciding experiment for H1: start P without and D with `--speculative-config` and diff the two `kv_cache_groups` dumps (engine init only, no
request needs to be served).
