# G1: Is `only_restores_canonical_kv()=True` safe for Mooncake / MoRIIO(+#68513) / NIXL on DSv4.1 ROCm?

Date 2026-11-19. Source: `V` = `source/vllm` @ 7a62ca0666. No GPU, no run. Tags: [FH] = read in code here, [INF] = inferred
from FH lines, [OPEN] = not established. PR #68513 diff saved at `research/notes/g1/vllm_pr68513.diff` (gh, 215 lines).

## 0. Verdicts

1. **The fast-path kernels read only slots their index lists mark valid; invalid lanes are masked to 0 in the vLLM-side
   loaders** [FH: V/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:2756-2787, 2790-2830, 3328-3330, 3539-3541, 3592-3593,
   3656]. So "NaN in an unwritten slot" can only reach the math if (a) a valid index points at a slot nobody wrote for that
   position, or (b) the AITER kernel (`rocm_aiter_ops.triton_sparse_mla_fwd`, external package, not in this tree)
   loads unmasked. (b) is [OPEN]: the aiter source is not available here (`import aiter` fails locally).
2. **What "NaN-free" means for the canonical writer** [FH]: on gfx950 the compressed-KV writer clamps the UE8M0 scale to
   254 (255 decodes to +inf, and inf x 0 = NaN) and replaces NaN RoPE with 0 (`SANITIZE_CACHE_NANS=_ON_GFX950`)
   [V/vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py:156, 533, 625-627, 636-637; decode of 255 ->
   0x7F800000 at rocm_aiter_mla_sparse.py:2750-2753]. The Triton path otherwise scrubs `k != k` to 0 per tile; with
   `TRUST_EXTRA_CACHE_NAN_FREE` it skips that only for the compressed ("extra") cache [rocm_aiter_mla_sparse.py:3369-3374,
   3604-3611]; the SWA ("main") segment is always scrubbed in the Triton kernel (constexpr `False`) [rocm_aiter_mla_sparse.py:3557],
   while the AITER path reads both SWA and compressed caches with no scrub [V/vllm/models/deepseek_v41/amd/rocm.py:1524-1574].
3. **Bytes Mooncake delivers into indexed slots are P's canonical bytes if and only if P and D have an identical cache
   layout and writer** [INF]. Transfers are whole kernel blocks (`kv_block_len` / page) per transfer group
   [mooncake_connector.py:2160-2200, 2410-2470]. But **Mooncake does not verify layout identity for MLA**:
   `_validate_asymmetric_region_lengths` returns before any length comparison when `total_num_kv_heads is None` (forced
   for `use_mla`) and the producer cache is replicated (`is_mla` => replicated) [FH: mooncake_connector.py:381-384,
   1854-1865; V/vllm/distributed/kv_transfer/kv_connector/utils.py:659-661], and Mooncake / MoRIIO carry no
   compatibility hash (grep for `compat|hash|version` finds none in the handshake) [FH]. NIXL does hash vLLM version, model,
   `cache_dtype`, attention backend, HMA, spec config [FH: nixl/metadata.py:154-224] but not `indexer_kv_dtype` nor GPU
   arch (gfx942 FNUZ vs gfx950 OCP, and `SANITIZE_CACHE_NANS` is gfx950-only) [FH-derived]. Whether DSv4.1 sets `use_mla`
   is [INF] from SYNTHESIS C2 (MLA-style replication), not re-read here.
4. **Coverage of indexed slots after a Mooncake load (same layout)** [FH + INF]:
   - D pulls `N-1` tokens under DSv4.1 bounded replay and P is truncated to `N-1` [mooncake_connector.py:1106-1114,
     1116-1147, 1149-1157, 1210-1216]; D computes token `N-1` itself, so the slot it writes in the received partial last
     block is overwritten by D's own writer before any read [INF].
   - SWA groups are clipped to `cdiv(window, block)+1` tail blocks on both sides [1075-1104, 1238-1250]; the decode/
     replay SWA window `[N-128, N-1]` lies in that tail [INF]. Blocks before the clip are never in a SWA index list [INF].
   - Compressed KV, indexer K and the `CircularBufferManager` ring are transfer groups by default
     (`enable_kv_transfer=True`) [V/vllm/v1/kv_cache_interface.py:708, 890-896]; Mooncake sends the ring as an unpacked
     region [mooncake_connector.py:2436-2443] and declares every transfer group loaded [869-870], so the scheduler marks the
     replay window as restored and replays nothing beyond the last token [V/vllm/v1/core/sched/scheduler.py:3615-3644,
     3680-3690].
   - P and D blocks are paired by tail ("Partial prefix cache hit: just read uncomputed blocks") [2142-2156]. Same
     `block_size` both sides is required for positions to line up; Mooncake only checks `block_size` against the kernel block
     size locally [1610]; a cross-side block_size check was not found [OPEN].
   - Tail slots (positions >= N-1) of the last received block carry P's bytes. On D they are written by D before being
     indexed [INF]. If `needs_kv_cache_zeroing` is on, D skips zeroing exactly the loaded groups' external blocks
     [scheduler.py:1507-1511; kv_cache_manager.py:446-447], and P zeroed its own fresh blocks, so the tail arrives as zeros
     or P-canonical bytes [INF]. `needs_kv_cache_zeroing = has_mamba_layers or has_mixed_precision_kv_cache`
     [kv_cache_interface.py:878-887, 966-977]; its value for DSv4.1 is [OPEN] (read it from a startup or a config dump).
   - Load failure: HMA requests go to `_failed_recv_reqs` [mooncake_connector.py:2850-2860]; with
     `kv_load_failure_policy=recompute` the scheduler resets `num_computed_tokens=0` and recomputes, else the request fails
     [scheduler.py:3653-3675, 3867-3880]. No path found where a failed or partial pull is read as computed [INF].
5. **Net for Mooncake**: under a same-build, same-arch (gfx950), same `kv_cache_dtype`/`indexer_kv_dtype`/`block_size`/
   HMA config P and D, no code path read here puts non-canonical bytes into a slot the fast path indexes [INF]. Safety is
   **not enforced** by the connector (verdict 3), and verdict 1(b) is open. So "safe" holds only as an operator-asserted
   precondition, not as a property the connector can claim unconditionally.
6. **MoRIIO (+#68513)**: #68513 only lets `CircularBufferManager` groups through in READ mode and pulls state groups on a full
   hit [notes/g1/vllm_pr68513.diff:126-215]. MoRIIO backs off one token only for Mamba
   (`num_external_tokens - 1` under `_has_mamba`) [moriio_connector.py:889-892]; it has no `_bounded_replay` handling (grep
   finds none) and #68513 does not add one. So D loads all `N` tokens and then re-computes the last one on a full hit
   [scheduler.py:3680-3690], which `nixl/base_scheduler.py:400-411` and `mooncake_connector.py:1106-1114` say is wrong for
   the DSv4.1 SWA window ("must be laid out for that token"). Consequence of that (wrong values vs NaN) is [OPEN]; it is a
   correctness precondition to settle before any fast-path override for MoRIIO. Layout identity is also unchecked [FH:
   per-layer `block_len` divergence is allowed, moriio_connector.py:2777-2785].
7. **NIXL**: has bounded-replay backoff [nixl/base_scheduler.py:105-106, 400-411] and a compat hash (verdict 3). Same
   coverage argument as Mooncake was not re-read for NIXL's write path [OPEN].

## 1. Exact conditions under which foreign KV can leave NaN/garbage in slots the fast path reads

From the reads above, each one is sufficient [INF unless tagged]:
- C1 P and D differ in record format while byte widths match or are unchecked: `kv_cache_dtype` (fp8_ds_mla 584 B vs
  fp8_e4m3 512 B, unchecked for MLA in Mooncake, verdict 3), `indexer_kv_dtype` (MXFP4 vs FP8; not in NIXL hash), GPU arch
  (gfx942 writes FNUZ, where 0x80 is NaN, and does not run `SANITIZE_CACHE_NANS`) [FH: fused_compress_quant_cache.py:156, 533;
  amd/mono/common/arch.py:10].
- C2 P runs a different vLLM build whose writer does not clamp scale<=254 / sanitize RoPE (no version check in Mooncake/MoRIIO).
- C3 `block_size` (or compress-ratio block geometry) differs, so tail-paired blocks map positions wrongly (values wrong;
  NaN only if C1/C2 also hold).
- C4 Bounded-replay mismatch (MoRIIO today): D recomputes the last token over a window laid out by P; effect [OPEN].
- C5 The AITER kernel reads unmasked lanes of a tile/page (verdict 1b) [OPEN]; then P-side stale bytes in tail slots
  matter. Colocated D has the same exposure via pages reused across overlaid groups [V/vllm/v1/worker/utils.py:604-606]
  unless zeroing is on, so C5 would not be connector-specific except where zeroing is skipped for loaded blocks.

## 2. Minimal patch candidates (none written, none tested)

- M1 (opt-in override, Mooncake only): `MooncakeConnector.only_restores_canonical_kv` returns
  `extra.get("trust_canonical_kv", False)`, mirroring UMBP's config-gated override [umbp/connector.py:110-117]. Plus a handshake
  fingerprint: hash of every per-layer `KVCacheSpec` (dtype, record width, block_size, quant mode) + `cache_dtype` +
  `indexer_kv_dtype` + gfx arch + vLLM version, rejected on mismatch. This turns C1-C3 into a hard error. UMBP already derives
  such a per-spec fingerprint for its namespace (comment at umbp/connector.py:112-115; implementation not read).
- M2 (scrub on receive): after `finished_recving`, run a sanitize pass over received compressed-cache blocks
  (scale 255 -> 254, NaN fp8/bf16 -> 0) and zero slots `>= N-1` in the last block. Bytes touched per request = the
  per-request KV (0.29-0.43 GB at ISL 173k, SYNTHESIS s0.5); cost unmeasured. M2 covers C1-C2 for NaN only (values stay
  wrong) and C5; it does not fix C3/C4.
- Neither is shown to be sufficient while C5 is open. MoRIIO additionally needs the bounded-replay backoff (C4) before an
  override is considered.

## 3. Tests that would establish it

- T1 Byte-identity (decides C1-C3 on a given deployment): 1P1D, same prompt; dump D's blocks for all transfer groups after
  load, and a colocated reference prefill of the same `N-1` tokens; compare bytes for positions `< N-1` per group (SWA tail
  window, compressed, indexer K, ring). Prompts with `N-1` mod block_size in {0, 1, B-1} and mod compress ratio 2 in {0, 1}.
- T2 Poison test (decides C5): fill P's KV allocation with 0xFF (NaN in fp8 e4m3fn, inf scale) instead of zeros
  (`allocate=` hook, V/vllm/v1/worker/utils.py:618-621), run 1P1D with the override forced; D greedy outputs/logprobs must
  equal the non-poisoned run and contain no NaN. Repeat poisoning D's pool to cover colocated page reuse.
- T3 End to end: greedy 1P1D with override vs Triton path vs colocated AITER path on long AgentX-like prompts; token-exact
  or logprob-diff bound, plus a NaN counter on attention outputs.
- T4 Negative: P `fp8_ds_mla` vs D `fp8_e4m3` (and MXFP4 vs FP8 indexer) must be rejected at handshake after M1; on main
  today the Mooncake MLA path is expected not to reject [FH-derived from verdict 3; not run].

## Verification (adversarial)

Re-read 2026-11-19 against `V` @ 7a62ca0666. No GPU, no run. Each verdict: CONFIRMED / CORRECTED / REFUTED.

- V1 (masked loaders) CONFIRMED [FH]. `_load_fp8_ds_mla_gfx950_nope_exact_chunk` loads data with `mask=valid, other=0`,
  scales with `other=127` (decodes to 1.0), then `tl.where(valid, value, 0)` [rocm_aiter_mla_sparse.py:2766-2787];
  invalid lanes are redirected to slot 0 [:3328]. (b) stays [OPEN], narrowed: the wrapper docstring says the AITER kernel
  masks `-1` indices when `has_invalid=True` [V/vllm/_aiter_ops.py:4083-4086], and DSv4.1 does not pass `has_invalid`, so the
  default `True` applies [_aiter_ops.py:4074; deepseek_v41/amd/rocm.py:1559-1574]. The kernel itself
  (`aiter.ops.triton.attention.sparse_mla.sparse_mla_fwd`, or FlyDSL when eligible [_aiter_ops.py:4104-4135]) is not in this
  tree and was not found under /home/yihou/dev or ~/.cache, so the docstring is second-hand.
- V2 (NaN-free) CORRECTED in scope. Clamp to 254 and RoPE NaN to 0 are CONFIRMED [fused_compress_quant_cache.py:156, 533,
  625-627, 636-637; `_ON_GFX950` comes from vllm.platforms.rocm, :11-14]. The `TRUST_EXTRA_CACHE_NAN_FREE` gate [:3369-3374]
  and the hard-coded `False` for the SWA segment [:3557] are CONFIRMED. Correction: "the AITER path reads both caches with
  no scrub" holds only when `_plain_fp8_kv` is False. With `kv_cache_torch_dtype == float8_e4m3fn`, `_aiter_sparse_mla`
  dispatches to `_plain_fp8_sparse_mla` [deepseek_v41/amd/rocm.py:627, 1540-1553; defined at deepseek_v4/amd/rocm.py:184]
  and never reaches `triton_sparse_mla_fwd`. That branch's scrub behaviour was not read [OPEN].
- V3 (no layout check for MLA) CONFIRMED, and the [INF] is upgraded to [FH]. The `use_mla` path passes
  `total_num_kv_heads=None` [mooncake_connector.py:1860-1864], and a replicated producer returns before any length
  comparison [:379-384]. Only the region *count* is compared [:373-377]. MLA is always replicated [utils.py:659-661].
  `use_mla` is True for DSv4.1: `is_deepseek_mla` lists `"deepseek_v41"`
  [V/vllm/transformers_utils/model_arch_config_convertor.py:296-305], `use_mla` returns it [V/vllm/config/model.py:2207-2220],
  and the config sets `model_type = "deepseek_v41"` [transformers_utils/configs/*deepseek_v41*:18]. No compat/hash symbol
  appears in the mooncake/ or moriio/ dirs (grep) [FH]. The NIXL factors are confirmed as vllm_version, model, dtype, kv
  heads, head size, layers, backend, `cache_dtype`, HMA, spec, mode [nixl/metadata.py:193-210]. `indexer_kv_dtype` lives in
  `AttentionConfig` [V/vllm/config/attention.py:102] and is absent from those factors. GPU arch is absent too [FH].
- V4 (coverage, same layout) CONFIRMED, with additions:
  - The N-1 backoff and the P truncation are gated on `_has_mamba or _bounded_replay` [mooncake_connector.py:1106-1157].
    `_bounded_replay` is true iff some group has `prefix_replay_tokens>0` [:1055-1057], which holds only for
    `SlidingWindowManager` with `bounded_replay` [single_type_kv_cache_manager.py:1119-1131].
  - On async load D schedules `num_new_tokens=0` [scheduler.py:1335-1338], so D allocates for N-1 tokens only. P and D
    block counts therefore match, and the tail pairing [mooncake:2142-2156] does not shift blocks in the equal-config case
    [INF].
  - The DSv4.1 compressed and indexer groups use `FullAttentionManager` [deepseek_v41/attention.py:1297, 1405]. They get
    `blocks_per_sw=0` and are sent unclipped [mooncake:1075-1104] [FH].
  - `needs_kv_cache_zeroing` is no longer [OPEN]; it is now [FH-derived] True for DSv4.1. The compressor ring is an
    `AttentionSpec` with `dtype=float32` [compressor.py:251, 262-269]. The compressed spec is `uint8` (fp8_ds_mla) or the kv
    torch dtype [attention.py:1306-1316]. Two precisions mean `has_mixed_precision_kv_cache` is True
    [kv_cache_interface.py:966-977, 884-887]. This assumes those groups are not `host_resident`, which was not checked.
  - Failure handling CONFIRMED. An HMA request goes to `_failed_recv_reqs` [mooncake:2856-2857]. `_handle_failed_recving`
    then sets `num_computed_tokens=0` under recompute, otherwise the request fails [scheduler.py:3867-3880]. The block-level
    path raises for multi-group configs [scheduler.py:3890-3900].
- V5 (Mooncake net) CONFIRMED as stated: conditional, not enforced. The plain-fp8 branch from V2 adds a second kernel whose
  NaN behaviour is [OPEN].
- V6 (MoRIIO + #68513) REFUTED as worded, then CORRECTED.
  - In READ mode the MoRIIO scheduler does back D off by one token: `return max(num_external_tokens - 1, 0), False`
    [moriio_connector.py:895-901]. Lines 889-892 are the WRITE-mode branch only. So "D loads all N tokens" is wrong for the
    READ mode that #68513 enables.
  - The real asymmetry is on P. MoRIIO truncates P's prompt only when `is_producer and _has_mamba` [:913-917], and #68513
    does not touch truncation (its hunks are at :628-680, :1114, :1203, :1573; grep for `truncat|on_new_request` finds
    nothing in the diff). For DSv4.1 (no mamba), P therefore computes all N tokens while D pulls N-1 and recomputes token N-1.
    P's transferred compressed KV and compressor ring can already include token N-1's contribution [INF].
  - Whether D's recompute then double-applies it to the ring/compressed state, and whether P's extra block (when N-1 is a
    block multiple) trips MoRIIO's block pairing, is [OPEN].
  - Also confirmed: on main (without #68513), MoRIIO rejects CircularBuffer groups without mamba
    [moriio_connector.py:654-661]. The conclusion stands: MoRIIO needs a bounded-replay P-side truncation before any override.
- V7 (NIXL) CONFIRMED [nixl/base_scheduler.py:105-107, 398-411]. The write path remains unread [OPEN].
- Section 1 C4 CORRECTED to: "MoRIIO READ: P not truncated for bounded replay, while D pulls N-1". The other conditions are
  unchanged. Section 2 M1: UMBP's actual gate is `key_namespace == "auto"` [umbp/connector.py:110-117]. M1 is a new key
  modelled on it, not a copy.
