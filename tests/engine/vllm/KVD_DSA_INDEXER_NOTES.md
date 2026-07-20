# DSA indexer key cache — round-trip finding (§5 foundation)

**Claim validated (unit, deterministic, CPU):** the DSA sparse-attention
`indexer.k_cache` round-trips byte+scale-exact through the kvd connector's
`gather -> pack -> unpack -> scatter`. See `test_dsa_indexer_roundtrip.py` (PASS).

## Why it's cacheable
Indexer K layout (vLLM 0.23 `deepseek_v4/attention.py`: `head_dim bytes = 128
fp8 + 4 fp32 scale = 132`): each token is a CONTIGUOUS 132-byte run — 128 fp8
index bytes + a 4-byte fp32 (ue8m0) scale, co-packed per token. A raw-byte
gather/scatter moves both as one coherent unit -> no scale desync. This is the
key difference from the earlier "byte-offload desyncs" framing: for the INDEXER
the scale travels WITH its data.

## Where the real desync likely is (main latent, NOT the indexer)
`deepseek_v4/common/ops/cache_utils.py` main K block layout (block_size=64):
```
[0, 64*576)            token data  (448 fp8 + 128 bf16 per token)
[64*576, 64*576+64*8)  SCALES      (8 uint8 ue8m0 scales per token) — SEPARATE region
[..., block_stride)    padding
```
The registered main-latent tensor is per-token 576 (the token-data region only).
If the connector's gather/scatter moves only the 576-byte token data and NOT the
separate per-block scale region (bytes `64*576 .. 64*576+64*8`), the fp8 data is
restored but its ue8m0 scales are stale -> garbage dequant -> garbage attention.
That matches the GPU-observed "reload to garbage" far better than any indexer
issue. NEXT STEP: confirm whether the registered main K tensor's stride includes
the scale region, and if not, extend the save/restore to carry it.

## Remaining for full §5 (follow-up, needs GPU E2E)
1. Connector: offload the `indexer.k_cache` sub-spec (byte copy — proven safe here)
   instead of whole-group-skipping the DSA group.
2. Main latent: carry the SEPARATE per-block ue8m0 scale region (above).
3. Validate on DeepSeek-V3.2: the prompts that garbage today -> reload==cold 3/3,
   indexer gets/hits>0, output == KVD=0 ground truth; negative control reproduces
   garbage.

## UPDATE — root-cause investigation (ROCm/MI355X target platform)

Inspected vLLM 0.23 layout for deepseek_v32 **on the backend we actually run**
(`ROCM_AITER_MLA_SPARSE`, not CUDA flashmla_sparse):

- indexer `k_cache`: `MLAAttentionSpec(head_size = 128 + 128//128*4 = 132, uint8)`
  → 128 fp8 + 4-byte fp32 ue8m0 scale, **co-packed per token** (deepseek_v2.py Indexer).
- main MLA latent: `rocm_aiter_mla_sparse.py get_kv_cache_shape` → `(num_blocks,
  block_size, head_size)`, head_size=576 — a **single contiguous 576-byte/token
  blob** for BOTH bf16 and fp8. ROCm does NOT use the CUDA "656-byte FP8-with-
  separate-scale" format (`flashmla_sparse.py:142` returns 656 — CUDA only).

**Conclusion (corrects the initial "main-latent separate scale" hypothesis for
this platform):** on ROCm both caches are single co-packed blobs, so the
connector's raw-byte gather/scatter round-trips them faithfully (confirmed by
KVSUM save-sha==load-sha earlier + the layout/indexer UTs). The "separate scale
region" desync is a **CUDA-only** concern (656 format); it does NOT apply here.

So DSA reload garbage on ROCm is **NOT a KV-byte desync** — it is the §3
mechanism: with the DSA group whole-group-skipped, the indexer `k_cache` is never
restored on an external L3 hit, so the sparse-attention scan reads a stale
indexer -> wrong top-k -> garbage. **Fix = actually offload+restore the indexer
sub-spec (byte transport already proven sound), not skip it**, and confirm the
scan consumes the restored keys. Remaining risk is the scan/metadata wiring on an
external hit (needs GPU E2E), NOT the bytes.

## UPDATE 2 — native L1 prefix-cache boundary (the authoritative spec for L3)

Traced the vLLM 0.23 DSA prefill scan (`v1/attention/backends/mla/indexer.py`):
`build_prefill_chunk_metadata` builds `cu_seqlen_ks/ke` from the FULL
`compressed_seq_lens` (whole sequence). So for each new query token the indexer
scan reads K over the ENTIRE sequence, straight from `indexer.k_cache`.

Native L1 prefix-cache hit therefore:
- REUSES (does NOT recompute) the cached prefix's **main MLA latent AND
  indexer.k_cache** — they are still resident in the paged cache;
- computes only the NEW tokens' K + the query-side scan/top-k (O(N), always run).

**Consequence (reverses "needs a vLLM scan change"):** the scan already reads the
prefix K from the cache — it does not care whether those bytes came from an
L1-resident block or an L3 restore, as long as they are in the correct
`indexer.k_cache` slots before the scan runs (start_load_kv completes before the
model forward). So **connector-side restore of BOTH the main latent and the
indexer.k_cache is the correct AND sufficient fix** — it exactly replicates the
native L1 store/recompute boundary. No vLLM scan/metadata change needed.

The earlier `#122` split-and-offload garbaged not because the approach is wrong
but almost certainly because the indexer restore had an implementation bug
(mis-stride of the (nb,1,132) layout / lost scale). The indexer byte+scale
round-trip is proven faithful (`test_dsa_indexer_roundtrip`), so a CORRECT
byte-faithful offload of main latent + indexer should reproduce native behavior.

### L3 spec (what to store vs recompute) — mirrors native L1
| per-token state | native L1 on hit | L3 must |
|---|---|---|
| main MLA latent (576, co-packed) | reuse from cache | **restore** (byte copy) |
| indexer.k_cache (132, co-packed fp8+scale) | reuse from cache | **restore** (byte copy) |
| new-token K / query scan / top-k / main attn | recompute (new tokens) | let vLLM recompute |

## UPDATE 3 — correctness confirmed; SPEED is not a short-context win (JIT caveat)

GPU-E2E (deconfounded: high-entropy prefixes, batch of 8, base-baseline):
**reload==cold 8/8, misses=0** — DSA L3 prefix reuse is CORRECT.

Speed, measured properly (warm up kernels first — the first request pays
Triton/aiter JIT + autotune and both inflates timing and perturbs output):
| prefix tokens | cold ms | reload ms | speedup |
|---|---|---|---|
| 1205 | 180 | 142 | 1.27x |
| 3599 | 314 | 306 | ~1.0x |
| 6002 | 1747 | 1841 | 0.95x |
| 8099 | 1930 | 809 | 2.39x (noisy) |

Reading: a naive "1609ms→292ms = 5.5x" was almost entirely first-request JIT.
With warmup, at achievable context (≤ max_model_len 8k) L3 reuse is roughly
break-even — sparse-attn prefill is already cheap, so the NVMe fetch + scatter
costs about as much as recomputing. Measurements are also noisy because a new
sequence length re-triggers shape-specific autotune. The FLOP model predicts the
real win only at LONG context (~128k, where the O(N²) indexer selection
dominates prefill) — untestable here without raising --max-model-len far above 8k.
So: ship for CORRECTNESS (and long-context workloads); do NOT claim a
short-context speedup.

## UPDATE 4 — long-context speed + GPU-direct code audit (why it's still break-even)

Long context did NOT surface a speedup. Warm + N=3 median, max_model_len 40960:
| prefix | cold ms | reload ms | speedup |
|---|---|---|---|
| 8k  | 637  | 640  | 0.99x |
| 16k | 1391 | 1393 | 1.00x |
| 32k | 3306 | 3300 | 1.00x |

Root cause is the LOAD path, not compute: without P2PDMA the connector forces a
single load worker ("P2PDMA not detected ... Forcing 1 worker"); the 1-worker
CPU-bounce load of the DSA KV (~708 B/token/layer × 61 layers ≈ 1.4 GB at 32k)
runs ~420 MB/s, which happens to match the sparse-prefill recompute → break-even.

GPU-direct (hipFile AIS DMA, multi-worker, 20+ GB/s) is the lever, but:
- **Driver blocker.** On the aus MI355X nodes the amdgpu driver (dkms
  6.14.14 / release 30.10.1) does NOT export the AIS op — `kfd_ais_rw_file`
  has 0 hits in `/proc/kallsyms` AND in `nm amdgpu.ko`, and the DKMS source
  tree has 0 files mentioning it. `ais-check` reports `amdgpu: False`. This is
  a genuine driver gap (AIS lands in amdgpu-dkms 30.30.0+), not a container
  artifact — copying `/boot/config` only flips the unrelated Kernel-P2PDMA line
  to True; `amdgpu` stays False. So the GPU-direct speedup is UNTESTABLE here.
- **Code audit (static, since we can't run it).** The GPU-direct load path
  (`_prepare_chunk_for_prefetch_load`) is CORRECT for DSA: per-gid spec by
  `cache_group_id`, per-layer size/hidden_dim/num_kv_channels from each chunk's
  own header, shape-validated; indexer gid=1000 aliases group-0 block ids
  (`cache_group_id < len(page_ids) else page_ids[0]`); scatter generic over
  hidden_dim. BUT the DMA fast path is gated on `per_layer_nbytes % 4096 == 0`.
  Indexer per-layer = `chunk_tokens × 132`, 4 KiB-aligned only when
  `chunk_tokens % 1024 == 0`. At the default 256/512 the indexer silently drops
  to mmap+H2D while the main latent DMAs — half the win. `register_kv_caches`
  now WARNs on this; set `INFERA_KVD_CHUNK_TOKENS` to a multiple of 1024 for a
  real GPU-direct DSA run.

## UPDATE 5 — GLM-5.2 mixed-group OOB fix (worker-side save/load entry split)

Validating on GLM-5.2-FP8 (glm_moe_dsa) surfaced a real bug that DeepSeek-V3.2
never hit: DSA L3 reuse produced `kv_chunk OOB GUARD gid=0 hidden_dim=576
cap_rows=2200` on every save, saves failed, and reload never used L3
(`gets=0`, External prefix cache hit rate 0.0%) — output matched cold only by
deterministic recompute.

Root cause (KVDBG-confirmed): vLLM groups DeepSeek-V3.2's main latent and
indexer into TWO native kv_cache_groups, but puts GLM-5.2's into ONE mixed
group (`UniformTypeKVCacheSpecs`, 156 layers = 78 `.self_attn.attn` @576 + 78
`.self_attn.indexer.k_cache` @132). The worker's register_kv_caches DOES split
it (gid 0 main + gid 1000 indexer, layer_to_group correct). But the SCHEDULER
does not split — its chunk emit keeps gid 0 with ALL 156 layer names. The
worker's save/load then gathered all 156 at gid-0's hidden_dim 576; the 132-wide
indexer tensors (numel 9600×132) give cap_rows = 9600×132/576 = 2200 < needed →
OOB. (So the DSA-split path had never actually been exercised end-to-end —
DeepSeek's two-group layout bypassed it.)

Fix: `InferaKvdConnector._split_dsa_entry` — on the worker, expand a mixed
gid-0 save/load entry into main (gid 0, orig key) + indexer (gid 1000, key with
trailing gid byte swapped to 1000&0xFF=232) sub-entries, each gathered at its
own hidden_dim. Reuses the same per-page block ids (indexer aliases group-0's
blocks; the gather falls back to page_ids[0] for the synthetic gid). Applied at
the top of start_load_kv (keeping load_chunk_req_ids aligned) and the save
flush. No-op unless the worker registered an indexer sub-spec (plain MLA /
regular attn / DeepSeek two-group all unchanged). Scheduler chunk_tokens
untouched (avoids the earlier bootstrap-split desync).

GPU-E2E after fix (GLM-5.2-FP8, deconfounded, 30-evict): OOB count 0,
reload_actually_hit_L3 8/8, RELOAD_diverged 0/8, misses 0 — PASS. Test:
test_split_dsa_entry_expands_mixed_save_and_load + _noop_when_not_mixed.
