# DSA DCP (GLM-5.2): which phase benefits (Q1), and MTP / CUDA graph per phase (Q2)

Scope: the DSA DCP the authors claim — sglang #43735 (merged, CUDA trtllm) + #44322 (AMD triton, merged
into this branch) + #45205 (AMD HiCache, branch `yihou/pr45205`, not merged here). Runtime bugs ignored.
All citations are on branch `dev/yihou/glm52_dcp_pd` (`4792dc0e52`) unless marked `pr45205:`.
Model numbers: GLM-5.2 config (copy at `~/dev/git/infera.glm52.pd/yihou/yihou-expert8-pd.packup_20260924/results/model.config.json`):
64 heads, 78 layers, kv_lora_rank 512, rope 64, index_topk 2048, index head dim 128,
`indexer_types` = 21 `full` + 57 `shared` layers, q_b_proj excluded from MXFP4 quant.

## Short answer

- **Q1: DCP is a decode-phase feature.** It shrinks per-rank latent KV (capacity) and per-rank sparse-KV
  reads per decode/verify step. In prefill it saves no GEMM/MoE/attention FLOPs (attention FLOPs are
  redistributed, not reduced) and it adds collectives; the one prefill compute saving is #44322's
  ROCm-only split of the prefill indexer rows across DCP ranks. In mixed (non-PD) serving, prefill
  benefits *indirectly*: the larger pool raises the prefix-cache hit rate, so less prefill is recomputed
  (this is what the #44322 agentic table shows). The code itself warns that DCP on a PD prefill server
  "usually adds communication overhead without improving prefill performance"
  (`python/sglang/srt/arg_groups/pd_disaggregation_hook.py:57-62`).
- **Q2: MTP and decode CUDA graph can both be on under DSA DCP** (EAGLE chain, `topk 1` only). Prefill
  CUDA graph is off under DCP: on CUDA a rule disables breakable prefill graph for `dcp_size > 1`; on ROCm
  prefill graph is off by default anyway. Details and AMD/CUDA differences below.

## Q1. What DSA DCP does, per forward mode

### Layout (all modes)

| Buffer | Layout under DCP size W | Code |
|---|---|---|
| Target latent KV (fp8 on HIP: nope+rope) | striped by token slot: rank `r` stores slot `s` iff `s % W == r`, at row `s // W`. Per-rank pool rows = logical / W. | `memory_pool.py:4925-4946` (`_write_loc_dcp_span`, `_scatter_mla_rows`), HIP fp8 write `memory_pool.py:4997-5009` (#44322 added `dcp_world_size/dcp_rank`) |
| Index-K cache (fp8 128 + scale, only the 21 `full` layers) | **replicated**: every rank holds the whole logical sequence, addressed by the widened slot; size `(size + page) * W - page` rows | `memory_pool.py:5344-5352`; doc `docs/docs/advanced_features/dcp.mdx:177-179` |
| MTP draft KV | HIP: sharded like the target (#44322). CUDA: replicated over the widened slots, draft attends with `dcp_size 1`. | `configs/model_config.py:214-220` (`dsa_draft_shards_under_dcp`), `kv_cache_configurator.py:491-502` (`loc_space_scale`), `models/deepseek_v2.py:2047-2053`, `dsa_backend.py:544-550` |
| Radix / allocator page | widened logical page = `page_size * W` (page_size 64 for DSA → 256 at DCP4) | `arg_groups/overrides.py:213-219`; `model_overrides/deepseek_v2.py:170-176` |

Every rank still computes the latent KV and the index K for **all** tokens of the batch (tokens are
replicated across the attention-TP group; the indexer weights are `ReplicatedLinear`,
`dsa_indexer.py:339-363`); only the latent KV *write* is filtered to owned slots.

### Decode, target verify (and on HIP, draft extend)

`is_dcp_mla_decode_phase` is true for decode, target verify, and (DSA) extend unless a gathered prefill
was planned; on HIP it also includes draft-extend-v2 (`forward_mla.py:122-139`). Per layer:

1. **Q**: either all-gather Q heads across the DCP group (`all_gather_q_for_mla_decode`,
   `layers/dcp/comm.py:258-272`; called at `forward_mla_rocm.py:772-777`), or, with
   `dcp_replicate_q_proj` (defaulted True for a2a/fi_a2a, `model_overrides/deepseek_v2.py:33-34`), each
   rank projects all heads itself from all-gathered full-head `q_b_proj`/`w_kc` weights
   (`model_executor/model_runner.py:1120-1157`; decode/verify only, `forward_mla.py:141-147`). It is skipped
   for quantized q-proj or non-16-bit `w_kc` (`model_runner.py:1139-1149`). Whether it is active on
   GLM-5.2-MXFP4 on ROCm is **unknown** (q_b_proj is bf16 in the checkpoint, but `w_kc` dtype on gfx950
   is not checked) → decide by grepping the server log for `dcp_replicate_q_proj: prepared ... for N MLA layers`.
2. **Indexer**: paged indexer over the full replicated index K, identical on every rank
   (`dsa_indexer.py:2231-2237` → `_get_topk_paged`). This is the same work as plain TP (it is replicated
   across TP ranks there too). Not reduced by DCP.
3. **Attention**: each rank attends **all W×(H/attn_tp) heads** over only the top-k slots it owns:
   `localize_dcp_page_table` (`dsa/dsa_dcp.py:41-63`) → `_forward_triton_dcp`
   (`dsa_backend.py:3566-3609`, scan bounded by the owned count) on HIP; `dcp_trtllm_sparse_attention`
   (`dsa/dsa_dcp.py:66-118`) on CUDA. Shared-top-k layers reuse the remap (21 remaps instead of 78,
   `dsa/dsa_dcp.py:29-32`).
4. **Merge**: a2a of head partials + LSE then local combine (`dcp_a2a_lse_reduce`, `comm.py:474`;
   called at `forward_mla_rocm.py:1018-1031`), or ag_rs (`cp_lse_ag_out_rs_mla`, `comm.py:85`).

Net per decode/verify step, per rank, vs TP with the same attn_tp:
- **Saved**: latent KV memory (÷W) → larger logical capacity; sparse-attention KV reads ≈ top-k/W rows
  per query (top-k = 2048, so this saving is bounded by 2048 rows/query/layer, independent of context).
- **Unchanged**: attention FLOPs (W× heads × 1/W keys), indexer scoring of the whole context
  (O(context), replicated), all GEMM/MoE.
- **Added**: Q all-gather (or W× q_b/w_kc GEMM work with replicate-q), one a2a/ag_rs of partial outputs
  + LSE per layer, the top-k remap kernel, index-K memory W× its non-DCP share.

### Extend / prefill

- **Indexer** (ragged, prefill only): on HIP, #44322 splits the query rows evenly across the DCP group and
  all-gathers the top-k (`dsa_indexer.py:1562-1576, 1718-1724`; `dsa/dsa_dcp.py:156-176`). This is the
  only prefill compute that DCP reduces (÷W on the indexer logits). CUDA computes it W times.
- **Attention**, two paths on HIP:
  - *Owned path* (default for long/unaligned prefixes and inside graphs): same as decode — gather Q for
    all extend tokens, attend owned top-k slots, a2a merge. No FLOP saving; Q gather + partial-output a2a
    scale with extend tokens × heads.
  - *Gathered path* (`dcp_use_gathered_prefill`, `dsa/dsa_dcp.py:123-142`, planned at
    `dsa_backend.py:1405-1466`): all-gather the prefix latent KV into `dcp_kv_buffer` and attend with this
    rank's heads only, i.e. plain TP attention plus a KV all-gather. Chosen when
    `sum(prefix) < SGLANG_DSA_DCP_OWNED_PREFILL_RATIO (3.0) × gathered_heads × extend_tokens` and every
    prefix is a multiple of W; excluded in graph capture (`dsa_backend.py:1424-1430`).
  - CUDA (#43735): extend always takes the owned path ("no prefix-KV gather", PR body; `dcp.mdx:177`).
- **Write**: every rank computes KV for all extend tokens, writes only owned slots.
- PR evidence on prefill cost: #44322 "a 360K-token prefill drops from 70 seconds to 18 seconds" — that
  is DCP-before vs DCP-after the gathered/split changes, not DCP vs no-DCP. **No DCP-vs-no-DCP prefill
  latency is published** → measure TTFT of a cold 32K/128K prompt with `--dcp-size 1` vs `4` on the same
  build.

### Numbers stated by the authors (quoted, not re-measured)

- #44322 (MI355X, GLM-5.2-MXFP4, TP4/EP4, fp8 KV, MTP): **KV capacity 2.83M → 8.81M tokens
  ("2.20M rows per rank x 4")**; MTP draft KV "shrinks from 5.85 GB to 2.26 GB per GPU" after sharding it.
  The per-rank row count drops 2.83M → 2.20M because the replicated index K (and draft) eat into the budget.
- #43735 doc: index K "costs `c` times its usual share of each rank's KV budget (about 6% at DCP2 and
  17% at DCP4 on GLM-5.3)" (`dcp.mdx:179`).
- #44322 agentic replay (TP4/EP4+DCP4+MTP vs TP4+HiCache): c24 25,302 vs 10,872 tok/s/GPU; c40 27,087 vs
  5,144; chip cache hit 94.8-96.9% vs 18.6-31.2%. At c48 the DCP pool fills (79.4% hit) and P90
  interactivity drops to 3.26. #45205 (+HiCache L1/L2 on ROCm) c48: 31,935 tok/s/GPU, P90 19.90.
- #43735 (B300, GLM-5.3, TP8, no PD): AIME gen tok/s TP8 1140, DCP2 1041, DCP8 838 (no MTP) — at low
  concurrency DCP alone is slower per request; the win is capacity.

### Verdict

Decode. The mechanisms DCP changes (KV bytes stored per rank, sparse KV read per step) are decode-side;
prefill gets comm overhead and, on ROCm only, an indexer split. The mixed-mode throughput gains reported
are attributed by the authors to cache hit rate (capacity), which in a PD deployment lives on the
**decode** side (decode KV pool / decode radix) and on the prefill node's own cache, not on prefill DCP.
**Unmeasured**: per-step decode ITL DCP4 vs TP4 at fixed batch (does the KV-read saving beat the added
a2a + Q-gather?); prefill TTFT DCP4 vs TP4. Both are single-run measurements on 054.

## Q2. MTP and CUDA graph per phase

| Phase | MTP (EAGLE) under DSA DCP | CUDA graph under DCP |
|---|---|---|
| Prefill / extend (target) | Yes; extend of the target runs normally (owned or gathered path). | **Off.** CUDA: breakable prefill graph disabled when `dcp_size > 1` (`arg_groups/cuda_graph_hook.py:212-215`, reason: capture builds an extend with `attn_dcp_metadata=None`). ROCm: prefill graph default is DISABLED on non-CUDA (`model_executor/cuda_graph_config.py:119-123`). The gathered-prefill planner refuses inside PCG/BCG (`dsa_backend.py:1427-1428`). |
| Decode (target) | n/a (with MTP the target runs verify) | **Allowed**: no rule disables the decode graph for DCP (searched all `arg_groups/*hook.py`); decode default FULL (`cuda_graph_config.py:150`). DCP decode ops (remap kernel, a2a `all_to_all_single`) run under capture; fi_a2a workspace is initialized pre-capture (`runner/base_runner.py:341-348`). Evidence it is exercised: #45205 e2e test uses default graph settings (`pr45205:test/registered/dcp/test_glm52_dcp4_hicache_amd.py`, fixture `dsa_mtp_fixture.py:64-84` passes no graph flags); the pre-#44322 ROCm failures were reported *in* decode graph capture. |
| Target verify | **Supported, chain only**: `--speculative-eagle-topk 1` enforced (`model_overrides/deepseek_v2.py:25-26`); draft tokens = steps+1 (doc `dcp.mdx:184`). Same DCP owned path as decode. | Allowed (captured with decode graphs, same rules). |
| Draft decode steps | CUDA: draft has `dcp_size 1` and a replicated pool (no DCP comm; pool `W×` larger). HIP (#44322): draft runs **under DCP**, sharded like the target, Q gather + a2a merge every draft step (`dsa_backend.py:544-550`; `deepseek_v2.py:2047-2053`). | Allowed; no DCP rule on draft graphs. |
| Draft extend (v2) | HIP: DCP owned path with LSE merge (`forward_mla.py:133-137`, `include_draft_extend_v2=_is_hip`); translator bound for DCP reads (`speculative/eagle_worker_v2.py:397-408`). CUDA: replicated, no DCP. | Allowed; graph replays keep the owned path (gathered prefill excluded in graphs). |

Other constraints that apply to the PD question: DSA DCP rejects PD disaggregation
(`model_overrides/deepseek_v2.py:23-24`), HiSparse, and (on this branch) HiCache
(`model_overrides/deepseek_v2.py:21-22`). #45205 relaxes HiCache to "ROCm only", L1/L2 only, with the MTP
draft packed into the target's DCP host rows (`pr45205:arg_groups/model_overrides/deepseek_v2.py:21-24`,
`pr45205:arg_groups/hicache_hook.py:124-171`, `pr45205:mem_cache/pool_host/dsa.py:196-199` pages the
index-K host mirror at the widened size). It does not lift the PD rejection.

AMD-specific: ROCm DCP requires the triton DSA backends (`arg_groups/overrides.py:569-576`; default
`triton` on HIP at `model_overrides/deepseek_v2.py:27-33`); the ROCm fused rope+cache write is disabled under
DCP (#44322 body); the draft is sharded only on HIP; the prefill indexer split and gathered prefill are
HIP-only.

Unknown / to measure: whether `dcp_replicate_q_proj` is active on ROCm GLM-5.2-MXFP4 (log line above);
decode graph capture time and memory under DCP4 (watch capture log); whether the draft-extend graph is
captured on ROCm (log `Capture draft extend cuda graph` lines).
