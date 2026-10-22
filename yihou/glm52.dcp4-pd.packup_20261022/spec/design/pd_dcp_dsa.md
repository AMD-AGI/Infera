# Design: PD disaggregation with DCP1 prefill -> DCP4 decode for GLM-5.2 (DSA) on ROCm

Status: design, not implemented. Branch `dev/yihou/glm52_dcp_pd` (`4792dc0e52` = main `fda84f80b7` + #44322).
All file:line citations are on this branch. "Verified" = read in code; "UNVERIFIED" = inferred, to be
checked by the tests in section 7.

## 0. Summary

- Target topology (one MI355X node, Mooncake over XGMI as in the known-good non-DCP PD pack-up):
  prefill GPUs 0-3 **TP4 (+ optional DP-attention 4), no DCP**; decode GPUs 4-7 **TP4/EP4 + DCP4 + EAGLE 5/1/6**,
  triton DSA backends, fp8_e4m3 KV, page_size 64.
- Most of the machinery already exists on main: the DCP1 -> DCP-N token relayout for MLA latent KV with
  dest-contiguous pack buffers (Mooncake/NIXL/Mori), logical-capacity admission, and a generic page path
  for the DSA index-K state.
- What blocks the topology today, in launch order:
  1. **Launch guard**: `_dsa_dcp_overrides` raises "DSA DCP does not support PD disaggregation"
     (`arg_groups/model_overrides/deepseek_v2.py:23-24`).
  2. **MTP draft latent KV layout**: on ROCm the DSA draft is DCP-sharded like the target (#44322), but the
     PD relayout assumes every draft entry is replicated over the widened page. On ROCm the transfer
     fails with `ValueError("PD DCP draft head slicing is unsupported for pure MLA ...")`
     (`disaggregation/mooncake/conn.py:1490-1514`). This is inferred from code, not run.
- **Index-K cache**: the generic `StateType.DSA` page path appears geometry-compatible on ROCm without
  changes (section 3.2). UNVERIFIED; a unit simulation and an e2e check are planned.
- Estimated code change: about 60-120 lines in 3-4 files, plus tests and docs. No kernel changes.

## 1. What exists on main (verified)

| Piece | Where | What it does |
|---|---|---|
| DCP1 -> DCP-N relayout decision | `disaggregation/common/conn.py:506-524` (`requires_dcp_relayout`) | Equal DCP sizes use the page path (ranks must match). DCP1 -> DCP-N is allowed for MLA / hybrid-MLA pools (the DSA pool is an `MLATokenToKVPool` subclass, so `is_mla_backend` is true, `disaggregation/utils.py:819-823`). Anything else is rejected. |
| Per-entry token geometry check | `common/conn.py:526-555` (`prepare_dcp_token_item_lens`) | Target entries: dst bytes/page == src bytes/page. Draft entries are checked at page scale `page*W`, i.e. assumed **replicated** (from #37709 `a207786205`). |
| Token plan | `disaggregation/common/utils.py` `build_dcp_token_transfer_plan` | Target rows: position p goes to rank `p % W`, row `page[p/W/P]*P + (p/W)%P`. Draft rows: identity over the virtual page `P*W`, i.e. replicated. Requires `decode_prefix_len % (P*W) == 0`. |
| Pack buffers | `common/dcp_pack.py` (#35762 `3760296be8`, #40376 `b3e4d198af`, #40284 `113f6f080e`, #40500 `018b73c7a0`) | Gathers the rank's strided rows into a dest-contiguous staging buffer, so the transfer uses large blocks. One buffer per transfer queue; size = `chunked_prefill_size x sum(token bytes)`. Cached-prefix transfers are bounded by pack capacity. |
| Mooncake send | `mooncake/conn.py:1386-1600` (`send_kvcache_dcp`), dispatch `2575-2643`, registration `2944-2963` | Uses the plan and the pack; state components (incl. `StateType.DSA`) are sent afterwards by `maybe_send_extra` on the last chunk (`2724-2729`). |
| Decode registration | `mooncake/conn.py:3373-3423` | Publishes `dst_dcp_size/rank` and the full `dst_kv_item_lens` list (target + draft entries, msg[19]). |
| Rank mapping | `common/conn.py:1315-1374` | Equal attn-TP: 1:1. Prefill attn_tp < decode attn_tp (DPA prefill): one prefill rank sends to `decode_attn_tp/prefill_attn_tp` decode ranks. |
| Decode DCP checks | `common/conn.py:1294-1310`, `arg_groups/pd_disaggregation_hook.py:82-96` | Decode DCP requires an MLA pool, prefill attn-CP 1, no prefill KV sharding, and transport mooncake/nixl/mori/ascend/fake. |
| Logical capacity | #39731 `50e50e13ab` | PD admission and load reporting use logical (widened) tokens. |
| Decode radix/HiCache with DCP | #40263 `10b0bcfd18` | Allowed in general. We keep them **off** in v1. |
| Prefill DCP warning | `pd_disaggregation_hook.py:57-62` | DCP on a PD prefill is supported only with the same layout, and "usually adds communication overhead without improving prefill performance". |
| Ascend precedent | `disaggregation/ascend/conn.py:454-497`, `hardware_backend/npu/memory_pool_npu.py:930-959` | NPU DSA PD+DCP carries a per-entry `uses_global_slots` flag: the index K is replicated over global slots, the latent is sharded. This is the same split we need. |
| Kimi-Linear PD DCP tests | `test/registered/e2e/disaggregation/test_kimi_linear_pd_dcp4.py` (CUDA B200) | Template for the e2e test: GSM8K, needle-in-haystack at 32K and 3 depths, cached-prefix flush. |

## 2. Buffers and layouts per side

P = 64 (DSA page), W = 4 (decode DCP). Widened allocator page = `P*W` = 256 logical slots
(`mem_cache/kv_cache_builder.py:379-386`). Widened slot `s` lives on rank `s % W` at row `s // W`.

| Buffer (per layer) | Prefill (DCP1, TP4 or TP4+DPA) | Decode (TP4 + DCP4, ROCm) | Transfer |
|---|---|---|---|
| Target latent KV, 78 layers, fp8 576 B/token | full, page 64, row = slot | striped, page 64, rank r holds slots `s % 4 == r` at row `s // 4` (`memory_pool.py:4925-4946, 4997-5009`) | existing token relayout + pack: each decode rank gets about 1/4 of the tokens |
| Draft (NextN) latent KV, 1 layer | full, page 64 | **ROCm: striped like the target** (`configs/model_config.py:214-220`, `kv_cache_configurator.py:491-502`); CUDA: replicated over widened slots, page 256 | **NEW**: on ROCm, treat it as a target entry (same rows as the target) |
| Target index-K, 21 `full` layers (57 `shared` layers elided), 132 B/token | page 64, row = slot // 64, 8448 B/row | **replicated**: rows addressed by widened slot // 64, 8448 B/row (`memory_pool.py:5344-5352`, `index_key_cache.py:21-46`) | existing `StateType.DSA` generic page path: prefill page list = slot//64 per 64 positions, decode list = widened slot//64 per 64 positions (`decode.py:1672-1682`, `prefill.py:1583-1594`). Every decode rank receives the full index K. |
| Draft index-K | page 64 | ROCm sharded draft: same geometry as the target (`dcp_replicated=False`, rows of 64 widened slots). CUDA replicated draft: rows of `64*W` slots (#45499 body: "A replicated DSA draft's stored index-K row holds dcp_size kernel pages") | ROCm: same page path as the target. CUDA: mismatched, so keep it rejected. |
| DSA tail (kpool compress) | n/a for GLM-5.2 (index_kpool 1; kpool>1 rejected under DCP, `model_overrides/deepseek_v2.py:164-169`) | n/a | none |
| Aux metadata: tokens, hidden states, EAGLE topk, `output_dsa_topk_indices` seed | per request | per request | existing aux path. Under DCP the decode does **not** remap the seed to local slots (`layers/attention/dsa/utils.py:173-181`). It keeps request-relative positions and uses unfused TopK (`utils.py:184-199`). UNVERIFIED for correctness under DCP; see the accept-length check. |

Why the index-K page path should line up (to be proven by T3): a decode allocator page holds 256
consecutive positions at consecutive widened slots. Index pages are 64 slots wide on both sides, so
positions `[64k, 64k+64)` occupy exactly one 64-slot index page on prefill (page = slot//64) and one on
decode (widened slot//64). `kv_to_page_indices(..., 64)` on both sides emits one id per 64 positions in
the same order. `decode_prefix_len` is either 0 (decode radix off) or a multiple of 256, which is also a
multiple of 64.

## 3. Changes

### 3.1 Launch guard (required)
`python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py:23-24`

- Allow `disaggregation_mode == "decode"` when `get_platform().is_hip`, so ROCm decode can run DCP with
  a DCP1 prefill.
- Keep these combinations rejected:
  - DSA DCP on a **prefill** server (`disaggregation_mode == "prefill" and dcp_size > 1`). It is unneeded
    and unvalidated for DSA.
  - CUDA (the replicated draft index-K rows do not match the prefill page; see section 2).
  - Decode HiCache (as today).
- The prefill server has dcp_size 1, so this function returns `{}` there; no change is needed for it.
- Error text: "DSA DCP with PD disaggregation is supported only as a DCP1 prefill feeding a ROCm DCP
  decode."

### 3.2 Index-K (expected: no code change)
- Keep the generic page path. Add T3 (section 7) to prove the pairing.
- If T3 fails, the fallback is to compute the DSA page list on decode at index-page granularity
  explicitly in `_dsa_pages_payload` (`decode.py:1672-1682`), the way Ascend's `uses_global_slots`
  does it.

### 3.3 Sharded draft latent in the relayout (required on ROCm with MTP)
1. **Registration** (`mooncake/conn.py:2944-2963`; same for `nixl/conn.py:1527-1545` and Mori). When
   `requires_dcp_relayout`:
   - Compute `dcp_draft_sharded` from the published `dst_kv_item_lens[num_target:]`:
     - sharded if each draft entry's dst bytes/page `== src bytes/page` (page scale P);
     - replicated if it equals `src * W` at page scale `P*W`;
     - otherwise raise. The two cases are distinct for W > 1.
   - Store the flag on `KVArgsRegisterInfo`.
   - Pass the per-entry page scale into `prepare_dcp_token_item_lens` (new kwarg
     `draft_page_scale: int`, default `dst_dcp_size`) so the geometry check uses the right scale.
2. **Send** (`mooncake/conn.py:1433-1470`): if `dcp_draft_sharded`, set `num_target = len(src_kv_ptrs)`
   and `num_draft = 0`. The draft then rides the target plan rows and the same pack: one gather, one set
   of transfer blocks. The pack buffer is already sized with `include_draft=True` (`common/conn.py:561-580`).
3. NIXL and Mori: I did not read how they handle a sharded draft (they have no equivalent of the
   Mooncake raise). Port the same two lines, or raise `NotImplementedError` on `dcp_draft_sharded` until
   ported. Mooncake is the transport we use.

### 3.4 Docs
- `docs/docs/advanced_features/dcp.mdx:155-196`: add "DCP x DSA x PD" (DCP1 prefill -> ROCm DCP decode;
  index-K replicated and sent whole to every decode rank; draft sharded on ROCm).
- GLM-5.2 cookbook: only after the e2e test passes, per the repo rule.

### 3.5 Not in v1 (guarded or left off)
- Decode radix cache and decode HiCache (#45205): the plan requires `decode_prefix_len % 256 == 0`, and
  the index-K suffix pairing then starts mid-prefix. Revisit after v1.
- Prefill DCP > 1, CUDA, NIXL/Mori sharded draft (unless ported), `--disaggregation-decode-host-receive-threshold < 1`
  (it forces `host_pool` retraction, which is rejected with DCP, `arg_groups/kv_cache_hook.py:361-364`),
  and `prefill_complete` allocation (rejected for `dcp_size != 1`, `pd_disaggregation_hook.py:205`).

## 4. Transfer volume (derived from config; not measured)

Per prompt token, GLM-5.2, fp8 KV:
- Latent: (78 + 1 draft) layers x 576 B = 45.5 KB.
- Index K: (21 target + draft) x 132 B ≈ 2.9 KB.

| Prefill -> decode | Bytes per token, summed over decode ranks |
|---|---|
| TP4 -> TP4 (non-DCP, MLA replicated, 1:1) | 4 x 48.4 KB ≈ 194 KB |
| TP4+DPA -> TP4+DPA (known-good: one DP rank to one DP rank) | ≈ 48.4 KB |
| TP4 -> TP4+DCP4 (this design, 1:1) | 45.5 KB (striped) + 4 x 2.9 KB (replicated index K) ≈ 57 KB, about 14 KB per rank pair |
| TP4+DPA -> TP4+DCP4 (one prefill rank fans out to 4) | same 57 KB, but all from one prefill GPU |

Recommendation: start with a **TP4 prefill** (balanced 1:1 transfer). DPA prefill (the known-good
prefill) is supported by the rank mapping (`common/conn.py:1340-1350`) and is the second arm. Which
prefill shape is faster is unmeasured.

Pack-buffer memory on prefill: 4 queues (`SGLANG_DISAGGREGATION_QUEUE_SIZE=4`, `environ.py:722`) x
`chunked_prefill_size` x 45.5 KB/token. At chunked 32768 that is about 6.0 GB per prefill GPU. It is
allocated lazily on the first decode registration (`common/conn.py:561-580`), i.e. after KV pool sizing
→ risk R3.

## 5. Risks

| # | Risk | Evidence status | Mitigation / check |
|---|---|---|---|
| R1 | The index-K page pairing is wrong somewhere: elided `shared` layers, the draft index layer, or the last partial page. | inferred | T3 unit simulation; e2e passkey at 8K/34K/96K (a wrong index K garbles long-context recall, as in #44322's "PR before" column). |
| R2 | The PD MTP seed (`output_dsa_topk_indices`, request-relative positions) is mishandled by the DCP-sharded draft on ROCm. | inferred; remap is explicitly off under DCP | e2e accept length with real draft vs mixed DCP4 (#44322 reports 3.47). A drop to ~1 points here. Fallback: drop the seed under DCP (decode recomputes). |
| R3 | Pack buffers (~6 GB/GPU) OOM the prefill when allocated after KV sizing. | arithmetic only | Lower prefill `--mem-fraction-static` by ~0.03, or `--chunked-prefill-size 16384`; watch the "PD DCP pack buffers allocated" log line. |
| R4 | Mooncake HIP IPC/XGMI cannot register the torch-allocated StagingBuffer (sub-allocation of a caching-allocator segment). | unknown | Smoke test with one 32K request; check the mooncake log for registration errors. The fallback to per-token RDMA blocks (`dcp_pack.py:61-71`) is correct but slow. |
| R5 | Decode with DCP cannot use DP-attention (`attn_tp % dcp == 0`), so decode is TP4 rather than the known-good DPA4. Decode step cost differs. | code (#43735 doc) | Measured in the perf phase; not a correctness risk. |
| R6 | The prefill `requires_dcp_relayout` dummy-rank logic with DPA prefill (one sender, 4 receivers) under-reports `required_dst_info_num`. | code read says 4 (`common/conn.py:1345-1349`) | e2e with the DPA prefill arm. |
| R7 | Infera image/branch skew: the known-good image is v0.5.21-based; this branch is main + #44322. | fact | Run both legs from this branch, mounted or rebuilt on `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020`. Leader decision. |

## 6. Implementation order

1. Unit tests T1-T3 (CPU), written to fail first where a code change is expected (T1, T2).
2. Guard (3.1) and sharded-draft relayout (3.3, Mooncake), then NIXL/Mori guards.
3. Mixed (non-PD) DCP4 baseline on 054 for the reference numbers (leader's deliverable 2).
4. PD smoke, then the e2e checks (T4), then AgentX.

## 7. Test plan

Unit (CPU, `test/registered/unit/disaggregation/test_dcp_pack.py` plus a new `test_dsa_pd_dcp.py`):
- **T1** `prepare_dcp_token_item_lens` / draft layout resolution:
  - sharded draft dst item len (`P*576`) → `dcp_draft_sharded=True`, no raise;
  - replicated (`P*W*576`) → False, as today;
  - anything else → raise.
- **T2** Sharded-draft plan: the draft entries get exactly the target's (src, dst) rows. Simulate with
  numpy buffers:
  - fill each prefill row with its position id;
  - apply the blocks for each dst rank r;
  - assert that decode rank r, row `s//4`, holds position p for every p with `s(p) % 4 == r`;
  - cover the target and the draft, chunked sends (`src_page_offset > 0`), and a partial last page
    (`num_kv_tokens` not a multiple of 64).
- **T3** Index-K pairing:
  - build a prefill `req_to_token` (random page-granular slots) and a decode `req_to_token` (random
    256-slot widened pages);
  - compute `suffix_page_indices(..., 64)` on both sides and copy 8448-B rows page-to-page;
  - assert that for every position p, decode index row `widened(p)//64` at offset `widened(p)%64` holds
    prefill's entry for p;
  - cover `decode_prefix_len` 0 and 256, sequence lengths 1, 63, 64, 65, 255, 257 and 10,000, and the
    elided layers (item len 0).

E2E (054, one node, PD 1P1D: prefill GPUs 0-3 TP4, decode GPUs 4-7 TP4/EP4 + DCP4 + EAGLE 5/1/6, Mooncake;
eager first if needed, then with CUDA graphs):
- **T4a**: math prompt answer `29`; passkey 8K / 34K / 96K → 3/3 each (the #44322 checks).
- **T4b**: GSM8K with sgl-eval, 200 examples, GLM-5.2 preset. Compare against the mixed DCP4 server and
  the non-DCP PD baseline on the same harness.
- **T4c**: MTP accept length with real draft (no `SGLANG_SIMULATE_ACC_LEN`), 4 prompts x 400 tokens
  greedy. Compare against mixed DCP4.
- **T4d**: teacher-forced logprob agreement. Score the mixed DCP4 server's greedy outputs on the PD
  stack; report top-1 agreement and mean |dlogprob|, plus a mixed-vs-mixed control.
- **T4e**: the DPA prefill arm (TP4+DPA4 → DCP4) repeats T4a + T4b.
- **T4f**: stress: a 128K prompt (pack-capacity bounded chunks), 32 concurrent 32K prompts, and a
  prefill-side prefix hit (decode receives the full sequence).

Pass criteria: T1-T3 green; T4a 3/3; T4b within 2 points of the mixed DCP4 score; T4c within 0.3 of
mixed; T4d top-1 agreement within the control's spread.
