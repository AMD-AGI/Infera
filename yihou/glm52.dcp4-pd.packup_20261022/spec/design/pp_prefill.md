# Phase 7 design: PP prefill (PP4 x TP1) -> TP4/EP4 + DCP4 decode, GLM-5.2, ROCm, Mooncake

Code read on branch `a04c3f8e0c` (main `fda84f80b7` + #44322 + recipes + our DCP PD commits). **Nothing
here is measured** except the numbers quoted from PR bodies, which are marked as such. The prefill runs
without DCP, as on the main path.

## 0. Summary

- On main, **PP prefill feeding an unpipelined DCP decode looks supported by code without new changes.**
  - Every PP stage is its own KV sender.
  - Per-stage layer subsets are paired with the decode's full layer list by layer id (latent KV relayout)
    or by `prefill_start_layer` offset (`StateType.DSA` index K).
  - Top-k for shared indexer layers crosses stage boundaries in the PP proxy tensors.
- What is **not** covered by any test or run on ROCm:
  - PP + DSA triton on MI355X;
  - PP + DCP decode;
  - PP + HiCache L2 on the prefill;
  - non-final chunk hand-over timing under PP (no overlap schedule, so the recipe event does not apply).
  These are the risks to check, not known blockers.
- Minimal change set: **none required to try it** (flags only). One optional guard or fix may be needed
  after the first run (section 4).

## 1. What main supports for PP prefill in PD

| Item | Where | Status |
|---|---|---|
| PP requires the overlap schedule off | `arg_groups/overrides.py:1612-1616` (auto) and `validation_hook.py:118-121` (assert) | auto-resolved |
| PP + EAGLE on a PD prefill (#39378, `3a64faa1f2`; the current form is `validation_hook.py:118-167`) | EAGLE single-layer only, and without `SGLANG_ENABLE_PP_SPEC` only with `disaggregation_mode == "prefill"`. #39378's architecture allowlist is no longer in the current function. Also `--min-free-slots-delay` must be unset. | only needed for mode S. **The main path is mode N (prefill without EAGLE), so not needed.** |
| Last-stage `embed_tokens` for the NextN draft; `dsa_topk_indices` through the PP ring | #39378 (`deepseek_v2.py`, `scheduler_pp_mixin.py`) | mode S only |
| Shared-indexer top-k across stage boundaries | `models/deepseek_v2.py:3106` (read `topk_indices` from proxy), `:3221-3224` (send) | in place. Needed for GLM-5.2 with partition 20,20,20,18: layer 20 is `shared` and reuses layer 18's top-k from stage 0. |
| Prefill breakable CUDA graph | `cuda_graph_hook.py:150-161` disables it for PP | ROCm prefill graph is off anyway; #4300 also runs `disable-cuda-graph: true` |
| Unified memory + PD + PP | `kv_cache_hook.py:542-546` rejects | not used |
| `prefill_complete` allocation | `pd_disaggregation_hook.py:205` rejects `pp != 1` | not used (`early`) |
| PD role switch | rejects PP | not used |
| HiCache L3 keys per PP rank | `mem_cache/hicache_storage.py:418-419` | L2 (our use) with PP: no guard found and no run on ROCm. **Unverified.** |
| `SGLANG_PP_LAYER_PARTITION` | env (#4300 uses `20,20,20,18`, #39768 `10,…,8`) | expected to work; check the partition log at startup |

## 2. PP prefill -> DCP decode transfer path

Rank mapping (decode side, `common/conn.py` `_resolve_rank_mapping`):
- prefill attn_tp 1 vs decode attn_tp 4 gives `target_tp_rank = 0` and `required_dst_info_num = 4`;
- PP: `target_pp_ranks = [0..3]` and `required_prefill_response_num *= 4`;
- so each decode rank waits for all four stages, and each stage sends to all four decode DCP ranks.

| Buffer | Per-stage mechanism | Status |
|---|---|---|
| Latent KV (target) | `send_kvcache_dcp` pairs src and dst entries by layer id (`resolve_dcp_dst_entry_indices`, `disaggregation/utils.py:1128-1150`). The decode publishes `dst_kv_layer_ids` (Mooncake registration msg[12]), and the stage's `kv_layer_ids` come from `build_kv_layer_ids`. The DCP token plan and pack are per stage. | supported by code; the same pairing is used by #39768 (PP8 -> DCP2, NVIDIA) |
| Pack buffers | sized from the stage's own `kv_item_lens` (`dcp_pack_buffer_bytes`). At chunked 16384 and 20 layers x 576 B: 16384 x 20 x 576 = 189 MB per buffer, 4 per stage GPU. | arithmetic |
| Index K (`StateType.DSA`) | registered without layer ids, so the positional path applies: `get_mla_kv_ptrs_with_pp` slices the decode list `[prefill_start_layer, +n_src)` (`common/conn.py:1764-1806`). Elided shared layers are 0-length on both sides (`allow_dst_elision`). The page pairing is the same as T3 (page 64 over widened slots). | supported by code. **Unverified run.** |
| Draft KV and draft index K (mode S only) | on the last stage only: latent pairs by layer id (draft id 78); index-K entries follow the target in both lists, so the offset slice `[60, 79)` lands on decode `[60..77] + draft`. `dcp_draft_sharded` is true only on the last stage (it needs `num_draft_entries > 0`). | by code; mode S debug only |
| Aux / metadata | sent by the last stage | generic |

Router / bootstrap: a PP prefill is DP1.
- The router target has no `dp_rank`, so the room is not realigned, and the decode body gets no
  `disagg_prefill_dp_rank`.
- The decode resolves `prefill_info.dp_size == 1 -> 0` (`decode.py:868-869`).
- The prefill's follow_bootstrap_room check is skipped because `num_dp_ranks == 1`
  (`common/conn.py:2055`).
- Bootstrap is fetched per (dp 0, cp 0, tp 0, pp k) for k = 0..3, which is generic code.
- **Router affinity must stay off** (decode DP1).
- In the infera harness this is prefill `--pp-size 4 --tp-size 1`, no `--dp-size` /
  `--enable-dp-attention`, plus `SGLANG_PP_LAYER_PARTITION`.

## 3. Interaction with the recipe patches and HiCache

- **dsa_indexer rows / dsa_dp_sync:** they target DP-attention padding and DP-rank divergence. With PP4 x
  TP1 there is no DP attention, so they are expected to be inert. Not verified.
- **dsa_page_table_rows:** decode-side; no change.
- **Non-final chunk wait event** (`e11f97495d`): it only patches the overlap-schedule hand-over
  (`prefill.py:1097-1107`). PP forces the overlap schedule off, so non-final chunks go through
  `process_prefill_chunk -> send_kv_chunk` (`prefill.py:1394-1426`) with no recorded event.
  - **Risk:** the transfer worker reads a stage's KV before that stage's forward finished. Unverified;
    whether the PP loop's ordering already guarantees completion is not established.
  - Correctness checks (long passkey with chunked prefill) will show it.
  - Fix if needed: record the same event in the non-overlap branch (about 5 lines).
- **HiCache L2 on a PP prefill:** each stage has its own host pool for its layers. No PP-specific guard
  was found for L2; never run on ROCm. Start debug with HiCache off, then on and small.
- **noblock, absorb-bmm off, responses, staged write-back:** PP-independent.

## 4. Minimal change set

1. **None for a first try**: flags only (section 5).
2. If the non-final chunk race shows up: an event in the non-overlap `send_kv_chunk` path.
3. If HiCache L2 + PP fails: run without prefill HiCache, or fix after diagnosis.
4. Mode S (prefill EAGLE) is debug-only and already covered by #39378 plus our sharded-draft relayout.

## 5. Test plan

Flags: prefill `--tp-size 1 --pp-size 4`, `SGLANG_PP_LAYER_PARTITION=20,20,20,18` (78 layers), triton DSA,
fp8 KV, `--chunked-prefill-size 16384 --max-prefill-tokens 16384` (#4300), `--mem-fraction-static` about
0.82, no EAGLE (mode N). The decode is unchanged from pdBe.

1. **Startup:**
   - each stage logs its layer range;
   - each stage logs `PD DCP pack buffers allocated: 4 x <MB>` at the first request (about 189 MB at
     chunked 16384 for 20 layers);
   - no `geometry differs` / `item length mismatch` / `Rejecting state transfers`.
2. **Correctness (sim off):**
   - math `29`;
   - passkey 8K / 34K / 96K (crosses chunk boundaries, which exercises the non-final chunk hand-over);
   - a 128K prompt;
   - GSM8K with sgl-eval 200, same settings as pdBe.
3. **Unit (CPU, optional):** extend `TestDpaPrefillFanOut` with a PP case. Four stage managers with
   `kv_layer_ids` 0-19 / 20-39 / 40-59 / 60-77 send to a decode with 78 entries (+ draft 78), and the
   union of written rows must cover every layer exactly once per owned token. This is the same harness
   as `_relayout`, with `prefill_start_layer` set for the index-K positional slice.
4. **Perf:** AgentX at the pdBe concurrency, compared with pdBe (DPA4 prefill); prefill TTFT and the decode
   parser.

## 6. Expected perf rationale (no measurement of ours exists)

Prefill only; the decode is unchanged.

| | DPA4 prefill (22k) | PP4 x TP1 prefill |
|---|---|---|
| Attention / indexer / MoE FLOPs per token | each GPU: its DP rank's tokens, all 78 layers | each GPU: all tokens, 20 or 18 layers; the same total |
| Per-layer collectives | DP gather / scatter around MoE every layer (MoE TP4) | none inside a stage; one hidden-state hand-off per stage boundary (6144 x 2 B per token, 3 boundaries) |
| Prefix-cache pool | 4 separate DP pools (a prefix may be cached on one DP rank only; router stickiness matters) | one pool; each GPU stores 1/4 of the layers for every cached token |
| Latency of one request | 1 GPU-pass per chunk | 4 sequential stages per chunk; pipelining needs several chunks or requests in flight (bubbles) |
| KV hand-off | one DP rank -> 4 decode ranks | 4 stages -> 4 decode ranks each (stage-local volume ~1/4) |

Reported elsewhere, not ours:
- #39768 (NVIDIA, GPU unstated): PP8 vs TP8 prefill, 100K cold C32 output 52.8 -> 181.4 tok/s, LongBench
  378 -> 1093. TP8 replicates attention work, so this does not translate to DPA4.
- #4045 (ATOM, MI355X, PP4 prefill + DCP4 decode, 1P1D same node): 31,067 / 38,487 / 42,884 tok/s/GPU at
  c56 / c72 / c96 (PR body, run 39028510364).
- #4300 gives no performance rationale for PP.

Whether PP4 beats DPA4 prefill at AgentX C40 is open. Expected levers: collective savings and a unified
prefix cache, against pipeline bubbles and per-request latency.

## 7. Pre-work done (branch commits; CPU only)

- **Risk 1 confirmed by code and fixed.**
  - In `event_loop_pp_disagg_prefill` (`managers/scheduler_pp_mixin.py:217-366`), `process_prefill_chunk`
    runs at the start of microbatch `i`, before results are processed.
  - `self.chunked_req.extend_end` was set when chunk `k` was scheduled in microbatch `i-1`
    (`schedule_policy.py`). That microbatch's forward on this stage may still be queued; on stages > 0 it
    waits for proxy tensors.
  - The non-overlap branch called `send_kv_chunk(req)` with no event, and `start_send_idx` then advances
    past those pages. So a read-before-write would never be re-sent.
  - Fix `6ee35f3617`: when `pp_size > 1`, record a `forward_stream` event and set
    `req.disagg_kv_sender._early_send_wait_event` before the send. The Mooncake transfer worker already
    synchronizes on it (`mooncake/conn.py:2498-2509`).
- **Tests `646dc699f3`** (`test_dsa_pd_dcp.py`; 36 DCP/PD CPU tests pass):
  - 4 stages (20/20/20/18 layers, attn TP1) relay onto a 79-entry DCP4 decode; every rank holds exactly
    its owned rows per layer, and the NextN entry is untouched;
  - the DSA index-K state slices from `prefill_start_layer`, and the last stage's draft index-K follows
    its target;
  - the non-final chunk send carries a forward-stream event under PP and none without PP. This failed on
    the pre-fix code, which was checked.

## 8. First-launch checks (Phase 7)

1. **PP + DSA triton on ROCm has never been run.** At startup, confirm each stage loads its layer range
   (`SGLANG_PP_LAYER_PARTITION=20,20,20,18`) and that `server_args` shows `dsa_prefill_backend='triton'`.
   The first prefill must complete on every stage. Any triton/aiter error is new ground. Also check the
   layer-20 shared-indexer handoff: no top-k shape errors on stage 1.
2. The prefill log shows per-stage `PD DCP pack buffers allocated`, and there is no
   `geometry` / `item length` / `Rejecting state` line.
3. Passkey 8K/34K/96K with `--chunked-prefill-size 16384`: multiple chunks exercise the non-final send
   (with fix `6ee35f3617`).
4. HiCache off first; then on with a small size.

### 7.1 Ordering review of `6ee35f3617` (code read): the event covers chunk k's forward

Within one pass of `event_loop_pp_disagg_prefill`:
- `process_prefill_chunk` runs first (`scheduler_pp_mixin.py:284`);
- then `get_new_batch_prefill` schedules chunk k (`:288`);
- then the proxy recv (`:299`);
- then `_pp_launch_batch` (`:312`), all on the CPU in the same pass.

The launch enqueues `run_batch` under `forward_stream_ctx`, i.e. on `self.forward_stream`
(`:1644-1651`; `scheduler.py:1683-1685`). The dependency on the previous stage is **stream-side**:
- `recv_tensor_dict` runs on the PP comm stream and returns an event (`:906-911`);
- `_pp_wait_forward_dependencies` adds `forward_stream.wait_stream` / `wait_event` (`:714-721`).

So by the time the next pass (microbatch i) calls `process_prefill_chunk`, the forward of microbatch
i-1 (chunk k) has been launched host-side on `forward_stream` on every stage, including non-first stages.
The event recorded there orders after it. No change to the commit is needed.
