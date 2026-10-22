# T4 runbook: GLM-5.2 PD, DP-attention prefill (no DCP) -> DCP4 decode (ROCm, Mooncake)

Code: branch `dev/yihou/glm52_dcp_pd` at **`906b0915a9`** = `4792dc0e52`
+ 9 Infera recipe commits (`e75479a189`..`4d1d6a647e`; same changed lines as pdrun's
`overlay-vs-src.4792dc0e52.yihou.diff`)
+ the DCP commits:
- `a6bdc7894d` sharded-draft relayout (Mooncake; NIXL/Mori raise);
- `228263d116` guard: ROCm DCP decode allowed in PD;
- `ae529c96bf` env `SGLANG_DSA_DCP_PD_DROP_MTP_SEED`;
- `906b0915a9` unit test for the DPA fan-out (test only; python/sglang is identical to `ae529c96bf`).

The old sha `d36aee0e1e` is kept as ref `yihou/backup-glm52_dcp_pd-d36aee0e1e`.
Mount the branch's `python/sglang` directly; no recipe re-patching is needed. Both legs must run it.

Unit tests pass on CPU: `test/registered/unit/disaggregation/test_dsa_pd_dcp.py`, `test_dcp_pack.py` and
`test/registered/unit/test_dsa_dcp_backend_defaults.py` (27 passed, 18 subtests).

Eager-mode caveat from recipe `dsa_dp_sync`: when `seq_lens_cpu` is absent, the page table is built at
full `req_to_token` width (context_len columns, repeated per draft token for verify). With
`--disable-cuda-graph` at a large batch this costs memory and time. It is not DCP-specific; keep eager
steps at low concurrency.

## Main path (user decision 2026-10-21)

| Leg | GPUs | Configuration |
|---|---|---|
| Prefill | 0-3 | **The 22k known-good prefill, unchanged**: TP4 + `--dp-size 4 --enable-dp-attention`, EP1, no DCP, **no EAGLE (mode N)**. Same flags as the 22k e2g prefill: `--chunked-prefill-size 8192 --max-prefill-tokens 16384` effective, `--mem-fraction-static 0.85`, HiCache, `--enable-fused-qk-norm-rope`, `--enable-aiter-allreduce-fusion`, `--json-model-override-args '{"index_share_for_mtp_iteration":false}'`, triton DSA, fp8 KV. Use pdrun's harness env **pdBe**. |
| Decode | 4-7 | TP4 / EP4 + DCP4 + EAGLE 5/1/6, **no DP-attention** (DCP needs `attn_tp % dcp == 0`). Router: decode DP1, so PD DP-rank affinity off (pdrun P005 `PD_DP_RANK_AFFINITY=0`). |
| Acceptance | | Perf runs: decode `SGLANG_SIMULATE_ACC_LEN=3.61`, `SGLANG_SIMULATE_ACC_METHOD=match-expected`, `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token` (as in the 22k run). Correctness runs (passkey, GSM8K): no `SIMULATE_*`. Mode N keeps the target exact; only the draft is degraded. |

Debug-only shapes:
- Mode **S**: the prefill also runs EAGLE 5/1/6 and sends draft KV and seed. It exercises the
  sharded-draft relayout; use it for the R2 seed A/B and for real accept length.
- Arm **P1**: TP4 prefill without DP-attention (1:1 transfer).

### Main-path code read: one DPA prefill rank fans out to 4 DCP decode ranks

All of this is from reading code. The CPU tests below cover the mapping and the relayout; no GPU run yet.

1. **Rank mapping.** The decode has attn_tp 4 and the prefill has attn_tp 1, which takes the branch at
   `common/conn.py` `_resolve_rank_mapping` (`attn_tp_size > info.attn_tp_size`).
   - Every decode rank targets prefill tp rank 0 of the request's prefill DP group.
   - `required_dst_info_num = 4`, `required_prefill_response_num = 1`.
   - The prefill concludes a room after all 4 decode peers' aux sends.
   - The prefill DP rank comes from the request or `/query_dp_ranks`, the same generic path as the 22k run.
   - Test: `TestDpaPrefillFanOut.test_rank_mapping_fans_out_to_every_dcp_rank`.
2. **Latent KV.** Each decode peer registers its own `dst_dcp_rank` (0..3). `requires_dcp_relayout` is
   True for each (prefill DCP1 -> DCP4, MLA pool).
   - The transfer worker loops over the room's 4 peers in one chunk (`mooncake/conn.py`
     `reqs_to_be_processed = self.transfer_infos[room].values()`).
   - Each peer gets `send_kvcache_dcp` with its own owner rows.
   - The worker's pack buffer is reused peer after peer. This is safe because the gather is synchronized
     before the send and `_transfer_data` is `batch_transfer_sync`.
   - Test: `test_target_only_prefill_to_dcp4_decode_with_draft` (45 tokens, 2 chunks, all 4 ranks).
3. **Draft entries in mode N.** The prefill has 0 draft entries. The decode's NextN entry (layer 78) has
   no source and is skipped by the layer-id pairing; the same test asserts it stays untouched.
4. **Index K.** `maybe_send_extra` runs per peer on the last chunk. For an MLA pool with mismatched attn
   TP there is no head slicing (`slice_kv_heads` requires non-MLA).
   - So each decode rank receives the full index-K pages for the request: 4 copies from one prefill GPU,
     about 4 x 2.8 KB/token.
   - Each decode rank builds its own page list from widened slots; the TP ranks share the allocator, so
     the lists are identical.
   - Pairing proof: `TestIndexKPagePairing`.
5. **Pack buffers.** Each prefill rank allocates its own 4 (one per transfer queue) on its first DCP peer
   registration.
   - Size per buffer: `chunked_prefill_size` x 78 entries x 576 B = **351.0 MB** at chunked 8192 (mode N),
     so 1.4 GB per prefill GPU.
   - They are allocated after KV sizing, at the known-good `--mem-fraction-static 0.85`. Check for OOM at
     the first request.
6. **Per-token volume** (derived, not measured): about 45 KB latent striped over 4 peers, plus 4 x 2.8 KB
   index K, all leaving one prefill GPU per request. The 22k run sent about 45 KB + 2.8 KB from one DP rank
   to one DP rank.

## Environment (both legs; same as the known-good PD plus the mixrun DCP env)

```
SGLANG_USE_AITER=1  SGLANG_OPT_USE_TOPK_V2=true  SGLANG_TIMEOUT_KEEP_ALIVE=900
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800  SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
MC_GID_INDEX=1  MC_TE_FILTERS=yihou_no_such_rdma_device  MC_DISABLE_HIP_TRANSPORT=1
MC_ENABLE_DEST_DEVICE_AFFINITY=1  MOONCAKE_DISABLE_HIP_DMABUF=0  HSA_ENABLE_IPC_MODE_LEGACY=1
MC_LOG_LEVEL=INFO   # keep INFO: R4 (pack-buffer registration over HIP IPC) shows up here
```

- `SGLANG_OPT_USE_TOPK_V2`: the known-good PD used `false`; mixrun DCP4 uses `true` (InferenceX recipe).
  Main path: the prefill keeps the 22k env unchanged (`false`); the decode uses mixrun's DCP4 value. Note
  both in the log header.
- Give each container its own aiter JIT dir (mixrun R1 lesson).
- Correctness runs: do **not** set `SGLANG_SIMULATE_ACC_*`. Perf runs: set them on the decode only (see Main path).

## Prefill (GPUs 0-3) -- debug arm P1 / mode S template; the main path uses the 22k prefill (pdBe)

```
python3 -m sglang.launch_server --model-path /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4 \
  --trust-remote-code --tp-size 4 --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton \
  --mem-fraction-static 0.82 --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
  --max-running-requests 64 --watchdog-timeout 3600 \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-cache-report --enable-metrics \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device '{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3"}' \
  --disaggregation-bootstrap-port <BOOT> \
  [mode S:  --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6] \
  [arm P2:  --dp-size 4 --enable-dp-attention] \
  [HiCache as known-good, small for debug: --enable-hierarchical-cache --hicache-ratio 1.5 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --hicache-size 48]
```

- `--mem-fraction-static` is 0.82, not the known-good 0.85. The DCP pack buffers (4 queues x
  `chunked_prefill_size` x 45.5 KB/token, about 1.5 GB at chunked 8192, about 6 GB at 32768) are allocated
  on the first decode registration, after KV sizing. Grep `PD DCP pack buffers allocated` in the log.
- Keep `--json-model-override-args '{"index_share_for_mtp_iteration":false}'` only if mixrun DCP4 uses
  it; both legs and the reference must agree.

## Decode (GPUs 4-7)

```
python3 -m sglang.launch_server --model-path /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4 \
  --trust-remote-code --tp-size 4 --ep-size 4 --dcp-size 4 \
  --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton \
  --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --mem-fraction-static 0.85 --max-running-requests 64 --cuda-graph-max-bs-decode 64 \
  --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 --enable-cache-report --enable-metrics \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device '{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3"}' \
  --disable-custom-all-reduce \
  [step 1 only: --disable-cuda-graph]
```

Do **not** pass any of these on decode:

| Flag | Why |
|---|---|
| `--enable-hierarchical-cache` | rejected for DSA DCP on this branch (#45205 not merged) |
| `--disaggregation-decode-enable-radix-cache` | v1 scope |
| `--disaggregation-decode-host-receive-threshold < 1` | forces host_pool retraction, which is rejected with DCP |
| `--disaggregation-decode-allocation-policy prefill_complete` | rejected for `dcp_size != 1` |
| `--enable-dp-attention` | DCP needs `attn_tp % dcp == 0` |

Expected decode startup log:
- `dcp_size=4`;
- DSA backends `triton`;
- `max_total_num_tokens` ≈ 9.2M (mixrun DCP4 showed 9,267,456 at mem 0.85);
- draft pool not replicated (draft KV ≈ 2.4 GB per rank in mixrun).

Expected prefill log on the first registration: `PD DCP pack buffers allocated`. There must be no
`draft head slicing` error and no `geometry differs` error.

## Steps (main path first; stop at the first failure; capture both logs)

1. **Smoke, main path, eager decode, no SIMULATE.** One request "What is 13+16? Answer with the number
   only." → `29`. Then one 32K random-token prompt with `max_new_tokens 16`.
   - Prefill log: `PD DCP pack buffers allocated: 4 x 351.0 MB`, no Mooncake registration error (R4),
     no `falling back to per-token RDMA`.
   - Decode log: no `geometry differs`, `item length mismatch` or `Rejecting state transfers`.
2. **T4a, eager**: passkey at 8K / 34K / 96K → 3/3 each. Use the same prompts as mixrun's DCP4 check if
   available. A wrong index-K or latent relayout shows up here as garbled long-context recall.
   - Send each prompt several times so the router spreads them over different prefill DP ranks.
3. Repeat 1-2 with the decode CUDA graph on (remove `--disable-cuda-graph`). Capture can take ~30 min;
   watch the JIT cache dir.
4. **T4b**: GSM8K with sgl-eval 200 examples, GLM-5.2 preset, no SIMULATE, against the PD router URL.
   Compare with mixed DCP4 and the 22k-shape non-DCP PD reference, same harness and settings.
5. **Perf smoke**: SIMULATE on (3.61), AgentX short window at C16. Then go to the full protocol.
6. **T4f** (stress):
   - one 128K prompt (pack-capacity chunking; grep `pack` warnings);
   - 32 concurrent 32K prompts;
   - one prefill-side prefix hit (send twice, `flush_cache` on decode only in between).
7. Debug only, if acceptance looks wrong or for the R2 A/B: mode S (prefill + EAGLE).
   - Real accept length, 4 prompts x 400 tokens, temperature 0, vs the mixed DCP4 server (#44322: 3.47).
   - Then the decode with `SGLANG_DSA_DCP_PD_DROP_MTP_SEED=1`.
   - If accept length is about 1 without the env and normal with it, the PD seed path is broken under DCP.

## What to send back to analyst on failure

- The traceback, with both logs' last 200 lines.
- Prefill: `PD DCP pack buffers allocated` line, any `falling back`, `geometry`, `item length mismatch`,
  `state layout` or `Rejecting state transfers` lines.
- Decode: `max_total_num_tokens`, pool sizes, and any `dcp` warning.

## Mode N (prefill without EAGLE): what the DCP decode's draft gets

Code read, branch `ae529c96bf`. Nothing here has been measured yet.

1. **The prefill registers no draft KV entries.**
   - The prefill appends draft entries only when it has a draft pool (`disaggregation/prefill.py:297-312`).
   - So `num_draft_entries = 0`, and `dcp_draft_sharded()` returns False (`common/conn.py`, `num_draft_entries > 0` check).
   - The relayout pairs entries by layer id. The decode's extra NextN entry has no source and is skipped
     (`disaggregation/utils.py:1128-1150`, "n_dst may exceed n_src when the decode side runs speculative
     decoding and the prefill side does not").
   - The new sharded-draft code is therefore **not exercised** in mode N.
2. **The decode never computes draft KV for the prompt.**
   - The first decode batch is `PREBUILT`, which "never enters a model forward"
     (`disaggregation/decode_schedule_batch_mixin.py:23-35`).
   - The draft latent KV and draft index-K rows for prompt positions keep whatever the pool held
     (stale rows from earlier requests, or zeros).
   - Only tokens the decode generates get draft KV, written by draft-extend after each verify.
   - This is the same with and without DCP. Under DCP the stale rows are still valid slots on their
     owning rank, so nothing new goes out of bounds.
3. **The first draft input is zeros.**
   - The prefill writes `output_topk_p`, `output_topk_index`, `output_hidden_states` and the DSA seed
     only when `hidden_states_tensor` is set, i.e. only with EAGLE (`prefill.py:1009-1022`,
     `disaggregation/utils.py:602-623`).
   - The buffers start as zeros, and the seed as -1 (`disaggregation/utils.py:436-450`).
   - On decode, an all -1 seed becomes `dsa_topk_indices = None`
     (`speculative/eagle_disaggregation.py`, the `all(< 0)` check). That makes the first draft step
     not graph-compatible, so it runs eager.
4. **Consequence.** The target output is still exact (verify), but draft quality on long prompts depends
   on stale KV.
   - The 22k baseline used simulated acceptance, so this did not affect its score.
   - The real-acceptance pack-up `glm52-agentx-c40-realacc.packup_20260918` also had prefill
     `speculative_algorithm: None` (its prefill log). It measured accept length 2.15-3.07 per rank.
   - Mode S is the configuration where the decode draft sees the prompt.
5. **dsa_page_table_rows** (recipe `344765b3bc`) acts on decode MTP steps and does not depend on the
   prefill mode.

Checks that prove it (all cheap):
- **(a) Draft entry count in the prefill log.** `PD DCP pack buffers allocated: 4 x <MB>`.
  - The size is `chunked_prefill_size x sum(per-token bytes)`, at fp8 576 B/token per layer.
  - Mode S includes the NextN entry: 79 entries, 8192 x 79 x 576 B = **355.5 MB** per buffer at
    chunked 8192.
  - Mode N has only target layers: 78 entries, **351.0 MB**.
  - (The 78 + 1 entry count is from the GLM-5.2 config; if the log differs, report the number.)
- **(b) Real accept length at C1**, same 4 prompts x 400 tokens, temperature 0, mode S vs mode N on the
  same DCP decode. Mode N should be clearly lower on long prompts.
  - Also compare against a short prompt (< 64 tokens), where little stale prompt KV is involved.
- **(c) Decode log in mode N.** There must be no `dcp_draft_sharded` / `draft head slicing` /
  `geometry differs` error. The first step of each request runs the draft eager (no seed); this is
  visible as non-graph draft steps if graph debug logging is on.

## Phase 7 (PP prefill -> DCP decode): first-launch checks

See `design/pp_prefill.md` sections 5 and 8.
- **PP4 x TP1 with the triton DSA backend on ROCm has never been run.** Confirm each stage's layer range,
  `dsa_prefill_backend='triton'`, and that the first prefill completes on all 4 stages, before any
  correctness or perf work.
- The branch must include `6ee35f3617` (wait event for non-final chunks under PP).
