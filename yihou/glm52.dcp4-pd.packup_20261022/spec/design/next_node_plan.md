# Next-node plan (after 054 ends ~01:00Z 10-22): ordered PD experiments toward 20k+ tok/s/chip

Resources:
- One OAI 8× MI355X node (maybe a second) for PD scoring.
- Two v3 burst nodes for mixed/decodebench. They need `HSA_ENABLE_SDMA=0`, which changes the copy path, so PD numbers from them are not comparable with 054/OAI. Use them only for mixed A/Bs.

Code: `dev/yihou/glm52_dcp_pd` HEAD 5aacce1f1c, or later. Every new knob is env-gated and default off.

Per-run cost on a fresh stack, from ledger.pdrun r2/r4:
- bring-up about 25-35 min (prefill HiCache 450 GB host pools; graph capture about 3-5 min warm)
- AgentX warmup about 16-20 min, plus 3600 s profiling
- **about 1 h 50 min per fresh run; about 1 h 20 min for a client-only rerun on a live stack** (flush, then bench)

Reference: r4 A1b 17,473 (fresh, custom AR on); r2 A2b 17,053 (warm, scored). r4 A2 and C56 results are pending as of 22:30Z. The target needs about +15-17 %.

## E1. r5 = full validated knob set, C40 then C56 (first run on the new node)

| | |
|---|---|
| Config delta vs r4 | Decode: `SGLANG_DSA_DCP_KEEP_KV_B_BF16=1`; `--mem-fraction-static 0.90`; `--tokenizer-worker-num 4 --detokenizer-worker-num 4` (if absent); `SGLANG_LOG_DECODE_SEQ_LENS=1`; graph bs 64 = max-running 64. Hang fix, idlefix and DP fixes are in the sha. Image `infera-sglang:v0522-1020-yihou-dcp-mc4340`: docker load from `PD/images/*.tar` and record the new image ID. |
| Expected | kv_b bf16 −1.6..−1.7 ms/step at N 16-32, 34K (benchrun T10, measured). Custom AR is already in r4. Mem and workers: no measured step effect. Overall about +2-3 % over r4: **est.**, from 69.4 → ~67.7 ms/step at ~20 running, assuming throughput ∝ 1/step. C56 adds concurrency. Mixed step time grows sublinearly with N (T7: 53 → 59 ms from N 24 to 32), so more tokens per step; size of the gain unmeasured. |
| Risk | Graph capture memory with mem 0.90 + custom AR + q-rep buffers (q-rep measured +0.34 GB weights; when its buffers are allocated relative to KV sizing is unverified). The hang fix is untested on GPU, so run the cancel repro (`tools/pd_cancel_repro.yihou.py`, about 3 min) right after the first bench. |
| Fallback | Drop kv_b (unset env). Mem 0.85. If the decode raises "transfer queues diverged", set `SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE=0` for the remaining runs. |
| Time | Fresh C40: about 1 h 50 min. C56 client-only: about 1 h 20 min. |

## E2. PD decode DP2 + DCP2 (only after mixrun's graph-mode DP2+DCP2 run is clean and its decodebench beats DCP4)

| | |
|---|---|
| Config delta | Decode `--tp 4 --ep 4 --dp 2 --enable-dp-attention --dcp-size 2`. Prefill unchanged (DPA4). Router `PD_DP_RANK_AFFINITY=0`, as for DCP4. |
| Rank mapping (code read) | Decode attn_tp 2 > prefill attn_tp 1, so `common/conn.py:1368-1378` gives `required_dst_info_num = 2` and target prefill tp rank 0 of the request's prefill DP group. The DCP1→DCP2 relayout is the same per-token plan with W = 2. The prefill DP rank comes from `disagg_prefill_dp_rank` / room % 4 (router 68f06e29, analyst read). The decode DP group is chosen by the router/load balancer. Affinity must stay 0: affinity would pick decode DP = prefill DP (0-3), and DP2 has only 0-1. |
| Hang fix under DP2 | The deferred-release reduce and the transfer-queue length reduce run on `attn_tp_group.cpu_group`, i.e. per DP group (2 ranks). Each DP group has its own queues, so the consensus stays within the group (code read). The cross-DP `prepare_mlp_sync_batch` collective is separate and unchanged. |
| Needed first | A CPU test: DPA4 prefill fan-out to an attn_tp 2 / DCP2 decode, a W=2 variant of `TestDpaPrefillFanOut` (about 30 min, analyst). pdBe-style gate: sim off, math/passkey, eager → graph. |
| Expected | Unknown. Attention collectives run over 2 ranks on half the rows, while MoE still spans 4. Per-DP KV pool is 4.55M tokens (mixrun R12), vs one 8.89M pool for DCP4. Decision input: mixrun decodebench DP2+DCP2 vs DCP4 (pending). |
| Risk | Three DP-padding bugs were just fixed (§4 of RESULTS.md). Graph mode is untested. The PD prebuilt + IDLE batches on an empty DP group are a new path. |
| Time | CPU test 0.5 h; gate + C40 about 2 h. |

## E3. Split-triton decode indexer (only if benchrun's graph A/B has split-triton < ref at 150K)

| | |
|---|---|
| Config delta | `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1 SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1` on the decode (sha 0347a6d47e+). |
| Expected | Eager est. −3.5 ms/step vs split-torch at 150K, i.e. below ref; about +2.5 ms vs ref at 34K (benchrun 21:39Z eager kernels). The AgentX benefit depends on the context distribution: run `tools/agentx_context_load.yihou.py` on r2/r4 first (prior C40 sample: about 181 K/row). Radix-select merge (est. 102 → 25-40 µs/layer) if the A/B is positive. |
| Risk | Gate passed (graph, sim off, 40/40); not yet run in PD. |
| Time | Client-only? No: decode restart needed. About 1 h 50 min, or decode-only restart about 1 h 30 min. Note r3: decode-only restart under a live prefill coincided with "no content" errors, cause not established. Prefer a full restart. |

## E4. Phase 6: the other session's optimizations + cross-node (135/136)

| | |
|---|---|
| Config delta (prefill) | GTT HiCache (#42553 in base) with `HSA_USERPTR_FOR_PAGED_MEM=0`, `--hicache-size 260`, OffloaderV2 G4K1 step 1 (needs c9394c114f, in the sha), chunk 32768. Do NOT mount their files (`design/phase6_merge.md` §0). |
| Config delta (decode) | As E1. index_share: no measured step effect (T11), keep the infera setting. |
| Cross-node | A DPA4 (or PP4) prefill fans out to all 4 decode ranks, so 3 of 4 pairs are off-rail. The other session measured that any GPU reaches the peer when BOTH ends pin the same ionic_k (poll 19:48Z, second-hand). Options: (a) **TP4 no-DPA prefill**: decode rank j pairs with prefill rank j (`conn.py:1363-1367`), rail-aligned with the existing per-GPU NIC map; simplest, but the prefill throughput is untuned. (b) Per-pair common rail: needs multi-NIC registration per rank and a per-destination NIC choice; Mooncake support unverified (`MC_ENABLE_DEST_DEVICE_AFFINITY` never exercised off-rail). Do (a) first; (b) only with a Mooncake code read + a rail probe. |
| Expected | Their same-node baseline idxf1a 21,814 (DPA decode). Effect on our DCP decode: unmeasured. |
| Risk | Offload G4 is untested with our overlay (is_shuffled port is CPU-tested only); TP4 prefill is an untuned arm; cross-node RDMA env from `config.yihou.xnode.sh`. |
| Time | First cross-node bring-up + gate about 1 h, then about 1 h 50 min per scored run. |

## E5. Phase 7: PP prefill (PP4 × TP1) → DCP4 decode

| | |
|---|---|
| Config delta | Prefill `--tp-size 1 --pp-size 4`, `SGLANG_PP_LAYER_PARTITION=20,20,20,18`, triton DSA (`design/pp_prefill.md` §8 first-launch checks). Same node only, because PP stages fan out to all decode ranks, which is off-rail cross-node. |
| Code | 6ee35f3617 (non-final chunk wait event) + 646dc699f3 (stage tests), CPU only. PP + DSA triton on ROCm and HiCache + PP: never run. |
| Expected | ATOM reference #4045: PP4 prefill + DCP4 decode, 1P1D, 31,067 tok/s/GPU at c56 (PR body, different engine). No SGLang data. |
| Risk | High (first run of the combination). |
| Time | Bring-up + gate 1-1.5 h, then a scored run. |

## Order and budget (one node, about 12 h)
1. E1 C40 (fresh) + cancel repro + E1 C56 (client-only): about 3.5 h.
2. E3 if positive: about 2 h.
3. E2 if mixrun's graph DP2+DCP2 is clean and faster: about 2.5 h.
4. E4 (needs 135/136 from the other session) and E5 in parallel on a second node if available.

Burst v3 nodes in parallel: mixrun/benchrun A/Bs that feed E2/E3 (DP2+DCP2 graph decodebench, split-triton radix select, decodebench at AgentX-like contexts).
