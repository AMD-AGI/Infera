# Optimizations in the dcpx / dpax rounds (cross-node 135 prefill -> 136 decode, GLM-5.2-MXFP4, AgentX C56)

Each entry: what, measured effect (one A1 run unless said otherwise; node-to-node / run-to-run spread is ~3 %), how to
apply, caveats. Knobs refer to `scripts/harness/config.yihou.dcpx.sh`.

## 1. DCP pack fix (prefill)  - the decisive one for DCP
- What: `disaggregation/common/dcp_pack.py` `try_pack_dcp_src`. Stock: pageable H2D of row indices + `torch.tensor(
  metadata, device=)` inside `copy_mla_rows_into_pack`, gather stream waits on the device default stream, then
  synchronize. Fixed: row indices pinned + non_blocking on the gather stream; metadata built once per (ptrs, widths,
  strides) and kept on the device, pack offset = prefix * n in the kernel; gather on `torch.cuda.Stream(priority=-1)`;
  no default-stream wait.
- Measured (`[yihou-xfer]`, all 4 prefill ranks, profiling windows): ms/pack 77.6 (dcpx8c) -> 40.9 (no-wait only, dcpx9;
  sub-timers launch 20.8 + sync 19.9) -> 0.6 (dcpx10; launch 0.06, sync 0.36). pack share of send time 91 % -> 9 %.
  KV transfer p50 12.3 s -> 0.93 s; decode running ~17 -> ~23; score 19,997 -> 24,753 (+23.8 %).
- Apply: `DCPX_SHARD=xstats DCPX_PACKNOWAIT=2` (mounts `patches/dcp_pack.yihou-nowait2.py` + stats conn). Kernel output
  equality vs stock checked on one GPU (`patches/tests`).
- Caveats: which half (cached metadata vs priority stream) matters is not separated. Ordering now relies on
  `TransferKVChunk.wait_event` or on the chunk being queued after the forward (same reasoning as upstream
  `staging_buffer.producer_stream`, sglang 8e3e2980db). Under a PP prefill (overlap off) the early cached-prefix send has
  no wait event - run a correctness check before trusting a PP + no-wait setup. Upstream PR planned.

## 2. Transfer queue shard by hashed bootstrap_room (prefill)
- What: `MooncakeKVManager.add_transfer_request` picks the queue by `sum(dst session ports) % 4`; with a DP1 (DCP) decode
  the set never changes, so every chunk of a prefill rank lands on one worker that sends to the 4 decode ranks one after
  another. `bootstrap_room % 4` is no fix: it equals the prefill DP rank (data_parallel_controller.py:883). Fixed:
  `((room * 0x9E3779B97F4A7C15) >> 32) % 4` when a destination needs DCP relayout (all chunks of a room stay in one queue).
- Measured: py-spy 1 -> 2-4 active workers per rank; KV transfer p50 21.3 -> 11.3 s; score 19,691 -> 21,302 (dcpx6 ->
  dcpx7b). The room%4 attempt (dcpx7) changed nothing.
- Apply: `DCPX_SHARD=hash` (or `xstats`, which includes it). Open upstream PR #42163 does the same with splitmix64.

## 3. Prefill MTP (llying)
- What: prefill carries the decode's EAGLE flags (`--speculative-algorithm EAGLE --speculative-num-steps 5
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`). Without it the decode's draft-layer KV for prompt tokens
  is never written and every new request is seedless (first draft step eager; under DPA one seedless request makes all
  DP ranks eager) - code read, see the workflow notes in `notes.md`.
- Measured: dcpx7b 21,302 -> dcpx8c 19,997 (-6 %), but dcpx8c also added the stats logging; not separated. Its value is
  correctness of the draft input, which sim acceptance hides.
- Apply: `DCPX_PMTP=1` (also exports `PREFILL_MTP=1`; `audit_live` and `agentx_env` were taught to expect EAGLE on the
  prefill). Prefill must set `--max-running-requests` (else capped at 48 with EAGLE).

## 4. Decode cuda graph 128 -> 96, 13 capture sizes (DCP decode)
- Measured: free device memory after init 24.12 -> 28.47 GB at mem 0.85; score 24,753 -> 25,197 (+1.8 %, inside spread);
  TPOT 17.7 -> 18.4 ms. Apply: `DECODE_GRAPH_MAX_BS=96 DCPX_GRAPH_BS=1,2,4,8,12,16,24,32,40,48,64,80,96`.

## 5. Decode mem 0.85 -> 0.87 (DCP)
- Measured: KV tokens 8.87 M -> 9.26 M, 23.24 GB free after init, ran clean; score 24,651 (no gain at C56, KV not
  binding). Above 0.87 needs the other session's empty_cache patch (`../glm52.dcp4-decode-gmu.packup_20261023`).

## 6. DPA4 decode on the same v0522 stack (dpax1) = current best 26,321
- Same image/overlay/prefill/rails/stats as dcpx10; decode TP4/DP4 DPA, mem 0.95, max-running 96, graph 64, router
  affinity on. KV per DP rank 3.04 M tokens (12.2 M total) vs DCP 2.22 M x 4 ranks = 8.87 M at 0.85. Vs g3x1 (v0521) the
  +3.1 % mixes image, prefill MTP, Mooncake dest-local rails and MC_MAX_MR_SIZE.

## Not optimizations but needed
- Mooncake `-destpin` image (dest-local rail, pinned HCA, MC_IB_TIMEOUT=18) for cross-node DCP fan-out.
- decode mem <= 0.87 without the empty_cache patch (0.90 / 0.92 abort `HSA_STATUS_ERROR_OUT_OF_RESOURCES`).
- Round names must be new while stopped containers are kept (harness refuses existing names).
