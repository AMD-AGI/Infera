# Notes — findings, traps, open questions (what / why / how / context)

## 1. dcpx6 A1 vs g3x1 A1 (C56, same prefill, same nodes)
| | dcpx6 (DCP4 decode) | g3x1 (DPA4 decode) |
|---|---|---|
| score tok/s/chip | 19691 | 25524 |
| client TPOT mean / p90 | 15.6 / 19.1 ms | 15.9 ms mean |
| TTFT p50 / p90 | 19.8 / 45.5 s | 4.3 / 22.0 s |
| kv_transfer_latency p50 / p90 | 21.3 / 39.5 s | 1.0 / 1.95 s |
| prefill transfer_kv stage p50 / p90 | 18.3 / 49.4 s | 1.06 / 1.97 s |
| kv_transfer_total_mb p50 (metric) | 25,345 | 4,529 |
| kv_transfer_speed_gb_s p50 (metric) | 1.17 | 4.41 |
| decode running (log lines, 55 min) | p50 14, p90 27, max 44 | ~24 (4 DP sum) |
| prefill inflight gauge, per DP | 4.7 / 6.1 / 5.7 / 9.0 | ~1.0 each |
| prefill waiting queue gauge, per DP | 0.18-0.6 | 2-2.75 |
| decode transfer-queue gauge | 26 | ~3 per DP |
| GPU prefix hit | 0.949 | 0.906 |
- What: per-request decode is faster under DCP (93.8-100 tok/s/req vs 72.7-80), but half as many requests decode at once; the
  KV-transfer stage takes ~20x longer and requests pile up in prefill-inflight / decode-transfer queues.
- Not yet known: whether the transfer itself is slow (bytes/s on the wire) or requests wait before/after it (decode
  allocation, bootstrap, sender worker serialisation); whether `kv_transfer_total_mb` is inflated by the DCP replica factor
  (`common/conn.py:2131-2139` multiplies by `get_kv_replica_factor`). RDMA counters during profiling were not captured
  (the one 60 s sample fell into the A2 warm-up: ~0 bytes).

## 2. Memory: why decode mem 0.85 and what it costs
- dcpx3 (mem 0.92, 5.87 GB free after init) and dcpx4 (0.90, 11.11 GB free) aborted with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`
  7 and 12 min into profiling, right after triton kernels (`_fused_dsa_draft_extend_metadata_kernel`,
  `_fused_dsa_decode_metadata_kernel`, `alloc_extend_kernel`, ...) were first loaded while serving, with 0.01-0.67 GiB free.
  Tails in `results/rounds/dcpx{3,4}/decode-0.a1.log.tail200`. dcpx6 (0.85, 24.12 GB free) ran clean.
- The late kernel loads were seen only with index_share=true (dcpx3/4), not in dcpx2 (false). Mechanism not established.
- Weights: DCP decode weights are ~17.6 GB/GPU smaller than DPA4 (attention q_b/kv_b/o_proj split over attn-TP4 instead of
  replicated per DP rank; not EP vs TP). Graph + buffers outside GMU: ~13.4 GB at bs64, ~19.3 GB at bs128 vs ~4.8 GB DPA4.
- KV capacity: dcpx6 max_total_num_tokens 8,869,120 (one DP); g3x1 ~2.63M per DP x 4 = ~10.5M (a request lives on one DP).
  So at 0.85 the weight saving does not turn into more KV. Decode KV usage mean 0.786 in dcpx6 A1: capacity was not binding.

## 3. Rails (cross-node)
- v0522 Mooncake had no `MC_ENABLE_DEST_LOCAL_RAIL` / `MC_IB_TIMEOUT` (strings check), hence the `-destpin` rebuild.
- destlocal mode: both legs register ionic_0..3, `MC_ENABLE_DEST_LOCAL_RAIL=1`, `MC_IB_TIMEOUT=18`: the sender posts on the
  receiver's pinned HCA, so each decode GPU is fed through one NIC (llying measured fan-in drops with 4 NICs into one GPU).
- DCP fan-out: decode DP1 means `required_dst_info_num=4`; each prefill DP rank sends its KV to all four decode ranks
  (DPA4: prefill DP i -> decode DP i). No transfer failures and no per-token fallback in dcpx6 A1.

## 4. Traps seen
- Our base sets decode `NCCL_DEBUG=INFO`; DCP capture logged ~24k lines (dcpx1). Overridden with
  `DECODE_EXTRA_ENV_BASE=MC_LOG_LEVEL=INFO`.
- An inherited `PD_DP_RANK_AFFINITY=0` broke the xnode dryrun check; `config.yihou.dcpx.sh` clears it before the base.
- A mounts tsv without a trailing newline silently drops its last row.
- AgentX warm-up requests are max_tokens=1: decode `Decode batch` lines and gauges start only at profiling. (An earlier
  claim that 4 tokenizer workers drop decode logs was wrong.)
- `env.dcpx6` header comment is stale (see REPRODUCE §2).
