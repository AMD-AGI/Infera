# Notes (what / why / how)

1. **Noise.** llr14 vs its replicate llr14r: 43,043 vs 42,366 (-1.6 %). Treat any single-run difference below ~2 % as
   unresolved. The AgentX closed loop also shifts the ISL mix by 1-4 % per run; report qps / TTFT / TPOT / queue series
   and ISL with every score.
2. **Decode graph capture.** Per-rank capture is capped at max_running / dp = 64 (llr7 captured 27 sizes 1..64 despite
   max bs 256). A coarse list (11 sizes) costs decode step time (50.5 -> 60.1 ms) because batches of 9-23 pad up to the
   next size with 6 verify rows per request; it frees only ~0.9 GB. Keep fine steps in the 6-24 range.
3. **Decode mem fraction.** 0.95 aborts (llr8, 2.2 GB free after capture). 0.93 (7.66 GB free) survived five runs but
   aborted in llr17 when the decode admitted more (DP3 running 33, usage 0.98): HSA_STATUS_ERROR_OUT_OF_RESOURCES
   "Available Free mem: 0 MB". Other session (S1m, DPA4 0.92 same-node): torch reserve rose ~1.4 GiB/rank above live
   tensors. 0.92 is the robust value; above it needs the empty_cache port (`analysis/ec_port/`, not run).
4. **pp-async-batch-depth.** 1 helps (+2.5 %, prefill PP0 1.74 -> 1.99 batch/s, TTFT p90 -16 %); 2 hurts
   (prefill_forward p50 +22 %). Overlap schedule is forced off for pp > 1 (`overrides.py:1625`); the kit's
   `--disable-overlap-schedule` is redundant.
5. **Decode radix cache.** Works mechanically (transfer MB p50 864 -> 45) but needs session -> decode-rank stickiness
   (`--session-affinity both`), which overrides demand balancing and unbalances ranks (prealloc waits up to 0.42 of the
   time on one rank). Needs a router change that trades hit vs load before it can pay off.
6. **HiCache io direct** on this host/image: prefill collapses (0.21 batch/s). Keep `kernel` / `page_first`.
7. **KV transfer latency** is a load-dependent floor (~4 PP iterations), not bandwidth: see
   `analysis/kv_transfer_latency_data.md`. llr17b (chunk 4096) tests the iteration-period hypothesis.
8. **Ops lessons.** guard v1 stopped llr9b in warm-up (the decode logs no batch lines during warm-up); v2 arms after the
   profiling phase starts. `pkill -f '<pattern>'` inside `ssh bash -c '...<pattern>...; docker stop ...'` kills its own
   shell: the llr16 stop silently failed once (30 min lost). Stop by PID.
9. **Not correctness-checked:** custom AR, KF, async depth, stream-interval, CLIP 8 all ran with sim acceptance.
10. **Unexplained:** AgentX gpu_hit swings 0.37-0.50 between runs while the prefill log hit stays 0.964.
