# KV transfer latency: DATA analysis (verified; the CODE part hung and was stopped)

Verification of the DATA report on the llr11/14/12 KV-transfer latency

I re-read every code citation in image `v0522-fdbafb60` with `docker run --rm` on 136, CPU only, using the extracted copies under `/tmp/yihou/xferlat/`. I re-ran or recomputed the numbers from the saved `*.pairs.npy`, `*.win60.npy` and `*.pp0batch.csv.gz`. My check scripts are in `/tmp/yihou/xferlat/v/`, on local a16-19 only. I deleted nothing and did not touch the `yihou-llr-*` containers.

## Definitions

- **CONFIRMED.** If Mooncake leaves `transfer_latency_s` unset, latency = `completion_time - prefill_transfer_queue_entry_time` (`req_time_stats.py:967-975`). The "only capture the last chunk time" comment is at line 972.
- **CONFIRMED.** `kv_transfer_speed_gb_s` = (total_mb/1024)/latency (`req_time_stats.py:985-986`). It is a whole-request size divided by the last-chunk interval, not a link speed. Units are GiB/s.
- **CONFIRMED, with a precision added.**
  - The entry time is set when this PP rank processes the batch result of the last chunk, right after `send_kv_chunk(last_chunk=True)` (`prefill.py:1029-1031`).
  - The finish time and `completion_time` are set inside `process_disagg_prefill_inflight_queue` on PP consensus Success (`prefill.py:1199, 1208-1209`).
  - Metrics are observed right after that (`prefill.py:1219-1220`).
  - So both the latency and the `prefill_transfer_kv_cache` stage include the wait for polling and PP consensus.
- **CONFIRMED.** `total_mb` is accumulated across chunks (`common/conn.py:2050-2069`). The report did not mention that it is also multiplied by `get_kv_replica_factor()`, which is above 1 only for MLA replication (`conn.py:2056`, `962-977`). The factor's value in this run is UNVERIFIED.
- **CONFIRMED.**
  - Release happens only at `scheduler_pp_mixin.py:341-342`, gated on `tmbs[next_mb_id]`.
  - Each rank intersects with the previous stage's transferred set (`:611-634`). The last rank then sends release rids round the ring (`:659-679`).
  - The loop has `pp_loop_size = pp_size + pp_async_batch_depth` slots (`:558-561`). That is 4 + 1 = 5, because the engine config sets `--pp-async-batch-depth 1`.
- **CONFIRMED.** Latency and MB are logged in one call (`metrics_collector.py:1423-1431`), so 1-second, count==1 timeslices give exact per-request pairs.
- **CONFIRMED.** The allocation policy is `early` (from the logs). `#optimistic-req` is 0 in all 10,295 PP0 batch lines, so the pending_bootstrap path at `prefill.py:1156` never ran.
- **PARTLY WRONG.** `decode_transferred` starts after `send_metadata` (`decode.py:1830-1831`). Under the `early` policy that point is before the prefill forward. So it covers prefill queue, forward and transfer, not just "the prefill-side wait". That is consistent with its mean of 6.43 s.

## Numbers

- **1) Pairs: CONFIRMED.**
  - Counts: llr11 has 2,508 pairs out of 44,160 observations.
  - Fits: OLS 2,099 + 0.099·MB (r 0.095), 2,067 + 0.079·MB (r 0.072), 1,846 + 0.277·MB (r 0.20).
  - Bin medians match.
  - llr12 with MB < 10: n=492, p10/p50/p90 = 12 / 1,041 / 3,031 ms.
  - The <100 ms fractions of 11 / 13 / 17 % match. They are the same on all four PP ranks: 10.6-12.7 / 12.9-13.6 / 16.7-17.7 %.
  - The bucket shares (4.6 / 5.1 / 6.5 % under 100 ms; 23.4 / 24.5 / 28.9 % under 1 s) match.
- **"Large FIXED floor at every size": WRONG as worded.** The floor does not depend on size, but it does depend on load.
  - In llr11 minutes 1-4 (decode usage about 0.1, prefill_forward 0.53-0.86 s), the average latency is 205-408 ms.
  - llr14 and llr12 show the same: 270-566 and 200-592 ms.
  - The ~1.5 s floor exists only at the C80 operating point.
- **Missing from the report.** On 60-second windows (usage > 0.3), corr(latency, MB) is 0.42 / 0.43 / 0.75. MB itself correlates with prefill_forward at 0.49 / 0.49 / 0.76, so this is confounded by load. It is not a per-request size effect, since per-request r ≤ 0.2.
- **Iteration period: CONFIRMED.** 433 / 431 / 433 ms. The latency-vs-batches table and the `#inflight-req` table reproduce exactly.
- **Latency in iteration units: approximately confirmed.** My p25/p50/p75 is 3.2/4.26/5.05, 3.3/4.36/5.09 and 2.77/3.94/4.88, against the report's 4.5/4.5/4.1. I used a 4-second window, which explains the difference. This is about pp_size to pp_loop_size (4-5) iterations.
- **2) Prefill transfer vs prefill forward: CONFIRMED.**
  - Correlations 0.951 / 0.959 / 0.927.
  - Fits -186 + 934·pf, -151 + 904·pf, -251 + 894·pf.
  - Means by PP rank 1.930 / 1.868 / 1.826 / 1.788 s.
  - Stage means match.
  - I did not re-check that llr12's bootstrap mean is 1.93 s (UNVERIFIED).
- **3) Per-request timestamps:** I accept that none exist; the prefill log has no per-request transfer lines.
- **4) Time of run: CONFIRMED within split noise.**
  - My thirds for llr11 are 1,424 / 1,739 / 2,728 ms, against the report's 1,378 / 1,778 / 2,665.
  - At usage ≥ 0.88: 2,626 ms and 3.01 s.
  - Correlation of the fit residual: 0.25 / 0.27 / 0.05 with decode usage and 0.17 / 0.25 / 0.18 with the decode transfer queue.
  - New check: the latency/forward ratio is 0.80 when the decode prealloc queue is under 0.2 and 0.84-0.86 when it is ≥ 1. The decode-side effect is small.
- **The fast mode (<100 ms) stays UNKNOWN.** The mean PP0 batch count in [t-3, t] is about 8 for fast pairs, the same as for slow pairs. Plain iteration-count arithmetic does not explain it.
- **Housekeeping.** The NFS copy exists in `/shared_nfs/yihou_glm52/llrepro_20261023/xferlat.yihou/` (checked from 135). Contents differ from the local set:
  - NFS has `*.extract.log` and an uncompressed `*.metrics.json`.
  - NFS has no `pairs`/`win60`/`pp0batch` files; those are only in the local `/tmp/yihou/xferlat/`.
  - The stray `/tmp/*` overwrite on a16-19 is as reported; I did not inspect it.

## Top 3 candidate causes of the ~1.5 s floor at C80

1. **Release is held for the PP consensus round trip (INFERENCE, strongest).**
   - What it is: release needs pp0 to poll Success, then the intersection pass 0→1→2→3, then the ring back. That is roughly pp_size to pp_loop_size scheduler iterations of about 430 ms each.
   - Why I suspect it: latency is a near-constant ~4 iterations across loads, it has a slope of about 0.9 on prefill_forward, and it is about 1 s even when MB < 10.
   - Cheapest test: change only `--pp-async-batch-depth` from 1 to 0, which makes pp_loop_size 4. If latency in iteration units drops by about 1, consensus is the cause.
   - Alternative: halve `chunked_prefill_size` (8192 to 4096). If latency in ms roughly halves while staying near 4 iterations, the iteration period is the driver.
2. **The scheduler iteration itself is long (INFERENCE).**
   - What it is: about 430 ms per PP0 batch, including the blocking DtoH page-index copies seen in llr7r (125-129 per stage, 486-1566 ms recorded).
   - Why it matters: every poll or consensus hop costs one iteration, so shorter iterations would shrink the floor in proportion.
   - Cheapest measurement: correlate per-iteration CPU stall time (a py-spy dump or the existing llr7r trace) with iteration period. A/B a build that removes the blocking `.cpu()`/`.tolist()` on page indices, holding everything else fixed.
3. **Actual completion of the Mooncake last chunk is late, from queueing on transfer worker threads or per-request fixed work such as aux/state/index-K/draft metadata (UNKNOWN).**
   - Weakened by: the llr12 MB < 10 latency, and fast-mode pairs moving about 1 GB in under 30 ms.
   - Not excluded, because no artifact records when the local poll first turned Success.
   - Cheapest decisive measurement: a log-only patch in `prefill.py` that records, per rid and PP rank, (a) the first local `KVPoll.Success` time inside `get_rids` (`scheduler_pp_mixin.py:1673-1693`) and (b) the release time at `prefill.py:1199`. If (a) minus entry is small and (b) minus (a) is about 4 iterations, cause 1 is confirmed. If (a) minus entry is large, the cause is in Mooncake.