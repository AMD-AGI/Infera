# Why decode TPOT is higher cross-node (g4x1) than same-node (idxf1a/b/c)

## 1. What was measured

TPOT and qps are the benchmark numbers from the task brief. Client ITL comes from `aiperf_artifacts/profile_export.jsonl`. Running-req and gen throughput come from the 5 s `/metrics` series `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/<round>/metrics.{A1,A2}.tsv` over the aiperf profiling window. The latency proxy is token-weighted `sum(running)/sum(gen_throughput)`, computed by the skeptic in `/tmp/yihou/rev/a.py`. Its earlier check against client ITL agreed within about 0.2 ms.

| round / leg | idx_share | bench TPOT ms | client ITL mean ms | proxy ms | decode running, 4-DP sum, mean / p90 | per-DP running mean | busiest-rank running mean | gen tput sum tok/s | qps |
|---|---|---|---|---|---|---|---|---|---|
| g4x1 A1 (cross-node) | true | 15.7 | 15.72 | 15.32 | 23.5 / 35 | 5.4 / 6.5 / 5.4 / 6.3 | 9.69 | 1559 | 1.47-1.51 |
| g4x1 A2 | true | 15.7 | n/r | 15.54 | 23.0 / 35 | n/r | 9.76 | 1525 | 1.47-1.51 |
| idxf1a A1 / A2 | true | 12.2-12.7 | 12.21-12.72 (range over idxf1a/b/c) | 11.88 / 12.46 | 15.0 / 16.5, p90 24-27 | about 3.6-4.2 | 6.07 / 6.66 | 1285-1345 (range) | 1.19-1.25 |
| idxf1b A1 / A2 | true | 12.2-12.7 | (as above) | 12.42 / 11.70 | 16.1 / 15.4 | 3.7 / 3.9 / 3.9 / 4.0 (A1, log) | 6.34 / 6.00 | (as above) | (as above) |
| idxf1c A1 / A2 | true | 12.2-12.7 | (as above) | 11.90 / 12.07 | 15.4 / 15.3 | about 3.6-4.2 | 6.01 / 6.17 | (as above) | (as above) |
| g4n3 A1 / A2 | **false** | 15.1 / 12.8 | n/r | 14.79 / 12.35 | 18.5 / 16.6 | 4.5 / 4.4 / 3.9 / 4.5 (A1, log) | 7.12 / 6.66 | n/r | n/r |

Other measured facts:

- **Same in every leg:**
  - Spec accept length is 3.60-3.61 (simulated).
  - Decode `#queue-req` is about 0; retracted and paused are 0.
  - Output-length distribution is the same (OSL mean about 1011-1043, p50 436-446).
  - Client in-flight requests are about 40 everywhere (39.6-39.9 cross-node vs 41.1-42.6 same-node).
- **Where the in-flight requests sit differs.**

  | | g4x1 A1 / A2 | same-node idxf1a/b/c |
  |---|---|---|
  | requests in decode phase (client side) | 23.1 / 22.7 | 14.5-16.0 |
  | decode transfer queue | 15.8 / 16.4 | 24.3-26.4 |
  | prefill inflight queue | 4.95 / 5.05 | 14.3-16.2 |

- **Eager fraction (target-verify graph):** the decode log field `cuda graph: False` appears 0 times in every leg on every DP rank. That field covers only the target-verify graph. Draft-decode eager is not logged (code read, see section 3).
- **Graph capture limit:** CUDA graphs were captured for bs 1..24 per rank. The maximum observed per-rank running-req is 24, so no step exceeded the captured sizes.
- **Config:**
  - Decode server-info is identical between g4x1 and idxf1a/b/c apart from host, seed, pids and startup timings.
  - `json_model_override_args = {"index_share_for_mtp_iteration":true}` in g4x1 and idxf1*; g4n3 has `false`.
  - Container differences: decode GPUs 0-3 on NUMA0 for g4x1 vs GPUs 4-7 on NUMA1 same-node; RDMA filter; image ID e8e2838f under overlay2 vs 9d755dd6 under containerd.

## 2. Leading explanation

**At equal load, g4x1 decode costs the same per token as same-node. TPOT is higher because g4x1 decodes more requests at once.** The skeptic tried to refute this with a different method and failed.

**Measured: per-token cost is the same at matched load.** Binned by the busiest DP rank's running-req, g4x1 A1 vs the idxf1 range, in ms:

| busiest-rank running | g4x1 A1 | idxf1 range |
|---|---|---|
| [3,5) | 9.0 | 8.9-9.1 |
| [5,7) | 11.1 | 10.6-11.1 |
| [7,9) | 12.8 | 12.5-12.7 |
| [9,12) | 16.0 | 15.4-16.0 |
| [12,16) | 18.6 | 18.2-19.2 |

**Measured: the pooled same-node curve predicts g4x1 almost exactly.** Applying the pooled idxf1a/b/c curve to g4x1's own load distribution:

| leg | actual ms | predicted ms | residual |
|---|---|---|---|
| g4x1 A1 | 15.32 | 15.18 | +0.15 |
| g4x1 A2 | 15.54 | 15.46 | +0.08 |
| idxf1 legs (noise floor) | | | -0.17 to +0.13 |

62-69 % of g4x1 tokens are generated at busiest-rank running ≥ 9, against 13-24 % for idxf1.

**Measured independently from log-line step time.** Step time = the gap between consecutive "Decode batch" lines of a rank / 40.

- Overall mean step: g4x1 50.3 ms vs idxf1a/b/c 39.7-40.1 ms. The ratio 1.27 matches the TPOT ratio of about 1.27.
- Bucketed by busiest-rank running, per-bucket step times agree within about 1 ms across g4x1, idxf1b and idxf1c.

**Code-read: index_share is ON in g4x1.** Its loss is not the cause.

**Hypothesis, not tested: why more requests are decoding.** Cross-node KV transfer is faster (transfer_kv p50 about 1.1 s vs about 7.5 s, from the brief, not re-measured). With about 40 requests in flight in the closed loop, fewer wait in the transfer and prefill-inflight queues and more decode concurrently (23 vs 15). That raises per-step cost and TPOT, together with +15-20 % gen throughput and higher qps. No run has varied transfer speed alone or held decode running fixed.

## 3. Claims that are not supported

- **"Every step has cuda graph: True, so eager does not explain the gap."** The log field covers only target-verify (code read: eagle_worker_common.py:607 → eagle_utils.py:614 → metrics_reporter.py:756). The forced-eager draft-decode loop is invisible in the log.
  - Code read: every newly admitted request is seedless, because prefill runs with `speculative_algorithm=None` and writes -1 into the seed row. The next draft-decode step then runs eager on all 4 DP ranks (dp_attn.py:54-70, eagle_disaggregation.py:63-108, decode.py:2529-2534, 3020-3045).
  - Measured upper bound on such steps: about 12.5 % (g4x1) vs about 8.6 % (idxf1b).
  - The per-step eager cost is **not measured**. Eager is ruled out as the main factor only because the matched-load residual is about +0.1 ms.
- **The "unexplained 1.6-2.0 ms/step" and "0.4 ms/token" residuals.** Both come from binning by total running. Binned by busiest rank, the residual is inside the noise.
- **"DP ranks step in lockstep, so the busiest rank sets the pace."** This is consistent with busiest-rank binning fitting best, but it was not verified in code.
- **"Faster transfer causes the larger decode batch."** Hypothesis (see section 2).
- **"index_share works / is not implicated."** What was actually measured is weaker:
  - Per-token cost is equal between g4x1 and the idx=true same-node runs.
  - The idx=false leg g4n3 A2 also sits on the same curve (residual -0.24 ms). **At this granularity, these data show no measurable per-step benefit from index_share.**
  - g4n3 A1 is +1.82 ms above the curve. That is an unexplained outlier, not evidence for or against index_share.
- **Image e8e2838f (node 136) is content-identical to 9d755dd6.** This rests on a comment in the harness config. No checksum or `docker image inspect` was run.
- **Benchmark TPOT vs log step time.** Benchmark TPOT is about 1.5 ms/token above log step / 3.61. That mapping was not checked; only the ratio agrees.

## 4. Next measurements to settle what remains open

1. **Matched-load test of the mechanism** (fits the user's request to raise conc on 135/136 in steps of 4).
   - Sweep conc on cross-node 135→136, and on a same-node pair, for example conc 56→60→64…
   - Read decode `num_running_reqs` from the metrics TSV.
   - If same-node at a higher conc reaches a busiest-rank running of about 9.7 with TPOT about 15.3, and cross-node at a lower conc reaches about 6.2 with TPOT about 12, then load is the cause and the transfer→batch link is confirmed.
2. **index_share A/B at matched load on 135/136.** Run idx=true vs idx=false with everything else fixed, compared per busiest-rank bin. This is needed because the current data show no per-step gain, and g4n3 A1 vs A2 disagree with each other.
3. **g4x2 replicate.** Once it has decode data (its metrics.A1.tsv showed decode running 0 so far), rerun `/tmp/yihou/rev/a.py`. Expect total running about 23 and a residual of about 0.
4. **Forced-eager cost.** Add a counter and timer around the eager branch in `EAGLEWorkerV2.draft()`, or run py-spy or rocprof on one decode rank. That gives the number and cost of eager draft-decode steps. Separately, decide whether prefill should run EAGLE so that the DSA seed is sent.
5. **g4n3 A1 outlier** (+1.8 ms at equal load): compare it against A2 for node and NUMA placement, clocks, and co-running jobs.

## Rule deviation (from the env-hw investigation)

`spur006:/tmp/xnode_load.awk` (873 bytes) was written outside `/tmp/yihou` by mistake. It has not been deleted, per the no-delete rule. Nothing else was written outside `/tmp/yihou`.

Scripts and copied data are under `/tmp/yihou/tpotdb/`, `/tmp/yihou/cgeager/`, `/tmp/yihou/tpotcmp/scripts/`, `/tmp/yihou/rev/a.py` and `/tmp/yihou/xnode_env/`.