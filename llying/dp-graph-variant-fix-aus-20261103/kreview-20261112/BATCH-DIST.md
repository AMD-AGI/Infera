# Decode batch-size distribution from raw decode logs (GLM-5.2, SGLang PD-disaggregated, MI355X)

Regenerate everything (stdlib only, ~1 min, streams the logs): `python3 /perf_apps/liyingli/bench_agentx/kreview-20261112/scripts/logs_batch_dist.py`  
Hand check of the tick alignment against raw lines: `python3 /perf_apps/liyingli/bench_agentx/kreview-20261112/scripts/logs_batch_dist.py --check runs-n04-33/c208-1109T1152-kopt-pp8`  
Machine-readable copy of every number below (and more): `batch_dist.json` in this directory.

## Findings (written 2026-11-12 from the tables below; re-read the tables if the inputs change)

* At C208 the TP8/DP8 decode never runs anywhere near 208/8 = 26 requests per rank. Total running p50 is 47-50 with one PP8 prefill (kopt-pp8, MegaMoeTP mt4q2), 63 in the truncated MegaMoeTP run, 75-89 with two prefill instances (2P1D). Busiest-rank p50 is 10-15, p90 15-23, p99 21-28.
* TP8/DP8 graph batch ranges by engine steps: 1P(PP8)+1D runs 80% in 7-16 and 95% in 4-18 (5-18 for mt4q2; 7-18 / 4-20 for the truncated MegaMoeTP run); 2P1D runs 80% in 10-22 and 95% in 5-24 (the 25-minute dprof run 5-22 / 3-26). Request-weighted the ranges move up by one or two sizes: 7-16 / 5-20 for kopt-pp8, 12-20 / 6-24 for 2P1D. In MoE tokens (M = 32 x graph batch) by steps: 224-512 (80%) and 128-576 (95%) for the PP8 runs, 320-704 and 160-768 for 2P1D.
* TP4/DP4 sweep, 80% / 95% of engine steps: K=5 C32 2-5 / 1-6; K=5 C48 3-8 / 2-10; K=5 C64 6-16 / 2-16; K=5 C88 (30-minute window, mem-fraction 0.93) 6-20 / 4-24; K=3 C88 10-22 / 5-24; K=3 C104 10-22..24 / 6-26..30 (seven 60-minute runs; the two v2-1node runs with 10 s TTFT sit lower at 7-20 / 5-24); K=3 C112 10-22 / 7-28. With DP balance on, the upper ends drop: K=5 C32 2-4 / 1-5, C48 3-7 / 2-8, C64 6-12 / 2-12, K=3 C88 10-18 / 5-20, C104 10-20 / 6-22. The 1P2D runs (two TP4 engines) sit at 3-10 / 2-12 (C96) and 5-12 / 4-16..18 (C128).
* Captured sizes above 32 exist only on TP4 (40, 44, ... 64) and carry at most 2.4% of request-steps and 1.6% of steps in any run; sizes above 24 carry at most 13% of request-steps in the 60-minute runs. The busiest rank never exceeded the largest captured size and `cuda graph` is True on every line.
* A large share of the rows a graph executes is padding: real / padded rows is 0.58 for the two complete PP8 TP8 runs, 0.63-0.67 for the other TP8 runs, 0.59-0.75 for TP4 without DP balance and 0.78-0.82 with it. The busiest rank holds 1.42-1.66x the mean rank (p50) on TP8, 1.26-1.60x on TP4 without DP balance, 1.17-1.33x with it.
* Step time is a staircase in the graph batch at K=3 ('Median step-time increment' table): 6 to 7 costs +0.1 ms, 7 to 8 +0.9, 8 to 10 +4.3 (minimum +2.8 over 24 engines), then +1.5, +2.1, +1.1, +2.9, +1.7, +2.1, +1.5 for the following sizes. The largest single cost is the move from 8 to 10 (32 to 40 attention rows per rank), at the same graph batch on TP4 and TP8. One run (runs-n02-33/c104-1105T0937, fp8 per-token config) is non-monotonic: 12 is slower than 14-16 and 20 slower than 22. Which kernel causes this cannot be seen in these logs.
* Context per KV-holding request grows through the window: p50 95-116K tokens in the first 10 minutes, 165-250K in the last 10 minutes of the 60-minute runs (132-174K at the end of the 25-30 minute runs). Indexer cost measured early in a run is not representative of the end.
* The decode starts empty at the window start in every run (first `Decode batch` line 2-5 s after the profiling phase starts), so the first minutes of the scored window are a ramp at small batches (first 10 minutes: busiest p50 7 in the TP8 runs against 9-18 later).
* In the kopt-pp8 run each rank's KV pool is 92% full at the median (99% at p90) while the rank runs 6.3 requests and holds 11.2 more in the transfer queue; 9% of engine ticks have at least one rank with nothing running. In the two 60-minute 2P1D runs the pool is 71% full at the median with 4.8-6.8 requests in transfer.
* Check on the whole chain: logged accept length divided by the request-weighted step time (p50 and p90) reproduces the client interactivity p50 / p90 of every scored run to within 3.1 tok/s (ratio 0.99-1.03). The decode summary lines in the score files agree with these numbers once weighted by wall time instead of steps (their medians are 1-2 higher for that reason); their step-ms-by-busiest lines agree to within 1.2 ms.
* Not all group B runs are single-node: by the node IPs in engine-config.env the runs-n06-25 c104 runs, v2-4p4 and v2-1p2d have prefill and decode on different nodes (layout column).

## Method and what is / is not verified

* **Source**: only `logs/*-decode*.log` (`Decode batch` lines), `engine-config.env`, the AgentX phase markers and `agentx/agentx_conc*.json`. No earlier summary is used as input; the `score-*.txt` numbers appear only in comparison columns.
* **Steps per log line**: `decode_log_interval` is printed in each log's `server_args` (40 in every run here). In the source (`sglang/srt/managers/scheduler_components/batch_result_processor.py` + `metrics_reporter.py::report_decode_stats`) the per-rank counter `forward_ct_decode` advances once per processed decode batch and a line is logged when it is a multiple of the interval. A rank with no running request processes no decode batch, so it neither counts nor logs: a line is 40 of *that rank's non-idle* steps. Empirical check per run (`steps/tick check` in the detail tables): 40 x running x accept-len / logged throughput divided by the timestamp gap, on ticks where the rank's running count did not change, must be 1.0.
* **Capture list**: read from each log's `Capture target verify CUDA graph end. bs=[...]` line, not assumed. TP8 runs: 1-8, 10-32 step 2 (max 32 because max_running_requests 256 / 8 ranks). TP4 runs: 1-8, 10-32 step 2, **40, 44, 48, ... 64** (step 4 after 40; there is no 36). Rows per request come from `num_tokens_per_req` on the `Capture target verify CUDA graph begin` line (4 for K=3, 6 for K=5).
* **Graph batch is inferred, not logged**: graph batch = smallest captured size >= the busiest rank's `#running-req`. That all ranks pad to the busiest rank is taken from the task statement; the log only shows `cuda graph: True` (it is True on every line of every run, and the busiest rank never exceeds the largest captured size). Indirect support from the logs ('Padding check' in the detail sections): above 8, where only even sizes are captured, the step time grows about 0.3-0.7 ms from busiest 2k-1 to 2k (same graph) and about 1.0-1.7 ms from 2k to 2k+1 (next graph), in every run. It is not zero inside a pair, so either the step time also depends on the real request count or the +-20-step alignment blurs it; the logs cannot separate the two.
* **Alignment**: each `Decode batch` line of rank r is one rank-tick at its timestamp (Docker nanosecond prefix where present). The other ranks' running counts at that tick are taken from their log line nearest in time, before or after; if a rank has no line within `tol` = max(1.5 x median line gap, 1 s) it is counted idle (0 running). Ranks log at different phases of the 40-step cycle, so the nearest line is at most about 20 steps away. Every rank's lines are used as ticks; each is weighted 1 / (#ranks that logged within tol) so that 40 engine steps count once.
* **Parser / alignment check (2026-11-12)**: `--check runs-n04-33/c208-1109T1152-kopt-pp8` prints 5 ticks spread over the window with the raw line used for every rank. In all 5 the per-rank counts, total, busiest and graph batch equal what the raw lines say, and the nearest lines are within 0.72 s (half a 40-step interval). One of them (12:45:26.449, DP5) was re-derived independently with grep/awk on the raw log: [9, 8, 5, 8, 8, 9, 3, 7], total 57, busiest 9, graph batch 10. DP0 and DP2 log within 0.2 ms of each other in every sampled tick, which fits all ranks stepping together. Gaps in a rank's lines (DP3, DP7 in that run) have `#running-req: 1` before and after, which fits 'no line = idle'.
* **Weightings**: *ticks* = that tick weight (~ engine steps). *requests* = tick weight x total running at the tick (~ request-steps, i.e. output tokens; this is what per-request interactivity sees). *time* = tick weight x the rank's line gap (wall time spent at that graph batch; lines with gap > 10 s dropped).
* **Ranges**: '80% range' = the contiguous run of captured batch sizes with the fewest distinct sizes holding >= 80% of the weight (ties: most weight); the share actually covered is in parentheses.
* **Step time** = gap between consecutive lines of the same rank / 40. 'stable' = ticks whose graph batch was the same at the previous line of that rank. It is wall time per engine step, so it includes whatever the scheduler does between forwards. `opt/partial/c208-1110T2109` has no Docker timestamp prefix (1 s resolution only); for it the step time is 40 x running x accept-len / logged throughput on ticks with unchanged running count. In the other logs that estimate agrees with the timestamp gaps to a few tenths of a ms per graph batch (both columns are in the detail tables).
* **Link to the client score**: accept length / request-weighted step time (p50 and p90) reproduces the client interactivity p50 / p90 of every scored run to within about 2 tok/s ('TP4/DP4 sweep' table), so the step-time and weighting logic is consistent with what AgentX measured.
* **#token**: `max_total_num_tokens - (available + evictable)` of that rank's KV pool (`pool_stats_observer.py::_get_token_info`), i.e. KV slots of running requests plus requests in the transfer queue (their KV is pre-allocated: `pre-allocated usage` sums `extend_end` over the transfer queue, `disaggregation/decode.py::num_tokens_pre_allocated`). Requests in `#prealloc-req` hold no KV yet. So context per request is reported as #token / (running + transfer + queue) ('per holder'); the task's formula #token / (running + prealloc + transfer) and a running-only estimate (#token - pre-allocated usage x max_total_num_tokens) / running (usage is logged to 2 decimals, so +-0.5% of the pool) are in the JSON and detail tables. These are per-rank averages over requests, not per-request values. The source tree read is the kreview copy; it was not checked to be byte-identical to the images that produced the logs.
* **Window**: AgentX `Phase profiling (profiling) started` .. `sending complete` (the same markers `score_summary.sh` uses), taken from the run's own agentx log / `benchmark.log`. Decode activity starts at the window start in every run (warm-up produces no decode batches), so the ramp-up of running requests is inside the window. Per-run window source is in the detail section.

## Envelope by configuration (complete runs; min of the range starts .. max of the range ends over the runs of the group)

| decode | K (rows/req) | C | DP-bal | engines | total run p50 | busiest p50 / p90 / p99 | gbs p50 ticks / req | gbs 80% ticks | gbs 95% ticks | gbs 80% req | gbs 95% req | attn rows/rank 95% req | MoE M 80% req | MoE M 95% req | fill | ctx/req p50 K first / last 10 min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TP8/DP8 | 3 (4) | 208 | off | 5 | 47-89 | 10-15 / 15-23 / 21-28 | 10-16 / 12-18 | 5-22 | 3-26 | 7-26 | 5-28 | 20-112 | 224-832 | 160-896 | 0.58-0.67 | 108-114 / 174-216 |
| TP4/DP4 | 3 (4) | 88 | off | 1 | 44 | 14 / 20 / 27 | 14 / 16 | 10-22 | 5-24 | 12-22 | 7-26 | 28-104 | 192-352 | 112-416 | 0.74 | 103 / 210 |
| TP4/DP4 | 3 (4) | 88 | on | 1 | 43 | 13 / 17 / 20 | 14 / 14 | 10-18 | 5-20 | 10-18 | 7-20 | 28-80 | 160-288 | 112-320 | 0.82 | 95 / 210 |
| TP4/DP4 | 3 (4) | 96 | off | 2 | 18-19 | 7 / 10 / 14 | 7 / 7-8 | 3-10 | 2-12 | 5-12 | 3-14 | 12-56 | 80-192 | 48-224 | 0.67-0.68 | 102-113 / 184-191 |
| TP4/DP4 | 3 (4) | 104 | off | 10 | 34-58 | 12-19 / 19-26 / 25-35 | 12-20 / 14-20 | 7-26 | 5-32 | 10-30 | 6-40 | 24-160 | 160-480 | 96-640 | 0.67-0.75 | 95-105 / 134-223 |
| TP4/DP4 | 3 (4) | 104 | on | 2 | 46-47 | 14 / 19 / 23 | 14 / 16 | 10-20 | 6-22 | 12-20 | 7-24 | 28-96 | 192-320 | 112-384 | 0.80-0.80 | 96-99 / 167-171 |
| TP4/DP4 | 3 (4) | 112 | off | 1 | 49 | 16 / 23 / 29 | 16 / 18 | 10-22 | 7-28 | 12-24 | 8-28 | 32-112 | 192-384 | 128-448 | 0.75 | 105 / 170 |
| TP4/DP4 | 3 (4) | 128 | off | 2 | 21 | 8 / 12-13 / 18-20 | 8 / 10 | 5-12 | 4-18 | 5-14 | 4-18 | 16-72 | 80-224 | 64-288 | 0.59-0.64 | 95-100 / 231-250 |
| TP4/DP4 | 5 (6) | 32 | off | 1 | 9 | 3 / 6 / 8 | 3 / 4 | 2-5 | 1-6 | 2-5 | 2-8 | 12-48 | 48-120 | 48-192 | 0.70 | 114 / 243 |
| TP4/DP4 | 5 (6) | 32 | on | 1 | 9 | 3 / 4 / 6 | 3 / 3 | 2-4 | 1-5 | 2-4 | 2-6 | 12-36 | 48-96 | 48-144 | 0.78 | 112 / 245 |
| TP4/DP4 | 5 (6) | 48 | off | 1 | 16 | 5 / 8 / 12 | 5 / 6 | 3-8 | 2-10 | 4-10 | 3-12 | 18-72 | 96-240 | 72-288 | 0.70 | 116 / 207 |
| TP4/DP4 | 5 (6) | 48 | on | 1 | 15 | 5 / 7 / 9 | 5 / 5 | 3-7 | 2-8 | 4-7 | 2-8 | 12-48 | 96-168 | 48-192 | 0.81 | 114 / 197 |
| TP4/DP4 | 5 (6) | 64 | off | 1 | 29 | 10 / 14 / 20 | 10 / 10 | 6-16 | 2-16 | 8-16 | 5-20 | 30-120 | 192-384 | 120-480 | 0.69 | 115 / 172 |
| TP4/DP4 | 5 (6) | 64 | on | 1 | 27 | 8 / 11 / 14 | 8 / 10 | 6-12 | 2-12 | 7-12 | 5-14 | 30-84 | 168-288 | 120-336 | 0.82 | 112 / 167 |
| TP4/DP4 | 5 (6) | 88 | off | 1 | 43 | 14 / 19 / 26 | 14 / 16 | 6-20 | 4-24 | 12-22 | 6-24 | 36-144 | 288-528 | 144-576 | 0.74 | 98 / 132 |

## Summary (one row per decode engine)

Columns: per-rank / total / busiest are `#running-req`; `gbs` = implied captured graph batch; ranges give the share covered in parentheses; `fill` = real rows / padded rows (ratio of sums); `step ms` = wall time per engine step; `ctx/req` = #token per KV-holding request. p-values are tick-weighted unless marked req. Scores are read from each run's `agentx/agentx_conc*.json`. Group A = TP8/DP8 decode on its own node; B = TP4/DP4 decode (runs asked for); B-extra = other TP4 C104 runs found next to them. TRUNCATED = the profiling phase did not finish (crash / abort / partial copy), no score.

| run | grp | layout | K (rows/req) | C | DP-bal | window min | tok/s/GPU | out tok/s/GPU | intvty p50/p90 | TTFT p50/p90 s | total run p50/p90 | per-rank p10/p50/p90/max | busiest p10/p50/p90/p99/max | busiest/mean p50/p90 | gbs 80% ticks | gbs 95% ticks | gbs 80% req | gbs 95% req | fill | step ms p50 | ctx/req p50 K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| partial:c208-1110T2109 [MegaMoeTP(mt4q2),kopt] | A | 1P(8GPU,PP8)+1D(TP8/DP8) 2-node | 3 (4) | 208 | off | 60 | n/a | n/a | n/a / n/a | n/a / n/a | 50 / 70 | 2/6/11/29 | 7/10/15/21/29 | 1.66 / 2.07 | 7-16 (85%) | 5-18 (96%) | 8-16 (81%) | 6-20 (96%) | 0.58 | 30.9 | 176 |
| n04-33/c208-1104T2113 [prefill-hrrn] | A | 2P(4GPU,PP4)+1D(TP8/DP8) 2-node | 3 (4) | 208 | off | 60 | 51,804 | 382.3 | 80.6 / 69.1 | 6.88 / 17.11 | 75 / 100 | 4/9/14/29 | 7/13/18/23/29 | 1.42 / 1.69 | 10-20 (83%) | 5-22 (96%) | 12-20 (81%) | 7-22 (95%) | 0.67 | 35.5 | 168 |
| n04-33/c208-1109T1152-kopt-pp8 [kopt,max-total-tokens=3680000] | A | 1P(8GPU,PP8)+1D(TP8/DP8) 2-node | 3 (4) | 208 | off | 60 | 53,341 | 401.1 | 94.2 / 80.7 | 8.54 / 20.26 | 47 / 71 | 2/6/11/32 | 6/10/15/21/32 | 1.64 / 2.05 | 7-16 (84%) | 4-18 (96%) | 7-16 (85%) | 5-20 (96%) | 0.58 | 30.7 | 176 |
| n06-25/c208-1104T0523 | A | 2P(4GPU,PP4)+1D(TP8/DP8) 2-node | 3 (4) | 208 | off | 60 | 54,094 | 403.3 | 77.2 / 65.7 | 4.16 / 8.01 | 84 / 106 | 5/10/15/32 | 7/15/20/27/32 | 1.42 / 1.70 | 12-22 (81%) | 5-24 (95%) | 12-20 (82%) | 6-24 (95%) | 0.67 | 36.7 | 179 |
| n06-25/dprof-c208-1104T0758 | A | 2P(4GPU,PP4)+1D(TP8/DP8) 2-node | 3 (4) | 208 | off | 25 | 48,866 | 389.7 | 70.8 / 61.0 | 3.79 / 7.74 | 89 / 122 | 3/10/17/32 | 5/15/23/28/32 | 1.46 / 1.75 | 5-22 (81%) | 3-26 (96%) | 14-26 (81%) | 5-28 (97%) | 0.67 | 37.5 | 151 |
| partial:c208-1110T0900 [MegaMoeTP,kopt] **TRUNCATED** | A | 1P(8GPU,PP8)+1D(TP8/DP8) 2-node | 3 (4) | 208 | off | 28 | n/a | n/a | n/a / n/a | n/a / n/a | 63 / 87 | 3/7/13/27 | 6/12/17/22/27 | 1.52 / 1.85 | 7-18 (81%) | 4-20 (96%) | 10-20 (84%) | 6-22 (96%) | 0.63 | 33.5 | 153 |
| v2:v2-k3-hi-1106T0358-c88 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 3 (4) | 88 | off | 60 | 50,316 | 413.1 | 73.3 / 63.9 | 3.03 / 5.61 | 44 / 60 | 5/11/17/39 | 8/14/20/27/39 | 1.29 / 1.53 | 10-22 (82%) | 5-24 (95%) | 12-22 (82%) | 7-26 (96%) | 0.74 | 39.4 | 158 |
| v2:v2-dpbal-k3-c88-1109T0501-c88 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 3 (4) | 88 | on(400000) | 60 | 50,650 | 413.7 | 76.1 / 69.3 | 3.17 / 6.91 | 43 / 58 | 6/11/15/24 | 7/13/17/20/24 | 1.17 / 1.33 | 10-18 (80%) | 5-20 (96%) | 10-18 (86%) | 7-20 (95%) | 0.82 | 38.2 | 153 |
| v2:v2-1p2d-1106T1933-c96:decode | B | 1P(4GPU,PP4)+2D(TP4/DP4) 2-node | 3 (4) | 96 | off | 60 | 38,094 | 304.7 | 93.8 / 84.3 | 3.28 / 6.07 | 19 / 28 | 2/4/8/19 | 3/7/10/14/19 | 1.41 / 1.78 | 3-10 (87%) | 2-12 (97%) | 5-12 (85%) | 3-14 (97%) | 0.68 | 30.8 | 158 |
| v2:v2-1p2d-1106T1933-c96:decode1 | B | 1P(4GPU,PP4)+2D(TP4/DP4) 2-node | 3 (4) | 96 | off | 60 | 38,094 | 304.7 | 93.8 / 84.3 | 3.28 / 6.07 | 18 / 27 | 2/4/8/18 | 3/7/10/14/18 | 1.43 / 1.78 | 3-10 (86%) | 2-12 (97%) | 5-12 (84%) | 3-14 (97%) | 0.67 | 30.6 | 161 |
| n02-33/c104-1105T0937 [nohip,fp8-pertoken] | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 3 (4) | 104 | off | 60 | 53,176 | 434.9 | 63.4 / 55.4 | 3.67 / 7.36 | 52 / 72 | 6/12/20/38 | 9/16/24/31/38 | 1.26 / 1.51 | 10-24 (84%) | 6-30 (96%) | 12-24 (82%) | 10-30 (95%) | 0.75 | 42.9 | 154 |
| n06-25/c104-1103T1909 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 2-node | 3 (4) | 104 | off | 60 | 53,609 | 438.9 | 69.1 / 60.9 | 4.01 / 9.36 | 49 / 67 | 6/12/19/39 | 8/15/23/29/39 | 1.28 / 1.51 | 10-24 (83%) | 6-28 (96%) | 12-24 (81%) | 8-30 (96%) | 0.74 | 40.1 | 156 |
| v2:v2-1node-c104-1104T2011-c104 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 3 (4) | 104 | off | 60 | 41,124 | 357.3 | 75.6 / 64.5 | 10.74 / 24.88 | 35 / 54 | 4/8/15/38 | 7/13/20/27/38 | 1.42 / 1.75 | 7-20 (85%) | 5-24 (97%) | 10-22 (82%) | 6-26 (97%) | 0.67 | 37.7 | 167 |
| v2:v2-1nodeswap-c104-1104T2207-c104 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 3 (4) | 104 | off | 60 | 41,395 | 352.8 | 76.8 / 67.4 | 10.75 / 25.41 | 34 / 52 | 4/8/15/35 | 7/12/19/25/35 | 1.39 / 1.74 | 8-20 (85%) | 6-22 (95%) | 10-20 (81%) | 7-24 (96%) | 0.68 | 36.5 | 168 |
| v2:v2-4p4-c104-1107T0344-c104 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 2-node | 3 (4) | 104 | off | 60 | 53,055 | 434.3 | 70.4 / 61.2 | 4.29 / 10.55 | 48 / 67 | 6/11/18/39 | 9/15/23/30/39 | 1.29 / 1.56 | 10-22 (80%) | 7-28 (95%) | 12-24 (82%) | 8-30 (96%) | 0.74 | 39.9 | 154 |
| v2:v2-nohip-c104-1105T1056-c104 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 3 (4) | 104 | off | 60 | 52,409 | 431.1 | 70.5 / 60.6 | 4.77 / 11.00 | 47 / 67 | 6/11/18/43 | 9/15/23/33/43 | 1.29 / 1.57 | 10-22 (81%) | 6-28 (96%) | 12-24 (82%) | 7-30 (96%) | 0.73 | 40.0 | 152 |
| v2:v2-dpbal-k3-1106T0845-c104 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 3 (4) | 104 | on(400000) | 60 | 53,460 | 437.8 | 73.0 / 65.4 | 4.26 / 11.27 | 47 / 64 | 6/12/17/26 | 8/14/19/23/26 | 1.20 / 1.38 | 10-20 (82%) | 6-22 (96%) | 12-20 (81%) | 7-22 (96%) | 0.80 | 39.5 | 150 |
| v2:v2-k3-hi-1106T0358-c112 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 3 (4) | 112 | off | 60 | 53,026 | 440.8 | 69.0 / 60.9 | 5.14 / 13.63 | 49 / 69 | 6/12/19/32 | 9/16/23/29/32 | 1.28 / 1.54 | 10-22 (80%) | 7-28 (95%) | 12-24 (82%) | 8-28 (96%) | 0.75 | 40.5 | 155 |
| v2:v2-1p2d-c128-1107T0201-c128:decode | B | 1P(4GPU,PP4)+2D(TP4/DP4) 2-node | 3 (4) | 128 | off | 60 | 37,999 | 328.8 | 88.5 / 78.2 | 3.93 / 7.22 | 21 / 30 | 2/5/10/25 | 5/8/13/20/25 | 1.60 / 2.00 | 5-12 (80%) | 4-18 (95%) | 5-14 (86%) | 4-18 (96%) | 0.59 | 32.1 | 205 |
| v2:v2-1p2d-c128-1107T0201-c128:decode1 | B | 1P(4GPU,PP4)+2D(TP4/DP4) 2-node | 3 (4) | 128 | off | 60 | 37,999 | 328.8 | 88.5 / 78.2 | 3.93 / 7.22 | 21 / 31 | 2/5/9/26 | 5/8/12/18/26 | 1.48 / 1.87 | 5-12 (85%) | 4-16 (96%) | 6-14 (85%) | 4-16 (96%) | 0.64 | 31.6 | 194 |
| v2:v2-k5-lo-1106T0121-c32 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 5 (6) | 32 | off | 60 | 23,369 | 146.1 | 118.0 / 101.5 | 1.01 / 2.26 | 9 / 15 | 1/2/4/11 | 2/3/6/8/11 | 1.33 / 1.71 | 2-5 (85%) | 1-6 (96%) | 2-5 (81%) | 2-8 (97%) | 0.70 | 29.2 | 200 |
| v2:v2-dpbal-k5-c32-1107T0209-c32 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 5 (6) | 32 | on(400000) | 60 | 23,530 | 146.7 | 121.6 / 106.8 | 1.07 / 2.39 | 9 / 14 | 1/2/4/7 | 2/3/4/6/7 | 1.33 / 1.60 | 2-4 (86%) | 1-5 (97%) | 2-4 (82%) | 2-6 (97%) | 0.78 | 28.5 | 193 |
| v2:v2-k5-lo-1106T0121-c48 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 5 (6) | 48 | off | 60 | 32,384 | 214.0 | 103.9 / 90.5 | 1.33 / 2.76 | 16 / 23 | 2/4/7/19 | 3/5/8/12/19 | 1.39 / 1.71 | 3-8 (84%) | 2-10 (96%) | 4-10 (87%) | 3-12 (96%) | 0.70 | 33.1 | 181 |
| v2:v2-dpbal-k5-c48-1107T0335-c48 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 5 (6) | 48 | on(400000) | 60 | 32,819 | 216.7 | 108.0 / 97.8 | 1.32 / 2.76 | 15 / 23 | 2/4/6/12 | 3/5/7/9/12 | 1.23 / 1.47 | 3-7 (89%) | 2-8 (98%) | 4-7 (80%) | 2-8 (97%) | 0.81 | 32.3 | 176 |
| v2:v2-k5-mid-1106T0727-c64 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 5 (6) | 64 | off | 60 | 42,284 | 309.7 | 86.3 / 74.0 | 2.02 / 3.86 | 29 / 39 | 3/7/11/26 | 4/10/14/20/26 | 1.37 / 1.67 | 6-16 (83%) | 2-16 (96%) | 8-16 (81%) | 5-20 (96%) | 0.69 | 39.8 | 165 |
| v2:v2-dpbal-k5-1106T0724-c64 | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 5 (6) | 64 | on(400000) | 60 | 43,245 | 316.0 | 91.3 / 81.6 | 1.97 / 3.92 | 27 / 37 | 3/7/10/16 | 4/8/11/14/16 | 1.19 / 1.33 | 6-12 (83%) | 2-12 (96%) | 7-12 (84%) | 5-14 (96%) | 0.82 | 38.0 | 160 |
| v2:v2-k5-c88-mf093-1106T2356-c88 [memfrac=0.93] | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 5 (6) | 88 | off | 30 | 45,081 | 390.7 | 74.8 / 57.8 | 2.74 / 4.80 | 43 / 56 | 4/10/16/30 | 6/14/19/26/30 | 1.28 / 1.52 | 6-20 (86%) | 4-24 (95%) | 12-22 (81%) | 6-24 (95%) | 0.74 | 45.6 | 132 |
| v2:v2-k5-c88r-1106T0958-c88 **TRUNCATED** | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 5 (6) | 88 | off | 6 | n/a | n/a | n/a / n/a | n/a / n/a | 19 / 27 | 2/5/8/24 | 3/6/9/20/24 | 1.33 / 1.60 | 3-8 (81%) | 1-16 (97%) | 3-10 (81%) | 3-20 (96%) | 0.75 | 33.1 | 83 |
| v2:v2-k5-c88s-1106T1028-c88 **TRUNCATED** | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 5 (6) | 88 | off | 9 | n/a | n/a | n/a / n/a | n/a / n/a | 22 / 54 | 2/6/14/25 | 4/7/18/25/25 | 1.26 / 1.51 | 4-18 (83%) | 2-22 (96%) | 5-20 (83%) | 5-26 (95%) | 0.76 | 34.1 | 95 |
| v2:v2-k5-mid-1106T0727-c88 **TRUNCATED** | B | 1P(4GPU,PP4)+1D(TP4/DP4) 1-node | 5 (6) | 88 | off | 8 | n/a | n/a | n/a / n/a | n/a / n/a | 21 / 52 | 2/5/12/28 | 3/7/18/28/28 | 1.33 / 1.68 | 1-10 (80%) | 2-24 (96%) | 5-22 (81%) | 4-28 (97%) | 0.72 | 33.8 | 89 |
| n06-25/c104-1103T1737 | B-extra | 1P(4GPU,PP4)+1D(TP4/DP4) 2-node | 3 (4) | 104 | off | 60 | 51,477 | 425.9 | 63.9 / 56.2 | 4.29 / 10.93 | 52 / 71 | 6/12/19/37 | 9/16/24/32/37 | 1.28 / 1.50 | 10-24 (82%) | 6-28 (95%) | 14-26 (80%) | 8-30 (95%) | 0.75 | 43.7 | 151 |
| n06-25/c104-1103T2037 | B-extra | 1P(4GPU,PP4)+1D(TP4/DP4) 2-node | 3 (4) | 104 | off | 60 | 50,797 | 419.1 | 63.7 / 55.7 | 4.55 / 11.41 | 51 / 70 | 6/12/19/41 | 9/16/24/33/41 | 1.30 / 1.52 | 10-24 (82%) | 6-28 (95%) | 12-26 (84%) | 10-40 (96%) | 0.73 | 44.2 | 152 |
| n06-25/c104-1104T0206 | B-extra | 1P(4GPU,PP4)+1D(TP4/DP4) 2-node | 3 (4) | 104 | off | 60 | 52,830 | 432.8 | 68.3 / 61.3 | 4.30 / 10.17 | 49 / 67 | 6/12/19/31 | 9/16/23/28/31 | 1.29 / 1.52 | 10-22 (81%) | 6-26 (96%) | 12-24 (85%) | 10-28 (95%) | 0.74 | 41.2 | 156 |
| n06-25/dprof-c104-1103T2244 | B-extra | 1P(4GPU,PP4)+1D(TP4/DP4) 2-node | 3 (4) | 104 | off | 25 | 48,027 | 443.6 | 65.1 / 57.6 | 4.02 / 10.88 | 58 / 73 | 5/13/21/41 | 7/19/26/35/41 | 1.31 / 1.61 | 7-26 (81%) | 5-32 (96%) | 14-30 (82%) | 7-32 (95%) | 0.72 | 43.9 | 123 |
| n06-25/c104-1103T2336 | B-extra | 1P(4GPU,PP4)+1D(TP4/DP4) 2-node | 3 (4) | 104 | on(1000000) | 60 | 53,107 | 436.7 | 73.9 / 65.6 | 4.50 / 12.99 | 46 / 63 | 6/11/17/26 | 8/14/19/23/26 | 1.19 / 1.38 | 10-20 (82%) | 6-22 (96%) | 12-20 (80%) | 8-24 (96%) | 0.80 | 39.1 | 151 |

## Shapes (target-verify graph): attention rows per rank = rows/req x gbs; MoE tokens M = n_ranks x rows/req x gbs

| run | ranks x rows/req | gbs p50 ticks / req | attn rows 80% ticks | attn rows 95% ticks | attn rows 80% req | attn rows 95% req | M 80% ticks | M 95% ticks | M 80% req | M 95% req | M 80% time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| partial:c208-1110T2109 | 8 x 4 | 10 / 12 | 28-64 | 20-72 | 32-64 | 24-80 | 224-512 | 160-576 | 256-512 | 192-640 | 224-512 |
| n04-33/c208-1104T2113 | 8 x 4 | 14 / 14 | 40-80 | 20-88 | 48-80 | 28-88 | 320-640 | 160-704 | 384-640 | 224-704 | 320-640 |
| n04-33/c208-1109T1152-kopt-pp8 | 8 x 4 | 10 / 12 | 28-64 | 16-72 | 28-64 | 20-80 | 224-512 | 128-576 | 224-512 | 160-640 | 224-512 |
| n06-25/c208-1104T0523 | 8 x 4 | 16 / 16 | 48-88 | 20-96 | 48-80 | 24-96 | 384-704 | 160-768 | 384-640 | 192-768 | 384-704 |
| n06-25/dprof-c208-1104T0758 | 8 x 4 | 16 / 18 | 20-88 | 12-104 | 56-104 | 20-112 | 160-704 | 96-832 | 448-832 | 160-896 | 192-768 |
| partial:c208-1110T0900 | 8 x 4 | 12 / 14 | 28-72 | 16-80 | 40-80 | 24-88 | 224-576 | 128-640 | 320-640 | 192-704 | 224-576 |
| v2:v2-k3-hi-1106T0358-c88 | 4 x 4 | 14 / 16 | 40-88 | 20-96 | 48-88 | 28-104 | 160-352 | 80-384 | 192-352 | 112-416 | 160-352 |
| v2:v2-dpbal-k3-c88-1109T0501-c88 | 4 x 4 | 14 / 14 | 40-72 | 20-80 | 40-72 | 28-80 | 160-288 | 80-320 | 160-288 | 112-320 | 160-288 |
| v2:v2-1p2d-1106T1933-c96:decode | 4 x 4 | 7 / 8 | 12-40 | 8-48 | 20-48 | 12-56 | 48-160 | 32-192 | 80-192 | 48-224 | 64-192 |
| v2:v2-1p2d-1106T1933-c96:decode1 | 4 x 4 | 7 / 7 | 12-40 | 8-48 | 20-48 | 12-56 | 48-160 | 32-192 | 80-192 | 48-224 | 64-160 |
| n02-33/c104-1105T0937 | 4 x 4 | 16 / 18 | 40-96 | 24-120 | 48-96 | 40-120 | 160-384 | 96-480 | 192-384 | 160-480 | 192-384 |
| n06-25/c104-1103T1909 | 4 x 4 | 16 / 18 | 40-96 | 24-112 | 48-96 | 32-120 | 160-384 | 96-448 | 192-384 | 128-480 | 160-384 |
| v2:v2-1node-c104-1104T2011-c104 | 4 x 4 | 14 / 14 | 28-80 | 20-96 | 40-88 | 24-104 | 112-320 | 80-384 | 160-352 | 96-416 | 128-320 |
| v2:v2-1nodeswap-c104-1104T2207-c104 | 4 x 4 | 12 / 14 | 32-80 | 24-88 | 40-80 | 28-96 | 128-320 | 96-352 | 160-320 | 112-384 | 128-320 |
| v2:v2-4p4-c104-1107T0344-c104 | 4 x 4 | 16 / 18 | 40-88 | 28-112 | 48-96 | 32-120 | 160-352 | 112-448 | 192-384 | 128-480 | 160-352 |
| v2:v2-nohip-c104-1105T1056-c104 | 4 x 4 | 16 / 18 | 40-88 | 24-112 | 48-96 | 28-120 | 160-352 | 96-448 | 192-384 | 112-480 | 160-352 |
| v2:v2-dpbal-k3-1106T0845-c104 | 4 x 4 | 14 / 16 | 40-80 | 24-88 | 48-80 | 28-88 | 160-320 | 96-352 | 192-320 | 112-352 | 160-320 |
| v2:v2-k3-hi-1106T0358-c112 | 4 x 4 | 16 / 18 | 40-88 | 28-112 | 48-96 | 32-112 | 160-352 | 112-448 | 192-384 | 128-448 | 160-352 |
| v2:v2-1p2d-c128-1107T0201-c128:decode | 4 x 4 | 8 / 10 | 20-48 | 16-72 | 20-56 | 16-72 | 80-192 | 64-288 | 80-224 | 64-288 | 80-224 |
| v2:v2-1p2d-c128-1107T0201-c128:decode1 | 4 x 4 | 8 / 10 | 20-48 | 16-64 | 24-56 | 16-64 | 80-192 | 64-256 | 96-224 | 64-256 | 80-192 |
| v2:v2-k5-lo-1106T0121-c32 | 4 x 6 | 3 / 4 | 12-30 | 6-36 | 12-30 | 12-48 | 48-120 | 24-144 | 48-120 | 48-192 | 48-120 |
| v2:v2-dpbal-k5-c32-1107T0209-c32 | 4 x 6 | 3 / 3 | 12-24 | 6-30 | 12-24 | 12-36 | 48-96 | 24-120 | 48-96 | 48-144 | 48-96 |
| v2:v2-k5-lo-1106T0121-c48 | 4 x 6 | 5 / 6 | 18-48 | 12-60 | 24-60 | 18-72 | 72-192 | 48-240 | 96-240 | 72-288 | 72-192 |
| v2:v2-dpbal-k5-c48-1107T0335-c48 | 4 x 6 | 5 / 5 | 18-42 | 12-48 | 24-42 | 12-48 | 72-168 | 48-192 | 96-168 | 48-192 | 72-168 |
| v2:v2-k5-mid-1106T0727-c64 | 4 x 6 | 10 / 10 | 36-96 | 12-96 | 48-96 | 30-120 | 144-384 | 48-384 | 192-384 | 120-480 | 168-384 |
| v2:v2-dpbal-k5-1106T0724-c64 | 4 x 6 | 8 / 10 | 36-72 | 12-72 | 42-72 | 30-84 | 144-288 | 48-288 | 168-288 | 120-336 | 144-288 |
| v2:v2-k5-c88-mf093-1106T2356-c88 | 4 x 6 | 14 / 16 | 36-120 | 24-144 | 72-132 | 36-144 | 144-480 | 96-576 | 288-528 | 144-576 | 240-576 |
| v2:v2-k5-c88r-1106T0958-c88 | 4 x 6 | 6 / 7 | 18-48 | 6-96 | 18-60 | 18-120 | 72-192 | 24-384 | 72-240 | 72-480 | 72-240 |
| v2:v2-k5-c88s-1106T1028-c88 | 4 x 6 | 7 / 14 | 24-108 | 12-132 | 30-120 | 30-156 | 96-432 | 48-528 | 120-480 | 120-624 | 96-480 |
| v2:v2-k5-mid-1106T0727-c88 | 4 x 6 | 7 / 8 | 6-60 | 12-144 | 30-132 | 24-168 | 24-240 | 48-576 | 120-528 | 96-672 | 72-480 |
| n06-25/c104-1103T1737 | 4 x 4 | 16 / 18 | 40-96 | 24-112 | 56-104 | 32-120 | 160-384 | 96-448 | 224-416 | 128-480 | 160-384 |
| n06-25/c104-1103T2037 | 4 x 4 | 16 / 18 | 40-96 | 24-112 | 48-104 | 40-160 | 160-384 | 96-448 | 192-416 | 160-640 | 160-384 |
| n06-25/c104-1104T0206 | 4 x 4 | 16 / 18 | 40-88 | 24-104 | 48-96 | 40-112 | 160-352 | 96-416 | 192-384 | 160-448 | 192-384 |
| n06-25/dprof-c104-1103T2244 | 4 x 4 | 20 / 20 | 28-104 | 20-128 | 56-120 | 28-128 | 112-416 | 80-512 | 224-480 | 112-512 | 128-448 |
| n06-25/c104-1103T2336 | 4 x 4 | 14 / 16 | 40-80 | 24-88 | 48-80 | 32-96 | 160-320 | 96-352 | 192-320 | 128-384 | 160-320 |

Other captured graphs seen in the logs (the draft model runs them inside the same step; how often each runs per step was not derived here): TP4 K=3: draft decode 1 row(s)/req, draft extend 4 row(s)/req, target verify 4 row(s)/req; TP4 K=5: draft decode 1 row(s)/req, draft extend 6 row(s)/req, target verify 6 row(s)/req; TP8 K=3: draft decode 1 row(s)/req, draft extend 4 row(s)/req, target verify 4 row(s)/req

## TP4/DP4 sweep by concurrency (complete runs only), and TP8/DP8 runs

TP4/DP4:

| K | C | DP-bal | run | out tok/s/GPU | intvty p50/p90 | total run p50 | busiest p50/p90/p99 | gbs p50 ticks/req | gbs 80% ticks | gbs 95% ticks | gbs 80% req | gbs 95% req | M 95% req | fill | step ms p50/p90 (req-weighted) | accept_len/step: p50 / p90-slow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 88 | off | v2:v2-k3-hi-1106T0358-c88 | 413.1 | 73.3 / 63.9 | 44 | 14/20/27 | 14 / 16 | 10-22 (82%) | 5-24 (95%) | 12-22 (82%) | 7-26 (96%) | 112-416 | 0.74 | 40.0 / 46.4 | 74.7 / 64.4 |
| 3 | 88 | on(400000) | v2:v2-dpbal-k3-c88-1109T0501-c88 | 413.7 | 76.1 / 69.3 | 43 | 13/17/20 | 14 / 14 | 10-18 (80%) | 5-20 (96%) | 10-18 (86%) | 7-20 (95%) | 112-320 | 0.82 | 39.0 / 42.9 | 76.7 / 69.7 |
| 3 | 96 | off | v2:v2-1p2d-1106T1933-c96:decode | 304.7 | 93.8 / 84.3 | 19 | 7/10/14 | 7 / 8 | 3-10 (87%) | 2-12 (97%) | 5-12 (85%) | 3-14 (97%) | 48-224 | 0.68 | 31.3 / 35.2 | 95.6 / 84.9 |
| 3 | 96 | off | v2:v2-1p2d-1106T1933-c96:decode1 | 304.7 | 93.8 / 84.3 | 18 | 7/10/14 | 7 / 7 | 3-10 (86%) | 2-12 (97%) | 5-12 (84%) | 3-14 (97%) | 48-224 | 0.67 | 31.1 / 35.1 | 96.2 / 85.1 |
| 3 | 104 | off | n02-33/c104-1105T0937 | 434.9 | 63.4 / 55.4 | 52 | 16/24/31 | 16 / 18 | 10-24 (84%) | 6-30 (96%) | 12-24 (82%) | 10-30 (95%) | 160-480 | 0.75 | 46.7 / 53.8 | 64.0 / 55.6 |
| 3 | 104 | off | n06-25/c104-1103T1737 | 425.9 | 63.9 / 56.2 | 52 | 16/24/32 | 16 / 18 | 10-24 (82%) | 6-28 (95%) | 14-26 (80%) | 8-30 (95%) | 128-480 | 0.75 | 46.1 / 52.6 | 64.9 / 56.8 |
| 3 | 104 | off | n06-25/c104-1103T1909 | 438.9 | 69.1 / 60.9 | 49 | 15/23/29 | 16 / 18 | 10-24 (83%) | 6-28 (96%) | 12-24 (81%) | 8-30 (96%) | 128-480 | 0.74 | 42.5 / 48.7 | 70.4 / 61.4 |
| 3 | 104 | off | n06-25/c104-1103T2037 | 419.1 | 63.7 / 55.7 | 51 | 16/24/33 | 16 / 18 | 10-24 (82%) | 6-28 (95%) | 12-26 (84%) | 10-40 (96%) | 160-640 | 0.73 | 46.2 / 53.0 | 64.8 / 56.4 |
| 3 | 104 | off | n06-25/c104-1104T0206 | 432.8 | 68.3 / 61.3 | 49 | 16/23/28 | 16 / 18 | 10-22 (81%) | 6-26 (96%) | 12-24 (85%) | 10-28 (95%) | 160-448 | 0.74 | 43.2 / 48.6 | 69.3 / 61.5 |
| 3 | 104 | off | n06-25/dprof-c104-1103T2244 | 443.6 | 65.1 / 57.6 | 58 | 19/26/35 | 20 / 20 | 7-26 (81%) | 5-32 (96%) | 14-30 (82%) | 7-32 (95%) | 112-512 | 0.72 | 45.6 / 50.9 | 65.6 / 58.7 |
| 3 | 104 | off | v2:v2-1node-c104-1104T2011-c104 | 357.3 | 75.6 / 64.5 | 35 | 13/20/27 | 14 / 14 | 7-20 (85%) | 5-24 (97%) | 10-22 (82%) | 6-26 (97%) | 96-416 | 0.67 | 38.9 / 46.0 | 76.9 / 65.0 |
| 3 | 104 | off | v2:v2-1nodeswap-c104-1104T2207-c104 | 352.8 | 76.8 / 67.4 | 34 | 12/19/25 | 12 / 14 | 8-20 (85%) | 6-22 (95%) | 10-20 (81%) | 7-24 (96%) | 112-384 | 0.68 | 38.2 / 44.0 | 78.3 / 68.0 |
| 3 | 104 | off | v2:v2-4p4-c104-1107T0344-c104 | 434.3 | 70.4 / 61.2 | 48 | 15/23/30 | 16 / 18 | 10-22 (80%) | 7-28 (95%) | 12-24 (82%) | 8-30 (96%) | 128-480 | 0.74 | 41.9 / 48.4 | 71.4 / 61.8 |
| 3 | 104 | off | v2:v2-nohip-c104-1105T1056-c104 | 431.1 | 70.5 / 60.6 | 47 | 15/23/33 | 16 / 18 | 10-22 (81%) | 6-28 (96%) | 12-24 (82%) | 7-30 (96%) | 112-480 | 0.73 | 41.8 / 48.8 | 71.6 / 61.3 |
| 3 | 104 | on(1000000) | n06-25/c104-1103T2336 | 436.7 | 73.9 / 65.6 | 46 | 14/19/23 | 14 / 16 | 10-20 (82%) | 6-22 (96%) | 12-20 (80%) | 8-24 (96%) | 128-384 | 0.80 | 39.8 / 45.1 | 75.1 / 66.2 |
| 3 | 104 | on(400000) | v2:v2-dpbal-k3-1106T0845-c104 | 437.8 | 73.0 / 65.4 | 47 | 14/19/23 | 14 / 16 | 10-20 (82%) | 6-22 (96%) | 12-20 (81%) | 7-22 (96%) | 112-352 | 0.80 | 40.2 / 45.3 | 74.4 / 66.0 |
| 3 | 112 | off | v2:v2-k3-hi-1106T0358-c112 | 440.8 | 69.0 / 60.9 | 49 | 16/23/29 | 16 / 18 | 10-22 (80%) | 7-28 (95%) | 12-24 (82%) | 8-28 (96%) | 128-448 | 0.75 | 42.5 / 48.9 | 70.3 / 61.1 |
| 3 | 128 | off | v2:v2-1p2d-c128-1107T0201-c128:decode | 328.8 | 88.5 / 78.2 | 21 | 8/13/20 | 8 / 10 | 5-12 (80%) | 4-18 (95%) | 5-14 (86%) | 4-18 (96%) | 64-288 | 0.59 | 33.9 / 38.1 | 88.3 / 78.4 |
| 3 | 128 | off | v2:v2-1p2d-c128-1107T0201-c128:decode1 | 328.8 | 88.5 / 78.2 | 21 | 8/12/18 | 8 / 10 | 5-12 (85%) | 4-16 (96%) | 6-14 (85%) | 4-16 (96%) | 64-256 | 0.64 | 32.9 / 37.2 | 90.8 / 80.4 |
| 5 | 32 | off | v2:v2-k5-lo-1106T0121-c32 | 146.1 | 118.0 / 101.5 | 9 | 3/6/8 | 3 / 4 | 2-5 (85%) | 1-6 (96%) | 2-5 (81%) | 2-8 (97%) | 48-192 | 0.70 | 31.0 / 34.7 | 116.4 / 104.0 |
| 5 | 32 | on(400000) | v2:v2-dpbal-k5-c32-1107T0209-c32 | 146.7 | 121.6 / 106.8 | 9 | 3/4/6 | 3 / 3 | 2-4 (86%) | 1-5 (97%) | 2-4 (82%) | 2-6 (97%) | 48-144 | 0.78 | 29.6 / 32.8 | 121.9 / 109.9 |
| 5 | 48 | off | v2:v2-k5-lo-1106T0121-c48 | 214.0 | 103.9 / 90.5 | 16 | 5/8/12 | 5 / 6 | 3-8 (84%) | 2-10 (96%) | 4-10 (87%) | 3-12 (96%) | 72-288 | 0.70 | 34.2 / 39.3 | 105.5 / 91.8 |
| 5 | 48 | on(400000) | v2:v2-dpbal-k5-c48-1107T0335-c48 | 216.7 | 108.0 / 97.8 | 15 | 5/7/9 | 5 / 5 | 3-7 (89%) | 2-8 (98%) | 4-7 (80%) | 2-8 (97%) | 48-192 | 0.81 | 33.0 / 36.5 | 109.5 / 98.8 |
| 5 | 64 | off | v2:v2-k5-mid-1106T0727-c64 | 309.7 | 86.3 / 74.0 | 29 | 10/14/20 | 10 / 10 | 6-16 (83%) | 2-16 (96%) | 8-16 (81%) | 5-20 (96%) | 120-480 | 0.69 | 41.2 / 47.5 | 87.7 / 76.0 |
| 5 | 64 | on(400000) | v2:v2-dpbal-k5-1106T0724-c64 | 316.0 | 91.3 / 81.6 | 27 | 8/11/14 | 8 / 10 | 6-12 (83%) | 2-12 (96%) | 7-12 (84%) | 5-14 (96%) | 120-336 | 0.82 | 39.3 / 43.5 | 92.0 / 82.9 |
| 5 | 88 | off | v2:v2-k5-c88-mf093-1106T2356-c88 | 390.7 | 74.8 / 57.8 | 43 | 14/19/26 | 14 / 16 | 6-20 (86%) | 4-24 (95%) | 12-22 (81%) | 6-24 (95%) | 144-576 | 0.74 | 47.4 / 62.0 | 76.0 / 58.1 |

TP8/DP8:

| K | C | DP-bal | run | out tok/s/GPU | intvty p50/p90 | total run p50 | busiest p50/p90/p99 | gbs p50 ticks/req | gbs 80% ticks | gbs 95% ticks | gbs 80% req | gbs 95% req | M 95% req | fill | step ms p50/p90 (req-weighted) | accept_len/step: p50 / p90-slow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 208 | off | partial:c208-1110T0900 TRUNCATED | n/a | n/a / n/a | 63 | 12/17/22 | 12 / 14 | 7-18 (81%) | 4-20 (96%) | 10-20 (84%) | 6-22 (96%) | 192-704 | 0.63 | 34.6 / 39.4 | 86.3 / 75.9 |
| 3 | 208 | off | partial:c208-1110T2109 | n/a | n/a / n/a | 50 | 10/15/21 | 10 / 12 | 7-16 (85%) | 5-18 (96%) | 8-16 (81%) | 6-20 (96%) | 192-640 | 0.58 | 31.2 / 34.5 | 95.8 / 86.6 |
| 3 | 208 | off | n04-33/c208-1104T2113 | 382.3 | 80.6 / 69.1 | 75 | 13/18/23 | 14 / 14 | 10-20 (83%) | 5-22 (96%) | 12-20 (81%) | 7-22 (95%) | 224-704 | 0.67 | 36.2 / 42.9 | 82.5 / 69.6 |
| 3 | 208 | off | n04-33/c208-1109T1152-kopt-pp8 | 401.1 | 94.2 / 80.7 | 47 | 10/15/21 | 10 / 12 | 7-16 (84%) | 4-18 (96%) | 7-16 (85%) | 5-20 (96%) | 160-640 | 0.58 | 31.2 / 36.2 | 95.8 / 82.7 |
| 3 | 208 | off | n06-25/c208-1104T0523 | 403.3 | 77.2 / 65.7 | 84 | 15/20/27 | 16 / 16 | 12-22 (81%) | 5-24 (95%) | 12-20 (82%) | 6-24 (95%) | 192-768 | 0.67 | 37.4 / 45.0 | 79.9 / 66.4 |
| 3 | 208 | off | n06-25/dprof-c208-1104T0758 | 389.7 | 70.8 / 61.0 | 89 | 15/23/28 | 16 / 18 | 5-22 (81%) | 3-26 (96%) | 14-26 (81%) | 5-28 (97%) | 160-896 | 0.67 | 42.1 / 48.4 | 71.1 / 61.8 |

`accept_len/step` is what a request would get if it sat on steps of the request-weighted p50 / p90 duration for its whole life; it is an upper-bound style cross-check of the client interactivity columns, not a replacement (a request averages over many steps, and the client number includes streaming and network).

## Step ms by implied graph batch (median over ticks whose graph batch did not change since the rank's previous line; blank = fewer than 40 such ticks)

| run | K | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| partial:c208-1110T2109 | 3 |  |  |  | 22.5 | 24.8 | 27.1 | 27.1 | 28.0 | 30.8 | 31.4 | 33.0 | 34.0 | 36.4 |  |  |  |  |  |  |  |
| n04-33/c208-1104T2113 | 3 |  | 19.4 | 20.8 | 21.6 | 23.5 | 26.2 | 26.4 | 27.1 | 31.8 | 33.3 | 35.8 | 36.9 | 41.4 | 43.2 | 45.7 |  |  |  |  |  |
| n04-33/c208-1109T1152-kopt-pp8 | 3 |  | 19.6 | 20.8 | 22.2 | 23.9 | 25.9 | 26.3 | 27.3 | 30.5 | 31.5 | 33.9 | 35.5 | 38.1 |  | 42.8 |  |  |  |  |  |
| n06-25/c208-1104T0523 | 3 |  |  | 20.6 | 21.6 | 23.6 | 26.3 | 26.4 | 27.0 | 31.9 | 33.4 | 35.9 | 37.1 | 41.6 | 43.1 | 45.3 | 46.8 |  |  |  |  |
| n06-25/dprof-c208-1104T0758 | 3 |  |  | 20.6 | 21.7 | 23.4 | 25.8 | 26.3 | 26.8 | 30.4 |  | 36.0 | 37.1 | 41.9 | 43.3 | 46.1 | 47.1 | 48.6 |  |  |  |
| partial:c208-1110T0900 | 3 |  | 20.3 |  | 22.8 | 24.2 | 26.7 | 26.6 | 28.3 | 32.4 | 33.0 | 34.9 | 35.9 | 38.9 | 40.3 |  |  |  |  |  |  |
| v2:v2-k3-hi-1106T0358-c88 | 3 |  |  |  | 25.5 |  | 29.6 | 29.5 | 30.3 | 35.6 | 36.8 | 38.9 | 39.8 | 42.6 | 44.3 | 46.2 | 48.1 |  |  |  |  |
| v2:v2-dpbal-k3-c88-1109T0501-c88 | 3 |  |  | 23.3 |  | 27.4 | 29.7 | 29.5 | 32.4 | 35.4 | 36.9 | 39.0 | 40.1 | 42.9 | 44.6 |  |  |  |  |  |  |
| v2:v2-1p2d-1106T1933-c96:decode | 3 | 17.9 | 21.8 | 23.7 | 25.8 | 28.6 | 30.6 | 30.6 | 31.3 | 34.1 | 35.3 | 36.8 |  |  |  |  |  |  |  |  |  |
| v2:v2-1p2d-1106T1933-c96:decode1 | 3 |  | 21.6 | 23.6 | 26.7 | 28.4 | 30.7 | 30.6 | 31.4 | 34.1 | 35.3 | 36.9 |  |  |  |  |  |  |  |  |  |
| n02-33/c104-1105T0937 | 3 | 16.4 |  |  |  |  | 26.5 | 28.1 | 29.1 | 37.9 | 41.9 | 38.0 | 39.0 | 47.3 | 48.8 | 46.7 | 52.4 | 53.7 |  |  |  |
| n06-25/c104-1103T1909 | 3 |  |  |  |  |  | 29.0 | 28.9 | 30.1 | 34.9 | 36.6 | 39.0 | 40.0 | 42.8 | 44.5 | 46.7 | 48.0 | 49.2 |  |  |  |
| v2:v2-1node-c104-1104T2011-c104 | 3 |  |  |  |  | 27.3 | 30.7 | 30.9 | 31.9 | 34.9 | 36.3 | 38.4 | 39.5 | 42.2 | 43.9 | 46.0 | 47.5 |  |  |  |  |
| v2:v2-1nodeswap-c104-1104T2207-c104 | 3 |  |  |  |  | 28.7 | 30.8 | 30.9 | 31.8 | 34.7 | 36.1 | 38.2 | 39.2 | 42.0 | 43.8 | 45.3 |  |  |  |  |  |
| v2:v2-4p4-c104-1107T0344-c104 | 3 |  |  |  |  |  | 29.1 | 29.5 | 30.4 | 35.0 | 36.8 | 38.8 | 39.9 | 42.6 | 44.2 | 46.3 | 47.9 | 49.5 |  |  |  |
| v2:v2-nohip-c104-1105T1056-c104 | 3 |  |  |  |  |  | 29.4 | 29.6 | 30.6 | 35.2 | 37.0 | 39.0 | 39.9 | 42.7 | 44.2 | 46.2 | 48.1 | 49.8 |  |  |  |
| v2:v2-dpbal-k3-1106T0845-c104 | 3 |  |  |  |  | 27.4 | 29.7 | 29.8 | 30.5 | 35.5 | 37.1 | 39.2 | 40.2 | 43.2 | 44.5 | 46.7 | 48.5 |  |  |  |  |
| v2:v2-k3-hi-1106T0358-c112 | 3 |  |  |  |  |  |  | 29.1 | 30.1 | 35.2 | 36.9 | 39.1 | 40.1 | 42.8 | 44.6 | 46.7 | 48.0 | 49.8 | 51.5 |  |  |
| v2:v2-1p2d-c128-1107T0201-c128:decode | 3 |  |  | 23.5 | 25.7 | 28.5 | 30.6 | 30.6 | 31.4 | 34.3 | 35.5 | 37.3 | 38.3 | 41.2 | 43.1 |  |  |  |  |  |  |
| v2:v2-1p2d-c128-1107T0201-c128:decode1 | 3 |  |  | 23.8 | 25.7 | 28.3 | 30.7 | 30.5 | 31.2 | 34.1 | 35.5 | 37.2 | 38.3 |  |  |  |  |  |  |  |  |
| v2:v2-k5-lo-1106T0121-c32 | 5 | 20.4 | 24.8 | 28.4 | 31.4 | 32.0 | 33.8 | 35.5 | 36.5 |  |  |  |  |  |  |  |  |  |  |  |  |
| v2:v2-dpbal-k5-c32-1107T0209-c32 | 5 | 20.5 | 25.0 | 28.4 | 31.6 | 32.2 | 34.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| v2:v2-k5-lo-1106T0121-c48 | 5 | 19.9 | 24.3 | 27.9 | 31.3 | 32.2 | 34.2 | 35.6 | 37.2 | 39.3 | 42.7 |  |  |  |  |  |  |  |  |  |  |
| v2:v2-dpbal-k5-c48-1107T0335-c48 | 5 |  | 24.3 | 28.2 | 31.5 | 32.3 | 34.3 | 36.0 | 37.2 | 39.6 |  |  |  |  |  |  |  |  |  |  |  |
| v2:v2-k5-mid-1106T0727-c64 | 5 |  | 24.3 | 27.1 | 29.6 | 31.2 | 34.0 | 36.1 | 37.1 | 39.5 | 42.7 | 45.1 | 48.0 | 51.3 |  |  |  |  |  |  |  |
| v2:v2-dpbal-k5-1106T0724-c64 | 5 | 20.2 | 24.4 | 27.3 | 30.0 | 32.0 | 34.6 | 36.1 | 37.5 | 40.1 | 43.5 | 46.3 |  |  |  |  |  |  |  |  |  |
| v2:v2-k5-c88-mf093-1106T2356-c88 | 5 |  |  |  |  | 30.5 | 32.4 | 33.8 | 35.2 | 39.6 | 42.6 | 45.0 | 47.8 | 51.0 | 52.8 |  |  |  |  |  |  |
| n06-25/c104-1103T1737 | 3 |  |  |  |  |  |  | 30.8 | 31.6 | 36.9 | 39.0 | 41.3 | 42.5 | 45.7 | 47.5 | 50.0 | 50.9 | 52.9 |  |  |  |
| n06-25/c104-1103T2037 | 3 |  |  |  |  |  | 30.8 | 30.7 | 31.6 | 36.4 | 38.8 | 41.4 | 42.5 | 45.5 | 47.6 | 49.8 | 51.1 | 53.1 |  |  |  |
| n06-25/c104-1104T0206 | 3 |  |  |  |  |  | 29.7 | 30.2 | 30.9 | 34.2 | 37.6 | 39.6 | 40.8 | 43.5 | 44.9 | 47.0 | 48.4 | 50.0 |  |  |  |
| n06-25/dprof-c104-1103T2244 | 3 |  |  |  |  | 25.9 |  | 28.6 | 29.8 | 33.0 |  |  | 40.0 | 42.9 | 44.7 | 46.6 | 47.9 |  | 50.8 |  |  |
| n06-25/c104-1103T2336 | 3 |  |  |  |  |  | 28.7 | 28.9 | 30.2 | 34.9 | 36.6 | 39.0 | 40.1 | 43.0 | 44.5 | 46.7 |  |  |  |  |  |

Median step-time increment between neighbouring captured sizes over the engines above that have both sizes (ms; min .. max; n engines):

| K (rows/req) | 1 to 2 | 2 to 3 | 3 to 4 | 4 to 5 | 5 to 6 | 6 to 7 | 7 to 8 | 8 to 10 | 10 to 12 | 12 to 14 | 14 to 16 | 16 to 18 | 18 to 20 | 20 to 22 | 22 to 24 | 24 to 26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 (4) | +3.83 (+3.8 .. +3.8; 1) | +1.65 (+1.2 .. +2.1; 4) | +1.67 (+0.9 .. +3.1; 8) | +1.86 (+1.4 .. +2.8; 9) | +2.27 (+2.0 .. +3.4; 13) | +0.12 (-0.2 .. +1.5; 21) | +0.87 (+0.5 .. +2.9; 24) | +4.31 (+2.8 .. +8.9; 24) | +1.49 (+0.6 .. +3.9; 22) | +2.06 (-3.9 .. +2.6; 22) | +1.10 (+0.9 .. +1.6; 21) | +2.87 (+2.6 .. +8.3; 21) | +1.68 (+1.3 .. +2.1; 20) | +2.13 (-2.2 .. +2.8; 17) | +1.46 (+0.9 .. +5.7; 14) | +1.65 (+1.3 .. +2.0; 9) |
| 5 (6) | +4.39 (+4.2 .. +4.5; 4) | +3.46 (+2.8 .. +3.9; 6) | +3.12 (+2.6 .. +3.4; 6) | +0.80 (+0.5 .. +2.1; 6) | +2.02 (+1.8 .. +2.7; 7) | +1.57 (+1.4 .. +2.1; 6) | +1.32 (+1.0 .. +1.6; 6) | +2.42 (+2.1 .. +4.3; 5) | +3.31 (+3.0 .. +3.5; 4) | +2.45 (+2.4 .. +2.8; 3) | +2.83 (+2.8 .. +2.9; 2) | +3.25 (+3.1 .. +3.4; 2) | +1.80 (+1.8 .. +1.8; 1) |  |  |  |

## Cross-check against the `score-*.txt` lines

The score-file decode numbers come from analyze_pd_logs.py, which bins by 10 s of wall time (last line per rank in the bin, busiest = max over the ranks seen). Mine are per 40 engine steps. Wall-time weighting gives more weight to large batches (their steps are slower), so the score-file medians sit 1-2 above the step-weighted ones; the time-weighted column shows that this accounts for the difference.

| run | score file | window agrees | tok/s/GPU json / score | total run p50 step-wtd / time-wtd / score | busiest p50 step-wtd / time-wtd / score | busiest p90 step-wtd / time-wtd / score | step ms by busiest: max abs diff (n sizes) | largest diffs @busiest mine/score |
|---|---|---|---|---|---|---|---|---|
| n04-33/c208-1109T1152-kopt-pp8 | kopt-pp8-score | yes | 53,341 / 53,341 | 47 / 49 / 49 | 10 / 10 / 10 | 15 / 15 / 15 | 0.6 (21) | @20 39.4/40, @8 27.6/27, @19 39.5/40 |
| n04-33/c208-1104T2113 | 2p1d-hrrn | yes | 51,804 / 51,804 | 75 / 78 / 76 | 13 / 14 / 14 | 18 / 19 / 19 | 0.5 (24) | @18 41.5/42, @13 35.5/36, @2 19.5/20 |
| n06-25/c208-1104T0523 | 2p1d | yes | 54,094 / 54,094 | 84 / 86 / 86 | 15 / 15 / 15 | 20 / 21 / 21 | 1.2 (28) | @2 19.8/21, @20 43.6/43, @22 45.4/45 |
| partial:c208-1110T0900 | megatp-pp8-score9 | yes | n/a / n/a | 63 / 65 / n/a | 12 / 12 / n/a | 17 / 17 / n/a | n/a | n/a |
| partial:c208-1110T2109 | megatp-mt4q2-pp8-score2 | yes | n/a / n/a | 50 / 51 / n/a | 10 / 11 / n/a | 15 / 15 / n/a | n/a | n/a |
| n06-25/c104-1103T1909 | vfix-C | yes | 53,609 / 53,609 | 49 / 51 / 51 | 15 / 16 / 16 | 23 / 24 / 24 | 0.8 (27) | @27 50.2/51, @11 36.5/36, @30 51.5/52 |
| n06-25/c104-1103T1737 | vfix-B | yes | 51,477 / 51,477 | 52 / 54 / 54 | 16 / 17 / 17 | 24 / 25 / 25 | 0.6 (27) | @10 37.6/37, @14 41.5/41, @32 65.5/66 |
| n06-25/c104-1103T2037 | E10 | yes | 50,797 / 50,797 | 51 / 54 / 53 | 16 / 17 / 17 | 24 / 25 / 24 | 0.6 (28) | @31 56.6/56, @25 52.5/52, @9 35.5/36 |
| n06-25/c104-1103T2336 | Cdpbal | yes | 53,107 / 53,107 | 46 / 48 / 48 | 14 / 14 / 14 | 19 / 20 / 20 | 0.5 (21) | @8 30.5/30, @23 47.5/48, @10 35.6/36 |
| n06-25/c104-1104T0206 | Cptpc0 | yes | 52,830 / 52,830 | 49 / 51 / 51 | 16 / 16 / 16 | 23 / 23 / 23 | 0.6 (25) | @21 46.4/47, @3 23.5/23, @13 39.5/40 |

## All score files (summary row + decode lines as written in the file; decode log availability)

| score file | run | window | C | tok/s/GPU | out tok/s/GPU | intvty p50/p90 | ITL p50/p90 ms | TTFT p50/p90/mean s | decode: total med / busiest med / p90 | decode log analysed here |
|---|---|---|---|---|---|---|---|---|---|---|
| 2p1d-hrrn | c208-1104T2113 | 2026-11-04 21:53:28 .. 2026-11-04 22:53:28 | 208 | 51,804 | 382.3 | 80.6 / 69.1 | 12.4 / 14.5 | 6.88 / 17.11 / 9.36 | 76 / 14 / 19 | n04-33/c208-1104T2113 |
| 2p1d | c208-1104T0523 | 2026-11-04 05:49:06 .. 2026-11-04 06:49:06 | 208 | 54,094 | 403.3 | 77.2 / 65.7 | 12.9 / 15.2 | 4.16 / 8.01 / 6.31 | 86 / 15 / 21 | n06-25/c208-1104T0523 |
| Cdpbal | c104-1103T2336 | 2026-11-03 23:58:00 .. 2026-11-04 00:58:00 | 104 | 53,107 | 436.7 | 73.9 / 65.6 | 13.5 / 15.2 | 4.50 / 12.99 / 6.77 | 48 / 14 / 20 | n06-25/c104-1103T2336 |
| Cfix-c112 | c112-1104T0854 | 2026-11-04 09:20:57 .. 2026-11-04 10:20:57 | 112 | 52,478 | 434.8 | 70.1 / 61.8 | 14.3 / 16.2 | 5.53 / 14.74 / 8.19 | 49 / 16 / 24 | no (log not kept) |
| Cfix-dpbal400k | c104-1104T2303 | 2026-11-04 23:27:13 .. 2026-11-05 | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| Cptpc0 | c104-1104T0206 | 2026-11-04 02:28:49 .. 2026-11-04 03:28:49 | 104 | 52,830 | 432.8 | 68.3 / 61.3 | 14.6 / 16.3 | 4.30 / 10.17 / 5.98 | 51 / 16 / 23 | n06-25/c104-1104T0206 |
| Cptpcfix-1node | c104-1105T0247 | 2026-11-05 03:13:43 .. 2026-11-05 04:13:43 | 104 | 40,807 | 350.6 | 78.5 / 67.3 | 12.7 / 14.8 | 11.39 / 26.52 / 15.65 | 34 / 12 / 19 | no (log not kept) |
| Cptpcfix | c104-1104T0344 | 2026-11-04 04:06:37 .. 2026-11-04 05:06:37 | 104 | 53,666 | 439.2 | 70.8 / 61.9 | 14.1 / 16.2 | 4.12 / 9.28 / 5.70 | 49 / 16 / 23 | no (log not kept) |
| E10 | c104-1103T2037 | 2026-11-03 21:00:20 .. 2026-11-03 22:00:20 | 104 | 50,797 | 419.1 | 63.7 / 55.7 | 15.7 / 17.9 | 4.55 / 11.41 / 6.46 | 53 / 17 / 24 | n06-25/c104-1103T2037 |
| densefp8-1node | c104-1105T0434 | 2026-11-05 04:58:30 .. 2026-11-05 05:58:30 | 104 | 42,968 | 367.0 | 70.8 / 61.5 | 14.1 / 16.3 | 7.40 / 21.67 / 12.53 | 42 / 14 / 21 | no (log not kept) |
| densefp8-nohip-1node |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| densefp8pf-nohip-1node | c104-1105T1229 | 2026-11-05 12:53:33 .. 2026-11-05 13:53:33 | 104 | 52,992 | 434.6 | 68.8 / 61.2 | 14.5 / 16.3 | 4.37 / 10.38 / 6.02 | 52 / 17 / 24 | no (log not kept) |
| dpbal400k-nohip-1node | c104-1105T1105 | 2026-11-05 11:28:32 .. 2026-11-05 12:28:32 | 104 | 52,813 | 433.2 | 73.4 / 65.0 | 13.6 / 15.4 | 4.61 / 12.39 / 6.99 | 46 / 14 / 20 | no (log not kept) |
| k3-nohip-1node-c64 | c64-1105T1235 | 2026-11-05 12:51:39 .. 2026-11-05 13:51:39 | 64 | 41,557 | 305.0 | 83.1 / 76.2 | 12.0 / 13.1 | 2.04 / 3.90 / 2.27 | 30 / 10 / 14 | no (log not kept) |
| k5-nohip-1node-c64 | c64-1105T1529 | 2026-11-05 15:46:06 .. 2026-11-05 16:46:06 | 64 | 42,356 | 310.3 | 85.8 / 75.5 | 11.7 / 13.2 | 2.02 / 3.83 / 2.25 | 29 / 10 / 14 | no (log not kept) |
| kopt-pp8-nocap-score | c208-1109T1336 | 2026-11-08 14:11:48 .. 2026-11-09 | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| kopt-pp8-nocap-score2 |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| kopt-pp8-nocap-score3 | c208-1109T1710 | 2026-11-09 17:44:16 .. 2026-11-09 18:44:16 | 208 | 56,595 | 427.7 | 91.6 / 78.2 | 10.9 / 12.8 | 7.51 / 14.53 / 11.43 | 54 / 11 / 15 | no (log not kept) |
| kopt-pp8-nocap-score4 |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| kopt-pp8-nocap-score5 |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| kopt-pp8-nocap-score6 |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| kopt-pp8-nocap-score7 | c208-1111T0749 | 2026-11-11 08:22:36 .. 2026-11-11 09:22:36 | 208 | 54,506 | 407.5 | 92.9 / 78.4 | 10.8 / 12.8 | 8.14 / 14.74 / 11.71 | 51 / 10 / 15 | no (log not kept) |
| kopt-pp8-nocap-score8 | c208-1111T1347 | 2026-11-11 .. 2026-11-11 | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| kopt-pp8-score | c208-1109T1152 | 2026-11-09 12:25:26 .. 2026-11-09 13:25:26 | 208 | 53,341 | 401.1 | 94.2 / 80.7 | 10.6 / 12.4 | 8.54 / 20.26 / 14.29 | 49 / 10 / 15 | n04-33/c208-1109T1152-kopt-pp8 |
| megatp-mt4-pp8-score |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| megatp-mt4q2-pp8-score | c208-1110T1940 | 2026-11-10 .. 2026-11-10 | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| megatp-mt4q2-pp8-score2 | c208-1110T2109 | 2026-11-10 21:40:02 .. 2026-11-10 22:40:02 | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | partial:c208-1110T2109 |
| megatp-mt4q2-pp8-score3 | c208-1111T0932 | 2026-11-11 10:04:43 .. 2026-11-11 11:04:43 | 208 | 54,743 | 411.0 | 93.6 / 84.4 | 10.7 / 11.8 | 8.01 / 15.81 / 12.18 | 52 / 10 / 14 | no (log not kept) |
| megatp-mt4q2-pp8-score4 | c208-1111T1157 | 2026-11-11 12:37:10 .. 2026-11-11 13:37:10 | 208 | 55,412 | 416.6 | 93.4 / 83.1 | 10.7 / 12.0 | 7.72 / 14.74 / 11.87 | 53 / 11 / 15 | no (log not kept) |
| megatp-pp8-score-d1500 | c208-1110T0634 | 2026-11-10 07:06:47 .. 2026-11-10 07:31:47 | 208 | 64,944 | 510.4 | 81.1 / 71.2 | 12.3 / 14.0 | 4.79 / 8.04 / 7.03 | 82 / 14 / 20 | no (log not kept) |
| megatp-pp8-score10 | c208-1110T1110 | 2026-11-10 .. 2026-11-10 | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| megatp-pp8-score11 |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| megatp-pp8-score2 | c208-1109T2306 | 2026-11-09 .. 2026-11-09 | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| megatp-pp8-score3 |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| megatp-pp8-score4 |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| megatp-pp8-score5 | c208-1110T0158 | 2026-11-09 02:32:23 .. 2026-11-10 | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| megatp-pp8-score6 |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| megatp-pp8-score7 | c208-1110T0530 | 2026-11-09 06:10:43 .. 2026-11-10 | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| megatp-pp8-score8 |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| megatp-pp8-score9 | c208-1110T0900 | 2026-11-09 09:37:01 .. 2026-11-10 | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | partial:c208-1110T0900 |
| nohip-1node-full | c104-1105T0807 | 2026-11-05 08:30:35 .. 2026-11-05 09:30:35 | 104 | 53,209 | 435.7 | 70.3 / 61.0 | 14.2 / 16.4 | 4.25 / 9.64 / 5.99 | 50 / 16 / 23 | no (log not kept) |
| nohip-1node | c104-1105T0612 | 2026-11-05 06:35:48 .. 2026-11-05 07:00:48 | 104 | 48,635 | 448.2 | 65.4 / 59.1 | 15.3 / 16.9 | 4.14 / 9.81 / 5.80 | 59 / 19 / 28 | no (log not kept) |
| nohip-n0433-c104 | c104-1105T1400 | 2026-11-05 14:23:07 .. 2026-11-05 15:23:07 | 104 | 53,057 | 436.0 | 69.7 / 60.4 | 14.3 / 16.6 | 4.28 / 9.46 / 5.93 | 50 / 16 / 24 | no (log not kept) |
| pertok-nohip-1node | c104-1105T2350 | 2026-11-06 00:12:09 .. 2026-11-06 01:12:09 | 104 | 53,129 | 436.8 | 69.3 / 60.7 | 14.4 / 16.5 | 4.26 / 10.06 / 5.90 | 51 / 17 / 24 | no (log not kept) |
| pp8-base-score |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| pp8-base-score2 |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| pp8-base-score3 | c208-1109T1854 | 2026-11-09 .. 2026-11-09 | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| pp8 | c208-1106T2153 | 2026-11-06 22:27:05 .. 2026-11-06 23:27:05 | 208 | 56,593 | 428.9 | 86.0 / 70.9 | 11.6 / 14.1 | 7.50 / 14.47 / 10.82 | 58 / 12 / 17 | no (log not kept) |
| score1500-base | c104-1109T1413 | 2026-11-09 14:35:12 .. 2026-11-09 15:00:12 | 104 | 49,493 | 454.0 | 65.3 / 58.8 | 15.3 / 17.0 | 3.85 / 9.37 / 5.27 | 60 / 19 / 26 | no (log not kept) |
| score1500-base2 | c104-1110T0017 | 2026-11-10 00:40:15 .. 2026-11-10 01:05:15 | 104 | 49,242 | 450.9 | 65.7 / 60.1 | 15.2 / 16.6 | 4.02 / 9.72 / 5.71 | 60 / 19 / 25 | no (log not kept) |
| score1500-kopt | c104-1109T1505 | 2026-11-09 15:28:01 .. 2026-11-09 15:53:01 | 104 | 49,594 | 459.9 | 69.4 / 62.6 | 14.4 / 16.0 | 4.05 / 9.74 / 5.87 | 57 / 18 / 25 | no (log not kept) |
| score1500-megatp-mt2q2 |  |  ..  | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| score1500-megatp | c104-1109T2043 | 2026-11-09 .. 2026-11-09 | n/a | n/a | n/a | n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a (scoring failed or line missing) | no (log not kept) |
| score1500-megatp2 | c104-1109T2324 | 2026-11-09 23:47:14 .. 2026-11-10 00:12:14 | 104 | 49,016 | 449.2 | 66.1 / 61.3 | 15.1 / 16.3 | 4.18 / 10.48 / 5.90 | 60 / 19 / 25 | no (log not kept) |
| vfix-B | c104-1103T1737 | 2026-11-03 18:02:36 .. 2026-11-03 19:02:36 | 104 | 51,477 | 425.9 | 63.9 / 56.2 | 15.7 / 17.8 | 4.29 / 10.93 / 6.04 | 54 / 17 / 25 | n06-25/c104-1103T1737 |
| vfix-C | c104-1103T1909 | 2026-11-03 19:31:19 .. 2026-11-03 20:31:19 | 104 | 53,609 | 438.9 | 69.1 / 60.9 | 14.5 / 16.4 | 4.01 / 9.36 / 5.53 | 51 / 16 / 24 | n06-25/c104-1103T1909 |

`decode step ms by busiest-rank running` lines of the score files:

```
2p1d-hrrn                          @2 20, @3 21, @4 22, @5 24, @6 26, @7 26, @8 28, @9 32, @10 32, @11 33, @12 34, @13 36, @14 36, @15 37, @16 38, @17 40, @18 42, @19 43, @20 43, @21 45, @22 46, @23 47, @24 47, @27 50
2p1d                               @2 21, @3 21, @4 22, @5 24, @6 26, @7 26, @8 27, @9 31, @10 32, @11 33, @12 34, @13 36, @14 36, @15 37, @16 38, @17 41, @18 42, @19 43, @20 43, @21 45, @22 45, @23 46, @24 47, @25 48, @26 49, @27 50, @28 50, @30 51
Cdpbal                             @4 26, @5 27, @6 29, @7 29, @8 30, @9 35, @10 36, @11 36, @12 37, @13 39, @14 39, @15 40, @16 40, @17 43, @18 43, @19 44, @20 45, @21 46, @22 47, @23 48, @24 48
Cfix-c112                          @3 23, @4 25, @5 28, @6 29, @7 30, @8 31, @9 33, @10 35, @11 37, @12 37, @13 39, @14 39, @15 40, @16 40, @17 42, @18 43, @19 44, @20 45, @21 46, @22 46, @23 47, @24 48, @25 49, @26 49, @27 50, @29 51, @32 56
Cptpc0                             @3 23, @5 28, @6 30, @7 30, @8 31, @9 34, @10 36, @11 37, @12 38, @13 40, @14 40, @15 41, @16 41, @17 43, @18 44, @19 45, @20 45, @21 47, @22 47, @23 48, @24 49, @25 49, @26 50, @27 51, @28 51
Cptpcfix-1node                     @3 25, @4 26, @5 28, @6 31, @7 31, @8 32, @9 34, @10 35, @11 36, @12 36, @13 38, @14 38, @15 39, @16 39, @17 42, @18 42, @19 43, @20 43, @21 45, @22 46, @23 47, @24 48, @25 48
Cptpcfix                           @4 25, @5 27, @6 29, @7 30, @8 30, @9 33, @10 35, @11 37, @12 37, @13 39, @14 39, @15 40, @16 40, @17 42, @18 43, @19 44, @20 44, @21 46, @22 46, @23 47, @24 48, @25 49, @26 49, @27 50, @28 50, @30 51
E10                                @4 26, @5 29, @6 31, @7 31, @8 32, @9 36, @10 38, @11 39, @12 39, @13 41, @14 42, @15 42, @16 43, @17 45, @18 46, @19 47, @20 48, @21 49, @22 50, @23 51, @24 51, @25 52, @26 53, @27 54, @28 54, @29 55, @30 55, @31 56
densefp8-1node                     @2 20, @3 22, @4 24, @5 27, @6 28, @7 29, @8 31, @9 39, @10 40, @11 41, @12 41, @13 38, @14 38, @15 38, @16 39, @17 46, @18 47, @19 48, @20 48, @21 47, @22 47, @23 50, @24 51
densefp8pf-nohip-1node             @2 21, @3 24, @4 25, @5 27, @6 29, @7 30, @8 32, @9 34, @10 36, @11 37, @12 37, @13 39, @14 39, @15 40, @16 41, @17 42, @18 43, @19 44, @20 45, @21 46, @22 47, @23 48, @24 48, @25 49, @26 50, @27 51, @28 51, @29 51
dpbal400k-nohip-1node              @4 25, @5 27, @6 30, @7 30, @8 31, @9 35, @10 36, @11 37, @12 37, @13 39, @14 39, @15 40, @16 40, @17 42, @18 43, @19 44, @20 45, @21 46, @22 47, @23 48, @24 48, @25 49
k3-nohip-1node-c64                 @1 18, @2 22, @3 24, @4 25, @5 28, @6 31, @7 31, @8 32, @9 35, @10 35, @11 36, @12 36, @13 38, @14 38, @15 39, @16 39, @17 41, @18 42, @19 43
k5-nohip-1node-c64                 @1 20, @2 24, @3 27, @4 30, @5 33, @6 35, @7 36, @8 38, @9 39, @10 40, @11 43, @12 43, @13 45, @14 46, @15 47, @16 48, @17 51, @18 51, @19 51
kopt-pp8-nocap-score3              @1 17, @2 20, @3 21, @4 22, @5 24, @6 26, @7 27, @8 28, @9 31, @10 31, @11 32, @12 32, @13 34, @14 35, @15 36, @16 36, @17 38, @18 39, @19 40, @20 42, @21 43, @22 44, @23 45, @24 45, @25 46
kopt-pp8-nocap-score7              @2 20, @3 21, @4 22, @5 24, @6 26, @7 27, @8 28, @9 31, @10 31, @11 32, @12 32, @13 34, @14 35, @15 36, @16 37, @17 39, @18 39, @19 40, @20 42, @21 43, @22 44
kopt-pp8-score                     @2 20, @3 21, @4 22, @5 24, @6 26, @7 27, @8 27, @9 30, @10 31, @11 31, @12 32, @13 33, @14 34, @15 35, @16 36, @17 37, @18 38, @19 40, @20 40, @21 42, @22 43
megatp-mt4q2-pp8-score3            @2 21, @3 22, @4 23, @5 25, @6 27, @7 28, @8 28, @9 31, @10 31, @11 32, @12 32, @13 33, @14 34, @15 34, @16 35, @17 36, @18 37, @19 39, @20 39, @21 40, @22 40, @23 41, @25 42
megatp-mt4q2-pp8-score4            @2 20, @3 22, @4 23, @5 25, @6 27, @7 27, @8 28, @9 31, @10 31, @11 31, @12 32, @13 33, @14 33, @15 34, @16 35, @17 36, @18 37, @19 38, @20 39, @21 40, @27 44
megatp-pp8-score-d1500             @1 17, @2 20, @3 22, @4 23, @5 24, @6 26, @7 27, @8 28, @9 30, @10 33, @11 33, @12 34, @13 35, @14 36, @15 36, @16 37, @17 38, @18 39, @19 40, @20 41, @21 42, @22 43, @23 43, @24 43, @25 45
nohip-1node-full                   @4 25, @5 27, @6 29, @7 30, @8 30, @9 33, @10 36, @11 37, @12 37, @13 39, @14 39, @15 40, @16 40, @17 42, @18 43, @19 44, @20 45, @21 46, @22 47, @23 48, @24 48, @25 49, @26 50, @27 50, @28 51, @29 51, @30 51
nohip-1node                        @4 25, @5 28, @6 29, @7 29, @8 30, @9 33, @10 33, @13 39, @14 39, @15 40, @16 41, @17 43, @18 43, @19 44, @20 45, @21 46, @22 47, @23 48, @24 48, @25 49, @26 50, @27 50, @28 51, @29 51, @31 52
nohip-n0433-c104                   @5 27, @6 30, @7 30, @8 30, @9 34, @10 36, @11 37, @12 38, @13 39, @14 39, @15 40, @16 41, @17 42, @18 43, @19 44, @20 45, @21 46, @22 47, @23 48, @24 48, @25 49, @26 50, @27 51, @28 51, @29 51, @30 51
pertok-nohip-1node                 @2 21, @4 26, @5 28, @6 30, @7 30, @8 31, @9 33, @10 36, @11 37, @12 37, @13 39, @14 39, @15 40, @16 40, @17 42, @18 43, @19 44, @20 45, @21 46, @22 47, @23 47, @24 48, @25 49, @26 50, @27 51, @28 51, @29 51
pp8                                @2 20, @3 21, @4 22, @5 24, @6 26, @7 27, @8 28, @9 31, @10 32, @11 33, @12 33, @13 35, @14 36, @15 36, @16 37, @17 40, @18 41, @19 42, @20 43, @21 45, @22 46, @23 46, @24 47, @25 48, @26 49
score1500-base                     @4 25, @5 28, @6 29, @7 30, @8 30, @9 33, @10 33, @11 34, @12 35, @13 38, @14 39, @15 40, @16 41, @17 43, @18 43, @19 44, @20 45, @21 46, @22 47, @23 48, @24 48, @25 49, @26 50, @27 51, @28 51, @29 51
score1500-base2                    @5 28, @6 29, @7 30, @8 30, @9 33, @10 34, @12 37, @13 38, @14 39, @15 40, @16 41, @17 43, @18 43, @19 44, @20 45, @21 46, @22 47, @23 47, @24 48, @25 49, @26 49, @27 50, @28 51
score1500-kopt                     @3 23, @4 25, @5 27, @6 29, @7 29, @8 30, @9 32, @10 33, @11 34, @12 36, @13 37, @14 38, @15 39, @16 40, @17 41, @18 42, @19 43, @20 43, @21 45, @22 45, @23 46, @24 46, @25 47, @26 48, @27 49
score1500-megatp2                  @4 29, @5 31, @6 32, @7 32, @8 32, @9 34, @10 35, @11 35, @12 38, @13 39, @14 40, @15 41, @16 41, @17 43, @18 43, @19 44, @20 45, @21 46, @22 47, @23 47, @24 48, @25 48, @26 49, @27 50
vfix-B                             @4 27, @5 29, @6 31, @7 31, @8 32, @9 35, @10 37, @11 39, @12 39, @13 41, @14 41, @15 42, @16 43, @17 45, @18 46, @19 47, @20 48, @21 49, @22 50, @23 51, @24 51, @25 52, @26 53, @27 54, @28 54, @29 55, @32 66
vfix-C                             @4 25, @5 27, @6 29, @7 29, @8 30, @9 34, @10 35, @11 36, @12 37, @13 39, @14 39, @15 40, @16 41, @17 42, @18 43, @19 44, @20 45, @21 46, @22 47, @23 48, @24 48, @25 49, @26 49, @27 51, @28 51, @29 51, @30 52
```

## Skipped

* `v2/runs/v2-1p2d-1106T1933-c128:decode`: decode log has no Decode batch lines
* `v2/runs/v2-1p2d-1106T1933-c128:decode1`: decode log has no Decode batch lines
* `v2/runs/v2-1p2d-gate-1106T1916:decode`: no AgentX profiling phase and no score-file window (gate / probe run); not a scoring run
* `v2/runs/v2-1p2d-gate-1106T1916:decode1`: no AgentX profiling phase and no score-file window (gate / probe run); not a scoring run
* `v2/runs/v2-dpbal-k3-c88-1107T0453-c88`: no decode log
* `v2/runs/v2-dpbal-k5-c32-1107T0005-c32`: decode log has no Decode batch lines
* `v2/runs/v2-dpbal-k5-c48-1107T0107-c48`: decode log has no Decode batch lines
* `v2/runs/v2-gate-1node-1104T1952`: no AgentX profiling phase and no score-file window (gate / probe run); not a scoring run
* `v2/runs/v2-nohip-gate-1105T1045`: no AgentX profiling phase and no score-file window (gate / probe run); not a scoring run

## Per-run detail

Not included in this copy (about 230 KB, one section per decode engine: per-graph-batch shares and step time, per-rank running requests, context / KV by period, 10-minute evolution). Full file: `/perf_apps/liyingli/bench_agentx/kreview-20261112/out/logs/BATCH-DIST.md`; machine-readable: `batch_dist.json` next to it. Both regenerate in about 6 s with `scripts/logs_batch_dist.py`.
