# A/B: runshare v2 (tb56) vs no patch (tb56n) — same node 085, same image, same config otherwise

Evidence tags: [M] measured from files in this packup / the run dirs; [I] inferred.

## 1. AgentX (from `../results/results.csv`) [M]

| run | CONC | round | score | qps | ISL mean | TTFT mean / p50 / p95 s | TPOT ms | GPU / CPU / overall hit | ok / err |
|---|---|---|---|---|---|---|---|---|---|
| A1 fresh | 56 | tb56 | 17,545 | 1.038 | 134,256 | 28.22 / 19.44 / 85.95 | 14.97 | 0.814 / 0.082 / 0.896 | 3764 / 3 |
| A1 fresh | 56 | tb56n | 17,791 | 1.057 | 133,693 | 27.76 / 17.89 / 87.55 | 14.92 | 0.819 / 0.078 / 0.897 | 3822 / 2 |
| A2 warm | 56 | tb56 | 17,611 | 1.053 | 132,674 | 31.08 / 18.56 / 98.68 | 12.80 | 0.718 / 0.166 / 0.884 | 3824 / 3 |
| A2 warm | 56 | tb56n | 18,016 | 1.041 | 137,357 | 30.09 / 22.38 / 92.78 | 12.54 | 0.680 / 0.204 / 0.885 | 3780 / 4 |
| A3 warm | 64 | tb56 | 15,070 | 0.926 | 129,104 | 51.22 / 43.51 / 135.36 | 11.13 | 0.561 / 0.276 / 0.837 | 3363 / 2 |
| A3 warm | 64 | tb56n | 15,108 | 0.930 | 128,977 | 50.84 / 44.14 / 134.50 | 11.50 | 0.544 / 0.296 / 0.841 | 3374 / 1 |

Deltas (tb56 vs tb56n): A1 -1.4 %, A2 -2.2 %, A3 -0.3 %. At A2 the patch round had +1.2 % qps but -3.4 % mean ISL
(the AgentX workload mix of completed requests differs between runs) [M]; TTFT mean +1.0 s, p50 -3.8 s, p95 +5.9 s [M].

## 2. Decode state under load (/metrics, 5 s; `../results/metrics/*.tsv.gz`) [M]

Max over each bench window, per decode DP rank 0/1/2/3:

| run | round | token_usage max | running reqs max | transfer-queue reqs max | retracted |
|---|---|---|---|---|---|
| A1 | tb56 | 0.98 / 0.98 / 0.91 / 0.99 | 16 / 20 / 15 / 16 | 21 / 25 / 13 / 24 | 0 |
| A1 | tb56n | 0.97 / 0.99 / 0.96 / 1.00 | 16 / 14 / 17 / 16 | 20 / 25 / 22 / 23 | 0 |
| A2 | tb56 | 0.95 / 0.92 / 1.00 / 0.97 | 19 / 20 / 15 / 13 | 21 / 20 / 19 / 14 | 0 |
| A2 | tb56n | 0.99 / 1.00 / 0.99 / 0.99 | 20 / 21 / 20 / 12 | 24 / 16 / 19 / 25 | 0 |
| A3 | tb56 | 0.99 / 0.99 / 0.98 / 1.00 | 15 / 13 / 11 / 10 | 20 / 19 / 23 / 16 | 0 |
| A3 | tb56n | 1.00 / 1.00 / 1.00 / 1.00 | 16 / 13 / 17 / 14 | 33 / 24 / 26 / 21 | 0 |

Decode log "pre-allocated usage" max over the whole stack lifetime: tb56 **0.50**, tb56n **0.98** [M grep].
Running reqs per DP peaked at 21, below the 24/DP implied by `--max-running-requests 96` with DP4 [M peak; per-DP split I].

## 3. Cap activity in tb56 (decode log; 1 line per 100th block per DP rank) [M]

| window | DP0 | DP1 | DP2 | DP3 | skip | stop |
|---|---|---|---|---|---|---|
| A1 (<= 08:22:58, incl. bring-up/smoke/warmup) | 19,554 | 18,588 | 17,236 | 19,931 | 22,693 | 52,616 |
| A2 (08:22:58-09:52:48) | 19,385 | 17,656 | 22,599 | 18,969 | 22,730 | 55,879 |
| A3 (09:52:48-11:33:42) | 31,147 | 29,047 | 25,986 | 26,882 | 29,247 | 83,815 |
| total | | | | | 74,670 | 192,310 |

First block 06:56:57 (DP3: nonrunning 1,295,424 + cand 329,308 >= cap 1,381,088, running 0, xfer 10). `stop` =
anti-starvation FIFO break (head request waited > 5 s).

## 4. Against the old sweep (t2f, `../results/reference/t2f-v0519-crossnode/`) [M numbers, I comparability]

| | C40 | C56 | C64 | C72 | C96 |
|---|---|---|---|---|---|
| t2f (v0.5.19, cross-node P4D4, HiCache ratio 1.5 = 3,423,488 tok/rank, decode mem 0.85, max_running 128, tilelang DSA) | 19,799 | 14,676 (-25.9 %) | — | 8,764 | 3,028 |
| tb56n (v0.5.21, same-node, HiCache ratio 1.5 = 3,649,664 tok/rank, decode mem 0.95, max_running 96, triton DSA) | not run | 18,016 (A2) | 15,108 (A3) | — | — |
| tb56 (same + runshare v2) | not run | 17,611 (A2) | 15,070 (A3) | — | — |

Absolute C56 is +20-23 % above t2f C56, but with and without the patch alike, so that gap is not attributable to the
patch. The C40->C56 regression of the tb56/tb56n config is **not measured** (no C40 at this config on 085). The
v0.5.21 points with HiCache 450 (q1, node 137, `../results/reference/q1-137/`): C44 21,862 / C48 22,306 / C52 21,418.

## 5. Reading (suspended)

- The patch enforced its cap (0.50) and was busy throughout, but neither CONC showed a throughput gain; single runs,
  differences of -2.2 % and -0.3 %.
- Decode KV was near full (token_usage 0.91-1.00) with 0 retractions in both rounds; prefill-side and transfer-side
  residence at C56/C64 were not decomposed for this packup [open]. The measurement that would locate the C56/C64
  limiter: Little's-law residence per phase (`residence.yihou.py` in the kvcap packup) on these TSVs, plus a C40 point
  at this config on the same node.
- Whether the -2.2 % at C56 is noise: open; needs replicate A2 runs (or alternating A/B on one node).
