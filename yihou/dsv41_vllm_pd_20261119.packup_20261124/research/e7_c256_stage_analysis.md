# E7 PD c256 (e7-pd-c256-11210342) — where does a request spend its time? (2026-11-21, first-hand from metrics_before/after)

Window 3,629 s; 30,038 requests finished on each role. Engine 0 = prefill (DP2, GPU0,1), engine 1 = decode (DP2, GPU2,3).
Source: runs/e7-pd-c256-11210342/bench/metrics_{before,after}_{0,1}.txt (vLLM Prometheus counters, summed over DP ranks).

| stage | per-request mean | Little's-law occupancy (sum/window) |
|---|---:|---:|
| P queue  (request_queue_time)   | 8.04 s  | 66.5 requests waiting on P |
| P prefill (request_prefill_time)| 0.92 s  | 7.6 requests in prefill |
| D queue                          | 0.82 s  | 6.8 requests waiting on D |
| D decode (request_decode_time)   | 15.78 s | 130.6 requests decoding |

- P prefix-cache hit 94.5% (3.937e9 / 4.167e9 prompt tokens); uncached prefill work 229.8 M tokens / 3,629 s = 63.3k tok/s on 2 GPUs
  (31.7k new tok/s per prefill GPU).
- D output 27.0 M tokens / 3,629 s = 7.44k tok/s on 2 GPUs; ITL p50 18.3 ms; D KV avg 32% (headroom).
- Reading (inference, not proven): the P queue (8.0 s, ~1/3 of the ~25.5 s server-side request time) is the largest controllable
  stage, and D has KV headroom -> the next discriminating experiment is more prefill capacity at the same D (e.g. 2P1D: P DP2 x2 or
  P DP4) and/or prefill-side settings from llying's B300 list (long-prefill-token-threshold 8192, max-num-seqs 64, mem 0.93),
  compared at equal total GPUs per tok/s/GPU. Decode-side: 130 concurrent at 18 ms ITL — whether D is near its own limit is unknown
  (needs a D-only rate test, E13).
