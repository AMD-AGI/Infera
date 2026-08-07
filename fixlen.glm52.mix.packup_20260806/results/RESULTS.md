# Results — GLM-5.2-MXFP4 mix, fixed-length sweep

12 points, **all completed, no failures, no error entries**. `completed` equals
`conc × 10` on every row (the InferenceX `--num-prompts` convention), and the
per-request `errors[]` array is entirely empty strings on all 12 — verified
programmatically against the shipped jsonl, not asserted.

Measured 2026-08-06, 07:13:48 → 09:45:14 UTC on `chi2835`. Per-point timestamps
are the jsonl file mtimes, listed at the bottom.

**ISL is the 10 % fresh remainder of Case A's prompt, not the full prompt.** See
`../README.md` and `../notes.md` §5 before reading any latency number.

## Full-width table

Latencies in ms. `cache_hit_%` is `cache_report.cache_hit_rate_pct`;
`max_conc_req` is the peak concurrent requests the client observed.

| arm | isl | osl | conc | completed | dur_s | req/s | in_tok/s | out_tok/s | tot_tok/s | ttft_p50 | ttft_p90 | ttft_p95 | ttft_p99 | tpot_p50 | tpot_p90 | tpot_p99 | itl_p50 | itl_p99 | e2e_p50 | e2e_p90 | e2e_p99 | cache_hit_% | max_conc_req |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p50 | 7400 | 320 | 1 | 10 | 38.8 | 0.258 | 1909.3 | 82.56 | 1991.9 | 1047 | 1058 | 1063 | 1067 | 9.13 | 9.86 | 10.07 | 25.87 | 31.72 | 3967 | 4193 | 4259 | 9.95 | 2 |
| p50 | 7400 | 320 | 8 | 80 | 64.7 | 1.236 | 9147.7 | 395.58 | 9543.3 | 2012 | 2685 | 3968 | 4233 | 13.55 | 16.30 | 22.31 | 28.31 | 81.05 | 6271 | 7818 | 9654 | 12.38 | 14 |
| p50 | 7400 | 320 | 16 | 160 | 75.3 | 2.124 | 15720.8 | 679.82 | 16400.6 | 1978 | 2557 | 2766 | 3333 | 16.56 | 24.47 | 28.37 | 30.97 | 899.37 | 7090 | 10038 | 11367 | 49.51 | 27 |
| p50 | 7400 | 320 | 24 | 240 | 102.8 | 2.334 | 17268.7 | 746.75 | 18015.4 | 2276 | 5608 | 6002 | 7226 | 21.70 | 31.32 | 36.39 | 32.97 | 1173.89 | 10313 | 13696 | 15944 | 45.86 | 35 |
| p90 | 15500 | 3300 | 1 | 10 | 294.2 | 0.034 | 526.9 | 112.18 | 639.1 | 1328 | 1654 | 2376 | 2954 | 8.14 | 9.78 | 9.85 | 25.76 | 27.00 | 29153 | 33586 | 33904 | 52.52 | 2 |
| p90 | 15500 | 3300 | 8 | 80 | 435.3 | 0.184 | 2848.9 | 606.55 | 3455.5 | 4523 | 5968 | 6277 | 6885 | 11.87 | 13.24 | 15.63 | 28.58 | 59.63 | 42588 | 49211 | 55857 | 20.92 | 10 |
| p90 | 15500 | 3300 | 16 | 160 | 517.5 | 0.309 | 4792.6 | 1020.36 | 5812.9 | 3208 | 6233 | 6878 | 7870 | 14.03 | 17.13 | 18.80 | 31.48 | 63.10 | 50259 | 60915 | 65726 | 39.27 | 19 |
| p90 | 15500 | 3300 | 24 | 240 | 594.7 | 0.404 | 6255.1 | 1331.73 | 7586.8 | 2691 | 5848 | 6421 | 7377 | 16.52 | 20.79 | 23.83 | 34.27 | 976.59 | 57663 | 71966 | 81354 | 43.05 | 30 |
| p99 | 23500 | 17000 | 1 | 10 | 1258.3 | 0.008 | 186.8 | 135.10 | 321.9 | 1440 | 1873 | 1893 | 1909 | 7.13 | 7.82 | 8.23 | 26.22 | 27.45 | 122381 | 134511 | 141626 | 68.98 | 2 |
| p99 | 23500 | 17000 | 8 | 80 | 1599.4 | 0.050 | 1175.5 | 850.34 | 2025.8 | 5368 | 7284 | 8279 | 8558 | 8.86 | 9.77 | 11.05 | 28.87 | 30.30 | 155093 | 171834 | 195106 | 18.28 | 10 |
| p99 | 23500 | 17000 | 16 | 160 | 1837.2 | 0.087 | 2046.6 | 1480.51 | 3527.1 | 4045 | 6229 | 7119 | 7839 | 10.15 | 11.50 | 12.69 | 31.71 | 35.38 | 176185 | 202216 | 219991 | 42.08 | 19 |
| p99 | 23500 | 17000 | 24 | 240 | 2108.0 | 0.114 | 2675.6 | 1935.51 | 4611.1 | 3522 | 5897 | 6972 | 9206 | 11.57 | 13.46 | 16.77 | 35.47 | 58.07 | 200049 | 232657 | 288194 | 46.64 | 27 |

Higher-precision values (full float, plus `mean_*` and `std_*` for every metric)
are in `summary.csv` and in each `fixlen/*.jsonl`.

## Observations, stated as observations

Each of these is a reading. Where a mechanism is not established, none is given.

1. **Output throughput scales sub-linearly with concurrency on every arm.**
   p50: 82.6 → 395.6 → 679.8 → 746.8 tok/s across conc 1/8/16/24 (a 24× rise in
   concurrency buys 9.0× throughput). p90: 112.2 → 1331.7 (11.9×). p99:
   135.1 → 1935.5 (14.3×). The p50 arm flattens hardest between conc 16 and 24
   (**+9.8 %**), while p90 and p99 are still gaining there (+30.5 % and
   +30.7 %).

2. **TPOT p50 rises with concurrency on every arm**, monotonically:
   p50 9.13 → 21.70 ms, p90 8.14 → 16.52 ms, p99 7.13 → 11.57 ms.

3. **TPOT p50 at fixed concurrency falls as OSL grows.** At conc 1: 9.13
   (osl 320) → 8.14 (3300) → 7.13 ms (17000). At conc 24: 21.70 → 16.52 →
   11.57 ms. No mechanism claimed.

4. **ITL p50 sits in a narrow 25.76–35.47 ms band across all 12 points**, while
   TPOT p50 ranges 7.13–21.70 ms. ITL p99 is the volatile one: 1174 ms
   (p50/c24), 977 ms (p90/c24), 899 ms (p50/c16), 81 ms (p50/c8) — the
   remaining eight points all sit at ≤63.1 ms.

5. **TTFT p50 is non-monotonic on the p90 and p99 arms** — it rises c1→c8, then
   falls at c16 and again at c24 (p90: 1328/4523/3208/2691; p99:
   1440/5368/4045/3522). **No explanation is offered.** `notes.md` §8 records
   what would settle it.

6. **`cache_hit_%` is emergent, not controlled, and is not a workload hit rate.**
   `--dataset-name random` builds prompts independently, so any hit is
   coincidental overlap plus radix retention across requests. It moves
   non-monotonically (p90: 52.5 / 20.9 / 39.3 / 43.1 %). Reported to confirm the
   mechanism is wired, not to be read as the agentic workload's 89–90 %.

7. **The kvd storage tier served tokens on exactly two points** — p50/c16
   (`storage_cached_tokens` 14,528) and p50/c24 (28,736), both reporting
   `storage_backend: "InferaKvdBackend"`. On the other ten, `storage_cached_tokens`
   is null. Host-tier hits appear on four points: p50 c8 (36,736), c16 (249,088),
   c24 (146,624), and p99 c24 (1,600); all p90 points and the remaining p99
   points are device-only. No claim is made about why the p50 arm is where the
   host and storage tiers engaged.

8. **Client-observed peak concurrency exceeds the requested cap on every arm.**
   `max_concurrent_requests` from `bench_serving` reads 2 at `--max-concurrency 1`
   on all three arms; at 16 it reads 27 (p50) / 19 (p90) / 19 (p99); at 24 it
   reads 35 (p50) / 30 (p90) / 27 (p99). Recorded as measured; not investigated.
   Note this is a client-side counter, and the `completed` counts confirm no
   extra requests were sent.

## Per-point timestamps

File mtime = end of that point's run (UTC, 2026-08-06).

| point | finished |
|---|---|
| p50 c1 | 07:13:58 |
| p50 c8 | 07:15:18 |
| p50 c16 | 07:16:48 |
| p50 c24 | 07:18:46 |
| p90 c1 | 07:23:56 |
| p90 c8 | 07:31:31 |
| p90 c16 | 07:40:24 |
| p90 c24 | 07:50:35 |
| p99 c1 | 08:11:50 |
| p99 c8 | 08:38:49 |
| p99 c16 | 09:09:46 |
| p99 c24 | 09:45:14 |

## What is in `fixlen/*.jsonl`, and what was trimmed

Each file holds **one json object** — the run's full `bench_serving` record:
every aggregate metric (`mean`/`median`/`std`/`p90`/`p95`/`p99` for e2e, TTFT,
TPOT, ITL), the `cache_report` block, and these index-aligned per-request arrays
of length `completed`:

`input_lens`, `output_lens`, **`ttfts`**, **`cached_tokens`**,
`cached_tokens_details`, `errors`.

**Two arrays were removed to keep the packup committable:** `itls`
(per-request inter-token latency lists) and `generated_texts`. Together they are
**96 MB of the 96 MB** — everything else is 145 KB. `ttfts` and `cached_tokens`
were **kept**, because they are what the open TTFT question in `notes.md` §8
needs.

The untouched originals remain on the cluster at

```
/mnt/vast/c_huggingface/glm52_mix_20260806/results/fixlen/*.jsonl
```

(96 MB raw / 28 MB gzipped). Nothing there was deleted or modified.

## Re-deriving this table

```bash
python3 ../scripts/summarize_fixlen.py ./fixlen
```

Rewrites `summary.csv` and prints the narrow markdown table. It reads the
**last** json object per file, because `bench_serving` appends. It works
unchanged on the trimmed files — it only ever touched scalar fields.

The full-width table above was built from the same jsonl; the extra columns are
straight field reads (`p95_ttft_ms`, `p99_itl_ms`, `input_throughput`,
`total_throughput`, `cache_report.cache_hit_rate_pct`, `max_concurrent_requests`).
