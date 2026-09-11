# Unprofiled concurrency sweep

Per-request decode rate=(output_tokens-1)/(request_latency-TTFT); SLA goodput counts only tokens in requests that meet this rate threshold.

Native SGLang ITL distributes an SSE chunk gap equally among new tokens in that chunk; raw chunk gaps are separately retained.

| Point | Conc | Requests | Output tok/s | TPOT P50/P90 ms | ITL P50/P90/P99 ms | Decode tok/s P50 | ≥70 / ≥80 pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| high128_c64_r01 | 64 | 512/512 | 3736.31 | 16.579 / 16.893 | 14.923 / 19.864 / 20.868 | 60.32 | 0.0% / 0.0% |
| high128_c128_r01 | 128 | 1024/1024 | 4456.20 | 25.883 / 46.575 | 23.244 / 30.943 / 32.336 | 38.64 | 0.0% / 0.0% |

## Maximum tested concurrency

All repeats must pass the selected criterion.

| Target | TPOT limit | P50 criterion | P90 criterion | ≥90% requests meet target |
|---|---:|---:|---:|---:|
| 70 tok/s | 14.2857 ms | None | None | None |
| 80 tok/s | 12.5000 ms | None | None | None |
