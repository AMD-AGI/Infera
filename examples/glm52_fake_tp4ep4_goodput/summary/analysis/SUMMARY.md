# Unprofiled concurrency sweep

Per-request decode rate=(output_tokens-1)/(request_latency-TTFT); SLA goodput counts only tokens in requests that meet this rate threshold.

Decode rate P90 is the 90th percentile of successful per-request rates in ascending order, using linear interpolation. The rate met by at least 90% of requests is described by rate P10 (approximately 1000 / TPOT P90 in ms).

Native SGLang ITL distributes an SSE chunk gap equally among new tokens in that chunk; raw chunk gaps are separately retained.

| Point | Conc | Requests | Output tok/s | TPOT P50/P90 ms | ITL P50/P90/P99 ms | Decode tok/s P50/P90 | ≥70 / ≥80 pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| sweep_c01_r01 | 1 | 128/128 | 239.46 | 4.160 / 4.213 | 3.752 / 4.995 / 5.159 | 240.39 / 244.86 | 100.0% / 100.0% |
| sweep_c02_r01 | 2 | 128/128 | 396.56 | 4.989 / 5.099 | 4.529 / 6.021 / 6.284 | 200.45 / 203.96 | 100.0% / 100.0% |
| sweep_c04_r01 | 4 | 128/128 | 679.89 | 5.856 / 5.928 | 5.289 / 7.037 / 7.344 | 170.77 / 173.86 | 100.0% / 100.0% |
| sweep_c08_r01 | 8 | 128/128 | 1047.28 | 7.556 / 7.703 | 6.866 / 9.101 / 9.549 | 132.34 / 134.66 | 100.0% / 100.0% |
| sweep_c16_r01 | 16 | 128/128 | 1810.50 | 8.744 / 8.906 | 7.949 / 10.555 / 10.849 | 114.36 / 115.56 | 100.0% / 100.0% |
| confirm_c32_r01 | 32 | 512/512 | 2671.75 | 11.879 / 12.041 | 10.686 / 14.200 / 15.005 | 84.18 / 85.39 | 100.0% / 100.0% |
| confirm_c32_r02 | 32 | 512/512 | 2662.38 | 11.953 / 12.088 | 10.696 / 14.199 / 14.903 | 83.66 / 85.61 | 100.0% / 100.0% |
| sweep_full_c32_r01 | 32 | 256/256 | 2685.27 | 11.849 / 12.022 | 10.669 / 14.189 / 14.824 | 84.40 / 86.78 | 100.0% / 100.0% |
| confirm_c40_r01 | 40 | 640/640 | 2880.61 | 13.751 / 13.977 | 12.416 / 16.500 / 17.029 | 72.72 / 73.69 | 100.0% / 0.0% |
| confirm_c40_r02 | 40 | 640/640 | 2904.52 | 13.588 / 13.915 | 12.394 / 16.470 / 16.775 | 73.59 / 74.23 | 100.0% / 0.0% |
| sweep_full_c40_r01 | 40 | 320/320 | 2875.11 | 13.730 / 14.029 | 12.381 / 16.474 / 16.865 | 72.83 / 74.29 | 100.0% / 0.0% |
| boundary_c48_r01 | 48 | 768/768 | 3275.08 | 14.475 / 14.916 | 13.143 / 17.448 / 18.090 | 69.09 / 69.93 | 4.2% / 0.0% |
| high128_c64_r01 | 64 | 512/512 | 3736.31 | 16.579 / 16.893 | 14.923 / 19.864 / 20.868 | 60.32 / 61.59 | 0.0% / 0.0% |
| sweep_full_c64_r01 | 64 | 512/512 | 3833.24 | 16.562 / 16.866 | 14.939 / 19.869 / 21.280 | 60.38 / 62.02 | 0.0% / 0.0% |
| high128_c128_r01 | 128 | 1024/1024 | 4456.20 | 25.883 / 46.575 | 23.244 / 30.943 / 32.336 | 38.64 / 39.14 | 0.0% / 0.0% |

## Maximum tested concurrency

All repeats must pass the selected criterion.

| Target | TPOT limit | P50 criterion | P90 criterion | ≥90% requests meet target |
|---|---:|---:|---:|---:|
| 70 tok/s | 14.2857 ms | 40 | 40 | 40 |
| 80 tok/s | 12.5000 ms | 32 | 32 | 32 |
