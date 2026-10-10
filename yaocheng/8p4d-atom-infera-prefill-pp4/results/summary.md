# GLM-5.2 MXFP4 8P4D, prefill 2 x TP1/PP4 (ATOM + Infera) AgentX

| conc | GPUs | total tok/s/GPU | output tok/s/GPU | TTFT p50 s | TTFT p90 s | ITL p50 ms | intvty p50 | profiled | error dropped | duration s | run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 48 | 12 | 18214.7 | 127.01 | 1.29 | 2.889 | 13.89 | 71.99 | 5406 | 0 | 3629 | crs-pp4-full-20261009-c48 |
| 80 | 12 | 21863.19 | 188.18 | 6.139 | 34.548 | 17.47 | 57.23 | 8374 | 3 | 3630 | crs-pp4-full-20261009-c80 |
| 120 | 12 | 8985.74 | 91.26 | 11.756 | 469.271 | 11.73 | 85.24 | 3929 | 0 | 3622 | crs-pp4-full-20261009-c120 |
