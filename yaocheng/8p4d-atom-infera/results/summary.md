# GLM-5.2 MXFP4 8P4D (ATOM + Infera) AgentX

| conc | GPUs | total tok/s/GPU | output tok/s/GPU | TTFT p50 s | TTFT p90 s | ITL p50 ms | intvty p50 | profiled | error dropped | duration s | run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 48 | 12 | 17528.52 | 122.15 | 2.239 | 4.893 | 13.96 | 71.62 | 5227 | 0 | 3629 | mem095-b64-c48 |
| 80 | 12 | 24223.04 | 206.44 | 4.646 | 9.917 | 18.61 | 53.73 | 9225 | 4 | 3629 | c80-mem095-b64-c80 |
| 120 | 12 | 22026.78 | 200.87 | 9.734 | 44.753 | 17.77 | 56.27 | 8972 | 1 | 3629 | mem095-b64-c120 |
