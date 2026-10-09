# Round003 — fixed-length C32 acceptance

PASS2026-09-23.32 warmup +256 measured, each4096input/1024output from server usage.288/288 records successful, peak inflight32, client exit0. Measured duration184.0876s; output1424.0176tok/s; input+output7120.0878tok/s. MeanTTFT1.12591s; meanTPOT20.8694ms; mean latency22.4763s.

Before measurement: round004 single128/16 and burst256/32,C8,16requests smoke passed. After measurement all3 health endpoints HTTP200. Effective server args TP4/DP4/EP4/DPA1/MTPoff/HiCacheoff/fp8KV/triton confirmed. Successful full logs no traceback/native fault/NCCL error/NaN markers. Prefill transfer count deltas73+72+72+71=288 across4ranks; health-related stage counters can include additional probes.

Artifacts: benchmark/summary.json, requests.jsonl, config.json; benchmark.exit-code; after-benchmark/ full snapshots; verification.json. This is synthetic model execution/performance, not quality validation or fully occupied KV capacity soak. No further benchmark started.
