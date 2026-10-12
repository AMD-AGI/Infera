# Results — GLM-5.2 same-node 1P1D P4D4 on sglang v0.5.21 (node crsuse2-m2m-102)

Draft, filled in as rounds finish. Ledger: `debug/working_process.md` (R001-R026); polls: `debug/poll.md`.
Runtime: `/shared_nfs/yihou_glm52/v0521dbg_20261010/` (`runs/<round>/`, `harness/env.<round>-102.yihou.sh`).

## HEADLINE (2026-10-11 08:35 UTC) — 22,114 tok/s/chip at C40, PASS (R051)

`runs/e2g/agentx-c40-A2/agentx_conc40.json`: target config (above chain), CONC 40, **176,909 tok/s total = 22,114/chip**,
4286 ok / 1 err-dropped, TTFT p50 3.33 / p95 19.21 s, TPOT mean 12.1 ms. Beats the triton packup floor (21,032/chip,
`infera.glm52.pd/yihou/glm52.samenode-triton-idxshare.c40-full.packup_20260923`) by +5.1 % and 22k.
**Protocol matters:** scored run = 2nd AgentX run on the same stack after `/flush_cache` on both legs (cold caches);
the stack's 1st run scores ~19.6k (A1, R048) — a first-run decode-step penalty of +7-12 % at matched load (R046-R049),
mechanism still under investigation (e2i). The previous best 21,696 (R030) used the same protocol by accident.
Warm-protocol replicates (R053, R057): e2i R2 21,921 (decode debug agent on), e2j R2 21,681 (decode host-variant +
cuda-graph bs 1..16) → four warm runs 22,114 / 21,921 / 21,696 / 21,681, mean 21,853 (spread −0.8…+1.2 %).
Pack-up: `glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/`.

## Common chain (all passing rounds)

- Image `infera-sglang:v0521-yihou-1010-mc4340` = v0521 recipe + Mooncake PR #4340 (HIP IPC sub-allocation relocation).
- KV over HIP IPC / XGMI (`MC_TE_FILTERS=<none>` → 0 HCAs). Default ROCr IPC, private PID namespaces,
  **`DECODE_PID_OFFSET=30000`** (decode rank PIDs disjoint from prefill: CLR same-PID guard R020, ROCr
  `@hsa<pid>` socket R012; spur-authz denies `--pid host`, R022).
- Fixed ports < 30000 (R015). Mounts: non-blocking kv-events PUSH (R001), absorb-bmm config off (R003), REGBASE diag.
- `index_share_for_mtp_iteration=false` (H2). Triton DSA prefill+decode. EAGLE 5/1/6, simulated acceptance 3.61
  (as the 0923 AgentX reference; output text garbled by construction).
- AgentX full mode: 3600 s profiling, warmup 10/lane.

## Rounds

| round | delta vs chain | CONC | verdict | ok / err | total tok/s | out tok/s | TTFT p50 / p95 s | ITL p50 ms |
|---|---|---|---|---|---|---|---|---|
| e1c | hicache ratio 1.5 (~154 GB), decode mem 0.85, fused indexer OFF | 40 | **PASS** | 4174 / 2 | 168,389 | 1,213 | 2.85 / 17.6 | 13.2 |
| e2b4 | hicache 450 + CHUNK_GB=512, decode mem 0.95, fused indexer OFF | 44 | **PASS** | 4138 / 1 | 162,774 | 1,170 | 5.41 / 30.5 | 14.0 |
| e2c | **target**: e2b4 + fused indexer ON | 44 | **PASS** | 4179 / 2 | 164,194 | 1,182 | 4.63 / 28.65 | 14.7 |

Reference (v0519, 0923 c40-full, node 137, index_share ON): 168,260 tok/s, 4157 req.
