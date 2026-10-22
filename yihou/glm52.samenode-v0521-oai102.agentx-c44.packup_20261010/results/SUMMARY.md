# Results — three passing AgentX full rounds on crsuse2-m2m-102, sglang v0.5.21

Source of every number: `results/<round>/agentx_conc*.json` (machine-readable, copied from
`/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/<round>/`). Full aiperf artifacts (~1.7 GB/round) are
left at that NFS path (`runs/<round>/agentx-c4x*/aiperf_artifacts/`).

All rounds share one chain:
- image mc4340;
- default IPC, private PID namespaces + `DECODE_PID_OFFSET=30000`;
- ports < 30000;
- mounts R001/R003 (+ REGBASE diag);
- `index_share=false`;
- triton DSA;
- EAGLE 5/1/6 with simulated acceptance 3.61;
- AgentX full: 3600 s, warmup 10/lane.

| round | config delta | CONC | profiling window (UTC) | verdict | ok / err | total tok/s (per GPU) | in / out tok/s | TTFT p50 / p90 / p95 s | ITL p50 / p95 ms | intvty p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| e1c | hicache ratio 1.5 (153.81 GB/rank), decode mem 0.85, fused indexer OFF | 40 | 08:19:31-09:20:11 | **PASS** | 4174 / 2 | 168,389 (21,049) | 167,176 / 1,213 | 2.85 / 11.63 / 17.62 | 13.2 / 18.3 | 75.8 |
| e2b4 | hicache 450 + CHUNK_GB=512, decode mem 0.95, fused indexer OFF | 44 | 10:04:41-11:05:21 | **PASS** | 4138 / 1 | 162,774 (20,347) | 161,604 / 1,170 | 5.41 / 21.5 / 30.49 | 14.0 / 19.7 | 71.5 |
| **e2c** | **target**: e2b4 + fused indexer ON (v0521 default) | 44 | 12:12:35-13:13:15 | **PASS** | 4179 / 2 | 164,194 (20,524) | 163,011 / 1,182 | 4.63 / 19.95 / 28.65 | 14.7 / 21.6 | 67.9 |

Errors are all `InvalidInferenceResultError`, below the 0.10 threshold.

Health after every bench, both legs (grep of the full server logs):
- Memory access fault 0;
- Traceback 0;
- KV send/get failures 0;
- `Error code: 201` 0.

In e2b4 and e2c the 450 GB host pool reached ~100 % `cpu_kv_usage`, crossing the H1 boundary
(~61 %) at ~10:20 and ~12:26.

Census (bring-up) per round: `results/<round>/census.txt`. Live-config audit: `results/<round>/audit.txt`
(`AUDIT PASS`). Rendered launch commands: `results/<round>/dryrun.*.txt`.

Reference (v0.5.19, `glm52.samenode-triton-idxshare.c40-full.packup_20260923`, node 137,
index_share ON): 168,260 tok/s, 4157 requests. Not like-for-like.

Side analyses (`results/analysis/`):
- warmup first wave (`firstwave.yihou.txt`);
- router dispatch (`router-firstwave.yihou.txt`);
- DP0 / per-DP prefill windows (`prefill-tput-window18.yihou.txt`, `prefill-perdp-windows.yihou.txt`);
- e2b4 vs proven v0519 r013 param diff (`diff.e2b4-vs-v0519r013.yihou.txt`).
