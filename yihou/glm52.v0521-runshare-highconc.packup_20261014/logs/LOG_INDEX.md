# Log index — NOT copied (size); absolute paths on /shared_nfs, sizes in bytes (2026-10-14)

| path | bytes | note |
|---|---|---|
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56/server-logs/decode-0.a1.log | 89,163,045 | runshare v2 decode; `YIHOU-RUNSHARE block` lines, "pre-allocated usage" |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56/server-logs/prefill-0.log | 23,395,199 | |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56n/server-logs/decode-0.a1.log | 37,552,437 | no patch |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56n/server-logs/prefill-0.log | 22,935,634 | |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56/agentx-c56-A1/aiperf_artifacts/ | 1,713,938,833 | aiperf raw per-request records |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56/agentx-c56-A2/aiperf_artifacts/ | 1,827,017,179 | |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56/agentx-c64-A3/aiperf_artifacts/ | 2,108,731,237 | |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56n/agentx-c56-A1/aiperf_artifacts/ | 1,710,367,203 | |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56n/agentx-c56-A2/aiperf_artifacts/ | 1,757,269,255 | |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56n/agentx-c64-A3/aiperf_artifacts/ | 2,024,809,603 | |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56/agentx-c56-A1/runner.log | 230,557 | in-dir runner log (round-level copy is in results as runner-tail .gz) |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56/agentx-c56-A2/runner.log | 217,702 | |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56/agentx-c64-A3/runner.log | 222,836 | |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56n/agentx-c56-A1/runner.log | 216,281 | |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56n/agentx-c56-A2/runner.log | 217,202 | |
| /shared_nfs/yihou_glm52/v0521dbg_20261010/runs/tb56n/agentx-c64-A3/runner.log | 222,934 | |
| /shared_nfs/yihou_glm52/v0521cap2_20261013/logs/imgload.crsuse2-m2m-085.yihou.log | small | image load on 085 (05:10-05:14 UTC) |
| /shared_nfs/yihou_glm52/v0521cap2_20261013/logs/seed.085.yihou.log | small | JIT seed SEED_OK 06:26:33 |

Raw /metrics TSVs (11.3-11.9 MB each) are in `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/{tb56,tb56n}/metrics.{A1,A2,A3}.tsv`;
gzipped copies (< 1 MB each) are in `../results/metrics/`.

Gzipped server logs were not attempted (task rule: raw server logs are indexed, not copied); the user decides.
