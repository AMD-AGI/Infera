# Log index

## Included (`logs/server/<round>/`, gzip of the full NFS copy; every .gz < 4 MB, no excerpts were needed)

| round | file | raw size (bytes) | covers |
|---|---|---|---|
| pf90 | prefill-0.log, decode-0.a1.log | 16,774,428 / 32,389,102 | 01:55 → 05:16 (full round) |
| pf92 | prefill-0.log, decode-0.a1.log | 16,049,147 / 31,922,188 | 02:17 → 05:07:19 (ends at the node failure) |
| pf88 | prefill-0.log, decode-0.a1.log | 1,101,210 / 23,241,140 | 02:05 → 02:32:36 (ends at the node failure) |
| pc64 | prefill-0.log, decode-0.a1.log | 7,839,411 / 27,654,275 | 05:22 → 07:51 (A1 + aborted A2) |
| wts48 | prefill-0.log, decode-0.a1.log | 255,353 / 22,908,820 | **startup only, until 03:57** (NFS copy; runtime logs are node-local on 136) |

## Not included

| what | where | why |
|---|---|---|
| wts48 / wts64 runtime server logs | node-local on `crsuse2-m2m-136` under `/mnt/m2m_nobackup/yihou_v0521dbg/` [inferred location from NODE_RW_ROOT] | `/mnt` not reachable for packing |
| aiperf raw artifacts | `$R/<round>/agentx-*/aiperf_artifacts/{profile_export.jsonl (~30 MB), profile_export_aiperf_timeslices.{csv,json} (~131/55 MB), server_metrics_export.json (~1.4 GB)}` | size |
| aiperf internal logs | `$R/<round>/agentx-*/aiperf_artifacts/logs/` | size / redundancy with benchmark.log |
| cx90 / wts64 / pf94 outputs | `$R/cx90`, `$N/runs/cx90`, `$R/wts48/agentx-c64-A2`, `$R/pf94` | in flight at packing time |

`$R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs`, `$N=/shared_nfs/yihou_glm52/v0521cap2_20261013`.
