# Log index — artifacts NOT copied into this packup (size or policy); sizes read 2026-10-14 ~02:03 UTC

Raw listing at pack-up time: `logs/ls_raw.txt`. Nothing below was moved or modified.

## hc50 server logs

| path | size (B) | note |
|---|---|---|
| `crsuse2-m2m-136:/mnt/m2m_nobackup/yihou_v0521dbg/runs/hc50/server-logs/prefill-0.log` | 15,911,382 | **full** prefill log (mtime 2026-10-13 19:22); node-local, plain ssh; not backed up (`m2m_nobackup`) |
| `crsuse2-m2m-136:/mnt/m2m_nobackup/yihou_v0521dbg/runs/hc50/server-logs/decode-0.a1.log` | 26,973,122 | **full** decode log (mtime 19:22) |
| `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/hc50/server-logs/prefill-0.log` | 278,854 | snapshot copied back at bring-up (16:28) — covers startup + census only |
| `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/hc50/server-logs/decode-0.a1.log` | 17,258,600 | snapshot at 16:28 (startup + CUDA-graph capture) |

The node-local copies are the only full logs of the A1/A2 benchmark hours. If they matter, copy them to NFS (or
gzip into a packup) before 136's local disk is reused.

## hc50 aiperf raw artifacts (per bench; A1 / A2)

Under `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/hc50/agentx-c40-A{1,2}/aiperf_artifacts/`:

| file | A1 size (B) | A2 size (B) |
|---|---|---|
| `server_metrics_export.json` | 1,414,205,056 | 1,388,295,977 |
| `profile_export_aiperf_timeslices.csv` | 117,904,438 | 121,759,440 |
| `profile_export_aiperf_timeslices.json` | 49,595,991 | 51,225,223 |
| `profile_export.jsonl` | 25,413,794 | 27,047,960 |

Copied (small): `profile_export_aiperf.{json,csv}`, `profile_export_console.txt`, `server_metrics_export.csv` (gz),
`logs/aiperf.log` (gz) -> `results/hc50/A{1,2}-C40/aiperf/`. Node-side twin of the output dir (client wrote there):
`crsuse2-m2m-136:/mnt/m2m_nobackup/yihou_v0521dbg/runs/hc50/agentx-c40-A{1,2}/` [contents not listed].

## /metrics sampler TSVs

| path | size (B) | in packup |
|---|---|---|
| `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/hc50/metrics.A1.tsv` | 11,012,144 | yes, gz 955,420 B (`results/metrics/`) |
| `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/hc50/metrics.A2.tsv` | 11,097,956 | yes, gz 957,997 B |

## Reference e2g server logs (102)

| path | size (B) |
|---|---|
| `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/e2g/server-logs/prefill-0.log` | 15,512,306 |
| `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/e2g/server-logs/decode-0.a1.log` | 26,789,303 |

## Driver logs (copied)

`hc50.chain.log`, `hc50.pipeline.log` -> `results/hc50/round/`; `agentx-c40-A{1,2}.log` (bench wrapper, ~210-224 KB)
gzipped there; `e2g.chain.log`, `e2g.pipeline.log`, `e2g.A2.log` -> `results/reference-e2g/`.
