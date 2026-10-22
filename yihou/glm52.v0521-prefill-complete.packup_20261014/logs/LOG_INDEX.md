# Uncopied logs and large artifacts (sizes in bytes, `stat` on 2026-10-14)

Not copied per the pack-up rules (server logs 15-32 MB; aiperf exports 25 MB-1.6 GB). All still on `/shared_nfs` (read-only use).
Ask the user before adding any of them (gzipped, < 4 MB each) to this kit.

## Engine server logs (container stdout, UTC-timestamped; contain the `Prefill batch` / `Decode batch` lines used by
`scripts/tools/logstats.yihou.awk`)

| bytes | path | note |
|---|---|---|
| 15,318,339 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pc48b/server-logs/prefill-0.log` | 10:12:51 → stack stop 2026-10-13 15:10 |
| 32,221,614 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pc48b/server-logs/decode-0.a1.log` | same |
| 15,424,151 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pc48e/server-logs/prefill-0.log` | 15:22 → stack stop 2026-10-14 01:53 |
| 32,353,232 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pc48e/server-logs/decode-0.a1.log` | same |
| 7,508 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pc48/server-logs/prefill-0.log` | **copied** as `results/rounds/pc48/prefill-0.startup-ValueError.log` |

## Bench client logs (≈210 KB each; the gzipped `benchmark.log` per bench IS copied)

| bytes | path |
|---|---|
| 213,053 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pc48b/agentx-c48-A1.log` |
| 213,198 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pc48b/agentx-c48-A2.log` |
| 214,763 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pc48e/agentx-c48-A1.log` |
| 212,414 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pc48e/agentx-c48-A2.log` |
| 211,845-212,629 | `.../runs/pc48{b,e}/agentx-c48-A{1,2}/runner.log` and `aiperf_artifacts/logs/aiperf.log` |

## aiperf raw exports (per bench dir `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/<round>/agentx-c48-<A>/aiperf_artifacts/`)

| round / run | profile_export.jsonl | timeslices.csv | timeslices.json | server_metrics_export.json |
|---|---|---|---|---|
| pc48b A1 | 24,922,214 | 114,462,066 | 48,172,655 | 1,442,575,547 |
| pc48b A2 | 26,597,383 | 120,750,632 | 50,824,084 | 1,490,498,344 |
| pc48e A1 | 26,340,811 | 119,918,676 | 50,475,239 | 1,580,242,119 |
| pc48e A2 | 26,174,704 | 117,427,761 | 49,450,693 | 1,474,498,203 |

Copied instead: `profile_export_aiperf.{csv,json}`, `profile_export_console.txt`, `server_metrics_export.csv.gz`.

## `/metrics` sampler TSVs (5 s) — copied gzipped (`results/metrics/*.tsv.gz`, 0.98-1.01 MB each)

| bytes (raw) | path |
|---|---|
| 11,259,953 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/pc48b/metrics.A1.tsv` |
| 11,616,935 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/pc48b/metrics.A2.tsv` |
| 11,205,323 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/pc48e/metrics.A1.tsv` |
| 11,376,078 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/pc48e/metrics.A2.tsv` |

(`/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/pc48/` is empty — pc48 never reached a bench.)
