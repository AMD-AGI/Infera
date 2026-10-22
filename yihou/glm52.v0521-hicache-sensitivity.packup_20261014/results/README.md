# results/

| path | what |
|---|---|
| `results.csv` | all runs of this packup, one row each (score, qps, ISL/OSL, TTFT, TPOT, GPU/CPU/overall hit, token accounting, pool sizes, like-for-like flag, source JSON path). Regenerate: `python3 -B scripts/tools/extract_results.yihou.py > results.csv` |
| `summary.v0521cap2.yihou.md` | whole-session table (all v0521cap2 rounds) from `tools_summ.yihou.py`, as of 2026-10-14 01:50 |
| `hc50/round/` | status.tsv, census.txt (host pool 50.00 GB x4), audit.txt (AUDIT PASS), workers.json, smoke, spec-accept after bench, engine launch logs, chain + pipeline logs, bench wrapper logs (gz), `audit/` live container + server-info JSONs |
| `hc50/A1-C40/`, `hc50/A2-C40/` | `agentx_conc40.json` (score of record), benchmark_command.txt, runtime(.node).env, workload summary + plots (PNG), `service/` container/server-info/hardware/workers JSONs, benchmark.log.gz, runner.log.gz, `aiperf/` (profile_export_aiperf.json/csv, console, server_metrics_export.csv.gz, aiperf.log.gz) |
| `metrics/hc50.metrics.A{1,2}.tsv.gz` | 5-s `/metrics` samples of both legs during each bench (epoch, role, rank, metric, value) |
| `reference-e2g/` | e2g (450 GB, 102) A1/A2 `agentx_conc40.json` + command/env/workload summary, round status/census/audit, chain/pipeline/A2 logs, `audit/prefill-server-info.json` |
| `reference-other/` | k1 A1/A2 (360 GB, C44), e1c r2 (154 GB, C40, decode 0.85), 8-expert v0.5.19 C56 arms (154/274/450 GB) JSONs + params |
| `dryrun/` | rendered launch of hc50 and e2g, unified diff, and the per-leg token diff (`tokdiff.e2g-102_vs_hc50-136.txt`) |

Score = `request_metrics.throughput.per_gpu.total_tput_tps` (tok/s/chip, 8 GPUs).
