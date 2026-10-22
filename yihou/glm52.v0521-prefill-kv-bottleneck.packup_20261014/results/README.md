# Results

- `results.csv` — one row per bench (12 rows: q1 A3 + 11 rounds' benches incl. the crashed x8d A2), built from the copied
  AgentX JSONs by `scripts/session/results_csv.yihou.py results/`. Columns: round, run, node, job, variant, conc, routed
  experts per leg, KV tokens per DP rank per leg, score (`request_metrics.throughput.per_gpu.total_tput_tps`), qps,
  ISL/OSL means, TTFT mean/p50/p95, TPOT, E2E latency, GPU/CPU/overall cache hit, prompt tokens total/computed, HiCache
  host pool tokens, ok / error-dropped / error categories, duration, JSON path.
- `<round>/` — `status.tsv` (pipeline stages), `audit.txt` (audit v3, live server args + mounts per role), `census.txt`,
  `smoke-response.json`, `workers.json`, `spec_accept.after-bench.txt`, `engine-*.log` (docker run lines), `audit/*.json.gz`
  (container inspect + /server_info per leg), `<round>.chain.log` / `.pipeline.log` / `.handover.log` (driver logs),
  `agentx-c<C>-A<n>.driver.log.gz`, `metrics/metrics.<tag>.tsv.gz` (5 s /metrics, both legs, all DPs).
- `<round>/agentx-c<C>-A<n>/` — `agentx_conc<C>.json` (THE result), `benchmark_command.txt`, `runtime.env` (no secrets),
  `metrics_plots.png`, `workload_distribution_*`, `benchmark.log.gz`, `runner.log.gz`, `aiperf_artifacts/`
  (`profile_export_aiperf.{json,csv}`, `profile_export_console.txt`, `server_metrics_export.csv.gz`). Raw aiperf exports
  (up to 1.4 GB) are only indexed in `logs/LOG_INDEX.md`.
- `q1/` — the reference: q1 round files and its C48 A3 run (137, 2026-10-12) + its /metrics TSV (`metrics.A3c48.tsv.gz`).
- `tokdiff/` — `tokdiff.yihou.sh dryrun.q1-137 vs dryrun.<round>` output (rendered argv/env deltas per leg).

| round | run | C | score | qps | ISL | TTFT mean/p95 s | TPOT ms | GPU hit | CPU hit | ok | err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q1 | A3 | 48 | 22,306 | 1.220 | 145,303 | 11.97 / 38.19 | 12.1 | 0.839 | 0.088 | 4425 | 2 |
| x8d | A1 (suspect HW) | 48 | 20,230 | 1.146 | 140,330 | 14.03 / 46.28 | 14.0 | 0.893 | 0.041 | 4155 | 3 |
| x8d | A2 | 48 | crashed (0 ok, 59 Bad Gateway) | | | | | | | 0 | 60 |
| x8p | A1 | 48 | 25,347 | 1.392 | 144,802 | 5.03 / 16.05 | 14.5 | 0.963 | 0.002 | 5043 | 2 |
| x8p | A2 | 48 | 25,905 | 1.409 | 146,053 | 5.45 / 17.02 | 13.2 | 0.950 | 0.002 | 5115 | 1 |
| x8d2 | A1 | 48 | 22,511 | 1.231 | 145,265 | 9.95 / 37.04 | 14.3 | 0.903 | 0.043 | 4464 | 1 |
| x8d2 | A2 | 48 | 23,591 | 1.286 | 145,686 | 10.43 / 33.44 | 11.9 | 0.857 | 0.076 | 4669 | 2 |
| x8d2 | A3 | 52 | 21,713 | 1.268 | 136,044 | 16.12 / 52.11 | 11.3 | 0.800 | 0.118 | 4599 | 5 |
| x8pc | A1 | 48 | 21,591 | 1.202 | 142,780 | 11.41 / 38.23 | 14.0 | 0.899 | 0.043 | 4347 | 0 |
| x8pc | A2 | 48 | 22,489 | 1.234 | 144,759 | 12.28 / 43.83 | 12.4 | 0.847 | 0.080 | 4477 | 1 |
| x128p | A1 | 48 | 24,316 | 1.346 | 143,619 | 5.64 / 16.74 | 15.1 | 0.959 | 0.004 | 4878 | 180 |
| x128p | A2 | 48 | 25,297 | 1.387 | 145,013 | 6.23 / 19.47 | 13.3 | 0.944 | 0.005 | 5030 | 180 |

HiCache host pool = 10,016,064 tokens per rank (450 GB) in every run.
