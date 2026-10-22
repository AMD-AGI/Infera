# Results index
- `analysis/runs_summary.csv` all runs; `analysis/q1_residence_c44_c48_c52.txt` Little's-law phase residence (C44/C48/C52).
- `<run>/` agentx_conc*.json (score: request_metrics.throughput.per_gpu.total_tput_tps), benchmark.log.gz, command, env.
- `round-<r>/` census/audit/status/burst smoke outputs. `metrics/` 5-s /metrics TSVs (gz; cols ts leg dp metric value).
- `pc40d.metrics.A2-sampler-only-no-bench.tsv.gz`: sampler ran but the A2 bench never started (driver host lost).
- `*.chain.log`, `dryrun.*.txt`: drivers' logs and rendered argv per round. aiperf_artifacts not copied (on NFS runs/).
