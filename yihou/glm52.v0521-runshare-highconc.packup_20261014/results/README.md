# Results

- `results.csv` — one row per run (tb56 A1-A3, tb56n A1-A3) plus references (q1 C44/C48/C52 on 137, pc40d A1 C40 runshare
  v2 on 102, t2f C40/C56/C72/C96 v0.5.19). Regenerate: `python3 -I ../scripts/make_results_csv.yihou.py <packup dir>`.
  Columns: score = `request_metrics.throughput.per_gpu.total_tput_tps`; qps = `request_metrics.qps.mean`; ISL =
  `tokens.input.mean`; TTFT/TPOT from `latency`; hits from `server_metrics.cache`; `cpu_pool_tok` = `server_metrics.kv_cache.cpu_total_tokens`.
- `summary.v0521cap2.yihou.md` — session-wide table (all 2026-10-13/14 rounds) from `tools_summ.yihou.py`, as of 01:50 UTC 10-14.
- `tb56/`, `tb56n/` — per round: `status.tsv`, `audit.txt` (+ `audit/` JSONs), `census.txt`, `smoke-response.json`,
  `spec_accept.after-bench.txt`, `workers.json`, `engine-{prefill,decode.a1}.log`, `<round>.chain.log`,
  `<round>.pipeline.log` (tb56n also `tb56n.handover.log`); per run `agentx-c<C>-A<n>/`: `agentx_conc<C>.json`,
  `benchmark.log.gz`, `benchmark_command.txt`, `runtime.env` (checked: no secrets), `workload_distribution_summary.txt`,
  `metrics_plots.png`, `workload_distribution_plots.png`, exit codes, `service/` (container + server-info JSONs; `api_key`/`password` are null);
  `agentx-c<C>-A<n>.runner-tail.log.gz` = the round-level runner log.
- `metrics/<round>.metrics.<A1|A2|A3>.tsv.gz` — /metrics sampler, 5 s, decode :29357 and prefill :29101, columns
  `epoch  role  dp  metric  value`.
- `reference/` — comparators copied from sibling packups (see `../README.md`).
