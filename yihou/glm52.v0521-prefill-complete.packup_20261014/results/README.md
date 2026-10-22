# Results

- `results.csv` — one row per bench (pc48b/pc48e A1/A2 + q1 A3 reference): score, qps, ISL/OSL, TTFT/TPOT/E2EL, GPU/CPU/overall
  cache hit, ok/error counts, HiCache host-pool tokens, source JSON path. Built by `scripts/tools/results_csv.yihou.py`.
- `pc48{b,e}-A{1,2}-C48/` — `agentx_conc48.json` (the score file), `benchmark_command.txt`, `runtime.env` (no secrets),
  `benchmark.log.gz`, `metrics_plots.png`, `workload_distribution_*`, `aiperf/` (summary CSV/JSON/console, server metrics CSV gz),
  `service/` (workers, hardware). A1 = first run on a fresh stack (unscored); A2 = after flushing decode then prefill (scored).
- `rounds/pc48/` — the failed startup: `status.tsv`, chain/pipeline/handover logs, `engine-prefill.log` (full docker run argv),
  `prefill-0.startup-ValueError.log`.
- `rounds/pc48b/`, `rounds/pc48e/` — `status.tsv`, `*.chain.log`, `*.pipeline.log` (pc48e also `*.handover.log`), `audit.txt`
  (pc48b: 2 FAIL = known false negative, see notes §3; pc48e: AUDIT PASS), `census.txt`, `smoke-response.json` (garbled by
  construction), `spec_accept.after-bench.txt`, `engine-*.log` (docker argv), `audit/*-server-info.json.gz` (live engine args).
- `metrics/` — 5-s `/metrics` samples of both legs during each bench (sampler `scripts/tools/metrics_sampler.yihou.py`).
- `dryruns/` — rendered engine argv per round; `tokdiff.yihou.txt` = q1-137 → pc48 → pc48b → pc48e token diffs.
- `env/` — `collect_env.*.20261014.txt` (274 is this experiment's node), image load / JIT seed logs for 274, pc48 stop log.

Timeline (UTC, 2026-10-13, node 274): 09:53 pc48 start (after x8p) → 10:03 prefill ValueError → 10:11 pc48b start → 10:24 audit STOP
(false negative) → 10:30 A1 → 11:54 A2 → 13:18 done (stack idle) → 15:10 pc48b stopped, pc48e start → 15:53 A1 → 17:26 A2 → 18:50 done.
