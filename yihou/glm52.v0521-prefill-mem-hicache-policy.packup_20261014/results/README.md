# results/

- `results.csv` — one row per bench (built by `scripts/tools/results_csv.yihou.py` from `$R/<round>/agentx-*/agentx_conc*.json`,
  read-only). Lost / in-flight benches have empty metric columns and a note in the `json` column.
- `<round>-c<C>-A<n>/` — per-bench copies: `agentx_conc*.json`, `metrics_plots.png`, `workload_distribution_*`, `benchmark_command.txt`,
  `runtime.env` (incl. `IMAGE_IDS`), `service/{hardware,workers}.json` + `*-server-info.json.gz` (live server args),
  `aiperf/` (`profile_export_aiperf.{csv,json}`, `profile_export_console.txt`, `server_metrics_export.csv.gz`), `benchmark.log.gz`, `runner.log.gz`.
  Not copied (size): `profile_export.jsonl`, `*_timeslices.*`, `server_metrics_export.json` (≈1.4 GB) — see `logs/LOG_INDEX.md`.
  Partial dirs: `pf88-c48-A1` (node failure, 0 profiled), `pf92-c48-A2` (lost), `wts48-c48-A1` (killed, no JSON),
  `wts48-c64-A2` (= wts64, **in flight**, runner.log snapshot ~07:59), `pc64-c64-A2` (aborted).
- `rounds/<round>/` — `status.tsv`, chain / pipeline / handover / cont logs, `audit.txt` + `audit/*.json.gz`, `census.txt`, smoke response,
  `spec_accept.after-bench.txt`, `agentx-*.log.gz`; `rounds/pf88b.handover.log`, `rounds/pf94.handover.log` (armed handovers).
  `rounds/cx90/` = in flight (status + chain log only at packing time; results will land in `$N/runs/cx90/` and `$R/cx90/`).
- `metrics/<round>.metrics.A<n>.tsv.gz` — 5-s `/metrics` samples (epoch, leg, dp rank, metric, value). `wts48.metrics.A2` contains
  wts48 A2 **and** wts64 samples (split at epoch 1791962230 = 07:17:10 UTC); snapshot while still growing.
- `vram/vram.089.tsv` — rocm-smi per-GPU used GiB every 5 s on 089 from 07:29:26 (pc64 tail; after 07:51 = pc64 stop / cx90 bring-up). Snapshot; the sampler was still running.
- `dryruns/` — dryruns of every round today and `tokdiff.*.txt` against the reference.
