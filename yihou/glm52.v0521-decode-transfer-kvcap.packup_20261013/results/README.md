# Results index
- `runs_summary.csv`: pc40c A1/A2 (v1), pc40d A1 (v2). Reference `reference-e2g-A2/agentx_conc40.json` (22,114).
- `pc40c-*/`, `pc40d-*/`: agentx_conc40.json, benchmark.log.gz, command, runtime.env.
- `round-pc40c/`, `round-pc40d/`: census, audit, status, burst*.txt (64-prompt smoke outputs).
- `pc40c.smoke2.log`: v1 230k-token smoke (cap 0.50). `pc40d.chain.log`: v2 bring-up + smoke (cap 0.42) + A1; stops at
  "stage bench A2" (driver host lost). `pc40.chain.log` / `pc40b.chain.log`: burst-node attempts cancelled at 20 min.
- `metrics/`: 5-s /metrics TSVs (gz). `pc40d.metrics.A2-sampler-only-no-bench.tsv.gz` has no bench behind it.
