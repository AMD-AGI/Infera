# Results index

All numbers first-hand from the AgentX result JSONs (`agentx_conc40.json`, field `request_metrics`), copied verbatim
from `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/`. The 1.5 GB `aiperf_artifacts/` per run were NOT copied (per-request
records, timeslices); they remain at the original path.

| dir | run | what |
|---|---|---|
| `e2g/A2-scored/` | **e2g A2 — headline 22,114/chip** | warm 2nd run after flush; `agentx_conc40.json`, `benchmark.log`, `benchmark_command.txt`, `runtime.env`, plots, aiperf summary |
| `e2g/A1-first-run/` | e2g A1 — 19,640 | fresh 1st run on the same stack (the warm-up) |
| `e2g/` | round files | `dryrun.e2g-102.yihou.txt` (rendered argv/env), `audit.txt` (AUDIT PASS), `census.txt`, `engine-*.log`, `smoke-response.json`, `spec_accept.after-bench.txt`, `status.tsv`, `workers.json`, `e2g.pipeline.log`, `e2g.A2.log`, `e2g.chain.log` |
| `e2i/R2-warm/`, `e2i/R1-first-run/` | e2i — 21,921 / 20,371 | replicate of the protocol with the decode debug-agent env (R052/R053) |
| `e2c-C40-warm/` | e2c C40 — 21,696 | first warm-protocol run (after the C44 run on the same stack; R030) |
| `e2j/R1-first-run/` (+ `R2-warm/` if present) | e2j — 19,335 / R2 | decode host-variant + cuda-graph bs 1..16 (R054) |
| `other-rounds/` | e1c, e2b4, e2c-C44, e2d, e2e, e2f | earlier rounds referenced in the ledger |
| `analysis/runs_summary.csv` | all runs | tok/s/chip, TTFT/TPOT/E2E, ok/err, stack state |
| `analysis/decode_stats.txt` | all runs | decode-log stats per profiling window: ms/step/DP, running-req, tokens, accept len, gen rate, transfer share, DP imbalance, prefill batches |
| `env/` | node captures 2026-10-10 | `collect_env.sh` output and image/in-container versions |
