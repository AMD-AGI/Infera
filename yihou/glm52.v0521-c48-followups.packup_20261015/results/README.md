# results/

- `results.csv` — pf90 A1/A2 (ref), tw90 A1/A2, wts48 A2 (ref), wts64. Built by `scripts/tools/results_csv_v15.yihou.py followups`.
- `tw90-c48-A1`, `tw90-c48-A2`, `wts48-c64-A2` (= wts64), `pf90-c48-A1`, `pf90-c48-A2` — AgentX outputs (json, aiperf exports, runner/benchmark logs gzip, service info).
- `correctness/cx90/` — `results.jsonl.gz` (one record per test: test, sampling, expect, pass, content, reasoning) and `correctness.log.gz`.
- `rounds/{tw90,cx90,wts48}/` — status.tsv, chain/pipeline/cont logs, audit, census, engine argv logs.
- `dryruns/` — dryrun renders and `tokdiff.pf90-vs-{tw90,cx90}.txt`.
- `metrics/` — `/metrics` sampler TSVs (tw90 A1/A2; the wts48 sampler file mixes the wts48 A2 and wts64 samples, split at 07:17:10 UTC).
