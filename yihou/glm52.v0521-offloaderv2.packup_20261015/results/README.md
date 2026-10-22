# results/

- `results.csv` — AgentX scores for pf90 (ref), ofs0 and the planned ofg4/ofg2 benches.
  Built by `scripts/tools/results_csv_v15.yihou.py offload`. Rows marked "(not available at packing)" were in flight.
- `ofs0-c48-A1`, `ofs0-c48-A2`, `pf90-c48-A1`, `pf90-c48-A2` — per-bench AgentX output (json, aiperf exports, runner/benchmark logs gzip, service info).
- `correctness/<round>/results.jsonl.gz`, `correctness.log.gz` — 23-test suite (`scripts/session/correctness_tests.yihou.py`): cx90 (no offload, ref), ofs0c, ofg3c, ofg2c.
  ofg4c was still running at packing.
- `rounds/<round>/` — `status.tsv`, chain/pipeline/seq logs, audit, census, engine argv logs:
  - ofs0c, ofs0, ofg3c, ofg2c complete.
  - ofg4c and ofg2 are snapshots.
  - ofg3 was aborted (node 266 released): `stop.release266.log`.
  - `ofg4c.hang1/` is a copy taken 10-15 ~02:57 while ofg4c's prefill PD warm-up had not finished (it finished at 02:42:09; the copy includes it).
- `dryruns/` — dryrun argv renders per round + `tokdiff.*` against the reference round.
- `metrics/ofs0.metrics.A{1,2}.tsv.gz` — `/metrics` sampler (5 s) for both legs.
- `h2d/h2d_bench.089.json` — H2D bandwidth gate (notes.md §3).
