# results/

- `results.csv` — all rows of this kit plus the references, built by `scripts/session/results_csv_v15b.yihou.py highconc`. Rows marked "(not available at packing)" were in flight or aborted.
- Bench dirs `<round>-c<C>-<tag>/`:
  - Complete: ofg2 c48-A1, ofg4b c48-A1/A2 + c56-A3, ofg3b c48-A1/A2.
  - Partial snapshots: ofg2 c48-A2, ofg4 c48-A1 (aborted at the oai release), ofg2h3 c56-A1 (in flight), ofg2h3 c56-A1.fail1 (client failure).
- `rounds/<round>/` — status.tsv, chain/pipeline/cont/pyspy logs, audit, census, engine argv, stop logs. `rounds/ofs136.*.log` and `rounds/ofg{4,2}.seq.log` hold the drivers.
- `pyspy/ofg2h2/` — dump1/dump2 (watcher, top-level processes only); dump3.sched.091626.txt (the four schedulers, the evidence in notes §1). `pyspy/ofg2h3/` is empty because the warm-up ended in 61 s.
- `correctness/ofg4c/` — 22/23 (needle_128k_d50 sampled: finish=length, reasoning 4098 tokens vs max_tokens 4096).
- `dryruns/` — renders and `tokdiff.*` for every new round.
- `metrics/` — `/metrics` samplers (5 s).
