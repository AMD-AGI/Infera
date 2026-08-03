# Analysis

| file | what |
|---|---|
| `lat1_latency_floor.md` | **the report.** Ladders with CIs, the TTFT-vs-length curve and its fit, the bin-matched Case A comparison, the sweep cross-check, engine health. |

Companion machine-readable outputs live in `../results/`:

* `lat1_analysis.txt` — the analyzer's full stdout, verbatim
* `lat1_ladders.json` — every ladder as JSON
* `summary.json` — the driver's own computation (the authoritative TPOT source)
* `metrics.jsonl.gz` — 124 raw per-request samples; everything above is
  recomputable from this

## Which number to trust for what

| quantity | use | why |
|---|---|---|
| TTFT p50 / p90 | `lat1_latency_floor.md` ladders | ±11 % / ±12 % at n=120 |
| TTFT p99 | **the range 5.8–9.3 s**, not the point | ±25 %; it is the 3rd-largest of 120 samples |
| TPOT (any percentile) | `summary.json` `tpot_ms` | the driver persists no per-request TPOT array; nothing finer is recoverable |
| MTP acceptance | the **decode leg log** (2.846) | the driver's 2.2 averages SSE chunk sizes and undercounts |
| prefill scaling | the fit, not any single percentile | R² 0.956 over all 124 points beats a percentile of a mixed distribution |
