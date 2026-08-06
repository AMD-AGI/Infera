# Results

| file | what |
|---|---|
| `summary.json` | the driver's own summary — every number in `../analysis/sli_percentiles.md` comes from here |
| `metrics.jsonl.gz` | per-sample time series (1 s cadence) over the whole 4,007 s |
| `metadata.json` | the workload parameters the driver resolved at startup |
| `armA2_prefill.kvd_before.json` / `_after.json` | **the kvd delta** — 0 → 47,975 entries / 84.6 GB host / 297 GB long / 14,864 gets, all hits |
| `armA2_decode.kvd_before.json` / `_after.json` | all-zero both ends, **by design** (`_skip_kvd_on_decode_leg`) |
| `armA2_router.log.gz` | **the pick log** — 290–291 picks to each of 16 targets; the primary round-robin evidence |
| `armA2_router_metrics.txt` | the `/metrics` scrape. `infera_router_picks_total` is EMPTY here: that counter lives in the kv-aware path only |

Unlike the arm B kit, **nothing is missing** — the after-state, the router log and
the `collect_env.sh` snapshots were all pulled immediately after the run, before
the allocations were released. `../REPRODUCE.md` step 11 is why.
