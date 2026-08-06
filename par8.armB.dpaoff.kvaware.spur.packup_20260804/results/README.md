# Results

| file | what |
|---|---|
| `summary.json` | the driver's own summary — every number in `../analysis/sli_percentiles.md` comes from here |
| `metrics.jsonl.gz` | per-sample time series (1 s cadence) over the whole 4,007 s |
| `metadata.json` | the workload parameters the driver resolved at startup |
| `armB_prefill.kvd_before.json` | kvd counters on the prefill node, immediately pre-run (all zero) |
| `armB_decode.kvd_before.json` | same, decode node |

## Missing, and why

| absent | cause |
|---|---|
| `*.kvd_after.json` | both allocations were reclaimed at the 24 h wall clock (SIGTERM) ~11 h after the run, before `statctl` could be re-read |
| `router.log.gz` / `router_metrics.txt` | both live **inside** the prefill container and died with it |
| `env/env_<node>.txt` | `collect_env.sh` was never run against the live nodes |

`../REPRODUCE.md` step 11 exists specifically so a rerun does not repeat this.
