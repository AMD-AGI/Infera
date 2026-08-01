# Results

Run of record: **`caseA_full/2026-08-01-13-34-46`** — ramp 400 s + sustain 3600 s,
2,988 requests sent, 0.988 success. Decode leg TAG `p6` (**patched**, see
`../patches/`); prefill TAG `p4`.

## Files

| file | what |
|---|---|
| `summary.json` | the driver's own aggregates + the 3-phase breakdown. **The `sustain` row is the result** — ramp is warm-up, drain is the tail. |
| `metadata.json` | run config as the driver recorded it. ⚠️ echoes unused argparse defaults (`think_time_mean`, `session_lifetime_mean`, `generation_length_mean: 1`) that are **not** this run's settings. |
| `metrics.jsonl.gz` | 1 Hz time series, 4.4 MB raw → 543 KB. Carries the **raw per-request arrays** that make the full percentile ladders recomputable. |
| `caseA_full.kvd_before.json` / `_after.json` | `statctl` snapshots bracketing the run |
| `raw/probe_*` | the 1,000 s probe run (same schema) — the Step-3 E2E measurement |

## The per-request arrays in `metrics.jsonl.gz`

Each 1 Hz row carries the requests that *completed* in that second, so concatenating
a field across rows gives the full per-request sample:

| field | n (sustain) | used for |
|---|---|---|
| `new_ttfts` | 2,702 | the TTFT ladder in `analysis/sli_percentiles.md` |
| `new_prompt_lengths` | 2,702 | input distribution vs the spec triple |
| `new_generation_lengths` | 2,702 | output distribution + the E2E back-solve |
| `new_acceptance_lengths` | 2,702 | **the MTP claim** (mean 2.04 vs config's 1.56) |
| `new_cache_hit_rates` / `new_ideal_cache_hit_rates` | 2,702 | cache efficiency |
| `new_inter_arrival_times` | 2,336 | duty cycle → the second E2E route |
| `in_flight`, `num_sessions_active` | per row | cap-binding checks, population trend |

**There is no `new_tpots`.** The driver keeps per-request TPOT in memory but
persists only p50/p90/p99 to `summary.json`. A full TPOT ladder needs a driver
change and a re-run.

Recompute anything:

```bash
zcat metrics.jsonl.gz | python3 -c "
import sys,json,math
def P(a,p):
    a=sorted(a); k=(len(a)-1)*p/100.0; lo,hi=math.floor(k),math.ceil(k)
    return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(k-lo)
t=[]
for line in sys.stdin:
    r=json.loads(line)
    if r.get('phase')=='sustain': t+=r.get('new_ttfts') or []
print('n=',len(t),[round(P(t,p)*1000) for p in [50,90,99]])"
# n= 2702 [4531, 9170, 16706]
```

## Headline numbers

| | sustain | whole run |
|---|---|---|
| requests | 2,702 | 2,988 sent / 2,952 completed |
| success | — | **0.988** (18 errors, all client 240 s timeouts) |
| qps (emergent) | 0.75 | 0.75 |
| TTFT p50 / p90 / p99 | 4,531 / 9,170 / 16,706 ms | 4,378 / 8,940 / 16,492 |
| TPOT p50 / p90 / p99 | 14.8 / 17.8 ms | 14.9 / 17.9 / 20.9 |
| cache actual / ideal | 89.0 % | **89.2 % / 89.0 %**, efficiency 100.3 %, eviction 0.0 % |
| uncached TPM | 425,522 (53,190/GPU) | — |
| peak prefill | — | 686,052 tok/s (85,756/GPU) |
| in-flight p50 / max | 17 / **30** | cap 48, **never bound** |
| live sessions p50 / p90 | 27 / 37 | target 32, cap 128, never bound |

## kvd delta

| counter | before | after | delta |
|---|---|---|---|
| entries | 29,750 | 47,732 | +17,982 |
| gets | 808 | 1,260 | **+452** |
| hits | 808 | 1,260 | **+452** |
| misses | 0 | 0 | **+0** |
| sets | 35,910 | 108,754 | +72,844 |
| evictions | 0 | 53,576 | +53,576 |
| host_bytes | 52.6 GB | 84.4 GB | +31.9 GB |

**452 gets / 452 hits / 0 misses.** Evictions began once the L3 tier reached its 64 G
cap — correct, deliberate behaviour (the cap keeps the tier off the node's root disk).

Note the 72,844-sets-to-452-gets asymmetry: every Case A request nests inside the
*same* shared prefix, so the GPU radix tree serves nearly everything and kvd is
written far more than it is read. This verifies kvd **correctness**, not cache
**tiering** — the guide says so explicitly.

## Known gaps

1. **No `new_tpots` field** — see above.
2. **The two crashed attempts have no `results/` of their own worth keeping.** Their
   driver logs ship in `../logs/caseA_full_attempt{1,2}_crashed.log.gz`; the
   partial `metrics.jsonl` they wrote is not included because the runs aborted
   mid-window and the phase accounting is meaningless.
3. **`generation_length.mean` prints "(target: 1)"** in `summary.json`. That
   "target" is a stale argparse default (`--new-tokens-mean`), not a setting of this
   run. The measured 1,085 is real.
