# Results — raw evidence

Every number in `../analysis/` is recomputed from these files. Nothing here is
hand-edited.

| file | what | size |
|---|---|---|
| `metrics.jsonl.gz` | **the primary artifact** — 3,992 one-second rows, each carrying that window's new per-request samples | 596 K |
| `metadata.json` | driver invocation, resolved workload, start time | 740 B |
| `summary.txt` | the driver's own end-of-run summary, verbatim | 2 K |
| `router.log.gz` | Rust router policy log — 5,794 lines, 5,768 scoped pick decisions | 98 K |
| `par8_full.kvd_before.json` | decode kvd counters, pre-run (all zero) | |
| `par8_full.kvd_before_prefill.json` | prefill kvd counters, pre-run (all zero) | |
| `par8_full.kvd_after_prefill.json` | prefill kvd counters, post-run | |
| `par8_full.kvd_after_decode.json` | decode kvd counters, post-run (all zero, by design) | |

`../env/env_chi2835.txt` and `../env/env_chi2879.txt` hold the node snapshots,
captured 16 min after the run with both legs still live — so the recorded
command lines are the ones that served it.

## metrics.jsonl schema

One JSON object per ~1 s window. Cumulative counters plus `new_*` arrays holding
the samples that *completed* in that window:

| field | note |
|---|---|
| `phase` | `ramp` (400 rows) / `sustain` (3,600) / `drain` (5) — **analysis uses sustain** |
| `new_ttfts` | seconds |
| `new_e2es`, `new_tpots` | seconds — **present only because of the `SOLO_M1` driver patch** |
| `new_generation_lengths` | client-side **re-tokenisation** of the accumulated stream, not `usage.completion_tokens` |
| `new_prompt_lengths` | from `usage.prompt_tokens` (engine-reported) |
| `new_cache_hit_rates` / `new_ideal_*` | per-request |
| `new_acceptance_lengths` | MTP accept length, per request |
| `in_flight`, `num_sessions_active` | instantaneous |

> Case A's `metrics.jsonl` has `new_ttfts` but **zero** `new_e2es`/`new_tpots` —
> it predates `SOLO_M1`. Cross-run E2E/TPOT comparison is impossible from raw
> samples; see `../environment.md`.

Recompute any ladder:

```bash
python3 -c "
import json,gzip
rows=[json.loads(l) for l in gzip.open('metrics.jsonl.gz','rt') if l.strip()]
v=[x for r in rows if r.get('phase')=='sustain' for x in (r.get('new_ttfts') or [])]
s=sorted(v); print({f'p{int(f*100)}': round(s[int(len(s)*f)]*1000) for f in (.5,.9,.99)})
"
```

## router.log.gz — read it scoped

```bash
zcat router.log.gz > /tmp/router.log
python3 ../scripts/cv_scoped.py /tmp/router.log 2026-08-03T09:28
```

**The timestamp argument is mandatory.** The log is appended across leg restarts
and an unscoped read mixes in picks to `10.2.122.3` (the previous decode leg on
a different node) — it reports **9** distinct targets for an 8-rank leg. The log
is also ANSI-colourised, so a bare `grep -c "picked="` returns 0; both
`cv_scoped.py` and `cache_view.py` strip escapes first.

## kvd counters

| counter | prefill before → after | decode |
|---|---|---|
| entries | 0 → 47,677 | 0 → 0 |
| **gets** | 0 → **0** | 0 → 0 |
| hits / misses | 0 / 0 | 0 / 0 |
| sets | 0 → **57,870** | 0 → 0 |
| evictions | 0 → 10,193 | 0 → 0 |
| host_bytes | 0 → 84.4 GB | 0 |
| long_bytes | 0 → 68.7 GB | 0 |

Both daemons started **cold** — a clean baseline, unlike the solo kits which
inherited a warm tier.

**`gets = 0` is the notable number**: kvd wrote 57,870 entries and read back
none (Case A: +452 gets / +452 hits). Decode is all-zero **by design** — the
image deliberately skips kvd wiring on a PD decode leg. **Why prefill never read
back is not established**; see `../environment.md`.

## Logs (`../logs/`, gzipped)

| file | what |
|---|---|
| `par8_full.log.gz` | driver stdout — the per-second dashboard and the final summary |
| `q1_prefill.log.gz` | prefill engine (chi2835), `strings`-filtered |
| `q4_decode.log.gz` | decode engine (chi2879), `strings`-filtered |

Engine logs are `strings`-filtered because they contain binary bytes that make
bare `grep` go blind. They are also **appended across runs** — scope by
timestamp (`^\[2026-08-03 (09|10):`) before counting anything, and read matches
before trusting a count: `grep -ic error` on the decode log returns **1,492**
hits, all from one `server_args=` dump containing
`abort_on_priority_when_disabled`. Excluding `server_args=|aiter|fused_moe`
leaves **0** real faults on both legs.
