# Results — agentic at concurrency 1, three Case-A request shapes

All latency figures are **sustain phase only**. Ramp is a warm-up exclusion
window (see `../notes.md` §3), so its requests — which pay the cold-prefix TTFT —
are excluded by construction.

Produced by [`../scripts/analyze_solo.py`](../scripts/analyze_solo.py) over the
`solo/<arm>/metrics.jsonl` files in this directory. **Every number below was
re-derived from those files while assembling this packup**, with no cluster
access; the reproduction command is in [`../REPRODUCE.md`](../REPRODUCE.md) §6.

Machine-readable: [`summary.csv`](summary.csv), one row per arm, 33 columns.

## Latency

| arm | n | prompt p50 | gen p50 | gen mean | cache hit (mean) | TTFT p50 | TTFT p90 | TTFT p99 | TTFT mean | E2E p50 | E2E p90 | E2E p99 | E2E mean | TPOT p50 | TPOT p90 | TPOT p99 | TPOT mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p50 | 106 | 74,013 | 317 | 317.5 | 0.8898 | **1811.7** | 2815.9 | 3413.3 | 2022.4 | **5111.1** | 5910.2 | 6798.8 | 5170.0 | **9.96** | 10.71 | 11.49 | 9.95 |
| p90 | 46 | 155,013 | 3,297 | 3,296.1 | 0.8897 | **3674.7** | 4730.9 | 5349.1 | 3902.4 | **30781.4** | 33478.7 | 39312.8 | 31389.3 | **8.13** | 9.40 | 10.84 | 8.34 |
| p99 | 28 | 235,013 | 16,993 | 16,980.6 | 0.8900 | **5663.6** | 5854.7 | 7486.4 | 5746.7 | **147404.6** | 155157.9 | 166228.8 | 146988.7 | **8.34** | 8.87 | 9.45 | 8.33 |

All times in ms. TPOT is per *output* token, excluding the first.

**No TPOT sample was filtered on any arm** — `analyze_solo.py` reports `n=106 /
46 / 28` for TPOT, equal to the request count, so its `NOTE:` line about dropped
0.0 markers never printed. The SOLO_M1 `0.0` convention was in force but never
had to be exercised here.

### These figures differ slightly from the driver's own `summary.json` — by percentile convention only

The table above is `analyze_solo.py`'s. Each `summary.json` also reports a
sustain-phase TTFT/TPOT percentile block, and the two disagree by up to 2.5 %:

| arm | metric | driver `summary.json` | `analyze_solo.py` |
|---|---|---|---|
| p50 | TTFT p50 | 1812.6 | 1811.7 |
| p50 | TTFT p90 | 2857.6 | 2815.9 |
| p90 | TTFT p90 | 4850.0 | 4730.9 |
| p99 | TTFT p90 | 6039.4 | 5854.7 |

**This is an index convention, not a data disagreement — established by test, not
assumed.** Both read the same array. The driver picks `sorted[int(q*n)]` (the
"higher" / ceiling convention); `analyze_solo.py` picks
`sorted[round(q*(n-1))]` (nearest-rank on `n-1`). Checked against every standard
convention on all three arms: the driver's value matches `numpy`'s
`method='higher'` exactly on every point tested, and never matches `linear`.
At n = 28–106 the two conventions can straddle a whole sample, which is the
entire size of the gap. p99 and p50 values coincide wherever the two indices land
on the same element.

Neither is more correct. The table above uses `analyze_solo.py` throughout so
that TTFT, E2E and TPOT are computed identically — the driver's block has no E2E
column at all, so mixing the two sources would compare percentiles taken by
different rules. If you cite `summary.json` figures instead, cite them for all
three metrics or say which convention you used.

## Run-level totals

| arm | duration | sent | completed | errors | success rate | qps |
|---|---|---|---|---|---|---|
| p50 | 785.7 s | 133 | 133 | **0** | 1.0000 | 0.1693 |
| p90 | 1685.6 s | 52 | 52 | **0** | 1.0000 | 0.0308 |
| p99 | 4385.9 s | 30 | 29 | **0** | 0.9667* | 0.0068 |

\* Clock truncation, not a failure — the 30th request was sent at elapsed
4302.7 s and was still in flight when the 4380 s budget expired. `errors` is 0 on
every tick. Full trace in [`../REPRODUCE.md`](../REPRODUCE.md) §7 and
[`../notes.md`](../notes.md) §7. It never entered the sustain arrays (n=28).

## Cache — the controlled variable

The workloads request `cache_hit_rate: 0.89`. The driver computes an *ideal* hit
rate for the prompts it built, and an *actual* rate from the server's own
`cached_tokens` (`--enable-cache-report` is on).

| arm | ideal | actual (whole run) | **sustain-phase** | efficiency | eviction rate |
|---|---|---|---|---|---|
| p50 | 0.8898 | 0.8831 | **0.8898** | 0.9924 | 0.0076 |
| p90 | 0.8899 | 0.8808 | **0.8897** | 0.9897 | 0.0103 |
| p99 | 0.8900 | 0.8795 | **0.8795→0.8900** | 0.9883 | 0.0117 |

The sustain-phase figure is the one that matters and it lands on target on all
three arms — 0.8898 / 0.8897 / 0.8900 against an ideal of ~0.8899. The whole-run
`actual` sits slightly lower only because it includes the ramp, where the prefix
was still being built.

## Token totals

| arm | input | cached | uncached | generated | reasoning | visible |
|---|---|---|---|---|---|---|
| p50 | 9,843,728 | 8,692,992 | 1,150,736 | 42,239 | 23,817 | 18,422 |
| p90 | 8,060,675 | 7,099,776 | 960,899 | 171,386 | 31,439 | 139,947 |
| p99 | 6,815,377 | 5,994,176 | 821,201 | 492,453 | 38,609 | 453,844 |

Reasoning tokens are billed against the same budget as content
(`--reasoning-parser glm45`). On the p50 arm reasoning is the *majority* of
generation (23,817 of 42,239).

## Ramp vs sustain — the evidence the warm-up worked

| arm | ramp completed | ramp cache hit | ramp TTFT p50 | ramp TTFT p99 | sustain cache hit | sustain TTFT p50 | sustain TTFT p99 |
|---|---|---|---|---|---|---|---|
| p50 | 26 | 0.8556 | 2069.8 | **20875.8** | 0.8898 | 1812.6 | 3413.3 |
| p90 | 5 | 0.7968 | 4113.4 | **16015.1** | 0.8897 | 3679.1 | 5349.1 |
| p99 | 1 | 0.5869 | 18462.2 | **18462.2** | 0.8900 | 5663.6 | 7486.4 |

The first request of each arm pays a 16–21 s cold TTFT. None of it reaches the
reported table.

## Against the stated SLA line

`sla.e2e_p50_ms: 4500` in the workload files is **documentation only** — the
driver parses `args.sla_cfg` and never consumes it. Recorded here as a
measurement against a stated bar:

| arm | E2E p50 | stated bar | difference |
|---|---|---|---|
| p50 | 5111.1 ms | 4500 ms | **+611.1 ms over** |
| p90 | 30781.4 ms | (no bar stated for this shape) | — |
| p99 | 147404.6 ms | (no bar stated for this shape) | — |

**No explanation is offered for the p50 gap.** See [`../notes.md`](../notes.md)
§6 for what would settle it.

## Two observations recorded without explanation

1. **TPOT p50 falls as the arms get longer**: 9.96 → 8.13 → 8.34 ms, while the
   prompt grows 74K → 155K → 235K and generation grows 317 → 3,297 → 16,993
   tokens. The whole distribution shifts (TPOT p99: 11.49 → 10.84 → 9.45).
   [`../notes.md`](../notes.md) §5 names the measurements that would settle it —
   the cheapest is offline and uses only the files in this directory.
2. **The p50 arm's E2E misses the stated 4500 ms bar** (above).
   [`../notes.md`](../notes.md) §6.

Neither is explained here on purpose. Nothing was measured that identifies a
mechanism for either.

## What is in `solo/<arm>/`

Each of `p50/`, `p90/`, `p99/` holds the driver's output verbatim, untrimmed:

| file | what |
|---|---|
| `metadata.json` | the run's resolved knobs + the timestamp that named the original directory |
| `summary.json` | run totals, the three phase blocks (ramp / sustain / drain), cache accounting |
| `metrics.jsonl` | **one JSON object per ~1 s tick**, 783 / 1,681 / 4,375 rows |

`metrics.jsonl` is the reproducible core. Per-tick arrays it carries, all
index-aligned with each other:

```
new_ttfts                 new_e2es (SOLO_M1)       new_tpots (SOLO_M1)
new_prompt_lengths        new_generation_lengths   new_cache_hit_rates
new_ideal_cache_hit_rates new_acceptance_lengths   new_acceptance_rates
new_inter_arrival_times   new_session_times        new_planned_prompt_lengths
```

plus per-tick scalars: `phase`, `elapsed_seconds`, `in_flight`, `errors`,
`requests_sent`, `requests_completed`, `num_sessions_active`, `cache_hit_rate`,
`generation_tps`, `prefill_tps`.

**Nothing was trimmed.** The original timestamped directory names
(`2026-08-06-09-53-41`, `-10-07-44`, `-10-38-57`) are preserved in each
`metadata.json`; the directories themselves were flattened to `p50/p90/p99` so
the paths in `REPRODUCE.md` are stable. Originals remain on the cluster at
`/mnt/vast/c_huggingface/glm52_mix_20260806/results/agentic_solo_<arm>/`.

Total size 6.5 MB, largest single file 4.1 MB (`p99/metrics.jsonl`).
