# Results

## `fixlen_summary.csv`

One row per round, 8 rounds. Regenerate from `raw/` with:

    python3 scripts/extract_results.py

Columns beyond the obvious:

| column | meaning |
|---|---|
| `cache_hit_pct` | server-reported, needs `--enable-cache-report` |
| `cached_device_tok` | hits served from the **in-GPU radix cache** |
| `cached_host_tok` | hits served from kvd **L2** (host RAM) |
| `cached_storage_tok` | hits served from kvd **L3** |

**`cached_host_tok` ≈ 0 in every round is the load-bearing observation.** It is why the
nonzero `cache_hit_pct` values (up to 49.8 %) must be read as GPU-radix residue from the
*previous* round, not as kvd doing work. `--dataset-name random` has no shared prefix by
construction. See `notes.md` §11.

## `raw/`

Per round, four files:

| file | what |
|---|---|
| `fixlen_<pair>_c<C>.jsonl.gz` | the bench artifact — a **single JSON object** despite the `.jsonl` name |
| `fixlen_<pair>_c<C>.log` | full `bench_serving` stdout, including the printed summary table |
| `fixlen_<pair>_c<C>.kvd_before.json` | `statctl` snapshot immediately before the round |
| `fixlen_<pair>_c<C>.kvd_after.json` | `statctl` snapshot immediately after |

The `.jsonl.gz` files carry **both** the aggregates and the raw per-request arrays —
`ttfts`, `itls` (per-token latencies), `input_lens`, `output_lens`, `cached_tokens`,
`generated_texts`. That is what makes a full percentile ladder (p1 … p99.9) recomputable
later without re-running anything:

    python3 -c "
    import json,gzip,statistics as s
    d=json.load(gzip.open('results/raw/fixlen_p90_c128.jsonl.gz'))
    t=sorted(d['ttfts'])
    print('n=',len(t),'p50=',s.median(t))"

Gzipped because the set is 15 MB raw (7.2 MB for `p90_c128` alone, 93 % of it `itls`).

## Known gap: `fixlen_p90_c128.kvd_after.json`

That one file contains an error string, not counters:

    OCI runtime exec failed: write /tmp/runc-process995146436: no space left on device

It was captured at the exact moment the node's root disk hit 100 % — the kvd L3 disk trap
described in `notes.md` §5 and fixed in `patches/0003`. The round's **benchmark** result
is unaffected and complete (128/128 requests, in the `.jsonl.gz` and `.log`); only the
post-round kvd counter snapshot is missing.

Recorded rather than silently regenerated: the counters could not be re-read afterwards
because the fix involved clearing `/tmp/kvd-long`, which resets them.
