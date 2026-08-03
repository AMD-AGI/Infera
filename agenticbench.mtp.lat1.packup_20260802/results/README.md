# Results

## The reported run — `lat1_full/2026-08-02-05-39-24`

| file | what |
|---|---|
| `summary.json` | the driver's own computation. **The authoritative TPOT source** — no finer TPOT exists |
| `metrics.jsonl.gz` | 1,981 per-second records carrying the 124 raw per-request samples (`new_ttfts`, `new_prompt_lengths`, `new_generation_lengths`, `new_cache_hit_rates`, `new_acceptance_lengths`). Everything in `analysis/` is recomputable from this |
| `metadata.json` | the config the driver actually ran with |
| `lat1_analysis.txt` | the analyzer's full stdout, verbatim |
| `lat1_ladders.json` | every ladder as JSON |
| `lat1_kvd_before.json` / `lat1_kvd_after.json` | kvd counters bracketing the run |

## The probe — `lat1_probe/2026-08-02-05-33-32`

| file | what |
|---|---|
| `probe_summary.json`, `probe_metrics.jsonl.gz` | the 6-min gate run that cleared cache, concurrency and artifacts before the real one |

## Evidence of the defect

| file | what |
|---|---|
| `CONTAMINATED_seed1337_metrics.jsonl.gz` | the **aborted** first full run. Its first 17 requests read cache hit 99.9–100.0 % with 14–62 uncached tokens, against a configured 0.89 — and its prompt-length sequence matches the probe's exactly. This is the raw evidence for `../notes/notes.lat1.md` § "Defect 1" |

Recreate the contamination comparison:

```bash
for f in CONTAMINATED_seed1337_metrics.jsonl.gz metrics.jsonl.gz; do
  echo "== $f"; zcat $f | python3 -c "
import sys,json
p=[];c=[]
for l in sys.stdin:
    r=json.loads(l); p+=r.get('new_prompt_lengths') or []; c+=r.get('new_cache_hit_rates') or []
for i in range(min(6,len(p))): print(f'  {p[i]:>9,}  cache={c[i]*100:.1f}%  uncached={p[i]*(1-c[i]):>8,.0f}')"
done
```

## kvd, read correctly

`gets_total` **did not move** (15,210 → 15,210), `misses` stayed 0, `sets` rose
+3,792. That is not kvd failing: prefill kvd only fetches on a radix-tree miss,
and at 89 % planned hit with one session the in-GPU tree serves everything. The
flat `gets` is also what proved the cache contamination was *not* kvd — see the
notes.
