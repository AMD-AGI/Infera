# Logs

| file | what | size |
|---|---|---|
| `lat1_full.driver.log.gz` | the bench driver's full console transcript for the reported run (per-second `Sessions:/In-flight:` line — the concurrency-1 evidence in raw form) | 13 KB |
| `lat1_probe.driver.log.gz` | the 6-min probe that gated the real run | 3 KB |
| `CONTAMINATED_seed1337.driver.log.gz` | the **aborted** first attempt — evidence for the cache-contamination defect | 3 KB |
| `lat1_prefill_tail6000.log.gz` | last 6,000 lines of the prefill leg, covering the lat1 window | 26 KB |
| `lat1_decode_tail6000.log.gz` | last 6,000 lines of the decode leg — this is where `accept len` comes from | 102 KB |

## These logs carry the seed provenance, and `metadata.json` does not

The driver's `metadata.json` records **neither `random_seed` nor the `profile:`
block** — it only persists the flat CLI-shaped knobs (verified: the file has 25
keys and none is a seed). For this experiment that is a real gap, because the
seed is *the* critical knob: a shared seed is what contaminated the first attempt.

The driver **log** does record it, so the chain is intact — just not where you
would look first:

```bash
for f in *.driver.log.gz; do
  echo "$f: $(zcat $f | grep -oE 'Random seed set to: [0-9]+')"
done
# CONTAMINATED_seed1337.driver.log.gz: Random seed set to: 1337
# lat1_full.driver.log.gz:             Random seed set to: 20260802
# lat1_probe.driver.log.gz:            Random seed set to: 2026080299
```

Each log also names the exact workload YAML it loaded
(`Loaded workload config from: …/lat1_full.yaml`), which is what binds
`../results/summary.json` to `../spec/lat1_full.yaml`.

**The leg logs are tails, not full logs, and that is deliberate.** Both legs have
been appending to the *same* files since 2026-08-01 08:36 (they were never
restarted between Case A and lat1), so the full prefill log is 14 MB and the
decode log 6 MB — and both are already carried, for the Case A window, in
`../../agenticbench.mtp.caseA.packup_20260801/logs/`. Duplicating 20 MB to
capture a 33-minute window would be waste. The tails cover the lat1 window.

Logs contain binary bytes. **Grep them through `strings`** — plain `grep` reports
`Binary file matches` and counts nothing:

```bash
zcat lat1_decode_tail6000.log.gz | grep -oE 'accept len: [0-9.]+' | ...
```

(The `.gz` here was produced from an already-`strings`-filtered stream, so it is
plain text; the caveat applies to the originals on the nodes.)
