# Logs

Two measured arms, each with its own driver transcript and leg tails.

| file | arm | what |
|---|---|---|
| `chunk65536_full.driver.log.gz` | **MAIN** | bench driver console for the reported run — the per-second `Sessions:/In-flight:` line is the concurrency-1 evidence in raw form |
| `chunk65536_probe.driver.log.gz` | MAIN | the 6-min probe that gated it |
| `chunk65536_prefill_tail6000.log.gz` | MAIN | last 6,000 lines of the prefill leg (DPA **off**, global chunk **65,536**) |
| `chunk65536_decode_tail6000.log.gz` | MAIN | last 6,000 lines of the decode leg — source of `accept len` |
| `chunk8192_full.driver.log.gz` | control | the chunk-control run (global chunk 8,192) |
| `chunk8192_probe.driver.log.gz` | control | its probe |
| `chunk8192_prefill_tail6000.log.gz` | control | prefill leg at global chunk 8,192 |
| `chunk8192_decode_tail6000.log.gz` | control | decode leg (identical config to MAIN — decode never changed) |

## Telling the two arms apart from the logs alone

```bash
for a in 65536 8192; do
  echo -n "chunk$a prefill: max #new-token = "
  zcat chunk${a}_prefill_tail6000.log.gz | grep -oE '#new-token: [0-9]+' \
    | awk '{print $2}' | sort -n | tail -1
done
```

    chunk65536 prefill: max #new-token = 65536
    chunk8192  prefill: max #new-token = 8192

**This is the check that needs no flag reasoning at all.** `#new-token` is the
token count actually processed in one prefill step. The control arm never exceeds
8,192 — its ceiling *is* 8,192 — while the MAIN arm reaches 65,536, which is
impossible under an 8,192 budget. Two arms, an 8× difference in per-step budget,
demonstrated from the engine's own counters.

(Only the MAIN prefill tail happens to contain a `server_args=` line; the control
arm's boot predates its tail window. The authoritative flag dumps for **both**
arms are in `../env/chunk{65536,8192}_prefill_server_args.txt`.)

## Seed provenance lives here, not in `metadata.json`

The driver's `metadata.json` records **neither `random_seed` nor the `profile:`
block** — only the flat CLI-shaped knobs. For this family of experiments that is a
real gap, since a shared seed silently contaminated an earlier lat1 attempt.

The driver **log** does record it:

```bash
for f in *.driver.log.gz; do
  echo "$f: $(zcat $f | grep -oE 'Random seed set to: [0-9]+')"
done
# chunk65536_full.driver.log.gz:  Random seed set to: 2026080211
# chunk65536_probe.driver.log.gz: Random seed set to: 2026080212
# chunk8192_full.driver.log.gz:   Random seed set to: 2026080201
# chunk8192_probe.driver.log.gz:  Random seed set to: 2026080202
```

Four distinct seeds, none shared with lat1's three (`1337`, `20260802`,
`2026080299`). Each log also names the workload YAML it loaded
(`Loaded workload config from: …/nodpa_full.yaml`), binding
`../results/chunk65536_MAIN/summary.json` to `../spec/nodpa_full.yaml`.

**Caveat on the shipped `spec/` yamls:** they carry the MAIN arm's seeds
(2026080211 / 2026080212). The control arm's yamls differed *only* in
`random_seed` and are not separately shipped — the two values are recorded above
and in `../notes/notes.nodpa.md` §4.

## The leg logs are tails, and one boot log is missing

**Tails, deliberately.** The full prefill log is ~3 MB and decode ~1.5 MB; the
tails cover the measured window, which is where the numbers come from.

**Missing: the `GMU=0.80` OOM crash.** The leg script redirects with `>`, so
rebooting at 0.70 truncated the file. The crash is quoted verbatim in
`../notes/notes.nodpa.md` §5 and listed as a gap in `../environment.md`. It
reproduces by booting this arm at 0.80.

**Also note the leg logs are cumulative across both arms** — the legs were never
restarted between them except for the prefill chunk reboot. The `chunk8192_*`
tails were captured before that reboot, the `chunk65536_*` tails after.

## Grep them through `strings`

The originals on the nodes contain binary bytes — plain `grep` reports
`Binary file matches` and counts nothing. These `.gz` files were produced from an
already-`strings`-filtered stream, so they are plain text; the caveat applies to
the originals.
