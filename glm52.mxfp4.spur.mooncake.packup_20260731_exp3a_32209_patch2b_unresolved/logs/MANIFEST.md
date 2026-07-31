# Log manifest

Which round produced which log, and which probes were live in it. Verified by
counting markers in the shipped (gzipped) files, not from memory.

All logs are gzipped; they contain binary bytes, so read them with
`zcat X.gz | strings | grep -a ...` — a plain `grep -c` returns 0 and reads as
"clean".

## e3c (#32209 trim + upstream slice hunk, our patch 4)

| file | round | `GLM52_DSTEP` | `GLM52_MULTI` | `GLM52_HSO` | `GLM52_PAD` |
|---|---|---|---|---|---|
| `decode.round1.log.gz` | 4 | 2252 | 0 | 0 | 0 |
| `decode.round2.log.gz` | 5 | 1843 | 1799 | 0 | 0 |
| `decode.round3.log.gz` | 6 | 2260 | 1800 | 908 | 0 |
| `decode.log.gz` | **7 (final)** | 1874 | 1902 | 202 | 632 |

`decode.log.gz` is the run all four probe families were live in, and is the one
the README and REPRODUCE quote. The `seq=18` record appears **only** there:

```
GLM52_MULTI gather seq=18 rank=0 local_rows=6 global_rows=32
  plan=[4,4,4,4,4,4,4,4] orig=[2,1,3,2,2,2,2,4] bs=4 fwd=2 capture=1 ndt=1
  spec_cls=EagleDraftInput hs_id=358144 inp_rows=4
```

`decode.round3.log.gz` is the second run used in the pooled analysis (the one
that refuted the `local == plan + 2` rule — see `notes.md` §4.5).

## e3a (#32209 trim only, our patch 4)

| file | round | note |
|---|---|---|
| `decode.round1.log.gz` | 1 | `instr_e3` read a nonexistent field; logged `None` throughout |
| `decode.round2.log.gz` | 2 | probe field names fixed; trim-fire counter added |
| `decode.log.gz` | 3 | rebuilt image, third node pair, `(rank, step)` probe |

`prefill*.log.gz` and `router.log.gz` are included for completeness; the crash
is entirely on the decode leg.

## Results ↔ logs

| jsonl | arm | round |
|---|---|---|
| `stress_c32.jsonl` | e3a | 1 |
| `stress_c32_r2.jsonl` | e3a | 2 |
| `stress_c32_r3.jsonl` | e3a | 3 |
| `e3c_stress_c32.jsonl` | e3c | 4 |
| `e3c_stress_c32_multi.jsonl` | e3c | 5 |
| `e3c_stress_c32_hso.jsonl` | e3c | 6 |
| `e3c_stress_c32_pad.jsonl` | e3c | 7 |

All seven are **0/32**.
