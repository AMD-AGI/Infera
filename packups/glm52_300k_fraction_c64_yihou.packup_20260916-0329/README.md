# What does raising the share of 300K-token requests do to a C=64 decode batch?

**2026-09-15/16.** GLM-5.2-MXFP4, TP8 / EP1 / DP-attention, C=64, OSL 10000, 8x MI355X.
Six points sweeping the 300K share of the batch from 0 % to 100 %.

## Answer

TPOT rises monotonically with the 300K share, **+6.61 % from 0 % to 100 %**.

| 300K share | run order | TPOT ms | vs 0 % | step | accept gate | iters | peak GB |
|---|---|---|---|---|---|---|---|
| 0 % | 2nd | 11.9839 | — | — | exact | 2768 | 236.0 |
| 12.5 % | 6th | 12.1261 | +1.19 % | +0.1423 | exact | 2768 | 236.0 |
| 25 % | 4th | 12.1853 | +1.68 % | +0.0592 | exact | 2768 | 236.0 |
| 50 % | 1st | 12.3450 | +3.01 % | +0.1597 | exact | 2768 | 236.0 |
| 75 % | 5th | 12.5503 | +4.73 % | +0.2053 | exact | 2768 | 236.0 |
| 100 % | 3rd | 12.7757 | +6.61 % | +0.2254 | exact | 2768 | 236.0 |

"accept gate: exact" = `realized_accept_length == 3.6134393063583814` to the last bit and
`verify_iterations == 2768`, on every point. All six `complete = True`.

## Why the ordering is the claim, and the magnitudes are not

**The series was executed shuffled — 50, 0, 100, 25, 75, 12.5 — and came out monotone in the 300K
share.** Six points landing in exactly the right order by chance is 1/720 ≈ 0.14 %. Had the series
been run 0 → 100 in order, thermal or clock drift would have been perfectly collinear with the
effect and no later analysis could have separated the two. That is the whole reason for shuffling,
and it is why the *ordering* is what this packup claims.

**The per-point numbers are single-shot.** No repeats, so no interval can be attached to any TPOT,
and the step column is not a shape — the 12.5 → 25 step is less than half the 0 → 12.5 step, which
no smooth story explains. **Do not quote a slope from this table.**

**And there is a 4.75 % elephant.** An earlier all-70K run at byte-identical configuration
(`results/points/c64_all70k_control_yihou/`, 12:00 UTC) gave **12.5537 ms** where this series' 0 %
point gave 11.9839 ms. Config was compared field by field and is the same. That gap is 72 % of the
entire 0 → 100 % effect, it is **unexplained**, and until the run-to-run spread at fixed
configuration is measured, no single pair of rows above can be defended on its own.

## Quality controls that did run

- **Tenancy verified before and after every run.** All twelve `rocm-smi --showpids` captures read
  `No KFD PIDs currently running` (`results/tenancy/`). The node is shared and containers on it
  ignore Slurm, so this is not a formality — a colleague's container appeared mid-session earlier
  the same day.
- **Arguments asserted after every run.** `ARG CHECK: ok` on all six, read back from
  `config_yihou.json` and `server_cli`. See `notes.md` for the bug that made this necessary.
- **Kernel identity asserted from the logs.** `FlyDSL sparse MLA decode declined` on all 8 ranks,
  `Loading tilelang libs` on all 8 — **TileLang executed**, same as the published baseline
  (`evidence/key_log_lines_yihou.md`).
- **An internal control for the most obvious confound.** The 0 % point reserves 640512 KV tokens
  against 2480128 for every point containing a 300K request (allocation is uniform-max), so it is
  structurally different from the rest. The five points that share an identical reservation are
  *still* monotone, 12.1261 → 12.7757 (+5.44 %), also under shuffled order. The trend is not an
  artifact of the 0 % point.

## Claims from earlier in the session that this supersedes

| earlier claim | status |
|---|---|
| "a 300K request makes the batch 3.62 % *faster*" | **wrong** — it compared against the 12.5537 outlier |
| "the earlier control was contaminated by a colleague's load" | **unsupported** — their container started 15 s after that run finished |
| "the 4.3 % gap shows contention" | **wrong** — the other side of that comparison was missing four optimisation flags |

## What is still open

**Why.** Attention is the obvious suspect and has not been profiled. One line appears in every
runtime log and is recorded here without interpretation:

```
[dense-decode] DSA dual-graph enabled: capturing dense (k-only) + sparse (full indexer)
decode graphs; dispatch on max_kv_len vs index_topk=2048.
```

Both 70000 and 300000 are far above `index_topk = 2048`, so both should take the sparse path and
this dispatch is probably not the mechanism — but "probably" is a guess, and profiling (graph on)
was deliberately not started.

**The noise floor.** Repeating the 0 % point three or four times in one contiguous window would
settle whether the per-point magnitudes mean anything. Not done.

## Layout

| file | what |
|---|---|
| `results/300k_fraction_series_yihou.md` | the full analysis, longer than this README |
| `results/points/` | `result_yihou.json` + config + command for all 8 runs referenced |
| `results/tenancy/` | the 12 before/after `--showpids` captures |
| `results/flagbug_evidence/` | configs of the three discarded runs, showing which flags were lost |
| `evidence/key_log_lines_yihou.md` | extracted log lines; full logs stay on disk, path inside |
| `scripts/run_300k_fraction_series_yihou.sh` | the driver, including its ARG CHECK |
| `REPRODUCE.md` | ordered steps |
| `notes.md` | the flag bug and three other traps, with the reasoning |

## Code

The `--input-len-spec` feature this experiment depends on is committed:

```
bce29162 bench: support heterogeneous input lengths in the decode harness
41020f5f packups: heterogeneous-ISL batch support, with its verification runs
```

and packed separately in `packups/glm52_isl_heterogeneous_batch_yihou.packup_20260915-1215/`.
This packup contains no source changes — only the experiment that used them.
