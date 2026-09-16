# 300K-fraction series — C=64, EP1, dpa, OSL 10000

2026-09-15, image `sha256:bdd7835…` (sglang `dev_glm52_0907` + PR #51 + #50), container
`yihou-glm52-tp8ep1-pr50-51`, node `smci355-ccs-aus-n10-29`.

Six points. `local_batch_size = 8` at C=64 / dp=8, so the 300K share lands on whole requests.
Every point is a full run to completion (`--output-len 10000`, no `--max-steps`).

## Result

| 300K share | run order | TPOT ms | vs 0 % | step | accept gate | iters | peak GB |
|---|---|---|---|---|---|---|---|
| 0 % | 2nd | 11.9839 | — | — | exact | 2768 | 236.0 |
| 12.5 % | 6th | 12.1261 | +1.19 % | +0.1423 | exact | 2768 | 236.0 |
| 25 % | 4th | 12.1853 | +1.68 % | +0.0592 | exact | 2768 | 236.0 |
| 50 % | 1st | 12.3450 | +3.01 % | +0.1597 | exact | 2768 | 236.0 |
| 75 % | 5th | 12.5503 | +4.73 % | +0.2053 | exact | 2768 | 236.0 |
| 100 % | 3rd | 12.7757 | +6.61 % | +0.2254 | exact | 2768 | 236.0 |

"accept gate: exact" means `realized_accept_length == 3.6134393063583814` to the last bit, and
`verify_iterations == 2768`, on every point. All six `complete = True`.

## Why the ordering is the important part

**The series was executed shuffled — 50, 0, 100, 25, 75, 12.5 — and came out monotone in the 300K
fraction.** That was the reason for shuffling. Had it been run 0 → 100 in order, a thermal or clock
drift trend would have been perfectly collinear with the ISL trend and nothing afterwards could
have separated them. Under a shuffled schedule, six points landing in exactly the correct order has
probability 1/720 ≈ 0.14 % if TPOT were independent of the fraction.

**Tenancy was verified before and after every single run.** All twelve `rocm-smi --showpids`
captures read `No KFD PIDs currently running` (`logs/smi_s*_yihou.txt`). No point in this series
was measured while anyone else held a GPU.

**Every point passed an argument assertion.** `ARG CHECK: ok` on all six, read back from
`config_yihou.json` and `server_cli` — see "the flag bug" below for why that check exists.

## An internal control that removes the most obvious confound

The 0 % point reserves less KV than the others: its `max(ISL)` is 70000, so `reserved_tokens` is
640512 against 2480128 for every point that contains at least one 300K request (allocation is
uniform-max). That makes the 0 % point structurally different from the rest.

**It does not matter to the trend.** The five points that share an identical reservation of
2480128 tokens — 12.5 %, 25 %, 50 %, 75 %, 100 % — are *still* monotone, spanning
12.1261 → 12.7757 (+5.44 %), and their run order was also shuffled (6th, 4th, 1st, 5th, 3rd).
Five in correct order by chance is 1/120.

Peak allocated memory is 236.0 GB at every point, including 0 %, so the torch-side footprint is
set by `--mem-fraction-static 0.85` and not by the workload.

## What is NOT established

**The per-point magnitudes are single-shot.** There are no repeats, so no confidence interval can
be attached to any individual TPOT, and the step sizes should not be read as a shape. The steps per
12.5 % of share are +0.142, +0.059, +0.080, +0.080, +0.103, +0.103, +0.113, +0.113 — suggestive of
a slope that grows with the fraction, but the 12.5 → 25 step is less than half the 0 → 12.5 step,
which no smooth story explains. **Do not quote a slope.**

**An earlier all-70K run at byte-identical configuration disagrees by 4.75 %.**
`c64_all70k_control_yihou`, run at 12:00 UTC, gave **12.5537 ms** where this series' 0 % point gave
11.9839 ms. Its config was checked field by field against the series point — `warmup_steps=10`,
`batch_size=64`, `output_len=10000`, both fusion flags present, `mem-fraction-static 0.85` — and
they are the same. It ran immediately after a C=256 / ISL-70000 run that had four times the batch
and ran for six minutes; whether that is the explanation is **not known**. Tenancy for that run was
not captured, because the before/after `--showpids` discipline only started with this series.

That outlier is 72 % of the entire 0 → 100 % effect. It is reported, not explained, and not folded
into the table. What can be said is that it sits outside the series, whose six points were all
taken under verified-clean tenancy within one contiguous window.

**Whether the trend is attributable to attention is not established.** One relevant line is in
every runtime log, recorded without interpretation:

```
[dense-decode] DSA dual-graph enabled: capturing dense (k-only) + sparse (full indexer)
decode graphs; dispatch on max_kv_len vs index_topk=2048.
```

Both 70000 and 300000 are far above `index_topk = 2048`, so both should take the sparse path and
this dispatch is probably not the mechanism — but "probably" is a guess and the profile has not
been run.

## Superseded claims from earlier in this session

- "The 300K arm is 3.62 % faster than all-70K." **Wrong.** That compared `c64_one300k_yihou`
  (12.0989) against the 12.5537 outlier. Against this series, 12.5 % of 300K is *slower* than 0 %,
  not faster.
- "The earlier control may have been contaminated by a colleague's load." **Unsupported.** The
  timeline shows the colleague's container started 15 s after that control finished, and the
  processes observed later started ~28 min after it. The 4.75 % gap is unexplained, not attributed.
- "The 4.3 % gap between 12.5537 and 12.0310 shows contention." **Wrong** — 12.0310 came from the
  flag-bug series and had four optimisation flags missing.

## The flag bug, and what it cost

The first attempt at this series silently dropped `--warmup-steps 10`, `--mem-fraction-static 0.85`,
`--enable-aiter-allreduce-fusion` and `--enable-fused-qk-norm-rope`. `COMMON` was written as a
multi-line shell string and interpolated into a double-quoted `ssh "…"` command: it is not
word-split locally, so the remote shell received embedded newlines and executed lines 2 and 3 as
separate commands.

**Three runs completed with `complete=True`, an exact acceptance gate and 2768 iterations, and
nothing in the output indicated anything was wrong.** The only visible symptom was a non-zero exit
from the stray lines, which is easy to dismiss.

Two consequences, both now permanent:

1. `COMMON` must stay on one line, with a comment saying why.
2. The driver reads `config_yihou.json` and `server_cli` back after every run and prints
   `ARG CHECK: ok` or the specific failures. An experiment harness that cannot prove its own
   parameters reached the binary is not measuring what its filename says.

The three affected runs are preserved, not deleted, in `logs/aborted_flagbug_yihou/`. For the
record, the flags moved TPOT less than the run-to-run spread does: 50 % was 12.3742 without them
and 12.3450 with (0.24 % apart); 0 % was 12.0310 and 11.9839 (0.39 %).

## Suggested next step, not taken

Repeat the 0 % point three or four times in one contiguous window with tenancy captures, to
characterise the run-to-run spread at fixed configuration. Until that number exists, the 4.75 %
outlier means no single pair of points in this table can be defended on its own — only the ordering
can.
