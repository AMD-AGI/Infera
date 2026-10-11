# Notes — what went wrong, and what the design choices were for

What / why / how, so the reasoning survives and not just the instruction.

---

## 1. The flag bug — four options silently dropped, and nothing said so

**What.** The first attempt at this series ran without `--warmup-steps 10`,
`--mem-fraction-static 0.85`, `--enable-aiter-allreduce-fusion` and `--enable-fused-qk-norm-rope`.

**Why.** `COMMON` was written across three lines:

```bash
COMMON="--tp-size 8 ... --max-running-requests 64
        --output-len 10000 --accept-length 3.61 --warmup-steps 10
        --enable-aiter-allreduce-fusion ... --mem-fraction-static 0.85"
```

and interpolated into `ssh "cd … && run_decode.sh $NAME $SPEC $COMMON"`. Inside double quotes there
is no local word splitting, so the string reaches the **remote** shell with its newlines intact —
and there a newline is a command separator, not whitespace. Lines 2 and 3 were executed as separate
remote commands.

**Why it was dangerous rather than merely annoying.** The three affected runs produced
`complete = True`, a bit-exact acceptance gate, and 2768 iterations. Every quality signal the
harness emits said the runs were good. The only symptom was a non-zero exit code from the stray
lines being run as commands — easy to attribute to a warning and move on. Had it not been caught,
the series would have been internally consistent and quietly incomparable to every other number in
this project.

**How it is prevented now.** Two changes, both permanent:

1. `COMMON` stays on one line, with a comment saying why. Reflowing it for readability
   reintroduces the bug.
2. The driver reads `config_yihou.json` and `server_cli` back after every run and prints
   `ARG CHECK: ok` or the specific failures. **A harness that cannot prove its own parameters
   reached the binary is not measuring what its filename says.**

**Cost, for calibration.** The four flags moved TPOT by less than the run-to-run spread does:
50 % was 12.3742 without them and 12.3450 with (0.24 % apart); 0 % was 12.0310 and 11.9839
(0.39 %). The bug mattered because it destroyed comparability, not because it was large.

The three discarded runs are preserved in `results/flagbug_evidence/` — their
`config_yihou.json` files are the primary evidence of which flags were missing.

---

## 2. Shuffled run order is the experiment, not a detail

**What.** The six points were executed 50, 0, 100, 25, 75, 12.5 %.

**Why.** Within-run drift of 0.35–0.61 % was measured in the previous task, and this series' whole
effect is 6.61 %. Run ascending, a monotone drift would be **perfectly collinear** with the ISL
trend: the data would look identical whether the effect were real or entirely an artifact, and no
amount of post-hoc analysis could tell them apart. Shuffled, the monotone result is evidence —
1/720 by chance for six points.

**How to keep it.** Do not "tidy" the series into ascending order. The script comments say so.

---

## 3. Tenancy has to be in the record, not in memory

**What.** Every run captures `rocm-smi --showpids` before and after, into `results/tenancy/`.

**Why.** The node is shared and its containers ignore Slurm. Earlier the same day the GPUs were
found fully occupied by three other workloads **while Slurm showed all 8 GPUs allocated to us**.
Worse, a colleague's container appeared partway through this session, leaving one earlier
measurement impossible to adjudicate after the fact — the container existed during the run but the
GPU processes observed later had started after it, and there was no capture to settle it.

**The earlier instrument was useless for this.** A previous series captured
`--showtemp --showpower --showclocks --showuse`, which says nothing about who else is on the card.
Capture `--showpids`.

**How to read it.** All twelve captures here say `No KFD PIDs currently running`. A run whose
capture shows other PIDs should be rerun, not adjusted.

---

## 4. The 0 % point reserves less KV than the others — and why that does not explain the trend

**What.** `reserved_tokens` is 640512 at 0 % and 2480128 at every other point.

**Why.** Allocation is uniform-max: rows are sized for `max(ISL)` before the per-request prefix is
truncated. A batch containing even one 300K request reserves as if all of them were 300K. So the
0 % point is structurally different from the rest, which is a real confound for a 0 %-vs-rest
comparison.

**How it was controlled.** The five points that share an identical 2480128-token reservation —
12.5, 25, 50, 75, 100 % — are still monotone across a +5.44 % span, under shuffled order. The trend
survives dropping the structurally-different point entirely.

Peak allocated memory is 236.0 GB at all six points, so the torch-side footprint is set by
`--mem-fraction-static 0.85`, not by the workload.

---

## 5. The unexplained 4.75 %

**What.** `c64_all70k_control_yihou` (12:00 UTC) gave 12.5537 ms; the series' 0 % point gave
11.9839 ms. Byte-identical configuration, verified field by field.

**Why it matters.** That gap is 72 % of the entire 0 → 100 % effect. If the run-to-run spread at
fixed configuration really is ~4.75 %, then no individual pair of rows in the result table can be
defended on its own, and only the ordering argument survives.

**What is known.** The earlier run followed a C=256 / ISL-70000 run — four times the batch, six
minutes long — by about eleven minutes. Whether that is the cause is **not known**. That run
predates the before/after `--showpids` discipline, so its tenancy cannot be checked retrospectively.

**What would settle it.** Three or four repeats of the 0 % point in one contiguous window with
tenancy captures. Not done; it is the first thing to do before quoting any magnitude from this
table.

---

## 6. Two operational traps that fired during this work

**`pkill -f <pattern>` self-matches.** `pkill -f 'run_300k_fraction_series_yihou.sh'` killed the
shell that was running it, because that shell's own command line contains the pattern. This fired
**three times** in one session before the lesson stuck. Get the PID with `ps` first, then `kill` by
number.

**Killing the outer driver leaves the in-container run alive.** After the driver was killed, the
current point was still executing inside the container (`03:50` elapsed). This is documented in the
project CLAUDE.md and it still caught us. After any interruption, check
`docker top <container> -o pid,etime,args` and clean up explicitly — then confirm
`rocm-smi --showpids` is empty before starting anything new.

---

## 7. Why the acceptance gate still applies to ragged batches

**What.** Every point here, uniform or mixed, must produce
`realized_accept_length == 3.6134393063583814` and `verify_iterations == 2768`.

**Why that is not a coincidence.** Acceptance is simulated as a single scalar per iteration
broadcast across the whole batch (`spec_utils.py`), and every request shares `output_len`. So all
requests emit the same number of tokens per iteration and finish on the same iteration, regardless
of where their prefixes started. ISL does not enter the acceptance accounting at all.

**Why it is worth keeping as a gate.** It is the cheapest available proof that a run executed the
same logical work as every other run in the comparison. A point that misses it is not a slow run,
it is a different run.
